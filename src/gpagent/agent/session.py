"""The commentary agent: capture events in, speech out.

`CommentaryAgent.handle()` takes an explicit `now`, defaulting to an injected
clock. That one signature is what makes the whole component testable: unit tests
call it synchronously with made-up times, `drive_from_session` calls it with the
timestamp recorded in the file (no sleeping, no wall clock, deterministic), and
`drive_from_bus` lets it default to `time.monotonic` for live capture. Nothing
below this layer knows which of the three it is running under.

Conversation shape, per response:

    conversation.item.create   one user item: accumulated controller summaries
    conversation.item.create   one user item: the screenshots -- ordering note,
                               trail, current frame (see context.py)
    input_audio_buffer.append  the player's utterance, PCM16 24 kHz mono --
    input_audio_buffer.commit  exactly what capture produced, no conversion
    conversation.item.create   one system item: which reason we are speaking
                               for (see persona.reason_note)
    response.create            bare -- persona, language and output cap are all
                               constant, so they live in session.update

The images have their own item rather than riding along with the text because
they are the half that gets deleted again: an item is the unit `conversation
.item.delete` operates on, so anything that has to outlive a screenshot cannot
share an item with one. See `_maybe_prune`.

Only `reply` sends audio. `react` and `ambient` are the same shape without it.

None of that changes with `agent.output`. What changes is the *answer*: with
`output = "text"` the session asks for `["text"]` on `session.update`, so the
model writes the remark instead of speaking it, it arrives on
`response.output_text.*` rather than as audio deltas plus a transcript, and it
goes to the HUD instead of the speakers. That is a session-wide choice, not a
per-turn one -- see `AgentConfig.output`.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from ..config import CaptureConfig
from ..events import (
    AgentResponse,
    Event,
    GamepadActivity,
    ScreenFrame,
    SessionEnd,
    SessionStart,
    SpeechSegment,
)
from ..tokens import UsageMeter
from .context import ContextBuffer, Frame
from .hud import NullHud, hold_for
from .persona import instructions, reason_note
from .playback import NullPlayer
from .policy import SpeakPolicy

log = logging.getLogger(__name__)

__all__ = [
    "CommentaryAgent",
    "ReplayClock",
    "drive_from_bus",
    "drive_from_session",
]

SAMPLE_RATE = 24000
#: base64 of ~32 KB of PCM per frame keeps individual websocket messages small
AUDIO_CHUNK_BYTES = 32768


class ReplayClock:
    """A clock driven by the timestamps in a recording, not by the wall."""

    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def set(self, t: float) -> None:
        # Monotonic: a recording is ordered, and cooldown arithmetic on a clock
        # that can go backwards is a debugging afternoon nobody needs.
        self.now = max(self.now, t)


@dataclass
class Line:
    """One line of the session's story, for the log and the console."""

    t: float
    kind: str
    text: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"t": round(self.t, 3), "kind": self.kind, "text": self.text, **self.extra}


class CommentaryAgent:
    def __init__(
        self,
        cfg: CaptureConfig,
        transport,
        player=None,
        hud=None,
        *,
        clock: Callable[[], float] = time.monotonic,
        log_path: str | Path | None = None,
        on_line: Callable[[Line], None] | None = None,
        recorder: Callable[[Event], None] | None = None,
    ):
        self.cfg = cfg.agent
        self.speak_cfg = cfg.speak
        self.hud_cfg = cfg.hud
        self.clock = clock
        self.transport = transport
        self.player = player if player is not None else NullPlayer(clock)
        # Where a text remark lands. Null unless the caller opened a real one:
        # a session with `output = "text"` and no display still runs, still
        # logs and still costs money, it just has nowhere to put the words.
        self.hud = hud if hud is not None else NullHud()
        self.policy = SpeakPolicy(cfg.speak, clock)
        self.context = ContextBuffer(cfg.agent)
        self.meter = UsageMeter(model=cfg.agent.model)
        self.lines: list[Line] = []
        self._on_line = on_line
        self._log_path = Path(log_path) if log_path else None
        self._log_fh: IO[str] | None = None

        #: unanswered utterances, oldest first: (t, pcm, dur_ms)
        self._pending: list[tuple[float, bytes, int]] = []
        self._response_active = False
        self._response_started: float | None = None
        self._audio_item_id: str | None = None
        #: False between cancelling a response and starting the next one, so
        #: whatever is still in flight for the cancelled one is not played or
        #: shown. Text needs this as much as audio does: the API emits
        #: `response.output_text.done` for an interrupted response too.
        self._accept_output = True
        #: Items this client created that it is willing to delete again, oldest
        #: first: (t, item_id, kind). Screenshots and per-turn nudges only --
        #: see `_maybe_prune` for why the conversation proper is not in here.
        self._disposable: list[tuple[float, str, str]] = []
        self._item_seq = 0
        self._last_prune = 0.0
        #: how many of each kind have been deleted, for `report()`
        self.pruned: Counter[str] = Counter()
        self._prune_rounds = 0
        #: age of each frame at the moment it was attached, for tuning capture
        self._frame_ages: list[float] = []

        # -- recording ----------------------------------------------------
        self._recorder = recorder
        #: clock() - event.t, so agent events land on the capture timeline.
        #: Live these are different bases (bus time vs time.monotonic); in
        #: deterministic replay they are the same and this is zero.
        #: Estimated as the SMALLEST lag seen, not the first: see `handle`.
        self._t_offset: float | None = None
        #: agent events get their own seq range, well clear of capture's
        self._agent_seq = 1_000_000
        self._response_audio = bytearray()
        self._response_reason = ""
        self._response_transcript = ""
        self._response_asked_at: float | None = None
        self._response_first_out: float | None = None
        self._response_usage: dict[str, Any] = {}
        self._pump: asyncio.Task | None = None
        self._watchdog: asyncio.Task | None = None
        #: set by close(), so the pump stops reopening the session
        self._closing = False
        self.spoke = 0

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self._log_path, "w", buffering=1)
        await self.transport.connect()
        try:
            await self.player.start()
        except Exception:
            # No sound card, no session bus, no GStreamer: worth a loud warning,
            # not worth taking the whole agent down for.
            log.warning("playback failed to start; continuing silent", exc_info=True)
            self.player = NullPlayer(self.clock)
        await self.transport.send(self.session_update())
        self._pump = asyncio.create_task(self._pump_events(), name="realtime-pump")
        self._emit(Line(self.clock(), "session", f"session open ({self.cfg.model})"))

    async def close(self) -> None:
        self._closing = True
        if self._watchdog is not None:
            self._watchdog.cancel()
        if self._pump is not None:
            self._pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pump
            self._pump = None
        await self.transport.close()
        await self.player.stop()
        self.hud.close()
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def session_update(self) -> dict:
        audio_input: dict[str, Any] = {
            # 24 kHz PCM16 mono is exactly what `speech.segment` carries.
            "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
            # We drive turns ourselves: capture already ran a VAD, and a
            # commentator has to be able to speak with no user turn at all.
            "turn_detection": None,
        }
        if self.cfg.transcribe_player:
            transcription: dict[str, Any] = {"model": self.cfg.transcribe_model}
            if self.cfg.language:
                # The one place language is a real API setting: it improves
                # transcription accuracy and latency. Output language has no
                # such parameter and is handled in the instructions.
                transcription["language"] = self.cfg.language
            audio_input["transcription"] = transcription
        # The one line that decides whether this session is heard or read. It
        # is not a rendering choice: it changes what the model generates, so it
        # can only be made once, here, for the whole session.
        text_out = self.cfg.text_output
        audio: dict[str, Any] = {"input": audio_input}
        if not text_out:
            audio["output"] = {
                # `rate` is required here even though the SDK's TypedDict
                # marks it optional and the docs example omits it: the
                # server rejects the session without it.
                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                "voice": self.cfg.voice,
                "speed": self.cfg.voice_speed,
            }
        session: dict[str, Any] = {
            "type": "realtime",
            "output_modalities": ["text"] if text_out else ["audio"],
            "instructions": instructions(self.cfg),
            "max_output_tokens": self.cfg.max_output_tokens,
            "truncation": {
                "type": "retention_ratio",
                "retention_ratio": self.cfg.truncation_retention_ratio,
            },
            # Input audio stays either way: a text session still listens, and
            # still transcribes what it hears for the log.
            "audio": audio,
        }
        if self.cfg.reasoning_effort is not None:
            # Non-reasoning models (gpt-realtime, gpt-realtime-1.5) reject this
            # field, so it's only sent when explicitly configured.
            session["reasoning"] = {"effort": self.cfg.reasoning_effort}
        return {"type": "session.update", "session": session}

    # -- capture events ----------------------------------------------------

    async def handle(self, event: Event, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        # The two clocks differ by a constant plus however long this event sat
        # in the queue, so the smallest lag ever seen is the best estimate of
        # the constant. Taking the *first* sample instead put every recorded
        # response 3.5 s early in a real session: the agent's first event came
        # out of a backlog that built up while `start()` opened the realtime
        # session, and every later event was handled promptly.
        lag = now - event.t
        if self._t_offset is None or lag < self._t_offset:
            self._t_offset = lag
        if self._recorder is not None:
            self._recorder(event)

        if isinstance(event, GamepadActivity):
            self.context.add_summary(now, event.summary)
            self.policy.on_gamepad(event.intensity, event.apm, now=now)
        elif isinstance(event, ScreenFrame):
            if event.data:
                self.context.add_frame(Frame.from_event(event, now))
            self.policy.on_scene(event.scene_score, now=now)
        elif isinstance(event, SpeechSegment):
            if event.data:
                self._pending.append((now, event.data, event.dur_ms))
            self.policy.on_speech_segment(event.dur_ms, now=now)
            self._emit(
                Line(now, "player", f"speech {event.dur_ms} ms", {"rms_dbfs": event.rms_dbfs})
            )
        elif isinstance(event, SessionStart):
            self._emit(Line(now, "session", "capture started"))
        elif isinstance(event, SessionEnd):
            self._emit(Line(now, "session", "capture ended"))

        await self.tick(now)

    async def on_speech_start(self, now: float | None = None) -> None:
        """The player has started talking. Get out of the way."""
        now = self.clock() if now is None else now
        self.policy.on_speech_start(now)
        if self._speaking(now) and self._audible(now):
            await self._cancel_response(now, why="barge-in")

    def _speaking(self, now: float) -> bool:
        """Is the agent's voice coming out of the speakers right now?

        Not the same as "a response is in flight": offline the response
        completes instantly but the sentence it produced still occupies time,
        and that is precisely the window barge-in has to cover.

        A text session has no such window. The remark is on screen for as long
        as it takes to read, but it is not *occupying* anything -- a second one
        stacks under it, and cutting the first one off would be deleting words
        the player may be halfway through. So here the question is only whether
        the model is still writing.
        """
        if self.cfg.text_output:
            return self._response_active
        return self._response_active or self.player.timer.remaining_ms(now) > 0

    def _audible(self, now: float) -> bool:
        """Has any of the current sentence actually reached the player?

        Barge-in exists to stop the agent talking over someone. A response
        that has not made a sound yet is not talking over anyone, and killing
        it throws away a turn already paid for. In sess7 a player talking in
        short bursts cancelled four consecutive replies at 0.0 s each and got
        answers to none of their questions; ten of sixty-six responses were
        generated, billed and discarded without a sound. Let it through -- the
        new utterance is already queued and folds into the next turn.

        Text is never audible in that sense, and that is the whole answer for a
        text session: nobody is being talked over, so nothing needs cutting off.
        The player's utterance still lands and still gets answered on the next
        turn -- it just does not cost a response to make room for itself.
        """
        if self.cfg.text_output:
            return False
        return self.player.timer.played_ms(now) > 0

    async def tick(self, now: float | None = None) -> None:
        """Speak if the policy says so. Safe to call as often as you like."""
        now = self.clock() if now is None else now
        await self._maybe_prune(now)
        reason = self.policy.decide(now)
        if reason is None:
            return
        await self._speak(reason, now)

    # -- speaking ----------------------------------------------------------

    async def _speak(self, reason: str, now: float) -> None:
        if self._speaking(now):
            # The previous sentence is still coming out of the speakers. A
            # reply is exempt from the quiet floor, so this happens whenever the
            # player asks something the moment the model stops generating.
            # Pushing the new utterance into the same appsrc would queue it
            # behind the old one and play both, late.
            await self._cancel_response(now, why="superseded")
        # A new utterance starts here, so nothing the last one left on the
        # playback clock belongs to it. See `PlaybackTimer.discard`.
        self.player.discard()
        self._accept_output = True
        self._audio_item_id = None
        with_image = self.cfg.image_on_unprompted or reason == "reply"
        ctx = self.context.take(now, with_image=with_image)

        detail: list[str] = []
        if ctx:
            if ctx.text:
                await self.transport.send(self._text_item(now, ctx.text))
                detail.append(f'"{ctx.text}"')
            if ctx.frame:
                await self.transport.send(self._images_item(now, ctx))
                # Frame age is the whole point of the capture-side trigger
                # interval now that at most one frame is sent per response:
                # it buys freshness, not tokens.
                age = now - ctx.frame.t
                self._frame_ages.append(age)
                detail.append(
                    f"frame {ctx.frame.w}x{ctx.frame.h} {self.cfg.image_detail} "
                    f"({age:.1f}s old)"
                )
            if ctx.trail:
                reach = now - ctx.trail[0].t
                detail.append(
                    f"trail {len(ctx.trail)} {self.cfg.image_trail_detail} "
                    f"(back to {reach:.1f}s)"
                )

        if reason == "reply":
            utterances = self._take_pending(now)
            if utterances:
                total_ms = sum(dur for _, dur in utterances)
                await self._send_audio(b"".join(pcm for pcm, _ in utterances))
                detail.append(
                    f"audio {total_ms} ms"
                    + (f" ({len(utterances)} utterances)" if len(utterances) > 1 else "")
                )
            else:
                # Metadata without a blob: the recording was read without
                # payloads, or capture dropped one. Say so rather than sending
                # a reply turn with nothing in it.
                log.warning("reply with no audio payload; sending context only")

        # The reason goes in as a system item rather than on
        # `response.create.instructions`, which would replace the session
        # persona and, because it varies per turn, break the cached prefix.
        # See `persona.reason_note`.
        await self.transport.send(self._reason_item(now, reason))
        await self.transport.send({"type": "response.create"})
        self.policy.mark_spoken(reason, now)
        self._response_active = True
        self._response_started = now
        self._response_reason = reason
        self._response_asked_at = now
        self.spoke += 1
        self._arm_watchdog()

        cause = self.policy.react_cause if reason == "react" else ""
        extra: dict[str, Any] = {"sent": detail, "state": self.policy.state(now)}
        if ctx.frame:
            # Identities, not just counts. `sent` above is the console story;
            # this is what lets a reader pull the exact blobs back out of
            # `events.jsonl` and see what the model was looking at, at which
            # detail. See `gpagent inspect --sent-sheet`.
            extra["frames"] = {
                "current": ctx.frame.seq,
                "detail": self.cfg.image_detail,
                "trail": [f.seq for f in ctx.trail],
                "trail_detail": self.cfg.image_trail_detail,
            }
        self._emit(
            Line(now, "ask", f"-> {reason}" + (f" ({cause})" if cause else ""), extra)
        )

    def _take_pending(self, now: float) -> list[tuple[bytes, int]]:
        """Every utterance still worth answering, oldest first.

        The player often gets two or three sentences out before the agent has a
        turn available. Sending only the last one answers the tail of a thought;
        sending all of them costs a few audio tokens and answers the thought.
        Anything past the reply TTL is dropped on the same rule the policy uses.
        """
        cutoff = now - self.speak_cfg.reply_ttl_s
        fresh = [(t, pcm, dur) for t, pcm, dur in self._pending if t >= cutoff]
        self._pending = []

        # Newest-first while trimming to the cap, then back into speaking order.
        kept: list[tuple[bytes, int]] = []
        total = 0
        for _, pcm, dur in reversed(fresh):
            if total + dur > self.speak_cfg.max_reply_audio_ms and kept:
                break
            kept.append((pcm, dur))
            total += dur
        kept.reverse()
        return kept

    def _text_item(self, now: float, text: str) -> dict:
        """The controller summaries. Not disposable: it is the cheap half."""
        return self._item(now, "gpctx", "user", [{"type": "input_text", "text": text}], None)

    def _images_item(self, now: float, ctx) -> dict:
        """Every screenshot this turn sends, in one item so one delete drops them.

        Trail and current frame share the item deliberately. They are created in
        the same turn and pruned on the same cutoff, so splitting them would buy
        two `conversation.item.delete` calls and two chances to invalidate the
        prompt cache in exchange for a distinction nothing ever draws.
        """
        content: list[dict] = []
        # A turn with no trail sends the current frame alone and says nothing
        # extra about it: the note exists to order *several* images, and one
        # image needs no ordering. That is most first turns and any turn whose
        # gap held nothing new -- 7 of 68 on sess7.
        if ctx.trail and ctx.frame:
            # Images in one item carry no timestamps and nothing says which way
            # round they go, so a bare pile of screenshots reads as ambiguous at
            # best and as "the screen is flickering" at worst. One line of text
            # ahead of them costs ~25 tokens and makes them a sequence.
            span = ctx.frame.t - ctx.trail[0].t
            n = len(ctx.trail)
            template = self.cfg.trail_note_template or (
                "[screen] {n} earlier {frame_word} from the last {span:.0f}s, "
                "oldest first, then the current screen last."
            )
            content.append(
                {
                    "type": "input_text",
                    "text": template.format(
                        n=n, frame_word="frame" if n == 1 else "frames", span=span
                    ),
                }
            )
            for f in ctx.trail:
                content.append(self._image_part(f, self.cfg.image_trail_detail))
        if ctx.frame:
            content.append(self._image_part(ctx.frame, self.cfg.image_detail))
        return self._item(now, "gpimg", "user", content, "image")

    def _item(
        self, now: float, prefix: str, role: str, content: list[dict], kind: str | None
    ) -> dict:
        """One `conversation.item.create`, with an id we chose ourselves.

        Client-assigned so an item can be deleted later without waiting for the
        server to say what it called it -- and, since `conversation.item.created`
        arrives asynchronously, without a race between deciding to prune and
        learning the name of the thing to prune. `kind` names the disposable
        ones; None means the item stays for the life of the session.
        """
        self._item_seq += 1
        item_id = f"{prefix}{self._item_seq:012d}"
        if kind is not None:
            self._disposable.append((now, item_id, kind))
        return {
            "type": "conversation.item.create",
            "item": {"id": item_id, "type": "message", "role": role, "content": content},
        }

    def _image_part(self, frame, detail: str) -> dict:
        b64 = base64.b64encode(frame.data).decode("ascii")
        return {
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{b64}",
            "detail": detail,
        }

    def _reason_item(self, now: float, reason: str) -> dict:
        """The per-turn nudge, as a system message appended before the response.

        Disposable for the same reason the screenshots are, and less obviously:
        each one is only ~30 tokens, but there is one per turn and they are all
        but identical to each other, so by the hundredth turn the conversation
        is carrying a hundred copies of three sentences that only ever applied
        to the turn they were written for.
        """
        content = [{"type": "input_text", "text": reason_note(reason, self.cfg)}]
        return self._item(now, "gprsn", "system", content, "note")

    async def _send_audio(self, pcm: bytes) -> None:
        for start in range(0, len(pcm), AUDIO_CHUNK_BYTES):
            chunk = pcm[start : start + AUDIO_CHUNK_BYTES]
            await self.transport.send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
            )
        # Commit turns the buffer into a user message item; with turn_detection
        # null nothing else would.
        await self.transport.send({"type": "input_audio_buffer.commit"})

    # -- server events -----------------------------------------------------

    async def _pump_events(self) -> None:
        delay = self.cfg.reconnect_min_s
        while not self._closing:
            try:
                async for event in self.transport:
                    delay = self.cfg.reconnect_min_s
                    try:
                        await self.on_server_event(event)
                    except Exception:
                        log.exception("failed handling %s", event.get("type"))
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("realtime connection dropped")
            if not await self._reconnect(delay):
                return
            delay = min(delay * 2, self.cfg.reconnect_max_s)

    async def _reconnect(self, delay: float) -> bool:
        """Open a fresh session after the socket goes away.

        Not an edge case: the API caps a session at 60 minutes, so on any long
        session this is the expected end of every hour. sess7 ran 61 minutes
        and spent its last 82 seconds mute, because the pump simply exited --
        the quietest possible failure, and the one the timeouts elsewhere in
        this file exist to avoid.

        History does not come back with us. Reseeding instructions only is a
        deliberate choice from the design: a couch commentator that forgets
        what happened four minutes ago is fine, and it keeps recovery to one
        `session.update`.
        """
        if self._closing:
            return False
        self._emit(Line(self.clock(), "session", f"connection lost; reopening in {delay:.0f}s"))
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            raise
        if self._closing:
            return False
        try:
            await self.transport.close()
            await self.transport.connect()
            await self.transport.send(self.session_update())
        except asyncio.CancelledError:
            raise
        except Exception:
            log.warning("reconnect failed", exc_info=True)
            return True  # keep trying; close() is what ends this loop
        self._after_reconnect()
        self._emit(Line(self.clock(), "session", f"session reopened ({self.cfg.model})"))
        return True

    def _after_reconnect(self) -> None:
        """Drop everything that referred to the session that just died."""
        now = self.clock()
        # Item ids belonged to the old conversation; deleting or truncating
        # them on the new one would earn an error for each.
        self._disposable.clear()
        self._audio_item_id = None
        self.player.flush(now)
        self._accept_output = True
        self._reset_response_record()
        if self._response_active:
            # The gate is absolute while set, and nothing is ever going to
            # finish the response it is holding open.
            self._response_active = False
            self._response_started = None
            self.policy.on_response_cancelled(now)
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None

    async def on_server_event(self, event: dict) -> None:
        kind = event.get("type")
        now = self.clock()

        if kind == "response.output_audio.delta":
            if not self._accept_output:
                # Deltas already in flight when we cancelled. Playing them would
                # emit a fragment of the sentence we just cut off, a second
                # after cutting it off.
                return
            self._audio_item_id = event.get("item_id") or self._audio_item_id
            delta = event.get("delta")
            if delta:
                pcm = base64.b64decode(delta)
                if self._response_first_out is None:
                    self._response_first_out = now
                if self._recorder is not None:
                    self._response_audio += pcm
                self.player.push(pcm)
        elif kind == "response.output_item.added":
            self._audio_item_id = (event.get("item") or {}).get("id") or self._audio_item_id
        elif kind == "response.output_audio_transcript.done":
            self._response_transcript = event.get("transcript") or ""
            self._emit(Line(now, "say", self._response_transcript))
        elif kind == "response.output_text.delta":
            # Nothing is shown per delta -- half a remark appearing and then
            # growing is a worse thing to catch out of the corner of an eye
            # than one that arrives whole. This is here for the latency
            # measurement, which is "when did the answer start", not "when was
            # it finished".
            first = self._accept_output and event.get("delta")
            if first and self._response_first_out is None:
                self._response_first_out = now
        elif kind == "response.output_text.done":
            # Also emitted for an interrupted or cancelled response, which is
            # exactly what `_accept_output` is for: showing it would put a
            # remark on screen that the session already decided against.
            if not self._accept_output:
                return
            self._response_transcript = event.get("text") or ""
            if self._response_transcript:
                self._emit(Line(now, "say", self._response_transcript))
                self.hud.show(self._response_transcript)
        elif kind == "conversation.item.input_audio_transcription.completed":
            self._emit(Line(now, "heard", event.get("transcript") or ""))
        elif kind == "response.done":
            await self._finish_response(event)
        elif kind == "error":
            err = event.get("error") or {}
            if _is_benign(err):
                # The response finished server-side between our deciding to
                # cancel and the cancel arriving. Nothing was wrong and nothing
                # needs doing; the local playback stop already happened.
                log.debug("ignoring benign realtime error: %s", err)
                return
            self._emit(Line(now, "error", str(err.get("message") or err)))
            log.error("realtime error: %s", err)

    async def _finish_response(self, event: dict) -> None:
        response = event.get("response") or {}
        self._response_usage = response.get("usage") or {}
        self.meter.add(response.get("usage"))
        spoken_ms = await self.player.drain()
        if self.cfg.text_output:
            # What a written remark "takes" is how long it is on screen, which
            # is reading time -- the same number the HUD holds it for. The
            # quiet floor is then measured from when the player is done with
            # it, exactly as it is measured from the end of a sentence.
            spoken_ms = hold_for(self._response_transcript, self.hud_cfg) * 1000.0
        elif spoken_ms <= 0:
            # Dry runs and --no-playback never push audio; take the duration the
            # API reported instead, so the cooldown still reflects a real pause.
            details = (response.get("usage") or {}).get("output_token_details") or {}
            spoken_ms = int(details.get("audio_tokens") or 0) * 50.0
        now = self.clock()
        # A response can finish without having said anything -- cancelled,
        # failed, or stopped at max_output_tokens. Without this the turn is
        # recorded as a 0 ms response and the player's question just goes
        # unanswered with nothing in the log to say why.
        status = response.get("status")
        detail = response.get("status_details") or {}
        why = detail.get("reason") or (detail.get("error") or {}).get("message") or ""
        # `client_cancelled` is the server agreeing to a barge-in we asked for,
        # already logged as `cut`. Noting it too made it two thirds of the
        # notes in sess7 and buried the two that mattered.
        if status and status != "completed" and why != "client_cancelled":
            self._emit(Line(now, "note", f"response {status}" + (f": {why}" if why else "")))
        end = now
        if self.cfg.text_output:
            # The remark is up now and done being read later, so the floor
            # starts later. Nothing goes on the playback clock: there is no
            # audio to model, and `_speaking` deliberately does not consult it
            # in a text session.
            end = now + spoken_ms / 1000.0
        elif not self.player.realtime:
            # Nothing waited for the audio, so the sentence ends in the future.
            # Hand the duration to the player's timer as well, so the agent can
            # still tell it is mid-sentence when the player interrupts.
            self.player.simulate(spoken_ms, now)
            end = now + spoken_ms / 1000.0
        self._response_active = False
        self._response_started = None
        self._record_response(now, cut=False)
        self.policy.on_response_finished(spoken_ms=spoken_ms, now=end)
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None

    async def _cancel_response(self, now: float, *, why: str) -> None:
        # Only if the server is still generating. Audio keeps playing locally
        # for seconds after `response.done`, and cancelling then earns a
        # "no active response found" error for no benefit -- the part that
        # matters is stopping the speakers and truncating the item.
        if self._response_active:
            await self.transport.send({"type": "response.cancel"})
        heard_ms = self.player.flush(now)
        if self._audio_item_id is not None and heard_ms > 0:
            # Tell the model how much of its sentence actually reached the
            # player; without this it believes it said the whole thing.
            await self.transport.send(
                {
                    "type": "conversation.item.truncate",
                    "item_id": self._audio_item_id,
                    "content_index": 0,
                    "audio_end_ms": int(heard_ms),
                }
            )
        self._accept_output = False
        self._response_active = False
        self._response_started = None
        self._audio_item_id = None
        self._record_response(now, cut=True, heard_ms=heard_ms)
        self.policy.on_response_cancelled(now)
        self._emit(Line(now, "cut", f"{why} after {heard_ms / 1000:.1f}s"))

    def _record_response(self, now: float, *, cut: bool, heard_ms: float = 0.0) -> None:
        """Write what the agent just said into the session, audio and all.

        Emitted for cancelled responses too, marked `cut`, since "it started
        saying this and got interrupted" is exactly what you want to see when
        reading a session back.
        """
        if self._recorder is None or self._response_asked_at is None:
            self._reset_response_record()
            return

        pcm = bytes(self._response_audio)
        dur_ms = int(len(pcm) / 2 / (SAMPLE_RATE / 1000.0))
        latency = (
            int((self._response_first_out - self._response_asked_at) * 1000)
            if self._response_first_out is not None
            else 0
        )
        event = AgentResponse(
            data=pcm or None,
            reason=self._response_reason,
            transcript=self._response_transcript,
            # A text remark has no blob and no duration: `transcript` is the
            # whole of it, and `modality` is how a reader knows that is not a
            # recording with its audio missing.
            modality="text" if self.cfg.text_output else "audio",
            dur_ms=0 if self.cfg.text_output else (int(heard_ms) if cut and heard_ms else dur_ms),
            sample_rate=SAMPLE_RATE,
            latency_ms=max(0, latency),
            cut=cut,
            usage=self._response_usage,
        )
        event.t = now - (self._t_offset or 0.0)
        event.seq = self._agent_seq
        self._agent_seq += 1
        self._recorder(event)
        self._reset_response_record()

    def _reset_response_record(self) -> None:
        self._response_audio = bytearray()
        self._response_transcript = ""
        self._response_asked_at = None
        self._response_first_out = None
        self._response_usage = {}

    def _arm_watchdog(self) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()
        if not self.player.realtime:
            return  # nothing to wait for; offline runs are not real time
        self._watchdog = asyncio.create_task(self._cut_off_rambling())

    async def _cut_off_rambling(self) -> None:
        try:
            await asyncio.sleep(self.cfg.max_response_s)
        except asyncio.CancelledError:
            return
        if self._response_active:
            log.info("response ran past %.0fs; cutting it off", self.cfg.max_response_s)
            await self._cancel_response(self.clock(), why="too long")

    # -- context pruning ---------------------------------------------------

    async def _maybe_prune(self, now: float) -> None:
        """Delete the disposable half of the conversation, in one round.

        Two decisions, and both are about the prompt cache rather than about
        what the model needs to remember.

        **One cutoff.** Screenshots -- the current frame and the trail behind it
        -- and the per-turn system nudge all go on the same `prune_after_s`, so
        every deletion the session ever wants to make is available at the same
        moment. Three independent windows would mean three rounds and three
        invalidations to remove things that were created together.

        **In bulk.** A delete truncates the cached prefix at the position of the
        deleted item, and we always delete from the oldest end, so *any* round
        costs essentially the whole prefix: the next request is billed at the
        fresh rate. That price is per round, not per item, which turns the whole
        problem into "how often", not "how much". `prune_interval_s` is the
        floor; nine items in one round cost a ninth of nine rounds of one.

        The conversation proper is left alone. What the player said and what the
        agent answered is the thread of the session, it is not something the
        client can regenerate, and it is a fraction of the tokens; the server's
        `truncation.retention_ratio` is the mechanism aimed at that, and it
        drops history in amortized batches that leave the prefix intact.

        Why do it at all when the cost measurement says trimming loses (see
        `AgentConfig.prune_after_s`): cached tokens count against TPM in full.
        The rate limit does not care that history is cheap, only that it is
        re-sent, and on sess_movie2 that ended the session thirteen requests
        before the film did.
        """
        if self.cfg.prune_after_s <= 0:
            return
        if now - self._last_prune < self.cfg.prune_interval_s:
            return
        cutoff = now - self.cfg.prune_after_s
        stale = [entry for entry in self._disposable if entry[0] < cutoff]
        if not stale:
            # Deliberately without touching `_last_prune`: the interval limits
            # how often the cache may be thrown away, so a round that threw
            # nothing away must not push the next real one further out.
            return
        self._last_prune = now
        self._disposable = [entry for entry in self._disposable if entry[0] >= cutoff]
        for _, item_id, _kind in stale:
            await self.transport.send({"type": "conversation.item.delete", "item_id": item_id})
        counts = Counter(kind for _, _, kind in stale)
        self.pruned.update(counts)
        self._prune_rounds += 1
        self._emit(
            Line(
                now,
                "prune",
                f"dropped {len(stale)} items older than {self.cfg.prune_after_s:.0f}s: "
                + ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items())),
                {"pruned": dict(counts)},
            )
        )

    # -- logging -----------------------------------------------------------

    def _emit(self, line: Line) -> None:
        self.lines.append(line)
        if self._log_fh is not None:
            self._log_fh.write(json.dumps(line.to_dict(), separators=(",", ":")) + "\n")
        if self._on_line is not None:
            self._on_line(line)

    async def settle(self, rounds: int = 200) -> None:
        """Let queued server events be processed before deciding again.

        Only meaningful for the local transports, whose replies are already in
        memory; live, the pump task runs on its own.
        """
        pending = getattr(self.transport, "pending", None)
        if pending is None:
            return
        for _ in range(rounds):
            if not pending():
                return
            await asyncio.sleep(0)

    def report(self) -> dict[str, Any]:
        return {
            "output": self.cfg.output,
            "responses": self.spoke,
            "by_reason": dict(self.policy.counts),
            "declined": dict(self.policy.declined),
            "frames_seen": self.context.frames_seen,
            "frames_sent": self.context.frames_sent,
            "trail_sent": self.context.trail_sent,
            # Rounds, not just items: the round is what the prompt cache is
            # billed for, so "48 items in 6 rounds" is the number to read.
            "pruned": {**self.pruned, "rounds": self._prune_rounds},
            "frame_age_s": {
                "mean": round(sum(self._frame_ages) / len(self._frame_ages), 2),
                "max": round(max(self._frame_ages), 2),
            }
            if self._frame_ages
            else None,
            "usage": self.meter.to_dict(),
        }


def _is_benign(err: dict) -> bool:
    """Errors that are races we already handled, not faults."""
    message = str(err.get("message") or "").lower()
    return "cancellation failed" in message or "no active response" in message


# -- drivers ---------------------------------------------------------------


async def drive_from_bus(agent: CommentaryAgent, bus, *, tick_s: float = 0.25) -> None:
    """Live capture (or wall-clock replay) through `CaptureBus`.

    The periodic tick is what lets the agent speak with no event to react to:
    an ambient remark has, by definition, nothing arriving to trigger it.
    """
    from ..capture.audio import SIGNAL_SPEECH_START

    loop = asyncio.get_running_loop()

    def on_speech_start(_payload=None) -> None:
        loop.create_task(agent.on_speech_start())

    bus.on_signal(SIGNAL_SPEECH_START, on_speech_start)

    async def ticker() -> None:
        while True:
            await asyncio.sleep(tick_s)
            await agent.tick()

    tick_task = asyncio.create_task(ticker(), name="agent-tick")
    try:
        async for event in bus:
            await agent.handle(event)
    finally:
        tick_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await tick_task


async def drive_from_session(
    agent: CommentaryAgent,
    directory: str | Path,
    clock: ReplayClock,
    *,
    tick_s: float = 0.5,
) -> None:
    """Deterministic replay: no bus, no sleeping, time comes from the file.

    Uses the same cue reconstruction the live-shaped `ReplaySource` does, so
    `speech.start` lands at the *start* of each utterance and barge-in is
    exercised offline exactly as it would be live.
    """
    from ..replay import SIGNAL_SPEECH_START, build_cues
    from ..sinks.jsonl import read_session

    events = list(read_session(directory, load_blobs=True))
    cues = build_cues(events)
    next_tick = cues[0][0] if cues else 0.0

    for when, kind, payload in cues:
        # Catch up on ticks between events, so ambient remarks can happen in
        # the gaps rather than only when something else arrives.
        while next_tick < when:
            clock.set(next_tick)
            await agent.tick(next_tick)
            await agent.settle()
            next_tick += tick_s
        clock.set(when)
        if kind == "event":
            await agent.handle(payload, now=when)
        elif kind == SIGNAL_SPEECH_START:
            await agent.on_speech_start(now=when)
        await agent.settle()
