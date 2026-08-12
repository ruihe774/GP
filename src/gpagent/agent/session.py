"""The commentary agent: capture events in, speech out.

`CommentaryAgent.handle()` takes an explicit `now`, defaulting to an injected
clock. That one signature is what makes the whole component testable: unit tests
call it synchronously with made-up times, `drive_from_session` calls it with the
timestamp recorded in the file (no sleeping, no wall clock, deterministic), and
`drive_from_bus` lets it default to `time.monotonic` for live capture. Nothing
below this layer knows which of the three it is running under.

Conversation shape, once at the start of the session:

    session.update             persona, language, output modality
    conversation.item.create   one system item: the whole subtitle file, if one
                               was configured (see subtitles.py). Sent here,
                               before the first turn, so it sits in the cached
                               prefix for the rest of the session.

...and then, per response:

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
they are the half that gets removed again: an item is the unit `conversation
.item.delete` operates on, so anything that has to outlive a screenshot cannot
share an item with one. See `_prune_round`, which does not so much delete that
item as swap it for a sentence saying what used to be there.

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
from collections.abc import Callable, Sequence
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
from .persona import instructions, reason_note, subtitle_note
from .playback import NullPlayer
from .policy import SpeakPolicy
from .subtitles import Script, load_script

# `clock` is the injected time source everywhere in this file; the subtitle
# module's is a formatter, so it comes in under a name that says which.
from .subtitles import clock as film_clock

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

#: What stands in for an item a pruning round removed. Every one of these is
#: permanent -- a stub is not itself disposable except at the very last rung of
#: `_prune_now` -- so they are written to be the shortest sentence that still
#: means something. A screenshot at high detail is ~765 tokens and an utterance
#: a couple of hundred; these are a dozen.
STUBS = {
    #: `[dropped]` and not `[screen]`: the trail note that introduces real
    #: frames owns that marker, and nothing left in the conversation should
    #: point at images the model can no longer look at.
    "image": "[dropped] {n} {frame_word} of what was on screen here.",
    #: The bracket is load-bearing. Without it a transcript reads as typed
    #: context, which is a different thing from the player having said it.
    "heard": "[they said] {transcript}",
    #: No marker: the agent did say these words, and bracketing them would put
    #: a stage direction in the model's own mouth for it to imitate.
    "said": "{transcript}",
}
#: ...and what stands in when the transcript never arrived.
STUB_FALLBACKS = {
    "heard": "[they said something here.]",
    "said": "[a remark here was not recorded.]",
}


@dataclass
class Disposable:
    """One item this client is willing to remove again, and what replaces it.

    Mutable on purpose. An image knows its replacement text the moment it is
    created; a piece of audio does not. The transcript arrives on a later
    event, and audio deleted before its transcript lands takes the only record
    of what was said with it -- so the id is registered as soon as the item is
    known to exist, `replacement` is filled in afterwards, and `pending` is
    what stops a round touching the entry in between.
    """

    #: agent clock when the item entered the conversation
    t: float
    item_id: str
    #: "image" | "note" | "heard" | "said" | "stub". The session's own
    #: vocabulary rather than the API's: `heard` and `said` are already the
    #: `Line` kinds for the two sides of a spoken exchange, so `report()` reads
    #: "12 image, 9 note, 4 heard, 4 said".
    kind: str
    #: role the replacement item takes; assistant replacements carry
    #: `output_text` content, everything else `input_text`
    role: str = "user"
    #: None means delete outright, with nothing put back in its place
    replacement: str | None = None
    #: replacement not known yet -- never removable while true
    pending: bool = False
    #: template arguments for `STUBS[kind]`
    fields: dict[str, Any] = field(default_factory=dict)


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
        self.subtitle_cfg = cfg.subtitles
        # Read here, not in `start()`: a missing or unparseable file should fail
        # before a socket is opened and before the first dollar is spent.
        self.script: Script | None = load_script(cfg.subtitles)
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
        #: Everything removable, in conversation order. Screenshots, per-turn
        #: nudges, and both sides of a spoken exchange -- see `_prune_round`.
        self._disposable: list[Disposable] = []
        #: the same records by item id, so an event that arrives later (a
        #: transcript) can fill one in
        self._by_item: dict[str, Disposable] = {}
        #: Ids belonging to the turn being answered right now. No round may
        #: touch these, however badly it needs the room: deleting an item out
        #: from under an in-flight response is the one thing pruning has never
        #: been allowed to do.
        self._live_items: set[str] = set()
        self._item_seq = 0
        self._last_prune = 0.0
        #: how many of each kind have been removed, for `report()`
        self.pruned: Counter[str] = Counter()
        self._prune_rounds = 0
        self._forced_rounds = 0
        self._replaced = 0
        #: set when a response bills past `context_budget_tokens`; consumed by
        #: the next `tick`, because pruning belongs on the tick and not in the
        #: middle of reading a usage block
        self._prune_forced = False
        #: retries spent on the turn in flight, and the flag that collapses the
        #: two ways one refusal can be reported (see `_on_context_full`)
        self._context_full_retries = 0
        self._retry_pending = False
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
        await self._seed_subtitles()
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
            "truncation": self._truncation(),
            # Input audio stays either way: a text session still listens, and
            # still transcribes what it hears for the log.
            "audio": audio,
        }
        if self.cfg.reasoning_effort is not None:
            # Non-reasoning models (gpt-realtime, gpt-realtime-1.5) reject this
            # field, so it's only sent when explicitly configured.
            session["reasoning"] = {"effort": self.cfg.reasoning_effort}
        return {"type": "session.update", "session": session}

    def _truncation(self) -> Any:
        """Who is allowed to drop history, as `session.truncation`.

        "disabled" by default, and the reason is one item: the whole film's
        dialogue, sent once and deliberately placed at the head of the
        conversation so it keys the cached prefix of every request after it
        (see `_seed_subtitles`). Server truncation drops from the oldest end.
        It drops in amortized batches that leave the cache alone, which is a
        good trade right up to the batch that takes the script -- and nothing
        tells the client it happened, so the first sign would be an agent that
        has quietly stopped knowing what is being said on screen.

        Disabling it means the server refuses a response rather than trimming
        one, which is a failure the client can see and answer (see
        `_on_context_full`). The other two modes hand the job back.
        """
        if self.cfg.truncation != "retention_ratio":
            return self.cfg.truncation
        truncation: dict[str, Any] = {
            "type": "retention_ratio",
            "retention_ratio": self.cfg.truncation_retention_ratio,
        }
        if self.cfg.truncation_post_instruction_tokens > 0:
            truncation["token_limits"] = {
                "post_instructions": self.cfg.truncation_post_instruction_tokens
            }
        return truncation

    # -- subtitles ---------------------------------------------------------

    async def _seed_subtitles(self) -> None:
        """Send the whole subtitle file, once, before the first turn.

        Position in the conversation is the entire design. This item is created
        immediately after `session.update` and never deleted, so it is part of
        the cached prefix of every request in the session: a two-hour film's
        dialogue is ~8-10k tokens, billed as fresh input exactly once and at the
        cached rate every turn after that. The same text fed a cue at a time
        would land at the *tail* of each turn instead, where nothing is cached
        yet, and would be re-billed as fresh input for the rest of the session.

        It is not disposable. Pruning exists to stop the request growing, and
        this item does not grow -- it is a constant, and it is the one piece of
        context the agent cannot reconstruct from anything else it is sent.
        """
        if self.script is None:
            return
        text = f"{subtitle_note(self.subtitle_cfg)}\n\n{self.script.text}"
        content = [{"type": "input_text", "text": text}]
        now = self.clock()
        await self.transport.send(self._item(now, "gpsub", "system", content, None))
        offset = self.subtitle_cfg.offset_s
        self._emit(
            Line(
                now,
                "session",
                f"subtitles: {len(self.script.cues)} cues, ~{self.script.tokens} tok, "
                f"{film_clock(self.script.runtime_s)} of film"
                + (f", starting {film_clock(offset)} in" if offset else ""),
                {"subtitles": self.subtitle_summary()},
            )
        )

    def subtitle_summary(self) -> dict[str, Any] | None:
        if self.script is None:
            return None
        return {
            "path": self.script.path,
            "cues": len(self.script.cues),
            "tokens": self.script.tokens,
            "runtime_s": round(self.script.runtime_s, 1),
            "offset_s": self.subtitle_cfg.offset_s,
            "skipped_effects": self.script.skipped,
        }

    def _film_position(self, now: float) -> float | None:
        """Seconds into the film, or None while that is not yet knowable.

        The session clock and the film's clock differ by two things: the offset
        between the agent's clock and capture's (`_t_offset`, learned from the
        events themselves), and how far into the film capture started
        (`subtitles.offset_s`, which only the user knows). Before the first
        event there is no estimate of the first, and a position invented from
        `time.monotonic()` would be the machine's uptime -- so nothing is said
        rather than something wrong.
        """
        if self.script is None or self._t_offset is None:
            return None
        return now - self._t_offset + self.subtitle_cfg.offset_s

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
        # A fresh turn gets a fresh retry budget, and the items of the turn
        # before it stop being untouchable now that it is over.
        self._context_full_retries = 0
        self._retry_pending = False
        self._live_items.clear()
        with_image = self.cfg.image_on_unprompted or reason == "reply"
        ctx = self.context.take(now, with_image=with_image)

        detail: list[str] = []
        if ctx:
            if ctx.text:
                await self.transport.send(self._text_item(now, ctx.text))
                detail.append(f'"{ctx.text}"')
            if ctx.frame:
                await self.transport.send(self._live(self._images_item(now, ctx)))
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
        await self.transport.send(self._live(self._reason_item(now, reason)))
        await self.transport.send({"type": "response.create"})
        # Nothing may await between that send and this line: a transport that
        # answers instantly (RecordingTransport does) can refuse the response
        # before the gate is up, and `_on_context_full` reads the gate to tell
        # a rejected turn from a stray error.
        self.policy.mark_spoken(reason, now)
        self._response_active = True
        self._response_started = now
        self._response_reason = reason
        self._response_asked_at = now
        self.spoke += 1
        self._arm_watchdog()

        # The position note is part of what this turn showed the model, so it is
        # reported with the rest of it -- and reading a session back, "which
        # minute of the film was this" is the first thing anyone asks.
        position = self._film_position(now) if self.subtitle_cfg.position_note else None
        if position is not None:
            detail.append(f"film {film_clock(position)}")

        cause = self.policy.react_cause if reason == "react" else ""
        extra: dict[str, Any] = {"sent": detail, "state": self.policy.state(now)}
        if position is not None:
            extra["film_t"] = round(position, 1)
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
        """Every screenshot this turn sends, in one item so one round takes them.

        Trail and current frame share the item deliberately. They are created in
        the same turn and pruned on the same cutoff, so splitting them would buy
        two `conversation.item.delete` calls and two chances to invalidate the
        prompt cache in exchange for a distinction nothing ever draws. It also
        means one stub covers the lot when they go.
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
        shown = sum(1 for part in content if part["type"] == "input_image")
        return self._item(
            now,
            "gpimg",
            "user",
            content,
            "image",
            n=shown,
            frame_word="frame" if shown == 1 else "frames",
        )

    def _item(
        self,
        now: float,
        prefix: str,
        role: str,
        content: list[dict],
        kind: str | None,
        *,
        after: str | None = None,
        **stub_fields: Any,
    ) -> dict:
        """One `conversation.item.create`, with an id we chose ourselves.

        Client-assigned so an item can be removed later without waiting for the
        server to say what it called it -- and, since `conversation.item.created`
        arrives asynchronously, without a race between deciding to prune and
        learning the name of the thing to prune. `kind` names the disposable
        ones; None means the item stays for the life of the session.

        `after` is `previous_item_id`, which puts the new item immediately
        after an existing one instead of at the end. Only pruning uses it, to
        land a replacement exactly where the thing it replaces was.
        """
        self._item_seq += 1
        item_id = f"{prefix}{self._item_seq:012d}"
        if kind is not None:
            self._register(Disposable(now, item_id, kind, role=role, fields=stub_fields))
        event: dict[str, Any] = {
            "type": "conversation.item.create",
            "item": {"id": item_id, "type": "message", "role": role, "content": content},
        }
        if after is not None:
            event["previous_item_id"] = after
        return event

    def _live(self, event: dict) -> dict:
        """Mark an item as belonging to the turn in flight, and pass it on.

        Nothing removes an item the current response is being asked to look at,
        not even the forced round that runs *because* the server refused that
        very response. Freeing room by deleting the question is not freeing
        room; it is asking a different question.
        """
        self._live_items.add(event["item"]["id"])
        return event

    def _register(self, entry: Disposable) -> None:
        """Take note of an item a later round may remove.

        Order matters and is creation order: `_prune_round` walks this list to
        decide what goes, and a replacement has to be able to name the item it
        follows, so the list has to be the conversation's own order.
        """
        if entry.item_id in self._by_item:
            return
        if entry.replacement is None and not entry.pending:
            entry.replacement = self._stub_text(entry)
        self._disposable.append(entry)
        self._by_item[entry.item_id] = entry

    def _stub_text(self, entry: Disposable, transcript: str | None = None) -> str | None:
        """What goes back in when this entry comes out, or None to leave a gap.

        A nudge leaves a gap on purpose: it was a sentence about the turn it
        was written for and it has been wrong ever since. Everything else says
        what was there, because a conversation that skips from a question to an
        answer with nothing in between is one the model has to guess at.
        """
        template = {
            "image": self.cfg.prune_image_stub_template,
            "heard": self.cfg.prune_speech_stub_template,
            "said": self.cfg.prune_reply_stub_template,
        }.get(entry.kind) or STUBS.get(entry.kind)
        if template is None:
            return None
        if entry.kind in STUB_FALLBACKS and not (transcript or "").strip():
            return STUB_FALLBACKS[entry.kind]
        return template.format(transcript=(transcript or "").strip(), **entry.fields)

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

        The only kind that is deleted rather than replaced, and the only one a
        round takes regardless of age: what a nudge says stopped being true the
        moment the next turn started, so there is nothing worth standing in for
        and no age at which it becomes worth keeping.
        """
        text = reason_note(reason, self.cfg)
        # How long the film has been running, when someone has told us that the
        # clock and the film are the same thing (`position_note`, off by
        # default -- a pause or a seek makes this a confident lie). It rides
        # here rather than anywhere else because this item is appended at the
        # tail of the conversation every turn: a value that changes every turn
        # costs nothing at a position nothing is cached after.
        position = self._film_position(now) if self.subtitle_cfg.position_note else None
        if position is not None:
            text += self.subtitle_cfg.position_template.format(clock=film_clock(position))
        content = [{"type": "input_text", "text": text}]
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

        The subtitle file is the exception, and the only one: it is not history,
        it is reference material the session cannot work without, and a film is
        very likely to outlive the API's 60-minute session cap. It goes back in
        at the same position as before -- ahead of everything -- so the new
        session's prefix is cacheable from the first turn.
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
            await self._seed_subtitles()
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
        # them on the new one would earn an error for each. The stubs go with
        # them: they were history, and history does not come back.
        self._disposable.clear()
        self._by_item.clear()
        self._live_items.clear()
        self._audio_item_id = None
        self._retry_pending = False
        self._context_full_retries = 0
        self._prune_forced = False
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
            item_id = (event.get("item") or {}).get("id")
            self._audio_item_id = item_id or self._audio_item_id
            if item_id and not self.cfg.text_output:
                # The agent's own spoken turn. Registered as removable and as
                # pending: what replaces it is its transcript, and that has not
                # arrived yet. A *text* session's assistant item is already the
                # words themselves, so there is nothing there to replace.
                self._live_items.add(item_id)
                self._register(
                    Disposable(now, item_id, "said", role="assistant", pending=True)
                )
        elif kind == "response.output_audio_transcript.done":
            said = event.get("transcript") or ""
            self._add_transcript(said)
            self._settle(event.get("item_id") or self._audio_item_id, said)
            self._emit(Line(now, "say", said))
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
            said = event.get("text") or ""
            if said:
                self._add_transcript(said)
                self._emit(Line(now, "say", said))
                self.hud.show(said)
        elif kind == "input_audio_buffer.committed":
            # The commit we sent has become a user message item, and this is
            # the only place its id is ever said. Without it the player's
            # utterance is unremovable for the life of the session -- which it
            # was, and which made audio a permanent floor under the request.
            item_id = event.get("item_id")
            if item_id:
                self._register(
                    Disposable(
                        now, item_id, "heard", role="user", pending=self.cfg.transcribe_player
                    )
                )
        elif kind == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript") or ""
            self._settle(event.get("item_id"), transcript)
            self._emit(Line(now, "heard", transcript))
        elif kind == "conversation.item.input_audio_transcription.failed":
            # Nothing to show, but the entry waiting on this transcript has to
            # stop waiting or it pins the audio in the conversation forever.
            self._settle(event.get("item_id"), "")
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
            if _is_context_full(err, self.cfg.context_full_codes):
                await self._on_context_full(now, err)
                return
            self._emit(Line(now, "error", str(err.get("message") or err)))
            log.error("realtime error: %s", err)

    def _add_transcript(self, said: str) -> None:
        """Fold one output part into the transcript of the response in flight.

        A response can produce more than one text (or transcript) part: on
        sess_movie3 one of thirty-two came back as a short preamble followed by
        the remark, two `response.output_text.done` events a hundredth of a
        second apart. Both were shown and both were logged, but this field was
        overwritten rather than appended, so the `agent.response` in
        `events.jsonl` recorded only the second half of what the viewer read.

        Joining also makes the reading-time hold in `_finish_response` cover
        everything that went on screen, which is what the quiet floor is
        supposed to be measured from.
        """
        if not said:
            return
        self._response_transcript = f"{self._response_transcript} {said}".strip()

    async def _finish_response(self, event: dict) -> None:
        response = event.get("response") or {}
        self._response_usage = response.get("usage") or {}
        self.meter.add(response.get("usage"))
        detail_error = (response.get("status_details") or {}).get("error") or {}
        if response.get("status") != "completed" and _is_context_full(
            detail_error, self.cfg.context_full_codes
        ):
            # The same refusal reaches us two ways depending on when the server
            # notices. `_on_context_full` collapses them.
            await self._on_context_full(self.clock(), detail_error)
            return
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
        # A response that said nothing still has an item in the conversation
        # waiting on a transcript that is never coming.
        self._settle(self._audio_item_id, self._response_transcript)
        self._live_items.clear()
        self._record_response(now, cut=False)
        self.policy.on_response_finished(spoken_ms=spoken_ms, now=end)
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None
        # After `_record_response`, which resets the usage it reads -- so the
        # number is taken from the event rather than from the field.
        self._check_budget(now, int((response.get("usage") or {}).get("input_tokens") or 0))

    def _check_budget(self, now: float, billed: int) -> None:
        """Ask for a round when the request is getting close to the ceiling.

        With `truncation = "disabled"` nothing else is watching the size of the
        conversation, and the alternative to noticing here is noticing when a
        `response.create` comes back refused -- which costs a turn the player
        was waiting on, and arrives at the least convenient moment there is.

        The number is what the *server* billed as input, not an estimate: it
        already includes the instructions, the script and everything the
        conversation is carrying, which is the whole quantity the ceiling is
        about.
        """
        budget = self.cfg.context_budget_tokens
        if budget <= 0 or billed <= budget:
            return
        self._prune_forced = True
        self._emit(
            Line(
                now,
                "note",
                f"last request billed {billed} input tokens against a budget of "
                f"{budget}; pruning on the next tick",
                {"input_tokens": billed, "context_budget_tokens": budget},
            )
        )

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
        # No transcript event is coming for a response that was cut off, so
        # whatever it managed to say before the cut is what stands in for it.
        self._settle(self._audio_item_id, self._response_transcript)
        self._live_items.clear()
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
        """The scheduled round: is anything old enough to be worth a round?

        Two decisions, and both are about the prompt cache rather than about
        what the model needs to remember.

        **One cutoff.** Screenshots -- the current frame and the trail behind it
        -- the per-turn system nudge, and both sides of a spoken exchange all go
        on the same `prune_after_s`, so every removal the session ever wants to
        make is available at the same moment. Independent windows would mean a
        round and an invalidation each for things that were created together.

        **In bulk.** A delete truncates the cached prefix at the position of the
        deleted item, and we always remove from the oldest end, so *any* round
        costs essentially the whole prefix: the next request is billed at the
        fresh rate. That price is per round, not per item, which turns the whole
        problem into "how often", not "how much". `prune_interval_s` is the
        floor; nine items in one round cost a ninth of nine rounds of one.

        The trigger here is deliberately the *old* one -- something past the
        cutoff -- even though a round now also sweeps every stale nudge whatever
        its age. Sweeping is not triggering: if two nudges were reason enough to
        prune, a round would fire every `prune_interval_s` for the rest of the
        session and pay the prefix each time, to save thirty tokens. Rounds stay
        exactly as rare as they were; each one does strictly more.

        Why prune at all when the cost measurement says trimming loses (see
        `AgentConfig.prune_after_s`): cached tokens count against TPM in full.
        The rate limit does not care that history is cheap, only that it is
        re-sent, and on sess_movie2 that ended the session thirteen requests
        before the film did.
        """
        if self._prune_forced:
            # A response billed past the budget. Deliberately consumed here
            # rather than acted on where it was set: pruning belongs on the
            # tick, not in the middle of reading a usage block.
            self._prune_forced = False
            await self._prune_now(now, why="over budget")
            return
        if self.cfg.prune_after_s <= 0:
            return
        if now - self._last_prune < self.cfg.prune_interval_s:
            return
        # Before the trigger is evaluated, not just before the sweep: an entry
        # still waiting on a transcript is not prunable, so a deadline that
        # expired since the last round has to be able to start one.
        self._expire_pending(now)
        cutoff = now - self.cfg.prune_after_s
        if not any(
            entry.t < cutoff and self._prunable(entry, now, cutoff=cutoff)
            for entry in self._disposable
        ):
            # Deliberately without touching `_last_prune`: the interval limits
            # how often the cache may be thrown away, so a round that threw
            # nothing away must not push the next real one further out.
            return
        await self._prune_round(
            now, cutoff=cutoff, why=f"older than {self.cfg.prune_after_s:.0f}s"
        )

    async def _prune_now(self, now: float, *, why: str) -> int:
        """A round that has to free something, and what it will spend to.

        Called when the conversation is up against the context window rather
        than merely growing: the token budget saw a response bill past
        `context_budget_tokens`, or the server refused one outright. Neither
        respects `prune_interval_s` -- the interval is a cost control, and a
        request that cannot be made costs more than a cold cache -- and neither
        respects `prune_after_s = 0`, which is a statement about routine
        spending and not a refusal to survive.

        Three rungs, stopping at the first that frees anything, because each
        one gives up more than the last:

        1. the scheduled cutoff, exactly as a timed round would take it;
        2. everything removable except the turn being answered right now --
           the conversation collapses to the script, the controller text and
           the stubs, and the next request is small;
        3. the stubs themselves, oldest first, keeping the newest
           `context_full_keep_stubs`. This is the wall: once every image and
           every utterance has already been replaced, the replacements are the
           only thing left to spend, and spending the oldest of them beats a
           session that can never speak again.

        Returns how many items went, so a caller can tell the difference
        between "made room" and "there is no room to make".
        """
        rungs: list[dict[str, Any]] = [
            {"cutoff": now - self.cfg.prune_after_s if self.cfg.prune_after_s > 0 else now},
            {"cutoff": now},
            {"cutoff": now, "allow_stubs": True},
        ]
        for rung in rungs:
            removed = await self._prune_round(now, why=why, forced=True, **rung)
            if removed:
                return removed
        return 0

    def _prunable(
        self,
        entry: Disposable,
        now: float,
        *,
        cutoff: float,
        allow_stubs: bool = False,
    ) -> bool:
        """May this entry come out, right now, on this cutoff?

        The subtitle script never appears in `_disposable` at all, which is
        structural rather than a check here: `_seed_subtitles` passes no kind.
        """
        if entry.pending:
            # No transcript yet. Deleting the audio now would take the only
            # record of what was said with it; `audio_stub_wait_s` is what
            # stops that being an indefinite reprieve.
            return False
        if self._turn_in_flight(now):
            if entry.item_id in self._live_items:
                return False
            if self._response_asked_at is not None and entry.t >= self._response_asked_at:
                # Server-named items registered mid-turn (the committed
                # utterance) only enter `_live_items` once the server names
                # them, so age against the turn in flight covers the gap.
                return False
        if entry.kind == "stub":
            return allow_stubs
        if entry.kind == "note":
            # Requirement of its own: a nudge is a sentence about the turn it
            # was written for, so every one but the current turn's has been
            # wrong since the moment the next turn started. Age does not come
            # into it, and they are identical to each other besides.
            return entry is not self._newest_note()
        return entry.t < cutoff

    def _turn_in_flight(self, now: float) -> bool:
        """Is a response still plausibly being generated for this turn?

        Bounded by `speak.response_gate_timeout_s` for the same reason the
        policy's own gates are: `_response_active` is only cleared by a
        `response.done` that may never arrive, and an absolute gate that can
        never clear turns one lost event into a conversation that can never be
        pruned again -- which, with the server no longer truncating, is a
        session that eventually cannot make a request at all.
        """
        if not self._response_active or self._response_asked_at is None:
            return False
        return now - self._response_asked_at < self.speak_cfg.response_gate_timeout_s

    def _newest_note(self) -> Disposable | None:
        for entry in reversed(self._disposable):
            if entry.kind == "note":
                return entry
        return None

    async def _prune_round(
        self,
        now: float,
        *,
        cutoff: float,
        why: str,
        allow_stubs: bool = False,
        forced: bool = False,
    ) -> int:
        """Swap the expensive half of the conversation for the cheap version.

        A round does not so much delete as re-state. A screenshot becomes a
        line saying screenshots were here; the player's utterance and the
        agent's own reply become their transcripts, which is the same thread of
        conversation at a twentieth of the tokens. Only the per-turn nudge goes
        without a replacement, because it is the one thing here that stopped
        being true when its turn ended.

        Each victim is handled in two events, in this order and never the other
        way round::

            conversation.item.create  previous_item_id=<victim>, id=gpstb...
            conversation.item.delete  item_id=<victim>

        `previous_item_id` has to name an item that still exists, so deleting
        first loses the replacement and earns an error for it. Inserting after
        the *victim* rather than after the victim's predecessor is also what
        keeps a multi-victim round in order with no bookkeeping -- and it is
        the only option for the server-named audio items, whose predecessor
        this client never learns.

        The insert is free in cache terms. A prompt cache is a prefix match, so
        the round's ceiling is already fixed by its earliest *delete*; every
        insert lands at or after that point and cannot lower it further. That
        is what makes replacing affordable at all: it costs tokens, which are
        few, and not another invalidation, which is the expensive thing.
        """
        self._expire_pending(now)
        victims = [
            entry
            for entry in self._disposable
            if self._prunable(entry, now, cutoff=cutoff, allow_stubs=allow_stubs)
        ]
        if allow_stubs and self.cfg.context_full_keep_stubs > 0:
            keep = {
                entry.item_id
                for entry in [e for e in victims if e.kind == "stub"][
                    -self.cfg.context_full_keep_stubs :
                ]
            }
            victims = [entry for entry in victims if entry.item_id not in keep]
        if not victims:
            return 0
        self._last_prune = now
        gone = {entry.item_id for entry in victims}
        self._disposable = [e for e in self._disposable if e.item_id not in gone]
        for item_id in gone:
            self._by_item.pop(item_id, None)

        replaced = 0
        for entry in victims:
            if entry.replacement:
                await self.transport.send(self._stub_item(now, entry))
                replaced += 1
            await self.transport.send(
                {"type": "conversation.item.delete", "item_id": entry.item_id}
            )
        counts = Counter(entry.kind for entry in victims)
        self.pruned.update(counts)
        self._prune_rounds += 1
        self._replaced += replaced
        if forced:
            self._forced_rounds += 1
        self._emit(
            Line(
                now,
                "prune",
                f"{why}: removed {len(victims)} items ({replaced} replaced by text) -- "
                + ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items())),
                {"pruned": dict(counts), "replaced": replaced, "forced": forced},
            )
        )
        return len(victims)

    def _stub_item(self, now: float, entry: Disposable) -> dict:
        """The few tokens that stand in for what is about to be deleted."""
        part = (
            {"type": "output_text", "text": entry.replacement}
            if entry.role == "assistant"
            # `conversation.item.create` cannot populate assistant *audio*, so
            # a spoken reply comes back as the assistant having written it.
            else {"type": "input_text", "text": entry.replacement}
        )
        return self._item(now, "gpstb", entry.role, [part], "stub", after=entry.item_id)

    def _expire_pending(self, now: float) -> None:
        """Give up waiting for a transcript that is not coming.

        `pending` is an absolute "never remove this", and a gate that can never
        clear is the quietest failure mode in this file: one transcription lost
        to a network hiccup would pin its audio in the conversation for the
        rest of the session, which is precisely the growth pruning exists to
        stop.
        """
        for entry in self._disposable:
            if entry.pending and now - entry.t > self.cfg.audio_stub_wait_s:
                entry.replacement = self._stub_text(entry)
                entry.pending = False

    def _settle(self, item_id: str | None, transcript: str) -> None:
        """Fill in the words that stand in for a piece of audio."""
        entry = self._by_item.get(item_id or "")
        if entry is None or not entry.pending:
            return
        entry.replacement = self._stub_text(entry, transcript)
        entry.pending = False

    # -- the context ceiling -----------------------------------------------

    async def _on_context_full(self, now: float, err: dict) -> None:
        """The server will not answer until the conversation is smaller.

        This is the bill for `truncation = "disabled"`, and it is the one we
        chose: the server would have trimmed from the oldest end and taken the
        script with it, silently. A refusal is at least a thing the client can
        answer.

        The answer, in order: make room, ask again, and if neither is possible
        stop asking. `context_full_retries` bounds the second -- one refused
        turn must never become a loop of prune-and-retry, which spends money on
        every lap -- and a round that freed nothing is not retried at all,
        because the request that just failed would be sent again unchanged.

        When there is nothing left to free the session reopens, which is the
        same recovery it already runs every hour when the API's 60-minute cap
        drops the socket: closing the transport ends the pump's loop, and
        `_reconnect` puts back the persona and the script. It costs the history
        -- but a session that cannot make a request has lost that anyway, and
        this way it keeps talking.
        """
        if self._retry_pending:
            return
        log.warning("conversation too long: %s", err)
        if not self._response_active:
            # Nothing is waiting on this one: the watchdog or a reconnect got
            # there first. The conversation is still too long, so still prune;
            # just do not conjure a response nobody asked for.
            await self._prune_now(now, why="conversation full")
            return
        self._retry_pending = True
        if self._context_full_retries >= self.cfg.context_full_retries:
            self._abandon_turn(now, "conversation full, out of retries")
            return
        freed = await self._prune_now(now, why="conversation full")
        if freed == 0:
            self._abandon_turn(now, "conversation full and nothing left to remove")
            self._emit(Line(now, "session", "nothing left to prune; reopening the session"))
            # Ending the socket rather than calling `_reconnect` here: this
            # runs inside the pump's own iteration, and the pump is the thing
            # that owns reconnecting. See `_pump_events`.
            await self.transport.close()
            return
        self._context_full_retries += 1
        self._retry_pending = False
        await self.transport.send({"type": "response.create"})
        self._arm_watchdog()
        self._emit(Line(now, "note", f"conversation full; freed {freed} items and asked again"))

    def _abandon_turn(self, now: float, why: str) -> None:
        """Give up on a response the server never started.

        `_cancel_response`'s bookkeeping without any of its wire traffic: there
        is nothing to cancel and nothing to truncate, because nothing was ever
        generated. Releasing the gate is the whole point -- `_speak` sets
        `_response_active` optimistically and only a `response.done` clears it,
        so a refused turn left alone goes quiet until
        `speak.response_gate_timeout_s` notices, which is 45 seconds of a
        commentator saying nothing for no visible reason.

        The player's utterance is already committed into the conversation, so
        the next turn still answers it. What is lost is a beat, not the thought.
        """
        self._accept_output = False
        self._response_active = False
        self._response_started = None
        self._settle(self._audio_item_id, self._response_transcript)
        self._live_items.clear()
        self._audio_item_id = None
        self._record_response(now, cut=True)
        self.policy.on_response_cancelled(now)
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None
        self._emit(Line(now, "note", why))

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
            "subtitles": self.subtitle_summary(),
            "responses": self.spoke,
            "by_reason": dict(self.policy.counts),
            "declined": dict(self.policy.declined),
            "frames_seen": self.context.frames_seen,
            "frames_sent": self.context.frames_sent,
            "trail_sent": self.context.trail_sent,
            # Rounds, not just items: the round is what the prompt cache is
            # billed for, so "48 items in 6 rounds" is the number to read.
            # `forced` counts the ones the budget or the server asked for, and
            # `replaced` how many of the items left a sentence behind.
            "pruned": {
                **self.pruned,
                "rounds": self._prune_rounds,
                "forced": self._forced_rounds,
                "replaced": self._replaced,
            },
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


#: substrings of `error.code` that mean the conversation itself is too big
_FULL_CODE_HINTS = ("too_long", "too_large", "context_length", "token_limit")
#: ...and, for the message fallback, one word about size and one about what is
#: oversized. Both are required: plenty of errors are about size.
_FULL_SIZE_WORDS = ("too long", "too large", "exceed", "over the limit")
_FULL_SUBJECT_WORDS = ("conversation", "context window", "context length")


def _is_context_full(err: dict, extra_codes: Sequence[str] = ()) -> bool:
    """Is this the server saying the conversation will not fit?

    Disabling truncation buys a conversation the server will not trim, and the
    documented price is "an error will be returned if the Conversation is too
    long to create a Response". The *code* for that error is not documented
    anywhere, so this matches on code first, falls back to the message, and
    takes `agent.context_full_codes` from the user for the day the real one
    turns out to be something else.

    The match is deliberately narrow at the edges rather than generous. A false
    negative is a mute session, which is bad; a false positive is a full
    pruning round plus a duplicate `response.create`, which costs money and
    throws away context to fix a problem that was not there. `rate_limit` is
    excluded by name because it is the closest miss there is -- also a
    complaint about size, but about a *window* of requests, and it wants
    backoff rather than a smaller conversation.
    """
    code = str(err.get("code") or "").lower()
    if code and code in {c.lower() for c in extra_codes}:
        return True
    if code.startswith("rate_limit"):
        return False
    if any(hint in code for hint in _FULL_CODE_HINTS):
        return True
    message = str(err.get("message") or "").lower()
    if not any(word in message for word in _FULL_SUBJECT_WORDS):
        return False
    return any(word in message for word in _FULL_SIZE_WORDS)


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
