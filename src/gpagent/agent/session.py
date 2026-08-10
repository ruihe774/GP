"""The commentary agent: capture events in, speech out.

`CommentaryAgent.handle()` takes an explicit `now`, defaulting to an injected
clock. That one signature is what makes the whole component testable: unit tests
call it synchronously with made-up times, `drive_from_session` calls it with the
timestamp recorded in the file (no sleeping, no wall clock, deterministic), and
`drive_from_bus` lets it default to `time.monotonic` for live capture. Nothing
below this layer knows which of the three it is running under.

Conversation shape, per response:

    conversation.item.create   one user item: accumulated controller summaries,
                               plus at most one screenshot (see context.py)
    input_audio_buffer.append  the player's utterance, PCM16 24 kHz mono --
    input_audio_buffer.commit  exactly what capture produced, no conversion
    response.create            with a per-reason instruction override

Only `reply` sends audio. `react` and `ambient` are the same shape without it.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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
from .persona import instructions, instructions_for
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
        *,
        clock: Callable[[], float] = time.monotonic,
        log_path: str | Path | None = None,
        on_line: Callable[[Line], None] | None = None,
        recorder: Callable[[Event], None] | None = None,
    ):
        self.cfg = cfg.agent
        self.speak_cfg = cfg.speak
        self.clock = clock
        self.transport = transport
        self.player = player if player is not None else NullPlayer(clock)
        self.policy = SpeakPolicy(cfg.speak, clock)
        self.context = ContextBuffer(cfg.agent)
        self.meter = UsageMeter(model=cfg.agent.model)
        self.lines: list[Line] = []
        self._on_line = on_line
        self._log_path = Path(log_path) if log_path else None
        self._log_fh = None

        #: unanswered utterances, oldest first: (t, pcm, dur_ms)
        self._pending: list[tuple[float, bytes, int]] = []
        self._response_active = False
        self._response_started: float | None = None
        self._audio_item_id: str | None = None
        #: False between cancelling a response and starting the next one, so
        #: in-flight deltas for the cancelled one are not played
        self._accept_audio = True
        self._items: list[tuple[float, str]] = []
        self._image_items: list[str] = []
        self._item_seq = 0
        self._last_prune = 0.0
        #: age of each frame at the moment it was attached, for tuning capture
        self._frame_ages: list[float] = []

        # -- recording ----------------------------------------------------
        self._recorder = recorder
        #: clock() - event.t, so agent events land on the capture timeline.
        #: Live these are different bases (bus time vs time.monotonic); in
        #: deterministic replay they are the same and this is zero.
        self._t_offset: float | None = None
        #: agent events get their own seq range, well clear of capture's
        self._agent_seq = 1_000_000
        self._response_audio = bytearray()
        self._response_reason = ""
        self._response_transcript = ""
        self._response_asked_at: float | None = None
        self._response_first_audio: float | None = None
        self._response_usage: dict[str, Any] = {}
        self._pump: asyncio.Task | None = None
        self._watchdog: asyncio.Task | None = None
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
        if self._watchdog is not None:
            self._watchdog.cancel()
        if self._pump is not None:
            self._pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pump
            self._pump = None
        await self.transport.close()
        await self.player.stop()
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
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": ["audio"],
                "instructions": instructions(self.cfg),
                "truncation": {
                    "type": "retention_ratio",
                    "retention_ratio": self.cfg.truncation_retention_ratio,
                },
                "audio": {
                    "input": audio_input,
                    "output": {
                        # `rate` is required here even though the SDK's TypedDict
                        # marks it optional and the docs example omits it: the
                        # server rejects the session without it.
                        "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                        "voice": self.cfg.voice,
                    },
                },
            },
        }

    # -- capture events ----------------------------------------------------

    async def handle(self, event: Event, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        if self._t_offset is None:
            self._t_offset = now - event.t
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
        if self._speaking(now):
            await self._cancel_response(now, why="barge-in")

    def _speaking(self, now: float) -> bool:
        """Is the agent's voice coming out of the speakers right now?

        Not the same as "a response is in flight": offline the response
        completes instantly but the sentence it produced still occupies time,
        and that is precisely the window barge-in has to cover.
        """
        return self._response_active or self.player.timer.remaining_ms(now) > 0

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
        self._accept_audio = True
        self._audio_item_id = None
        with_image = self.cfg.image_on_unprompted or reason == "reply"
        ctx = self.context.take(now, with_image=with_image)

        detail: list[str] = []
        if ctx:
            item = self._context_item(ctx)
            await self.transport.send(item)
            if ctx.frame:
                await self._retire_old_images(item["item"]["id"])
            if ctx.text:
                detail.append(f'"{ctx.text}"')
            if ctx.frame:
                # Frame age is the whole point of the capture-side trigger
                # interval now that at most one frame is sent per response:
                # it buys freshness, not tokens.
                age = now - ctx.frame.t
                self._frame_ages.append(age)
                detail.append(
                    f"frame {ctx.frame.w}x{ctx.frame.h} {self.cfg.image_detail} "
                    f"({age:.1f}s old)"
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

        await self.transport.send(
            {
                "type": "response.create",
                "response": {
                    "instructions": instructions_for(self.cfg, reason),
                    "max_output_tokens": self.cfg.max_output_tokens,
                },
            }
        )
        self.policy.mark_spoken(reason, now)
        self._response_active = True
        self._response_started = now
        self._response_reason = reason
        self._response_asked_at = now
        self.spoke += 1
        self._arm_watchdog()

        cause = self.policy.react_cause if reason == "react" else ""
        self._emit(
            Line(
                now,
                "ask",
                f"-> {reason}" + (f" ({cause})" if cause else ""),
                {"sent": detail, "state": self.policy.state(now)},
            )
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

    def _context_item(self, ctx) -> dict:
        content: list[dict] = []
        if ctx.text:
            content.append({"type": "input_text", "text": ctx.text})
        if ctx.frame:
            b64 = base64.b64encode(ctx.frame.data).decode("ascii")
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{b64}",
                    "detail": self.cfg.image_detail,
                }
            )
        # Client-assigned so the item can be pruned without waiting for the
        # server to tell us what it called it.
        self._item_seq += 1
        return {
            "type": "conversation.item.create",
            "item": {
                "id": f"gpctx{self._item_seq:012d}",
                "type": "message",
                "role": "user",
                "content": content,
            },
        }

    async def _retire_old_images(self, newest_id: str) -> None:
        """Keep only the last few screenshots in the conversation.

        The whole conversation is re-billed as input on every turn, so a
        screenshot left in context is not paid for once but once per response
        for as long as it stays. Two minutes of stale frames is the most
        expensive thing the agent can carry, and the oldest of them is also the
        least useful.
        """
        if self.cfg.keep_images <= 0:
            return
        self._image_items.append(newest_id)
        while len(self._image_items) > self.cfg.keep_images:
            stale = self._image_items.pop(0)
            await self.transport.send(
                {"type": "conversation.item.delete", "item_id": stale}
            )
            self._items = [(t, i) for t, i in self._items if i != stale]

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
        try:
            async for event in self.transport:
                try:
                    await self.on_server_event(event)
                except Exception:
                    log.exception("failed handling %s", event.get("type"))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("realtime connection dropped")

    async def on_server_event(self, event: dict) -> None:
        kind = event.get("type")
        now = self.clock()

        if kind == "response.output_audio.delta":
            if not self._accept_audio:
                # Deltas already in flight when we cancelled. Playing them would
                # emit a fragment of the sentence we just cut off, a second
                # after cutting it off.
                return
            self._audio_item_id = event.get("item_id") or self._audio_item_id
            delta = event.get("delta")
            if delta:
                pcm = base64.b64decode(delta)
                if self._response_first_audio is None:
                    self._response_first_audio = now
                if self._recorder is not None:
                    self._response_audio += pcm
                self.player.push(pcm)
        elif kind == "response.output_item.added":
            self._audio_item_id = (event.get("item") or {}).get("id") or self._audio_item_id
        elif kind == "conversation.item.created":
            item_id = (event.get("item") or {}).get("id")
            if item_id:
                self._items.append((now, item_id))
        elif kind == "response.output_audio_transcript.done":
            self._response_transcript = event.get("transcript") or ""
            self._emit(Line(now, "say", self._response_transcript))
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
        if spoken_ms <= 0:
            # Dry runs and --no-playback never push audio; take the duration the
            # API reported instead, so the cooldown still reflects a real pause.
            details = (response.get("usage") or {}).get("output_token_details") or {}
            spoken_ms = int(details.get("audio_tokens") or 0) * 50.0
        now = self.clock()
        end = now
        if not self.player.realtime:
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
        self._accept_audio = False
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
            int((self._response_first_audio - self._response_asked_at) * 1000)
            if self._response_first_audio is not None
            else 0
        )
        event = AgentResponse(
            data=pcm or None,
            reason=self._response_reason,
            transcript=self._response_transcript,
            dur_ms=int(heard_ms) if cut and heard_ms else dur_ms,
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
        self._response_first_audio = None
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
        """Delete conversation items older than the retention window.

        The server's `truncation` setting is the backstop; this is the cheap
        deterministic half, and it matters most for player audio items, which
        are by far the heaviest things in the conversation.
        """
        if self.cfg.prune_after_s <= 0 or now - self._last_prune < 30.0:
            return
        self._last_prune = now
        cutoff = now - self.cfg.prune_after_s
        stale = [(t, i) for t, i in self._items if t < cutoff]
        if not stale:
            return
        self._items = [(t, i) for t, i in self._items if t >= cutoff]
        for _, item_id in stale:
            await self.transport.send({"type": "conversation.item.delete", "item_id": item_id})
        log.debug("pruned %d conversation items", len(stale))

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
            "responses": self.spoke,
            "by_reason": dict(self.policy.counts),
            "declined": dict(self.policy.declined),
            "frames_seen": self.context.frames_seen,
            "frames_sent": self.context.frames_sent,
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
