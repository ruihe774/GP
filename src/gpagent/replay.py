"""Replay a recorded session as if it were live capture.

`ReplaySource` implements the same `CaptureSource` protocol the real sources do,
so `CaptureBus([ReplaySource("sess5")])` is indistinguishable from live capture
to anything downstream -- which is what makes Component B developable without a
game running, a controller plugged in, or a single dollar spent.

Two things are easy to get wrong and are handled here:

* The live sources publish *hints* on the signal bus (`gamepad.intensity`,
  `speech.start`) that never appear in `events.jsonl`. A replay that only
  re-emits events would silently starve any consumer that subscribes to those,
  so they are reconstructed. `speech.start` in particular fires at VAD detection
  time, i.e. *before* the segment exists, which is exactly what barge-in hangs
  off; here it is issued one hangover+preroll ahead of the segment event.

* `CaptureBus.emit` restamps `t` at emit time, so wall-clock replay only
  preserves the original timeline at `speed=1`. For deterministic tests, drive
  the agent from `read_session()` directly with `now=event.t` instead of going
  through the bus at all (see `agent.session.drive_from_session`).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from pathlib import Path
from typing import Any

from .bus import BusContext
from .events import AgentResponse, Event, GamepadActivity, SpeechSegment
from .sinks.jsonl import read_session

log = logging.getLogger(__name__)

__all__ = ["ReplaySource", "build_cues", "SIGNAL_INTENSITY", "SIGNAL_SPEECH_START"]

#: mirrors capture.screen / capture.audio, which own these names live
SIGNAL_INTENSITY = "gamepad.intensity"
SIGNAL_SPEECH_START = "speech.start"


class ReplaySource:
    name = "replay"

    def __init__(
        self,
        directory: str | Path,
        *,
        speed: float = 1.0,
        load_blobs: bool = True,
        signals: bool = True,
    ) -> None:
        self.directory = Path(directory)
        #: 0 means "as fast as possible" -- no sleeping between events
        self.speed = speed
        self.load_blobs = load_blobs
        self.signals = signals
        self.finished = asyncio.Event()
        self._ctx: BusContext | None = None
        self._task: asyncio.Task | None = None
        self._emitted = 0
        self._total = 0

    # -- lifecycle ---------------------------------------------------------

    async def start(self, ctx: BusContext) -> None:
        self._ctx = ctx
        events = list(read_session(self.directory, load_blobs=self.load_blobs))
        if not events:
            raise RuntimeError(f"no events in {self.directory}")
        self._total = len(events)
        self._task = asyncio.create_task(self._run(events), name="replay")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def describe(self) -> dict[str, Any]:
        return {
            "source": str(self.directory),
            "speed": self.speed,
            "events_total": self._total,
            "events_emitted": self._emitted,
        }

    # -- the replay loop ---------------------------------------------------

    async def _run(self, events: list[Event]) -> None:
        assert self._ctx is not None, "_run only starts after start() sets _ctx"
        cues = build_cues(events, signals=self.signals)
        start = time.monotonic()
        origin = cues[0][0]
        try:
            for when, kind, payload in cues:
                if self.speed > 0:
                    target = (when - origin) / self.speed
                    delay = target - (time.monotonic() - start)
                    if delay > 0:
                        await asyncio.sleep(delay)
                else:
                    # Yield so the consumer keeps up rather than filling the
                    # bus queue and having frames evicted as droppable.
                    await asyncio.sleep(0)
                if kind == "event":
                    self._ctx.emit(payload)
                    self._emitted += 1
                else:
                    self._ctx.signal(kind, payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("replay failed")
        finally:
            self.finished.set()


def build_cues(
    events: list[Event], *, signals: bool = True
) -> list[tuple[float, str, Any]]:
    """Interleave events with the signals the live sources would have published.

    Split out and pure so the reconstruction is testable without a bus.

    `speech.start` is placed at the *start* of the utterance (the segment event
    lands after the whole utterance plus hangover), not immediately before the
    segment. Getting this wrong would make barge-in untestable offline: the
    "player is talking" gate would open and close in the same instant, and a
    replay would never catch the agent talking over the player.

    A recording made by `commentate` also holds what the agent said, and that
    is output, not capture. Replaying it would hand the new agent the old one's
    speech: `handle()` records every event it is given, so the new session came
    out holding two interleaved conversations, and the console printed the old
    agent's lines as if they were this run's. Dropped here rather than at the
    recorder so the new agent never sees them at all.
    """
    cues: list[tuple[float, int, str, Any]] = []
    for order, event in enumerate(events):
        if isinstance(event, AgentResponse):
            continue
        cues.append((event.t, order, "event", event))
        if not signals:
            continue
        if isinstance(event, GamepadActivity):
            cues.append((event.t, order - 1, SIGNAL_INTENSITY, event.intensity))
        elif isinstance(event, SpeechSegment):
            onset = max(0.0, event.t - event.dur_ms / 1000.0)
            cues.append((onset, order - 1, SIGNAL_SPEECH_START, None))
    # `order` breaks ties so a signal stays ahead of the event it belongs to.
    cues.sort(key=lambda c: (c[0], c[1]))
    return [(when, kind, payload) for when, _, kind, payload in cues]
