"""Screen-frame trigger policy.

Pure logic with an injectable clock so the timing rules are testable without
hardware or real waiting.

Two rules stack. `min_interval_s` is a global floor binding every trigger, and
each trigger additionally throttles its own kind. The global rule is what keeps
the *combined* rate sane: three triggers each obeying only their own throttle
will happily take turns and emit far more often than any of them permits.

The gamepad rule is a *leading-edge throttle*, not a debounce: gameplay activity
is near-continuous, so a debounce would either fire late or starve frames
entirely. The first frame of a burst is the informative one.
"""

from __future__ import annotations

import time
from typing import Callable

from ..config import TriggerConfig

__all__ = ["TriggerPolicy"]


class TriggerPolicy:
    def __init__(self, cfg: TriggerConfig, clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.clock = clock
        self._last_emit: float | None = None
        self._last_gamepad: float | None = None
        self._last_scene: float | None = None
        self._speech_pending = False
        self._gamepad_pending = False
        self.counts: dict[str, int] = {}

    # -- inputs from other sources ----------------------------------------

    def on_intensity(self, intensity: float) -> None:
        if intensity >= self.cfg.gamepad_intensity:
            self._gamepad_pending = True

    def on_speech_start(self, _payload=None) -> None:
        self._speech_pending = True

    # -- decision ----------------------------------------------------------

    def decide(self, now: float | None = None, scene_score: float = 0.0) -> str | None:
        """Which trigger, if any, wants a frame right now."""
        now = self.clock() if now is None else now

        # The global floor is checked first and binds every trigger, speech
        # included. Per-trigger throttles govern only their own kind, so
        # anything exempted here raises the combined rate for free.
        if self._last_emit is not None and now - self._last_emit < self.cfg.min_interval_s:
            # Consume pending hints rather than queueing them, so a burst
            # cannot produce a ghost frame long after the fact.
            self._speech_pending = False
            self._gamepad_pending = False
            return None

        if self._speech_pending:
            self._speech_pending = False
            if self.cfg.on_speech:
                return "speech"

        if self._gamepad_pending:
            self._gamepad_pending = False
            if self._last_gamepad is None or now - self._last_gamepad >= self.cfg.gamepad_throttle_s:
                return "gamepad"

        if scene_score >= self.cfg.scene_threshold and (
            self._last_scene is None or now - self._last_scene >= self.cfg.scene_throttle_s
        ):
            return "scene"

        if self._last_emit is None or now - self._last_emit >= self.cfg.heartbeat_s:
            return "heartbeat"

        return None

    def mark_emitted(self, trigger: str, now: float | None = None) -> None:
        """Record that a frame actually went out under this trigger."""
        now = self.clock() if now is None else now
        self._last_emit = now
        if trigger == "gamepad":
            self._last_gamepad = now
        elif trigger == "scene":
            self._last_scene = now
        self.counts[trigger] = self.counts.get(trigger, 0) + 1

    @property
    def last_emit(self) -> float | None:
        return self._last_emit
