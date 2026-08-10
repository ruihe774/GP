"""When the agent should talk.

Pure logic with an injectable clock, mirroring `capture/triggers.py::TriggerPolicy`,
which solved the structurally identical problem for screen frames. The shape is
the same on purpose: inputs are pushed in, `decide()` names a reason or returns
None, and the caller confirms with `mark_spoken()` only if it actually spoke.

Three reasons stack:

    reply    the player said something to us
    react    something happened -- a scene change, or a burst of input
    ambient  nothing has happened for a while and the silence is getting long

and two rules bind them:

    a global sliding-window cap (`max_per_min`) over *every* reason, and
    a quiet floor (`min_gap_s`) over the two unprompted ones.

The global cap is the important one, and it is here because of Component A's
frame triggers: three rules each obeying only their own cooldown took turns and
produced 36 frames/min when each individually allowed far fewer. Per-reason
rules compose badly. `reply` is exempt from the *floor* -- if the player asks a
question two seconds after the agent finished a sentence, answering is correct
and refusing is broken -- but nothing is exempt from the cap.

Cooldowns are adaptive. Every unprompted remark the player does not answer
lengthens them (`backoff_factor`); talking to the agent shortens them
(`engagement_boost`). An agent being ignored gets quieter on its own.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable

from .config import SpeakConfig

__all__ = ["SpeakPolicy", "REASONS"]

#: in priority order
REASONS = ("reply", "react", "ambient")

#: reasons the player did not ask for, and which the quiet floor binds
UNPROMPTED = ("react", "ambient")


class SpeakPolicy:
    def __init__(self, cfg: SpeakConfig, clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.clock = clock
        self.counts: dict[str, int] = {}
        #: why we stayed quiet, for the decision log -- the interesting half
        self.declined: dict[str, int] = {}

        # Read at construction, so "how long has it been quiet" is measured from
        # the start of the session and not from whenever the first event landed.
        # Deterministic drivers construct the policy with their clock already
        # positioned at t=0.
        self._session_start: float = clock()
        self._reply_pending = False
        self._reply_since: float | None = None
        self._react_pending = False
        self._react_cause = ""
        self._hot_windows = 0

        self._talking_since: float | None = None
        self._response_since: float | None = None
        self._last_spoken_end: float | None = None
        self._last_reason: dict[str, float] = {}
        self._recent: deque[float] = deque()

        self._last_activity: float | None = None
        self._last_player_speech: float | None = None
        self._unanswered = 0

    # -- inputs from the event stream --------------------------------------

    def on_gamepad(self, intensity: float, apm: int = 0, now: float | None = None) -> None:
        now = self._now(now)
        self._last_activity = now
        if intensity >= self.cfg.intensity_threshold:
            self._hot_windows += 1
            if self._hot_windows >= self.cfg.burst_windows:
                self._hot_windows = 0
                self._react_pending = True
                self._react_cause = f"input burst (intensity {intensity:.2f})"
        else:
            self._hot_windows = 0

    def on_scene(self, scene_score: float, now: float | None = None) -> None:
        now = self._now(now)
        self._last_activity = now
        if scene_score >= self.cfg.scene_threshold:
            self._react_pending = True
            self._react_cause = f"scene change ({scene_score:.2f})"

    def on_speech_start(self, now: float | None = None) -> None:
        """The player opened their mouth. Nothing may be said until they stop."""
        self._talking_since = self._now(now)

    def on_speech_segment(self, dur_ms: int = 0, now: float | None = None) -> None:
        """A complete utterance is ready to send."""
        now = self._now(now)
        self._talking_since = None
        self._reply_pending = True
        # Refreshed on every utterance: while they are still talking the reply
        # is still wanted, and the TTL runs from the most recent thing said.
        self._reply_since = now
        self._last_player_speech = now
        # They are engaged with us; stop backing off.
        self._unanswered = 0

    # -- inputs from our own responses -------------------------------------

    def on_response_finished(self, spoken_ms: float = 0.0, now: float | None = None) -> None:
        now = self._now(now)
        self._response_since = None
        self._last_spoken_end = now

    def on_response_cancelled(self, now: float | None = None) -> None:
        # Being interrupted is not a reason to sulk: the floor still starts
        # here, but the player is plainly engaged.
        self.on_response_finished(now=now)

    # -- decision ----------------------------------------------------------

    def decide(self, now: float | None = None) -> str | None:
        """Name the reason to speak right now, or None."""
        now = self._now(now)
        cfg = self.cfg

        # Hard gates. Both expire, because a gate that sticks is a permanently
        # mute agent and nothing else in the system would notice.
        if self._talking_since is not None:
            if now - self._talking_since <= cfg.speech_gate_timeout_s:
                return self._decline("player_talking")
            self._talking_since = None
        if self._response_since is not None:
            if now - self._response_since <= cfg.response_gate_timeout_s:
                return self._decline("response_active")
            self._response_since = None

        # The global cap, checked before any reason and binding on all of them.
        self._prune(now)
        if len(self._recent) >= cfg.max_per_min:
            self._consume_pending()
            return self._decline("rate_cap")

        if self._reply_pending:
            if now - (self._reply_since or now) <= cfg.reply_ttl_s:
                self._reply_pending = False
                return "reply"
            # Too late to be an answer. Drop it rather than replying to a
            # question the player has already moved on from.
            self._reply_pending = False
            self._decline("reply_expired")

        factor = self.cooldown_factor(now)

        if self._last_spoken_end is not None and now - self._last_spoken_end < cfg.min_gap_s * factor:
            # Consume rather than queue, so a burst during the floor cannot
            # produce a ghost remark about it long after the fact.
            self._consume_pending()
            return self._decline("quiet_floor")

        if self._react_pending:
            self._react_pending = False
            last = self._last_reason.get("react")
            if last is None or now - last >= cfg.event_cooldown_s * factor:
                return "react"
            self._decline("react_cooldown")

        if now - self._last_exchange(now) >= cfg.ambient_after_s * factor:
            if self._room_is_alive(now):
                return "ambient"
            self._decline("room_is_empty")

        return None

    def mark_spoken(self, reason: str, now: float | None = None) -> None:
        """Record that we actually opened our mouth for this reason."""
        now = self._now(now)
        self._recent.append(now)
        self._last_reason[reason] = now
        self._response_since = now
        self.counts[reason] = self.counts.get(reason, 0) + 1
        if reason in UNPROMPTED:
            self._unanswered += 1

    # -- adaptation --------------------------------------------------------

    def cooldown_factor(self, now: float | None = None) -> float:
        """Multiplier on both unprompted cooldowns.

        >1 when the agent is being ignored, <1 while the player is talking to it.
        """
        now = self._now(now)
        cfg = self.cfg
        factor = cfg.backoff_factor ** min(self._unanswered, cfg.backoff_max)
        if (
            self._last_player_speech is not None
            and now - self._last_player_speech <= cfg.engagement_window_s
        ):
            factor *= cfg.engagement_boost
        return factor

    # -- introspection -----------------------------------------------------

    @property
    def response_active(self) -> bool:
        return self._response_since is not None

    @property
    def player_talking(self) -> bool:
        return self._talking_since is not None

    @property
    def react_cause(self) -> str:
        return self._react_cause

    def state(self, now: float | None = None) -> dict[str, Any]:
        now = self._now(now)
        return {
            "spoken_last_min": len(self._recent),
            "unanswered": self._unanswered,
            "cooldown_factor": round(self.cooldown_factor(now), 2),
            "quiet_for_s": round(now - self._last_exchange(now), 1),
        }

    # -- internals ---------------------------------------------------------

    def _now(self, now: float | None) -> float:
        return self.clock() if now is None else now

    def _prune(self, now: float) -> None:
        while self._recent and now - self._recent[0] >= 60.0:
            self._recent.popleft()

    def _consume_pending(self) -> None:
        self._react_pending = False

    def _last_exchange(self, now: float) -> float:
        """When the conversation -- either side of it -- last had anything in it."""
        return max(
            t
            for t in (self._session_start, self._last_spoken_end, self._last_player_speech)
            if t is not None
        )

    def _room_is_alive(self, now: float) -> bool:
        """Is anyone still there?

        Ambient remarks exist to fill conversational silence during play, not to
        talk at a paused game in an empty room.
        """
        if not self.cfg.ambient_requires_activity:
            return True
        if self._last_activity is None:
            return False
        return now - self._last_activity <= self.cfg.ambient_idle_horizon_s

    def _decline(self, why: str) -> None:
        self.declined[why] = self.declined.get(why, 0) + 1
        return None
