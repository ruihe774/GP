"""Frame trigger policy, driven by a virtual clock."""

from __future__ import annotations

import pytest

from gpagent.capture.triggers import TriggerPolicy
from gpagent.config import TriggerConfig


class Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make(**kwargs):
    base = dict(
        min_interval_s=5.0,
        heartbeat_s=10.0,
        gamepad_intensity=0.35,
        gamepad_throttle_s=2.0,
        scene_threshold=0.06,
        scene_throttle_s=3.0,
        on_speech=True,
        dedup_threshold=0.01,
    )
    base.update(kwargs)
    clock = Clock()
    return TriggerPolicy(TriggerConfig(**base), clock), clock


def emit(policy, clock, **kwargs) -> str | None:
    trigger = policy.decide(scene_score=kwargs.get("scene_score", 0.0))
    if trigger is not None:
        policy.mark_emitted(trigger)
    return trigger


class TestGamepadThrottle:
    def test_fires_immediately_on_first_activity(self):
        """Regression guard: a debounce here would fire late or never.

        Gameplay activity is near-continuous, so waiting for it to stop would
        starve frames entirely. The first frame of a burst is the informative one.
        """
        policy, clock = make()
        policy.on_intensity(0.9)
        assert emit(policy, clock) == "gamepad"

    def test_sustained_activity_is_rate_limited_not_suppressed(self):
        # Global floor set below the gamepad throttle so this isolates the
        # per-trigger rule; the combined rate is covered in TestGlobalFloor.
        policy, clock = make(min_interval_s=0.5, gamepad_throttle_s=2.0, heartbeat_s=1000)
        fired = []
        for _ in range(40):  # 20 s of unbroken activity, sampled at 2 Hz
            policy.on_intensity(0.9)
            trigger = emit(policy, clock)
            if trigger:
                fired.append((clock.now, trigger))
            clock.advance(0.5)

        gamepad_times = [t for t, name in fired if name == "gamepad"]
        assert len(gamepad_times) >= 8, "throttle must keep firing during activity"
        gaps = [b - a for a, b in zip(gamepad_times, gamepad_times[1:])]
        assert all(gap >= 2.0 - 1e-9 for gap in gaps), "and must respect the interval"

    def test_below_threshold_intensity_does_not_trigger(self):
        policy, clock = make(gamepad_intensity=0.35)
        policy.on_intensity(0.2)
        assert emit(policy, clock) != "gamepad"

    def test_stale_hint_cannot_fire_late(self):
        policy, clock = make()
        policy.on_intensity(0.9)
        assert emit(policy, clock) == "gamepad"
        clock.advance(0.2)
        policy.on_intensity(0.9)
        assert policy.decide(scene_score=0.0) is None  # inside min_interval
        clock.advance(5.0)
        # The hint was consumed, not queued: no ghost frame long after the fact.
        assert policy.decide(scene_score=0.0) != "gamepad"


class TestGlobalFloor:
    def test_min_interval_blocks_a_scene_change(self):
        policy, clock = make()
        assert emit(policy, clock, scene_score=1.0) is not None
        clock.advance(0.5)
        assert policy.decide(scene_score=0.9) is None

    def test_after_the_floor_a_scene_change_fires(self):
        policy, clock = make(min_interval_s=5.0)
        emit(policy, clock, scene_score=1.0)
        clock.advance(6.0)
        assert policy.decide(scene_score=0.9) == "scene"

    def test_combined_rate_is_bounded_when_everything_fires_at_once(self):
        """The sess3 regression: three triggers taking turns produced 36
        frames/min while each individually respected its own throttle."""
        policy, clock = make(min_interval_s=5.0)
        fired = []
        for _ in range(240):  # 120 s sampled at 2 Hz, everything always active
            policy.on_intensity(1.0)
            policy.on_speech_start()
            trigger = policy.decide(scene_score=1.0)
            if trigger:
                policy.mark_emitted(trigger)
                fired.append(clock.now)
            clock.advance(0.5)

        gaps = [b - a for a, b in zip(fired, fired[1:])]
        assert all(gap >= 5.0 - 1e-9 for gap in gaps), f"floor violated: {gaps}"
        per_minute = len(fired) / 2.0
        assert per_minute <= 12.5, f"{per_minute}/min still too many"

    def test_the_shipped_defaults_are_bounded(self):
        """Guards the *default* config, not a config the test invented.

        The floor was lowered from 5.0 to 2.0 once Component B stopped billing
        per captured frame. That is a real relaxation of the rule that fixed the
        36 frames/min regression, so the shipped numbers get their own ceiling.
        """
        clock = Clock()
        policy = TriggerPolicy(TriggerConfig(), clock)
        fired = []
        for _ in range(480):  # 4 minutes at 2 Hz, everything always active
            policy.on_intensity(1.0)
            policy.on_speech_start()
            trigger = policy.decide(scene_score=1.0)
            if trigger:
                policy.mark_emitted(trigger)
                fired.append(clock.now)
            clock.advance(0.5)

        assert fired, "the defaults must still capture something"
        gaps = [b - a for a, b in zip(fired, fired[1:])]
        assert all(gap >= 2.0 - 1e-9 for gap in gaps), f"floor violated: {gaps}"
        per_minute = len(fired) / 4.0
        assert per_minute <= 30.0, f"{per_minute}/min is more than the floor allows"

    def test_floor_applies_across_different_trigger_kinds(self):
        policy, clock = make(min_interval_s=5.0)
        policy.on_intensity(1.0)
        assert emit(policy, clock) == "gamepad"
        clock.advance(1.0)
        # A different kind of trigger must not slip under the shared floor.
        policy.on_speech_start()
        assert policy.decide(scene_score=1.0) is None


class TestSpeech:
    def test_speech_obeys_the_global_floor(self):
        """Speech used to bypass the floor, which let frames land 0.5 s apart."""
        policy, clock = make(min_interval_s=5.0)
        emit(policy, clock, scene_score=1.0)
        clock.advance(0.5)
        policy.on_speech_start()
        assert policy.decide(scene_score=0.0) is None

    def test_speech_fires_once_the_floor_clears(self):
        policy, clock = make(min_interval_s=5.0)
        emit(policy, clock, scene_score=1.0)
        clock.advance(6.0)
        policy.on_speech_start()
        assert policy.decide(scene_score=0.0) == "speech"

    def test_speech_outranks_gamepad(self):
        policy, clock = make()
        policy.on_speech_start()
        policy.on_intensity(0.9)
        assert policy.decide(scene_score=0.0) == "speech"

    def test_speech_can_be_disabled(self):
        policy, clock = make(on_speech=False)
        clock.advance(100)
        policy.on_speech_start()
        assert policy.decide(scene_score=0.0) != "speech"


class TestHeartbeat:
    def test_heartbeat_fires_on_a_quiet_session(self):
        policy, clock = make(heartbeat_s=10.0)
        emit(policy, clock, scene_score=1.0)
        clock.advance(9.0)
        assert policy.decide(scene_score=0.0) is None
        clock.advance(1.5)
        assert policy.decide(scene_score=0.0) == "heartbeat"

    def test_heartbeat_spacing_over_a_long_idle(self):
        policy, clock = make(heartbeat_s=10.0)
        emit(policy, clock, scene_score=1.0)
        times = []
        for _ in range(120):  # 60 s at 2 Hz
            clock.advance(0.5)
            if emit(policy, clock) == "heartbeat":
                times.append(clock.now)
        gaps = [b - a for a, b in zip(times, times[1:])]
        assert times, "an idle session must still send periodic frames"
        assert all(gap >= 10.0 - 1e-9 for gap in gaps)


class TestSceneChange:
    def test_below_threshold_does_not_fire(self):
        policy, clock = make(scene_threshold=0.06, heartbeat_s=1000)
        emit(policy, clock, scene_score=1.0)
        clock.advance(5.0)
        assert policy.decide(scene_score=0.02) is None

    def test_scene_throttle_applies(self):
        policy, clock = make(
            min_interval_s=0.5, scene_threshold=0.06, scene_throttle_s=3.0, heartbeat_s=1000
        )
        emit(policy, clock, scene_score=1.0)
        clock.advance(2.0)
        assert policy.decide(scene_score=0.5) is None  # inside scene throttle
        clock.advance(2.0)
        assert policy.decide(scene_score=0.5) == "scene"


def test_counts_are_tracked():
    policy, clock = make()
    policy.on_intensity(0.9)
    emit(policy, clock)
    clock.advance(20)
    emit(policy, clock)
    assert policy.counts["gamepad"] == 1
    assert sum(policy.counts.values()) == 2
