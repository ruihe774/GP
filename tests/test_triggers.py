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
        min_interval_s=1.5,
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
        policy, clock = make()
        fired = []
        for tick in range(40):  # 20 s of unbroken activity, sampled at 2 Hz
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
        policy, clock = make()
        emit(policy, clock, scene_score=1.0)
        clock.advance(3.5)
        assert policy.decide(scene_score=0.9) == "scene"


class TestSpeech:
    def test_speech_bypasses_the_floor(self):
        policy, clock = make()
        emit(policy, clock, scene_score=1.0)
        clock.advance(0.1)
        policy.on_speech_start()
        assert policy.decide(scene_score=0.0) == "speech"

    def test_speech_outranks_gamepad(self):
        policy, clock = make()
        policy.on_speech_start()
        policy.on_intensity(0.9)
        assert policy.decide(scene_score=0.0) == "speech"

    def test_speech_can_be_disabled(self):
        policy, clock = make(on_speech=False)
        emit(policy, clock, scene_score=1.0)
        clock.advance(0.1)
        policy.on_speech_start()
        assert policy.decide(scene_score=0.0) is None


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
        policy, clock = make(scene_threshold=0.06, heartbeat_s=1000)
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
