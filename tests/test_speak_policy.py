"""The speaking policy, driven by a virtual clock.

Note the shape of the assertions. Component A's most expensive bug was a VAD
that returned zero for every input and passed every test, because every test
asserted a *negative* ("rejects silence"). A policy that never speaks would pass
"stays quiet while the player is talking", "respects the cooldown" and "obeys
the rate cap" perfectly, and would be completely broken. So each reason gets a
test that it **does** fire, first, before anything asserts silence.
"""

from __future__ import annotations

from gpagent.agent.config import SpeakConfig
from gpagent.agent.policy import SpeakPolicy


class Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make(**kwargs):
    base = dict(
        min_gap_s=8.0,
        max_per_min=6.0,
        scene_threshold=0.35,
        intensity_threshold=0.6,
        burst_windows=2,
        event_cooldown_s=30.0,
        ambient_after_s=75.0,
        ambient_requires_activity=True,
        ambient_idle_horizon_s=45.0,
        reply_min_gap_s=3.0,
        reply_ttl_s=8.0,
        backoff_factor=1.6,
        backoff_max=3,
        engagement_boost=0.6,
        engagement_window_s=60.0,
        speech_gate_timeout_s=31.0,
        response_gate_timeout_s=45.0,
    )
    base.update(kwargs)
    clock = Clock()
    return SpeakPolicy(SpeakConfig(**base), clock), clock


def speak(policy, clock, reason_expected=None):
    """decide + confirm + finish, i.e. one complete short utterance."""
    reason = policy.decide()
    if reason is not None:
        policy.mark_spoken(reason)
        policy.on_response_finished(spoken_ms=1500)
    return reason


def keep_alive(policy, clock, intensity=0.1):
    """A window of ordinary play, so the room does not look empty."""
    policy.on_gamepad(intensity)


# -- it speaks -------------------------------------------------------------


class TestItSpeaks:
    """The positive half. Everything below this class is worthless without it."""

    def test_answers_the_player(self):
        policy, clock = make()
        policy.on_speech_start()
        clock.advance(2.0)
        policy.on_speech_segment(dur_ms=2000)
        assert policy.decide() == "reply"

    def test_answers_shortly_after_speaking(self):
        """A question 4 s after the agent stopped talking still gets answered.

        The quiet floor exists to stop the agent chattering at itself, not to
        make it ignore the player, so it must not apply here. Only the much
        shorter `reply_min_gap_s` beat does -- see TestBurstsDoNotBecomeBursts,
        which covers a question that arrives inside that beat (deferred by a
        second, never dropped).
        """
        policy, clock = make(min_gap_s=8.0, reply_min_gap_s=3.0)
        policy.on_scene(1.0)
        assert speak(policy, clock) == "react"
        clock.advance(4.0)
        policy.on_speech_start()
        policy.on_speech_segment(dur_ms=1200)
        assert policy.decide() == "reply"

    def test_reacts_to_a_scene_change(self):
        policy, clock = make()
        policy.on_scene(0.9)
        assert policy.decide() == "react"

    def test_reacts_to_a_burst_of_input(self):
        policy, clock = make(intensity_threshold=0.6, burst_windows=2)
        policy.on_gamepad(0.8)
        clock.advance(0.5)
        policy.on_gamepad(0.9)
        assert policy.decide() == "react"

    def test_remarks_unprompted_on_a_quiet_session(self):
        policy, clock = make(ambient_after_s=75.0)
        keep_alive(policy, clock)
        clock.advance(80.0)
        keep_alive(policy, clock)
        assert policy.decide() == "ambient"

    def test_keeps_remarking_over_a_long_quiet_session(self):
        """It must stay alive, not fire once and go silent forever."""
        policy, clock = make(ambient_after_s=75.0, backoff_factor=1.0)
        said = []
        for _ in range(1200):  # 10 minutes at 2 Hz
            clock.advance(0.5)
            keep_alive(policy, clock)
            if speak(policy, clock) == "ambient":
                said.append(clock.now)
        assert len(said) >= 4, f"only {len(said)} ambient remarks in 10 minutes"
        gaps = [b - a for a, b in zip(said, said[1:])]
        assert all(gap >= 75.0 - 1e-9 for gap in gaps), gaps

    def test_a_whole_conversation_runs(self):
        """End to end: talk to it, it answers; leave it alone, it remarks."""
        policy, clock = make()
        heard = []
        for step in range(600):  # 5 minutes at 2 Hz
            clock.advance(0.5)
            keep_alive(policy, clock)
            if step in (20, 200):  # the player says something twice
                policy.on_speech_start()
                policy.on_speech_segment(dur_ms=1500)
            if step == 100:
                policy.on_scene(0.9)
            reason = speak(policy, clock)
            if reason:
                heard.append(reason)
        assert heard.count("reply") == 2
        assert "react" in heard
        assert "ambient" in heard


# -- it stays quiet --------------------------------------------------------


class TestNeverTalksOverThePlayer:
    def test_silent_while_the_player_is_talking(self):
        policy, clock = make()
        policy.on_scene(1.0)
        policy.on_speech_start()
        assert policy.decide() is None
        assert policy.declined["player_talking"] == 1

    def test_the_gate_lifts_when_the_utterance_arrives(self):
        policy, clock = make()
        policy.on_speech_start()
        clock.advance(3.0)
        assert policy.decide() is None
        policy.on_speech_segment(dur_ms=3000)
        assert policy.decide() == "reply"

    def test_a_false_start_cannot_mute_the_agent_forever(self):
        """A speech.start whose segment is dropped as too short must expire.

        Without this the agent goes permanently silent and looks merely boring.
        """
        policy, clock = make(speech_gate_timeout_s=31.0)
        policy.on_speech_start()
        clock.advance(20.0)
        assert policy.decide() is None
        clock.advance(15.0)  # past the timeout, no segment ever arrived
        policy.on_scene(1.0)
        assert policy.decide() == "react"


class TestOneAtATime:
    def test_silent_while_a_response_is_in_flight(self):
        policy, clock = make()
        policy.on_scene(1.0)
        assert policy.decide() == "react"
        policy.mark_spoken("react")
        clock.advance(1.0)
        policy.on_speech_segment(dur_ms=1000)
        assert policy.decide() is None
        assert policy.declined["response_active"] == 1

    def test_a_lost_response_cannot_mute_the_agent_forever(self):
        policy, clock = make(response_gate_timeout_s=45.0)
        policy.on_scene(1.0)
        policy.decide()
        policy.mark_spoken("react")  # response.done never arrives
        clock.advance(50.0)
        policy.on_speech_segment(dur_ms=1000)
        assert policy.decide() == "reply"

    def test_a_cancelled_response_releases_the_gate(self):
        policy, clock = make()
        policy.on_scene(1.0)
        policy.decide()
        policy.mark_spoken("react")
        clock.advance(1.0)
        policy.on_response_cancelled()
        assert not policy.response_active


class TestQuietFloor:
    def test_no_unprompted_remark_inside_the_floor(self):
        policy, clock = make(min_gap_s=8.0)
        policy.on_scene(1.0)
        assert speak(policy, clock) == "react"
        clock.advance(2.0)
        policy.on_scene(1.0)
        assert policy.decide() is None
        assert policy.declined["quiet_floor"] == 1

    def test_the_floor_binds_ambient_too(self):
        policy, clock = make(min_gap_s=30.0, ambient_after_s=10.0)
        keep_alive(policy, clock)
        clock.advance(15.0)
        assert speak(policy, clock) == "ambient"
        clock.advance(11.0)  # ambient interval has passed, the floor has not
        keep_alive(policy, clock)
        assert policy.decide() is None

    def test_a_burst_during_the_floor_is_consumed_not_queued(self):
        """No ghost remark about something that happened 20 s ago."""
        policy, clock = make(min_gap_s=8.0, event_cooldown_s=0.0)
        policy.on_scene(1.0)
        speak(policy, clock)
        clock.advance(1.0)
        policy.on_scene(1.0)  # happens during the floor
        assert policy.decide() is None
        clock.advance(20.0)
        assert policy.decide() != "react"

    def test_react_has_its_own_cooldown_above_the_floor(self):
        # backoff off, to isolate the per-reason rule from the adaptation
        policy, clock = make(
            min_gap_s=2.0, event_cooldown_s=30.0, ambient_after_s=1e6, backoff_factor=1.0
        )
        policy.on_scene(1.0)
        assert speak(policy, clock) == "react"
        clock.advance(10.0)
        policy.on_scene(1.0)
        assert policy.decide() is None
        clock.advance(25.0)
        policy.on_scene(1.0)
        assert policy.decide() == "react"


class TestEmptyRoom:
    def test_no_ambient_remark_when_nobody_is_playing(self):
        policy, clock = make(ambient_after_s=75.0)
        clock.advance(200.0)  # nothing ever touched the controller
        assert policy.decide() is None

    def test_ambient_resumes_when_the_player_comes_back(self):
        policy, clock = make(ambient_after_s=75.0)
        clock.advance(200.0)
        assert policy.decide() is None
        keep_alive(policy, clock)
        clock.advance(10.0)
        keep_alive(policy, clock)
        assert policy.decide() == "ambient"

    def test_the_check_can_be_disabled(self):
        policy, clock = make(ambient_after_s=75.0, ambient_requires_activity=False)
        clock.advance(200.0)
        assert policy.decide() == "ambient"


# -- adaptation ------------------------------------------------------------


class TestAdaptiveCooldown:
    def test_being_ignored_makes_it_quieter(self):
        policy, clock = make(event_cooldown_s=30.0, min_gap_s=1.0, backoff_factor=2.0)
        policy.on_scene(1.0)
        assert speak(policy, clock) == "react"
        clock.advance(35.0)  # would clear a 30 s cooldown, but it is now 60 s
        policy.on_scene(1.0)
        assert policy.decide() is None
        clock.advance(30.0)
        policy.on_scene(1.0)
        assert policy.decide() == "react"

    def test_backoff_is_capped(self):
        policy, clock = make(backoff_factor=2.0, backoff_max=3)
        for _ in range(10):
            policy.mark_spoken("ambient")
            policy.on_response_finished()
        assert policy.cooldown_factor() == 2.0**3

    def test_talking_to_it_makes_it_chattier(self):
        policy, clock = make(engagement_boost=0.5, engagement_window_s=60.0)
        assert policy.cooldown_factor() == 1.0
        policy.on_speech_segment(dur_ms=1000)
        assert policy.cooldown_factor() == 0.5
        clock.advance(90.0)
        assert policy.cooldown_factor() == 1.0

    def test_answering_the_player_resets_the_backoff(self):
        policy, clock = make(backoff_factor=2.0, engagement_boost=1.0)
        policy.mark_spoken("ambient")
        policy.on_response_finished()
        assert policy.cooldown_factor() == 2.0
        policy.on_speech_segment(dur_ms=1000)
        assert policy.cooldown_factor() == 1.0

    def test_replies_do_not_count_as_being_ignored(self):
        policy, clock = make(backoff_factor=2.0, engagement_boost=1.0)
        for _ in range(3):
            policy.on_speech_segment(dur_ms=1000)
            assert speak(policy, clock) == "reply"
            clock.advance(20.0)
        assert policy.cooldown_factor() == 1.0


# -- the global cap --------------------------------------------------------


class TestBurstsDoNotBecomeBursts:
    """A player who talks continuously must not pull back-to-back answers.

    `reply` is exempt from the quiet floor so a direct question always gets
    answered, which left the global per-minute cap as the only thing spacing
    replies out -- and a cap permits its whole allowance in one burst.
    """

    def test_a_reply_waits_a_beat_after_the_last_response(self):
        policy, clock = make(reply_min_gap_s=3.0)
        policy.on_speech_segment(dur_ms=800)
        assert speak(policy, clock) == "reply"
        clock.advance(1.0)
        policy.on_speech_segment(dur_ms=800)
        assert policy.decide() is None
        assert policy.declined["reply_spacing"] >= 1

    def test_it_is_answered_as_soon_as_the_beat_passes(self):
        """Held, not dropped: the question still gets an answer."""
        policy, clock = make(reply_min_gap_s=3.0)
        policy.on_speech_segment(dur_ms=800)
        speak(policy, clock)
        clock.advance(1.0)
        policy.on_speech_segment(dur_ms=800)
        assert policy.decide() is None
        clock.advance(2.5)
        assert policy.decide() == "reply"

    def test_the_beat_is_much_shorter_than_the_unprompted_floor(self):
        policy, clock = make(min_gap_s=8.0, reply_min_gap_s=3.0)
        policy.on_scene(1.0)
        assert speak(policy, clock) == "react"
        clock.advance(4.0)  # past the reply beat, well inside the quiet floor
        policy.on_speech_segment(dur_ms=800)
        assert policy.decide() == "reply"

    def test_interrupting_the_agent_skips_the_beat(self):
        """Someone who cuts you off is not asking you to pause first."""
        policy, clock = make(reply_min_gap_s=3.0)
        policy.on_scene(1.0)
        assert policy.decide() == "react"
        policy.mark_spoken("react")
        clock.advance(1.0)
        policy.on_speech_start()
        policy.on_response_cancelled()  # barge-in
        clock.advance(1.0)
        policy.on_speech_segment(dur_ms=900)
        assert policy.decide() == "reply"

    def test_continuous_talking_over_a_long_fight(self):
        """Two minutes of shouting at a boss, one utterance every 1.5 s."""
        policy, clock = make(reply_min_gap_s=3.0, max_per_min=6.0)
        said = []
        for step in range(480):  # 2 minutes at 4 Hz
            policy.on_gamepad(1.0)  # sustained heavy input, too
            if step % 6 == 0:
                policy.on_speech_segment(dur_ms=1200)
            reason = policy.decide()
            if reason:
                policy.mark_spoken(reason)
                policy.on_response_finished(spoken_ms=2500, now=clock.now + 2.5)
                said.append(clock.now)
            clock.advance(0.25)

        assert said, "it must still answer a player who is talking to it"
        gaps = [b - a for a, b in zip(said, said[1:])]
        assert min(gaps) >= 3.0, f"back-to-back replies got through: {gaps}"
        assert len(said) / 2.0 <= 6.0, "per-minute cap still holds"


class TestGlobalCap:
    def test_combined_rate_when_everything_fires_at_once(self):
        """The frame-trigger regression, ported.

        Three reasons each obeying only their own cooldown will take turns and
        exceed what any of them allows. Only a global rule bounds the total.
        """
        policy, clock = make(max_per_min=6.0)
        said = []
        for _ in range(2400):  # 10 minutes at 4 Hz, everything always pending
            policy.on_gamepad(1.0)
            policy.on_scene(1.0)
            policy.on_speech_segment(dur_ms=500)
            reason = policy.decide()
            if reason:
                policy.mark_spoken(reason)
                policy.on_response_finished(spoken_ms=1000)
                said.append(clock.now)
            clock.advance(0.25)

        assert said, "a policy that never speaks passes every rate test"
        for i, t in enumerate(said):
            window = [s for s in said[i:] if s - t < 60.0]
            assert len(window) <= 6, f"{len(window)} in the minute from {t}"
        assert len(said) / 10.0 <= 6.0

    def test_the_cap_binds_replies_too(self):
        # advance past reply_min_gap_s each time, so this isolates the cap
        policy, clock = make(max_per_min=2.0, reply_min_gap_s=3.0)
        for _ in range(2):
            policy.on_speech_segment(dur_ms=500)
            assert speak(policy, clock) == "reply"
            clock.advance(4.0)
        policy.on_speech_segment(dur_ms=500)
        assert policy.decide() is None
        assert policy.declined["rate_cap"] == 1

    def test_the_cap_slides(self):
        policy, clock = make(max_per_min=2.0, reply_min_gap_s=3.0)
        for _ in range(2):
            policy.on_speech_segment(dur_ms=500)
            speak(policy, clock)
            clock.advance(4.0)
        clock.advance(60.0)
        policy.on_speech_segment(dur_ms=500)
        assert policy.decide() == "reply"


def test_counts_are_tracked():
    policy, clock = make()
    policy.on_speech_segment(dur_ms=500)
    speak(policy, clock)
    clock.advance(40.0)
    policy.on_scene(1.0)
    speak(policy, clock)
    assert policy.counts == {"reply": 1, "react": 1}
