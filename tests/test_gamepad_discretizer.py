"""Gamepad discretization: windows, taps vs holds, suppression, sticks modes."""

from __future__ import annotations

import pytest

from conftest import make_gamepad
from gpagent.capture import evdev_raw as ev
from gpagent.capture.gamepad import GamepadDiscretizer, resolve_layout
from gpagent.config import GamepadConfig
from gpagent.events import GamepadActivity, GamepadIdle


def event(type_: int, code: int, value: int) -> ev.InputEvent:
    return ev.InputEvent(sec=0, usec=0, type=type_, code=code, value=value)


def make(sticks_mode: str = "intensity", **kwargs) -> GamepadDiscretizer:
    info = make_gamepad()
    cfg = GamepadConfig(sticks_mode=sticks_mode, **kwargs)
    return GamepadDiscretizer("pad0", resolve_layout(info), info, cfg)


def tap(disc: GamepadDiscretizer, code: int, at: float, duration: float = 0.05) -> float:
    disc.feed(event(ev.EV_KEY, code, 1), at)
    disc.feed(event(ev.EV_KEY, code, 0), at + duration)
    return at + duration


class TestButtons:
    def test_three_taps_are_counted(self):
        disc = make()
        t = 0.0
        for _ in range(3):
            t = tap(disc, ev.BTN_SOUTH, t) + 0.02
        out = disc.flush(0.5)
        assert isinstance(out, GamepadActivity)
        assert out.buttons["A"] == {"taps": 3, "held_ms": 0}
        assert out.summary == "tapped A x3"

    def test_single_tap_has_no_multiplier(self):
        disc = make()
        tap(disc, ev.BTN_TR, 0.0)
        out = disc.flush(0.5)
        assert out.summary == "tapped RB"

    def test_press_and_release_inside_one_window_reports_duration(self):
        disc = make(tap_max_ms=220)
        disc.feed(event(ev.EV_KEY, ev.BTN_SOUTH, 1), 0.0)
        disc.feed(event(ev.EV_KEY, ev.BTN_SOUTH, 0), 0.9)
        out = disc.flush(1.0)
        assert out.buttons["A"]["taps"] == 0
        assert out.buttons["A"]["held_ms"] == pytest.approx(900, abs=5)
        # Never announced as ongoing, so "held ... for", not "released ... after".
        assert out.summary == "held A for 0.9s"

    def test_summary_orders_holds_before_taps(self):
        disc = make()
        disc.feed(event(ev.EV_ABS, ev.ABS_RZ, 1023), 0.0)
        tap(disc, ev.BTN_SOUTH, 0.1)
        out = disc.flush(0.5)
        assert out.summary.index("holding RT") < out.summary.index("tapped A")


class TestEdgeTriggeredHolds:
    """A hold is announced once when it starts and once when it ends.

    Repeating "held RT (0.5s)" every window is token cost with no information.
    """

    def test_hold_is_announced_once_then_goes_quiet(self):
        disc = make(tap_max_ms=200)
        disc.feed(event(ev.EV_KEY, ev.BTN_SOUTH, 1), 0.0)

        first = disc.flush(0.5)
        assert first.summary == "holding A"

        for at in (1.0, 1.5, 2.0, 2.5):
            assert disc.flush(at) is None, "an ongoing hold has nothing new to say"

    def test_release_reports_total_duration(self):
        disc = make(tap_max_ms=200)
        disc.feed(event(ev.EV_KEY, ev.BTN_SOUTH, 1), 0.0)
        assert disc.flush(0.5).summary == "holding A"
        assert disc.flush(1.0) is None
        disc.feed(event(ev.EV_KEY, ev.BTN_SOUTH, 0), 1.2)
        out = disc.flush(1.5)
        assert out.summary == "released A after 1.2s"
        assert out.buttons["A"]["held_ms"] == pytest.approx(1200, abs=5)

    def test_trigger_hold_is_announced_once(self):
        disc = make()
        disc.feed(event(ev.EV_ABS, ev.ABS_RZ, 1023), 0.0)
        assert disc.flush(0.5).summary == "holding RT"
        assert disc.flush(1.0) is None
        assert disc.flush(1.5) is None

    def test_trigger_release_reports_how_long_it_was_held(self):
        disc = make()
        disc.feed(event(ev.EV_ABS, ev.ABS_RZ, 1023), 0.0)
        disc.flush(0.5)   # the hold is timed from this announcement
        assert disc.flush(1.0) is None
        disc.feed(event(ev.EV_ABS, ev.ABS_RZ, 0), 1.4)
        assert disc.flush(1.5).summary == "released RT after 1.0s"

    def test_holding_field_reports_current_state(self):
        disc = make()
        disc.feed(event(ev.EV_ABS, ev.ABS_RZ, 1023), 0.0)
        disc.feed(event(ev.EV_KEY, ev.BTN_TL, 1), 0.0)
        first = disc.flush(0.5)
        assert first.holding == ["LB", "RT"]
        # A later window that has news carries the still-held state with it.
        tap(disc, ev.BTN_SOUTH, 1.0)
        later = disc.flush(1.5)
        assert later.summary == "tapped A"
        assert later.holding == ["LB", "RT"]

    def test_dpad_direction_is_edge_triggered(self):
        disc = make()
        disc.feed(event(ev.EV_ABS, ev.ABS_HAT0Y, -1), 0.0)
        assert disc.flush(0.5).summary == "dpad N"
        assert disc.flush(1.0) is None, "still N -- no news"
        disc.feed(event(ev.EV_ABS, ev.ABS_HAT0X, 1), 1.2)
        assert disc.flush(1.5).summary == "dpad NE"

    def test_intensity_stays_level_based_during_a_hold(self):
        """Frame triggers must keep seeing the hold even while the summary is quiet."""
        disc = make()
        disc.feed(event(ev.EV_ABS, ev.ABS_RZ, 1023), 0.0)
        disc.flush(0.5)
        assert disc.flush(1.0) is None
        assert disc.last_intensity > 0.0

    def test_a_held_button_keeps_the_pad_out_of_idle(self):
        disc = make(tap_max_ms=200)
        disc.feed(event(ev.EV_KEY, ev.BTN_SOUTH, 1), 0.0)
        disc.flush(0.5)
        for at in (1.0, 1.5, 2.0):
            assert disc.flush(at) is None
        disc.feed(event(ev.EV_KEY, ev.BTN_SOUTH, 0), 2.2)
        assert disc.flush(2.5).summary.startswith("released A")
        assert isinstance(disc.flush(3.0), GamepadIdle), "idle only after the release"


class TestTriggers:
    @pytest.mark.parametrize(
        "raw,expected", [(0, "idle"), (50, "idle"), (400, "light"), (1023, "full")]
    )
    def test_trigger_buckets(self, raw, expected):
        disc = make()
        disc.feed(event(ev.EV_ABS, ev.ABS_RZ, raw), 0.0)
        out = disc.flush(0.5)
        if expected == "idle":
            assert out is None or out.triggers["RT"] == "idle"
        else:
            assert out.triggers["RT"] == expected

    def test_full_trigger_reads_as_holding(self):
        disc = make()
        disc.feed(event(ev.EV_ABS, ev.ABS_RZ, 1023), 0.0)
        assert disc.flush(0.5).summary == "holding RT"


class TestDpad:
    def test_hat_maps_to_compass(self):
        disc = make()
        disc.feed(event(ev.EV_ABS, ev.ABS_HAT0Y, -1), 0.0)  # up
        out = disc.flush(0.5)
        assert out.dpad == "N"
        assert "dpad N" in out.summary

    def test_diagonal(self):
        disc = make()
        disc.feed(event(ev.EV_ABS, ev.ABS_HAT0X, 1), 0.0)
        disc.feed(event(ev.EV_ABS, ev.ABS_HAT0Y, -1), 0.0)
        assert disc.flush(0.5).dpad == "NE"


class TestSuppression:
    def test_idle_window_emits_nothing(self):
        disc = make()
        assert disc.flush(0.5) is None
        assert disc.flush(1.0) is None

    def test_idle_event_emitted_once_after_activity(self):
        disc = make()
        tap(disc, ev.BTN_SOUTH, 0.0)
        assert isinstance(disc.flush(0.5), GamepadActivity)
        assert isinstance(disc.flush(1.0), GamepadIdle)
        assert disc.flush(1.5) is None, "idle must not repeat"

    def test_held_trigger_does_not_keep_emitting(self):
        """Regression: this used to re-announce the hold twice a second."""
        disc = make()
        disc.feed(event(ev.EV_ABS, ev.ABS_RZ, 1023), 0.0)
        assert disc.flush(0.5).triggers["RT"] == "full"
        assert disc.flush(1.0) is None
        assert disc.flush(1.5) is None


class TestSticksModes:
    def _push_stick_ne(self, disc):
        disc.feed(event(ev.EV_ABS, ev.ABS_X, 32767), 0.0)
        disc.feed(event(ev.EV_ABS, ev.ABS_Y, -32768), 0.0)

    def test_full_mode_describes_direction(self):
        disc = make("full")
        self._push_stick_ne(disc)
        out = disc.flush(0.5)
        assert out.sticks["left"] == {"dir": "NE", "mag": "full"}
        assert "left stick full NE" in out.summary

    def test_intensity_mode_emits_no_event_for_stick_only_motion(self):
        disc = make("intensity")
        self._push_stick_ne(disc)
        out = disc.flush(0.5)
        assert out is None, "a bare stick push has nothing describable to say"
        assert disc.last_intensity > 0.0, "but it must still drive frame triggers"

    def test_intensity_mode_omits_sticks_from_a_real_event(self):
        disc = make("intensity")
        self._push_stick_ne(disc)
        tap(disc, ev.BTN_SOUTH, 0.1)
        out = disc.flush(0.5)
        assert out.sticks is None
        assert "stick" not in out.summary
        assert out.summary == "tapped A"

    def test_off_mode_ignores_sticks_entirely(self):
        disc = make("off")
        self._push_stick_ne(disc)
        assert disc.flush(0.5) is None
        assert disc.last_intensity == 0.0

    def test_sticks_field_absent_from_json_unless_full(self):
        disc = make("intensity")
        tap(disc, ev.BTN_SOUTH, 0.0)
        assert "sticks" not in disc.flush(0.5).to_dict()


class TestDeadzone:
    def test_inside_deadzone_is_idle(self):
        disc = make("full", deadzone=0.12)
        disc.feed(event(ev.EV_ABS, ev.ABS_X, int(32767 * 0.10)), 0.0)
        assert disc.flush(0.5) is None

    def test_outside_deadzone_registers(self):
        disc = make("full", deadzone=0.12)
        disc.feed(event(ev.EV_ABS, ev.ABS_X, int(32767 * 0.30)), 0.0)
        out = disc.flush(0.5)
        assert out is not None and out.sticks["left"]["dir"] == "E"


class TestIntensity:
    def test_intensity_rises_with_activity(self):
        quiet = make()
        tap(quiet, ev.BTN_SOUTH, 0.0)
        quiet.flush(0.5)

        busy = make()
        t = 0.0
        for code in (ev.BTN_SOUTH, ev.BTN_EAST, ev.BTN_NORTH, ev.BTN_WEST, ev.BTN_TL):
            t = tap(busy, code, t) + 0.01
        busy.feed(event(ev.EV_ABS, ev.ABS_RZ, 1023), 0.0)
        busy.flush(0.5)

        assert busy.last_intensity > quiet.last_intensity

    def test_intensity_is_bounded(self):
        disc = make("full")
        t = 0.0
        for code in (ev.BTN_SOUTH, ev.BTN_EAST, ev.BTN_NORTH, ev.BTN_WEST, ev.BTN_TL, ev.BTN_TR):
            t = tap(disc, code, t) + 0.001
        disc.feed(event(ev.EV_ABS, ev.ABS_RZ, 1023), 0.0)
        disc.feed(event(ev.EV_ABS, ev.ABS_Z, 1023), 0.0)
        disc.feed(event(ev.EV_ABS, ev.ABS_X, 32767), 0.0)
        disc.flush(0.5)
        assert 0.0 <= disc.last_intensity <= 1.0
