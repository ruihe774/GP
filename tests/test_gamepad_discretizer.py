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

    def test_long_press_is_a_hold_not_a_tap(self):
        disc = make(tap_max_ms=220)
        disc.feed(event(ev.EV_KEY, ev.BTN_SOUTH, 1), 0.0)
        disc.feed(event(ev.EV_KEY, ev.BTN_SOUTH, 0), 0.9)
        out = disc.flush(1.0)
        assert out.buttons["A"]["taps"] == 0
        assert out.buttons["A"]["held_ms"] == pytest.approx(680, abs=5)
        assert "held A" in out.summary

    def test_hold_spanning_windows_is_not_double_counted(self):
        disc = make(tap_max_ms=200)
        disc.feed(event(ev.EV_KEY, ev.BTN_SOUTH, 1), 0.0)
        first = disc.flush(1.0)
        second = disc.flush(2.0)
        # 1.0s held minus the 0.2s tap grace, then a further full second.
        assert first.buttons["A"]["held_ms"] == pytest.approx(800, abs=5)
        assert second.buttons["A"]["held_ms"] == pytest.approx(1000, abs=5)

    def test_summary_orders_holds_before_taps(self):
        disc = make()
        disc.feed(event(ev.EV_ABS, ev.ABS_RZ, 1023), 0.0)
        tap(disc, ev.BTN_SOUTH, 0.1)
        out = disc.flush(0.5)
        assert out.summary.index("held RT") < out.summary.index("tapped A")


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

    def test_full_trigger_reads_as_held(self):
        disc = make()
        disc.feed(event(ev.EV_ABS, ev.ABS_RZ, 1023), 0.0)
        assert disc.flush(0.5).summary == "held RT"


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

    def test_held_trigger_keeps_emitting(self):
        disc = make()
        disc.feed(event(ev.EV_ABS, ev.ABS_RZ, 1023), 0.0)
        assert disc.flush(0.5).triggers["RT"] == "full"
        assert disc.flush(1.0).triggers["RT"] == "full"


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
