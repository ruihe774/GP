"""Device discovery must key off capabilities, never paths or names.

This is the portability guard: if it passes on synthetic devices, discovery
works on a machine whose hardware nobody has seen.
"""

from __future__ import annotations

import pytest

from conftest import (
    make_gamepad,
    make_joystick_flightstick,
    make_keyboard,
    make_mouse,
    make_touchpad,
)
from gpagent.capture import evdev_raw as ev
from gpagent.capture.gamepad import resolve_layout


def test_gamepad_is_classified_as_gamepad():
    assert ev.classify(make_gamepad()) == "gamepad"
    assert ev.is_gamepad(make_gamepad())


def test_flight_stick_is_a_gamepad():
    # BTN_JOYSTICK range, not BTN_GAMEPAD -- still a pad for our purposes.
    assert ev.is_gamepad(make_joystick_flightstick())


@pytest.mark.parametrize(
    "factory,expected",
    [(make_mouse, "mouse"), (make_keyboard, "keyboard"), (make_touchpad, "touchpad")],
)
def test_non_gamepads_are_rejected(factory, expected):
    info = factory()
    assert ev.classify(info) == expected
    assert not ev.is_gamepad(info)


def test_touchpad_with_xy_is_not_mistaken_for_a_pad():
    # A touchpad has ABS_X/ABS_Y and BTN_MOUSE; only BTN_TOUCH separates it.
    assert not ev.is_gamepad(make_touchpad())


def test_device_id_is_stable_and_path_independent():
    a = make_gamepad(path="/dev/input/event19")
    b = make_gamepad(path="/dev/input/event23")
    assert a.device_id == b.device_id, "replug must not change identity"


def test_device_id_distinguishes_two_identical_pads():
    a = make_gamepad()
    b = make_gamepad()
    object.__setattr__(b, "phys", "usb-test/input1")
    assert a.device_id != b.device_id


class TestCodeNames:
    """`monitor` prints these beside the resolved label to verify the mapping."""

    def test_face_buttons(self):
        assert ev.key_name(ev.BTN_SOUTH) == "BTN_SOUTH"
        assert ev.key_name(ev.BTN_NORTH) == "BTN_NORTH"
        assert ev.key_name(ev.KEY_RECORD) == "KEY_RECORD"

    def test_trigger_happy_range_is_expanded(self):
        assert ev.key_name(ev.BTN_TRIGGER_HAPPY) == "BTN_TRIGGER_HAPPY1"
        assert ev.key_name(ev.BTN_TRIGGER_HAPPY + 3) == "BTN_TRIGGER_HAPPY4"

    def test_unknown_code_falls_back_to_hex(self):
        # 0x1ff sits outside both the named table and the TRIGGER_HAPPY range.
        assert ev.key_name(0x1FF) == "KEY_0x1ff"

    def test_axis_names(self):
        assert ev.abs_name(ev.ABS_Z) == "ABS_Z"
        assert ev.abs_name(ev.ABS_HAT0X) == "ABS_HAT0X"
        assert ev.abs_name(0x0A) == "ABS_BRAKE"
        assert ev.abs_name(0x3F) == "ABS_0x3f"


class TestLayoutResolution:
    def test_xpad_layout(self):
        layout = resolve_layout(make_gamepad())
        assert layout.triggers == {"LT": ev.ABS_Z, "RT": ev.ABS_RZ}
        assert layout.left_stick == (ev.ABS_X, ev.ABS_Y)
        assert layout.right_stick == (ev.ABS_RX, ev.ABS_RY)
        assert layout.hat == (ev.ABS_HAT0X, ev.ABS_HAT0Y)

    def test_north_is_y_and_west_is_x(self):
        # The mapping everyone gets backwards.
        layout = resolve_layout(make_gamepad())
        assert layout.buttons[ev.BTN_NORTH] == "Y"
        assert layout.buttons[ev.BTN_WEST] == "X"
        assert layout.buttons[ev.BTN_SOUTH] == "A"
        assert layout.buttons[ev.BTN_EAST] == "B"

    def test_nintendo_override_swaps_face_labels(self):
        info = make_gamepad(vid=0x057E, pid=0x2009)
        layout = resolve_layout(info)
        assert layout.buttons[ev.BTN_SOUTH] == "B"
        assert layout.buttons[ev.BTN_EAST] == "A"

    def test_signed_z_axes_are_a_stick_not_triggers(self):
        """Many DInput pads use ABS_Z/RZ as the right stick; sign discriminates."""
        info = make_gamepad()
        info.absinfo[ev.ABS_Z] = ev.AbsInfo(0, -32768, 32767, 0, 0, 0)
        info.absinfo[ev.ABS_RZ] = ev.AbsInfo(0, -32768, 32767, 0, 0, 0)
        object.__setattr__(
            info, "abses", frozenset(info.abses - {ev.ABS_RX, ev.ABS_RY})
        )
        layout = resolve_layout(info)
        assert layout.triggers == {}
        assert layout.right_stick == (ev.ABS_Z, ev.ABS_RZ)
