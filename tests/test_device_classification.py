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
from gpagent.capture.gamepad import parse_button_map, resolve_layout
from gpagent.config import GamepadConfig


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


class TestButtonCodeParsing:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("BTN_NORTH", ev.BTN_NORTH),
            ("btn_north", ev.BTN_NORTH),
            ("  BTN_WEST  ", ev.BTN_WEST),
            ("0x133", ev.BTN_NORTH),
            ("307", ev.BTN_NORTH),
            ("BTN_TRIGGER_HAPPY1", ev.BTN_TRIGGER_HAPPY),
            ("BTN_TRIGGER_HAPPY4", ev.BTN_TRIGGER_HAPPY + 3),
        ],
    )
    def test_accepts_names_and_literals(self, text, expected):
        assert ev.key_code(text) == expected

    def test_round_trips_with_key_name(self):
        assert ev.key_code(ev.key_name(ev.BTN_THUMBR)) == ev.BTN_THUMBR

    @pytest.mark.parametrize("bad", ["BTN_NOPE", "", "   ", "not-a-code"])
    def test_rejects_nonsense(self, bad):
        with pytest.raises(ValueError):
            ev.key_code(bad)


class TestRemapping:
    def test_swapping_x_and_y(self):
        """The reported case: a pad whose X and Y come out the wrong way round."""
        cfg = GamepadConfig(button_map={"BTN_NORTH": "X", "BTN_WEST": "Y"})
        layout = resolve_layout(make_gamepad(), cfg)
        assert layout.buttons[ev.BTN_NORTH] == "X"
        assert layout.buttons[ev.BTN_WEST] == "Y"
        assert layout.buttons[ev.BTN_SOUTH] == "A", "untouched buttons stay put"

    def test_hex_codes_work_too(self):
        cfg = GamepadConfig(button_map={"0x133": "X"})
        assert resolve_layout(make_gamepad(), cfg).buttons[ev.BTN_NORTH] == "X"

    def test_no_config_leaves_the_spec_mapping(self):
        assert resolve_layout(make_gamepad()).buttons[ev.BTN_NORTH] == "Y"

    def test_device_map_applies_to_matching_pad(self):
        cfg = GamepadConfig(device_button_map={"2dc8:200f": {"BTN_NORTH": "X"}})
        layout = resolve_layout(make_gamepad(vid=0x2DC8, pid=0x200F), cfg)
        assert layout.buttons[ev.BTN_NORTH] == "X"

    def test_device_map_ignores_other_pads(self):
        cfg = GamepadConfig(device_button_map={"dead:beef": {"BTN_NORTH": "X"}})
        layout = resolve_layout(make_gamepad(vid=0x2DC8, pid=0x200F), cfg)
        assert layout.buttons[ev.BTN_NORTH] == "Y"

    def test_device_map_beats_global_map(self):
        cfg = GamepadConfig(
            button_map={"BTN_NORTH": "GLOBAL"},
            device_button_map={"2dc8:200f": {"BTN_NORTH": "DEVICE"}},
        )
        layout = resolve_layout(make_gamepad(vid=0x2DC8, pid=0x200F), cfg)
        assert layout.buttons[ev.BTN_NORTH] == "DEVICE"

    def test_user_map_beats_builtin_vendor_quirk(self):
        cfg = GamepadConfig(button_map={"BTN_SOUTH": "MINE"})
        layout = resolve_layout(make_gamepad(vid=0x057E, pid=0x2009), cfg)
        assert layout.buttons[ev.BTN_SOUTH] == "MINE"

    def test_half_finished_swap_is_warned_about(self, caplog):
        cfg = GamepadConfig(button_map={"BTN_NORTH": "X"})  # BTN_WEST is still X
        with caplog.at_level("WARNING"):
            resolve_layout(make_gamepad(), cfg)
        assert "both report" in caplog.text

    def test_a_complete_swap_is_not_warned_about(self, caplog):
        cfg = GamepadConfig(button_map={"BTN_NORTH": "X", "BTN_WEST": "Y"})
        with caplog.at_level("WARNING"):
            resolve_layout(make_gamepad(), cfg)
        assert "both report" not in caplog.text

    def test_bad_button_name_is_rejected(self):
        with pytest.raises(ValueError):
            parse_button_map({"BTN_NOPE": "X"})


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
