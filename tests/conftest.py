"""Shared fixtures: synthetic devices so nothing here needs real hardware."""

from __future__ import annotations

import pytest

from gpagent.capture import evdev_raw as ev


def absinfo(minimum: int, maximum: int, value: int = 0) -> ev.AbsInfo:
    return ev.AbsInfo(value=value, minimum=minimum, maximum=maximum, fuzz=0, flat=0, resolution=0)


def make_gamepad(path: str = "/dev/input/event99", vid: int = 0x2DC8, pid: int = 0x200F) -> ev.DeviceInfo:
    """An xpad-style pad: two sticks, two analog triggers, a hat, 11 buttons."""
    return ev.DeviceInfo(
        path=path,
        name="Test Gamepad",
        bus=3,
        vid=vid,
        pid=pid,
        version=0x200,
        phys="usb-test/input0",
        keys=frozenset(
            {
                ev.BTN_SOUTH, ev.BTN_EAST, ev.BTN_NORTH, ev.BTN_WEST,
                ev.BTN_TL, ev.BTN_TR, ev.BTN_SELECT, ev.BTN_START,
                ev.BTN_MODE, ev.BTN_THUMBL, ev.BTN_THUMBR,
            }
        ),
        rels=frozenset(),
        abses=frozenset(
            {ev.ABS_X, ev.ABS_Y, ev.ABS_RX, ev.ABS_RY, ev.ABS_Z, ev.ABS_RZ,
             ev.ABS_HAT0X, ev.ABS_HAT0Y}
        ),
        absinfo={
            ev.ABS_X: absinfo(-32768, 32767),
            ev.ABS_Y: absinfo(-32768, 32767),
            ev.ABS_RX: absinfo(-32768, 32767),
            ev.ABS_RY: absinfo(-32768, 32767),
            ev.ABS_Z: absinfo(0, 1023),
            ev.ABS_RZ: absinfo(0, 1023),
            ev.ABS_HAT0X: absinfo(-1, 1),
            ev.ABS_HAT0Y: absinfo(-1, 1),
        },
    )


def make_mouse() -> ev.DeviceInfo:
    return ev.DeviceInfo(
        path="/dev/input/event98",
        name="Test Mouse",
        bus=3, vid=0x1532, pid=0x0099, version=0x111,
        keys=frozenset({ev.BTN_MOUSE, ev.BTN_MOUSE + 1, ev.BTN_MOUSE + 2}),
        rels=frozenset({ev.REL_X, ev.REL_Y}),
        abses=frozenset(),
    )


def make_keyboard() -> ev.DeviceInfo:
    return ev.DeviceInfo(
        path="/dev/input/event97",
        name="Test Keyboard",
        bus=3, vid=0x05AC, pid=0x024F, version=0x111,
        keys=frozenset(range(1, 120)),
        rels=frozenset(),
        abses=frozenset(),
    )


def make_touchpad() -> ev.DeviceInfo:
    return ev.DeviceInfo(
        path="/dev/input/event96",
        name="Test Touchpad",
        bus=0x18, vid=0x06CB, pid=0x1234, version=0x100,
        keys=frozenset({ev.BTN_TOUCH, ev.BTN_TOOL_FINGER, ev.BTN_MOUSE}),
        rels=frozenset(),
        abses=frozenset({ev.ABS_X, ev.ABS_Y}),
        absinfo={ev.ABS_X: absinfo(0, 1300), ev.ABS_Y: absinfo(0, 700)},
    )


def make_joystick_flightstick() -> ev.DeviceInfo:
    """A classic joystick: BTN_JOYSTICK range rather than BTN_GAMEPAD."""
    return ev.DeviceInfo(
        path="/dev/input/event95",
        name="Test Flight Stick",
        bus=3, vid=0x044F, pid=0xB10A, version=0x100,
        keys=frozenset({ev.BTN_JOYSTICK, ev.BTN_JOYSTICK + 1, ev.BTN_JOYSTICK + 2}),
        rels=frozenset(),
        abses=frozenset({ev.ABS_X, ev.ABS_Y, ev.ABS_RZ}),
        absinfo={
            ev.ABS_X: absinfo(-512, 511),
            ev.ABS_Y: absinfo(-512, 511),
            ev.ABS_RZ: absinfo(0, 255),
        },
    )


@pytest.fixture
def gamepad_info() -> ev.DeviceInfo:
    return make_gamepad()
