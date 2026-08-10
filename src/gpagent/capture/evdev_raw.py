"""Minimal pure-Python evdev bindings.

`python-evdev` has no cp314 wheel and would require a build toolchain, but the
parts we need are small: a fixed 24-byte struct and three ioctls. Everything
here is capability-based so device discovery never keys off a path or a name.
"""

from __future__ import annotations

import fcntl
import glob
import os
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field

__all__ = [
    "InputEvent",
    "AbsInfo",
    "DeviceInfo",
    "EV_SYN", "EV_KEY", "EV_REL", "EV_ABS",
    "ABS_X", "ABS_Y", "ABS_Z", "ABS_RX", "ABS_RY", "ABS_RZ",
    "ABS_HAT0X", "ABS_HAT0Y",
    "list_event_paths",
    "probe_device",
    "classify",
    "is_gamepad",
    "open_device",
    "read_events",
    "EVENT_SIZE",
]

# -- event types ----------------------------------------------------------

EV_SYN, EV_KEY, EV_REL, EV_ABS = 0x00, 0x01, 0x02, 0x03

KEY_MAX, REL_MAX, ABS_MAX = 0x2FF, 0x0F, 0x3F

# -- axis codes -----------------------------------------------------------

ABS_X, ABS_Y, ABS_Z = 0x00, 0x01, 0x02
ABS_RX, ABS_RY, ABS_RZ = 0x03, 0x04, 0x05
ABS_HAT0X, ABS_HAT0Y = 0x10, 0x11
REL_X, REL_Y = 0x00, 0x01

# -- button codes ---------------------------------------------------------

BTN_MOUSE = 0x110
BTN_JOYSTICK = 0x120
BTN_GAMEPAD = 0x130
BTN_SOUTH, BTN_EAST, BTN_C, BTN_NORTH = 0x130, 0x131, 0x132, 0x133
BTN_WEST, BTN_Z, BTN_TL, BTN_TR = 0x134, 0x135, 0x136, 0x137
BTN_TL2, BTN_TR2, BTN_SELECT, BTN_START = 0x138, 0x139, 0x13A, 0x13B
BTN_MODE, BTN_THUMBL, BTN_THUMBR = 0x13C, 0x13D, 0x13E
BTN_TOOL_FINGER, BTN_TOUCH = 0x145, 0x14A
BTN_TRIGGER_HAPPY = 0x2C0
KEY_RECORD = 0x0A7

#: Reverse lookups, for diagnostics that need to show the raw kernel code
#: alongside the label we resolved it to.
KEY_CODE_NAMES: dict[int, str] = {
    BTN_MOUSE: "BTN_MOUSE",
    BTN_JOYSTICK: "BTN_JOYSTICK",
    BTN_SOUTH: "BTN_SOUTH",
    BTN_EAST: "BTN_EAST",
    BTN_C: "BTN_C",
    BTN_NORTH: "BTN_NORTH",
    BTN_WEST: "BTN_WEST",
    BTN_Z: "BTN_Z",
    BTN_TL: "BTN_TL",
    BTN_TR: "BTN_TR",
    BTN_TL2: "BTN_TL2",
    BTN_TR2: "BTN_TR2",
    BTN_SELECT: "BTN_SELECT",
    BTN_START: "BTN_START",
    BTN_MODE: "BTN_MODE",
    BTN_THUMBL: "BTN_THUMBL",
    BTN_THUMBR: "BTN_THUMBR",
    BTN_TOUCH: "BTN_TOUCH",
    BTN_TOOL_FINGER: "BTN_TOOL_FINGER",
    KEY_RECORD: "KEY_RECORD",
}

ABS_CODE_NAMES: dict[int, str] = {
    ABS_X: "ABS_X",
    ABS_Y: "ABS_Y",
    ABS_Z: "ABS_Z",
    ABS_RX: "ABS_RX",
    ABS_RY: "ABS_RY",
    ABS_RZ: "ABS_RZ",
    ABS_HAT0X: "ABS_HAT0X",
    ABS_HAT0Y: "ABS_HAT0Y",
    0x09: "ABS_GAS",
    0x0A: "ABS_BRAKE",
}


def key_name(code: int) -> str:
    if code in KEY_CODE_NAMES:
        return KEY_CODE_NAMES[code]
    if BTN_TRIGGER_HAPPY <= code < BTN_TRIGGER_HAPPY + 0x40:
        return f"BTN_TRIGGER_HAPPY{code - BTN_TRIGGER_HAPPY + 1}"
    return f"KEY_{code:#x}"


def abs_name(code: int) -> str:
    return ABS_CODE_NAMES.get(code, f"ABS_{code:#x}")


_KEY_CODES_BY_NAME = {name: code for code, name in KEY_CODE_NAMES.items()}


def key_code(name: str | int) -> int:
    """Resolve a button to its kernel code.

    Accepts what `monitor` prints -- a name like ``BTN_NORTH``, or a literal
    like ``0x133`` / ``307`` -- so a remap can be written straight from the
    output that revealed the problem.
    """
    if isinstance(name, int):
        return name
    text = name.strip()
    if not text:
        raise ValueError("empty button code")
    upper = text.upper()
    if upper in _KEY_CODES_BY_NAME:
        return _KEY_CODES_BY_NAME[upper]
    if upper.startswith("BTN_TRIGGER_HAPPY"):
        suffix = upper[len("BTN_TRIGGER_HAPPY") :]
        if suffix.isdigit() and 1 <= int(suffix) <= 0x40:
            return BTN_TRIGGER_HAPPY + int(suffix) - 1
    try:
        return int(text, 0)
    except ValueError:
        raise ValueError(
            f"unknown button {name!r}; use a name like BTN_NORTH or a code like 0x133 "
            f"(run `gpagent monitor` to see the codes your pad reports)"
        ) from None


#: struct input_event { struct timeval time; __u16 type; __u16 code; __s32 value; }
_EVENT_FMT = "llHHi"
EVENT_SIZE = struct.calcsize(_EVENT_FMT)  # 24 on 64-bit

_IOC_READ = 2


def _ioc(direction: int, type_char: str, nr: int, size: int) -> int:
    return (direction << 30) | (size << 16) | (ord(type_char) << 8) | nr


def _EVIOCGID() -> int:
    return _ioc(_IOC_READ, "E", 0x02, 8)


def _EVIOCGNAME(length: int) -> int:
    return _ioc(_IOC_READ, "E", 0x06, length)


def _EVIOCGPHYS(length: int) -> int:
    return _ioc(_IOC_READ, "E", 0x07, length)


def _EVIOCGUNIQ(length: int) -> int:
    return _ioc(_IOC_READ, "E", 0x08, length)


def _EVIOCGBIT(ev_type: int, length: int) -> int:
    return _ioc(_IOC_READ, "E", 0x20 + ev_type, length)


def _EVIOCGABS(axis: int) -> int:
    return _ioc(_IOC_READ, "E", 0x40 + axis, 24)


@dataclass(frozen=True)
class InputEvent:
    sec: int
    usec: int
    type: int
    code: int
    value: int

    @property
    def timestamp(self) -> float:
        return self.sec + self.usec / 1_000_000


@dataclass(frozen=True)
class AbsInfo:
    value: int
    minimum: int
    maximum: int
    fuzz: int
    flat: int
    resolution: int

    def normalize(self, raw: int) -> float:
        """Map a raw reading to [-1, 1] for signed axes, [0, 1] for unsigned."""
        span = self.maximum - self.minimum
        if span == 0:
            return 0.0
        if self.minimum < 0:
            # Split scaling keeps 0 at exactly 0 on asymmetric ranges.
            return max(-1.0, raw / -self.minimum) if raw < 0 else min(1.0, raw / self.maximum)
        return (raw - self.minimum) / span


@dataclass
class DeviceInfo:
    path: str
    name: str
    bus: int
    vid: int
    pid: int
    version: int
    phys: str = ""
    uniq: str = ""
    keys: frozenset[int] = field(default_factory=frozenset)
    rels: frozenset[int] = field(default_factory=frozenset)
    abses: frozenset[int] = field(default_factory=frozenset)
    absinfo: dict[int, AbsInfo] = field(default_factory=dict)

    @property
    def device_id(self) -> str:
        """Stable identity across replug; falls back to path when unavailable."""
        base = f"{self.vid:04x}:{self.pid:04x}"
        if self.uniq:
            return f"{base}:{self.uniq}"
        if self.phys:
            return f"{base}:{self.phys}"
        return f"{base}:{os.path.basename(self.path)}"


def _read_string(fd: int, request_fn, length: int = 256) -> str:
    buf = bytearray(length)
    try:
        fcntl.ioctl(fd, request_fn(length), buf)
    except OSError:
        return ""
    return buf.split(b"\x00", 1)[0].decode("utf-8", "replace")


def _read_bits(fd: int, ev_type: int, max_code: int) -> frozenset[int]:
    nbytes = (max_code + 8) // 8
    buf = bytearray(nbytes)
    try:
        fcntl.ioctl(fd, _EVIOCGBIT(ev_type, nbytes), buf)
    except OSError:
        return frozenset()
    return frozenset(i for i in range(max_code + 1) if buf[i >> 3] >> (i & 7) & 1)


def list_event_paths() -> list[str]:
    """Every /dev/input/event* on the system, in stable numeric order."""

    def index(path: str) -> int:
        digits = "".join(c for c in os.path.basename(path) if c.isdigit())
        return int(digits) if digits else 0

    return sorted(glob.glob("/dev/input/event*"), key=index)


def probe_device(path: str) -> DeviceInfo | None:
    """Read a device's identity and capabilities. None if it can't be opened."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        ident = bytearray(8)
        try:
            fcntl.ioctl(fd, _EVIOCGID(), ident)
            bus, vid, pid, version = struct.unpack("HHHH", ident)
        except OSError:
            bus = vid = pid = version = 0

        abses = _read_bits(fd, EV_ABS, ABS_MAX)
        absinfo: dict[int, AbsInfo] = {}
        for axis in abses:
            buf = bytearray(24)
            try:
                fcntl.ioctl(fd, _EVIOCGABS(axis), buf)
            except OSError:
                continue
            absinfo[axis] = AbsInfo(*struct.unpack("iiiiii", buf))

        return DeviceInfo(
            path=path,
            name=_read_string(fd, _EVIOCGNAME),
            bus=bus,
            vid=vid,
            pid=pid,
            version=version,
            phys=_read_string(fd, _EVIOCGPHYS),
            uniq=_read_string(fd, _EVIOCGUNIQ),
            keys=_read_bits(fd, EV_KEY, KEY_MAX),
            rels=_read_bits(fd, EV_REL, REL_MAX),
            abses=abses,
            absinfo=absinfo,
        )
    finally:
        os.close(fd)


def classify(info: DeviceInfo) -> str:
    """Classify a device by capability, mirroring udev's input_id heuristic."""
    has_xy = ABS_X in info.abses and ABS_Y in info.abses
    gamepad_buttons = any(
        BTN_JOYSTICK <= key < BTN_JOYSTICK + 0x20 for key in info.keys
    ) or any(BTN_TRIGGER_HAPPY <= key < BTN_TRIGGER_HAPPY + 0x40 for key in info.keys)
    pointer_like = BTN_MOUSE in info.keys and REL_X in info.rels and REL_Y in info.rels
    touch_like = BTN_TOOL_FINGER in info.keys or BTN_TOUCH in info.keys

    if has_xy and gamepad_buttons and not pointer_like and not touch_like:
        return "gamepad"
    if touch_like and has_xy:
        return "touchpad"
    if pointer_like:
        return "mouse"
    if any(key < BTN_MOUSE for key in info.keys):
        return "keyboard"
    return "other"


def is_gamepad(info: DeviceInfo) -> bool:
    return classify(info) == "gamepad"


def find_gamepads() -> list[DeviceInfo]:
    """All readable gamepads. Unreadable nodes are skipped, not fatal."""
    found = []
    for path in list_event_paths():
        info = probe_device(path)
        if info is not None and is_gamepad(info):
            found.append(info)
    return found


def open_device(path: str) -> int:
    """Open an event node non-blocking, suitable for loop.add_reader."""
    return os.open(path, os.O_RDONLY | os.O_NONBLOCK)


def read_events(fd: int) -> Iterator[InputEvent]:
    """Drain all currently-available events. Stops cleanly at EAGAIN."""
    while True:
        try:
            chunk = os.read(fd, EVENT_SIZE * 64)
        except BlockingIOError:
            return
        except OSError:
            return
        if not chunk:
            return
        for offset in range(0, len(chunk) - EVENT_SIZE + 1, EVENT_SIZE):
            yield InputEvent(*struct.unpack_from(_EVENT_FMT, chunk, offset))
        if len(chunk) < EVENT_SIZE * 64:
            return
