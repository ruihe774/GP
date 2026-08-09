"""Gamepad capture: capability-based discovery, hotplug, and discretization.

Raw evdev is a firehose — a stick push at the device's poll rate is hundreds of
events describing one intention. This module resolves events into semantic state
and aggregates that state into fixed windows, emitting a compact `summary` string
that is what Component B actually spends tokens on.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..config import GamepadConfig
from ..events import GamepadActivity, GamepadConnected, GamepadDisconnected, GamepadIdle
from . import evdev_raw as ev

log = logging.getLogger(__name__)

SIGNAL_INTENSITY = "gamepad.intensity"

# -- layout ---------------------------------------------------------------

#: Positional names follow the Linux gamepad spec; the labels are Xbox
#: convention. Note BTN_NORTH is Y and BTN_WEST is X -- easy to get backwards.
_BASE_BUTTONS: dict[int, str] = {
    ev.BTN_SOUTH: "A",
    ev.BTN_EAST: "B",
    ev.BTN_NORTH: "Y",
    ev.BTN_WEST: "X",
    ev.BTN_C: "C",
    ev.BTN_Z: "Z",
    ev.BTN_TL: "LB",
    ev.BTN_TR: "RB",
    ev.BTN_TL2: "LT",
    ev.BTN_TR2: "RT",
    ev.BTN_SELECT: "View",
    ev.BTN_START: "Menu",
    ev.BTN_MODE: "Guide",
    ev.BTN_THUMBL: "LS",
    ev.BTN_THUMBR: "RS",
    ev.KEY_RECORD: "Share",
}

#: D-pad reported as buttons rather than a hat, on some pads.
_HAPPY_DPAD = {
    ev.BTN_TRIGGER_HAPPY + 0: "DLeft",
    ev.BTN_TRIGGER_HAPPY + 1: "DRight",
    ev.BTN_TRIGGER_HAPPY + 2: "DUp",
    ev.BTN_TRIGGER_HAPPY + 3: "DDown",
}

ABS_GAS, ABS_BRAKE = 0x09, 0x0A

#: Per-vendor label overrides. The positional codes are identical; only the
#: printed face-button names differ. Keyed by vendor id, then product id or None.
_VENDOR_OVERRIDES: dict[int, dict[int | None, dict[int, str]]] = {
    # Nintendo: the bottom button is B and the right button is A.
    0x057E: {
        None: {
            ev.BTN_SOUTH: "B",
            ev.BTN_EAST: "A",
            ev.BTN_NORTH: "X",
            ev.BTN_WEST: "Y",
        }
    },
}

_DIRECTIONS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


@dataclass
class Layout:
    """Resolved control layout for one device. Probed, never assumed."""

    name: str
    buttons: dict[int, str] = field(default_factory=dict)
    left_stick: tuple[int, int] | None = None
    right_stick: tuple[int, int] | None = None
    triggers: dict[str, int] = field(default_factory=dict)  # label -> abs code
    hat: tuple[int, int] | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "buttons": sorted(self.buttons.values()),
            "left_stick": self.left_stick is not None,
            "right_stick": self.right_stick is not None,
            "analog_triggers": sorted(self.triggers),
            "hat": self.hat is not None,
        }


def resolve_layout(info: ev.DeviceInfo) -> Layout:
    """Derive the control layout from probed capabilities."""
    labels = dict(_BASE_BUTTONS)
    labels.update(_HAPPY_DPAD)
    vendor = _VENDOR_OVERRIDES.get(info.vid)
    if vendor is not None:
        labels.update(vendor.get(info.pid) or vendor.get(None) or {})

    buttons = {code: labels[code] for code in info.keys if code in labels}
    for code in info.keys:
        if code not in buttons and ev.BTN_JOYSTICK <= code < ev.BTN_GAMEPAD:
            buttons[code] = f"BTN{code - ev.BTN_JOYSTICK + 1}"

    left = (
        (ev.ABS_X, ev.ABS_Y)
        if ev.ABS_X in info.abses and ev.ABS_Y in info.abses
        else None
    )

    def unsigned(axis: int) -> bool:
        ai = info.absinfo.get(axis)
        return ai is not None and ai.minimum >= 0

    # ABS_Z/ABS_RZ are analog triggers when unsigned (xpad, hid-playstation),
    # but a second stick when signed (many DInput pads). The sign discriminates.
    triggers: dict[str, int] = {}
    if ev.ABS_Z in info.abses and ev.ABS_RZ in info.abses and unsigned(ev.ABS_Z):
        triggers = {"LT": ev.ABS_Z, "RT": ev.ABS_RZ}
    elif ABS_BRAKE in info.abses and ABS_GAS in info.abses:
        triggers = {"LT": ABS_BRAKE, "RT": ABS_GAS}

    if ev.ABS_RX in info.abses and ev.ABS_RY in info.abses:
        right = (ev.ABS_RX, ev.ABS_RY)
    elif ev.ABS_Z in info.abses and ev.ABS_RZ in info.abses and not triggers:
        right = (ev.ABS_Z, ev.ABS_RZ)
    else:
        right = None

    hat = (
        (ev.ABS_HAT0X, ev.ABS_HAT0Y)
        if ev.ABS_HAT0X in info.abses and ev.ABS_HAT0Y in info.abses
        else None
    )

    return Layout(
        name=info.name,
        buttons=buttons,
        left_stick=left,
        right_stick=right,
        triggers=triggers,
        hat=hat,
    )


def _direction(x: float, y: float) -> str:
    """8-way compass direction. evdev Y is inverted (up is negative)."""
    import math

    angle = math.degrees(math.atan2(-y, x)) % 360.0
    return _DIRECTIONS[int((angle + 22.5) % 360.0 // 45.0)]


# -- discretizer ----------------------------------------------------------


@dataclass
class _Press:
    start: float
    accounted: float  # ms already attributed to held_ms in earlier windows


class GamepadDiscretizer:
    """Folds raw events into fixed windows of semantic activity."""

    def __init__(self, device_id: str, layout: Layout, info: ev.DeviceInfo, cfg: GamepadConfig):
        self.device_id = device_id
        self.layout = layout
        self.info = info
        self.cfg = cfg

        self._down: dict[str, _Press] = {}
        self._taps: dict[str, int] = {}
        self._held_ms: dict[str, float] = {}
        self._raw_axis: dict[int, int] = {}
        self._window_start: float | None = None
        self._was_active = False
        self._active_since: float | None = None
        self._prev_discrete: tuple | None = None
        self._transitions = 0
        self._pending_flush = False
        #: intensity of the most recent window, whether or not an event was emitted
        self.last_intensity = 0.0

    # -- input ------------------------------------------------------------

    def feed(self, event: ev.InputEvent, now: float) -> None:
        if self._window_start is None:
            self._window_start = now
        if event.type == ev.EV_KEY:
            self._feed_key(event, now)
        elif event.type == ev.EV_ABS:
            self._feed_abs(event, now)

    def _feed_key(self, event: ev.InputEvent, now: float) -> None:
        name = self.layout.buttons.get(event.code)
        if name is None:
            return
        if event.value == 1:
            self._down[name] = _Press(start=now, accounted=0.0)
            self._transitions += 1
            self._pending_flush = True
        elif event.value == 0:
            press = self._down.pop(name, None)
            if press is None:
                return
            duration_ms = (now - press.start) * 1000.0
            if duration_ms <= self.cfg.tap_max_ms:
                self._taps[name] = self._taps.get(name, 0) + 1
            else:
                self._held_ms[name] = self._held_ms.get(name, 0.0) + max(
                    0.0, duration_ms - press.accounted - self.cfg.tap_max_ms
                )
            self._transitions += 1

    def _feed_abs(self, event: ev.InputEvent, now: float) -> None:
        before = self._discrete_axes()
        self._raw_axis[event.code] = event.value
        if self._discrete_axes() != before:
            self._transitions += 1
            self._pending_flush = True

    # -- derived state ----------------------------------------------------

    def _axis_norm(self, axis: int) -> float:
        info = self.info.absinfo.get(axis)
        raw = self._raw_axis.get(axis, info.value if info else 0)
        return info.normalize(raw) if info else 0.0

    def _stick(self, axes: tuple[int, int] | None) -> tuple[str, str, float]:
        """(direction, magnitude bucket, magnitude) for one stick."""
        if axes is None or self.cfg.sticks_mode == "off":
            return "idle", "idle", 0.0
        x, y = self._axis_norm(axes[0]), self._axis_norm(axes[1])
        magnitude = min(1.0, (x * x + y * y) ** 0.5)
        if magnitude < self.cfg.deadzone:
            return "idle", "idle", 0.0
        bucket = (
            "full"
            if magnitude >= self.cfg.stick_full
            else "light"
            if magnitude < self.cfg.stick_light
            else "mid"
        )
        return _direction(x, y), bucket, magnitude

    def _trigger(self, label: str) -> tuple[str, float]:
        axis = self.layout.triggers.get(label)
        if axis is None:
            # Fall back to the digital shoulder button of the same name.
            return ("full", 1.0) if label in self._down else ("idle", 0.0)
        value = self._axis_norm(axis)
        if value < self.cfg.trigger_light:
            return "idle", value
        if value >= self.cfg.trigger_full:
            return "full", value
        return "light", value

    def _dpad(self) -> str:
        if self.layout.hat is None:
            pressed = [n for n in ("DUp", "DDown", "DLeft", "DRight") if n in self._down]
            if not pressed:
                return "idle"
            x = ("DRight" in pressed) - ("DLeft" in pressed)
            y = ("DDown" in pressed) - ("DUp" in pressed)
            return _direction(float(x), float(y)) if (x or y) else "idle"
        hx, hy = self.layout.hat
        x, y = self._raw_axis.get(hx, 0), self._raw_axis.get(hy, 0)
        return _direction(float(x), float(y)) if (x or y) else "idle"

    def _discrete_axes(self) -> tuple:
        """The quantized analog state; window flush compares against this."""
        left = self._stick(self.layout.left_stick)[:2]
        right = self._stick(self.layout.right_stick)[:2]
        sticks = (left, right) if self.cfg.sticks_mode == "full" else ()
        return (
            self._trigger("LT")[0],
            self._trigger("RT")[0],
            self._dpad(),
            sticks,
        )

    # -- output -----------------------------------------------------------

    def flush(self, now: float) -> GamepadActivity | GamepadIdle | None:
        window_start = self._window_start if self._window_start is not None else now
        window_ms = max(1.0, (now - window_start) * 1000.0)

        held: dict[str, float] = dict(self._held_ms)
        for name, press in self._down.items():
            total_ms = (now - press.start) * 1000.0
            billable = max(0.0, total_ms - press.accounted - self.cfg.tap_max_ms)
            if billable > 0:
                held[name] = held.get(name, 0.0) + billable
                press.accounted += billable

        lt_bucket, lt_value = self._trigger("LT")
        rt_bucket, rt_value = self._trigger("RT")
        dpad = self._dpad()
        left_dir, left_mag, left_value = self._stick(self.layout.left_stick)
        right_dir, right_mag, right_value = self._stick(self.layout.right_stick)

        discrete = self._discrete_axes()
        analog_idle = (
            lt_bucket == "idle"
            and rt_bucket == "idle"
            and dpad == "idle"
            and left_value == 0.0
            and right_value == 0.0
        )
        has_events = bool(self._taps or held)
        changed = self._prev_discrete is not None and discrete != self._prev_discrete
        active = has_events or not analog_idle or changed

        self._prev_discrete = discrete
        intensity = self._intensity(held, lt_value, rt_value, left_value, right_value, window_ms)
        self.last_intensity = intensity
        apm = int(round(self._transitions * 60000.0 / window_ms))

        buttons = {
            name: {"taps": self._taps.get(name, 0), "held_ms": int(round(held.get(name, 0.0)))}
            for name in sorted(set(self._taps) | set(held))
        }
        summary = self._summarize(buttons, lt_bucket, rt_bucket, dpad, left_dir, left_mag, right_dir, right_mag)

        self._reset_window(now)

        if not active:
            if self._was_active:
                self._was_active = False
                active_ms = int(round((now - (self._active_since or now)) * 1000.0))
                self._active_since = None
                return GamepadIdle(device=self.device_id, active_ms=active_ms)
            return None

        if not self._was_active:
            self._was_active = True
            self._active_since = window_start

        # In "intensity" mode a bare stick push produces no describable content.
        # It still counts as activity (and still drives frame triggers via
        # last_intensity), but emitting an empty summary would cost tokens for
        # nothing, so the event itself is suppressed.
        if not buttons and not summary:
            return None

        event = GamepadActivity(
            device=self.device_id,
            window_ms=int(round(window_ms)),
            buttons=buttons,
            triggers={"LT": lt_bucket, "RT": rt_bucket},
            dpad=dpad,
            intensity=round(intensity, 3),
            apm=apm,
            summary=summary,
        )
        if self.cfg.sticks_mode == "full":
            event.sticks = {
                "left": {"dir": left_dir, "mag": left_mag},
                "right": {"dir": right_dir, "mag": right_mag},
            }
        return event

    def _intensity(
        self,
        held: dict[str, float],
        lt: float,
        rt: float,
        left: float,
        right: float,
        window_ms: float,
    ) -> float:
        actions = sum(self._taps.values()) + len(held)
        button_part = min(1.0, actions / 5.0)
        trigger_part = max(lt, rt)
        # Sticks feed intensity even in "intensity" mode -- that is the point of
        # keeping them: they carry activity signal without costing tokens.
        stick_part = 0.0 if self.cfg.sticks_mode == "off" else max(left, right)
        return min(1.0, 0.45 * button_part + 0.30 * trigger_part + 0.35 * stick_part)

    def _summarize(
        self,
        buttons: dict[str, dict[str, int]],
        lt: str,
        rt: str,
        dpad: str,
        left_dir: str,
        left_mag: str,
        right_dir: str,
        right_mag: str,
    ) -> str:
        parts: list[str] = []
        for label, bucket in (("LT", lt), ("RT", rt)):
            if bucket == "full":
                parts.append(f"held {label}")
            elif bucket in ("mid", "light"):
                parts.append(f"{label} {bucket}")

        held_buttons = [n for n, v in buttons.items() if v["held_ms"] > 0]
        for name in held_buttons:
            seconds = buttons[name]["held_ms"] / 1000.0
            parts.append(f"held {name} ({seconds:.1f}s)")

        if dpad != "idle":
            parts.append(f"dpad {dpad}")

        if self.cfg.sticks_mode == "full":
            for label, direction, magnitude in (
                ("left", left_dir, left_mag),
                ("right", right_dir, right_mag),
            ):
                if direction != "idle":
                    parts.append(f"{label} stick {magnitude} {direction}")

        tapped = [(n, v["taps"]) for n, v in buttons.items() if v["taps"] > 0]
        if tapped:
            rendered = ", ".join(f"{n} x{c}" if c > 1 else n for n, c in tapped)
            parts.append(f"tapped {rendered}")

        return ", ".join(parts)

    def _reset_window(self, now: float) -> None:
        self._taps = {}
        self._held_ms = {}
        self._transitions = 0
        self._window_start = now
        self._pending_flush = False

    @property
    def wants_flush(self) -> bool:
        return self._pending_flush


# -- source ---------------------------------------------------------------


class _Device:
    def __init__(self, info: ev.DeviceInfo, fd: int, discretizer: GamepadDiscretizer):
        self.info = info
        self.fd = fd
        self.discretizer = discretizer


class GamepadSource:
    """Discovers gamepads by capability, reads them, emits windowed activity."""

    name = "gamepad"

    def __init__(self, cfg: GamepadConfig):
        self.cfg = cfg
        self._ctx = None
        self._devices: dict[str, _Device] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: list[asyncio.Task] = []
        self._netlink: socket.socket | None = None
        self._rescan_handle: asyncio.TimerHandle | None = None

    async def start(self, ctx) -> None:
        self._ctx = ctx
        self._loop = asyncio.get_running_loop()
        self._rescan()
        self._start_netlink()
        self._tasks.append(asyncio.create_task(self._flush_loop(), name="gamepad-flush"))
        self._tasks.append(asyncio.create_task(self._rescan_loop(), name="gamepad-rescan"))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
        if self._netlink is not None and self._loop is not None:
            with contextlib.suppress(Exception):
                self._loop.remove_reader(self._netlink.fileno())
            self._netlink.close()
            self._netlink = None
        for device_id in list(self._devices):
            self._remove_device(device_id, notify=False)

    def describe(self) -> dict[str, Any]:
        return {
            "devices": [
                {
                    "id": device_id,
                    "name": device.info.name,
                    "path": device.info.path,
                    "vid": f"{device.info.vid:04x}",
                    "pid": f"{device.info.pid:04x}",
                    "layout": device.discretizer.layout.describe(),
                }
                for device_id, device in self._devices.items()
            ],
            "sticks_mode": self.cfg.sticks_mode,
            "window_ms": self.cfg.window_ms,
        }

    # -- discovery --------------------------------------------------------

    def _rescan(self) -> None:
        seen: set[str] = set()
        for path in ev.list_event_paths():
            info = ev.probe_device(path)
            if info is None:
                continue
            if not ev.is_gamepad(info):
                continue
            device_id = info.device_id
            seen.add(device_id)
            if device_id not in self._devices:
                self._add_device(info)
        for device_id in list(self._devices):
            if device_id not in seen:
                self._remove_device(device_id)

    def _add_device(self, info: ev.DeviceInfo) -> None:
        try:
            fd = ev.open_device(info.path)
        except OSError as exc:
            log.warning("cannot open %s (%s): %s", info.path, info.name, exc)
            return
        layout = resolve_layout(info)
        discretizer = GamepadDiscretizer(info.device_id, layout, info, self.cfg)
        device = _Device(info, fd, discretizer)
        self._devices[info.device_id] = device
        assert self._loop is not None
        self._loop.add_reader(fd, self._on_readable, info.device_id)
        log.info("gamepad connected: %s (%s)", info.name, info.path)
        self._ctx.emit(
            GamepadConnected(
                device=info.device_id,
                name=info.name,
                vid=info.vid,
                pid=info.pid,
                path=info.path,
                layout=layout.name,
                caps=layout.describe(),
            )
        )

    def _remove_device(self, device_id: str, *, notify: bool = True) -> None:
        device = self._devices.pop(device_id, None)
        if device is None:
            return
        if self._loop is not None:
            with contextlib.suppress(Exception):
                self._loop.remove_reader(device.fd)
        with contextlib.suppress(OSError):
            os.close(device.fd)
        if notify:
            log.info("gamepad disconnected: %s", device.info.name)
        if notify and self._ctx is not None:
            self._ctx.emit(
                GamepadDisconnected(device=device_id, name=device.info.name)
            )

    # -- reading ----------------------------------------------------------

    def _on_readable(self, device_id: str) -> None:
        device = self._devices.get(device_id)
        if device is None:
            return
        now = time.monotonic()
        try:
            for event in ev.read_events(device.fd):
                device.discretizer.feed(event, now)
        except OSError:
            self._remove_device(device_id)
            return
        if device.discretizer.wants_flush:
            # A transition happened; the periodic flush will pick it up. We do
            # not flush inline so a burst still coalesces into one window.
            pass

    async def _flush_loop(self) -> None:
        interval = self.cfg.window_ms / 1000.0
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            peak = 0.0
            for device in list(self._devices.values()):
                event = device.discretizer.flush(now)
                # Read intensity from the discretizer, not the event: a window
                # can be genuinely active yet emit nothing (stick-only motion).
                peak = max(peak, device.discretizer.last_intensity)
                if event is not None:
                    self._ctx.emit(event)
            if peak > 0.0:
                self._ctx.signal(SIGNAL_INTENSITY, peak)

    # -- hotplug ----------------------------------------------------------

    def _start_netlink(self) -> None:
        """Netlink is only a latency accelerator; rescan is the source of truth."""
        try:
            sock = socket.socket(
                socket.AF_NETLINK, socket.SOCK_DGRAM | socket.SOCK_CLOEXEC, 15
            )
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            sock.setblocking(False)
            sock.bind((0, 2))  # group 2 = udev, readable unprivileged
        except OSError as exc:
            log.info("netlink hotplug unavailable (%s); polling every %.1fs", exc, self.cfg.rescan_s)
            return
        self._netlink = sock
        assert self._loop is not None
        self._loop.add_reader(sock.fileno(), self._on_netlink)

    def _on_netlink(self) -> None:
        assert self._netlink is not None
        interesting = False
        while True:
            try:
                data = self._netlink.recv(8192)
            except BlockingIOError:
                break
            except OSError:
                break
            if b"input" in data:
                interesting = True
        if interesting and self._loop is not None:
            if self._rescan_handle is not None:
                self._rescan_handle.cancel()
            # Debounce: udev emits several messages per physical plug event.
            self._rescan_handle = self._loop.call_later(0.2, self._rescan)

    async def _rescan_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.rescan_s)
            self._rescan()
