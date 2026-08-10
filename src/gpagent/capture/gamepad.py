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
from typing import Any

from ..bus import BusContext
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


def parse_button_map(mapping: dict[str, str] | None) -> dict[int, str]:
    """Turn a configured {code: label} map into {kernel code: label}."""
    if not mapping:
        return {}
    return {ev.key_code(code): str(label) for code, label in mapping.items()}


def resolve_layout(info: ev.DeviceInfo, cfg: GamepadConfig | None = None) -> Layout:
    """Derive the control layout from probed capabilities.

    Labels are applied in increasing order of specificity: the Linux gamepad
    spec, then built-in vendor quirks, then the user's global remap, then the
    user's per-device remap. The last word belongs to whoever was most specific.
    """
    labels = dict(_BASE_BUTTONS)
    labels.update(_HAPPY_DPAD)
    vendor = _VENDOR_OVERRIDES.get(info.vid)
    if vendor is not None:
        labels.update(vendor.get(info.pid) or vendor.get(None) or {})

    if cfg is not None:
        labels.update(parse_button_map(cfg.button_map))
        key = f"{info.vid:04x}:{info.pid:04x}"
        labels.update(parse_button_map(cfg.device_button_map.get(key)))

    buttons = {code: labels[code] for code in info.keys if code in labels}
    for code in info.keys:
        if code not in buttons and ev.BTN_JOYSTICK <= code < ev.BTN_GAMEPAD:
            buttons[code] = f"BTN{code - ev.BTN_JOYSTICK + 1}"

    # Two buttons sharing a label is almost always a half-finished swap: the
    # user remapped one direction and forgot the other.
    seen: dict[str, int] = {}
    for code, label in sorted(buttons.items()):
        if label in seen:
            log.warning(
                "%s: %s and %s both report %r -- a remap swap needs both directions",
                info.name,
                ev.key_name(seen[label]),
                ev.key_name(code),
                label,
            )
        seen[label] = code

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


class GamepadDiscretizer:
    """Folds raw events into fixed windows of semantic activity.

    The summary is *edge triggered*. A hold is announced once when it begins and
    once when it ends; the windows in between say nothing about it and are
    suppressed entirely if nothing else happened. Repeating "held RT" twice a
    second for a trigger the player is leaning on is pure token cost and reads
    as noise to the model.

    Level state still exists where it is useful: the structured `triggers` field
    and `holding` list describe the present moment, and `intensity` stays
    level-based because it drives screen-frame triggers.
    """

    def __init__(self, device_id: str, layout: Layout, info: ev.DeviceInfo, cfg: GamepadConfig):
        self.device_id = device_id
        self.layout = layout
        self.info = info
        self.cfg = cfg

        self._down: dict[str, float] = {}  # name -> press start
        self._taps: dict[str, int] = {}
        self._ended: dict[str, tuple[float, bool]] = {}  # name -> (total ms, announced)
        self._hold_started: list[str] = []
        self._announced_holds: set[str] = set()
        self._announced_triggers: dict[str, str] = {}
        self._trigger_full_since: dict[str, float] = {}
        self._announced_dpad = "idle"
        self._announced_sticks: dict[str, tuple[str, str]] = {}
        self._raw_axis: dict[int, int] = {}
        self._window_start: float | None = None
        self._was_active = False
        self._active_since: float | None = None
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
            self._down[name] = now
            self._transitions += 1
            self._pending_flush = True
        elif event.value == 0:
            start = self._down.pop(name, None)
            if start is None:
                return
            duration_ms = (now - start) * 1000.0
            if duration_ms <= self.cfg.tap_max_ms:
                self._taps[name] = self._taps.get(name, 0) + 1
            else:
                announced = name in self._announced_holds
                self._announced_holds.discard(name)
                previous = self._ended.get(name, (0.0, announced))
                self._ended[name] = (previous[0] + duration_ms, announced)
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

    def _discrete_axes(self) -> tuple:
        """The quantized analog state, used to count meaningful transitions."""
        left = self._stick(self.layout.left_stick)[:2]
        right = self._stick(self.layout.right_stick)[:2]
        sticks = (left, right) if self.cfg.sticks_mode == "full" else ()
        return (
            self._trigger("LT")[0],
            self._trigger("RT")[0],
            self._dpad(),
            sticks,
        )

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

    # -- output -----------------------------------------------------------

    def flush(self, now: float) -> GamepadActivity | GamepadIdle | None:
        window_start = self._window_start if self._window_start is not None else now
        window_ms = max(1.0, (now - window_start) * 1000.0)

        # A press crosses into "hold" the moment it outlives the tap threshold;
        # announce it once, here, and stay quiet until it ends.
        for name, start in self._down.items():
            if name in self._announced_holds:
                continue
            if (now - start) * 1000.0 > self.cfg.tap_max_ms:
                self._announced_holds.add(name)
                self._hold_started.append(name)

        lt_bucket, lt_value = self._trigger("LT")
        rt_bucket, rt_value = self._trigger("RT")
        dpad = self._dpad()
        left_dir, left_mag, left_value = self._stick(self.layout.left_stick)
        right_dir, right_mag, right_value = self._stick(self.layout.right_stick)

        trigger_edges: list[tuple[str, str, float | None]] = []
        for label, bucket in (("LT", lt_bucket), ("RT", rt_bucket)):
            previous = self._announced_triggers.get(label, "idle")
            if previous == bucket:
                continue
            self._announced_triggers[label] = bucket
            duration = None
            if bucket == "full":
                self._trigger_full_since[label] = now
            elif previous == "full":
                started = self._trigger_full_since.pop(label, None)
                if started is not None:
                    duration = (now - started) * 1000.0
            trigger_edges.append((label, bucket, duration))

        dpad_edge = dpad if dpad != self._announced_dpad else None
        self._announced_dpad = dpad

        stick_edges: list[tuple[str, str, str]] = []
        if self.cfg.sticks_mode == "full":
            for label, direction, magnitude in (
                ("left", left_dir, left_mag),
                ("right", right_dir, right_mag),
            ):
                if self._announced_sticks.get(label) != (direction, magnitude):
                    self._announced_sticks[label] = (direction, magnitude)
                    if direction != "idle":
                        stick_edges.append((label, direction, magnitude))

        holding = sorted(
            self._announced_holds
            | {label for label, bucket in self._announced_triggers.items() if bucket == "full"}
        )

        intensity = self._intensity(lt_value, rt_value, left_value, right_value)
        self.last_intensity = intensity
        apm = int(round(self._transitions * 60000.0 / window_ms))

        buttons = {
            name: {
                "taps": self._taps.get(name, 0),
                "held_ms": int(round(self._ended.get(name, (0.0, False))[0])),
            }
            for name in sorted(set(self._taps) | set(self._ended))
        }
        summary = self._summarize(trigger_edges, dpad_edge, stick_edges)

        self._reset_window(now)

        # Edge-triggered means an ongoing hold contributes nothing to say. The
        # controller is still active, but there is no news.
        has_news = bool(buttons or summary)
        still_active = bool(self._down) or holding or dpad != "idle" or left_value or right_value

        if not has_news and not still_active:
            if self._was_active:
                self._was_active = False
                active_ms = int(round((now - (self._active_since or now)) * 1000.0))
                self._active_since = None
                return GamepadIdle(device=self.device_id, active_ms=active_ms)
            return None

        if not self._was_active:
            self._was_active = True
            self._active_since = window_start

        if not has_news:
            return None

        event = GamepadActivity(
            device=self.device_id,
            window_ms=int(round(window_ms)),
            buttons=buttons,
            triggers={"LT": lt_bucket, "RT": rt_bucket},
            dpad=dpad,
            holding=holding,
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

    #: relative contribution of each input family to the activity scalar
    W_BUTTONS = 0.45
    W_TRIGGERS = 0.30
    W_STICKS = 0.35

    def _intensity(self, lt: float, rt: float, left: float, right: float) -> float:
        # Level-based on purpose: this drives screen-frame triggers, so a button
        # the player is holding down should keep counting even though the
        # summary has already said its piece.
        actions = sum(self._taps.values()) + len(self._down) + len(self._ended)
        button_part = min(1.0, actions / 5.0)
        trigger_part = max(lt, rt)
        # Sticks feed intensity even in "intensity" mode -- that is the point of
        # keeping them: they carry activity signal without costing tokens.
        sticks_on = self.cfg.sticks_mode != "off"
        stick_part = max(left, right) if sticks_on else 0.0

        # Renormalise over the families that are actually enabled, so the scalar
        # means the same thing whatever `sticks_mode` is. Without this, the
        # default `sticks_mode="off"` silently zeroes the largest weight and
        # caps intensity at 0.75: a trigger held flat out reached only 0.30
        # against a 0.35 trigger threshold, so `triggers.gamepad_intensity` was
        # unreachable by anything short of mashing four buttons inside one
        # 500 ms window. sess5 recorded zero gamepad-triggered frames in 151 s
        # of play, with a peak intensity of 0.27.
        total = self.W_BUTTONS + self.W_TRIGGERS + self.W_STICKS
        enabled = total if sticks_on else total - self.W_STICKS
        scale = total / enabled
        raw = (
            self.W_BUTTONS * button_part
            + self.W_TRIGGERS * trigger_part
            + self.W_STICKS * stick_part
        )
        return min(1.0, scale * raw)

    def _summarize(
        self,
        trigger_edges: list[tuple[str, str, float | None]],
        dpad_edge: str | None,
        stick_edges: list[tuple[str, str, str]],
    ) -> str:
        """Describe only what changed this window."""
        parts: list[str] = []

        for name in self._hold_started:
            parts.append(f"holding {name}")
        for label, bucket, duration_ms in trigger_edges:
            if bucket == "full":
                parts.append(f"holding {label}")
            elif duration_ms is not None:
                parts.append(f"released {label} after {duration_ms / 1000.0:.1f}s")
            elif bucket == "idle":
                parts.append(f"released {label}")
            else:
                parts.append(f"{label} {bucket}")

        if dpad_edge is not None and dpad_edge != "idle":
            parts.append(f"dpad {dpad_edge}")
        for label, direction, magnitude in stick_edges:
            parts.append(f"{label} stick {magnitude} {direction}")

        for name, (total_ms, announced) in sorted(self._ended.items()):
            seconds = total_ms / 1000.0
            # "released" only makes sense if the hold was announced earlier;
            # one that began and ended inside a single window never was.
            verb = "released" if announced else "held"
            preposition = "after" if announced else "for"
            parts.append(f"{verb} {name} {preposition} {seconds:.1f}s")

        tapped = [(n, c) for n, c in sorted(self._taps.items()) if c > 0]
        if tapped:
            rendered = ", ".join(f"{n} x{c}" if c > 1 else n for n, c in tapped)
            parts.append(f"tapped {rendered}")

        return ", ".join(parts)

    def _reset_window(self, now: float) -> None:
        self._taps = {}
        self._ended = {}
        self._hold_started = []
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
        self._ctx: BusContext | None = None
        self._devices: dict[str, _Device] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: list[asyncio.Task] = []
        self._netlink: socket.socket | None = None
        self._rescan_handle: asyncio.TimerHandle | None = None

    async def start(self, ctx: BusContext) -> None:
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
        layout = resolve_layout(info, self.cfg)
        discretizer = GamepadDiscretizer(info.device_id, layout, info, self.cfg)
        device = _Device(info, fd, discretizer)
        self._devices[info.device_id] = device
        assert self._loop is not None
        assert self._ctx is not None
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
        assert self._ctx is not None
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
            log.info(
                "netlink hotplug unavailable (%s); polling every %.1fs", exc, self.cfg.rescan_s
            )
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
