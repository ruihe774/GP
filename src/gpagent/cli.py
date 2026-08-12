"""gpagent command line: devices | record | inspect | replay."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
import time
import tomllib
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import CaptureConfig
from .events import (
    AgentResponse,
    GamepadActivity,
    GamepadConnected,
    GamepadDisconnected,
    GamepadIdle,
    ScreenFrame,
    SessionEnd,
    SessionStart,
    SpeechSegment,
)
from .sinks.jsonl import JsonlSink, read_session, session_dir_name
from .tokens import DEFAULT_RATES, Estimate, audio_tokens, image_tokens, text_tokens

log = logging.getLogger("gpagent")

#: voices the Realtime API accepts, as of openai 2.53
VOICES = [
    "alloy", "ash", "ballad", "cedar", "coral", "echo", "marin", "sage",
    "shimmer", "verse",
]


# -- helpers ---------------------------------------------------------------


def _looks_like_ssh() -> bool:
    return bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"))


def _has_display() -> bool:
    return bool(os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY"))


def _parse_set(values: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in values or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects section.key=value, got {item!r}")
        try:
            out[key.strip()] = json.loads(raw)
        except ValueError:
            out[key.strip()] = raw
    return out


def _build_config(args) -> CaptureConfig:
    cfg = CaptureConfig()
    replay_dir = getattr(args, "replay", None)
    if replay_dir:
        saved = _read_manifest(Path(replay_dir)).get("config")
        if isinstance(saved, dict):
            # Lenient: this is what the recording was made under, not what the
            # user is asking for. A key this version has dropped must not make
            # an old session unreplayable.
            stale = cfg.merge_dict(saved, strict=False)
            if stale:
                print(
                    f"note: {replay_dir}/manifest.json has settings this version "
                    f"no longer has, ignored: {', '.join(stale)}"
                )
    if getattr(args, "config", None):
        with open(args.config, "rb") as fh:
            try:
                # Strict, unlike the manifest above: here an unknown key is a
                # typo, and a setting silently ignored for a whole session is
                # worse than a message before it starts.
                cfg.merge_dict(tomllib.load(fh))
            except ValueError as exc:
                raise SystemExit(f"bad config in {args.config}: {exc}") from exc
    try:
        cfg.apply_overrides(_parse_set(getattr(args, "set", []) or []))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    sections = (("gamepad", "no_gamepad"), ("audio", "no_audio"), ("screen", "no_screen"))
    for section, flag in sections:
        if getattr(args, flag, False):
            getattr(cfg, section).enabled = False
    if getattr(args, "pick_screen", False):
        cfg.screen.reselect_source = True

    for item in getattr(args, "map", None) or []:
        code, sep, label = item.partition("=")
        if not sep or not label.strip():
            raise SystemExit(f"--map expects CODE=LABEL, got {item!r}")
        cfg.gamepad.button_map[code.strip()] = label.strip()

    # Fail on a mistyped button now, rather than after opening devices.
    from .capture.gamepad import parse_button_map

    try:
        parse_button_map(cfg.gamepad.button_map)
        for device in cfg.gamepad.device_button_map.values():
            parse_button_map(device)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return cfg


def _add_mapping_args(parser) -> None:
    parser.add_argument("-c", "--config", help="TOML config file")
    parser.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE")
    parser.add_argument(
        "--map",
        action="append",
        metavar="CODE=LABEL",
        help="relabel a button, e.g. --map BTN_NORTH=X (repeatable)",
    )


# -- devices ---------------------------------------------------------------


def cmd_devices(args) -> int:
    from .capture import evdev_raw as ev
    from .capture.gamepad import resolve_layout

    gamepad_cfg = _build_config(args).gamepad
    print("session")
    session_type = os.environ.get("XDG_SESSION_TYPE", "?")
    print(f"  type            {session_type}")
    display = os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY") or "none"
    print(f"  display         {display}")
    if _looks_like_ssh() or not _has_display():
        print("  WARNING: this looks like an SSH/headless session.")
        print("           Gamepad ACLs (uaccess) and the portal both bind to the")
        print("           active seat session; run gpagent from the desktop.")

    print("\ngamepads")
    unreadable = 0
    found = 0
    for path in ev.list_event_paths():
        info = ev.probe_device(path)
        if info is None:
            unreadable += 1
            continue
        if not ev.is_gamepad(info):
            continue
        found += 1
        layout = resolve_layout(info, gamepad_cfg)
        print(f"  {path}  {info.name}")
        print(f"    id            {info.device_id}")
        print(f"    vid:pid       {info.vid:04x}:{info.pid:04x}")
        print(f"    buttons       {' '.join(sorted(layout.buttons.values()))}")
        sticks = f"left={layout.left_stick is not None} right={layout.right_stick is not None}"
        print(f"    sticks        {sticks}")
        print(f"    triggers      {', '.join(sorted(layout.triggers)) or 'digital only'}")
        print(f"    dpad          {'hat' if layout.hat else 'buttons'}")
    if not found:
        print("  none detected")
    if unreadable:
        print(
            f"  ({unreadable} input nodes not readable by this user -- normal for keyboards/mice)"
        )

    print("\naudio (pipewire)")
    try:
        _print_audio_devices()
    except Exception as exc:
        print(f"  device enumeration failed: {exc}")

    print("\nscreencast portal")
    try:
        _print_portal()
    except Exception as exc:
        print(f"  unavailable: {exc}")
    return 0


def _print_audio_devices() -> None:
    from .capture.gst_util import ensure_gst

    ensure_gst()
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    defaults = _pipewire_defaults()

    monitor = Gst.DeviceMonitor.new()
    monitor.add_filter("Audio/Source", None)
    monitor.add_filter("Audio/Sink", None)
    monitor.start()
    try:
        seen: set[tuple[str, str]] = set()
        rows: list[tuple[str, str, str, bool]] = []
        for device in monitor.get_devices():
            props = device.get_properties()
            # The alsa and pulse providers duplicate every pipewire node; only
            # the pipewire provider sets node.name, so use it to deduplicate.
            node = props.get_string("node.name") if props else None
            if not node:
                continue
            media = (props.get_string("media.class") or device.get_device_class() or "").strip()
            label = device.get_display_name()
            # A sink and its monitor share one node.name, so the class has to be
            # part of the key or the monitor silently disappears.
            key = (media, node)
            if key in seen:
                continue
            seen.add(key)
            is_monitor = node.endswith(".monitor") or label.startswith("Monitor of")
            rows.append((media, node, label, is_monitor))

        for media, node, label, is_monitor in sorted(rows):
            marks = []
            if is_monitor:
                marks.append("monitor")
            if node == defaults.get("source"):
                marks.append("DEFAULT SOURCE")
            if node == defaults.get("sink"):
                marks.append("DEFAULT SINK")
            print(f"  {media:12s} {label[:42]:42s} {' '.join(marks)}")
        if not rows:
            print("  none detected")
    finally:
        monitor.stop()

    print("  capture plan:")
    mic = defaults.get("source") or "default source"
    sink = defaults.get("sink") or "default sink"
    print(f"    microphone    {mic}")
    print(f"    AEC reference monitor of {sink}")


def _pipewire_defaults() -> dict[str, str]:
    """Resolve the default source/sink node names via pw-dump.

    Diagnostics only -- capture itself never names a device, it just leaves the
    target unset and lets WirePlumber route it (and re-route it on change).
    """
    import subprocess

    try:
        raw = subprocess.run(
            ["pw-dump", "-N"], capture_output=True, timeout=5, text=True, check=True
        ).stdout
        objects = json.loads(raw)
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}

    out: dict[str, str] = {}
    for obj in objects:
        if obj.get("type") != "PipeWire:Interface:Metadata":
            continue
        for entry in obj.get("metadata") or []:
            key, value = entry.get("key"), entry.get("value")
            name = value.get("name") if isinstance(value, dict) else value
            if key == "default.audio.source" and name:
                out["source"] = name
            elif key == "default.audio.sink" and name:
                out["sink"] = name
    return out


def _print_portal() -> None:
    import gi

    gi.require_version("Gio", "2.0")
    from gi.repository import Gio, GLib

    conn = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    reply = conn.call_sync(
        "org.freedesktop.portal.Desktop",
        "/org/freedesktop/portal/desktop",
        "org.freedesktop.DBus.Properties",
        "GetAll",
        GLib.Variant("(s)", ("org.freedesktop.portal.ScreenCast",)),
        GLib.VariantType("(a{sv})"),
        Gio.DBusCallFlags.NONE,
        3000,
        None,
    )
    props = reply.unpack()[0]
    version = props.get("version")
    print(f"  version         {version}")
    print(f"  source types    {props.get('AvailableSourceTypes')} (1=monitor 2=window 4=virtual)")
    print(f"  cursor modes    {props.get('AvailableCursorModes')} (1=hidden 2=embedded 4=metadata)")
    if version is not None and version >= 4:
        print("  persist         supported (restore_token -- consent is first-run only)")
    else:
        print("  persist         NOT supported; expect a dialog every run")

    from .capture.portal import default_token_path

    token = default_token_path()
    print(f"  token cache     {token} ({'present' if token.exists() else 'absent'})")


# -- monitor ---------------------------------------------------------------


def cmd_monitor(args) -> int:
    """Live per-input view, for checking that buttons map to the right labels."""
    import select
    import time

    from .capture import evdev_raw as ev
    from .capture.gamepad import parse_button_map, resolve_layout

    # Output is watched live and is often piped to a file; keep it unbuffered.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    cfg = _build_config(args).gamepad
    pads = ev.find_gamepads()
    if not pads:
        print("no gamepad detected (run `gpagent devices`)", file=sys.stderr)
        return 1

    devices = []
    for info in pads:
        layout = resolve_layout(info, cfg)
        remapped = set(parse_button_map(cfg.button_map)) | set(
            parse_button_map(cfg.device_button_map.get(f"{info.vid:04x}:{info.pid:04x}"))
        )
        print(f"{info.name}  [{info.vid:04x}:{info.pid:04x}]  {info.path}")
        print("  resolved button map (kernel code -> label used in events):")
        for code in sorted(layout.buttons):
            note = "  (remapped)" if code in remapped else ""
            print(f"    {ev.key_name(code):<24s} {code:#06x}  ->  {layout.buttons[code]}{note}")
        for label, axis in sorted(layout.triggers.items()):
            print(f"    {ev.abs_name(axis):<24s} {axis:#06x}  ->  {label} (analog)")
        if layout.hat:
            print(f"    {ev.abs_name(layout.hat[0])}/{ev.abs_name(layout.hat[1])}       ->  dpad")
        try:
            devices.append((info, layout, ev.open_device(info.path)))
        except OSError as exc:
            print(f"  cannot open: {exc}", file=sys.stderr)

    if not devices:
        return 1

    print("\npress buttons; Ctrl-C to stop\n")
    print(f"  {'time':>8s}  {'event':<9s} {'label':<8s} {'detail'}")

    down: dict[tuple[int, int], float] = {}
    state: dict[tuple[int, int], str] = {}
    seen: set[str] = set()
    start = time.monotonic()
    fds = [fd for _, _, fd in devices]

    try:
        while True:
            ready, _, _ = select.select(fds, [], [], 0.2)
            now = time.monotonic()
            for info, layout, fd in devices:
                if fd not in ready:
                    continue
                # A throwaway discretizer gives the same bucketing the real
                # pipeline uses, so what is printed is what events will say.
                for raw in ev.read_events(fd):
                    _print_input(raw, info, layout, cfg, now - start, down, state, seen, args.raw)
    except KeyboardInterrupt:
        pass
    finally:
        for _, _, fd in devices:
            with contextlib.suppress(OSError):
                os.close(fd)

    print(f"\nbuttons seen: {' '.join(sorted(seen)) or 'none'}")
    all_labels = {label for _, layout, _ in devices for label in layout.buttons.values()}
    missing = sorted(all_labels - seen)
    if missing:
        print(f"not pressed:  {' '.join(missing)}")
    return 0


def _print_input(raw, info, layout, cfg, elapsed, down, state, seen, show_raw) -> None:
    from .capture import evdev_raw as ev

    key = (id(info), raw.code)
    if show_raw and raw.type != ev.EV_SYN:
        kind = {ev.EV_KEY: "KEY", ev.EV_ABS: "ABS"}.get(raw.type, str(raw.type))
        print(f"  {elapsed:8.2f}  raw       {kind:<8s} code={raw.code:#06x} value={raw.value}")

    if raw.type == ev.EV_KEY:
        label = layout.buttons.get(raw.code)
        if label is None:
            print(
                f"  {elapsed:8.2f}  UNMAPPED  {'?':<8s} "
                f"{ev.key_name(raw.code)} {raw.code:#06x} value={raw.value}"
            )
            return
        if raw.value == 1:
            down[key] = elapsed
            seen.add(label)
            print(
                f"  {elapsed:8.2f}  press     {label:<8s} "
                f"{ev.key_name(raw.code)} {raw.code:#06x}"
            )
        elif raw.value == 0:
            held = (elapsed - down.pop(key, elapsed)) * 1000.0
            kind = "tap" if held <= cfg.tap_max_ms else "hold"
            print(f"  {elapsed:8.2f}  release   {label:<8s} held {held:.0f} ms -> {kind}")
        return

    if raw.type != ev.EV_ABS:
        return

    absinfo = info.absinfo.get(raw.code)
    value = absinfo.normalize(raw.value) if absinfo else 0.0

    for label, axis in layout.triggers.items():
        if axis == raw.code:
            bucket = (
                "idle" if value < cfg.trigger_light
                else "full" if value >= cfg.trigger_full
                else "light"
            )
            if state.get(key) != bucket:
                state[key] = bucket
                seen.add(label)
                print(
                    f"  {elapsed:8.2f}  trigger   {label:<8s} {bucket:<5s} "
                    f"({ev.abs_name(raw.code)} raw={raw.value} -> {value:.2f})"
                )
            return

    if layout.hat and raw.code in layout.hat:
        axis = "x" if raw.code == layout.hat[0] else "y"
        if state.get(key) != str(raw.value):
            state[key] = str(raw.value)
            if raw.value:
                direction_names = {
                    ("x", -1): "left",
                    ("x", 1): "right",
                    ("y", -1): "up",
                    ("y", 1): "down",
                }
                direction = direction_names.get((axis, raw.value), str(raw.value))
                seen.add(f"dpad-{direction}")
                print(
                    f"  {elapsed:8.2f}  dpad      {direction:<8s} "
                    f"({ev.abs_name(raw.code)} = {raw.value})"
                )
        return

    for name, axes in (("left", layout.left_stick), ("right", layout.right_stick)):
        if axes and raw.code in axes:
            magnitude = abs(value)
            bucket = "idle" if magnitude < cfg.deadzone else "moved"
            if state.get(key) != bucket:
                state[key] = bucket
                if bucket != "idle":
                    print(
                        f"  {elapsed:8.2f}  stick     {name:<8s} "
                        f"{ev.abs_name(raw.code)} {value:+.2f}"
                    )
            return


# -- record ----------------------------------------------------------------


def cmd_record(args) -> int:
    cfg = _build_config(args)
    directory = Path(args.output or session_dir_name())
    return asyncio.run(_record(cfg, directory, args))


async def _record(cfg: CaptureConfig, directory: Path, args) -> int:
    from .bus import CaptureBus, CaptureSource

    sources: list[CaptureSource] = []
    if cfg.gamepad.enabled:
        from .capture.gamepad import GamepadSource

        sources.append(GamepadSource(cfg.gamepad))
    if cfg.audio.enabled:
        from .capture.audio import AudioSource

        sources.append(AudioSource(cfg.audio))
    if cfg.screen.enabled:
        from .capture.screen import ScreenSource

        sources.append(ScreenSource(cfg.screen, cfg.triggers))

    if not sources:
        raise SystemExit("all sources disabled; nothing to record")

    bus = CaptureBus(sources)
    sink = JsonlSink(directory, inline=args.inline)
    sink.open()
    print(f"recording to {directory}")
    if cfg.screen.enabled:
        print("(the portal may ask for screen-capture permission -- approve it once)")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    started = datetime.now(UTC)
    try:
        await bus.start()
    except Exception as exc:
        sink.close({"error": str(exc)})
        print(f"failed to start capture: {exc}", file=sys.stderr)
        return 1

    sink.write(
        SessionStart(
            t=0.0,
            seq=-1,
            started_at=started.isoformat(),
            config=cfg.to_dict(),
            devices=bus.describe(),
        )
    )

    async def consume() -> None:
        async for event in bus:
            sink.write(event)
            if not args.quiet:
                _print_live(event)

    consumer = asyncio.create_task(consume())
    timer = asyncio.create_task(asyncio.sleep(args.duration)) if args.duration else None
    waiters = [asyncio.create_task(stop.wait())] + ([timer] if timer else [])
    await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
    for task in waiters:
        task.cancel()

    print("\nstopping...")
    # Snapshot diagnostics before teardown: sources drop their device handles
    # in stop(), which would otherwise report an empty, useless manifest.
    devices = bus.describe()
    await bus.stop()
    with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
        await asyncio.wait_for(consumer, timeout=5)
    consumer.cancel()

    duration = (datetime.now(UTC) - started).total_seconds()
    end = SessionEnd(duration_s=round(duration, 2), counters=dict(sink.counts))
    end.t = duration
    end.seq = 10**9
    sink.write(end)
    sink.close(
        {
            "started_at": started.isoformat(),
            "duration_s": round(duration, 2),
            "config": cfg.to_dict(),
            "devices": devices,
            "dropped_events": bus.dropped,
        }
    )
    print(f"wrote {sum(sink.counts.values())} events to {directory}")
    _warn_about_capture(devices, duration)
    print(f"inspect with: gpagent inspect {directory}")
    return 0


def _warn_about_capture(devices: dict[str, Any], duration: float) -> None:
    """Say plainly when a source produced nothing, and why."""
    audio = devices.get("audio") or {}
    screen = devices.get("screen") or {}
    notes: list[str] = []

    if audio and audio.get("segments") == 0:
        peak = audio.get("input_peak_dbfs", -120.0)
        best = audio.get("vad_max_probability", 0.0)
        threshold = audio.get("vad_threshold", 0.5)
        if peak <= -60.0:
            notes.append(
                f"no speech captured, and the microphone never rose above {peak:.0f} dBFS "
                "-- check that the default source is a real microphone (gpagent devices)"
            )
        else:
            notes.append(
                f"no speech captured, though the microphone peaked at {peak:.0f} dBFS; "
                f"the VAD topped out at {best:.2f} against a {threshold} threshold"
            )

    handshake = screen.get("portal_handshake_s") or 0.0
    if handshake > 5.0:
        notes.append(
            f"the portal consent dialog took {handshake:.0f}s to answer, so the first "
            f"{handshake:.0f}s of a {duration:.0f}s session has no screen frames"
        )
    if screen.get("stall_events"):
        notes.append(
            f"the compositor stopped delivering frames {screen['stall_events']} time(s) "
            "-- a fullscreen game can suspend the screencast"
        )

    for note in notes:
        print(f"  note: {note}")


def _print_live(event) -> None:
    stamp = f"[{event.t:7.2f}]"
    if isinstance(event, GamepadActivity):
        print(f"{stamp} pad  {event.summary}  (intensity {event.intensity:.2f})")
    elif isinstance(event, GamepadIdle):
        print(f"{stamp} pad  idle")
    elif isinstance(event, SpeechSegment):
        print(f"{stamp} mic  speech {event.dur_ms} ms  ({event.rms_dbfs:.1f} dBFS)")
    elif isinstance(event, ScreenFrame):
        size = len(event.data or b"")
        print(f"{stamp} scr  frame {event.w}x{event.h} via {event.trigger}  ({size // 1024} KiB)")
    elif isinstance(event, GamepadConnected):
        print(f"{stamp} pad  connected: {event.name}")
    elif isinstance(event, GamepadDisconnected):
        print(f"{stamp} pad  disconnected: {event.name}")


# -- inspect ---------------------------------------------------------------


def cmd_inspect(args) -> int:
    directory = Path(args.directory)
    events = list(
        read_session(directory, load_blobs=args.wav or args.contact_sheet or args.sent_sheet)
    )
    if not events:
        print("no events")
        return 1

    manifest = _read_manifest(directory)
    estimate = Estimate()
    speech_ms = 0.0
    said_ms = 0.0
    triggers: dict[str, int] = {}
    counts: dict[str, int] = {}
    duration = 0.0

    if not args.quiet:
        print("timeline")
    for event in events:
        counts[event.TYPE] = counts.get(event.TYPE, 0) + 1
        duration = max(duration, event.t)
        line = None
        if isinstance(event, SessionStart):
            line = f"session start ({event.started_at})"
        elif isinstance(event, GamepadConnected):
            line = f"gamepad connected: {event.name}"
        elif isinstance(event, GamepadDisconnected):
            line = f"gamepad disconnected: {event.name}"
        elif isinstance(event, GamepadActivity):
            estimate.text_tokens += text_tokens(event.summary)
            line = f"pad  {event.summary}  [i={event.intensity:.2f} apm={event.apm}]"
        elif isinstance(event, GamepadIdle):
            line = f"pad  idle (after {event.active_ms} ms active)"
        elif isinstance(event, SpeechSegment):
            speech_ms += event.dur_ms
            estimate.audio_tokens += audio_tokens(event.dur_ms)
            estimate.audio_ms += event.dur_ms
            line = f"mic  speech {event.dur_ms} ms  {event.rms_dbfs:.1f} dBFS"
        elif isinstance(event, ScreenFrame):
            triggers[event.trigger] = triggers.get(event.trigger, 0) + 1
            estimate.image_tokens += image_tokens(event.w, event.h)
            estimate.frames += 1
            line = (
                f"scr  frame {event.w}x{event.h} via {event.trigger} "
                f"(scene {event.scene_score:.3f})"
            )
        elif isinstance(event, AgentResponse):
            said_ms += event.dur_ms
            mark = " CUT" if event.cut else ""
            text = event.transcript or "(no transcript)"
            # A text remark has no duration to report -- it was written, not
            # spoken, and "(0 ms)" would read as a response that said nothing.
            tail = "" if event.modality == "text" else f"  ({event.dur_ms} ms)"
            line = f"say  [{event.reason}{mark}] {text}{tail}"
        elif isinstance(event, SessionEnd):
            duration = max(duration, event.duration_s)
            line = f"session end ({event.duration_s:.1f}s)"
        if line and not args.quiet:
            print(f"  [{event.t:7.2f}] {line}")

    if manifest.get("duration_s"):
        duration = float(manifest["duration_s"])
    minutes = duration / 60.0 if duration else 0.0

    print("\nsummary")
    print(f"  duration        {duration:.1f}s")
    for name in sorted(counts):
        print(f"  {name:15s} {counts[name]}")
    if speech_ms:
        share = speech_ms / (duration * 1000.0) * 100 if duration else 0.0
        print(f"  speech audio    {speech_ms / 1000:.1f}s ({share:.1f}% of session)")
    if said_ms:
        share = said_ms / (duration * 1000.0) * 100 if duration else 0.0
        print(f"  agent spoke     {said_ms / 1000:.1f}s ({share:.1f}% of session)")
    if triggers:
        print(f"  frames by trigger  {', '.join(f'{k}={v}' for k, v in sorted(triggers.items()))}")
    if manifest.get("dropped_events"):
        print(f"  dropped events  {manifest['dropped_events']}")

    cost = estimate.cost()
    print("\nestimated input cost (approximate -- see tokens.py)")
    print(f"  audio           {estimate.audio_tokens:>8d} tok  ${cost['audio_usd']:.4f}")
    print(
        f"  images          {estimate.image_tokens:>8d} tok  ${cost['image_usd']:.4f}"
        f"  ({estimate.frames} frames)"
    )
    print(f"  text summaries  {estimate.text_tokens:>8d} tok  ${cost['text_usd']:.4f}")
    print(f"  total                        ${cost['total_usd']:.4f}", end="")
    if minutes:
        print(f"   (${cost['total_usd'] / minutes:.4f}/min)")
    else:
        print()
    print(
        f"  rates: audio ${DEFAULT_RATES.audio_per_mtok}/Mtok @1 tok/100ms, "
        f"image ${DEFAULT_RATES.image_per_mtok}/Mtok, text ${DEFAULT_RATES.text_per_mtok}/Mtok"
    )

    if args.wav:
        _write_wavs(directory, events)
    elif speech_ms or said_ms:
        rate = next(
            (e.sample_rate for e in events if isinstance(e, SpeechSegment)), 24000
        )
        print(
            f"\nspeech blobs are headerless PCM (s16le, mono, {rate} Hz)."
            f"\n  gpagent inspect {directory} --wav      # write .wav files"
            f"\n  ffplay -f s16le -ar {rate} -ac 1 {directory}/blobs/<seq>-speech.pcm"
            f"\n  ffmpeg -f s16le -ar {rate} -ac 1 -i in.pcm out.wav"
        )
    if args.contact_sheet:
        _write_contact_sheet(directory, events)
    if args.sent_sheet:
        _write_sent_sheet(directory, events, Path(args.agent_log) if args.agent_log else None)
    return 0


def _read_asks(path: Path) -> list[dict[str, Any]]:
    """The `ask` lines of an agent log, which say what each turn was shown."""
    asks = []
    with open(path) as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                line = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if line.get("kind") == "ask" and line.get("frames"):
                asks.append(line)
    return asks


def _read_manifest(directory: Path) -> dict[str, Any]:
    path = directory / "manifest.json"
    if not path.exists():
        return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except ValueError:
        return {}


def _write_wavs(directory: Path, events: list) -> None:
    """Both sides of the conversation, plus one file of the whole thing."""
    out = directory / "wav"
    out.mkdir(exist_ok=True)
    written = 0
    rate = 24000
    for event in events:
        if isinstance(event, SpeechSegment):
            kind = "speech"
        elif isinstance(event, AgentResponse):
            kind = "agent"
        else:
            continue
        if not event.data:
            continue
        rate = event.sample_rate
        with wave.open(str(out / f"{event.seq:06d}-{kind}.wav"), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(event.sample_rate)
            fh.writeframes(event.data)
        written += 1

    # A single file laid out on the session timeline, silence in the gaps, so
    # the exchange can be listened to as a conversation rather than as a pile
    # of clips in the wrong order.
    #
    # Both event types are stamped when they FINISH -- a speech segment when
    # VAD closes it, a response when the server stops streaming -- so a clip
    # starts `dur_ms` before its own `t`. Laying them out at `t` put every one
    # of them a whole utterance late and made the two sides look interleaved
    # when they were not.
    timed = [
        e for e in events
        if isinstance(e, (SpeechSegment, AgentResponse)) and e.data
    ]
    if timed:
        def span(e) -> tuple[float, float]:
            assert e.data is not None
            start = e.t - (e.dur_ms or 0) / 1000.0
            return start, start + len(e.data) / 2 / e.sample_rate

        end = max(span(e)[1] for e in timed)
        track = bytearray(int(end * rate) * 2 + 2)
        for event in timed:
            assert event.data is not None
            at = max(0, int(span(event)[0] * rate)) * 2
            track[at : at + len(event.data)] = event.data
        with wave.open(str(out / "conversation.wav"), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(rate)
            fh.writeframes(bytes(track))
        written += 1

    print(f"\nwrote {written} wav file(s) to {out}")


def _write_contact_sheet(directory: Path, events: list, columns: int = 4) -> None:
    from io import BytesIO

    from PIL import Image, ImageDraw

    frames = [e for e in events if isinstance(e, ScreenFrame) and e.data]
    if not frames:
        print("\nno frames to montage")
        return

    thumbs = []
    for event in frames:
        assert event.data is not None
        image = Image.open(BytesIO(event.data)).convert("RGB")
        image.thumbnail((480, 480))
        thumbs.append((image, event))

    cell_w = max(t.width for t, _ in thumbs)
    cell_h = max(t.height for t, _ in thumbs)
    label_h = 18
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * (cell_h + label_h)), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)

    for index, (thumb, event) in enumerate(thumbs):
        x = (index % columns) * cell_w
        y = (index // columns) * (cell_h + label_h)
        sheet.paste(thumb, (x, y))
        draw.text(
            (x + 4, y + cell_h + 4),
            f"t={event.t:.1f}s  {event.trigger}  {event.w}x{event.h}",
            fill=(200, 200, 200),
        )

    path = directory / "contact-sheet.png"
    sheet.save(path)
    print(f"\nwrote contact sheet with {len(thumbs)} frames to {path}")


def _write_sent_sheet(directory: Path, events: list, agent_log: Path | None = None) -> None:
    """One row per turn: exactly the images that turn put on the wire.

    `--contact-sheet` montages everything capture recorded, which is the wrong
    question once a turn carries a trail -- a session captures a thousand frames
    and sends a few hundred. This reads the `frames` field of the agent log's
    `ask` lines, which names the `seq` of every image sent and the detail it was
    sent at, and pulls those blobs back out of the recording.

    The agent log and the blobs need not live together: a replay recorded with
    `--no-media` has the log but no payloads, so point `--agent-log` at the run
    and `directory` at the original recording.
    """
    from io import BytesIO

    from PIL import Image, ImageDraw

    log = agent_log or directory / "agent.jsonl"
    if not log.exists():
        print(f"\nno agent log at {log}; run `commentate -o DIR` or pass --agent-log")
        return
    asks = _read_asks(log)
    if not asks:
        print(f"\nno turns with frames in {log}")
        return

    by_seq = {e.seq: e for e in events if isinstance(e, ScreenFrame) and e.data}
    rows: list[tuple[dict, list[tuple[int, str]]]] = []
    for ask in asks:
        meta = ask["frames"]
        cells = [(seq, meta.get("trail_detail", "low")) for seq in meta.get("trail", [])]
        cells.append((meta["current"], meta.get("detail", "high")))
        rows.append((ask, cells))

    print(f"\nturns and the images they sent, from {log}")
    missing = 0
    for ask, cells in rows:
        shown = " ".join(f"#{seq}:{detail}" for seq, detail in cells)
        missing += sum(1 for seq, _ in cells if seq not in by_seq)
        print(f"  t={ask['t']:>8.1f}s  {ask['text']:<28s} {shown}")
    if missing:
        print(f"  ({missing} blobs not in {directory}; drawn as placeholders)")

    columns = max(len(cells) for _, cells in rows)
    cell_w, cell_h, label_h, pad, margin = 320, 180, 16, 4, 190
    sheet = Image.new(
        "RGB",
        (margin + columns * (cell_w + pad), len(rows) * (cell_h + label_h + pad)),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(sheet)

    for row, (ask, cells) in enumerate(rows):
        y = row * (cell_h + label_h + pad)
        # Right-align, so the current frame is the last column on every row and
        # the trail reads left-to-right into it however long it is.
        offset = columns - len(cells)
        draw.text((6, y + cell_h // 2 - 12), f"t={ask['t']:.1f}s", fill=(170, 170, 170))
        draw.text((6, y + cell_h // 2 + 2), ask["text"][:28], fill=(120, 120, 120))
        for col, (seq, detail) in enumerate(cells):
            x = margin + (offset + col) * (cell_w + pad)
            event = by_seq.get(seq)
            if event is not None and event.data:
                thumb = Image.open(BytesIO(event.data)).convert("RGB")
                thumb.thumbnail((cell_w, cell_h))
                sheet.paste(thumb, (x, y))
                box = (x, y, x + thumb.width - 1, y + thumb.height - 1)
            else:
                box = (x, y, x + cell_w - 1, y + cell_h - 1)
                draw.rectangle(box, fill=(40, 40, 40))
                draw.text((x + 8, y + 8), "(blob not here)", fill=(140, 140, 140))
            # High detail is what the turn is really being asked about; the
            # trail is context. The border says which at a glance, the label
            # says it in words.
            hot = detail == "high"
            draw.rectangle(box, outline=(90, 200, 120) if hot else (90, 90, 110), width=3)
            label = f"#{seq} {detail}" + ("" if hot else "  trail")
            draw.text(
                (x + 4, y + cell_h + 2), label, fill=(210, 210, 210) if hot else (150, 150, 150)
            )

    path = directory / "sent-sheet.png"
    sheet.save(path)
    print(f"\nwrote {len(rows)} turns to {path}")


# -- hud -------------------------------------------------------------------

#: A reel that exercises what actually breaks: a one-liner, something long
#: enough to wrap, a burst that pushes an entry off the top, and a run with no
#: spaces in it. The last line only renders as words with a font that has the
#: glyphs -- `--set hud.font_lang=ja` is how you get one.
DEMO_LINES = [
    "Oh no.",
    "That jump was optimistic — you had the speed, just not the angle.",
    "Okay, going left this time.",
    "Wait.",
    "That is the third time that exact barrel has ended a run, and I want you "
    "to know I have been counting.",
    "ナイス、そのショートカットで一周分ぐらい稼いだよ。",
]


def cmd_hud(args) -> int:
    cfg = _build_config(args)
    cfg.hud.enabled = True
    if args.anchor:
        cfg.hud.anchor = args.anchor
    if args.monitor:
        cfg.hud.monitor = args.monitor
    if args.hold:
        cfg.hud.hold_min_s = cfg.hud.hold_max_s = args.hold

    lines = DEMO_LINES if args.demo else list(args.text)
    if args.render:
        return _render_hud(cfg, lines or DEMO_LINES[:3], args)
    if not lines and not args.stdin:
        raise SystemExit("give some text, or --demo, or --stdin")

    from .agent.hud import hold_for, make_hud

    hud = make_hud(cfg.hud)
    print(f"hud: {hud.name}  ({cfg.hud.anchor} of {cfg.hud.monitor})")
    if hud.name == "null":
        print("  nothing will appear: no X display, or the window would not open")
    last = ""
    try:
        for index, text in enumerate(lines):
            print(f"  {text}")
            hud.show(text)
            last = text
            if args.stdin or index < len(lines) - 1:
                time.sleep(args.interval)
        if args.stdin:
            for line in sys.stdin:
                text = line.strip()
                if text:
                    print(f"  {text}")
                    hud.show(text)
                    last = text
        # Stay alive until the last remark has had its time on screen.
        if last:
            time.sleep(hold_for(last, cfg.hud) + cfg.hud.fade_ms / 1000.0)
    except KeyboardInterrupt:
        print()
    finally:
        hud.close()
    return 0


def _render_hud(cfg: CaptureConfig, lines: list[str], args) -> int:
    """Draw the card to a file. No X server, no compositor, no human."""
    from .agent.hud import Entry, Metrics, Monitor, load_font, place, render_card

    try:
        width, _, height = args.render_size.partition("x")
        screen = Monitor(0, 0, int(width), int(height), scale=cfg.hud.scale or 1.0, name="render")
    except ValueError:
        raise SystemExit(f"--render-size expects WxH, got {args.render_size!r}") from None
    metrics = Metrics.for_monitor(cfg.hud, screen)
    shown = lines[-max(1, cfg.hud.max_entries) :]
    entries = [Entry(text=text, shown_at=0.0, expires_at=60.0) for text in shown]
    font = load_font(cfg.hud, metrics.font_px)
    card = render_card(entries, [1.0] * len(entries), cfg.hud, metrics, font)
    card.save(args.render)
    x, y = place(card.size, screen, metrics, cfg.hud)
    print(f"wrote {card.width}x{card.height} to {args.render}")
    print(f"  {screen.width}x{screen.height} at {metrics.scale:g}x, {cfg.hud.anchor} at ({x}, {y})")
    return 0


# -- commentate ------------------------------------------------------------


def cmd_commentate(args) -> int:
    cfg = _build_config(args)
    if args.model:
        cfg.agent.model = args.model
    if args.voice:
        cfg.agent.voice = args.voice
    if args.voice_speed is not None:
        cfg.agent.voice_speed = args.voice_speed
    if args.image_detail:
        cfg.agent.image_detail = args.image_detail
    if args.image_trail is not None:
        if args.image_trail < 0:
            raise SystemExit("--image-trail cannot be negative")
        cfg.agent.image_trail = args.image_trail
    if args.volume is not None:
        cfg.agent.volume = args.volume
    if args.language:
        cfg.agent.language = args.language
    if args.text_output is not None:
        cfg.agent.output = args.text_output
    if cfg.agent.text_output:
        # Nothing will be spoken, so there is nothing to open a sink for, and
        # the HUD is the only place the words can go.
        cfg.agent.playback = False
        cfg.hud.enabled = True
    if args.no_playback or args.dry_run:
        cfg.agent.playback = False
    if args.replay is None and args.replay_speed == 0.0:
        raise SystemExit("--replay-speed 0 only makes sense with --replay")
    return asyncio.run(_commentate(cfg, args))


async def _commentate(cfg: CaptureConfig, args) -> int:
    from .agent.hud import make_hud
    from .agent.playback import make_player
    from .agent.session import CommentaryAgent, ReplayClock, drive_from_bus, drive_from_session
    from .agent.transport import OpenAITransport, RecordingTransport, Transport

    out_dir = Path(args.output) if args.output else None
    deterministic = args.replay is not None and args.replay_speed == 0.0
    clock = ReplayClock() if deterministic else None

    transport: Transport
    if args.dry_run:
        transport = RecordingTransport(text=cfg.agent.text_output)
    else:
        from .agent.env import MissingAPIKey, load_api_key

        try:
            transport = OpenAITransport(cfg.agent.model, load_api_key())
        except MissingAPIKey as exc:
            print(str(exc), file=sys.stderr)
            return 1

    player = make_player(
        cfg.agent.playback,
        clock or time.monotonic,
        sink=cfg.agent.audio_sink,
        volume=cfg.agent.volume,
    )
    # A text session opens the overlay here for the same reason it opens the
    # sink for a spoken one: it is the output device. `make_hud` degrades to a
    # null one when there is no display, which keeps a headless replay running.
    hud = make_hud(cfg.hud) if cfg.agent.text_output else None

    # One directory holding both halves of the conversation: what was captured
    # and what the agent said back, in one ordered stream with the audio beside
    # it, so `gpagent inspect` reads a session the same way whether it came from
    # `record` or from `commentate`.
    sink = None
    if out_dir is not None:
        sink = JsonlSink(out_dir, inline=args.inline, media=not args.no_media)
        sink.open()
        if args.replay is None:
            # A replayed recording re-emits its own session.start; only live
            # capture needs one written here.
            sink.write(
                SessionStart(
                    t=0.0,
                    seq=-1,
                    started_at=datetime.now(UTC).isoformat(),
                    config=cfg.to_dict(),
                    devices={"source": "live capture", "model": cfg.agent.model},
                )
            )

    agent = CommentaryAgent(
        cfg,
        transport,
        player,
        hud,
        clock=clock or time.monotonic,
        log_path=(out_dir / "agent.jsonl") if out_dir else None,
        on_line=None if args.quiet else _print_line,
        recorder=sink.write if sink is not None else None,
    )

    mode = "dry run" if args.dry_run else cfg.agent.model
    source = f"replay {args.replay}" if args.replay else "live capture"
    print(f"commentating: {source} -> {mode}")
    if args.replay and isinstance(_read_manifest(Path(args.replay)).get("config"), dict):
        print(f"  using recorded config from {args.replay} (pass -c/--set/flags to override)")
    if hud is not None:
        print(f"  text output -> hud: {hud.name} ({cfg.hud.anchor} of {cfg.hud.monitor})")
        if hud.name == "null":
            print("  nothing will appear on screen: no X display, or the window would not open")
    elif not args.dry_run and not cfg.agent.playback:
        print("(playback disabled: nothing will come out of the speakers)")

    await agent.start()
    try:
        if deterministic:
            assert clock is not None, "deterministic replay always constructs a ReplayClock"
            await drive_from_session(agent, args.replay, clock)
        else:
            bus = await _build_bus(cfg, args)
            try:
                await drive_from_bus(agent, bus)
            finally:
                await bus.stop()
    except KeyboardInterrupt:
        pass
    finally:
        await agent.close()
        if sink is not None:
            sink.close(
                {
                    "model": cfg.agent.model,
                    "source": args.replay or "live capture",
                    "config": cfg.to_dict(),
                    "agent": agent.report(),
                }
            )

    _print_agent_report(agent, args, out_dir)
    if sink is not None:
        print(f"  full session ({sum(sink.counts.values())} events) -> gpagent inspect {out_dir}")
    return 0


async def _build_bus(cfg: CaptureConfig, args):
    """Either a recording played back in real time, or the real thing."""
    from .bus import CaptureBus, CaptureSource

    if args.replay is not None:
        from .replay import ReplaySource

        source = ReplaySource(args.replay, speed=args.replay_speed)
        bus = CaptureBus([source])
        await bus.start()
        # Stop the bus once the recording runs out, so the run terminates.
        asyncio.create_task(_stop_when_finished(source, bus))
        return bus

    sources: list[CaptureSource] = []
    if cfg.gamepad.enabled:
        from .capture.gamepad import GamepadSource

        sources.append(GamepadSource(cfg.gamepad))
    if cfg.audio.enabled:
        from .capture.audio import AudioSource

        sources.append(AudioSource(cfg.audio))
    if cfg.screen.enabled:
        from .capture.screen import ScreenSource

        sources.append(ScreenSource(cfg.screen, cfg.triggers))
    if not sources:
        raise SystemExit("all sources disabled; nothing to commentate on")
    bus = CaptureBus(sources)
    await bus.start()
    return bus


async def _stop_when_finished(source, bus) -> None:
    await source.finished.wait()
    # Let anything already queued drain before pulling the rug out.
    await asyncio.sleep(1.0)
    await bus.stop()


def _print_line(line) -> None:
    marks = {
        "ask": "    ",
        "say": "say ",
        "heard": "you ",
        "player": "mic ",
        "cut": "CUT ",
        "note": "??  ",
        "prune": "--  ",
        "error": "ERR ",
        "session": "--  ",
    }
    prefix = marks.get(line.kind, "    ")
    text = line.text
    if line.kind == "ask":
        sent = line.extra.get("sent") or []
        if sent:
            text += "  |  " + ", ".join(sent)
    print(f"[{line.t:7.2f}] {prefix} {text}")


def _print_agent_report(agent, args, out_dir: Path | None) -> None:
    report = agent.report()
    usage = report["usage"]
    print("\nagent")
    print(f"  responses       {report['responses']}  {report['by_reason']}")
    if report["declined"]:
        print(f"  stayed quiet    {report['declined']}")
    print(
        f"  frames          {report['frames_sent']} sent of {report['frames_seen']} captured"
        + (f", +{report['trail_sent']} as trail" if report.get("trail_sent") else "")
    )
    if report["frame_age_s"]:
        age = report["frame_age_s"]
        print(f"  frame age       {age['mean']}s mean, {age['max']}s worst at send time")
    pruned = report.get("pruned") or {}
    if pruned.get("rounds"):
        # Rounds is the number that matters: one round is one invalidated
        # prompt-cache prefix, whether it dropped two items or twenty.
        kinds = ", ".join(f"{n} {kind}" for kind, n in sorted(pruned.items()) if kind != "rounds")
        print(f"  pruned          {kinds} in {pruned['rounds']} rounds")
    if report.get("output") == "text":
        print(f"  wrote           {len(agent.hud.shown)} remarks to the {agent.hud.name} hud")
    else:
        print(f"  spoke           {usage['spoken_s']}s")

    tok = usage["tokens"]
    cost = usage["cost"]
    print(f"\nreported usage ({usage['model']})")
    print(f"  input audio     {tok['input_audio']:>8d} tok  ${cost['in_audio_usd']:.4f}")
    print(f"  input image     {tok['input_image']:>8d} tok  ${cost['in_image_usd']:.4f}")
    print(f"  input text      {tok['input_text']:>8d} tok  ${cost['in_text_usd']:.4f}")
    print(f"  input cached    {tok['input_cached']:>8d} tok  ${cost['in_cached_usd']:.4f}")
    if tok["input_unattributed"]:
        print(
            f"  input other     {tok['input_unattributed']:>8d} tok  ${cost['in_other_usd']:.4f}"
        )
    print(f"  output audio    {tok['output_audio']:>8d} tok  ${cost['out_audio_usd']:.4f}")
    print(f"  output text     {tok['output_text']:>8d} tok  ${cost['out_text_usd']:.4f}")
    print(f"  total                        ${cost['total_usd']:.4f}", end="")

    # Span, not the last timestamp: live, the clock is `time.monotonic`, whose
    # absolute value is the machine's uptime and made every $/min read as $0.0001.
    duration = (agent.lines[-1].t - agent.lines[0].t) if len(agent.lines) > 1 else 0.0
    if duration > 0:
        print(f"   (${cost['total_usd'] / (duration / 60.0):.4f}/min)")
    else:
        print()
    if args.dry_run:
        print("  (dry run: token counts are estimates, not a bill)")

    if args.replay:
        _compare_with_estimate(args.replay, agent, tok, cost)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "agent-usage.json", "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {out_dir}/agent.jsonl and {out_dir}/agent-usage.json")


def _compare_with_estimate(directory: str, agent, tok: dict, cost: dict) -> None:
    """The reconciliation Component A never got to do.

    Both sides are priced at the *same* model's rates, so the comparison is of
    policy against policy and not of one model's price list against another's.
    """
    from .tokens import estimate_events

    estimate = estimate_events(read_session(directory))
    capture = estimate.cost(agent.meter.rates)
    print("\nagainst `gpagent inspect` (everything captured, sent in full)")
    print(f"  {'':16s}{'captured':>12s}{'sent':>12s}")
    print(f"  {'image tok':16s}{estimate.image_tokens:>12d}{tok['input_image']:>12d}")
    print(f"  {'audio tok':16s}{estimate.audio_tokens:>12d}{tok['input_audio']:>12d}")
    print(f"  {'text tok':16s}{estimate.text_tokens:>12d}{tok['input_text']:>12d}")
    print(
        f"  {'input usd':16s}{capture['total_usd']:>12.4f}{cost['input_usd']:>12.4f}"
        f"   (same rates)"
    )
    if capture["total_usd"] > 0:
        share = cost["input_usd"] / capture["total_usd"] * 100
        print(f"  the agent's input cost was {share:.0f}% of sending everything")
    print(
        "  note: 'sent' can exceed 'captured'. `inspect` counts each item once;\n"
        "        the API re-bills the whole conversation as input on every turn,\n"
        "        which is what agent.prune_after_s controls."
    )


# -- replay ----------------------------------------------------------------


def cmd_replay(args) -> int:
    directory = Path(args.directory)
    events = list(read_session(directory, load_blobs=True))
    return asyncio.run(_replay(events, args.replay_speed))


async def _replay(events: list, speed: float) -> int:
    import time

    if not events:
        print("no events")
        return 1
    start = time.monotonic()
    origin = events[0].t
    for event in events:
        target = (event.t - origin) / max(0.01, speed)
        delay = target - (time.monotonic() - start)
        if delay > 0:
            await asyncio.sleep(delay)
        payload = len(event.data) if event.data else 0
        print(f"[{event.t:7.2f}] {event.TYPE:20s} {payload if payload else ''}")
    return 0


# -- entrypoint ------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpagent", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    devices = sub.add_parser("devices", help="enumerate inputs and verify permissions")
    _add_mapping_args(devices)
    devices.set_defaults(func=cmd_devices)

    monitor = sub.add_parser(
        "monitor", help="live per-button view, to verify the mapping is right"
    )
    monitor.add_argument("--raw", action="store_true", help="also dump every evdev event")
    _add_mapping_args(monitor)
    monitor.set_defaults(func=cmd_monitor)

    record = sub.add_parser("record", help="capture a session")
    record.add_argument("-o", "--output", help="session directory (default: session-<timestamp>)")
    record.add_argument("-d", "--duration", type=float, default=0.0, help="stop after N seconds")
    _add_mapping_args(record)
    record.add_argument("--inline", action="store_true", help="base64 blobs into the JSONL")
    record.add_argument(
        "--pick-screen",
        action="store_true",
        help="show the screen-capture picker instead of reusing the last choice",
    )
    record.add_argument("--no-gamepad", action="store_true")
    record.add_argument("--no-audio", action="store_true")
    record.add_argument("--no-screen", action="store_true")
    record.add_argument("-q", "--quiet", action="store_true", help="no live output")
    record.set_defaults(func=cmd_record)

    inspect = sub.add_parser("inspect", help="summarize a recorded session")
    inspect.add_argument("directory")
    inspect.add_argument("--wav", action="store_true", help="convert speech blobs to wav")
    inspect.add_argument("--contact-sheet", action="store_true", help="montage frames to png")
    inspect.add_argument(
        "--sent-sheet",
        action="store_true",
        help="montage only what the agent sent, one row per turn, labelled by detail",
    )
    inspect.add_argument(
        "--agent-log",
        metavar="PATH",
        help="agent.jsonl for --sent-sheet, if not in DIR (e.g. a --no-media replay run)",
    )
    inspect.add_argument("-q", "--quiet", action="store_true", help="stats only, no timeline")
    inspect.set_defaults(func=cmd_inspect)

    commentate = sub.add_parser(
        "commentate", help="watch, and talk about it (Component B)"
    )
    commentate.add_argument(
        "--replay",
        metavar="DIR",
        help="drive from a recorded session instead of live capture",
    )
    commentate.add_argument(
        "--replay-speed",
        type=float,
        default=1.0,
        help="replay speed; 0 means as fast as possible, deterministically",
    )
    commentate.add_argument(
        "--dry-run",
        action="store_true",
        help="never connect: print what would be sent, and what it would cost",
    )
    commentate.add_argument("--model", help="override agent.model")
    commentate.add_argument(
        "--voice",
        choices=VOICES,
        help="override agent.voice (default: %(default)s)",
        default=None,
    )
    commentate.add_argument(
        "--voice-speed",
        type=float,
        default=None,
        help="override agent.voice_speed; output playback rate, clamped to [0.25, 1.5]",
    )
    commentate.add_argument(
        "--language",
        metavar="CODE",
        help="ISO-639-1 code to speak, e.g. ja. Default: follow the player.",
    )
    commentate.add_argument(
        "--no-media",
        action="store_true",
        help="with -o, write the event stream but no audio or frame payloads",
    )
    commentate.add_argument(
        "--inline", action="store_true", help="base64 blobs into the JSONL"
    )
    commentate.add_argument(
        "--image-detail", choices=["auto", "low", "high"], help="override agent.image_detail"
    )
    commentate.add_argument(
        "--image-trail",
        type=int,
        metavar="N",
        help="earlier low-detail frames to send with each response (0 disables)",
    )
    commentate.add_argument(
        "--text",
        dest="text_output",
        action="store_const",
        const="text",
        default=None,
        help="read, not heard: the model writes each remark and it goes on the HUD",
    )
    commentate.add_argument(
        "--no-text",
        dest="text_output",
        action="store_const",
        const="voice",
        help="speak, overriding agent.output from the config",
    )
    commentate.add_argument(
        "--no-playback", action="store_true", help="do not open the audio sink"
    )
    commentate.add_argument(
        "--volume",
        type=float,
        default=None,
        help="override agent.volume; linear gain, 1.0 is unity (default: %(default)s)",
    )
    commentate.add_argument("-o", "--output", help="directory for agent.jsonl and usage")
    commentate.add_argument("-q", "--quiet", action="store_true", help="no live output")
    _add_mapping_args(commentate)
    commentate.add_argument("--no-gamepad", action="store_true")
    commentate.add_argument("--no-audio", action="store_true")
    commentate.add_argument("--no-screen", action="store_true")
    commentate.add_argument(
        "--pick-screen",
        action="store_true",
        help="show the screen-capture picker instead of reusing the last choice",
    )
    commentate.set_defaults(func=cmd_commentate)

    hud = sub.add_parser(
        "hud", help="put commentary on screen instead of through the speakers"
    )
    hud.add_argument("text", nargs="*", help="remarks to show, in order")
    hud.add_argument("--demo", action="store_true", help="a scripted reel instead of TEXT")
    hud.add_argument("--stdin", action="store_true", help="one line in, one toast out, until EOF")
    hud.add_argument("--render", metavar="PATH", help="draw to a PNG and exit; needs no display")
    hud.add_argument(
        "--render-size",
        metavar="WxH",
        default="1920x1080",
        help="screen --render lays the card out against (default: 1920x1080)",
    )
    hud.add_argument("--interval", type=float, default=2.0, help="seconds between remarks")
    hud.add_argument(
        "--hold", type=float, default=0.0, help="fixed seconds on screen, instead of reading speed"
    )
    hud.add_argument(
        "--anchor",
        choices=["top-right", "top-left", "top", "bottom-right", "bottom-left", "bottom"],
        help="which corner the stack sits in",
    )
    hud.add_argument("--monitor", help='"primary", an output name like HDMI-1, or an index')
    _add_mapping_args(hud)
    hud.set_defaults(func=cmd_hud)

    replay = sub.add_parser("replay", help="re-emit a session with original timing")
    replay.add_argument("directory")
    replay.add_argument("--replay-speed", type=float, default=1.0)
    replay.set_defaults(func=cmd_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
