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
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import CaptureConfig
from .events import (
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
    cfg = CaptureConfig.load(getattr(args, "config", None))
    try:
        cfg.apply_overrides(_parse_set(getattr(args, "set", []) or []))
    except ValueError as exc:
        raise SystemExit(str(exc))
    for section, flag in (("gamepad", "no_gamepad"), ("audio", "no_audio"), ("screen", "no_screen")):
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
        raise SystemExit(str(exc))
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
    print(f"  display         {os.environ.get('WAYLAND_DISPLAY') or os.environ.get('DISPLAY') or 'none'}")
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
        print(f"    sticks        left={layout.left_stick is not None} right={layout.right_stick is not None}")
        print(f"    triggers      {', '.join(sorted(layout.triggers)) or 'digital only'}")
        print(f"    dpad          {'hat' if layout.hat else 'buttons'}")
    if not found:
        print("  none detected")
    if unreadable:
        print(f"  ({unreadable} input nodes not readable by this user -- normal for keyboards/mice)")

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
                name = {("x", -1): "left", ("x", 1): "right", ("y", -1): "up", ("y", 1): "down"}
                direction = name.get((axis, raw.value), str(raw.value))
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
    from .bus import CaptureBus

    sources = []
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

    started = datetime.now(timezone.utc)
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

    duration = (datetime.now(timezone.utc) - started).total_seconds()
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
    events = list(read_session(directory, load_blobs=args.wav or args.contact_sheet))
    if not events:
        print("no events")
        return 1

    manifest = _read_manifest(directory)
    estimate = Estimate()
    speech_ms = 0.0
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
            line = f"scr  frame {event.w}x{event.h} via {event.trigger} (scene {event.scene_score:.3f})"
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
    if triggers:
        print(f"  frames by trigger  {', '.join(f'{k}={v}' for k, v in sorted(triggers.items()))}")
    if manifest.get("dropped_events"):
        print(f"  dropped events  {manifest['dropped_events']}")

    cost = estimate.cost()
    print("\nestimated input cost (approximate -- see tokens.py)")
    print(f"  audio           {estimate.audio_tokens:>8d} tok  ${cost['audio_usd']:.4f}")
    print(f"  images          {estimate.image_tokens:>8d} tok  ${cost['image_usd']:.4f}  ({estimate.frames} frames)")
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
    elif speech_ms:
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
    return 0


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
    out = directory / "wav"
    out.mkdir(exist_ok=True)
    written = 0
    for event in events:
        if not isinstance(event, SpeechSegment) or not event.data:
            continue
        path = out / f"{event.seq:06d}-speech.wav"
        with wave.open(str(path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(event.sample_rate)
            fh.writeframes(event.data)
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


# -- commentate ------------------------------------------------------------


def cmd_commentate(args) -> int:
    cfg = _build_config(args)
    if args.model:
        cfg.agent.model = args.model
    if args.voice:
        cfg.agent.voice = args.voice
    if args.image_detail:
        cfg.agent.image_detail = args.image_detail
    if args.no_playback or args.dry_run:
        cfg.agent.playback = False
    if args.replay is None and args.speed == 0.0:
        raise SystemExit("--speed 0 only makes sense with --replay")
    return asyncio.run(_commentate(cfg, args))


async def _commentate(cfg: CaptureConfig, args) -> int:
    from .agent.playback import make_player
    from .agent.session import CommentaryAgent, ReplayClock, drive_from_bus, drive_from_session
    from .agent.transport import OpenAITransport, RecordingTransport

    out_dir = Path(args.output) if args.output else None
    deterministic = args.replay is not None and args.speed == 0.0
    clock = ReplayClock() if deterministic else None

    if args.dry_run:
        transport = RecordingTransport()
    else:
        from .agent.env import MissingAPIKey, load_api_key

        try:
            transport = OpenAITransport(cfg.agent.model, load_api_key())
        except MissingAPIKey as exc:
            print(str(exc), file=sys.stderr)
            return 1

    player = make_player(
        cfg.agent.playback, clock or time.monotonic, sink=cfg.agent.audio_sink
    )
    agent = CommentaryAgent(
        cfg,
        transport,
        player,
        clock=clock or time.monotonic,
        log_path=(out_dir / "agent.jsonl") if out_dir else None,
        on_line=None if args.quiet else _print_line,
    )

    mode = "dry run" if args.dry_run else cfg.agent.model
    source = f"replay {args.replay}" if args.replay else "live capture"
    print(f"commentating: {source} -> {mode}")
    if not args.dry_run and not cfg.agent.playback:
        print("(playback disabled: nothing will come out of the speakers)")

    await agent.start()
    try:
        if deterministic:
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

    _print_agent_report(agent, args, out_dir)
    return 0


async def _build_bus(cfg: CaptureConfig, args):
    """Either a recording played back in real time, or the real thing."""
    from .bus import CaptureBus

    if args.replay is not None:
        from .replay import ReplaySource

        source = ReplaySource(args.replay, speed=args.speed)
        bus = CaptureBus([source])
        await bus.start()
        # Stop the bus once the recording runs out, so the run terminates.
        asyncio.create_task(_stop_when_finished(source, bus))
        return bus

    sources = []
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
    )
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
        "        which is what agent.keep_images and agent.prune_after_s control."
    )


# -- replay ----------------------------------------------------------------


def cmd_replay(args) -> int:
    directory = Path(args.directory)
    events = list(read_session(directory, load_blobs=True))
    return asyncio.run(_replay(events, args.speed))


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
        "--speed",
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
    commentate.add_argument("--voice", help="override agent.voice")
    commentate.add_argument(
        "--image-detail", choices=["auto", "low", "high"], help="override agent.image_detail"
    )
    commentate.add_argument(
        "--no-playback", action="store_true", help="do not open the audio sink"
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

    replay = sub.add_parser("replay", help="re-emit a session with original timing")
    replay.add_argument("directory")
    replay.add_argument("--speed", type=float, default=1.0)
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
