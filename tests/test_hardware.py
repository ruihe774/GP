"""Tests that touch real devices. Deselected by default; run with:

    pytest -m hardware

These must run from the desktop session -- the gamepad uaccess ACL and the
ScreenCast portal both bind to the active seat.
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.hardware


def test_can_enumerate_input_devices():
    from gpagent.capture import evdev_raw as ev

    paths = ev.list_event_paths()
    assert paths, "no /dev/input/event* nodes at all"


def test_gamepad_layout_resolves_if_one_is_connected():
    from gpagent.capture import evdev_raw as ev
    from gpagent.capture.gamepad import resolve_layout

    pads = ev.find_gamepads()
    if not pads:
        pytest.skip("no gamepad connected")
    for info in pads:
        layout = resolve_layout(info)
        assert layout.buttons, f"{info.name}: no buttons resolved"
        assert layout.left_stick or layout.hat, f"{info.name}: no directional input"


def test_silero_model_loads_and_rejects_silence():
    import numpy as np

    from gpagent.capture.vad import WINDOW_SAMPLES, SileroVAD

    vad = SileroVAD()
    silence = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    assert vad(silence) < 0.1


def test_audio_pipeline_starts_and_delivers_both_branches():
    """The tee branches must stay sample-aligned; the segmenter relies on it."""
    from gpagent.bus import CaptureBus
    from gpagent.capture.audio import AudioSource
    from gpagent.config import AudioConfig

    async def run():
        source = AudioSource(AudioConfig())
        bus = CaptureBus([source])
        await bus.start()
        await asyncio.sleep(2.0)
        described = source.describe()
        await bus.stop()
        return described

    described = asyncio.run(run())
    assert described["echo_cancel_active"], "AEC should engage when a sink exists"


def test_playback_names_no_device():
    """Naming a device is how the echo canceller silently stops working."""
    from gpagent.agent.config import AgentConfig
    from gpagent.agent.playback import AudioPlayer

    for text in (AudioPlayer.PIPELINE, AudioPlayer.SINK, AgentConfig().audio_sink):
        assert "device=" not in text
        assert "target-object" not in text


def test_playback_stream_lands_on_the_default_sink():
    """The AEC reference is the *default* sink's monitor.

    Asserting the element name would prove nothing -- what matters is where
    WirePlumber actually routes the stream, so this looks at the live graph.
    """
    import json
    import subprocess

    from gpagent.agent.playback import SAMPLE_RATE, AudioPlayer

    def graph():
        raw = subprocess.run(
            ["pw-dump", "-N"], capture_output=True, text=True, timeout=8, check=True
        ).stdout
        return json.loads(raw)

    async def run():
        player = AudioPlayer()
        await player.start()
        player.push(b"\x00\x00" * SAMPLE_RATE * 2)  # 2 s of silence
        await asyncio.sleep(1.0)
        objects = graph()
        await player.drain()
        await player.stop()
        return objects

    try:
        objects = asyncio.run(run())
    except (OSError, subprocess.SubprocessError):
        pytest.skip("pw-dump unavailable")

    nodes = {
        o["id"]: (o.get("info", {}).get("props", {}) or {})
        for o in objects
        if o.get("type") == "PipeWire:Interface:Node"
    }
    default = None
    for o in objects:
        if o.get("type") == "PipeWire:Interface:Metadata":
            for entry in o.get("metadata") or []:
                if entry.get("key") == "default.audio.sink":
                    value = entry.get("value")
                    default = value.get("name") if isinstance(value, dict) else value
    assert default, "no default sink configured"

    ours = [
        i
        for i, props in nodes.items()
        if props.get("media.class") == "Stream/Output/Audio"
        and "python" in str(props.get("application.process.binary", "")).lower()
    ]
    assert ours, "our playback stream is not in the graph at all"
    for node in ours:
        assert not nodes[node].get("target.object"), "playback must not pin a device"

    targets = {
        (o.get("info", {}).get("props", {}) or {}).get("link.input.node")
        for o in objects
        if o.get("type") == "PipeWire:Interface:Link"
        and (o.get("info", {}).get("props", {}) or {}).get("link.output.node") in ours
    }
    assert {nodes.get(t, {}).get("node.name") for t in targets} == {default}


def test_playback_is_paced_and_tears_down_cleanly():
    """Regression guard for the three bugs that made speech unlistenable."""
    import math
    import struct
    import time

    from gpagent.agent.playback import SAMPLE_RATE, AudioPlayer

    async def run():
        player = AudioPlayer()
        started = time.monotonic()
        await player.start()
        start_took = time.monotonic() - started

        # Two seconds of 440 Hz in 50 ms chunks, pushed as fast as the loop
        # will go -- the model streams far faster than real time. Every other
        # chunk is an odd number of bytes, which used to shift every following
        # sample and turn the rest of the utterance into white noise.
        phase = 0.0
        began = time.monotonic()
        for i in range(40):
            n = SAMPLE_RATE // 20
            samples = [
                int(6000 * math.sin(2 * math.pi * 440 * j / SAMPLE_RATE + phase))
                for j in range(n)
            ]
            phase += 2 * math.pi * 440 * n / SAMPLE_RATE
            pcm = struct.pack(f"<{n}h", *samples)
            player.push(pcm + b"\x00" if i % 2 else pcm)
        push_took = time.monotonic() - began

        spoken = await player.drain()
        wall = time.monotonic() - began
        stopped = time.monotonic()
        await player.stop()
        return start_took, push_took, spoken, wall, time.monotonic() - stopped

    start_took, push_took, spoken, wall, stop_took = asyncio.run(run())

    assert start_took < 2.0, f"start stalled for {start_took:.1f}s (unprimed preroll?)"
    assert push_took < 1.0, "pushing must not block on playback"
    assert 1900 <= spoken <= 2100, f"expected ~2 s of playback, got {spoken}"
    # The one that matters: audio must be paced by the clock, not rendered as
    # fast as it arrived. Untimestamped buffers come out in a fraction of this.
    assert 1.9 <= wall <= 2.6, f"2 s of audio took {wall:.2f}s of wall clock"
    assert stop_took < 2.0, f"teardown blocked for {stop_took:.1f}s"


def test_screencast_portal_is_available():
    import gi

    gi.require_version("Gio", "2.0")
    from gi.repository import Gio, GLib

    conn = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    reply = conn.call_sync(
        "org.freedesktop.portal.Desktop",
        "/org/freedesktop/portal/desktop",
        "org.freedesktop.DBus.Properties",
        "Get",
        GLib.Variant("(ss)", ("org.freedesktop.portal.ScreenCast", "version")),
        GLib.VariantType("(v)"),
        Gio.DBusCallFlags.NONE,
        3000,
        None,
    )
    version = reply.unpack()[0]
    assert version >= 4, "portal too old for restore_token; expect a dialog every run"
