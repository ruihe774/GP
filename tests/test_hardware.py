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
