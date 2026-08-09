"""Shared GStreamer helpers."""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

_init_lock = threading.Lock()
_initialized = False


def ensure_gst():
    """Import and initialize GStreamer exactly once. Returns the Gst module."""
    global _initialized
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    from gi.repository import Gst

    with _init_lock:
        if not _initialized:
            Gst.init(None)
            _initialized = True
    return Gst


def pull_bytes(sample) -> bytes:
    """Copy a GstSample's buffer out as bytes."""
    from gi.repository import Gst

    buffer = sample.get_buffer()
    ok, info = buffer.map(Gst.MapFlags.READ)
    if not ok:
        return b""
    try:
        return bytes(info.data)
    finally:
        buffer.unmap(info)


def drain_bus_errors(pipeline, name: str) -> list[str]:
    """Non-blocking bus drain. Returns fatal error descriptions."""
    from gi.repository import Gst

    errors: list[str] = []
    bus = pipeline.get_bus()
    while True:
        msg = bus.timed_pop_filtered(
            0, Gst.MessageType.ERROR | Gst.MessageType.WARNING | Gst.MessageType.EOS
        )
        if msg is None:
            break
        if msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            log.error("%s pipeline error: %s (%s)", name, err.message, debug)
            errors.append(err.message)
        elif msg.type == Gst.MessageType.EOS:
            log.error("%s pipeline reached EOS unexpectedly", name)
            errors.append("EOS")
        else:
            err, _ = msg.parse_warning()
            log.warning("%s pipeline warning: %s", name, err.message)
    return errors
