"""Event contract between Component A (capture) and Component B (realtime agent).

Every event carries `t` (monotonic seconds since session start) and `seq` (global
ordering counter); both are stamped by `CaptureBus`, not by the sources.

Events that carry bulk payloads (PCM, JPEG) keep them in `data`, which is *never*
serialized inline by default. A sink writes `data` to a blob file and records the
relative path in `blob`; `--inline` mode base64s it into the JSON instead.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field, fields
from typing import Any, Callable, ClassVar

__all__ = [
    "Event",
    "SessionStart",
    "SessionEnd",
    "GamepadConnected",
    "GamepadDisconnected",
    "GamepadActivity",
    "GamepadIdle",
    "SpeechSegment",
    "ScreenFrame",
    "AgentResponse",
    "decode",
    "EVENT_TYPES",
]


@dataclass(kw_only=True)
class Event:
    """Base event. Subclasses declare TYPE and their own body fields."""

    TYPE: ClassVar[str] = "event"
    #: extension used when this event's payload is written to a blob file
    BLOB_EXT: ClassVar[str | None] = None
    #: short name used in the blob filename
    BLOB_KIND: ClassVar[str] = "blob"

    t: float = 0.0
    seq: int = 0

    #: bulk payload, excluded from JSON; sinks route it to `blob`
    data: bytes | None = field(default=None, repr=False, compare=False)
    #: relative path of the written blob, or None
    blob: str | None = None

    # -- serialization ----------------------------------------------------

    def body(self) -> dict[str, Any]:
        """Type-specific fields, excluding the common envelope."""
        skip = {"t", "seq", "data", "blob"}
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in skip
        }

    def to_dict(self, *, inline: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {"t": round(self.t, 4), "seq": self.seq, "type": self.TYPE}
        out.update(self.body())
        if inline and self.data is not None:
            out["data_b64"] = base64.b64encode(self.data).decode("ascii")
        elif self.blob is not None:
            out["blob"] = self.blob
        return out


@dataclass(kw_only=True)
class SessionStart(Event):
    TYPE: ClassVar[str] = "session.start"

    started_at: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    devices: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class SessionEnd(Event):
    TYPE: ClassVar[str] = "session.end"

    duration_s: float = 0.0
    counters: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class GamepadConnected(Event):
    TYPE: ClassVar[str] = "gamepad.connected"

    device: str = ""
    name: str = ""
    vid: int = 0
    pid: int = 0
    path: str = ""
    layout: str = ""
    caps: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class GamepadDisconnected(Event):
    TYPE: ClassVar[str] = "gamepad.disconnected"

    device: str = ""
    name: str = ""


@dataclass(kw_only=True)
class GamepadActivity(Event):
    TYPE: ClassVar[str] = "gamepad.activity"

    device: str = ""
    window_ms: int = 0
    buttons: dict[str, dict[str, int]] = field(default_factory=dict)
    triggers: dict[str, str] = field(default_factory=dict)
    dpad: str = "idle"
    #: buttons and triggers currently held. The summary announces a hold once at
    #: each edge, so this is how a reader knows what is still down.
    holding: list[str] = field(default_factory=list)
    #: present only when sticks_mode == "full"
    sticks: dict[str, dict[str, Any]] | None = None
    intensity: float = 0.0
    apm: int = 0
    summary: str = ""

    def body(self) -> dict[str, Any]:
        b = super().body()
        if b.get("sticks") is None:
            b.pop("sticks", None)
        if not b.get("holding"):
            b.pop("holding", None)
        return b


@dataclass(kw_only=True)
class GamepadIdle(Event):
    TYPE: ClassVar[str] = "gamepad.idle"

    device: str = ""
    active_ms: int = 0


@dataclass(kw_only=True)
class SpeechSegment(Event):
    TYPE: ClassVar[str] = "speech.segment"
    BLOB_EXT: ClassVar[str | None] = "pcm"
    BLOB_KIND: ClassVar[str] = "speech"

    dur_ms: int = 0
    sample_rate: int = 24000
    encoding: str = "pcm_s16le"
    preroll_ms: int = 0
    hangover_ms: int = 0
    rms_dbfs: float = 0.0


@dataclass(kw_only=True)
class AgentResponse(Event):
    """Something the agent said. Component B's only outbound event.

    Carries the spoken audio as a blob in the same format `SpeechSegment` uses,
    so a recorded session holds both halves of the conversation and `inspect`
    can play back either.
    """

    TYPE: ClassVar[str] = "agent.response"
    BLOB_EXT: ClassVar[str | None] = "pcm"
    BLOB_KIND: ClassVar[str] = "response"

    #: which speaking rule fired: reply | react | ambient
    reason: str = ""
    transcript: str = ""
    dur_ms: int = 0
    sample_rate: int = 24000
    encoding: str = "pcm_s16le"
    #: request sent -> first audio out
    latency_ms: int = 0
    #: true when the player interrupted and the rest was never heard
    cut: bool = False
    #: what the API reported for this response
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class ScreenFrame(Event):
    TYPE: ClassVar[str] = "screen.frame"
    BLOB_EXT: ClassVar[str | None] = "jpg"
    BLOB_KIND: ClassVar[str] = "frame"

    w: int = 0
    h: int = 0
    format: str = "jpeg"
    trigger: str = ""
    scene_score: float = 0.0


EVENT_TYPES: dict[str, type[Event]] = {
    cls.TYPE: cls
    for cls in (
        SessionStart,
        SessionEnd,
        GamepadConnected,
        GamepadDisconnected,
        GamepadActivity,
        GamepadIdle,
        SpeechSegment,
        ScreenFrame,
        AgentResponse,
    )
}


def decode(
    obj: dict[str, Any],
    *,
    blob_loader: Callable[[str], bytes] | None = None,
) -> Event:
    """Rebuild an Event from a decoded JSONL line.

    `blob_loader` resolves a relative blob path to bytes; without it, `data`
    stays None and only the metadata is available.
    """
    d = dict(obj)
    type_name = d.pop("type", None)
    cls = EVENT_TYPES.get(type_name)
    if cls is None:
        raise ValueError(f"unknown event type: {type_name!r}")

    data: bytes | None = None
    if (b64 := d.pop("data_b64", None)) is not None:
        data = base64.b64decode(b64)
    blob = d.pop("blob", None)
    if data is None and blob is not None and blob_loader is not None:
        data = blob_loader(blob)

    known = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in d.items() if k in known}
    return cls(**kwargs, data=data, blob=blob)
