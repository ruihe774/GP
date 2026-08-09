"""Capture configuration.

Defaults are tuned for the cost model described in the plan: audio input is the
expensive modality, so the mic is gated hard; frames are sampled from a cheap
constant-rate pipeline rather than encoded on demand.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "GamepadConfig",
    "AudioConfig",
    "ScreenConfig",
    "TriggerConfig",
    "CaptureConfig",
]

SticksMode = Literal["off", "intensity", "full"]
VadBackend = Literal["silero", "webrtc"]


@dataclass
class GamepadConfig:
    enabled: bool = True
    #: aggregation window; a stick push at 180 Hz is hundreds of events
    window_ms: int = 500
    #: our own radial deadzone; device-reported `flat` is far too small
    deadzone: float = 0.12
    stick_light: float = 0.55
    stick_full: float = 0.90
    trigger_light: float = 0.10
    trigger_full: float = 0.60
    #: max press duration still counted as a tap rather than a hold
    tap_max_ms: int = 220
    #: "off" ignores sticks; "intensity" feeds the scalar only (default);
    #: "full" also describes direction/magnitude in the summary
    sticks_mode: SticksMode = "intensity"
    #: rescan interval when the netlink hotplug socket is unavailable
    rescan_s: float = 2.0


@dataclass
class AudioConfig:
    enabled: bool = True
    #: webrtcdsp/webrtcechoprobe require a shared, supported rate
    dsp_rate: int = 48000
    vad_rate: int = 16000
    out_rate: int = 24000
    echo_cancel: bool = True
    noise_suppression: bool = True
    #: "low" | "moderate" | "high" | "very-high"
    noise_suppression_level: str = "moderate"
    #: The monitor tap's latency relative to the acoustic path is unknown, so
    #: the canceller has to estimate it; both of these exist for that case.
    delay_agnostic: bool = True
    extended_filter: bool = True
    #: "low" | "moderate" | "high". Keep this at moderate: the docs note a
    #: higher level "trades off double-talk performance", and talking over game
    #: audio *is* double-talk, so "high" suppresses the player along with the
    #: echo. Raise it only if game audio is leaking into segments.
    echo_suppression_level: str = "moderate"
    high_pass_filter: bool = True
    #: "silero" (ONNX, accurate, ~2 MB model) or "webrtc" (webrtcdsp's built-in
    #: detector, zero extra cost but noticeably more permissive on game audio)
    vad_backend: VadBackend = "silero"
    vad_threshold: float = 0.5
    preroll_ms: int = 300
    hangover_ms: int = 500
    min_speech_ms: int = 250
    max_segment_ms: int = 30000
    model_path: str | None = None


@dataclass
class ScreenConfig:
    enabled: bool = True
    #: target long edge; height follows from the negotiated aspect ratio
    long_edge: int = 1024
    jpeg_quality: int = 75
    #: constant cheap rate into the latest-frame holder
    pipeline_fps: int = 2
    cursor_mode: int = 2  # 1=hidden 2=embedded 4=metadata
    source_types: int = 3  # 1=monitor 2=window (bitmask)
    restore_token_path: str | None = None


@dataclass
class TriggerConfig:
    #: no two frames closer than this, except speech
    min_interval_s: float = 1.5
    heartbeat_s: float = 10.0
    gamepad_intensity: float = 0.35
    #: leading-edge throttle: fire at once, then at most once per interval
    gamepad_throttle_s: float = 2.0
    scene_threshold: float = 0.06
    scene_throttle_s: float = 3.0
    #: speech-start frames bypass min_interval_s
    on_speech: bool = True
    #: drop frames too similar to the last emitted one, whatever the trigger
    dedup_threshold: float = 0.01


@dataclass
class CaptureConfig:
    gamepad: GamepadConfig = field(default_factory=GamepadConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    triggers: TriggerConfig = field(default_factory=TriggerConfig)

    # -- (de)serialization -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaptureConfig:
        cfg = cls()
        for f in fields(cls):
            section = data.get(f.name)
            if not isinstance(section, dict):
                continue
            target = getattr(cfg, f.name)
            known = {sf.name for sf in fields(target)}
            for key, value in section.items():
                if key in known:
                    setattr(target, key, value)
                else:
                    raise ValueError(f"unknown config key: {f.name}.{key}")
        return cfg

    @classmethod
    def load(cls, path: str | Path | None) -> CaptureConfig:
        if path is None:
            return cls()
        with open(path, "rb") as fh:
            return cls.from_dict(tomllib.load(fh))

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply dotted `section.key=value` overrides from the CLI."""
        for dotted, value in overrides.items():
            section_name, _, key = dotted.partition(".")
            if not key:
                raise ValueError(f"override must be section.key: {dotted!r}")
            section = getattr(self, section_name, None)
            if section is None or not is_dataclass(section):
                raise ValueError(f"unknown config section: {section_name!r}")
            if not any(sf.name == key for sf in fields(section)):
                raise ValueError(f"unknown config key: {dotted!r}")
            setattr(section, key, value)
