"""Microphone capture with acoustic echo cancellation and VAD gating.

Two PipeWire streams, both resolved to defaults so nothing keys off a device
name (WirePlumber re-links untargeted streams when the default changes, so this
follows the user switching headsets):

  reference -- the default *sink's monitor*, i.e. everything the system plays.
               This is the right AEC reference: the game is coming out of the
               speakers into the mic too, not just the agent's voice. On a
               combined speaker/mic device that echo would otherwise be
               detected as speech and billed as audio tokens.

  capture   -- the default source, cleaned by webrtcdsp, then split into a VAD
               branch and a 24 kHz payload branch (the Realtime API's rate).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from typing import Any

from ..config import AudioConfig
from ..events import SpeechSegment
from .gst_util import drain_bus_errors, ensure_gst, pull_bytes
from .segmenter import Segment, SpeechSegmenter

log = logging.getLogger(__name__)

SIGNAL_SPEECH_START = "speech.start"


def _bool(value: bool) -> str:
    return "true" if value else "false"


class AudioSource:
    name = "audio"

    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self._ctx = None
        self._gst = None
        self._pipeline = None
        self._lock = threading.Lock()
        self._segmenter: SpeechSegmenter | None = None
        self._vad = None
        self._task: asyncio.Task | None = None
        self._aec_active = False
        self._webrtc_voice = False
        self._voice_transitions = 0
        self._segments = 0
        self._peak = 0.0

    # -- lifecycle ---------------------------------------------------------

    async def start(self, ctx) -> None:
        self._ctx = ctx
        Gst = ensure_gst()
        self._gst = Gst

        vad_rate = self.cfg.dsp_rate if self.cfg.vad_backend == "webrtc" else self.cfg.vad_rate
        if self.cfg.vad_backend == "silero":
            from .vad import SileroVAD

            self._vad = SileroVAD(self.cfg.model_path, sample_rate=self.cfg.vad_rate)
            log.info("VAD backend: silero (%s)", self._vad.model_path)
        else:
            log.info("VAD backend: webrtc (webrtcdsp built-in detector)")

        self._segmenter = SpeechSegmenter(
            self.cfg,
            self._vad,
            vad_rate=vad_rate,
            on_speech_start=self._on_speech_start,
            on_segment=self._on_segment,
        )

        self._pipeline = self._build_pipeline(aec=self.cfg.echo_cancel)
        if not await self._play():
            # webrtcdsp refuses to start if echo-cancel is on and the probe is
            # missing (e.g. no default sink), so retry once without AEC.
            log.warning("audio pipeline failed with AEC; retrying without it")
            await self._teardown()
            self._pipeline = self._build_pipeline(aec=False)
            if not await self._play():
                raise RuntimeError("audio pipeline failed to start")
        else:
            self._aec_active = self.cfg.echo_cancel

        self._task = asyncio.create_task(self._watch_bus(), name="audio-bus")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self._teardown()
        if self._segmenter is not None:
            with self._lock:
                self._segmenter.close()

    async def _teardown(self) -> None:
        if self._pipeline is not None:
            self._pipeline.set_state(self._gst.State.NULL)
            self._pipeline = None

    def describe(self) -> dict[str, Any]:
        import math

        seg = self._segmenter
        peak_dbfs = round(20 * math.log10(self._peak), 1) if self._peak > 1e-9 else -120.0
        info: dict[str, Any] = {
            "vad_backend": self.cfg.vad_backend,
            "echo_cancel_active": self._aec_active,
            "echo_suppression_level": self.cfg.echo_suppression_level,
            "reference": "default sink monitor (stream.capture.sink=true)",
            "capture": "default source",
            "out_rate": self.cfg.out_rate,
            "segments": self._segments,
            "input_peak_dbfs": peak_dbfs,
            "voice_transitions": self._voice_transitions,
        }
        if seg is not None:
            info.update(
                vad_windows=seg.windows,
                vad_speech_windows=seg.speech_windows,
                vad_max_probability=round(seg.max_probability, 3),
                vad_threshold=self.cfg.vad_threshold,
                segments_dropped_short=seg.dropped_short,
            )
        return info

    # -- pipeline ----------------------------------------------------------

    def _build_pipeline(self, *, aec: bool):
        Gst = self._gst
        cfg = self.cfg
        caps_dsp = f"audio/x-raw,format=S16LE,rate={cfg.dsp_rate},channels=1"
        chains: list[str] = []

        if aec:
            chains.append(
                f'pipewiresrc name=ref stream-properties="props,stream.capture.sink=true" '
                f"! audio/x-raw ! audioconvert ! audioresample ! {caps_dsp} "
                f"! webrtcechoprobe name=aecprobe ! fakesink sync=false"
            )

        dsp = [
            "webrtcdsp name=dsp",
            f"echo-cancel={_bool(aec)}",
            f"noise-suppression={_bool(cfg.noise_suppression)}",
            f"noise-suppression-level={cfg.noise_suppression_level}",
            f"high-pass-filter={_bool(cfg.high_pass_filter)}",
            f"voice-detection={_bool(cfg.vad_backend == 'webrtc')}",
        ]
        if aec:
            dsp += [
                "probe=aecprobe",
                f"echo-suppression-level={cfg.echo_suppression_level}",
                f"delay-agnostic={_bool(cfg.delay_agnostic)}",
                f"extended-filter={_bool(cfg.extended_filter)}",
            ]

        mic = (
            f"pipewiresrc name=mic ! audio/x-raw ! audioconvert ! audioresample ! {caps_dsp} "
            f"! {' '.join(dsp)} ! tee name=t "
            f"t. ! queue ! audioresample "
            f"! audio/x-raw,format=S16LE,rate={cfg.out_rate},channels=1 "
            f"! appsink name=outsink emit-signals=true sync=false max-buffers=64 drop=false"
        )
        if cfg.vad_backend == "silero":
            mic += (
                f" t. ! queue ! audioresample "
                f"! audio/x-raw,format=S16LE,rate={cfg.vad_rate},channels=1 "
                f"! appsink name=vadsink emit-signals=true sync=false max-buffers=64 drop=false"
            )
        chains.append(mic)

        pipeline = Gst.parse_launch("\n".join(chains))
        pipeline.get_by_name("outsink").connect("new-sample", self._on_out_sample)
        if cfg.vad_backend == "silero":
            pipeline.get_by_name("vadsink").connect("new-sample", self._on_vad_sample)
        else:
            # The detector is edge-reported: webrtcdsp attaches a level meta
            # only when voice activity flips, so we hold the level in between.
            pad = pipeline.get_by_name("dsp").get_static_pad("src")
            pad.add_probe(Gst.PadProbeType.BUFFER, self._on_dsp_buffer)
        return pipeline

    async def _play(self) -> bool:
        Gst = self._gst
        self._pipeline.set_state(Gst.State.PLAYING)
        result, _, _ = self._pipeline.get_state(5 * Gst.SECOND)
        if result == Gst.StateChangeReturn.FAILURE:
            drain_bus_errors(self._pipeline, "audio")
            return False
        await asyncio.sleep(0.05)
        return not drain_bus_errors(self._pipeline, "audio")

    async def _watch_bus(self) -> None:
        while True:
            await asyncio.sleep(0.5)
            if self._pipeline is None:
                continue
            if drain_bus_errors(self._pipeline, "audio"):
                log.error("audio pipeline is in error; capture has stopped")

    # -- streaming-thread callbacks ---------------------------------------

    def _on_vad_sample(self, sink):
        Gst = self._gst
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        data = pull_bytes(sample)
        if data:
            with self._lock:
                self._segmenter.feed_vad(data)
        return Gst.FlowReturn.OK

    def _on_out_sample(self, sink):
        Gst = self._gst
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        data = pull_bytes(sample)
        if data:
            self._track_level(data)
            with self._lock:
                self._segmenter.feed_out(data)
        return Gst.FlowReturn.OK

    def _track_level(self, pcm: bytes) -> None:
        """Running peak of what actually reaches the segmenter.

        A dead microphone and a microphone the VAD simply never liked look the
        same from the outside; this separates them.
        """
        import numpy as np

        samples = np.frombuffer(pcm, dtype="<i2")
        if not samples.size:
            return
        peak = float(np.abs(samples).max()) / 32768.0
        if peak > self._peak:
            self._peak = peak

    def _on_dsp_buffer(self, pad, info):
        Gst = self._gst
        from gi.repository import GstAudio

        buffer = info.get_buffer()
        meta = GstAudio.buffer_get_audio_level_meta(buffer)
        if meta is not None:
            self._webrtc_voice = bool(meta.voice_activity)
            self._voice_transitions += 1
        samples = buffer.get_size() // 2
        if samples:
            with self._lock:
                self._segmenter.feed_probability(1.0 if self._webrtc_voice else 0.0, samples)
        return Gst.PadProbeReturn.OK

    # -- segmenter callbacks ----------------------------------------------

    def _on_speech_start(self) -> None:
        # Fired at detection time, not at segment end, so a screen frame
        # captures what was on screen when the player started talking.
        if self._ctx is not None:
            self._ctx.signal(SIGNAL_SPEECH_START, None)

    def _on_segment(self, segment: Segment) -> None:
        self._segments += 1
        if self._ctx is None:
            return
        self._ctx.emit(
            SpeechSegment(
                data=segment.pcm,
                dur_ms=segment.dur_ms,
                sample_rate=self.cfg.out_rate,
                encoding="pcm_s16le",
                preroll_ms=self.cfg.preroll_ms,
                hangover_ms=self.cfg.hangover_ms,
                rms_dbfs=segment.rms_dbfs,
            )
        )
