"""Screen capture via the ScreenCast portal.

The pipeline runs at a constant, cheap rate into a *latest-frame holder*; the
trigger policy merely samples that holder. This decouples what the model sees
from pipeline plumbing: no JPEG encoding happens at the display's refresh rate,
and a trigger never has to wait for a frame to arrive.

A second tee branch produces a 32x32 grayscale thumbnail so scene-change
detection never has to decode a JPEG.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ScreenConfig, TriggerConfig
from ..events import ScreenFrame
from .gst_util import drain_bus_errors, ensure_gst, pull_bytes
from .portal import ScreenCastSession, open_screencast
from .triggers import TriggerPolicy

log = logging.getLogger(__name__)

THUMB = 32


def _even(value: int) -> int:
    """Encoders are happier with even dimensions."""
    return value - (value & 1)


class ScreenSource:
    name = "screen"

    def __init__(self, cfg: ScreenConfig, triggers: TriggerConfig):
        self.cfg = cfg
        self.policy = TriggerPolicy(triggers)
        self._ctx = None
        self._gst = None
        self._pipeline = None
        self._session: ScreenCastSession | None = None
        self._lock = threading.Lock()
        self._latest: tuple[bytes, int, int] | None = None
        self._thumb: np.ndarray | None = None
        self._last_sent_thumb: np.ndarray | None = None
        self._task: asyncio.Task | None = None
        self._frames_seen = 0
        self._deduped = 0

    # -- lifecycle ---------------------------------------------------------

    async def start(self, ctx) -> None:
        self._ctx = ctx
        Gst = ensure_gst()
        self._gst = Gst

        loop = asyncio.get_running_loop()
        self._session = await loop.run_in_executor(
            None,
            lambda: open_screencast(
                source_types=self.cfg.source_types,
                cursor_mode=self.cfg.cursor_mode,
                token_path=(
                    Path(self.cfg.restore_token_path)
                    if self.cfg.restore_token_path
                    else None
                ),
            ),
        )
        log.info(
            "screencast: node=%s size=%sx%s restored=%s",
            self._session.node_id,
            self._session.width,
            self._session.height,
            self._session.restore_token is not None,
        )

        self._pipeline = self._build_pipeline(self._session)
        self._pipeline.set_state(Gst.State.PLAYING)
        result, _, _ = self._pipeline.get_state(10 * Gst.SECOND)
        if result == Gst.StateChangeReturn.FAILURE:
            drain_bus_errors(self._pipeline, "screen")
            raise RuntimeError("screen pipeline failed to start")

        ctx.on_signal("gamepad.intensity", self.policy.on_intensity)
        ctx.on_signal("speech.start", self.policy.on_speech_start)
        self._task = asyncio.create_task(self._sample_loop(), name="screen-sample")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._pipeline is not None:
            self._pipeline.set_state(self._gst.State.NULL)
            self._pipeline = None
        if self._session is not None:
            self._session.close()
            self._session = None

    def describe(self) -> dict[str, Any]:
        return {
            "session": self._session.describe() if self._session else None,
            "long_edge": self.cfg.long_edge,
            "jpeg_quality": self.cfg.jpeg_quality,
            "pipeline_fps": self.cfg.pipeline_fps,
            "frames_seen": self._frames_seen,
            "frames_emitted": sum(self.policy.counts.values()),
            "frames_deduped": self._deduped,
            "by_trigger": dict(self.policy.counts),
        }

    # -- pipeline ----------------------------------------------------------

    def _scale_caps(self, session: ScreenCastSession) -> str:
        """Constrain the long edge, preserving aspect ratio.

        The portal may hand back a monitor, a single window, or a virtual
        display at any size, so the target is computed from the negotiated
        source size rather than assumed. Both dimensions are fixed explicitly:
        pinning only one and forcing pixel-aspect-ratio makes videoscale
        fixate the other to 1.
        """
        edge = self.cfg.long_edge
        width, height = session.width, session.height
        if not width or not height:
            # Size unknown: constrain width only and let DAR carry the rest.
            return f"video/x-raw,width={edge}"

        scale = min(1.0, edge / max(width, height))
        target_w = max(2, _even(round(width * scale)))
        target_h = max(2, _even(round(height * scale)))
        return f"video/x-raw,width={target_w},height={target_h}"

    def _build_pipeline(self, session: ScreenCastSession):
        Gst = self._gst
        # dup the fd: pipewiresrc takes ownership of what it is given.
        fd = os.dup(session.fd)
        description = (
            f"pipewiresrc fd={fd} path={session.node_id} "
            f"! videorate max-rate={self.cfg.pipeline_fps} drop-only=true "
            f"! videoconvert ! tee name=vt "
            f"vt. ! queue leaky=downstream max-size-buffers=2 "
            f"! videoscale add-borders=false ! {self._scale_caps(session)} "
            f"! jpegenc quality={self.cfg.jpeg_quality} "
            f"! appsink name=framesink emit-signals=true sync=false "
            f"max-buffers=1 drop=true "
            f"vt. ! queue leaky=downstream max-size-buffers=2 "
            f"! videoscale add-borders=false ! videoconvert "
            f"! video/x-raw,format=GRAY8,width={THUMB},height={THUMB} "
            f"! appsink name=thumbsink emit-signals=true sync=false "
            f"max-buffers=1 drop=true"
        )
        pipeline = Gst.parse_launch(description)
        pipeline.get_by_name("framesink").connect("new-sample", self._on_frame)
        pipeline.get_by_name("thumbsink").connect("new-sample", self._on_thumb)
        return pipeline

    # -- streaming-thread callbacks ---------------------------------------

    def _on_frame(self, sink):
        Gst = self._gst
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        data = pull_bytes(sample)
        if data:
            structure = sample.get_caps().get_structure(0)
            ok_w, width = structure.get_int("width")
            ok_h, height = structure.get_int("height")
            with self._lock:
                self._latest = (data, width if ok_w else 0, height if ok_h else 0)
                self._frames_seen += 1
        return Gst.FlowReturn.OK

    def _on_thumb(self, sink):
        Gst = self._gst
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        data = pull_bytes(sample)
        # GStreamer pads GRAY8 rows to a 4-byte stride.
        stride = (THUMB + 3) & ~3
        if len(data) >= stride * THUMB:
            arr = np.frombuffer(data, dtype=np.uint8, count=stride * THUMB)
            arr = arr.reshape(THUMB, stride)[:, :THUMB].astype(np.int16)
            with self._lock:
                self._thumb = arr
        return Gst.FlowReturn.OK

    # -- sampling ----------------------------------------------------------

    def _scene_score(self, thumb: np.ndarray | None) -> float:
        if thumb is None or self._last_sent_thumb is None:
            return 1.0
        return float(np.abs(thumb - self._last_sent_thumb).mean() / 255.0)

    async def _sample_loop(self) -> None:
        interval = 1.0 / max(1, self.cfg.pipeline_fps)
        while True:
            await asyncio.sleep(interval)
            if self._pipeline is not None:
                drain_bus_errors(self._pipeline, "screen")

            with self._lock:
                latest = self._latest
                thumb = None if self._thumb is None else self._thumb.copy()

            if latest is None:
                continue

            score = self._scene_score(thumb)
            trigger = self.policy.decide(scene_score=score)
            if trigger is None:
                continue

            # A frame the model has already seen is not worth paying for again,
            # whichever trigger asked for it.
            if self._last_sent_thumb is not None and score < self.policy.cfg.dedup_threshold:
                self._deduped += 1
                continue

            data, width, height = latest
            self._last_sent_thumb = thumb
            self.policy.mark_emitted(trigger)
            self._ctx.emit(
                ScreenFrame(
                    data=data,
                    w=width,
                    h=height,
                    format="jpeg",
                    trigger=trigger,
                    scene_score=round(score, 4),
                )
            )
