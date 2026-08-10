"""Speech segmentation — pure logic, no GStreamer.

Two synchronized streams arrive from a `tee`: a 16 kHz branch that drives the
VAD and a 24 kHz branch that becomes the payload (the Realtime API's rate).
Positions are tracked as sample counts in each domain and converted by the fixed
rate ratio, so the two never need wall-clock alignment.

The 24 kHz branch can lag the VAD branch by a queue's worth of buffers, so a
finalized segment waits until the payload has actually caught up to its end —
otherwise every segment loses its tail.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import numpy as np

from ..config import AudioConfig
from .vad import WINDOW_SAMPLES

log = logging.getLogger(__name__)

__all__ = ["SpeechSegmenter", "Segment"]

_INT16_SCALE = 1.0 / 32768.0


class Segment:
    __slots__ = ("pcm", "dur_ms", "speech_ms", "rms_dbfs", "forced")

    def __init__(self, pcm: bytes, dur_ms: int, speech_ms: int, rms_dbfs: float, forced: bool):
        self.pcm = pcm
        self.dur_ms = dur_ms
        self.speech_ms = speech_ms
        self.rms_dbfs = rms_dbfs
        self.forced = forced


class SpeechSegmenter:
    """VAD state machine over paired 16 kHz / 24 kHz streams."""

    def __init__(
        self,
        cfg: AudioConfig,
        vad: Callable[[np.ndarray], float] | None = None,
        *,
        vad_rate: int | None = None,
        on_speech_start: Callable[[], None] | None = None,
        on_segment: Callable[[Segment], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.vad = vad
        self.on_speech_start = on_speech_start
        self.on_segment = on_segment

        # The webrtc backend reports at the DSP rate, not the Silero rate.
        self.vad_rate = vad_rate if vad_rate is not None else cfg.vad_rate
        self._ratio = cfg.out_rate / self.vad_rate
        self._preroll = _ms_to_samples(cfg.preroll_ms, self.vad_rate)
        self._hangover = _ms_to_samples(cfg.hangover_ms, self.vad_rate)
        self._min_speech = _ms_to_samples(cfg.min_speech_ms, self.vad_rate)
        self._max_segment = _ms_to_samples(cfg.max_segment_ms, self.vad_rate)

        self._vad_tail = bytearray()
        self._vad_pos = 0  # samples consumed from the VAD branch

        self._out_buf = bytearray()
        self._out_start = 0  # sample index of _out_buf[0]

        self._speaking = False
        self._speech_start = 0
        self._speech_samples = 0
        self._silence_run = 0
        self._pending: tuple[int, int, int, bool] | None = None
        #: set after a forced split so the next chunk resumes contiguously
        self._resume_at: int | None = None

        # Diagnostics: without these, "no speech was captured" is impossible to
        # tell apart from "the VAD never came close to firing".
        self.windows = 0
        self.speech_windows = 0
        self.max_probability = 0.0
        self.dropped_short = 0

        #: how much 24 kHz history to retain
        retain_ms = cfg.max_segment_ms + cfg.preroll_ms + cfg.hangover_ms + 1000
        self._retain = _ms_to_samples(retain_ms, cfg.out_rate)

    # -- ingestion ---------------------------------------------------------

    def feed_vad(self, pcm: bytes) -> None:
        """16 kHz mono s16le from the Silero branch."""
        if self.vad is None:
            raise RuntimeError("feed_vad requires a vad callable")
        self._vad_tail.extend(pcm)
        window_bytes = WINDOW_SAMPLES * 2
        while len(self._vad_tail) >= window_bytes:
            chunk = bytes(self._vad_tail[:window_bytes])
            del self._vad_tail[:window_bytes]
            frame = np.frombuffer(chunk, dtype="<i2").astype(np.float32) * _INT16_SCALE
            try:
                probability = self.vad(frame)
            except Exception:
                log.exception("VAD inference failed; treating window as silence")
                probability = 0.0
            self.feed_probability(probability, WINDOW_SAMPLES)

    def feed_probability(self, probability: float, n_samples: int) -> None:
        """Advance the VAD clock by `n_samples` carrying this probability.

        The webrtc backend uses this directly: its detector is edge-reported, so
        the caller holds the level between transitions and pushes it per buffer.
        """
        self._vad_pos += n_samples
        self.windows += 1
        self.max_probability = max(self.max_probability, probability)
        if probability >= self.cfg.vad_threshold:
            self.speech_windows += 1
        self._step(probability, self._vad_pos, n_samples)

    def feed_out(self, pcm: bytes) -> None:
        """24 kHz mono s16le from the payload branch."""
        self._out_buf.extend(pcm)
        self._trim()
        self._try_complete()

    def close(self) -> None:
        """Flush any segment still open at shutdown."""
        if self._speaking:
            self._finalize(self._vad_pos, forced=True)
        self._try_complete(force=True)

    # -- state machine -----------------------------------------------------

    def _step(self, probability: float, end: int, n_samples: int) -> None:
        speech = probability >= self.cfg.vad_threshold
        if not self._speaking:
            if speech:
                self._speaking = True
                self._speech_start = end - n_samples
                self._speech_samples = n_samples
                self._silence_run = 0
                if self.on_speech_start is not None:
                    self.on_speech_start()
            return

        if speech:
            self._speech_samples += n_samples
            self._silence_run = 0
        else:
            self._silence_run += n_samples

        if self._silence_run >= self._hangover:
            self._finalize(end, forced=False)
        elif end - self._speech_start >= self._max_segment:
            self._finalize(end, forced=True)
            # Speech is still going: resume immediately and contiguously. No
            # second onset signal, and no pre-roll on the next chunk -- that
            # audio has already been sent once.
            self._speaking = True
            self._speech_start = end
            self._speech_samples = 0
            self._silence_run = 0

    def _finalize(self, end: int, *, forced: bool) -> None:
        continuation = self._resume_at is not None
        if self._resume_at is not None:
            start_v = self._resume_at
            self._resume_at = None
        else:
            start_v = max(0, self._speech_start - self._preroll)
        end_v = min(end, end - self._silence_run + self._hangover)
        speech_samples = self._speech_samples

        self._speaking = False
        self._speech_samples = 0
        self._silence_run = 0
        if forced:
            self._resume_at = end_v

        # A continuation is the tail of an utterance already in flight; the
        # min-speech filter exists to drop clicks, not to truncate speech.
        if not continuation and speech_samples < self._min_speech:
            self.dropped_short += 1
            log.debug(
                "dropping %d ms segment below min_speech",
                _samples_to_ms(speech_samples, self.vad_rate),
            )
            return

        self._pending = (start_v, end_v, speech_samples, forced)
        self._try_complete()

    def _try_complete(self, *, force: bool = False) -> None:
        """Emit a finalized segment once the 24 kHz branch has caught up."""
        if self._pending is None:
            return
        start_v, end_v, speech_samples, forced = self._pending

        start_o = int(round(start_v * self._ratio))
        end_o = int(round(end_v * self._ratio))
        available = self._out_start + len(self._out_buf) // 2
        if available < end_o and not force:
            return  # payload still in flight; wait for more

        pcm = self._slice(start_o, min(end_o, available))
        self._pending = None
        if not pcm:
            return

        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) * _INT16_SCALE
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        rms_dbfs = 20.0 * math.log10(rms) if rms > 1e-9 else -120.0

        segment = Segment(
            pcm=pcm,
            dur_ms=_samples_to_ms(len(pcm) // 2, self.cfg.out_rate),
            speech_ms=_samples_to_ms(speech_samples, self.vad_rate),
            rms_dbfs=round(rms_dbfs, 1),
            forced=forced,
        )
        if self.on_segment is not None:
            self.on_segment(segment)

    # -- ring buffer -------------------------------------------------------

    def _slice(self, start: int, end: int) -> bytes:
        lo = max(start, self._out_start)
        hi = min(end, self._out_start + len(self._out_buf) // 2)
        if hi <= lo:
            return b""
        return bytes(self._out_buf[(lo - self._out_start) * 2 : (hi - self._out_start) * 2])

    def _trim(self) -> None:
        excess = len(self._out_buf) // 2 - self._retain
        if excess <= 0:
            return
        # Never discard audio an open or pending segment still needs.
        floor = None
        if self._pending is not None:
            floor = int(round(self._pending[0] * self._ratio))
        elif self._speaking:
            floor = int(round(max(0, self._speech_start - self._preroll) * self._ratio))
        cut = self._out_start + excess
        if floor is not None:
            cut = min(cut, floor)
        drop = cut - self._out_start
        if drop > 0:
            del self._out_buf[: drop * 2]
            self._out_start += drop


def _ms_to_samples(ms: float, rate: int) -> int:
    return int(round(ms * rate / 1000.0))


def _samples_to_ms(samples: int, rate: int) -> int:
    return int(round(samples * 1000.0 / rate))
