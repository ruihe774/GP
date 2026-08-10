"""Speech segmentation: boundaries, pre-roll, hangover, min duration, lag."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from gpagent.capture.segmenter import Segment, SpeechSegmenter
from gpagent.config import AudioConfig

VAD_RATE = 16000
OUT_RATE = 24000
WINDOW = 512
OUT_PER_WINDOW = WINDOW * OUT_RATE // VAD_RATE  # 768


def config(**kwargs: Any) -> AudioConfig:
    base: dict[str, Any] = dict(
        vad_rate=VAD_RATE,
        out_rate=OUT_RATE,
        preroll_ms=300,
        hangover_ms=500,
        min_speech_ms=250,
        max_segment_ms=30000,
        vad_threshold=0.5,
    )
    base.update(kwargs)
    return AudioConfig(**base)


class Harness:
    """Drives both branches in lockstep with identifiable payload samples."""

    def __init__(self, cfg: AudioConfig, out_lag_windows: int = 0):
        self.segments: list[Segment] = []
        self.starts = 0
        self.seg = SpeechSegmenter(
            cfg,
            vad=None,
            on_speech_start=self._on_start,
            on_segment=self.segments.append,
        )
        self.out_position = 0
        self.pending: list[bytes] = []
        self.out_lag = out_lag_windows

    def _on_start(self):
        self.starts += 1

    def _payload(self, n: int) -> bytes:
        idx = np.arange(self.out_position, self.out_position + n)
        self.out_position += n
        return ((idx % 30000) - 15000).astype("<i2").tobytes()

    def drive(self, probability: float, windows: int) -> None:
        for _ in range(windows):
            self.pending.append(self._payload(OUT_PER_WINDOW))
            # Release payload only after `out_lag` windows, emulating the
            # queue depth difference between the two tee branches.
            if len(self.pending) > self.out_lag:
                self.seg.feed_out(self.pending.pop(0))
            self.seg.feed_probability(probability, WINDOW)

    def flush_pending(self) -> None:
        while self.pending:
            self.seg.feed_out(self.pending.pop(0))


class TestBoundaries:
    def test_single_utterance(self):
        h = Harness(config())
        h.drive(0.0, 200)
        h.drive(0.9, 40)   # 1.28 s of speech
        h.drive(0.0, 20)   # past the 500 ms hangover
        h.flush_pending()

        assert len(h.segments) == 1
        seg = h.segments[0]
        # 1280 ms speech + 300 ms pre-roll + 500 ms hangover
        assert seg.dur_ms == pytest.approx(2080, abs=2)
        assert seg.speech_ms == pytest.approx(1280, abs=2)
        assert not seg.forced

    def test_speech_start_fires_once_at_onset(self):
        h = Harness(config())
        h.drive(0.0, 20)
        assert h.starts == 0
        h.drive(0.9, 40)
        assert h.starts == 1, "must fire at onset, not at segment end"
        assert not h.segments, "segment is not finished yet"
        h.drive(0.0, 20)
        h.flush_pending()
        assert h.starts == 1
        assert len(h.segments) == 1

    def test_two_utterances_produce_two_segments(self):
        h = Harness(config())
        h.drive(0.0, 50)
        h.drive(0.9, 30)
        h.drive(0.0, 30)
        h.drive(0.9, 30)
        h.drive(0.0, 30)
        h.flush_pending()
        assert len(h.segments) == 2
        assert h.starts == 2

    def test_brief_dip_does_not_split_a_segment(self):
        h = Harness(config(hangover_ms=500))
        h.drive(0.0, 20)
        h.drive(0.9, 20)
        h.drive(0.0, 5)    # 160 ms gap, under the 500 ms hangover
        h.drive(0.9, 20)
        h.drive(0.0, 20)
        h.flush_pending()
        assert len(h.segments) == 1


class TestPreroll:
    def test_preroll_audio_precedes_the_onset(self):
        h = Harness(config(preroll_ms=300))
        h.drive(0.0, 100)
        h.drive(0.9, 40)
        h.drive(0.0, 20)
        h.flush_pending()

        seg = h.segments[0]
        speech_start_v = 100 * WINDOW
        expected_start_o = int(round((speech_start_v - 300 * VAD_RATE // 1000) * 1.5))
        first = np.frombuffer(seg.pcm[:2], dtype="<i2")[0]
        assert first == (expected_start_o % 30000) - 15000

    def test_zero_preroll_starts_at_onset(self):
        h = Harness(config(preroll_ms=0))
        h.drive(0.0, 100)
        h.drive(0.9, 40)
        h.drive(0.0, 20)
        h.flush_pending()
        expected = int(round(100 * WINDOW * 1.5))
        first = np.frombuffer(h.segments[0].pcm[:2], dtype="<i2")[0]
        assert first == (expected % 30000) - 15000

    def test_preroll_clamped_at_stream_start(self):
        h = Harness(config(preroll_ms=300))
        h.drive(0.9, 40)   # speech from the very first window
        h.drive(0.0, 20)
        h.flush_pending()
        assert len(h.segments) == 1
        assert h.segments[0].dur_ms > 0


class TestFiltering:
    def test_segment_below_min_speech_is_dropped(self):
        h = Harness(config(min_speech_ms=250))
        h.drive(0.0, 50)
        h.drive(0.9, 3)    # 96 ms -- a click, not speech
        h.drive(0.0, 20)
        h.flush_pending()
        assert h.segments == []

    def test_segment_at_min_speech_is_kept(self):
        h = Harness(config(min_speech_ms=250))
        h.drive(0.0, 50)
        h.drive(0.9, 9)    # 288 ms
        h.drive(0.0, 20)
        h.flush_pending()
        assert len(h.segments) == 1

    def test_silence_alone_produces_nothing(self):
        h = Harness(config())
        h.drive(0.0, 500)
        h.flush_pending()
        assert h.segments == []
        assert h.starts == 0

    def test_threshold_is_respected(self):
        h = Harness(config(vad_threshold=0.8))
        h.drive(0.0, 20)
        h.drive(0.6, 40)   # above 0.5 but below the configured 0.8
        h.drive(0.0, 20)
        h.flush_pending()
        assert h.segments == []


class TestLongSpeech:
    def test_max_segment_forces_a_flush(self):
        h = Harness(config(max_segment_ms=1000))
        h.drive(0.0, 10)
        h.drive(0.9, 100)  # 3.2 s of continuous speech
        h.flush_pending()
        assert len(h.segments) >= 2
        assert h.segments[0].forced
        # The cap bounds the speech span; pre-roll is added on top of it.
        assert h.segments[0].dur_ms == pytest.approx(1300, abs=50)

    def test_forced_split_only_signals_onset_once(self):
        h = Harness(config(max_segment_ms=1000))
        h.drive(0.0, 10)
        h.drive(0.9, 100)
        h.flush_pending()
        assert h.starts == 1, "a forced split is not a new utterance"

    def test_forced_split_chunks_are_contiguous(self):
        """No overlap (duplicate tokens) and no gap (lost speech) at the seam."""
        h = Harness(config(max_segment_ms=1000))
        h.drive(0.0, 10)
        h.drive(0.9, 100)
        h.flush_pending()

        first, second = h.segments[0], h.segments[1]
        last_of_first = np.frombuffer(first.pcm[-2:], dtype="<i2")[0]
        first_of_second = np.frombuffer(second.pcm[:2], dtype="<i2")[0]
        expected_next = int(last_of_first) + 1
        if expected_next > 14999:
            expected_next -= 30000
        assert first_of_second == expected_next

    def test_continuation_is_not_dropped_by_min_speech(self):
        # The tail after a forced split can be short; it must survive.
        h = Harness(config(max_segment_ms=1000, min_speech_ms=250))
        h.drive(0.0, 10)
        h.drive(0.9, 34)   # just past one forced split
        h.drive(0.0, 20)
        h.flush_pending()
        assert len(h.segments) == 2
        assert h.segments[1].dur_ms > 0


class TestPayloadLag:
    def test_late_payload_still_yields_the_full_tail(self):
        """The 24 kHz branch lags the VAD branch; the tail must not be clipped."""
        lagged = Harness(config(), out_lag_windows=12)
        lagged.drive(0.0, 100)
        lagged.drive(0.9, 40)
        lagged.drive(0.0, 20)
        lagged.flush_pending()

        prompt = Harness(config(), out_lag_windows=0)
        prompt.drive(0.0, 100)
        prompt.drive(0.9, 40)
        prompt.drive(0.0, 20)
        prompt.flush_pending()

        assert len(lagged.segments) == 1
        assert lagged.segments[0].dur_ms == prompt.segments[0].dur_ms
        assert lagged.segments[0].pcm == prompt.segments[0].pcm

    def test_close_flushes_an_open_segment(self):
        h = Harness(config())
        h.drive(0.0, 20)
        h.drive(0.9, 40)   # still speaking when the session ends
        h.flush_pending()
        assert h.segments == []
        h.seg.close()
        assert len(h.segments) == 1
        assert h.segments[0].forced


class TestSileroPath:
    def test_feed_vad_chunks_into_windows(self):
        calls = []

        def stub(frame):
            assert frame.shape[-1] == WINDOW
            calls.append(frame)
            return 0.9 if len(calls) > 10 else 0.0

        cfg = config()
        segments: list[Segment] = []
        seg = SpeechSegmenter(cfg, vad=stub, on_segment=segments.append)

        # Feed in awkward sizes to prove re-chunking works.
        pcm = np.zeros(WINDOW * 60, dtype="<i2").tobytes()
        for offset in range(0, len(pcm), 999):
            seg.feed_vad(pcm[offset : offset + 999])
        assert len(calls) == 60
