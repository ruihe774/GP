"""Silero VAD contract tests.

These run by default (skipped only if the model has not been fetched) rather
than living behind the `hardware` marker, because the bug they guard against
was invisible to every negative test: feeding the ONNX graph a bare 512-sample
window instead of 64 context samples + 512 new ones makes it return ~0 for all
input. It still "rejected silence" perfectly. Only a positive assertion catches
that, so there is one here.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpagent.capture.vad import CONTEXT_SAMPLES, WINDOW_SAMPLES, SileroVAD, default_model_path

pytestmark = pytest.mark.skipif(
    not default_model_path().exists(),
    reason="silero_vad.onnx not fetched; see README setup",
)

SR = 16000


def voiced_vowel(duration: float = 1.6, f0: float = 118.0) -> np.ndarray:
    """A synthetic voiced vowel: harmonic stack shaped by three formants.

    Not real speech, but periodic and formant-structured enough that a working
    Silero scores it well above threshold and a broken one does not.
    """
    t = np.arange(int(SR * duration)) / SR
    vibrato = f0 * (1 + 0.02 * np.sin(2 * np.pi * 4.5 * t))
    phase = np.cumsum(2 * np.pi * vibrato / SR)
    signal = np.zeros_like(t)
    for harmonic in range(1, 45):
        frequency = harmonic * f0
        if frequency > SR / 2:
            break
        amplitude = sum(
            np.exp(-((frequency - centre) ** 2) / (2 * width**2))
            for centre, width in ((730, 110), (1090, 180), (2440, 300))
        )
        signal += (amplitude / harmonic) * np.sin(harmonic * phase)
    envelope = np.minimum(1, np.minimum(t / 0.06, (duration - t) / 0.12))
    return (0.35 * signal / np.max(np.abs(signal)) * envelope).astype(np.float32)


def probabilities(vad: SileroVAD, signal: np.ndarray) -> list[float]:
    return [
        vad(signal[i : i + WINDOW_SAMPLES])
        for i in range(0, len(signal) - WINDOW_SAMPLES + 1, WINDOW_SAMPLES)
    ]


class SpySession:
    """Wraps the ONNX session to record the shapes it is actually fed."""

    def __init__(self, inner):
        self.inner = inner
        self.inputs: list[np.ndarray] = []

    def run(self, outputs, feeds):
        self.inputs.append(feeds["input"].copy())
        return self.inner.run(outputs, feeds)


class TestModelContract:
    def test_input_is_context_plus_window(self):
        vad = SileroVAD()
        spy = SpySession(vad._session)
        vad._session = spy
        probabilities(vad, np.zeros(WINDOW_SAMPLES * 3, dtype=np.float32))

        expected = CONTEXT_SAMPLES[SR] + WINDOW_SAMPLES
        assert all(fed.shape == (1, expected) for fed in spy.inputs)
        assert expected == 576

    def test_context_carries_between_calls(self):
        vad = SileroVAD()
        spy = SpySession(vad._session)
        vad._session = spy

        # A ramp makes it obvious which samples were carried forward.
        signal = np.arange(WINDOW_SAMPLES * 3, dtype=np.float32) / (WINDOW_SAMPLES * 3)
        probabilities(vad, signal)

        context = CONTEXT_SAMPLES[SR]
        first, second = spy.inputs[0], spy.inputs[1]
        assert np.allclose(second[0, :context], first[0, -context:]), (
            "each call must be prefixed with the tail of the previous chunk"
        )

    def test_first_call_has_zero_context(self):
        vad = SileroVAD()
        spy = SpySession(vad._session)
        vad._session = spy
        probabilities(vad, np.ones(WINDOW_SAMPLES, dtype=np.float32))
        assert np.all(spy.inputs[0][0, : CONTEXT_SAMPLES[SR]] == 0.0)

    def test_reset_clears_context(self):
        vad = SileroVAD()
        probabilities(vad, voiced_vowel(0.5))
        vad.reset()
        spy = SpySession(vad._session)
        vad._session = spy
        probabilities(vad, np.ones(WINDOW_SAMPLES, dtype=np.float32))
        assert np.all(spy.inputs[0][0, : CONTEXT_SAMPLES[SR]] == 0.0)

    def test_wrong_window_size_is_rejected(self):
        vad = SileroVAD()
        with pytest.raises(ValueError):
            vad(np.zeros(400, dtype=np.float32))


class TestDetection:
    def test_detects_voiced_speech(self):
        """The regression guard. A VAD that always returns 0 passes every
        negative test in this file; only this one fails."""
        vad = SileroVAD()
        signal = np.concatenate(
            [np.zeros(SR // 2, np.float32), voiced_vowel(), np.zeros(SR // 2, np.float32)]
        )
        scores = probabilities(vad, signal)
        assert max(scores) > 0.5, f"VAD never fired on voiced audio (max {max(scores):.3f})"
        assert sum(score > 0.5 for score in scores) >= 3

    @pytest.mark.parametrize(
        "name,make",
        [
            ("silence", lambda: np.zeros(SR, np.float32)),
            ("white noise", lambda: np.random.default_rng(0).normal(0, 0.05, SR).astype(np.float32)),
            ("bass rumble", lambda: (0.4 * np.sin(2 * np.pi * 80 * np.arange(SR) / SR)).astype(np.float32)),
            ("mid tone", lambda: (0.3 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype(np.float32)),
        ],
    )
    def test_rejects_non_speech(self, name, make):
        vad = SileroVAD()
        assert max(probabilities(vad, make())) < 0.5, f"false positive on {name}"

    def test_speech_scores_far_above_noise(self):
        vad = SileroVAD()
        speech = max(probabilities(vad, voiced_vowel()))
        vad.reset()
        noise = max(
            probabilities(vad, np.random.default_rng(1).normal(0, 0.05, SR).astype(np.float32))
        )
        assert speech > noise * 5
