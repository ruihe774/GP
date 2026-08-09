"""Silero VAD (v5) wrapper.

The model is deliberately restricted to a single thread: this runs while a game
is running, and grabbing extra cores for a 2 MB model is a bad trade.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["SileroVAD", "default_model_path", "WINDOW_SAMPLES"]

#: Chunk of new audio per call at 16 kHz.
WINDOW_SAMPLES = 512

#: Samples of the *previous* chunk that must be prepended to the new one. The
#: exported silero_vad.onnx does not carry this internally -- the reference
#: wrapper concatenates it before every inference, so the tensor is 576 wide at
#: 16 kHz. Feeding a bare 512 is accepted by the graph (its input is [None,
#: None]) but yields a near-zero probability for *all* input, speech included.
CONTEXT_SAMPLES = {16000: 64, 8000: 32}


def default_model_path() -> Path:
    """models/silero_vad.onnx next to the repo root, overridable via config."""
    return Path(__file__).resolve().parents[3] / "models" / "silero_vad.onnx"


class SileroVAD:
    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        sample_rate: int = 16000,
        threads: int = 1,
    ) -> None:
        import onnxruntime as ort

        path = Path(model_path) if model_path else default_model_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Silero VAD model not found at {path}. "
                "Fetch it with: scripts/fetch_model.sh"
            )

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3

        self._session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.sample_rate = sample_rate
        self.window = 512 if sample_rate == 16000 else 256
        self._context_size = CONTEXT_SAMPLES.get(sample_rate, 64)
        self._sr = np.array(sample_rate, dtype=np.int64)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)
        self.model_path = path

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Speech probability for exactly `window` float32 samples."""
        if frame.shape[-1] != self.window:
            raise ValueError(f"expected {self.window} samples, got {frame.shape[-1]}")
        chunk = frame.reshape(1, self.window).astype(np.float32, copy=False)
        # Prepend the tail of the previous chunk; without it the model scores
        # everything at ~0 and the VAD silently never fires.
        batch = np.concatenate((self._context, chunk), axis=1)
        out, self._state = self._session.run(
            None, {"input": batch, "state": self._state, "sr": self._sr}
        )
        self._context = batch[:, -self._context_size :]
        return float(out[0][0])
