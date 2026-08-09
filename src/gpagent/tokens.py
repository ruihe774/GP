"""Token and cost estimation for captured sessions.

These are *estimates* derived from published tokenization rules, not measured
against the API. They exist to make the cost consequences of a capture policy
visible while tuning; reconcile against real usage once Component B runs.

Rates as of the gpt-realtime-2.1 pricing page:
  audio input  $32 / 1M tokens, 1 token per 100 ms of input audio
  image input  $5  / 1M tokens
  text input   $4  / 1M tokens
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Rates", "audio_tokens", "image_tokens", "text_tokens", "Estimate"]


@dataclass(frozen=True)
class Rates:
    audio_per_mtok: float = 32.0
    image_per_mtok: float = 5.0
    text_per_mtok: float = 4.0
    #: audio input is billed at 1 token per 100 ms
    audio_ms_per_token: float = 100.0


DEFAULT_RATES = Rates()


def audio_tokens(duration_ms: float, rates: Rates = DEFAULT_RATES) -> int:
    return int(math.ceil(duration_ms / rates.audio_ms_per_token))


def image_tokens(width: int, height: int, *, detail: str = "high") -> int:
    """OpenAI patch-based image tokenization.

    `detail="low"` is a flat 85 tokens regardless of size -- a large lever for
    Component B if HUD legibility turns out not to matter.
    """
    if detail == "low":
        return 85
    if width <= 0 or height <= 0:
        return 85

    # Fit within 2048x2048, then bring the shortest side down to 768.
    # This downscales only: whether OpenAI also scales small images *up* to a
    # 768 short side is not clearly documented and would raise a 1024x576 frame
    # from 765 to 1105 tokens. The conservative reading is used here, which is
    # the main reason these numbers are labelled estimates.
    w, h = float(width), float(height)
    if max(w, h) > 2048:
        scale = 2048 / max(w, h)
        w, h = w * scale, h * scale
    if min(w, h) > 768:
        scale = 768 / min(w, h)
        w, h = w * scale, h * scale

    tiles = math.ceil(w / 512) * math.ceil(h / 512)
    return 85 + 170 * tiles


def text_tokens(text: str) -> int:
    """Rough character-based approximation; summaries are short and ASCII."""
    return max(1, int(math.ceil(len(text) / 4)))


@dataclass
class Estimate:
    audio_tokens: int = 0
    image_tokens: int = 0
    text_tokens: int = 0
    audio_ms: float = 0.0
    frames: int = 0

    def cost(self, rates: Rates = DEFAULT_RATES) -> dict[str, float]:
        audio = self.audio_tokens * rates.audio_per_mtok / 1e6
        image = self.image_tokens * rates.image_per_mtok / 1e6
        text = self.text_tokens * rates.text_per_mtok / 1e6
        return {
            "audio_usd": audio,
            "image_usd": image,
            "text_usd": text,
            "total_usd": audio + image + text,
        }
