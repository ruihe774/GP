"""Token and cost estimation for captured sessions.

These are *estimates* derived from published tokenization rules, not measured
against the API. They exist to make the cost consequences of a capture policy
visible while tuning; reconcile against real usage once Component B runs.

Rates as of the gpt-realtime-2.1 pricing page:
  audio input  $32 / 1M tokens, 1 token per 100 ms of input audio
  image input  $5  / 1M tokens
  text input   $4  / 1M tokens

`Estimate` is Component A's forward-looking guess from a capture. `UsageMeter` is
Component B's record of what the API actually charged, folded from `response.done`.
Keeping both in one module is the point: the estimates had never been reconciled
against a bill, and now they can be, in the same units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "Rates",
    "MODEL_RATES",
    "rates_for",
    "audio_tokens",
    "image_tokens",
    "text_tokens",
    "Estimate",
    "UsageMeter",
]


@dataclass(frozen=True)
class Rates:
    audio_per_mtok: float = 32.0
    image_per_mtok: float = 5.0
    text_per_mtok: float = 4.0
    #: audio input is billed at 1 token per 100 ms
    audio_ms_per_token: float = 100.0

    # Output and cached-input rates matter only once Component B is running:
    # what the agent says is the half of the bill capture cannot see.
    audio_out_per_mtok: float = 64.0
    text_out_per_mtok: float = 24.0
    cached_audio_per_mtok: float = 0.4
    cached_text_per_mtok: float = 0.4
    cached_image_per_mtok: float = 0.5
    #: output audio is billed at roughly 1 token per 50 ms spoken
    audio_out_ms_per_token: float = 50.0


DEFAULT_RATES = Rates()

#: developers.openai.com/api/docs/pricing, checked 2026-08-09. The token counts
#: the API reports are authoritative; these only turn them into dollars, so a
#: stale entry misprices a report but never misreports usage.
MODEL_RATES: dict[str, Rates] = {
    "gpt-realtime-2.1": DEFAULT_RATES,
    "gpt-realtime-2.1-mini": Rates(
        audio_per_mtok=10.0,
        image_per_mtok=0.8,
        text_per_mtok=0.6,
        audio_out_per_mtok=20.0,
        text_out_per_mtok=2.4,
        cached_audio_per_mtok=0.3,
        cached_text_per_mtok=0.06,
        cached_image_per_mtok=0.08,
    ),
}


def rates_for(model: str) -> Rates:
    """Rates for a model name, falling back to the full-price ones.

    Falling back *up* is deliberate: an unknown model reporting flagship prices
    overstates the bill, which is the safe direction to be wrong in.
    """
    if model in MODEL_RATES:
        return MODEL_RATES[model]
    for name, rates in MODEL_RATES.items():
        if model.startswith(name):
            return rates
    return DEFAULT_RATES


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


def estimate_events(events) -> Estimate:
    """What a captured session would cost to send in full, if it all went.

    The *upper bound* on Component B's input cost: the agent sends at most one
    frame per response and drops summaries it never had a reason to use, so what
    it actually spends is lower, and by how much is the interesting number.
    """
    from .events import GamepadActivity, ScreenFrame, SpeechSegment

    estimate = Estimate()
    for event in events:
        if isinstance(event, GamepadActivity):
            estimate.text_tokens += text_tokens(event.summary)
        elif isinstance(event, SpeechSegment):
            estimate.audio_tokens += audio_tokens(event.dur_ms)
            estimate.audio_ms += event.dur_ms
        elif isinstance(event, ScreenFrame):
            estimate.image_tokens += image_tokens(event.w, event.h)
            estimate.frames += 1
    return estimate


@dataclass
class UsageMeter:
    """What the API actually billed, folded from `response.done`.

    Cached input tokens are a *subset* of input tokens, not an addition to them,
    so uncached counts are derived by subtraction. Getting that backwards
    double-counts the cheapest tokens in the session.
    """

    model: str = "gpt-realtime-2.1"
    responses: int = 0
    in_text: int = 0
    in_audio: int = 0
    in_image: int = 0
    in_cached: int = 0
    out_text: int = 0
    out_audio: int = 0
    #: cached_tokens_details, when the server breaks it down for us
    cached_text: int = 0
    cached_audio: int = 0
    cached_image: int = 0
    unattributed_input: int = 0
    raw: list[dict] = field(default_factory=list, repr=False)

    @property
    def rates(self) -> Rates:
        return rates_for(self.model)

    def add(self, usage: dict | None) -> None:
        """Fold one `response.done` usage object in. Missing fields are zero."""
        if not usage:
            return
        self.responses += 1
        self.raw.append(usage)
        into = usage.get("input_token_details") or {}
        out = usage.get("output_token_details") or {}
        cached = into.get("cached_tokens_details") or {}

        self.in_text += _int(into.get("text_tokens"))
        self.in_audio += _int(into.get("audio_tokens"))
        self.in_image += _int(into.get("image_tokens"))
        self.in_cached += _int(into.get("cached_tokens"))
        self.cached_text += _int(cached.get("text_tokens"))
        self.cached_audio += _int(cached.get("audio_tokens"))
        self.cached_image += _int(cached.get("image_tokens"))
        self.out_text += _int(out.get("text_tokens"))
        self.out_audio += _int(out.get("audio_tokens"))

        # The per-modality breakdown is optional; keep the total honest when it
        # is absent rather than silently reporting a session that cost nothing.
        detailed = _int(into.get("text_tokens")) + _int(into.get("audio_tokens")) + _int(
            into.get("image_tokens")
        )
        self.unattributed_input += max(0, _int(usage.get("input_tokens")) - detailed)

    def cost(self) -> dict[str, float]:
        r = self.rates
        # Split cached out of each modality where the server told us how; where
        # it did not, charge the remainder at the uncached rate.
        known_cached = self.cached_text + self.cached_audio + self.cached_image
        other_cached = max(0, self.in_cached - known_cached)

        text = max(0, self.in_text - self.cached_text) * r.text_per_mtok / 1e6
        audio = max(0, self.in_audio - self.cached_audio) * r.audio_per_mtok / 1e6
        image = max(0, self.in_image - self.cached_image) * r.image_per_mtok / 1e6
        cached = (
            self.cached_text * r.cached_text_per_mtok
            + self.cached_audio * r.cached_audio_per_mtok
            + self.cached_image * r.cached_image_per_mtok
            + other_cached * r.cached_text_per_mtok
        ) / 1e6
        other = self.unattributed_input * r.text_per_mtok / 1e6
        out_text = self.out_text * r.text_out_per_mtok / 1e6
        out_audio = self.out_audio * r.audio_out_per_mtok / 1e6
        return {
            "in_text_usd": text,
            "in_audio_usd": audio,
            "in_image_usd": image,
            "in_cached_usd": cached,
            "in_other_usd": other,
            "out_text_usd": out_text,
            "out_audio_usd": out_audio,
            "input_usd": text + audio + image + cached + other,
            "output_usd": out_text + out_audio,
            "total_usd": text + audio + image + cached + other + out_text + out_audio,
        }

    @property
    def spoken_ms(self) -> float:
        """Audio the agent produced, inferred from output audio tokens."""
        return self.out_audio * self.rates.audio_out_ms_per_token

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "responses": self.responses,
            "tokens": {
                "input_text": self.in_text,
                "input_audio": self.in_audio,
                "input_image": self.in_image,
                "input_cached": self.in_cached,
                "input_unattributed": self.unattributed_input,
                "output_text": self.out_text,
                "output_audio": self.out_audio,
            },
            "spoken_s": round(self.spoken_ms / 1000.0, 1),
            "cost": {k: round(v, 6) for k, v in self.cost().items()},
        }


def _int(value: object) -> int:
    return int(value) if isinstance(value, (int, float)) else 0
