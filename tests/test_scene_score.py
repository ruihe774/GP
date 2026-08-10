"""Scene-change detection vs deduplication -- two different questions.

`scene_score` asks "did something just happen", `dedup` asks "has the model
already seen this". They were the same measurement (both against the last frame
sent), which made the trigger a clock rather than a detector: the longer the
frame policy suppressed frames, the more the screen diverged, so the comparison
cleared any fixed threshold given enough time. In sess5 that fired 25 of 28
frames, every one at exactly the 5 s floor, every recorded score above the
threshold -- `scene_threshold` selected nothing and `heartbeat` never fired.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpagent.capture.screen import ScreenSource
from gpagent.config import ScreenConfig, TriggerConfig


def source() -> ScreenSource:
    return ScreenSource(ScreenConfig(), TriggerConfig())


def thumb(value: int) -> np.ndarray:
    return np.full((32, 32), value, dtype=np.int16)


class TestSceneScore:
    def test_the_first_sample_always_counts_as_a_change(self):
        assert source()._scene_score(thumb(10)) == 1.0

    def test_a_still_screen_scores_zero(self):
        src = source()
        src._prev_thumb = thumb(10)
        assert src._scene_score(thumb(10)) == 0.0

    def test_it_measures_against_the_previous_sample(self):
        src = source()
        src._prev_thumb = thumb(10)
        src._last_sent_thumb = thumb(200)  # a long-stale sent frame
        # A quiet screen must score quiet even when the model last saw
        # something completely different -- that is the bug.
        assert src._scene_score(thumb(12)) < 0.02

    def test_a_real_cut_scores_high(self):
        src = source()
        src._prev_thumb = thumb(0)
        assert src._scene_score(thumb(255)) == pytest.approx(1.0)

    def test_a_static_screen_never_accumulates_a_score(self):
        """The regression: no amount of waiting should manufacture a change."""
        src = source()
        src._prev_thumb = thumb(10)
        for _ in range(100):  # 50 s of samples at 2 Hz
            score = src._scene_score(thumb(10))
            src._prev_thumb = thumb(10)
        assert score == 0.0


class TestDedupStillAsksTheOtherQuestion:
    def test_it_measures_against_the_last_sent_frame(self):
        src = source()
        src._prev_thumb = thumb(200)
        src._last_sent_thumb = thumb(10)
        # Unchanged since the model saw it, however much it moved in between.
        assert src._change_since_sent(thumb(10)) == 0.0

    def test_drift_since_the_model_last_looked_is_visible(self):
        src = source()
        src._last_sent_thumb = thumb(10)
        assert src._change_since_sent(thumb(60)) > TriggerConfig().dedup_threshold

    def test_nothing_sent_yet_is_never_a_duplicate(self):
        assert source()._change_since_sent(thumb(10)) == 1.0


class TestTheBaselineAdvances:
    def test_scene_score_is_relative_to_the_most_recent_sample(self):
        """If the baseline only advanced on *sent* frames it would decay back
        into the old behaviour, silently."""
        src = source()
        src._prev_thumb = thumb(10)
        # a gradual pan: each sample differs slightly from the one before
        for value in range(12, 60, 2):
            score = src._scene_score(thumb(value))
            src._prev_thumb = thumb(value)
            assert score < 0.02, "a slow pan is not a scene change"
