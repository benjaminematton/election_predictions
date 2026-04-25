"""Tests for scores/impact.combine_scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oath_score.scores.impact import combine_scores


class TestCombineScores:
    def test_geometric_mean(self):
        c = pd.Series([1.0, 0.5, 0.0, 0.25])
        s = pd.Series([1.0, 0.5, 1.0, 0.16])
        out = combine_scores(c, s)
        assert out.iloc[0] == pytest.approx(1.0)
        assert out.iloc[1] == pytest.approx(0.5)
        assert out.iloc[2] == pytest.approx(0.0)
        assert out.iloc[3] == pytest.approx(0.2)

    def test_zero_on_either_dimension(self):
        c = pd.Series([0.9, 0.0, 0.0])
        s = pd.Series([0.0, 0.9, 0.0])
        out = combine_scores(c, s)
        assert (out == 0.0).all()

    def test_nan_treated_as_zero(self):
        c = pd.Series([0.5, np.nan])
        s = pd.Series([0.5, 0.5])
        out = combine_scores(c, s)
        assert out.iloc[0] == pytest.approx(0.5)
        assert out.iloc[1] == 0.0

    def test_clips_above_one(self):
        # Inputs >1 (shouldn't happen but defensive) get clipped
        c = pd.Series([2.0])
        s = pd.Series([2.0])
        out = combine_scores(c, s)
        assert out.iloc[0] == pytest.approx(1.0)

    def test_index_mismatch_raises(self):
        c = pd.Series([0.5], index=[0])
        s = pd.Series([0.5], index=[1])
        with pytest.raises(ValueError, match="share an index"):
            combine_scores(c, s)

    def test_output_named(self):
        c = pd.Series([0.5, 0.5])
        s = pd.Series([0.5, 0.5])
        out = combine_scores(c, s)
        assert out.name == "score_impact_base"
