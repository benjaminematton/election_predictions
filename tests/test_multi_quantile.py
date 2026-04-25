"""Tests for scores/multi_quantile.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oath_score.scores.multi_quantile import (
    CLOSE_THRESHOLD,
    MultiQuantileCompetitiveness,
    QUANTILES,
)
from oath_score.scores.competitiveness import SCORE_COL


def _synthetic_dataset(n_dem: int = 200, *, seed: int = 0) -> pd.DataFrame:
    """Synthetic Dem-only dataset where margin is correlated with cook_rating."""
    rng = np.random.default_rng(seed)
    cook = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], size=n_dem)
    incumbent = rng.choice([0, 1], size=n_dem, p=[0.4, 0.6])
    # Margin: shape U-shape AROUND tossup-like cook=4
    margin = (cook - 4) / 6 + rng.normal(0, 0.15, size=n_dem)
    margin = np.clip(margin, -0.95, 0.95)
    return pd.DataFrame({
        "party_major": ["D"] * n_dem,
        "cook_rating": cook,
        "incumbent": incumbent,
        "margin_pct": margin,
    })


class TestFitPredict:
    def test_fit_predict_returns_probabilities(self):
        train = _synthetic_dataset(seed=1)
        model = MultiQuantileCompetitiveness().fit(train)
        scores = model.predict_proba(train)
        assert ((scores >= 0) & (scores <= 1)).all()
        assert scores.name == SCORE_COL

    def test_one_model_per_quantile(self):
        train = _synthetic_dataset(seed=2)
        model = MultiQuantileCompetitiveness().fit(train)
        assert len(model._models) == len(QUANTILES)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            MultiQuantileCompetitiveness().predict_proba(pd.DataFrame())

    def test_too_small_train_raises(self):
        train = _synthetic_dataset(n_dem=10, seed=3)
        with pytest.raises(ValueError, match="too small"):
            MultiQuantileCompetitiveness().fit(train)

    def test_scores_higher_for_tossup_than_safe(self):
        """Sanity: a Dem in a Tossup district should score higher P(close-race)
        than a Dem in a Solid D blowout district."""
        train = _synthetic_dataset(seed=4)
        model = MultiQuantileCompetitiveness().fit(train)
        test = pd.DataFrame([
            {"party_major": "D", "cook_rating": 4.0, "incumbent": 0, "margin_pct": 0.0},
            {"party_major": "D", "cook_rating": 7.0, "incumbent": 0, "margin_pct": 0.5},
        ])
        scores = model.predict_proba(test)
        assert scores.iloc[0] > scores.iloc[1], (
            f"Tossup should score higher P(close); got {scores.tolist()}"
        )

    def test_non_dems_get_zero(self):
        train = _synthetic_dataset(seed=5)
        model = MultiQuantileCompetitiveness().fit(train)
        test = pd.DataFrame([
            {"party_major": "R", "cook_rating": 4.0, "incumbent": 0, "margin_pct": 0.0},
            {"party_major": "D", "cook_rating": 4.0, "incumbent": 0, "margin_pct": 0.0},
        ])
        scores = model.predict_proba(test)
        assert scores.iloc[0] == 0.0
        assert scores.iloc[1] > 0.0

    def test_naive_set_requires_non_nan_cook(self):
        train = _synthetic_dataset(seed=6)
        model = MultiQuantileCompetitiveness().fit(train)
        test = pd.DataFrame([{
            "party_major": "D", "cook_rating": np.nan,
            "incumbent": 0, "margin_pct": 0.0,
        }])
        scores = model.predict_proba(test)
        assert scores.iloc[0] == 0.0


class TestCrossings:
    def test_crossing_rate_accessible(self):
        train = _synthetic_dataset(seed=7)
        model = MultiQuantileCompetitiveness().fit(train)
        # Run prediction to populate the diagnostic
        model.predict_proba(train)
        rate = model.crossing_rate
        assert 0.0 <= rate <= 1.0

    def test_crossings_repaired_in_output(self):
        """Even when the underlying quantile fits cross, the predicted P
        should remain in [0, 1] thanks to the per-row sort."""
        train = _synthetic_dataset(seed=8)
        model = MultiQuantileCompetitiveness().fit(train)
        # Force a noisy test to maximize crossings
        rng = np.random.default_rng(9)
        n = 100
        test = pd.DataFrame({
            "party_major": ["D"] * n,
            "cook_rating": rng.uniform(1, 7, size=n),
            "incumbent": rng.choice([0, 1], size=n),
            "margin_pct": rng.normal(0, 0.3, size=n),
        })
        scores = model.predict_proba(test)
        assert ((scores >= 0) & (scores <= 1)).all()


class TestProbabilityMakesSense:
    def test_close_threshold_value(self):
        """Sanity-check the close threshold matches the constant in the module."""
        assert CLOSE_THRESHOLD == 0.05
