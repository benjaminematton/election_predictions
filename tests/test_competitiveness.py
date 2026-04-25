"""Tests for scores/competitiveness.py — naive baseline + Cook imputation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oath_score.scores.competitiveness import (
    NaiveCompetitiveness,
    SCORE_COL,
    impute_cook_rating,
)


def _row(party: str, margin: float, *, cook=None, incumbent: int = 0) -> dict:
    return {
        "party_major": party,
        "margin_pct": margin,
        "cook_rating": cook,
        "incumbent": incumbent,
    }


class TestImputation:
    def test_already_present_passes_through(self):
        df = pd.DataFrame([
            _row("D", 0.05, cook=4.0),
            _row("R", -0.05, cook=4.0),
        ])
        out = impute_cook_rating(df)
        assert list(out) == [4.0, 4.0]

    def test_d_incumbent_imputed_solid_d(self):
        df = pd.DataFrame([_row("D", 0.4, incumbent=1)])
        assert impute_cook_rating(df).iloc[0] == 7.0

    def test_r_incumbent_imputed_solid_r(self):
        df = pd.DataFrame([_row("R", -0.4, incumbent=1)])
        assert impute_cook_rating(df).iloc[0] == 1.0

    def test_open_seat_solid_d_by_margin(self):
        df = pd.DataFrame([_row("D", 0.20, incumbent=0)])
        assert impute_cook_rating(df).iloc[0] == 7.0

    def test_open_seat_likely_d(self):
        df = pd.DataFrame([_row("D", 0.05, incumbent=0)])
        assert impute_cook_rating(df).iloc[0] == 6.0

    def test_open_seat_likely_r(self):
        df = pd.DataFrame([_row("D", -0.05, incumbent=0)])
        assert impute_cook_rating(df).iloc[0] == 2.0

    def test_open_seat_solid_r(self):
        df = pd.DataFrame([_row("D", -0.20, incumbent=0)])
        assert impute_cook_rating(df).iloc[0] == 1.0

    def test_no_margin_data_falls_back_to_tossup(self):
        df = pd.DataFrame([{
            "party_major": "D", "margin_pct": np.nan,
            "cook_rating": np.nan, "incumbent": 0,
        }])
        assert impute_cook_rating(df).iloc[0] == 4.0


def _synthetic_training_data(seed: int = 0) -> pd.DataFrame:
    """Synthetic 200-candidate frame: roughly bimodal margins, varied ratings."""
    rng = np.random.default_rng(seed)
    n = 200
    cook = rng.choice([1, 2, 3, 4, 5, 6, 7], size=n)
    incumbent = rng.choice([0, 1], size=n, p=[0.4, 0.6])
    party = rng.choice(["D", "R"], size=n)
    # Margin is positively correlated with cook_rating (1=R, 7=D); add noise
    margin = (cook - 4) / 6 + rng.normal(0, 0.10, size=n)
    margin = np.clip(margin, -0.99, 0.99)
    return pd.DataFrame({
        "party_major": party,
        "cook_rating": cook.astype(float),
        "incumbent": incumbent,
        "margin_pct": margin,
    })


class TestNaiveCompetitiveness:
    def test_fit_predict_returns_probabilities(self):
        train = _synthetic_training_data()
        model = NaiveCompetitiveness().fit(train)
        scores = model.predict_proba(train)
        assert ((scores >= 0) & (scores <= 1)).all()
        assert scores.name == SCORE_COL

    def test_non_dems_get_zero(self):
        train = _synthetic_training_data()
        model = NaiveCompetitiveness().fit(train)
        scores = model.predict_proba(train)
        # Republican rows score 0
        r_scores = scores[train["party_major"] == "R"]
        assert (r_scores == 0.0).all()
        # Democrats get nonzero (some at least)
        d_scores = scores[train["party_major"] == "D"]
        assert (d_scores > 0).any()

    def test_score_attaches_column(self):
        train = _synthetic_training_data()
        model = NaiveCompetitiveness().fit(train)
        scored = model.score(train)
        assert SCORE_COL in scored.columns
        assert len(scored) == len(train)

    def test_d_outscores_r_in_same_district(self):
        """Sanity: a Dem in a Tossup should outscore an R in the same district.

        We don't assert on relative ordering between different cook_ratings —
        the naive baseline collapses 'close-D-win' towards 'D-win' due to the
        class imbalance noted in phase3.md (Phase 4 fixes this with multi-
        quantile regression on signed margin).
        """
        train = _synthetic_training_data()
        model = NaiveCompetitiveness().fit(train)

        test = pd.DataFrame([
            {"party_major": "D", "cook_rating": 4.0, "incumbent": 0, "margin_pct": 0.0},
            {"party_major": "R", "cook_rating": 4.0, "incumbent": 0, "margin_pct": 0.0},
        ])
        scores = model.predict_proba(test)
        # The R candidate scores 0 by construction; Dem scores should be > 0.
        assert scores.iloc[1] == 0.0
        assert scores.iloc[0] > 0.0

    def test_predict_before_fit_raises(self):
        model = NaiveCompetitiveness()
        with pytest.raises(RuntimeError):
            model.predict_proba(pd.DataFrame())

    def test_no_dems_in_train_raises(self):
        train = pd.DataFrame([
            {"party_major": "R", "cook_rating": 1.0, "incumbent": 1, "margin_pct": -0.3},
        ])
        with pytest.raises(ValueError, match="No scorable Democratic"):
            NaiveCompetitiveness().fit(train)

    def test_too_few_positives_raises(self):
        # All Dems lose by huge margins → no close-D-win positives
        train = pd.DataFrame([
            {"party_major": "D", "cook_rating": 1.0, "incumbent": 0, "margin_pct": -0.4}
            for _ in range(20)
        ])
        with pytest.raises(ValueError, match="Too few positive"):
            NaiveCompetitiveness().fit(train)

    def test_nan_cook_in_test_scores_zero(self):
        """Per the leakage discussion in phase3.md, rows with NaN cook_rating
        score 0 — they're outside the Wikipedia-tracked universe and a
        sophisticated donor doesn't need help with them."""
        train = _synthetic_training_data()
        model = NaiveCompetitiveness().fit(train)

        test = pd.DataFrame([
            {"party_major": "D", "cook_rating": np.nan, "incumbent": 1, "margin_pct": 0.4},
            {"party_major": "D", "cook_rating": 4.0, "incumbent": 0, "margin_pct": 0.0},
        ])
        scores = model.predict_proba(test)
        assert scores.iloc[0] == 0.0
        assert scores.iloc[1] > 0.0
