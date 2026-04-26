"""Tests for scores/financial_need.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oath_score.scores.financial_need import (
    MIN_FLOOR,
    SCORE_COL,
    FinancialNeed,
)


def _synthetic_train(n: int = 100, *, seed: int = 0) -> pd.DataFrame:
    """Synthetic D winners of close races for fit().

    Floor scales monotonically with opp_raised_log so the model has signal.
    """
    rng = np.random.default_rng(seed)
    opp = rng.uniform(11.0, 15.5, size=n)
    ie_against = rng.uniform(0.0, 14.0, size=n)
    income = rng.uniform(35_000, 110_000, size=n)
    # Winner spent: scales with opp + ie + a bit of income
    spent = np.exp(opp + 0.05 * ie_against + 0.000005 * income + rng.normal(0, 0.4, size=n))
    margin = rng.uniform(-0.09, 0.09, size=n)
    return pd.DataFrame({
        "party_major": ["D"] * n,
        "winner": [True] * n,
        "margin_pct": margin,
        "opp_raised_log": opp,
        "ie_against_log": ie_against,
        "acs_median_income": income,
        "total_trans": spent,
    })


def _synthetic_test(n: int = 50, *, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    opp = rng.uniform(11.0, 15.5, size=n)
    return pd.DataFrame({
        "party_major": ["D"] * n,
        "winner": rng.choice([True, False], size=n),
        "margin_pct": rng.uniform(-0.30, 0.30, size=n),
        "opp_raised_log": opp,
        "ie_against_log": rng.uniform(0.0, 14.0, size=n),
        "acs_median_income": rng.uniform(35_000, 110_000, size=n),
        "total_trans": np.exp(rng.uniform(8.0, 16.0, size=n)),
    })


class TestFit:
    def test_fits_and_is_marked(self):
        m = FinancialNeed().fit(_synthetic_train())
        assert m.is_fitted

    def test_too_few_rows_raises(self):
        df = _synthetic_train(n=10)
        with pytest.raises(ValueError, match="Too few"):
            FinancialNeed(min_train_rows=30).fit(df)

    def test_filter_drops_non_dems(self):
        df = _synthetic_train()
        df.loc[:50, "party_major"] = "R"
        # Should still fit because we have ~50 D winners
        FinancialNeed(min_train_rows=30).fit(df)

    def test_filter_drops_losers(self):
        df = _synthetic_train()
        df["winner"] = False
        with pytest.raises(ValueError, match="Too few"):
            FinancialNeed().fit(df)

    def test_filter_drops_far_margins(self):
        df = _synthetic_train()
        df["margin_pct"] = 0.5  # all blowouts → none close
        with pytest.raises(ValueError, match="Too few"):
            FinancialNeed().fit(df)


class TestPredictFloor:
    def test_returns_series_aligned_with_input(self):
        m = FinancialNeed().fit(_synthetic_train())
        test = _synthetic_test()
        floor = m.predict_floor(test)
        assert isinstance(floor, pd.Series)
        assert len(floor) == len(test)
        assert floor.index.equals(test.index)
        assert floor.name == "viable_floor"

    def test_floor_always_positive(self):
        m = FinancialNeed().fit(_synthetic_train())
        floor = m.predict_floor(_synthetic_test())
        assert (floor > 0).all()

    def test_floor_clipped_to_min(self):
        m = FinancialNeed().fit(_synthetic_train())
        # Inputs that might push floor very low
        test = pd.DataFrame({
            "opp_raised_log": [0.0, 0.0],
            "ie_against_log": [0.0, 0.0],
            "acs_median_income": [10_000.0, 10_000.0],
        })
        floor = m.predict_floor(test)
        assert (floor >= MIN_FLOOR).all()

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            FinancialNeed().predict_floor(_synthetic_test())

    def test_missing_features_raise(self):
        m = FinancialNeed().fit(_synthetic_train())
        bad = pd.DataFrame({"opp_raised_log": [12.0]})
        with pytest.raises(KeyError, match="features missing"):
            m.predict_floor(bad)


class TestPredictNeed:
    def test_in_unit_interval(self):
        m = FinancialNeed().fit(_synthetic_train())
        need = m.predict_need(_synthetic_test())
        assert (need >= 0).all() and (need <= 1).all()
        assert need.name == SCORE_COL

    def test_above_floor_yields_zero_need(self):
        m = FinancialNeed().fit(_synthetic_train())
        # Build a test row with very high own spend
        test = pd.DataFrame({
            "opp_raised_log": [12.0],
            "ie_against_log": [5.0],
            "acs_median_income": [60_000.0],
            "total_trans": [1e9],  # extremely overfunded
        })
        need = m.predict_need(test)
        assert need.iloc[0] == 0.0

    def test_zero_spend_yields_unit_need(self):
        m = FinancialNeed().fit(_synthetic_train())
        test = pd.DataFrame({
            "opp_raised_log": [13.5],
            "ie_against_log": [10.0],
            "acs_median_income": [70_000.0],
            "total_trans": [0.0],
        })
        need = m.predict_need(test)
        assert need.iloc[0] == 1.0

    def test_nan_spend_treated_as_most_needy(self):
        m = FinancialNeed().fit(_synthetic_train())
        test = pd.DataFrame({
            "opp_raised_log": [13.5],
            "ie_against_log": [10.0],
            "acs_median_income": [70_000.0],
            "total_trans": [np.nan],
        })
        need = m.predict_need(test)
        assert need.iloc[0] == 1.0


class TestNoTargetLeak:
    def test_predict_path_does_not_use_margin_or_winner(self):
        """Pull a row with arbitrary margin / winner, swap them; predictions identical."""
        m = FinancialNeed().fit(_synthetic_train())
        test_a = _synthetic_test().head(5).copy()
        test_b = test_a.copy()
        test_b["margin_pct"] = -test_a["margin_pct"]
        test_b["winner"] = ~test_a["winner"].astype(bool)
        floor_a = m.predict_floor(test_a)
        floor_b = m.predict_floor(test_b)
        assert (floor_a.values == floor_b.values).all()
        need_a = m.predict_need(test_a)
        need_b = m.predict_need(test_b)
        assert (need_a.values == need_b.values).all()
