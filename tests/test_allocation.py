"""Tests for allocation.py — pure function, easy to lock down."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oath_score.allocation import allocate, metric_pct_to_close_races


def _candidates(n: int, *, scores=None, margins=None) -> pd.DataFrame:
    if scores is None:
        scores = list(range(1, n + 1))
    if margins is None:
        margins = [0.01] * n
    return pd.DataFrame({
        "id": [f"C{i}" for i in range(n)],
        "score": scores,
        "margin_pct": margins,
    })


class TestAllocate:
    def test_top_n_respected(self):
        df = _candidates(20)
        out = allocate(df, score_col="score", n=5, total_dollars=100.0)
        assert len(out) == 5
        # Highest-scoring 5 should be kept
        assert set(out["id"]) == {f"C{i}" for i in range(15, 20)}

    def test_allocations_sum_to_total(self):
        df = _candidates(10, scores=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        out = allocate(df, score_col="score", n=5, total_dollars=1000.0)
        assert out["allocation"].sum() == pytest.approx(1000.0, rel=1e-9)

    def test_score_weighting(self):
        # Two candidates, scores 1 and 3; top-2; expect 25% / 75%.
        df = pd.DataFrame({"id": ["A", "B"], "score": [1.0, 3.0], "margin_pct": [0, 0]})
        out = allocate(df, score_col="score", n=2, total_dollars=100.0)
        out = out.sort_values("id").reset_index(drop=True)
        assert out.loc[0, "allocation"] == pytest.approx(25.0)
        assert out.loc[1, "allocation"] == pytest.approx(75.0)

    def test_n_larger_than_candidate_pool(self):
        df = _candidates(3)
        out = allocate(df, score_col="score", n=10, total_dollars=100.0)
        assert len(out) == 3
        assert out["allocation"].sum() == pytest.approx(100.0)

    def test_all_zero_scores_falls_back_to_equal_split(self):
        df = pd.DataFrame({"id": ["A", "B", "C"], "score": [0, 0, 0], "margin_pct": [0, 0, 0]})
        out = allocate(df, score_col="score", n=3, total_dollars=99.0)
        assert out["allocation"].nunique() == 1
        assert out["allocation"].iloc[0] == pytest.approx(33.0)

    def test_negative_scores_clipped_to_zero(self):
        df = pd.DataFrame({"id": ["A", "B"], "score": [-5.0, 10.0], "margin_pct": [0, 0]})
        out = allocate(df, score_col="score", n=2, total_dollars=100.0)
        # Negative score becomes 0; B gets all the money via the
        # "all-zero-falls-back-to-equal-split" path? No — B has positive score
        # so we pick top-2 of (clipped 0, 10). Sum = 10, B gets 100% of weight.
        b_alloc = out.loc[out["id"] == "B", "allocation"].iloc[0]
        a_alloc = out.loc[out["id"] == "A", "allocation"].iloc[0]
        assert b_alloc == pytest.approx(100.0)
        assert a_alloc == pytest.approx(0.0)

    def test_nan_scores_treated_as_zero(self):
        df = pd.DataFrame({"id": ["A", "B"], "score": [np.nan, 5.0], "margin_pct": [0, 0]})
        out = allocate(df, score_col="score", n=2, total_dollars=100.0)
        assert out.loc[out["id"] == "B", "allocation"].iloc[0] == pytest.approx(100.0)

    def test_unknown_score_col_raises(self):
        df = _candidates(3)
        with pytest.raises(KeyError):
            allocate(df, score_col="nonexistent", n=3)

    def test_zero_n_raises(self):
        df = _candidates(3)
        with pytest.raises(ValueError):
            allocate(df, score_col="score", n=0)


class TestNeedCap:
    def test_need_cap_disabled_by_default(self):
        # Without need_col, identical to plain weighting
        df = pd.DataFrame({
            "id": ["A", "B"], "score": [1.0, 1.0],
            "margin_pct": [0, 0], "need_remaining": [10.0, 1000.0],
        })
        out = allocate(df, score_col="score", n=2, total_dollars=100.0)
        # Equal score → equal allocation
        assert out["allocation"].nunique() == 1

    def test_need_cap_caps_low_need(self):
        # A has very low need (10), B has high need (1000); both score the same.
        # With total_dollars=100, A's effective weight = 1 * min(1, 10/100) = 0.1,
        # B's = 1 * min(1, 1000/100) = 1.0. So B should get ~10x more.
        df = pd.DataFrame({
            "id": ["A", "B"], "score": [1.0, 1.0],
            "margin_pct": [0, 0], "need_remaining": [10.0, 1000.0],
        })
        out = allocate(df, score_col="score", n=2, total_dollars=100.0,
                       need_col="need_remaining")
        out = out.set_index("id")
        ratio = out.loc["B", "allocation"] / out.loc["A", "allocation"]
        assert ratio == pytest.approx(10.0, rel=1e-6)


class TestMetric:
    def test_all_close_races(self):
        df = pd.DataFrame({
            "allocation": [50.0, 30.0, 20.0],
            "margin_pct": [0.001, -0.02, 0.04],
        })
        assert metric_pct_to_close_races(df) == pytest.approx(1.0)

    def test_no_close_races(self):
        df = pd.DataFrame({
            "allocation": [50.0, 50.0],
            "margin_pct": [0.30, -0.40],
        })
        assert metric_pct_to_close_races(df) == pytest.approx(0.0)

    def test_partial_close(self):
        # 50 to a race at 0.01 (close), 50 to a race at 0.20 (not close)
        df = pd.DataFrame({"allocation": [50.0, 50.0], "margin_pct": [0.01, 0.20]})
        assert metric_pct_to_close_races(df) == pytest.approx(0.5)

    def test_threshold_inclusive(self):
        # At exactly 0.05 (the threshold), `<` is strict — should NOT count
        df = pd.DataFrame({"allocation": [100.0], "margin_pct": [0.05]})
        assert metric_pct_to_close_races(df) == pytest.approx(0.0)
        # Just below should count
        df2 = pd.DataFrame({"allocation": [100.0], "margin_pct": [0.0499]})
        assert metric_pct_to_close_races(df2) == pytest.approx(1.0)

    def test_empty_df(self):
        df = pd.DataFrame(columns=["allocation", "margin_pct"])
        assert metric_pct_to_close_races(df) == 0.0

    def test_zero_total_allocation(self):
        df = pd.DataFrame({"allocation": [0.0, 0.0], "margin_pct": [0.01, 0.02]})
        assert metric_pct_to_close_races(df) == 0.0
