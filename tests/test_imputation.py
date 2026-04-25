"""Tests for _imputation.py — target-free PVI-based cook imputation.

The whole point: imputed cook_rating must depend ONLY on cpvi (cycle-static
input), never on margin_pct (the prediction target). Anything that uses
margin to backfill cook would be target leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oath_score.scores._imputation import (
    fill_remaining_with_tossup,
    impute_cook_from_pvi,
)


def _row(*, cook=None, cpvi=None, margin=0.0, party="D") -> dict:
    return {
        "cook_rating": cook,
        "cpvi": cpvi,
        "party_major": party,
        "margin_pct": margin,
    }


class TestImputeFromPvi:
    def test_existing_cook_passes_through(self):
        df = pd.DataFrame([_row(cook=4.0, cpvi=10)])
        out = impute_cook_from_pvi(df)
        assert out.iloc[0] == 4.0

    def test_strong_d_pvi_imputes_solid_d(self):
        df = pd.DataFrame([_row(cook=None, cpvi=15)])
        assert impute_cook_from_pvi(df).iloc[0] == 7.0

    def test_strong_r_pvi_imputes_solid_r(self):
        df = pd.DataFrame([_row(cook=None, cpvi=-15)])
        assert impute_cook_from_pvi(df).iloc[0] == 1.0

    def test_lean_d_pvi(self):
        df = pd.DataFrame([_row(cook=None, cpvi=5)])
        assert impute_cook_from_pvi(df).iloc[0] == 6.0

    def test_lean_r_pvi(self):
        df = pd.DataFrame([_row(cook=None, cpvi=-5)])
        assert impute_cook_from_pvi(df).iloc[0] == 2.0

    def test_neutral_pvi_imputes_tossup(self):
        df = pd.DataFrame([_row(cook=None, cpvi=0)])
        assert impute_cook_from_pvi(df).iloc[0] == 4.0

    def test_no_pvi_leaves_nan(self):
        df = pd.DataFrame([_row(cook=None, cpvi=None)])
        assert pd.isna(impute_cook_from_pvi(df).iloc[0])

    def test_no_target_leakage_via_margin(self):
        """Identical PVIs and missing cooks → identical imputed cooks
        regardless of the margin column. Documents the invariant."""
        df = pd.DataFrame([
            _row(cook=None, cpvi=5, margin=0.4),
            _row(cook=None, cpvi=5, margin=-0.4),
            _row(cook=None, cpvi=5, margin=0.0),
        ])
        out = impute_cook_from_pvi(df)
        assert out.nunique() == 1
        assert out.iloc[0] == 6.0


class TestFillRemainingWithTossup:
    def test_nans_become_4(self):
        s = pd.Series([1.0, np.nan, 7.0, np.nan])
        out = fill_remaining_with_tossup(s)
        assert out.tolist() == [1.0, 4.0, 7.0, 4.0]

    def test_no_nans_passes_through(self):
        s = pd.Series([1.0, 2.0, 3.0])
        out = fill_remaining_with_tossup(s)
        assert out.tolist() == [1.0, 2.0, 3.0]
