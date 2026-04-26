"""Tests for scores/deciling.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oath_score.scores.deciling import (
    DecileThresholds,
    N_BINS,
    calibrate,
)


class TestCalibrate:
    def test_uniform_distribution_gives_evenly_spaced(self):
        scores = pd.Series(np.linspace(0, 1, 1000))
        d = calibrate(scores, cycles_calibrated_on=(2014, 2016, 2022))
        assert len(d.cutpoints) == N_BINS - 1
        # Roughly 0.1, 0.2, ..., 0.9 for a uniform distribution
        for c, expected in zip(d.cutpoints, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
            assert abs(c - expected) < 0.02

    def test_returns_sorted(self):
        scores = pd.Series(np.random.default_rng(0).random(500))
        d = calibrate(scores, cycles_calibrated_on=(2024,))
        assert list(d.cutpoints) == sorted(d.cutpoints)

    def test_handles_tied_scores(self):
        # All zero — cutpoints should be made strictly increasing
        scores = pd.Series(np.zeros(100))
        d = calibrate(scores, cycles_calibrated_on=(2024,))
        assert list(d.cutpoints) == sorted(set(d.cutpoints))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            calibrate(pd.Series([], dtype=float), cycles_calibrated_on=())

    def test_drops_nans(self):
        scores = pd.Series([np.nan, np.nan, *np.linspace(0, 1, 10)])
        d = calibrate(scores, cycles_calibrated_on=(2024,))
        assert len(d.cutpoints) == N_BINS - 1


class TestApply:
    def test_round_trip_uniform(self):
        scores = pd.Series(np.linspace(0, 1, 1000))
        d = calibrate(scores, cycles_calibrated_on=(2024,))
        bins = d.apply(scores)
        # Each bin should have ~100 elements
        counts = bins.value_counts().to_dict()
        for b in range(1, N_BINS + 1):
            assert 80 <= counts.get(b, 0) <= 120, counts

    def test_returns_series_aligned(self):
        scores = pd.Series([0.05, 0.55, 0.95], index=[10, 20, 30])
        d = calibrate(pd.Series(np.linspace(0, 1, 1000)), cycles_calibrated_on=(2024,))
        bins = d.apply(scores)
        assert list(bins.index) == [10, 20, 30]

    def test_clipped_to_1_10(self):
        scores = pd.Series([-0.5, 1.5, 0.0, 1.0])
        d = calibrate(pd.Series(np.linspace(0, 1, 100)), cycles_calibrated_on=(2024,))
        bins = d.apply(scores)
        assert (bins >= 1).all() and (bins <= N_BINS).all()

    def test_nan_input_treated_as_zero(self):
        d = calibrate(pd.Series(np.linspace(0, 1, 100)), cycles_calibrated_on=(2024,))
        bins = d.apply(pd.Series([np.nan, 0.0]))
        assert bins.iloc[0] == bins.iloc[1] == 1

    def test_constructor_validates_count(self):
        with pytest.raises(ValueError, match="9 cutpoints"):
            DecileThresholds(cutpoints=(0.5,), cycles_calibrated_on=(2024,))

    def test_constructor_validates_sorted(self):
        with pytest.raises(ValueError, match="sorted"):
            DecileThresholds(
                cutpoints=(0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
                cycles_calibrated_on=(2024,),
            )


class TestRoundTripIO:
    def test_save_load(self, tmp_path):
        d = calibrate(pd.Series(np.linspace(0, 1, 100)), cycles_calibrated_on=(2014, 2024))
        p = tmp_path / "deciles.json"
        d.save(p)
        loaded = DecileThresholds.load(p)
        assert loaded.cutpoints == d.cutpoints
        assert loaded.cycles_calibrated_on == d.cycles_calibrated_on
