"""Tests for calibration.py — α grid LOO and ablation helpers.

Pure unit tests on the small helpers; the heavyweight runners are exercised
by the smoke run in 7.5 (would require synthetic 4-cycle parquets to fully
unit-test, which is overkill for the planning-and-summary helpers).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oath_score.calibration import (
    DEFAULT_ALPHA_GRID,
    DEFAULT_TRAIN_CYCLES,
    best_alpha,
)


class TestBestAlpha:
    def test_picks_highest_close_race(self):
        df = pd.DataFrame({
            "alpha": [0.0, 0.3, 1.0],
            "n_folds": [3, 3, 3],
            "mean_close_race": [0.5, 0.7, 0.6],
            "std_close_race": [0.1, 0.1, 0.1],
            "mean_pivotal": [1.0, 1.0, 1.0],
            "mean_floor_sat": [0.0, 0.0, 0.0],
        })
        assert best_alpha(df) == 0.3

    def test_tie_break_to_smaller_alpha(self):
        df = pd.DataFrame({
            "alpha": [0.1, 0.5, 1.0],
            "n_folds": [3, 3, 3],
            "mean_close_race": [0.7, 0.7, 0.7],
            "std_close_race": [0.1, 0.1, 0.1],
            "mean_pivotal": [1.0, 1.0, 1.0],
            "mean_floor_sat": [0.0, 0.0, 0.0],
        })
        assert best_alpha(df) == 0.1

    def test_all_nan_returns_default(self):
        df = pd.DataFrame({
            "alpha": [0.1, 0.5],
            "n_folds": [0, 0],
            "mean_close_race": [float("nan"), float("nan")],
            "std_close_race": [float("nan"), float("nan")],
            "mean_pivotal": [float("nan"), float("nan")],
            "mean_floor_sat": [float("nan"), float("nan")],
        })
        from oath_score.backtest import DEFAULT_NEED_ALPHA
        assert best_alpha(df) == DEFAULT_NEED_ALPHA

    def test_some_nan_skipped(self):
        df = pd.DataFrame({
            "alpha": [0.1, 0.5, 1.0],
            "n_folds": [0, 3, 3],
            "mean_close_race": [float("nan"), 0.6, 0.5],
            "std_close_race": [float("nan"), 0.1, 0.1],
            "mean_pivotal": [float("nan"), 1.0, 1.0],
            "mean_floor_sat": [float("nan"), 0.0, 0.0],
        })
        assert best_alpha(df) == 0.5  # NaN α=0.1 ignored


class TestDefaults:
    def test_alpha_grid_includes_zero(self):
        assert 0.0 in DEFAULT_ALPHA_GRID

    def test_alpha_grid_monotonic(self):
        assert list(DEFAULT_ALPHA_GRID) == sorted(DEFAULT_ALPHA_GRID)

    def test_default_train_cycles_three_cycles(self):
        assert len(DEFAULT_TRAIN_CYCLES) == 3
        assert 2014 in DEFAULT_TRAIN_CYCLES
        assert 2024 not in DEFAULT_TRAIN_CYCLES  # never train on test
