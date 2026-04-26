"""Tests for backtest.py — orchestrator, BacktestRow, bootstrap CI, multi-N.

allocation.metric_pct_to_close_races is unit-tested in test_allocation.py.
This file tests the integration: train on synthetic 2022, test on synthetic
2024, verify the row format, multi-N expansion, and JSONL output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oath_score.backtest import (
    BacktestRow,
    NMetric,
    N_GRID,
    append_jsonl,
    format_row,
    run_backtest,
)


def _synthetic_cycle_parquet(cycle: int, snapshot: str, dest: Path, *, seed: int) -> Path:
    """Build a tiny per-cycle feature parquet matching features.py output schema."""
    rng = np.random.default_rng(seed)
    n_districts = 300  # 600 candidates total — enough close-D-wins to fit on
    rows = []
    for d in range(n_districts):
        cook = float(rng.choice([1, 2, 3, 4, 5, 6, 7]))
        true_margin = (cook - 4) / 6 + rng.normal(0, 0.15)
        true_margin = float(np.clip(true_margin, -0.95, 0.95))
        for party in ("D", "R"):
            sign = 1 if party == "D" else -1
            rows.append({
                "cycle": cycle,
                "snapshot": snapshot,
                "snapshot_date": f"{cycle}-09-09",
                "state_abbr": "ZZ",
                "district": d,
                "party_major": party,
                "last_name": f"{party}{d}",
                "candidate_name": f"{party} {d}",
                "cand_id": f"{party}{cycle}{d:03d}",
                "cook_rating": cook,
                "incumbent": int(rng.integers(0, 2)),
                "margin_pct": true_margin * sign,
                "winner": (sign * true_margin) > 0,
                "total_trans": float(rng.integers(50_000, 5_000_000)),
            })
    df = pd.DataFrame(rows)
    out = dest / f"candidates_{cycle}_{snapshot}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def _make_row(**overrides) -> BacktestRow:
    """Helper: build a BacktestRow with sensible defaults for tests that don't care."""
    defaults = dict(
        feature_set="naive",
        model="logistic",
        combine="competitiveness",
        snapshot="T-60",
        test_cycle=2024,
        train_cycles=(2022,),
        universe="all",
        n_dem_candidates=200,
        metrics=tuple(NMetric(n=n, model_score=0.5, fundraising_score=0.3, oracle_score=1.0)
                      for n in N_GRID),
        headline_n=10,
        headline_model_ci=(0.40, 0.60),
        headline_fund_ci=(0.20, 0.40),
        headline_cook_final_ci=None,
        pivotal_dollar_share=None,
        pivotal_ci=None,
        floor_saturation_efficiency=None,
        floor_saturation_ci=None,
        need_alpha=None,
        bootstrap_reps=100,
        notes="",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    defaults.update(overrides)
    return BacktestRow(**defaults)


class TestRunBacktest:
    @pytest.fixture
    def synthetic_dir(self, tmp_path):
        _synthetic_cycle_parquet(2022, "T-60", tmp_path, seed=1)
        _synthetic_cycle_parquet(2024, "T-60", tmp_path, seed=2)
        return tmp_path

    def test_naive_end_to_end(self, synthetic_dir):
        row = run_backtest(
            feature_set="naive",
            snapshot="T-60",
            train_cycles=(2022,),
            test_cycle=2024,
            processed_dir=synthetic_dir,
            bootstrap_reps=50,
        )
        assert isinstance(row, BacktestRow)
        assert row.feature_set == "naive"
        assert row.universe == "all"
        assert len(row.metrics) == len(N_GRID)
        assert {m.n for m in row.metrics} == set(N_GRID)
        for m in row.metrics:
            assert 0 <= m.model_score <= 1
            assert 0 <= m.fundraising_score <= 1
            assert 0 <= m.oracle_score <= 1
            # Oracle is the upper bound at every N
            assert m.oracle_score >= m.model_score - 1e-9
            assert m.oracle_score >= m.fundraising_score - 1e-9

    def test_bootstrap_ci_brackets_point_estimate(self, synthetic_dir):
        row = run_backtest(
            feature_set="naive",
            snapshot="T-60",
            train_cycles=(2022,),
            test_cycle=2024,
            processed_dir=synthetic_dir,
            bootstrap_reps=200,
        )
        head = row.headline
        lo, hi = row.headline_model_ci
        # Point estimate should usually fall inside its CI; allow a tiny margin
        # for the discreteness of bootstrap percentiles on small reps.
        assert lo - 0.05 <= head.model_score <= hi + 0.05
        assert lo <= hi

    def test_universe_wikipedia_smaller(self, synthetic_dir):
        # Synthetic data has cook_rating for all rows (no NaN), so wikipedia
        # universe should equal all universe in size. Real data differs.
        row_all = run_backtest(
            feature_set="naive", snapshot="T-60", train_cycles=(2022,),
            test_cycle=2024, processed_dir=synthetic_dir, universe="all",
            bootstrap_reps=20,
        )
        row_wiki = run_backtest(
            feature_set="naive", snapshot="T-60", train_cycles=(2022,),
            test_cycle=2024, processed_dir=synthetic_dir, universe="wikipedia",
            bootstrap_reps=20,
        )
        assert row_all.n_dem_candidates >= row_wiki.n_dem_candidates

    def test_unknown_feature_set_raises(self, synthetic_dir):
        with pytest.raises(KeyError):
            run_backtest(
                feature_set="not-a-real-set",
                snapshot="T-60",
                train_cycles=(2022,),
                test_cycle=2024,
                processed_dir=synthetic_dir,
                bootstrap_reps=10,
            )

    def test_unknown_universe_raises(self, synthetic_dir):
        with pytest.raises(ValueError):
            run_backtest(
                feature_set="naive",
                snapshot="T-60",
                train_cycles=(2022,),
                test_cycle=2024,
                processed_dir=synthetic_dir,
                universe="bananas",
                bootstrap_reps=10,
            )

    def test_headline_n_must_be_in_grid(self, synthetic_dir):
        with pytest.raises(ValueError, match="headline_n"):
            run_backtest(
                feature_set="naive",
                snapshot="T-60",
                train_cycles=(2022,),
                test_cycle=2024,
                processed_dir=synthetic_dir,
                headline_n=7,  # not in default N_GRID
                bootstrap_reps=10,
            )

    def test_missing_parquet_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_backtest(
                feature_set="naive",
                snapshot="T-60",
                train_cycles=(2022,),
                test_cycle=2024,
                processed_dir=tmp_path,
                bootstrap_reps=10,
            )


class TestJsonlOutput:
    def test_append_creates_file(self, tmp_path):
        row = _make_row()
        out = tmp_path / "results.jsonl"
        append_jsonl(row, out)
        assert out.exists()
        with out.open() as fh:
            d = json.loads(fh.readline())
        assert d["feature_set"] == "naive"
        assert d["universe"] == "all"
        assert d["headline_n"] == 10
        assert "metrics" in d and len(d["metrics"]) == len(N_GRID)
        assert d["headline_model_ci"] == [0.40, 0.60]

    def test_append_preserves_existing_rows(self, tmp_path):
        out = tmp_path / "results.jsonl"
        for i in range(3):
            append_jsonl(_make_row(feature_set=f"f{i}"), out)
        with out.open() as fh:
            lines = fh.readlines()
        assert len(lines) == 3
        assert json.loads(lines[0])["feature_set"] == "f0"
        assert json.loads(lines[2])["feature_set"] == "f2"


class TestFormatRow:
    def test_renders_with_n_sweep_and_ci(self):
        text = format_row(_make_row())
        assert "naive" in text
        assert "universe=all" in text
        assert "headline" in text
        assert "95% CI" in text
        # Each N should appear as a row in the sweep table
        for n in N_GRID:
            assert f"  {n:>4} " in text or f" {n:>4}\t" in text or str(n) in text


class TestBacktestRowAccessors:
    def test_headline_property(self):
        row = _make_row()
        assert row.headline.n == row.headline_n
