"""Tests for backtest.py — orchestrator and BacktestRow.

allocation.metric_pct_to_close_races is unit-tested in test_allocation.py.
This file tests the integration: train on synthetic 2022, test on synthetic
2024, verify the row format and that all three reference lines compute.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oath_score.backtest import (
    BacktestRow,
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
        # Margin correlated with cook: cook=4 → ~0, cook=7 → ~+0.5; wider noise
        # so we get enough close races even at high/low cook ratings.
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


class TestRunBacktest:
    def test_naive_end_to_end(self, tmp_path):
        _synthetic_cycle_parquet(2022, "T-60", tmp_path, seed=1)
        _synthetic_cycle_parquet(2024, "T-60", tmp_path, seed=2)

        row = run_backtest(
            feature_set="naive",
            snapshot="T-60",
            train_cycles=(2022,),
            test_cycle=2024,
            processed_dir=tmp_path,
            n=10,
        )

        assert isinstance(row, BacktestRow)
        assert row.feature_set == "naive"
        assert row.snapshot == "T-60"
        assert row.test_cycle == 2024
        assert row.train_cycles == (2022,)
        assert 0 <= row.model_score <= 1
        assert 0 <= row.fundraising_score <= 1
        assert 0 <= row.oracle_score <= 1
        # Oracle should always be ≥ both other lines (it's the upper bound)
        assert row.oracle_score >= row.model_score - 1e-9
        assert row.oracle_score >= row.fundraising_score - 1e-9

    def test_unknown_feature_set_raises(self, tmp_path):
        _synthetic_cycle_parquet(2022, "T-60", tmp_path, seed=1)
        _synthetic_cycle_parquet(2024, "T-60", tmp_path, seed=2)
        with pytest.raises(NotImplementedError):
            run_backtest(
                feature_set="full",
                snapshot="T-60",
                train_cycles=(2022,),
                test_cycle=2024,
                processed_dir=tmp_path,
            )

    def test_missing_parquet_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_backtest(
                feature_set="naive",
                snapshot="T-60",
                train_cycles=(2022,),
                test_cycle=2024,
                processed_dir=tmp_path,
            )


class TestJsonlOutput:
    def test_append_creates_file(self, tmp_path):
        row = BacktestRow(
            feature_set="naive", snapshot="T-60", test_cycle=2024,
            train_cycles=(2022,), n=10, n_dem_candidates=50,
            model_score=0.5, fundraising_score=0.3, oracle_score=1.0,
            notes="", timestamp="2026-01-01T00:00:00+00:00",
        )
        out = tmp_path / "results.jsonl"
        append_jsonl(row, out)
        assert out.exists()
        with out.open() as fh:
            line = fh.readline()
        d = json.loads(line)
        assert d["feature_set"] == "naive"
        assert d["model_score"] == 0.5
        assert d["train_cycles"] == [2022]

    def test_append_preserves_existing_rows(self, tmp_path):
        out = tmp_path / "results.jsonl"
        for i in range(3):
            append_jsonl(BacktestRow(
                feature_set=f"f{i}", snapshot="T-60", test_cycle=2024,
                train_cycles=(2022,), n=10, n_dem_candidates=50,
                model_score=0.1 * i, fundraising_score=0.0, oracle_score=1.0,
                notes="", timestamp="2026-01-01T00:00:00+00:00",
            ), out)
        with out.open() as fh:
            lines = fh.readlines()
        assert len(lines) == 3
        assert json.loads(lines[0])["feature_set"] == "f0"
        assert json.loads(lines[2])["feature_set"] == "f2"


class TestFormatRow:
    def test_renders_human_readable(self):
        row = BacktestRow(
            feature_set="naive", snapshot="T-60", test_cycle=2024,
            train_cycles=(2022,), n=10, n_dem_candidates=200,
            model_score=0.42, fundraising_score=0.31, oracle_score=1.0,
            notes="", timestamp="2026-01-01T00:00:00+00:00",
        )
        text = format_row(row)
        assert "naive" in text
        assert "T-60" in text
        assert "0.4200" in text
        assert "+0.1100" in text  # delta
