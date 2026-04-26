"""Phase 7 calibration: α grid + N sweep + 2014-cycle ablation.

All three rely on `run_backtest` and never touch the held-out test cycle
(2024) for parameter selection. The selection happens via leave-one-out
across the training cycles, then refit + report on 2024.

Public functions:
  * alpha_grid_search — LOO across train_cycles, sweep α, return per-α metric
  * best_n_per_alpha   — pick top-N from existing JSONL or LOO results
  * cycle_ablation     — train with vs without 2014, compare 2024 metric

The CLI runs the full sweep and writes summary CSVs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from oath_score.backtest import (
    BOOTSTRAP_REPS,
    DEFAULT_NEED_ALPHA,
    HEADLINE_N,
    N_GRID,
    run_backtest,
)
from oath_score.constants import CYCLES, SNAPSHOT_OFFSETS_DAYS

DEFAULT_ALPHA_GRID: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0)
DEFAULT_TRAIN_CYCLES: tuple[int, ...] = (2014, 2016, 2022)


def alpha_grid_search(
    *,
    feature_set: str,
    snapshot: str,
    train_cycles: tuple[int, ...],
    processed_dir: Path,
    raw_dir: Path,
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
    headline_n: int = HEADLINE_N,
    bootstrap_reps: int = 100,
) -> pd.DataFrame:
    """Leave-one-out across `train_cycles`; sweep α; return one row per α.

    Each row = mean (across LOO folds) headline close-race metric for the
    multi-quantile + impact backtest at that α value.
    """
    if len(train_cycles) < 2:
        raise ValueError("Need >=2 train cycles for leave-one-out")

    rows: list[dict] = []
    for alpha in alpha_grid:
        fold_metrics: list[float] = []
        fold_pivotal: list[float] = []
        fold_floor_sat: list[float] = []
        for held_out in train_cycles:
            train_subset = tuple(c for c in train_cycles if c != held_out)
            try:
                row = run_backtest(
                    feature_set=feature_set,
                    snapshot=snapshot,
                    train_cycles=train_subset,
                    test_cycle=held_out,
                    processed_dir=processed_dir,
                    raw_dir=raw_dir,
                    model="multi-quantile",
                    universe="all",
                    combine="impact",
                    headline_n=headline_n,
                    bootstrap_reps=bootstrap_reps,
                    need_alpha=alpha,
                    notes=f"phase7-alpha-loo (alpha={alpha}, held_out={held_out})",
                )
            except Exception as exc:
                # If the held-out cycle is back-filled-only (e.g. 2014), the
                # ratings predictive may be too thin. Skip the fold but flag.
                print(f"[calibration] α={alpha} held_out={held_out} FAILED: {exc}")
                continue
            fold_metrics.append(row.headline.model_score)
            if row.pivotal_dollar_share is not None:
                fold_pivotal.append(row.pivotal_dollar_share)
            if row.floor_saturation_efficiency is not None:
                fold_floor_sat.append(row.floor_saturation_efficiency)

        rows.append({
            "alpha": alpha,
            "n_folds": len(fold_metrics),
            "mean_close_race": float(np.mean(fold_metrics)) if fold_metrics else float("nan"),
            "std_close_race": float(np.std(fold_metrics)) if fold_metrics else float("nan"),
            "mean_pivotal": float(np.mean(fold_pivotal)) if fold_pivotal else float("nan"),
            "mean_floor_sat": float(np.mean(fold_floor_sat)) if fold_floor_sat else float("nan"),
        })
    return pd.DataFrame(rows)


def best_alpha(grid_results: pd.DataFrame) -> float:
    """Pick α that maximizes mean_close_race; tie-break to smaller α (simpler)."""
    df = grid_results.dropna(subset=["mean_close_race"]).copy()
    if df.empty:
        return DEFAULT_NEED_ALPHA
    df = df.sort_values(["mean_close_race", "alpha"], ascending=[False, True])
    return float(df.iloc[0]["alpha"])


def n_sensitivity_table(
    *,
    feature_set: str,
    snapshot: str,
    train_cycles: tuple[int, ...],
    test_cycle: int,
    processed_dir: Path,
    raw_dir: Path,
    need_alpha: float,
    bootstrap_reps: int = 100,
) -> pd.DataFrame:
    """Single backtest run; expose its N-grid metrics as a tidy DataFrame."""
    row = run_backtest(
        feature_set=feature_set,
        snapshot=snapshot,
        train_cycles=train_cycles,
        test_cycle=test_cycle,
        processed_dir=processed_dir,
        raw_dir=raw_dir,
        model="multi-quantile",
        universe="all",
        combine="impact",
        bootstrap_reps=bootstrap_reps,
        need_alpha=need_alpha,
        notes=f"phase7-n-sensitivity (alpha={need_alpha})",
    )
    return pd.DataFrame([asdict(m) for m in row.metrics])


def cycle_ablation(
    *,
    feature_set: str,
    snapshot: str,
    test_cycle: int,
    processed_dir: Path,
    raw_dir: Path,
    need_alpha: float,
    bootstrap_reps: int = 100,
) -> pd.DataFrame:
    """Compare train=[2014, 2016, 2022] vs train=[2016, 2022] at fixed α."""
    rows = []
    for label, cycles in [
        ("with_2014", (2014, 2016, 2022)),
        ("without_2014", (2016, 2022)),
    ]:
        row = run_backtest(
            feature_set=feature_set,
            snapshot=snapshot,
            train_cycles=cycles,
            test_cycle=test_cycle,
            processed_dir=processed_dir,
            raw_dir=raw_dir,
            model="multi-quantile",
            universe="all",
            combine="impact",
            bootstrap_reps=bootstrap_reps,
            need_alpha=need_alpha,
            notes=f"phase7-ablation ({label}, alpha={need_alpha})",
        )
        rows.append({
            "label": label,
            "train_cycles": list(cycles),
            "headline_close_race": row.headline.model_score,
            "headline_cook_final": row.headline.cook_final_score,
            "pivotal_dollar_share": row.pivotal_dollar_share,
            "floor_saturation_efficiency": row.floor_saturation_efficiency,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 calibration sweeps.")
    parser.add_argument("--features", default="full")
    parser.add_argument("--snapshot", choices=list(SNAPSHOT_OFFSETS_DAYS), default="T-60")
    parser.add_argument("--train", nargs="+", default=[str(c) for c in DEFAULT_TRAIN_CYCLES])
    parser.add_argument("--test", type=int, default=2024)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--bootstrap-reps", type=int, default=100)
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=list(DEFAULT_ALPHA_GRID))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    train_cycles = tuple(int(c) for c in args.train)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 7 calibration")
    print(f"  features={args.features} snapshot={args.snapshot}")
    print(f"  train={list(train_cycles)} test={args.test}")
    print("=" * 60)

    # 1. α grid search via LOO
    print("\n[1/3] α grid search (leave-one-out across train_cycles)")
    alpha_results = alpha_grid_search(
        feature_set=args.features,
        snapshot=args.snapshot,
        train_cycles=train_cycles,
        processed_dir=args.processed_dir,
        raw_dir=args.raw_dir,
        alpha_grid=tuple(args.alpha_grid),
        bootstrap_reps=args.bootstrap_reps,
    )
    print(alpha_results.to_string(index=False))
    alpha_results.to_csv(args.out_dir / "phase7_alpha_grid.csv", index=False)

    alpha_star = best_alpha(alpha_results)
    print(f"\n→ best α* = {alpha_star}")

    # 2. N sensitivity at α* on actual test cycle
    print(f"\n[2/3] N sensitivity at α*={alpha_star} on {args.test}")
    n_results = n_sensitivity_table(
        feature_set=args.features,
        snapshot=args.snapshot,
        train_cycles=train_cycles,
        test_cycle=args.test,
        processed_dir=args.processed_dir,
        raw_dir=args.raw_dir,
        need_alpha=alpha_star,
        bootstrap_reps=args.bootstrap_reps,
    )
    print(n_results.to_string(index=False))
    n_results.to_csv(args.out_dir / "phase7_n_sensitivity.csv", index=False)

    # 3. 2014-cycle ablation at α*
    print(f"\n[3/3] 2014-cycle ablation at α*={alpha_star}")
    abl = cycle_ablation(
        feature_set=args.features,
        snapshot=args.snapshot,
        test_cycle=args.test,
        processed_dir=args.processed_dir,
        raw_dir=args.raw_dir,
        need_alpha=alpha_star,
        bootstrap_reps=args.bootstrap_reps,
    )
    print(abl.to_string(index=False))
    abl.to_csv(args.out_dir / "phase7_cycle_ablation.csv", index=False)

    summary = {
        "alpha_star": alpha_star,
        "alpha_grid": list(args.alpha_grid),
        "snapshot": args.snapshot,
        "feature_set": args.features,
        "train_cycles": list(train_cycles),
        "test_cycle": args.test,
    }
    (args.out_dir / "phase7_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n→ summary written to {args.out_dir / 'phase7_summary.json'}")


if __name__ == "__main__":
    main()
