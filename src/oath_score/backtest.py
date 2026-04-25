"""End-to-end backtest harness.

For one (feature_set, snapshot, train_cycles, test_cycle, universe) tuple:
  * Fit the competitiveness model on training cycles.
  * Score Dem candidates in the test cycle.
  * For each N in N_GRID, run the allocation function and compute the
    headline metric for the model and two reference lines (fundraising
    baseline, hindsight oracle).
  * Bootstrap a 95% CI on the model and fundraising scores at the default
    headline N, so the curve has uncertainty bars from day one.

Universe options (`--universe` / `universe=`):
  * "all"       - every Dem candidate in a contested two-party general
                  election (default; the honest measurement).
  * "wikipedia" - restrict to Dems in Wikipedia-tracked races (matches
                  Oath's product surface but inflates fundraising baseline).

The Cook-final benchmark is deferred to Phase 7 — it needs a separate
ratings fetch at election-1week.

One row per call is appended to data/processed/backtest_results.jsonl,
git-tracked so the improvement curve is visible in repo history.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from oath_score.allocation import allocate, metric_pct_to_close_races
from oath_score.constants import CYCLES, SNAPSHOT_OFFSETS_DAYS
from oath_score.feature_sets import get as get_feature_set
from oath_score.scores.competitiveness import NaiveCompetitiveness, SCORE_COL


# Range of top-N values to sweep per backtest row.
N_GRID: tuple[int, ...] = (1, 3, 5, 10, 20, 50)
HEADLINE_N: int = 10
BOOTSTRAP_REPS: int = 1000
BOOTSTRAP_SEED: int = 42

ORACLE_COL = "__oracle_score__"
FUND_COL = "total_trans"

UniverseChoice = str  # "all" | "wikipedia"


# ----- result row -----

@dataclass(frozen=True)
class NMetric:
    n: int
    model_score: float
    fundraising_score: float
    oracle_score: float


@dataclass(frozen=True)
class BacktestRow:
    feature_set: str
    snapshot: str
    test_cycle: int
    train_cycles: tuple[int, ...]
    universe: UniverseChoice
    n_dem_candidates: int
    metrics: tuple[NMetric, ...]            # one entry per N in N_GRID
    headline_n: int                          # which N in metrics is the canonical one
    headline_model_ci: tuple[float, float]   # (low, high) at 95% confidence
    headline_fund_ci: tuple[float, float]
    bootstrap_reps: int
    notes: str
    timestamp: str

    def as_dict(self) -> dict:
        return {
            "feature_set": self.feature_set,
            "snapshot": self.snapshot,
            "test_cycle": self.test_cycle,
            "train_cycles": list(self.train_cycles),
            "universe": self.universe,
            "n_dem_candidates": self.n_dem_candidates,
            "metrics": [asdict(m) for m in self.metrics],
            "headline_n": self.headline_n,
            "headline_model_ci": list(self.headline_model_ci),
            "headline_fund_ci": list(self.headline_fund_ci),
            "bootstrap_reps": self.bootstrap_reps,
            "notes": self.notes,
            "timestamp": self.timestamp,
        }

    @property
    def headline(self) -> NMetric:
        return next(m for m in self.metrics if m.n == self.headline_n)


# ----- public API -----

def load_processed(cycle: int, snapshot: str, processed_dir: Path) -> pd.DataFrame:
    """Read a per-(cycle, snapshot) feature parquet."""
    path = processed_dir / f"candidates_{cycle}_{snapshot}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No feature matrix at {path}. Run "
            f"`python -m oath_score.features --cycle {cycle} --snapshot {snapshot}` first."
        )
    return pd.read_parquet(path)


def _filter_universe(df: pd.DataFrame, universe: UniverseChoice) -> pd.DataFrame:
    """Apply the universe filter to scored Dem candidates."""
    dems = df.loc[df["party_major"] == "D"].copy()
    if universe == "wikipedia":
        dems = dems.loc[dems["cook_rating"].notna()].copy()
    elif universe != "all":
        raise ValueError(f"unknown universe={universe!r}; want 'all' or 'wikipedia'")
    return dems


def _scores_at_n(dems: pd.DataFrame, n: int) -> NMetric:
    """Compute model / fund / oracle metrics at one N."""
    model_alloc = allocate(dems, score_col=SCORE_COL, n=n)
    fund_alloc = allocate(dems, score_col=FUND_COL, n=n)
    oracle_alloc = allocate(dems, score_col=ORACLE_COL, n=n)
    return NMetric(
        n=n,
        model_score=round(metric_pct_to_close_races(model_alloc), 4),
        fundraising_score=round(metric_pct_to_close_races(fund_alloc), 4),
        oracle_score=round(metric_pct_to_close_races(oracle_alloc), 4),
    )


def _bootstrap_ci(
    dems: pd.DataFrame,
    score_col: str,
    n: int,
    reps: int,
    seed: int,
) -> tuple[float, float]:
    """95% percentile bootstrap CI on `score_col` -> top-N -> close-race metric."""
    rng = np.random.default_rng(seed)
    samples = np.empty(reps, dtype=float)
    for i in range(reps):
        boot = dems.sample(n=len(dems), replace=True, random_state=rng.integers(1, 10**9))
        alloc = allocate(boot, score_col=score_col, n=n)
        samples[i] = metric_pct_to_close_races(alloc)
    return (
        round(float(np.percentile(samples, 2.5)), 4),
        round(float(np.percentile(samples, 97.5)), 4),
    )


def run_backtest(
    *,
    feature_set: str,
    snapshot: str,
    train_cycles: tuple[int, ...],
    test_cycle: int,
    processed_dir: Path,
    universe: UniverseChoice = "all",
    headline_n: int = HEADLINE_N,
    n_grid: tuple[int, ...] = N_GRID,
    bootstrap_reps: int = BOOTSTRAP_REPS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    notes: str = "",
) -> BacktestRow:
    """Fit on `train_cycles`, score `test_cycle`, return a BacktestRow.

    Currently only the `naive` feature_set is wired up — Phase 4 will register
    additional models in feature_sets.py and dispatch here.
    """
    if feature_set != "naive":
        raise NotImplementedError(
            f"feature_set={feature_set!r} not implemented yet. Phase 4 will add the rest."
        )
    if headline_n not in n_grid:
        raise ValueError(f"headline_n={headline_n} must be in n_grid={n_grid}")

    train_df = pd.concat(
        [load_processed(c, snapshot, processed_dir) for c in train_cycles],
        ignore_index=True,
    )
    test_df = load_processed(test_cycle, snapshot, processed_dir)

    model = NaiveCompetitiveness(feature_set_name=feature_set).fit(train_df)
    scored = model.score(test_df)
    dems = _filter_universe(scored, universe)
    dems[ORACLE_COL] = (dems["margin_pct"].abs() < 0.05).astype(float)
    n_dem = len(dems)

    metrics = tuple(_scores_at_n(dems, n) for n in n_grid)

    model_ci = _bootstrap_ci(dems, SCORE_COL, headline_n, bootstrap_reps, bootstrap_seed)
    fund_ci = _bootstrap_ci(dems, FUND_COL, headline_n, bootstrap_reps, bootstrap_seed)

    return BacktestRow(
        feature_set=feature_set,
        snapshot=snapshot,
        test_cycle=test_cycle,
        train_cycles=tuple(train_cycles),
        universe=universe,
        n_dem_candidates=n_dem,
        metrics=metrics,
        headline_n=headline_n,
        headline_model_ci=model_ci,
        headline_fund_ci=fund_ci,
        bootstrap_reps=bootstrap_reps,
        notes=notes,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def append_jsonl(row: BacktestRow, out_path: Path) -> None:
    """Append one row to the JSONL log. Creates parent dir if needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row.as_dict()) + "\n")


def format_row(row: BacktestRow) -> str:
    """Pretty-print a row including N sweep + headline CI."""
    lines = [
        f"[backtest] feature_set={row.feature_set} snapshot={row.snapshot} "
        f"test={row.test_cycle} train={list(row.train_cycles)} universe={row.universe}",
        f"  universe size: {row.n_dem_candidates} Dems",
        f"  {'N':>4} {'model':>8} {'fund':>8} {'oracle':>8} {'Δ':>8}",
    ]
    for m in row.metrics:
        delta = m.model_score - m.fundraising_score
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"  {m.n:>4} {m.model_score:>8.4f} {m.fundraising_score:>8.4f} "
            f"{m.oracle_score:>8.4f} {sign}{delta:>7.4f}"
        )
    head = row.headline
    head_delta = head.model_score - head.fundraising_score
    head_sign = "+" if head_delta >= 0 else ""
    lines.append(
        f"  headline (N={row.headline_n}): "
        f"model {head.model_score:.4f} (95% CI [{row.headline_model_ci[0]:.4f}, "
        f"{row.headline_model_ci[1]:.4f}]) vs fund {head.fundraising_score:.4f} "
        f"(95% CI [{row.headline_fund_ci[0]:.4f}, {row.headline_fund_ci[1]:.4f}]) "
        f"→ Δ {head_sign}{head_delta:.4f}"
    )
    return "\n".join(lines)


# ----- CLI -----

def _parse_cycles(s: Iterable[str]) -> tuple[int, ...]:
    out = []
    for x in s:
        c = int(x)
        if c not in CYCLES:
            raise argparse.ArgumentTypeError(f"cycle {c} not in {list(CYCLES)}")
        out.append(c)
    return tuple(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one backtest row.")
    parser.add_argument("--features", required=True,
                        help="feature-set name from feature_sets.REGISTRY")
    parser.add_argument("--snapshots", nargs="+", required=True,
                        choices=list(SNAPSHOT_OFFSETS_DAYS),
                        help="one or more snapshots (T-110, T-60, T-20)")
    parser.add_argument("--train", nargs="+", required=True,
                        help="training cycles (e.g. 2022 or 2014 2016 2022)")
    parser.add_argument("--test", type=int, required=True,
                        help="held-out test cycle")
    parser.add_argument("--universe", choices=("all", "wikipedia"), default="all",
                        help="candidate universe; default 'all' (Dems in any "
                             "contested two-party general); 'wikipedia' "
                             "restricts to Dems in Wikipedia-tracked races")
    parser.add_argument("--headline-n", type=int, default=HEADLINE_N,
                        help=f"top-N for the headline number; default {HEADLINE_N}")
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS,
                        help=f"bootstrap resamples for CI; default {BOOTSTRAP_REPS}")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/processed/backtest_results.jsonl"))
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    get_feature_set(args.features)  # validate name early

    train_cycles = _parse_cycles(args.train)
    if args.test not in CYCLES:
        parser.error(f"--test {args.test} not in {list(CYCLES)}")

    for snap in args.snapshots:
        row = run_backtest(
            feature_set=args.features,
            snapshot=snap,
            train_cycles=train_cycles,
            test_cycle=args.test,
            processed_dir=args.processed_dir,
            universe=args.universe,
            headline_n=args.headline_n,
            bootstrap_reps=args.bootstrap_reps,
            notes=args.notes,
        )
        print(format_row(row))
        append_jsonl(row, args.out)
        print(f"  → appended to {args.out}")


if __name__ == "__main__":
    main()
