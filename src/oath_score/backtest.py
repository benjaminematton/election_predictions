"""End-to-end backtest harness.

Loads per-(cycle, snapshot) feature matrices, fits a competitiveness model on
the training cycle(s), scores Dem candidates in the test cycle, runs the
allocation function, and computes the headline metric for the model and two
reference lines (fundraising-proportional baseline, hindsight oracle).

The Cook-final benchmark is deferred to Phase 7 — it needs a separate ratings
fetch at election-1week, which we'll add when we close out the calibration step.

One row per call is appended to data/processed/backtest_results.jsonl, which is
git-tracked so the improvement curve is visible in repo history.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from oath_score.allocation import allocate, metric_pct_to_close_races
from oath_score.constants import CYCLES, SNAPSHOT_OFFSETS_DAYS
from oath_score.feature_sets import get as get_feature_set
from oath_score.scores.competitiveness import NaiveCompetitiveness, SCORE_COL


# ----- result row -----

@dataclass(frozen=True)
class BacktestRow:
    feature_set: str
    snapshot: str
    test_cycle: int
    train_cycles: tuple[int, ...]
    n: int                              # top-N used in allocation
    n_dem_candidates: int               # universe size before top-N
    model_score: float
    fundraising_score: float
    oracle_score: float
    notes: str
    timestamp: str

    def as_dict(self) -> dict:
        d = asdict(self)
        d["train_cycles"] = list(self.train_cycles)
        return d


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


def run_backtest(
    *,
    feature_set: str,
    snapshot: str,
    train_cycles: tuple[int, ...],
    test_cycle: int,
    processed_dir: Path,
    n: int = 10,
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

    train_df = pd.concat(
        [load_processed(c, snapshot, processed_dir) for c in train_cycles],
        ignore_index=True,
    )
    test_df = load_processed(test_cycle, snapshot, processed_dir)

    model = NaiveCompetitiveness(feature_set_name=feature_set).fit(train_df)
    scored = model.score(test_df)

    # Restrict universe to Dem candidates in Wikipedia-tracked races. All three
    # reference lines (model / fundraising / oracle) run on the *same* universe
    # so the comparison is apples-to-apples. Oath wouldn't recommend a Solid-X
    # race anyway, so Wikipedia coverage = the donor-relevant universe.
    dems = scored.loc[
        (scored["party_major"] == "D") & scored["cook_rating"].notna()
    ].copy()
    n_dem = len(dems)

    # --- Three reference lines, all using the same allocation function ---

    # 1. Model: score by competitiveness output
    model_alloc = allocate(dems, score_col=SCORE_COL, n=n)
    model_metric = metric_pct_to_close_races(model_alloc)

    # 2. Fundraising baseline: score by snapshot fundraising
    fund_alloc = allocate(dems, score_col="total_trans", n=n)
    fund_metric = metric_pct_to_close_races(fund_alloc)

    # 3. Hindsight oracle: score = 1 if race finished <5%, else 0
    dems = dems.copy()
    dems["__oracle_score__"] = (dems["margin_pct"].abs() < 0.05).astype(float)
    oracle_alloc = allocate(dems, score_col="__oracle_score__", n=n)
    oracle_metric = metric_pct_to_close_races(oracle_alloc)

    return BacktestRow(
        feature_set=feature_set,
        snapshot=snapshot,
        test_cycle=test_cycle,
        train_cycles=tuple(train_cycles),
        n=n,
        n_dem_candidates=n_dem,
        model_score=round(model_metric, 4),
        fundraising_score=round(fund_metric, 4),
        oracle_score=round(oracle_metric, 4),
        notes=notes,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def append_jsonl(row: BacktestRow, out_path: Path) -> None:
    """Append one row to the JSONL log. Creates parent dir if needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row.as_dict()) + "\n")


def format_row(row: BacktestRow) -> str:
    delta = row.model_score - row.fundraising_score
    sign = "+" if delta >= 0 else ""
    return (
        f"[backtest] feature_set={row.feature_set} snapshot={row.snapshot} "
        f"test={row.test_cycle} train={list(row.train_cycles)}\n"
        f"  model_score:        {row.model_score:.4f}  (top-{row.n} of {row.n_dem_candidates} Dems)\n"
        f"  fundraising_score:  {row.fundraising_score:.4f}\n"
        f"  oracle_score:       {row.oracle_score:.4f}\n"
        f"  Δ vs baseline:      {sign}{delta:.4f}"
    )


# ----- CLI -----

def _parse_cycles(s: Iterable[str]) -> tuple[int, ...]:
    out = []
    for x in s:
        c = int(x)
        if c not in CYCLES:
            raise argparse.ArgumentTypeError(
                f"cycle {c} not in {list(CYCLES)}"
            )
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
    parser.add_argument("--n", type=int, default=10, help="top-N for allocation")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/processed/backtest_results.jsonl"))
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    # Validate feature set name (helpful error before we get into the model)
    get_feature_set(args.features)

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
            n=args.n,
            notes=args.notes,
        )
        print(format_row(row))
        append_jsonl(row, args.out)
        print(f"  → appended to {args.out}")


if __name__ == "__main__":
    main()
