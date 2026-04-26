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
from oath_score.feature_sets import REGISTRY as FEATURE_REGISTRY
from oath_score.feature_sets import get as get_feature_set
from oath_score.scores.competitiveness import LogisticCompetitiveness, SCORE_COL


# Range of top-N values to sweep per backtest row.
N_GRID: tuple[int, ...] = (1, 3, 5, 10, 20, 50)
HEADLINE_N: int = 10
BOOTSTRAP_REPS: int = 1000
BOOTSTRAP_SEED: int = 42

ORACLE_COL = "__oracle_score__"
FUND_COL = "total_trans"

UniverseChoice = str  # "all" | "wikipedia"
ModelChoice = str      # "logistic" | "multi-quantile"
CombineChoice = str    # "competitiveness" | "base" | "impact"

PIVOTAL_COL = "__is_pivotal__"
UNDER_FLOOR_COL = "__is_under_floor__"

# Hard-coded weight on the financial-need adjustment for Phase 6.
# Phase 7 grid-searches this against the headline + secondary metrics.
NEED_ALPHA: float = 0.3


def _build_model(model: ModelChoice, feature_set: str):
    """Dispatch on model name. Imports the multi-quantile model lazily so
    Phase 4.3 (logistic curve) can run before Phase 4.4 lands."""
    if model == "logistic":
        return LogisticCompetitiveness(feature_set_name=feature_set)
    if model == "multi-quantile":
        from oath_score.scores.multi_quantile import MultiQuantileCompetitiveness
        return MultiQuantileCompetitiveness(feature_set_name=feature_set)
    raise ValueError(f"unknown model={model!r}; want 'logistic' or 'multi-quantile'")


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
    model: str                                # "logistic" | "multi-quantile"
    combine: str                              # "competitiveness" | "base"
    snapshot: str
    test_cycle: int
    train_cycles: tuple[int, ...]
    universe: UniverseChoice
    n_dem_candidates: int
    metrics: tuple[NMetric, ...]            # one entry per N in N_GRID
    headline_n: int                          # which N in metrics is the canonical one
    headline_model_ci: tuple[float, float]   # (low, high) at 95% confidence
    headline_fund_ci: tuple[float, float]
    pivotal_dollar_share: float | None       # secondary metric, None if combine=competitiveness
    pivotal_ci: tuple[float, float] | None
    floor_saturation_efficiency: float | None  # only set when combine=impact
    floor_saturation_ci: tuple[float, float] | None
    bootstrap_reps: int
    notes: str
    timestamp: str

    def as_dict(self) -> dict:
        return {
            "feature_set": self.feature_set,
            "model": self.model,
            "combine": self.combine,
            "snapshot": self.snapshot,
            "test_cycle": self.test_cycle,
            "train_cycles": list(self.train_cycles),
            "universe": self.universe,
            "n_dem_candidates": self.n_dem_candidates,
            "metrics": [asdict(m) for m in self.metrics],
            "headline_n": self.headline_n,
            "headline_model_ci": list(self.headline_model_ci),
            "headline_fund_ci": list(self.headline_fund_ci),
            "pivotal_dollar_share": self.pivotal_dollar_share,
            "pivotal_ci": list(self.pivotal_ci) if self.pivotal_ci is not None else None,
            "floor_saturation_efficiency": self.floor_saturation_efficiency,
            "floor_saturation_ci": (
                list(self.floor_saturation_ci)
                if self.floor_saturation_ci is not None else None
            ),
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


def _filter_universe(df: pd.DataFrame, universe: UniverseChoice, *, naive: bool) -> pd.DataFrame:
    """Apply the universe filter to scored Dem candidates.

    For the naive feature set, "all" still requires non-NaN cook_rating —
    the model can't say anything useful otherwise (and would just emit 0).
    For richer feature sets, "all" means the full Dem contested-race
    universe; the model imputes cook from PVI internally.
    """
    dems = df.loc[df["party_major"] == "D"].copy()
    if universe == "wikipedia":
        dems = dems.loc[dems["cook_rating"].notna()].copy()
    elif universe == "all":
        if naive:
            dems = dems.loc[dems["cook_rating"].notna()].copy()
    else:
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
    model: ModelChoice = "logistic",
    universe: UniverseChoice = "all",
    combine: CombineChoice = "competitiveness",
    raw_dir: Path = Path("data/raw"),
    headline_n: int = HEADLINE_N,
    n_grid: tuple[int, ...] = N_GRID,
    bootstrap_reps: int = BOOTSTRAP_REPS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    notes: str = "",
) -> BacktestRow:
    """Fit on `train_cycles`, score `test_cycle`, return a BacktestRow.

    `combine="base"` adds chamber-control stakes to the score:
      base = sqrt(competitiveness * stakes_normalized)
    Requires `model="multi-quantile"` (only that model exposes the per-race
    predictive distribution stakes needs).
    """
    if feature_set not in FEATURE_REGISTRY:
        raise KeyError(
            f"unknown feature_set={feature_set!r}; have {sorted(FEATURE_REGISTRY)}"
        )
    if headline_n not in n_grid:
        raise ValueError(f"headline_n={headline_n} must be in n_grid={n_grid}")
    if combine in ("base", "impact") and model != "multi-quantile":
        raise ValueError(
            f"combine={combine!r} requires model='multi-quantile' "
            "(logistic doesn't expose the per-race predictive distribution)"
        )

    train_df = pd.concat(
        [load_processed(c, snapshot, processed_dir) for c in train_cycles],
        ignore_index=True,
    )
    test_df = load_processed(test_cycle, snapshot, processed_dir)

    fitted = _build_model(model, feature_set).fit(train_df)
    scored = fitted.score(test_df)
    naive = (feature_set == "naive")
    dems = _filter_universe(scored, universe, naive=naive)
    dems[ORACLE_COL] = (dems["margin_pct"].abs() < 0.05).astype(float)
    n_dem = len(dems)

    pivotal_share: float | None = None
    pivotal_ci: tuple[float, float] | None = None
    floor_saturation: float | None = None
    floor_saturation_ci: tuple[float, float] | None = None
    if combine in ("base", "impact"):
        dems = _apply_stakes_combine(
            dems=dems, fitted_model=fitted, snapshot=snapshot,
            test_cycle=test_cycle, raw_dir=raw_dir,
        )
        pivotal_share, pivotal_ci = _pivotal_metric_with_ci(
            dems, headline_n, bootstrap_reps, bootstrap_seed
        )
    if combine == "impact":
        dems = _apply_need_combine(dems=dems, train_df=train_df)
        floor_saturation, floor_saturation_ci = _floor_saturation_with_ci(
            dems, headline_n, bootstrap_reps, bootstrap_seed
        )

    metrics = tuple(_scores_at_n(dems, n) for n in n_grid)

    model_ci = _bootstrap_ci(dems, SCORE_COL, headline_n, bootstrap_reps, bootstrap_seed)
    fund_ci = _bootstrap_ci(dems, FUND_COL, headline_n, bootstrap_reps, bootstrap_seed)

    return BacktestRow(
        feature_set=feature_set,
        model=model,
        combine=combine,
        snapshot=snapshot,
        test_cycle=test_cycle,
        train_cycles=tuple(train_cycles),
        universe=universe,
        n_dem_candidates=n_dem,
        metrics=metrics,
        headline_n=headline_n,
        headline_model_ci=model_ci,
        headline_fund_ci=fund_ci,
        pivotal_dollar_share=pivotal_share,
        pivotal_ci=pivotal_ci,
        floor_saturation_efficiency=floor_saturation,
        floor_saturation_ci=floor_saturation_ci,
        bootstrap_reps=bootstrap_reps,
        notes=notes,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _apply_stakes_combine(
    *,
    dems: pd.DataFrame,
    fitted_model,
    snapshot: str,
    test_cycle: int,
    raw_dir: Path,
) -> pd.DataFrame:
    """Run the stakes MC and replace SCORE_COL with sqrt(comp * stakes)."""
    from oath_score.scores.chamber import build_chamber
    from oath_score.scores.stakes import (
        PIVOTAL_THRESHOLD, StakesSimulator, sigma_for_snapshot,
    )
    import numpy as np

    chamber = build_chamber(test_cycle, raw_dir)
    uncontested_d = chamber.deterministic_d_count()

    quantile_levels = np.array(fitted_model.quantiles)
    quantile_preds = fitted_model.predict_quantiles(dems)  # (n_dems, n_q)

    # Dynamic-median threshold (chamber_threshold=None): asks "is this seat
    # pivotal to whether D performs above- or below-median across MC draws?"
    # The literal 218-seat threshold is often unreachable in our truncated
    # contested universe, which would zero out all stakes. The median-relative
    # framing is the donor-relevant question anyway.
    sim = StakesSimulator(sigma=sigma_for_snapshot(snapshot), chamber_threshold=None)
    result = sim.simulate(
        contested_quantiles=quantile_preds,
        quantile_levels=quantile_levels,
        uncontested_d_count=uncontested_d,
    )

    dems = dems.copy()
    competitiveness = dems[SCORE_COL].copy()
    stakes_norm = pd.Series(result.stakes_normalized, index=dems.index)
    dems["__stakes_raw__"] = result.stakes_raw
    dems["__stakes_norm__"] = stakes_norm
    dems[PIVOTAL_COL] = (np.abs(result.stakes_raw) > PIVOTAL_THRESHOLD).astype(float)
    # Combined score replaces SCORE_COL so the existing allocation/metric path
    # picks up the impact base score automatically.
    dems[SCORE_COL] = np.sqrt(
        competitiveness.clip(lower=0.0, upper=1.0).fillna(0.0)
        * stakes_norm.clip(lower=0.0, upper=1.0).fillna(0.0)
    )
    return dems


def _apply_need_combine(*, dems: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Layer the financial-need adjustment on top of the existing base score.

    Expects `dems` to already contain SCORE_COL set to `base = sqrt(comp * stakes)`
    (i.e., `_apply_stakes_combine` has already run). Adds the need_raw column
    and overwrites SCORE_COL with `clip(base * (1 + alpha * need_raw), 0, 1)`.

    Trains FinancialNeed on `train_df` (close-race D winners only).
    """
    from oath_score.scores.financial_need import FinancialNeed, MIN_FLOOR

    need_model = FinancialNeed().fit(train_df)
    floor = need_model.predict_floor(dems)
    need_raw = need_model.predict_need(dems)

    own_spend = pd.to_numeric(dems["total_trans"], errors="coerce").fillna(0.0).clip(lower=0.0)

    dems = dems.copy()
    dems["__viable_floor__"] = floor.values
    dems["__need_raw__"] = need_raw.values
    dems[UNDER_FLOOR_COL] = (own_spend.values < floor.values).astype(float)

    base = dems[SCORE_COL].clip(lower=0.0, upper=1.0).fillna(0.0)
    dems[SCORE_COL] = (base * (1.0 + NEED_ALPHA * need_raw.values)).clip(lower=0.0, upper=1.0)
    return dems


def _floor_saturation_metric(allocations: pd.DataFrame) -> float:
    """Fraction of dollars sent to candidates spending below their viable floor."""
    if UNDER_FLOOR_COL not in allocations.columns or allocations.empty:
        return 0.0
    total = float(allocations["allocation"].sum())
    if total <= 0:
        return 0.0
    under_dollars = float((allocations["allocation"] * allocations[UNDER_FLOOR_COL]).sum())
    return under_dollars / total


def _floor_saturation_with_ci(
    dems: pd.DataFrame, n: int, reps: int, seed: int
) -> tuple[float, tuple[float, float]]:
    """Bootstrap CI of floor_saturation_efficiency at top-N model allocation."""
    rng = np.random.default_rng(seed)
    samples = np.empty(reps, dtype=float)
    for i in range(reps):
        boot = dems.sample(n=len(dems), replace=True, random_state=rng.integers(1, 10**9))
        alloc = allocate(boot, score_col=SCORE_COL, n=n)
        samples[i] = _floor_saturation_metric(alloc)
    point = float(_floor_saturation_metric(allocate(dems, score_col=SCORE_COL, n=n)))
    ci = (
        round(float(np.percentile(samples, 2.5)), 4),
        round(float(np.percentile(samples, 97.5)), 4),
    )
    return round(point, 4), ci


def _pivotal_metric(allocations: pd.DataFrame) -> float:
    """Fraction of dollars sent to seats flagged pivotal."""
    if PIVOTAL_COL not in allocations.columns or allocations.empty:
        return 0.0
    total = float(allocations["allocation"].sum())
    if total <= 0:
        return 0.0
    pivotal_dollars = float((allocations["allocation"] * allocations[PIVOTAL_COL]).sum())
    return pivotal_dollars / total


def _pivotal_metric_with_ci(
    dems: pd.DataFrame, n: int, reps: int, seed: int
) -> tuple[float, tuple[float, float]]:
    """Bootstrap-CI of pivotal_dollar_share at top-N model allocation."""
    rng = np.random.default_rng(seed)
    samples = np.empty(reps, dtype=float)
    for i in range(reps):
        boot = dems.sample(n=len(dems), replace=True, random_state=rng.integers(1, 10**9))
        alloc = allocate(boot, score_col=SCORE_COL, n=n)
        samples[i] = _pivotal_metric(alloc)
    point = float(_pivotal_metric(allocate(dems, score_col=SCORE_COL, n=n)))
    ci = (
        round(float(np.percentile(samples, 2.5)), 4),
        round(float(np.percentile(samples, 97.5)), 4),
    )
    return round(point, 4), ci


def append_jsonl(row: BacktestRow, out_path: Path) -> None:
    """Append one row to the JSONL log. Creates parent dir if needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row.as_dict()) + "\n")


def format_row(row: BacktestRow) -> str:
    """Pretty-print a row including N sweep + headline CI."""
    lines = [
        f"[backtest] feature_set={row.feature_set} model={row.model} "
        f"combine={row.combine} snapshot={row.snapshot} "
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
    if row.pivotal_dollar_share is not None and row.pivotal_ci is not None:
        lines.append(
            f"  pivotal_dollar_share: {row.pivotal_dollar_share:.4f} "
            f"(95% CI [{row.pivotal_ci[0]:.4f}, {row.pivotal_ci[1]:.4f}])"
        )
    if row.floor_saturation_efficiency is not None and row.floor_saturation_ci is not None:
        lines.append(
            f"  floor_saturation_efficiency: {row.floor_saturation_efficiency:.4f} "
            f"(95% CI [{row.floor_saturation_ci[0]:.4f}, {row.floor_saturation_ci[1]:.4f}])"
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
    parser.add_argument("--model", choices=("logistic", "multi-quantile"),
                        default="logistic",
                        help="competitiveness model class; default 'logistic'")
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
    parser.add_argument("--combine", choices=("competitiveness", "base", "impact"),
                        default="competitiveness",
                        help="score combination; 'base' adds chamber-control "
                             "stakes; 'impact' additionally adds the "
                             "financial-need adjustment (requires --model multi-quantile)")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"),
                        help="raw data dir for chamber view (used by --combine base)")
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
            model=args.model,
            combine=args.combine,
            snapshot=snap,
            train_cycles=train_cycles,
            test_cycle=args.test,
            processed_dir=args.processed_dir,
            raw_dir=args.raw_dir,
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
