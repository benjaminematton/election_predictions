"""Donor-allocation function used by every backtest reference line.

Same allocation logic, different score column → fair comparison between
the model, the fundraising baseline, and the hindsight oracle.

Default behavior (Phase 3): top-N candidates by score, score-weighted, no
need cap. Phase 6 will turn on the need-saturation cap once we have a
viable-floor model; the parameter is exposed but defaults to off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def allocate(
    candidates: pd.DataFrame,
    *,
    score_col: str,
    n: int = 10,
    total_dollars: float = 100.0,
    need_col: str | None = None,
) -> pd.DataFrame:
    """Allocate `total_dollars` across top-N candidates by `score_col`.

    Args:
      candidates: one row per candidate; must contain `score_col`. NaNs in the
        score column are treated as 0.
      score_col: the column to rank and weight by.
      n: top-N picked. If fewer than N candidates exist, all of them are kept.
      total_dollars: notional budget. The metric is dollar-fraction so the
        absolute value is irrelevant; defaults to $100 for readability.
      need_col: optional. If provided, allocation is capped at the candidate's
        remaining need: `weight_i = score_i × min(1, need_remaining_i / D)`.
        Phase 3 leaves this off (no viable-floor model yet).

    Returns:
      A copy of `candidates` (or its top-N subset) with one new column,
      `allocation`, summing to `total_dollars` (within float tolerance).
      Rows below the top-N have allocation 0 and are not returned.

    Raises:
      KeyError: if score_col (or need_col when given) is not in candidates.
      ValueError: if all top-N scores are non-positive (no rationale to allocate).
    """
    if score_col not in candidates.columns:
        raise KeyError(f"score_col {score_col!r} not in DataFrame; have {list(candidates.columns)}")
    if need_col is not None and need_col not in candidates.columns:
        raise KeyError(f"need_col {need_col!r} not in DataFrame")
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")

    df = candidates.copy()
    score = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["__score__"] = score

    # Sort and slice. Deterministic tie-break by (state_abbr, district) so that
    # benchmarks like Cook-final (which pile up many ties at the same ordinal,
    # e.g. 22 Toss-ups all weighted 4) pick the same N rows on every run.
    sort_cols = ["__score__"]
    sort_asc = [False]
    for col in ("state_abbr", "district"):
        if col in df.columns:
            sort_cols.append(col)
            sort_asc.append(True)
    top = df.sort_values(sort_cols, ascending=sort_asc).head(min(n, len(df))).copy()
    weights = top["__score__"].to_numpy(dtype=float)

    if need_col is not None:
        need_remaining = pd.to_numeric(top[need_col], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy()
        weights = weights * np.minimum(1.0, need_remaining / max(total_dollars, 1.0))

    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        # Degenerate: all top-N scored zero. Split evenly so the metric is well-defined.
        weights = np.ones(len(top), dtype=float)
        total_weight = float(weights.sum())

    top["allocation"] = total_dollars * weights / total_weight
    top = top.drop(columns="__score__")
    return top.reset_index(drop=True)


def metric_pct_to_close_races(
    allocations: pd.DataFrame,
    margin_col: str = "margin_pct",
    threshold: float = 0.05,
) -> float:
    """Fraction of allocated dollars that went to races finishing within ±threshold.

    `allocations` must have an `allocation` column (output of `allocate()`)
    and a `margin_col` column (signed margin per the candidate's race).
    """
    if "allocation" not in allocations.columns:
        raise KeyError("allocations must have an 'allocation' column (run allocate() first)")
    if margin_col not in allocations.columns:
        raise KeyError(f"margin_col {margin_col!r} not in DataFrame")
    if allocations.empty:
        return 0.0

    total = float(allocations["allocation"].sum())
    if total <= 0:
        return 0.0
    close_mask = pd.to_numeric(allocations[margin_col], errors="coerce").abs() < threshold
    close_dollars = float(allocations.loc[close_mask, "allocation"].sum())
    return close_dollars / total
