"""Bake the precomputed parquet that the deployed Streamlit app reads.

Replicates the Phase 7 final calibrated config (full feature set,
multi-quantile, combine=impact, alpha=0.3, train=[2016, 2022], universe=all)
for all three snapshots and writes one combined parquet keyed by
(snapshot, state_abbr, district).

The deployed app loads this parquet on cold start and never re-runs the MC,
the multi-quantile fit, or the FinancialNeed regression. All heavy compute
happens here, offline.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/bake_app_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oath_score.constants import GENERAL_ELECTION_DATES, SNAPSHOT_OFFSETS_DAYS
from oath_score.scores.chamber import build_chamber
from oath_score.scores.competitiveness import SCORE_COL
from oath_score.scores.deciling import DecileThresholds
from oath_score.scores.financial_need import FinancialNeed
from oath_score.scores.multi_quantile import MultiQuantileCompetitiveness
from oath_score.scores.stakes import StakesSimulator, sigma_for_snapshot

# Final calibrated config (per Phase 7).
FEATURE_SET = "full"
TRAIN_CYCLES = (2016, 2022)
TEST_CYCLE = 2024
NEED_ALPHA = 0.3

PROC_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
OUT_PATH = PROC_DIR / "app_candidates_2024.parquet"


def bake_one_snapshot(
    snapshot: str,
    *,
    deciles: DecileThresholds,
) -> pd.DataFrame:
    """Run the full Phase 7 pipeline for one snapshot, return scored test rows."""
    print(f"  baking {snapshot}...", flush=True)

    train_df = pd.concat(
        [pd.read_parquet(PROC_DIR / f"candidates_{c}_{snapshot}.parquet") for c in TRAIN_CYCLES],
        ignore_index=True,
    )
    test_df = pd.read_parquet(PROC_DIR / f"candidates_{TEST_CYCLE}_{snapshot}.parquet")

    # Competitiveness
    comp_model = MultiQuantileCompetitiveness(feature_set_name=FEATURE_SET).fit(train_df)
    scored = comp_model.score(test_df)
    dems = scored.loc[scored["party_major"] == "D"].copy()

    # Median predicted margin (q=0.5) — used for the detail panel
    quantile_levels = np.array(comp_model.quantiles)
    quantile_preds = comp_model.predict_quantiles(dems)
    q50_idx = int(np.where(quantile_levels == 0.5)[0][0])
    dems["predicted_margin_median"] = quantile_preds[:, q50_idx]

    # Stakes (correlated MC)
    chamber = build_chamber(TEST_CYCLE, RAW_DIR)
    sim = StakesSimulator(sigma=sigma_for_snapshot(snapshot), chamber_threshold=None)
    stakes_result = sim.simulate(
        contested_quantiles=quantile_preds,
        quantile_levels=quantile_levels,
        uncontested_d_count=chamber.deterministic_d_count(),
    )
    stakes_norm = pd.Series(stakes_result.stakes_normalized, index=dems.index)
    dems["stakes_normalized"] = stakes_norm.values
    dems["stakes_raw"] = stakes_result.stakes_raw

    # Need
    need_model = FinancialNeed().fit(train_df)
    dems["viable_floor"] = need_model.predict_floor(dems).values
    need_raw = need_model.predict_need(dems)
    dems["need_raw"] = need_raw.values

    # Combine: base = sqrt(comp * stakes); impact = clip(base * (1 + alpha * need), 0, 1)
    competitiveness = dems[SCORE_COL].clip(lower=0.0, upper=1.0).fillna(0.0)
    base = np.sqrt(competitiveness * stakes_norm.clip(lower=0.0, upper=1.0).fillna(0.0))
    impact = (base * (1.0 + NEED_ALPHA * need_raw.values)).clip(0.0, 1.0)
    dems["competitiveness"] = competitiveness.values
    dems["impact_continuous"] = impact.values
    dems["impact_decile"] = deciles.apply(pd.Series(impact, index=dems.index)).values

    # Trim to display columns
    dems["snapshot"] = snapshot
    keep = [
        "snapshot", "state_abbr", "district",
        "candidate_name", "last_name", "party_major", "incumbent",
        "impact_continuous", "impact_decile",
        "competitiveness", "stakes_normalized", "stakes_raw", "need_raw",
        "predicted_margin_median",
        "viable_floor", "total_trans",
        "opp_raised", "ie_for_total", "ie_against_total",
        "cook_rating", "cpvi",
        "margin_pct", "winner",
    ]
    available = [c for c in keep if c in dems.columns]
    return dems[available].copy()


def main() -> None:
    deciles = DecileThresholds.load(PROC_DIR / "decile_thresholds.json")
    print(f"Loaded decile cutpoints from {PROC_DIR / 'decile_thresholds.json'}")

    pieces: list[pd.DataFrame] = []
    for snapshot in SNAPSHOT_OFFSETS_DAYS:
        pieces.append(bake_one_snapshot(snapshot, deciles=deciles))

    out = pd.concat(pieces, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print()
    print(f"Wrote {len(out)} rows ({len(pieces)} snapshots × ~{len(pieces[0])} candidates)")
    print(f"  → {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024:.1f} KB)")
    print()
    print("Decile distribution per snapshot (T-60):")
    snap_t60 = out[out["snapshot"] == "T-60"]
    print(snap_t60["impact_decile"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
