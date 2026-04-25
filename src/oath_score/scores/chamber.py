"""Per-cycle 435-seat House view for the chamber-control Monte Carlo.

The contested-race filter in features.py drops uncontested districts. The
stakes MC needs ALL 435 seats to compute chamber control: contested seats
are drawn each iteration; locked seats (one major-party candidate ran)
contribute deterministic D or R wins to the baseline.

Build the chamber view directly from MIT Election Lab results — same source
the contested filter uses, just with a different filter (drop only special
elections and write-ins, never drop on contested-ness).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from oath_score.features import DEM_PARTY_LABELS, REP_PARTY_LABELS
from oath_score.ingest import results

ChamberStatus = Literal["contested", "d_lock", "r_lock", "other_lock"]


@dataclass(frozen=True)
class ChamberView:
    """One row per (state, district) for one cycle.

    Columns of `df`:
      * state_abbr, district
      * status: 'contested' | 'd_lock' | 'r_lock' | 'other_lock'
      * winner_party: 'D' | 'R' | 'OTHER' | None  — actual winner from results
      * d_two_party_share: float ∈ [0, 1] — for contested races (NaN otherwise)
    """
    df: pd.DataFrame
    cycle: int

    @property
    def n_seats(self) -> int:
        return len(self.df)

    @property
    def n_d_locks(self) -> int:
        return int((self.df["status"] == "d_lock").sum())

    @property
    def n_r_locks(self) -> int:
        return int((self.df["status"] == "r_lock").sum())

    @property
    def n_contested(self) -> int:
        return int((self.df["status"] == "contested").sum())

    def deterministic_d_count(self) -> int:
        """How many seats already locked for Dems before any MC draw.

        Used as the baseline that contested-seat MC outcomes add to.
        """
        return self.n_d_locks


def build_chamber(cycle: int, raw_dir: Path) -> ChamberView:
    """Build the 435-seat chamber view for one cycle.

    Logic:
      1. Pull all general-election House results from MIT.
      2. For each (state, district), classify by which major parties had a
         non-trivial candidate (>1% of total or simply present, depending
         on data quality).
      3. Status:
           * Both D and R present → 'contested'
           * Only D present → 'd_lock'
           * Only R present → 'r_lock'
           * Neither (independents/third-party only, e.g., VT-AL Sanders) → 'other_lock'
      4. Winner determined by max votes per district.
    """
    raw_df = results.fetch_results(cycle, raw_dir)

    # Identify each candidate's major-party class
    raw_df = raw_df.copy()
    raw_df["party_major"] = raw_df["party"].astype(str).str.upper().map(
        lambda p: "D" if p in DEM_PARTY_LABELS
        else "R" if p in REP_PARTY_LABELS
        else "OTHER"
    )

    # Per-district aggregation
    rows: list[dict] = []
    for (state, district), grp in raw_df.groupby(["state_abbr", "district"], sort=True):
        d_present = (grp["party_major"] == "D").any()
        r_present = (grp["party_major"] == "R").any()

        if d_present and r_present:
            status: ChamberStatus = "contested"
        elif d_present:
            status = "d_lock"
        elif r_present:
            status = "r_lock"
        else:
            status = "other_lock"

        # Winner: candidate with most votes in this district
        top = grp.nlargest(1, "candidate_votes").iloc[0]
        winner_party = top["party_major"] if top["party_major"] != "OTHER" else "OTHER"

        # D two-party share (only meaningful for contested)
        d_two = grp.loc[grp["party_major"] == "D", "candidate_votes"].sum()
        r_two = grp.loc[grp["party_major"] == "R", "candidate_votes"].sum()
        denom = d_two + r_two
        d_share = float(d_two / denom) if denom > 0 else float("nan")

        rows.append({
            "state_abbr": state,
            "district": int(district),
            "status": status,
            "winner_party": winner_party,
            "d_two_party_share": d_share,
        })

    df = pd.DataFrame(rows).sort_values(["state_abbr", "district"]).reset_index(drop=True)
    return ChamberView(df=df, cycle=cycle)
