"""FEC Schedule E (independent expenditures) ingestion with snapshot discipline.

Schedule E reports cover money spent by entities OTHER than the candidate's
own committee, either supporting or opposing a specific candidate. This is
the canonical, snapshot-filterable IE source — preferred over OpenSecrets.

Key columns from FEC's Schedule E bulk file:
  CMTE_ID         filer (the spender)
  CAND_ID         candidate the spending was for/against
  EXP_DATE        disbursement date (MMDDYYYY) — snapshot filter target
  EXP_AMO         expenditure amount
  SUP_OPP         "S" (support) or "O" (oppose)
  PAY_DATE, etc.  not used here

Each cycle's bulk file is downloadable from
  https://www.fec.gov/files/bulk-downloads/{cycle}/independent_expenditure_{cycle}.csv

Header columns are inline (the file is a true CSV with a header row, unlike
the other FEC bulk files which need an external header dictionary).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from oath_score.ingest._download import download_file

IE_URL = "https://www.fec.gov/files/bulk-downloads/{cycle}/independent_expenditure_{cycle}.csv"

EXP_DATE_FMT = "%m/%d/%Y"  # FEC's Schedule E CSV uses slashes; differs from indiv/pas


@dataclass(frozen=True)
class IePaths:
    data: Path

    @classmethod
    def for_cycle(cls, cycle: int, raw_dir: Path) -> "IePaths":
        return cls(data=Path(raw_dir) / str(cycle) / "fec" / "schedule_e.csv")


def download_for(cycle: int, raw_dir: Path) -> IePaths:
    """Download Schedule E for one cycle. Idempotent."""
    paths = IePaths.for_cycle(cycle, raw_dir)
    if not paths.data.exists():
        url = IE_URL.format(cycle=cycle)
        print(f"[fec_ie] downloading cycle {cycle}: {url}")
        download_file(url, paths.data)
    return paths


def fetch_independent_expenditures(
    cycle: int, snapshot: date, raw_dir: Path
) -> pd.DataFrame:
    """Aggregate Schedule E into per-candidate IE-for and IE-against totals.

    Returns one row per CAND_ID with columns:
      cycle, snapshot_date, cand_id, ie_for_total, ie_against_total
    """
    paths = IePaths.for_cycle(cycle, raw_dir)
    if not paths.data.exists():
        raise FileNotFoundError(
            f"Schedule E file missing at {paths.data}. "
            f"Run download_for({cycle}, raw_dir) first."
        )

    df = pd.read_csv(paths.data, low_memory=False, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # Normalize required columns. FEC has used different casings across cycles.
    rename = {}
    for source, target in (("cand_id", "CAND_ID"), ("exp_date", "EXP_DATE"),
                            ("exp_amo", "EXP_AMO"), ("sup_opp", "SUP_OPP")):
        if source in df.columns:
            rename[source] = target
        elif source.upper() in df.columns:
            rename[source.upper()] = target
    df = df.rename(columns=rename)

    required = {"CAND_ID", "EXP_DATE", "EXP_AMO", "SUP_OPP"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Schedule E for cycle {cycle} is missing columns {missing}. "
            f"Available: {list(df.columns)[:20]}"
        )

    # Snapshot filter — disbursement date must be on or before snapshot.
    parsed = pd.to_datetime(df["EXP_DATE"], format=EXP_DATE_FMT, errors="coerce")
    df = df.loc[parsed.notna() & (parsed.dt.date <= snapshot)].copy()

    df["EXP_AMO"] = pd.to_numeric(df["EXP_AMO"], errors="coerce").fillna(0)
    df["SUP_OPP"] = df["SUP_OPP"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["CAND_ID"])

    grouped = (
        df.groupby(["CAND_ID", "SUP_OPP"])["EXP_AMO"]
        .sum()
        .unstack(fill_value=0.0)
        .rename(columns={"S": "ie_for_total", "O": "ie_against_total"})
        .reset_index()
        .rename(columns={"CAND_ID": "cand_id"})
    )
    for col in ("ie_for_total", "ie_against_total"):
        if col not in grouped.columns:
            grouped[col] = 0.0
        grouped[col] = grouped[col].astype(np.int64)

    grouped["cycle"] = cycle
    grouped["snapshot_date"] = snapshot.isoformat()
    return grouped[["cycle", "snapshot_date", "cand_id", "ie_for_total", "ie_against_total"]]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate FEC Schedule E IEs at a snapshot.")
    parser.add_argument("--cycle", type=int, required=True, choices=[2014, 2016, 2022, 2024])
    parser.add_argument("--snapshot", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--download", action="store_true", help="Pull the bulk file first")
    args = parser.parse_args()

    snap = datetime.strptime(args.snapshot, "%Y-%m-%d").date()
    if args.download:
        download_for(args.cycle, args.raw_dir)
    df = fetch_independent_expenditures(args.cycle, snap, args.raw_dir)
    print(f"Cycle {args.cycle} snapshot {args.snapshot}: {len(df)} candidates with IE activity")
    print(df.head())


if __name__ == "__main__":
    main()
