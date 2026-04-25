"""FEC bulk-file ingestion with snapshot-date discipline.

Reads four FEC bulk files per cycle:
  * indiv (itcont):  individual itemized contributions  → date-filterable
  * pas   (itpas2):  committee-to-candidate transactions → date-filterable
  * ccl   (ccl):     candidate ↔ committee linkages    → master, no date
  * cn    (cn):      candidate master                  → master, no date

The candidate ↔ committee ↔ contribution join logic is preserved from the
original fec.py. The new contribution: every contribution row is filtered by
TRANSACTION_DT <= snapshot_date BEFORE aggregation, so the per-candidate
totals reflect only money raised by the snapshot date.

Bulk files must be staged at:  data/raw/{cycle}/fec/{indiv,pas,ccl,cn}.txt
Headers must be staged at:     data/raw/{cycle}/fec/{indiv,pas,ccl,cn}_headers.csv

Download from https://www.fec.gov/data/browse-data/?tab=bulk-data
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# FEC bulk files use '|' as delimiter and ship without headers; we apply them
# from a sibling .csv file.
FEC_DELIM = "|"

# FEC date format inside bulk files: MMDDYYYY (8-char string)
FEC_DATE_FMT = "%m%d%Y"


@dataclass(frozen=True)
class FecPaths:
    """Resolved paths to the four FEC bulk files for one cycle."""

    indiv_data: Path
    indiv_headers: Path
    pas_data: Path
    pas_headers: Path
    ccl_data: Path
    ccl_headers: Path
    cn_data: Path
    cn_headers: Path

    @classmethod
    def for_cycle(cls, cycle: int, raw_dir: Path) -> "FecPaths":
        base = Path(raw_dir) / str(cycle) / "fec"
        return cls(
            indiv_data=base / "indiv.txt",
            indiv_headers=base / "indiv_headers.csv",
            pas_data=base / "pas.txt",
            pas_headers=base / "pas_headers.csv",
            ccl_data=base / "ccl.txt",
            ccl_headers=base / "ccl_headers.csv",
            cn_data=base / "cn.txt",
            cn_headers=base / "cn_headers.csv",
        )


def _read_headers(path: Path) -> list[str]:
    """Read a comma-separated single-row header file; return column names."""
    return list(pd.read_csv(path).columns)


def _read_pipe_table(data_path: Path, header_path: Path) -> pd.DataFrame:
    headers = _read_headers(header_path)
    # on_bad_lines='skip' replaces deprecated error_bad_lines=False
    return pd.read_csv(
        data_path,
        sep=FEC_DELIM,
        names=headers,
        on_bad_lines="skip",
        low_memory=False,
        dtype=str,  # parse everything as string; cast targeted cols afterward
    )


def _filter_by_snapshot(
    df: pd.DataFrame, snapshot: date, date_col: str = "TRANSACTION_DT"
) -> pd.DataFrame:
    """Keep only rows whose date_col is <= snapshot. Drops rows with bad dates.

    FEC stores dates as MMDDYYYY strings; bad/missing values exist.
    """
    parsed = pd.to_datetime(df[date_col], format=FEC_DATE_FMT, errors="coerce")
    mask = parsed.notna() & (parsed.dt.date <= snapshot)
    return df.loc[mask].copy()


def _sum_by_committee(df: pd.DataFrame, out_col: str) -> pd.Series:
    """Sum TRANSACTION_AMT per CMTE_ID, returning a Series."""
    amt = pd.to_numeric(df["TRANSACTION_AMT"], errors="coerce").fillna(0)
    return amt.groupby(df["CMTE_ID"]).sum().rename(out_col)


def aggregate_to_candidate(
    indiv: pd.DataFrame,
    pas: pd.DataFrame,
    ccl: pd.DataFrame,
    cn: pd.DataFrame,
) -> pd.DataFrame:
    """Join contributions through CCL → CN to produce per-candidate totals.

    Returns one row per CAND_ID with columns:
      cand_id, cand_name, cand_pty_affiliation, state, district,
      trans_by_indiv, trans_by_cmte, total_trans, last_name
    """
    indiv_by_cmte = _sum_by_committee(indiv, "trans_by_indiv")
    pas_by_cmte = _sum_by_committee(pas, "trans_by_cmte")

    by_cmte = pd.concat([indiv_by_cmte, pas_by_cmte], axis=1).fillna(0)
    by_cmte["total_trans"] = by_cmte["trans_by_indiv"] + by_cmte["trans_by_cmte"]
    by_cmte["CMTE_ID"] = by_cmte.index

    # Join contribution totals to candidates via the candidate-committee linkage.
    by_cand = (
        by_cmte.merge(ccl[["CMTE_ID", "CAND_ID"]], on="CMTE_ID", how="left")
        .dropna(subset=["CAND_ID"])
        .groupby("CAND_ID")[["total_trans", "trans_by_indiv", "trans_by_cmte"]]
        .sum()
        .astype(np.int64)
        .reset_index()
    )

    # Attach candidate metadata.
    cn_cols = [
        "CAND_ID", "CAND_NAME", "CAND_PTY_AFFILIATION", "CAND_OFFICE",
        "CAND_OFFICE_ST", "CAND_OFFICE_DISTRICT", "CAND_ELECTION_YR",
    ]
    fec = by_cand.merge(cn[cn_cols], on="CAND_ID", how="left")
    fec = fec.rename(columns={
        "CAND_ID": "cand_id",
        "CAND_NAME": "cand_name",
        "CAND_PTY_AFFILIATION": "party",
        "CAND_OFFICE": "office",
        "CAND_OFFICE_ST": "state",
        "CAND_OFFICE_DISTRICT": "district",
        "CAND_ELECTION_YR": "election_year",
    })
    fec["last_name"] = (
        fec["cand_name"].fillna("").str.split(",").str[0].str.strip().str.upper()
    )
    return fec


def fetch_fec(cycle: int, snapshot: date, raw_dir: Path) -> pd.DataFrame:
    """Top-level entry point: load + snapshot-filter + aggregate one cycle.

    Returns a DataFrame with one row per House candidate active in `cycle`,
    with `total_trans`, `trans_by_indiv`, `trans_by_cmte` reflecting only
    contributions dated <= `snapshot`.
    """
    paths = FecPaths.for_cycle(cycle, raw_dir)

    indiv_raw = _read_pipe_table(paths.indiv_data, paths.indiv_headers)
    pas_raw = _read_pipe_table(paths.pas_data, paths.pas_headers)
    ccl = _read_pipe_table(paths.ccl_data, paths.ccl_headers)
    cn = _read_pipe_table(paths.cn_data, paths.cn_headers)

    indiv = _filter_by_snapshot(indiv_raw, snapshot)
    pas = _filter_by_snapshot(pas_raw, snapshot)

    fec = aggregate_to_candidate(indiv, pas, ccl, cn)
    fec["cycle"] = cycle
    fec["snapshot_date"] = snapshot.isoformat()

    # House candidates only.
    return fec.loc[fec["office"] == "H"].copy()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate FEC contributions per candidate at snapshot.")
    parser.add_argument("--cycle", type=int, required=True, choices=[2014, 2016, 2022, 2024])
    parser.add_argument("--snapshot", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    snapshot = datetime.strptime(args.snapshot, "%Y-%m-%d").date()
    df = fetch_fec(args.cycle, snapshot, args.raw_dir)

    out = args.out_dir / f"fec_{args.cycle}_{args.snapshot}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} candidate rows to {out}")


if __name__ == "__main__":
    main()
