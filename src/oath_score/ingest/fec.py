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

Bulk files are downloaded by `download_for(cycle, raw_dir)` from
https://www.fec.gov/files/bulk-downloads/{cycle}/, unzipped to:
  data/raw/{cycle}/fec/{indiv,pas,ccl,cn}.txt
Headers are downloaded once into:
  data/raw/{cycle}/fec/{indiv,pas,ccl,cn}_headers.csv

The `indiv` file in particular can exceed 1 GB; we use chunked reads with
in-loop snapshot filtering and per-committee aggregation, so peak memory
stays bounded regardless of cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from oath_score.ingest._download import download_file, unzip

# FEC bulk files use '|' as delimiter and ship without headers; we apply them
# from a sibling .csv file.
FEC_DELIM = "|"

# FEC date format inside bulk files: MMDDYYYY (8-char string)
FEC_DATE_FMT = "%m%d%Y"

# Chunk size for streaming reads of the multi-GB indiv file.
CHUNK_ROWS = 500_000

# Bulk file URL pattern. `cycle` and 2-digit-year `yy` get substituted.
BULK_BASE = "https://www.fec.gov/files/bulk-downloads/{cycle}/{name}{yy}.zip"
HEADER_BASE = "https://www.fec.gov/files/bulk-downloads/data_dictionaries/{name}_header_file.csv"

# (file-key, FEC bulk archive prefix, name inside the zip after extraction)
BULK_SPECS: tuple[tuple[str, str, str], ...] = (
    ("indiv", "indiv", "itcont.txt"),
    ("pas",   "pas2",  "itpas2.txt"),
    ("ccl",   "ccl",   "ccl.txt"),
    ("cn",    "cn",    "cn.txt"),
)

HEADER_NAMES: dict[str, str] = {
    # Map our internal file-key → FEC's "data dictionary" header file basename.
    "indiv": "indiv",
    "pas":   "pas2",
    "ccl":   "ccl",
    "cn":    "cn",
}


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
    """Single-shot read for small master files (ccl, cn). Use _stream_pipe_table for indiv/pas."""
    headers = _read_headers(header_path)
    return pd.read_csv(
        data_path,
        sep=FEC_DELIM,
        names=headers,
        on_bad_lines="skip",
        low_memory=False,
        dtype=str,
    )


def _stream_pipe_table(
    data_path: Path, header_path: Path, chunksize: int = CHUNK_ROWS
) -> Iterable[pd.DataFrame]:
    """Yield DataFrame chunks of `chunksize` rows. For multi-GB FEC files."""
    headers = _read_headers(header_path)
    yield from pd.read_csv(
        data_path,
        sep=FEC_DELIM,
        names=headers,
        on_bad_lines="skip",
        low_memory=False,
        dtype=str,
        chunksize=chunksize,
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


def _sum_by_committee_streaming(
    data_path: Path, header_path: Path, snapshot: date, out_col: str
) -> pd.Series:
    """Stream a multi-GB pipe table, snapshot-filter each chunk, aggregate per committee.

    Returns a Series indexed by CMTE_ID with the summed TRANSACTION_AMT.
    Peak memory stays bounded by chunksize regardless of file size.
    """
    running: dict[str, int] = {}
    for chunk in _stream_pipe_table(data_path, header_path):
        filtered = _filter_by_snapshot(chunk, snapshot)
        if filtered.empty:
            continue
        amt = pd.to_numeric(filtered["TRANSACTION_AMT"], errors="coerce").fillna(0)
        sums = amt.groupby(filtered["CMTE_ID"]).sum()
        for cmte, total in sums.items():
            running[cmte] = running.get(cmte, 0) + int(total)
    return pd.Series(running, name=out_col, dtype=np.int64)


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

    Streams indiv and pas (multi-GB), single-shot reads ccl and cn (small).

    Returns a DataFrame with one row per House candidate active in `cycle`,
    with `total_trans`, `trans_by_indiv`, `trans_by_cmte` reflecting only
    contributions dated <= `snapshot`.
    """
    paths = FecPaths.for_cycle(cycle, raw_dir)

    indiv_by_cmte = _sum_by_committee_streaming(
        paths.indiv_data, paths.indiv_headers, snapshot, "trans_by_indiv"
    )
    pas_by_cmte = _sum_by_committee_streaming(
        paths.pas_data, paths.pas_headers, snapshot, "trans_by_cmte"
    )
    ccl = _read_pipe_table(paths.ccl_data, paths.ccl_headers)
    cn = _read_pipe_table(paths.cn_data, paths.cn_headers)

    fec = _join_committees_to_candidates(indiv_by_cmte, pas_by_cmte, ccl, cn)
    fec["cycle"] = cycle
    fec["snapshot_date"] = snapshot.isoformat()

    return fec.loc[fec["office"] == "H"].copy()


def _join_committees_to_candidates(
    indiv_by_cmte: pd.Series,
    pas_by_cmte: pd.Series,
    ccl: pd.DataFrame,
    cn: pd.DataFrame,
) -> pd.DataFrame:
    """Combine per-committee aggregates into per-candidate totals via CCL → CN."""
    by_cmte = pd.concat([indiv_by_cmte, pas_by_cmte], axis=1).fillna(0).astype(np.int64)
    by_cmte["total_trans"] = by_cmte["trans_by_indiv"] + by_cmte["trans_by_cmte"]
    by_cmte["CMTE_ID"] = by_cmte.index

    by_cand = (
        by_cmte.merge(ccl[["CMTE_ID", "CAND_ID"]], on="CMTE_ID", how="left")
        .dropna(subset=["CAND_ID"])
        .groupby("CAND_ID")[["total_trans", "trans_by_indiv", "trans_by_cmte"]]
        .sum()
        .astype(np.int64)
        .reset_index()
    )

    cn_cols = [
        "CAND_ID", "CAND_NAME", "CAND_PTY_AFFILIATION", "CAND_OFFICE",
        "CAND_OFFICE_ST", "CAND_OFFICE_DISTRICT", "CAND_ELECTION_YR",
        "CAND_ICI",
    ]
    available = [c for c in cn_cols if c in cn.columns]
    fec = by_cand.merge(cn[available], on="CAND_ID", how="left")
    fec = fec.rename(columns={
        "CAND_ID": "cand_id",
        "CAND_NAME": "cand_name",
        "CAND_PTY_AFFILIATION": "party",
        "CAND_OFFICE": "office",
        "CAND_OFFICE_ST": "state",
        "CAND_OFFICE_DISTRICT": "district",
        "CAND_ELECTION_YR": "election_year",
        "CAND_ICI": "cand_ici",  # I=incumbent, C=challenger, O=open
    })
    fec["last_name"] = (
        fec["cand_name"].fillna("").str.split(",").str[0].str.strip().str.upper()
    )
    return fec


# ---------- bulk download orchestration ----------

def download_for(cycle: int, raw_dir: Path) -> FecPaths:
    """Download and unzip all four FEC bulk files for a cycle.

    Idempotent: skips files that already exist on disk.
    Returns FecPaths so callers can use the result immediately.
    """
    yy = f"{cycle % 100:02d}"
    fec_dir = Path(raw_dir) / str(cycle) / "fec"
    fec_dir.mkdir(parents=True, exist_ok=True)

    paths = FecPaths.for_cycle(cycle, raw_dir)
    name_to_data: dict[str, Path] = {
        "indiv": paths.indiv_data,
        "pas":   paths.pas_data,
        "ccl":   paths.ccl_data,
        "cn":    paths.cn_data,
    }
    name_to_header: dict[str, Path] = {
        "indiv": paths.indiv_headers,
        "pas":   paths.pas_headers,
        "ccl":   paths.ccl_headers,
        "cn":    paths.cn_headers,
    }

    for key, archive_prefix, member_in_zip in BULK_SPECS:
        data_path = name_to_data[key]
        if not data_path.exists():
            zip_url = BULK_BASE.format(cycle=cycle, name=archive_prefix, yy=yy)
            zip_path = fec_dir / f"{key}.zip"
            print(f"[fec] downloading {key} cycle={cycle}: {zip_url}")
            download_file(zip_url, zip_path)
            extracted = unzip(zip_path, fec_dir, members=[member_in_zip])
            # FEC zips contain the raw file with its native name; rename to our convention.
            extracted[0].rename(data_path)
            zip_path.unlink()

        header_path = name_to_header[key]
        if not header_path.exists():
            header_url = HEADER_BASE.format(name=HEADER_NAMES[key])
            print(f"[fec] downloading {key} headers: {header_url}")
            download_file(header_url, header_path)

    return paths


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
