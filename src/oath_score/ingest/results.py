"""District-level election results from MIT Election Lab.

Replaces the broken Politico scraper. Source dataset:
  "U.S. House 1976-202x" — Harvard Dataverse
  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2

The dataset is a single CSV with one row per (year, state, district, candidate)
covering general and primary races. We filter to general-election House races,
compute the signed two-party margin per district, and mark winners.

The exact file URL on Dataverse changes when the dataset is updated; the
download function tries the canonical access endpoint and falls back to a
manual-download instruction.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from oath_score.constants import STATE_ABBR

# Canonical filename used in this project tree.
LOCAL_FILENAME = "1976_2024_house.csv"

# Dataverse file access endpoint. The "persistentId" form is stable; the
# fileId of the latest CSV may change. If this fails, the user should
# download manually and place the CSV at data/raw/elections/{LOCAL_FILENAME}.
_DATAVERSE_URL = (
    "https://dataverse.harvard.edu/api/access/datafile"
    "/:persistentId?persistentId=doi:10.7910/DVN/IG0UN2"
)

# Two-party label set used for signed-margin computation.
DEM_LABELS = frozenset({"DEMOCRAT", "DEMOCRATIC", "DEMOCRATIC-FARMER-LABOR", "DFL"})
REP_LABELS = frozenset({"REPUBLICAN"})

# Reverse-lookup: state name (uppercase) → abbreviation. MIT data uses uppercase names.
_NAME_TO_ABBR = {name.upper(): abbr for name, abbr in STATE_ABBR.items()}


def fetch_results(cycle: int, raw_dir: Path) -> pd.DataFrame:
    """Return one row per (state, district, candidate) for the cycle's general election.

    Output columns:
      cycle, state_abbr, district, candidate_name, last_name, party,
      candidate_votes, total_votes_in_race, vote_share,
      margin_pct_signed,  # D% - R% over the two-party total, per district
      winner              # bool
    """
    raw = _load_master_csv(raw_dir)
    df = _filter_general_house(raw, cycle)
    df = _normalize(df)
    df = _compute_two_party_margin(df)
    df["cycle"] = cycle
    return df


# ---------- file load ----------

def _load_master_csv(raw_dir: Path) -> pd.DataFrame:
    """Load the staged MIT Election Lab CSV. If missing, attempt download."""
    raw_dir = Path(raw_dir) / "elections"
    raw_dir.mkdir(parents=True, exist_ok=True)
    local = raw_dir / LOCAL_FILENAME
    if not local.exists():
        _try_download(local)
    return pd.read_csv(local, encoding="utf-8", low_memory=False)


def _try_download(dest: Path) -> None:
    """Attempt to fetch the Dataverse CSV; raise with manual fallback instructions on failure."""
    from oath_score.ingest._download import download_file
    try:
        download_file(_DATAVERSE_URL, dest)
    except Exception as exc:
        raise RuntimeError(
            f"Could not auto-download the MIT Election Lab CSV.\n"
            f"  Tried: {_DATAVERSE_URL}\n"
            f"  Original error: {exc}\n\n"
            f"Manual fallback: visit\n"
            f"  https://dataverse.harvard.edu/dataset.xhtml"
            f"?persistentId=doi:10.7910/DVN/IG0UN2\n"
            f"download the latest 'U.S. House' CSV, and place it at\n"
            f"  {dest}"
        ) from exc


# ---------- filtering & normalization ----------

def _filter_general_house(raw: pd.DataFrame, cycle: int) -> pd.DataFrame:
    """Keep only general-election U.S. House rows for the requested cycle."""
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    out = df.loc[
        (df["year"] == cycle)
        & (df["office"].astype(str).str.upper() == "US HOUSE")
        & (df["stage"].astype(str).str.upper() == "GEN")
    ].copy()
    if "special" in out.columns:
        out = out.loc[~out["special"].astype(bool)].copy()
    if out.empty:
        raise RuntimeError(f"No general-election House rows found for cycle {cycle}.")
    return out


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize state, district, party, candidate-name columns."""
    df = df.copy()
    df["state_abbr"] = df["state_po"].astype(str).str.upper().fillna(
        df["state"].astype(str).str.upper().map(_NAME_TO_ABBR)
    )
    # MIT encodes at-large districts as 0 (or sometimes 1 for single-district states); accept both.
    df["district"] = pd.to_numeric(df["district"], errors="coerce").fillna(0).astype(int)
    df["party"] = df["party"].astype(str).str.upper().str.strip()
    df["candidate_name"] = df["candidate"].astype(str).str.strip()
    df["last_name"] = df["candidate_name"].apply(_extract_last_name)
    df["candidate_votes"] = pd.to_numeric(df["candidatevotes"], errors="coerce").fillna(0).astype(int)
    df["total_votes_in_race"] = pd.to_numeric(df["totalvotes"], errors="coerce").fillna(0).astype(int)
    return df[[
        "state_abbr", "district", "candidate_name", "last_name", "party",
        "candidate_votes", "total_votes_in_race",
    ]]


_PUNCT_RE = re.compile(r"[^A-Z\s\-']")


def _extract_last_name(s: str) -> str:
    """MIT format is 'LAST, FIRST' or 'LAST, FIRST MIDDLE'. Take the comma-prefix."""
    if not s:
        return ""
    upper = s.upper().strip()
    head = upper.split(",", 1)[0]
    return _PUNCT_RE.sub("", head).strip()


# ---------- margin computation ----------

def _compute_two_party_margin(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-district signed margin (D% - R%) and a winner flag.

    Where multiple D or R candidates exist in a single district (rare), keep
    only the highest-vote candidate per party for the margin numerator.
    Third-party candidates remain in the output but get the same district-level
    margin assigned.
    """
    parts = []
    for (state, district), grp in df.groupby(["state_abbr", "district"], sort=False):
        dem_top = grp.loc[grp["party"].isin(DEM_LABELS)].nlargest(1, "candidate_votes")
        rep_top = grp.loc[grp["party"].isin(REP_LABELS)].nlargest(1, "candidate_votes")

        dem_v = int(dem_top["candidate_votes"].sum())
        rep_v = int(rep_top["candidate_votes"].sum())
        two_party_total = dem_v + rep_v

        if two_party_total == 0:
            margin = np.nan
        else:
            margin = (dem_v - rep_v) / two_party_total

        # winner = highest-vote candidate in district overall
        max_v = grp["candidate_votes"].max()
        out = grp.copy()
        out["margin_pct_signed"] = margin
        out["vote_share"] = (
            grp["candidate_votes"] / grp["total_votes_in_race"].replace(0, np.nan)
        )
        out["winner"] = grp["candidate_votes"] == max_v
        parts.append(out)

    return pd.concat(parts, ignore_index=True)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Load MIT Election Lab House results for one cycle.")
    parser.add_argument("--cycle", type=int, required=True, choices=[2014, 2016, 2022, 2024])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    df = fetch_results(args.cycle, args.raw_dir)
    print(f"Loaded {len(df)} candidate-rows for cycle {args.cycle}")
    print(df.groupby("state_abbr")["district"].nunique().sum(), "unique (state, district) pairs")
    print(df.head(10))


if __name__ == "__main__":
    main()
