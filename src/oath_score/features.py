"""Build the per-(cycle, snapshot) candidate feature matrix.

Joins all six ingestion sources, applies the contested-race filter, computes
derived columns, and writes one parquet to data/processed/.

Output schema (matches feature_sets.full plus the regression target and a
few audit columns):

  cycle, snapshot, snapshot_date,
  state_abbr, district, party, last_name, candidate_name,
  cand_id, cand_ici,
  margin_pct, winner,                                       # regression target / outcome
  cook_rating, cpvi, incumbent,                             # naive + pvi
  acs_median_age, acs_race_white, acs_race_black, acs_race_asian,
  acs_edu_bachelors, acs_edu_grad_degree,
  acs_median_income, acs_below_100pct_fpl,                  # demographics
  self_raised_log, self_raised_pct, opp_raised_log,         # snapshot fundraising
  ie_for_log, ie_against_log                                # IE
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from oath_score.constants import snapshot_date_for, SNAPSHOT_OFFSETS_DAYS, CYCLES
from oath_score.ingest import census, fec, fec_ie, pvi, ratings, results

DEM_PARTY_LABELS = frozenset({"DEM", "DEMOCRAT", "DEMOCRATIC", "DFL", "DEMOCRATIC-FARMER-LABOR"})
REP_PARTY_LABELS = frozenset({"REP", "REPUBLICAN"})


def _safe_fetch_pvi(cycle: int, raw_dir: Path) -> pd.DataFrame:
    """Fetch Daily Kos PVI; return empty schema-correct DataFrame if unavailable.

    Wikipedia ratings already supply CPVI for the ~180 competitive races, which
    is the contested-race universe `features.py` filters to. Daily Kos is a
    fallback for the remaining ~250 safe seats — useful for descriptive plots
    but not strictly required for the model.
    """
    try:
        return pvi.fetch_pvi(cycle, raw_dir)
    except Exception as exc:
        print(f"[features] Daily Kos PVI unavailable ({exc.__class__.__name__}); "
              "proceeding with Wikipedia-CPVI only. This is fine for contested races.")
        return pd.DataFrame(columns=["state_abbr", "district", "cpvi_signed", "map_version"])


def build_features(cycle: int, snapshot: str, raw_dir: Path) -> pd.DataFrame:
    """End-to-end feature pipeline for one (cycle, snapshot)."""
    snap_date = snapshot_date_for(cycle, snapshot)

    print(f"[features] cycle={cycle} snapshot={snapshot} -> {snap_date}")
    print("[features] loading sources...")
    res_df = results.fetch_results(cycle, raw_dir)
    rat_res = ratings.fetch_ratings(cycle, snap_date, raw_dir)
    rat_df = rat_res.df
    pvi_df = _safe_fetch_pvi(cycle, raw_dir)
    acs_df = census.fetch_acs(cycle)
    fec_df = fec.fetch_fec(cycle, snap_date, raw_dir)
    ie_df = fec_ie.fetch_independent_expenditures(cycle, snap_date, raw_dir)

    if rat_res.backfilled:
        print(
            f"[features] WARNING: cycle={cycle} ratings page is back-filled "
            "(March 2021 creation); snapshot column reflects final ratings, "
            "not contemporaneous data."
        )

    print(
        f"[features] sizes: results={len(res_df)} ratings={len(rat_df)} "
        f"pvi={len(pvi_df)} acs={len(acs_df)} fec={len(fec_df)} ie={len(ie_df)}"
    )

    df = _normalize_results(res_df)
    df = _attach_ratings(df, rat_df)
    df = _attach_pvi_fallback(df, pvi_df)
    df = _attach_demographics(df, acs_df)
    df = _attach_fec(df, fec_df)
    df = _attach_ie(df, ie_df)

    before = len(df)
    df = _apply_contested_race_filter(df)
    print(f"[features] contested-race filter: {before} -> {len(df)} candidate-rows")

    df = _compute_derived_columns(df)

    df["cycle"] = cycle
    df["snapshot"] = snapshot
    df["snapshot_date"] = snap_date.isoformat()

    _assert_invariants(df)
    return df


# ---------- per-source attach ----------

def _normalize_results(res_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot results so we have one row per (state, district, party-major) pair."""
    df = res_df.copy()
    df["party_major"] = df["party"].map(_party_major)
    # Keep only D / R rows; we'll re-attach third-party-only districts info later
    # as part of the contested-race filter (those will be excluded anyway).
    df = df.loc[df["party_major"].isin(["D", "R"])].copy()
    # Per (state, district, party), keep the highest-vote candidate (in rare
    # cases multiple D or R candidates appear in the general).
    df = (
        df.sort_values("candidate_votes", ascending=False)
          .drop_duplicates(["state_abbr", "district", "party_major"], keep="first")
          .reset_index(drop=True)
    )
    return df


def _party_major(p: object) -> str | None:
    """Map a raw party string (or NaN) to one of {'D', 'R', None}."""
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return None
    pu = str(p).upper().strip()
    if pu in DEM_PARTY_LABELS:
        return "D"
    if pu in REP_PARTY_LABELS:
        return "R"
    return None


def _attach_ratings(df: pd.DataFrame, rat: pd.DataFrame) -> pd.DataFrame:
    """Left-join the ratings table on (state_abbr, district)."""
    cols = [c for c in (
        "state_abbr", "district",
        "cpvi_signed", "cook_ordinal", "sabato_ordinal", "inside_ordinal",
        "incumbent_raw", "flip_predicted",
    ) if c in rat.columns]
    out = df.merge(rat[cols], on=["state_abbr", "district"], how="left")
    return out


def _attach_pvi_fallback(df: pd.DataFrame, pvi_df: pd.DataFrame) -> pd.DataFrame:
    """Where ratings table didn't supply CPVI, fall back to Daily Kos."""
    out = df.merge(
        pvi_df[["state_abbr", "district", "cpvi_signed"]].rename(
            columns={"cpvi_signed": "cpvi_dailykos"}
        ),
        on=["state_abbr", "district"],
        how="left",
    )
    out["cpvi"] = out.get("cpvi_signed").combine_first(out["cpvi_dailykos"])
    return out.drop(columns=[c for c in ("cpvi_signed", "cpvi_dailykos") if c in out.columns])


def _attach_demographics(df: pd.DataFrame, acs: pd.DataFrame) -> pd.DataFrame:
    """Left-join ACS columns on (state_abbr, district), prefixing with acs_."""
    keep = [
        "median_age", "race_white", "race_black", "race_asian",
        "edu_bachelors", "edu_grad_degree", "median_income", "below_100pct_fpl",
    ]
    avail = [c for c in keep if c in acs.columns]
    sub = acs[["state_abbr", "district", *avail]].rename(
        columns={c: f"acs_{c}" for c in avail}
    )
    return df.merge(sub, on=["state_abbr", "district"], how="left")


def _attach_fec(df: pd.DataFrame, fec_df: pd.DataFrame) -> pd.DataFrame:
    """Match candidates between MIT results and FEC by (state, district, last_name).

    Falls back to fuzzy last-name match within a district when exact fails.
    """
    fec_h = fec_df.copy()
    fec_h["state"] = fec_h["state"].astype(str).str.upper()
    # FEC district is zero-padded string e.g. '01' / '00' (at-large); convert to int.
    fec_h["district"] = pd.to_numeric(fec_h["district"], errors="coerce").fillna(0).astype(int)
    fec_h["last_name"] = fec_h["last_name"].astype(str).str.upper().str.strip()
    fec_h = fec_h.rename(columns={"state": "state_abbr"})

    # First pass: exact match on (state, district, last_name)
    exact_cols = ["state_abbr", "district", "last_name"]
    merged = df.merge(
        fec_h[[*exact_cols, "cand_id", "total_trans", "trans_by_indiv",
               "trans_by_cmte", "cand_ici"]],
        on=exact_cols,
        how="left",
    )

    # Second pass: fuzzy fill for unmatched rows
    missing = merged["cand_id"].isna()
    if missing.any():
        merged = _fuzzy_fill_fec(merged, fec_h, missing)

    return merged


def _fuzzy_fill_fec(merged: pd.DataFrame, fec_h: pd.DataFrame, missing_mask: pd.Series) -> pd.DataFrame:
    """For rows without an exact FEC match, try fuzzy + substring within district.

    Three passes for the unmatched rows:
      1. difflib close-match at cutoff 0.85 (handles minor spelling variants)
      2. substring containment in either direction (handles compound surnames
         like "GLUESENKAMP PEREZ" vs "PEREZ" — MIT takes the trailing token,
         FEC keeps the full pre-comma string)
      3. give up
    """
    from difflib import get_close_matches

    fec_by_district = fec_h.groupby(["state_abbr", "district"])

    def find_match(row: pd.Series) -> dict[str, object]:
        key = (row["state_abbr"], row["district"])
        if key not in fec_by_district.groups:
            return {}
        candidates = fec_h.loc[fec_by_district.groups[key]]
        if candidates.empty:
            return {}
        # Restrict to FEC candidates of the same major party where possible —
        # avoids cross-party false matches when names are similar.
        same_party = candidates
        if "party" in candidates.columns and row.get("party_major") in {"D", "R"}:
            party_filter = {"D": ["DEM"], "R": ["REP"]}[row["party_major"]]
            party_match = candidates[candidates["party"].isin(party_filter)]
            if not party_match.empty:
                same_party = party_match

        names = same_party["last_name"].tolist()
        target = str(row["last_name"]).strip().upper()

        # Pass 1: difflib close-match
        match = get_close_matches(target, names, n=1, cutoff=0.85)
        if match:
            rec = same_party.loc[same_party["last_name"] == match[0]].iloc[0]
        else:
            # Pass 2: substring containment in either direction
            substring_hits = [
                n for n in names
                if n and target and (target in n or n in target)
            ]
            if not substring_hits:
                return {}
            # Prefer the shortest substring hit (most specific match)
            best = min(substring_hits, key=len)
            rec = same_party.loc[same_party["last_name"] == best].iloc[0]

        return {
            "cand_id": rec["cand_id"],
            "total_trans": rec["total_trans"],
            "trans_by_indiv": rec["trans_by_indiv"],
            "trans_by_cmte": rec["trans_by_cmte"],
            "cand_ici": rec["cand_ici"],
        }

    fills = merged.loc[missing_mask].apply(find_match, axis=1)
    for col in ("cand_id", "total_trans", "trans_by_indiv", "trans_by_cmte", "cand_ici"):
        if not fills.empty:
            merged.loc[missing_mask, col] = merged.loc[missing_mask, col].combine_first(
                fills.map(lambda d: d.get(col))
            )
    return merged


def _attach_ie(df: pd.DataFrame, ie_df: pd.DataFrame) -> pd.DataFrame:
    return df.merge(
        ie_df[["cand_id", "ie_for_total", "ie_against_total"]],
        on="cand_id",
        how="left",
    )


# ---------- contested-race filter ----------

def _apply_contested_race_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only contested two-party general elections.

    Drops:
      * Districts with fewer than 2 major-party candidates
      * Candidates whose party isn't D or R (already filtered upstream, but defensive)
      * Candidates without a FEC record (couldn't have filed Q2)
      * Candidates whose race had margin == ±1.0 (uncontested)
    """
    df = df.loc[df["party_major"].isin(["D", "R"])].copy()
    df = df.loc[df["cand_id"].notna()].copy()
    df = df.loc[df["margin_pct_signed"].between(-0.999999, 0.999999)].copy()
    if df.empty:
        return df

    # FEC fuzzy matches can attach multiple committees to one MIT candidate;
    # keep the one with the highest total_trans (most active committee) per
    # (state, district, party_major).
    df["total_trans"] = pd.to_numeric(df["total_trans"], errors="coerce").fillna(0)
    df = (
        df.sort_values("total_trans", ascending=False)
          .drop_duplicates(["state_abbr", "district", "party_major"], keep="first")
          .reset_index(drop=True)
    )

    # Each (state, district) must have one D and one R after the above.
    party_sets = df.groupby(["state_abbr", "district"])["party_major"].apply(set)
    valid_mask = party_sets.apply(lambda s: s == {"D", "R"})
    valid_districts = party_sets[valid_mask].index
    if len(valid_districts) == 0:
        return df.iloc[0:0]
    df = df.set_index(["state_abbr", "district"]).loc[valid_districts].reset_index()
    return df


# ---------- derived columns ----------

def _compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log fundraising, opponent features, incumbent flag, signed margin."""
    df = df.copy()
    if df.empty:
        # Filter dropped everything; return empty frame with the expected columns
        # so downstream code (assertions, parquet write) still has a valid schema.
        for col in ("margin_pct", "incumbent", "self_raised_log", "opp_raised",
                    "opp_raised_log", "self_raised_pct", "ie_for_log",
                    "ie_against_log", "cook_rating", "total_trans"):
            if col not in df.columns:
                df[col] = pd.Series(dtype="float64")
        return df

    # Single source of truth: signed margin (D% - R%)
    df["margin_pct"] = df["margin_pct_signed"]

    # Per-row: this candidate's party determines the sign of their stake
    # (a Dem in a D+10 race has the same `margin_pct` regardless; the model
    # learns the party axis through the `party_major` column).

    # Incumbency: prefer FEC's I/C/O code; fall back to Wikipedia incumbent_raw.
    df["incumbent"] = (
        df["cand_ici"].astype(str).str.upper().eq("I").astype("Int64")
    )
    if "incumbent_raw" in df.columns:
        # If FEC didn't say but Wikipedia row hints it's the incumbent,
        # try matching candidate's last name against the Wikipedia incumbent string.
        wiki_match = df.apply(
            lambda r: int(
                isinstance(r.get("incumbent_raw"), str)
                and r["last_name"] in str(r["incumbent_raw"]).upper()
            ),
            axis=1,
        )
        df["incumbent"] = df["incumbent"].fillna(wiki_match.astype("Int64"))

    # Snapshot fundraising (already snapshot-filtered upstream by fec.fetch_fec)
    df["total_trans"] = pd.to_numeric(df["total_trans"], errors="coerce").fillna(0)
    df["self_raised_log"] = np.log1p(df["total_trans"])

    # Opponent fundraising — paired by district
    opp = (
        df[["state_abbr", "district", "party_major", "total_trans"]]
        .pivot_table(
            index=["state_abbr", "district"],
            columns="party_major",
            values="total_trans",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    df = df.merge(
        opp.rename(columns={"D": "raised_D", "R": "raised_R"}),
        on=["state_abbr", "district"],
        how="left",
    )
    # Defensive: pivot only produces columns for parties present; backfill any missing.
    for col in ("raised_D", "raised_R"):
        if col not in df.columns:
            df[col] = 0.0
    df["opp_raised"] = np.where(df["party_major"] == "D", df["raised_R"], df["raised_D"])
    df["opp_raised_log"] = np.log1p(df["opp_raised"])
    denom = df["total_trans"] + df["opp_raised"]
    df["self_raised_pct"] = np.where(denom > 0, df["total_trans"] / denom, 0.5)

    # Independent expenditures
    df["ie_for_total"] = pd.to_numeric(df.get("ie_for_total"), errors="coerce").fillna(0)
    df["ie_against_total"] = pd.to_numeric(df.get("ie_against_total"), errors="coerce").fillna(0)
    df["ie_for_log"] = np.log1p(df["ie_for_total"])
    df["ie_against_log"] = np.log1p(df["ie_against_total"])

    # Cook ordinal → cook_rating column expected by feature_sets
    if "cook_ordinal" in df.columns:
        df["cook_rating"] = df["cook_ordinal"]

    return df.drop(columns=[c for c in ("raised_D", "raised_R") if c in df.columns])


# ---------- invariants ----------

def _assert_invariants(df: pd.DataFrame) -> None:
    """Hard checks that catch the most common failure modes."""
    if df.empty:
        return
    assert df["margin_pct"].between(-1, 1).all(), "margin_pct out of [-1, 1]"
    assert df["margin_pct"].abs().lt(1.0).all(), "uncontested race leaked through filter"
    counts = df.groupby(["state_abbr", "district"]).size()
    assert (counts == 2).all(), f"districts must have exactly 2 candidates; got: {counts[counts != 2].head()}"
    # Two candidates per district, opposing parties
    for (s, d), grp in df.groupby(["state_abbr", "district"]):
        parties = sorted(grp["party_major"].unique())
        assert parties == ["D", "R"], f"district {s}-{d}: parties were {parties}"


# ---------- CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the per-(cycle, snapshot) feature matrix.")
    parser.add_argument("--cycle", type=int, required=True, choices=list(CYCLES))
    parser.add_argument("--snapshot", type=str, required=True, choices=list(SNAPSHOT_OFFSETS_DAYS))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    df = build_features(args.cycle, args.snapshot, args.raw_dir)

    out = args.out_dir / f"candidates_{args.cycle}_{args.snapshot}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[features] wrote {len(df)} rows to {out}")
    print(df.head())


if __name__ == "__main__":
    main()
