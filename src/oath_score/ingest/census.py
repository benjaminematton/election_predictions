"""American Community Survey ingestion.

Pulls demographic + economic variables per congressional district from the Census
ACS 5-year estimates for a given year. Uses the `census` PyPI package as a thin
wrapper around the Census API.

ACS variables are cycle-static within an election cycle, so no snapshot_date
parameter is needed here. The data is frozen at the most recent ACS 5-year
estimate available before the election.

Required env: CENSUS_API_KEY  (request one at https://api.census.gov/data/key_signup.html)
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from oath_score.constants import STATE_ABBR, STATE_FIPS

# 33 ACS variables, carried over from the original acs.py (variable selection
# rationale: demographics, language, education, income, poverty by race/sex).
# Variable IDs are stable across ACS years; relabel here once.
ACS_VARIABLES: dict[str, str] = {
    "B01003_001E": "total_pop",
    "B01002_001E": "median_age",
    "B02001_002E": "race_white",
    "B02001_003E": "race_black",
    "B02001_004E": "race_amer_ind",
    "B02001_005E": "race_asian",
    "B02001_006E": "race_pacific",
    "B02001_008E": "race_two_plus",
    "B05001_006E": "not_us_citizen",
    "B05001_005E": "us_citizen_naturalized",
    "B05002_003E": "born_in_state",
    "B05012_001E": "born_in_us",
    "B06007_001E": "language_universe",
    "B06007_002E": "speaks_only_english",
    "B06007_003E": "speaks_spanish",
    "B06009_002E": "edu_less_than_hs",
    "B06009_003E": "edu_hs_grad",
    "B06009_004E": "edu_some_college",
    "B06009_011E": "edu_bachelors",
    "B06009_012E": "edu_grad_degree",
    "B06010_001E": "income_universe",
    "B06010_002E": "income_none",
    "B06011_001E": "median_income",
    "B06012_001E": "poverty_universe",
    "B06012_002E": "below_100pct_fpl",
    "B06012_003E": "fpl_100_to_149",
    "B17001A_003E": "poverty_white_male_below",
    "B17001A_017E": "poverty_white_female_below",
    "B17001A_032E": "poverty_white_male_above",
    "B17001A_046E": "poverty_white_female_above",
    "B17001B_003E": "poverty_black_male_below",
    "B17001B_017E": "poverty_black_female_below",
    "B17001B_032E": "poverty_black_male_above",
    "B17001B_046E": "poverty_black_female_above",
}

# Map cycle → ACS 5-year endpoint year.
# Conservative pick: ACS data ending the year before the election (so it would
# have been available at any snapshot date during the cycle).
ACS_YEAR_FOR_CYCLE: dict[int, int] = {
    2014: 2013,
    2016: 2015,
    2022: 2021,
    2024: 2023,
}


def fetch_acs(cycle: int, api_key: str | None = None) -> pd.DataFrame:
    """Download ACS 5-year estimates for all U.S. congressional districts.

    Returns a DataFrame with one row per (state, district) and columns for each
    ACS variable plus `state_fips`, `state_abbr`, `district`.

    Raises if `census` is not installed or if the API key is missing.
    """
    from census import Census  # local import: optional dep at import time

    api_key = api_key or os.environ.get("CENSUS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "CENSUS_API_KEY env var not set. Get one at "
            "https://api.census.gov/data/key_signup.html"
        )

    acs_year = ACS_YEAR_FOR_CYCLE[cycle]
    c = Census(api_key, year=acs_year)
    var_ids = list(ACS_VARIABLES.keys())

    rows: list[dict] = []
    for fips in STATE_FIPS:
        # acs5.state_congressional_district returns one row per CD in the state.
        results = c.acs5.state_congressional_district(
            ("NAME", *var_ids), str(fips).zfill(2), Census.ALL
        )
        rows.extend(results)

    df = pd.DataFrame(rows)
    df = df.rename(columns=ACS_VARIABLES)
    df = df.rename(columns={"state": "state_fips", "congressional district": "district"})
    df["state_fips"] = df["state_fips"].astype(int)
    df["district"] = pd.to_numeric(df["district"], errors="coerce").astype("Int64")
    df["state_abbr"] = df["state_fips"].map(_fips_to_abbr())
    df["cycle"] = cycle
    return df


def _fips_to_abbr() -> dict[int, str]:
    import us  # local import: optional dep
    return {int(s.fips): s.abbr for s in us.states.STATES if s.fips}


def save_parquet(df: pd.DataFrame, out_dir: Path, cycle: int) -> Path:
    """Persist the raw ACS frame to data/raw/{cycle}/acs.parquet."""
    out_dir = Path(out_dir) / str(cycle)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "acs.parquet"
    df.to_parquet(path, index=False)
    return path


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Fetch ACS demographics for one cycle.")
    parser.add_argument("--cycle", type=int, required=True, choices=[2014, 2016, 2022, 2024])
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    df = fetch_acs(args.cycle)
    path = save_parquet(df, args.out_dir, args.cycle)
    print(f"Wrote {len(df)} rows to {path}")


if __name__ == "__main__":
    main()
