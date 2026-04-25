"""Cook Partisan Voting Index per congressional district (Daily Kos source).

Used as the comprehensive CPVI source covering all 435 districts. The
Wikipedia ratings page (ingest/ratings.py) provides CPVI for the ~180
competitive races; this module fills the gap for everything else.

Daily Kos publishes one Google-Sheet-as-CSV per district map version:
  pre-2020 maps: used for 2014, 2016 cycles
  post-2020 maps: used for 2022, 2024 cycles

The Daily Kos URL has moved several times; if the canonical URL fails, fall
back to a manual download placed at:
  data/raw/pvi/dailykos_pvi_{map_version}.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from oath_score.constants import STATE_ABBR
from oath_score.ingest._download import download_file
from oath_score.ingest.ratings import _parse_cpvi  # reuse same parser

# Daily Kos hosting moves periodically. These URLs are best-known-good as of
# 2026; if either fails, the auto-download function raises with manual-fallback
# instructions and the user stages the CSV by hand.
PVI_URLS: dict[str, str] = {
    "pre2020":  "https://docs.google.com/spreadsheets/d/1iVcL3DvWjugrjYfBGodVcIoEQ2iN7ULzvvJaoOO_Mh8/export?format=csv",
    "post2020": "https://docs.google.com/spreadsheets/d/1izhJrh4WTVOxztpEU_ZmPwl5amJp6V2hDe0kE6nAyXo/export?format=csv",
}

CYCLE_MAP_VERSION: dict[int, str] = {
    2014: "pre2020", 2016: "pre2020",
    2022: "post2020", 2024: "post2020",
}

# Reverse-lookup: state name (uppercase) → abbreviation.
_NAME_TO_ABBR = {name.upper(): abbr for name, abbr in STATE_ABBR.items()}


def fetch_pvi(cycle: int, raw_dir: Path) -> pd.DataFrame:
    """Return one row per congressional district with signed CPVI.

    Map version selected automatically by cycle (pre/post 2020 redistricting).

    Columns: state_abbr, district, cpvi_signed (D positive)
    """
    map_ver = CYCLE_MAP_VERSION[cycle]
    csv_path = _ensure_csv(map_ver, raw_dir)
    return _parse_csv(csv_path, map_ver)


def _ensure_csv(map_ver: str, raw_dir: Path) -> Path:
    """Download Daily Kos PVI CSV if not staged. Returns the local path."""
    raw_dir = Path(raw_dir) / "pvi"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"dailykos_pvi_{map_ver}.csv"
    if dest.exists():
        return dest

    url = PVI_URLS[map_ver]
    try:
        download_file(url, dest)
    except Exception as exc:
        raise RuntimeError(
            f"Could not auto-download Daily Kos PVI for map_version={map_ver}.\n"
            f"  Tried: {url}\n"
            f"  Original error: {exc}\n\n"
            f"Manual fallback: search 'Daily Kos PVI by congressional district' "
            f"for the {map_ver} CSV and place it at:\n  {dest}"
        ) from exc
    return dest


def _parse_csv(path: Path, map_ver: str) -> pd.DataFrame:
    """Parse the Daily Kos CSV — tolerant of small column-name variations."""
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # Daily Kos columns vary slightly across publications; locate the relevant
    # ones by fuzzy keyword match.
    dist_col = _find_col(df, ["district", "cd"])
    pvi_col = _find_col(df, ["pvi", "cpvi"])
    if dist_col is None or pvi_col is None:
        raise RuntimeError(
            f"Could not identify district/PVI columns in {path}. "
            f"Got: {list(df.columns)[:15]}"
        )

    parsed = df[dist_col].astype(str).apply(_parse_dailykos_district)
    out = pd.DataFrame({
        "state_abbr": parsed.map(lambda x: x[0] if x else None),
        "district":   parsed.map(lambda x: x[1] if x else None),
        "cpvi_signed": df[pvi_col].astype(str).apply(_parse_cpvi),
    })
    out = out.dropna(subset=["state_abbr", "district"]).copy()
    out["district"] = out["district"].astype(int)
    out["map_version"] = map_ver
    return out


def _find_col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """Return the first column name containing any of `keywords` (case-insensitive)."""
    for col in df.columns:
        low = col.lower()
        if any(k in low for k in keywords):
            return col
    return None


def _parse_dailykos_district(s: str) -> tuple[str, int] | None:
    """Daily Kos district format varies: 'AL-01', 'AL-1', 'Alabama 1', 'AK-AL'."""
    if not s or s.lower() == "nan":
        return None
    s = s.strip()

    if "-" in s:
        head, tail = s.split("-", 1)
        head = head.strip().upper()
        tail = tail.strip().upper()
        # State part: 2-letter abbr, or full name
        abbr = head if len(head) == 2 else _NAME_TO_ABBR.get(head)
        if abbr is None:
            return None
        if tail in ("AL", "AT-LARGE", "ATLARGE"):
            return abbr, 0
        try:
            return abbr, int(tail)
        except ValueError:
            return None

    # "Alabama 1" / "Alaska At-Large" form
    parts = s.rsplit(" ", 1)
    if len(parts) != 2:
        return None
    state, num = parts
    abbr = _NAME_TO_ABBR.get(state.strip().upper())
    if abbr is None:
        return None
    if num.lower() in ("at-large", "atlarge", "al"):
        return abbr, 0
    try:
        return abbr, int(num)
    except ValueError:
        return None


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Load Daily Kos PVI for one cycle.")
    parser.add_argument("--cycle", type=int, required=True, choices=[2014, 2016, 2022, 2024])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    df = fetch_pvi(args.cycle, args.raw_dir)
    print(f"Cycle {args.cycle}: {len(df)} districts")
    print(df.head(10))


if __name__ == "__main__":
    main()
