"""Snapshot-aware race ratings ingestion via MediaWiki revision API.

For each cycle, the dedicated Wikipedia page
  "<cycle> United States House of Representatives election ratings"
contains a structured table with race ratings from up to 9 forecasters
(Cook, Sabato, Inside Elections, Politico, RealClearPolitics, Fox News,
Decision Desk HQ, FiveThirtyEight, The Economist), plus CPVI, incumbent,
and last-cycle margin.

We resolve a (cycle, snapshot_date) -> revision_id via the API, fetch the
page HTML at that revision, and parse the ratings table.

KNOWN LIMITATION: The 2014 and 2016 ratings pages were created on Wikipedia
in March 2021 (back-fill, not contemporaneous). Snapshot queries against
those cycles return the same back-filled data regardless of date. The
public function `is_backfilled(cycle)` exposes this so downstream code can
warn or down-weight.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

from oath_score.ingest._download import USER_AGENT

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_INDEX = "https://en.wikipedia.org/w/index.php"

PAGE_TITLE_FMT = "{cycle}_United_States_House_of_Representatives_election_ratings"

# Cycles whose ratings pages were created retroactively (March 2021) and
# therefore have no real snapshot history. Determined empirically — see
# Phase 1 spike notes in .claude/plans/phase2.md.
BACKFILLED_CYCLES: frozenset[int] = frozenset({2014, 2016})


# ---------- public API ----------

@dataclass(frozen=True)
class RatingsResult:
    df: pd.DataFrame
    revision_id: int
    revision_timestamp: str
    backfilled: bool


def fetch_ratings(
    cycle: int,
    snapshot: date,
    raw_dir: Path,
    *,
    force_refresh: bool = False,
) -> RatingsResult:
    """Fetch the ratings table as it stood on `snapshot` for `cycle`.

    Caches both the revision-id mapping (JSON) and the raw HTML at
      data/raw/{cycle}/ratings_revisions.json
      data/raw/{cycle}/ratings_<snapshot>.html
    """
    raw_dir = Path(raw_dir) / str(cycle)
    raw_dir.mkdir(parents=True, exist_ok=True)

    revid, ts = _resolve_revision(cycle, snapshot, raw_dir, force_refresh=force_refresh)
    html = _fetch_html(cycle, revid, snapshot, raw_dir, force_refresh=force_refresh)
    df = _parse_ratings_table(html, cycle=cycle, snapshot=snapshot, revision_id=revid)
    return RatingsResult(
        df=df,
        revision_id=revid,
        revision_timestamp=ts,
        backfilled=is_backfilled(cycle),
    )


def is_backfilled(cycle: int) -> bool:
    """True for cycles whose ratings page was created retroactively."""
    return cycle in BACKFILLED_CYCLES


# ---------- revision resolution ----------

def _resolve_revision(
    cycle: int, snapshot: date, raw_dir: Path, *, force_refresh: bool = False
) -> tuple[int, str]:
    """Return (revision_id, timestamp_iso) for the latest revision <= snapshot.

    For back-filled cycles (page created retroactively, after the cycle
    happened) there ARE no revisions <= snapshot. In that case, fall back to
    the earliest-ever revision of the page — which is the back-filled content
    we want for those cycles. The caller can check `is_backfilled(cycle)` to
    know whether the timestamp is contemporaneous.

    Caches the (cycle, snapshot) -> (revid, ts) mapping in
    data/raw/{cycle}/ratings_revisions.json.
    """
    cache_path = raw_dir / "ratings_revisions.json"
    cache: dict[str, list] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    key = snapshot.isoformat()
    if not force_refresh and key in cache:
        revid, ts = cache[key]
        return int(revid), ts

    title = PAGE_TITLE_FMT.format(cycle=cycle)

    # Primary query: latest revision at or before the snapshot.
    revid, ts = _query_revision(
        title, rvstart=f"{snapshot.isoformat()}T23:59:59Z", rvdir="older"
    )

    # Back-fill fallback: if no pre-snapshot revision exists, take the
    # earliest revision ever — that's the initial back-fill upload.
    if revid is None and is_backfilled(cycle):
        revid, ts = _query_revision(title, rvdir="newer")

    if revid is None:
        raise RuntimeError(
            f"No revisions found for {title!r} at or before {snapshot}; page may not exist."
        )

    cache[key] = [revid, ts]
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    return revid, ts


def _query_revision(
    title: str, *, rvstart: str | None = None, rvdir: str = "older"
) -> tuple[int | None, str | None]:
    """Single-revision MediaWiki API call. Returns (None, None) if no revisions found."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": title,
        "rvprop": "ids|timestamp",
        "rvlimit": "1",
        "rvdir": rvdir,
    }
    if rvstart is not None:
        params["rvstart"] = rvstart
    resp = requests.get(WIKI_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    revs = page.get("revisions") or []
    if not revs:
        return None, None
    rev = revs[0]
    return int(rev["revid"]), rev["timestamp"]


def _fetch_html(
    cycle: int, revid: int, snapshot: date, raw_dir: Path, *, force_refresh: bool = False
) -> str:
    cache_path = raw_dir / f"ratings_{snapshot.isoformat()}_rev{revid}.html"
    if cache_path.exists() and not force_refresh:
        return cache_path.read_text(encoding="utf-8")

    title = PAGE_TITLE_FMT.format(cycle=cycle)
    params = {"title": title, "oldid": str(revid)}
    resp = requests.get(WIKI_INDEX, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    cache_path.write_text(resp.text, encoding="utf-8")
    return resp.text


# ---------- HTML table parsing ----------

# Map text rating → ordinal on a 1..7 scale (Solid R = 1, Solid D = 7).
# "Tilt" is treated as a half-step between Lean and Toss-up.
RATING_ORDINAL: dict[str, float] = {
    "solid r": 1.0, "safe r": 1.0,
    "likely r": 2.0,
    "lean r": 3.0,
    "tilt r": 3.5,
    "tossup": 4.0, "toss-up": 4.0, "toss up": 4.0,
    "tilt d": 4.5,
    "lean d": 5.0,
    "likely d": 6.0,
    "solid d": 7.0, "safe d": 7.0,
}


def _parse_ratings_table(
    html: str, *, cycle: int, snapshot: date, revision_id: int
) -> pd.DataFrame:
    """Extract the per-district ratings table from the Wikipedia page HTML.

    Strategy: pd.read_html returns all tables on the page. We look for the
    one whose columns include 'District' (or starts with state/dist), 'CPVI'
    (or 'Cook PVI'), and at least one of {'Cook', 'Sabato', 'Inside'}.
    """
    # pandas >=3.0 requires StringIO for in-memory HTML; older versions
    # accepted raw strings. Wrap defensively.
    tables = pd.read_html(io.StringIO(html), flavor="lxml")
    target = _select_ratings_table(tables)
    if target is None:
        raise RuntimeError(
            f"Could not identify the ratings table on cycle={cycle} snapshot={snapshot} "
            f"rev={revision_id}. Found {len(tables)} tables."
        )

    df = _normalize_columns(target)
    df = _parse_district_rows(df)
    df["cycle"] = cycle
    df["snapshot_date"] = snapshot.isoformat()
    df["revision_id"] = revision_id
    return df


def _select_ratings_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    """Pick the per-district ratings table from a list of all page tables.

    Heuristic: must have >50 rows AND mention CPVI or Cook PVI in column names
    AND mention at least one of Cook/Sabato/Inside as a column.
    """
    for tbl in tables:
        cols = [_normalize_col(c) for c in tbl.columns]
        col_str = " ".join(cols)
        if len(tbl) < 50:
            continue
        has_pvi = "cpvi" in col_str or "cook_pvi" in col_str
        has_forecaster = any(f in col_str for f in ("cook", "sabato", "inside"))
        has_district = any("district" in c or c == "race" for c in cols)
        if has_pvi and has_forecaster and has_district:
            return tbl
    return None


def _normalize_col(col: object) -> str:
    """Snake-case a column name, handling MultiIndex tuples and noisy whitespace."""
    if isinstance(col, tuple):
        # MultiIndex headers — take the deepest non-blank level.
        for part in reversed(col):
            if part and not str(part).startswith("Unnamed"):
                col = part
                break
        else:
            col = "_".join(str(p) for p in col)
    s = str(col).strip().lower()
    s = re.sub(r"\[[^\]]*\]", "", s)  # drop [12] reference brackets
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_col(c) for c in df.columns]

    # Wikipedia ratings tables often use date-stamped column names like
    # "Cook (September 6, 2024)". Our normalizer turns those into
    # "cook_september_6_2024". Strip everything after a prefix match.
    forecaster_prefixes = (
        ("cook", "cook"),
        ("sabato", "sabato"),
        ("inside", "inside"),
        ("ie", "inside"),  # "IE" = Inside Elections in some cycles
        ("politico", "politico"),
        ("rcp", "rcp"),
        ("real_clear", "rcp"),
        ("race_to", "rcp"),
        ("fox", "fox"),
        ("ddhq", "ddhq"),
        ("decision_desk", "ddhq"),
        ("five38", "five38"),
        ("538", "five38"),
        ("fivethirtyeight", "five38"),
        ("economist", "economist"),
        ("the_economist", "economist"),
        ("cnalysis", "cnalysis"),
    )

    new_names: dict[str, str] = {}
    for col in df.columns:
        # First, exact-prefix aliases for non-dated columns
        if col in ("cook_pvi", "pvi"):
            new_names[col] = "cpvi"
            continue
        if col == "incumbent":
            new_names[col] = "incumbent_raw"
            continue
        # Then prefix-match against forecaster names
        matched = None
        for prefix, canonical in forecaster_prefixes:
            if col == prefix or col.startswith(prefix + "_"):
                matched = canonical
                break
        if matched is not None:
            new_names[col] = matched
    return df.rename(columns=new_names)


_DISTRICT_RE = re.compile(
    r"^(?P<state>[A-Za-z .]+?)\s+(?P<num>at[-\s]?large|\d+)$"
)
_CPVI_RE = re.compile(r"^(?P<sign>[RD])\s*\+\s*(?P<mag>\d+)$")


def _parse_district_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Parse district names → (state_abbr, district_int); CPVI → signed int; ratings → ordinals."""
    from oath_score.constants import STATE_ABBR

    # Find which column holds the district name.
    dist_col = next(
        (c for c in df.columns if "district" in c or c == "race"), df.columns[0]
    )

    parsed = df[dist_col].astype(str).apply(_parse_district_string)
    df["state_abbr"] = parsed.map(lambda x: x[0] if x else None)
    df["district"] = parsed.map(lambda x: x[1] if x else None)

    # Drop rows we couldn't parse (separator rows, etc.)
    df = df.dropna(subset=["state_abbr", "district"]).copy()
    df["district"] = df["district"].astype(int)

    if "cpvi" in df.columns:
        df["cpvi_signed"] = df["cpvi"].astype(str).apply(_parse_cpvi)

    for fc in ("cook", "sabato", "inside", "rcp", "politico", "fox", "ddhq", "five38", "economist"):
        if fc in df.columns:
            df[f"{fc}_ordinal"] = df[fc].astype(str).apply(_rating_to_ordinal)

    # Surface flip prediction from any forecaster suffix "(flip)"
    flip_cols = [c for c in df.columns if c in ("cook", "sabato", "inside")]
    if flip_cols:
        df["flip_predicted"] = df[flip_cols].apply(
            lambda row: any("flip" in str(v).lower() for v in row), axis=1
        )

    return df


def _parse_district_string(s: str) -> tuple[str, int] | None:
    """E.g. 'Pennsylvania 8' -> ('PA', 8); 'Alaska at-large' -> ('AK', 0)."""
    from oath_score.constants import STATE_ABBR
    if not s or s == "nan":
        return None
    m = _DISTRICT_RE.match(s.strip())
    if not m:
        return None
    state_name = m.group("state").strip()
    num_raw = m.group("num").lower().replace(" ", "-")
    abbr = STATE_ABBR.get(state_name)
    if abbr is None:
        return None
    if num_raw in ("at-large", "atlarge"):
        return abbr, 0
    try:
        return abbr, int(num_raw)
    except ValueError:
        return None


def _parse_cpvi(s: str) -> float | None:
    """'R+8' -> -8.0, 'D+12' -> +12.0, 'EVEN' -> 0.0, 'New seat' -> nan."""
    if not s:
        return None
    s = s.strip()
    if s.lower() == "even":
        return 0.0
    m = _CPVI_RE.match(s)
    if not m:
        return None
    mag = float(m.group("mag"))
    return mag if m.group("sign") == "D" else -mag


def _rating_to_ordinal(s: str) -> float | None:
    """Convert a rating cell to a 1-7 ordinal. Strips '(flip)' suffix."""
    if not s:
        return None
    cleaned = re.sub(r"\(.*?\)", "", str(s)).strip().lower()
    return RATING_ORDINAL.get(cleaned)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Fetch ratings table for one (cycle, snapshot).")
    parser.add_argument("--cycle", type=int, required=True, choices=[2014, 2016, 2022, 2024])
    parser.add_argument("--snapshot", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true", help="Bypass HTML/JSON cache")
    args = parser.parse_args()

    snap = datetime.strptime(args.snapshot, "%Y-%m-%d").date()
    res = fetch_ratings(args.cycle, snap, args.raw_dir, force_refresh=args.force)
    print(f"Cycle {args.cycle}, snapshot {args.snapshot}")
    print(f"  revision_id={res.revision_id}  ts={res.revision_timestamp}  backfilled={res.backfilled}")
    print(f"  rows: {len(res.df)}")
    print(res.df[["state_abbr", "district", "cpvi_signed", "cook_ordinal", "sabato_ordinal", "inside_ordinal"]].head(10))


if __name__ == "__main__":
    main()
