"""Contemporaneous Cook race-ratings via the Wayback Machine.

Background: the Wikipedia ratings pages for 2014, 2016, and 2018 were created
*after* their respective elections (March 2021 for 2014/2016; Dec 2018 for
2018). Snapshot queries against `ingest/ratings.py` for those cycles return
the *final* back-filled ratings, which is perfect-foresight leakage when the
cycle is in the training set.

This module fetches contemporaneous Cook race ratings from
`web.archive.org/web/{ts}/cookpolitical.com/house/charts/race-ratings` for any
(cycle, snapshot_date) pair, parses them into the same per-district DataFrame
shape that `ingest/ratings.py` produces, and caches the raw HTML + parsed
parquet under `data/raw/wayback_cook/`.

Workflow:
    cook_df = fetch_wayback_cook(cycle=2016, snapshot=date(2016, 9, 9), raw_dir=...)

The returned frame has the same canonical columns the Wikipedia parser emits:
`state_abbr, district, cook_ordinal`. CPVI is *not* included (Cook's
ratings page didn't carry CPVI on these older pages); for CPVI, the existing
`ingest/pvi.py` (or Wikipedia for cycles where it's tracked) is the source.

Audit-trail metadata is exposed via the returned `WaybackCookResult` dataclass
so `features.py` can record which Wayback snapshot timestamp the ratings
came from.
"""

from __future__ import annotations

import io
import json
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

USER_AGENT = "oath_score/0.1 (audit; benjaminematton@gmail.com)"

# Cook's house race-ratings URL hasn't changed across the 2014/2016 cycles.
# 2018 may live at a different path — verified via spike on 2026-04-25.
COOK_URL_PATH = "cookpolitical.com/house/charts/race-ratings"

# Cook's bucket label -> ordinal scale matching the Wikipedia parser:
# 1=Solid R, 2=Likely R, 3=Lean R, 4=Toss-up, 5=Lean D, 6=Likely D, 7=Solid D.
# Cook splits Toss-up into "Republican Toss Up" (3.5) and "Democratic Toss Up"
# (4.5) on the older pages; fold to nearest integer for parity with Wikipedia.
RATING_TO_ORDINAL: dict[str, float] = {
    "Solid Republican":    1.0,
    "Likely Republican":   2.0,
    "Lean Republican":     3.0,
    "Republican Toss Up":  3.5,
    "Toss Up":             4.0,
    "Democratic Toss Up":  4.5,
    "Lean Democratic":     5.0,
    "Likely Democratic":   6.0,
    "Solid Democratic":    7.0,
}


@dataclass(frozen=True)
class WaybackCookResult:
    df: pd.DataFrame
    wayback_timestamp: str
    cook_url: str
    cycle: int
    snapshot: date
    days_off_target: int  # |snapshot - wayback_ts.date()|; non-negative


def fetch_wayback_cook(
    cycle: int,
    snapshot: date,
    raw_dir: Path,
    *,
    force_refresh: bool = False,
) -> WaybackCookResult:
    """Return contemporaneous Cook ratings for `(cycle, snapshot)` from Wayback."""
    cache_dir = Path(raw_dir) / "wayback_cook"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ts_index = _resolve_wayback_ts(cycle, snapshot, cache_dir, force_refresh=force_refresh)
    parquet_cache = cache_dir / f"cook_{cycle}_{snapshot.isoformat()}_wayback_{ts_index}.parquet"
    html_cache = cache_dir / f"cook_{cycle}_{snapshot.isoformat()}_wayback_{ts_index}.html"

    if not parquet_cache.exists() or force_refresh:
        if not html_cache.exists() or force_refresh:
            html = _fetch_wayback_html(ts_index)
            html_cache.write_text(html)
        else:
            html = html_cache.read_text()
        df = _parse_wayback_cook(html)
        df.to_parquet(parquet_cache, index=False)
    else:
        df = pd.read_parquet(parquet_cache)

    days_off = abs((snapshot - _ts_to_date(ts_index)).days)
    cook_url = f"https://web.archive.org/web/{ts_index}/http://{COOK_URL_PATH}"
    return WaybackCookResult(
        df=df, wayback_timestamp=ts_index, cook_url=cook_url,
        cycle=cycle, snapshot=snapshot, days_off_target=days_off,
    )


def _resolve_wayback_ts(
    cycle: int,
    snapshot: date,
    cache_dir: Path,
    *,
    force_refresh: bool,
) -> str:
    """Find the closest Wayback snapshot (ts string YYYYMMDDHHMMSS) <= snapshot date.

    Caches the (cycle, snapshot_iso) -> ts mapping in cache_dir/wayback_index.json.
    """
    index_path = cache_dir / "wayback_index.json"
    index: dict[str, str] = {}
    if index_path.exists():
        index = json.loads(index_path.read_text())
    key = f"{cycle}_{snapshot.isoformat()}"
    if not force_refresh and key in index:
        return index[key]

    # CDX API: list all snapshots in the cycle's window, pick latest <= snapshot
    cycle_start = f"{cycle - 1}1101"  # day after prior election year's election
    snap_str = snapshot.strftime("%Y%m%d")
    cdx_url = (
        "https://web.archive.org/cdx/search/cdx"
        f"?url={COOK_URL_PATH}"
        f"&from={cycle_start}&to={snap_str}235959"
        "&output=json&collapse=timestamp:8"
    )
    rows = _retry_json(cdx_url)
    if not rows or len(rows) <= 1:
        raise RuntimeError(
            f"Wayback CDX returned no snapshots of {COOK_URL_PATH} "
            f"in window [{cycle_start}, {snap_str}] for cycle={cycle} snapshot={snapshot}."
        )
    header, data = rows[0], rows[1:]
    ts_idx = header.index("timestamp")
    candidates = [r[ts_idx] for r in data if r[ts_idx][:8] <= snap_str]
    if not candidates:
        raise RuntimeError(
            f"Wayback has no snapshots <= {snap_str} for cycle={cycle}; "
            "earliest available is later than the target snapshot."
        )
    chosen = max(candidates)
    index[key] = chosen
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
    return chosen


def _fetch_wayback_html(ts: str) -> str:
    """Stream the archived Cook page HTML at the given Wayback timestamp."""
    url = f"https://web.archive.org/web/{ts}/http://{COOK_URL_PATH}"
    return _retry_text(url)


def _parse_wayback_cook(html: str) -> pd.DataFrame:
    """Parse the Cook race-ratings HTML into per-district rows.

    Cook's page renders one HTML table per rating bucket, each with columns
    [DIST, REPRESENTATIVE, PVI]. Only the first column matters for our
    purpose (district code → ordinal mapping); REPRESENTATIVE and PVI are
    kept for the audit trail / debugging.
    """
    tables = pd.read_html(io.StringIO(html))
    rows = []
    for t in tables:
        if t.shape[0] < 2 or t.shape[1] != 3:
            continue
        bucket = t.columns[0]
        if bucket not in RATING_TO_ORDINAL:
            continue
        ord_val = RATING_TO_ORDINAL[bucket]
        for _, row in t.iloc[1:].iterrows():
            dist = str(row[t.columns[0]]).strip()
            m = re.match(r"^([A-Z]{2})-(\d{1,2})$", dist)
            if not m:
                continue
            rows.append({
                "state_abbr": m.group(1),
                "district":   int(m.group(2)),
                "cook_ordinal": ord_val,
                "cook_rating_label": bucket,
                "incumbent_raw": str(row[t.columns[1]]).strip(),
                "pvi_text":  str(row[t.columns[2]]).strip(),
            })
    if not rows:
        raise RuntimeError(
            "Could not parse any per-district rows from the Cook Wayback HTML "
            "(no tables with Solid/Lean/Likely/Toss Up bucket headers)."
        )
    return pd.DataFrame(rows)


def _retry_json(url: str, *, retries: int = 3, timeout: int = 120) -> list:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code} from {url}")
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _retry_text(url: str, *, retries: int = 3, timeout: int = 120) -> str:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code} from {url}")
            return r.text
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _ts_to_date(ts: str) -> date:
    """Wayback timestamp YYYYMMDDHHMMSS -> date."""
    return date(int(ts[:4]), int(ts[4:6]), int(ts[6:8]))
