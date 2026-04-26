"""End-to-end leakage integration test.

Asserts the snapshot-date discipline of the full feature pipeline by inspecting
every committed `candidates_{cycle}_{snapshot}.parquet`:

* `leaky_ratings` is the source of truth — a True flag is documented leakage
  (back-filled Wikipedia where Wayback fell back), False means the ratings
  are contemporaneous and the test enforces `ratings_revision_ts <= snapshot
  + WAYBACK_SLOP` (Wayback returns the closest archived snapshot, which can
  be a few days off the target).
* When `ratings_source == "wikipedia"` and the cycle is in BACKFILLED_CYCLES,
  the row MUST carry `leaky_ratings=True`. Test FAILS if a back-filled
  Wikipedia source silently un-flags itself.
* When `ratings_source == "wayback_cook"`, the row MUST have
  `leaky_ratings=False` (the whole point of the Wayback fallback is to clear
  the leak).

This test is the load-bearing piece of the README "Validity checks" section's
claim that leakage is structurally enforced, not just promised. Parquets
predating the audit columns are SKIPPED (with a reason).
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from oath_score.constants import snapshot_date_for, SNAPSHOT_OFFSETS_DAYS
from oath_score.ingest.ratings import BACKFILLED_CYCLES

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Wayback returns the closest archived snapshot to the target. The Wayback
# index can lag the target snapshot date by a few weeks if archive coverage
# is thin. Allow up to 30 days of slop on the Wayback side specifically;
# Wikipedia revisions for clean cycles must be exact.
WAYBACK_SLOP_DAYS = 30


def _all_candidate_parquets() -> list[Path]:
    if not PROCESSED_DIR.exists():
        return []
    return sorted(PROCESSED_DIR.glob("candidates_*.parquet"))


def _parse_filename(path: Path) -> tuple[int, str]:
    """`candidates_2024_T-60.parquet` -> (2024, 'T-60')."""
    stem = path.stem
    parts = stem.split("_", 2)
    if len(parts) != 3 or parts[0] != "candidates":
        raise ValueError(f"unexpected parquet name {path.name!r}")
    cycle = int(parts[1])
    snapshot = parts[2]
    if snapshot not in SNAPSHOT_OFFSETS_DAYS:
        raise ValueError(f"unknown snapshot {snapshot!r} in {path.name!r}")
    return cycle, snapshot


def _parse_ts(ts: str) -> date | None:
    """Accept both ISO ('2024-09-06T21:46:12Z') and YYYYMMDDHHMMSS Wayback formats."""
    if not isinstance(ts, str):
        return None
    # Wayback compact format
    if len(ts) >= 8 and ts[:8].isdigit():
        try:
            return date(int(ts[:4]), int(ts[4:6]), int(ts[6:8]))
        except ValueError:
            pass
    # ISO
    parsed = pd.to_datetime(ts, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.date()


@pytest.mark.parametrize("parquet", _all_candidate_parquets(), ids=lambda p: p.name)
def test_no_post_snapshot_leakage(parquet: Path) -> None:
    """Audit one (cycle, snapshot) parquet for ratings-date leakage."""
    cycle, snapshot = _parse_filename(parquet)
    snap_date = snapshot_date_for(cycle, snapshot)
    df = pd.read_parquet(parquet)

    if "ratings_revision_ts" not in df.columns or "leaky_ratings" not in df.columns:
        pytest.skip(
            f"{parquet.name} predates audit columns (ratings_revision_ts, "
            f"leaky_ratings); re-bake to enable."
        )

    src = df["ratings_source"].unique() if "ratings_source" in df.columns else ["unknown"]
    assert len(src) == 1, f"{parquet.name}: mixed ratings_source values {src}"
    source = str(src[0])

    if source == "wikipedia" and cycle in BACKFILLED_CYCLES:
        # Back-filled Wikipedia must be flagged leaky.
        assert df["leaky_ratings"].all(), (
            f"{parquet.name}: source=wikipedia cycle={cycle} is in "
            f"BACKFILLED_CYCLES but {(~df['leaky_ratings']).sum()}/{len(df)} "
            f"rows are missing leaky_ratings=True"
        )
        return

    if source == "wayback_cook":
        # Wayback fallback clears the leak.
        assert (~df["leaky_ratings"]).all(), (
            f"{parquet.name}: source=wayback_cook should have leaky_ratings=False "
            f"on every row; got {df['leaky_ratings'].sum()}/{len(df)} flagged leaky"
        )
        # And the ts must be close to the snapshot (Wayback returns nearest archive).
        ts_dates = df["ratings_revision_ts"].apply(_parse_ts)
        assert ts_dates.notna().all(), (
            f"{parquet.name}: un-parseable ratings_revision_ts values"
        )
        max_ts = ts_dates.max()
        assert max_ts <= snap_date + timedelta(days=WAYBACK_SLOP_DAYS), (
            f"{parquet.name}: Wayback ts {max_ts} is more than "
            f"{WAYBACK_SLOP_DAYS} days after snapshot {snap_date}"
        )
        return

    # Clean Wikipedia cycle: leaky=False, ts <= snapshot strictly
    assert (~df["leaky_ratings"]).all(), (
        f"{parquet.name}: clean Wikipedia cycle {cycle} has rows flagged "
        f"leaky_ratings=True"
    )
    ts_dates = df["ratings_revision_ts"].apply(_parse_ts)
    assert ts_dates.notna().all(), (
        f"{parquet.name}: un-parseable ratings_revision_ts values"
    )
    assert (ts_dates <= snap_date).all(), (
        f"{parquet.name}: clean cycle {cycle} snapshot={snapshot} ({snap_date}) "
        f"has rows whose ratings_revision_ts is AFTER the snapshot. "
        f"Max revision ts: {ts_dates.max()}. Temporal leak."
    )


def test_backfilled_cycles_constant_documented() -> None:
    """Pin the BACKFILLED_CYCLES set so a careless edit doesn't silently un-flag a leaky cycle."""
    assert BACKFILLED_CYCLES == frozenset({2014, 2016, 2018})
