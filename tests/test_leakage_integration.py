"""End-to-end leakage integration test.

Asserts the snapshot-date discipline of the full feature pipeline by inspecting
every committed `candidates_{cycle}_{snapshot}.parquet` file:

  * For CLEAN cycles (not in BACKFILLED_CYCLES): the ratings_revision_ts column
    must be <= snapshot_date for every row. Test FAILS on regression.
  * For LEAKY cycles (in BACKFILLED_CYCLES): every row must carry
    leaky_ratings=True. Test FAILS if a back-filled cycle is silently un-flagged.

Parquets baked before the audit columns landed are SKIPPED (with a reason),
so the test isn't a hard fail until parquets are re-baked. New parquets must
have the audit columns to pass.

This test is the load-bearing piece of the README "Validity checks" section's
claim that leakage is structurally enforced, not just promised.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from oath_score.constants import snapshot_date_for, SNAPSHOT_OFFSETS_DAYS
from oath_score.ingest.ratings import BACKFILLED_CYCLES

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def _all_candidate_parquets() -> list[Path]:
    if not PROCESSED_DIR.exists():
        return []
    return sorted(PROCESSED_DIR.glob("candidates_*.parquet"))


def _parse_filename(path: Path) -> tuple[int, str]:
    """`candidates_2024_T-60.parquet` -> (2024, 'T-60')."""
    stem = path.stem  # candidates_2024_T-60
    parts = stem.split("_", 2)
    if len(parts) != 3 or parts[0] != "candidates":
        raise ValueError(f"unexpected parquet name {path.name!r}")
    cycle = int(parts[1])
    snapshot = parts[2]
    if snapshot not in SNAPSHOT_OFFSETS_DAYS:
        raise ValueError(f"unknown snapshot {snapshot!r} in {path.name!r}")
    return cycle, snapshot


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

    if cycle in BACKFILLED_CYCLES:
        # Every row must carry the leaky flag — the back-fill is documented in
        # the data layer, not just hidden in a print statement.
        assert df["leaky_ratings"].all(), (
            f"{parquet.name}: cycle {cycle} is in BACKFILLED_CYCLES but "
            f"{(~df['leaky_ratings']).sum()}/{len(df)} rows are missing leaky_ratings=True"
        )
        return

    # Clean cycle: every row's ratings revision must be <= snapshot date.
    assert (~df["leaky_ratings"]).all(), (
        f"{parquet.name}: clean cycle {cycle} has rows flagged leaky_ratings=True"
    )
    rev_ts = pd.to_datetime(df["ratings_revision_ts"], errors="coerce").dt.date
    assert rev_ts.notna().all(), (
        f"{parquet.name}: ratings_revision_ts has un-parseable values"
    )
    assert (rev_ts <= snap_date).all(), (
        f"{parquet.name}: clean cycle {cycle} snapshot={snapshot} ({snap_date}) "
        f"has rows whose ratings_revision_ts is AFTER the snapshot. "
        f"Max revision ts: {rev_ts.max()}. This is a temporal leak."
    )


def test_backfilled_cycles_constant_documented() -> None:
    """Pin the BACKFILLED_CYCLES set so a careless edit doesn't silently un-flag a leaky cycle."""
    # Established by querying MediaWiki page-creation timestamps on 2026-04-25.
    # If you add or remove a cycle, also update the comment block above
    # BACKFILLED_CYCLES in src/oath_score/ingest/ratings.py with the page
    # creation date and reasoning.
    assert BACKFILLED_CYCLES == frozenset({2014, 2016, 2018})
