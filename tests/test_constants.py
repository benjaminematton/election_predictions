"""Pin snapshot-date arithmetic across all (cycle, snapshot) pairs.

Guards against accidental edits to GENERAL_ELECTION_DATES or SNAPSHOT_OFFSETS_DAYS
that would silently shift every backtest.
"""

from __future__ import annotations

from datetime import date

import pytest

from oath_score.constants import (
    CYCLES,
    GENERAL_ELECTION_DATES,
    SNAPSHOT_OFFSETS_DAYS,
    snapshot_date_for,
)

# Verified by hand against (election_date - offset_days) at the time of writing.
EXPECTED: dict[tuple[int, str], date] = {
    (2014, "T-110"): date(2014, 7, 17),
    (2014, "T-60"):  date(2014, 9, 5),
    (2014, "T-20"):  date(2014, 10, 15),
    (2016, "T-110"): date(2016, 7, 21),
    (2016, "T-60"):  date(2016, 9, 9),
    (2016, "T-20"):  date(2016, 10, 19),
    (2022, "T-110"): date(2022, 7, 21),
    (2022, "T-60"):  date(2022, 9, 9),
    (2022, "T-20"):  date(2022, 10, 19),
    (2024, "T-110"): date(2024, 7, 18),
    (2024, "T-60"):  date(2024, 9, 6),
    (2024, "T-20"):  date(2024, 10, 16),
}


@pytest.mark.parametrize(("cycle", "snapshot", "expected"),
                         [(c, s, d) for (c, s), d in EXPECTED.items()])
def test_snapshot_date_for(cycle: int, snapshot: str, expected: date) -> None:
    assert snapshot_date_for(cycle, snapshot) == expected


def test_all_pairs_covered() -> None:
    pairs = {(c, s) for c in CYCLES for s in SNAPSHOT_OFFSETS_DAYS}
    assert pairs == set(EXPECTED.keys())


def test_election_dates_are_tuesdays() -> None:
    # Federal general elections are the Tuesday after the first Monday in November.
    for cycle, dt in GENERAL_ELECTION_DATES.items():
        assert dt.weekday() == 1, f"{cycle} election date {dt} is not a Tuesday"


def test_snapshot_strictly_before_election() -> None:
    for cycle in CYCLES:
        for snapshot in SNAPSHOT_OFFSETS_DAYS:
            assert snapshot_date_for(cycle, snapshot) < GENERAL_ELECTION_DATES[cycle]


def test_unknown_snapshot_raises() -> None:
    with pytest.raises(KeyError):
        snapshot_date_for(2024, "T-999")
