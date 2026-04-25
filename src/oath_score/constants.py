"""Shared constants: state codes, snapshot offsets, election dates."""

from __future__ import annotations

from datetime import date

STATE_ABBR: dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}

# 50 states; FIPS codes used by Census API (DC excluded — no House district)
STATE_FIPS: list[int] = [
    1, 2, 4, 5, 6, 8, 9, 10, 12, 13,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 44, 45,
    46, 47, 48, 49, 50, 51, 53, 54, 55, 56,
]

GENERAL_ELECTION_DATES: dict[int, date] = {
    2014: date(2014, 11, 4),
    2016: date(2016, 11, 8),
    2022: date(2022, 11, 8),
    2024: date(2024, 11, 5),
}

SNAPSHOT_OFFSETS_DAYS: dict[str, int] = {
    "T-110": 110,
    "T-60": 60,
    "T-20": 20,
}

CYCLES: tuple[int, ...] = (2014, 2016, 2022, 2024)


def snapshot_date_for(cycle: int, snapshot: str) -> date:
    """Return the absolute calendar date for a (cycle, snapshot) combo.

    `snapshot` must be a key in SNAPSHOT_OFFSETS_DAYS (e.g. 'T-60').
    """
    election = GENERAL_ELECTION_DATES[cycle]
    offset = SNAPSHOT_OFFSETS_DAYS[snapshot]
    return date.fromordinal(election.toordinal() - offset)
