"""Pure-Python tests for the ratings parser. No network, no pandas required."""

from __future__ import annotations

import pytest

from oath_score.ingest.ratings import (
    BACKFILLED_CYCLES,
    RATING_ORDINAL,
    _parse_cpvi,
    _parse_district_string,
    _rating_to_ordinal,
    is_backfilled,
)


class TestDistrictParsing:
    def test_numbered_district(self):
        assert _parse_district_string("Pennsylvania 8") == ("PA", 8)

    def test_double_digit(self):
        assert _parse_district_string("California 45") == ("CA", 45)

    def test_at_large(self):
        assert _parse_district_string("Alaska at-large") == ("AK", 0)

    def test_at_large_montana(self):
        # Montana had at-large pre-2022 redistricting
        assert _parse_district_string("Montana at-large") == ("MT", 0)

    def test_unknown_state_returns_none(self):
        assert _parse_district_string("Puerto Rico 1") is None

    def test_separator_row_returns_none(self):
        assert _parse_district_string("") is None
        assert _parse_district_string("nan") is None
        assert _parse_district_string("Total") is None


class TestCpviParsing:
    def test_r_lean(self):
        assert _parse_cpvi("R+8") == -8.0

    def test_d_lean(self):
        assert _parse_cpvi("D+12") == 12.0

    def test_even(self):
        assert _parse_cpvi("EVEN") == 0.0
        assert _parse_cpvi("Even") == 0.0

    def test_new_seat(self):
        assert _parse_cpvi("New seat") is None

    def test_with_whitespace(self):
        assert _parse_cpvi(" R+5 ") == -5.0

    def test_empty(self):
        assert _parse_cpvi("") is None
        assert _parse_cpvi(None) is None


class TestRatingOrdinal:
    @pytest.mark.parametrize("text,expected", [
        ("Solid R", 1.0),
        ("Likely R", 2.0),
        ("Lean R", 3.0),
        ("Tilt R", 3.5),
        ("Tossup", 4.0),
        ("Toss-up", 4.0),
        ("Tilt D", 4.5),
        ("Lean D", 5.0),
        ("Likely D", 6.0),
        ("Solid D", 7.0),
        ("Safe D", 7.0),
    ])
    def test_known_ratings(self, text, expected):
        assert _rating_to_ordinal(text) == expected

    def test_strips_flip_suffix(self):
        # "Tossup (flip)" should still resolve to 4.0
        assert _rating_to_ordinal("Tossup (flip)") == 4.0
        assert _rating_to_ordinal("Lean R (flip)") == 3.0

    def test_unknown_returns_none(self):
        assert _rating_to_ordinal("Some random text") is None
        assert _rating_to_ordinal("") is None

    def test_ordinal_monotonic(self):
        """Verify the rating ordinal scale is strictly monotonic from R to D."""
        ladder = ["solid r", "likely r", "lean r", "tilt r", "tossup",
                  "tilt d", "lean d", "likely d", "solid d"]
        values = [RATING_ORDINAL[r] for r in ladder]
        assert values == sorted(values), "Ordinal scale must be monotone increasing R→D"


class TestBackfilledCycles:
    def test_2014_is_backfilled(self):
        assert is_backfilled(2014)
        assert 2014 in BACKFILLED_CYCLES

    def test_2016_is_backfilled(self):
        assert is_backfilled(2016)

    def test_2022_is_real(self):
        assert not is_backfilled(2022)

    def test_2024_is_real(self):
        assert not is_backfilled(2024)
