"""Tests for the Daily Kos PVI district-string parser."""

from __future__ import annotations

from oath_score.ingest.pvi import _parse_dailykos_district


class TestDailyKosDistrictParsing:
    def test_dash_format(self):
        assert _parse_dailykos_district("AL-01") == ("AL", 1)
        assert _parse_dailykos_district("CA-45") == ("CA", 45)

    def test_dash_no_zero_pad(self):
        assert _parse_dailykos_district("PA-8") == ("PA", 8)

    def test_at_large_dash_form(self):
        assert _parse_dailykos_district("AK-AL") == ("AK", 0)
        assert _parse_dailykos_district("MT-AT-LARGE") == ("MT", 0)

    def test_full_state_name_dash(self):
        assert _parse_dailykos_district("ALABAMA-1") == ("AL", 1)

    def test_space_format(self):
        assert _parse_dailykos_district("Pennsylvania 8") == ("PA", 8)

    def test_at_large_space_form(self):
        assert _parse_dailykos_district("Alaska At-Large") == ("AK", 0)

    def test_unknown_state(self):
        assert _parse_dailykos_district("Mars 1") is None

    def test_empty(self):
        assert _parse_dailykos_district("") is None
        assert _parse_dailykos_district("nan") is None
