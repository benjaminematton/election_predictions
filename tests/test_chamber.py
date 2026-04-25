"""Tests for scores/chamber.py — the 435-seat chamber view."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from oath_score.scores.chamber import ChamberView, build_chamber


@pytest.fixture(scope="module")
def chamber_2024() -> ChamberView:
    """Build once per test session — the MIT data load is the slow step."""
    return build_chamber(2024, Path("data/raw"))


class TestChamberShape:
    def test_435_seats(self, chamber_2024: ChamberView):
        """Chamber should have 435 House seats. Allow small slack for vacancies/specials."""
        assert 430 <= chamber_2024.n_seats <= 440

    def test_status_counts_sum(self, chamber_2024: ChamberView):
        s = chamber_2024.df["status"]
        assert (s == "contested").sum() + (s == "d_lock").sum() + (s == "r_lock").sum() + (s == "other_lock").sum() == len(s)

    def test_status_values_only_canonical(self, chamber_2024: ChamberView):
        valid = {"contested", "d_lock", "r_lock", "other_lock"}
        assert set(chamber_2024.df["status"].unique()) <= valid


class TestKnownDistricts:
    def test_ma_7_is_d_lock(self, chamber_2024: ChamberView):
        """MA-7 (Pressley) — solidly D, often uncontested by R.
        If 2024 had a Republican challenger, this is contested; otherwise d_lock.
        Either way it should NOT be an R lock."""
        row = chamber_2024.df.query("state_abbr == 'MA' and district == 7").iloc[0]
        assert row["status"] in {"d_lock", "contested"}

    def test_al_1_is_r_dominant(self, chamber_2024: ChamberView):
        """AL-1 — solidly R."""
        row = chamber_2024.df.query("state_abbr == 'AL' and district == 1").iloc[0]
        # Could be contested (D challenger) or r_lock; either way winner is R
        assert row["winner_party"] == "R"

    def test_ca_13_is_contested(self, chamber_2024: ChamberView):
        """CA-13 — Gray vs Duarte, the closest 2024 race."""
        row = chamber_2024.df.query("state_abbr == 'CA' and district == 13").iloc[0]
        assert row["status"] == "contested"


class TestDeterministicCount:
    def test_d_lock_count_reasonable(self, chamber_2024: ChamberView):
        """In modern cycles most districts are formally contested; only the
        deepest-blue districts have no R challenger. Expect roughly 10-80
        d_locks. The number contributes to the chamber MC baseline."""
        n = chamber_2024.deterministic_d_count()
        assert 5 <= n <= 100

    def test_r_lock_count_reasonable(self, chamber_2024: ChamberView):
        n = chamber_2024.n_r_locks
        assert 5 <= n <= 100


class TestProperties:
    def test_n_seats_matches_df(self, chamber_2024: ChamberView):
        assert chamber_2024.n_seats == len(chamber_2024.df)

    def test_cycle_attached(self, chamber_2024: ChamberView):
        assert chamber_2024.cycle == 2024
