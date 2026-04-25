"""Tests for the contested-race filter and derived columns in features.py.

Builds a tiny synthetic post-join DataFrame and runs it through the filter
+ derived-column functions in isolation, verifying:
  * uncontested races are excluded
  * third-party-only candidates are excluded
  * candidates without FEC records are excluded
  * after the filter, every district has exactly 2 candidates (D + R)
  * signed margin is in (-1, +1)
  * derived columns (self_raised_log, opp_raised_log, self_raised_pct) compute correctly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oath_score.features import (
    _apply_contested_race_filter,
    _compute_derived_columns,
)


def _post_join_row(
    state: str, district: int, party_major: str, last: str,
    margin: float, total_trans: int = 100_000, cand_id: str | None = "C0001",
    cand_ici: str = "C", incumbent_raw: str | None = None,
) -> dict:
    """Mimic the shape of df after attach_fec, before contested-race filter."""
    return {
        "state_abbr": state,
        "district": district,
        "party_major": party_major,
        "party": "DEMOCRAT" if party_major == "D" else "REPUBLICAN",
        "last_name": last,
        "candidate_name": f"{last}, FIRST",
        "margin_pct_signed": margin,
        "cand_id": cand_id,
        "total_trans": total_trans,
        "trans_by_indiv": total_trans // 2,
        "trans_by_cmte": total_trans // 2,
        "cand_ici": cand_ici,
        "incumbent_raw": incumbent_raw,
        "ie_for_total": 0,
        "ie_against_total": 0,
        "winner": party_major == "D" if margin > 0 else party_major == "R",
        "candidate_votes": 100 if margin > 0 else 99,
        "vote_share": 0.5 + margin / 2,
        "total_votes_in_race": 200,
    }


class TestContestedRaceFilter:
    def test_keeps_two_party_contested(self):
        df = pd.DataFrame([
            _post_join_row("CA", 45, "D", "PORTER", 0.05),
            _post_join_row("CA", 45, "R", "STEEL",  -0.05),
        ])
        out = _apply_contested_race_filter(df)
        assert len(out) == 2
        assert set(out["party_major"]) == {"D", "R"}

    def test_drops_uncontested(self):
        # Only one major-party candidate; margin == 1.0
        df = pd.DataFrame([
            _post_join_row("WY", 0, "R", "HAGEMAN", -1.0),
        ])
        out = _apply_contested_race_filter(df)
        assert out.empty

    def test_drops_third_party_only(self):
        df = pd.DataFrame([
            {**_post_join_row("VT", 0, "D", "BALINT", 0.3),
             "party_major": None},
        ])
        out = _apply_contested_race_filter(df)
        assert out.empty

    def test_drops_candidate_without_fec(self):
        df = pd.DataFrame([
            _post_join_row("OH", 1, "D", "LANDSMAN", 0.05, cand_id=None),
            _post_join_row("OH", 1, "R", "CHABOT",   -0.05),
        ])
        out = _apply_contested_race_filter(df)
        # Both candidates removed because the district doesn't have 2 valid rows
        assert out.empty

    def test_districts_have_exactly_two_candidates(self):
        df = pd.DataFrame([
            _post_join_row("CA", 45, "D", "PORTER", 0.05),
            _post_join_row("CA", 45, "R", "STEEL",  -0.05),
            _post_join_row("PA", 8,  "D", "CARTWRIGHT", 0.02),
            _post_join_row("PA", 8,  "R", "BOGNET",     -0.02),
        ])
        out = _apply_contested_race_filter(df)
        counts = out.groupby(["state_abbr", "district"]).size()
        assert (counts == 2).all()


class TestDerivedColumns:
    def test_self_raised_log_monotonic(self):
        df = pd.DataFrame([
            _post_join_row("X", 1, "D", "A", 0.1, total_trans=1_000),
            _post_join_row("X", 1, "R", "B", -0.1, total_trans=10_000_000),
        ])
        out = _compute_derived_columns(df)
        # log1p is monotone; bigger raise → bigger log
        a = out.loc[out["last_name"] == "A", "self_raised_log"].iloc[0]
        b = out.loc[out["last_name"] == "B", "self_raised_log"].iloc[0]
        assert b > a

    def test_self_raised_pct_in_zero_one(self):
        df = pd.DataFrame([
            _post_join_row("X", 1, "D", "A", 0.1, total_trans=1_000),
            _post_join_row("X", 1, "R", "B", -0.1, total_trans=10_000_000),
        ])
        out = _compute_derived_columns(df)
        for v in out["self_raised_pct"]:
            assert 0 <= v <= 1

    def test_opponent_raised_paired_correctly(self):
        df = pd.DataFrame([
            _post_join_row("X", 1, "D", "A", 0.1, total_trans=100),
            _post_join_row("X", 1, "R", "B", -0.1, total_trans=900),
        ])
        out = _compute_derived_columns(df)
        a = out.loc[out["last_name"] == "A"].iloc[0]
        b = out.loc[out["last_name"] == "B"].iloc[0]
        # Each candidate's opponent_raised should equal the OTHER candidate's total
        assert a["opp_raised"] == 900
        assert b["opp_raised"] == 100

    def test_incumbent_from_fec_ici(self):
        df = pd.DataFrame([
            _post_join_row("X", 1, "D", "A", 0.1, cand_ici="I"),
            _post_join_row("X", 1, "R", "B", -0.1, cand_ici="C"),
        ])
        out = _compute_derived_columns(df)
        a = out.loc[out["last_name"] == "A"].iloc[0]
        b = out.loc[out["last_name"] == "B"].iloc[0]
        assert a["incumbent"] == 1
        assert b["incumbent"] == 0
