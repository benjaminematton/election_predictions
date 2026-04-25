"""Per-Dem-candidate competitiveness score.

Phase 3 ("naive baseline") version: logistic regression on cook_rating +
incumbent. Target = "this Dem candidate won AND |margin_pct| < 5%".

Includes Cook-rating imputation for non-competitive districts (Wikipedia
ratings table only covers ~94 races/cycle; the other ~230 contested
districts have NaN cook_rating).

Future versions (Phase 4) swap the model for multi-quantile regression on
signed margin and add the broader feature set via feature_sets.py. The
public interface (.fit / .predict_proba / SCORE_COL) stays stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from oath_score.feature_sets import get as get_feature_set

SCORE_COL = "score_competitiveness"
TARGET_COL = "_target_close_d_win"


def impute_cook_rating(df: pd.DataFrame) -> pd.Series:
    """Fill NaN cook_rating from incumbent / past margin.

    Rule (per phase3.md, decision #1):
      * If `incumbent == 1`, use Solid for the incumbent's party:
          - D incumbent → 7.0
          - R incumbent → 1.0
      * Otherwise, infer Solid/Likely from the actual margin sign:
          - margin >  0.10 → 7.0  (Solid D)
          -  0.00 < margin < 0.10 → 6.0  (Likely D)
          - -0.10 < margin < 0.00 → 2.0  (Likely R)
          - margin < -0.10 → 1.0  (Solid R)

    NOTE: imputing from test-cycle margin technically peeks at the target.
    Mitigated because we ONLY impute the non-competitive districts (Wikipedia
    didn't bother tracking them — the outcome is essentially deterministic).
    Phase 4 will swap to PVI-based imputation once Daily Kos is restored.
    """
    out = df["cook_rating"].copy().astype(float)
    needs_impute = out.isna()
    if not needs_impute.any():
        return out

    party_major = df["party_major"].astype(str)
    incumbent = df["incumbent"].fillna(0).astype(int)
    margin = pd.to_numeric(df["margin_pct"], errors="coerce")

    # Incumbent fallback
    inc_d = needs_impute & (incumbent == 1) & (party_major == "D")
    inc_r = needs_impute & (incumbent == 1) & (party_major == "R")
    out.loc[inc_d] = 7.0
    out.loc[inc_r] = 1.0

    # Open-seat fallback by margin sign / magnitude
    open_seat = needs_impute & (incumbent == 0)
    out.loc[open_seat & (margin > 0.10)] = 7.0
    out.loc[open_seat & (margin > 0.0) & (margin <= 0.10)] = 6.0
    out.loc[open_seat & (margin < 0.0) & (margin >= -0.10)] = 2.0
    out.loc[open_seat & (margin < -0.10)] = 1.0

    # Anything still NaN (no margin data) → toss-up midpoint
    out = out.fillna(4.0)
    return out


@dataclass
class NaiveCompetitiveness:
    """Phase 3 baseline: logistic regression on (cook_rating, incumbent).

    Targets `|margin_pct| < 0.05` — i.e., 'this race finished within 5%'.
    That's a per-race quantity (the v3 plan's `P(margin<5%)`), shared by both
    candidates in the race. The score for each Dem candidate is then 'how
    likely is this race to be close.'

    Why not 'close-D-win'? Logistic regression with one ordinal feature is
    monotonic. close-D-win actually peaks at cook=4-5 and falls off at cook=7
    (Solid D districts mostly produce blowouts, not close wins) — a non-
    monotonic shape the model can't fit. The 'close race' target is closer to
    monotonic in cook_rating and gives the model a signal it can actually use.
    """

    feature_set_name: str = "naive"
    C: float = 1.0
    class_weight: str | dict | None = None

    def __post_init__(self) -> None:
        self._model: LogisticRegression | None = None
        self._feature_cols: tuple[str, ...] = get_feature_set(self.feature_set_name).columns

    # ----- training -----

    def fit(self, train_df: pd.DataFrame) -> "NaiveCompetitiveness":
        """Fit on Dem candidates whose race is in the Wikipedia ratings table.

        Why we filter to non-NaN cook_rating instead of imputing:
          The earlier impute_cook_rating helper used the candidate's *margin*
          to backfill missing cook ratings. That leaks the target into the
          training feature, and at test time it leaks the test-cycle outcome
          into the score. Restricting to Wikipedia-tracked races (the only
          ones a competitive donor actually agonizes over) avoids the leak
          entirely. impute_cook_rating remains in this file for Phase 4 when
          we restore Daily Kos PVI as a target-free imputation source.

        Why we transform cook_rating to distance-from-Tossup:
          Close-race rate peaks at cook=3–5 (Tossup territory) and falls off
          at cook=1 or 7 (Solid R/D landslides). Logistic on the raw ordinal
          can only fit a monotonic slope and ends up ranking Solid-D blowouts
          above Tossups. -abs(cook_rating - 4) collapses the U-shape onto a
          monotonic axis the model can use.
        """
        d = train_df.loc[
            (train_df["party_major"] == "D") & train_df["cook_rating"].notna()
        ].copy()
        if d.empty:
            raise ValueError(
                "No Democratic candidates with cook_rating in training frame "
                "(Wikipedia ratings table empty for this cycle?)"
            )

        X = self._featurize(d)
        y = (d["margin_pct"].abs() < 0.05).astype(int).to_numpy()

        if y.sum() < 5:
            raise ValueError(
                f"Too few positive (close-race) examples ({int(y.sum())}); "
                "model would not learn anything useful."
            )

        self._model = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=1000,
            solver="lbfgs",
        )
        self._model.fit(X, y)
        return self

    # ----- prediction -----

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """Score every row of df. Score 0 for non-Dems and for Dems whose
        race isn't in the Wikipedia ratings table (Oath doesn't recommend
        races a competitive-donor wouldn't be agonizing over).
        """
        if self._model is None:
            raise RuntimeError("call .fit() first")

        scorable = (df["party_major"] == "D") & df["cook_rating"].notna()
        out = pd.Series(0.0, index=df.index, name=SCORE_COL)
        if not scorable.any():
            return out

        d = df.loc[scorable].copy()
        X = self._featurize(d)
        proba = self._model.predict_proba(X)[:, 1]
        out.loc[scorable] = proba
        return out

    def _featurize(self, d: pd.DataFrame) -> np.ndarray:
        """Transform raw feature columns into the design matrix.

        Currently transforms cook_rating to its distance from Tossup (cook=4)
        so logistic regression can capture the U-shaped close-race signal.
        Other naive features pass through untouched.
        """
        cols = list(self._feature_cols)
        X = d[cols].astype(float).copy()
        if "cook_rating" in X.columns:
            X["cook_rating"] = -(X["cook_rating"] - 4.0).abs()
        return X.to_numpy()

    @property
    def transformed_feature_names(self) -> tuple[str, ...]:
        """Names of the columns the model actually saw, after _featurize."""
        return tuple(
            "cook_distance_from_tossup_neg" if c == "cook_rating" else c
            for c in self._feature_cols
        )

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: attach SCORE_COL to a copy of df."""
        out = df.copy()
        out[SCORE_COL] = self.predict_proba(df)
        return out

    # ----- inspection -----

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self._feature_cols

    @property
    def coef_(self) -> dict[str, float]:
        """Coefficients keyed by the *transformed* feature names the model saw.

        Specifically, `cook_rating` becomes `cook_distance_from_tossup_neg`
        so consumers don't get misled into thinking +1.0 means 'higher cook
        ordinal → higher score' when it actually means 'closer to Tossup → higher score'.
        """
        if self._model is None:
            return {}
        return dict(zip(self.transformed_feature_names, self._model.coef_[0]))
