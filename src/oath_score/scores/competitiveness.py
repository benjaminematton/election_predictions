"""Competitiveness models for the improvement curve.

Two model classes share the same fit/predict_proba/score interface:

  * ``LogisticCompetitiveness`` — logistic regression on a feature_set.
    Targets ``|margin_pct| < 0.05`` (per-race close-race indicator).
    Used for the first improvement curve.

  * ``MultiQuantileCompetitiveness`` (separate module) — fits 9 quantile
    regressors on signed margin and derives close-race probability from
    the empirical CDF. Used for the second improvement curve.

Universe handling:
  * For the ``naive`` set (cook + incumbent only), score 0 if cook_rating is
    NaN — the model can't say anything useful without it.
  * For richer sets that include cpvi, demographics, fundraising, etc.,
    impute NaN cook from PVI sign (target-free) and score every Dem.

The cook-distance-from-Tossup transform is naive-only — richer feature sets
have enough capacity to learn the U-shape from interactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from oath_score.feature_sets import get as get_feature_set
from oath_score.scores._imputation import (
    fill_remaining_with_tossup,
    impute_cook_from_pvi,
)

SCORE_COL = "score_competitiveness"
TARGET_COL = "_target_close_race"

# Feature sets where we should NOT use the cook-distance transform — the
# extra features give the model enough capacity to learn cook_rating's
# U-shape directly via interactions.
_NAIVE_ONLY_SETS = frozenset({"naive"})


def impute_cook_rating(df: pd.DataFrame) -> pd.Series:
    """LEGACY (Phase 3) imputation that uses the test-cycle margin.

    Kept for the regression test that asserts it works as documented; not
    used by any current model. Phase 4 models call ``impute_cook_from_pvi``
    instead, which doesn't leak the target.
    """
    out = df["cook_rating"].copy().astype(float)
    needs_impute = out.isna()
    if not needs_impute.any():
        return out

    party_major = df["party_major"].astype(str)
    incumbent = df["incumbent"].fillna(0).astype(int)
    margin = pd.to_numeric(df["margin_pct"], errors="coerce")

    inc_d = needs_impute & (incumbent == 1) & (party_major == "D")
    inc_r = needs_impute & (incumbent == 1) & (party_major == "R")
    out.loc[inc_d] = 7.0
    out.loc[inc_r] = 1.0

    open_seat = needs_impute & (incumbent == 0)
    out.loc[open_seat & (margin > 0.10)] = 7.0
    out.loc[open_seat & (margin > 0.0) & (margin <= 0.10)] = 6.0
    out.loc[open_seat & (margin < 0.0) & (margin >= -0.10)] = 2.0
    out.loc[open_seat & (margin < -0.10)] = 1.0

    return out.fillna(4.0)


@dataclass
class LogisticCompetitiveness:
    """Logistic regression on any feature_set, target = ``|margin| < 0.05``.

    Args:
        feature_set_name: matches a key in ``feature_sets.REGISTRY``.
        C: inverse regularization strength (sklearn default 1.0).
        class_weight: pass through to sklearn; default None (the close-race
            target is balanced enough that weighting hurts more than helps).
    """

    feature_set_name: str = "naive"
    C: float = 1.0
    class_weight: str | dict | None = None

    def __post_init__(self) -> None:
        self._model: LogisticRegression | None = None
        self._feature_cols: tuple[str, ...] = get_feature_set(self.feature_set_name).columns

    @property
    def is_naive(self) -> bool:
        return self.feature_set_name in _NAIVE_ONLY_SETS

    # ----- training -----

    def fit(self, train_df: pd.DataFrame) -> "LogisticCompetitiveness":
        """Fit on Dem candidates from the training cycles."""
        d = self._scorable_dems(train_df).copy()
        if d.empty:
            raise ValueError(
                f"No scorable Democratic candidates for feature_set={self.feature_set_name!r}"
            )

        X = self._featurize(d)
        y = (d["margin_pct"].abs() < 0.05).astype(int).to_numpy()

        if y.sum() < 5:
            raise ValueError(
                f"Too few positive (close-race) examples ({int(y.sum())}); "
                "model would not learn anything useful."
            )

        # StandardScaler is essential when richer feature sets mix scales —
        # ACS median_income (~60k), self_raised_log (~14), incumbent (0/1).
        # Without it L2-regularized logistic puts almost all the weight on
        # the large-magnitude features regardless of signal.
        self._model = Pipeline([
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(
                C=self.C,
                class_weight=self.class_weight,
                max_iter=2000,
                solver="lbfgs",
            )),
        ])
        self._model.fit(X, y)
        return self

    # ----- prediction -----

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """Score every row of df. Score 0 for non-Dems and rows we can't featurize."""
        if self._model is None:
            raise RuntimeError("call .fit() first")

        scorable_mask = self._scorable_mask(df)
        out = pd.Series(0.0, index=df.index, name=SCORE_COL)
        if not scorable_mask.any():
            return out

        d = df.loc[scorable_mask].copy()
        X = self._featurize(d)
        proba = self._model.predict_proba(X)[:, 1]
        out.loc[scorable_mask] = proba
        return out

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: attach SCORE_COL to a copy of df."""
        out = df.copy()
        out[SCORE_COL] = self.predict_proba(df)
        return out

    # ----- universe + featurize -----

    def _scorable_mask(self, df: pd.DataFrame) -> pd.Series:
        """Boolean mask of rows the model can score.

        For the naive set (cook + incumbent), require non-NaN cook_rating
        — without ratings the model has no signal. For richer sets, rely on
        PVI-based imputation in _featurize and score every Dem.
        """
        is_dem = df["party_major"] == "D"
        if self.is_naive:
            return is_dem & df["cook_rating"].notna()
        return is_dem

    def _scorable_dems(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[self._scorable_mask(df)]

    def _featurize(self, d: pd.DataFrame) -> np.ndarray:
        """Transform raw feature columns into the design matrix.

        For naive set: apply cook-distance-from-Tossup transform to get the
        U-shape onto a monotonic axis logistic can fit.

        For richer sets: PVI-impute NaN cook_rating, then pass features
        through. Numeric NaNs in any feature column → 0 (median-ish for
        scaled features; logistic is robust to this).
        """
        cols = list(self._feature_cols)
        X = d[cols].astype(float).copy()

        if "cook_rating" in X.columns:
            if self.is_naive:
                X["cook_rating"] = -(X["cook_rating"] - 4.0).abs()
            else:
                # Target-free PVI imputation, then any leftover NaNs to Tossup
                imputed = impute_cook_from_pvi(d)
                X["cook_rating"] = fill_remaining_with_tossup(imputed).astype(float).values

        # Any remaining NaNs in other columns: median imputation done lazily
        # via fillna(0). For demographics and fundraising features this is
        # crude but safe given everything is log-scaled or proportions.
        X = X.fillna(0.0)
        return X.to_numpy()

    @property
    def transformed_feature_names(self) -> tuple[str, ...]:
        """Names of the columns the model actually saw, after _featurize."""
        if self.is_naive:
            return tuple(
                "cook_distance_from_tossup_neg" if c == "cook_rating" else c
                for c in self._feature_cols
            )
        return tuple(
            "cook_rating_imputed" if c == "cook_rating" else c
            for c in self._feature_cols
        )

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self._feature_cols

    @property
    def coef_(self) -> dict[str, float]:
        """Standardized coefficients keyed by transformed feature names.

        Because we wrap LogisticRegression in a StandardScaler pipeline, these
        are coefficients on z-scored features — comparable across columns by
        magnitude. Original-scale coefficients aren't recovered (would need
        the per-feature mean/std from the scaler step).
        """
        if self._model is None:
            return {}
        # _model is now a Pipeline; the LogisticRegression step is named "logreg"
        coef = self._model.named_steps["logreg"].coef_[0]
        return dict(zip(self.transformed_feature_names, coef))


# Backwards-compat alias for one commit window. To be removed after Phase 4.
NaiveCompetitiveness = LogisticCompetitiveness
