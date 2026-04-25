"""Multi-quantile regression on signed margin → close-race probability.

Per phase4.md: instead of fitting a binary logistic on "is the race close?",
fit nine independent QuantileRegressor models on the *continuous* signed
margin (D% − R%). Each model produces a quantile of the predictive
distribution. Together they form an empirical predictive CDF per candidate;
P(|margin| < 0.05) = CDF(+0.05) − CDF(−0.05).

Key design choices:
  * Quantile crossings (q=0.5 below q=0.25 for some inputs) are post-hoc
    repaired by isotonic projection of each row's quantile vector.
  * Same scorable-mask + featurize logic as LogisticCompetitiveness — the
    naive set requires non-NaN cook_rating, richer sets impute from PVI.
  * On small-sample feature sets (especially the wikipedia-restricted
    universe with ~80 training rows), 9 quantile fits per model are
    data-thin. Mitigation: each QuantileRegressor uses ``alpha=0.1`` (mild
    L1) and ``solver_options={'sparse': True}``.

Public interface mirrors LogisticCompetitiveness:
  * ``.fit(train_df)``       — returns self
  * ``.predict_proba(df)``   — returns SCORE_COL series of P(|margin|<0.05)
  * ``.score(df)``           — copy of df with SCORE_COL attached
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

from oath_score.scores._imputation import (
    fill_remaining_with_tossup,
    impute_cook_from_pvi,
)
from oath_score.scores.competitiveness import SCORE_COL, _NAIVE_ONLY_SETS
from oath_score.feature_sets import get as get_feature_set

# Quantiles to fit. The CDF window for "close race" is [+0.05, −0.05], so
# 0.025 / 0.05 / 0.10 cover the tails near the close-race threshold.
QUANTILES: tuple[float, ...] = (0.025, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975)

CLOSE_THRESHOLD = 0.05
ALPHA = 0.1  # L1 strength on each QuantileRegressor


@dataclass
class MultiQuantileCompetitiveness:
    """9-quantile regression on signed margin → close-race probability."""

    feature_set_name: str = "naive"
    alpha: float = ALPHA
    quantiles: tuple[float, ...] = QUANTILES

    def __post_init__(self) -> None:
        self._models: list[QuantileRegressor] = []
        self._scaler: StandardScaler | None = None
        self._feature_cols: tuple[str, ...] = get_feature_set(self.feature_set_name).columns
        self._crossing_rate: float = 0.0  # last-call diagnostic

    @property
    def is_naive(self) -> bool:
        return self.feature_set_name in _NAIVE_ONLY_SETS

    # ----- training -----

    def fit(self, train_df: pd.DataFrame) -> "MultiQuantileCompetitiveness":
        d = self._scorable_dems(train_df).copy()
        if d.empty:
            raise ValueError(
                f"No scorable Democratic candidates for feature_set={self.feature_set_name!r}"
            )
        if len(d) < max(20, 2 * len(self._feature_cols)):
            raise ValueError(
                f"Training set too small ({len(d)} rows) for {len(self.quantiles)} "
                "quantile fits."
            )

        X = self._featurize(d)
        y = d["margin_pct"].astype(float).to_numpy()

        # Scale once; reuse for every quantile fit and at predict time.
        self._scaler = StandardScaler().fit(X)
        X_scaled = self._scaler.transform(X)

        self._models = []
        for q in self.quantiles:
            m = QuantileRegressor(
                quantile=q,
                alpha=self.alpha,
                solver="highs",  # default; explicit for clarity
            )
            m.fit(X_scaled, y)
            self._models.append(m)
        return self

    # ----- prediction -----

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        if not self._models:
            raise RuntimeError("call .fit() first")

        scorable = self._scorable_mask(df)
        out = pd.Series(0.0, index=df.index, name=SCORE_COL)
        if not scorable.any():
            return out

        d = df.loc[scorable].copy()
        X = self._featurize(d)
        X_scaled = self._scaler.transform(X) if self._scaler is not None else X

        # Predict each quantile, stack into (n_rows, n_quantiles)
        preds = np.column_stack([m.predict(X_scaled) for m in self._models])

        # Track crossings BEFORE we fix them — for the diagnostic notebook
        self._crossing_rate = float(
            ((preds[:, 1:] - preds[:, :-1]) < 0).any(axis=1).mean()
        )

        # Repair crossings: per-row monotone projection (isotonic).
        # For each row, sort the predicted quantile vector and zip it back
        # to the requested quantile order. This is the simplest valid fix
        # and is what most quantile-regression libraries do internally.
        preds_fixed = np.sort(preds, axis=1)

        # P(|margin| < 0.05) = CDF(+0.05) − CDF(−0.05).
        # We have inverse-CDF values (margin at each quantile), so we need to
        # invert: for each row, find what quantile +0.05 corresponds to and
        # similarly for −0.05; the difference is the close-race probability.
        proba = np.empty(preds_fixed.shape[0], dtype=float)
        q_arr = np.array(self.quantiles)
        for i, row in enumerate(preds_fixed):
            cdf_upper = np.interp(+CLOSE_THRESHOLD, row, q_arr,
                                  left=0.0, right=1.0)
            cdf_lower = np.interp(-CLOSE_THRESHOLD, row, q_arr,
                                  left=0.0, right=1.0)
            proba[i] = max(0.0, cdf_upper - cdf_lower)

        out.loc[scorable] = proba
        return out

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[SCORE_COL] = self.predict_proba(df)
        return out

    # ----- universe + featurize (mirror LogisticCompetitiveness) -----

    def _scorable_mask(self, df: pd.DataFrame) -> pd.Series:
        is_dem = df["party_major"] == "D"
        if self.is_naive:
            return is_dem & df["cook_rating"].notna()
        return is_dem

    def _scorable_dems(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[self._scorable_mask(df)]

    def _featurize(self, d: pd.DataFrame) -> np.ndarray:
        cols = list(self._feature_cols)
        X = d[cols].astype(float).copy()

        if "cook_rating" in X.columns:
            if self.is_naive:
                # Naive set: distance-from-Tossup transform, same as logistic.
                X["cook_rating"] = -(X["cook_rating"] - 4.0).abs()
            else:
                imputed = impute_cook_from_pvi(d)
                X["cook_rating"] = fill_remaining_with_tossup(imputed).astype(float).values

        X = X.fillna(0.0)
        return X.to_numpy()

    # ----- diagnostics + raw predictions -----

    @property
    def crossing_rate(self) -> float:
        """Fraction of rows in the most-recent predict_proba call where any
        adjacent pair of quantile predictions was out of order. 0 = perfectly
        monotone; >0.05 = worth investigating before trusting the model."""
        return self._crossing_rate

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self._feature_cols

    def predict_quantiles(self, df: pd.DataFrame) -> np.ndarray:
        """Return the full per-row quantile-prediction matrix.

        Shape: (len(df), len(self.quantiles)). Rows that aren't in the
        scorable universe (non-Dem, or Dem without cook for the naive set)
        get a row of NaN. Quantiles are post-sorted to repair crossings.

        Stakes MC consumes these directly via inverse-CDF sampling.
        """
        if not self._models:
            raise RuntimeError("call .fit() first")

        out = np.full((len(df), len(self.quantiles)), np.nan, dtype=float)
        scorable = self._scorable_mask(df)
        if not scorable.any():
            return out

        d = df.loc[scorable].copy()
        X = self._featurize(d)
        X_scaled = self._scaler.transform(X) if self._scaler is not None else X
        preds = np.column_stack([m.predict(X_scaled) for m in self._models])
        preds_fixed = np.sort(preds, axis=1)
        out[scorable.to_numpy(), :] = preds_fixed
        return out
