"""Financial-need sub-score: viable-floor quantile regression.

Per phase 6 of the master plan. Trains a 25th-percentile QuantileRegressor on
the winning side of historical close races (|margin| <= 10%) to predict the
viable spending floor for a contested race. Then for any contested-race
candidate, computes:

    need_raw = clip((viable_floor - own_snapshot_spend) / viable_floor, 0, 1)

Range [0, 1]. 1 = candidate has nothing relative to the floor; 0 = at/above.

The 25th-percentile-of-winners reading: "below this spend, even winners
struggle." It's the Oath-relevant 'minimum to be viable,' not a 'typical
spend' (which would be confounded by overspending and survivorship bias).

Target-leak discipline:
  * Training filter uses `winner` and `margin_pct` — fine, we have those at
    train time.
  * Prediction NEVER touches `winner` or `margin_pct`. The features are
    cycle-static / snapshot-deterministic.

Three features, all already in features.py output:
  * opp_raised_log   — opponent's snapshot fundraising (need to match)
  * ie_against_log   — outside spending against (need to counter)
  * acs_median_income — district cost-of-living proxy

(The master plan also called for district_VAP / media-market index, but
those aren't in our parquets; median income is a v1 substitute.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCORE_COL = "score_need"

# Cushion against floor going to zero or near-zero, which would make
# need_raw blow up. Floor predictions below this get clipped up.
MIN_FLOOR = 1.0


@dataclass
class FinancialNeed:
    """25th-percentile viable-floor quantile regression.

    Trains on D winners of close races; predicts a viable_floor for any
    contested-race candidate; derives need_raw from each candidate's own
    snapshot spend relative to the floor.
    """

    quantile: float = 0.25
    feature_cols: tuple[str, ...] = field(
        default_factory=lambda: ("opp_raised_log", "ie_against_log", "acs_median_income")
    )
    margin_threshold: float = 0.10
    alpha: float = 0.1  # L1 strength on QuantileRegressor
    min_train_rows: int = 30

    def __post_init__(self) -> None:
        self._pipeline: Pipeline | None = None

    # ----- training -----

    def fit(self, train_df: pd.DataFrame) -> "FinancialNeed":
        """Fit on D winners of close races.

        Filter: party_major == 'D' AND winner == True AND |margin_pct| <= margin_threshold.
        Target: total_trans (raw dollars at the snapshot).
        Pipeline: StandardScaler -> QuantileRegressor(quantile=self.quantile, alpha=self.alpha).
        """
        d = train_df.loc[
            (train_df["party_major"] == "D")
            & train_df["winner"].astype(bool)
            & (train_df["margin_pct"].abs() <= self.margin_threshold)
        ].copy()

        if len(d) < self.min_train_rows:
            raise ValueError(
                f"Too few training rows for FinancialNeed ({len(d)} < {self.min_train_rows}). "
                f"Try lowering `margin_threshold` or `quantile` to relax the filter."
            )

        X = self._featurize(d)
        y = pd.to_numeric(d["total_trans"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy()

        # The target distribution is heavy-tailed in dollars; QuantileRegressor
        # is fine with that, but we report the median for the diagnostic.
        self._pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("qreg", QuantileRegressor(quantile=self.quantile, alpha=self.alpha, solver="highs")),
        ])
        self._pipeline.fit(X, y)
        return self

    # ----- prediction -----

    def predict_floor(self, df: pd.DataFrame) -> pd.Series:
        """Predicted 25th-percentile-of-winners viable spend per row.

        Always positive (clipped at MIN_FLOOR for numerical safety in
        downstream division).
        """
        if self._pipeline is None:
            raise RuntimeError("call .fit() first")
        X = self._featurize(df)
        floor = self._pipeline.predict(X)
        floor = np.clip(floor, MIN_FLOOR, None)
        return pd.Series(floor, index=df.index, name="viable_floor")

    def predict_need(
        self,
        df: pd.DataFrame,
        *,
        own_spend_col: str = "total_trans",
    ) -> pd.Series:
        """need_raw = clip((floor - own_spend) / floor, 0, 1).

        Range [0, 1]. NaN own_spend -> 1.0 (most-needy assumption).
        """
        if self._pipeline is None:
            raise RuntimeError("call .fit() first")
        floor = self.predict_floor(df)
        own = pd.to_numeric(df[own_spend_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        need = (floor - own) / floor
        need = need.clip(lower=0.0, upper=1.0)
        return need.rename(SCORE_COL)

    # ----- inspection -----

    @property
    def is_fitted(self) -> bool:
        return self._pipeline is not None

    @property
    def coef_(self) -> dict[str, float]:
        """Standardized coefficients keyed by feature name."""
        if self._pipeline is None:
            return {}
        return dict(zip(self.feature_cols, self._pipeline.named_steps["qreg"].coef_))

    # ----- internals -----

    def _featurize(self, df: pd.DataFrame) -> np.ndarray:
        cols = list(self.feature_cols)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"FinancialNeed features missing from input: {missing}. "
                f"Available columns include: {list(df.columns)[:15]}..."
            )
        X = df[cols].astype(float).copy()
        X = X.fillna(0.0)
        return X.to_numpy()
