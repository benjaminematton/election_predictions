"""Target-free imputation helpers for the competitiveness models.

Per phase4.md decision #2: when a candidate's race isn't in the Wikipedia
ratings table (NaN cook_rating) but Cook PVI *is* present, impute cook_rating
from PVI sign instead of from the candidate's own margin. This avoids the
target-leak problem the Phase 3 imputation had.

PVI is a function of past presidential votes in the district — fixed for the
cycle, not derived from the test-cycle outcome. So it's safe to use as a
feature input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def impute_cook_from_pvi(df: pd.DataFrame) -> pd.Series:
    """Fill NaN cook_rating from cpvi sign. No target leakage.

    Mapping (PVI in cycle's units, e.g. D+5 = +5):
        cpvi >= +10  -> 7  (Solid D)
        cpvi >= +3   -> 6  (Likely D)
        cpvi >= -3   -> 4  (Tossup territory)
        cpvi >= -10  -> 2  (Likely R)
        cpvi <  -10  -> 1  (Solid R)

    Where cpvi is also missing, leave cook_rating as NaN (downstream code
    decides — usually fills with 4 = Tossup midpoint or drops the row).

    Returns:
        Series same length as df with cook_rating imputed where possible.
    """
    out = df["cook_rating"].copy().astype(float)
    needs_impute = out.isna()
    if not needs_impute.any():
        return out

    cpvi = pd.to_numeric(df.get("cpvi"), errors="coerce")
    have_pvi = needs_impute & cpvi.notna()
    if not have_pvi.any():
        return out

    p = cpvi[have_pvi]
    imputed = pd.Series(4.0, index=p.index)  # Tossup default
    imputed.loc[p >= 10] = 7.0
    imputed.loc[(p >= 3) & (p < 10)] = 6.0
    imputed.loc[(p > -3) & (p < 3)] = 4.0
    imputed.loc[(p > -10) & (p <= -3)] = 2.0
    imputed.loc[p <= -10] = 1.0

    out.loc[have_pvi] = imputed.values
    return out


def fill_remaining_with_tossup(s: pd.Series) -> pd.Series:
    """After PVI imputation, any still-NaN rows go to Tossup midpoint (4.0)."""
    return s.fillna(4.0)
