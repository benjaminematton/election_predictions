"""Combine sub-scores into the v1 Impact Score.

Per phase5.md decision #6: ``base = sqrt(competitiveness × stakes)``.
Phase 6 will add the additive financial-need adjustment on top of base.

Both inputs are expected in [0, 1]; the geometric mean penalizes low values
on either dimension. A safe-D incumbent in a chamber-pivotal-but-uncompetitive
race scores low; a Tossup in a non-pivotal race scores low; a Tossup that
decides chamber control scores high.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def combine_scores(competitiveness: pd.Series, stakes: pd.Series) -> pd.Series:
    """``base = sqrt(competitiveness * stakes)`` element-wise.

    Both inputs must share an index. NaN in either input → 0 in output (a
    candidate without one of the signals is conservatively scored as the
    Oath product would: don't recommend it).

    Output is clipped to [0, 1] for safety even though the inputs should
    already be there.
    """
    if not competitiveness.index.equals(stakes.index):
        raise ValueError("competitiveness and stakes must share an index")
    c = pd.to_numeric(competitiveness, errors="coerce").clip(lower=0.0, upper=1.0).fillna(0.0)
    s = pd.to_numeric(stakes, errors="coerce").clip(lower=0.0, upper=1.0).fillna(0.0)
    base = np.sqrt(c * s)
    return base.clip(lower=0.0, upper=1.0).rename("score_impact_base")
