"""Fixed-threshold map from impact_continuous ∈ [0, 1] → integer 1–10.

Per the master plan's "fixed thresholds, not deciles" decision: cutpoints
are calibrated ONCE on the training cycles' continuous impact scores, then
frozen. Applied to test-cycle scores to produce the displayed 1–10 score.

This is what the Streamlit UI shows next to each candidate.

The fixed-not-quantile choice means: in a year with no toss-ups, very few
candidates score 10. Scores are comparable across cycles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


N_BINS = 10  # always 1-10


@dataclass(frozen=True)
class DecileThresholds:
    """9 cutpoints define 10 bins; bin i covers (cutpoints[i-1], cutpoints[i]]."""

    cutpoints: tuple[float, ...]
    cycles_calibrated_on: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.cutpoints) != N_BINS - 1:
            raise ValueError(
                f"DecileThresholds expects {N_BINS - 1} cutpoints, got {len(self.cutpoints)}"
            )
        if list(self.cutpoints) != sorted(self.cutpoints):
            raise ValueError("cutpoints must be sorted ascending")

    def apply(self, scores: pd.Series) -> pd.Series:
        """Assign integer 1..10 per the frozen cutpoints.

        Boundaries:
          score <= cutpoints[0]      -> 1
          cutpoints[0] < score <= cutpoints[1] -> 2
          ...
          score > cutpoints[-1]      -> 10
        """
        s = pd.to_numeric(scores, errors="coerce").fillna(0.0)
        # np.searchsorted(side="right") gives index where score would insert,
        # which is the bin index (0..N_BINS-1); add 1 for 1-indexed.
        bins = np.searchsorted(np.array(self.cutpoints), s.to_numpy(), side="right") + 1
        bins = np.clip(bins, 1, N_BINS)
        return pd.Series(bins, index=scores.index, name="impact_decile").astype(int)

    def to_dict(self) -> dict:
        return {
            "cutpoints": list(self.cutpoints),
            "cycles_calibrated_on": list(self.cycles_calibrated_on),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DecileThresholds":
        return cls(
            cutpoints=tuple(d["cutpoints"]),
            cycles_calibrated_on=tuple(d["cycles_calibrated_on"]),
        )

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "DecileThresholds":
        return cls.from_dict(json.loads(Path(path).read_text()))


def calibrate(
    scores: pd.Series,
    *,
    cycles_calibrated_on: tuple[int, ...],
) -> DecileThresholds:
    """Compute 9 percentile-based cutpoints (10/20/.../90) from training scores."""
    s = pd.to_numeric(scores, errors="coerce").dropna()
    if s.empty:
        raise ValueError("calibrate() requires at least one non-NaN training score")
    # Use percentiles; sklearn's quantile uses linear interpolation by default
    cutpoints = tuple(
        float(np.percentile(s, q)) for q in (10, 20, 30, 40, 50, 60, 70, 80, 90)
    )
    # If many scores are tied (e.g. 0), cutpoints may not be strictly increasing.
    # Bump duplicates by an epsilon so the dataclass invariant holds.
    fixed = list(cutpoints)
    for i in range(1, len(fixed)):
        if fixed[i] <= fixed[i - 1]:
            fixed[i] = fixed[i - 1] + 1e-9
    return DecileThresholds(
        cutpoints=tuple(fixed),
        cycles_calibrated_on=cycles_calibrated_on,
    )
