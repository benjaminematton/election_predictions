"""Chamber-control stakes via correlated Monte Carlo.

For each Democratic candidate in a contested race, computes
    stakes_i = P(D controls chamber | D wins seat_i)
            − P(D controls chamber | R wins seat_i)
via a single MC run with 10k iterations, decomposed post-hoc by seat outcome.

Single-MC-run decomposition (per phase5.md decision #2):
  1. Each iteration t draws ε_t ~ Normal(0, σ) — the national environment shift
     applied to every contested race (correlated draws).
  2. For each contested race i, draws u_{t,i} ~ Uniform(0, 1) and looks up
     margin_{t,i} via inverse CDF on the multi-quantile predictions, then
     adds ε_t.
  3. d_won_{t,i} = (margin_{t,i} > 0).
  4. total_d_seats_t = uncontested_d_count + sum_i d_won_{t,i}.
     chamber_d_t = (total_d_seats_t >= chamber_threshold).
  5. Per seat: stakes_i = mean(chamber_d_t | d_won_{t,i})
                          − mean(chamber_d_t | not d_won_{t,i}).

Snapshot-dependent sigma per phase5.md decision #1; cited from 538/Economist
methodology rather than estimated from 4 cycles (too noisy).

The MC is fully vectorized; 10k iter × 320 contested seats runs in <1 sec on
a laptop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Snapshot → national-environment-forecast-error sigma in percentage points.
# These are the published 538/Economist values used by mainstream forecasters,
# not estimated from our 4 training cycles. See phase5.md for rationale.
SIGMA_BY_SNAPSHOT: dict[str, float] = {
    "T-110": 4.0,
    "T-60":  3.0,
    "T-20":  2.0,
}

# House majority threshold (218 of 435).
CHAMBER_THRESHOLD = 218

# A seat is "pivotal" for the secondary metric if its stakes magnitude exceeds
# this threshold. Hard-coded; document in the notebook.
PIVOTAL_THRESHOLD = 0.05

DEFAULT_N_ITER = 10_000
DEFAULT_SEED = 42


@dataclass
class StakesResult:
    stakes_raw: np.ndarray            # per contested race, ∈ [-1, +1]
    stakes_normalized: np.ndarray     # per contested race, ∈ [0, 1]
    chamber_d_rate: float              # P(D controls chamber) overall
    chamber_d_iters: np.ndarray       # bool array, len = n_iter
    seat_d_won: np.ndarray            # (n_iter, n_seats) bool

    @property
    def n_iter(self) -> int:
        return self.chamber_d_iters.shape[0]

    @property
    def n_pivotal(self) -> int:
        """Number of seats with |stakes_raw| > PIVOTAL_THRESHOLD."""
        return int((np.abs(self.stakes_raw) > PIVOTAL_THRESHOLD).sum())


@dataclass
class StakesSimulator:
    sigma: float                              # in margin units (0.04 = 4pp)
    n_iter: int = DEFAULT_N_ITER
    chamber_threshold: int | None = CHAMBER_THRESHOLD
    """If int, literal majority threshold (default 218). If None, use the
    median of the MC chamber-composition draws as the threshold — this makes
    stakes informative when the model predicts the literal threshold is out
    of reach (e.g., 2024 where the model puts D's expected total at ~169,
    far below 218). Treats stakes as 'marginal contribution to relative D
    performance,' which is the donor-relevant question."""
    seed: int = DEFAULT_SEED

    def simulate(
        self,
        contested_quantiles: np.ndarray,      # (n_seats, n_quantiles) signed-margin quantiles
        quantile_levels: np.ndarray,          # (n_quantiles,) — e.g. [0.025, ..., 0.975]
        uncontested_d_count: int,
    ) -> StakesResult:
        """Run the correlated MC and decompose into per-seat stakes.

        ``contested_quantiles[i, q]`` is the predicted signed margin (D% − R%)
        at quantile ``quantile_levels[q]`` for seat i. Quantile predictions
        should already be sorted (the multi-quantile model post-sorts them).
        """
        if contested_quantiles.ndim != 2:
            raise ValueError("contested_quantiles must be 2D (n_seats, n_quantiles)")
        if contested_quantiles.shape[1] != len(quantile_levels):
            raise ValueError("quantile_levels must match contested_quantiles second dim")

        n_seats = contested_quantiles.shape[0]
        rng = np.random.default_rng(self.seed)

        # Draw national-environment shifts; one per iteration, applied to every seat.
        eps = rng.normal(0.0, self.sigma, size=self.n_iter)            # (n_iter,)

        # Draw per-iteration, per-seat uniforms; invert the predictive CDF to get
        # base margin samples, then add the national shift.
        u = rng.uniform(size=(self.n_iter, n_seats))                   # (n_iter, n_seats)
        margins = np.empty_like(u, dtype=float)
        for i in range(n_seats):
            margins[:, i] = np.interp(u[:, i], quantile_levels, contested_quantiles[i, :])
        margins = margins + eps[:, None]

        # Per-iter, per-seat outcome.
        seat_d_won = margins > 0.0                                     # (n_iter, n_seats) bool

        # Per-iteration chamber outcome.
        total_d_per_iter = seat_d_won.sum(axis=1) + uncontested_d_count
        if self.chamber_threshold is None:
            # Use median of the MC distribution as the dynamic threshold —
            # asks "is this iteration above-median for D performance?" rather
            # than the literal 218 which may be unreachable in some cycles.
            effective_threshold = float(np.median(total_d_per_iter))
        else:
            effective_threshold = float(self.chamber_threshold)
        chamber_d_iters = total_d_per_iter >= effective_threshold      # (n_iter,) bool

        # Per-seat conditional probabilities.
        # mean(chamber_d_t | d_won_{t,i}=True) − mean(chamber_d_t | d_won_{t,i}=False)
        stakes_raw = np.zeros(n_seats, dtype=float)
        for i in range(n_seats):
            d_won_i = seat_d_won[:, i]
            n_d = d_won_i.sum()
            n_r = self.n_iter - n_d
            if n_d == 0 or n_r == 0:
                # Degenerate: race outcome essentially fixed → stakes = 0
                stakes_raw[i] = 0.0
                continue
            p_d = chamber_d_iters[d_won_i].mean()
            p_r = chamber_d_iters[~d_won_i].mean()
            stakes_raw[i] = float(p_d - p_r)

        stakes_normalized = _min_max(stakes_raw)

        return StakesResult(
            stakes_raw=stakes_raw,
            stakes_normalized=stakes_normalized,
            chamber_d_rate=float(chamber_d_iters.mean()),
            chamber_d_iters=chamber_d_iters,
            seat_d_won=seat_d_won,
        )


def _min_max(x: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1]. Returns all-0.5 if x is constant."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5, dtype=float)
    return (x - lo) / (hi - lo)


def sigma_for_snapshot(snapshot: str) -> float:
    """Lookup helper: returns sigma in margin units (e.g. 0.03 for T-60)."""
    pp = SIGMA_BY_SNAPSHOT[snapshot]
    return pp / 100.0
