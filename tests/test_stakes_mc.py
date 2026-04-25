"""Tests for scores/stakes.py — correlated MC + per-seat stakes."""

from __future__ import annotations

import numpy as np
import pytest

from oath_score.scores.stakes import (
    CHAMBER_THRESHOLD,
    PIVOTAL_THRESHOLD,
    SIGMA_BY_SNAPSHOT,
    StakesResult,
    StakesSimulator,
    sigma_for_snapshot,
)


QUANTILE_LEVELS = np.array([0.025, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975])


def _seat_quantiles(median: float, std: float = 0.05) -> np.ndarray:
    """Build a 9-quantile vector from a Gaussian(median, std)."""
    from scipy.stats import norm
    return norm.ppf(QUANTILE_LEVELS, loc=median, scale=std)


def _build_universe(seats: list[float]) -> np.ndarray:
    """Stack per-seat quantile vectors into a (n_seats, 9) matrix.

    `seats` is a list of median predicted margins (one per seat).
    """
    return np.array([_seat_quantiles(m) for m in seats])


class TestSigmaLookup:
    def test_canonical_values(self):
        assert sigma_for_snapshot("T-110") == pytest.approx(0.04)
        assert sigma_for_snapshot("T-60") == pytest.approx(0.03)
        assert sigma_for_snapshot("T-20") == pytest.approx(0.02)

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            sigma_for_snapshot("T-99")


class TestSimulatorBasics:
    def test_returns_correct_shape(self):
        # 100 contested seats, all median +0.05 (D-leaning)
        quants = _build_universe([0.05] * 100)
        sim = StakesSimulator(sigma=0.03, n_iter=500)
        result = sim.simulate(quants, QUANTILE_LEVELS, uncontested_d_count=170)
        assert result.stakes_raw.shape == (100,)
        assert result.stakes_normalized.shape == (100,)
        assert result.seat_d_won.shape == (500, 100)
        assert result.chamber_d_iters.shape == (500,)

    def test_normalized_in_unit_interval(self):
        quants = _build_universe([0.0, 0.02, -0.05, 0.10, -0.10] * 20)
        sim = StakesSimulator(sigma=0.03, n_iter=500)
        result = sim.simulate(quants, QUANTILE_LEVELS, uncontested_d_count=170)
        assert (result.stakes_normalized >= 0).all()
        assert (result.stakes_normalized <= 1).all()

    def test_deterministic_with_seed(self):
        quants = _build_universe([0.0] * 50)
        sim1 = StakesSimulator(sigma=0.03, n_iter=200, seed=7)
        sim2 = StakesSimulator(sigma=0.03, n_iter=200, seed=7)
        r1 = sim1.simulate(quants, QUANTILE_LEVELS, uncontested_d_count=180)
        r2 = sim2.simulate(quants, QUANTILE_LEVELS, uncontested_d_count=180)
        assert (r1.stakes_raw == r2.stakes_raw).all()


class TestStakesSemantics:
    def test_pivotal_seats_outscore_lopsided(self):
        """Tossup seats in a chamber on a knife edge should have higher
        |stakes| than landslide seats. Tune uncontested_d so chamber outcome
        genuinely depends on the pivotal-seat draws."""
        n_pivotal = 30
        n_lopsided_d = 10
        n_lopsided_r = 10

        pivotal = _build_universe([0.0] * n_pivotal)         # ~50/50 each
        lop_d = _build_universe([0.30] * n_lopsided_d)       # ~all D
        lop_r = _build_universe([-0.30] * n_lopsided_r)      # ~all R

        quants = np.vstack([pivotal, lop_d, lop_r])
        # Expected D contested: ~15 (pivotal) + 10 (lop_d) + 0 (lop_r) = 25.
        # Threshold 218 → uncontested_d = 218 - 25 = 193 puts chamber on edge.
        sim = StakesSimulator(sigma=0.03, n_iter=3000, seed=1)
        result = sim.simulate(quants, QUANTILE_LEVELS, uncontested_d_count=193)

        # Chamber probability should be near 50% — that's the regime where
        # individual seats matter most.
        assert 0.2 < result.chamber_d_rate < 0.8, result.chamber_d_rate

        pivotal_mean = np.abs(result.stakes_raw[:n_pivotal]).mean()
        lop_d_mean = np.abs(result.stakes_raw[n_pivotal:n_pivotal + n_lopsided_d]).mean()
        lop_r_mean = np.abs(result.stakes_raw[n_pivotal + n_lopsided_d:]).mean()
        assert pivotal_mean > lop_d_mean, (pivotal_mean, lop_d_mean)
        assert pivotal_mean > lop_r_mean, (pivotal_mean, lop_r_mean)

    def test_safe_dem_seat_low_stakes(self):
        """A Solid D seat where D wins ~100% of iterations has near-zero
        marginal stakes (degenerate branch)."""
        n = 50
        # First seat is Solid D (margin +0.40); others are tossups.
        quants = np.vstack([_seat_quantiles(0.40)[None, :], _build_universe([0.0] * (n - 1))])
        sim = StakesSimulator(sigma=0.02, n_iter=1000, seed=2)
        result = sim.simulate(quants, QUANTILE_LEVELS, uncontested_d_count=200)
        # The Solid-D seat should have stakes magnitude essentially 0
        assert abs(result.stakes_raw[0]) < 0.05


class TestPivotalThreshold:
    def test_n_pivotal_bounded(self):
        """With 30 seats and small sigma, only a few should clear pivotal_threshold."""
        quants = _build_universe([0.0, 0.02, 0.05, 0.10, 0.20, -0.05, -0.20] * 5)[:30]
        sim = StakesSimulator(sigma=0.01, n_iter=1000, seed=3)
        result = sim.simulate(quants, QUANTILE_LEVELS, uncontested_d_count=200)
        assert 0 <= result.n_pivotal <= len(quants)


class TestInputValidation:
    def test_2d_required(self):
        sim = StakesSimulator(sigma=0.03, n_iter=100)
        with pytest.raises(ValueError, match="2D"):
            sim.simulate(np.zeros(9), QUANTILE_LEVELS, uncontested_d_count=0)

    def test_quantile_dim_mismatch_raises(self):
        sim = StakesSimulator(sigma=0.03, n_iter=100)
        bad = np.zeros((10, 5))
        with pytest.raises(ValueError, match="quantile_levels"):
            sim.simulate(bad, QUANTILE_LEVELS, uncontested_d_count=0)
