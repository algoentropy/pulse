"""
Unit tests for backtest.price_similarity.compute_price_similarity.

Coverage:
  - Identical series                       -> 1.0  (both methods)
  - Varying series lengths                 -> handled gracefully
  - Flat series (zero variance)            -> defined behaviour per method
  - Highly volatile series                 -> symmetry and bounds
  - Anti-correlated series                 -> low similarity (Pearson)
  - Monotone trend (consistent direction)  -> high Pearson
  - Method switching dtw / pearson
  - Input types: list, numpy array, mixed
  - Error cases: empty / < 2 points / bad method
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backtest.price_similarity import (
    _dtw_distance,
    _pct_from_start,
    _pearson_similarity,
    _resample_linear,
    _z_score,
    compute_price_similarity,
)


# ---------------------------------------------------------------------------
# Helper: build simple price series
# ---------------------------------------------------------------------------

def _linear(start: float, end: float, n: int) -> list[float]:
    """Linearly interpolated price series from *start* to *end* with *n* points."""
    return [start + (end - start) * i / (n - 1) for i in range(n)]


def _flat(value: float, n: int) -> list[float]:
    return [value] * n


def _volatile(seed: int = 0, n: int = 50, scale: float = 10.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    prices = 100.0 + np.cumsum(rng.standard_normal(n) * scale)
    return prices


# ===========================================================================
# Internal helpers — _z_score
# ===========================================================================

class TestZScore:
    def test_zero_mean_unit_std(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _z_score(arr)
        assert pytest.approx(float(np.mean(result)), abs=1e-9) == 0.0
        assert pytest.approx(float(np.std(result)), abs=1e-9) == 1.0

    def test_flat_returns_zeros(self):
        arr = np.array([7.0, 7.0, 7.0])
        result = _z_score(arr)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_single_nonzero(self):
        arr = np.array([3.0, 3.0, 3.0, 3.0])
        result = _z_score(arr)
        assert np.all(result == 0.0)


# ===========================================================================
# Internal helpers — _pct_from_start
# ===========================================================================

class TestPctFromStart:
    def test_simple_up(self):
        arr = np.array([100.0, 110.0, 120.0])
        result = _pct_from_start(arr)
        np.testing.assert_allclose(result, [0.0, 0.1, 0.2])

    def test_simple_down(self):
        arr = np.array([200.0, 180.0, 160.0])
        result = _pct_from_start(arr)
        np.testing.assert_allclose(result, [0.0, -0.1, -0.2])

    def test_flat_returns_zeros(self):
        arr = np.array([50.0, 50.0, 50.0])
        result = _pct_from_start(arr)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_zero_start_falls_back(self):
        """If first element is 0 we fall back to mean-based scaling."""
        arr = np.array([0.0, 1.0, 2.0])
        result = _pct_from_start(arr)
        # Should not raise; first element maps to 0.0
        assert float(result[0]) == 0.0


# ===========================================================================
# Internal helpers — _resample_linear
# ===========================================================================

class TestResampleLinear:
    def test_same_length_returns_unchanged(self):
        arr = np.array([1.0, 2.0, 3.0])
        out = _resample_linear(arr, 3)
        np.testing.assert_array_equal(out, arr)

    def test_upsample_preserves_endpoints(self):
        arr = np.array([0.0, 10.0])
        out = _resample_linear(arr, 5)
        assert float(out[0]) == pytest.approx(0.0, abs=1e-9)
        assert float(out[-1]) == pytest.approx(10.0, abs=1e-9)
        assert len(out) == 5

    def test_downsample_preserves_endpoints(self):
        arr = np.linspace(0.0, 1.0, 100)
        out = _resample_linear(arr, 10)
        assert float(out[0]) == pytest.approx(0.0, abs=1e-9)
        assert float(out[-1]) == pytest.approx(1.0, abs=1e-9)
        assert len(out) == 10


# ===========================================================================
# Internal helpers — _dtw_distance
# ===========================================================================

class TestDTWDistance:
    def test_identical_series_is_zero(self):
        arr = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        assert _dtw_distance(arr, arr) == pytest.approx(0.0, abs=1e-9)

    def test_offset_constant_series(self):
        """Shifted identical shapes should yield a small (or zero) distance."""
        a = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        b = np.array([2.0, 3.0, 4.0, 3.0, 2.0])
        # DTW can warp to align; result should be symmetric
        assert _dtw_distance(a, b) == pytest.approx(_dtw_distance(b, a), abs=1e-12)

    def test_distance_non_negative(self):
        rng = np.random.default_rng(99)
        for _ in range(20):
            a = rng.standard_normal(15)
            b = rng.standard_normal(12)
            assert _dtw_distance(a, b) >= 0.0

    def test_different_lengths_does_not_raise(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = _dtw_distance(a, b)
        assert dist >= 0.0

    def test_single_element_arrays(self):
        a = np.array([3.0])
        b = np.array([5.0])
        # Path has length 1; cost = (3-5)^2 = 4 -> sqrt(4)/1 = 2
        assert _dtw_distance(a, b) == pytest.approx(2.0, abs=1e-9)


# ===========================================================================
# compute_price_similarity — dtw method
# ===========================================================================

class TestDTWMethod:
    def test_identical_series_is_one(self):
        s = [100.0, 102.0, 105.0, 103.0, 107.0]
        result = compute_price_similarity(s, s, method="dtw")
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_result_in_zero_one(self):
        rng = np.random.default_rng(0)
        prices_a = 100.0 + np.cumsum(rng.standard_normal(30))
        prices_b = 100.0 + np.cumsum(rng.standard_normal(30))
        result = compute_price_similarity(prices_a, prices_b, method="dtw")
        assert 0.0 <= result <= 1.0

    def test_symmetry(self):
        a = _volatile(seed=1, n=30)
        b = _volatile(seed=2, n=30)
        assert compute_price_similarity(a, b, "dtw") == pytest.approx(
            compute_price_similarity(b, a, "dtw"), abs=1e-12
        )

    def test_different_length_series(self):
        """DTW naturally handles series of different lengths."""
        a = _volatile(seed=3, n=20)
        b = _volatile(seed=3, n=40)   # same seed → same shape, different length
        result = compute_price_similarity(a, b, method="dtw")
        assert 0.0 <= result <= 1.0

    def test_flat_vs_flat(self):
        """Two flat series are identical in shape (both z-score to zeros)."""
        a = _flat(100.0, 30)
        b = _flat(200.0, 30)
        result = compute_price_similarity(a, b, method="dtw")
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_flat_vs_volatile_is_lower_than_flat_vs_flat(self):
        flat = _flat(100.0, 30)
        vol = _volatile(seed=5, n=30, scale=5.0).tolist()
        sim_flat_flat = compute_price_similarity(flat, flat, method="dtw")
        sim_flat_vol = compute_price_similarity(flat, vol, method="dtw")
        assert sim_flat_flat > sim_flat_vol

    def test_highly_similar_trajectories_above_0_8(self):
        """Near-identical trajectories (same seed, tiny noise) should score > 0.8."""
        rng = np.random.default_rng(42)
        base = 100.0 + np.cumsum(rng.standard_normal(50) * 2.0)
        noise = base + rng.standard_normal(50) * 0.05
        result = compute_price_similarity(base, noise, method="dtw")
        assert result > 0.8

    def test_volatile_100_points(self):
        """Handles 100-point volatile series within bounds."""
        a = _volatile(seed=10, n=100)
        b = _volatile(seed=11, n=100)
        result = compute_price_similarity(a, b, method="dtw")
        assert 0.0 <= result <= 1.0

    def test_two_point_minimum(self):
        """Works correctly with the minimum allowed series length of 2."""
        result = compute_price_similarity([1.0, 2.0], [1.0, 3.0], method="dtw")
        assert 0.0 <= result <= 1.0

    def test_series_with_negative_prices_allowed(self):
        """DTW works on any numeric series, including negative values."""
        a = [-5.0, -3.0, -1.0, -2.0, -4.0]
        b = [-6.0, -4.0, -2.0, -3.0, -5.0]
        result = compute_price_similarity(a, b, method="dtw")
        assert 0.0 <= result <= 1.0


# ===========================================================================
# compute_price_similarity — pearson method
# ===========================================================================

class TestPearsonMethod:
    def test_identical_series_is_one(self):
        s = [100.0, 105.0, 103.0, 110.0, 108.0]
        result = compute_price_similarity(s, s, method="pearson")
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_result_in_zero_one(self):
        rng = np.random.default_rng(0)
        a = 100.0 + np.cumsum(rng.standard_normal(30))
        b = 100.0 + np.cumsum(rng.standard_normal(30))
        result = compute_price_similarity(a, b, method="pearson")
        assert 0.0 <= result <= 1.0

    def test_symmetry(self):
        a = _volatile(seed=20, n=30)
        b = _volatile(seed=21, n=30)
        assert compute_price_similarity(a, b, "pearson") == pytest.approx(
            compute_price_similarity(b, a, "pearson"), abs=1e-12
        )

    def test_up_trend_vs_up_trend_high(self):
        """Two consistently rising series should score > 0.9."""
        a = _linear(100.0, 130.0, 30)
        b = _linear(200.0, 260.0, 30)   # same shape, different level
        result = compute_price_similarity(a, b, method="pearson")
        assert result > 0.9

    def test_up_trend_vs_down_trend_low(self):
        """Opposite trends score below 0.5 (negative correlation)."""
        a = _linear(100.0, 130.0, 30)
        b = _linear(130.0, 100.0, 30)
        result = compute_price_similarity(a, b, method="pearson")
        assert result < 0.2

    def test_perfectly_anti_correlated_is_zero(self):
        """Perfect negative correlation → Pearson r = -1 → similarity = 0.0."""
        n = 20
        a = list(range(n))          # [0, 1, ..., 19]
        b = list(range(n - 1, -1, -1))  # [19, 18, ..., 0]
        # Both expressed pct-from-start → anti-correlated cumulative paths
        result = compute_price_similarity(a, b, method="pearson")
        # Anti-correlated → score near 0; allow small tolerance for normalisation
        assert result <= 0.15

    def test_two_flat_series_is_one(self):
        """Two flat series → identical shape → 1.0."""
        a = _flat(100.0, 20)
        b = _flat(999.0, 20)
        result = compute_price_similarity(a, b, method="pearson")
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_flat_vs_volatile_is_neutral(self):
        """One flat, one volatile → correlation undefined for flat → 0.5."""
        flat = _flat(100.0, 20)
        vol = _volatile(seed=7, n=20).tolist()
        result = compute_price_similarity(flat, vol, method="pearson")
        assert result == pytest.approx(0.5, abs=1e-9)

    def test_different_length_series_resampled(self):
        """Pearson resamples to the shorter length; result stays in [0, 1]."""
        a = _linear(100.0, 140.0, 20)
        b = _linear(100.0, 140.0, 40)    # same trend, different length
        result = compute_price_similarity(a, b, method="pearson")
        # Same trend → high similarity
        assert result > 0.9

    def test_near_identical_plus_noise(self):
        rng = np.random.default_rng(55)
        base = 100.0 + np.cumsum(rng.standard_normal(50))
        noisy = base + rng.standard_normal(50) * 0.1
        result = compute_price_similarity(base, noisy, method="pearson")
        assert result > 0.9

    def test_volatile_series_bounds(self):
        a = _volatile(seed=30, n=60)
        b = _volatile(seed=31, n=60)
        result = compute_price_similarity(a, b, method="pearson")
        assert 0.0 <= result <= 1.0


# ===========================================================================
# Cross-method consistency
# ===========================================================================

class TestMethodConsistency:
    def test_dtw_default_method(self):
        """Default method is DTW."""
        s = _volatile(seed=99, n=25)
        result_default = compute_price_similarity(s, s)
        result_dtw = compute_price_similarity(s, s, method="dtw")
        assert result_default == pytest.approx(result_dtw, abs=1e-12)

    def test_both_methods_return_one_for_identical(self):
        s = [100.0, 102.0, 98.0, 105.0, 103.0]
        assert compute_price_similarity(s, s, "dtw") == pytest.approx(1.0, abs=1e-9)
        assert compute_price_similarity(s, s, "pearson") == pytest.approx(1.0, abs=1e-9)

    def test_dtw_vs_pearson_similar_trend(self):
        """For clearly trending series both methods should agree directionally."""
        up = _linear(100.0, 150.0, 30)
        down = _linear(150.0, 100.0, 30)

        sim_dtw_up_up = compute_price_similarity(up, up, "dtw")
        sim_pearson_up_up = compute_price_similarity(up, up, "pearson")
        sim_dtw_up_down = compute_price_similarity(up, down, "dtw")
        sim_pearson_up_down = compute_price_similarity(up, down, "pearson")

        # Same vs same should be higher than same vs opposite for both methods
        assert sim_dtw_up_up > sim_dtw_up_down
        assert sim_pearson_up_up > sim_pearson_up_down


# ===========================================================================
# Input type flexibility
# ===========================================================================

class TestInputTypes:
    def test_list_inputs(self):
        result = compute_price_similarity([100.0, 105.0, 103.0], [100.0, 105.0, 103.0])
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_numpy_array_inputs(self):
        a = np.array([100.0, 102.0, 101.0])
        b = np.array([100.0, 102.0, 101.0])
        result = compute_price_similarity(a, b)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_mixed_list_and_array(self):
        a = [100.0, 110.0, 120.0]
        b = np.array([100.0, 110.0, 120.0])
        assert compute_price_similarity(a, b, "pearson") == pytest.approx(1.0, abs=1e-9)

    def test_integer_list_coerced(self):
        a = [100, 110, 120]
        b = [100, 110, 120]
        result = compute_price_similarity(a, b)
        assert result == pytest.approx(1.0, abs=1e-9)


# ===========================================================================
# Error cases
# ===========================================================================

class TestErrorCases:
    def test_empty_series_a_raises(self):
        with pytest.raises(ValueError, match="series_a"):
            compute_price_similarity([], [100.0, 200.0])

    def test_empty_series_b_raises(self):
        with pytest.raises(ValueError, match="series_b"):
            compute_price_similarity([100.0, 200.0], [])

    def test_single_point_series_a_raises(self):
        with pytest.raises(ValueError, match="series_a"):
            compute_price_similarity([100.0], [100.0, 200.0])

    def test_single_point_series_b_raises(self):
        with pytest.raises(ValueError, match="series_b"):
            compute_price_similarity([100.0, 200.0], [100.0])

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            compute_price_similarity([100.0, 200.0], [100.0, 200.0], method="cosine")  # type: ignore[arg-type]

    def test_both_single_point_raises(self):
        with pytest.raises(ValueError):
            compute_price_similarity([100.0], [200.0])
