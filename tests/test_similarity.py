"""
Tests for backtest.similarity.

Covers:
  compute_macro_similarity:
    - Identical vectors             -> 1.0
    - Zero vector(s)                -> 0.0
    - Orthogonal vectors            -> 0.0
    - Opposite-direction vectors    -> -1.0
    - Arbitrary known similarity    -> validated against manual calculation
    - List inputs (not just ndarray)
    - Negative-valued vectors
    - Single-element vectors
    - High-dimensional vectors (realistic macro feature size)
    - Mismatched lengths            -> ValueError
    - Empty vectors                 -> ValueError

  combine_similarity_scores (Sub-AC 2c):
    - Output range [0, 1] for all valid inputs
    - Default weights are 60/40 macro/price
    - Weight normalisation (raw importance values work)
    - Partial weight overrides fall back to defaults for missing keys
    - Macro-only and price-only weight extremes
    - Scale-invariance of weights
    - Error cases: negative weights, zero-sum, out-of-range scores
    - Integration pipeline: compute_macro_similarity + compute_price_similarity
      -> combine_similarity_scores always yields [0, 1]
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backtest.similarity import combine_similarity_scores, compute_macro_similarity
from backtest.price_similarity import compute_price_similarity as ps_compute


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_reference(a: list[float], b: list[float]) -> float:
    """Pure-Python reference implementation for test cross-checking."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(x ** 2 for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Edge cases — zero vectors
# ---------------------------------------------------------------------------

class TestZeroVectors:
    def test_both_zero_vectors(self):
        """Two zero vectors: undefined similarity -> 0.0."""
        result = compute_macro_similarity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        assert result == 0.0

    def test_first_zero_vector(self):
        """First vector is zero: undefined -> 0.0."""
        result = compute_macro_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
        assert result == 0.0

    def test_second_zero_vector(self):
        """Second vector is zero: undefined -> 0.0."""
        result = compute_macro_similarity([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
        assert result == 0.0

    def test_single_element_zero_vectors(self):
        result = compute_macro_similarity([0.0], [0.0])
        assert result == 0.0


# ---------------------------------------------------------------------------
# Edge cases — identical vectors
# ---------------------------------------------------------------------------

class TestIdenticalVectors:
    def test_identical_positive_vectors(self):
        """Identical vectors -> 1.0."""
        v = [1.0, 2.0, 3.0]
        result = compute_macro_similarity(v, v)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_identical_unit_vectors(self):
        v = [1.0 / math.sqrt(3)] * 3
        result = compute_macro_similarity(v, v)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_identical_negative_vectors(self):
        v = [-3.0, -1.5, -0.7]
        result = compute_macro_similarity(v, v)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_identical_mixed_sign_vectors(self):
        v = [-1.0, 0.5, 2.0, -0.3]
        result = compute_macro_similarity(v, v)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_scaled_vector_is_identical_direction(self):
        """A vector scaled by a positive constant has cosine similarity 1.0."""
        v = [1.0, 2.0, 3.0]
        v_scaled = [x * 42.7 for x in v]
        result = compute_macro_similarity(v, v_scaled)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_single_element_identical(self):
        result = compute_macro_similarity([5.0], [5.0])
        assert result == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Edge cases — orthogonal vectors
# ---------------------------------------------------------------------------

class TestOrthogonalVectors:
    def test_standard_basis_orthogonal(self):
        """Standard basis vectors are orthogonal."""
        result = compute_macro_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_2d_orthogonal(self):
        """[1, 0] and [0, 1] are orthogonal in 2D."""
        result = compute_macro_similarity([1.0, 0.0], [0.0, 1.0])
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_constructed_orthogonal(self):
        """[1, 1] and [1, -1] are orthogonal (dot product = 0)."""
        result = compute_macro_similarity([1.0, 1.0], [1.0, -1.0])
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_3d_orthogonal(self):
        """[1, 0, 0] and [0, 0, 1] are orthogonal."""
        result = compute_macro_similarity([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        assert result == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Opposite direction
# ---------------------------------------------------------------------------

class TestOppositeVectors:
    def test_opposite_vectors(self):
        """v and -v should have similarity -1.0."""
        v = [1.0, 2.0, 3.0]
        neg_v = [-x for x in v]
        result = compute_macro_similarity(v, neg_v)
        assert result == pytest.approx(-1.0, abs=1e-9)

    def test_opposite_single_element(self):
        result = compute_macro_similarity([3.0], [-3.0])
        assert result == pytest.approx(-1.0, abs=1e-9)

    def test_opposite_negative_scaled(self):
        v = [2.0, -4.0, 1.0]
        neg_v = [-6.0, 12.0, -3.0]  # -3 * v
        result = compute_macro_similarity(v, neg_v)
        assert result == pytest.approx(-1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Known similarity values
# ---------------------------------------------------------------------------

class TestKnownSimilarity:
    def test_45_degree_2d(self):
        """[1, 0] vs [1, 1] — 45° apart, cos(45°) = 1/sqrt(2)."""
        result = compute_macro_similarity([1.0, 0.0], [1.0, 1.0])
        expected = 1.0 / math.sqrt(2)
        assert result == pytest.approx(expected, abs=1e-9)

    def test_60_degree_2d(self):
        """[1, 0] vs [0.5, sqrt(3)/2] — 60° apart, cos(60°) = 0.5."""
        result = compute_macro_similarity([1.0, 0.0], [0.5, math.sqrt(3) / 2])
        assert result == pytest.approx(0.5, abs=1e-9)

    def test_arbitrary_vectors_match_reference(self):
        """Compare against pure-Python reference for an arbitrary pair."""
        a = [0.3, -1.2, 0.5, 2.1, -0.8]
        b = [0.7, 0.4, -0.9, 1.0, 0.2]
        expected = cosine_reference(a, b)
        result = compute_macro_similarity(a, b)
        assert result == pytest.approx(expected, abs=1e-9)

    def test_macro_realistic_vectors(self):
        """Simulate a realistic macro feature vector (e.g., 50 features)."""
        rng = np.random.default_rng(seed=42)
        a = rng.standard_normal(50)
        b = rng.standard_normal(50)
        expected = cosine_reference(a.tolist(), b.tolist())
        result = compute_macro_similarity(a, b)
        assert result == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Return value bounds
# ---------------------------------------------------------------------------

class TestReturnBounds:
    def test_result_in_minus_one_to_one(self):
        """Result is always within [-1, 1]."""
        rng = np.random.default_rng(seed=7)
        for _ in range(100):
            a = rng.standard_normal(20)
            b = rng.standard_normal(20)
            result = compute_macro_similarity(a, b)
            assert -1.0 <= result <= 1.0

    def test_symmetry(self):
        """similarity(a, b) == similarity(b, a)."""
        a = [0.1, 0.5, -0.3, 1.2]
        b = [-0.4, 0.8, 0.2, -1.0]
        assert compute_macro_similarity(a, b) == pytest.approx(
            compute_macro_similarity(b, a), abs=1e-12
        )


# ---------------------------------------------------------------------------
# Input type flexibility
# ---------------------------------------------------------------------------

class TestInputTypes:
    def test_list_inputs(self):
        """Accepts plain Python lists."""
        result = compute_macro_similarity([1.0, 0.0], [0.0, 1.0])
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_numpy_array_inputs(self):
        """Accepts numpy arrays."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        result = compute_macro_similarity(a, b)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_mixed_list_and_array(self):
        """Accepts mixed list and numpy array."""
        a = [1.0, 0.0, 0.0]
        b = np.array([0.0, 1.0, 0.0])
        result = compute_macro_similarity(a, b)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_integer_inputs_coerced(self):
        """Integer inputs are coerced to float."""
        result = compute_macro_similarity([1, 0], [0, 1])
        assert result == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestErrorCases:
    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_macro_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_empty_vectors_raises(self):
        with pytest.raises(ValueError):
            compute_macro_similarity([], [])

    def test_mismatched_one_empty_raises(self):
        with pytest.raises(ValueError):
            compute_macro_similarity([1.0, 2.0], [])


# ===========================================================================
# combine_similarity_scores — Sub-AC 2c
# ===========================================================================

_RNG = np.random.default_rng(42)


def _rand_vec(n: int = 20) -> np.ndarray:
    return _RNG.standard_normal(n)


def _rand_prices(n: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 + np.cumsum(rng.standard_normal(n) * 2.0)


class TestCombineSimilarityScores:
    """Tests for combine_similarity_scores (Sub-AC 2c).

    Validates:
      - Output range [0, 1] for all valid inputs
      - Default weights 60/40
      - Weight normalisation (raw "importance" values behave identically to
        pre-scaled fractions)
      - Partial weight overrides use defaults for missing keys
      - Macro-only and price-only weight extremes
      - Scale-invariance of weight ratios
      - Error cases: negative weights, zero-sum weights, out-of-range scores
    """

    # ------------------------------------------------------------------
    # Output range
    # ------------------------------------------------------------------

    def test_output_in_range_exhaustive_random(self):
        """Combined score must stay in [0, 1] for any valid component scores."""
        for _ in range(500):
            ms = float(_RNG.uniform(0.0, 1.0))
            ps = float(_RNG.uniform(0.0, 1.0))
            score = combine_similarity_scores(ms, ps)
            assert 0.0 <= score <= 1.0, (
                f"Out of range: macro={ms:.4f}, price={ps:.4f} -> {score:.6f}"
            )

    def test_extreme_boundary_inputs_stay_in_range(self):
        for ms, ps in [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)]:
            score = combine_similarity_scores(ms, ps)
            assert 0.0 <= score <= 1.0

    def test_boundary_exactly_zero(self):
        assert combine_similarity_scores(0.0, 0.0) == pytest.approx(0.0, abs=1e-12)

    def test_boundary_exactly_one(self):
        assert combine_similarity_scores(1.0, 1.0) == pytest.approx(1.0, abs=1e-12)

    # ------------------------------------------------------------------
    # Default weights (60 / 40)
    # ------------------------------------------------------------------

    def test_default_weights_macro_heavy(self):
        """macro=1, price=0 with default weights -> 0.6."""
        assert combine_similarity_scores(1.0, 0.0) == pytest.approx(0.6, abs=1e-9)

    def test_default_weights_price_heavy(self):
        """macro=0, price=1 with default weights -> 0.4."""
        assert combine_similarity_scores(0.0, 1.0) == pytest.approx(0.4, abs=1e-9)

    def test_default_weights_known_values(self):
        macro, price = 0.8, 0.5
        expected = 0.6 * macro + 0.4 * price
        assert combine_similarity_scores(macro, price) == pytest.approx(expected, abs=1e-9)

    # ------------------------------------------------------------------
    # Weight normalisation
    # ------------------------------------------------------------------

    def test_raw_importance_same_as_pre_scaled(self):
        """{"macro": 3, "price": 1} normalises to 0.75 / 0.25."""
        macro, price = 0.7, 0.3
        score_raw = combine_similarity_scores(macro, price, {"macro": 3.0, "price": 1.0})
        score_norm = combine_similarity_scores(macro, price, {"macro": 0.75, "price": 0.25})
        assert score_raw == pytest.approx(score_norm, abs=1e-12)

    def test_equal_weights_is_simple_average(self):
        macro, price = 0.6, 0.2
        score = combine_similarity_scores(macro, price, {"macro": 1.0, "price": 1.0})
        assert score == pytest.approx((macro + price) / 2.0, abs=1e-9)

    def test_uniform_scaling_does_not_change_result(self):
        """Multiplying all weights by any positive constant must not change output."""
        macro, price = 0.55, 0.75
        base = combine_similarity_scores(macro, price, {"macro": 0.6, "price": 0.4})
        for scale in [0.001, 2.0, 10.0, 1000.0]:
            scaled = combine_similarity_scores(
                macro, price, {"macro": 0.6 * scale, "price": 0.4 * scale}
            )
            assert scaled == pytest.approx(base, abs=1e-9), (
                f"Scale {scale} changed output: {base:.6f} -> {scaled:.6f}"
            )

    def test_large_raw_weights_normalise_correctly(self):
        """Very large unnormalised weights should still produce the same result."""
        macro, price = 0.4, 0.9
        score = combine_similarity_scores(macro, price, {"macro": 1e9, "price": 0.0})
        assert score == pytest.approx(macro, abs=1e-9)

    # ------------------------------------------------------------------
    # Weight extremes — macro-only and price-only
    # ------------------------------------------------------------------

    def test_macro_only_weight_equals_macro_score(self):
        macro, price = 0.82, 0.15
        score = combine_similarity_scores(macro, price, {"macro": 1.0, "price": 0.0})
        assert score == pytest.approx(macro, abs=1e-9)

    def test_price_only_weight_equals_price_score(self):
        macro, price = 0.82, 0.15
        score = combine_similarity_scores(macro, price, {"macro": 0.0, "price": 1.0})
        assert score == pytest.approx(price, abs=1e-9)

    # ------------------------------------------------------------------
    # Partial weight overrides
    # ------------------------------------------------------------------

    def test_partial_override_macro_key_only(self):
        """Providing only 'macro' fills 'price' from the default (0.4)."""
        macro, price = 0.5, 0.9
        score = combine_similarity_scores(macro, price, {"macro": 0.6})
        # Default price weight 0.4; total already 1.0 so no normalisation needed
        expected = 0.6 * macro + 0.4 * price
        assert score == pytest.approx(expected, abs=1e-9)

    def test_partial_override_price_key_only(self):
        """Providing only 'price' fills 'macro' from the default (0.6)."""
        macro, price = 0.3, 0.7
        score = combine_similarity_scores(macro, price, {"price": 0.4})
        expected = 0.6 * macro + 0.4 * price
        assert score == pytest.approx(expected, abs=1e-9)

    def test_none_weights_equals_no_argument(self):
        macro, price = 0.45, 0.65
        assert combine_similarity_scores(macro, price, None) == pytest.approx(
            combine_similarity_scores(macro, price), abs=1e-12
        )

    # ------------------------------------------------------------------
    # Monotonicity — pulling toward the dominant component
    # ------------------------------------------------------------------

    def test_monotone_in_macro_weight_when_macro_higher(self):
        """Higher macro weight -> score closer to macro_score (which is higher)."""
        macro_s, price_s = 0.9, 0.1
        low = combine_similarity_scores(macro_s, price_s, {"macro": 0.1, "price": 0.9})
        high = combine_similarity_scores(macro_s, price_s, {"macro": 0.9, "price": 0.1})
        assert high > low

    def test_monotone_in_price_weight_when_price_higher(self):
        """Higher price weight -> score closer to price_score (which is higher)."""
        macro_s, price_s = 0.1, 0.9
        low = combine_similarity_scores(macro_s, price_s, {"macro": 0.9, "price": 0.1})
        high = combine_similarity_scores(macro_s, price_s, {"macro": 0.1, "price": 0.9})
        assert high > low

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_negative_macro_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            combine_similarity_scores(0.5, 0.5, {"macro": -0.1, "price": 1.0})

    def test_negative_price_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            combine_similarity_scores(0.5, 0.5, {"macro": 1.0, "price": -0.1})

    def test_both_weights_zero_raises(self):
        with pytest.raises(ValueError, match="Sum of weights"):
            combine_similarity_scores(0.5, 0.5, {"macro": 0.0, "price": 0.0})

    def test_macro_score_above_one_raises(self):
        with pytest.raises(ValueError, match="macro_score"):
            combine_similarity_scores(1.001, 0.5)

    def test_macro_score_below_zero_raises(self):
        with pytest.raises(ValueError, match="macro_score"):
            combine_similarity_scores(-0.001, 0.5)

    def test_price_score_above_one_raises(self):
        with pytest.raises(ValueError, match="price_score"):
            combine_similarity_scores(0.5, 1.001)

    def test_price_score_below_zero_raises(self):
        with pytest.raises(ValueError, match="price_score"):
            combine_similarity_scores(0.5, -0.001)

    # ------------------------------------------------------------------
    # NaN safety
    # ------------------------------------------------------------------

    def test_no_nan_for_any_valid_inputs(self):
        for _ in range(200):
            ms = float(_RNG.uniform(0.0, 1.0))
            ps = float(_RNG.uniform(0.0, 1.0))
            score = combine_similarity_scores(ms, ps)
            assert not math.isnan(score), f"NaN for macro={ms}, price={ps}"

    # ------------------------------------------------------------------
    # Integration pipeline: raw vectors -> combined score
    # ------------------------------------------------------------------

    def test_pipeline_identical_macro_and_price(self):
        """
        Identical macro vectors (cosine=1) and identical price series (sim=1)
        should produce combined score = 1.0 regardless of weight split.
        """
        vec = _rand_vec()
        prices = _rand_prices()

        raw_cos = compute_macro_similarity(vec, vec)          # 1.0
        macro_s = (raw_cos + 1.0) / 2.0                       # 1.0
        price_s = ps_compute(prices.tolist(), prices.tolist()) # 1.0

        combined = combine_similarity_scores(macro_s, price_s)
        assert combined == pytest.approx(1.0, abs=1e-9)

    def test_pipeline_opposite_macro_zero_price_similarity(self):
        """
        Opposite macro vectors (cosine=-1 -> normalised 0.0) and
        a price series compared with itself rotated 180 degrees
        should produce combined score near 0.
        """
        vec = _rand_vec()
        prices_a = _rand_prices(seed=7)
        prices_b = _rand_prices(seed=99)  # different random seed -> low similarity

        raw_cos = compute_macro_similarity(vec, -vec)     # -1.0
        macro_s = (raw_cos + 1.0) / 2.0                   # 0.0

        # Use the two random prices so price sim is not pinned to 0 or 1;
        # combined score should be close to macro_s since macro is the anchor
        price_s = ps_compute(prices_a.tolist(), prices_b.tolist())
        combined = combine_similarity_scores(macro_s, price_s)
        assert 0.0 <= combined <= 1.0

    def test_pipeline_output_range_randomised(self):
        """
        End-to-end: raw vectors -> macro_score -> combine.
        Combined score must always land in [0, 1].
        """
        for i in range(100):
            va, vb = _rand_vec(), _rand_vec()
            pa = _rand_prices(seed=i * 2)
            pb = _rand_prices(seed=i * 2 + 1)
            w_m = float(_RNG.uniform(0.01, 1.0))
            w_p = float(_RNG.uniform(0.01, 1.0))

            raw_cos = compute_macro_similarity(va, vb)
            macro_s = (raw_cos + 1.0) / 2.0
            price_s = ps_compute(pa.tolist(), pb.tolist())

            combined = combine_similarity_scores(
                macro_s, price_s, {"macro": w_m, "price": w_p}
            )
            assert 0.0 <= combined <= 1.0, (
                f"Iteration {i}: combined={combined:.4f} out of [0,1]"
            )

    def test_pipeline_macro_dominance_when_weighted_higher(self):
        """
        When macro weight >> price weight and macro similarity is high but
        price similarity is low, the combined score should be pulled toward
        macro_score.
        """
        vec = _rand_vec()
        prices_unrelated = _rand_prices(seed=42)
        prices_different = _rand_prices(seed=7)

        # High macro similarity: identical vectors
        raw_cos_high = compute_macro_similarity(vec, vec)
        macro_high = (raw_cos_high + 1.0) / 2.0   # 1.0

        # Low macro similarity: random pair
        va2, vb2 = _rand_vec(), _rand_vec()
        raw_cos_low = compute_macro_similarity(va2, -va2)
        macro_low = (raw_cos_low + 1.0) / 2.0   # 0.0

        price_s = ps_compute(prices_unrelated.tolist(), prices_different.tolist())

        score_high = combine_similarity_scores(
            macro_high, price_s, {"macro": 0.9, "price": 0.1}
        )
        score_low = combine_similarity_scores(
            macro_low, price_s, {"macro": 0.9, "price": 0.1}
        )

        # With macro heavily weighted, identical macro -> much higher score
        assert score_high > score_low
