"""
Tests for backtest.analogue_engine.find_analogues.

Covers:
  - Correct return type and count
  - All scores are in [0, 1]
  - combined_score is consistent with macro_score, price_score, and the weights
  - Results are sorted by combined_score descending
  - Minimum separation: no two results overlap by < min_separation_days
  - Query date filtering: analogues all predate the query date
  - Top features list is populated and has correct fields
  - Forward return fields are present (may be None near end of data)
  - Raises FileNotFoundError when parquet doesn't exist
  - Raises ValueError for bad lookback / insufficient history
  - Weight extremes (macro-only, price-only)
  - Single analogue (n=1)
  - Reproducibility: calling twice gives same results

Uses a synthetic feature matrix so tests run without the real parquet file.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------


def _make_synthetic_df(
    n_rows: int = 500,
    seed: int = 42,
    start_date: str = "2015-01-01",
) -> pd.DataFrame:
    """
    Build a minimal synthetic feature matrix that looks like macro_features.parquet.

    Columns produced:
      ^GSPC_ret_1d, ^GSPC_ret_5d, ^GSPC_ret_21d, ^GSPC_ret_63d
      ^VIX_ret_1d, ^VIX_ret_5d
      ^TNX_ret_1d
      macro_copper_gold_ratio
      macro_vix_tnx_ratio
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_rows)

    sp500_daily = rng.standard_normal(n_rows) * 0.01  # daily returns ~1%
    cumulative  = np.cumprod(1.0 + sp500_daily)

    data: dict[str, Any] = {
        "^GSPC_ret_1d":  sp500_daily,
        "^GSPC_ret_5d":  pd.Series(sp500_daily).rolling(5).sum().values,
        "^GSPC_ret_21d": pd.Series(sp500_daily).rolling(21).sum().values,
        "^GSPC_ret_63d": pd.Series(sp500_daily).rolling(63).sum().values,
        "^VIX_ret_1d":   rng.standard_normal(n_rows) * 0.02,
        "^VIX_ret_5d":   rng.standard_normal(n_rows) * 0.04,
        "^TNX_ret_1d":   rng.standard_normal(n_rows) * 0.005,
        "macro_copper_gold_ratio":      rng.uniform(0.3, 0.6, n_rows),
        "macro_vix_tnx_ratio":          rng.uniform(2.0, 8.0, n_rows),
    }

    df = pd.DataFrame(data, index=dates)
    df = df.fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Fixture: patch the module so it reads our synthetic df
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_features(tmp_path: Path):
    """
    Write a synthetic parquet to a temp file and patch _FEATURES_PATH so the
    engine picks it up.
    """
    df = _make_synthetic_df(n_rows=600)
    parquet_path = tmp_path / "macro_features.parquet"
    df.to_parquet(parquet_path)
    return df, parquet_path


def _run_find_analogues(parquet_path: Path, **kwargs):
    """Helper: call find_analogues with the given parquet path patched in."""
    import backtest.analogue_engine as eng

    with patch.object(eng, "_FEATURES_PATH", parquet_path), \
         patch.object(eng, "_DB_PATH", parquet_path.parent / "nonexistent.db"):
        return eng.find_analogues(**kwargs)


# ===========================================================================
# Basic shape and type checks
# ===========================================================================


class TestReturnType:
    def test_returns_list(self, synthetic_features):
        df, parquet = synthetic_features
        results = _run_find_analogues(parquet, n=5)
        assert isinstance(results, list)

    def test_returns_n_results(self, synthetic_features):
        # Use small min_separation so 5 non-overlapping windows always fit in
        # the 600-row synthetic dataset.
        df, parquet = synthetic_features
        results = _run_find_analogues(parquet, n=5, min_separation_days=5)
        assert len(results) == 5

    def test_returns_fewer_when_not_enough_candidates(self, synthetic_features):
        df, parquet = synthetic_features
        # Very large min_separation means fewer unique periods available
        results = _run_find_analogues(parquet, n=10, min_separation_days=200)
        assert len(results) <= 10

    def test_single_analogue(self, synthetic_features):
        df, parquet = synthetic_features
        results = _run_find_analogues(parquet, n=1)
        assert len(results) == 1

    def test_result_fields_present(self, synthetic_features):
        df, parquet = synthetic_features
        r = _run_find_analogues(parquet, n=1)[0]
        assert hasattr(r, "start_date")
        assert hasattr(r, "end_date")
        assert hasattr(r, "macro_score")
        assert hasattr(r, "price_score")
        assert hasattr(r, "combined_score")
        assert hasattr(r, "forward_ret_5d")
        assert hasattr(r, "forward_ret_21d")
        assert hasattr(r, "forward_ret_63d")
        assert hasattr(r, "top_features")


# ===========================================================================
# Score ranges and consistency
# ===========================================================================


class TestScoreRanges:
    def test_macro_score_in_range(self, synthetic_features):
        df, parquet = synthetic_features
        for r in _run_find_analogues(parquet, n=5):
            assert 0.0 <= r.macro_score <= 1.0, f"macro_score={r.macro_score} out of range"

    def test_price_score_in_range(self, synthetic_features):
        df, parquet = synthetic_features
        for r in _run_find_analogues(parquet, n=5):
            assert 0.0 <= r.price_score <= 1.0, f"price_score={r.price_score} out of range"

    def test_combined_score_in_range(self, synthetic_features):
        df, parquet = synthetic_features
        for r in _run_find_analogues(parquet, n=5):
            assert 0.0 <= r.combined_score <= 1.0, f"combined_score={r.combined_score} out of range"

    def test_combined_score_consistent_with_components_default_weights(self, synthetic_features):
        """
        With default weights (macro=0.6, price=0.4), combined ≈ 0.6*macro + 0.4*price.
        """
        df, parquet = synthetic_features
        for r in _run_find_analogues(parquet, n=5):
            expected = 0.6 * r.macro_score + 0.4 * r.price_score
            assert r.combined_score == pytest.approx(expected, abs=0.01), (
                f"combined={r.combined_score:.4f}, expected≈{expected:.4f}"
            )

    def test_no_nan_scores(self, synthetic_features):
        df, parquet = synthetic_features
        for r in _run_find_analogues(parquet, n=5):
            assert not math.isnan(r.macro_score)
            assert not math.isnan(r.price_score)
            assert not math.isnan(r.combined_score)

    def test_scores_not_all_identical(self, synthetic_features):
        """Scores should not be degenerate (all 0.0 or all 1.0) for random data."""
        df, parquet = synthetic_features
        results = _run_find_analogues(parquet, n=5)
        scores = [r.combined_score for r in results]
        # There should be some variance in the scores
        assert max(scores) - min(scores) > 0.0 or len(scores) == 1


# ===========================================================================
# Sorted by combined_score descending
# ===========================================================================


class TestOrdering:
    def test_results_sorted_descending(self, synthetic_features):
        df, parquet = synthetic_features
        results = _run_find_analogues(parquet, n=5)
        scores = [r.combined_score for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Results not sorted descending: {scores}"
        )

    def test_first_result_is_best(self, synthetic_features):
        df, parquet = synthetic_features
        results = _run_find_analogues(parquet, n=5)
        if len(results) > 1:
            assert results[0].combined_score >= results[1].combined_score


# ===========================================================================
# Minimum separation
# ===========================================================================


class TestMinSeparation:
    def _trading_days_between(self, end_date_a: str, end_date_b: str) -> int:
        """Approximate trading days between two ISO date strings."""
        d1 = pd.Timestamp(end_date_a)
        d2 = pd.Timestamp(end_date_b)
        return int(abs((d2 - d1).days) * 5 / 7)  # rough conversion

    def test_no_overlapping_periods(self, synthetic_features):
        """No two results should have end_dates closer than min_separation_days."""
        df, parquet = synthetic_features
        min_sep = 63  # trading days
        results = _run_find_analogues(parquet, n=5, min_separation_days=min_sep)

        end_dates = [pd.Timestamp(r.end_date) for r in results]
        for i in range(len(end_dates)):
            for j in range(i + 1, len(end_dates)):
                delta = abs((end_dates[i] - end_dates[j]).days)
                assert delta >= min_sep, (
                    f"Results {i} ({end_dates[i].date()}) and {j} ({end_dates[j].date()}) "
                    f"are only {delta} calendar days apart (min_sep={min_sep})"
                )

    def test_smaller_sep_allows_more_results(self, synthetic_features):
        """Smaller min_separation should allow at least as many results."""
        df, parquet = synthetic_features
        results_tight = _run_find_analogues(parquet, n=5, min_separation_days=252)
        results_loose = _run_find_analogues(parquet, n=5, min_separation_days=5)
        assert len(results_loose) >= len(results_tight)


# ===========================================================================
# Query date filtering
# ===========================================================================


class TestQueryDate:
    def test_analogues_predate_query(self, synthetic_features):
        """All analogue end dates must be strictly before the query date."""
        df, parquet = synthetic_features
        # Use a date somewhere in the middle of the dataset
        query_date = str(df.index[400].date())
        results = _run_find_analogues(parquet, n=5, date=query_date)
        q_dt = pd.Timestamp(query_date)
        for r in results:
            assert pd.Timestamp(r.end_date) < q_dt, (
                f"Analogue {r.end_date} is not before query date {query_date}"
            )

    def test_date_string_accepted(self, synthetic_features):
        df, parquet = synthetic_features
        query_date = str(df.index[300].date())
        results = _run_find_analogues(parquet, n=3, date=query_date)
        assert isinstance(results, list)

    def test_date_before_any_data_raises(self, synthetic_features):
        df, parquet = synthetic_features
        import backtest.analogue_engine as eng

        with patch.object(eng, "_FEATURES_PATH", parquet), \
             patch.object(eng, "_DB_PATH", parquet.parent / "nonexistent.db"), \
             pytest.raises(ValueError):
            eng.find_analogues(date="1990-01-01")


# ===========================================================================
# Top features
# ===========================================================================


class TestTopFeatures:
    def test_top_features_is_list(self, synthetic_features):
        df, parquet = synthetic_features
        r = _run_find_analogues(parquet, n=1)[0]
        assert isinstance(r.top_features, list)

    def test_top_features_not_empty(self, synthetic_features):
        df, parquet = synthetic_features
        r = _run_find_analogues(parquet, n=1)[0]
        assert len(r.top_features) > 0

    def test_top_features_have_required_keys(self, synthetic_features):
        df, parquet = synthetic_features
        r = _run_find_analogues(parquet, n=1)[0]
        required_keys = {"feature", "query_value", "hist_value", "cosine_contribution"}
        for feat in r.top_features:
            assert required_keys.issubset(feat.keys()), (
                f"Missing keys in top_feature: {feat}"
            )

    def test_top_features_feature_names_are_strings(self, synthetic_features):
        df, parquet = synthetic_features
        r = _run_find_analogues(parquet, n=1)[0]
        for feat in r.top_features:
            assert isinstance(feat["feature"], str)

    def test_top_features_values_are_floats(self, synthetic_features):
        df, parquet = synthetic_features
        r = _run_find_analogues(parquet, n=1)[0]
        for feat in r.top_features:
            assert isinstance(feat["query_value"], float)
            assert isinstance(feat["hist_value"], float)
            assert isinstance(feat["cosine_contribution"], float)


# ===========================================================================
# Forward returns
# ===========================================================================


class TestForwardReturns:
    def test_forward_returns_numeric_or_none(self, synthetic_features):
        df, parquet = synthetic_features
        for r in _run_find_analogues(parquet, n=5):
            for attr in ("forward_ret_5d", "forward_ret_21d", "forward_ret_63d"):
                val = getattr(r, attr)
                assert val is None or isinstance(val, float), (
                    f"{attr}={val!r} is not float or None"
                )

    def test_forward_returns_not_nan(self, synthetic_features):
        df, parquet = synthetic_features
        for r in _run_find_analogues(parquet, n=5):
            for attr in ("forward_ret_5d", "forward_ret_21d", "forward_ret_63d"):
                val = getattr(r, attr)
                if val is not None:
                    assert not math.isnan(val), f"{attr} is NaN"

    def test_forward_returns_present_for_early_analogues(self, synthetic_features):
        """
        Analogues that end well before the query date should have forward returns.

        The engine slices the dataframe to [start .. query_date], so forward
        returns are only available for analogues whose end_date is at least
        fwd_days trading rows before query_date.  We query at row 200 and look
        for analogues in the first third of history, guaranteeing plenty of
        headroom inside the sliced window.
        """
        df, parquet = synthetic_features
        # Query at row 200 — plenty of future rows remain inside the slice.
        query_date = str(df.index[200].date())
        # Restrict analogues to end before row 180 to ensure ≥ 20 rows of
        # headroom for forward return look-up (5d ≤ 20 rows).
        end_date_cutoff = df.index[180]
        results = _run_find_analogues(
            parquet, n=3, date=query_date, min_separation_days=5
        )
        for r in results:
            end_dt = pd.Timestamp(r.end_date)
            # Only assert for analogues clearly before the query slice boundary
            if end_dt <= pd.Timestamp(end_date_cutoff):
                assert r.forward_ret_5d is not None, (
                    f"Expected 5d return for analogue at {r.end_date} "
                    f"(query={query_date}) but got None"
                )


# ===========================================================================
# Date strings
# ===========================================================================


class TestDateStrings:
    def test_start_end_dates_are_iso_strings(self, synthetic_features):
        df, parquet = synthetic_features
        for r in _run_find_analogues(parquet, n=3):
            # Should parse without error
            pd.Timestamp(r.start_date)
            pd.Timestamp(r.end_date)

    def test_start_date_before_end_date(self, synthetic_features):
        df, parquet = synthetic_features
        for r in _run_find_analogues(parquet, n=5):
            assert pd.Timestamp(r.start_date) <= pd.Timestamp(r.end_date), (
                f"start={r.start_date} is after end={r.end_date}"
            )

    def test_window_length_approximately_correct(self, synthetic_features):
        """end_date - start_date should be approximately lookback_days in calendar days."""
        df, parquet = synthetic_features
        lookback = 30
        results = _run_find_analogues(parquet, n=3, lookback_days=lookback)
        for r in results:
            start = pd.Timestamp(r.start_date)
            end   = pd.Timestamp(r.end_date)
            cal_days = (end - start).days
            # 30 trading days ≈ 42 calendar days; allow generous range
            assert 0 <= cal_days <= 60, (
                f"Window spans {cal_days} calendar days for lookback={lookback}"
            )


# ===========================================================================
# Weight customisation
# ===========================================================================


class TestWeights:
    def test_macro_only_weight_macro_score_equals_combined(self, synthetic_features):
        """
        When price_weight=0, combined_score should equal macro_score exactly.
        """
        df, parquet = synthetic_features
        results = _run_find_analogues(
            parquet, n=3, macro_weight=1.0, price_weight=0.0
        )
        for r in results:
            assert r.combined_score == pytest.approx(r.macro_score, abs=1e-4)

    def test_price_only_weight_price_score_equals_combined(self, synthetic_features):
        """
        When macro_weight=0, combined_score should equal price_score exactly.
        """
        df, parquet = synthetic_features
        results = _run_find_analogues(
            parquet, n=3, macro_weight=0.0, price_weight=1.0
        )
        for r in results:
            assert r.combined_score == pytest.approx(r.price_score, abs=1e-4)

    def test_equal_weights_is_average(self, synthetic_features):
        """
        When macro_weight == price_weight, combined == (macro + price) / 2.
        """
        df, parquet = synthetic_features
        results = _run_find_analogues(
            parquet, n=3, macro_weight=1.0, price_weight=1.0
        )
        for r in results:
            expected = (r.macro_score + r.price_score) / 2.0
            assert r.combined_score == pytest.approx(expected, abs=1e-4)

    def test_unnormalised_weights_same_as_normalised(self, synthetic_features):
        """
        {"macro": 3, "price": 2} should give same ranking as {"macro": 0.6, "price": 0.4}.
        """
        df, parquet = synthetic_features
        r1 = _run_find_analogues(parquet, n=5, macro_weight=3.0, price_weight=2.0)
        r2 = _run_find_analogues(parquet, n=5, macro_weight=0.6, price_weight=0.4)
        dates1 = [r.end_date for r in r1]
        dates2 = [r.end_date for r in r2]
        assert dates1 == dates2, "Different unnormalised/normalised weights give different rankings"


# ===========================================================================
# Error cases
# ===========================================================================


class TestErrorCases:
    def test_missing_parquet_raises_file_not_found(self, tmp_path: Path):
        import backtest.analogue_engine as eng

        nonexistent = tmp_path / "missing.parquet"
        with patch.object(eng, "_FEATURES_PATH", nonexistent):
            with pytest.raises(FileNotFoundError):
                eng.find_analogues()

    def test_lookback_less_than_2_raises(self, synthetic_features):
        import backtest.analogue_engine as eng

        df, parquet = synthetic_features
        with patch.object(eng, "_FEATURES_PATH", parquet), \
             patch.object(eng, "_DB_PATH", parquet.parent / "nonexistent.db"), \
             pytest.raises(ValueError, match="lookback_days"):
            eng.find_analogues(lookback_days=1)

    def test_insufficient_history_raises(self, tmp_path: Path):
        import backtest.analogue_engine as eng

        # Only 10 rows but lookback=63
        df = _make_synthetic_df(n_rows=10)
        small_parquet = tmp_path / "small.parquet"
        df.to_parquet(small_parquet)

        with patch.object(eng, "_FEATURES_PATH", small_parquet), \
             patch.object(eng, "_DB_PATH", tmp_path / "nonexistent.db"), \
             pytest.raises(ValueError):
            eng.find_analogues(lookback_days=63)


# ===========================================================================
# Reproducibility
# ===========================================================================


class TestReproducibility:
    def test_same_results_on_two_calls(self, synthetic_features):
        df, parquet = synthetic_features
        r1 = _run_find_analogues(parquet, n=5)
        r2 = _run_find_analogues(parquet, n=5)
        assert [r.end_date for r in r1] == [r.end_date for r in r2]
        assert [r.combined_score for r in r1] == [r.combined_score for r in r2]


# ===========================================================================
# to_dict serialisation
# ===========================================================================


class TestToDict:
    def test_to_dict_returns_dict(self, synthetic_features):
        df, parquet = synthetic_features
        r = _run_find_analogues(parquet, n=1)[0]
        d = r.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_json_serialisable(self, synthetic_features):
        df, parquet = synthetic_features
        r = _run_find_analogues(parquet, n=1)[0]
        d = r.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["end_date"] == r.end_date

    def test_to_dict_all_fields_present(self, synthetic_features):
        df, parquet = synthetic_features
        r = _run_find_analogues(parquet, n=1)[0]
        d = r.to_dict()
        expected_keys = {
            "start_date", "end_date",
            "macro_score", "price_score", "combined_score",
            "forward_ret_5d", "forward_ret_21d", "forward_ret_63d",
            "top_features",
        }
        assert expected_keys.issubset(d.keys())


# ===========================================================================
# Integration: _batch_cosine_similarities internal helper
# ===========================================================================


class TestBatchCosineSimilarities:
    """Verify the vectorised cosine helper matches the per-row scalar version."""

    def test_matches_scalar_implementation(self):
        from backtest.analogue_engine import _batch_cosine_similarities
        from backtest.similarity import compute_macro_similarity

        rng = np.random.default_rng(7)
        query = rng.standard_normal(20)
        matrix = rng.standard_normal((50, 20))

        batch = _batch_cosine_similarities(query, matrix)
        scalar = np.array([compute_macro_similarity(query, row) for row in matrix])

        np.testing.assert_allclose(batch, scalar, atol=1e-9)

    def test_zero_query_vector_returns_zeros(self):
        from backtest.analogue_engine import _batch_cosine_similarities

        query = np.zeros(10)
        matrix = np.random.default_rng(1).standard_normal((20, 10))
        result = _batch_cosine_similarities(query, matrix)
        np.testing.assert_array_equal(result, np.zeros(20))

    def test_output_in_minus_one_to_one(self):
        from backtest.analogue_engine import _batch_cosine_similarities

        rng = np.random.default_rng(3)
        query = rng.standard_normal(15)
        matrix = rng.standard_normal((100, 15))
        result = _batch_cosine_similarities(query, matrix)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)


# ===========================================================================
# Integration: _top_feature_drivers
# ===========================================================================


class TestTopFeatureDrivers:
    def test_returns_list_of_dicts(self):
        from backtest.analogue_engine import _top_feature_drivers

        rng = np.random.default_rng(0)
        q = rng.standard_normal(10)
        h = rng.standard_normal(10)
        names = [f"feat_{i}" for i in range(10)]
        result = _top_feature_drivers(q, h, names, top_k=3)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_identical_vectors_all_positive_contributions(self):
        from backtest.analogue_engine import _top_feature_drivers

        rng = np.random.default_rng(5)
        vec = rng.standard_normal(10)
        names = [f"f{i}" for i in range(10)]
        result = _top_feature_drivers(vec, vec, names, top_k=5)
        for item in result:
            assert item["cosine_contribution"] > 0.0

    def test_zero_vector_returns_empty(self):
        from backtest.analogue_engine import _top_feature_drivers

        zero = np.zeros(5)
        normal = np.ones(5)
        result = _top_feature_drivers(zero, normal, [f"f{i}" for i in range(5)])
        assert result == []
