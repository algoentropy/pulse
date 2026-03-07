"""
Tests for backtest.price_pattern_scanner.

Coverage
--------
extract_price_windows:
  - Returns correct number of windows for various (n, window_size, step)
  - Each window has the right length
  - start_date / end_date match the source dates at correct indices
  - start_idx is correct
  - step > 1 reduces the count correctly
  - Error: window_size < 2
  - Error: mismatched dates / prices lengths
  - Error: series shorter than window_size
  - Error: step < 1

score_price_windows:
  - Returns a list with the same length as historical_windows
  - Scores are in [0, 1]
  - Best match is sorted first (identical window gets score 1.0 at index 0)
  - Works with both "pearson" and "dtw" methods
  - Error: current_prices fewer than 2 points
  - Error: empty historical_windows list

find_price_analogues:
  - Returns at most top_k results
  - All similarity scores are in [0, 1]
  - forward_returns keys match forward_horizons
  - Results are sorted by descending similarity
  - min_gap_days enforcement: no two results are within min_gap_days
  - Identical sub-series is found as best analogue (sanity check)
  - Handles edge-case: series just barely long enough
  - Error: series too short

_days_between (internal helper):
  - Same date -> 0
  - Known interval
  - Cross month / year boundary

Integration:
  - end-to-end on a deterministic series: injected copy of current window
    is retrieved as the top analogue
  - forward_returns are None when horizon exceeds series length
  - forward_returns are correct fractional values
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from backtest.price_pattern_scanner import (
    PriceWindow,
    ScoredWindow,
    _compute_forward_returns,
    _days_between,
    _deduplicate,
    extract_price_windows,
    find_price_analogues,
    score_price_windows,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dates(n: int, start: str = "2010-01-04") -> list[str]:
    """Generate *n* sequential calendar dates starting from *start*."""
    y, m, d = (int(x) for x in start.split("-"))
    base = date(y, m, d)
    return [(base + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n)]


def _linear_prices(start: float, end: float, n: int) -> list[float]:
    return [start + (end - start) * i / (n - 1) for i in range(n)]


def _rw_prices(n: int, seed: int = 42, start: float = 100.0) -> list[float]:
    rng = np.random.default_rng(seed)
    prices = start + np.cumsum(rng.standard_normal(n) * 2.0)
    return prices.tolist()


# ===========================================================================
# extract_price_windows
# ===========================================================================


class TestExtractPriceWindows:
    # --- Basic count / shape ---

    def test_count_step_1(self):
        n, ws = 20, 5
        dates = _make_dates(n)
        prices = _rw_prices(n)
        windows = extract_price_windows(dates, prices, window_size=ws, step=1)
        # n - ws + 1 complete windows
        assert len(windows) == n - ws + 1

    def test_count_step_2(self):
        n, ws = 21, 5
        dates = _make_dates(n)
        prices = _rw_prices(n)
        windows = extract_price_windows(dates, prices, window_size=ws, step=2)
        expected = math.ceil((n - ws + 1) / 2)
        assert len(windows) == expected

    def test_count_step_equals_window_size(self):
        """Non-overlapping tiles."""
        n, ws = 20, 5
        dates = _make_dates(n)
        prices = _rw_prices(n)
        windows = extract_price_windows(dates, prices, window_size=ws, step=ws)
        assert len(windows) == n // ws

    def test_window_length_matches_window_size(self):
        ws = 7
        dates = _make_dates(30)
        prices = _rw_prices(30)
        for win in extract_price_windows(dates, prices, window_size=ws):
            assert len(win.prices) == ws

    def test_exact_series_length_equals_window_size(self):
        """Only one complete window when n == window_size."""
        ws = 10
        dates = _make_dates(ws)
        prices = _rw_prices(ws)
        windows = extract_price_windows(dates, prices, window_size=ws)
        assert len(windows) == 1
        assert windows[0].start_date == dates[0]
        assert windows[0].end_date == dates[-1]

    # --- Date / index correctness ---

    def test_start_date_and_end_date_correct(self):
        n, ws = 10, 4
        dates = _make_dates(n)
        prices = _rw_prices(n)
        windows = extract_price_windows(dates, prices, window_size=ws)

        for i, win in enumerate(windows):
            assert win.start_date == dates[i]
            assert win.end_date == dates[i + ws - 1]

    def test_start_idx_correct(self):
        n, ws = 15, 5
        dates = _make_dates(n)
        prices = _rw_prices(n)
        for i, win in enumerate(extract_price_windows(dates, prices, window_size=ws)):
            assert win.start_idx == i

    def test_prices_are_correct_slice(self):
        n, ws = 10, 4
        dates = _make_dates(n)
        prices = list(range(n))  # [0, 1, 2, ..., 9]
        windows = extract_price_windows(dates, prices, window_size=ws)
        for i, win in enumerate(windows):
            assert win.prices == prices[i : i + ws]

    def test_windows_chronological_order(self):
        n, ws = 20, 5
        dates = _make_dates(n)
        prices = _rw_prices(n)
        windows = extract_price_windows(dates, prices, window_size=ws)
        for a, b in zip(windows, windows[1:]):
            assert a.start_date < b.start_date

    # --- Error cases ---

    def test_window_size_less_than_2_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            extract_price_windows(_make_dates(10), _rw_prices(10), window_size=1)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            extract_price_windows(_make_dates(10), _rw_prices(5), window_size=3)

    def test_series_shorter_than_window_raises(self):
        with pytest.raises(ValueError, match="shorter"):
            extract_price_windows(_make_dates(4), _rw_prices(4), window_size=5)

    def test_step_zero_raises(self):
        with pytest.raises(ValueError, match="step"):
            extract_price_windows(_make_dates(10), _rw_prices(10), window_size=3, step=0)

    def test_step_negative_raises(self):
        with pytest.raises(ValueError, match="step"):
            extract_price_windows(_make_dates(10), _rw_prices(10), window_size=3, step=-1)


# ===========================================================================
# score_price_windows
# ===========================================================================


class TestScorePriceWindows:
    def _make_windows(self, n_windows: int, ws: int, seed_offset: int = 0) -> list[PriceWindow]:
        out = []
        for i in range(n_windows):
            prices = _rw_prices(ws, seed=i + seed_offset)
            out.append(PriceWindow(
                start_date=f"2010-{i+1:02d}-01",
                end_date=f"2010-{i+1:02d}-28",
                prices=prices,
                start_idx=i * ws,
            ))
        return out

    def test_length_matches_input(self):
        ws = 10
        current = _rw_prices(ws, seed=99)
        windows = self._make_windows(5, ws)
        result = score_price_windows(current, windows)
        assert len(result) == 5

    def test_scores_in_zero_one(self):
        ws = 20
        current = _rw_prices(ws, seed=7)
        windows = self._make_windows(10, ws)
        for _, score in score_price_windows(current, windows):
            assert 0.0 <= score <= 1.0

    def test_sorted_best_first(self):
        ws = 20
        current = _rw_prices(ws, seed=7)
        windows = self._make_windows(10, ws)
        result = score_price_windows(current, windows)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_identical_window_scores_one(self):
        ws = 15
        current = _rw_prices(ws, seed=42)
        identical_win = PriceWindow(
            start_date="2020-01-01",
            end_date="2020-01-15",
            prices=list(current),
            start_idx=0,
        )
        other_windows = self._make_windows(5, ws, seed_offset=100)
        all_windows = [identical_win] + other_windows
        result = score_price_windows(current, all_windows)
        # Identical window must be first with score 1.0
        assert result[0][0].start_date == "2020-01-01"
        assert result[0][1] == pytest.approx(1.0, abs=1e-9)

    def test_pearson_method_works(self):
        ws = 15
        current = _rw_prices(ws, seed=1)
        windows = self._make_windows(3, ws)
        result = score_price_windows(current, windows, method="pearson")
        assert len(result) == 3

    def test_dtw_method_works(self):
        ws = 10
        current = _rw_prices(ws, seed=2)
        windows = self._make_windows(3, ws)
        result = score_price_windows(current, windows, method="dtw")
        assert len(result) == 3

    def test_current_prices_too_short_raises(self):
        windows = self._make_windows(3, 10)
        with pytest.raises(ValueError, match="current_prices"):
            score_price_windows([100.0], windows)

    def test_empty_historical_windows_raises(self):
        with pytest.raises(ValueError, match="empty"):
            score_price_windows(_rw_prices(10), [])


# ===========================================================================
# _days_between (internal)
# ===========================================================================


class TestDaysBetween:
    def test_same_date_is_zero(self):
        assert _days_between("2020-06-15", "2020-06-15") == 0

    def test_one_day_apart(self):
        assert _days_between("2020-06-15", "2020-06-16") == 1
        assert _days_between("2020-06-16", "2020-06-15") == 1  # symmetric

    def test_known_interval(self):
        # 2020 is a leap year; 2020-01-01 to 2020-03-01 = 60 days
        assert _days_between("2020-01-01", "2020-03-01") == 60

    def test_cross_year_boundary(self):
        # 2019-12-31 to 2020-01-01 = 1 day
        assert _days_between("2019-12-31", "2020-01-01") == 1

    def test_multi_year(self):
        # 365 * 2 = 730 for 2 non-leap years (2021, 2022)
        assert _days_between("2021-01-01", "2023-01-01") == 730


# ===========================================================================
# _deduplicate (internal)
# ===========================================================================


class TestDeduplicate:
    def _make_scored(self, end_dates: list[str], scores: list[float]) -> list[tuple[PriceWindow, float]]:
        result = []
        for d, s in zip(end_dates, scores):
            win = PriceWindow(
                start_date="2000-01-01",  # unused here
                end_date=d,
                prices=[100.0, 101.0],
                start_idx=0,
            )
            result.append((win, s))
        return result

    def test_top_k_respected(self):
        # 5 windows far apart; ask for 3
        dates = ["2010-01-01", "2011-01-01", "2012-01-01", "2013-01-01", "2014-01-01"]
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        result = _deduplicate(self._make_scored(dates, scores), top_k=3, min_gap_days=30)
        assert len(result) == 3

    def test_min_gap_enforced(self):
        # Two windows only 5 days apart; min_gap=30 → only first should be returned
        dates = ["2010-01-01", "2010-01-06"]
        scores = [0.9, 0.85]
        result = _deduplicate(self._make_scored(dates, scores), top_k=5, min_gap_days=30)
        assert len(result) == 1
        assert result[0][0].end_date == "2010-01-01"

    def test_enough_spread_returns_all(self):
        dates = ["2010-01-01", "2010-05-01", "2010-09-01"]
        scores = [0.8, 0.7, 0.6]
        result = _deduplicate(self._make_scored(dates, scores), top_k=5, min_gap_days=90)
        assert len(result) == 3

    def test_best_score_wins_when_clustered(self):
        # Three dates within 10 days of each other; only the best should survive
        dates = ["2010-01-01", "2010-01-05", "2010-01-09"]
        scores = [0.7, 0.9, 0.5]  # already best-first sorted? No — deduplicate receives sorted input
        # Simulate sorted input: best score first
        scored = sorted(self._make_scored(dates, scores), key=lambda x: x[1], reverse=True)
        result = _deduplicate(scored, top_k=5, min_gap_days=30)
        assert len(result) == 1
        assert result[0][1] == 0.9


# ===========================================================================
# _compute_forward_returns (internal)
# ===========================================================================


class TestComputeForwardReturns:
    def test_basic_positive_return(self):
        prices = [100.0, 105.0, 110.0, 115.0, 120.0]
        result = _compute_forward_returns(0, prices, horizons=(2, 4))
        assert result[2] == pytest.approx(0.10, abs=1e-9)   # 110/100 - 1
        assert result[4] == pytest.approx(0.20, abs=1e-9)   # 120/100 - 1

    def test_negative_return(self):
        prices = [200.0, 180.0, 160.0]
        result = _compute_forward_returns(0, prices, horizons=(2,))
        assert result[2] == pytest.approx(-0.20, abs=1e-9)

    def test_returns_none_beyond_series_end(self):
        prices = [100.0, 105.0, 110.0]
        result = _compute_forward_returns(0, prices, horizons=(2, 5))
        assert result[2] == pytest.approx(0.10, abs=1e-9)
        assert result[5] is None

    def test_end_of_series_all_none(self):
        prices = [100.0, 110.0]
        result = _compute_forward_returns(1, prices, horizons=(1, 3))
        assert result[1] is None
        assert result[3] is None

    def test_mid_series_returns_correct(self):
        # Base at index 2 (price=110), forward 1 (price=115)
        prices = [100.0, 105.0, 110.0, 115.0, 120.0]
        result = _compute_forward_returns(2, prices, horizons=(1,))
        assert result[1] == pytest.approx(5.0 / 110.0, abs=1e-9)


# ===========================================================================
# find_price_analogues
# ===========================================================================


class TestFindPriceAnalogues:
    # ------------------------------------------------------------------
    # Basic output shape
    # ------------------------------------------------------------------

    def test_returns_at_most_top_k(self):
        n = 300
        dates = _make_dates(n)
        prices = _rw_prices(n)
        results = find_price_analogues(dates, prices, window_size=21, top_k=3)
        assert len(results) <= 3

    def test_returns_scored_window_instances(self):
        n = 200
        dates = _make_dates(n)
        prices = _rw_prices(n)
        results = find_price_analogues(dates, prices, window_size=21, top_k=5)
        for r in results:
            assert isinstance(r, ScoredWindow)

    def test_similarity_scores_in_range(self):
        n = 300
        dates = _make_dates(n)
        prices = _rw_prices(n)
        for r in find_price_analogues(dates, prices, window_size=21, top_k=5):
            assert 0.0 <= r.similarity <= 1.0

    def test_sorted_by_descending_similarity(self):
        n = 300
        dates = _make_dates(n)
        prices = _rw_prices(n)
        results = find_price_analogues(dates, prices, window_size=21, top_k=5)
        sims = [r.similarity for r in results]
        assert sims == sorted(sims, reverse=True)

    # ------------------------------------------------------------------
    # Forward returns
    # ------------------------------------------------------------------

    def test_forward_return_keys_match_horizons(self):
        n = 300
        dates = _make_dates(n)
        prices = _rw_prices(n)
        horizons = (10, 20, 40)
        results = find_price_analogues(
            dates, prices, window_size=21, top_k=3, forward_horizons=horizons
        )
        for r in results:
            assert set(r.forward_returns.keys()) == set(horizons)

    def test_forward_returns_none_at_series_end(self):
        """The last historical window may not have data 126 days out."""
        n = 250
        dates = _make_dates(n)
        prices = _rw_prices(n)
        results = find_price_analogues(
            dates, prices, window_size=21, top_k=5, forward_horizons=(21, 63, 126)
        )
        # At least some windows near the end should have None for the 126-day horizon
        # (We can't guarantee which ones, but we can verify the code doesn't crash)
        for r in results:
            for v in r.forward_returns.values():
                assert v is None or isinstance(v, float)

    # ------------------------------------------------------------------
    # min_gap_days enforcement
    # ------------------------------------------------------------------

    def test_min_gap_days_no_two_results_within_gap(self):
        n = 500
        dates = _make_dates(n)
        prices = _rw_prices(n, seed=77)
        min_gap = 90
        results = find_price_analogues(
            dates, prices, window_size=21, top_k=5, min_gap_days=min_gap
        )
        for i, a in enumerate(results):
            for b in results[i + 1:]:
                gap = _days_between(a.window.end_date, b.window.end_date)
                assert gap >= min_gap, (
                    f"Windows {a.window.end_date} and {b.window.end_date} "
                    f"are only {gap} days apart (min_gap={min_gap})"
                )

    # ------------------------------------------------------------------
    # Sanity / correctness
    # ------------------------------------------------------------------

    def test_injected_copy_is_best_analogue(self):
        """
        Inject an exact copy of the current window far back in history.
        It should be found as the top analogue.
        """
        ws = 30
        base_n = 400
        dates = _make_dates(base_n + ws)
        prices = _rw_prices(base_n + ws, seed=10)

        # The current window is the last *ws* bars
        current_window_prices = prices[-ws:]

        # Inject that exact pattern at position 50 (far from the current window)
        inject_start = 50
        prices_injected = prices[::]  # copy
        for i in range(ws):
            prices_injected[inject_start + i] = current_window_prices[i]

        results = find_price_analogues(
            dates, prices_injected, window_size=ws, top_k=3, min_gap_days=30
        )
        assert len(results) > 0
        top = results[0]
        assert top.similarity == pytest.approx(1.0, abs=1e-9)
        # The injected window should start on the right date
        assert top.window.start_date == dates[inject_start]

    def test_dtw_method_accepted(self):
        """find_price_analogues works with method='dtw' without error."""
        n = 200
        dates = _make_dates(n)
        prices = _rw_prices(n)
        results = find_price_analogues(
            dates, prices, window_size=21, top_k=3, method="dtw"
        )
        assert len(results) <= 3

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_minimum_viable_series(self):
        """Series just barely long enough: 2 * window_size bars."""
        ws = 10
        n = 2 * ws
        dates = _make_dates(n)
        prices = _rw_prices(n)
        results = find_price_analogues(dates, prices, window_size=ws, top_k=5)
        # Should produce exactly 1 non-overlapping historical window
        assert len(results) == 1

    def test_series_too_short_raises(self):
        ws = 10
        n = ws + 1  # only ws+1 bars; need at least 2*ws
        dates = _make_dates(n)
        prices = _rw_prices(n)
        with pytest.raises(ValueError):
            find_price_analogues(dates, prices, window_size=ws, top_k=3)

    def test_forward_return_value_correct(self):
        """
        Build a simple linear price series and verify a specific forward return.
        """
        # 100 prices: 1, 2, ..., 100 (linear)
        n = 100
        prices = list(range(1, n + 1))
        dates = _make_dates(n)

        ws = 10
        results = find_price_analogues(
            dates, prices, window_size=ws, top_k=1,
            min_gap_days=0, forward_horizons=(5,)
        )
        assert len(results) == 1
        sw = results[0]
        h = 5
        end_idx = sw.window.start_idx + ws - 1
        expected_base = prices[end_idx]
        expected_future = prices[end_idx + h] if end_idx + h < n else None
        if expected_future is not None:
            expected_ret = (expected_future - expected_base) / abs(expected_base)
            assert sw.forward_returns[h] == pytest.approx(expected_ret, abs=1e-9)

    def test_window_prices_match_source_series(self):
        """Prices in the returned windows must match the original series."""
        n = 200
        prices = _rw_prices(n)
        dates = _make_dates(n)
        ws = 21
        results = find_price_analogues(dates, prices, window_size=ws, top_k=5)
        for sw in results:
            start = sw.window.start_idx
            expected = prices[start : start + ws]
            assert sw.window.prices == pytest.approx(expected, abs=1e-9)

    def test_no_current_window_overlap_in_results(self):
        """None of the returned windows should overlap with the current window."""
        n = 400
        dates = _make_dates(n)
        prices = _rw_prices(n)
        ws = 63
        results = find_price_analogues(dates, prices, window_size=ws, top_k=5)
        current_start_date = dates[n - ws]
        for sw in results:
            # The analogue's window must end before the current window starts
            assert sw.window.end_date < current_start_date, (
                f"Analogue end {sw.window.end_date} overlaps with current "
                f"window start {current_start_date}"
            )
