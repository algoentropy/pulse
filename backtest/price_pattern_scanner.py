"""
Price pattern scanner for the Historical Analogues feature.

Extracts the current S&P 500 price pattern (a fixed-width trailing window of
close prices) and scans all historical windows of the same width to find the
most similar past periods.

Public API
----------
PriceWindow
    Dataclass describing one candidate window extracted from the price series.

ScoredWindow
    Dataclass pairing a PriceWindow with its similarity score and the forward
    returns observed after that window ended.

extract_price_windows(dates, prices, window_size, step) -> list[PriceWindow]
    Slide a fixed-width window across the series and return every complete
    window.

score_price_windows(current_prices, historical_windows, method)
    -> list[tuple[PriceWindow, float]]
    Score each historical window against the current pattern.  Returns the
    list sorted by descending similarity score.

find_price_analogues(dates, prices, ...) -> list[ScoredWindow]
    End-to-end: extract the current window, scan history, de-duplicate
    near-overlapping results, attach forward returns, and return the top-K
    analogues.

Notes
-----
- Default comparison method is "pearson": O(n) per comparison, sub-second for
  a full 15-year scan.  Use method="dtw" for robustness to temporal offsets
  at the cost of ~8x more compute (~8s for a 15-year scan at window_size=63).
- The "current" window is always the most recent *window_size* trading days in
  the supplied series.  Any historical window that overlaps with the current
  window is excluded from the candidate set.
- After scoring, a greedy de-duplication pass enforces a minimum calendar-day
  gap between returned analogues so that the results span distinct market
  episodes rather than adjacent days of the same episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from backtest.price_similarity import compute_price_similarity


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PriceWindow:
    """A fixed-width slice of a price series."""

    start_date: str          # "YYYY-MM-DD" — first day in the window
    end_date: str            # "YYYY-MM-DD" — last day in the window
    prices: list[float]      # raw close prices, length == window_size
    start_idx: int           # position of the first bar in the source series


@dataclass
class ScoredWindow:
    """A historical PriceWindow annotated with its similarity to the current
    pattern and the forward returns observed after the window closed."""

    window: PriceWindow
    similarity: float        # [0, 1]; 1.0 = shape-identical to current window

    # Keyed by horizon in trading days; value is the fractional return
    # (e.g. 0.05 = +5 %) or None when the series ends before that horizon.
    forward_returns: dict[int, float | None] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------


def extract_price_windows(
    dates: Sequence[str],
    prices: Sequence[float],
    window_size: int = 63,
    step: int = 1,
) -> list[PriceWindow]:
    """
    Slide a fixed-width window over a price series and return every complete
    window.

    Parameters
    ----------
    dates:
        Ordered sequence of date strings ("YYYY-MM-DD"), one per bar.
        Must have the same length as *prices*.
    prices:
        Ordered sequence of close prices, one per bar.
    window_size:
        Number of trading days in each window.  Default is 63 (≈ 3 months).
    step:
        Number of bars to advance the window on each iteration.  Default is 1
        (every possible starting position).  Increase to speed up scanning at
        the cost of resolution.

    Returns
    -------
    list[PriceWindow]
        All complete windows, in chronological order (oldest first).

    Raises
    ------
    ValueError
        If *window_size* < 2, or if *dates* and *prices* have different
        lengths, or if the series is shorter than *window_size*.
    """
    dates_list = list(dates)
    prices_list = list(prices)

    if len(dates_list) != len(prices_list):
        raise ValueError(
            f"dates and prices must have the same length; "
            f"got {len(dates_list)} dates and {len(prices_list)} prices"
        )
    if window_size < 2:
        raise ValueError(f"window_size must be >= 2, got {window_size}")
    if len(prices_list) < window_size:
        raise ValueError(
            f"Series length ({len(prices_list)}) is shorter than "
            f"window_size ({window_size})"
        )
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    n = len(prices_list)
    windows: list[PriceWindow] = []

    for start in range(0, n - window_size + 1, step):
        end = start + window_size - 1  # inclusive index
        windows.append(
            PriceWindow(
                start_date=dates_list[start],
                end_date=dates_list[end],
                prices=prices_list[start : end + 1],
                start_idx=start,
            )
        )

    return windows


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_price_windows(
    current_prices: Sequence[float] | np.ndarray,
    historical_windows: list[PriceWindow],
    method: Literal["dtw", "pearson"] = "pearson",
) -> list[tuple[PriceWindow, float]]:
    """
    Score each historical window against the current price pattern.

    Parameters
    ----------
    current_prices:
        The current (most recent) price window — an ordered sequence of close
        prices of the same length as each PriceWindow in *historical_windows*.
    historical_windows:
        Candidate windows returned by :func:`extract_price_windows`.
    method:
        Similarity method passed to :func:`compute_price_similarity`:
        ``"pearson"`` (fast, default) or ``"dtw"`` (slower, time-warp aware).

    Returns
    -------
    list[tuple[PriceWindow, float]]
        Each element is (window, similarity_score).  The list is sorted by
        descending similarity (best match first).

    Raises
    ------
    ValueError
        If *current_prices* has fewer than 2 points, or if
        *historical_windows* is empty.
    """
    current = list(current_prices)
    if len(current) < 2:
        raise ValueError(
            f"current_prices must have at least 2 points, got {len(current)}"
        )
    if not historical_windows:
        raise ValueError("historical_windows must not be empty")

    scored: list[tuple[PriceWindow, float]] = []
    for win in historical_windows:
        sim = compute_price_similarity(current, win.prices, method=method)
        scored.append((win, sim))

    # Sort best-first
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Forward-return helper
# ---------------------------------------------------------------------------


def _compute_forward_returns(
    end_idx: int,
    prices: list[float],
    horizons: tuple[int, ...],
) -> dict[int, float | None]:
    """
    Compute percentage returns from *prices[end_idx]* to *prices[end_idx + h]*
    for each horizon *h* in *horizons*.

    Returns None for any horizon where insufficient future data exists.
    """
    base = prices[end_idx]
    result: dict[int, float | None] = {}
    for h in horizons:
        future_idx = end_idx + h
        if future_idx < len(prices) and base != 0.0:
            result[h] = (prices[future_idx] - base) / abs(base)
        else:
            result[h] = None
    return result


# ---------------------------------------------------------------------------
# De-duplication
# ---------------------------------------------------------------------------


def _days_between(date_a: str, date_b: str) -> int:
    """Return the absolute number of calendar days between two 'YYYY-MM-DD' strings."""
    from datetime import date

    y1, m1, d1 = (int(x) for x in date_a.split("-"))
    y2, m2, d2 = (int(x) for x in date_b.split("-"))
    delta = date(y2, m2, d2) - date(y1, m1, d1)
    return abs(delta.days)


def _deduplicate(
    scored: list[tuple[PriceWindow, float]],
    top_k: int,
    min_gap_days: int,
) -> list[tuple[PriceWindow, float]]:
    """
    Greedily select up to *top_k* windows from *scored* (best-first) such
    that every pair of selected windows is at least *min_gap_days* calendar
    days apart (measured between their *end_date*s).
    """
    selected: list[tuple[PriceWindow, float]] = []

    for window, sim in scored:
        # Check gap against all already-selected windows
        too_close = any(
            _days_between(window.end_date, sel_win.end_date) < min_gap_days
            for sel_win, _ in selected
        )
        if not too_close:
            selected.append((window, sim))
            if len(selected) >= top_k:
                break

    return selected


# ---------------------------------------------------------------------------
# End-to-end entry point
# ---------------------------------------------------------------------------


def find_price_analogues(
    dates: Sequence[str],
    prices: Sequence[float],
    window_size: int = 63,
    top_k: int = 5,
    min_gap_days: int = 90,
    forward_horizons: tuple[int, ...] = (21, 63, 126),
    method: Literal["dtw", "pearson"] = "pearson",
    exclude_last_days: int | None = None,
) -> list[ScoredWindow]:
    """
    Find the top-K historical price windows most similar to the current pattern.

    The "current" pattern is the trailing *window_size* trading days of the
    supplied series.  Every historical window that would overlap with the
    current window is excluded from the search.

    Parameters
    ----------
    dates:
        Ordered sequence of "YYYY-MM-DD" strings, one per bar.
    prices:
        Corresponding close prices, same length as *dates*.
    window_size:
        Length of the comparison window in trading days.  Default 63 (≈ 3 months).
    top_k:
        Maximum number of analogues to return.
    min_gap_days:
        Minimum calendar-day gap between the *end_date*s of any two returned
        analogues.  Prevents the result set from being dominated by adjacent
        dates in the same episode.  Default 90 days (≈ 1 quarter).
    forward_horizons:
        Tuple of trading-day horizons at which forward returns are computed.
        Default (21, 63, 126) ≈ 1-month, 3-month, 6-month.
    method:
        Price similarity method — ``"pearson"`` (default, fast) or ``"dtw"``
        (time-warp aware, ~8× slower).
    exclude_last_days:
        How many trailing bars to exclude from the historical candidate set.
        Defaults to *window_size* (the current window itself).  Increase to
        also exclude the run-up to the current window.

    Returns
    -------
    list[ScoredWindow]
        Up to *top_k* analogues, sorted by descending similarity score.
        Each ScoredWindow includes:
          - .window.start_date / .window.end_date — the analogue period
          - .window.prices                         — raw close prices
          - .similarity                            — [0, 1] shape-match score
          - .forward_returns                       — {horizon: pct_return | None}

    Raises
    ------
    ValueError
        If the series is too short to contain a current window plus at least
        one non-overlapping historical window.
    """
    dates_list = list(dates)
    prices_list = list(prices)

    n = len(prices_list)

    if exclude_last_days is None:
        exclude_last_days = window_size

    if n < window_size + window_size:
        raise ValueError(
            f"Series has only {n} bars; need at least {window_size * 2} "
            f"to compare the current window against at least one historical window."
        )

    # --- Extract the current (most recent) window ---
    current_start = n - window_size
    current_prices = prices_list[current_start:]

    # --- Extract all historical windows (excluding current & overlap) ---
    # The candidate pool ends at (n - exclude_last_days - window_size + 1)
    # so that no candidate window's bars overlap with the excluded trailing period.
    max_hist_start = n - exclude_last_days - window_size
    if max_hist_start < 0:
        raise ValueError(
            f"Not enough historical data to find non-overlapping windows. "
            f"Series length={n}, window_size={window_size}, "
            f"exclude_last_days={exclude_last_days}"
        )

    # Trim series to the historical candidate region
    hist_dates = dates_list[: max_hist_start + window_size]
    hist_prices = prices_list[: max_hist_start + window_size]

    historical_windows = extract_price_windows(
        hist_dates, hist_prices, window_size=window_size, step=1
    )

    # --- Score all historical windows ---
    scored = score_price_windows(current_prices, historical_windows, method=method)

    # --- De-duplicate: keep windows that are min_gap_days apart ---
    top_scored = _deduplicate(scored, top_k=top_k, min_gap_days=min_gap_days)

    # --- Attach forward returns ---
    result: list[ScoredWindow] = []
    for win, sim in top_scored:
        # end_idx in the *full* prices list
        end_idx = win.start_idx + window_size - 1
        fwd = _compute_forward_returns(end_idx, prices_list, forward_horizons)
        result.append(ScoredWindow(window=win, similarity=sim, forward_returns=fwd))

    return result
