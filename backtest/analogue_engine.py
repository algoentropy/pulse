"""
Historical Analogues engine for the Pulse macro-market dashboard.

Finds past time periods most similar to current market conditions using a
weighted combination of:
  1. Macro feature cosine similarity  — the macro regime (rates, vol, FX, etc.)
  2. S&P 500 price trajectory DTW     — the recent price pattern / momentum shape

Public API
----------
  find_analogues(
      n                   = 5,
      date                = None,          # ISO string or None -> most recent
      lookback_days       = 63,            # price window in trading days
      min_separation_days = 126,           # min days between returned matches
      macro_weight        = 0.6,
      price_weight        = 0.4,
      price_prefilter_k   = 200,           # DTW only run on top-k macro candidates
  ) -> list[AnalogueResult]

  fetch_analogues(**kwargs) -> list[dict]  # cached, JSON-serialisable

Algorithm
---------
  1. Load macro_features.parquet (produced by data_pipeline.build_features).
  2. Resolve query date (latest row or caller-supplied date).
  3. Compute all macro cosine similarities in one vectorised pass (O(T*F)).
  4. Pre-filter top `price_prefilter_k` candidates by macro score only.
  5. For each pre-filtered candidate run DTW on the S&P 500 price window.
  6. Merge into combined_score via combine_similarity_scores.
  7. Greedy minimum-separation selection to avoid overlapping windows.
  8. Return top-N AnalogueResult objects with forward returns & feature drivers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.price_similarity import compute_price_similarity
from backtest.similarity import combine_similarity_scores, compute_macro_similarity

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE_PATH = Path(__file__).parent
_FEATURES_PATH = _BASE_PATH / "macro_features.parquet"
_DB_PATH = _BASE_PATH / "market_data.db"
_CACHE_DIR = _BASE_PATH.parent / "pulse_cache"
_CACHE_FILE = _CACHE_DIR / "analogues.json"
_CACHE_TTL_SECONDS = 30 * 60  # 30 minutes

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SP500_TICKER = "^GSPC"

# Columns that must never appear in the macro feature vector
_EXCLUDE_PATTERNS = ("future", "target")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AnalogueResult:
    """A single historical analogue match with full metadata."""

    # Window identification
    start_date: str     # ISO date — first day of the lookback window
    end_date: str       # ISO date — last day of the lookback window (match point)

    # Similarity scores (all in [0, 1]; higher = more similar)
    macro_score: float    # Normalised cosine similarity of macro feature vectors
    price_score: float    # DTW shape-similarity of S&P 500 price trajectories
    combined_score: float # Weighted combination of the two scores

    # What happened AFTER this analogue period (S&P 500 total return)
    forward_ret_5d: float | None   # ~1 trading week ahead
    forward_ret_21d: float | None  # ~1 month ahead
    forward_ret_63d: float | None  # ~3 months ahead

    # Interpretability: features that drove the macro match (top by cosine contribution)
    top_features: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "macro_score": self.macro_score,
            "price_score": self.price_score,
            "combined_score": self.combined_score,
            "forward_ret_5d": self.forward_ret_5d,
            "forward_ret_21d": self.forward_ret_21d,
            "forward_ret_63d": self.forward_ret_63d,
            "top_features": self.top_features,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature column names, excluding targets and forward-return cols."""
    return [
        col
        for col in df.columns
        if not any(pat in col.lower() for pat in _EXCLUDE_PATTERNS)
    ]


def _load_sp500_prices_raw(df: pd.DataFrame) -> pd.Series:
    """
    Load S&P 500 raw close prices.

    Tries market_data.db first; falls back to reconstructing a cumulative
    return series from ``^GSPC_ret_1d`` in the feature matrix.
    The returned series is aligned to the feature matrix index.
    """
    if _DB_PATH.exists():
        try:
            import sqlite3

            conn = sqlite3.connect(str(_DB_PATH))
            prices = pd.read_sql(
                f'SELECT Date, "{_SP500_TICKER}" FROM raw_prices ORDER BY Date',
                conn,
                index_col="Date",
                parse_dates=["Date"],
            )[_SP500_TICKER].dropna()
            conn.close()
            # Reindex to match the feature matrix (inner join on dates)
            prices = prices.reindex(df.index).ffill()
            if not prices.isna().all():
                return prices
        except Exception:
            pass

    # Fallback: reconstruct cumulative prices from daily returns
    ret_col = f"{_SP500_TICKER}_ret_1d"
    if ret_col not in df.columns:
        raise ValueError(
            f"S&P 500 return column {ret_col!r} not found in feature matrix "
            "and market_data.db is unavailable."
        )
    returns = df[ret_col].fillna(0.0)
    return (1.0 + returns).cumprod()


def _price_window(prices: np.ndarray, end_idx: int, window: int) -> np.ndarray | None:
    """
    Extract a relative (percent-from-start) price window.

    Parameters
    ----------
    prices:
        1-D array of prices aligned to the feature matrix rows.
    end_idx:
        Row index of the window's last element.
    window:
        Number of data points in the window.

    Returns
    -------
    numpy array of length `window`, or None if there is not enough data.
    """
    start_idx = end_idx - window + 1
    if start_idx < 0:
        return None
    segment = prices[start_idx : end_idx + 1]
    if len(segment) < 2 or np.isnan(segment).any():
        return None
    return segment


def _batch_cosine_similarities(
    query_vec: np.ndarray,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarities between `query_vec` and every row of
    `feature_matrix` in a single vectorised pass.

    Parameters
    ----------
    query_vec:
        Shape (F,) — the macro feature vector for the query date.
    feature_matrix:
        Shape (T, F) — historical macro feature vectors, one per row.

    Returns
    -------
    numpy array of shape (T,) with cosine similarities in [-1, 1].
    Rows where the norm is zero receive 0.0 (undefined similarity).
    """
    q_norm = float(np.linalg.norm(query_vec))
    if q_norm == 0.0:
        return np.zeros(len(feature_matrix))

    q_unit = query_vec / q_norm

    # Row norms
    row_norms = np.linalg.norm(feature_matrix, axis=1)  # (T,)
    valid = row_norms > 0.0

    cosines = np.zeros(len(feature_matrix))
    cosines[valid] = (feature_matrix[valid] @ q_unit) / row_norms[valid]

    # Clamp for floating-point rounding
    return np.clip(cosines, -1.0, 1.0)


def _top_feature_drivers(
    query_vec: np.ndarray,
    hist_vec: np.ndarray,
    feat_cols: list[str],
    top_k: int = 5,
) -> list[dict]:
    """
    Identify which features drive the cosine similarity between the query
    vector and a historical candidate.

    The per-feature cosine contribution is:
        contribution_i = q_norm_i * h_norm_i
    where q_norm and h_norm are the L2-normalised versions of each vector.

    A higher contribution means the feature is pointing in the same direction
    in both periods — i.e., it's a key reason why the periods are similar.

    Returns the top_k features sorted by contribution descending, with
    query and historical values included for interpretability.
    """
    q_norm_val = float(np.linalg.norm(query_vec))
    h_norm_val = float(np.linalg.norm(hist_vec))

    if q_norm_val == 0.0 or h_norm_val == 0.0:
        return []

    q_unit = query_vec / q_norm_val
    h_unit = hist_vec / h_norm_val

    contributions = q_unit * h_unit  # element-wise; sums to cosine similarity

    # Sort descending by absolute contribution (most impactful features first)
    indices = np.argsort(contributions)[::-1][:top_k]

    return [
        {
            "feature": feat_cols[i],
            "query_value": round(float(query_vec[i]), 6),
            "hist_value": round(float(hist_vec[i]), 6),
            "cosine_contribution": round(float(contributions[i]), 6),
        }
        for i in indices
    ]


def _get_forward_return(
    df: pd.DataFrame,
    end_row_idx: int,
    col: str,
    fwd_days: int,
) -> float | None:
    """
    Retrieve the forward return of S&P 500 starting from ``end_row_idx``.

    ``^GSPC_ret_5d`` at row i+5 is ``(price[i+5] - price[i]) / price[i]``,
    which is exactly the total return from row i forward 5 days.
    """
    fwd_idx = end_row_idx + fwd_days
    if fwd_idx >= len(df) or col not in df.columns:
        return None
    val = float(df.iloc[fwd_idx][col])
    return None if np.isnan(val) else val


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_analogues(
    n: int = 5,
    date: str | None = None,
    lookback_days: int = 63,
    min_separation_days: int = 126,
    macro_weight: float = 0.6,
    price_weight: float = 0.4,
    price_prefilter_k: int = 200,
) -> list[AnalogueResult]:
    """
    Find the top-N historical periods most similar to current market conditions.

    Parameters
    ----------
    n:
        Number of top analogues to return (default 5).
    date:
        ISO date string for the query (e.g. "2022-10-15").  Defaults to the
        most recent available row in the feature matrix.
    lookback_days:
        Length of the S&P 500 price trajectory window in trading days (default 63).
        Must be at least 2.  The same window length is used for the current period
        and every historical candidate.
    min_separation_days:
        Minimum number of trading days separating the end dates of any two
        returned analogues.  Prevents near-identical overlapping periods from
        both appearing in the top-N (default 126, approx 6 months).
    macro_weight:
        Relative weight for the macro-state component (default 0.6).
        Normalised against ``price_weight`` before combining, so raw importance
        values (e.g. 3 / 1) work equally well.
    price_weight:
        Relative weight for the price-trajectory component (default 0.4).
    price_prefilter_k:
        How many candidates to retain after the fast macro-only pass before
        running the more expensive DTW price similarity (default 200).

    Returns
    -------
    list[AnalogueResult]
        Top-N analogues sorted by combined_score descending.

    Raises
    ------
    FileNotFoundError
        If ``backtest/macro_features.parquet`` does not exist.
    ValueError
        If the feature matrix has insufficient history for the requested
        lookback window.
    """
    if lookback_days < 2:
        raise ValueError(f"lookback_days must be >= 2, got {lookback_days}")

    # ------------------------------------------------------------------
    # 1. Load feature matrix
    # ------------------------------------------------------------------
    if not _FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Macro features file not found: {_FEATURES_PATH}. "
            "Run POST /api/train to build the feature matrix first."
        )
    df = pd.read_parquet(_FEATURES_PATH)

    # ------------------------------------------------------------------
    # 2. Resolve query date
    # ------------------------------------------------------------------
    if date is not None:
        target_dt = pd.to_datetime(date)
        df = df.loc[:target_dt]
        if df.empty:
            raise ValueError(f"No feature data available on or before {date!r}.")

    query_idx = len(df) - 1  # index of the "current" row

    if query_idx < lookback_days:
        raise ValueError(
            f"Insufficient history: lookback_days={lookback_days} but only "
            f"{query_idx + 1} rows available."
        )

    feat_cols = _feature_cols(df)

    # ------------------------------------------------------------------
    # 3. Query state
    # ------------------------------------------------------------------
    query_vec: np.ndarray = df.iloc[query_idx][feat_cols].values.astype(float)
    query_date_str = pd.Timestamp(df.index[query_idx]).strftime("%Y-%m-%d")

    # Load S&P 500 prices for the price window
    sp500_prices: np.ndarray = _load_sp500_prices_raw(df).values.astype(float)

    query_price_window = _price_window(sp500_prices, query_idx, lookback_days)
    if query_price_window is None:
        raise ValueError(
            f"Could not extract S&P 500 price window of {lookback_days} days "
            f"ending at query date {query_date_str}."
        )

    # ------------------------------------------------------------------
    # 4. Build candidate pool (all historical rows with sufficient lookback)
    # ------------------------------------------------------------------
    # Candidates run from row `lookback_days` (first row with a full window)
    # up to row `query_idx - 1` (exclude the current period itself).
    candidate_end = query_idx - 1
    first_valid = lookback_days  # inclusive

    if candidate_end < first_valid:
        raise ValueError(
            "Not enough historical data to find any analogues. "
            f"Need at least {lookback_days + 1} rows."
        )

    candidate_indices = np.arange(first_valid, candidate_end + 1)  # [first_valid, query_idx)

    # Feature matrix for all candidates — shape (T, F)
    feature_matrix: np.ndarray = df.iloc[first_valid : candidate_end + 1][feat_cols].values.astype(float)

    # ------------------------------------------------------------------
    # 5. Vectorised macro cosine similarity for ALL candidates (fast)
    # ------------------------------------------------------------------
    cosine_scores = _batch_cosine_similarities(query_vec, feature_matrix)
    macro_scores_all = (cosine_scores + 1.0) / 2.0  # map [-1,1] -> [0,1]

    # ------------------------------------------------------------------
    # 6. Pre-filter: keep top price_prefilter_k candidates by macro score
    # ------------------------------------------------------------------
    k = min(price_prefilter_k, len(candidate_indices))
    top_macro_local = np.argpartition(macro_scores_all, -k)[-k:]  # local indices
    # Sort these k candidates by macro score descending for determinism
    top_macro_local = top_macro_local[np.argsort(macro_scores_all[top_macro_local])[::-1]]

    # ------------------------------------------------------------------
    # 7. DTW price similarity for pre-filtered candidates
    # ------------------------------------------------------------------
    weights: dict[str, float] = {"macro": macro_weight, "price": price_weight}

    scored: list[tuple[float, float, float, int]] = []  # (combined, macro, price, global_idx)

    for local_idx in top_macro_local:
        global_idx = int(candidate_indices[local_idx])
        macro_s = float(macro_scores_all[local_idx])

        # Extract price window for this candidate
        hist_price_window = _price_window(sp500_prices, global_idx, lookback_days)
        if hist_price_window is None:
            continue

        # DTW similarity
        price_s = compute_price_similarity(
            query_price_window, hist_price_window, method="dtw"
        )

        combined = combine_similarity_scores(macro_s, price_s, weights)
        scored.append((combined, macro_s, price_s, global_idx))

    if not scored:
        return []

    # Sort by combined score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # ------------------------------------------------------------------
    # 8. Greedy minimum-separation selection
    # ------------------------------------------------------------------
    selected: list[tuple[float, float, float, int]] = []
    selected_end_indices: list[int] = []

    for combined, macro_s, price_s, idx in scored:
        too_close = any(
            abs(idx - sel_idx) < min_separation_days
            for sel_idx in selected_end_indices
        )
        if not too_close:
            selected.append((combined, macro_s, price_s, idx))
            selected_end_indices.append(idx)
        if len(selected) >= n:
            break

    # ------------------------------------------------------------------
    # 9. Build AnalogueResult objects
    # ------------------------------------------------------------------
    sp500_ret_5d_col  = f"{_SP500_TICKER}_ret_5d"
    sp500_ret_21d_col = f"{_SP500_TICKER}_ret_21d"
    sp500_ret_63d_col = f"{_SP500_TICKER}_ret_63d"

    results: list[AnalogueResult] = []

    for combined_score, macro_score, price_score, end_row_idx in selected:
        end_date_str   = pd.Timestamp(df.index[end_row_idx]).strftime("%Y-%m-%d")
        start_row_idx  = max(0, end_row_idx - lookback_days + 1)
        start_date_str = pd.Timestamp(df.index[start_row_idx]).strftime("%Y-%m-%d")

        hist_vec = df.iloc[end_row_idx][feat_cols].values.astype(float)

        # Forward returns: ret_Nd at row (end_row_idx + N) equals the return
        # from end_row_idx to end_row_idx+N
        fwd_5d  = _get_forward_return(df, end_row_idx, sp500_ret_5d_col,  5)
        fwd_21d = _get_forward_return(df, end_row_idx, sp500_ret_21d_col, 21)
        fwd_63d = _get_forward_return(df, end_row_idx, sp500_ret_63d_col, 63)

        top_feats = _top_feature_drivers(query_vec, hist_vec, feat_cols, top_k=5)

        results.append(
            AnalogueResult(
                start_date=start_date_str,
                end_date=end_date_str,
                macro_score=round(macro_score, 4),
                price_score=round(price_score, 4),
                combined_score=round(combined_score, 4),
                forward_ret_5d=round(fwd_5d, 6) if fwd_5d is not None else None,
                forward_ret_21d=round(fwd_21d, 6) if fwd_21d is not None else None,
                forward_ret_63d=round(fwd_63d, 6) if fwd_63d is not None else None,
                top_features=top_feats,
            )
        )

    return results


def fetch_analogues(
    force_refresh: bool = False,
    **kwargs,
) -> list[dict]:
    """
    Cached wrapper around ``find_analogues`` for the FastAPI endpoint.

    Results are stored in ``pulse_cache/analogues.json`` and refreshed
    automatically when the cache is older than 30 minutes or when
    ``force_refresh=True``.

    Parameters
    ----------
    force_refresh:
        Bypass cache and recompute.
    **kwargs:
        Passed through to ``find_analogues`` (n, date, lookback_days, etc.).

    Returns
    -------
    list[dict]
        JSON-serialisable list of analogue dicts (same fields as AnalogueResult).
    """
    cache_file = _CACHE_FILE
    cache_ttl  = _CACHE_TTL_SECONDS

    if not force_refresh and cache_file.exists():
        mtime = cache_file.stat().st_mtime
        if time.time() - mtime < cache_ttl:
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass  # Fall through to recompute

    analogues = find_analogues(**kwargs)
    result = [a.to_dict() for a in analogues]

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(result, f)

    return result
