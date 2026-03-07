"""
Historical Analogues — macro feature vector builder and cache layer.

Builds and caches a standardised macro feature vector matrix derived from
macro_features.parquet.  Each row in the matrix is a z-scored snapshot of
market conditions on that trading day.  This matrix is the primary input for
analogue search: cosine similarity between the current row and every
historical row identifies periods whose macro regime most resembles today.

Feature selection
-----------------
All ticker features are included EXCEPT sub-day noise (1-day returns), which
are too sensitive to short-term events to characterise a macro regime.  The
selected set is:

    - {ticker}_ret_5d      (5-day momentum across all 14 tickers)
    - {ticker}_ret_21d     (1-month momentum)
    - {ticker}_ret_63d     (3-month momentum)
    - {ticker}_vol_21d     (rolling realised volatility)
    - macro_copper_gold_ratio          (risk-appetite proxy, absolute level)
    - macro_copper_gold_ratio_ret_21d  (momentum of that ratio)
    - macro_vix_tnx_ratio              (fear / rate stress, absolute level)
    - macro_vix_tnx_ratio_ret_21d      (momentum of that ratio)

This yields 68 features (14 tickers × 4 + 4 composite) per trading day.

Standardisation
---------------
All features are z-scored using the full-history mean and standard deviation
so that absolute level differences (e.g. BTC vol >> VIX vol) don't dominate
cosine similarity and so that the current environment is fairly compared
against every historical period.

Cache
-----
The standardised matrix is written to:
    pulse_cache/analogue_vectors.parquet

It is automatically regenerated whenever macro_features.parquet is newer
than the cache file, or when force_refresh=True.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BACKTEST_DIR = Path(__file__).parent
_FEATURES_PATH = _BACKTEST_DIR / "macro_features.parquet"
_CACHE_PATH = _BACKTEST_DIR.parent / "pulse_cache" / "analogue_vectors.parquet"

# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

# Column name substrings that mark non-regime features to exclude.
# "1d" returns carry intraday/overnight noise rather than regime signal.
# "future" and "target" are forward-looking labels used during model training.
_EXCLUDE_PATTERNS: tuple[str, ...] = ("_ret_1d", "target", "future")


def get_feature_cols(df: pd.DataFrame | None = None) -> list[str]:
    """
    Return the ordered list of macro feature column names used for analogue
    comparison.

    Parameters
    ----------
    df:
        The raw macro features DataFrame.  If None the parquet file is loaded
        automatically.  Pass an already-loaded DataFrame to avoid double I/O.

    Returns
    -------
    list[str]
        Column names from macro_features.parquet that survive the exclusion
        filter, in their original order.

    Raises
    ------
    FileNotFoundError
        If df is None and macro_features.parquet does not exist.
    """
    if df is None:
        if not _FEATURES_PATH.exists():
            raise FileNotFoundError(
                f"macro_features.parquet not found at {_FEATURES_PATH}. "
                "Run POST /api/train to build the feature matrix first."
            )
        df = pd.read_parquet(_FEATURES_PATH)

    return [
        col
        for col in df.columns
        if not any(pattern in col for pattern in _EXCLUDE_PATTERNS)
    ]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_is_fresh() -> bool:
    """
    Return True when the analogue vector cache exists and is at least as new
    as the source macro_features.parquet file.

    Uses file modification timestamps — no TTL clock needed because the
    vectors only change when the underlying feature matrix changes.
    """
    if not _CACHE_PATH.exists() or not _FEATURES_PATH.exists():
        return False
    return _CACHE_PATH.stat().st_mtime >= _FEATURES_PATH.stat().st_mtime


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_analogue_vectors(force_refresh: bool = False) -> pd.DataFrame:
    """
    Build (or load from cache) the standardised macro feature vector matrix.

    Each row is a trading day; each column is a z-scored macro feature.
    The scaler is fitted on the entire available history so that the current
    row sits in the same feature space as every historical row.

    Parameters
    ----------
    force_refresh:
        When True, recompute the matrix from source even if the cache is
        already fresh.

    Returns
    -------
    pd.DataFrame
        - Index  : DatetimeIndex of trading days (same as macro_features.parquet)
        - Columns: selected macro feature names (see module docstring)
        - Values : z-scored floats

    Raises
    ------
    FileNotFoundError
        If macro_features.parquet does not exist.
    """
    if not force_refresh and _cache_is_fresh():
        try:
            return pd.read_parquet(_CACHE_PATH)
        except Exception:
            # Corrupt cache — fall through to rebuild
            pass

    if not _FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"macro_features.parquet not found at {_FEATURES_PATH}. "
            "Run POST /api/train to build the feature matrix first."
        )

    raw = pd.read_parquet(_FEATURES_PATH)
    feature_cols = get_feature_cols(raw)

    X = raw[feature_cols].copy()

    # Z-score each feature across the entire history.
    # fit_transform uses the global mean/std, so:
    #  - The current row is scaled by the same parameters as old rows.
    #  - A VIX reading of 80 maps to the same z-score whether it appears in
    #    2008, 2020, or today — making cross-era comparisons meaningful.
    scaler = StandardScaler()
    X_scaled_values = scaler.fit_transform(X)

    vectors = pd.DataFrame(
        X_scaled_values,
        index=X.index,
        columns=feature_cols,
    )

    # Persist to cache
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    vectors.to_parquet(_CACHE_PATH)

    return vectors


def get_current_vector(
    vectors: pd.DataFrame | None = None,
) -> tuple[np.ndarray, pd.Timestamp]:
    """
    Return the macro feature vector for the most recent trading day.

    Parameters
    ----------
    vectors:
        Pre-loaded standardised feature matrix from build_analogue_vectors().
        Pass this in to avoid a redundant disk read when you've already loaded
        the matrix in the calling scope.  If None, the matrix is loaded
        automatically.

    Returns
    -------
    tuple[np.ndarray, pd.Timestamp]
        - vector : 1-D float64 array of length ``len(get_feature_cols())``.
        - date   : The trading-day timestamp this vector represents.
    """
    if vectors is None:
        vectors = build_analogue_vectors()

    latest_row = vectors.iloc[-1]
    date = pd.Timestamp(vectors.index[-1])
    return latest_row.to_numpy(dtype=np.float64), date


def get_historical_vectors(
    vectors: pd.DataFrame | None = None,
    min_lookback_days: int = 63,
) -> pd.DataFrame:
    """
    Return the subset of the feature matrix eligible for analogue matching.

    Excludes the most recent *min_lookback_days* rows so that every candidate
    analogue date has at least that many subsequent trading days in the dataset.
    This ensures the "what happened next" forward price window is always
    available for display.

    Parameters
    ----------
    vectors:
        Pre-loaded standardised feature matrix from build_analogue_vectors().
        If None, the matrix is loaded automatically.
    min_lookback_days:
        Minimum forward trading-day horizon required after an analogue date.
        Default 63 (≈3 calendar months).  Set to 0 to disable exclusion.

    Returns
    -------
    pd.DataFrame
        Rows are valid analogue candidate dates; columns are z-scored macro
        features.  Always contains at least one row when the source data is
        long enough.
    """
    if vectors is None:
        vectors = build_analogue_vectors()

    if min_lookback_days > 0 and len(vectors) > min_lookback_days:
        return vectors.iloc[:-min_lookback_days]

    return vectors


# ---------------------------------------------------------------------------
# Scoring pipeline helpers
# ---------------------------------------------------------------------------


def _reconstruct_prices(rets: np.ndarray, start: float = 100.0) -> np.ndarray:
    """
    Convert a sequence of fractional daily returns into an indexed price series.

    price[0] = start
    price[i] = price[i-1] * (1 + rets[i])

    NaN returns are treated as 0 (no change) so the reconstruction stays finite.
    """
    safe = np.nan_to_num(rets, nan=0.0)
    prices = np.empty(len(safe), dtype=np.float64)
    prices[0] = start
    for i in range(1, len(safe)):
        prices[i] = prices[i - 1] * (1.0 + safe[i])
    return prices


def _macro_similarities_vec(
    ref_vec: np.ndarray,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    """
    Vectorised cosine similarity between *ref_vec* and every row of
    *feature_matrix*.

    Returns an array of shape (n_rows,) with values in [-1.0, 1.0].
    Rows with zero norm (and the reference, if zero) get similarity 0.0.
    """
    ref_norm = float(np.linalg.norm(ref_vec))
    if ref_norm == 0.0:
        return np.zeros(len(feature_matrix), dtype=np.float64)

    ref_unit = ref_vec / ref_norm

    row_norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)  # (n, 1)
    safe_norms = np.where(row_norms == 0.0, 1.0, row_norms)
    unit_matrix = feature_matrix / safe_norms  # (n, d)

    sims = unit_matrix @ ref_unit  # (n,)
    sims[row_norms.ravel() == 0.0] = 0.0
    return np.clip(sims, -1.0, 1.0)


def _pct_from_start(arr: np.ndarray) -> np.ndarray:
    """Express each element as fractional change from arr[0]."""
    first = float(arr[0])
    if first == 0.0:
        mean = float(np.mean(arr))
        denom = abs(mean) if mean != 0.0 else 1.0
        return (arr - first) / denom
    return (arr - first) / abs(first)


def _price_similarities_vec(
    ref_prices: np.ndarray,
    candidate_matrix: np.ndarray,
) -> np.ndarray:
    """
    Vectorised Pearson-correlation price similarity.

    Both *ref_prices* and every row of *candidate_matrix* are expressed as
    percent-change-from-start so that absolute price level and magnitude do
    not bias the shape comparison.

    Returns an array of shape (n_candidates,) with values in [0.0, 1.0].
    Flat series (zero std) get 0.5 (neutral); two flat series get 1.0.
    """
    ref_norm = _pct_from_start(ref_prices)
    ref_std = float(np.std(ref_norm))

    # Normalise candidate series
    first_col = candidate_matrix[:, 0:1]  # (n, 1)
    abs_first = np.abs(first_col)
    row_means = np.abs(np.mean(candidate_matrix, axis=1, keepdims=True))
    denom = np.where(
        abs_first == 0.0,
        np.where(row_means == 0.0, 1.0, row_means),
        abs_first,
    )
    cand_norm = (candidate_matrix - first_col) / denom  # (n, window)
    cand_stds = np.std(cand_norm, axis=1)  # (n,)

    if ref_std == 0.0:
        return np.where(cand_stds == 0.0, 1.0, 0.5).astype(np.float64)

    ref_centered = ref_norm - float(np.mean(ref_norm))  # (window,)
    cand_centered = cand_norm - np.mean(cand_norm, axis=1, keepdims=True)

    numerator = cand_centered @ ref_centered  # (n,)
    window = ref_prices.shape[0]
    denominator = cand_stds * ref_std * window

    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(cand_stds == 0.0, 0.5, numerator / denominator)

    r = np.clip(r, -1.0, 1.0)
    return (r + 1.0) / 2.0


def _top_feature_diffs(
    ref_vec: np.ndarray,
    cand_vec: np.ndarray,
    feat_names: list[str],
    top_k: int = 5,
) -> list[dict]:
    """
    Return the *top_k* features with the largest absolute difference between
    the reference and the candidate z-scored vectors.

    This highlights which regime dimensions are most different, providing
    interpretability for why each analogue was — or was not — selected.
    """
    diffs = np.abs(ref_vec - cand_vec)
    top_idx = np.argsort(diffs)[::-1][:top_k]
    return [
        {
            "feature": feat_names[i],
            "ref_value": round(float(ref_vec[i]), 6),
            "analogue_value": round(float(cand_vec[i]), 6),
            "abs_diff": round(float(diffs[i]), 6),
        }
        for i in top_idx
    ]


# ---------------------------------------------------------------------------
# Core scoring pipeline
# ---------------------------------------------------------------------------

_MAX_FWD_DAYS = 63  # maximum forward horizon needed for forward-return data


def find_analogues(
    n: int = 5,
    window: int = 63,
    macro_weight: float = 0.6,
    price_weight: float = 0.4,
    reference_date: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    price_method: Literal["pearson", "dtw"] = "pearson",
) -> dict:
    """
    Find the top-N historical periods most similar to the current market state.

    Similarity is a weighted combination of:
      * Macro regime similarity  — cosine similarity between z-scored feature
        vectors (vectors pre-standardised via :func:`build_analogue_vectors`).
      * Price trajectory similarity — Pearson correlation (default) or DTW on
        the trailing S&P 500 price window, normalised to percent-from-start.

    Parameters
    ----------
    n:
        Number of analogues to return (>= 1).
    window:
        Lookback window in trading days for the S&P 500 price comparison
        (~63 ≈ one quarter, ~21 ≈ one month).  The macro vector is always
        the single snapshot at the window's end date.
    macro_weight:
        Unnormalised weight for the macro-feature similarity component.
    price_weight:
        Unnormalised weight for the S&P 500 price-trajectory similarity.
    reference_date:
        ISO-8601 date string used as "today".  Defaults to the most recent
        row in the features parquet.
    date_from:
        Earliest candidate end-date to include.  Defaults to the first date
        with enough lookback history.
    date_to:
        Latest candidate end-date to include.  Defaults to one full *window*
        before the reference to prevent recent data from trivially dominating.
    price_method:
        ``"pearson"`` (fast, recommended) or ``"dtw"`` (handles temporal
        warping; uses a per-row loop and is considerably slower).

    Returns
    -------
    dict
        ``status``, ``reference_date``, ``window_days``, ``macro_weight``,
        ``price_weight``, ``n_candidates_scored``, ``analogues``,
        ``reference_prices``.

        Each analogue entry: ``rank``, ``start_date``, ``end_date``,
        ``combined_score``, ``macro_score``, ``price_score``,
        ``forward_returns`` (5d/21d/63d), ``sp500_prices`` (chart series),
        ``top_feature_diffs`` (top 5 most-different z-scored features).
    """
    from backtest.similarity import combine_similarity_scores

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    vectors = build_analogue_vectors()          # z-scored macro feature matrix
    raw = pd.read_parquet(_FEATURES_PATH)       # raw features (returns, prices)
    feat_cols = list(vectors.columns)
    n_rows = len(vectors)

    # Align raw to the same index as vectors (build_analogue_vectors may drop rows)
    raw = raw.loc[vectors.index]

    # ------------------------------------------------------------------
    # Resolve reference index
    # ------------------------------------------------------------------
    if reference_date is not None:
        ref_ts = pd.to_datetime(reference_date)
        mask = vectors.index <= ref_ts
        if not mask.any():
            return {
                "status": "error",
                "message": f"No data available on or before {reference_date}",
                "reference_date": reference_date,
                "window_days": window,
                "analogues": [],
            }
        ref_idx = int(np.nonzero(mask)[0][-1])
    else:
        ref_idx = n_rows - 1

    ref_ts = vectors.index[ref_idx]

    if ref_idx < window - 1:
        return {
            "status": "error",
            "message": f"Not enough history for window={window} at {ref_ts.date()}",
            "reference_date": str(ref_ts.date()),
            "window_days": window,
            "analogues": [],
        }

    # ------------------------------------------------------------------
    # Reference macro vector and price series
    # ------------------------------------------------------------------
    ref_vec = vectors.iloc[ref_idx].values.astype(np.float64)
    ref_vec = np.nan_to_num(ref_vec, nan=0.0)

    ref_start_idx = ref_idx - window + 1
    ref_gspc_rets = raw["^GSPC_ret_1d"].values[ref_start_idx : ref_idx + 1]
    ref_prices = _reconstruct_prices(ref_gspc_rets)

    # ------------------------------------------------------------------
    # Candidate index range
    # ------------------------------------------------------------------
    # Constraints:
    #   lo: need `window` rows of lookback before the candidate end date
    #   hi: non-overlapping with the reference window AND enough forward data
    lo = window - 1
    hi = ref_idx - window  # leaves `window`-day gap before reference

    if date_from is not None:
        dt_from = pd.to_datetime(date_from)
        idx_from = int(vectors.index.searchsorted(dt_from))
        lo = max(lo, idx_from)

    if date_to is not None:
        dt_to = pd.to_datetime(date_to)
        idx_to = int(vectors.index.searchsorted(dt_to, side="right")) - 1
        hi = min(hi, idx_to)

    if lo > hi:
        return {
            "status": "error",
            "message": "No candidate dates found in the specified range",
            "reference_date": ref_ts.strftime("%Y-%m-%d"),
            "window_days": window,
            "analogues": [],
        }

    candidate_indices = np.arange(lo, hi + 1, dtype=np.int64)
    n_candidates = int(len(candidate_indices))

    # ------------------------------------------------------------------
    # Macro similarity  (vectorised cosine on z-scored features)
    # ------------------------------------------------------------------
    feature_matrix = vectors.iloc[candidate_indices].values.astype(np.float64)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)

    raw_cosines = _macro_similarities_vec(ref_vec, feature_matrix)
    macro_scores = (raw_cosines + 1.0) / 2.0  # → [0, 1]

    # ------------------------------------------------------------------
    # Price similarity
    # ------------------------------------------------------------------
    gspc_rets_all = raw["^GSPC_ret_1d"].values

    if price_method == "pearson":
        cand_price_matrix = np.empty((n_candidates, window), dtype=np.float64)
        for local_i, cand_idx in enumerate(candidate_indices):
            rets = gspc_rets_all[int(cand_idx) - window + 1 : int(cand_idx) + 1]
            cand_price_matrix[local_i] = _reconstruct_prices(rets)

        price_scores = _price_similarities_vec(ref_prices, cand_price_matrix)

    else:
        # DTW — per-row loop
        from backtest.price_similarity import compute_price_similarity as _ps

        price_scores = np.empty(n_candidates, dtype=np.float64)
        for local_i, cand_idx in enumerate(candidate_indices):
            rets = gspc_rets_all[int(cand_idx) - window + 1 : int(cand_idx) + 1]
            cand_prices = _reconstruct_prices(rets)
            try:
                price_scores[local_i] = _ps(ref_prices, cand_prices, method="dtw")
            except Exception:
                price_scores[local_i] = 0.0

    # ------------------------------------------------------------------
    # Combined score
    # ------------------------------------------------------------------
    weights = {"macro": macro_weight, "price": price_weight}
    combined_scores = np.array(
        [
            combine_similarity_scores(float(m), float(p), weights)
            for m, p in zip(macro_scores, price_scores)
        ],
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    # Select top-N with non-overlapping windows
    # ------------------------------------------------------------------
    sorted_local = np.argsort(combined_scores)[::-1]
    analogues: list[dict] = []
    used_ranges: list[tuple[int, int]] = []

    for local_idx in sorted_local:
        if len(analogues) >= n:
            break

        cand_idx = int(candidate_indices[local_idx])
        cand_start = cand_idx - window + 1
        cand_end = cand_idx

        # Skip windows that overlap with any already-chosen analogue
        if any(not (cand_end < s or cand_start > e) for s, e in used_ranges):
            continue

        used_ranges.append((cand_start, cand_end))

        cand_end_date = vectors.index[cand_end]
        cand_start_date = vectors.index[cand_start]

        # Forward returns: what happened in the 5 / 21 / 63 trading days after
        forward_returns: dict[str, float] = {}
        for fwd_days in (5, 21, 63):
            fwd_idx = cand_end + fwd_days
            if fwd_idx < n_rows:
                col = f"^GSPC_ret_{fwd_days}d"
                if col in raw.columns:
                    val = float(raw.iloc[fwd_idx][col])
                    if not np.isnan(val):
                        forward_returns[f"{fwd_days}d"] = round(val, 6)

        # Normalised S&P 500 price series for this window (for charting)
        cand_rets = gspc_rets_all[cand_start : cand_end + 1]
        cand_prices_chart = _reconstruct_prices(cand_rets)
        cand_dates = vectors.index[cand_start : cand_end + 1]

        sp500_prices = [
            {
                "time": pd.Timestamp(ts).strftime("%Y-%m-%d"),
                "value": round(float(v), 4),
            }
            for ts, v in zip(cand_dates, cand_prices_chart)
        ]

        # Top 5 feature differences for interpretability
        cand_vec = feature_matrix[local_idx]
        top_diffs = _top_feature_diffs(ref_vec, cand_vec, feat_cols, top_k=5)

        analogues.append(
            {
                "rank": len(analogues) + 1,
                "start_date": cand_start_date.strftime("%Y-%m-%d"),
                "end_date": cand_end_date.strftime("%Y-%m-%d"),
                "combined_score": round(float(combined_scores[local_idx]), 4),
                "macro_score": round(float(macro_scores[local_idx]), 4),
                "price_score": round(float(price_scores[local_idx]), 4),
                "forward_returns": forward_returns,
                "sp500_prices": sp500_prices,
                "top_feature_diffs": top_diffs,
            }
        )

    # ------------------------------------------------------------------
    # Reference price series for charting
    # ------------------------------------------------------------------
    ref_prices_chart = _reconstruct_prices(ref_gspc_rets)
    ref_dates = vectors.index[ref_start_idx : ref_idx + 1]
    reference_prices = [
        {
            "time": pd.Timestamp(ts).strftime("%Y-%m-%d"),
            "value": round(float(v), 4),
        }
        for ts, v in zip(ref_dates, ref_prices_chart)
    ]

    return {
        "status": "success",
        "reference_date": ref_ts.strftime("%Y-%m-%d"),
        "window_days": window,
        "macro_weight": macro_weight,
        "price_weight": price_weight,
        "n_candidates_scored": n_candidates,
        "analogues": analogues,
        "reference_prices": reference_prices,
    }


# ---------------------------------------------------------------------------
# File-cached wrapper
# ---------------------------------------------------------------------------

_CACHE_DIR = _BACKTEST_DIR.parent / "pulse_cache"
_CACHE_TTL = 30 * 60  # 30 minutes — matches the rest of the Pulse cache layer


def _result_cache_key(
    n: int,
    window: int,
    macro_weight: float,
    price_weight: float,
    reference_date: str | None,
    date_from: str | None,
    date_to: str | None,
    price_method: str,
) -> str:
    """Stable SHA-1 hash of all call parameters for cache filenames."""
    payload = (
        f"{n}|{window}|{macro_weight}|{price_weight}|"
        f"{reference_date}|{date_from}|{date_to}|{price_method}"
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def get_analogues_cached(
    n: int = 5,
    window: int = 63,
    macro_weight: float = 0.6,
    price_weight: float = 0.4,
    reference_date: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    price_method: Literal["pearson", "dtw"] = "pearson",
    force_refresh: bool = False,
) -> dict:
    """
    Cached wrapper around :func:`find_analogues`.

    Results are written to ``pulse_cache/analogues_<hash>.json`` and reused
    within a 30-minute TTL (matching the rest of the Pulse cache layer).
    Pass ``force_refresh=True`` to bypass the cache and overwrite it.
    """
    key = _result_cache_key(
        n, window, macro_weight, price_weight,
        reference_date, date_from, date_to, price_method,
    )
    cache_file = _CACHE_DIR / f"analogues_{key}.json"

    if not force_refresh and cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < _CACHE_TTL:
            try:
                with open(cache_file) as fh:
                    return json.load(fh)
            except Exception:
                pass  # corrupted cache — fall through to recompute

    result = find_analogues(
        n=n,
        window=window,
        macro_weight=macro_weight,
        price_weight=price_weight,
        reference_date=reference_date,
        date_from=date_from,
        date_to=date_to,
        price_method=price_method,
    )

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_file, "w") as fh:
            json.dump(result, fh)
    except Exception as exc:
        print(f"[analogues] Cache write failed: {exc}")

    return result
