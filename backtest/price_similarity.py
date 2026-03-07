"""
Price trajectory similarity scoring for the Historical Analogues feature.

Exposes compute_price_similarity(series_a, series_b, method) -> float
returning a score in [0, 1] where 1.0 = identical shape, 0.0 = maximally different.

Two methods are supported:
  - "dtw"     : Dynamic Time Warping distance on z-scored series, handles
                different series lengths and small temporal offsets.
  - "pearson" : Pearson correlation on percent-from-start normalised series,
                fast and interpretable; best for same-length comparisons.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_array(series: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert any sequence to a 1-D float64 numpy array."""
    return np.asarray(series, dtype=np.float64).ravel()


def _z_score(arr: np.ndarray) -> np.ndarray:
    """
    Z-score normalise an array.

    If the series has zero variance (flat), return a zero vector so that DTW
    distance between two flat series is 0 and distance between a flat and a
    non-flat series is non-trivially large.
    """
    std = float(np.std(arr))
    if std == 0.0:
        return np.zeros(len(arr), dtype=np.float64)
    return (arr - float(np.mean(arr))) / std


def _pct_from_start(arr: np.ndarray) -> np.ndarray:
    """
    Express each element as a fractional change from the first value.

    e.g. [100, 110, 90] → [0.0, 0.1, -0.1]

    If the first element is 0 we fall back to mean-centred differencing to
    avoid division-by-zero.
    """
    first = float(arr[0])
    if first == 0.0:
        mean = float(np.mean(arr))
        if mean == 0.0:
            return np.zeros(len(arr), dtype=np.float64)
        return (arr - first) / abs(mean)
    return (arr - first) / abs(first)


def _resample_linear(arr: np.ndarray, target_len: int) -> np.ndarray:
    """
    Linearly interpolate *arr* to exactly *target_len* points.

    Used by the Pearson method to align series of different lengths.
    """
    if len(arr) == target_len:
        return arr
    src_x = np.linspace(0.0, 1.0, len(arr))
    dst_x = np.linspace(0.0, 1.0, target_len)
    return np.interp(dst_x, src_x, arr)


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Dynamic Time Warping (DTW) distance between two 1-D arrays.

    Uses the standard O(n·m) DP formulation with squared Euclidean cost and
    a final square-root so the result lives in the same unit as the inputs.

    The distance is then divided by the warp-path length (n + m − 1) to make
    it comparable across series of varying lengths.
    """
    n, m = len(a), len(b)
    # Initialise cost matrix with ∞; [0, 0] starts at 0
    dtw: np.ndarray = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    raw_dist = float(np.sqrt(dtw[n, m]))
    # Normalise by warp-path length so longer series don't dominate
    path_len = n + m - 1
    return raw_dist / path_len if path_len > 0 else 0.0


def _pearson_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation converted to a [0, 1] similarity.

    Both series are first resampled to the same length (shorter one is the
    reference) and expressed as percent-change from start so that absolute
    price level and overall trend magnitude don't bias the result.

    Returns 0.5 when one or both series is flat (correlation undefined).
    """
    target_len = min(len(a), len(b))
    a_r = _pct_from_start(_resample_linear(a, target_len))
    b_r = _pct_from_start(_resample_linear(b, target_len))

    std_a = float(np.std(a_r))
    std_b = float(np.std(b_r))

    # Flat series → correlation undefined; return 0.5 (neutral)
    if std_a == 0.0 or std_b == 0.0:
        # Two flat series are perfectly "similar in shape" — both go nowhere
        if std_a == 0.0 and std_b == 0.0:
            return 1.0
        return 0.5

    r = float(np.corrcoef(a_r, b_r)[0, 1])

    # Clamp for floating-point edge cases
    r = max(-1.0, min(1.0, r))

    # Map [-1, 1] → [0, 1]
    return (r + 1.0) / 2.0


def _dtw_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    DTW distance converted to a [0, 1] similarity via 1 / (1 + distance).

    Both series are z-scored first so amplitude differences don't dominate
    the shape comparison.  Zero distance → similarity = 1.0.
    """
    a_z = _z_score(a)
    b_z = _z_score(b)

    dist = _dtw_distance(a_z, b_z)
    # 1/(1+d) maps [0, ∞) → (0, 1]; equals 1.0 only when dist=0
    return 1.0 / (1.0 + dist)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_price_similarity(
    series_a: Sequence[float] | np.ndarray,
    series_b: Sequence[float] | np.ndarray,
    method: Literal["dtw", "pearson"] = "dtw",
) -> float:
    """
    Compute the shape-similarity between two price series.

    Parameters
    ----------
    series_a, series_b:
        Ordered sequences of price (or price-like) values.  May be different
        lengths when method="dtw".  For method="pearson" the longer series is
        linearly resampled to match the shorter.
    method:
        "dtw"     — Dynamic Time Warping on z-scored series.  Handles
                    different lengths and small temporal offsets.
        "pearson" — Pearson correlation on pct-from-start series.  Fast;
                    best when series are the same length.

    Returns
    -------
    float in [0, 1]
        1.0 means the two series have identical shape.
        0.0 means maximally dissimilar.

    Raises
    ------
    ValueError
        If either series is empty or contains fewer than 2 points.
    """
    a = _to_array(series_a)
    b = _to_array(series_b)

    if len(a) < 2:
        raise ValueError(f"series_a must have at least 2 points, got {len(a)}")
    if len(b) < 2:
        raise ValueError(f"series_b must have at least 2 points, got {len(b)}")

    if method == "dtw":
        return _dtw_similarity(a, b)
    elif method == "pearson":
        return _pearson_similarity(a, b)
    else:
        raise ValueError(f"Unknown method {method!r}. Choose 'dtw' or 'pearson'.")
