"""
Similarity scoring functions for the Historical Analogues feature.

Provides:
  compute_macro_similarity  — cosine similarity between macro feature vectors,
                              returns float in [-1.0, 1.0].
  combine_similarity_scores — weighted merge of a normalised macro score and a
                              price-trajectory score, both expected in [0.0, 1.0],
                              returning a combined float in [0.0, 1.0].

Typical pipeline
----------------
  raw_cosine  = compute_macro_similarity(vec_now, vec_hist)    # [-1, 1]
  macro_score = (raw_cosine + 1.0) / 2.0                       # [0, 1]
  price_score = compute_price_similarity(prices_now, prices_hist)  # [0, 1]
  score       = combine_similarity_scores(macro_score, price_score)
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np


# ---------------------------------------------------------------------------
# Macro-state similarity
# ---------------------------------------------------------------------------


def compute_macro_similarity(
    vector_a: np.ndarray | list[float],
    vector_b: np.ndarray | list[float],
) -> float:
    """
    Compute cosine similarity between two macro feature vectors.

    Cosine similarity measures the angle between vectors in feature space,
    making it scale-invariant — useful for macro feature vectors where
    absolute magnitude varies wildly (e.g. VIX vs copper prices).

    Returns a value in [-1.0, 1.0]:
      - 1.0 : identical direction (maximally similar macro state)
      - 0.0 : orthogonal (no linear relationship)
      - -1.0: opposite direction (maximally dissimilar)

    Edge cases:
      - Zero vector(s): returns 0.0 (undefined similarity, treated as no match)
      - Single-element vectors: behaves correctly (+1 or -1 depending on signs)
      - NaN values in input will propagate; caller should sanitize first

    Args:
        vector_a: First macro feature vector (1-D array-like).
        vector_b: Second macro feature vector (1-D array-like, same length).

    Returns:
        Cosine similarity as a float in [-1.0, 1.0].

    Raises:
        ValueError: If vectors have different lengths or are not 1-D.
    """
    a = np.asarray(vector_a, dtype=float).ravel()
    b = np.asarray(vector_b, dtype=float).ravel()

    if a.shape != b.shape:
        raise ValueError(
            f"Vectors must have the same length, got {a.shape[0]} and {b.shape[0]}"
        )

    if a.ndim != 1 or a.shape[0] == 0:
        raise ValueError("Vectors must be non-empty 1-D arrays")

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    # Zero vector has undefined cosine similarity — treat as no similarity
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    cosine_sim = float(np.dot(a, b) / (norm_a * norm_b))

    # Clamp to [-1, 1] to guard against floating-point rounding beyond bounds
    return max(-1.0, min(1.0, cosine_sim))


# ---------------------------------------------------------------------------
# Combined similarity
# ---------------------------------------------------------------------------


class SimilarityWeights(TypedDict, total=False):
    """
    Configurable weight mapping for combine_similarity_scores.

    Keys
    ----
    macro : Weight for the macro-state component (default 0.6).
    price : Weight for the price-trajectory component (default 0.4).

    Weights are automatically normalised so they always sum to 1.0, meaning
    raw "importance" numbers work just as well as pre-scaled fractions:
        {"macro": 3, "price": 1}  →  effective {"macro": 0.75, "price": 0.25}
    """

    macro: float
    price: float


def combine_similarity_scores(
    macro_score: float,
    price_score: float,
    weights: SimilarityWeights | None = None,
) -> float:
    """
    Merge a macro-state similarity score and a price-trajectory similarity
    score into a single combined score using a configurable weighting scheme.

    Both input scores must be in [0.0, 1.0]:
      - macro_score  : normalised cosine similarity, e.g. ``(raw_cosine + 1) / 2``
                       where ``raw_cosine`` comes from ``compute_macro_similarity``.
      - price_score  : shape similarity from ``compute_price_similarity``
                       (backtest.price_similarity), which already returns [0, 1].

    The function returns a weighted average of the two components, also in
    [0.0, 1.0], where 1.0 is the best possible match and 0.0 is the worst.

    Weight normalisation
    --------------------
    Weights are always normalised to sum to 1.0 before combining, so callers
    can pass raw "importance" values without pre-scaling:

        {"macro": 3, "price": 1}  →  effective weights 0.75 / 0.25
        {"macro": 0.6, "price": 0.4}  →  already normalised, unchanged

    Default weights
    ---------------
    macro=0.6, price=0.4 — macro conditions drive the primary market regime;
    price action refines the match within that regime.  Override per-call via
    the ``weights`` argument.

    Parameters
    ----------
    macro_score:
        Normalised macro-state similarity in [0.0, 1.0].
    price_score:
        Price-trajectory similarity in [0.0, 1.0].
    weights:
        Optional ``SimilarityWeights`` mapping with keys ``"macro"`` and/or
        ``"price"``.  Missing keys fall back to the defaults (0.6 / 0.4)
        before normalisation.  Both values must be non-negative.

    Returns
    -------
    float
        Combined similarity in [0.0, 1.0].

    Raises
    ------
    ValueError
        If any weight is negative.
    ValueError
        If both weights are zero (cannot normalise).
    ValueError
        If either score is outside [0.0, 1.0].
    """
    # --- Validate score ranges ---
    for name, score in (("macro_score", macro_score), ("price_score", price_score)):
        if not (0.0 <= score <= 1.0):
            raise ValueError(
                f"{name} must be in [0.0, 1.0], got {score:.6f}"
            )

    # --- Resolve weights with defaults ---
    _weights: dict[str, float] = {"macro": 0.6, "price": 0.4}
    if weights is not None:
        _weights.update(weights)

    w_macro = float(_weights.get("macro", 0.6))
    w_price = float(_weights.get("price", 0.4))

    if w_macro < 0.0 or w_price < 0.0:
        raise ValueError(
            f"Weights must be non-negative, got macro={w_macro}, price={w_price}"
        )

    total = w_macro + w_price
    if total == 0.0:
        raise ValueError("Sum of weights must be > 0; got macro=0, price=0")

    # --- Normalise weights ---
    w_macro_norm = w_macro / total
    w_price_norm = w_price / total

    # --- Weighted average ---
    combined = w_macro_norm * macro_score + w_price_norm * price_score

    # Clamp to [0, 1] to absorb any floating-point drift
    return max(0.0, min(1.0, combined))
