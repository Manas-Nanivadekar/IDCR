"""
Ground truth precision matrix generation and optimal allocation computation.

Each document d contributes a user-dependent precision matrix Λ_d(x):
    Λ_d(x) = λ_d · relevance(d, x) · P_d

where λ_d is the document's base informativeness, relevance measures
user-document alignment, and P_d is a PSD projection onto informed dimensions.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from src.data.user_profiles import ASSET_CLASSES
from src.utils.linear_algebra import nearest_psd, project_simplex


D = len(ASSET_CLASSES)  # 8

# ── Realistic market covariance template ──────────────────────────────────────

def get_market_covariance() -> np.ndarray:
    """Return a realistic 8×8 covariance matrix for the asset classes.

    Based on typical historical correlations between major asset classes:
    US equity, international equity, emerging markets, bonds, real estate,
    commodities, cash, alternatives.

    Returns:
        (8, 8) symmetric PSD covariance matrix.
    """
    # Annualised volatilities (std dev)
    vols = np.array([0.16, 0.18, 0.24, 0.05, 0.14, 0.20, 0.01, 0.12])

    # Correlation matrix (stylised but realistic)
    corr = np.array([
        #  USEq  IntlEq  EM    Bonds REst  Comm  Cash  Alt
        [1.00,  0.80,  0.70, -0.20, 0.55,  0.35, 0.00, 0.60],   # US_equity
        [0.80,  1.00,  0.75, -0.15, 0.50,  0.40, 0.00, 0.55],   # intl_equity
        [0.70,  0.75,  1.00, -0.10, 0.45,  0.50, 0.00, 0.50],   # emerging_markets
        [-0.20, -0.15, -0.10, 1.00, 0.10, -0.10, 0.30, -0.05],  # bonds
        [0.55,  0.50,  0.45,  0.10, 1.00,  0.30, 0.00, 0.45],   # real_estate
        [0.35,  0.40,  0.50, -0.10, 0.30,  1.00, 0.00, 0.35],   # commodities
        [0.00,  0.00,  0.00,  0.30, 0.00,  0.00, 1.00, 0.00],   # cash
        [0.60,  0.55,  0.50, -0.05, 0.45,  0.35, 0.00, 1.00],   # alternatives
    ])

    sigma = np.outer(vols, vols) * corr
    return nearest_psd(sigma)


def get_expected_returns() -> np.ndarray:
    """Return expected annual returns for the 8 asset classes.

    Returns:
        (8,) vector of expected returns.
    """
    return np.array([0.10, 0.09, 0.12, 0.03, 0.08, 0.06, 0.02, 0.07])


# ── Sector → dimension mapping ───────────────────────────────────────────────

def get_relevant_dims(relevant_sectors: list[str]) -> list[int]:
    """Map sector names to indices in the d=8 asset class space.

    Args:
        relevant_sectors: list of asset class names.

    Returns:
        List of integer indices.
    """
    return [ASSET_CLASSES.index(s) for s in relevant_sectors if s in ASSET_CLASSES]


# ── User–document relevance ──────────────────────────────────────────────────

def compute_relevance(doc: dict[str, Any], user: dict[str, Any]) -> float:
    """Compute relevance score in [0, 1] between a document and user profile.

    Based on:
    1. Sector overlap: fraction of doc's relevant sectors in user's preferences
    2. Horizon match: penalise mismatch between doc and user time horizons
    3. Risk alignment: how well the doc's content aligns with user risk level

    Args:
        doc: document dict.
        user: user profile dict.

    Returns:
        Scalar relevance in [0, 1].
    """
    # Sector overlap
    user_sectors = set(user["sector_preferences"])
    doc_sectors = set(doc["relevant_sectors"])
    # Include ASSET_CLASSES that match user holdings > threshold
    for ac in ASSET_CLASSES:
        if user["current_holdings"].get(ac, 0) > 0.1:
            user_sectors.add(ac)
    overlap = len(user_sectors & doc_sectors) / max(len(doc_sectors), 1)

    # Horizon match
    horizon_map = {"short_term": 6, "medium_term": 24, "long_term": 72}
    doc_horizon = horizon_map.get(doc.get("time_horizon", "medium_term"), 24)
    user_horizon = user["investment_horizon_months"]
    horizon_diff = abs(user_horizon - doc_horizon) / max(user_horizon, doc_horizon, 1)
    horizon_match = 1.0 - min(horizon_diff, 1.0)

    # Risk alignment
    sentiment_risk = {"bullish": 0.8, "bearish": 0.3, "neutral": 0.5, "mixed": 0.5}
    doc_risk = sentiment_risk.get(doc.get("sentiment", "neutral"), 0.5)
    risk_match = 1.0 - abs(user["risk_tolerance"] - doc_risk)

    # Weighted combination
    relevance = 0.5 * overlap + 0.3 * horizon_match + 0.2 * risk_match
    return float(np.clip(relevance, 0.05, 1.0))


# ── Precision matrix construction ────────────────────────────────────────────

def compute_ground_truth_precision(
    doc: dict[str, Any],
    user: dict[str, Any],
    d: int = D,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Compute the ground truth precision matrix Λ_d(x) for a (doc, user) pair.

    Λ_d(x) = λ_d · relevance(d, x) · P_d

    where P_d is a PSD projection onto the dimensions the document informs.

    Args:
        doc: document dict.
        user: user profile dict.
        d: dimensionality (number of asset classes).
        rng: optional random generator for off-diagonal correlations.

    Returns:
        (d, d) symmetric PSD precision matrix.
    """
    if rng is None:
        rng = np.random.default_rng(doc["doc_id"] * 1000 + user["user_id"])

    # Build projection matrix onto relevant dimensions
    relevant_dims = get_relevant_dims(doc["relevant_sectors"])
    P = np.zeros((d, d))
    for i in relevant_dims:
        P[i, i] = 1.0

    # Add small off-diagonal correlations within relevant dims
    for i in relevant_dims:
        for j in relevant_dims:
            if i != j:
                P[i, j] = 0.1 * rng.standard_normal()

    # Symmetrise and ensure PSD
    P = (P + P.T) / 2
    P = nearest_psd(P)

    # Scale by informativeness and relevance
    relevance = compute_relevance(doc, user)
    Lambda = doc["informativeness"] * relevance * P

    return Lambda


def compute_prior_covariance(user: dict[str, Any], d: int = D,
                             base_var: float = 1.0) -> np.ndarray:
    """Compute the prior covariance Σ_0(x) for a user.

    Higher risk tolerance → larger prior variance (more uncertain about
    risky assets). Lower risk tolerance → tighter priors (closer to bonds/cash).

    Args:
        user: user profile dict.
        d: dimensionality.
        base_var: base variance scale.

    Returns:
        (d, d) PSD prior covariance matrix.
    """
    risk = user["risk_tolerance"]
    # Scale variance per asset class based on risk tolerance
    # Riskier assets get higher variance for low-risk users (more uncertain)
    asset_riskiness = np.array([0.7, 0.8, 0.9, 0.2, 0.5, 0.7, 0.1, 0.6])
    var_scale = base_var * (1.0 + 2.0 * (1.0 - risk) * asset_riskiness)
    # Add small correlations
    sigma = np.diag(var_scale)
    sigma_market = get_market_covariance()
    # Blend: 70% diagonal + 30% market-based
    sigma = 0.7 * sigma + 0.3 * sigma_market * (base_var / 0.02)  # rescale market cov
    return nearest_psd(sigma)


# ── Optimal allocation ───────────────────────────────────────────────────────

def compute_optimal_allocation(
    user: dict[str, Any],
    sigma_market: np.ndarray | None = None,
    expected_returns: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the ground truth optimal allocation y*(x) for a user.

    y* = argmax_w  E[r'w] - (risk_aversion/2) * w'Σ_market*w
    s.t. w >= 0, 1'w = 1

    Uses analytical unconstrained solution projected to simplex.

    Args:
        user: user profile dict.
        sigma_market: (d, d) market covariance. None → use default.
        expected_returns: (d,) expected returns. None → use default.

    Returns:
        (d,) optimal portfolio weights on the simplex.
    """
    if sigma_market is None:
        sigma_market = get_market_covariance()
    if expected_returns is None:
        expected_returns = get_expected_returns()

    risk_aversion = 1.0 / (user["risk_tolerance"] + 0.01)

    # Unconstrained mean-variance solution: w* = (1/γ) Σ^{-1} μ
    sigma_inv = np.linalg.inv(sigma_market)
    w_unconstrained = (1.0 / risk_aversion) * sigma_inv @ expected_returns

    # Project onto simplex
    w_optimal = project_simplex(w_unconstrained)

    return w_optimal
