"""
Evaluation metrics for portfolio recommendations.

Covers: MSE, cosine similarity, Sharpe ratio, max drawdown, and
portfolio-specific metrics.
"""

from __future__ import annotations

import numpy as np
from typing import Any


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error between allocations.

    Args:
        y_true: (n, d) or (d,) ground truth.
        y_pred: (n, d) or (d,) predictions.

    Returns:
        Scalar MSE.
    """
    return float(np.mean((y_true - y_pred) ** 2))


def cosine_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean cosine similarity between allocation vectors.

    Args:
        y_true: (n, d) ground truth.
        y_pred: (n, d) predictions.

    Returns:
        Mean cosine similarity in [-1, 1].
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_pred = y_pred.reshape(1, -1)

    norms_true = np.linalg.norm(y_true, axis=1, keepdims=True)
    norms_pred = np.linalg.norm(y_pred, axis=1, keepdims=True)

    # Avoid division by zero
    norms_true = np.maximum(norms_true, 1e-10)
    norms_pred = np.maximum(norms_pred, 1e-10)

    cos_sim = np.sum(y_true * y_pred, axis=1) / (norms_true.squeeze() * norms_pred.squeeze())
    return float(np.mean(cos_sim))


def sharpe_ratio(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.02,
) -> float:
    """Compute Sharpe ratio for a portfolio.

    Args:
        weights: (d,) portfolio weights.
        expected_returns: (d,) expected returns.
        cov_matrix: (d, d) return covariance.
        risk_free_rate: risk-free rate.

    Returns:
        Sharpe ratio scalar.
    """
    portfolio_return = float(weights @ expected_returns)
    portfolio_vol = float(np.sqrt(weights @ cov_matrix @ weights))
    if portfolio_vol < 1e-10:
        return 0.0
    return (portfolio_return - risk_free_rate) / portfolio_vol


def max_drawdown(returns_series: np.ndarray) -> float:
    """Maximum drawdown from a series of cumulative returns.

    Args:
        returns_series: (T,) cumulative return values.

    Returns:
        Maximum drawdown as a positive fraction.
    """
    if len(returns_series) == 0:
        return 0.0
    peak = returns_series[0]
    max_dd = 0.0
    for val in returns_series:
        peak = max(peak, val)
        dd = (peak - val) / max(abs(peak), 1e-10)
        max_dd = max(max_dd, dd)
    return float(max_dd)


def portfolio_metrics(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    y_true: np.ndarray | None = None,
) -> dict[str, float]:
    """Comprehensive portfolio quality metrics.

    Args:
        weights: (d,) predicted portfolio weights.
        expected_returns: (d,) expected returns.
        cov_matrix: (d, d) covariance.
        y_true: (d,) ground truth weights (optional).

    Returns:
        Dict of metric name → value.
    """
    metrics: dict[str, float] = {}
    metrics["portfolio_return"] = float(weights @ expected_returns)
    metrics["portfolio_volatility"] = float(np.sqrt(weights @ cov_matrix @ weights))
    metrics["sharpe_ratio"] = sharpe_ratio(weights, expected_returns, cov_matrix)
    metrics["diversification"] = float(1.0 - np.max(weights))
    metrics["n_nonzero"] = int(np.sum(weights > 0.01))

    if y_true is not None:
        metrics["mse_to_optimal"] = mse(y_true, weights)
        metrics["cosine_to_optimal"] = cosine_similarity(y_true, weights)

    return metrics


def batch_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma_inv: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute aggregate metrics over a batch.

    Args:
        y_true: (n, d) ground truth.
        y_pred: (n, d) predictions.
        sigma_inv: (n, d, d) optional precision matrices for Mahalanobis.

    Returns:
        Dict of aggregated metrics.
    """
    from src.utils.linear_algebra import mahalanobis_batch

    metrics = {
        "mse": mse(y_true, y_pred),
        "cosine_similarity": cosine_similarity(y_true, y_pred),
        "mean_l2": float(np.mean(np.linalg.norm(y_true - y_pred, axis=1))),
    }

    if sigma_inv is not None:
        maha_dists = mahalanobis_batch(y_true, y_pred, sigma_inv)
        metrics["mean_mahalanobis"] = float(np.mean(maha_dists))
        metrics["median_mahalanobis"] = float(np.median(maha_dists))

    return metrics
