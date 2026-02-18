"""
Empirical coverage verification for conformal prediction.

Tests that all retrieval policies (random, greedy, RL) achieve ≥ (1-α) coverage
after conformal calibration, regardless of retrieval policy.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Callable

from src.models.conformal import ConformalPredictor
from src.utils.linear_algebra import posterior_covariance, log_det


def evaluate_coverage(
    retrieval_fn: Callable,
    users_cal: list[dict[str, Any]],
    users_test: list[dict[str, Any]],
    y_true_cal: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_cal: np.ndarray,
    y_pred_test: np.ndarray,
    sigma_inv_cal: np.ndarray,
    sigma_inv_test: np.ndarray,
    alpha: float = 0.10,
) -> dict[str, float]:
    """Evaluate conformal coverage for a given retrieval method.

    Args:
        retrieval_fn: function that returns sigma_inv for each user.
        users_cal: calibration user profiles.
        users_test: test user profiles.
        y_true_cal: (n_cal, d) calibration ground truth.
        y_true_test: (n_test, d) test ground truth.
        y_pred_cal: (n_cal, d) calibration predictions.
        y_pred_test: (n_test, d) test predictions.
        sigma_inv_cal: (n_cal, d, d) calibration precision matrices.
        sigma_inv_test: (n_test, d, d) test precision matrices.
        alpha: miscoverage level.

    Returns:
        Dict with coverage, mean_volume, q_hat.
    """
    cp = ConformalPredictor(alpha=alpha)
    q_hat = cp.calibrate(y_true_cal, y_pred_cal, sigma_inv_cal)

    coverage = cp.check_coverage(y_true_test, y_pred_test, sigma_inv_test)

    # Compute mean volume
    d = y_true_test.shape[1]
    volumes = []
    for i in range(len(y_true_test)):
        sigma_i = np.linalg.inv(sigma_inv_test[i])
        vol = cp.compute_volume(sigma_i)
        volumes.append(vol)

    return {
        "coverage": coverage,
        "mean_volume": float(np.mean(volumes)),
        "median_volume": float(np.median(volumes)),
        "q_hat": q_hat,
        "target_coverage": 1.0 - alpha,
        "coverage_gap": coverage - (1.0 - alpha),
    }


def multi_alpha_coverage(
    y_true_cal: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_cal: np.ndarray,
    y_pred_test: np.ndarray,
    sigma_inv_cal: np.ndarray,
    sigma_inv_test: np.ndarray,
    alphas: list[float] | None = None,
) -> dict[str, Any]:
    """Evaluate coverage across multiple alpha levels.

    Args:
        y_true_cal, y_true_test: ground truth arrays.
        y_pred_cal, y_pred_test: prediction arrays.
        sigma_inv_cal, sigma_inv_test: precision matrix arrays.
        alphas: list of miscoverage levels.

    Returns:
        Dict with per-alpha results.
    """
    if alphas is None:
        alphas = [0.01, 0.05, 0.10, 0.20]

    results = {}
    for alpha in alphas:
        cp = ConformalPredictor(alpha=alpha)
        cp.calibrate(y_true_cal, y_pred_cal, sigma_inv_cal)
        coverage = cp.check_coverage(y_true_test, y_pred_test, sigma_inv_test)

        sigma_test = np.array([np.linalg.inv(si) for si in sigma_inv_test])
        volumes = [cp.compute_volume(s) for s in sigma_test]

        results[f"alpha_{alpha}"] = {
            "coverage": coverage,
            "target": 1.0 - alpha,
            "valid": coverage >= (1.0 - alpha) - 0.01,
            "mean_volume": float(np.mean(volumes)),
            "q_hat": cp.q_hat,
        }

    return results
