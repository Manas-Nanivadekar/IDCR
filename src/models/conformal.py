"""
Split conformal prediction for ellipsoidal prediction sets.

Implements the conformal prediction pipeline using Mahalanobis nonconformity
scores, producing ellipsoidal prediction sets with distribution-free coverage
guarantees.
"""

from __future__ import annotations

import numpy as np
from typing import Any
import math

from src.utils.linear_algebra import mahalanobis_batch


class ConformalPredictor:
    """Split conformal prediction with ellipsoidal prediction sets.

    The nonconformity score is the Mahalanobis distance:
        s_i = (y_i - ŷ_i)^T Σ^{-1}_i (y_i - ŷ_i)

    The prediction set is an ellipsoid:
        C_α = {y : (y - ŷ)^T Σ^{-1} (y - ŷ) ≤ q̂}
    """

    def __init__(self, alpha: float = 0.10):
        """
        Args:
            alpha: miscoverage level. Prediction sets target (1-alpha) coverage.
        """
        self.alpha = alpha
        self.q_hat: float | None = None
        self.calibration_scores: np.ndarray | None = None

    def calibrate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sigma_inv: np.ndarray,
    ) -> float:
        """Compute calibration threshold from held-out data.

        Args:
            y_true: (n_cal, d) ground truth allocations.
            y_pred: (n_cal, d) predicted allocations.
            sigma_inv: (n_cal, d, d) inverse covariance for each point.

        Returns:
            q_hat: the calibration threshold.
        """
        scores = mahalanobis_batch(y_true, y_pred, sigma_inv)
        self.calibration_scores = scores

        n = len(scores)
        q_level = math.ceil((n + 1) * (1.0 - self.alpha)) / n
        self.q_hat = float(np.quantile(scores, min(q_level, 1.0)))
        return self.q_hat

    def predict_set(
        self, y_pred: np.ndarray, sigma: np.ndarray,
    ) -> dict[str, Any]:
        """Return ellipsoidal prediction set parameters.

        C_α = {y : (y - ŷ)^T Σ^{-1} (y - ŷ) ≤ q̂}

        Args:
            y_pred: (d,) or (n, d) predicted centres.
            sigma: (d, d) or (n, d, d) covariance matrices.

        Returns:
            Dict with 'center', 'sigma', 'threshold'.
        """
        assert self.q_hat is not None, "Must call calibrate() first."
        return {
            "center": y_pred,
            "sigma": sigma,
            "threshold": self.q_hat,
        }

    def compute_volume(self, sigma: np.ndarray) -> float:
        """Volume of the conformal ellipsoid.

        Vol = V_d · det(Σ)^{1/2} · q̂^{d/2}

        where V_d = π^{d/2} / Γ(d/2 + 1) is the unit ball volume.

        Args:
            sigma: (d, d) covariance matrix.

        Returns:
            Positive scalar volume.
        """
        assert self.q_hat is not None, "Must call calibrate() first."
        d = sigma.shape[0]
        vol_unit = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
        det_sigma = np.linalg.det(sigma)
        return vol_unit * np.sqrt(max(det_sigma, 0.0)) * self.q_hat ** (d / 2)

    def check_coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sigma_inv: np.ndarray,
    ) -> float:
        """Empirical coverage on a dataset.

        Args:
            y_true: (n, d) ground truth.
            y_pred: (n, d) predictions.
            sigma_inv: (n, d, d) precision matrices.

        Returns:
            Fraction of points covered (in [0, 1]).
        """
        assert self.q_hat is not None, "Must call calibrate() first."
        scores = mahalanobis_batch(y_true, y_pred, sigma_inv)
        return float(np.mean(scores <= self.q_hat))

    def score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sigma_inv: np.ndarray,
    ) -> np.ndarray:
        """Compute nonconformity scores.

        Args:
            y_true: (n, d) ground truth.
            y_pred: (n, d) predictions.
            sigma_inv: (n, d, d) precision matrices.

        Returns:
            (n,) Mahalanobis scores.
        """
        return mahalanobis_batch(y_true, y_pred, sigma_inv)
