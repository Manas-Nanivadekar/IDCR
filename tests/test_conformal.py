"""Unit tests for conformal prediction."""

import numpy as np
import pytest

from src.models.conformal import ConformalPredictor


class TestConformalCoverage:
    """Test that conformal prediction achieves ≥ (1-α) coverage on Gaussian data."""

    @pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
    def test_coverage_guarantee(self, alpha: float):
        """On synthetic Gaussian data, empirical coverage should be ≥ 1-α."""
        rng = np.random.default_rng(42)
        d = 4
        n_cal = 500
        n_test = 1000

        # Generate Gaussian data
        mu_true = rng.standard_normal(d)
        A = rng.standard_normal((d, d))
        sigma_true = A @ A.T + np.eye(d) * 0.1
        sigma_inv_true = np.linalg.inv(sigma_true)

        # Calibration data
        y_cal = rng.multivariate_normal(mu_true, sigma_true, size=n_cal)
        y_pred_cal = np.tile(mu_true, (n_cal, 1))  # predict mean
        sigma_inv_cal = np.stack([sigma_inv_true] * n_cal)

        # Test data
        y_test = rng.multivariate_normal(mu_true, sigma_true, size=n_test)
        y_pred_test = np.tile(mu_true, (n_test, 1))
        sigma_inv_test = np.stack([sigma_inv_true] * n_test)

        # Calibrate and check coverage
        cp = ConformalPredictor(alpha=alpha)
        q_hat = cp.calibrate(y_cal, y_pred_cal, sigma_inv_cal)
        assert q_hat > 0

        coverage = cp.check_coverage(y_test, y_pred_test, sigma_inv_test)
        # Coverage should be approximately (1-alpha).
        # Allow some slack for finite-sample effects.
        assert coverage >= (1.0 - alpha) - 0.05, \
            f"Coverage {coverage:.3f} < {1.0 - alpha - 0.05:.3f} for α={alpha}"

    def test_volume_positive(self):
        """Conformal ellipsoid volume should be positive."""
        rng = np.random.default_rng(42)
        d = 4
        n_cal = 200

        mu = np.zeros(d)
        sigma = np.eye(d) * 2.0
        sigma_inv = np.linalg.inv(sigma)

        y_cal = rng.multivariate_normal(mu, sigma, size=n_cal)
        y_pred_cal = np.tile(mu, (n_cal, 1))
        sigma_inv_cal = np.stack([sigma_inv] * n_cal)

        cp = ConformalPredictor(alpha=0.10)
        cp.calibrate(y_cal, y_pred_cal, sigma_inv_cal)

        vol = cp.compute_volume(sigma)
        assert vol > 0

    def test_calibration_scores_stored(self):
        """Calibration should store scores."""
        rng = np.random.default_rng(42)
        d, n = 3, 50
        y = rng.standard_normal((n, d))
        mu = np.zeros((n, d))
        sigma_inv = np.stack([np.eye(d)] * n)

        cp = ConformalPredictor(alpha=0.10)
        cp.calibrate(y, mu, sigma_inv)
        assert cp.calibration_scores is not None
        assert len(cp.calibration_scores) == n


class TestConformalSubmodularity:
    """Test that greedy marginal gains are non-increasing (submodularity proxy)."""

    def test_diminishing_gains_from_greedy(self):
        from src.data.user_profiles import generate_profiles
        from src.data.documents import generate_corpus
        from src.data.precision_ground_truth import (
            compute_ground_truth_precision,
            compute_prior_covariance,
        )
        from src.retrieval.greedy import GreedyRetrieval

        profiles = generate_profiles(3, seed=42)
        corpus = generate_corpus(30, seed=42)
        user = profiles[0]
        sigma_0 = compute_prior_covariance(user)

        def precision_fn(doc, usr):
            return compute_ground_truth_precision(doc, usr)

        greedy = GreedyRetrieval(sigma_0)
        result = greedy.retrieve(user, corpus, precision_fn, k=10)
        gains = result["marginal_gains"]

        # Check diminishing returns (allow small numerical noise)
        for i in range(1, len(gains)):
            assert gains[i] <= gains[i - 1] + 1e-8, \
                f"Gain increased at step {i}: {gains[i]:.6f} > {gains[i-1]:.6f}"
