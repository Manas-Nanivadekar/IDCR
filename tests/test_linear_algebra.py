"""Unit tests for linear algebra utilities."""

import numpy as np
import pytest

from src.utils.linear_algebra import (
    nearest_psd,
    matrix_sqrt,
    log_det,
    mahalanobis,
    mahalanobis_batch,
    project_simplex,
    posterior_covariance,
    marginal_gain,
    is_psd,
)


class TestNearestPSD:
    def test_already_psd(self):
        M = np.eye(4) * 2.0
        result = nearest_psd(M)
        assert is_psd(result)

    def test_non_psd_input(self):
        M = np.array([[1.0, 2.0], [2.0, 1.0]])  # has negative eigenvalue
        result = nearest_psd(M)
        assert is_psd(result)

    def test_symmetry(self):
        rng = np.random.default_rng(42)
        M = rng.standard_normal((5, 5))
        M = (M + M.T) / 2
        result = nearest_psd(M)
        np.testing.assert_allclose(result, result.T, atol=1e-12)

    def test_preserves_psd(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((4, 4))
        M = A @ A.T + np.eye(4) * 0.1  # PSD by construction
        result = nearest_psd(M)
        np.testing.assert_allclose(result, M, atol=1e-6)


class TestMatrixSqrt:
    def test_sqrt_squared(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((4, 4))
        M = A @ A.T + np.eye(4) * 0.1
        M_sqrt = matrix_sqrt(M)
        np.testing.assert_allclose(M_sqrt @ M_sqrt, M, atol=1e-8)

    def test_identity(self):
        M_sqrt = matrix_sqrt(np.eye(5))
        np.testing.assert_allclose(M_sqrt, np.eye(5), atol=1e-12)

    def test_psd_output(self):
        rng = np.random.default_rng(123)
        A = rng.standard_normal((6, 6))
        M = A @ A.T
        assert is_psd(matrix_sqrt(M))


class TestLogDet:
    def test_identity(self):
        assert abs(log_det(np.eye(5))) < 1e-12

    def test_diagonal(self):
        M = np.diag([1.0, 2.0, 3.0])
        expected = np.log(1.0) + np.log(2.0) + np.log(3.0)
        assert abs(log_det(M) - expected) < 1e-10

    def test_matches_numpy(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((5, 5))
        M = A @ A.T + np.eye(5)
        expected = np.log(np.linalg.det(M))
        assert abs(log_det(M) - expected) < 1e-8


class TestMahalanobis:
    def test_zero_distance(self):
        x = np.array([1.0, 2.0, 3.0])
        assert abs(mahalanobis(x, x, np.eye(3))) < 1e-12

    def test_euclidean(self):
        x = np.array([1.0, 0.0])
        mu = np.zeros(2)
        assert abs(mahalanobis(x, mu, np.eye(2)) - 1.0) < 1e-12

    def test_batch(self):
        rng = np.random.default_rng(42)
        n, d = 10, 4
        x = rng.standard_normal((n, d))
        mu = rng.standard_normal((n, d))
        sigma_inv = np.stack([np.eye(d)] * n)
        batch_result = mahalanobis_batch(x, mu, sigma_inv)
        for i in range(n):
            single = mahalanobis(x[i], mu[i], sigma_inv[i])
            assert abs(batch_result[i] - single) < 1e-10


class TestProjectSimplex:
    def test_already_on_simplex(self):
        w = np.array([0.3, 0.3, 0.4])
        result = project_simplex(w)
        assert np.all(result >= -1e-10)
        assert abs(np.sum(result) - 1.0) < 1e-10

    def test_negative_input(self):
        w = np.array([-1.0, 2.0, 0.5])
        result = project_simplex(w)
        assert np.all(result >= -1e-10)
        assert abs(np.sum(result) - 1.0) < 1e-10

    def test_uniform(self):
        w = np.array([5.0, 5.0, 5.0, 5.0])
        result = project_simplex(w)
        np.testing.assert_allclose(result, 0.25, atol=1e-10)


class TestPosteriorCovariance:
    def test_shrinks_variance(self):
        sigma = np.eye(4) * 2.0
        Lambda = np.eye(4) * 0.5
        sigma_new = posterior_covariance(sigma, Lambda)
        # Variance should decrease
        assert np.all(np.diag(sigma_new) < np.diag(sigma) + 1e-10)

    def test_psd_output(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((4, 4))
        sigma = A @ A.T + np.eye(4)
        B = rng.standard_normal((4, 4))
        Lambda = B @ B.T
        sigma_new = posterior_covariance(sigma, Lambda)
        assert is_psd(sigma_new)


class TestMarginalGain:
    def test_nonnegative(self):
        sigma = np.eye(4) * 2.0
        Lambda = np.eye(4) * 0.5
        gain = marginal_gain(sigma, Lambda)
        assert gain >= -1e-10

    def test_zero_precision(self):
        sigma = np.eye(4)
        Lambda = np.zeros((4, 4))
        gain = marginal_gain(sigma, Lambda)
        assert abs(gain) < 1e-10
