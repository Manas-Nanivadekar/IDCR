"""
Linear algebra utilities for PSD operations, log-determinant, and simplex projection.
All operations are designed to be numerically stable for the precision/covariance
matrices used throughout the IDCR framework.
"""

import numpy as np
from scipy import linalg


# ---------------------------------------------------------------------------
# PSD operations
# ---------------------------------------------------------------------------

def nearest_psd(M: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Project a symmetric matrix to the nearest PSD matrix.

    Uses eigenvalue clipping: negative eigenvalues are replaced by epsilon.

    Args:
        M: (d, d) symmetric matrix.
        epsilon: minimum eigenvalue in the output.

    Returns:
        (d, d) PSD matrix closest to M in Frobenius norm.
    """
    M_sym = (M + M.T) / 2
    eigvals, eigvecs = np.linalg.eigh(M_sym)
    eigvals = np.maximum(eigvals, epsilon)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def matrix_sqrt(M: np.ndarray) -> np.ndarray:
    """Symmetric PSD square root via eigendecomposition.

    Returns S such that S @ S = M (up to numerical precision).

    Args:
        M: (d, d) symmetric PSD matrix.

    Returns:
        (d, d) symmetric PSD square root.
    """
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def matrix_sqrt_inv(M: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Inverse of the symmetric PSD square root.

    Returns S^{-1} such that S^{-1} @ M @ S^{-1} = I.

    Args:
        M: (d, d) symmetric PSD matrix.
        epsilon: regularization for small eigenvalues.

    Returns:
        (d, d) inverse square root.
    """
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, epsilon)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


# ---------------------------------------------------------------------------
# Log-determinant
# ---------------------------------------------------------------------------

def log_det(M: np.ndarray) -> float:
    """Numerically stable log-determinant for PSD matrices via Cholesky.

    Falls back to eigenvalue computation if Cholesky fails.

    Args:
        M: (d, d) symmetric PSD matrix.

    Returns:
        log det(M).
    """
    try:
        L = np.linalg.cholesky(M)
        return 2.0 * np.sum(np.log(np.diag(L)))
    except np.linalg.LinAlgError:
        # Fall back to eigenvalue method
        eigvals = np.linalg.eigvalsh(M)
        eigvals = np.maximum(eigvals, 1e-30)
        return np.sum(np.log(eigvals))


def log_det_ratio(sigma_0: np.ndarray, sigma_s: np.ndarray) -> float:
    """Compute log det(Σ_0) - log det(Σ_S) = uncertainty reduction g(S).

    Args:
        sigma_0: (d, d) prior covariance.
        sigma_s: (d, d) posterior covariance after retrieval.

    Returns:
        Non-negative scalar measuring uncertainty reduction.
    """
    return log_det(sigma_0) - log_det(sigma_s)


# ---------------------------------------------------------------------------
# Mahalanobis distance
# ---------------------------------------------------------------------------

def mahalanobis(x: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray) -> float:
    """Mahalanobis distance: (x - μ)^T Σ^{-1} (x - μ).

    Args:
        x: (d,) observation vector.
        mu: (d,) mean vector.
        sigma_inv: (d, d) precision matrix.

    Returns:
        Non-negative scalar Mahalanobis distance.
    """
    diff = x - mu
    return float(diff @ sigma_inv @ diff)


def mahalanobis_batch(
    x: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray
) -> np.ndarray:
    """Batched Mahalanobis distance.

    Args:
        x: (n, d) observation vectors.
        mu: (n, d) mean vectors.
        sigma_inv: (n, d, d) precision matrices.

    Returns:
        (n,) Mahalanobis distances.
    """
    diff = x - mu  # (n, d)
    # Einsum: for each i, diff[i] @ sigma_inv[i] @ diff[i]
    return np.einsum("ij,ijk,ik->i", diff, sigma_inv, diff)


# ---------------------------------------------------------------------------
# Simplex projection
# ---------------------------------------------------------------------------

def project_simplex(w: np.ndarray) -> np.ndarray:
    """Project vector onto the probability simplex {w >= 0, sum(w) = 1}.

    Uses the algorithm from Duchi et al. (2008).

    Args:
        w: (d,) vector.

    Returns:
        (d,) vector on the simplex.
    """
    d = len(w)
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u * np.arange(1, d + 1) > cssv)[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(w - theta, 0.0)


# ---------------------------------------------------------------------------
# Posterior covariance update
# ---------------------------------------------------------------------------

def posterior_covariance(
    sigma: np.ndarray, precision_update: np.ndarray
) -> np.ndarray:
    """Bayesian covariance update: Σ_new = (Σ^{-1} + Λ)^{-1}.

    Args:
        sigma: (d, d) current covariance matrix.
        precision_update: (d, d) precision contribution Λ_d(x).

    Returns:
        (d, d) updated covariance matrix.
    """
    sigma_inv = np.linalg.inv(sigma)
    new_inv = sigma_inv + precision_update
    return np.linalg.inv(new_inv)


def marginal_gain(
    sigma_current: np.ndarray, precision_d: np.ndarray
) -> float:
    """Marginal log-det gain from adding one document.

    Δ(d | S) = log det(I + Σ^{1/2} Λ_d Σ^{1/2})

    Args:
        sigma_current: (d, d) current posterior covariance.
        precision_d: (d, d) document's precision contribution.

    Returns:
        Non-negative marginal gain scalar.
    """
    d = sigma_current.shape[0]
    sqrt_sigma = matrix_sqrt(sigma_current)
    M = sqrt_sigma @ precision_d @ sqrt_sigma
    return log_det(np.eye(d) + M)


def is_psd(M: np.ndarray, tol: float = -1e-8) -> bool:
    """Check whether a matrix is positive semidefinite.

    Args:
        M: (d, d) symmetric matrix.
        tol: tolerance for smallest eigenvalue.

    Returns:
        True if M is PSD (up to tolerance).
    """
    eigvals = np.linalg.eigvalsh(M)
    return bool(np.all(eigvals >= tol))
