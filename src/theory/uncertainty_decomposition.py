"""
Uncertainty decomposition into user, document, and irreducible components.

Decomposes the total uncertainty Σ(x, S) into:
1. User uncertainty: uncertainty from the profile itself
2. Document uncertainty: reducible uncertainty that retrieval can address
3. Irreducible uncertainty: fundamental noise in the recommendation
"""

from __future__ import annotations

import numpy as np
from typing import Any, Callable

from src.utils.linear_algebra import log_det, posterior_covariance, is_psd


def decompose_uncertainty(
    user: dict[str, Any],
    sigma_0: np.ndarray,
    sigma_post: np.ndarray,
    sigma_irreducible: np.ndarray | None = None,
) -> dict[str, float]:
    """Decompose total uncertainty into components.

    Total = User + Document-reducible + Irreducible

    In log-det space:
    - Total: log det Σ_0
    - After retrieval: log det Σ(x, S)
    - Document-reducible: log det Σ_0 - log det Σ(x, S)
    - Irreducible: log det Σ_irreducible (lower bound on achievable uncertainty)

    Args:
        user: user profile dict.
        sigma_0: (d, d) prior covariance (before any retrieval).
        sigma_post: (d, d) posterior covariance (after retrieval).
        sigma_irreducible: (d, d) irreducible noise covariance (optional).

    Returns:
        Dict with uncertainty components.
    """
    d = sigma_0.shape[0]

    total = log_det(sigma_0)
    after_retrieval = log_det(sigma_post)
    document_reduction = total - after_retrieval

    result = {
        "total_uncertainty": total,
        "posterior_uncertainty": after_retrieval,
        "document_reduction": document_reduction,
        "reduction_fraction": document_reduction / max(abs(total), 1e-10),
    }

    if sigma_irreducible is not None:
        irreducible = log_det(sigma_irreducible)
        result["irreducible_uncertainty"] = irreducible
        result["achievable_reduction"] = total - irreducible
        result["efficiency"] = document_reduction / max(total - irreducible, 1e-10)

    return result


def compute_per_dimension_uncertainty(
    sigma: np.ndarray,
) -> np.ndarray:
    """Extract per-dimension (diagonal) uncertainties.

    Args:
        sigma: (d, d) covariance matrix.

    Returns:
        (d,) vector of per-dimension standard deviations.
    """
    return np.sqrt(np.maximum(np.diag(sigma), 0.0))


def analyze_uncertainty_trajectory(
    sigma_trajectory: list[np.ndarray],
) -> dict[str, Any]:
    """Analyze how uncertainty evolves through retrieval steps.

    Args:
        sigma_trajectory: list of (d, d) covariance matrices,
                         one per retrieval step (including initial).

    Returns:
        Dict with per-step uncertainty metrics.
    """
    log_dets = [log_det(s) for s in sigma_trajectory]
    per_dim = [compute_per_dimension_uncertainty(s) for s in sigma_trajectory]

    marginal_reductions = [
        log_dets[i] - log_dets[i + 1]
        for i in range(len(log_dets) - 1)
    ]

    return {
        "log_dets": np.array(log_dets),
        "per_dim_std": np.array(per_dim),
        "marginal_reductions": np.array(marginal_reductions),
        "total_reduction": log_dets[0] - log_dets[-1] if len(log_dets) > 1 else 0.0,
        "n_steps": len(log_dets) - 1,
    }
