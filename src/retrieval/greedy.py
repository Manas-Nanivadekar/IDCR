"""
Greedy submodular retrieval maximising log-det uncertainty reduction.

At each step selects:
    d* = argmax_d  log det(I + Σ_current^{1/2} Λ_d(x) Σ_current^{1/2})

Updates posterior covariance:
    Σ_new = (Σ^{-1} + Λ_{d*}(x))^{-1}
"""

from __future__ import annotations

import numpy as np
from typing import Any, Callable

from src.utils.linear_algebra import (
    matrix_sqrt,
    log_det,
    posterior_covariance,
    marginal_gain,
)


class GreedyRetrieval:
    """Greedy submodular retrieval for uncertainty reduction."""

    def __init__(self, sigma_0: np.ndarray):
        """
        Args:
            sigma_0: (d, d) prior covariance matrix.
        """
        self.sigma_0 = sigma_0.copy()
        self.d = sigma_0.shape[0]

    def retrieve(
        self,
        user_profile: dict[str, Any],
        corpus: list[dict[str, Any]],
        precision_fn: Callable,
        k: int,
    ) -> dict[str, Any]:
        """Greedy retrieval maximising log-det uncertainty reduction.

        Args:
            user_profile: user profile dict.
            corpus: list of document dicts.
            precision_fn: callable(doc, user) → Λ_d(x) ∈ R^{d×d}.
            k: retrieval budget.

        Returns:
            Dict with:
                - 'retrieved': list of k documents
                - 'sigma_trajectory': list of covariance matrices (k+1 entries)
                - 'marginal_gains': list of marginal log-det gains (k entries)
                - 'total_gain': total uncertainty reduction g(S)
        """
        S: list[dict[str, Any]] = []
        sigma_current = self.sigma_0.copy()
        remaining = list(corpus)
        marginal_gains: list[float] = []
        sigma_trajectory = [sigma_current.copy()]

        for _t in range(k):
            best_doc = None
            best_gain = -np.inf

            for doc in remaining:
                Lambda_d = precision_fn(doc, user_profile)
                gain = marginal_gain(sigma_current, Lambda_d)
                if gain > best_gain:
                    best_gain = gain
                    best_doc = doc

            if best_doc is None:
                break

            S.append(best_doc)
            remaining.remove(best_doc)
            marginal_gains.append(float(best_gain))

            # Update posterior covariance
            Lambda_best = precision_fn(best_doc, user_profile)
            sigma_current = posterior_covariance(sigma_current, Lambda_best)
            sigma_trajectory.append(sigma_current.copy())

        total_gain = sum(marginal_gains)
        return {
            "retrieved": S,
            "sigma_trajectory": sigma_trajectory,
            "marginal_gains": marginal_gains,
            "total_gain": total_gain,
        }


def greedy_objective(
    doc_set: list[dict[str, Any]],
    user_profile: dict[str, Any],
    precision_fn: Callable,
    sigma_0: np.ndarray,
) -> float:
    """Compute total uncertainty reduction g(S) for a document set.

    g(S) = log det Σ_0 - log det Σ(x, S)

    Args:
        doc_set: list of documents.
        user_profile: user profile dict.
        precision_fn: callable(doc, user) → Λ_d(x).
        sigma_0: prior covariance.

    Returns:
        Scalar uncertainty reduction.
    """
    sigma_inv = np.linalg.inv(sigma_0)
    for doc in doc_set:
        sigma_inv = sigma_inv + precision_fn(doc, user_profile)
    sigma_s = np.linalg.inv(sigma_inv)
    return log_det(sigma_0) - log_det(sigma_s)
