"""
Random document retrieval baseline.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Callable

from src.utils.linear_algebra import log_det, posterior_covariance


class RandomRetrieval:
    """Uniform random document retrieval baseline."""

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
        seed: int = 0,
    ) -> dict[str, Any]:
        """Random retrieval of k documents.

        Args:
            user_profile: user profile dict.
            corpus: list of document dicts.
            precision_fn: callable(doc, user) → Λ_d(x).
            k: retrieval budget.
            seed: random seed.

        Returns:
            Dict with 'retrieved', 'sigma_trajectory', 'total_gain'.
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(corpus), size=min(k, len(corpus)), replace=False)
        selected = [corpus[i] for i in indices]

        sigma_current = self.sigma_0.copy()
        sigma_trajectory = [sigma_current.copy()]

        for doc in selected:
            Lambda_d = precision_fn(doc, user_profile)
            sigma_current = posterior_covariance(sigma_current, Lambda_d)
            sigma_trajectory.append(sigma_current.copy())

        total_gain = log_det(self.sigma_0) - log_det(sigma_current)
        return {
            "retrieved": selected,
            "sigma_trajectory": sigma_trajectory,
            "total_gain": float(total_gain),
        }
