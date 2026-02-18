"""
Cosine similarity retrieval baseline (standard RAG).

Retrieves documents by cosine similarity between user profile embedding
and document embeddings.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Callable

from src.utils.linear_algebra import log_det, posterior_covariance


class CosineRetrieval:
    """Standard RAG baseline: retrieve by cosine similarity."""

    def __init__(self, sigma_0: np.ndarray):
        self.sigma_0 = sigma_0.copy()
        self.d = sigma_0.shape[0]

    def _embed_user(self, user: dict[str, Any]) -> np.ndarray:
        """Simple user embedding from profile features."""
        from src.models.user_encoder import profile_to_features
        return profile_to_features(user)

    def _embed_docs(self, corpus: list[dict[str, Any]]) -> np.ndarray:
        """Simple document embeddings from metadata."""
        from src.models.document_encoder import document_to_features
        return np.stack([document_to_features(doc) for doc in corpus])

    def retrieve(
        self,
        user_profile: dict[str, Any],
        corpus: list[dict[str, Any]],
        precision_fn: Callable,
        k: int,
    ) -> dict[str, Any]:
        """Retrieve top-k documents by cosine similarity.

        Args:
            user_profile: user profile dict.
            corpus: list of document dicts.
            precision_fn: callable(doc, user) → Λ_d(x) (for computing gain).
            k: retrieval budget.

        Returns:
            Dict with 'retrieved', 'sigma_trajectory', 'total_gain', 'similarities'.
        """
        user_emb = self._embed_user(user_profile)
        doc_embs = self._embed_docs(corpus)

        # Pad to same dimension
        max_dim = max(len(user_emb), doc_embs.shape[1])
        user_padded = np.pad(user_emb, (0, max(0, max_dim - len(user_emb))))
        doc_padded = np.pad(doc_embs, ((0, 0), (0, max(0, max_dim - doc_embs.shape[1]))))

        # Cosine similarity
        user_norm = np.linalg.norm(user_padded)
        if user_norm > 0:
            user_padded = user_padded / user_norm
        doc_norms = np.linalg.norm(doc_padded, axis=1, keepdims=True)
        doc_norms = np.maximum(doc_norms, 1e-10)
        doc_padded = doc_padded / doc_norms

        similarities = doc_padded @ user_padded  # (n,)

        # Top-k
        top_k_idx = np.argsort(similarities)[::-1][:k]
        selected = [corpus[i] for i in top_k_idx]

        # Compute covariance trajectory
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
            "similarities": similarities[top_k_idx].tolist(),
        }
