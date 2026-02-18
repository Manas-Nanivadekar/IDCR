"""
Interaction tensor T_ijk computation and analysis.

The third-order interaction tensor captures document synergies:
    T_ijk = g({i,j,k}) - g({i,j}) - g({i,k}) - g({j,k}) + g({i}) + g({j}) + g({k})

where g(S) = log det Σ_0 - log det Σ(x, S) is the uncertainty reduction.

- T_ijk > 0: Synergy (documents together reveal more than sum of pairwise)
- T_ijk < 0: Redundancy (overlapping information)
- T_ijk = 0: No higher-order interaction
"""

from __future__ import annotations

import numpy as np
from typing import Any, Callable
from itertools import combinations
from tqdm import tqdm

from src.utils.linear_algebra import log_det


def _compute_g(
    doc_indices: list[int],
    corpus: list[dict[str, Any]],
    user_profile: dict[str, Any],
    precision_fn: Callable,
    sigma_0: np.ndarray,
    log_det_sigma_0: float,
) -> float:
    """Compute uncertainty reduction g(S) for a set of documents.

    g(S) = log det Σ_0 - log det (Σ_0^{-1} + Σ_{d∈S} Λ_d)^{-1}

    When precision_fn supports synergy (via retrieved_ids), documents
    are added sequentially so synergy bonuses activate.

    Args:
        doc_indices: list of indices into corpus.
        corpus: document list.
        user_profile: user profile dict.
        precision_fn: callable(doc, user[, retrieved_ids]) → Λ_d(x).
        sigma_0: prior covariance.
        log_det_sigma_0: precomputed log det Σ_0.

    Returns:
        Scalar g(S).
    """
    sigma_inv = np.linalg.inv(sigma_0)
    retrieved_so_far: set[int] = set()
    for idx in doc_indices:
        doc = corpus[idx]
        try:
            Lambda = precision_fn(doc, user_profile, retrieved_so_far)
        except TypeError:
            # precision_fn doesn't accept retrieved_ids
            Lambda = precision_fn(doc, user_profile)
        sigma_inv = sigma_inv + Lambda
        retrieved_so_far.add(doc.get("doc_id", idx))
    sigma_s = np.linalg.inv(sigma_inv)
    return log_det_sigma_0 - log_det(sigma_s)


def compute_interaction_tensor(
    corpus: list[dict[str, Any]],
    user_profile: dict[str, Any],
    precision_fn: Callable,
    sigma_0: np.ndarray,
    max_docs: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Compute third-order interaction tensor T_ijk.

    T_ijk = g({i,j,k}) - g({i,j}) - g({i,k}) - g({j,k})
            + g({i}) + g({j}) + g({k})

    For efficiency, only computes for the first max_docs documents.

    Args:
        corpus: list of document dicts.
        user_profile: user profile dict.
        precision_fn: callable(doc, user) → Λ_d(x).
        sigma_0: prior covariance.
        max_docs: maximum number of documents to consider (for efficiency).
        show_progress: if True, show tqdm progress bar.

    Returns:
        Dict with:
            - 'tensor': (n, n, n) interaction tensor
            - 'max_synergy': maximum T_ijk (positive = synergy)
            - 'min_synergy': minimum T_ijk (negative = redundancy)
            - 'synergy_fraction': fraction of triples with T_ijk > threshold
            - 'redundancy_fraction': fraction with T_ijk < -threshold
    """
    n = min(len(corpus), max_docs) if max_docs is not None else len(corpus)
    log_det_s0 = log_det(sigma_0)

    # Cache g values for singletons and pairs
    g_single: dict[int, float] = {}
    g_pair: dict[tuple[int, int], float] = {}
    g_triple: dict[tuple[int, int, int], float] = {}

    # Singletons
    for i in range(n):
        g_single[i] = _compute_g([i], corpus, user_profile, precision_fn,
                                  sigma_0, log_det_s0)

    # Pairs
    pairs = list(combinations(range(n), 2))
    for i, j in tqdm(pairs, desc="Computing pairs", disable=not show_progress):
        g_pair[(i, j)] = _compute_g([i, j], corpus, user_profile, precision_fn,
                                     sigma_0, log_det_s0)

    # Triples
    T = np.zeros((n, n, n))
    triples = list(combinations(range(n), 3))
    for i, j, k in tqdm(triples, desc="Computing triples", disable=not show_progress):
        g_ijk = _compute_g([i, j, k], corpus, user_profile, precision_fn,
                           sigma_0, log_det_s0)
        g_triple[(i, j, k)] = g_ijk

        t_val = (g_ijk
                 - g_pair[(i, j)] - g_pair[(i, k)] - g_pair[(j, k)]
                 + g_single[i] + g_single[j] + g_single[k])

        # Fill all permutations (tensor is symmetric)
        for pi, pj, pk in [
            (i, j, k), (i, k, j), (j, i, k),
            (j, k, i), (k, i, j), (k, j, i),
        ]:
            T[pi, pj, pk] = t_val

    # Statistics
    threshold = 1e-6
    nonzero_triples = T[np.triu_indices(n, k=1)]  # upper triangle entries
    # Actually we need the unique triples
    unique_vals = [T[i, j, k] for i, j, k in combinations(range(n), 3)]
    unique_vals = np.array(unique_vals) if unique_vals else np.array([0.0])

    return {
        "tensor": T,
        "max_synergy": float(np.max(unique_vals)),
        "min_synergy": float(np.min(unique_vals)),
        "mean_interaction": float(np.mean(unique_vals)),
        "synergy_fraction": float(np.mean(unique_vals > threshold)),
        "redundancy_fraction": float(np.mean(unique_vals < -threshold)),
        "g_single": g_single,
        "g_pair": g_pair,
        "g_triple": g_triple,
    }


def compute_pairwise_interaction(
    corpus: list[dict[str, Any]],
    user_profile: dict[str, Any],
    precision_fn: Callable,
    sigma_0: np.ndarray,
    max_docs: int | None = None,
) -> np.ndarray:
    """Compute pairwise interaction matrix (faster alternative for large corpora).

    I_ij = g({i,j}) - g({i}) - g({j})

    Positive: synergy. Negative: redundancy.

    Args:
        corpus, user_profile, precision_fn, sigma_0: as above.
        max_docs: maximum number of documents.

    Returns:
        (n, n) pairwise interaction matrix.
    """
    n = min(len(corpus), max_docs) if max_docs is not None else len(corpus)
    log_det_s0 = log_det(sigma_0)

    g_single = np.zeros(n)
    for i in range(n):
        g_single[i] = _compute_g([i], corpus, user_profile, precision_fn,
                                  sigma_0, log_det_s0)

    I_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            g_ij = _compute_g([i, j], corpus, user_profile, precision_fn,
                              sigma_0, log_det_s0)
            val = g_ij - g_single[i] - g_single[j]
            I_mat[i, j] = val
            I_mat[j, i] = val

    return I_mat
