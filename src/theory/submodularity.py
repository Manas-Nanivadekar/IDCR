"""
Submodularity verification utilities.

Empirically verifies the diminishing returns property of the log-det
uncertainty reduction function g(S) = log det Σ_0 - log det Σ(x, S).

Two verification modes:
1. Greedy marginal gains: verify g(S_t) - g(S_{t-1}) is non-increasing
   along the greedy path (consequence of submodularity).
2. Formal definition: for random S ⊆ T and random d ∉ T, check
   Δ(d|S) ≥ Δ(d|T).
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


def verify_submodularity(
    user_profile: dict[str, Any],
    corpus: list[dict[str, Any]],
    precision_fn: Callable,
    sigma_0: np.ndarray,
    max_k: int = 20,
    n_trials: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Empirically verify diminishing returns of the log-det objective.

    Uses two tests:
    1. Greedy path: run greedy retrieval for k steps and check that
       marginal gains are non-increasing (a necessary consequence of
       submodularity).
    2. Formal (S ⊆ T) test: for random S ⊆ T and d ∉ T, verify
       Δ(d|S) ≥ Δ(d|T).

    Args:
        user_profile: user profile dict.
        corpus: list of document dicts.
        precision_fn: callable(doc, user) → Λ_d(x).
        sigma_0: (d, d) prior covariance matrix.
        max_k: maximum number of documents to retrieve.
        n_trials: number of random (S, T, d) trials for formal test.
        seed: random seed.

    Returns:
        Dict with:
            - 'greedy_marginal_gains': (max_k,) gains along greedy path
            - 'greedy_is_diminishing': bool — True if greedy gains decrease
            - 'formal_violation_rate': fraction of S⊆T trials that violate
            - 'mean_margin': mean of Δ(d|S) - Δ(d|T) (should be ≥ 0)
            - 'all_margins': (n_trials,) raw margin values
    """
    rng = np.random.default_rng(seed)
    d = sigma_0.shape[0]
    k = min(max_k, len(corpus))

    # ── Test 1: Greedy path diminishing returns ──────────────────────────
    from src.retrieval.greedy import GreedyRetrieval
    greedy = GreedyRetrieval(sigma_0)
    result = greedy.retrieve(user_profile, corpus, precision_fn, k)
    greedy_gains = np.array(result["marginal_gains"])

    # Check monotonicity
    greedy_is_diminishing = True
    greedy_max_violation = 0.0
    for i in range(1, len(greedy_gains)):
        increase = greedy_gains[i] - greedy_gains[i - 1]
        if increase > 1e-10:
            greedy_is_diminishing = False
            greedy_max_violation = max(greedy_max_violation, increase)

    # ── Test 2: Formal S ⊆ T definition ─────────────────────────────────
    # For random S ⊂ T ⊂ corpus and random d ∉ T:
    #   Δ(d | S) ≥ Δ(d | T)
    margins = np.zeros(n_trials)
    n_violations = 0

    for trial in range(n_trials):
        # Pick a random set T of size t_size and a subset S of size s_size
        t_size = rng.integers(2, min(k, len(corpus) - 1) + 1)
        s_size = rng.integers(1, t_size)

        indices_T = rng.choice(len(corpus), size=t_size, replace=False)
        indices_S = rng.choice(indices_T, size=s_size, replace=False)

        # Pick d ∉ T
        remaining = [i for i in range(len(corpus)) if i not in indices_T]
        if len(remaining) == 0:
            continue
        d_idx = rng.choice(remaining)

        # Build Σ(S) by accumulating precision from S
        sigma_S = sigma_0.copy()
        sigma_inv_S = np.linalg.inv(sigma_S)
        for idx in indices_S:
            Lambda = precision_fn(corpus[idx], user_profile)
            sigma_inv_S = sigma_inv_S + Lambda
        sigma_S = np.linalg.inv(sigma_inv_S)

        # Build Σ(T) by accumulating precision from T
        sigma_T = sigma_0.copy()
        sigma_inv_T = np.linalg.inv(sigma_T)
        for idx in indices_T:
            Lambda = precision_fn(corpus[idx], user_profile)
            sigma_inv_T = sigma_inv_T + Lambda
        sigma_T = np.linalg.inv(sigma_inv_T)

        # Compute Δ(d|S) and Δ(d|T)
        Lambda_d = precision_fn(corpus[d_idx], user_profile)
        gain_S = marginal_gain(sigma_S, Lambda_d)
        gain_T = marginal_gain(sigma_T, Lambda_d)

        margins[trial] = gain_S - gain_T

        if gain_S < gain_T - 1e-10:
            n_violations += 1

    return {
        "greedy_marginal_gains": greedy_gains,
        "greedy_is_diminishing": greedy_is_diminishing,
        "greedy_max_violation": greedy_max_violation,
        "formal_violation_rate": n_violations / n_trials,
        "mean_margin": float(np.mean(margins)),
        "all_margins": margins,
    }


def verify_greedy_optimality_ratio(
    user_profile: dict[str, Any],
    corpus: list[dict[str, Any]],
    precision_fn: Callable,
    sigma_0: np.ndarray,
    k: int = 5,
    n_random_samples: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare greedy objective to random subsets to estimate approximation ratio.

    For budget k, compares g(S_greedy) against g(S_random) for many random
    subsets. Reports ratio relative to the best random sample found.

    Args:
        user_profile: user profile dict.
        corpus: list of document dicts.
        precision_fn: callable(doc, user) → Λ_d(x).
        sigma_0: prior covariance.
        k: retrieval budget.
        n_random_samples: number of random subsets to sample.
        seed: random seed.

    Returns:
        Dict with 'greedy_gain', 'best_random_gain', 'mean_random_gain',
        'ratio_vs_best', 'ratio_distribution'.
    """
    from src.retrieval.greedy import GreedyRetrieval, greedy_objective

    rng = np.random.default_rng(seed)

    # Greedy
    greedy = GreedyRetrieval(sigma_0)
    result = greedy.retrieve(user_profile, corpus, precision_fn, k)
    greedy_gain = result["total_gain"]

    # Random subsets
    random_gains = np.zeros(n_random_samples)
    for i in range(n_random_samples):
        indices = rng.choice(len(corpus), size=min(k, len(corpus)), replace=False)
        doc_set = [corpus[j] for j in indices]
        random_gains[i] = greedy_objective(doc_set, user_profile, precision_fn, sigma_0)

    best_random = float(np.max(random_gains))

    return {
        "greedy_gain": greedy_gain,
        "best_random_gain": best_random,
        "mean_random_gain": float(np.mean(random_gains)),
        "ratio_vs_best": greedy_gain / best_random if best_random > 0 else float("inf"),
        "random_gains": random_gains,
    }
