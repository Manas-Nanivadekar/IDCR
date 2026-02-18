"""
Efficiency-coverage Pareto frontier computation.

For IDCR with varying alpha and k, computes and plots the (coverage, volume)
Pareto-optimal frontier.
"""

from __future__ import annotations

import numpy as np
from typing import Any


def compute_pareto_frontier(
    coverages: np.ndarray,
    volumes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Pareto frontier for (coverage, -volume) bi-objective.

    Points on the frontier achieve the best tradeoff: highest coverage for
    a given volume, or smallest volume for a given coverage.

    Args:
        coverages: (n,) coverage values (higher is better).
        volumes: (n,) volume values (lower is better).

    Returns:
        (pareto_coverages, pareto_volumes) — sorted Pareto-optimal points.
    """
    n = len(coverages)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j has better or equal coverage AND better or equal volume
            if coverages[j] >= coverages[i] and volumes[j] <= volumes[i]:
                if coverages[j] > coverages[i] or volumes[j] < volumes[i]:
                    is_pareto[i] = False
                    break

    pareto_cov = coverages[is_pareto]
    pareto_vol = volumes[is_pareto]

    # Sort by coverage
    sort_idx = np.argsort(pareto_cov)
    return pareto_cov[sort_idx], pareto_vol[sort_idx]


def pareto_metrics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute Pareto frontier metrics from experiment results.

    Args:
        results: list of dicts, each with 'coverage', 'volume', and optionally
                 'method', 'alpha', 'k'.

    Returns:
        Dict with frontier points and dominated fraction.
    """
    coverages = np.array([r["coverage"] for r in results])
    volumes = np.array([r["volume"] for r in results])

    frontier_cov, frontier_vol = compute_pareto_frontier(coverages, volumes)

    # Area under Pareto frontier (higher = better)
    if len(frontier_cov) > 1:
        aupf = float(np.trapz(1.0 / np.maximum(frontier_vol, 1e-10), frontier_cov))
    else:
        aupf = 0.0

    return {
        "frontier_coverages": frontier_cov,
        "frontier_volumes": frontier_vol,
        "n_pareto_points": len(frontier_cov),
        "n_total_points": len(coverages),
        "dominated_fraction": 1.0 - len(frontier_cov) / max(len(coverages), 1),
        "area_under_pareto_frontier": aupf,
    }
