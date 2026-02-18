"""
Experiment 7: Pareto frontier analysis.

Sweep over (α, k) configurations and plot the Pareto frontier
of efficiency (volume) vs coverage.

Uses noisy posterior predictions to produce realistic coverage and volume.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.user_profiles import generate_splits
from src.data.documents import generate_corpus
from src.data.precision_ground_truth import (
    compute_ground_truth_precision,
    compute_prior_covariance,
    compute_optimal_allocation,
)
from src.models.conformal import ConformalPredictor
from src.retrieval.greedy import GreedyRetrieval
from src.evaluation.pareto import compute_pareto_frontier
from src.utils.linear_algebra import log_det, project_simplex


def _noisy_prediction(user, sigma_post, rng):
    """Produce a noisy prediction for realistic conformal calibration."""
    y_star = compute_optimal_allocation(user)
    noise = rng.multivariate_normal(np.zeros(len(y_star)), sigma_post)
    return project_simplex(y_star + noise)


def run_experiment(
    n_cal: int = 80,
    n_test: int = 80,
    n_docs: int = 100,
    k_values: list[int] | None = None,
    alphas: list[float] | None = None,
    seed: int = 42,
    output_dir: str = "outputs/exp7_pareto",
):
    """Run the Pareto frontier experiment."""
    if k_values is None:
        k_values = [1, 2, 3, 5, 8, 10, 15]
    if alphas is None:
        alphas = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    print("=== Experiment 7: Pareto Frontier ===")
    splits = generate_splits(200, n_cal, n_test, seed=seed)
    corpus = generate_corpus(n_docs, seed=seed)

    def precision_fn(doc, user):
        return compute_ground_truth_precision(doc, user)

    all_points = []

    for k in tqdm(k_values, desc="Budget k"):
        for alpha in alphas:
            # ── Calibration ──────────────────────────────────────────
            cal_y_true, cal_y_pred, cal_sigma_inv = [], [], []
            cal_sigmas = []

            for user in splits["calibration"]:
                sigma_0 = compute_prior_covariance(user)
                greedy = GreedyRetrieval(sigma_0)
                result = greedy.retrieve(user, corpus, precision_fn, k)
                sigma_post = result["sigma_trajectory"][-1]

                y_star = compute_optimal_allocation(user)
                y_pred = _noisy_prediction(user, sigma_post, rng)
                sigma_inv = np.linalg.inv(sigma_post)

                cal_y_true.append(y_star)
                cal_y_pred.append(y_pred)
                cal_sigma_inv.append(sigma_inv)

            # ── Test ─────────────────────────────────────────────────
            test_y_true, test_y_pred, test_sigma_inv = [], [], []
            test_sigmas = []

            for user in splits["test"]:
                sigma_0 = compute_prior_covariance(user)
                greedy = GreedyRetrieval(sigma_0)
                result = greedy.retrieve(user, corpus, precision_fn, k)
                sigma_post = result["sigma_trajectory"][-1]

                y_star = compute_optimal_allocation(user)
                y_pred = _noisy_prediction(user, sigma_post, rng)
                sigma_inv = np.linalg.inv(sigma_post)

                test_y_true.append(y_star)
                test_y_pred.append(y_pred)
                test_sigma_inv.append(sigma_inv)
                test_sigmas.append(sigma_post)

            cal_y_true = np.array(cal_y_true)
            cal_y_pred = np.array(cal_y_pred)
            cal_sigma_inv = np.array(cal_sigma_inv)
            test_y_true = np.array(test_y_true)
            test_y_pred = np.array(test_y_pred)
            test_sigma_inv = np.array(test_sigma_inv)

            cp = ConformalPredictor(alpha=alpha)
            cp.calibrate(cal_y_true, cal_y_pred, cal_sigma_inv)
            coverage = cp.check_coverage(test_y_true, test_y_pred, test_sigma_inv)

            vols = [cp.compute_volume(s) for s in test_sigmas]
            mean_vol = float(np.mean(vols))

            all_points.append({
                "k": k,
                "alpha": alpha,
                "coverage": coverage,
                "volume": mean_vol,
                "q_hat": cp.q_hat,
                "label": f"k={k}, α={alpha}",
            })

            print(f"  k={k}, α={alpha}: cov={coverage:.3f}, "
                  f"vol={mean_vol:.4e}, q̂={cp.q_hat:.4f}")

    coverages = np.array([p["coverage"] for p in all_points])
    volumes = np.array([p["volume"] for p in all_points])

    # Filter out zero-volume points for Pareto
    valid = volumes > 0
    if np.any(valid):
        frontier_cov, frontier_vol = compute_pareto_frontier(
            coverages[valid], volumes[valid])
    else:
        frontier_cov = coverages
        frontier_vol = volumes

    # ── Plot ──────────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))

    # Color by k
    k_unique = sorted(set(p["k"] for p in all_points))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(k_unique)))
    k_to_color = dict(zip(k_unique, cmap))

    for pt in all_points:
        ax.scatter(pt["coverage"], max(pt["volume"], 1e-20),
                   color=k_to_color[pt["k"]], s=50, alpha=0.7, zorder=2)

    if np.any(valid) and len(frontier_cov) > 1:
        ax.plot(frontier_cov, frontier_vol, "r-o", linewidth=2, markersize=6,
                zorder=3, label="Pareto Frontier")

    # Legend for k values
    for kv, color in k_to_color.items():
        ax.scatter([], [], color=color, s=50, label=f"k={kv}")

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Mean Ellipsoid Volume")
    if np.any(volumes > 0):
        ax.set_yscale("log")
    ax.set_title("Efficiency-Coverage Pareto Frontier")
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "pareto_frontier.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {fig_path}")

    # Summary
    if np.any(valid):
        print(f"\n  Pareto-optimal points: {len(frontier_cov)}/{np.sum(valid)}")
        for cov, vol in zip(frontier_cov, frontier_vol):
            print(f"    Coverage={cov:.3f}, Volume={vol:.4e}")
    else:
        print("\n  Warning: all volumes are zero — check predictor noise")


if __name__ == "__main__":
    run_experiment()
