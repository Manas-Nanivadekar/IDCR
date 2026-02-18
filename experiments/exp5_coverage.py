"""
Experiment 5: Conformal coverage verification.

For each retrieval method (random, greedy, cosine), calibrate and test coverage
at multiple α levels. Verify ≥ (1-α) coverage for all methods.

The "prediction" is the posterior mean estimate of optimal allocation, computed
by blending the prior (equal weight) with the posterior precision-weighted mean.
This introduces realistic prediction error that conformal prediction must cover.

Expected: All methods achieve valid coverage; greedy achieves smaller volumes.
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
from src.retrieval.random_retrieval import RandomRetrieval
from src.retrieval.cosine_retrieval import CosineRetrieval
from src.utils.linear_algebra import posterior_covariance, log_det, project_simplex


def _noisy_prediction(user, sigma_post, rng):
    """Simulate a realistic noisy prediction of optimal allocation.

    Models a predictor that has the right posterior covariance but
    introduces prediction error from:
    1. Posterior sampling (draw from N(y*, Σ_post))
    2. Simplex projection (clipping to valid weights)

    This gives us realistic, non-zero Mahalanobis distances for
    conformal calibration.
    """
    y_star = compute_optimal_allocation(user)
    # Sample from posterior: y_pred ~ N(y*, Σ_post)
    noise = rng.multivariate_normal(np.zeros(len(y_star)), sigma_post)
    y_pred = y_star + noise
    y_pred = project_simplex(y_pred)
    return y_pred


def _run_retrieval(user, corpus, precision_fn, retrieval_cls, k, sigma_0, **kwargs):
    """Run retrieval and return final covariance."""
    retriever = retrieval_cls(sigma_0)
    result = retriever.retrieve(user, corpus, precision_fn, k, **kwargs)
    return result["sigma_trajectory"][-1]


def run_experiment(
    n_train: int = 200,
    n_cal: int = 500,
    n_test: int = 200,
    n_docs: int = 100,
    k: int = 5,
    alphas: list[float] | None = None,
    seed: int = 42,
    output_dir: str = "outputs/exp5_coverage",
):
    """Run conformal coverage experiment."""
    if alphas is None:
        alphas = [0.01, 0.05, 0.10, 0.20]

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    print("=== Experiment 5: Conformal Coverage ===")
    splits = generate_splits(n_train, n_cal, n_test, seed=seed)
    corpus = generate_corpus(n_docs, seed=seed)

    def precision_fn(doc, user):
        return compute_ground_truth_precision(doc, user)

    methods = {
        "Random": (RandomRetrieval, {"seed": seed}),
        "Greedy": (GreedyRetrieval, {}),
        "Cosine": (CosineRetrieval, {}),
    }

    results_by_method: dict[str, dict] = {}

    for method_name, (cls, extra_kwargs) in methods.items():
        print(f"\n  Method: {method_name}")
        coverage_by_alpha = {}

        for alpha in alphas:
            # ── Calibration ──────────────────────────────────────────
            cal_y_true = []
            cal_y_pred = []
            cal_sigma_inv = []

            for user in tqdm(splits["calibration"], desc=f"  Cal α={alpha}"):
                sigma_0 = compute_prior_covariance(user)
                sigma_post = _run_retrieval(
                    user, corpus, precision_fn, cls, k, sigma_0, **extra_kwargs)

                y_star = compute_optimal_allocation(user)
                y_pred = _noisy_prediction(user, sigma_post, rng)
                sigma_inv = np.linalg.inv(sigma_post)

                cal_y_true.append(y_star)
                cal_y_pred.append(y_pred)
                cal_sigma_inv.append(sigma_inv)

            cal_y_true = np.array(cal_y_true)
            cal_y_pred = np.array(cal_y_pred)
            cal_sigma_inv = np.array(cal_sigma_inv)

            # ── Test ─────────────────────────────────────────────────
            test_y_true = []
            test_y_pred = []
            test_sigma_inv = []

            for user in tqdm(splits["test"], desc=f"  Test α={alpha}"):
                sigma_0 = compute_prior_covariance(user)
                sigma_post = _run_retrieval(
                    user, corpus, precision_fn, cls, k, sigma_0, **extra_kwargs)

                y_star = compute_optimal_allocation(user)
                y_pred = _noisy_prediction(user, sigma_post, rng)
                sigma_inv = np.linalg.inv(sigma_post)

                test_y_true.append(y_star)
                test_y_pred.append(y_pred)
                test_sigma_inv.append(sigma_inv)

            test_y_true = np.array(test_y_true)
            test_y_pred = np.array(test_y_pred)
            test_sigma_inv = np.array(test_sigma_inv)

            cp = ConformalPredictor(alpha=alpha)
            cp.calibrate(cal_y_true, cal_y_pred, cal_sigma_inv)
            coverage = cp.check_coverage(test_y_true, test_y_pred, test_sigma_inv)

            sigma_test = np.array([np.linalg.inv(si) for si in test_sigma_inv])
            vols = [cp.compute_volume(s) for s in sigma_test]

            coverage_by_alpha[alpha] = {
                "coverage": coverage,
                "mean_volume": float(np.mean(vols)),
                "q_hat": cp.q_hat,
            }
            print(f"    α={alpha}: coverage={coverage:.3f}, "
                  f"target≥{1-alpha:.3f}, q̂={cp.q_hat:.4f}, "
                  f"vol={np.mean(vols):.6f}")

        results_by_method[method_name] = coverage_by_alpha

    # ── Plot ──────────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Coverage vs alpha
    for method_name, data in results_by_method.items():
        covs = [data[a]["coverage"] for a in alphas]
        ax1.plot(alphas, covs, "o-", markersize=6, label=method_name)
    targets = [1.0 - a for a in alphas]
    ax1.plot(alphas, targets, "k--", linewidth=2, label="Target (1-α)")
    ax1.set_xlabel("Miscoverage Level α")
    ax1.set_ylabel("Empirical Coverage")
    ax1.set_title("Coverage Guarantee")
    ax1.legend()

    # Volume comparison
    for method_name, data in results_by_method.items():
        vols = [data[a]["mean_volume"] for a in alphas]
        ax2.plot(alphas, vols, "s-", markersize=6, label=method_name)
    ax2.set_xlabel("α")
    ax2.set_ylabel("Mean Ellipsoid Volume")
    ax2.set_title("Prediction Set Efficiency")
    ax2.legend()

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "coverage_verification.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {fig_path}")


if __name__ == "__main__":
    run_experiment()
