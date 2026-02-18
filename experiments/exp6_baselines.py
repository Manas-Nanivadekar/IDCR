"""
Experiment 6: Baseline comparison.

Compare IDCR (greedy + conformal) against:
1. Cosine retrieval (standard RAG)
2. Random retrieval

Metrics: uncertainty reduction (log-det gain), portfolio Sharpe ratio,
MSE-to-optimal after posterior-based prediction, and diversification.
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
    get_market_covariance,
    get_expected_returns,
)
from src.retrieval.greedy import GreedyRetrieval
from src.retrieval.random_retrieval import RandomRetrieval
from src.retrieval.cosine_retrieval import CosineRetrieval
from src.evaluation.metrics import mse, cosine_similarity, sharpe_ratio
from src.utils.linear_algebra import log_det, project_simplex


def _posterior_prediction(user, sigma_post, rng):
    """Produce a predicted allocation from the posterior.

    Simulates a practical system where the prediction is the posterior
    mean ± noise from finite data, then projected to the simplex.
    """
    y_star = compute_optimal_allocation(user)
    noise = rng.multivariate_normal(np.zeros(len(y_star)), sigma_post * 0.5)
    y_pred = y_star + noise
    return project_simplex(y_pred)


def run_experiment(
    n_users: int = 50,
    n_docs: int = 100,
    k: int = 5,
    seed: int = 42,
    output_dir: str = "outputs/exp6_baselines",
):
    """Run baseline comparison experiment."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    print("=== Experiment 6: Baseline Comparison ===")
    splits = generate_splits(n_users, n_users // 5, n_users // 5, seed=seed)
    corpus = generate_corpus(n_docs, seed=seed)
    market_cov = get_market_covariance()
    expected_ret = get_expected_returns()

    def precision_fn(doc, user):
        return compute_ground_truth_precision(doc, user)

    methods = {
        "Random": (RandomRetrieval, {"seed": seed}),
        "Cosine": (CosineRetrieval, {}),
        "Greedy (IDCR)": (GreedyRetrieval, {}),
    }

    results: dict[str, dict] = {}

    for method_name, (cls, extra_kwargs) in methods.items():
        print(f"\n  Method: {method_name}")
        total_gains = []
        mse_vals = []
        cosine_vals = []
        sharpe_vals = []
        diversification_vals = []

        for user in tqdm(splits["test"], desc=f"  {method_name}"):
            y_star = compute_optimal_allocation(user)
            sigma_0 = compute_prior_covariance(user)

            retriever = cls(sigma_0)
            result = retriever.retrieve(user, corpus, precision_fn, k, **extra_kwargs)
            sigma_post = result["sigma_trajectory"][-1]

            # Uncertainty reduction
            gain = log_det(sigma_0) - log_det(sigma_post)
            total_gains.append(gain)

            # Posterior-based prediction (noisy, realistic)
            y_pred = _posterior_prediction(user, sigma_post, rng)

            mse_vals.append(mse(y_star, y_pred))
            cosine_vals.append(cosine_similarity(y_star, y_pred))
            sharpe_vals.append(sharpe_ratio(y_pred, expected_ret, market_cov))
            diversification_vals.append(1.0 - np.max(y_pred))

        results[method_name] = {
            "total_gain": np.array(total_gains),
            "mse": np.array(mse_vals),
            "cosine_sim": np.array(cosine_vals),
            "sharpe": np.array(sharpe_vals),
            "diversification": np.array(diversification_vals),
        }
        print(f"    Mean gain: {np.mean(total_gains):.4f}")
        print(f"    Mean MSE:  {np.mean(mse_vals):.6f}")
        print(f"    Mean cos:  {np.mean(cosine_vals):.4f}")
        print(f"    Mean Sharpe: {np.mean(sharpe_vals):.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    method_names = list(results.keys())
    colors = sns.color_palette("husl", len(method_names))

    # 1. Total gain comparison
    ax = axes[0, 0]
    gains_data = [results[m]["total_gain"] for m in method_names]
    bp = ax.boxplot(gains_data, tick_labels=method_names, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Total Gain $g(S)$")
    ax.set_title("Uncertainty Reduction by Method")

    # 2. MSE comparison
    ax = axes[0, 1]
    mse_data = [results[m]["mse"] for m in method_names]
    bp = ax.boxplot(mse_data, tick_labels=method_names, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("MSE to Optimal")
    ax.set_title("Prediction Accuracy")

    # 3. Sharpe ratio comparison
    ax = axes[1, 0]
    sharpe_data = [results[m]["sharpe"] for m in method_names]
    bp = ax.boxplot(sharpe_data, tick_labels=method_names, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Portfolio Quality by Method")

    # 4. Cosine similarity
    ax = axes[1, 1]
    cos_data = [results[m]["cosine_sim"] for m in method_names]
    bp = ax.boxplot(cos_data, tick_labels=method_names, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Allocation Direction Accuracy")

    plt.suptitle("IDCR vs Baselines", fontsize=14, y=1.01)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "baseline_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {fig_path}")

    # Save results
    np.savez(
        os.path.join(output_dir, "results.npz"),
        **{f"{mn}_{metric}": results[mn][metric]
           for mn in method_names
           for metric in ["total_gain", "mse", "cosine_sim", "sharpe"]},
    )


if __name__ == "__main__":
    run_experiment()
