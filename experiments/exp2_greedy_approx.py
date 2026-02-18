"""
Experiment 2: Greedy Approximation Quality.

For small k, compare g(S_greedy) / g(S_best_found) by sampling many random
subsets. Verify the (1 - 1/e) ≈ 0.63 approximation guarantee.

Expected output: Ratio ≥ 0.63 across all settings.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.user_profiles import generate_profiles
from src.data.documents import generate_corpus
from src.data.precision_ground_truth import (
    compute_ground_truth_precision,
    compute_prior_covariance,
)
from src.theory.submodularity import verify_greedy_optimality_ratio


def run_experiment(
    n_users: int = 30,
    k_values: list[int] | None = None,
    n_random_samples: int = 10000,
    n_docs: int = 60,
    seed: int = 42,
    output_dir: str = "outputs/exp2_greedy_approx",
):
    """Run the greedy approximation quality experiment."""
    if k_values is None:
        k_values = [3, 4, 5, 6]

    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Experiment 2: Greedy Approximation Quality ===")
    print(f"  Users: {n_users}, k values: {k_values}")
    print(f"  Random samples/user: {n_random_samples}, Documents: {n_docs}")

    profiles = generate_profiles(n_users, seed=seed)
    corpus = generate_corpus(n_docs, seed=seed)

    def precision_fn(doc, user):
        return compute_ground_truth_precision(doc, user)

    results_by_k: dict[int, dict] = {}

    for k in k_values:
        print(f"\n  Processing k={k}...")
        ratios = []
        greedy_gains = []
        best_random_gains = []

        for i, user in enumerate(tqdm(profiles, desc=f"  k={k}")):
            sigma_0 = compute_prior_covariance(user)
            result = verify_greedy_optimality_ratio(
                user_profile=user,
                corpus=corpus,
                precision_fn=precision_fn,
                sigma_0=sigma_0,
                k=k,
                n_random_samples=n_random_samples,
                seed=seed + i,
            )
            ratios.append(result["ratio_vs_best"])
            greedy_gains.append(result["greedy_gain"])
            best_random_gains.append(result["best_random_gain"])

        results_by_k[k] = {
            "ratios": np.array(ratios),
            "greedy_gains": np.array(greedy_gains),
            "best_random_gains": np.array(best_random_gains),
        }
        print(f"    Mean ratio: {np.mean(ratios):.4f}")
        print(f"    Min ratio:  {np.min(ratios):.4f}")
        print(f"    All ≥ 0.63: {np.all(np.array(ratios) >= 0.63 - 0.01)}")

    # ── Plot ──────────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Box plot of ratios by k
    ax = axes[0]
    data_for_plot = [results_by_k[k]["ratios"] for k in k_values]
    bp = ax.boxplot(data_for_plot, tick_labels=[str(k) for k in k_values],
                    patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
    ax.axhline(1 - 1 / np.e, color="red", linestyle="--",
               label=f"1 - 1/e ≈ {1 - 1/np.e:.3f}")
    ax.axhline(1.0, color="green", linestyle=":", alpha=0.5, label="Optimal")
    ax.set_xlabel("Budget k")
    ax.set_ylabel("$g(S_{greedy}) / g(S_{best})$")
    ax.set_title("Greedy Approximation Ratio")
    ax.legend()

    # Right: Greedy vs best gain scatter
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(k_values)))
    for k, color in zip(k_values, colors):
        r = results_by_k[k]
        ax.scatter(r["best_random_gains"], r["greedy_gains"],
                   alpha=0.6, color=color, label=f"k={k}", s=40)
    lo = 0
    hi = max(max(r["greedy_gains"].max(), r["best_random_gains"].max())
             for r in results_by_k.values())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("Best Random $g(S)$")
    ax.set_ylabel("Greedy $g(S)$")
    ax.set_title("Greedy vs Best Random Subset")
    ax.legend()

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "greedy_approximation.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {fig_path}")

    # Summary
    print(f"\n=== Summary ===")
    for k in k_values:
        ratios = results_by_k[k]["ratios"]
        print(f"  k={k}: mean={np.mean(ratios):.4f}, "
              f"min={np.min(ratios):.4f}, "
              f"≥0.63 = {np.all(ratios >= 0.63 - 0.01)}")


if __name__ == "__main__":
    run_experiment()
