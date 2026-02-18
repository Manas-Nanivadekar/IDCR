"""
Experiment 1: Submodularity Verification.

For N users:
1. Run greedy retrieval k=1..max_k and verify marginal gains are non-increasing.
2. Run formal S⊆T tests to verify Δ(d|S) ≥ Δ(d|T).

Expected output:
- Monotonically decreasing greedy marginal gains curve.
- ~0 formal violations.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from tqdm import tqdm

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.user_profiles import generate_profiles
from src.data.documents import generate_corpus
from src.data.precision_ground_truth import (
    compute_ground_truth_precision,
    compute_prior_covariance,
)
from src.theory.submodularity import verify_submodularity


def run_experiment(
    n_users: int = 50,
    max_k: int = 20,
    n_trials: int = 100,
    n_docs: int = 200,
    seed: int = 42,
    output_dir: str = "outputs/exp1_submodularity",
):
    """Run the submodularity verification experiment."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Experiment 1: Submodularity Verification ===")
    print(f"  Users: {n_users}, Max k: {max_k}, Formal trials/user: {n_trials}")
    print(f"  Documents: {n_docs}")

    profiles = generate_profiles(n_users, seed=seed)
    corpus = generate_corpus(n_docs, seed=seed)

    def precision_fn(doc, user):
        return compute_ground_truth_precision(doc, user)

    all_greedy_gains = []
    all_greedy_diminishing = []
    all_formal_violation_rates = []
    all_mean_margins = []

    for i, user in enumerate(tqdm(profiles, desc="Processing users")):
        sigma_0 = compute_prior_covariance(user)
        result = verify_submodularity(
            user_profile=user,
            corpus=corpus,
            precision_fn=precision_fn,
            sigma_0=sigma_0,
            max_k=max_k,
            n_trials=n_trials,
            seed=seed + i,
        )
        all_greedy_gains.append(result["greedy_marginal_gains"])
        all_greedy_diminishing.append(result["greedy_is_diminishing"])
        all_formal_violation_rates.append(result["formal_violation_rate"])
        all_mean_margins.append(result["mean_margin"])

    # Aggregate statistics
    greedy_gains_matrix = np.array(all_greedy_gains)  # (n_users, max_k)
    grand_mean = np.mean(greedy_gains_matrix, axis=0)
    grand_std = np.std(greedy_gains_matrix, axis=0)
    pct_diminishing = np.mean(all_greedy_diminishing) * 100
    avg_formal_violation = np.mean(all_formal_violation_rates)
    avg_margin = np.mean(all_mean_margins)

    print(f"\n=== Results ===")
    print(f"  Greedy diminishing for {pct_diminishing:.1f}% of users")
    print(f"  Formal S⊆T violation rate: {avg_formal_violation:.4f}")
    print(f"  Mean margin Δ(d|S) - Δ(d|T): {avg_margin:.6f} (should be ≥ 0)")
    print(f"  Greedy gains (first 5 steps): {grand_mean[:5]}")

    # ── Plot ──────────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: Greedy marginal gains vs |S|
    ax = axes[0]
    steps = np.arange(1, len(grand_mean) + 1)
    ax.fill_between(steps, grand_mean - grand_std, grand_mean + grand_std,
                    alpha=0.3, color="steelblue")
    ax.plot(steps, grand_mean, "o-", color="steelblue", markersize=4,
            label="Mean greedy marginal gain")
    ax.set_xlabel("Step $|S|$")
    ax.set_ylabel("Marginal Gain $\\Delta(d^*|S)$")
    ax.set_title("Greedy Diminishing Returns")
    ax.legend()

    # Middle: Formal violation rate histogram
    ax = axes[1]
    ax.hist(all_formal_violation_rates, bins=20, color="coral",
            edgecolor="black", alpha=0.7)
    ax.axvline(avg_formal_violation, color="red", linestyle="--",
               label=f"Mean = {avg_formal_violation:.3f}")
    ax.set_xlabel("Formal Violation Rate per User")
    ax.set_ylabel("Count")
    ax.set_title("S⊆T Violation Rates")
    ax.legend()

    # Right: Margin distribution
    ax = axes[2]
    ax.hist(all_mean_margins, bins=20, color="mediumseagreen",
            edgecolor="black", alpha=0.7)
    ax.axvline(0, color="black", linestyle="-", linewidth=1)
    ax.set_xlabel("Mean Margin $\\Delta(d|S) - \\Delta(d|T)$")
    ax.set_ylabel("Count")
    ax.set_title("Submodularity Margin (should be ≥ 0)")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "submodularity_verification.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {fig_path}")

    # Save numerical results
    np.savez(
        os.path.join(output_dir, "results.npz"),
        greedy_gains=grand_mean,
        greedy_std=grand_std,
        pct_diminishing=pct_diminishing,
        formal_violation_rates=np.array(all_formal_violation_rates),
        mean_margins=np.array(all_mean_margins),
    )


if __name__ == "__main__":
    run_experiment()
