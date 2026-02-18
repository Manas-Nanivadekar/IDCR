"""
Experiment 4: Interaction tensor analysis.

Compute pairwise interactions for different corpus types and verify that:
1. Redundant corpus → mostly I_ij < 0
2. Synergistic corpus → mostly I_ij > 0
3. Mixed corpus → mixture of both
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.user_profiles import generate_profiles
from src.data.precision_ground_truth import compute_prior_covariance
from src.data.synergy_control import (
    create_synergistic_corpus,
    create_redundant_corpus,
    create_mixed_corpus,
    make_precision_fn,
)
from src.theory.interaction_tensor import (
    compute_interaction_tensor,
    compute_pairwise_interaction,
)


def run_experiment(
    n_docs: int = 12,
    seed: int = 42,
    output_dir: str = "outputs/exp4_interaction_tensor",
):
    """Run the interaction tensor analysis."""
    os.makedirs(output_dir, exist_ok=True)

    print("=== Experiment 4: Interaction Tensor Analysis ===")
    profiles = generate_profiles(5, seed=seed)
    user = profiles[0]
    sigma_0 = compute_prior_covariance(user)

    corpus_types = {
        "redundant": create_redundant_corpus(n=n_docs, seed=seed),
        "synergistic": create_synergistic_corpus(n=n_docs, seed=seed),
        "mixed": create_mixed_corpus(n=n_docs, seed=seed),
    }

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (name, corpus_data) in zip(axes, corpus_types.items()):
        print(f"\n  Corpus: {name}")
        corpus = corpus_data["documents"]
        # Pass full corpus_data dict to make_precision_fn
        precision_fn = make_precision_fn(corpus_data)

        # Pairwise interactions
        I_mat = compute_pairwise_interaction(
            corpus, user, precision_fn, sigma_0, max_docs=n_docs,
        )

        # Statistics
        upper_tri = I_mat[np.triu_indices(n_docs, k=1)]
        print(f"    Pairwise interactions:")
        print(f"      Mean: {np.mean(upper_tri):.6f}")
        print(f"      % positive (synergy): {np.mean(upper_tri > 1e-6)*100:.1f}%")
        print(f"      % negative (redundancy): {np.mean(upper_tri < -1e-6)*100:.1f}%")

        # Heatmap
        sns.heatmap(I_mat, cmap="RdBu_r", center=0, ax=ax,
                    xticklabels=False, yticklabels=False,
                    cbar_kws={"label": "$I_{ij}$"})
        ax.set_title(f"{name.capitalize()} Corpus")

    plt.suptitle("Pairwise Interaction Matrices", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "interaction_analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {fig_path}")


if __name__ == "__main__":
    run_experiment()
