"""
Experiment 3: RL vs Greedy characterization.

Train the RL policy and compare against greedy retrieval across corpus types
(redundant, synergistic, mixed) to validate the interaction tensor characterization.

Expected: RL > Greedy on synergistic corpora, RL ≈ Greedy on redundant corpora.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.user_profiles import generate_profiles, ASSET_CLASSES
from src.data.documents import generate_corpus
from src.data.precision_ground_truth import (
    compute_ground_truth_precision,
    compute_prior_covariance,
)
from src.data.synergy_control import (
    create_synergistic_corpus,
    create_redundant_corpus,
    create_mixed_corpus,
    make_precision_fn,
)
from src.retrieval.greedy import GreedyRetrieval
from src.retrieval.random_retrieval import RandomRetrieval
from src.retrieval.rl_policy import (
    RLRetrievalPolicy,
    train_rl_policy,
    evaluate_rl_policy,
    compute_document_embeddings,
)


def run_experiment(
    n_users: int = 20,
    budget: int = 5,
    n_episodes: int = 2000,
    n_docs: int = 40,
    seed: int = 42,
    output_dir: str = "outputs/exp3_rl_vs_greedy",
):
    """Run RL vs Greedy comparison across corpus types."""
    os.makedirs(output_dir, exist_ok=True)

    print("=== Experiment 3: RL vs Greedy ===")
    profiles = generate_profiles(n_users, seed=seed)

    corpus_types = {
        "redundant": create_redundant_corpus(n=n_docs, seed=seed),
        "synergistic": create_synergistic_corpus(n=n_docs, seed=seed),
        "mixed": create_mixed_corpus(n=n_docs, seed=seed),
    }

    results_all = {}

    for corpus_name, corpus_data in corpus_types.items():
        print(f"\n  Corpus: {corpus_name}")
        corpus = corpus_data["documents"]
        # Pass the FULL corpus_data dict, not just the precision_matrices
        precision_fn = make_precision_fn(corpus_data)

        # Greedy
        greedy_gains = []
        for user in profiles:
            sigma_0 = compute_prior_covariance(user)
            greedy = GreedyRetrieval(sigma_0)
            result = greedy.retrieve(user, corpus, precision_fn, budget)
            greedy_gains.append(result["total_gain"])

        # Random
        random_gains = []
        for user in profiles:
            sigma_0 = compute_prior_covariance(user)
            rand = RandomRetrieval(sigma_0)
            result = rand.retrieve(user, corpus, precision_fn, budget, seed=seed)
            random_gains.append(result["total_gain"])

        # RL
        user_dim = 12  # from _encode_user
        policy = RLRetrievalPolicy(
            user_dim=user_dim, d=8, doc_dim=16, hidden_dim=64,
            n_ensemble=3, ids_lambda=0.1,
        )
        train_result = train_rl_policy(
            policy, profiles, corpus, precision_fn,
            sigma_0_fn=compute_prior_covariance,
            budget=budget, n_episodes=n_episodes, lr=1e-3, seed=seed,
        )
        rl_result = evaluate_rl_policy(
            policy, profiles, corpus, precision_fn,
            sigma_0_fn=compute_prior_covariance, budget=budget,
        )

        results_all[corpus_name] = {
            "greedy": np.array(greedy_gains),
            "random": np.array(random_gains),
            "rl": np.array(rl_result["total_gains"]),
            "rl_training": train_result,
        }

        print(f"    Greedy mean: {np.mean(greedy_gains):.4f}")
        print(f"    Random mean: {np.mean(random_gains):.4f}")
        print(f"    RL mean:     {rl_result['mean_gain']:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (name, data) in zip(axes, results_all.items()):
        positions = [1, 2, 3]
        bp = ax.boxplot(
            [data["random"], data["greedy"], data["rl"]],
            positions=positions,
            tick_labels=["Random", "Greedy", "RL"],
            patch_artist=True,
        )
        colors = ["#FFB3BA", "#87CEEB", "#90EE90"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_ylabel("Total Gain $g(S)$")
        ax.set_title(f"{name.capitalize()} Corpus")

    plt.suptitle("RL vs Greedy vs Random by Corpus Type", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "rl_vs_greedy_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {fig_path}")

    # Save results
    np.savez(
        os.path.join(output_dir, "results.npz"),
        **{f"{cn}_{mn}": data[mn]
           for cn, data in results_all.items()
           for mn in ["greedy", "random", "rl"]},
    )


if __name__ == "__main__":
    run_experiment()
