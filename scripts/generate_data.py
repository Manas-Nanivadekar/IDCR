"""
Generate and cache all synthetic data needed by IDCR experiments.

Run this once before running experiments. All data is saved to data/generated/.
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.user_profiles import generate_profiles, generate_splits, ASSET_CLASSES
from src.data.documents import generate_corpus
from src.data.precision_ground_truth import (
    compute_ground_truth_precision,
    compute_prior_covariance,
    compute_optimal_allocation,
    get_market_covariance,
    get_expected_returns,
)
from src.data.synergy_control import (
    create_synergistic_corpus,
    create_redundant_corpus,
    create_mixed_corpus,
)
from src.data.llm_augmentation import create_synthetic_texts


DATA_DIR = Path("data/generated")
SEED = 42


def save_json(obj, path: Path):
    """Save a JSON-serialisable object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  ✓ Saved {path}")


def save_npz(path: Path, **arrays):
    """Save numpy arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    print(f"  ✓ Saved {path}")


def generate_user_data():
    """Generate user profiles and splits."""
    print("\n━━━ User Profiles ━━━")

    profiles_all = generate_profiles(1400, seed=SEED)
    splits = generate_splits(n_train=1000, n_cal=200, n_test=200, seed=SEED)

    save_json(profiles_all, DATA_DIR / "profiles_all.json")
    save_json(splits["train"], DATA_DIR / "splits" / "train.json")
    save_json(splits["calibration"], DATA_DIR / "splits" / "calibration.json")
    save_json(splits["test"], DATA_DIR / "splits" / "test.json")

    print(f"  Train: {len(splits['train'])}, Cal: {len(splits['calibration'])}, "
          f"Test: {len(splits['test'])}")


def generate_document_data():
    """Generate document corpus and synthetic texts."""
    print("\n━━━ Document Corpus ━━━")

    corpus = generate_corpus(200, seed=SEED)
    save_json(corpus, DATA_DIR / "corpus.json")
    print(f"  Documents: {len(corpus)}")

    # Synthetic texts (fallback, no LLM needed)
    texts = create_synthetic_texts(corpus)
    save_json(texts, DATA_DIR / "document_texts.json")
    print(f"  Texts generated: {len(texts)}")


def generate_ground_truth():
    """Pre-compute ground truth allocations, prior covariances, and market data."""
    print("\n━━━ Ground Truth ━━━")

    splits = generate_splits(n_train=1000, n_cal=200, n_test=200, seed=SEED)
    all_users = splits["train"] + splits["calibration"] + splits["test"]

    # Market data
    market_cov = get_market_covariance()
    expected_ret = get_expected_returns()
    save_npz(DATA_DIR / "market_data.npz",
             covariance=market_cov,
             expected_returns=expected_ret,
             asset_classes=np.array(ASSET_CLASSES))

    # Optimal allocations
    y_stars = {}
    sigma_0s = {}
    for split_name, users in [("train", splits["train"]),
                               ("calibration", splits["calibration"]),
                               ("test", splits["test"])]:
        allocations = []
        prior_covs = []
        for user in users:
            y_star = compute_optimal_allocation(user)
            sigma_0 = compute_prior_covariance(user)
            allocations.append(y_star)
            prior_covs.append(sigma_0)

        alloc_arr = np.array(allocations)
        cov_arr = np.array(prior_covs)
        save_npz(DATA_DIR / "ground_truth" / f"{split_name}.npz",
                 y_star=alloc_arr,
                 sigma_0=cov_arr)
        print(f"  {split_name}: {len(allocations)} allocations, "
              f"shape={alloc_arr.shape}")


def generate_precision_matrices():
    """Pre-compute precision matrices Λ_d(x) for all (user, doc) pairs in test set."""
    print("\n━━━ Precision Matrices (test set) ━━━")

    splits = generate_splits(n_train=1000, n_cal=200, n_test=200, seed=SEED)
    corpus = generate_corpus(200, seed=SEED)
    test_users = splits["test"]

    # Only cache for test users (200 users × 200 docs = 40K matrices, ~10MB)
    n_users = len(test_users)
    n_docs = len(corpus)
    d = 8

    Lambda_cache = np.zeros((n_users, n_docs, d, d))
    for i, user in enumerate(test_users):
        for j, doc in enumerate(corpus):
            Lambda_cache[i, j] = compute_ground_truth_precision(doc, user)

    save_npz(DATA_DIR / "precision_cache.npz", Lambda=Lambda_cache)
    print(f"  Cached: {n_users} × {n_docs} = {n_users * n_docs} matrices")


def generate_synergy_corpora():
    """Generate controlled corpus variants for interaction tensor experiments."""
    print("\n━━━ Synergy Corpora ━━━")

    for name, create_fn in [("redundant", create_redundant_corpus),
                             ("synergistic", create_synergistic_corpus),
                             ("mixed", create_mixed_corpus)]:
        data = create_fn(n=40, seed=SEED)
        corpus = data["documents"]
        prec = data["precision_matrices"]

        save_json(corpus, DATA_DIR / "synergy" / f"{name}_corpus.json")

        # Save precision matrices as npz
        n = len(prec)
        d = 8
        prec_arr = np.zeros((n, d, d))
        for doc_id, mat in prec.items():
            prec_arr[int(doc_id)] = mat
        save_npz(DATA_DIR / "synergy" / f"{name}_precision.npz", Lambda=prec_arr)
        print(f"  {name}: {len(corpus)} docs")


def main():
    print("╔══════════════════════════════════════╗")
    print("║   IDCR Data Generation Pipeline      ║")
    print("╚══════════════════════════════════════╝")

    generate_user_data()
    generate_document_data()
    generate_ground_truth()
    generate_precision_matrices()
    generate_synergy_corpora()

    print("\n" + "═" * 40)
    print("✅ All data generated successfully!")
    print(f"   Location: {DATA_DIR.resolve()}")

    # Summary
    total_files = sum(1 for _ in DATA_DIR.rglob("*") if _.is_file())
    total_size = sum(f.stat().st_size for f in DATA_DIR.rglob("*") if f.is_file())
    print(f"   Files: {total_files}")
    print(f"   Size: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
