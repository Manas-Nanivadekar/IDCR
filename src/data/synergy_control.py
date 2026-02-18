"""
Controlled synergy / redundancy structures for document corpora.

Creates three corpus variants:
- Redundant: overlapping eigenspaces, T_ijk < 0
- Synergistic: complementary dimensions with cross-dim information, T_ijk > 0
- Mixed: 60% independent, 20% redundant, 20% synergistic
"""

from __future__ import annotations

import numpy as np
from typing import Any

from src.data.user_profiles import ASSET_CLASSES
from src.utils.linear_algebra import nearest_psd

D = len(ASSET_CLASSES)  # 8


# ── Synergistic pair construction ────────────────────────────────────────────

def create_synergistic_pair(
    dim_i: int,
    dim_j: int,
    d: int = D,
    synergy_strength: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create two documents that individually inform separate dims but together
    inform their correlation.

    Args:
        dim_i: first dimension index.
        dim_j: second dimension index.
        d: total dimensionality.
        synergy_strength: off-diagonal synergy magnitude.

    Returns:
        (Lambda_1, Lambda_2, Lambda_synergy) — precision matrices.
        Lambda_synergy is added only when BOTH docs are in the retrieved set.
    """
    Lambda_1 = np.zeros((d, d))
    Lambda_1[dim_i, dim_i] = 1.0

    Lambda_2 = np.zeros((d, d))
    Lambda_2[dim_j, dim_j] = 1.0

    Lambda_synergy = np.zeros((d, d))
    Lambda_synergy[dim_i, dim_j] = synergy_strength
    Lambda_synergy[dim_j, dim_i] = synergy_strength
    # Make the synergy matrix PSD by adding enough diagonal
    Lambda_synergy[dim_i, dim_i] += synergy_strength
    Lambda_synergy[dim_j, dim_j] += synergy_strength
    Lambda_synergy = nearest_psd(Lambda_synergy)

    return Lambda_1, Lambda_2, Lambda_synergy


# ── Redundant pair construction ──────────────────────────────────────────────

def create_redundant_pair(
    dims: list[int],
    d: int = D,
    overlap: float = 0.9,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create two documents with overlapping precision matrices (>80% shared eigenspace).

    Args:
        dims: list of dimension indices both documents inform.
        d: total dimensionality.
        overlap: fraction of shared eigenspace.
        rng: random generator.

    Returns:
        (Lambda_1, Lambda_2) — precision matrices with high overlap.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Base projection onto dims
    P_base = np.zeros((d, d))
    for i in dims:
        P_base[i, i] = 1.0

    # Doc 1: base + small perturbation
    noise_1 = rng.standard_normal((d, d)) * (1.0 - overlap) * 0.1
    noise_1 = (noise_1 + noise_1.T) / 2
    Lambda_1 = nearest_psd(P_base + noise_1)

    # Doc 2: base + different small perturbation
    noise_2 = rng.standard_normal((d, d)) * (1.0 - overlap) * 0.1
    noise_2 = (noise_2 + noise_2.T) / 2
    Lambda_2 = nearest_psd(P_base + noise_2)

    return Lambda_1, Lambda_2


# ── Independent document ─────────────────────────────────────────────────────

def create_independent_doc(
    dim: int, d: int = D, informativeness: float = 1.0,
) -> np.ndarray:
    """Create a document that informs a single dimension independently.

    Args:
        dim: dimension index.
        d: total dimensionality.
        informativeness: diagonal value.

    Returns:
        (d, d) precision matrix.
    """
    Lambda = np.zeros((d, d))
    Lambda[dim, dim] = informativeness
    return Lambda


# ── Corpus builders ──────────────────────────────────────────────────────────

def create_redundant_corpus(
    n: int = 50, seed: int = 42,
) -> dict[str, Any]:
    """Create a corpus where documents have overlapping precision matrices.

    T_ijk < 0 by construction (redundant information).

    Returns:
        Dict with 'documents' (list of metadata dicts), 'precision_matrices'
        (dict doc_id → (d,d) ndarray), and 'synergy_matrices' (empty).
    """
    rng = np.random.default_rng(seed)
    documents = []
    precisions: dict[int, np.ndarray] = {}

    sector_groups = [
        [0, 1],    # equity group
        [2, 3],    # EM + bonds
        [4, 5],    # real estate + commodities
        [6, 7],    # cash + alternatives
    ]

    for doc_id in range(n):
        group = sector_groups[doc_id % len(sector_groups)]
        λ1, λ2 = create_redundant_pair(group, rng=rng)
        # Alternate between the two overlapping matrices
        precisions[doc_id] = λ1 if doc_id % 2 == 0 else λ2
        documents.append({
            "doc_id": doc_id,
            "doc_type": "sector_report",
            "relevant_sectors": [ASSET_CLASSES[i] for i in group],
            "informativeness": float(rng.gamma(2.0, 1.0)),
            "corpus_type": "redundant",
        })

    return {
        "documents": documents,
        "precision_matrices": precisions,
        "synergy_matrices": {},
    }


def create_synergistic_corpus(
    n: int = 50, seed: int = 42, synergy_strength: float = 0.5,
) -> dict[str, Any]:
    """Create a corpus with synergistic document pairs.

    T_ijk > 0 by construction (complementary information).

    Returns:
        Dict with 'documents', 'precision_matrices', and 'synergy_matrices'
        (dict (doc_i, doc_j) → synergy precision matrix).
    """
    rng = np.random.default_rng(seed)
    documents = []
    precisions: dict[int, np.ndarray] = {}
    synergies: dict[tuple[int, int], np.ndarray] = {}

    for doc_id in range(0, n, 2):
        # Pick two different dimensions for synergistic pair
        dim_i = doc_id % D
        dim_j = (doc_id + 3) % D
        if dim_i == dim_j:
            dim_j = (dim_j + 1) % D

        λ1, λ2, λ_syn = create_synergistic_pair(
            dim_i, dim_j, synergy_strength=synergy_strength
        )
        info_scale = float(rng.gamma(2.0, 1.0))
        precisions[doc_id] = λ1 * info_scale
        precisions[doc_id + 1] = λ2 * info_scale if doc_id + 1 < n else λ2
        synergies[(doc_id, doc_id + 1)] = λ_syn * info_scale

        for offset in range(2):
            did = doc_id + offset
            if did < n:
                dim_used = dim_i if offset == 0 else dim_j
                documents.append({
                    "doc_id": did,
                    "doc_type": "macro_analysis",
                    "relevant_sectors": [ASSET_CLASSES[dim_used]],
                    "informativeness": info_scale,
                    "corpus_type": "synergistic",
                    "synergy_partner": doc_id + (1 - offset),
                })

    return {
        "documents": documents,
        "precision_matrices": precisions,
        "synergy_matrices": synergies,
    }


def create_mixed_corpus(
    n: int = 50, seed: int = 42,
) -> dict[str, Any]:
    """Create a mixed corpus: 60% independent, 20% redundant, 20% synergistic.

    Returns:
        Dict with 'documents', 'precision_matrices', and 'synergy_matrices'.
    """
    rng = np.random.default_rng(seed)
    n_indep = int(n * 0.6)
    n_redun = int(n * 0.2)
    n_syner = n - n_indep - n_redun

    documents = []
    precisions: dict[int, np.ndarray] = {}
    synergies: dict[tuple[int, int], np.ndarray] = {}
    doc_id = 0

    # Independent documents
    for _ in range(n_indep):
        dim = rng.integers(0, D)
        info = float(rng.gamma(2.0, 1.0))
        precisions[doc_id] = create_independent_doc(int(dim), informativeness=info)
        documents.append({
            "doc_id": doc_id,
            "doc_type": "market_outlook",
            "relevant_sectors": [ASSET_CLASSES[int(dim)]],
            "informativeness": info,
            "corpus_type": "independent",
        })
        doc_id += 1

    # Redundant documents
    sector_groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    for i in range(0, n_redun, 2):
        group = sector_groups[i % len(sector_groups)]
        λ1, λ2 = create_redundant_pair(group, rng=rng)
        info = float(rng.gamma(2.0, 1.0))
        for offset, lam in enumerate([λ1, λ2]):
            if doc_id < n_indep + n_redun:
                precisions[doc_id] = lam * info
                documents.append({
                    "doc_id": doc_id,
                    "doc_type": "sector_report",
                    "relevant_sectors": [ASSET_CLASSES[j] for j in group],
                    "informativeness": info,
                    "corpus_type": "redundant",
                })
                doc_id += 1

    # Synergistic documents
    for i in range(0, n_syner, 2):
        dim_i = rng.integers(0, D)
        dim_j = (int(dim_i) + rng.integers(1, D)) % D
        λ1, λ2, λ_syn = create_synergistic_pair(int(dim_i), int(dim_j))
        info = float(rng.gamma(2.0, 1.0))
        id_a, id_b = doc_id, doc_id + 1
        for offset, lam in enumerate([λ1, λ2]):
            if doc_id < n_indep + n_redun + n_syner:
                precisions[doc_id] = lam * info
                documents.append({
                    "doc_id": doc_id,
                    "doc_type": "macro_analysis",
                    "relevant_sectors": [ASSET_CLASSES[int(dim_i if offset == 0 else dim_j)]],
                    "informativeness": info,
                    "corpus_type": "synergistic",
                    "synergy_partner": id_b if offset == 0 else id_a,
                })
                doc_id += 1
        synergies[(id_a, id_b)] = λ_syn * info

    return {
        "documents": documents,
        "precision_matrices": precisions,
        "synergy_matrices": synergies,
    }


# ── Precision function with synergy ──────────────────────────────────────────

def make_precision_fn(
    corpus_data: dict[str, Any],
) -> callable:
    """Create a precision function that accounts for synergy.

    The returned function computes the precision contribution of a document,
    including synergy bonuses if the document's partner is already in the
    retrieved set.

    Args:
        corpus_data: dict from create_*_corpus functions.

    Returns:
        Callable (doc, user, retrieved_set) → Λ_d(x).
    """
    precisions = corpus_data["precision_matrices"]
    synergies = corpus_data["synergy_matrices"]

    def precision_fn(doc: dict, user: dict,
                     retrieved_ids: set[int] | None = None) -> np.ndarray:
        base = precisions[doc["doc_id"]].copy()
        if retrieved_ids is None:
            return base

        doc_id = doc["doc_id"]
        # Check if any synergy partner is already retrieved
        for (a, b), syn_mat in synergies.items():
            if doc_id == a and b in retrieved_ids:
                base = base + syn_mat
            elif doc_id == b and a in retrieved_ids:
                base = base + syn_mat

        return base

    return precision_fn
