"""
Document encoder: document features → φ(d) ∈ R^m.

Simple encoder for synthetic documents. For real NLP documents, this would
wrap a pretrained transformer (e.g., all-MiniLM-L6-v2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Any

from src.data.user_profiles import ASSET_CLASSES
from src.data.documents import DOC_TYPES


def document_to_features(doc: dict[str, Any]) -> np.ndarray:
    """Convert a document dict to a flat feature vector.

    Args:
        doc: document dict.

    Returns:
        (m_raw,) float32 feature vector.
    """
    features = [doc["informativeness"] / 5.0]

    # Doc type one-hot
    for dt in DOC_TYPES:
        features.append(1.0 if doc["doc_type"] == dt else 0.0)

    # Sector coverage
    for ac in ASSET_CLASSES:
        features.append(1.0 if ac in doc["relevant_sectors"] else 0.0)

    # Key metrics (normalised)
    km = doc.get("key_metrics", {})
    features.append(km.get("revenue_growth_pct", 0.0) / 20.0)
    features.append(km.get("pe_ratio", 20.0) / 40.0)
    features.append(km.get("gdp_growth_pct", 2.0) / 5.0)
    features.append(km.get("volatility_index", 20.0) / 40.0)
    features.append(km.get("sharpe_ratio", 1.0) / 3.0)
    features.append(km.get("confidence_score", 0.5))

    return np.array(features, dtype=np.float32)


class DocumentEncoder(nn.Module):
    """MLP encoder for document features.

    doc → features → MLP → φ(d) ∈ R^m
    """

    def __init__(self, input_dim: int = 20, hidden_dim: int = 64,
                 output_dim: int = 32):
        """
        Args:
            input_dim: dimension of document_to_features output.
            hidden_dim: MLP hidden dimension.
            output_dim: embedding dimension m.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) document feature vectors.

        Returns:
            (B, output_dim) document embeddings.
        """
        return self.net(x)
