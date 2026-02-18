"""
User encoder: profile features → h_x ∈ R^p.

Simple MLP encoder that takes structured user profile features and produces
a fixed-size embedding for the precision predictor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Any

from src.data.user_profiles import ASSET_CLASSES


def profile_to_features(profile: dict[str, Any]) -> np.ndarray:
    """Convert a user profile dict to a flat feature vector.

    Args:
        profile: user profile dict.

    Returns:
        (p,) float32 feature vector.
    """
    features = [
        profile["risk_tolerance"],
        profile["investment_horizon_months"] / 120.0,
        profile["demographic"]["age"] / 80.0,
        len(profile["sector_preferences"]) / len(ASSET_CLASSES),
        profile["constraints"]["max_single_sector"],
    ]

    # Current holdings
    for ac in ASSET_CLASSES:
        features.append(profile["current_holdings"].get(ac, 0.0))

    # Sector preference one-hot (common sectors)
    all_sectors = ["technology", "healthcare", "bonds", "utilities", "dividends",
                   "REITs", "clean_energy", "growth", "crypto", "emerging_markets",
                   "real_estate", "commodities", "alternatives", "cash",
                   "US_equity", "intl_equity"]
    for s in all_sectors:
        features.append(1.0 if s in profile["sector_preferences"] else 0.0)

    return np.array(features, dtype=np.float32)


class UserEncoder(nn.Module):
    """MLP encoder for user profile features.

    profile → features → MLP → h_x ∈ R^p
    """

    def __init__(self, input_dim: int = 29, hidden_dim: int = 64, output_dim: int = 32):
        """
        Args:
            input_dim: dimension of profile_to_features output.
            hidden_dim: MLP hidden dimension.
            output_dim: embedding dimension p.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) user feature vectors.

        Returns:
            (B, output_dim) user embeddings.
        """
        return self.net(x)
