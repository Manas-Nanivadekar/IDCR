"""
Precision predictor: (user_profile, document) → Λ_d(x).

Architecture:
1. User encoder: profile features → h_x ∈ R^p
2. Document encoder: doc features → φ(d) ∈ R^m
3. Cross-attention: attend h_x to φ(d) → e_{x,d} ∈ R^q
4. Precision head: e_{x,d} → L ∈ R^{d×r}, then Λ = L L^T (PSD by construction)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Any

from src.models.user_encoder import UserEncoder, profile_to_features
from src.models.document_encoder import DocumentEncoder, document_to_features


class CrossAttention(nn.Module):
    """Cross-attention between user and document embeddings."""

    def __init__(self, user_dim: int, doc_dim: int, output_dim: int,
                 n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = output_dim // n_heads
        assert output_dim % n_heads == 0

        self.W_q = nn.Linear(user_dim, output_dim)
        self.W_k = nn.Linear(doc_dim, output_dim)
        self.W_v = nn.Linear(doc_dim, output_dim)
        self.W_o = nn.Linear(output_dim, output_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, h_user: torch.Tensor,
                h_doc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_user: (B, user_dim) user embeddings.
            h_doc: (B, doc_dim) document embeddings.

        Returns:
            (B, output_dim) cross-attended embedding.
        """
        Q = self.W_q(h_user)  # (B, output_dim)
        K = self.W_k(h_doc)
        V = self.W_v(h_doc)

        # Simple single-token attention (element-wise as K,V are single vectors)
        attn = (Q * K * self.scale).sum(dim=-1, keepdim=True)
        attn = torch.sigmoid(attn)  # gating instead of softmax for single-token
        out = attn * V
        return self.W_o(out)


class PrecisionHead(nn.Module):
    """Map cross-attended embedding to PSD precision matrix via L L^T."""

    def __init__(self, input_dim: int, d: int = 8, rank: int = 4):
        """
        Args:
            input_dim: dimension of cross-attended embedding.
            d: dimensionality (number of asset classes).
            rank: rank of the precision contribution.
        """
        super().__init__()
        self.d = d
        self.rank = rank
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, d * rank),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e: (B, input_dim) cross-attended embedding.

        Returns:
            (B, d, d) PSD precision matrix Λ = L L^T.
        """
        L = self.net(e).view(-1, self.d, self.rank)  # (B, d, r)
        Lambda = torch.bmm(L, L.transpose(1, 2))  # (B, d, d) — PSD by construction
        return Lambda


class PrecisionPredictor(nn.Module):
    """Full precision predictor: (user, doc) → Λ_d(x).

    Composes user encoder, document encoder, cross-attention, and precision head.
    """

    def __init__(
        self,
        user_input_dim: int = 29,
        doc_input_dim: int = 20,
        user_embed_dim: int = 32,
        doc_embed_dim: int = 32,
        cross_dim: int = 32,
        d: int = 8,
        rank: int = 4,
        n_heads: int = 4,
    ):
        super().__init__()
        self.user_encoder = UserEncoder(user_input_dim, 64, user_embed_dim)
        self.doc_encoder = DocumentEncoder(doc_input_dim, 64, doc_embed_dim)
        self.cross_attention = CrossAttention(
            user_embed_dim, doc_embed_dim, cross_dim, n_heads
        )
        self.precision_head = PrecisionHead(cross_dim, d, rank)
        self.d = d

    def forward(
        self, user_features: torch.Tensor, doc_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            user_features: (B, user_input_dim)
            doc_features: (B, doc_input_dim)

        Returns:
            (B, d, d) PSD precision matrices.
        """
        h_user = self.user_encoder(user_features)
        h_doc = self.doc_encoder(doc_features)
        e = self.cross_attention(h_user, h_doc)
        return self.precision_head(e)

    def predict(
        self, user: dict[str, Any], doc: dict[str, Any],
    ) -> np.ndarray:
        """Convenience method for single (user, doc) pair.

        Args:
            user: user profile dict.
            doc: document dict.

        Returns:
            (d, d) numpy PSD precision matrix.
        """
        self.eval()
        with torch.no_grad():
            u = torch.tensor(profile_to_features(user)).unsqueeze(0)
            d = torch.tensor(document_to_features(doc)).unsqueeze(0)
            Lambda = self.forward(u, d)
        return Lambda.squeeze(0).numpy()


# ── Training ─────────────────────────────────────────────────────────────────

class PrecisionPredictorLoss(nn.Module):
    """Composite loss for precision predictor training.

    L = L_efficiency + λ₁ · L_coverage + λ₂ · L_rank

    L_efficiency: (1/n) Σ log det Σ(x_i, S_i)
    L_coverage:   ((1/n) Σ 1[y_i ∉ C_α] - α)²
    L_rank:       Σ max(0, tr(Λ_worse) - tr(Λ_better) + γ)
    """

    def __init__(self, alpha: float = 0.10, lambda_1: float = 10.0,
                 lambda_2: float = 0.1, gamma: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.gamma = gamma

    def forward(
        self,
        Lambda_pred: torch.Tensor,
        Lambda_true: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            Lambda_pred: (B, d, d) predicted precision matrices.
            Lambda_true: (B, d, d) ground truth precision matrices.

        Returns:
            Dict with 'total', 'efficiency', 'rank' losses.
        """
        # Efficiency: MSE between predicted and true precision
        efficiency_loss = torch.mean((Lambda_pred - Lambda_true) ** 2)

        # Rank loss: predicted should preserve ranking of trace
        tr_pred = torch.diagonal(Lambda_pred, dim1=-2, dim2=-1).sum(-1)  # (B,)
        tr_true = torch.diagonal(Lambda_true, dim1=-2, dim2=-1).sum(-1)

        # Pairwise ranking
        B = tr_pred.shape[0]
        if B > 1:
            rank_loss = torch.tensor(0.0)
            n_pairs = 0
            for i in range(min(B, 32)):
                for j in range(i + 1, min(B, 32)):
                    if tr_true[i] > tr_true[j]:
                        rank_loss = rank_loss + torch.relu(
                            tr_pred[j] - tr_pred[i] + self.gamma
                        )
                    elif tr_true[j] > tr_true[i]:
                        rank_loss = rank_loss + torch.relu(
                            tr_pred[i] - tr_pred[j] + self.gamma
                        )
                    n_pairs += 1
            rank_loss = rank_loss / max(n_pairs, 1)
        else:
            rank_loss = torch.tensor(0.0)

        total = efficiency_loss + self.lambda_2 * rank_loss

        return {
            "total": total,
            "efficiency": efficiency_loss,
            "rank": rank_loss,
        }
