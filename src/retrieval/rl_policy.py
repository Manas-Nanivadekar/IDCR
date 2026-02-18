"""
RL policy for document retrieval: Actor-Critic with IDS objective.

MDP Environment:
- State: (user_embedding, retrieved_set_precision, remaining_budget)
- Action: index into remaining corpus
- Reward: marginal log-det gain (conditional mutual information)
- Terminal: budget exhausted → -log det Σ(x, S_k)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Callable
from dataclasses import dataclass

from src.utils.linear_algebra import (
    marginal_gain,
    posterior_covariance,
    log_det,
)


# ── Environment ──────────────────────────────────────────────────────────────

@dataclass
class RetrievalState:
    """State in the retrieval MDP."""
    user_features: np.ndarray     # (p,) user embedding
    sigma_current: np.ndarray     # (d, d) current covariance
    retrieved_ids: set[int]       # set of retrieved doc IDs
    remaining_budget: int         # how many more docs can retrieve
    step: int                     # current step


class RetrievalEnv:
    """Gym-like environment for sequential document retrieval."""

    def __init__(
        self,
        user_profile: dict[str, Any],
        corpus: list[dict[str, Any]],
        precision_fn: Callable,
        sigma_0: np.ndarray,
        budget: int,
        user_features: np.ndarray | None = None,
    ):
        self.user_profile = user_profile
        self.corpus = corpus
        self.precision_fn = precision_fn
        self.sigma_0 = sigma_0
        self.budget = budget
        self.d = sigma_0.shape[0]
        self.user_features = user_features if user_features is not None else self._encode_user()
        self.state: RetrievalState | None = None

    def _encode_user(self) -> np.ndarray:
        """Simple feature extraction from user profile."""
        p = self.user_profile
        features = [
            p["risk_tolerance"],
            p["investment_horizon_months"] / 120.0,
            p["demographic"]["age"] / 80.0,
            len(p["sector_preferences"]) / 8.0,
        ]
        # Holdings as features
        from src.data.user_profiles import ASSET_CLASSES
        for ac in ASSET_CLASSES:
            features.append(p["current_holdings"].get(ac, 0.0))
        return np.array(features, dtype=np.float32)

    def reset(self) -> RetrievalState:
        self.state = RetrievalState(
            user_features=self.user_features,
            sigma_current=self.sigma_0.copy(),
            retrieved_ids=set(),
            remaining_budget=self.budget,
            step=0,
        )
        return self.state

    def step(self, action: int) -> tuple[RetrievalState, float, bool, dict]:
        """Take action (select document by index).

        Returns:
            (next_state, reward, done, info)
        """
        assert self.state is not None, "Must call reset() first."
        assert action not in self.state.retrieved_ids, "Document already retrieved."
        assert 0 <= action < len(self.corpus), f"Invalid action {action}"

        doc = self.corpus[action]
        Lambda_d = self.precision_fn(doc, self.user_profile)

        # Reward = marginal log-det gain
        reward = marginal_gain(self.state.sigma_current, Lambda_d)

        # Update state
        new_sigma = posterior_covariance(self.state.sigma_current, Lambda_d)
        new_retrieved = self.state.retrieved_ids | {action}

        self.state = RetrievalState(
            user_features=self.user_features,
            sigma_current=new_sigma,
            retrieved_ids=new_retrieved,
            remaining_budget=self.state.remaining_budget - 1,
            step=self.state.step + 1,
        )

        done = self.state.remaining_budget <= 0
        info = {
            "marginal_gain": reward,
            "log_det_sigma": log_det(new_sigma),
        }

        return self.state, float(reward), done, info

    def get_available_actions(self) -> list[int]:
        """Get indices of documents not yet retrieved."""
        if self.state is None:
            return list(range(len(self.corpus)))
        return [i for i in range(len(self.corpus))
                if i not in self.state.retrieved_ids]


# ── State encoder ────────────────────────────────────────────────────────────

class StateEncoder(nn.Module):
    """Encode MDP state into a fixed-size vector."""

    def __init__(self, user_dim: int, d: int, hidden_dim: int = 128):
        super().__init__()
        # user_features + flattened precision summary + budget
        input_dim = user_dim + d * (d + 1) // 2 + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.hidden_dim = hidden_dim

    def forward(self, user_features: torch.Tensor,
                sigma_triu: torch.Tensor,
                budget_frac: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_features: (B, user_dim)
            sigma_triu: (B, d*(d+1)/2) upper-triangular of current sigma
            budget_frac: (B, 1) remaining budget / total budget

        Returns:
            (B, hidden_dim) state embedding
        """
        x = torch.cat([user_features, sigma_triu, budget_frac], dim=-1)
        return self.net(x)


# ── Actor-Critic ─────────────────────────────────────────────────────────────

class DocumentScorer(nn.Module):
    """Score documents given state embedding (actor head)."""

    def __init__(self, hidden_dim: int, doc_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, doc_dim)

    def forward(self, h: torch.Tensor,
                doc_embeddings: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            h: (B, hidden_dim) state embeddings
            doc_embeddings: (n_docs, doc_dim) document embeddings
            mask: (B, n_docs) bool mask, True = available

        Returns:
            (B, n_docs) log-probabilities over documents
        """
        q = self.query(h)  # (B, doc_dim)
        scores = q @ doc_embeddings.T  # (B, n_docs)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        return F.log_softmax(scores, dim=-1)


class Critic(nn.Module):
    """Value function estimator."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, hidden_dim) state embeddings

        Returns:
            (B, 1) value estimates
        """
        return self.net(h)


class RLRetrievalPolicy(nn.Module):
    """Actor-Critic policy for document retrieval with IDS exploration."""

    def __init__(
        self,
        user_dim: int,
        d: int = 8,
        doc_dim: int = 16,
        hidden_dim: int = 128,
        n_ensemble: int = 5,
        ids_lambda: float = 0.1,
    ):
        super().__init__()
        self.d = d
        self.hidden_dim = hidden_dim
        self.ids_lambda = ids_lambda

        self.state_encoder = StateEncoder(user_dim, d, hidden_dim)
        self.actor = DocumentScorer(hidden_dim, doc_dim)
        self.critic = Critic(hidden_dim)

        # Ensemble heads for IDS exploration bonus
        self.ensemble_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_ensemble)
        ])

    def _state_to_tensors(self, state: RetrievalState,
                          total_budget: int) -> tuple[torch.Tensor, ...]:
        """Convert numpy state to tensors."""
        user = torch.tensor(state.user_features, dtype=torch.float32).unsqueeze(0)

        # Upper-triangular of sigma
        sigma = state.sigma_current
        triu_idx = np.triu_indices(self.d)
        sigma_triu = torch.tensor(
            sigma[triu_idx], dtype=torch.float32
        ).unsqueeze(0)

        budget_frac = torch.tensor(
            [[state.remaining_budget / max(total_budget, 1)]],
            dtype=torch.float32,
        )
        return user, sigma_triu, budget_frac

    def get_action(
        self,
        state: RetrievalState,
        doc_embeddings: torch.Tensor,
        total_budget: int,
        available_actions: list[int],
        explore: bool = True,
        return_tensors: bool = False,
    ) -> tuple[int, dict]:
        """Select an action using the policy.

        Args:
            state: current MDP state.
            doc_embeddings: (n_docs, doc_dim) document embeddings.
            total_budget: total retrieval budget.
            available_actions: list of available doc indices.
            explore: if True, sample from policy; else take argmax.
            return_tensors: if True, return tensors with grad_fn for training.

        Returns:
            (action_idx, info_dict)
        """
        user, sigma_triu, budget_frac = self._state_to_tensors(state, total_budget)
        h = self.state_encoder(user, sigma_triu, budget_frac)

        # Create mask
        mask = torch.zeros(1, len(doc_embeddings), dtype=torch.bool)
        for idx in available_actions:
            mask[0, idx] = True

        log_probs = self.actor(h, doc_embeddings, mask)
        value = self.critic(h)

        if explore:
            # IDS exploration bonus: ensemble disagreement
            ensemble_preds = [head(h) for head in self.ensemble_heads]
            disagreement = torch.stack(ensemble_preds).var(dim=0)

            # Adjusted scores (exploitation + exploration)
            adjusted_log_probs = log_probs + self.ids_lambda * torch.log(
                disagreement + 1e-6
            )
            # Re-mask and re-normalise
            adjusted_log_probs = adjusted_log_probs.masked_fill(~mask, float("-inf"))
            probs = F.softmax(adjusted_log_probs, dim=-1)
            action = torch.multinomial(probs, 1).item()
        else:
            action = log_probs.argmax(dim=-1).item()

        if return_tensors:
            return int(action), {
                "log_prob": log_probs[0, action],
                "value": value.squeeze(),
            }
        return int(action), {
            "log_prob": log_probs[0, action].item(),
            "value": value.item(),
        }


# ── Training utilities ───────────────────────────────────────────────────────

def compute_document_embeddings(
    corpus: list[dict[str, Any]],
    d: int = 8,
    doc_dim: int = 16,
    seed: int = 42,
) -> torch.Tensor:
    """Compute simple document embeddings from metadata.

    Uses a deterministic mapping from document features to a fixed-size vector.
    For the full system, these would come from a learned document encoder.

    Args:
        corpus: list of document dicts.
        d: number of asset classes.
        doc_dim: embedding dimension.
        seed: random seed.

    Returns:
        (n_docs, doc_dim) tensor of document embeddings.
    """
    rng = np.random.default_rng(seed)
    from src.data.user_profiles import ASSET_CLASSES

    embeddings = []
    for doc in corpus:
        features = []
        features.append(doc["informativeness"] / 5.0)  # normalised

        # One-hot for doc_type
        from src.data.documents import DOC_TYPES
        for dt in DOC_TYPES:
            features.append(1.0 if doc["doc_type"] == dt else 0.0)

        # Sector coverage
        for ac in ASSET_CLASSES:
            features.append(1.0 if ac in doc["relevant_sectors"] else 0.0)

        # Pad or truncate to doc_dim
        feat = np.array(features[:doc_dim], dtype=np.float32)
        if len(feat) < doc_dim:
            feat = np.pad(feat, (0, doc_dim - len(feat)))
        embeddings.append(feat)

    return torch.tensor(np.stack(embeddings), dtype=torch.float32)


def train_rl_policy(
    policy: RLRetrievalPolicy,
    users: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    precision_fn: Callable,
    sigma_0_fn: Callable,
    budget: int = 5,
    n_episodes: int = 200,
    lr: float = 1e-3,
    gamma: float = 0.99,
    entropy_coeff: float = 0.01,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Train the RL policy using REINFORCE with baseline.

    Args:
        policy: the RL policy module.
        users: list of user profiles.
        corpus: document corpus.
        precision_fn: callable(doc, user) → Λ_d(x).
        sigma_0_fn: callable(user) → Σ_0.
        budget: retrieval budget per episode.
        n_episodes: number of training episodes.
        lr: learning rate.
        gamma: discount factor.
        entropy_coeff: coefficient for entropy regularization.
        seed: random seed.

    Returns:
        Dict with 'episode_returns', 'actor_losses', 'critic_losses'.
    """
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    doc_embeddings = compute_document_embeddings(corpus)

    episode_returns = []
    actor_losses = []
    critic_losses = []

    for ep in range(n_episodes):
        # Sample a random user
        user_idx = int(rng.integers(0, len(users)))
        user = users[user_idx]
        sigma_0 = sigma_0_fn(user)

        env = RetrievalEnv(user, corpus, precision_fn, sigma_0, budget)
        state = env.reset()

        log_probs_list: list[torch.Tensor] = []
        values_list: list[torch.Tensor] = []
        rewards = []

        for _t in range(budget):
            available = env.get_available_actions()
            action, info = policy.get_action(
                state, doc_embeddings, budget, available,
                explore=True, return_tensors=True,
            )
            state, reward, done, step_info = env.step(action)

            log_probs_list.append(info["log_prob"])
            values_list.append(info["value"])
            rewards.append(reward)

            if done:
                break

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        # Stack tensors preserving grad_fn
        log_probs_t = torch.stack(log_probs_list)
        values_t = torch.stack(values_list)

        # Advantage (detach values so policy gradient doesn't flow through critic)
        advantage = (returns_t - values_t.detach())

        # Policy gradient loss (REINFORCE with baseline)
        actor_loss = -(log_probs_t * advantage).mean()

        # Entropy bonus for exploration
        entropy = -(log_probs_t.exp() * log_probs_t).sum()

        # Value loss
        critic_loss = F.mse_loss(values_t, returns_t)

        # Total loss
        loss = actor_loss + 0.5 * critic_loss - entropy_coeff * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        ep_return = float(sum(rewards))
        episode_returns.append(ep_return)
        actor_losses.append(float(actor_loss.item()))
        critic_losses.append(float(critic_loss.item()))

        if (ep + 1) % 200 == 0:
            recent = episode_returns[-200:]
            print(f"    Episode {ep+1}/{n_episodes}: "
                  f"mean_return={np.mean(recent):.3f}")

    return {
        "episode_returns": episode_returns,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
    }


def evaluate_rl_policy(
    policy: RLRetrievalPolicy,
    users: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    precision_fn: Callable,
    sigma_0_fn: Callable,
    budget: int = 5,
) -> dict[str, Any]:
    """Evaluate the RL policy on a set of users.

    Returns:
        Dict with 'total_gains', 'final_log_dets', per user.
    """
    doc_embeddings = compute_document_embeddings(corpus)
    total_gains = []
    final_log_dets = []

    policy.eval()
    with torch.no_grad():
        for user in users:
            sigma_0 = sigma_0_fn(user)
            env = RetrievalEnv(user, corpus, precision_fn, sigma_0, budget)
            state = env.reset()

            for _t in range(budget):
                available = env.get_available_actions()
                action, _ = policy.get_action(
                    state, doc_embeddings, budget, available, explore=False
                )
                state, _, done, _ = env.step(action)
                if done:
                    break

            total_gain = log_det(sigma_0) - log_det(state.sigma_current)
            total_gains.append(float(total_gain))
            final_log_dets.append(float(log_det(state.sigma_current)))

    return {
        "total_gains": total_gains,
        "final_log_dets": final_log_dets,
        "mean_gain": float(np.mean(total_gains)),
        "mean_log_det": float(np.mean(final_log_dets)),
    }
