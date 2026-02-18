"""
Standard RAG baseline: retrieve by cosine similarity, include doc texts
in LLM prompt, ask for allocation.
"""

from __future__ import annotations

import json
import numpy as np
from typing import Any

from src.data.user_profiles import ASSET_CLASSES
from src.baselines.zero_shot import parse_weights
from src.utils.ollama_client import ollama_generate


def standard_rag_baseline(
    user_profile: dict[str, Any],
    retrieved_texts: list[str],
    model: str = "mistral",
) -> np.ndarray:
    """Standard RAG: embed profile, retrieve top-k docs by cosine sim,
    include doc texts in prompt, ask for allocation.

    Args:
        user_profile: user profile dict.
        retrieved_texts: list of retrieved document texts.
        model: Ollama model name.

    Returns:
        (8,) portfolio weight vector.
    """
    docs_text = "\n\n---\n\n".join(
        f"Document {i+1}:\n{text}" for i, text in enumerate(retrieved_texts[:5])
    )

    prompt = f"""You are a financial advisor. Based on the following market research documents
and client profile, recommend portfolio weights across these 8 asset classes: {ASSET_CLASSES}.

Client Profile:
- Risk tolerance: {user_profile['risk_tolerance']:.2f} (0=conservative, 1=aggressive)
- Investment horizon: {user_profile['investment_horizon_months']} months
- Sector preferences: {user_profile['sector_preferences']}
- Age: {user_profile['demographic']['age']}

Relevant Research:
{docs_text}

Based on the above research and client profile, provide portfolio weights.
Respond with ONLY a JSON object mapping asset class to weight (0-1, summing to 1)."""

    response = ollama_generate(prompt, model=model)
    return parse_weights(response)
