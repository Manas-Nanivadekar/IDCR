"""
Zero-shot LLM baseline: ask the LLM for portfolio allocation directly
without any retrieved documents.
"""

from __future__ import annotations

import json
import re
import numpy as np
from typing import Any

from src.data.user_profiles import ASSET_CLASSES
from src.utils.ollama_client import ollama_generate
from src.utils.linear_algebra import project_simplex


def zero_shot_baseline(
    user_profile: dict[str, Any],
    model: str = "mistral",
) -> np.ndarray:
    """Send profile to Ollama, ask for portfolio allocation directly.

    Args:
        user_profile: user profile dict.
        model: Ollama model name.

    Returns:
        (8,) portfolio weight vector on the simplex.
    """
    prompt = f"""You are a financial advisor. Given this client profile, recommend portfolio weights
across these 8 asset classes: {ASSET_CLASSES}.

Client Profile:
- Risk tolerance: {user_profile['risk_tolerance']:.2f} (0=conservative, 1=aggressive)
- Investment horizon: {user_profile['investment_horizon_months']} months
- Sector preferences: {user_profile['sector_preferences']}
- Age: {user_profile['demographic']['age']}

Respond with ONLY a JSON object mapping asset class to weight (0-1, summing to 1).
Example: {{"US_equity": 0.2, "intl_equity": 0.15, ...}}"""

    response = ollama_generate(prompt, model=model)
    return parse_weights(response)


def parse_weights(response: str) -> np.ndarray:
    """Parse LLM response into portfolio weight vector.

    Args:
        response: LLM text response.

    Returns:
        (8,) portfolio weights on the simplex.
    """
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response.replace('\n', ' '))
        if json_match:
            weights_dict = json.loads(json_match.group())
            weights = np.array([
                float(weights_dict.get(ac, 0.0))
                for ac in ASSET_CLASSES
            ])
        else:
            # Fallback: equal weights
            weights = np.ones(len(ASSET_CLASSES)) / len(ASSET_CLASSES)
    except (json.JSONDecodeError, ValueError):
        weights = np.ones(len(ASSET_CLASSES)) / len(ASSET_CLASSES)

    # Ensure simplex constraint
    weights = np.maximum(weights, 0.0)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(ASSET_CLASSES)) / len(ASSET_CLASSES)

    return weights
