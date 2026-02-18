"""
Few-shot LLM baseline: include example (profile → allocation) pairs in the prompt.
"""

from __future__ import annotations

import json
import numpy as np
from typing import Any

from src.data.user_profiles import ASSET_CLASSES
from src.baselines.zero_shot import parse_weights
from src.utils.ollama_client import ollama_generate


def few_shot_baseline(
    user_profile: dict[str, Any],
    examples: list[tuple[dict[str, Any], np.ndarray]],
    model: str = "mistral",
) -> np.ndarray:
    """Few-shot LLM baseline with example demonstrations.

    Args:
        user_profile: user profile dict.
        examples: list of (profile, allocation) pairs for few-shot prompting.
        model: Ollama model name.

    Returns:
        (8,) portfolio weight vector.
    """
    examples_text = ""
    for i, (prof, alloc) in enumerate(examples[:5]):
        alloc_dict = {ac: round(float(w), 3) for ac, w in zip(ASSET_CLASSES, alloc)}
        examples_text += f"""
Example {i+1}:
Profile: risk_tolerance={prof['risk_tolerance']:.2f}, horizon={prof['investment_horizon_months']}mo, age={prof['demographic']['age']}
Allocation: {json.dumps(alloc_dict)}
"""

    prompt = f"""You are a financial advisor. Given client profiles, recommend portfolio weights
across these 8 asset classes: {ASSET_CLASSES}.

Here are some examples:
{examples_text}

Now recommend for this new client:
- Risk tolerance: {user_profile['risk_tolerance']:.2f} (0=conservative, 1=aggressive)
- Investment horizon: {user_profile['investment_horizon_months']} months
- Sector preferences: {user_profile['sector_preferences']}
- Age: {user_profile['demographic']['age']}

Respond with ONLY a JSON object mapping asset class to weight (0-1, summing to 1)."""

    response = ollama_generate(prompt, model=model)
    return parse_weights(response)
