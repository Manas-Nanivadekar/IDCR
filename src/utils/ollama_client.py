"""
Ollama API client for LLM-based baselines.

Wraps the Ollama HTTP API for text generation. All responses are cached
to disk for reproducibility and efficiency.
"""

from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from typing import Any


OLLAMA_URL = "http://localhost:11434/api/generate"
CACHE_DIR = Path("data/llm_cache")


def _cache_key(prompt: str, model: str) -> str:
    """Generate a deterministic cache key from prompt + model."""
    h = hashlib.sha256(f"{model}::{prompt}".encode()).hexdigest()[:16]
    return f"{model}_{h}"


def ollama_generate(
    prompt: str,
    model: str = "mistral",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    use_cache: bool = True,
) -> str:
    """Generate text using Ollama.

    Args:
        prompt: the prompt to send.
        model: Ollama model name.
        temperature: sampling temperature.
        max_tokens: maximum tokens to generate.
        use_cache: if True, cache responses to disk.

    Returns:
        Generated text response.
    """
    # Check cache
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = _cache_key(prompt, model)
        cache_path = CACHE_DIR / f"{key}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)["response"]

    # Call Ollama API
    try:
        import requests
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()["response"]
    except Exception as e:
        # Fallback: return a placeholder if Ollama is not available
        result = f"[Ollama unavailable: {e}] Placeholder response for: {prompt[:100]}"

    # Cache
    if use_cache:
        cache_data = {"model": model, "prompt": prompt, "response": result}
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)

    return result


def ollama_available(model: str = "mistral") -> bool:
    """Check if Ollama is available and the model is loaded."""
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(model in m for m in models)
    except Exception:
        pass
    return False
