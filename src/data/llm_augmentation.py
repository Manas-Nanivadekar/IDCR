"""
LLM-augmented natural language document generation.

Uses Ollama to wrap synthetic document metadata in realistic financial text.
This is needed for zero-shot/few-shot/standard-RAG baselines.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from tqdm import tqdm

from src.utils.ollama_client import ollama_generate, ollama_available


CACHE_DIR = Path("data/generated_docs")


def generate_document_text(
    doc_metadata: dict[str, Any],
    model: str = "mistral",
) -> str:
    """Use Ollama to generate natural language document from structured metadata.

    Args:
        doc_metadata: document dict with doc_type, relevant_sectors, etc.
        model: Ollama model name.

    Returns:
        Generated document text (3-5 paragraphs).
    """
    prompt = f"""Generate a realistic financial {doc_metadata['doc_type']} document.

Relevant sectors: {doc_metadata['relevant_sectors']}
Key metrics to mention: {json.dumps(doc_metadata.get('key_metrics', {}))}
Sentiment: {doc_metadata.get('sentiment', 'neutral')}
Time horizon: {doc_metadata.get('time_horizon', 'medium_term')}

Write 3-5 paragraphs of realistic financial analysis text. Include specific numbers,
analyst opinions, and forward-looking statements. Do not include any headers or titles."""

    return ollama_generate(prompt, model=model)


def generate_all_document_texts(
    corpus: list[dict[str, Any]],
    model: str = "mistral",
    cache_dir: str | Path = CACHE_DIR,
) -> dict[int, str]:
    """Generate text for all documents in the corpus.

    Caches results to disk for efficiency.

    Args:
        corpus: list of document dicts.
        model: Ollama model name.
        cache_dir: directory to cache generated texts.

    Returns:
        Dict mapping doc_id → generated text.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"all_docs_{model}.json"

    # Load cache
    cached: dict[str, str] = {}
    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)

    results: dict[int, str] = {}

    for doc in tqdm(corpus, desc="Generating doc texts"):
        doc_id = doc["doc_id"]
        key = str(doc_id)

        if key in cached:
            results[doc_id] = cached[key]
        else:
            text = generate_document_text(doc, model=model)
            results[doc_id] = text
            cached[key] = text

    # Save cache
    with open(cache_file, "w") as f:
        json.dump(cached, f, indent=2)

    return results


def create_synthetic_texts(
    corpus: list[dict[str, Any]],
) -> dict[int, str]:
    """Create simple synthetic texts without LLM (fallback when Ollama unavailable).

    Args:
        corpus: list of document dicts.

    Returns:
        Dict mapping doc_id → synthetic text.
    """
    results = {}
    for doc in corpus:
        metrics = doc.get("key_metrics", {})
        metrics_str = ", ".join(f"{k}: {v}" for k, v in metrics.items())
        text = (
            f"This {doc['doc_type']} covers {', '.join(doc['relevant_sectors'])}. "
            f"The overall sentiment is {doc.get('sentiment', 'neutral')} "
            f"with a {doc.get('time_horizon', 'medium_term')} outlook. "
            f"Key metrics: {metrics_str}. "
            f"Informativeness rating: {doc['informativeness']:.2f}."
        )
        results[doc["doc_id"]] = text
    return results
