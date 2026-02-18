"""
Synthetic document corpus generation for the IDCR framework.

Each document represents a financial information source (sector report, macro
analysis, etc.) with known metadata that determines its precision contribution.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from src.data.user_profiles import ASSET_CLASSES

DOC_TYPES = [
    "sector_report",
    "macro_analysis",
    "earnings_summary",
    "risk_assessment",
    "market_outlook",
]

# Mapping from doc_type to which asset classes it's likely to inform
DOC_TYPE_SECTOR_MAP: dict[str, list[str]] = {
    "sector_report": ["US_equity", "intl_equity", "emerging_markets"],
    "macro_analysis": ["bonds", "cash", "commodities", "US_equity"],
    "earnings_summary": ["US_equity", "intl_equity"],
    "risk_assessment": ["bonds", "alternatives", "commodities", "cash"],
    "market_outlook": ASSET_CLASSES.copy(),  # can inform any
}

# Sentiment options (used for LLM augmentation later)
SENTIMENTS = ["bullish", "bearish", "neutral", "mixed"]


def generate_document(doc_id: int, rng: np.random.Generator) -> dict[str, Any]:
    """Generate a single synthetic document.

    Args:
        doc_id: unique identifier.
        rng: NumPy random generator.

    Returns:
        Document dict with metadata.
    """
    doc_type = str(rng.choice(DOC_TYPES))
    base_sectors = DOC_TYPE_SECTOR_MAP[doc_type]

    # Sample 1-4 relevant sectors from the doc_type's base sectors
    n_sectors = min(int(rng.integers(1, min(5, len(base_sectors) + 1))),
                    len(base_sectors))
    relevant_sectors = list(rng.choice(base_sectors, size=n_sectors, replace=False))

    # Sometimes add an extra sector not in the base (cross-cutting report)
    if rng.random() < 0.2:
        extra_pool = [s for s in ASSET_CLASSES if s not in relevant_sectors]
        if extra_pool:
            relevant_sectors.append(str(rng.choice(extra_pool)))

    # Base informativeness ~ Gamma(2, 1)
    informativeness = float(rng.gamma(2.0, 1.0))

    # Time horizon the document discusses
    time_horizon = str(rng.choice(["short_term", "medium_term", "long_term"]))

    # Sentiment
    sentiment = str(rng.choice(SENTIMENTS))

    # Key metrics (for later LLM augmentation)
    key_metrics = _sample_key_metrics(rng, doc_type)

    return {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "relevant_sectors": relevant_sectors,
        "informativeness": round(informativeness, 4),
        "time_horizon": time_horizon,
        "sentiment": sentiment,
        "key_metrics": key_metrics,
    }


def _sample_key_metrics(rng: np.random.Generator,
                        doc_type: str) -> dict[str, float]:
    """Sample realistic key metrics based on document type."""
    metrics: dict[str, float] = {}
    if doc_type in ("sector_report", "earnings_summary"):
        metrics["revenue_growth_pct"] = round(float(rng.normal(5, 15)), 2)
        metrics["pe_ratio"] = round(float(rng.uniform(8, 40)), 1)
        metrics["eps_surprise_pct"] = round(float(rng.normal(0, 5)), 2)
    elif doc_type == "macro_analysis":
        metrics["gdp_growth_pct"] = round(float(rng.normal(2, 1.5)), 2)
        metrics["inflation_pct"] = round(float(rng.uniform(0.5, 8)), 2)
        metrics["interest_rate_pct"] = round(float(rng.uniform(0, 6)), 2)
    elif doc_type == "risk_assessment":
        metrics["volatility_index"] = round(float(rng.uniform(10, 40)), 1)
        metrics["sharpe_ratio"] = round(float(rng.normal(1.0, 0.5)), 2)
        metrics["max_drawdown_pct"] = round(float(rng.uniform(5, 30)), 1)
    elif doc_type == "market_outlook":
        metrics["target_return_pct"] = round(float(rng.normal(8, 5)), 2)
        metrics["confidence_score"] = round(float(rng.uniform(0.3, 0.9)), 2)
    return metrics


def generate_corpus(n: int = 200, seed: int = 42) -> list[dict[str, Any]]:
    """Generate n synthetic documents.

    Args:
        n: number of documents.
        seed: random seed.

    Returns:
        List of document dicts.
    """
    rng = np.random.default_rng(seed)
    return [generate_document(i, rng) for i in range(n)]
