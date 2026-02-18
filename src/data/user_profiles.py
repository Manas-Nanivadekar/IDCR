"""
User profile generation for the IDCR framework.

Generates structured user profiles (profile cards) by sampling from a mixture
of K=5 archetypal investor profiles. Each profile is a dict with fields for
risk tolerance, investment horizon, sector preferences, constraints, holdings,
and demographic information.
"""

from __future__ import annotations

import numpy as np
from typing import Any

# ── 8 asset classes ──────────────────────────────────────────────────────────
ASSET_CLASSES = [
    "US_equity",
    "intl_equity",
    "emerging_markets",
    "bonds",
    "real_estate",
    "commodities",
    "cash",
    "alternatives",
]

# ── Sector tags (superset used by archetypes) ─────────────────────────────────
ALL_SECTORS = [
    "technology", "healthcare", "bonds", "utilities", "dividends",
    "REITs", "clean_energy", "growth", "crypto", "emerging_markets",
    "real_estate", "commodities", "alternatives", "cash",
    "US_equity", "intl_equity",
]

# ── Archetype definitions ────────────────────────────────────────────────────

ARCHETYPES: list[dict[str, Any]] = [
    {
        "name": "conservative_retiree",
        "risk_beta": (2, 8),
        "horizon_range": (6, 24),
        "sector_prefs": ["bonds", "utilities", "healthcare", "cash"],
        "weight_prior": np.array([0.05, 0.05, 0.02, 0.45, 0.05, 0.03, 0.30, 0.05]),
        "age_range": (55, 80),
        "income_bracket": "low_to_medium",
    },
    {
        "name": "aggressive_trader",
        "risk_beta": (8, 2),
        "horizon_range": (1, 6),
        "sector_prefs": ["technology", "crypto", "growth", "emerging_markets"],
        "weight_prior": np.array([0.30, 0.15, 0.20, 0.02, 0.03, 0.05, 0.05, 0.20]),
        "age_range": (22, 40),
        "income_bracket": "high",
    },
    {
        "name": "balanced_longterm",
        "risk_beta": (5, 5),
        "horizon_range": (36, 120),
        "sector_prefs": ["US_equity", "intl_equity", "bonds", "real_estate",
                         "commodities", "alternatives"],
        "weight_prior": np.array([0.20, 0.15, 0.10, 0.20, 0.10, 0.10, 0.05, 0.10]),
        "age_range": (30, 55),
        "income_bracket": "medium_to_high",
    },
    {
        "name": "income_focused",
        "risk_beta": (3, 6),
        "horizon_range": (12, 60),
        "sector_prefs": ["dividends", "REITs", "bonds", "utilities"],
        "weight_prior": np.array([0.10, 0.05, 0.03, 0.35, 0.20, 0.02, 0.15, 0.10]),
        "age_range": (40, 65),
        "income_bracket": "medium",
    },
    {
        "name": "esg_thematic",
        "risk_beta": (5, 4),
        "horizon_range": (24, 60),
        "sector_prefs": ["clean_energy", "healthcare", "technology"],
        "weight_prior": np.array([0.15, 0.15, 0.10, 0.15, 0.05, 0.05, 0.10, 0.25]),
        "age_range": (25, 45),
        "income_bracket": "medium_to_high",
    },
]


def _sample_holdings(rng: np.random.Generator, weight_prior: np.ndarray,
                     noise_scale: float = 0.08) -> dict[str, float]:
    """Sample portfolio holdings from a Dirichlet centred on the archetype prior."""
    # Concentration ~ prior * scale; higher scale = less noise
    alpha = np.maximum(weight_prior * 20.0 + noise_scale, 0.1)
    weights = rng.dirichlet(alpha)
    return {ac: float(round(w, 4)) for ac, w in zip(ASSET_CLASSES, weights)}


def _sample_constraints(rng: np.random.Generator,
                        archetype: dict) -> dict[str, Any]:
    """Generate portfolio constraints consistent with the archetype."""
    max_single = float(rng.uniform(0.25, 0.50))
    # Exclude 0-2 sectors that aren't in the archetype's preferences
    non_preferred = [s for s in ALL_SECTORS if s not in archetype["sector_prefs"]]
    n_excluded = rng.integers(0, min(3, len(non_preferred)))
    excluded = list(rng.choice(non_preferred, size=n_excluded, replace=False)) if n_excluded > 0 else []
    return {
        "max_single_sector": round(max_single, 2),
        "excluded_sectors": excluded,
    }


def generate_profile(user_id: int, rng: np.random.Generator) -> dict[str, Any]:
    """Generate a single user profile by sampling from an archetype.

    Args:
        user_id: unique integer identifier.
        rng: NumPy random generator for reproducibility.

    Returns:
        Profile dict with all required fields.
    """
    # Sample archetype
    k = int(rng.integers(0, len(ARCHETYPES)))
    arch = ARCHETYPES[k]

    a, b = arch["risk_beta"]
    risk_tolerance = float(rng.beta(a, b))

    lo, hi = arch["horizon_range"]
    horizon = int(rng.integers(lo, hi + 1))

    # Sector preferences: archetype core + possible random addition
    prefs = list(arch["sector_prefs"])
    if rng.random() < 0.3:
        extra = rng.choice([s for s in ALL_SECTORS if s not in prefs])
        prefs.append(str(extra))

    age_lo, age_hi = arch["age_range"]
    age = int(rng.integers(age_lo, age_hi + 1))

    return {
        "user_id": user_id,
        "archetype": arch["name"],
        "risk_tolerance": round(risk_tolerance, 4),
        "investment_horizon_months": horizon,
        "sector_preferences": prefs,
        "constraints": _sample_constraints(rng, arch),
        "current_holdings": _sample_holdings(rng, arch["weight_prior"]),
        "demographic": {
            "age": age,
            "income_bracket": arch["income_bracket"],
        },
    }


def generate_profiles(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """Generate n user profiles.

    Args:
        n: number of profiles.
        seed: random seed.

    Returns:
        List of profile dicts.
    """
    rng = np.random.default_rng(seed)
    return [generate_profile(uid, rng) for uid in range(n)]


def generate_splits(
    n_train: int = 1000,
    n_cal: int = 200,
    n_test: int = 200,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Generate train / calibration / test profile splits.

    Uses a single RNG stream so splits are deterministic and non-overlapping.

    Returns:
        Dict with keys 'train', 'calibration', 'test'.
    """
    total = n_train + n_cal + n_test
    all_profiles = generate_profiles(total, seed=seed)
    return {
        "train": all_profiles[:n_train],
        "calibration": all_profiles[n_train : n_train + n_cal],
        "test": all_profiles[n_train + n_cal :],
    }
