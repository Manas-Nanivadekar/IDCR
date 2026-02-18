"""Unit tests for data generation modules."""

import numpy as np
import pytest

from src.data.user_profiles import (
    ASSET_CLASSES,
    generate_profiles,
    generate_splits,
)
from src.data.documents import generate_corpus, DOC_TYPES
from src.data.precision_ground_truth import (
    compute_ground_truth_precision,
    compute_prior_covariance,
    compute_optimal_allocation,
    get_market_covariance,
)
from src.utils.linear_algebra import is_psd


class TestUserProfiles:
    def test_profile_count(self):
        profiles = generate_profiles(100, seed=42)
        assert len(profiles) == 100

    def test_required_fields(self):
        profiles = generate_profiles(10, seed=42)
        required = {"user_id", "risk_tolerance", "investment_horizon_months",
                     "sector_preferences", "constraints", "current_holdings",
                     "demographic"}
        for p in profiles:
            assert required.issubset(set(p.keys()))

    def test_risk_tolerance_range(self):
        profiles = generate_profiles(500, seed=42)
        for p in profiles:
            assert 0.0 <= p["risk_tolerance"] <= 1.0

    def test_holdings_sum_to_one(self):
        profiles = generate_profiles(100, seed=42)
        for p in profiles:
            total = sum(p["current_holdings"].values())
            assert abs(total - 1.0) < 0.01, f"Holdings sum to {total}"

    def test_splits_non_overlapping(self):
        splits = generate_splits(100, 20, 20, seed=42)
        assert len(splits["train"]) == 100
        assert len(splits["calibration"]) == 20
        assert len(splits["test"]) == 20
        train_ids = {p["user_id"] for p in splits["train"]}
        cal_ids = {p["user_id"] for p in splits["calibration"]}
        test_ids = {p["user_id"] for p in splits["test"]}
        assert len(train_ids & cal_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(cal_ids & test_ids) == 0

    def test_reproducibility(self):
        p1 = generate_profiles(10, seed=42)
        p2 = generate_profiles(10, seed=42)
        for a, b in zip(p1, p2):
            assert a["risk_tolerance"] == b["risk_tolerance"]


class TestDocuments:
    def test_corpus_size(self):
        corpus = generate_corpus(200, seed=42)
        assert len(corpus) == 200

    def test_valid_doc_types(self):
        corpus = generate_corpus(50, seed=42)
        for doc in corpus:
            assert doc["doc_type"] in DOC_TYPES

    def test_informativeness_positive(self):
        corpus = generate_corpus(50, seed=42)
        for doc in corpus:
            assert doc["informativeness"] > 0

    def test_relevant_sectors_valid(self):
        corpus = generate_corpus(50, seed=42)
        for doc in corpus:
            for sector in doc["relevant_sectors"]:
                assert sector in ASSET_CLASSES


class TestPrecisionGroundTruth:
    def test_precision_psd(self):
        profiles = generate_profiles(10, seed=42)
        corpus = generate_corpus(20, seed=42)
        for profile in profiles[:5]:
            for doc in corpus[:10]:
                Lambda = compute_ground_truth_precision(doc, profile)
                assert Lambda.shape == (8, 8)
                assert is_psd(Lambda), f"Precision not PSD for user {profile['user_id']}, doc {doc['doc_id']}"

    def test_precision_symmetric(self):
        profiles = generate_profiles(5, seed=42)
        corpus = generate_corpus(10, seed=42)
        for p in profiles:
            for d in corpus[:5]:
                Lambda = compute_ground_truth_precision(d, p)
                np.testing.assert_allclose(Lambda, Lambda.T, atol=1e-10)

    def test_prior_covariance_psd(self):
        profiles = generate_profiles(20, seed=42)
        for p in profiles:
            sigma = compute_prior_covariance(p)
            assert sigma.shape == (8, 8)
            assert is_psd(sigma)

    def test_market_covariance_psd(self):
        sigma = get_market_covariance()
        assert sigma.shape == (8, 8)
        assert is_psd(sigma)
        np.testing.assert_allclose(sigma, sigma.T, atol=1e-10)

    def test_optimal_allocation_simplex(self):
        profiles = generate_profiles(50, seed=42)
        for p in profiles:
            y_star = compute_optimal_allocation(p)
            assert y_star.shape == (8,)
            assert np.all(y_star >= -1e-10), f"Negative weight: {y_star}"
            assert abs(np.sum(y_star) - 1.0) < 1e-8, f"Sum = {np.sum(y_star)}"


class TestSynergyControl:
    def test_synergistic_pair_psd(self):
        from src.data.synergy_control import create_synergistic_pair
        Lambda_1, Lambda_2, Lambda_syn = create_synergistic_pair(0, 3)
        assert is_psd(Lambda_1)
        assert is_psd(Lambda_2)
        assert is_psd(Lambda_syn)

    def test_redundant_pair_psd(self):
        from src.data.synergy_control import create_redundant_pair
        Lambda_1, Lambda_2 = create_redundant_pair([0, 1])
        assert is_psd(Lambda_1)
        assert is_psd(Lambda_2)

    def test_corpus_types(self):
        from src.data.synergy_control import (
            create_redundant_corpus,
            create_synergistic_corpus,
            create_mixed_corpus,
        )
        for create_fn in [create_redundant_corpus, create_synergistic_corpus,
                          create_mixed_corpus]:
            data = create_fn(n=20, seed=42)
            assert "documents" in data
            assert "precision_matrices" in data
            assert len(data["documents"]) > 0
