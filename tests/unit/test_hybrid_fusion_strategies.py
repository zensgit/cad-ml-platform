"""Tests for hybrid fusion strategies."""

from __future__ import annotations

import pytest
import numpy as np

from src.ml.hybrid.fusion import (
    AttentionFusion,
    DempsterShaferFusion,
    FusionResult,
    FusionStrategy,
    MultiSourceFusion,
    SourcePrediction,
    VotingFusion,
    WeightedAverageFusion,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agreeing_predictions():
    """Three sources all predicting the same label."""
    return [
        SourcePrediction(source_name="ml", label="steel", confidence=0.9),
        SourcePrediction(source_name="rules", label="steel", confidence=0.8),
        SourcePrediction(source_name="llm", label="steel", confidence=0.85),
    ]


@pytest.fixture
def disagreeing_predictions():
    """Three sources predicting different labels."""
    return [
        SourcePrediction(source_name="ml", label="steel", confidence=0.9),
        SourcePrediction(source_name="rules", label="aluminum", confidence=0.7),
        SourcePrediction(source_name="llm", label="copper", confidence=0.6),
    ]


@pytest.fixture
def partial_agreement_predictions():
    """Two agree, one disagrees."""
    return [
        SourcePrediction(source_name="ml", label="steel", confidence=0.9),
        SourcePrediction(source_name="rules", label="steel", confidence=0.8),
        SourcePrediction(source_name="llm", label="aluminum", confidence=0.6),
    ]


@pytest.fixture
def single_prediction():
    return [SourcePrediction(source_name="ml", label="steel", confidence=0.9)]


@pytest.fixture
def empty_predictions():
    return []


@pytest.fixture
def invalid_predictions():
    """Predictions with invalid entries."""
    return [
        SourcePrediction(source_name="bad", label=None, confidence=0.0),
    ]


# ---------------------------------------------------------------------------
# WeightedAverageFusion
# ---------------------------------------------------------------------------

class TestWeightedAverageFusion:

    def test_agreeing_sources(self, agreeing_predictions):
        engine = WeightedAverageFusion()
        result = engine.fuse(agreeing_predictions)
        assert result.label == "steel"
        assert result.confidence > 0.0
        assert result.agreement_score == 1.0
        assert result.fusion_strategy == FusionStrategy.WEIGHTED_AVERAGE

    def test_disagreeing_sources(self, disagreeing_predictions):
        engine = WeightedAverageFusion()
        result = engine.fuse(disagreeing_predictions)
        assert result.label == "steel"  # highest confidence
        assert result.agreement_score < 1.0

    def test_custom_weights(self, partial_agreement_predictions):
        engine = WeightedAverageFusion()
        weights = {"ml": 2.0, "rules": 1.0, "llm": 0.5}
        result = engine.fuse(partial_agreement_predictions, weights=weights)
        assert result.label == "steel"

    def test_empty_predictions(self, empty_predictions):
        engine = WeightedAverageFusion()
        result = engine.fuse(empty_predictions)
        assert result.label == "unknown"
        assert result.confidence == 0.0

    def test_invalid_predictions(self, invalid_predictions):
        engine = WeightedAverageFusion()
        result = engine.fuse(invalid_predictions)
        assert result.label == "unknown"
        assert result.confidence == 0.0

    def test_single_source(self, single_prediction):
        engine = WeightedAverageFusion()
        result = engine.fuse(single_prediction)
        assert result.label == "steel"
        assert result.agreement_score == 1.0

    def test_agreement_bonus_applied(self, agreeing_predictions):
        engine = WeightedAverageFusion(agreement_bonus=0.1)
        result = engine.fuse(agreeing_predictions)
        # Agreement is 1.0 so bonus should be applied
        assert result.confidence > 0.0

    def test_probabilities_sum_to_one(self, partial_agreement_predictions):
        engine = WeightedAverageFusion()
        result = engine.fuse(partial_agreement_predictions)
        if result.probabilities:
            total = sum(result.probabilities.values())
            assert abs(total - 1.0) < 1e-6

    def test_min_sources_threshold(self):
        engine = WeightedAverageFusion(min_sources=3)
        preds = [SourcePrediction(source_name="ml", label="steel", confidence=0.9)]
        result = engine.fuse(preds)
        assert result.label == "unknown"


# ---------------------------------------------------------------------------
# VotingFusion
# ---------------------------------------------------------------------------

class TestVotingFusion:

    def test_hard_voting_majority(self, partial_agreement_predictions):
        engine = VotingFusion(voting_type="hard")
        result = engine.fuse(partial_agreement_predictions)
        assert result.label == "steel"  # 2 votes vs 1
        assert result.fusion_strategy == FusionStrategy.VOTING

    def test_soft_voting(self, partial_agreement_predictions):
        engine = VotingFusion(voting_type="soft")
        result = engine.fuse(partial_agreement_predictions)
        assert result.label == "steel"

    def test_empty_predictions(self, empty_predictions):
        engine = VotingFusion()
        result = engine.fuse(empty_predictions)
        assert result.label == "unknown"

    def test_min_votes_threshold(self):
        engine = VotingFusion(min_votes=5)
        preds = [SourcePrediction(source_name="ml", label="steel", confidence=0.9)]
        result = engine.fuse(preds)
        assert result.label == "unknown"

    def test_agreement_score(self, agreeing_predictions):
        engine = VotingFusion()
        result = engine.fuse(agreeing_predictions)
        assert result.agreement_score == 1.0

    def test_probabilities(self, partial_agreement_predictions):
        engine = VotingFusion(voting_type="soft")
        result = engine.fuse(partial_agreement_predictions)
        if result.probabilities:
            total = sum(result.probabilities.values())
            assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# DempsterShaferFusion
# ---------------------------------------------------------------------------

class TestDempsterShaferFusion:

    def test_agreeing_sources_high_confidence(self, agreeing_predictions):
        engine = DempsterShaferFusion()
        result = engine.fuse(agreeing_predictions)
        assert result.label == "steel"
        assert result.confidence > 0.5
        assert result.fusion_strategy == FusionStrategy.DEMPSTER_SHAFER

    def test_disagreeing_sources(self, disagreeing_predictions):
        engine = DempsterShaferFusion()
        result = engine.fuse(disagreeing_predictions)
        # Should still pick most supported label
        assert result.label in ("steel", "aluminum", "copper")

    def test_empty_predictions(self, empty_predictions):
        engine = DempsterShaferFusion()
        result = engine.fuse(empty_predictions)
        assert result.label == "unknown"

    def test_single_source(self, single_prediction):
        engine = DempsterShaferFusion()
        result = engine.fuse(single_prediction)
        assert result.label == "steel"
        assert result.confidence == 0.9

    def test_with_probabilities(self):
        preds = [
            SourcePrediction(
                source_name="ml", label="steel", confidence=0.8,
                probabilities={"steel": 0.8, "aluminum": 0.2},
            ),
            SourcePrediction(
                source_name="rules", label="steel", confidence=0.7,
                probabilities={"steel": 0.7, "aluminum": 0.3},
            ),
        ]
        engine = DempsterShaferFusion()
        result = engine.fuse(preds)
        assert result.label == "steel"
        assert result.confidence > 0.7

    def test_high_conflict_warning(self):
        """High conflict between sources should still produce result."""
        preds = [
            SourcePrediction(
                source_name="ml", label="steel", confidence=0.99,
                probabilities={"steel": 0.99, "aluminum": 0.01},
            ),
            SourcePrediction(
                source_name="rules", label="aluminum", confidence=0.99,
                probabilities={"aluminum": 0.99, "steel": 0.01},
            ),
        ]
        engine = DempsterShaferFusion(conflict_threshold=0.9)
        result = engine.fuse(preds)
        assert result.label in ("steel", "aluminum")


# ---------------------------------------------------------------------------
# AttentionFusion
# ---------------------------------------------------------------------------

class TestAttentionFusion:

    def test_basic_fusion(self, agreeing_predictions):
        engine = AttentionFusion()
        result = engine.fuse(agreeing_predictions)
        assert result.label == "steel"
        assert result.fusion_strategy == FusionStrategy.ATTENTION
        assert "attention_weights" in result.metadata

    def test_temperature_effect(self, partial_agreement_predictions):
        low_temp = AttentionFusion(temperature=0.1)
        high_temp = AttentionFusion(temperature=10.0)
        r_low = low_temp.fuse(partial_agreement_predictions)
        r_high = high_temp.fuse(partial_agreement_predictions)
        # Both should pick steel, but attention distributions differ
        assert r_low.label == "steel"
        assert r_high.label == "steel"

    def test_empty_predictions(self, empty_predictions):
        engine = AttentionFusion()
        result = engine.fuse(empty_predictions)
        assert result.label == "unknown"

    def test_learned_weights(self, agreeing_predictions):
        engine = AttentionFusion(learn_weights=True)
        engine.update_weights("ml", 2.0)
        engine.update_weights("rules", 0.5)
        result = engine.fuse(agreeing_predictions)
        assert result.label == "steel"
        # ML should have higher contribution
        assert result.source_contributions["ml"] > result.source_contributions["rules"]

    def test_source_contributions_present(self, agreeing_predictions):
        engine = AttentionFusion()
        result = engine.fuse(agreeing_predictions)
        assert len(result.source_contributions) == 3
        for name in ["ml", "rules", "llm"]:
            assert name in result.source_contributions


# ---------------------------------------------------------------------------
# MultiSourceFusion (auto-selection)
# ---------------------------------------------------------------------------

class TestMultiSourceFusion:

    def test_default_strategy(self, agreeing_predictions):
        fusion = MultiSourceFusion()
        result = fusion.fuse(agreeing_predictions)
        assert result.fusion_strategy == FusionStrategy.WEIGHTED_AVERAGE

    def test_explicit_strategy(self, agreeing_predictions):
        fusion = MultiSourceFusion()
        result = fusion.fuse(agreeing_predictions, strategy=FusionStrategy.VOTING)
        assert result.fusion_strategy == FusionStrategy.VOTING

    def test_auto_select_all_agree(self, agreeing_predictions):
        fusion = MultiSourceFusion(auto_select=True)
        result = fusion.fuse(agreeing_predictions)
        assert result.fusion_strategy == FusionStrategy.WEIGHTED_AVERAGE

    def test_auto_select_all_disagree(self, disagreeing_predictions):
        fusion = MultiSourceFusion(auto_select=True)
        result = fusion.fuse(disagreeing_predictions)
        assert result.fusion_strategy == FusionStrategy.DEMPSTER_SHAFER

    def test_auto_select_partial_agreement(self, partial_agreement_predictions):
        fusion = MultiSourceFusion(auto_select=True)
        result = fusion.fuse(partial_agreement_predictions)
        assert result.fusion_strategy == FusionStrategy.VOTING

    def test_unknown_strategy_raises(self):
        fusion = MultiSourceFusion()
        with pytest.raises(ValueError):
            fusion.get_engine(FusionStrategy.BAYESIAN)

    def test_engine_caching(self):
        fusion = MultiSourceFusion()
        e1 = fusion.get_engine(FusionStrategy.WEIGHTED_AVERAGE)
        e2 = fusion.get_engine(FusionStrategy.WEIGHTED_AVERAGE)
        assert e1 is e2


# ---------------------------------------------------------------------------
# FusionResult / SourcePrediction
# ---------------------------------------------------------------------------

class TestDataClasses:

    def test_source_prediction_is_valid(self):
        p = SourcePrediction(source_name="ml", label="steel", confidence=0.9)
        assert p.is_valid is True

    def test_source_prediction_invalid_no_label(self):
        p = SourcePrediction(source_name="ml", label=None, confidence=0.9)
        assert p.is_valid is False

    def test_source_prediction_invalid_zero_confidence(self):
        p = SourcePrediction(source_name="ml", label="steel", confidence=0.0)
        assert p.is_valid is False

    def test_fusion_result_to_dict(self):
        result = FusionResult(
            label="steel",
            confidence=0.9,
            probabilities={"steel": 0.9, "aluminum": 0.1},
            source_contributions={"ml": 0.5},
            fusion_strategy=FusionStrategy.WEIGHTED_AVERAGE,
            agreement_score=1.0,
        )
        d = result.to_dict()
        assert d["label"] == "steel"
        assert d["fusion_strategy"] == "weighted_average"
        assert d["confidence"] == 0.9
