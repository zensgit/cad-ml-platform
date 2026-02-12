"""
Multi-source fusion for HybridClassifier.

Provides advanced fusion strategies for combining predictions from multiple sources.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FusionStrategy(str, Enum):
    """Fusion strategy types."""
    WEIGHTED_AVERAGE = "weighted_average"
    VOTING = "voting"
    STACKING = "stacking"
    ATTENTION = "attention"
    DEMPSTER_SHAFER = "dempster_shafer"
    BAYESIAN = "bayesian"


@dataclass
class SourcePrediction:
    """Prediction from a single source."""
    source_name: str
    label: Optional[str]
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    features: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.label is not None and self.confidence > 0


@dataclass
class FusionResult:
    """Result of multi-source fusion."""
    label: str
    confidence: float
    probabilities: Dict[str, float]
    source_contributions: Dict[str, float]
    fusion_strategy: FusionStrategy
    agreement_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "source_contributions": self.source_contributions,
            "fusion_strategy": self.fusion_strategy.value,
            "agreement_score": self.agreement_score,
            "metadata": self.metadata,
        }


class FusionEngine(ABC):
    """Abstract base class for fusion engines."""

    @abstractmethod
    def fuse(
        self,
        predictions: List[SourcePrediction],
        weights: Optional[Dict[str, float]] = None,
    ) -> FusionResult:
        """Fuse predictions from multiple sources."""
        pass


class WeightedAverageFusion(FusionEngine):
    """Weighted average fusion strategy."""

    def __init__(
        self,
        normalize_weights: bool = True,
        min_sources: int = 1,
        agreement_bonus: float = 0.1,
    ):
        self.normalize_weights = normalize_weights
        self.min_sources = min_sources
        self.agreement_bonus = agreement_bonus

    def fuse(
        self,
        predictions: List[SourcePrediction],
        weights: Optional[Dict[str, float]] = None,
    ) -> FusionResult:
        """Fuse using weighted averaging."""
        valid_preds = [p for p in predictions if p.is_valid]

        if len(valid_preds) < self.min_sources:
            return FusionResult(
                label="unknown",
                confidence=0.0,
                probabilities={},
                source_contributions={},
                fusion_strategy=FusionStrategy.WEIGHTED_AVERAGE,
                agreement_score=0.0,
            )

        # Default equal weights
        if weights is None:
            weights = {p.source_name: 1.0 for p in valid_preds}

        # Normalize weights
        if self.normalize_weights:
            total = sum(weights.get(p.source_name, 1.0) for p in valid_preds)
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}

        # Aggregate scores per label
        label_scores: Dict[str, float] = {}
        source_contributions: Dict[str, float] = {}

        for pred in valid_preds:
            w = weights.get(pred.source_name, 1.0)
            score = pred.confidence * w

            if pred.label:
                label_scores[pred.label] = label_scores.get(pred.label, 0.0) + score
                source_contributions[pred.source_name] = score

        if not label_scores:
            return FusionResult(
                label="unknown",
                confidence=0.0,
                probabilities={},
                source_contributions=source_contributions,
                fusion_strategy=FusionStrategy.WEIGHTED_AVERAGE,
                agreement_score=0.0,
            )

        # Find best label
        best_label = max(label_scores.items(), key=lambda x: x[1])[0]

        # Calculate agreement score
        agreeing = sum(1 for p in valid_preds if p.label == best_label)
        agreement_score = agreeing / len(valid_preds) if valid_preds else 0.0

        # Apply agreement bonus
        confidence = label_scores[best_label]
        if agreement_score >= 0.5:
            confidence = min(1.0, confidence + self.agreement_bonus * agreement_score)

        # Normalize probabilities
        total_score = sum(label_scores.values())
        probabilities = {k: v / total_score for k, v in label_scores.items()} if total_score > 0 else {}

        return FusionResult(
            label=best_label,
            confidence=confidence,
            probabilities=probabilities,
            source_contributions=source_contributions,
            fusion_strategy=FusionStrategy.WEIGHTED_AVERAGE,
            agreement_score=agreement_score,
        )


class VotingFusion(FusionEngine):
    """Majority voting fusion strategy."""

    def __init__(
        self,
        voting_type: str = "soft",  # "hard" or "soft"
        min_votes: int = 1,
    ):
        self.voting_type = voting_type
        self.min_votes = min_votes

    def fuse(
        self,
        predictions: List[SourcePrediction],
        weights: Optional[Dict[str, float]] = None,
    ) -> FusionResult:
        """Fuse using voting."""
        valid_preds = [p for p in predictions if p.is_valid]

        if len(valid_preds) < self.min_votes:
            return FusionResult(
                label="unknown",
                confidence=0.0,
                probabilities={},
                source_contributions={},
                fusion_strategy=FusionStrategy.VOTING,
                agreement_score=0.0,
            )

        if weights is None:
            weights = {p.source_name: 1.0 for p in valid_preds}

        if self.voting_type == "hard":
            # Count votes per label
            votes: Dict[str, float] = {}
            for pred in valid_preds:
                if pred.label:
                    w = weights.get(pred.source_name, 1.0)
                    votes[pred.label] = votes.get(pred.label, 0.0) + w
        else:
            # Soft voting with confidence weighting
            votes: Dict[str, float] = {}
            for pred in valid_preds:
                if pred.label:
                    w = weights.get(pred.source_name, 1.0)
                    votes[pred.label] = votes.get(pred.label, 0.0) + pred.confidence * w

        if not votes:
            return FusionResult(
                label="unknown",
                confidence=0.0,
                probabilities={},
                source_contributions={},
                fusion_strategy=FusionStrategy.VOTING,
                agreement_score=0.0,
            )

        best_label = max(votes.items(), key=lambda x: x[1])[0]
        total_votes = sum(votes.values())
        confidence = votes[best_label] / total_votes if total_votes > 0 else 0.0

        # Agreement score
        agreeing = sum(1 for p in valid_preds if p.label == best_label)
        agreement_score = agreeing / len(valid_preds) if valid_preds else 0.0

        # Probabilities
        probabilities = {k: v / total_votes for k, v in votes.items()} if total_votes > 0 else {}

        # Source contributions
        source_contributions = {
            p.source_name: (weights.get(p.source_name, 1.0) * (p.confidence if self.voting_type == "soft" else 1.0))
            for p in valid_preds
        }

        return FusionResult(
            label=best_label,
            confidence=confidence,
            probabilities=probabilities,
            source_contributions=source_contributions,
            fusion_strategy=FusionStrategy.VOTING,
            agreement_score=agreement_score,
        )


class DempsterShaferFusion(FusionEngine):
    """Dempster-Shafer evidence theory fusion."""

    def __init__(self, conflict_threshold: float = 0.9):
        self.conflict_threshold = conflict_threshold

    def fuse(
        self,
        predictions: List[SourcePrediction],
        weights: Optional[Dict[str, float]] = None,
    ) -> FusionResult:
        """Fuse using Dempster-Shafer combination rule."""
        valid_preds = [p for p in predictions if p.is_valid]

        if not valid_preds:
            return FusionResult(
                label="unknown",
                confidence=0.0,
                probabilities={},
                source_contributions={},
                fusion_strategy=FusionStrategy.DEMPSTER_SHAFER,
                agreement_score=0.0,
            )

        # Convert predictions to mass functions
        mass_functions = []
        for pred in valid_preds:
            if pred.probabilities:
                mass = dict(pred.probabilities)
            elif pred.label:
                mass = {pred.label: pred.confidence, "uncertainty": 1 - pred.confidence}
            else:
                continue
            mass_functions.append(mass)

        if len(mass_functions) < 2:
            # Not enough for combination
            if valid_preds:
                p = valid_preds[0]
                return FusionResult(
                    label=p.label or "unknown",
                    confidence=p.confidence,
                    probabilities=p.probabilities or {},
                    source_contributions={p.source_name: p.confidence},
                    fusion_strategy=FusionStrategy.DEMPSTER_SHAFER,
                    agreement_score=1.0,
                )

        # Combine mass functions
        combined = mass_functions[0].copy()
        for mass in mass_functions[1:]:
            combined = self._combine_masses(combined, mass)

        # Remove uncertainty and normalize
        uncertainty = combined.pop("uncertainty", 0.0)
        if combined:
            total = sum(combined.values())
            if total > 0:
                combined = {k: v / total for k, v in combined.items()}

        if not combined:
            return FusionResult(
                label="unknown",
                confidence=0.0,
                probabilities={},
                source_contributions={},
                fusion_strategy=FusionStrategy.DEMPSTER_SHAFER,
                agreement_score=0.0,
            )

        best_label = max(combined.items(), key=lambda x: x[1])[0]
        confidence = combined[best_label]

        # Agreement
        agreeing = sum(1 for p in valid_preds if p.label == best_label)
        agreement_score = agreeing / len(valid_preds) if valid_preds else 0.0

        source_contributions = {p.source_name: p.confidence for p in valid_preds}

        return FusionResult(
            label=best_label,
            confidence=confidence,
            probabilities=combined,
            source_contributions=source_contributions,
            fusion_strategy=FusionStrategy.DEMPSTER_SHAFER,
            agreement_score=agreement_score,
            metadata={"uncertainty": uncertainty},
        )

    def _combine_masses(self, m1: Dict[str, float], m2: Dict[str, float]) -> Dict[str, float]:
        """Combine two mass functions using Dempster's rule."""
        combined: Dict[str, float] = {}
        conflict = 0.0

        for h1, v1 in m1.items():
            for h2, v2 in m2.items():
                if h1 == "uncertainty":
                    combined[h2] = combined.get(h2, 0.0) + v1 * v2
                elif h2 == "uncertainty":
                    combined[h1] = combined.get(h1, 0.0) + v1 * v2
                elif h1 == h2:
                    combined[h1] = combined.get(h1, 0.0) + v1 * v2
                else:
                    conflict += v1 * v2

        # Normalize
        if conflict < self.conflict_threshold:
            normalizer = 1.0 - conflict
            if normalizer > 0:
                combined = {k: v / normalizer for k, v in combined.items()}
        else:
            logger.warning(f"High conflict in DS fusion: {conflict:.3f}")

        return combined


class AttentionFusion(FusionEngine):
    """Attention-based fusion using learned weights."""

    def __init__(
        self,
        learn_weights: bool = False,
        temperature: float = 1.0,
    ):
        self.learn_weights = learn_weights
        self.temperature = temperature
        self._learned_weights: Optional[Dict[str, float]] = None

    def fuse(
        self,
        predictions: List[SourcePrediction],
        weights: Optional[Dict[str, float]] = None,
    ) -> FusionResult:
        """Fuse using attention mechanism."""
        valid_preds = [p for p in predictions if p.is_valid]

        if not valid_preds:
            return FusionResult(
                label="unknown",
                confidence=0.0,
                probabilities={},
                source_contributions={},
                fusion_strategy=FusionStrategy.ATTENTION,
                agreement_score=0.0,
            )

        # Compute attention scores based on confidence
        raw_scores = [p.confidence for p in valid_preds]

        # Softmax with temperature
        exp_scores = np.exp(np.array(raw_scores) / self.temperature)
        attention_weights = exp_scores / exp_scores.sum()

        # Apply learned weights if available
        if self._learned_weights:
            for i, pred in enumerate(valid_preds):
                if pred.source_name in self._learned_weights:
                    attention_weights[i] *= self._learned_weights[pred.source_name]
            attention_weights = attention_weights / attention_weights.sum()

        # Aggregate
        label_scores: Dict[str, float] = {}
        for pred, w in zip(valid_preds, attention_weights):
            if pred.label:
                label_scores[pred.label] = label_scores.get(pred.label, 0.0) + w * pred.confidence

        if not label_scores:
            return FusionResult(
                label="unknown",
                confidence=0.0,
                probabilities={},
                source_contributions={},
                fusion_strategy=FusionStrategy.ATTENTION,
                agreement_score=0.0,
            )

        best_label = max(label_scores.items(), key=lambda x: x[1])[0]
        confidence = label_scores[best_label]

        # Normalize
        total = sum(label_scores.values())
        probabilities = {k: v / total for k, v in label_scores.items()} if total > 0 else {}

        # Agreement
        agreeing = sum(1 for p in valid_preds if p.label == best_label)
        agreement_score = agreeing / len(valid_preds) if valid_preds else 0.0

        source_contributions = {
            pred.source_name: float(w)
            for pred, w in zip(valid_preds, attention_weights)
        }

        return FusionResult(
            label=best_label,
            confidence=confidence,
            probabilities=probabilities,
            source_contributions=source_contributions,
            fusion_strategy=FusionStrategy.ATTENTION,
            agreement_score=agreement_score,
            metadata={"attention_weights": attention_weights.tolist()},
        )

    def update_weights(self, source_name: str, weight: float) -> None:
        """Update learned weights for a source."""
        if self._learned_weights is None:
            self._learned_weights = {}
        self._learned_weights[source_name] = weight


class MultiSourceFusion:
    """
    Multi-source fusion manager.

    Supports multiple fusion strategies and automatic strategy selection.
    """

    STRATEGIES = {
        FusionStrategy.WEIGHTED_AVERAGE: WeightedAverageFusion,
        FusionStrategy.VOTING: VotingFusion,
        FusionStrategy.DEMPSTER_SHAFER: DempsterShaferFusion,
        FusionStrategy.ATTENTION: AttentionFusion,
    }

    def __init__(
        self,
        default_strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE,
        auto_select: bool = False,
    ):
        self.default_strategy = default_strategy
        self.auto_select = auto_select
        self._engines: Dict[FusionStrategy, FusionEngine] = {}

    def get_engine(self, strategy: FusionStrategy) -> FusionEngine:
        """Get or create fusion engine for strategy."""
        if strategy not in self._engines:
            engine_class = self.STRATEGIES.get(strategy)
            if engine_class is None:
                raise ValueError(f"Unknown fusion strategy: {strategy}")
            self._engines[strategy] = engine_class()
        return self._engines[strategy]

    def fuse(
        self,
        predictions: List[SourcePrediction],
        weights: Optional[Dict[str, float]] = None,
        strategy: Optional[FusionStrategy] = None,
    ) -> FusionResult:
        """
        Fuse predictions using specified or auto-selected strategy.

        Args:
            predictions: List of source predictions
            weights: Optional source weights
            strategy: Fusion strategy (None for auto-select or default)

        Returns:
            FusionResult
        """
        if strategy is None:
            if self.auto_select:
                strategy = self._select_strategy(predictions)
            else:
                strategy = self.default_strategy

        engine = self.get_engine(strategy)
        return engine.fuse(predictions, weights)

    def _select_strategy(self, predictions: List[SourcePrediction]) -> FusionStrategy:
        """Auto-select best fusion strategy based on predictions."""
        valid_preds = [p for p in predictions if p.is_valid]
        n_sources = len(valid_preds)

        if n_sources <= 1:
            return FusionStrategy.WEIGHTED_AVERAGE

        # Check agreement
        labels = [p.label for p in valid_preds if p.label]
        unique_labels = set(labels)

        if len(unique_labels) == 1:
            # All agree - simple average
            return FusionStrategy.WEIGHTED_AVERAGE
        elif len(unique_labels) == len(labels):
            # All disagree - use Dempster-Shafer for conflict handling
            return FusionStrategy.DEMPSTER_SHAFER
        else:
            # Partial agreement - use voting
            return FusionStrategy.VOTING
