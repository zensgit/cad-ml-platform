"""
Error case analysis for model evaluation.

Provides:
- Error case collection
- Pattern detection
- Boundary case identification
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ErrorCase:
    """A single prediction error case."""
    sample_id: str
    true_label: str
    pred_label: str
    confidence: float
    true_label_prob: Optional[float] = None
    features: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def confidence_gap(self) -> Optional[float]:
        """Gap between prediction confidence and true label probability."""
        if self.true_label_prob is None:
            return None
        return self.confidence - self.true_label_prob

    @property
    def is_high_confidence_error(self) -> bool:
        """Check if this is a high-confidence error."""
        return self.confidence >= 0.8

    @property
    def is_boundary_case(self) -> bool:
        """Check if this is a boundary case (low confidence)."""
        return self.confidence < 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sample_id": self.sample_id,
            "true_label": self.true_label,
            "pred_label": self.pred_label,
            "confidence": round(self.confidence, 4),
            "true_label_prob": round(self.true_label_prob, 4) if self.true_label_prob else None,
            "confidence_gap": round(self.confidence_gap, 4) if self.confidence_gap else None,
            "is_high_confidence_error": self.is_high_confidence_error,
            "is_boundary_case": self.is_boundary_case,
            "features": self.features,
            "metadata": self.metadata,
        }


@dataclass
class ErrorPattern:
    """A pattern of errors."""
    pattern_type: str  # e.g., "confusion_pair", "low_confidence", "systematic"
    description: str
    affected_classes: List[str]
    error_count: int
    examples: List[str]  # sample_ids
    severity: str  # "high", "medium", "low"
    suggested_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "affected_classes": self.affected_classes,
            "error_count": self.error_count,
            "examples": self.examples[:5],  # Limit examples
            "severity": self.severity,
            "suggested_action": self.suggested_action,
        }


@dataclass
class ErrorCaseCollection:
    """Collection of error cases with analysis."""
    errors: List[ErrorCase]
    patterns: List[ErrorPattern]
    total_predictions: int
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def error_rate(self) -> float:
        return self.error_count / self.total_predictions if self.total_predictions > 0 else 0.0

    @property
    def high_confidence_errors(self) -> List[ErrorCase]:
        return [e for e in self.errors if e.is_high_confidence_error]

    @property
    def boundary_cases(self) -> List[ErrorCase]:
        return [e for e in self.errors if e.is_boundary_case]

    def get_errors_by_true_label(self, label: str) -> List[ErrorCase]:
        """Get errors for a specific true label."""
        return [e for e in self.errors if e.true_label == label]

    def get_errors_by_pred_label(self, label: str) -> List[ErrorCase]:
        """Get errors for a specific predicted label."""
        return [e for e in self.errors if e.pred_label == label]

    def get_confusion_errors(self, true_label: str, pred_label: str) -> List[ErrorCase]:
        """Get errors for a specific confusion pair."""
        return [
            e for e in self.errors
            if e.true_label == true_label and e.pred_label == pred_label
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_count": self.error_count,
            "total_predictions": self.total_predictions,
            "error_rate": round(self.error_rate, 4),
            "high_confidence_error_count": len(self.high_confidence_errors),
            "boundary_case_count": len(self.boundary_cases),
            "patterns": [p.to_dict() for p in self.patterns],
            "errors": [e.to_dict() for e in self.errors],
            "created_at": self.created_at.isoformat(),
        }

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary (without individual errors)."""
        return {
            "error_count": self.error_count,
            "total_predictions": self.total_predictions,
            "error_rate": round(self.error_rate, 4),
            "high_confidence_error_count": len(self.high_confidence_errors),
            "boundary_case_count": len(self.boundary_cases),
            "patterns": [p.to_dict() for p in self.patterns],
            "created_at": self.created_at.isoformat(),
        }


class ErrorAnalyzer:
    """
    Analyzer for prediction errors.

    Provides:
    - Error collection from predictions
    - Pattern detection
    - Severity classification
    """

    def __init__(
        self,
        high_confidence_threshold: float = 0.8,
        boundary_threshold: float = 0.5,
    ):
        """
        Initialize error analyzer.

        Args:
            high_confidence_threshold: Threshold for high-confidence errors
            boundary_threshold: Threshold for boundary cases
        """
        self._high_conf_threshold = high_confidence_threshold
        self._boundary_threshold = boundary_threshold

    def collect_errors(
        self,
        predictions: List[Dict[str, Any]],
        labels: Optional[List[str]] = None,
    ) -> ErrorCaseCollection:
        """
        Collect and analyze errors from predictions.

        Args:
            predictions: List of prediction dicts with keys:
                - sample_id: str
                - true_label: int or str
                - pred_label: int or str
                - confidence: float
                - probs: Optional[List[float]]
                - features: Optional[Dict]
                - metadata: Optional[Dict]
            labels: Label names

        Returns:
            ErrorCaseCollection
        """
        errors = []
        total = len(predictions)

        for pred in predictions:
            true_label = pred["true_label"]
            pred_label = pred["pred_label"]

            # Convert to label names if needed
            if labels and isinstance(true_label, int):
                true_label = labels[true_label] if true_label < len(labels) else str(true_label)
            else:
                true_label = str(true_label)

            if labels and isinstance(pred_label, int):
                pred_label = labels[pred_label] if pred_label < len(labels) else str(pred_label)
            else:
                pred_label = str(pred_label)

            # Skip correct predictions
            if true_label == pred_label:
                continue

            # Get true label probability
            true_label_prob = None
            if "probs" in pred and pred.get("probs"):
                probs = pred["probs"]
                true_idx = pred.get("true_label_idx", pred.get("true_label", 0))
                if isinstance(true_idx, int) and true_idx < len(probs):
                    true_label_prob = probs[true_idx]

            error = ErrorCase(
                sample_id=str(pred.get("sample_id", "")),
                true_label=true_label,
                pred_label=pred_label,
                confidence=pred.get("confidence", 0.0),
                true_label_prob=true_label_prob,
                features=pred.get("features"),
                metadata=pred.get("metadata", {}),
            )
            errors.append(error)

        # Detect patterns
        patterns = self._detect_patterns(errors)

        return ErrorCaseCollection(
            errors=errors,
            patterns=patterns,
            total_predictions=total,
        )

    def _detect_patterns(self, errors: List[ErrorCase]) -> List[ErrorPattern]:
        """Detect error patterns."""
        patterns = []

        if not errors:
            return patterns

        # 1. High-confidence errors pattern
        high_conf_errors = [e for e in errors if e.is_high_confidence_error]
        if high_conf_errors:
            patterns.append(ErrorPattern(
                pattern_type="high_confidence_error",
                description=f"{len(high_conf_errors)} predictions with high confidence (>80%) were wrong",
                affected_classes=list(set(e.true_label for e in high_conf_errors)),
                error_count=len(high_conf_errors),
                examples=[e.sample_id for e in high_conf_errors[:10]],
                severity="high",
                suggested_action="Review training data for these classes, check for label noise",
            ))

        # 2. Confusion pair patterns
        confusion_counts: Dict[Tuple[str, str], List[ErrorCase]] = defaultdict(list)
        for e in errors:
            key = (e.true_label, e.pred_label)
            confusion_counts[key].append(e)

        for (true_label, pred_label), cases in sorted(
            confusion_counts.items(), key=lambda x: len(x[1]), reverse=True
        )[:5]:  # Top 5 confusion pairs
            if len(cases) >= 3:  # Only report if multiple occurrences
                severity = "high" if len(cases) >= 10 else "medium" if len(cases) >= 5 else "low"
                patterns.append(ErrorPattern(
                    pattern_type="confusion_pair",
                    description=f"'{true_label}' frequently misclassified as '{pred_label}' ({len(cases)} times)",
                    affected_classes=[true_label, pred_label],
                    error_count=len(cases),
                    examples=[e.sample_id for e in cases[:5]],
                    severity=severity,
                    suggested_action=f"Add more training examples to distinguish '{true_label}' from '{pred_label}'",
                ))

        # 3. Class-specific error patterns
        class_error_counts: Dict[str, int] = defaultdict(int)
        for e in errors:
            class_error_counts[e.true_label] += 1

        for true_label, count in sorted(class_error_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            class_errors = [e for e in errors if e.true_label == true_label]
            error_rate = count / sum(1 for e in errors) if errors else 0
            if error_rate >= 0.15:  # Class contributes >15% of errors
                patterns.append(ErrorPattern(
                    pattern_type="class_specific",
                    description=f"Class '{true_label}' has high error rate ({count} errors, {error_rate:.1%} of total)",
                    affected_classes=[true_label],
                    error_count=count,
                    examples=[e.sample_id for e in class_errors[:5]],
                    severity="high" if error_rate >= 0.25 else "medium",
                    suggested_action=f"Review and augment training data for class '{true_label}'",
                ))

        # 4. Boundary cases pattern
        boundary_cases = [e for e in errors if e.is_boundary_case]
        if len(boundary_cases) >= len(errors) * 0.3:  # >30% are boundary cases
            patterns.append(ErrorPattern(
                pattern_type="boundary_dominated",
                description=f"{len(boundary_cases)} errors ({len(boundary_cases)/len(errors):.1%}) are boundary cases with low confidence",
                affected_classes=list(set(e.true_label for e in boundary_cases)),
                error_count=len(boundary_cases),
                examples=[e.sample_id for e in boundary_cases[:10]],
                severity="medium",
                suggested_action="Consider using confidence thresholds or adding 'uncertain' handling",
            ))

        return patterns

    def get_worst_errors(
        self,
        collection: ErrorCaseCollection,
        top_k: int = 20,
        sort_by: str = "confidence",
    ) -> List[ErrorCase]:
        """
        Get worst errors.

        Args:
            collection: Error case collection
            top_k: Number of errors to return
            sort_by: "confidence" (high conf errors first) or "gap" (largest gap first)

        Returns:
            List of worst error cases
        """
        errors = collection.errors.copy()

        if sort_by == "confidence":
            errors.sort(key=lambda x: x.confidence, reverse=True)
        elif sort_by == "gap":
            errors.sort(
                key=lambda x: x.confidence_gap if x.confidence_gap else 0,
                reverse=True
            )
        else:
            raise ValueError(f"Unknown sort_by: {sort_by}")

        return errors[:top_k]

    def summarize_by_class(
        self,
        collection: ErrorCaseCollection,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Summarize errors by class.

        Args:
            collection: Error case collection

        Returns:
            Dict mapping class name to error summary
        """
        summaries: Dict[str, Dict[str, Any]] = {}

        # Group by true label
        by_true_label: Dict[str, List[ErrorCase]] = defaultdict(list)
        for error in collection.errors:
            by_true_label[error.true_label].append(error)

        for label, errors in by_true_label.items():
            confused_with = defaultdict(int)
            for e in errors:
                confused_with[e.pred_label] += 1

            avg_confidence = sum(e.confidence for e in errors) / len(errors)
            high_conf_count = sum(1 for e in errors if e.is_high_confidence_error)

            summaries[label] = {
                "error_count": len(errors),
                "avg_confidence": round(avg_confidence, 4),
                "high_confidence_errors": high_conf_count,
                "top_confusions": dict(sorted(confused_with.items(), key=lambda x: x[1], reverse=True)[:3]),
            }

        return summaries
