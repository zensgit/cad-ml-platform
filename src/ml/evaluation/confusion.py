"""
Confusion matrix analysis.

Provides:
- Confusion matrix generation
- Class-pair analysis
- Common confusion patterns
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClassConfusion:
    """Confusion data for a single class."""
    label: str
    label_id: int
    total_true: int
    total_predicted: int
    correct: int
    confused_with: Dict[str, int]  # label -> count
    confused_from: Dict[str, int]  # label -> count (other classes predicted as this)

    @property
    def error_rate(self) -> float:
        """Error rate for this class."""
        if self.total_true == 0:
            return 0.0
        return 1.0 - (self.correct / self.total_true)

    @property
    def top_confusions(self) -> List[Tuple[str, int]]:
        """Top confused classes (this class predicted as others)."""
        return sorted(self.confused_with.items(), key=lambda x: x[1], reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "label": self.label,
            "label_id": self.label_id,
            "total_true": self.total_true,
            "total_predicted": self.total_predicted,
            "correct": self.correct,
            "error_rate": round(self.error_rate, 4),
            "confused_with": self.confused_with,
            "confused_from": self.confused_from,
        }


@dataclass
class ConfusionPair:
    """A pair of commonly confused classes."""
    class_a: str
    class_b: str
    a_as_b: int  # class_a predicted as class_b
    b_as_a: int  # class_b predicted as class_a
    total_confusion: int

    @property
    def symmetry_ratio(self) -> float:
        """Ratio of symmetric confusion (both directions)."""
        if self.total_confusion == 0:
            return 0.0
        return min(self.a_as_b, self.b_as_a) / max(self.a_as_b, self.b_as_a)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "class_a": self.class_a,
            "class_b": self.class_b,
            "a_as_b": self.a_as_b,
            "b_as_a": self.b_as_a,
            "total_confusion": self.total_confusion,
            "symmetry_ratio": round(self.symmetry_ratio, 4),
        }


@dataclass
class ConfusionAnalysis:
    """Complete confusion analysis."""
    matrix: np.ndarray
    labels: List[str]
    per_class: Dict[str, ClassConfusion]
    top_confusion_pairs: List[ConfusionPair]
    total_samples: int
    total_errors: int

    @property
    def overall_accuracy(self) -> float:
        """Overall accuracy from confusion matrix."""
        return np.trace(self.matrix) / self.total_samples if self.total_samples > 0 else 0.0

    def get_confusion_rate(self, true_label: str, pred_label: str) -> float:
        """Get confusion rate between two classes."""
        try:
            true_idx = self.labels.index(true_label)
            pred_idx = self.labels.index(pred_label)
            total_true = self.matrix[true_idx].sum()
            if total_true == 0:
                return 0.0
            return self.matrix[true_idx, pred_idx] / total_true
        except (ValueError, IndexError):
            return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "matrix": self.matrix.tolist(),
            "labels": self.labels,
            "overall_accuracy": round(self.overall_accuracy, 4),
            "total_samples": self.total_samples,
            "total_errors": self.total_errors,
            "per_class": {
                label: cc.to_dict()
                for label, cc in self.per_class.items()
            },
            "top_confusion_pairs": [cp.to_dict() for cp in self.top_confusion_pairs],
        }

    def to_markdown_table(self) -> str:
        """Generate markdown confusion matrix table."""
        lines = []

        # Header
        header = "| Actual \\ Pred | " + " | ".join(self.labels) + " |"
        separator = "|" + "|".join(["---"] * (len(self.labels) + 1)) + "|"
        lines.extend([header, separator])

        # Rows
        for i, label in enumerate(self.labels):
            row_values = [str(int(self.matrix[i, j])) for j in range(len(self.labels))]
            row = f"| **{label}** | " + " | ".join(row_values) + " |"
            lines.append(row)

        return "\n".join(lines)


class ConfusionAnalyzer:
    """
    Analyzer for confusion matrices.

    Provides:
    - Confusion matrix generation
    - Per-class confusion analysis
    - Common confusion pair identification
    """

    def analyze(
        self,
        y_true: List[int],
        y_pred: List[int],
        labels: Optional[List[str]] = None,
    ) -> ConfusionAnalysis:
        """
        Analyze predictions and generate confusion analysis.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names

        Returns:
            ConfusionAnalysis
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        # Determine number of classes
        all_labels = sorted(set(y_true) | set(y_pred))
        num_classes = len(all_labels)

        if labels is None:
            labels = [str(i) for i in all_labels]

        # Build confusion matrix
        matrix = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            t_idx = all_labels.index(t)
            p_idx = all_labels.index(p)
            matrix[t_idx, p_idx] += 1

        # Per-class analysis
        per_class = {}
        for i, label_id in enumerate(all_labels):
            label = labels[i] if i < len(labels) else str(label_id)

            # Confused with (this class predicted as others)
            confused_with = {}
            for j, other_id in enumerate(all_labels):
                if i != j and matrix[i, j] > 0:
                    other_label = labels[j] if j < len(labels) else str(other_id)
                    confused_with[other_label] = int(matrix[i, j])

            # Confused from (other classes predicted as this)
            confused_from = {}
            for j, other_id in enumerate(all_labels):
                if i != j and matrix[j, i] > 0:
                    other_label = labels[j] if j < len(labels) else str(other_id)
                    confused_from[other_label] = int(matrix[j, i])

            per_class[label] = ClassConfusion(
                label=label,
                label_id=label_id,
                total_true=int(matrix[i].sum()),
                total_predicted=int(matrix[:, i].sum()),
                correct=int(matrix[i, i]),
                confused_with=confused_with,
                confused_from=confused_from,
            )

        # Find top confusion pairs
        confusion_pairs = []
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                a_as_b = int(matrix[i, j])
                b_as_a = int(matrix[j, i])
                total = a_as_b + b_as_a
                if total > 0:
                    label_a = labels[i] if i < len(labels) else str(all_labels[i])
                    label_b = labels[j] if j < len(labels) else str(all_labels[j])
                    confusion_pairs.append(ConfusionPair(
                        class_a=label_a,
                        class_b=label_b,
                        a_as_b=a_as_b,
                        b_as_a=b_as_a,
                        total_confusion=total,
                    ))

        # Sort by total confusion
        confusion_pairs.sort(key=lambda x: x.total_confusion, reverse=True)

        total_samples = len(y_true)
        total_errors = sum(1 for t, p in zip(y_true, y_pred) if t != p)

        return ConfusionAnalysis(
            matrix=matrix,
            labels=labels,
            per_class=per_class,
            top_confusion_pairs=confusion_pairs[:10],  # Top 10
            total_samples=total_samples,
            total_errors=total_errors,
        )

    def normalize_matrix(
        self,
        matrix: np.ndarray,
        mode: str = "true",
    ) -> np.ndarray:
        """
        Normalize confusion matrix.

        Args:
            matrix: Confusion matrix
            mode: "true" (normalize by true labels), "pred" (by predictions), "all"

        Returns:
            Normalized matrix
        """
        if mode == "true":
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            return matrix / row_sums
        elif mode == "pred":
            col_sums = matrix.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1
            return matrix / col_sums
        elif mode == "all":
            total = matrix.sum()
            return matrix / total if total > 0 else matrix
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

    def find_systematic_confusions(
        self,
        analysis: ConfusionAnalysis,
        threshold: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Find systematic confusion patterns.

        Args:
            analysis: Confusion analysis
            threshold: Minimum confusion rate to report

        Returns:
            List of systematic confusion patterns
        """
        patterns = []
        normalized = self.normalize_matrix(analysis.matrix, mode="true")

        for i, true_label in enumerate(analysis.labels):
            for j, pred_label in enumerate(analysis.labels):
                if i == j:
                    continue

                confusion_rate = normalized[i, j]
                if confusion_rate >= threshold:
                    patterns.append({
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "confusion_rate": float(confusion_rate),
                        "count": int(analysis.matrix[i, j]),
                        "severity": "high" if confusion_rate >= 0.3 else "medium",
                    })

        patterns.sort(key=lambda x: x["confusion_rate"], reverse=True)
        return patterns
