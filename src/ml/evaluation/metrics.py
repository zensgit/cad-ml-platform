"""
Metrics calculation for model evaluation.

Provides:
- Classification metrics (accuracy, precision, recall, F1)
- Per-class metrics
- Aggregated statistics
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ClassMetrics:
    """Metrics for a single class."""
    label: str
    label_id: int
    precision: float
    recall: float
    f1: float
    support: int  # Number of true instances
    predicted_count: int  # Number of predictions for this class
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def accuracy(self) -> float:
        """Class-specific accuracy (TP / (TP + FP + FN))."""
        total = self.true_positives + self.false_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "label": self.label,
            "label_id": self.label_id,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "support": self.support,
            "predicted_count": self.predicted_count,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


@dataclass
class MetricsReport:
    """Complete metrics report."""
    accuracy: float
    precision_macro: float
    precision_micro: float
    precision_weighted: float
    recall_macro: float
    recall_micro: float
    recall_weighted: float
    f1_macro: float
    f1_micro: float
    f1_weighted: float
    per_class: Dict[str, ClassMetrics]
    total_samples: int
    num_classes: int
    labels: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": {
                "macro": round(self.precision_macro, 4),
                "micro": round(self.precision_micro, 4),
                "weighted": round(self.precision_weighted, 4),
            },
            "recall": {
                "macro": round(self.recall_macro, 4),
                "micro": round(self.recall_micro, 4),
                "weighted": round(self.recall_weighted, 4),
            },
            "f1": {
                "macro": round(self.f1_macro, 4),
                "micro": round(self.f1_micro, 4),
                "weighted": round(self.f1_weighted, 4),
            },
            "total_samples": self.total_samples,
            "num_classes": self.num_classes,
            "per_class": {
                label: metrics.to_dict()
                for label, metrics in self.per_class.items()
            },
        }

    def get_worst_classes(self, metric: str = "f1", top_k: int = 5) -> List[Tuple[str, float]]:
        """Get classes with worst performance."""
        scores = []
        for label, cm in self.per_class.items():
            if metric == "f1":
                scores.append((label, cm.f1))
            elif metric == "precision":
                scores.append((label, cm.precision))
            elif metric == "recall":
                scores.append((label, cm.recall))
            else:
                raise ValueError(f"Unknown metric: {metric}")

        scores.sort(key=lambda x: x[1])
        return scores[:top_k]

    def get_best_classes(self, metric: str = "f1", top_k: int = 5) -> List[Tuple[str, float]]:
        """Get classes with best performance."""
        scores = []
        for label, cm in self.per_class.items():
            if metric == "f1":
                scores.append((label, cm.f1))
            elif metric == "precision":
                scores.append((label, cm.precision))
            elif metric == "recall":
                scores.append((label, cm.recall))
            else:
                raise ValueError(f"Unknown metric: {metric}")

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class MetricsCalculator:
    """
    Calculator for classification metrics.

    Supports:
    - Multi-class classification
    - Micro/macro/weighted averaging
    - Per-class metrics
    """

    def __init__(self, labels: Optional[List[str]] = None):
        """
        Initialize metrics calculator.

        Args:
            labels: List of class labels (auto-detected if not provided)
        """
        self._labels = labels

    def calculate(
        self,
        y_true: List[int],
        y_pred: List[int],
        labels: Optional[List[str]] = None,
    ) -> MetricsReport:
        """
        Calculate all metrics.

        Args:
            y_true: True labels (as integers)
            y_pred: Predicted labels (as integers)
            labels: Label names (optional)

        Returns:
            MetricsReport
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        if not y_true:
            raise ValueError("Empty predictions")

        # Determine labels
        all_labels = sorted(set(y_true) | set(y_pred))
        num_classes = len(all_labels)

        if labels is None:
            labels = self._labels or [str(i) for i in all_labels]

        label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

        # Calculate per-class metrics
        per_class = {}
        total_tp = 0
        total_fp = 0
        total_fn = 0

        true_counts = Counter(y_true)
        pred_counts = Counter(y_pred)

        for label_id in all_labels:
            label_name = labels[label_id] if label_id < len(labels) else str(label_id)

            # Count TP, FP, FN
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == label_id and p == label_id)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != label_id and p == label_id)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == label_id and p != label_id)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class[label_name] = ClassMetrics(
                label=label_name,
                label_id=label_id,
                precision=precision,
                recall=recall,
                f1=f1,
                support=true_counts.get(label_id, 0),
                predicted_count=pred_counts.get(label_id, 0),
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
            )

        # Calculate aggregated metrics
        accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

        # Micro averaging (global TP, FP, FN)
        precision_micro = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall_micro = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

        # Macro averaging (average of per-class metrics)
        precision_macro = sum(cm.precision for cm in per_class.values()) / num_classes if num_classes > 0 else 0.0
        recall_macro = sum(cm.recall for cm in per_class.values()) / num_classes if num_classes > 0 else 0.0
        f1_macro = sum(cm.f1 for cm in per_class.values()) / num_classes if num_classes > 0 else 0.0

        # Weighted averaging (weighted by support)
        total_support = sum(cm.support for cm in per_class.values())
        precision_weighted = sum(cm.precision * cm.support for cm in per_class.values()) / total_support if total_support > 0 else 0.0
        recall_weighted = sum(cm.recall * cm.support for cm in per_class.values()) / total_support if total_support > 0 else 0.0
        f1_weighted = sum(cm.f1 * cm.support for cm in per_class.values()) / total_support if total_support > 0 else 0.0

        return MetricsReport(
            accuracy=accuracy,
            precision_macro=precision_macro,
            precision_micro=precision_micro,
            precision_weighted=precision_weighted,
            recall_macro=recall_macro,
            recall_micro=recall_micro,
            recall_weighted=recall_weighted,
            f1_macro=f1_macro,
            f1_micro=f1_micro,
            f1_weighted=f1_weighted,
            per_class=per_class,
            total_samples=len(y_true),
            num_classes=num_classes,
            labels=labels,
        )

    def calculate_from_predictions(
        self,
        predictions: List[Tuple[Any, int, int]],  # (sample_id, true_label, pred_label)
        labels: Optional[List[str]] = None,
    ) -> MetricsReport:
        """
        Calculate metrics from prediction tuples.

        Args:
            predictions: List of (sample_id, true_label, pred_label)
            labels: Label names

        Returns:
            MetricsReport
        """
        y_true = [p[1] for p in predictions]
        y_pred = [p[2] for p in predictions]
        return self.calculate(y_true, y_pred, labels)

    def calculate_top_k_accuracy(
        self,
        y_true: List[int],
        y_pred_probs: List[List[float]],
        k: int = 5,
    ) -> float:
        """
        Calculate top-k accuracy.

        Args:
            y_true: True labels
            y_pred_probs: Predicted probabilities for each class
            k: Number of top predictions to consider

        Returns:
            Top-k accuracy
        """
        correct = 0
        for true_label, probs in zip(y_true, y_pred_probs):
            top_k_preds = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
            if true_label in top_k_preds:
                correct += 1

        return correct / len(y_true) if y_true else 0.0
