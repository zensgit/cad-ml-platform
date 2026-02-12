"""
Model Evaluator - Main interface for model evaluation.

Provides:
- Unified evaluation interface
- Integration with metrics, confusion, and error analysis
- Report generation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.ml.evaluation.metrics import MetricsCalculator, MetricsReport
from src.ml.evaluation.confusion import ConfusionAnalyzer, ConfusionAnalysis
from src.ml.evaluation.error_analysis import ErrorAnalyzer, ErrorCaseCollection
from src.ml.evaluation.reporter import EvaluationReporter, ReportConfig, ReportFormat

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A single prediction result."""
    sample_id: str
    true_label: int
    pred_label: int
    confidence: float
    probs: Optional[List[float]] = None
    features: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sample_id": self.sample_id,
            "true_label": self.true_label,
            "pred_label": self.pred_label,
            "confidence": self.confidence,
            "probs": self.probs,
            "features": self.features,
            "metadata": self.metadata,
        }


@dataclass
class EvalConfig:
    """Configuration for model evaluation."""
    compute_confusion: bool = True
    compute_errors: bool = True
    generate_report: bool = True
    report_format: ReportFormat = ReportFormat.MARKDOWN
    report_config: Optional[ReportConfig] = None
    labels: Optional[List[str]] = None
    top_k_accuracy: Optional[List[int]] = None  # e.g., [3, 5] for top-3 and top-5


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    metrics: MetricsReport
    confusion: Optional[ConfusionAnalysis] = None
    errors: Optional[ErrorCaseCollection] = None
    top_k_accuracy: Optional[Dict[int, float]] = None
    evaluation_time: float = 0.0
    model_name: Optional[str] = None
    dataset_name: Optional[str] = None

    @property
    def accuracy(self) -> float:
        return self.metrics.accuracy

    @property
    def f1_macro(self) -> float:
        return self.metrics.f1_macro

    @property
    def f1_weighted(self) -> float:
        return self.metrics.f1_weighted

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "metrics": self.metrics.to_dict(),
            "evaluation_time": round(self.evaluation_time, 4),
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
        }

        if self.confusion:
            result["confusion"] = self.confusion.to_dict()

        if self.errors:
            result["errors"] = self.errors.to_summary_dict()

        if self.top_k_accuracy:
            result["top_k_accuracy"] = self.top_k_accuracy

        return result

    def to_report(
        self,
        format: ReportFormat = ReportFormat.MARKDOWN,
        config: Optional[ReportConfig] = None,
    ) -> str:
        """Generate evaluation report."""
        reporter = EvaluationReporter(config)
        return reporter.generate(
            metrics=self.metrics,
            confusion=self.confusion,
            errors=self.errors,
            format=format,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
            extra_info={"evaluation_time_seconds": self.evaluation_time},
        )

    def save_report(
        self,
        path: Path,
        format: ReportFormat = ReportFormat.MARKDOWN,
        config: Optional[ReportConfig] = None,
    ) -> None:
        """Save report to file."""
        report = self.to_report(format, config)
        reporter = EvaluationReporter(config)
        reporter.save(report, path, format)


class ModelEvaluator:
    """
    Unified model evaluation interface.

    Provides:
    - Metrics calculation
    - Confusion matrix analysis
    - Error case analysis
    - Report generation
    """

    def __init__(self, config: Optional[EvalConfig] = None):
        """
        Initialize model evaluator.

        Args:
            config: Evaluation configuration
        """
        self._config = config or EvalConfig()
        self._metrics_calc = MetricsCalculator()
        self._confusion_analyzer = ConfusionAnalyzer()
        self._error_analyzer = ErrorAnalyzer()
        self._reporter = EvaluationReporter(self._config.report_config)

    def evaluate(
        self,
        predictions: List[Prediction],
        labels: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate model predictions.

        Args:
            predictions: List of Prediction objects
            labels: Label names
            model_name: Name of the model
            dataset_name: Name of the dataset

        Returns:
            EvaluationResult
        """
        start_time = time.time()

        labels = labels or self._config.labels

        # Extract true and predicted labels
        y_true = [p.true_label for p in predictions]
        y_pred = [p.pred_label for p in predictions]

        # Calculate metrics
        metrics = self._metrics_calc.calculate(y_true, y_pred, labels)

        # Confusion analysis
        confusion = None
        if self._config.compute_confusion:
            confusion = self._confusion_analyzer.analyze(y_true, y_pred, labels)

        # Error analysis
        errors = None
        if self._config.compute_errors:
            pred_dicts = [p.to_dict() for p in predictions]
            errors = self._error_analyzer.collect_errors(pred_dicts, labels)

        # Top-k accuracy
        top_k_accuracy = None
        if self._config.top_k_accuracy:
            top_k_accuracy = {}
            for k in self._config.top_k_accuracy:
                probs_list = [p.probs for p in predictions if p.probs]
                if probs_list:
                    top_k_accuracy[k] = self._metrics_calc.calculate_top_k_accuracy(
                        y_true, probs_list, k
                    )

        evaluation_time = time.time() - start_time

        logger.info(
            f"Evaluation completed: accuracy={metrics.accuracy:.4f}, "
            f"f1_macro={metrics.f1_macro:.4f}, time={evaluation_time:.2f}s"
        )

        return EvaluationResult(
            metrics=metrics,
            confusion=confusion,
            errors=errors,
            top_k_accuracy=top_k_accuracy,
            evaluation_time=evaluation_time,
            model_name=model_name,
            dataset_name=dataset_name,
        )

    def evaluate_from_arrays(
        self,
        y_true: List[int],
        y_pred: List[int],
        confidences: Optional[List[float]] = None,
        probs: Optional[List[List[float]]] = None,
        sample_ids: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate from arrays of predictions.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            confidences: Prediction confidences
            probs: Prediction probabilities
            sample_ids: Sample identifiers
            labels: Label names
            model_name: Name of the model
            dataset_name: Name of the dataset

        Returns:
            EvaluationResult
        """
        n = len(y_true)

        if confidences is None:
            if probs:
                confidences = [max(p) for p in probs]
            else:
                confidences = [1.0] * n

        if sample_ids is None:
            sample_ids = [str(i) for i in range(n)]

        predictions = []
        for i in range(n):
            predictions.append(Prediction(
                sample_id=sample_ids[i],
                true_label=y_true[i],
                pred_label=y_pred[i],
                confidence=confidences[i],
                probs=probs[i] if probs else None,
            ))

        return self.evaluate(predictions, labels, model_name, dataset_name)

    def evaluate_model(
        self,
        model: Any,
        dataset: Any,
        predict_fn: Optional[Callable] = None,
        labels: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        batch_size: int = 32,
    ) -> EvaluationResult:
        """
        Evaluate a model on a dataset.

        Args:
            model: Model to evaluate
            dataset: Dataset with (sample, label) items
            predict_fn: Custom prediction function
            labels: Label names
            model_name: Name of the model
            dataset_name: Name of the dataset
            batch_size: Batch size for inference

        Returns:
            EvaluationResult
        """
        predictions = []

        # Default prediction function
        if predict_fn is None:
            def predict_fn(model, batch):
                import torch
                model.eval()
                with torch.no_grad():
                    outputs = model(batch)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    probs = torch.softmax(outputs, dim=-1)
                    preds = torch.argmax(probs, dim=-1)
                    confs = torch.max(probs, dim=-1).values
                    return preds.tolist(), confs.tolist(), probs.tolist()

        # Iterate through dataset
        for i, item in enumerate(dataset):
            if isinstance(item, tuple):
                sample, true_label = item
            else:
                sample = item
                true_label = getattr(item, "y", 0)

            # Get prediction
            try:
                pred_label, confidence, probs = predict_fn(model, sample)
                if isinstance(pred_label, list):
                    pred_label = pred_label[0]
                if isinstance(confidence, list):
                    confidence = confidence[0]
                if isinstance(probs, list) and probs and isinstance(probs[0], list):
                    probs = probs[0]
            except Exception as e:
                logger.warning(f"Prediction failed for sample {i}: {e}")
                continue

            predictions.append(Prediction(
                sample_id=str(i),
                true_label=int(true_label) if not isinstance(true_label, int) else true_label,
                pred_label=int(pred_label),
                confidence=float(confidence),
                probs=probs if isinstance(probs, list) else None,
            ))

        return self.evaluate(predictions, labels, model_name, dataset_name)

    def compare_models(
        self,
        results: List[EvaluationResult],
        metric: str = "f1_macro",
    ) -> Dict[str, Any]:
        """
        Compare multiple model evaluation results.

        Args:
            results: List of evaluation results
            metric: Metric to compare

        Returns:
            Comparison summary
        """
        comparison = {
            "models": [],
            "best_model": None,
            "metric": metric,
        }

        best_value = None
        best_model = None

        for result in results:
            model_data = {
                "name": result.model_name or "unnamed",
                "accuracy": result.accuracy,
                "f1_macro": result.f1_macro,
                "f1_weighted": result.f1_weighted,
                "evaluation_time": result.evaluation_time,
            }

            # Get requested metric
            if hasattr(result.metrics, metric):
                value = getattr(result.metrics, metric)
            elif metric in ("accuracy", "f1_macro", "f1_weighted"):
                value = getattr(result, metric)
            else:
                value = None

            model_data["metric_value"] = value

            if value is not None:
                if best_value is None or value > best_value:
                    best_value = value
                    best_model = result.model_name

            comparison["models"].append(model_data)

        comparison["best_model"] = best_model
        comparison["best_value"] = best_value

        return comparison
