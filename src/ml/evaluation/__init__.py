"""
Model Evaluation Module.

Provides standardized model evaluation:
- Multi-dimensional metrics calculation
- Confusion matrix analysis
- Error case analysis
- Automated report generation
"""

from __future__ import annotations

from src.ml.evaluation.evaluator import (
    ModelEvaluator,
    EvalConfig,
    EvaluationResult,
    Prediction,
)
from src.ml.evaluation.metrics import (
    MetricsCalculator,
    ClassMetrics,
    MetricsReport,
)
from src.ml.evaluation.confusion import (
    ConfusionAnalyzer,
    ConfusionAnalysis,
    ClassConfusion,
)
from src.ml.evaluation.error_analysis import (
    ErrorAnalyzer,
    ErrorCase,
    ErrorCaseCollection,
    ErrorPattern,
)
from src.ml.evaluation.reporter import (
    EvaluationReporter,
    ReportConfig,
    ReportFormat,
)

__all__ = [
    # Evaluator
    "ModelEvaluator",
    "EvalConfig",
    "EvaluationResult",
    "Prediction",
    # Metrics
    "MetricsCalculator",
    "ClassMetrics",
    "MetricsReport",
    # Confusion
    "ConfusionAnalyzer",
    "ConfusionAnalysis",
    "ClassConfusion",
    # Error Analysis
    "ErrorAnalyzer",
    "ErrorCase",
    "ErrorCaseCollection",
    "ErrorPattern",
    # Reporter
    "EvaluationReporter",
    "ReportConfig",
    "ReportFormat",
]
