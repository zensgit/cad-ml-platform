"""
Experiment Tracking Module.

Provides unified experiment management infrastructure:
- Experiment parameter recording
- Metrics tracking and visualization
- Model version management
- Experiment comparison analysis
"""

from __future__ import annotations

from src.ml.experiment.tracker import (
    ExperimentTracker,
    get_tracker,
    set_tracker,
)
from src.ml.experiment.run import (
    Run,
    RunStatus,
    RunInfo,
)
from src.ml.experiment.metrics import (
    MetricsLogger,
    Metric,
    MetricType,
)
from src.ml.experiment.artifacts import (
    ArtifactStore,
    Artifact,
    ArtifactType,
)
from src.ml.experiment.registry import (
    ModelRegistry,
    ModelInfo,
    ModelStage,
)
from src.ml.experiment.comparison import (
    ExperimentComparison,
    ComparisonReport,
)

__all__ = [
    # Tracker
    "ExperimentTracker",
    "get_tracker",
    "set_tracker",
    # Run
    "Run",
    "RunStatus",
    "RunInfo",
    # Metrics
    "MetricsLogger",
    "Metric",
    "MetricType",
    # Artifacts
    "ArtifactStore",
    "Artifact",
    "ArtifactType",
    # Registry
    "ModelRegistry",
    "ModelInfo",
    "ModelStage",
    # Comparison
    "ExperimentComparison",
    "ComparisonReport",
]
