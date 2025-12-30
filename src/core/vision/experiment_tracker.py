"""Experiment Tracker Module for Vision System.

This module provides ML experiment tracking capabilities including:
- Experiment creation and organization
- Run management and parameter logging
- Metrics tracking and visualization
- Artifact storage and versioning
- Experiment comparison and analysis
- Collaboration and sharing features

Phase 18: Advanced ML Pipeline & AutoML
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from .base import VisionDescription, VisionProvider

# ========================
# Enums
# ========================


class RunStatus(str, Enum):
    """Experiment run status."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


class MetricGoal(str, Enum):
    """Optimization goal for metrics."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ArtifactType(str, Enum):
    """Types of artifacts."""

    MODEL = "model"
    DATASET = "dataset"
    PLOT = "plot"
    CONFIG = "config"
    LOG = "log"
    CHECKPOINT = "checkpoint"
    OTHER = "other"


class ComparisonMode(str, Enum):
    """Experiment comparison modes."""

    TABLE = "table"
    CHART = "chart"
    PARALLEL = "parallel"
    SCATTER = "scatter"


class TagType(str, Enum):
    """Tag types for experiments."""

    USER = "user"
    SYSTEM = "system"
    AUTO = "auto"


# ========================
# Dataclasses
# ========================


@dataclass
class Experiment:
    """An experiment grouping multiple runs."""

    experiment_id: str
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    artifact_location: Optional[str] = None
    lifecycle_stage: str = "active"


@dataclass
class Run:
    """An individual experiment run."""

    run_id: str
    experiment_id: str
    name: str = ""
    status: RunStatus = RunStatus.CREATED
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    source_name: str = ""
    source_version: str = ""
    user_id: str = "system"
    tags: Dict[str, str] = field(default_factory=dict)
    notes: str = ""


@dataclass
class Parameter:
    """A logged parameter."""

    key: str
    value: str
    run_id: str
    logged_at: datetime = field(default_factory=datetime.now)


@dataclass
class Metric:
    """A logged metric."""

    key: str
    value: float
    run_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    step: int = 0
    goal: MetricGoal = MetricGoal.MAXIMIZE


@dataclass
class MetricHistory:
    """History of a metric across steps."""

    key: str
    run_id: str
    values: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)


@dataclass
class Artifact:
    """An experiment artifact."""

    artifact_id: str
    run_id: str
    name: str
    artifact_type: ArtifactType
    path: str
    size_bytes: int = 0
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ComparisonResult:
    """Result of comparing experiments/runs."""

    comparison_id: str
    run_ids: List[str]
    parameters: Dict[str, Dict[str, str]]  # run_id -> param_key -> value
    metrics: Dict[str, Dict[str, float]]  # run_id -> metric_key -> value
    best_run: Optional[str] = None
    ranking: List[str] = field(default_factory=list)
    computed_at: datetime = field(default_factory=datetime.now)


@dataclass
class SearchQuery:
    """Query for searching experiments/runs."""

    filter_string: Optional[str] = None
    metric_filters: Dict[str, Tuple[str, float]] = field(
        default_factory=dict
    )  # metric -> (op, value)
    param_filters: Dict[str, str] = field(default_factory=dict)
    tag_filters: Dict[str, str] = field(default_factory=dict)
    order_by: Optional[str] = None
    ascending: bool = True
    max_results: int = 100


# ========================
# Core Classes
# ========================


class ExperimentStore:
    """Store for experiments and runs."""

    def __init__(self, storage_path: Optional[str] = None):
        self._storage_path = Path(storage_path) if storage_path else None
        self._experiments: Dict[str, Experiment] = {}
        self._runs: Dict[str, Run] = {}
        self._parameters: Dict[str, List[Parameter]] = defaultdict(list)
        self._metrics: Dict[str, List[Metric]] = defaultdict(list)
        self._artifacts: Dict[str, List[Artifact]] = defaultdict(list)
        self._lock = threading.RLock()

    def create_experiment(self, experiment: Experiment) -> Experiment:
        """Create a new experiment."""
        with self._lock:
            self._experiments[experiment.experiment_id] = experiment
        return experiment

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        return self._experiments.get(experiment_id)

    def get_experiment_by_name(self, name: str) -> Optional[Experiment]:
        """Get an experiment by name."""
        for exp in self._experiments.values():
            if exp.name == name:
                return exp
        return None

    def list_experiments(
        self,
        lifecycle_stage: str = "active",
    ) -> List[Experiment]:
        """List experiments."""
        return [e for e in self._experiments.values() if e.lifecycle_stage == lifecycle_stage]

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        with self._lock:
            if experiment_id in self._experiments:
                self._experiments[experiment_id].lifecycle_stage = "deleted"
                return True
        return False

    def create_run(self, run: Run) -> Run:
        """Create a new run."""
        with self._lock:
            self._runs[run.run_id] = run
        return run

    def get_run(self, run_id: str) -> Optional[Run]:
        """Get a run by ID."""
        return self._runs.get(run_id)

    def list_runs(
        self,
        experiment_id: str,
        status: Optional[RunStatus] = None,
    ) -> List[Run]:
        """List runs for an experiment."""
        runs = [r for r in self._runs.values() if r.experiment_id == experiment_id]
        if status:
            runs = [r for r in runs if r.status == status]
        return sorted(runs, key=lambda r: r.start_time or datetime.min, reverse=True)

    def update_run(self, run: Run) -> Run:
        """Update a run."""
        with self._lock:
            self._runs[run.run_id] = run
        return run

    def log_parameter(self, param: Parameter) -> None:
        """Log a parameter."""
        with self._lock:
            self._parameters[param.run_id].append(param)

    def log_metric(self, metric: Metric) -> None:
        """Log a metric."""
        with self._lock:
            self._metrics[metric.run_id].append(metric)

    def get_parameters(self, run_id: str) -> List[Parameter]:
        """Get parameters for a run."""
        return self._parameters.get(run_id, [])

    def get_metrics(self, run_id: str) -> List[Metric]:
        """Get metrics for a run."""
        return self._metrics.get(run_id, [])

    def get_metric_history(self, run_id: str, key: str) -> MetricHistory:
        """Get metric history for a run."""
        metrics = [m for m in self._metrics.get(run_id, []) if m.key == key]
        metrics.sort(key=lambda m: m.step)

        return MetricHistory(
            key=key,
            run_id=run_id,
            values=[m.value for m in metrics],
            steps=[m.step for m in metrics],
            timestamps=[m.timestamp for m in metrics],
        )

    def log_artifact(self, artifact: Artifact) -> None:
        """Log an artifact."""
        with self._lock:
            self._artifacts[artifact.run_id].append(artifact)

    def get_artifacts(self, run_id: str) -> List[Artifact]:
        """Get artifacts for a run."""
        return self._artifacts.get(run_id, [])


class RunContext:
    """Context manager for experiment runs."""

    def __init__(
        self,
        store: ExperimentStore,
        experiment_id: str,
        run_name: str = "",
    ):
        self._store = store
        self._experiment_id = experiment_id
        self._run_name = run_name
        self._run: Optional[Run] = None
        self._step = 0

    def __enter__(self) -> RunContext:
        """Start the run."""
        run_id = hashlib.sha256(f"{self._experiment_id}:{time.time()}".encode()).hexdigest()[:12]

        self._run = Run(
            run_id=run_id,
            experiment_id=self._experiment_id,
            name=self._run_name,
            status=RunStatus.RUNNING,
            start_time=datetime.now(),
        )

        self._store.create_run(self._run)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End the run."""
        if self._run:
            self._run.end_time = datetime.now()
            if exc_type is not None:
                self._run.status = RunStatus.FAILED
            else:
                self._run.status = RunStatus.COMPLETED
            self._store.update_run(self._run)

    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter."""
        if self._run:
            param = Parameter(
                key=key,
                value=str(value),
                run_id=self._run.run_id,
            )
            self._store.log_parameter(param)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        for key, value in params.items():
            self.log_param(key, value)

    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """Log a metric."""
        if self._run:
            if step is None:
                step = self._step
                self._step += 1

            metric = Metric(
                key=key,
                value=value,
                run_id=self._run.run_id,
                step=step,
            )
            self._store.log_metric(metric)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_artifact(
        self,
        name: str,
        path: str,
        artifact_type: ArtifactType = ArtifactType.OTHER,
    ) -> None:
        """Log an artifact."""
        if self._run:
            artifact_id = hashlib.sha256(
                f"{self._run.run_id}:{name}:{time.time()}".encode()
            ).hexdigest()[:8]

            artifact = Artifact(
                artifact_id=artifact_id,
                run_id=self._run.run_id,
                name=name,
                artifact_type=artifact_type,
                path=path,
            )
            self._store.log_artifact(artifact)

    def set_tag(self, key: str, value: str) -> None:
        """Set a run tag."""
        if self._run:
            self._run.tags[key] = value
            self._store.update_run(self._run)

    @property
    def run_id(self) -> Optional[str]:
        """Get the current run ID."""
        return self._run.run_id if self._run else None


class ExperimentTracker:
    """Main experiment tracking interface."""

    def __init__(self, storage_path: Optional[str] = None):
        self._store = ExperimentStore(storage_path)
        self._active_run: Optional[RunContext] = None

    def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Experiment:
        """Create a new experiment."""
        experiment_id = hashlib.sha256(f"{name}:{time.time()}".encode()).hexdigest()[:12]

        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            tags=tags or [],
        )

        return self._store.create_experiment(experiment)

    def get_experiment(
        self,
        experiment_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Optional[Experiment]:
        """Get an experiment by ID or name."""
        if experiment_id:
            return self._store.get_experiment(experiment_id)
        elif name:
            return self._store.get_experiment_by_name(name)
        return None

    def list_experiments(self) -> List[Experiment]:
        """List all active experiments."""
        return self._store.list_experiments()

    def start_run(
        self,
        experiment_id: str,
        run_name: str = "",
    ) -> RunContext:
        """Start a new run."""
        context = RunContext(
            store=self._store,
            experiment_id=experiment_id,
            run_name=run_name,
        )
        self._active_run = context
        return context

    def get_run(self, run_id: str) -> Optional[Run]:
        """Get a run by ID."""
        return self._store.get_run(run_id)

    def list_runs(
        self,
        experiment_id: str,
        status: Optional[RunStatus] = None,
    ) -> List[Run]:
        """List runs for an experiment."""
        return self._store.list_runs(experiment_id, status)

    def search_runs(
        self,
        experiment_ids: List[str],
        query: SearchQuery,
    ) -> List[Run]:
        """Search runs across experiments."""
        all_runs = []
        for exp_id in experiment_ids:
            all_runs.extend(self._store.list_runs(exp_id))

        # Apply filters
        filtered_runs = []
        for run in all_runs:
            params = {p.key: p.value for p in self._store.get_parameters(run.run_id)}
            metrics = {}
            for m in self._store.get_metrics(run.run_id):
                if m.key not in metrics:
                    metrics[m.key] = m.value

            # Check param filters
            param_match = all(params.get(k) == v for k, v in query.param_filters.items())

            # Check metric filters
            metric_match = True
            for key, (op, threshold) in query.metric_filters.items():
                if key in metrics:
                    value = metrics[key]
                    if op == ">" and not value > threshold:
                        metric_match = False
                    elif op == "<" and not value < threshold:
                        metric_match = False
                    elif op == ">=" and not value >= threshold:
                        metric_match = False
                    elif op == "<=" and not value <= threshold:
                        metric_match = False
                    elif op == "=" and not value == threshold:
                        metric_match = False

            # Check tag filters
            tag_match = all(run.tags.get(k) == v for k, v in query.tag_filters.items())

            if param_match and metric_match and tag_match:
                filtered_runs.append(run)

        # Apply ordering
        if query.order_by:
            if query.order_by.startswith("metric."):
                metric_key = query.order_by[7:]
                filtered_runs.sort(
                    key=lambda r: next(
                        (m.value for m in self._store.get_metrics(r.run_id) if m.key == metric_key),
                        float("-inf") if query.ascending else float("inf"),
                    ),
                    reverse=not query.ascending,
                )
            elif query.order_by.startswith("param."):
                param_key = query.order_by[6:]
                filtered_runs.sort(
                    key=lambda r: next(
                        (
                            p.value
                            for p in self._store.get_parameters(r.run_id)
                            if p.key == param_key
                        ),
                        "",
                    ),
                    reverse=not query.ascending,
                )

        return filtered_runs[: query.max_results]

    def get_metric_history(
        self,
        run_id: str,
        metric_key: str,
    ) -> MetricHistory:
        """Get metric history for a run."""
        return self._store.get_metric_history(run_id, metric_key)

    def compare_runs(
        self,
        run_ids: List[str],
        metric_key: Optional[str] = None,
        goal: MetricGoal = MetricGoal.MAXIMIZE,
    ) -> ComparisonResult:
        """Compare multiple runs."""
        comparison_id = hashlib.sha256(f"{':'.join(run_ids)}:{time.time()}".encode()).hexdigest()[:8]

        parameters: Dict[str, Dict[str, str]] = {}
        metrics: Dict[str, Dict[str, float]] = {}

        for run_id in run_ids:
            params = self._store.get_parameters(run_id)
            parameters[run_id] = {p.key: p.value for p in params}

            run_metrics = self._store.get_metrics(run_id)
            metrics[run_id] = {}
            for m in run_metrics:
                if m.key not in metrics[run_id]:
                    metrics[run_id][m.key] = m.value

        # Determine best run
        best_run = None
        ranking = []
        if metric_key:
            scores = []
            for run_id in run_ids:
                score = metrics.get(run_id, {}).get(metric_key)
                if score is not None:
                    scores.append((run_id, score))

            if scores:
                scores.sort(
                    key=lambda x: x[1],
                    reverse=(goal == MetricGoal.MAXIMIZE),
                )
                ranking = [s[0] for s in scores]
                best_run = ranking[0] if ranking else None

        return ComparisonResult(
            comparison_id=comparison_id,
            run_ids=run_ids,
            parameters=parameters,
            metrics=metrics,
            best_run=best_run,
            ranking=ranking,
        )


class MetricAggregator:
    """Aggregate metrics across runs."""

    def __init__(self, store: ExperimentStore):
        self._store = store

    def aggregate_metric(
        self,
        run_ids: List[str],
        metric_key: str,
    ) -> Dict[str, float]:
        """Aggregate a metric across runs."""
        values = []
        for run_id in run_ids:
            metrics = self._store.get_metrics(run_id)
            for m in metrics:
                if m.key == metric_key:
                    values.append(m.value)
                    break

        if not values:
            return {}

        import statistics

        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
        }

    def get_best_run(
        self,
        run_ids: List[str],
        metric_key: str,
        goal: MetricGoal = MetricGoal.MAXIMIZE,
    ) -> Optional[str]:
        """Get the best run by a metric."""
        best_run = None
        best_value = None

        for run_id in run_ids:
            metrics = self._store.get_metrics(run_id)
            for m in metrics:
                if m.key == metric_key:
                    if best_value is None:
                        best_value = m.value
                        best_run = run_id
                    elif goal == MetricGoal.MAXIMIZE and m.value > best_value:
                        best_value = m.value
                        best_run = run_id
                    elif goal == MetricGoal.MINIMIZE and m.value < best_value:
                        best_value = m.value
                        best_run = run_id
                    break

        return best_run


class HyperparameterAnalyzer:
    """Analyze hyperparameter impact."""

    def __init__(self, store: ExperimentStore):
        self._store = store

    def analyze_parameter_importance(
        self,
        run_ids: List[str],
        target_metric: str,
    ) -> Dict[str, float]:
        """Analyze parameter importance for a metric."""
        # Collect data
        param_values: Dict[str, List[str]] = defaultdict(list)
        metric_values: List[float] = []

        for run_id in run_ids:
            params = self._store.get_parameters(run_id)
            metrics = self._store.get_metrics(run_id)

            metric_value = None
            for m in metrics:
                if m.key == target_metric:
                    metric_value = m.value
                    break

            if metric_value is not None:
                metric_values.append(metric_value)
                for p in params:
                    param_values[p.key].append(p.value)

        if not metric_values:
            return {}

        # Simple correlation-based importance
        # In a real implementation, this would use more sophisticated methods
        importances = {}
        for param_key, values in param_values.items():
            if len(set(values)) > 1:
                # Non-constant parameter
                # Simple variance-based importance
                importances[param_key] = len(set(values)) / len(values)
            else:
                importances[param_key] = 0.0

        # Normalize
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}

        return importances

    def suggest_next_parameters(
        self,
        run_ids: List[str],
        target_metric: str,
        goal: MetricGoal = MetricGoal.MAXIMIZE,
    ) -> Dict[str, Any]:
        """Suggest next parameters based on analysis."""
        # Find best performing runs
        best_runs = []
        for run_id in run_ids:
            metrics = self._store.get_metrics(run_id)
            for m in metrics:
                if m.key == target_metric:
                    best_runs.append((run_id, m.value))
                    break

        if not best_runs:
            return {}

        # Sort by metric
        best_runs.sort(
            key=lambda x: x[1],
            reverse=(goal == MetricGoal.MAXIMIZE),
        )

        # Get parameters from top runs
        top_n = min(3, len(best_runs))
        suggestions: Dict[str, List[Any]] = defaultdict(list)

        for run_id, _ in best_runs[:top_n]:
            params = self._store.get_parameters(run_id)
            for p in params:
                suggestions[p.key].append(p.value)

        # Return most common values or averages
        result = {}
        for key, values in suggestions.items():
            # Try to convert to numeric
            try:
                numeric_values = [float(v) for v in values]
                result[key] = sum(numeric_values) / len(numeric_values)
            except (ValueError, TypeError):
                # Return most common
                result[key] = max(set(values), key=values.count)

        return result


# ========================
# Vision Provider
# ========================


class ExperimentTrackerVisionProvider(VisionProvider):
    """Vision provider for experiment tracking capabilities."""

    def __init__(self, storage_path: Optional[str] = None):
        self._storage_path = storage_path
        self._tracker: Optional[ExperimentTracker] = None

    def get_description(self) -> VisionDescription:
        """Get provider description."""
        return VisionDescription(
            name="Experiment Tracker Vision Provider",
            version="1.0.0",
            description="ML experiment tracking, metrics logging, and comparison",
            capabilities=[
                "experiment_management",
                "run_tracking",
                "metric_logging",
                "artifact_storage",
                "run_comparison",
                "hyperparameter_analysis",
            ],
        )

    def initialize(self) -> None:
        """Initialize the provider."""
        self._tracker = ExperimentTracker(self._storage_path)

    def shutdown(self) -> None:
        """Shutdown the provider."""
        self._tracker = None

    def get_tracker(self) -> ExperimentTracker:
        """Get the experiment tracker."""
        if self._tracker is None:
            self.initialize()
        return self._tracker


# ========================
# Factory Functions
# ========================


def create_experiment_tracker(
    storage_path: Optional[str] = None,
) -> ExperimentTracker:
    """Create an experiment tracker."""
    return ExperimentTracker(storage_path=storage_path)


def create_experiment(
    name: str,
    description: str = "",
    tags: Optional[List[str]] = None,
) -> Experiment:
    """Create an experiment."""
    experiment_id = hashlib.sha256(f"{name}:{time.time()}".encode()).hexdigest()[:12]

    return Experiment(
        experiment_id=experiment_id,
        name=name,
        description=description,
        tags=tags or [],
    )


def create_run(
    run_id: str,
    experiment_id: str,
    name: str = "",
    status: RunStatus = RunStatus.CREATED,
) -> Run:
    """Create a run."""
    return Run(
        run_id=run_id,
        experiment_id=experiment_id,
        name=name,
        status=status,
    )


def create_metric(
    key: str,
    value: float,
    run_id: str,
    step: int = 0,
) -> Metric:
    """Create a metric."""
    return Metric(
        key=key,
        value=value,
        run_id=run_id,
        step=step,
    )


def create_parameter(
    key: str,
    value: str,
    run_id: str,
) -> Parameter:
    """Create a parameter."""
    return Parameter(
        key=key,
        value=value,
        run_id=run_id,
    )


def create_artifact(
    artifact_id: str,
    run_id: str,
    name: str,
    path: str,
    artifact_type: ArtifactType = ArtifactType.OTHER,
) -> Artifact:
    """Create an artifact."""
    return Artifact(
        artifact_id=artifact_id,
        run_id=run_id,
        name=name,
        path=path,
        artifact_type=artifact_type,
    )


def create_search_query(
    filter_string: Optional[str] = None,
    max_results: int = 100,
) -> SearchQuery:
    """Create a search query."""
    return SearchQuery(
        filter_string=filter_string,
        max_results=max_results,
    )


def create_metric_aggregator(store: ExperimentStore) -> MetricAggregator:
    """Create a metric aggregator."""
    return MetricAggregator(store=store)


def create_hyperparameter_analyzer(store: ExperimentStore) -> HyperparameterAnalyzer:
    """Create a hyperparameter analyzer."""
    return HyperparameterAnalyzer(store=store)


def create_experiment_tracker_provider(
    storage_path: Optional[str] = None,
) -> ExperimentTrackerVisionProvider:
    """Create an experiment tracker vision provider."""
    return ExperimentTrackerVisionProvider(storage_path=storage_path)
