"""
A/B Testing framework for model deployment.

Provides:
- Traffic splitting strategies
- Experiment management
- Statistical analysis
- Automatic winner selection
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    """A/B experiment status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"


class TrafficSplitStrategy(str, Enum):
    """Traffic splitting strategy."""
    RANDOM = "random"
    WEIGHTED = "weighted"
    STICKY = "sticky"  # Same user always gets same variant
    GRADUAL = "gradual"  # Gradually increase traffic


class WinnerCriterion(str, Enum):
    """Criterion for selecting winner."""
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"


@dataclass
class Variant:
    """A/B test variant."""
    name: str
    model_name: str
    model_version: str
    weight: float = 0.5  # Traffic weight (0-1)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Runtime stats
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    metrics: Dict[str, List[float]] = field(default_factory=dict)

    @property
    def avg_latency_ms(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count

    @property
    def success_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.success_count / self.request_count

    @property
    def error_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count

    def record_request(
        self,
        success: bool,
        latency_ms: float,
        custom_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record a request result."""
        self.request_count += 1
        self.total_latency_ms += latency_ms

        if success:
            self.success_count += 1
        else:
            self.error_count += 1

        if custom_metrics:
            for metric, value in custom_metrics.items():
                if metric not in self.metrics:
                    self.metrics[metric] = []
                self.metrics[metric].append(value)

    def get_metric_stats(self, metric: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        values = self.metrics.get(metric, [])
        if not values:
            return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        import statistics
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "weight": self.weight,
            "description": self.description,
            "metadata": self.metadata,
            "request_count": self.request_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "avg_latency_ms": self.avg_latency_ms,
            "success_rate": self.success_rate,
        }


@dataclass
class Experiment:
    """A/B test experiment."""
    id: str
    name: str
    description: str = ""
    variants: List[Variant] = field(default_factory=list)
    status: ExperimentStatus = ExperimentStatus.DRAFT
    strategy: TrafficSplitStrategy = TrafficSplitStrategy.RANDOM
    primary_metric: str = "success_rate"
    winner_criterion: WinnerCriterion = WinnerCriterion.HIGHER_IS_BETTER
    min_sample_size: int = 100
    confidence_level: float = 0.95
    max_duration_hours: int = 168  # 1 week
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    winner: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def is_running(self) -> bool:
        return self.status == ExperimentStatus.RUNNING

    @property
    def total_requests(self) -> int:
        return sum(v.request_count for v in self.variants)

    @property
    def duration(self) -> Optional[timedelta]:
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now()
        return end - self.started_at

    def get_variant(self, name: str) -> Optional[Variant]:
        """Get variant by name."""
        for v in self.variants:
            if v.name == name:
                return v
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "variants": [v.to_dict() for v in self.variants],
            "status": self.status.value,
            "strategy": self.strategy.value,
            "primary_metric": self.primary_metric,
            "winner_criterion": self.winner_criterion.value,
            "min_sample_size": self.min_sample_size,
            "confidence_level": self.confidence_level,
            "max_duration_hours": self.max_duration_hours,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "winner": self.winner,
            "total_requests": self.total_requests,
            "tags": self.tags,
        }


@dataclass
class ABTestConfig:
    """A/B testing configuration."""
    storage_path: str = "./ab_tests"
    auto_stop_on_significance: bool = True
    auto_stop_on_error_spike: bool = True
    error_spike_threshold: float = 0.1  # 10% error rate
    min_requests_per_variant: int = 50
    check_interval_seconds: int = 60


class ABTestManager:
    """
    A/B testing manager for model experiments.

    Handles:
    - Experiment lifecycle management
    - Traffic routing
    - Statistical analysis
    - Winner determination
    """

    def __init__(self, config: Optional[ABTestConfig] = None):
        """
        Initialize A/B test manager.

        Args:
            config: A/B test configuration
        """
        self._config = config or ABTestConfig()
        self._experiments: Dict[str, Experiment] = {}
        self._lock = threading.RLock()
        self._user_assignments: Dict[str, Dict[str, str]] = {}  # user_id -> {experiment_id: variant_name}

        # Setup storage
        self._storage_path = Path(self._config.storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)

        self._load_experiments()

    def _load_experiments(self) -> None:
        """Load experiments from storage."""
        experiments_file = self._storage_path / "experiments.json"
        if experiments_file.exists():
            try:
                with open(experiments_file) as f:
                    data = json.load(f)

                for exp_data in data.get("experiments", []):
                    variants = [
                        Variant(
                            name=v["name"],
                            model_name=v["model_name"],
                            model_version=v["model_version"],
                            weight=v.get("weight", 0.5),
                            description=v.get("description", ""),
                        )
                        for v in exp_data.get("variants", [])
                    ]

                    experiment = Experiment(
                        id=exp_data["id"],
                        name=exp_data["name"],
                        description=exp_data.get("description", ""),
                        variants=variants,
                        status=ExperimentStatus(exp_data.get("status", "draft")),
                        strategy=TrafficSplitStrategy(exp_data.get("strategy", "random")),
                        primary_metric=exp_data.get("primary_metric", "success_rate"),
                        winner_criterion=WinnerCriterion(exp_data.get("winner_criterion", "higher_is_better")),
                    )

                    self._experiments[experiment.id] = experiment

                logger.info(f"Loaded {len(self._experiments)} experiments")
            except Exception as e:
                logger.warning(f"Failed to load experiments: {e}")

    def _save_experiments(self) -> None:
        """Save experiments to storage."""
        experiments_file = self._storage_path / "experiments.json"

        data = {
            "experiments": [exp.to_dict() for exp in self._experiments.values()],
            "updated_at": datetime.now().isoformat(),
        }

        with open(experiments_file, "w") as f:
            json.dump(data, f, indent=2)

    def create_experiment(
        self,
        name: str,
        variants: List[Dict[str, Any]],
        description: str = "",
        strategy: TrafficSplitStrategy = TrafficSplitStrategy.RANDOM,
        primary_metric: str = "success_rate",
        winner_criterion: WinnerCriterion = WinnerCriterion.HIGHER_IS_BETTER,
        min_sample_size: int = 100,
        confidence_level: float = 0.95,
        max_duration_hours: int = 168,
        tags: Optional[Dict[str, str]] = None,
    ) -> Experiment:
        """
        Create a new A/B experiment.

        Args:
            name: Experiment name
            variants: List of variant configurations
            description: Experiment description
            strategy: Traffic splitting strategy
            primary_metric: Primary metric for comparison
            winner_criterion: How to determine winner
            min_sample_size: Minimum samples per variant
            confidence_level: Statistical confidence level
            max_duration_hours: Maximum experiment duration
            tags: Experiment tags

        Returns:
            Created Experiment
        """
        with self._lock:
            experiment_id = f"exp_{int(time.time())}_{random.randint(1000, 9999)}"

            # Normalize weights
            total_weight = sum(v.get("weight", 1.0) for v in variants)
            variant_objects = [
                Variant(
                    name=v["name"],
                    model_name=v["model_name"],
                    model_version=v["model_version"],
                    weight=v.get("weight", 1.0) / total_weight,
                    description=v.get("description", ""),
                    metadata=v.get("metadata", {}),
                )
                for v in variants
            ]

            experiment = Experiment(
                id=experiment_id,
                name=name,
                description=description,
                variants=variant_objects,
                strategy=strategy,
                primary_metric=primary_metric,
                winner_criterion=winner_criterion,
                min_sample_size=min_sample_size,
                confidence_level=confidence_level,
                max_duration_hours=max_duration_hours,
                tags=tags or {},
            )

            self._experiments[experiment_id] = experiment
            self._save_experiments()

            logger.info(f"Created experiment: {name} ({experiment_id})")
            return experiment

    def start_experiment(self, experiment_id: str) -> Experiment:
        """Start an experiment."""
        with self._lock:
            experiment = self._experiments.get(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment not found: {experiment_id}")

            if experiment.status != ExperimentStatus.DRAFT:
                raise ValueError(f"Experiment must be in DRAFT status to start")

            experiment.status = ExperimentStatus.RUNNING
            experiment.started_at = datetime.now()
            self._save_experiments()

            logger.info(f"Started experiment: {experiment.name}")
            return experiment

    def pause_experiment(self, experiment_id: str) -> Experiment:
        """Pause an experiment."""
        with self._lock:
            experiment = self._experiments.get(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment not found: {experiment_id}")

            experiment.status = ExperimentStatus.PAUSED
            self._save_experiments()

            logger.info(f"Paused experiment: {experiment.name}")
            return experiment

    def resume_experiment(self, experiment_id: str) -> Experiment:
        """Resume a paused experiment."""
        with self._lock:
            experiment = self._experiments.get(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment not found: {experiment_id}")

            if experiment.status != ExperimentStatus.PAUSED:
                raise ValueError("Can only resume paused experiments")

            experiment.status = ExperimentStatus.RUNNING
            self._save_experiments()

            logger.info(f"Resumed experiment: {experiment.name}")
            return experiment

    def stop_experiment(
        self,
        experiment_id: str,
        winner: Optional[str] = None,
    ) -> Experiment:
        """Stop an experiment."""
        with self._lock:
            experiment = self._experiments.get(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment not found: {experiment_id}")

            experiment.status = ExperimentStatus.STOPPED
            experiment.completed_at = datetime.now()
            experiment.winner = winner
            self._save_experiments()

            logger.info(f"Stopped experiment: {experiment.name}, winner: {winner}")
            return experiment

    def complete_experiment(self, experiment_id: str) -> Experiment:
        """Complete an experiment and determine winner."""
        with self._lock:
            experiment = self._experiments.get(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment not found: {experiment_id}")

            # Analyze results
            winner = self._determine_winner(experiment)

            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_at = datetime.now()
            experiment.winner = winner
            self._save_experiments()

            logger.info(f"Completed experiment: {experiment.name}, winner: {winner}")
            return experiment

    def route_request(
        self,
        experiment_id: str,
        user_id: Optional[str] = None,
    ) -> Variant:
        """
        Route a request to a variant.

        Args:
            experiment_id: Experiment ID
            user_id: User ID for sticky routing

        Returns:
            Selected variant
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        if not experiment.is_running:
            raise ValueError(f"Experiment is not running: {experiment_id}")

        # Select variant based on strategy
        if experiment.strategy == TrafficSplitStrategy.STICKY and user_id:
            variant = self._get_sticky_variant(experiment, user_id)
        elif experiment.strategy == TrafficSplitStrategy.WEIGHTED:
            variant = self._get_weighted_variant(experiment)
        else:
            variant = self._get_random_variant(experiment)

        return variant

    def _get_random_variant(self, experiment: Experiment) -> Variant:
        """Select a random variant based on weights."""
        r = random.random()
        cumulative = 0.0

        for variant in experiment.variants:
            cumulative += variant.weight
            if r < cumulative:
                return variant

        return experiment.variants[-1]

    def _get_weighted_variant(self, experiment: Experiment) -> Variant:
        """Select variant using weighted random selection."""
        return self._get_random_variant(experiment)

    def _get_sticky_variant(self, experiment: Experiment, user_id: str) -> Variant:
        """Get consistent variant for a user."""
        # Check if user already assigned
        if user_id in self._user_assignments:
            if experiment.id in self._user_assignments[user_id]:
                variant_name = self._user_assignments[user_id][experiment.id]
                variant = experiment.get_variant(variant_name)
                if variant:
                    return variant

        # Assign based on user ID hash
        hash_val = int(
            hashlib.md5(f"{experiment.id}:{user_id}".encode()).hexdigest(), 16
        )  # nosec B324 - consistent variant bucketing
        bucket = (hash_val % 100) / 100.0

        cumulative = 0.0
        selected = experiment.variants[0]

        for variant in experiment.variants:
            cumulative += variant.weight
            if bucket < cumulative:
                selected = variant
                break

        # Store assignment
        if user_id not in self._user_assignments:
            self._user_assignments[user_id] = {}
        self._user_assignments[user_id][experiment.id] = selected.name

        return selected

    def record_result(
        self,
        experiment_id: str,
        variant_name: str,
        success: bool,
        latency_ms: float,
        custom_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record a request result.

        Args:
            experiment_id: Experiment ID
            variant_name: Variant name
            success: Whether request succeeded
            latency_ms: Request latency
            custom_metrics: Custom metrics to record
        """
        with self._lock:
            experiment = self._experiments.get(experiment_id)
            if not experiment:
                return

            variant = experiment.get_variant(variant_name)
            if variant:
                variant.record_request(success, latency_ms, custom_metrics)

            # Check for auto-stop conditions
            self._check_auto_stop(experiment)

    def _check_auto_stop(self, experiment: Experiment) -> None:
        """Check if experiment should auto-stop."""
        if not experiment.is_running:
            return

        # Check error spike
        if self._config.auto_stop_on_error_spike:
            for variant in experiment.variants:
                if variant.request_count >= self._config.min_requests_per_variant:
                    if variant.error_rate > self._config.error_spike_threshold:
                        logger.warning(f"Error spike detected in variant {variant.name}")
                        self.pause_experiment(experiment.id)
                        return

        # Check statistical significance
        if self._config.auto_stop_on_significance:
            if self._has_statistical_significance(experiment):
                winner = self._determine_winner(experiment)
                if winner:
                    self.complete_experiment(experiment.id)

    def _has_statistical_significance(self, experiment: Experiment) -> bool:
        """Check if experiment has statistical significance."""
        # Simple check: all variants have minimum samples
        for variant in experiment.variants:
            if variant.request_count < experiment.min_sample_size:
                return False

        # Perform statistical test
        try:
            from scipy import stats

            if len(experiment.variants) == 2:
                v1, v2 = experiment.variants
                metric = experiment.primary_metric

                if metric == "success_rate":
                    # Chi-squared test for proportions
                    table = [
                        [v1.success_count, v1.error_count],
                        [v2.success_count, v2.error_count],
                    ]
                    _, p_value, _, _ = stats.chi2_contingency(table)
                else:
                    # T-test for other metrics
                    values1 = v1.metrics.get(metric, [])
                    values2 = v2.metrics.get(metric, [])
                    if values1 and values2:
                        _, p_value = stats.ttest_ind(values1, values2)
                    else:
                        return False

                return p_value < (1 - experiment.confidence_level)

        except ImportError:
            # Fallback: simple comparison
            pass

        return False

    def _determine_winner(self, experiment: Experiment) -> Optional[str]:
        """Determine the winning variant."""
        if not experiment.variants:
            return None

        metric = experiment.primary_metric
        criterion = experiment.winner_criterion

        best_variant = None
        best_value = None

        for variant in experiment.variants:
            if variant.request_count < self._config.min_requests_per_variant:
                continue

            if metric == "success_rate":
                value = variant.success_rate
            elif metric == "latency":
                value = variant.avg_latency_ms
            else:
                stats = variant.get_metric_stats(metric)
                value = stats.get("mean", 0.0)

            if best_value is None:
                best_variant = variant
                best_value = value
            else:
                if criterion == WinnerCriterion.HIGHER_IS_BETTER:
                    if value > best_value:
                        best_variant = variant
                        best_value = value
                else:
                    if value < best_value:
                        best_variant = variant
                        best_value = value

        return best_variant.name if best_variant else None

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID."""
        return self._experiments.get(experiment_id)

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
    ) -> List[Experiment]:
        """List experiments."""
        experiments = list(self._experiments.values())
        if status:
            experiments = [e for e in experiments if e.status == status]
        return sorted(experiments, key=lambda e: e.created_at, reverse=True)

    def get_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment results."""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        results = {
            "experiment": experiment.to_dict(),
            "variants": {},
            "winner": experiment.winner,
            "total_requests": experiment.total_requests,
        }

        for variant in experiment.variants:
            results["variants"][variant.name] = {
                **variant.to_dict(),
                "metric_stats": {
                    m: variant.get_metric_stats(m)
                    for m in variant.metrics
                },
            }

        return results

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        with self._lock:
            if experiment_id not in self._experiments:
                return False

            del self._experiments[experiment_id]
            self._save_experiments()

            logger.info(f"Deleted experiment: {experiment_id}")
            return True

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        running = sum(1 for e in self._experiments.values() if e.is_running)
        completed = sum(1 for e in self._experiments.values() if e.status == ExperimentStatus.COMPLETED)

        return {
            "total_experiments": len(self._experiments),
            "running_experiments": running,
            "completed_experiments": completed,
            "storage_path": str(self._storage_path),
        }


def get_ab_test_manager(config: Optional[ABTestConfig] = None) -> ABTestManager:
    """Get or create A/B test manager singleton."""
    global _ab_manager
    if "_ab_manager" not in globals():
        _ab_manager = ABTestManager(config)
    return _ab_manager
