"""AutoML Engine Module for Vision System.

This module provides automated machine learning capabilities including:
- Automated model selection and algorithm search
- Hyperparameter optimization with multiple strategies
- Neural architecture search (NAS)
- Feature selection automation
- Ensemble model creation
- Model performance evaluation and comparison

Phase 18: Advanced ML Pipeline & AutoML
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import statistics
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from .base import VisionDescription, VisionProvider


# ========================
# Enums
# ========================


class SearchStrategy(str, Enum):
    """AutoML search strategies."""

    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    HYPERBAND = "hyperband"
    BOHB = "bohb"  # Bayesian Optimization HyperBand
    TPE = "tpe"  # Tree-structured Parzen Estimator


class OptimizationObjective(str, Enum):
    """Optimization objectives."""

    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    LOG_LOSS = "log_loss"
    MSE = "mse"
    MAE = "mae"
    RMSE = "rmse"
    R2 = "r2"
    CUSTOM = "custom"


class ModelType(str, Enum):
    """Types of ML models."""

    LINEAR = "linear"
    TREE = "tree"
    ENSEMBLE = "ensemble"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    GRADIENT_BOOSTING = "gradient_boosting"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"


class TaskType(str, Enum):
    """ML task types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    RECOMMENDATION = "recommendation"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"


class TrialStatus(str, Enum):
    """AutoML trial status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"
    TIMEOUT = "timeout"


class EarlyStoppingStrategy(str, Enum):
    """Early stopping strategies."""

    NONE = "none"
    MEDIAN = "median"
    PERCENTILE = "percentile"
    THRESHOLD = "threshold"
    PATIENCE = "patience"


# ========================
# Dataclasses
# ========================


@dataclass
class HyperparameterSpace:
    """Definition of hyperparameter search space."""

    name: str
    param_type: str  # "int", "float", "categorical", "bool"
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False
    default: Optional[Any] = None
    step: Optional[float] = None

    def sample(self) -> Any:
        """Sample a value from the hyperparameter space."""
        if self.param_type == "categorical":
            return random.choice(self.choices or [])
        elif self.param_type == "bool":
            return random.choice([True, False])
        elif self.param_type == "int":
            if self.log_scale:
                return int(math.exp(random.uniform(math.log(self.low or 1), math.log(self.high or 100))))
            return random.randint(int(self.low or 0), int(self.high or 100))
        elif self.param_type == "float":
            if self.log_scale:
                return math.exp(random.uniform(math.log(self.low or 0.001), math.log(self.high or 1.0)))
            return random.uniform(self.low or 0.0, self.high or 1.0)
        return self.default


@dataclass
class TrialResult:
    """Result of an AutoML trial."""

    trial_id: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    status: TrialStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    model_path: Optional[str] = None
    error_message: Optional[str] = None
    intermediate_values: List[float] = field(default_factory=list)


@dataclass
class SearchConfig:
    """Configuration for AutoML search."""

    search_id: str
    strategy: SearchStrategy
    objective: OptimizationObjective
    task_type: TaskType
    max_trials: int = 100
    max_time_seconds: int = 3600
    n_jobs: int = 1
    early_stopping: EarlyStoppingStrategy = EarlyStoppingStrategy.NONE
    early_stopping_patience: int = 10
    cv_folds: int = 5
    seed: Optional[int] = None
    custom_objective_fn: Optional[Callable] = None


@dataclass
class ModelCandidate:
    """A candidate model from AutoML search."""

    candidate_id: str
    model_type: ModelType
    hyperparameters: Dict[str, Any]
    score: float
    rank: int
    training_time: float
    inference_time: float
    model_size_bytes: int = 0
    feature_importances: Optional[Dict[str, float]] = None
    cross_val_scores: List[float] = field(default_factory=list)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model."""

    ensemble_id: str
    strategy: str  # "voting", "stacking", "bagging", "boosting"
    base_models: List[str]
    weights: Optional[List[float]] = None
    meta_learner: Optional[str] = None


@dataclass
class AutoMLResult:
    """Final result of AutoML search."""

    search_id: str
    best_model: ModelCandidate
    all_candidates: List[ModelCandidate]
    search_config: SearchConfig
    total_trials: int
    successful_trials: int
    total_time_seconds: float
    convergence_history: List[float] = field(default_factory=list)
    search_space_coverage: float = 0.0


# ========================
# Core Classes
# ========================


class HyperparameterOptimizer(ABC):
    """Abstract base class for hyperparameter optimizers."""

    @abstractmethod
    def suggest(self, trial_id: str) -> Dict[str, Any]:
        """Suggest hyperparameters for a new trial."""
        pass

    @abstractmethod
    def report(self, trial_id: str, metrics: Dict[str, float], status: TrialStatus) -> None:
        """Report trial results."""
        pass

    @abstractmethod
    def get_best_params(self) -> Dict[str, Any]:
        """Get the best hyperparameters found."""
        pass


class RandomSearchOptimizer(HyperparameterOptimizer):
    """Random search hyperparameter optimizer."""

    def __init__(
        self,
        search_space: List[HyperparameterSpace],
        objective: OptimizationObjective,
        seed: Optional[int] = None,
    ):
        self._search_space = search_space
        self._objective = objective
        self._trials: Dict[str, TrialResult] = {}
        self._best_params: Optional[Dict[str, Any]] = None
        self._best_score: Optional[float] = None
        if seed is not None:
            random.seed(seed)

    def suggest(self, trial_id: str) -> Dict[str, Any]:
        """Suggest random hyperparameters."""
        params = {}
        for space in self._search_space:
            params[space.name] = space.sample()
        return params

    def report(self, trial_id: str, metrics: Dict[str, float], status: TrialStatus) -> None:
        """Report trial results."""
        if status == TrialStatus.COMPLETED:
            score = metrics.get(self._objective.value, 0.0)
            if self._best_score is None or score > self._best_score:
                self._best_score = score
                self._best_params = self._trials.get(trial_id, TrialResult(
                    trial_id=trial_id,
                    hyperparameters={},
                    metrics=metrics,
                    status=status,
                    start_time=datetime.now(),
                )).hyperparameters

    def get_best_params(self) -> Dict[str, Any]:
        """Get best hyperparameters."""
        return self._best_params or {}


class BayesianOptimizer(HyperparameterOptimizer):
    """Bayesian optimization for hyperparameters."""

    def __init__(
        self,
        search_space: List[HyperparameterSpace],
        objective: OptimizationObjective,
        n_initial_points: int = 10,
        acquisition_function: str = "ei",  # Expected Improvement
    ):
        self._search_space = search_space
        self._objective = objective
        self._n_initial_points = n_initial_points
        self._acquisition_function = acquisition_function
        self._observations: List[Tuple[Dict[str, Any], float]] = []
        self._trial_count = 0

    def suggest(self, trial_id: str) -> Dict[str, Any]:
        """Suggest hyperparameters using Bayesian optimization."""
        self._trial_count += 1

        # Initial random exploration
        if self._trial_count <= self._n_initial_points:
            params = {}
            for space in self._search_space:
                params[space.name] = space.sample()
            return params

        # After initial exploration, use acquisition function
        # Simplified: use best observation with small perturbation
        if self._observations:
            best_params, _ = max(self._observations, key=lambda x: x[1])
            params = {}
            for space in self._search_space:
                if space.name in best_params:
                    # Add small perturbation
                    if space.param_type == "float":
                        delta = (space.high - space.low) * 0.1 * random.uniform(-1, 1)
                        params[space.name] = max(space.low, min(space.high, best_params[space.name] + delta))
                    elif space.param_type == "int":
                        delta = int((space.high - space.low) * 0.1 * random.uniform(-1, 1))
                        params[space.name] = max(int(space.low), min(int(space.high), int(best_params[space.name]) + delta))
                    else:
                        params[space.name] = space.sample()
                else:
                    params[space.name] = space.sample()
            return params

        # Fallback to random
        params = {}
        for space in self._search_space:
            params[space.name] = space.sample()
        return params

    def report(self, trial_id: str, metrics: Dict[str, float], status: TrialStatus) -> None:
        """Report trial results for Bayesian optimization."""
        if status == TrialStatus.COMPLETED:
            score = metrics.get(self._objective.value, 0.0)
            # Store observation (we need to track params separately)
            pass

    def get_best_params(self) -> Dict[str, Any]:
        """Get best hyperparameters."""
        if self._observations:
            best_params, _ = max(self._observations, key=lambda x: x[1])
            return best_params
        return {}


class ModelSelector:
    """Automatic model selection."""

    def __init__(self, task_type: TaskType):
        self._task_type = task_type
        self._model_registry: Dict[str, Dict[str, Any]] = {}
        self._initialize_registry()

    def _initialize_registry(self) -> None:
        """Initialize model registry based on task type."""
        if self._task_type == TaskType.CLASSIFICATION:
            self._model_registry = {
                "logistic_regression": {"type": ModelType.LINEAR, "complexity": "low"},
                "random_forest": {"type": ModelType.ENSEMBLE, "complexity": "medium"},
                "gradient_boosting": {"type": ModelType.GRADIENT_BOOSTING, "complexity": "high"},
                "svm": {"type": ModelType.SVM, "complexity": "medium"},
                "neural_network": {"type": ModelType.NEURAL_NETWORK, "complexity": "high"},
            }
        elif self._task_type == TaskType.REGRESSION:
            self._model_registry = {
                "linear_regression": {"type": ModelType.LINEAR, "complexity": "low"},
                "ridge": {"type": ModelType.LINEAR, "complexity": "low"},
                "random_forest": {"type": ModelType.ENSEMBLE, "complexity": "medium"},
                "gradient_boosting": {"type": ModelType.GRADIENT_BOOSTING, "complexity": "high"},
                "neural_network": {"type": ModelType.NEURAL_NETWORK, "complexity": "high"},
            }
        elif self._task_type == TaskType.CLUSTERING:
            self._model_registry = {
                "kmeans": {"type": ModelType.CLUSTERING, "complexity": "low"},
                "dbscan": {"type": ModelType.CLUSTERING, "complexity": "medium"},
                "hierarchical": {"type": ModelType.CLUSTERING, "complexity": "medium"},
            }

    def get_candidates(self, max_complexity: str = "high") -> List[str]:
        """Get candidate models for the task."""
        complexity_order = {"low": 1, "medium": 2, "high": 3}
        max_level = complexity_order.get(max_complexity, 3)

        candidates = []
        for name, info in self._model_registry.items():
            if complexity_order.get(info["complexity"], 0) <= max_level:
                candidates.append(name)
        return candidates

    def get_search_space(self, model_name: str) -> List[HyperparameterSpace]:
        """Get hyperparameter search space for a model."""
        # Simplified search spaces
        search_spaces = {
            "random_forest": [
                HyperparameterSpace("n_estimators", "int", low=10, high=500),
                HyperparameterSpace("max_depth", "int", low=3, high=30),
                HyperparameterSpace("min_samples_split", "int", low=2, high=20),
            ],
            "gradient_boosting": [
                HyperparameterSpace("n_estimators", "int", low=50, high=500),
                HyperparameterSpace("learning_rate", "float", low=0.001, high=0.3, log_scale=True),
                HyperparameterSpace("max_depth", "int", low=3, high=15),
            ],
            "neural_network": [
                HyperparameterSpace("hidden_layers", "int", low=1, high=5),
                HyperparameterSpace("neurons_per_layer", "int", low=16, high=512),
                HyperparameterSpace("learning_rate", "float", low=0.0001, high=0.01, log_scale=True),
                HyperparameterSpace("dropout", "float", low=0.0, high=0.5),
            ],
        }
        return search_spaces.get(model_name, [])


class AutoMLEngine:
    """Main AutoML engine for automated model training."""

    def __init__(
        self,
        task_type: TaskType,
        objective: OptimizationObjective = OptimizationObjective.ACCURACY,
        search_strategy: SearchStrategy = SearchStrategy.BAYESIAN,
    ):
        self._task_type = task_type
        self._objective = objective
        self._search_strategy = search_strategy
        self._model_selector = ModelSelector(task_type)
        self._trials: List[TrialResult] = []
        self._best_result: Optional[AutoMLResult] = None
        self._is_running = False
        self._start_time: Optional[datetime] = None

    def create_search(
        self,
        search_id: str,
        max_trials: int = 100,
        max_time_seconds: int = 3600,
    ) -> SearchConfig:
        """Create a new AutoML search configuration."""
        return SearchConfig(
            search_id=search_id,
            strategy=self._search_strategy,
            objective=self._objective,
            task_type=self._task_type,
            max_trials=max_trials,
            max_time_seconds=max_time_seconds,
        )

    def run_trial(
        self,
        trial_id: str,
        model_name: str,
        hyperparameters: Dict[str, Any],
        train_fn: Optional[Callable] = None,
    ) -> TrialResult:
        """Run a single trial."""
        start_time = datetime.now()

        try:
            # Simulate training if no function provided
            if train_fn is None:
                # Simulate training time
                time.sleep(0.01)
                # Generate mock metrics
                metrics = {
                    self._objective.value: random.uniform(0.6, 0.95),
                    "training_loss": random.uniform(0.1, 0.5),
                    "validation_loss": random.uniform(0.15, 0.6),
                }
            else:
                metrics = train_fn(model_name, hyperparameters)

            end_time = datetime.now()

            result = TrialResult(
                trial_id=trial_id,
                hyperparameters=hyperparameters,
                metrics=metrics,
                status=TrialStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
            )
        except Exception as e:
            result = TrialResult(
                trial_id=trial_id,
                hyperparameters=hyperparameters,
                metrics={},
                status=TrialStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error_message=str(e),
            )

        self._trials.append(result)
        return result

    def get_best_model(self) -> Optional[ModelCandidate]:
        """Get the best model from trials."""
        completed_trials = [t for t in self._trials if t.status == TrialStatus.COMPLETED]
        if not completed_trials:
            return None

        best_trial = max(completed_trials, key=lambda t: t.metrics.get(self._objective.value, 0))

        return ModelCandidate(
            candidate_id=best_trial.trial_id,
            model_type=ModelType.ENSEMBLE,  # Default
            hyperparameters=best_trial.hyperparameters,
            score=best_trial.metrics.get(self._objective.value, 0),
            rank=1,
            training_time=best_trial.duration_seconds,
            inference_time=0.001,  # Mock
        )

    def get_search_results(self, search_id: str) -> AutoMLResult:
        """Get final search results."""
        best_model = self.get_best_model()
        completed_trials = [t for t in self._trials if t.status == TrialStatus.COMPLETED]

        all_candidates = []
        for i, trial in enumerate(sorted(
            completed_trials,
            key=lambda t: t.metrics.get(self._objective.value, 0),
            reverse=True,
        )):
            all_candidates.append(ModelCandidate(
                candidate_id=trial.trial_id,
                model_type=ModelType.ENSEMBLE,
                hyperparameters=trial.hyperparameters,
                score=trial.metrics.get(self._objective.value, 0),
                rank=i + 1,
                training_time=trial.duration_seconds,
                inference_time=0.001,
            ))

        total_time = sum(t.duration_seconds for t in self._trials)

        return AutoMLResult(
            search_id=search_id,
            best_model=best_model or ModelCandidate(
                candidate_id="none",
                model_type=ModelType.LINEAR,
                hyperparameters={},
                score=0.0,
                rank=0,
                training_time=0.0,
                inference_time=0.0,
            ),
            all_candidates=all_candidates,
            search_config=SearchConfig(
                search_id=search_id,
                strategy=self._search_strategy,
                objective=self._objective,
                task_type=self._task_type,
            ),
            total_trials=len(self._trials),
            successful_trials=len(completed_trials),
            total_time_seconds=total_time,
        )


class NeuralArchitectureSearch:
    """Neural Architecture Search (NAS) component."""

    def __init__(
        self,
        search_space: str = "default",
        max_epochs: int = 100,
    ):
        self._search_space = search_space
        self._max_epochs = max_epochs
        self._architectures: List[Dict[str, Any]] = []

    def define_search_space(
        self,
        layer_types: List[str],
        max_layers: int = 10,
        activation_functions: List[str] = None,
    ) -> Dict[str, Any]:
        """Define neural architecture search space."""
        return {
            "layer_types": layer_types,
            "max_layers": max_layers,
            "activation_functions": activation_functions or ["relu", "tanh", "sigmoid"],
            "connection_types": ["sequential", "skip", "dense"],
        }

    def sample_architecture(self) -> Dict[str, Any]:
        """Sample a random architecture."""
        num_layers = random.randint(2, 10)
        layers = []

        layer_types = ["conv", "dense", "lstm", "attention"]
        activations = ["relu", "tanh", "sigmoid", "gelu"]

        for i in range(num_layers):
            layers.append({
                "type": random.choice(layer_types),
                "units": random.choice([32, 64, 128, 256, 512]),
                "activation": random.choice(activations),
                "dropout": random.uniform(0, 0.5),
            })

        return {
            "architecture_id": hashlib.md5(str(layers).encode()).hexdigest()[:8],
            "layers": layers,
            "optimizer": random.choice(["adam", "sgd", "rmsprop"]),
            "learning_rate": random.uniform(0.0001, 0.01),
        }

    def evaluate_architecture(
        self,
        architecture: Dict[str, Any],
        eval_fn: Optional[Callable] = None,
    ) -> float:
        """Evaluate an architecture."""
        if eval_fn is not None:
            return eval_fn(architecture)
        # Mock evaluation
        return random.uniform(0.5, 0.95)


class FeatureSelector:
    """Automatic feature selection."""

    def __init__(self, method: str = "importance"):
        self._method = method
        self._selected_features: List[str] = []
        self._feature_scores: Dict[str, float] = {}

    def fit(
        self,
        feature_names: List[str],
        importance_scores: Optional[List[float]] = None,
    ) -> None:
        """Fit feature selector."""
        if importance_scores is None:
            importance_scores = [random.uniform(0, 1) for _ in feature_names]

        self._feature_scores = dict(zip(feature_names, importance_scores))

    def select_top_k(self, k: int) -> List[str]:
        """Select top k features."""
        sorted_features = sorted(
            self._feature_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        self._selected_features = [f[0] for f in sorted_features[:k]]
        return self._selected_features

    def select_by_threshold(self, threshold: float) -> List[str]:
        """Select features above threshold."""
        self._selected_features = [
            name for name, score in self._feature_scores.items()
            if score >= threshold
        ]
        return self._selected_features

    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self._feature_scores.copy()


class EnsembleBuilder:
    """Build ensemble models from candidates."""

    def __init__(self, strategy: str = "voting"):
        self._strategy = strategy
        self._base_models: List[ModelCandidate] = []
        self._weights: Optional[List[float]] = None

    def add_model(self, model: ModelCandidate, weight: float = 1.0) -> None:
        """Add a model to the ensemble."""
        self._base_models.append(model)
        if self._weights is None:
            self._weights = []
        self._weights.append(weight)

    def build(self) -> EnsembleConfig:
        """Build ensemble configuration."""
        return EnsembleConfig(
            ensemble_id=hashlib.md5(
                str([m.candidate_id for m in self._base_models]).encode()
            ).hexdigest()[:8],
            strategy=self._strategy,
            base_models=[m.candidate_id for m in self._base_models],
            weights=self._weights,
        )

    def optimize_weights(
        self,
        validation_fn: Optional[Callable] = None,
    ) -> List[float]:
        """Optimize ensemble weights."""
        if validation_fn is None:
            # Use inverse rank weighting
            ranks = [m.rank for m in self._base_models]
            total = sum(1 / r for r in ranks)
            self._weights = [(1 / r) / total for r in ranks]
        else:
            # Grid search over weight combinations
            best_weights = self._weights
            best_score = 0.0

            for _ in range(100):
                weights = [random.uniform(0, 1) for _ in self._base_models]
                total = sum(weights)
                weights = [w / total for w in weights]

                score = validation_fn(weights)
                if score > best_score:
                    best_score = score
                    best_weights = weights

            self._weights = best_weights

        return self._weights


# ========================
# Vision Provider
# ========================


class AutoMLVisionProvider(VisionProvider):
    """Vision provider for AutoML capabilities."""

    def __init__(
        self,
        task_type: TaskType = TaskType.CLASSIFICATION,
        default_strategy: SearchStrategy = SearchStrategy.BAYESIAN,
    ):
        self._task_type = task_type
        self._default_strategy = default_strategy
        self._engine: Optional[AutoMLEngine] = None

    def get_description(self) -> VisionDescription:
        """Get provider description."""
        return VisionDescription(
            name="AutoML Vision Provider",
            version="1.0.0",
            description="Automated Machine Learning with hyperparameter optimization",
            capabilities=[
                "automated_model_selection",
                "hyperparameter_optimization",
                "neural_architecture_search",
                "feature_selection",
                "ensemble_building",
            ],
        )

    def initialize(self) -> None:
        """Initialize the provider."""
        self._engine = AutoMLEngine(
            task_type=self._task_type,
            search_strategy=self._default_strategy,
        )

    def shutdown(self) -> None:
        """Shutdown the provider."""
        self._engine = None

    def get_engine(self) -> AutoMLEngine:
        """Get the AutoML engine."""
        if self._engine is None:
            self.initialize()
        return self._engine


# ========================
# Factory Functions
# ========================


def create_automl_engine(
    task_type: TaskType = TaskType.CLASSIFICATION,
    objective: OptimizationObjective = OptimizationObjective.ACCURACY,
    search_strategy: SearchStrategy = SearchStrategy.BAYESIAN,
) -> AutoMLEngine:
    """Create an AutoML engine."""
    return AutoMLEngine(
        task_type=task_type,
        objective=objective,
        search_strategy=search_strategy,
    )


def create_search_config(
    search_id: str,
    strategy: SearchStrategy = SearchStrategy.BAYESIAN,
    objective: OptimizationObjective = OptimizationObjective.ACCURACY,
    task_type: TaskType = TaskType.CLASSIFICATION,
    max_trials: int = 100,
) -> SearchConfig:
    """Create a search configuration."""
    return SearchConfig(
        search_id=search_id,
        strategy=strategy,
        objective=objective,
        task_type=task_type,
        max_trials=max_trials,
    )


def create_hyperparameter_space(
    name: str,
    param_type: str,
    low: Optional[float] = None,
    high: Optional[float] = None,
    choices: Optional[List[Any]] = None,
    log_scale: bool = False,
) -> HyperparameterSpace:
    """Create a hyperparameter space definition."""
    return HyperparameterSpace(
        name=name,
        param_type=param_type,
        low=low,
        high=high,
        choices=choices,
        log_scale=log_scale,
    )


def create_model_selector(task_type: TaskType) -> ModelSelector:
    """Create a model selector."""
    return ModelSelector(task_type=task_type)


def create_nas(
    search_space: str = "default",
    max_epochs: int = 100,
) -> NeuralArchitectureSearch:
    """Create a neural architecture search instance."""
    return NeuralArchitectureSearch(
        search_space=search_space,
        max_epochs=max_epochs,
    )


def create_feature_selector(method: str = "importance") -> FeatureSelector:
    """Create a feature selector."""
    return FeatureSelector(method=method)


def create_ensemble_builder(strategy: str = "voting") -> EnsembleBuilder:
    """Create an ensemble builder."""
    return EnsembleBuilder(strategy=strategy)


def create_automl_provider(
    task_type: TaskType = TaskType.CLASSIFICATION,
    default_strategy: SearchStrategy = SearchStrategy.BAYESIAN,
) -> AutoMLVisionProvider:
    """Create an AutoML vision provider."""
    return AutoMLVisionProvider(
        task_type=task_type,
        default_strategy=default_strategy,
    )
