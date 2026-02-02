"""
Hyperparameter Optimizer.

Core optimization engine using Optuna.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.ml.tuning.search_space import SearchSpace
from src.ml.tuning.strategies import (
    SamplerType,
    PrunerType,
    get_sampler,
    get_pruner,
    get_strategy_config,
)
from src.ml.tuning.callbacks import TuningCallback, TrialInfo, CompositeCallback

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    n_trials: int = 100
    timeout: Optional[float] = None  # Timeout in seconds
    direction: str = "maximize"  # "maximize" or "minimize"
    directions: Optional[List[str]] = None  # For multi-objective
    sampler_type: SamplerType = SamplerType.TPE
    pruner_type: PrunerType = PrunerType.MEDIAN
    sampler_kwargs: Dict[str, Any] = field(default_factory=dict)
    pruner_kwargs: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    study_name: str = "optimization"
    storage: Optional[str] = None  # Optuna storage URL
    load_if_exists: bool = True
    show_progress: bool = True
    n_jobs: int = 1  # Number of parallel jobs

    @classmethod
    def from_strategy(cls, strategy: str, **overrides: Any) -> "OptimizationConfig":
        """Create config from predefined strategy."""
        config = get_strategy_config(strategy)
        return cls(
            sampler_type=config["sampler"],
            pruner_type=config["pruner"],
            sampler_kwargs=config["sampler_kwargs"],
            pruner_kwargs=config["pruner_kwargs"],
            **overrides,
        )


@dataclass
class TrialResult:
    """Result of a single optimization trial."""
    trial_number: int
    params: Dict[str, Any]
    value: Optional[float] = None
    values: Optional[List[float]] = None
    state: str = "complete"
    duration_seconds: float = 0.0
    user_attrs: Dict[str, Any] = field(default_factory=dict)
    intermediate_values: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_number": self.trial_number,
            "params": self.params,
            "value": self.value,
            "values": self.values,
            "state": self.state,
            "duration_seconds": self.duration_seconds,
            "user_attrs": self.user_attrs,
        }


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_value: Optional[float] = None
    best_values: Optional[List[float]] = None  # For multi-objective
    best_trial_number: int = 0
    n_trials: int = 0
    n_completed: int = 0
    n_pruned: int = 0
    n_failed: int = 0
    total_duration_seconds: float = 0.0
    trials: List[TrialResult] = field(default_factory=list)
    study_name: str = ""
    direction: str = "maximize"
    search_space: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "best_values": self.best_values,
            "best_trial_number": self.best_trial_number,
            "n_trials": self.n_trials,
            "n_completed": self.n_completed,
            "n_pruned": self.n_pruned,
            "n_failed": self.n_failed,
            "total_duration_seconds": self.total_duration_seconds,
            "study_name": self.study_name,
            "direction": self.direction,
            "trials": [t.to_dict() for t in self.trials],
            "search_space": self.search_space,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save result to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "OptimizationResult":
        """Load result from JSON file."""
        with open(path) as f:
            data = json.load(f)

        trials = [
            TrialResult(**trial_data)
            for trial_data in data.get("trials", [])
        ]

        return cls(
            best_params=data["best_params"],
            best_value=data.get("best_value"),
            best_values=data.get("best_values"),
            best_trial_number=data.get("best_trial_number", 0),
            n_trials=data.get("n_trials", 0),
            n_completed=data.get("n_completed", 0),
            n_pruned=data.get("n_pruned", 0),
            n_failed=data.get("n_failed", 0),
            total_duration_seconds=data.get("total_duration_seconds", 0.0),
            study_name=data.get("study_name", ""),
            direction=data.get("direction", "maximize"),
            trials=trials,
            search_space=data.get("search_space"),
        )

    def get_top_trials(self, n: int = 10) -> List[TrialResult]:
        """Get top N trials by value."""
        completed = [t for t in self.trials if t.state == "complete" and t.value is not None]
        reverse = self.direction == "maximize"
        return sorted(completed, key=lambda t: t.value or 0, reverse=reverse)[:n]

    def get_param_importance(self) -> Dict[str, float]:
        """
        Estimate parameter importance based on trial results.

        Simple variance-based importance estimation.
        """
        if not self.trials:
            return {}

        completed = [t for t in self.trials if t.state == "complete" and t.value is not None]
        if len(completed) < 5:
            return {}

        # Get all parameter names
        param_names = set()
        for trial in completed:
            param_names.update(trial.params.keys())

        importance = {}
        for param in param_names:
            # Get values for this parameter
            param_values = []
            objectives = []
            for trial in completed:
                if param in trial.params:
                    param_values.append(trial.params[param])
                    objectives.append(trial.value)

            if len(param_values) < 3:
                continue

            # Simple correlation-based importance
            try:
                # Normalize values
                if all(isinstance(v, (int, float)) for v in param_values):
                    mean_param = sum(param_values) / len(param_values)
                    std_param = (sum((v - mean_param) ** 2 for v in param_values) / len(param_values)) ** 0.5

                    mean_obj = sum(objectives) / len(objectives)
                    std_obj = (sum((v - mean_obj) ** 2 for v in objectives) / len(objectives)) ** 0.5

                    if std_param > 0 and std_obj > 0:
                        covariance = sum(
                            (p - mean_param) * (o - mean_obj)
                            for p, o in zip(param_values, objectives)
                        ) / len(param_values)
                        correlation = abs(covariance / (std_param * std_obj))
                        importance[param] = correlation
                else:
                    # For categorical, use variance in objective for each category
                    category_values: Dict[Any, List[float]] = {}
                    for v, obj in zip(param_values, objectives):
                        category_values.setdefault(v, []).append(obj)

                    between_var = 0.0
                    for values in category_values.values():
                        if values:
                            cat_mean = sum(values) / len(values)
                            between_var += len(values) * (cat_mean - mean_obj) ** 2

                    total_var = sum((o - mean_obj) ** 2 for o in objectives)
                    if total_var > 0:
                        importance[param] = between_var / total_var

            except (ValueError, ZeroDivisionError):
                continue

        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


class HyperOptimizer:
    """
    Hyperparameter optimizer using Optuna.

    Provides a high-level interface for hyperparameter optimization
    with support for callbacks, pruning, and experiment tracking.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        config: Optional[OptimizationConfig] = None,
    ):
        """
        Initialize hyperparameter optimizer.

        Args:
            search_space: Search space definition
            config: Optimization configuration
        """
        self._search_space = search_space
        self._config = config or OptimizationConfig()
        self._study = None
        self._callbacks: List[TuningCallback] = []
        self._start_time: Optional[float] = None

    @property
    def search_space(self) -> SearchSpace:
        return self._search_space

    @property
    def config(self) -> OptimizationConfig:
        return self._config

    def add_callback(self, callback: TuningCallback) -> "HyperOptimizer":
        """Add a callback."""
        self._callbacks.append(callback)
        return self

    def clear_callbacks(self) -> "HyperOptimizer":
        """Clear all callbacks."""
        self._callbacks.clear()
        return self

    def optimize(
        self,
        objective: Callable[[Any], float],
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> OptimizationResult:
        """
        Run hyperparameter optimization.

        Args:
            objective: Objective function that takes Optuna trial and returns value
            n_trials: Number of trials (overrides config)
            timeout: Timeout in seconds (overrides config)

        Returns:
            OptimizationResult
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("Optuna is required for hyperparameter optimization. Install with: pip install optuna")

        n_trials = n_trials or self._config.n_trials
        timeout = timeout or self._config.timeout

        # Create sampler and pruner
        sampler = get_sampler(
            self._config.sampler_type,
            seed=self._config.seed,
            **self._config.sampler_kwargs,
        )
        pruner = get_pruner(
            self._config.pruner_type,
            **self._config.pruner_kwargs,
        )

        # Create or load study
        if self._config.directions:
            self._study = optuna.create_study(
                study_name=self._config.study_name,
                storage=self._config.storage,
                load_if_exists=self._config.load_if_exists,
                sampler=sampler,
                directions=self._config.directions,
            )
        else:
            self._study = optuna.create_study(
                study_name=self._config.study_name,
                storage=self._config.storage,
                load_if_exists=self._config.load_if_exists,
                sampler=sampler,
                pruner=pruner,
                direction=self._config.direction,
            )

        # Create wrapped objective
        callback = CompositeCallback(self._callbacks) if self._callbacks else None
        wrapped_objective = self._create_wrapped_objective(objective, callback)

        # Run optimization
        self._start_time = time.time()

        try:
            self._study.optimize(
                wrapped_objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=self._config.n_jobs,
                show_progress_bar=self._config.show_progress,
                callbacks=[self._optuna_callback] if callback else None,
            )
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")

        return self._build_result()

    def _create_wrapped_objective(
        self,
        objective: Callable[[Any], float],
        callback: Optional[TuningCallback],
    ) -> Callable[[Any], float]:
        """Create wrapped objective function."""

        def wrapped(trial: Any) -> float:
            # Suggest parameters
            params = self._search_space.suggest(trial)

            # Validate constraints
            if not self._search_space.validate(params):
                raise ValueError("Parameters violate constraints")

            # Create trial info for callbacks
            trial_info = TrialInfo(
                trial_number=trial.number,
                params=params,
                start_time=datetime.utcnow(),
            )

            if callback:
                callback.on_trial_start(trial_info)
                if callback.should_stop():
                    trial.study.stop()

            start_time = time.time()

            try:
                value = objective(trial)
                trial_info.value = value
                trial_info.state = "complete"
            except Exception as e:
                trial_info.state = "failed"
                raise

            finally:
                trial_info.end_time = datetime.utcnow()
                trial_info.duration_seconds = time.time() - start_time
                trial_info.intermediate_values = dict(trial.intermediate_values)

                if callback:
                    callback.on_trial_end(trial_info)

            return value

        return wrapped

    def _optuna_callback(self, study: Any, trial: Any) -> None:
        """Optuna callback to check for early stopping."""
        if self._callbacks:
            callback = CompositeCallback(self._callbacks)
            if callback.should_stop():
                study.stop()

    def _build_result(self) -> OptimizationResult:
        """Build optimization result from study."""
        if self._study is None:
            return OptimizationResult(best_params={})

        trials = []
        for trial in self._study.trials:
            trial_result = TrialResult(
                trial_number=trial.number,
                params=trial.params,
                value=trial.value if hasattr(trial, "value") else None,
                values=trial.values if hasattr(trial, "values") else None,
                state=trial.state.name.lower(),
                duration_seconds=trial.duration.total_seconds() if trial.duration else 0.0,
                user_attrs=dict(trial.user_attrs),
                intermediate_values=dict(trial.intermediate_values),
            )
            trials.append(trial_result)

        # Get best trial
        try:
            best_trial = self._study.best_trial
            best_params = best_trial.params
            best_value = best_trial.value if hasattr(best_trial, "value") else None
            best_values = best_trial.values if hasattr(best_trial, "values") else None
            best_trial_number = best_trial.number
        except ValueError:
            # No completed trials
            best_params = {}
            best_value = None
            best_values = None
            best_trial_number = -1

        # Count trial states
        import optuna
        n_completed = sum(1 for t in self._study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        n_pruned = sum(1 for t in self._study.trials if t.state == optuna.trial.TrialState.PRUNED)
        n_failed = sum(1 for t in self._study.trials if t.state == optuna.trial.TrialState.FAIL)

        return OptimizationResult(
            best_params=best_params,
            best_value=best_value,
            best_values=best_values,
            best_trial_number=best_trial_number,
            n_trials=len(self._study.trials),
            n_completed=n_completed,
            n_pruned=n_pruned,
            n_failed=n_failed,
            total_duration_seconds=time.time() - (self._start_time or time.time()),
            trials=trials,
            study_name=self._config.study_name,
            direction=self._config.direction,
            search_space=self._search_space.to_dict(),
        )

    def get_study(self) -> Any:
        """Get the underlying Optuna study."""
        return self._study

    def get_param_importances(self) -> Dict[str, float]:
        """Get parameter importances from Optuna."""
        if self._study is None:
            return {}

        try:
            import optuna
            return optuna.importance.get_param_importances(self._study)
        except Exception as e:
            logger.warning(f"Could not compute param importances: {e}")
            return {}


# Global optimizer instance
_default_optimizer: Optional[HyperOptimizer] = None


def get_optimizer() -> Optional[HyperOptimizer]:
    """Get default optimizer instance."""
    return _default_optimizer


def set_optimizer(optimizer: HyperOptimizer) -> None:
    """Set default optimizer instance."""
    global _default_optimizer
    _default_optimizer = optimizer
