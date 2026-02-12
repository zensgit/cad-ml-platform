"""
Callbacks for Hyperparameter Tuning.

Provides callback interfaces for monitoring and controlling optimization.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.ml.experiment import ExperimentTracker

logger = logging.getLogger(__name__)


@dataclass
class TrialInfo:
    """Information about a trial."""
    trial_number: int
    params: Dict[str, Any]
    value: Optional[float] = None
    values: Optional[List[float]] = None  # For multi-objective
    state: str = "running"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    intermediate_values: Dict[int, float] = field(default_factory=dict)
    user_attrs: Dict[str, Any] = field(default_factory=dict)


class TuningCallback(ABC):
    """Base class for tuning callbacks."""

    @abstractmethod
    def on_trial_start(self, trial_info: TrialInfo) -> None:
        """Called when a trial starts."""
        pass

    @abstractmethod
    def on_trial_end(self, trial_info: TrialInfo) -> None:
        """Called when a trial ends."""
        pass

    def on_step(self, trial_info: TrialInfo, step: int, value: float) -> None:
        """Called after each optimization step (optional)."""
        pass

    def should_stop(self) -> bool:
        """Return True to stop optimization early."""
        return False


class EarlyStoppingCallback(TuningCallback):
    """
    Early stopping callback for tuning.

    Stops optimization when no improvement is seen for a number of trials.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        direction: str = "maximize",
    ):
        """
        Initialize early stopping callback.

        Args:
            patience: Number of trials without improvement before stopping
            min_delta: Minimum change to qualify as an improvement
            direction: "maximize" or "minimize"
        """
        self._patience = patience
        self._min_delta = min_delta
        self._direction = direction
        self._best_value: Optional[float] = None
        self._trials_without_improvement = 0
        self._should_stop = False

    def on_trial_start(self, trial_info: TrialInfo) -> None:
        pass

    def on_trial_end(self, trial_info: TrialInfo) -> None:
        if trial_info.value is None:
            return

        value = trial_info.value

        if self._best_value is None:
            self._best_value = value
            self._trials_without_improvement = 0
            return

        improved = False
        if self._direction == "maximize":
            if value > self._best_value + self._min_delta:
                improved = True
        else:
            if value < self._best_value - self._min_delta:
                improved = True

        if improved:
            self._best_value = value
            self._trials_without_improvement = 0
        else:
            self._trials_without_improvement += 1

        if self._trials_without_improvement >= self._patience:
            logger.info(
                f"Early stopping: no improvement for {self._patience} trials "
                f"(best={self._best_value:.4f})"
            )
            self._should_stop = True

    def should_stop(self) -> bool:
        return self._should_stop

    def reset(self) -> None:
        """Reset the callback state."""
        self._best_value = None
        self._trials_without_improvement = 0
        self._should_stop = False


class ExperimentTrackerCallback(TuningCallback):
    """
    Callback that integrates with experiment tracking (M1).

    Logs trials as experiment runs.
    """

    def __init__(
        self,
        tracker: "ExperimentTracker",
        experiment_name: str = "hyperparameter_tuning",
        log_params: bool = True,
        log_metrics: bool = True,
    ):
        """
        Initialize experiment tracker callback.

        Args:
            tracker: ExperimentTracker instance
            experiment_name: Name for the tuning experiment
            log_params: Whether to log hyperparameters
            log_metrics: Whether to log metrics
        """
        self._tracker = tracker
        self._experiment_name = experiment_name
        self._log_params = log_params
        self._log_metrics = log_metrics
        self._current_run = None

    def on_trial_start(self, trial_info: TrialInfo) -> None:
        run_name = f"trial_{trial_info.trial_number}"
        self._current_run = self._tracker.start_run(
            experiment_name=self._experiment_name,
            run_name=run_name,
            tags={"type": "tuning_trial"},
            config=trial_info.params if self._log_params else None,
        )

    def on_trial_end(self, trial_info: TrialInfo) -> None:
        if self._current_run is None:
            return

        if self._log_metrics and trial_info.value is not None:
            self._tracker.log_metrics({
                "objective_value": trial_info.value,
                "duration_seconds": trial_info.duration_seconds,
            })

        # Log intermediate values
        for step, value in trial_info.intermediate_values.items():
            self._tracker.log_metrics({"intermediate": value}, step=step)

        self._tracker.end_run(status="completed" if trial_info.state == "complete" else "failed")
        self._current_run = None

    def on_step(self, trial_info: TrialInfo, step: int, value: float) -> None:
        if self._current_run is not None and self._log_metrics:
            self._tracker.log_metrics({"step_value": value}, step=step)


class ProgressCallback(TuningCallback):
    """
    Progress reporting callback.

    Reports optimization progress to console or custom handler.
    """

    def __init__(
        self,
        n_trials: int,
        direction: str = "maximize",
        verbose: bool = True,
        report_interval: int = 1,
        custom_handler: Optional[callable] = None,
    ):
        """
        Initialize progress callback.

        Args:
            n_trials: Total number of trials
            direction: "maximize" or "minimize"
            verbose: Whether to print progress
            report_interval: Interval between reports
            custom_handler: Custom function to handle progress updates
        """
        self._n_trials = n_trials
        self._direction = direction
        self._verbose = verbose
        self._report_interval = report_interval
        self._custom_handler = custom_handler
        self._best_value: Optional[float] = None
        self._best_params: Optional[Dict[str, Any]] = None
        self._completed_trials = 0
        self._start_time = time.time()

    def on_trial_start(self, trial_info: TrialInfo) -> None:
        pass

    def on_trial_end(self, trial_info: TrialInfo) -> None:
        self._completed_trials += 1

        if trial_info.value is not None:
            is_best = False
            if self._best_value is None:
                is_best = True
            elif self._direction == "maximize" and trial_info.value > self._best_value:
                is_best = True
            elif self._direction == "minimize" and trial_info.value < self._best_value:
                is_best = True

            if is_best:
                self._best_value = trial_info.value
                self._best_params = trial_info.params.copy()

        if self._completed_trials % self._report_interval == 0:
            self._report_progress(trial_info)

    def _report_progress(self, trial_info: TrialInfo) -> None:
        elapsed = time.time() - self._start_time
        progress = self._completed_trials / self._n_trials * 100

        current_str = f"{trial_info.value:.4f}" if trial_info.value is not None else "N/A"
        best_str = f"{self._best_value:.4f}" if self._best_value is not None else "N/A"

        message = (
            f"Trial {self._completed_trials}/{self._n_trials} ({progress:.1f}%) | "
            f"Current: {current_str} | "
            f"Best: {best_str} | "
            f"Elapsed: {elapsed:.1f}s"
        )

        if self._verbose:
            logger.info(message)

        if self._custom_handler:
            self._custom_handler({
                "completed": self._completed_trials,
                "total": self._n_trials,
                "progress": progress,
                "current_value": trial_info.value,
                "best_value": self._best_value,
                "best_params": self._best_params,
                "elapsed_seconds": elapsed,
                "message": message,
            })

    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary."""
        elapsed = time.time() - self._start_time
        return {
            "completed_trials": self._completed_trials,
            "total_trials": self._n_trials,
            "best_value": self._best_value,
            "best_params": self._best_params,
            "elapsed_seconds": elapsed,
            "trials_per_second": self._completed_trials / max(1, elapsed),
        }


class CompositeCallback(TuningCallback):
    """Combines multiple callbacks."""

    def __init__(self, callbacks: List[TuningCallback]):
        """
        Initialize composite callback.

        Args:
            callbacks: List of callbacks to combine
        """
        self._callbacks = callbacks

    def on_trial_start(self, trial_info: TrialInfo) -> None:
        for callback in self._callbacks:
            callback.on_trial_start(trial_info)

    def on_trial_end(self, trial_info: TrialInfo) -> None:
        for callback in self._callbacks:
            callback.on_trial_end(trial_info)

    def on_step(self, trial_info: TrialInfo, step: int, value: float) -> None:
        for callback in self._callbacks:
            callback.on_step(trial_info, step, value)

    def should_stop(self) -> bool:
        return any(callback.should_stop() for callback in self._callbacks)
