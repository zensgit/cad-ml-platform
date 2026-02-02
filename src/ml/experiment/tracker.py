"""
Experiment Tracker - Main interface for experiment tracking.

Provides:
- Unified experiment management
- Run lifecycle management
- Integration with metrics, artifacts, and registry
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar

from src.ml.experiment.run import Run, RunContext, RunStatus
from src.ml.experiment.metrics import MetricsLogger
from src.ml.experiment.artifacts import ArtifactStore
from src.ml.experiment.registry import ModelRegistry, ModelStage
from src.ml.experiment.comparison import ExperimentComparison, ComparisonReport

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class TrackerConfig:
    """Configuration for experiment tracker."""
    base_path: str = "experiments"
    auto_log_git: bool = True
    auto_log_env: bool = True
    log_system_metrics: bool = False


class ExperimentTracker:
    """
    Unified experiment tracking interface.

    Provides:
    - Experiment and run management
    - Parameter and metric logging
    - Artifact storage
    - Model registration
    - Experiment comparison
    """

    def __init__(self, config: Optional[TrackerConfig] = None):
        """
        Initialize experiment tracker.

        Args:
            config: Tracker configuration
        """
        self._config = config or TrackerConfig()
        self._base_path = Path(self._config.base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)

        self._current_run: Optional[Run] = None
        self._experiments: Dict[str, List[str]] = {}  # experiment_name -> run_ids

        self._artifact_store = ArtifactStore(self._base_path / "artifacts")
        self._model_registry = ModelRegistry(self._base_path / "models")

        self._index_path = self._base_path / "index.json"
        if self._index_path.exists():
            self._load_index()

        logger.info(f"ExperimentTracker initialized at {self._base_path}")

    @property
    def current_run(self) -> Optional[Run]:
        """Get current active run."""
        return self._current_run

    @property
    def artifact_store(self) -> ArtifactStore:
        """Get artifact store."""
        return self._artifact_store

    @property
    def model_registry(self) -> ModelRegistry:
        """Get model registry."""
        return self._model_registry

    def start_run(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Run:
        """
        Start a new experiment run.

        Args:
            experiment_name: Name of the experiment
            run_name: Optional run name (generates ID if not provided)
            tags: Initial tags
            config: Configuration to log as parameters

        Returns:
            Started Run object
        """
        if self._current_run is not None:
            logger.warning(f"Ending previous run {self._current_run.run_id}")
            self.end_run()

        run = Run(
            experiment_name=experiment_name,
            _base_path=self._base_path,
        )

        if run_name:
            run.run_id = run_name

        if tags:
            run.set_tags(tags)

        # Auto-log git info
        if self._config.auto_log_git:
            git_info = self._get_git_info()
            if git_info:
                run.log_params({"git": git_info})

        # Auto-log environment
        if self._config.auto_log_env:
            env_info = self._get_env_info()
            run.log_params({"env": env_info})

        # Log config as params
        if config:
            run.log_params(config)

        run.start()
        self._current_run = run

        # Update index
        if experiment_name not in self._experiments:
            self._experiments[experiment_name] = []
        self._experiments[experiment_name].append(run.run_id)
        self._save_index()

        return run

    def end_run(self, status: RunStatus = RunStatus.COMPLETED) -> None:
        """
        End the current run.

        Args:
            status: Final run status
        """
        if self._current_run is None:
            return

        self._current_run.end(status)
        self._current_run = None

    @contextmanager
    def run(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Generator[Run, None, None]:
        """
        Context manager for runs.

        Args:
            experiment_name: Name of the experiment
            run_name: Optional run name
            tags: Initial tags
            config: Configuration to log

        Yields:
            Run object
        """
        run = self.start_run(experiment_name, run_name, tags, config)
        try:
            yield run
        except Exception:
            self.end_run(RunStatus.FAILED)
            raise
        else:
            self.end_run(RunStatus.COMPLETED)

    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter to current run."""
        if self._current_run:
            self._current_run.log_param(key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters to current run."""
        if self._current_run:
            self._current_run.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric to current run."""
        if self._current_run:
            self._current_run.log_metric(key, value, step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics to current run."""
        if self._current_run:
            self._current_run.log_metrics(metrics, step)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on current run."""
        if self._current_run:
            self._current_run.set_tag(key, value)

    def log_artifact(self, local_path: str, artifact_name: Optional[str] = None) -> Optional[str]:
        """Log an artifact to current run."""
        if self._current_run:
            return self._current_run.log_artifact(local_path, artifact_name)
        return None

    def log_model(
        self,
        model_path: str,
        model_name: str,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        register: bool = True,
    ) -> Optional[str]:
        """
        Log a model artifact and optionally register it.

        Args:
            model_path: Path to model file
            model_name: Model name for registration
            metrics: Model metrics
            params: Model parameters
            register: Whether to register in model registry

        Returns:
            Model version if registered, else artifact path
        """
        # Log as artifact
        artifact_path = self.log_artifact(model_path, f"{model_name}.pth")

        # Register in model registry
        if register:
            run_id = self._current_run.run_id if self._current_run else None
            version = self._model_registry.register(
                name=model_name,
                model_path=model_path,
                metrics=metrics,
                params=params,
                run_id=run_id,
            )
            return version.version

        return artifact_path

    def get_run(self, experiment_name: str, run_id: str) -> Optional[Run]:
        """
        Get a specific run.

        Args:
            experiment_name: Experiment name
            run_id: Run ID

        Returns:
            Run or None
        """
        run_dir = self._base_path / experiment_name / run_id
        if not run_dir.exists():
            return None

        try:
            return Run.load(run_dir)
        except Exception as e:
            logger.error(f"Failed to load run {run_id}: {e}")
            return None

    def list_experiments(self) -> List[str]:
        """List all experiment names."""
        return list(self._experiments.keys())

    def list_runs(self, experiment_name: str) -> List[str]:
        """List all run IDs for an experiment."""
        return self._experiments.get(experiment_name, [])

    def compare_runs(
        self,
        experiment_name: str,
        run_ids: Optional[List[str]] = None,
        mode: str = "max",
    ) -> ComparisonReport:
        """
        Compare runs in an experiment.

        Args:
            experiment_name: Experiment name
            run_ids: Specific runs to compare (all if None)
            mode: "max" or "min" for best run determination

        Returns:
            ComparisonReport
        """
        comparison = ExperimentComparison()

        target_runs = run_ids or self.list_runs(experiment_name)

        for run_id in target_runs:
            run = self.get_run(experiment_name, run_id)
            if run:
                comparison.add_run_from_dict(run.to_dict())

        return comparison.compare(mode)

    def delete_run(self, experiment_name: str, run_id: str) -> bool:
        """
        Delete a run.

        Args:
            experiment_name: Experiment name
            run_id: Run ID

        Returns:
            True if deleted
        """
        import shutil

        run_dir = self._base_path / experiment_name / run_id
        if not run_dir.exists():
            return False

        shutil.rmtree(run_dir)

        if experiment_name in self._experiments:
            self._experiments[experiment_name] = [
                r for r in self._experiments[experiment_name] if r != run_id
            ]
        self._save_index()

        logger.info(f"Deleted run {run_id} from {experiment_name}")
        return True

    def _get_git_info(self) -> Optional[Dict[str, str]]:
        """Get current git information."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
            )
            if result.returncode != 0:
                return None

            commit = result.stdout.strip()

            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
            )
            branch = result.stdout.strip() if result.returncode == 0 else "unknown"

            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
            )
            dirty = bool(result.stdout.strip()) if result.returncode == 0 else False

            return {
                "commit": commit,
                "branch": branch,
                "dirty": str(dirty).lower(),
            }
        except Exception:
            return None

    def _get_env_info(self) -> Dict[str, str]:
        """Get environment information."""
        import platform
        import sys

        info = {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": platform.node(),
        }

        # Check for common ML libraries
        try:
            import torch
            info["torch_version"] = torch.__version__
            info["cuda_available"] = str(torch.cuda.is_available()).lower()
        except ImportError:
            pass

        try:
            import numpy
            info["numpy_version"] = numpy.__version__
        except ImportError:
            pass

        return info

    def _save_index(self) -> None:
        """Save experiment index."""
        with open(self._index_path, "w") as f:
            json.dump(self._experiments, f, indent=2)

    def _load_index(self) -> None:
        """Load experiment index."""
        with open(self._index_path) as f:
            self._experiments = json.load(f)

    def __repr__(self) -> str:
        experiments = len(self._experiments)
        total_runs = sum(len(runs) for runs in self._experiments.values())
        return f"ExperimentTracker(experiments={experiments}, runs={total_runs})"


# Global tracker instance
_default_tracker: Optional[ExperimentTracker] = None


def get_tracker() -> Optional[ExperimentTracker]:
    """Get default experiment tracker."""
    return _default_tracker


def set_tracker(tracker: ExperimentTracker) -> None:
    """Set default experiment tracker."""
    global _default_tracker
    _default_tracker = tracker


def init_tracker(base_path: str = "experiments") -> ExperimentTracker:
    """
    Initialize and set default tracker.

    Args:
        base_path: Base path for experiment storage

    Returns:
        Initialized ExperimentTracker
    """
    config = TrackerConfig(base_path=base_path)
    tracker = ExperimentTracker(config)
    set_tracker(tracker)
    return tracker
