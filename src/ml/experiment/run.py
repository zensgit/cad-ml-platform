"""
Run management for experiment tracking.

Handles individual experiment runs with:
- Status tracking
- Parameter storage
- Metrics logging
- Artifact management
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RunStatus(str, Enum):
    """Status of an experiment run."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


@dataclass
class RunInfo:
    """Information about a run."""
    run_id: str
    experiment_name: str
    status: RunStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get run duration in seconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunInfo":
        """Create from dictionary."""
        return cls(
            run_id=data["run_id"],
            experiment_name=data["experiment_name"],
            status=RunStatus(data["status"]),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            tags=data.get("tags", {}),
        )


@dataclass
class Run:
    """
    Represents a single experiment run.

    A run tracks:
    - Parameters (hyperparameters, config)
    - Metrics (loss, accuracy, etc.)
    - Artifacts (models, plots, etc.)
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    experiment_name: str = ""
    status: RunStatus = RunStatus.CREATED
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None

    # Storage
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)

    # Internal
    _base_path: Optional[Path] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize run directory if base path is set."""
        if self._base_path:
            self._init_storage()

    def _init_storage(self) -> None:
        """Initialize storage directory."""
        if self._base_path is None:
            return

        run_dir = self._base_path / self.experiment_name / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (run_dir / "artifacts").mkdir(exist_ok=True)
        (run_dir / "metrics").mkdir(exist_ok=True)

    @property
    def run_dir(self) -> Optional[Path]:
        """Get run directory path."""
        if self._base_path is None:
            return None
        return self._base_path / self.experiment_name / self.run_id

    @property
    def info(self) -> RunInfo:
        """Get run info."""
        return RunInfo(
            run_id=self.run_id,
            experiment_name=self.experiment_name,
            status=self.status,
            start_time=self.start_time,
            end_time=self.end_time,
            tags=self.tags,
        )

    def start(self) -> None:
        """Start the run."""
        self.status = RunStatus.RUNNING
        self.start_time = datetime.utcnow()
        logger.info(f"Run {self.run_id} started for experiment {self.experiment_name}")

    def end(self, status: RunStatus = RunStatus.COMPLETED) -> None:
        """End the run."""
        self.status = status
        self.end_time = datetime.utcnow()
        self._save_metadata()
        logger.info(
            f"Run {self.run_id} ended with status {status.value}, "
            f"duration: {self.info.duration_seconds:.2f}s"
        )

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        self.params[key] = value
        logger.debug(f"Run {self.run_id}: param {key}={value}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        self.params.update(params)
        logger.debug(f"Run {self.run_id}: logged {len(params)} params")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric."""
        if key not in self.metrics:
            self.metrics[key] = []

        entry = {
            "value": value,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if step is not None:
            entry["step"] = step

        self.metrics[key].append(entry)
        logger.debug(f"Run {self.run_id}: metric {key}={value} step={step}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag."""
        self.tags[key] = value

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set multiple tags."""
        self.tags.update(tags)

    def log_artifact(self, local_path: str, artifact_name: Optional[str] = None) -> str:
        """Log an artifact (file)."""
        import shutil

        src_path = Path(local_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Artifact not found: {local_path}")

        artifact_name = artifact_name or src_path.name

        if self.run_dir:
            dst_path = self.run_dir / "artifacts" / artifact_name
            shutil.copy2(src_path, dst_path)
            artifact_path = str(dst_path)
        else:
            artifact_path = str(src_path)

        self.artifacts.append(artifact_path)
        logger.info(f"Run {self.run_id}: logged artifact {artifact_name}")
        return artifact_path

    def get_metric_history(self, key: str) -> List[Dict[str, Any]]:
        """Get history of a metric."""
        return self.metrics.get(key, [])

    def get_metric_value(self, key: str, aggregation: str = "last") -> Optional[float]:
        """Get aggregated metric value."""
        history = self.get_metric_history(key)
        if not history:
            return None

        values = [entry["value"] for entry in history]

        if aggregation == "last":
            return values[-1]
        elif aggregation == "first":
            return values[0]
        elif aggregation == "min":
            return min(values)
        elif aggregation == "max":
            return max(values)
        elif aggregation == "mean":
            return sum(values) / len(values)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert run to dictionary."""
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "params": self.params,
            "metrics": self.metrics,
            "tags": self.tags,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_path: Optional[Path] = None) -> "Run":
        """Create run from dictionary."""
        run = cls(
            run_id=data["run_id"],
            experiment_name=data["experiment_name"],
            status=RunStatus(data["status"]),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            params=data.get("params", {}),
            metrics=data.get("metrics", {}),
            tags=data.get("tags", {}),
            artifacts=data.get("artifacts", []),
            _base_path=base_path,
        )
        return run

    def _save_metadata(self) -> None:
        """Save run metadata to disk."""
        if self.run_dir is None:
            return

        # Ensure run directory exists
        self.run_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = self.run_dir / "run.json"
        with open(metadata_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, run_dir: Path) -> "Run":
        """Load run from directory."""
        metadata_path = run_dir / "run.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Run metadata not found: {metadata_path}")

        with open(metadata_path) as f:
            data = json.load(f)

        base_path = run_dir.parent.parent
        return cls.from_dict(data, base_path)


class RunContext:
    """Context manager for runs."""

    def __init__(self, run: Run):
        self._run = run

    def __enter__(self) -> Run:
        self._run.start()
        return self._run

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._run.end(RunStatus.FAILED)
        else:
            self._run.end(RunStatus.COMPLETED)
        return False
