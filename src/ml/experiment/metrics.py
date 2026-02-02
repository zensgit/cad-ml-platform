"""
Metrics logging for experiment tracking.

Provides:
- Metric recording with timestamps
- Step-based metric logging
- Metric aggregation utilities
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Type of metric."""
    SCALAR = "scalar"
    HISTOGRAM = "histogram"
    IMAGE = "image"
    TEXT = "text"


@dataclass
class Metric:
    """A single metric entry."""
    name: str
    value: Union[float, int, str, List[float]]
    step: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metric_type: MetricType = MetricType.SCALAR
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "step": self.step,
            "timestamp": self.timestamp.isoformat(),
            "metric_type": self.metric_type.value,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Metric":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            value=data["value"],
            step=data.get("step"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.utcnow(),
            metric_type=MetricType(data.get("metric_type", "scalar")),
            tags=data.get("tags", {}),
        )


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    name: str
    count: int
    min_value: float
    max_value: float
    mean_value: float
    last_value: float
    first_value: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "count": self.count,
            "min": self.min_value,
            "max": self.max_value,
            "mean": self.mean_value,
            "last": self.last_value,
            "first": self.first_value,
        }


class MetricsLogger:
    """
    Logger for experiment metrics.

    Supports:
    - Scalar metrics with optional steps
    - Metric history tracking
    - File-based persistence
    - Summary statistics
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize metrics logger.

        Args:
            storage_path: Path to store metrics (optional)
        """
        self._storage_path = storage_path
        self._metrics: Dict[str, List[Metric]] = {}

    @property
    def metric_names(self) -> List[str]:
        """Get all metric names."""
        return list(self._metrics.keys())

    def log(
        self,
        name: str,
        value: Union[float, int],
        step: Optional[int] = None,
        metric_type: MetricType = MetricType.SCALAR,
        tags: Optional[Dict[str, str]] = None,
    ) -> Metric:
        """
        Log a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
            metric_type: Type of metric
            tags: Optional tags

        Returns:
            The logged metric
        """
        metric = Metric(
            name=name,
            value=value,
            step=step,
            metric_type=metric_type,
            tags=tags or {},
        )

        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append(metric)

        logger.debug(f"Logged metric: {name}={value} step={step}")
        return metric

    def log_batch(
        self,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[Metric]:
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric name to value
            step: Optional step number
            tags: Optional tags to apply to all metrics

        Returns:
            List of logged metrics
        """
        logged = []
        for name, value in metrics.items():
            metric = self.log(name, value, step, tags=tags)
            logged.append(metric)
        return logged

    def get_history(self, name: str) -> List[Metric]:
        """
        Get history of a metric.

        Args:
            name: Metric name

        Returns:
            List of metric entries
        """
        return self._metrics.get(name, [])

    def get_values(self, name: str) -> List[float]:
        """
        Get values of a metric.

        Args:
            name: Metric name

        Returns:
            List of metric values
        """
        return [m.value for m in self.get_history(name)]

    def get_steps(self, name: str) -> List[Optional[int]]:
        """
        Get steps of a metric.

        Args:
            name: Metric name

        Returns:
            List of step numbers
        """
        return [m.step for m in self.get_history(name)]

    def get_latest(self, name: str) -> Optional[Metric]:
        """
        Get latest metric entry.

        Args:
            name: Metric name

        Returns:
            Latest metric or None
        """
        history = self.get_history(name)
        return history[-1] if history else None

    def get_best(self, name: str, mode: str = "min") -> Optional[Metric]:
        """
        Get best metric entry.

        Args:
            name: Metric name
            mode: "min" or "max"

        Returns:
            Best metric or None
        """
        history = self.get_history(name)
        if not history:
            return None

        if mode == "min":
            return min(history, key=lambda m: m.value)
        elif mode == "max":
            return max(history, key=lambda m: m.value)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_summary(self, name: str) -> Optional[MetricSummary]:
        """
        Get summary statistics for a metric.

        Args:
            name: Metric name

        Returns:
            MetricSummary or None
        """
        history = self.get_history(name)
        if not history:
            return None

        values = [m.value for m in history]
        return MetricSummary(
            name=name,
            count=len(values),
            min_value=min(values),
            max_value=max(values),
            mean_value=sum(values) / len(values),
            last_value=values[-1],
            first_value=values[0],
        )

    def get_all_summaries(self) -> Dict[str, MetricSummary]:
        """
        Get summaries for all metrics.

        Returns:
            Dictionary of metric name to summary
        """
        summaries = {}
        for name in self.metric_names:
            summary = self.get_summary(name)
            if summary:
                summaries[name] = summary
        return summaries

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Convert all metrics to dictionary."""
        return {
            name: [m.to_dict() for m in metrics]
            for name, metrics in self._metrics.items()
        }

    def save(self, path: Optional[Path] = None) -> None:
        """
        Save metrics to file.

        Args:
            path: File path (uses storage_path if not provided)
        """
        save_path = path or self._storage_path
        if save_path is None:
            raise ValueError("No path specified for saving metrics")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved metrics to {save_path}")

    @classmethod
    def load(cls, path: Path) -> "MetricsLogger":
        """
        Load metrics from file.

        Args:
            path: File path

        Returns:
            MetricsLogger instance
        """
        with open(path) as f:
            data = json.load(f)

        logger_instance = cls(storage_path=path)
        for name, metrics_data in data.items():
            logger_instance._metrics[name] = [
                Metric.from_dict(m) for m in metrics_data
            ]

        return logger_instance

    def clear(self) -> None:
        """Clear all metrics."""
        self._metrics.clear()

    def __len__(self) -> int:
        """Get total number of metric entries."""
        return sum(len(metrics) for metrics in self._metrics.values())

    def __repr__(self) -> str:
        return f"MetricsLogger(metrics={self.metric_names}, entries={len(self)})"
