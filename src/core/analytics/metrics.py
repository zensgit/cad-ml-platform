"""Metrics Aggregation.

Provides pre-aggregated metrics for dashboards:
- Counter, gauge, histogram metrics
- Dimensional aggregation
- Rate calculations
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """A metric value with labels."""
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HistogramBucket:
    """A histogram bucket."""
    le: float  # Less than or equal
    count: int


@dataclass
class HistogramValue:
    """Histogram metric value."""
    buckets: List[HistogramBucket]
    sum: float
    count: int
    labels: Dict[str, str] = field(default_factory=dict)


class Metric(ABC):
    """Base class for metrics."""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazily create lock."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _labels_key(self, labels: Dict[str, str]) -> str:
        """Create key from labels."""
        return "|".join(f"{k}={labels.get(k, '')}" for k in self.label_names)

    @abstractmethod
    def collect(self) -> List[MetricValue]:
        """Collect current metric values."""
        pass


class Counter(Metric):
    """A monotonically increasing counter."""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ):
        super().__init__(name, description, labels)
        self._values: Dict[str, float] = defaultdict(float)
        self._labels_map: Dict[str, Dict[str, str]] = {}

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the counter."""
        labels = labels or {}
        key = self._labels_key(labels)
        self._values[key] += value
        self._labels_map[key] = labels

    def labels(self, **labels: str) -> "Counter":
        """Return a counter with labels pre-set."""
        return _LabeledCounter(self, labels)

    def collect(self) -> List[MetricValue]:
        return [
            MetricValue(value=value, labels=self._labels_map.get(key, {}))
            for key, value in self._values.items()
        ]


class _LabeledCounter:
    """Counter with pre-set labels."""

    def __init__(self, counter: Counter, labels: Dict[str, str]):
        self._counter = counter
        self._labels = labels

    def inc(self, value: float = 1.0) -> None:
        self._counter.inc(value, self._labels)


class Gauge(Metric):
    """A gauge that can increase and decrease."""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ):
        super().__init__(name, description, labels)
        self._values: Dict[str, float] = defaultdict(float)
        self._labels_map: Dict[str, Dict[str, str]] = {}

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set the gauge value."""
        labels = labels or {}
        key = self._labels_key(labels)
        self._values[key] = value
        self._labels_map[key] = labels

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the gauge."""
        labels = labels or {}
        key = self._labels_key(labels)
        self._values[key] += value
        self._labels_map[key] = labels

    def dec(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement the gauge."""
        self.inc(-value, labels)

    def labels(self, **labels: str) -> "Gauge":
        """Return a gauge with labels pre-set."""
        return _LabeledGauge(self, labels)

    def collect(self) -> List[MetricValue]:
        return [
            MetricValue(value=value, labels=self._labels_map.get(key, {}))
            for key, value in self._values.items()
        ]


class _LabeledGauge:
    """Gauge with pre-set labels."""

    def __init__(self, gauge: Gauge, labels: Dict[str, str]):
        self._gauge = gauge
        self._labels = labels

    def set(self, value: float) -> None:
        self._gauge.set(value, self._labels)

    def inc(self, value: float = 1.0) -> None:
        self._gauge.inc(value, self._labels)

    def dec(self, value: float = 1.0) -> None:
        self._gauge.dec(value, self._labels)


class Histogram(Metric):
    """A histogram with configurable buckets."""

    DEFAULT_BUCKETS = [
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0,
        2.5, 5.0, 7.5, 10.0, float('inf')
    ]

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
    ):
        super().__init__(name, description, labels)
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        if self.buckets[-1] != float('inf'):
            self.buckets.append(float('inf'))

        self._bucket_counts: Dict[str, List[int]] = defaultdict(
            lambda: [0] * len(self.buckets)
        )
        self._sums: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)
        self._labels_map: Dict[str, Dict[str, str]] = {}

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a value observation."""
        labels = labels or {}
        key = self._labels_key(labels)
        self._labels_map[key] = labels

        self._sums[key] += value
        self._counts[key] += 1

        for i, bucket in enumerate(self.buckets):
            if value <= bucket:
                self._bucket_counts[key][i] += 1

    def labels(self, **labels: str) -> "Histogram":
        """Return a histogram with labels pre-set."""
        return _LabeledHistogram(self, labels)

    def time(self, labels: Optional[Dict[str, str]] = None) -> "_HistogramTimer":
        """Context manager for timing."""
        return _HistogramTimer(self, labels)

    def collect(self) -> List[HistogramValue]:
        result = []
        for key in self._bucket_counts.keys():
            buckets = [
                HistogramBucket(le=le, count=count)
                for le, count in zip(self.buckets, self._bucket_counts[key])
            ]
            result.append(HistogramValue(
                buckets=buckets,
                sum=self._sums[key],
                count=self._counts[key],
                labels=self._labels_map.get(key, {}),
            ))
        return result


class _LabeledHistogram:
    """Histogram with pre-set labels."""

    def __init__(self, histogram: Histogram, labels: Dict[str, str]):
        self._histogram = histogram
        self._labels = labels

    def observe(self, value: float) -> None:
        self._histogram.observe(value, self._labels)

    def time(self) -> "_HistogramTimer":
        return _HistogramTimer(self._histogram, self._labels)


class _HistogramTimer:
    """Context manager for timing operations."""

    def __init__(self, histogram: Histogram, labels: Optional[Dict[str, str]]):
        self._histogram = histogram
        self._labels = labels
        self._start: Optional[float] = None

    def __enter__(self) -> "_HistogramTimer":
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start is not None:
            duration = time.time() - self._start
            self._histogram.observe(duration, self._labels)


class MetricRegistry:
    """Registry for all metrics."""

    def __init__(self):
        self._metrics: Dict[str, Metric] = {}

    def register(self, metric: Metric) -> Metric:
        """Register a metric."""
        if metric.name in self._metrics:
            raise ValueError(f"Metric already registered: {metric.name}")
        self._metrics[metric.name] = metric
        return metric

    def counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> Counter:
        """Create and register a counter."""
        metric = Counter(name, description, labels)
        return self.register(metric)

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> Gauge:
        """Create and register a gauge."""
        metric = Gauge(name, description, labels)
        return self.register(metric)

    def histogram(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> Histogram:
        """Create and register a histogram."""
        metric = Histogram(name, description, labels, buckets)
        return self.register(metric)

    def get(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        return self._metrics.get(name)

    def collect_all(self) -> Dict[str, List[Any]]:
        """Collect all metrics."""
        return {
            name: metric.collect()
            for name, metric in self._metrics.items()
        }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for name, metric in self._metrics.items():
            # Type line
            if isinstance(metric, Counter):
                lines.append(f"# TYPE {name} counter")
            elif isinstance(metric, Gauge):
                lines.append(f"# TYPE {name} gauge")
            elif isinstance(metric, Histogram):
                lines.append(f"# TYPE {name} histogram")

            # Help line
            if metric.description:
                lines.append(f"# HELP {name} {metric.description}")

            # Values
            values = metric.collect()
            for value in values:
                if isinstance(value, MetricValue):
                    labels_str = ",".join(
                        f'{k}="{v}"' for k, v in value.labels.items()
                    )
                    if labels_str:
                        lines.append(f"{name}{{{labels_str}}} {value.value}")
                    else:
                        lines.append(f"{name} {value.value}")

                elif isinstance(value, HistogramValue):
                    labels_str = ",".join(
                        f'{k}="{v}"' for k, v in value.labels.items()
                    )

                    # Buckets
                    for bucket in value.buckets:
                        le_str = "+Inf" if bucket.le == float('inf') else str(bucket.le)
                        if labels_str:
                            lines.append(
                                f'{name}_bucket{{{labels_str},le="{le_str}"}} {bucket.count}'
                            )
                        else:
                            lines.append(f'{name}_bucket{{le="{le_str}"}} {bucket.count}')

                    # Sum and count
                    if labels_str:
                        lines.append(f"{name}_sum{{{labels_str}}} {value.sum}")
                        lines.append(f"{name}_count{{{labels_str}}} {value.count}")
                    else:
                        lines.append(f"{name}_sum {value.sum}")
                        lines.append(f"{name}_count {value.count}")

        return "\n".join(lines)


# Global metric registry
_registry: Optional[MetricRegistry] = None


def get_registry() -> MetricRegistry:
    """Get the global metric registry."""
    global _registry
    if _registry is None:
        _registry = MetricRegistry()
    return _registry
