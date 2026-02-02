"""Metrics Aggregator Core.

Provides metric types and collection:
- Counter, Gauge, Histogram
- Labels and dimensions
- Time series data
"""

from __future__ import annotations

import statistics
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricLabels:
    """Labels for a metric."""
    labels: Dict[str, str] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.labels.items())))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MetricLabels):
            return False
        return self.labels == other.labels

    def to_string(self) -> str:
        if not self.labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in sorted(self.labels.items())]
        return "{" + ",".join(parts) + "}"


@dataclass
class MetricValue:
    """A metric value at a point in time."""
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: MetricLabels = field(default_factory=MetricLabels)


@dataclass
class MetricMetadata:
    """Metadata for a metric."""
    name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    label_names: List[str] = field(default_factory=list)


class Metric(ABC):
    """Abstract base class for metrics."""

    def __init__(
        self,
        name: str,
        description: str = "",
        label_names: Optional[List[str]] = None,
    ):
        self._name = name
        self._description = description
        self._label_names = label_names or []
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    @abstractmethod
    def metric_type(self) -> MetricType:
        pass

    @abstractmethod
    def collect(self) -> List[MetricValue]:
        """Collect current metric values."""
        pass

    def labels(self, **kwargs: str) -> "LabeledMetric":
        """Return metric with specific labels."""
        return LabeledMetric(self, MetricLabels(kwargs))

    def _validate_labels(self, labels: MetricLabels) -> None:
        """Validate label names."""
        if set(labels.labels.keys()) != set(self._label_names):
            expected = set(self._label_names)
            got = set(labels.labels.keys())
            raise ValueError(f"Label mismatch: expected {expected}, got {got}")


class LabeledMetric:
    """A metric with specific labels."""

    def __init__(self, metric: Metric, labels: MetricLabels):
        self._metric = metric
        self._labels = labels

    def inc(self, value: float = 1) -> None:
        if hasattr(self._metric, '_inc'):
            self._metric._inc(self._labels, value)

    def dec(self, value: float = 1) -> None:
        if hasattr(self._metric, '_dec'):
            self._metric._dec(self._labels, value)

    def set(self, value: float) -> None:
        if hasattr(self._metric, '_set'):
            self._metric._set(self._labels, value)

    def observe(self, value: float) -> None:
        if hasattr(self._metric, '_observe'):
            self._metric._observe(self._labels, value)


class Counter(Metric):
    """Counter metric (monotonically increasing)."""

    def __init__(
        self,
        name: str,
        description: str = "",
        label_names: Optional[List[str]] = None,
    ):
        super().__init__(name, description, label_names)
        self._values: Dict[MetricLabels, float] = {}

    @property
    def metric_type(self) -> MetricType:
        return MetricType.COUNTER

    def inc(self, value: float = 1) -> None:
        """Increment counter."""
        self._inc(MetricLabels(), value)

    def _inc(self, labels: MetricLabels, value: float = 1) -> None:
        if value < 0:
            raise ValueError("Counter can only be incremented")

        with self._lock:
            current = self._values.get(labels, 0)
            self._values[labels] = current + value

    def collect(self) -> List[MetricValue]:
        with self._lock:
            return [
                MetricValue(value=v, labels=l)
                for l, v in self._values.items()
            ]


class Gauge(Metric):
    """Gauge metric (can go up and down)."""

    def __init__(
        self,
        name: str,
        description: str = "",
        label_names: Optional[List[str]] = None,
    ):
        super().__init__(name, description, label_names)
        self._values: Dict[MetricLabels, float] = {}

    @property
    def metric_type(self) -> MetricType:
        return MetricType.GAUGE

    def set(self, value: float) -> None:
        """Set gauge value."""
        self._set(MetricLabels(), value)

    def _set(self, labels: MetricLabels, value: float) -> None:
        with self._lock:
            self._values[labels] = value

    def inc(self, value: float = 1) -> None:
        """Increment gauge."""
        self._inc(MetricLabels(), value)

    def _inc(self, labels: MetricLabels, value: float = 1) -> None:
        with self._lock:
            current = self._values.get(labels, 0)
            self._values[labels] = current + value

    def dec(self, value: float = 1) -> None:
        """Decrement gauge."""
        self._dec(MetricLabels(), value)

    def _dec(self, labels: MetricLabels, value: float = 1) -> None:
        with self._lock:
            current = self._values.get(labels, 0)
            self._values[labels] = current - value

    def collect(self) -> List[MetricValue]:
        with self._lock:
            return [
                MetricValue(value=v, labels=l)
                for l, v in self._values.items()
            ]


class Histogram(Metric):
    """Histogram metric for distributions."""

    DEFAULT_BUCKETS = (
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5,
        0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float('inf'),
    )

    def __init__(
        self,
        name: str,
        description: str = "",
        label_names: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ):
        super().__init__(name, description, label_names)
        self._buckets = buckets or self.DEFAULT_BUCKETS
        # Per-label data: {labels: {bucket: count}}
        self._bucket_counts: Dict[MetricLabels, Dict[float, int]] = {}
        self._sums: Dict[MetricLabels, float] = {}
        self._counts: Dict[MetricLabels, int] = {}

    @property
    def metric_type(self) -> MetricType:
        return MetricType.HISTOGRAM

    def observe(self, value: float) -> None:
        """Observe a value."""
        self._observe(MetricLabels(), value)

    def _observe(self, labels: MetricLabels, value: float) -> None:
        with self._lock:
            # Initialize if needed
            if labels not in self._bucket_counts:
                self._bucket_counts[labels] = {b: 0 for b in self._buckets}
                self._sums[labels] = 0
                self._counts[labels] = 0

            # Update buckets
            for bucket in self._buckets:
                if value <= bucket:
                    self._bucket_counts[labels][bucket] += 1

            self._sums[labels] += value
            self._counts[labels] += 1

    def collect(self) -> List[MetricValue]:
        results = []

        with self._lock:
            for labels in self._bucket_counts:
                # Bucket values
                for bucket, count in self._bucket_counts[labels].items():
                    bucket_labels = MetricLabels({
                        **labels.labels,
                        "le": str(bucket) if bucket != float('inf') else "+Inf",
                    })
                    results.append(MetricValue(
                        value=count,
                        labels=bucket_labels,
                    ))

                # Sum
                sum_labels = MetricLabels({**labels.labels, "_type": "sum"})
                results.append(MetricValue(
                    value=self._sums[labels],
                    labels=sum_labels,
                ))

                # Count
                count_labels = MetricLabels({**labels.labels, "_type": "count"})
                results.append(MetricValue(
                    value=self._counts[labels],
                    labels=count_labels,
                ))

        return results

    def time(self) -> "Timer":
        """Return a timer context manager."""
        return Timer(self)


class Summary(Metric):
    """Summary metric for percentiles."""

    def __init__(
        self,
        name: str,
        description: str = "",
        label_names: Optional[List[str]] = None,
        quantiles: Optional[Tuple[float, ...]] = None,
        max_samples: int = 1000,
    ):
        super().__init__(name, description, label_names)
        self._quantiles = quantiles or (0.5, 0.9, 0.95, 0.99)
        self._max_samples = max_samples
        self._samples: Dict[MetricLabels, List[float]] = {}
        self._sums: Dict[MetricLabels, float] = {}
        self._counts: Dict[MetricLabels, int] = {}

    @property
    def metric_type(self) -> MetricType:
        return MetricType.SUMMARY

    def observe(self, value: float) -> None:
        """Observe a value."""
        self._observe(MetricLabels(), value)

    def _observe(self, labels: MetricLabels, value: float) -> None:
        with self._lock:
            if labels not in self._samples:
                self._samples[labels] = []
                self._sums[labels] = 0
                self._counts[labels] = 0

            samples = self._samples[labels]
            samples.append(value)

            # Keep only recent samples
            if len(samples) > self._max_samples:
                self._samples[labels] = samples[-self._max_samples:]

            self._sums[labels] += value
            self._counts[labels] += 1

    def collect(self) -> List[MetricValue]:
        results = []

        with self._lock:
            for labels in self._samples:
                samples = sorted(self._samples[labels])
                if not samples:
                    continue

                # Quantile values
                for q in self._quantiles:
                    idx = int(len(samples) * q)
                    idx = min(idx, len(samples) - 1)
                    q_labels = MetricLabels({
                        **labels.labels,
                        "quantile": str(q),
                    })
                    results.append(MetricValue(
                        value=samples[idx],
                        labels=q_labels,
                    ))

                # Sum and count
                sum_labels = MetricLabels({**labels.labels, "_type": "sum"})
                results.append(MetricValue(
                    value=self._sums[labels],
                    labels=sum_labels,
                ))

                count_labels = MetricLabels({**labels.labels, "_type": "count"})
                results.append(MetricValue(
                    value=self._counts[labels],
                    labels=count_labels,
                ))

        return results


class Timer:
    """Timer context manager for histograms."""

    def __init__(self, histogram: Histogram, labels: Optional[MetricLabels] = None):
        self._histogram = histogram
        self._labels = labels or MetricLabels()
        self._start: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start is not None:
            duration = time.time() - self._start
            self._histogram._observe(self._labels, duration)


@dataclass
class MetricSample:
    """A sample of metric data for export."""
    name: str
    metric_type: MetricType
    description: str
    values: List[MetricValue]
    timestamp: datetime = field(default_factory=datetime.utcnow)
