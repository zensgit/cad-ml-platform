"""
Real-time metrics collection for model monitoring.

Collects and tracks model performance metrics in production.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"  # Cumulative count
    GAUGE = "gauge"  # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution
    SUMMARY = "summary"  # Statistical summary


@dataclass
class MetricValue:
    """A single metric observation."""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class HistogramBucket:
    """A histogram bucket."""
    upper_bound: float
    count: int


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    name: str
    count: int
    sum: float
    mean: float
    std: float
    min: float
    max: float
    percentiles: Dict[int, float]  # p50, p90, p95, p99
    buckets: Optional[List[HistogramBucket]] = None


class Counter:
    """A monotonically increasing counter."""

    def __init__(self, name: str, labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.labels = labels or {}
        self._value: float = 0.0
        self._lock = threading.Lock()

    def inc(self, value: float = 1.0) -> None:
        """Increment counter."""
        if value < 0:
            raise ValueError("Counter increment must be non-negative")
        with self._lock:
            self._value += value

    def get(self) -> float:
        """Get current value."""
        with self._lock:
            return self._value

    def reset(self) -> None:
        """Reset counter (use with caution)."""
        with self._lock:
            self._value = 0.0


class Gauge:
    """A metric that can go up and down."""

    def __init__(self, name: str, labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.labels = labels or {}
        self._value: float = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        """Set gauge value."""
        with self._lock:
            self._value = value

    def inc(self, value: float = 1.0) -> None:
        """Increment gauge."""
        with self._lock:
            self._value += value

    def dec(self, value: float = 1.0) -> None:
        """Decrement gauge."""
        with self._lock:
            self._value -= value

    def get(self) -> float:
        """Get current value."""
        with self._lock:
            return self._value


class Histogram:
    """Distribution of values with configurable buckets."""

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ):
        self.name = name
        self.labels = labels or {}
        self._buckets = buckets or self.DEFAULT_BUCKETS
        self._bucket_counts: List[int] = [0] * len(self._buckets)
        self._sum: float = 0.0
        self._count: int = 0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        """Record an observation."""
        with self._lock:
            self._sum += value
            self._count += 1
            for i, bound in enumerate(self._buckets):
                if value <= bound:
                    self._bucket_counts[i] += 1

    def get_summary(self) -> MetricSummary:
        """Get histogram summary."""
        with self._lock:
            buckets = [
                HistogramBucket(upper_bound=b, count=c)
                for b, c in zip(self._buckets, self._bucket_counts)
            ]
            mean = self._sum / self._count if self._count > 0 else 0.0
            return MetricSummary(
                name=self.name,
                count=self._count,
                sum=self._sum,
                mean=mean,
                std=0.0,  # Would need to store values for this
                min=0.0,
                max=0.0,
                percentiles={},
                buckets=buckets,
            )


class SlidingWindowMetric:
    """Metric with sliding time window."""

    def __init__(
        self,
        name: str,
        window_seconds: float = 60.0,
        max_samples: int = 10000,
    ):
        self.name = name
        self.window_seconds = window_seconds
        self.max_samples = max_samples
        self._values: Deque[Tuple[float, float]] = deque(maxlen=max_samples)
        self._lock = threading.Lock()

    def add(self, value: float, timestamp: Optional[float] = None) -> None:
        """Add a value."""
        ts = timestamp or time.time()
        with self._lock:
            self._values.append((ts, value))
            self._prune_old()

    def _prune_old(self) -> None:
        """Remove values outside the window."""
        cutoff = time.time() - self.window_seconds
        while self._values and self._values[0][0] < cutoff:
            self._values.popleft()

    def get_values(self) -> List[float]:
        """Get current window values."""
        with self._lock:
            self._prune_old()
            return [v for _, v in self._values]

    def get_summary(self) -> MetricSummary:
        """Get summary statistics."""
        values = self.get_values()
        if not values:
            return MetricSummary(
                name=self.name,
                count=0,
                sum=0.0,
                mean=0.0,
                std=0.0,
                min=0.0,
                max=0.0,
                percentiles={50: 0, 90: 0, 95: 0, 99: 0},
            )

        arr = np.array(values)
        return MetricSummary(
            name=self.name,
            count=len(values),
            sum=float(np.sum(arr)),
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            percentiles={
                50: float(np.percentile(arr, 50)),
                90: float(np.percentile(arr, 90)),
                95: float(np.percentile(arr, 95)),
                99: float(np.percentile(arr, 99)),
            },
        )


class MetricsCollector:
    """
    Central metrics collection and management.

    Supports multiple metric types and label-based filtering.
    """

    def __init__(self):
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._sliding_windows: Dict[str, SlidingWindowMetric] = {}
        self._lock = threading.Lock()

        # Standard model metrics
        self._init_standard_metrics()

    def _init_standard_metrics(self) -> None:
        """Initialize standard model monitoring metrics."""
        # Request metrics
        self.counter("model_requests_total", {"status": "success"})
        self.counter("model_requests_total", {"status": "error"})

        # Latency metrics
        self.histogram("model_latency_seconds")
        self.sliding_window("model_latency_seconds_window")

        # Prediction metrics
        self.sliding_window("prediction_confidence")
        self.counter("predictions_by_class")

        # Resource metrics
        self.gauge("model_memory_bytes")
        self.gauge("model_loaded")

    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create unique key for metric."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> Counter:
        """Get or create a counter."""
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._counters:
                self._counters[key] = Counter(name, labels)
            return self._counters[key]

    def gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> Gauge:
        """Get or create a gauge."""
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._gauges:
                self._gauges[key] = Gauge(name, labels)
            return self._gauges[key]

    def histogram(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ) -> Histogram:
        """Get or create a histogram."""
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = Histogram(name, labels, buckets)
            return self._histograms[key]

    def sliding_window(
        self,
        name: str,
        window_seconds: float = 60.0,
    ) -> SlidingWindowMetric:
        """Get or create a sliding window metric."""
        with self._lock:
            if name not in self._sliding_windows:
                self._sliding_windows[name] = SlidingWindowMetric(name, window_seconds)
            return self._sliding_windows[name]

    def record_prediction(
        self,
        latency_seconds: float,
        success: bool = True,
        confidence: float = 0.0,
        label: Optional[str] = None,
    ) -> None:
        """Record a prediction event."""
        status = "success" if success else "error"
        self.counter("model_requests_total", {"status": status}).inc()

        self.histogram("model_latency_seconds").observe(latency_seconds)
        self.sliding_window("model_latency_seconds_window").add(latency_seconds)

        if confidence > 0:
            self.sliding_window("prediction_confidence").add(confidence)

        if label:
            self.counter("predictions_by_class", {"class": label}).inc()

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metric values."""
        result = {
            "counters": {},
            "gauges": {},
            "histograms": {},
            "sliding_windows": {},
        }

        with self._lock:
            for key, counter in self._counters.items():
                result["counters"][key] = counter.get()

            for key, gauge in self._gauges.items():
                result["gauges"][key] = gauge.get()

            for key, hist in self._histograms.items():
                result["histograms"][key] = hist.get_summary().__dict__

            for key, sw in self._sliding_windows.items():
                result["sliding_windows"][key] = sw.get_summary().__dict__

        return result

    def get_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        with self._lock:
            # Counters
            for key, counter in self._counters.items():
                lines.append(f"# TYPE {counter.name} counter")
                lines.append(f"{key} {counter.get()}")

            # Gauges
            for key, gauge in self._gauges.items():
                lines.append(f"# TYPE {gauge.name} gauge")
                lines.append(f"{key} {gauge.get()}")

            # Histograms
            for key, hist in self._histograms.items():
                summary = hist.get_summary()
                lines.append(f"# TYPE {hist.name} histogram")
                if summary.buckets:
                    for bucket in summary.buckets:
                        lines.append(f'{hist.name}_bucket{{le="{bucket.upper_bound}"}} {bucket.count}')
                lines.append(f"{hist.name}_sum {summary.sum}")
                lines.append(f"{hist.name}_count {summary.count}")

        return "\n".join(lines)


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
