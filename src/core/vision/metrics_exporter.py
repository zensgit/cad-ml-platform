"""Metrics exporter for vision providers.

Provides:
- Prometheus-compatible metrics
- StatsD export
- Custom metric backends
- Real-time monitoring support
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .base import VisionDescription, VisionProvider

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricLabel:
    """A label for a metric."""

    name: str
    value: str


@dataclass
class MetricValue:
    """A single metric value."""

    name: str
    metric_type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    help_text: str = ""

    def to_prometheus(self) -> str:
        """Format as Prometheus text format."""
        label_str = ""
        if self.labels:
            pairs = [f'{k}="{v}"' for k, v in self.labels.items()]
            label_str = "{" + ",".join(pairs) + "}"

        return f"{self.name}{label_str} {self.value}"


@dataclass
class HistogramValue:
    """A histogram metric with buckets."""

    name: str
    count: int = 0
    sum: float = 0.0
    buckets: Dict[float, int] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    help_text: str = ""

    def observe(self, value: float) -> None:
        """Observe a value."""
        self.count += 1
        self.sum += value
        for bucket in self.buckets:
            if value <= bucket:
                self.buckets[bucket] += 1

    def to_prometheus(self) -> List[str]:
        """Format as Prometheus text format."""
        lines = []
        label_str = ""
        if self.labels:
            pairs = [f'{k}="{v}"' for k, v in self.labels.items()]
            label_str = ",".join(pairs) + ","

        for bucket, count in sorted(self.buckets.items()):
            bucket_label = f'{label_str}le="{bucket}"'
            lines.append(f"{self.name}_bucket{{{bucket_label}}} {count}")

        lines.append(f'{self.name}_bucket{{{label_str}le="+Inf"}} {self.count}')
        lines.append(f"{self.name}_sum{{{label_str[:-1]}}} {self.sum}")
        lines.append(f"{self.name}_count{{{label_str[:-1]}}} {self.count}")

        return lines


class MetricBackend(ABC):
    """Abstract base class for metric backends."""

    @abstractmethod
    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter."""
        pass

    @abstractmethod
    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge value."""
        pass

    @abstractmethod
    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Observe a histogram value."""
        pass

    @abstractmethod
    def export(self) -> str:
        """Export all metrics."""
        pass


class InMemoryMetricBackend(MetricBackend):
    """In-memory metric storage for testing and simple deployments."""

    def __init__(
        self,
        histogram_buckets: Optional[List[float]] = None,
    ):
        """
        Initialize backend.

        Args:
            histogram_buckets: Bucket boundaries for histograms
        """
        self._counters: Dict[str, MetricValue] = {}
        self._gauges: Dict[str, MetricValue] = {}
        self._histograms: Dict[str, HistogramValue] = {}
        self._histogram_buckets = histogram_buckets or [
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
        ]

    def _key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Generate unique key for metric."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter."""
        key = self._key(name, labels)
        if key not in self._counters:
            self._counters[key] = MetricValue(
                name=name,
                metric_type=MetricType.COUNTER,
                value=0.0,
                labels=labels or {},
            )
        self._counters[key].value += value

    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge value."""
        key = self._key(name, labels)
        self._gauges[key] = MetricValue(
            name=name,
            metric_type=MetricType.GAUGE,
            value=value,
            labels=labels or {},
        )

    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Observe a histogram value."""
        key = self._key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = HistogramValue(
                name=name,
                buckets={b: 0 for b in self._histogram_buckets},
                labels=labels or {},
            )
        self._histograms[key].observe(value)

    def export(self) -> str:
        """Export all metrics in Prometheus format."""
        lines = []

        # Export counters
        for metric in self._counters.values():
            lines.append(f"# TYPE {metric.name} counter")
            lines.append(metric.to_prometheus())

        # Export gauges
        for metric in self._gauges.values():
            lines.append(f"# TYPE {metric.name} gauge")
            lines.append(metric.to_prometheus())

        # Export histograms
        for metric in self._histograms.values():
            lines.append(f"# TYPE {metric.name} histogram")
            lines.extend(metric.to_prometheus())

        return "\n".join(lines)

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get counter value."""
        key = self._key(name, labels)
        return self._counters.get(key, MetricValue(name, MetricType.COUNTER, 0)).value

    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get gauge value."""
        key = self._key(name, labels)
        return self._gauges.get(key, MetricValue(name, MetricType.GAUGE, 0)).value

    def reset(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


class CallbackMetricBackend(MetricBackend):
    """Backend that calls external callbacks for each metric."""

    def __init__(
        self,
        on_increment: Optional[Callable[[str, float, Dict[str, str]], None]] = None,
        on_gauge: Optional[Callable[[str, float, Dict[str, str]], None]] = None,
        on_histogram: Optional[Callable[[str, float, Dict[str, str]], None]] = None,
    ):
        """
        Initialize callback backend.

        Args:
            on_increment: Callback for counter increments
            on_gauge: Callback for gauge updates
            on_histogram: Callback for histogram observations
        """
        self._on_increment = on_increment
        self._on_gauge = on_gauge
        self._on_histogram = on_histogram
        self._inner = InMemoryMetricBackend()

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment counter and call callback."""
        self._inner.increment(name, value, labels)
        if self._on_increment:
            try:
                self._on_increment(name, value, labels or {})
            except Exception as e:
                logger.warning(f"Metric callback failed: {e}")

    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set gauge and call callback."""
        self._inner.gauge(name, value, labels)
        if self._on_gauge:
            try:
                self._on_gauge(name, value, labels or {})
            except Exception as e:
                logger.warning(f"Metric callback failed: {e}")

    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Observe histogram and call callback."""
        self._inner.histogram(name, value, labels)
        if self._on_histogram:
            try:
                self._on_histogram(name, value, labels or {})
            except Exception as e:
                logger.warning(f"Metric callback failed: {e}")

    def export(self) -> str:
        """Export metrics."""
        return self._inner.export()


@dataclass
class MetricsConfig:
    """Configuration for metrics exporter."""

    # Metric naming
    prefix: str = "vision"
    include_provider_label: bool = True
    include_operation_label: bool = True

    # Collection options
    collect_request_count: bool = True
    collect_request_duration: bool = True
    collect_error_count: bool = True
    collect_active_requests: bool = True
    collect_image_size: bool = True
    collect_confidence: bool = True

    # Histogram buckets for duration (in seconds)
    duration_buckets: List[float] = field(
        default_factory=lambda: [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
    )

    # Histogram buckets for image size (in KB)
    size_buckets: List[float] = field(
        default_factory=lambda: [10, 50, 100, 500, 1000, 5000, 10000]
    )


class MetricsExporter:
    """
    Metrics exporter for vision operations.

    Collects and exports metrics in various formats.
    """

    def __init__(
        self,
        config: Optional[MetricsConfig] = None,
        backend: Optional[MetricBackend] = None,
    ):
        """
        Initialize metrics exporter.

        Args:
            config: Metrics configuration
            backend: Metric backend for storage/export
        """
        self._config = config or MetricsConfig()
        self._backend = backend or InMemoryMetricBackend(
            histogram_buckets=self._config.duration_buckets
        )
        self._active_requests: Dict[str, int] = {}

    def _metric_name(self, name: str) -> str:
        """Generate full metric name."""
        return f"{self._config.prefix}_{name}"

    def record_request_start(
        self,
        provider: str,
        operation: str = "analyze_image",
    ) -> None:
        """Record start of a request."""
        labels = {}
        if self._config.include_provider_label:
            labels["provider"] = provider
        if self._config.include_operation_label:
            labels["operation"] = operation

        # Increment active requests
        if self._config.collect_active_requests:
            key = f"{provider}:{operation}"
            self._active_requests[key] = self._active_requests.get(key, 0) + 1
            self._backend.gauge(
                self._metric_name("active_requests"),
                self._active_requests[key],
                labels,
            )

    def record_request_end(
        self,
        provider: str,
        operation: str = "analyze_image",
        success: bool = True,
        duration_seconds: float = 0.0,
        image_size_bytes: int = 0,
        confidence: Optional[float] = None,
        error_type: Optional[str] = None,
    ) -> None:
        """
        Record end of a request.

        Args:
            provider: Provider name
            operation: Operation name
            success: Whether request succeeded
            duration_seconds: Request duration
            image_size_bytes: Size of processed image
            confidence: Result confidence score
            error_type: Type of error if failed
        """
        labels = {}
        if self._config.include_provider_label:
            labels["provider"] = provider
        if self._config.include_operation_label:
            labels["operation"] = operation

        # Decrement active requests
        if self._config.collect_active_requests:
            key = f"{provider}:{operation}"
            self._active_requests[key] = max(0, self._active_requests.get(key, 1) - 1)
            self._backend.gauge(
                self._metric_name("active_requests"),
                self._active_requests[key],
                labels,
            )

        # Record request count
        if self._config.collect_request_count:
            status_labels = {**labels, "status": "success" if success else "error"}
            self._backend.increment(
                self._metric_name("requests_total"),
                1.0,
                status_labels,
            )

        # Record error count
        if not success and self._config.collect_error_count:
            error_labels = {**labels}
            if error_type:
                error_labels["error_type"] = error_type
            self._backend.increment(
                self._metric_name("errors_total"),
                1.0,
                error_labels,
            )

        # Record duration
        if self._config.collect_request_duration and duration_seconds > 0:
            self._backend.histogram(
                self._metric_name("request_duration_seconds"),
                duration_seconds,
                labels,
            )

        # Record image size
        if self._config.collect_image_size and image_size_bytes > 0:
            size_kb = image_size_bytes / 1024
            self._backend.histogram(
                self._metric_name("image_size_kb"),
                size_kb,
                labels,
            )

        # Record confidence
        if self._config.collect_confidence and confidence is not None:
            self._backend.histogram(
                self._metric_name("confidence_score"),
                confidence,
                labels,
            )

    def export(self) -> str:
        """Export all metrics in Prometheus format."""
        return self._backend.export()

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary."""
        if isinstance(self._backend, InMemoryMetricBackend):
            return {
                "counters": {
                    k: v.value for k, v in self._backend._counters.items()
                },
                "gauges": {
                    k: v.value for k, v in self._backend._gauges.items()
                },
                "histograms": {
                    k: {"count": v.count, "sum": v.sum}
                    for k, v in self._backend._histograms.items()
                },
            }
        return {}

    def reset(self) -> None:
        """Reset all metrics."""
        if isinstance(self._backend, InMemoryMetricBackend):
            self._backend.reset()
        self._active_requests.clear()


class MetricsVisionProvider:
    """
    Wrapper that adds metrics collection to any VisionProvider.

    Automatically records metrics for all operations.
    """

    def __init__(
        self,
        provider: VisionProvider,
        exporter: MetricsExporter,
    ):
        """
        Initialize metrics provider.

        Args:
            provider: The underlying vision provider
            exporter: MetricsExporter instance
        """
        self._provider = provider
        self._exporter = exporter

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """
        Analyze image with metrics collection.

        Args:
            image_data: Raw image bytes
            include_description: Whether to generate description

        Returns:
            VisionDescription with analysis results
        """
        provider_name = self._provider.provider_name
        operation = "analyze_image"

        self._exporter.record_request_start(provider_name, operation)
        start_time = time.time()

        try:
            result = await self._provider.analyze_image(
                image_data, include_description
            )
            duration = time.time() - start_time

            self._exporter.record_request_end(
                provider=provider_name,
                operation=operation,
                success=True,
                duration_seconds=duration,
                image_size_bytes=len(image_data),
                confidence=result.confidence,
            )

            return result

        except Exception as e:
            duration = time.time() - start_time

            self._exporter.record_request_end(
                provider=provider_name,
                operation=operation,
                success=False,
                duration_seconds=duration,
                image_size_bytes=len(image_data),
                error_type=type(e).__name__,
            )

            raise

    @property
    def provider_name(self) -> str:
        """Return wrapped provider name."""
        return self._provider.provider_name

    @property
    def metrics_exporter(self) -> MetricsExporter:
        """Get the metrics exporter."""
        return self._exporter


# Global metrics exporter
_global_exporter: Optional[MetricsExporter] = None


def get_metrics_exporter() -> MetricsExporter:
    """
    Get the global metrics exporter instance.

    Returns:
        MetricsExporter singleton
    """
    global _global_exporter
    if _global_exporter is None:
        _global_exporter = MetricsExporter()
    return _global_exporter


def create_metrics_provider(
    provider: VisionProvider,
    config: Optional[MetricsConfig] = None,
    exporter: Optional[MetricsExporter] = None,
) -> MetricsVisionProvider:
    """
    Factory to create a metrics-collecting provider wrapper.

    Args:
        provider: The underlying vision provider
        config: Optional metrics configuration
        exporter: Optional existing metrics exporter

    Returns:
        MetricsVisionProvider wrapping the original

    Example:
        >>> provider = create_vision_provider("openai")
        >>> metered = create_metrics_provider(provider)
        >>> result = await metered.analyze_image(image_bytes)
        >>> print(metered.metrics_exporter.export())
    """
    if exporter is None:
        exporter = MetricsExporter(config=config)

    return MetricsVisionProvider(
        provider=provider,
        exporter=exporter,
    )
