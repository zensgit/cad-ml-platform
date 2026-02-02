"""Metrics Registry and Export.

Provides metric collection and export:
- Registry for metrics
- Multiple export formats
- Aggregation
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from src.core.metrics_aggregator.core import (
    Counter,
    Gauge,
    Histogram,
    Metric,
    MetricLabels,
    MetricSample,
    MetricType,
    MetricValue,
    Summary,
)

logger = logging.getLogger(__name__)


class MetricsRegistry:
    """Registry for all metrics."""

    def __init__(self, prefix: str = ""):
        self._prefix = prefix
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()

    def _full_name(self, name: str) -> str:
        if self._prefix:
            return f"{self._prefix}_{name}"
        return name

    def counter(
        self,
        name: str,
        description: str = "",
        label_names: Optional[List[str]] = None,
    ) -> Counter:
        """Get or create a counter."""
        full_name = self._full_name(name)

        with self._lock:
            if full_name in self._metrics:
                metric = self._metrics[full_name]
                if not isinstance(metric, Counter):
                    raise ValueError(f"Metric {name} is not a Counter")
                return metric

            counter = Counter(full_name, description, label_names)
            self._metrics[full_name] = counter
            return counter

    def gauge(
        self,
        name: str,
        description: str = "",
        label_names: Optional[List[str]] = None,
    ) -> Gauge:
        """Get or create a gauge."""
        full_name = self._full_name(name)

        with self._lock:
            if full_name in self._metrics:
                metric = self._metrics[full_name]
                if not isinstance(metric, Gauge):
                    raise ValueError(f"Metric {name} is not a Gauge")
                return metric

            gauge = Gauge(full_name, description, label_names)
            self._metrics[full_name] = gauge
            return gauge

    def histogram(
        self,
        name: str,
        description: str = "",
        label_names: Optional[List[str]] = None,
        buckets: Optional[tuple] = None,
    ) -> Histogram:
        """Get or create a histogram."""
        full_name = self._full_name(name)

        with self._lock:
            if full_name in self._metrics:
                metric = self._metrics[full_name]
                if not isinstance(metric, Histogram):
                    raise ValueError(f"Metric {name} is not a Histogram")
                return metric

            histogram = Histogram(full_name, description, label_names, buckets)
            self._metrics[full_name] = histogram
            return histogram

    def summary(
        self,
        name: str,
        description: str = "",
        label_names: Optional[List[str]] = None,
        quantiles: Optional[tuple] = None,
    ) -> Summary:
        """Get or create a summary."""
        full_name = self._full_name(name)

        with self._lock:
            if full_name in self._metrics:
                metric = self._metrics[full_name]
                if not isinstance(metric, Summary):
                    raise ValueError(f"Metric {name} is not a Summary")
                return metric

            summary = Summary(full_name, description, label_names, quantiles)
            self._metrics[full_name] = summary
            return summary

    def collect(self) -> List[MetricSample]:
        """Collect all metrics."""
        samples = []

        with self._lock:
            for name, metric in self._metrics.items():
                values = metric.collect()
                samples.append(MetricSample(
                    name=name,
                    metric_type=metric.metric_type,
                    description=metric.description,
                    values=values,
                ))

        return samples

    def get_metric(self, name: str) -> Optional[Metric]:
        """Get metric by name."""
        full_name = self._full_name(name)
        with self._lock:
            return self._metrics.get(full_name)

    def list_metrics(self) -> List[str]:
        """List all metric names."""
        with self._lock:
            return list(self._metrics.keys())


class MetricsExporter(ABC):
    """Abstract metrics exporter."""

    @abstractmethod
    def export(self, samples: List[MetricSample]) -> str:
        """Export metrics to string format."""
        pass


class PrometheusExporter(MetricsExporter):
    """Export metrics in Prometheus format."""

    def export(self, samples: List[MetricSample]) -> str:
        lines = []

        for sample in samples:
            # HELP line
            if sample.description:
                lines.append(f"# HELP {sample.name} {sample.description}")

            # TYPE line
            type_name = sample.metric_type.value
            lines.append(f"# TYPE {sample.name} {type_name}")

            # Values
            for value in sample.values:
                labels_str = value.labels.to_string()
                metric_name = sample.name

                # Handle histogram bucket suffix
                if "le" in value.labels.labels:
                    metric_name = f"{sample.name}_bucket"
                elif "_type" in value.labels.labels:
                    suffix = value.labels.labels["_type"]
                    metric_name = f"{sample.name}_{suffix}"
                    # Remove internal label
                    labels_str = MetricLabels({
                        k: v for k, v in value.labels.labels.items()
                        if k != "_type"
                    }).to_string()

                lines.append(f"{metric_name}{labels_str} {value.value}")

        return "\n".join(lines)


class JSONExporter(MetricsExporter):
    """Export metrics in JSON format."""

    def export(self, samples: List[MetricSample]) -> str:
        import json

        data = []

        for sample in samples:
            sample_data = {
                "name": sample.name,
                "type": sample.metric_type.value,
                "description": sample.description,
                "timestamp": sample.timestamp.isoformat(),
                "values": [
                    {
                        "value": v.value,
                        "labels": v.labels.labels,
                        "timestamp": v.timestamp.isoformat(),
                    }
                    for v in sample.values
                ],
            }
            data.append(sample_data)

        return json.dumps(data, indent=2)


class StatsDExporter(MetricsExporter):
    """Export metrics in StatsD format."""

    def export(self, samples: List[MetricSample]) -> str:
        lines = []

        for sample in samples:
            for value in sample.values:
                # Build metric name with labels
                name = sample.name
                if value.labels.labels:
                    label_parts = [f"{k}.{v}" for k, v in value.labels.labels.items()]
                    name = f"{name}.{'.'.join(label_parts)}"

                # Determine type suffix
                if sample.metric_type == MetricType.COUNTER:
                    lines.append(f"{name}:{value.value}|c")
                elif sample.metric_type == MetricType.GAUGE:
                    lines.append(f"{name}:{value.value}|g")
                elif sample.metric_type == MetricType.HISTOGRAM:
                    lines.append(f"{name}:{value.value}|h")

        return "\n".join(lines)


class MetricsAggregator:
    """Aggregates metrics from multiple sources."""

    def __init__(self):
        self._registries: List[MetricsRegistry] = []
        self._collectors: List[Callable[[], List[MetricSample]]] = []

    def add_registry(self, registry: MetricsRegistry) -> None:
        """Add a registry."""
        self._registries.append(registry)

    def add_collector(
        self,
        collector: Callable[[], List[MetricSample]],
    ) -> None:
        """Add a custom collector."""
        self._collectors.append(collector)

    def collect_all(self) -> List[MetricSample]:
        """Collect from all sources."""
        samples = []

        for registry in self._registries:
            samples.extend(registry.collect())

        for collector in self._collectors:
            try:
                samples.extend(collector())
            except Exception as e:
                logger.error(f"Collector error: {e}")

        return samples

    def export(self, exporter: MetricsExporter) -> str:
        """Export all metrics."""
        samples = self.collect_all()
        return exporter.export(samples)


class MetricsPusher:
    """Push metrics to remote endpoint."""

    def __init__(
        self,
        aggregator: MetricsAggregator,
        endpoint: str,
        interval_seconds: float = 10.0,
        exporter: Optional[MetricsExporter] = None,
    ):
        self._aggregator = aggregator
        self._endpoint = endpoint
        self._interval = interval_seconds
        self._exporter = exporter or PrometheusExporter()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start pushing metrics."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._push_loop, daemon=True)
        self._thread.start()
        logger.info(f"Metrics pusher started for {self._endpoint}")

    def stop(self) -> None:
        """Stop pushing metrics."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _push_loop(self) -> None:
        """Push loop."""
        while self._running:
            try:
                data = self._aggregator.export(self._exporter)
                self._send(data)
            except Exception as e:
                logger.error(f"Metrics push error: {e}")

            time.sleep(self._interval)

    def _send(self, data: str) -> None:
        """Send metrics data."""
        # Would use HTTP client in real implementation
        logger.debug(f"Would push {len(data)} bytes to {self._endpoint}")


# Default registry
_default_registry = MetricsRegistry()


def get_default_registry() -> MetricsRegistry:
    """Get default metrics registry."""
    return _default_registry


def counter(
    name: str,
    description: str = "",
    label_names: Optional[List[str]] = None,
) -> Counter:
    """Create counter in default registry."""
    return _default_registry.counter(name, description, label_names)


def gauge(
    name: str,
    description: str = "",
    label_names: Optional[List[str]] = None,
) -> Gauge:
    """Create gauge in default registry."""
    return _default_registry.gauge(name, description, label_names)


def histogram(
    name: str,
    description: str = "",
    label_names: Optional[List[str]] = None,
    buckets: Optional[tuple] = None,
) -> Histogram:
    """Create histogram in default registry."""
    return _default_registry.histogram(name, description, label_names, buckets)


def summary(
    name: str,
    description: str = "",
    label_names: Optional[List[str]] = None,
    quantiles: Optional[tuple] = None,
) -> Summary:
    """Create summary in default registry."""
    return _default_registry.summary(name, description, label_names, quantiles)
