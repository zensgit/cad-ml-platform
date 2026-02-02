"""Metrics Aggregator Module.

Provides metric collection and export:
- Counter, Gauge, Histogram, Summary
- Multiple export formats
- Registry management
"""

from src.core.metrics_aggregator.core import (
    MetricType,
    MetricLabels,
    MetricValue,
    MetricMetadata,
    Metric,
    LabeledMetric,
    Counter,
    Gauge,
    Histogram,
    Summary,
    Timer,
    MetricSample,
)
from src.core.metrics_aggregator.registry import (
    MetricsRegistry,
    MetricsExporter,
    PrometheusExporter,
    JSONExporter,
    StatsDExporter,
    MetricsAggregator,
    MetricsPusher,
    get_default_registry,
    counter,
    gauge,
    histogram,
    summary,
)

__all__ = [
    # Core
    "MetricType",
    "MetricLabels",
    "MetricValue",
    "MetricMetadata",
    "Metric",
    "LabeledMetric",
    "Counter",
    "Gauge",
    "Histogram",
    "Summary",
    "Timer",
    "MetricSample",
    # Registry
    "MetricsRegistry",
    "MetricsExporter",
    "PrometheusExporter",
    "JSONExporter",
    "StatsDExporter",
    "MetricsAggregator",
    "MetricsPusher",
    "get_default_registry",
    "counter",
    "gauge",
    "histogram",
    "summary",
]
