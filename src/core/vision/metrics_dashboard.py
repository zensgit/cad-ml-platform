"""Metrics Dashboard Module for Vision System.

This module provides real-time metrics dashboard capabilities including:
- Custom metrics definition and collection
- Time-series data storage and aggregation
- Dashboard configuration and layouts
- Widget management and visualization
- Real-time metric streaming
- Historical data analysis

Phase 17: Advanced Observability & Monitoring
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import statistics
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from .base import VisionDescription, VisionProvider

# ========================
# Enums
# ========================


class MetricType(str, Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class AggregationType(str, Enum):
    """Types of aggregations."""

    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    P50 = "p50"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"
    RATE = "rate"


class WidgetType(str, Enum):
    """Types of dashboard widgets."""

    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    GAUGE = "gauge"
    COUNTER = "counter"
    TABLE = "table"
    HEATMAP = "heatmap"
    SPARKLINE = "sparkline"
    STAT = "stat"
    TEXT = "text"


class TimeRange(str, Enum):
    """Predefined time ranges."""

    LAST_5_MINUTES = "5m"
    LAST_15_MINUTES = "15m"
    LAST_1_HOUR = "1h"
    LAST_6_HOURS = "6h"
    LAST_24_HOURS = "24h"
    LAST_7_DAYS = "7d"
    LAST_30_DAYS = "30d"
    CUSTOM = "custom"


class RefreshInterval(str, Enum):
    """Dashboard refresh intervals."""

    REAL_TIME = "1s"
    FAST = "5s"
    NORMAL = "30s"
    SLOW = "1m"
    MANUAL = "manual"


# ========================
# Data Classes
# ========================


@dataclass
class MetricValue:
    """A single metric value."""

    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricDefinition:
    """Definition of a metric."""

    name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    labels: List[str] = field(default_factory=list)
    buckets: List[float] = field(default_factory=list)  # For histograms
    quantiles: List[float] = field(default_factory=list)  # For summaries


@dataclass
class MetricSeries:
    """A time series of metric values."""

    metric_name: str
    labels: Dict[str, str]
    values: List[MetricValue] = field(default_factory=list)

    def add_value(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Add a value to the series."""
        self.values.append(
            MetricValue(
                value=value, timestamp=timestamp or datetime.now(), labels=self.labels.copy()
            )
        )


@dataclass
class AggregatedMetric:
    """Aggregated metric result."""

    metric_name: str
    aggregation: AggregationType
    value: float
    start_time: datetime
    end_time: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    sample_count: int = 0


@dataclass
class WidgetConfig:
    """Configuration for a dashboard widget."""

    widget_id: str
    widget_type: WidgetType
    title: str
    metrics: List[str]
    aggregation: AggregationType = AggregationType.AVG
    time_range: TimeRange = TimeRange.LAST_1_HOUR
    refresh_interval: RefreshInterval = RefreshInterval.NORMAL
    position: Tuple[int, int] = (0, 0)  # (row, col)
    size: Tuple[int, int] = (1, 1)  # (height, width)
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardConfig:
    """Configuration for a dashboard."""

    dashboard_id: str
    name: str
    description: str = ""
    widgets: List[WidgetConfig] = field(default_factory=list)
    layout: str = "grid"  # grid, flex, free
    refresh_interval: RefreshInterval = RefreshInterval.NORMAL
    time_range: TimeRange = TimeRange.LAST_1_HOUR
    variables: Dict[str, Any] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class WidgetData:
    """Data for rendering a widget."""

    widget_id: str
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class DashboardSnapshot:
    """A snapshot of dashboard state."""

    dashboard_id: str
    widgets: List[WidgetData]
    timestamp: datetime = field(default_factory=datetime.now)


# ========================
# Metric Collector
# ========================


class MetricCollector:
    """Collects and stores metrics."""

    def __init__(self, max_points: int = 10000, retention_hours: int = 24):
        self._definitions: Dict[str, MetricDefinition] = {}
        self._series: Dict[str, Dict[str, MetricSeries]] = defaultdict(dict)
        self._counters: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._histograms: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self._max_points = max_points
        self._retention_hours = retention_hours
        self._lock = threading.Lock()

    def register_metric(self, definition: MetricDefinition) -> bool:
        """Register a new metric definition."""
        with self._lock:
            if definition.name in self._definitions:
                return False
            self._definitions[definition.name] = definition
            return True

    def get_metric_definition(self, name: str) -> Optional[MetricDefinition]:
        """Get a metric definition."""
        with self._lock:
            return self._definitions.get(name)

    def increment(
        self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric."""
        labels = labels or {}
        labels_key = self._labels_to_key(labels)

        with self._lock:
            self._counters[name][labels_key] += value
            self._record_series(name, self._counters[name][labels_key], labels)

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric."""
        labels = labels or {}
        labels_key = self._labels_to_key(labels)

        with self._lock:
            self._gauges[name][labels_key] = value
            self._record_series(name, value, labels)

    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a value for histogram/summary."""
        labels = labels or {}
        labels_key = self._labels_to_key(labels)

        with self._lock:
            self._histograms[name][labels_key].append(value)
            self._record_series(name, value, labels)

            # Limit stored values
            if len(self._histograms[name][labels_key]) > self._max_points:
                self._histograms[name][labels_key] = self._histograms[name][labels_key][
                    -self._max_points :
                ]

    def _record_series(self, name: str, value: float, labels: Dict[str, str]) -> None:
        """Record a value in the time series."""
        labels_key = self._labels_to_key(labels)

        if labels_key not in self._series[name]:
            self._series[name][labels_key] = MetricSeries(metric_name=name, labels=labels)

        self._series[name][labels_key].add_value(value)

        # Limit series length
        series = self._series[name][labels_key]
        if len(series.values) > self._max_points:
            series.values = series.values[-self._max_points :]

    def get_series(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[MetricSeries]:
        """Get metric series."""
        with self._lock:
            if name not in self._series:
                return []

            result = []
            for labels_key, series in self._series[name].items():
                if labels and not self._labels_match(series.labels, labels):
                    continue

                filtered_values = series.values
                if start_time:
                    filtered_values = [v for v in filtered_values if v.timestamp >= start_time]
                if end_time:
                    filtered_values = [v for v in filtered_values if v.timestamp <= end_time]

                if filtered_values:
                    result.append(
                        MetricSeries(
                            metric_name=series.metric_name,
                            labels=series.labels.copy(),
                            values=filtered_values,
                        )
                    )

            return result

    def aggregate(
        self,
        name: str,
        aggregation: AggregationType,
        labels: Optional[Dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Optional[AggregatedMetric]:
        """Aggregate metric values."""
        series_list = self.get_series(name, labels, start_time, end_time)

        if not series_list:
            return None

        all_values = []
        for series in series_list:
            all_values.extend([v.value for v in series.values])

        if not all_values:
            return None

        value = self._compute_aggregation(all_values, aggregation)

        return AggregatedMetric(
            metric_name=name,
            aggregation=aggregation,
            value=value,
            start_time=start_time or datetime.min,
            end_time=end_time or datetime.now(),
            labels=labels or {},
            sample_count=len(all_values),
        )

    def _compute_aggregation(self, values: List[float], aggregation: AggregationType) -> float:
        """Compute aggregation on values."""
        if not values:
            return 0.0

        if aggregation == AggregationType.SUM:
            return sum(values)
        elif aggregation == AggregationType.AVG:
            return statistics.mean(values)
        elif aggregation == AggregationType.MIN:
            return min(values)
        elif aggregation == AggregationType.MAX:
            return max(values)
        elif aggregation == AggregationType.COUNT:
            return float(len(values))
        elif aggregation == AggregationType.P50:
            return self._percentile(values, 50)
        elif aggregation == AggregationType.P90:
            return self._percentile(values, 90)
        elif aggregation == AggregationType.P95:
            return self._percentile(values, 95)
        elif aggregation == AggregationType.P99:
            return self._percentile(values, 99)
        elif aggregation == AggregationType.RATE:
            if len(values) < 2:
                return 0.0
            return (values[-1] - values[0]) / len(values)

        return 0.0

    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile."""
        sorted_values = sorted(values)
        idx = (len(sorted_values) - 1) * p / 100
        lower = int(idx)
        upper = lower + 1
        if upper >= len(sorted_values):
            return sorted_values[-1]
        weight = idx - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

    def _labels_to_key(self, labels: Dict[str, str]) -> str:
        """Convert labels to a hashable key."""
        return json.dumps(labels, sort_keys=True)

    def _labels_match(self, series_labels: Dict[str, str], filter_labels: Dict[str, str]) -> bool:
        """Check if series labels match filter."""
        for key, value in filter_labels.items():
            if series_labels.get(key) != value:
                return False
        return True

    def cleanup_old_data(self) -> int:
        """Remove data older than retention period."""
        cutoff = datetime.now() - timedelta(hours=self._retention_hours)
        removed_count = 0

        with self._lock:
            for name in list(self._series.keys()):
                for labels_key in list(self._series[name].keys()):
                    series = self._series[name][labels_key]
                    original_len = len(series.values)
                    series.values = [v for v in series.values if v.timestamp >= cutoff]
                    removed_count += original_len - len(series.values)

                    if not series.values:
                        del self._series[name][labels_key]

                if not self._series[name]:
                    del self._series[name]

        return removed_count

    def get_all_metrics(self) -> List[str]:
        """Get all registered metric names."""
        with self._lock:
            return list(self._definitions.keys())


# ========================
# Dashboard Manager
# ========================


class DashboardManager:
    """Manages dashboards and their configurations."""

    def __init__(self, collector: MetricCollector):
        self._collector = collector
        self._dashboards: Dict[str, DashboardConfig] = {}
        self._lock = threading.Lock()

    def create_dashboard(self, config: DashboardConfig) -> bool:
        """Create a new dashboard."""
        with self._lock:
            if config.dashboard_id in self._dashboards:
                return False
            self._dashboards[config.dashboard_id] = config
            return True

    def update_dashboard(self, config: DashboardConfig) -> bool:
        """Update an existing dashboard."""
        with self._lock:
            if config.dashboard_id not in self._dashboards:
                return False
            config.updated_at = datetime.now()
            self._dashboards[config.dashboard_id] = config
            return True

    def delete_dashboard(self, dashboard_id: str) -> bool:
        """Delete a dashboard."""
        with self._lock:
            if dashboard_id not in self._dashboards:
                return False
            del self._dashboards[dashboard_id]
            return True

    def get_dashboard(self, dashboard_id: str) -> Optional[DashboardConfig]:
        """Get a dashboard configuration."""
        with self._lock:
            return self._dashboards.get(dashboard_id)

    def list_dashboards(self) -> List[DashboardConfig]:
        """List all dashboards."""
        with self._lock:
            return list(self._dashboards.values())

    def add_widget(self, dashboard_id: str, widget: WidgetConfig) -> bool:
        """Add a widget to a dashboard."""
        with self._lock:
            dashboard = self._dashboards.get(dashboard_id)
            if not dashboard:
                return False

            # Check for duplicate widget ID
            for existing in dashboard.widgets:
                if existing.widget_id == widget.widget_id:
                    return False

            dashboard.widgets.append(widget)
            dashboard.updated_at = datetime.now()
            return True

    def remove_widget(self, dashboard_id: str, widget_id: str) -> bool:
        """Remove a widget from a dashboard."""
        with self._lock:
            dashboard = self._dashboards.get(dashboard_id)
            if not dashboard:
                return False

            original_count = len(dashboard.widgets)
            dashboard.widgets = [w for w in dashboard.widgets if w.widget_id != widget_id]

            if len(dashboard.widgets) < original_count:
                dashboard.updated_at = datetime.now()
                return True
            return False

    def get_widget_data(
        self, widget: WidgetConfig, custom_time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> WidgetData:
        """Get data for a widget."""
        start_time, end_time = self._resolve_time_range(widget.time_range, custom_time_range)

        data = []
        for metric_name in widget.metrics:
            aggregated = self._collector.aggregate(
                metric_name, widget.aggregation, start_time=start_time, end_time=end_time
            )

            if aggregated:
                data.append(
                    {
                        "metric": metric_name,
                        "value": aggregated.value,
                        "aggregation": aggregated.aggregation.value,
                        "sample_count": aggregated.sample_count,
                    }
                )

            # Also get time series for charts
            series_list = self._collector.get_series(
                metric_name, start_time=start_time, end_time=end_time
            )

            for series in series_list:
                data.append(
                    {
                        "metric": metric_name,
                        "series": [
                            {
                                "timestamp": v.timestamp.isoformat(),
                                "value": v.value,
                                "labels": v.labels,
                            }
                            for v in series.values
                        ],
                    }
                )

        return WidgetData(
            widget_id=widget.widget_id,
            data=data,
            metadata={
                "time_range": widget.time_range.value,
                "aggregation": widget.aggregation.value,
            },
        )

    def get_dashboard_snapshot(
        self, dashboard_id: str, custom_time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Optional[DashboardSnapshot]:
        """Get a snapshot of dashboard data."""
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            return None

        widgets_data = []
        for widget in dashboard.widgets:
            widget_data = self.get_widget_data(widget, custom_time_range)
            widgets_data.append(widget_data)

        return DashboardSnapshot(dashboard_id=dashboard_id, widgets=widgets_data)

    def _resolve_time_range(
        self, time_range: TimeRange, custom_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Tuple[datetime, datetime]:
        """Resolve time range to start/end times."""
        if custom_range:
            return custom_range

        end_time = datetime.now()

        if time_range == TimeRange.LAST_5_MINUTES:
            start_time = end_time - timedelta(minutes=5)
        elif time_range == TimeRange.LAST_15_MINUTES:
            start_time = end_time - timedelta(minutes=15)
        elif time_range == TimeRange.LAST_1_HOUR:
            start_time = end_time - timedelta(hours=1)
        elif time_range == TimeRange.LAST_6_HOURS:
            start_time = end_time - timedelta(hours=6)
        elif time_range == TimeRange.LAST_24_HOURS:
            start_time = end_time - timedelta(hours=24)
        elif time_range == TimeRange.LAST_7_DAYS:
            start_time = end_time - timedelta(days=7)
        elif time_range == TimeRange.LAST_30_DAYS:
            start_time = end_time - timedelta(days=30)
        else:
            start_time = end_time - timedelta(hours=1)

        return start_time, end_time


# ========================
# Real-time Streaming
# ========================


class MetricStreamSubscriber(ABC):
    """Abstract subscriber for metric streams."""

    @abstractmethod
    async def on_metric(self, metric: MetricValue, definition: MetricDefinition) -> None:
        """Handle incoming metric."""
        pass


class MetricStreamer:
    """Streams metrics in real-time."""

    def __init__(self, collector: MetricCollector):
        self._collector = collector
        self._subscribers: Dict[str, List[MetricStreamSubscriber]] = defaultdict(list)
        self._running = False
        self._lock = threading.Lock()

    def subscribe(self, metric_name: str, subscriber: MetricStreamSubscriber) -> None:
        """Subscribe to a metric stream."""
        with self._lock:
            self._subscribers[metric_name].append(subscriber)

    def unsubscribe(self, metric_name: str, subscriber: MetricStreamSubscriber) -> None:
        """Unsubscribe from a metric stream."""
        with self._lock:
            if metric_name in self._subscribers:
                try:
                    self._subscribers[metric_name].remove(subscriber)
                except ValueError:
                    pass

    async def publish(self, metric_name: str, value: MetricValue) -> None:
        """Publish a metric value to subscribers."""
        definition = self._collector.get_metric_definition(metric_name)

        with self._lock:
            subscribers = self._subscribers.get(metric_name, [])[:]

        for subscriber in subscribers:
            try:
                await subscriber.on_metric(value, definition)
            except Exception:
                pass  # Don't let one subscriber failure affect others


# ========================
# Custom Metric Builder
# ========================


class CustomMetricBuilder:
    """Builder for custom metrics."""

    def __init__(self):
        self._name: str = ""
        self._type: MetricType = MetricType.GAUGE
        self._description: str = ""
        self._unit: str = ""
        self._labels: List[str] = []
        self._buckets: List[float] = []
        self._quantiles: List[float] = []

    def name(self, name: str) -> "CustomMetricBuilder":
        """Set metric name."""
        self._name = name
        return self

    def type(self, metric_type: MetricType) -> "CustomMetricBuilder":
        """Set metric type."""
        self._type = metric_type
        return self

    def description(self, description: str) -> "CustomMetricBuilder":
        """Set metric description."""
        self._description = description
        return self

    def unit(self, unit: str) -> "CustomMetricBuilder":
        """Set metric unit."""
        self._unit = unit
        return self

    def with_labels(self, *labels: str) -> "CustomMetricBuilder":
        """Add labels."""
        self._labels.extend(labels)
        return self

    def with_buckets(self, *buckets: float) -> "CustomMetricBuilder":
        """Set histogram buckets."""
        self._buckets.extend(buckets)
        return self

    def with_quantiles(self, *quantiles: float) -> "CustomMetricBuilder":
        """Set summary quantiles."""
        self._quantiles.extend(quantiles)
        return self

    def build(self) -> MetricDefinition:
        """Build the metric definition."""
        if not self._name:
            raise ValueError("Metric name is required")

        return MetricDefinition(
            name=self._name,
            metric_type=self._type,
            description=self._description,
            unit=self._unit,
            labels=self._labels,
            buckets=self._buckets,
            quantiles=self._quantiles,
        )


# ========================
# Metrics Dashboard Provider
# ========================


class MetricsDashboardVisionProvider(VisionProvider):
    """Vision provider with metrics dashboard integration."""

    def __init__(
        self,
        base_provider: VisionProvider,
        collector: Optional[MetricCollector] = None,
        dashboard_manager: Optional[DashboardManager] = None,
    ):
        self._base_provider = base_provider
        self._collector = collector or MetricCollector()
        self._dashboard_manager = dashboard_manager or DashboardManager(self._collector)
        self._setup_default_metrics()

    def _setup_default_metrics(self) -> None:
        """Set up default vision metrics."""
        metrics = [
            MetricDefinition(
                name="vision_requests_total",
                metric_type=MetricType.COUNTER,
                description="Total vision analysis requests",
                labels=["provider", "status"],
            ),
            MetricDefinition(
                name="vision_request_duration_seconds",
                metric_type=MetricType.HISTOGRAM,
                description="Vision request duration in seconds",
                unit="seconds",
                labels=["provider"],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            ),
            MetricDefinition(
                name="vision_confidence_score",
                metric_type=MetricType.GAUGE,
                description="Latest confidence score",
                labels=["provider"],
            ),
        ]

        for metric in metrics:
            self._collector.register_metric(metric)

    @property
    def provider_name(self) -> str:
        return f"metrics_dashboard_{self._base_provider.provider_name}"

    async def analyze_image(
        self, image_data: bytes, context: Optional[Dict[str, Any]] = None
    ) -> VisionDescription:
        """Analyze image with metrics collection."""
        start_time = time.time()
        provider = self._base_provider.provider_name

        try:
            result = await self._base_provider.analyze_image(image_data, context)

            duration = time.time() - start_time

            # Record metrics
            self._collector.increment(
                "vision_requests_total", labels={"provider": provider, "status": "success"}
            )
            self._collector.observe(
                "vision_request_duration_seconds", duration, labels={"provider": provider}
            )
            self._collector.set_gauge(
                "vision_confidence_score", result.confidence, labels={"provider": provider}
            )

            return result

        except Exception as e:
            self._collector.increment(
                "vision_requests_total", labels={"provider": provider, "status": "error"}
            )
            raise

    def get_collector(self) -> MetricCollector:
        """Get the metric collector."""
        return self._collector

    def get_dashboard_manager(self) -> DashboardManager:
        """Get the dashboard manager."""
        return self._dashboard_manager


# ========================
# Factory Functions
# ========================


def create_metric_collector(max_points: int = 10000, retention_hours: int = 24) -> MetricCollector:
    """Create a new metric collector."""
    return MetricCollector(max_points, retention_hours)


def create_dashboard_manager(collector: Optional[MetricCollector] = None) -> DashboardManager:
    """Create a new dashboard manager."""
    return DashboardManager(collector or create_metric_collector())


def create_metric_streamer(collector: Optional[MetricCollector] = None) -> MetricStreamer:
    """Create a new metric streamer."""
    return MetricStreamer(collector or create_metric_collector())


def create_custom_metric_builder() -> CustomMetricBuilder:
    """Create a new custom metric builder."""
    return CustomMetricBuilder()


def create_metrics_dashboard_provider(
    base_provider: VisionProvider,
    collector: Optional[MetricCollector] = None,
    dashboard_manager: Optional[DashboardManager] = None,
) -> MetricsDashboardVisionProvider:
    """Create a metrics dashboard vision provider."""
    return MetricsDashboardVisionProvider(base_provider, collector, dashboard_manager)
