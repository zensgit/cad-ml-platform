"""
Observability Hub for Vision Provider.

This module provides a central observability orchestration system including:
- Unified metrics collection and aggregation
- Distributed tracing coordination
- Health monitoring and status aggregation
- Alert correlation and management
- Observability pipeline management
- SLO/SLI tracking
- Anomaly detection

Phase 21 Feature.
"""

import asyncio
import logging
import statistics
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .base import VisionDescription, VisionProvider

logger = logging.getLogger(__name__)


# ============================================================================
# Observability Enums
# ============================================================================


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertState(Enum):
    """Alert states."""

    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class TraceStatus(Enum):
    """Trace status."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class SLOStatus(Enum):
    """SLO status."""

    MET = "met"
    AT_RISK = "at_risk"
    BREACHED = "breached"


class AnomalyType(Enum):
    """Types of anomalies."""

    SPIKE = "spike"
    DROP = "drop"
    TREND = "trend"
    SEASONALITY = "seasonality"
    OUTLIER = "outlier"


# ============================================================================
# Observability Data Classes
# ============================================================================


@dataclass
class MetricPoint:
    """A single metric data point."""

    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class MetricSeries:
    """A time series of metric points."""

    name: str
    metric_type: MetricType
    points: List[MetricPoint] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)

    def add_point(self, value: float) -> None:
        """Add a point to the series."""
        point = MetricPoint(
            name=self.name,
            value=value,
            metric_type=self.metric_type,
            labels=self.labels,
        )
        self.points.append(point)

    def get_latest(self) -> Optional[MetricPoint]:
        """Get the latest point."""
        return self.points[-1] if self.points else None

    def get_average(self, window_seconds: int = 60) -> Optional[float]:
        """Get average value over a time window."""
        cutoff = datetime.now() - timedelta(seconds=window_seconds)
        recent = [p.value for p in self.points if p.timestamp > cutoff]
        return statistics.mean(recent) if recent else None


@dataclass
class SpanContext:
    """Distributed tracing span context."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    sampled: bool = True


@dataclass
class Span:
    """A single span in a trace."""

    context: SpanContext
    operation_name: str
    service_name: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: TraceStatus = TraceStatus.UNSET
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0

    def finish(self, status: TraceStatus = TraceStatus.OK) -> None:
        """Finish the span."""
        self.end_time = datetime.now()
        self.status = status
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

    def log(self, message: str, **kwargs: Any) -> None:
        """Add a log entry to the span."""
        self.logs.append(
            {
                "timestamp": datetime.now().isoformat(),
                "message": message,
                **kwargs,
            }
        )


@dataclass
class HealthCheck:
    """Health check result."""

    name: str
    status: HealthStatus
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert definition and state."""

    alert_id: str
    name: str
    severity: AlertSeverity
    state: AlertState = AlertState.PENDING
    message: str = ""
    source: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    firing_since: Optional[datetime] = None
    value: float = 0.0


@dataclass
class SLODefinition:
    """Service Level Objective definition."""

    slo_id: str
    name: str
    target: float  # Target percentage (e.g., 99.9)
    window_days: int = 30
    metric_name: str = ""
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class SLOResult:
    """SLO status and burn rate result."""

    slo: SLODefinition
    current_value: float
    error_budget_remaining: float
    burn_rate: float
    status: str  # met, at_risk, breached
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Anomaly:
    """Detected anomaly."""

    anomaly_id: str
    anomaly_type: AnomalyType
    metric_name: str
    expected_value: float
    actual_value: float
    deviation: float
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservabilityConfig:
    """Observability hub configuration."""

    metrics_retention_hours: int = 24
    trace_sample_rate: float = 0.1
    health_check_interval_seconds: int = 30
    alert_evaluation_interval_seconds: int = 60
    anomaly_detection_enabled: bool = True
    slo_tracking_enabled: bool = True


# ============================================================================
# Metrics Collector
# ============================================================================


class MetricsCollector:
    """Collects and aggregates metrics."""

    def __init__(self, config: Optional[ObservabilityConfig] = None) -> None:
        """Initialize metrics collector."""
        self._config = config or ObservabilityConfig()
        self._series: Dict[str, MetricSeries] = {}
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._lock = threading.RLock()

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter."""
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value
            self._record_point(name, self._counters[key], MetricType.COUNTER, labels)

    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge value."""
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value
            self._record_point(name, value, MetricType.GAUGE, labels)

    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram value."""
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)
            self._record_point(name, value, MetricType.HISTOGRAM, labels)

    def timer(
        self,
        name: str,
        duration_ms: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a timer value."""
        self.histogram(name, duration_ms, labels)

    def get_metric(
        self, name: str, labels: Optional[Dict[str, str]] = None
    ) -> Optional[MetricSeries]:
        """Get a metric series."""
        key = self._make_key(name, labels)
        with self._lock:
            return self._series.get(key)

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get counter value."""
        key = self._make_key(name, labels)
        with self._lock:
            return self._counters.get(key, 0)

    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get gauge value."""
        key = self._make_key(name, labels)
        with self._lock:
            return self._gauges.get(key)

    def get_histogram_stats(
        self, name: str, labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """Get histogram statistics."""
        key = self._make_key(name, labels)
        with self._lock:
            values = self._histograms.get(key, [])
            if not values:
                return {}

            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95": self._percentile(values, 95),
                "p99": self._percentile(values, 99),
            }

    def list_metrics(self) -> List[str]:
        """List all metric names."""
        with self._lock:
            return list(set(s.name for s in self._series.values()))

    def export(self) -> List[MetricPoint]:
        """Export all current metrics."""
        with self._lock:
            points = []
            for series in self._series.values():
                if series.points:
                    points.append(series.points[-1])
            return points

    def _record_point(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        labels: Optional[Dict[str, str]],
    ) -> None:
        """Record a metric point."""
        key = self._make_key(name, labels)
        if key not in self._series:
            self._series[key] = MetricSeries(
                name=name,
                metric_type=metric_type,
                labels=labels or {},
            )
        self._series[key].add_point(value)

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a unique key for a metric."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


# ============================================================================
# Trace Manager
# ============================================================================


class TraceManager:
    """Manages distributed traces."""

    def __init__(self, config: Optional[ObservabilityConfig] = None) -> None:
        """Initialize trace manager."""
        self._config = config or ObservabilityConfig()
        self._traces: Dict[str, List[Span]] = {}
        self._active_spans: Dict[str, Span] = {}
        self._lock = threading.RLock()

    def start_span(
        self,
        operation_name: str,
        service_name: str,
        parent_context: Optional[SpanContext] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """Start a new span."""
        # Determine if we should sample this trace
        if parent_context:
            trace_id = parent_context.trace_id
            sampled = parent_context.sampled
        else:
            trace_id = str(uuid.uuid4())
            sampled = self._should_sample()

        span_id = str(uuid.uuid4())[:16]
        parent_span_id = parent_context.span_id if parent_context else None

        context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            sampled=sampled,
        )

        span = Span(
            context=context,
            operation_name=operation_name,
            service_name=service_name,
            tags=tags or {},
        )

        with self._lock:
            if trace_id not in self._traces:
                self._traces[trace_id] = []
            self._traces[trace_id].append(span)
            self._active_spans[span_id] = span

        return span

    def finish_span(self, span: Span, status: TraceStatus = TraceStatus.OK) -> None:
        """Finish a span."""
        span.finish(status)
        with self._lock:
            self._active_spans.pop(span.context.span_id, None)

    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        with self._lock:
            return self._traces.get(trace_id, [])

    def get_active_spans(self) -> List[Span]:
        """Get all active spans."""
        with self._lock:
            return list(self._active_spans.values())

    def inject_context(self, span: Span) -> Dict[str, str]:
        """Inject trace context into headers."""
        return {
            "X-Trace-ID": span.context.trace_id,
            "X-Span-ID": span.context.span_id,
            "X-Parent-Span-ID": span.context.parent_span_id or "",
            "X-Sampled": str(span.context.sampled).lower(),
        }

    def extract_context(self, headers: Dict[str, str]) -> Optional[SpanContext]:
        """Extract trace context from headers."""
        trace_id = headers.get("X-Trace-ID")
        span_id = headers.get("X-Span-ID")

        if not trace_id or not span_id:
            return None

        return SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=headers.get("X-Parent-Span-ID") or None,
            sampled=headers.get("X-Sampled", "true").lower() == "true",
        )

    def _should_sample(self) -> bool:
        """Determine if a trace should be sampled."""
        import random

        return random.random() < self._config.trace_sample_rate


# ============================================================================
# Health Monitor
# ============================================================================


class HealthMonitor:
    """Monitors system health."""

    def __init__(self, config: Optional[ObservabilityConfig] = None) -> None:
        """Initialize health monitor."""
        self._config = config or ObservabilityConfig()
        self._checks: Dict[str, Callable[[], HealthCheck]] = {}
        self._results: Dict[str, HealthCheck] = {}
        self._lock = threading.RLock()

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], HealthCheck],
    ) -> None:
        """Register a health check."""
        with self._lock:
            self._checks[name] = check_fn

    def unregister_check(self, name: str) -> bool:
        """Unregister a health check."""
        with self._lock:
            if name in self._checks:
                del self._checks[name]
                return True
            return False

    def run_check(self, name: str) -> Optional[HealthCheck]:
        """Run a specific health check."""
        with self._lock:
            check_fn = self._checks.get(name)

        if not check_fn:
            return None

        start_time = time.time()
        try:
            result = check_fn()
            result.duration_ms = (time.time() - start_time) * 1000
        except Exception as e:
            result = HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

        with self._lock:
            self._results[name] = result

        return result

    def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all health checks."""
        results = {}
        with self._lock:
            check_names = list(self._checks.keys())

        for name in check_names:
            result = self.run_check(name)
            if result:
                results[name] = result

        return results

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        with self._lock:
            if not self._results:
                return HealthStatus.UNKNOWN

            statuses = [r.status for r in self._results.values()]

            if all(s == HealthStatus.HEALTHY for s in statuses):
                return HealthStatus.HEALTHY
            if any(s == HealthStatus.UNHEALTHY for s in statuses):
                return HealthStatus.UNHEALTHY
            if any(s == HealthStatus.DEGRADED for s in statuses):
                return HealthStatus.DEGRADED

            return HealthStatus.UNKNOWN

    def get_check_result(self, name: str) -> Optional[HealthCheck]:
        """Get the latest result for a check."""
        with self._lock:
            return self._results.get(name)

    def list_checks(self) -> List[str]:
        """List all registered checks."""
        with self._lock:
            return list(self._checks.keys())


# ============================================================================
# Alert Manager
# ============================================================================


class AlertManager:
    """Manages alerts and notifications."""

    def __init__(self, config: Optional[ObservabilityConfig] = None) -> None:
        """Initialize alert manager."""
        self._config = config or ObservabilityConfig()
        self._alerts: Dict[str, Alert] = {}
        self._rules: Dict[str, Callable[[Dict[str, Any]], Optional[Alert]]] = {}
        self._handlers: List[Callable[[Alert], None]] = []
        self._suppression_rules: List[Callable[[Alert], bool]] = []
        self._lock = threading.RLock()

    def register_rule(
        self,
        name: str,
        rule_fn: Callable[[Dict[str, Any]], Optional[Alert]],
    ) -> None:
        """Register an alert rule."""
        with self._lock:
            self._rules[name] = rule_fn

    def unregister_rule(self, name: str) -> bool:
        """Unregister an alert rule."""
        with self._lock:
            if name in self._rules:
                del self._rules[name]
                return True
            return False

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert handler."""
        with self._lock:
            self._handlers.append(handler)

    def add_suppression_rule(self, rule: Callable[[Alert], bool]) -> None:
        """Add a suppression rule."""
        with self._lock:
            self._suppression_rules.append(rule)

    def fire_alert(self, alert: Alert) -> None:
        """Fire an alert."""
        # Check suppression
        for rule in self._suppression_rules:
            if rule(alert):
                alert.state = AlertState.SUPPRESSED
                return

        alert.state = AlertState.FIRING
        alert.firing_since = datetime.now()

        with self._lock:
            self._alerts[alert.alert_id] = alert

        # Notify handlers
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")

    def resolve_alert(self, alert_id: str) -> Optional[Alert]:
        """Resolve an alert."""
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert:
                alert.state = AlertState.RESOLVED
                alert.resolved_at = datetime.now()
            return alert

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get an alert by ID."""
        with self._lock:
            return self._alerts.get(alert_id)

    def get_firing_alerts(self) -> List[Alert]:
        """Get all firing alerts."""
        with self._lock:
            return [a for a in self._alerts.values() if a.state == AlertState.FIRING]

    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity."""
        with self._lock:
            return [a for a in self._alerts.values() if a.severity == severity]

    def evaluate_rules(self, context: Dict[str, Any]) -> List[Alert]:
        """Evaluate all rules against context."""
        fired_alerts = []

        with self._lock:
            rules = list(self._rules.items())

        for name, rule_fn in rules:
            try:
                alert = rule_fn(context)
                if alert:
                    self.fire_alert(alert)
                    fired_alerts.append(alert)
            except Exception as e:
                logger.error(f"Alert rule '{name}' error: {e}")

        return fired_alerts


# ============================================================================
# SLO Tracker
# ============================================================================


class SLOTracker:
    """Tracks Service Level Objectives."""

    def __init__(self, metrics: MetricsCollector) -> None:
        """Initialize SLO tracker."""
        self._metrics = metrics
        self._slos: Dict[str, SLODefinition] = {}
        self._good_events: Dict[str, int] = {}
        self._total_events: Dict[str, int] = {}
        self._lock = threading.RLock()

    def define_slo(self, slo: SLODefinition) -> None:
        """Define an SLO."""
        with self._lock:
            self._slos[slo.slo_id] = slo
            self._good_events[slo.slo_id] = 0
            self._total_events[slo.slo_id] = 0

    def record_event(self, slo_id: str, is_good: bool) -> None:
        """Record an event for SLO calculation."""
        with self._lock:
            if slo_id not in self._slos:
                return

            self._total_events[slo_id] = self._total_events.get(slo_id, 0) + 1
            if is_good:
                self._good_events[slo_id] = self._good_events.get(slo_id, 0) + 1

    def get_slo_status(self, slo_id: str) -> Optional[SLOResult]:
        """Get current SLO status."""
        with self._lock:
            slo = self._slos.get(slo_id)
            if not slo:
                return None

            total = self._total_events.get(slo_id, 0)
            good = self._good_events.get(slo_id, 0)

            if total == 0:
                current_value = 100.0
            else:
                current_value = (good / total) * 100

            error_budget_total = 100 - slo.target
            error_budget_used = 100 - current_value
            error_budget_remaining = max(0, error_budget_total - error_budget_used)

            # Calculate burn rate (simplified)
            burn_rate = error_budget_used / error_budget_total if error_budget_total > 0 else 0

            if current_value >= slo.target:
                status = "met"
            elif error_budget_remaining > 0:
                status = "at_risk"
            else:
                status = "breached"

            return SLOResult(
                slo=slo,
                current_value=current_value,
                error_budget_remaining=error_budget_remaining,
                burn_rate=burn_rate,
                status=status,
            )

    def list_slos(self) -> List[SLODefinition]:
        """List all SLOs."""
        with self._lock:
            return list(self._slos.values())


# ============================================================================
# Anomaly Detector
# ============================================================================


class AnomalyDetector:
    """Detects anomalies in metrics."""

    def __init__(
        self,
        metrics: MetricsCollector,
        sensitivity: float = 2.0,
    ) -> None:
        """Initialize anomaly detector."""
        self._metrics = metrics
        self._sensitivity = sensitivity
        self._baselines: Dict[str, Dict[str, float]] = {}
        self._detected: List[Anomaly] = []
        self._lock = threading.RLock()

    def calculate_baseline(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, float]]:
        """Calculate baseline for a metric."""
        series = self._metrics.get_metric(metric_name, labels)
        if not series or len(series.points) < 10:
            return None

        values = [p.value for p in series.points]
        baseline = {
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
        }

        key = self._make_key(metric_name, labels)
        with self._lock:
            self._baselines[key] = baseline

        return baseline

    def detect(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[Anomaly]:
        """Detect if a value is anomalous."""
        key = self._make_key(metric_name, labels)

        with self._lock:
            baseline = self._baselines.get(key)

        if not baseline or baseline["stdev"] == 0:
            return None

        # Z-score based detection
        z_score = abs(value - baseline["mean"]) / baseline["stdev"]

        if z_score > self._sensitivity:
            anomaly_type = AnomalyType.SPIKE if value > baseline["mean"] else AnomalyType.DROP
            deviation = (value - baseline["mean"]) / baseline["mean"] * 100

            anomaly = Anomaly(
                anomaly_id=str(uuid.uuid4()),
                anomaly_type=anomaly_type,
                metric_name=metric_name,
                expected_value=baseline["mean"],
                actual_value=value,
                deviation=deviation,
                confidence=min(1.0, z_score / 4),  # Normalize confidence
            )

            with self._lock:
                self._detected.append(anomaly)

            return anomaly

        return None

    def get_detected_anomalies(self, limit: int = 100) -> List[Anomaly]:
        """Get detected anomalies."""
        with self._lock:
            return self._detected[-limit:]

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a unique key."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


# ============================================================================
# Observability Hub
# ============================================================================


class ObservabilityHub:
    """
    Central observability orchestration hub.

    Integrates metrics, tracing, health, and alerting.
    """

    def __init__(self, config: Optional[ObservabilityConfig] = None) -> None:
        """Initialize observability hub."""
        self._config = config or ObservabilityConfig()
        self._metrics = MetricsCollector(self._config)
        self._traces = TraceManager(self._config)
        self._health = HealthMonitor(self._config)
        self._alerts = AlertManager(self._config)
        self._slo_tracker = SLOTracker(self._metrics)
        self._anomaly_detector = AnomalyDetector(self._metrics)
        self._running = False
        self._lock = threading.RLock()

    @property
    def metrics(self) -> MetricsCollector:
        """Get metrics collector."""
        return self._metrics

    @property
    def traces(self) -> TraceManager:
        """Get trace manager."""
        return self._traces

    @property
    def health(self) -> HealthMonitor:
        """Get health monitor."""
        return self._health

    @property
    def alerts(self) -> AlertManager:
        """Get alert manager."""
        return self._alerts

    @property
    def slo_tracker(self) -> SLOTracker:
        """Get SLO tracker."""
        return self._slo_tracker

    @property
    def anomaly_detector(self) -> AnomalyDetector:
        """Get anomaly detector."""
        return self._anomaly_detector

    def start(self) -> None:
        """Start the observability hub."""
        self._running = True
        logger.info("Observability hub started")

    def stop(self) -> None:
        """Stop the observability hub."""
        self._running = False
        logger.info("Observability hub stopped")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for observability dashboard."""
        return {
            "health": {
                "overall": self._health.get_overall_status().value,
                "checks": {
                    name: result.status.value for name, result in self._health._results.items()
                },
            },
            "metrics": {
                "count": len(self._metrics.list_metrics()),
                "latest": [p.name for p in self._metrics.export()[:10]],
            },
            "traces": {
                "active_spans": len(self._traces.get_active_spans()),
            },
            "alerts": {
                "firing": len(self._alerts.get_firing_alerts()),
                "critical": len(self._alerts.get_alerts_by_severity(AlertSeverity.CRITICAL)),
            },
            "slos": {
                slo.name: self._slo_tracker.get_slo_status(slo.slo_id)
                for slo in self._slo_tracker.list_slos()
            },
            "anomalies": {
                "recent": len(self._anomaly_detector.get_detected_anomalies(10)),
            },
            "timestamp": datetime.now().isoformat(),
        }


# ============================================================================
# Vision Provider Integration
# ============================================================================


class ObservableVisionProvider(VisionProvider):
    """Vision provider with full observability integration."""

    def __init__(
        self,
        base_provider: VisionProvider,
        hub: ObservabilityHub,
    ) -> None:
        """Initialize observable vision provider."""
        self._base_provider = base_provider
        self._hub = hub

        # Register health check
        self._hub.health.register_check(
            f"provider_{base_provider.provider_name}",
            self._health_check,
        )

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"observable_{self._base_provider.provider_name}"

    def _health_check(self) -> HealthCheck:
        """Health check for this provider."""
        return HealthCheck(
            name=f"provider_{self._base_provider.provider_name}",
            status=HealthStatus.HEALTHY,
            message="Provider is operational",
        )

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with full observability."""
        # Start span
        span = self._hub.traces.start_span(
            operation_name="analyze_image",
            service_name=self.provider_name,
            tags={
                "image_size": len(image_data),
                "include_description": include_description,
            },
        )

        # Record metrics
        self._hub.metrics.increment(
            "vision_requests_total",
            labels={"provider": self._base_provider.provider_name},
        )

        start_time = time.time()

        try:
            result = await self._base_provider.analyze_image(image_data, include_description)

            duration_ms = (time.time() - start_time) * 1000

            # Record success metrics
            self._hub.metrics.increment(
                "vision_requests_success",
                labels={"provider": self._base_provider.provider_name},
            )
            self._hub.metrics.timer(
                "vision_request_duration_ms",
                duration_ms,
                labels={"provider": self._base_provider.provider_name},
            )
            self._hub.metrics.gauge(
                "vision_confidence",
                result.confidence,
                labels={"provider": self._base_provider.provider_name},
            )

            # Check for anomalies
            self._hub.anomaly_detector.detect(
                "vision_request_duration_ms",
                duration_ms,
                labels={"provider": self._base_provider.provider_name},
            )

            # Record SLO event
            self._hub.slo_tracker.record_event(
                f"slo_{self._base_provider.provider_name}",
                is_good=True,
            )

            # Finish span
            self._hub.traces.finish_span(span, TraceStatus.OK)

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            # Record error metrics
            self._hub.metrics.increment(
                "vision_requests_error",
                labels={"provider": self._base_provider.provider_name},
            )

            # Record SLO event
            self._hub.slo_tracker.record_event(
                f"slo_{self._base_provider.provider_name}",
                is_good=False,
            )

            # Fire alert for errors
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                name=f"VisionProviderError",
                severity=AlertSeverity.ERROR,
                message=str(e),
                source=self.provider_name,
                labels={"provider": self._base_provider.provider_name},
            )
            self._hub.alerts.fire_alert(alert)

            # Finish span with error
            span.log("error", error=str(e))
            self._hub.traces.finish_span(span, TraceStatus.ERROR)

            raise


# ============================================================================
# Factory Functions
# ============================================================================


def create_observability_hub(
    metrics_retention_hours: int = 24,
    trace_sample_rate: float = 0.1,
    **kwargs: Any,
) -> ObservabilityHub:
    """Create an observability hub."""
    config = ObservabilityConfig(
        metrics_retention_hours=metrics_retention_hours,
        trace_sample_rate=trace_sample_rate,
        **kwargs,
    )
    return ObservabilityHub(config)


def create_metrics_collector(
    config: Optional[ObservabilityConfig] = None,
) -> MetricsCollector:
    """Create a metrics collector."""
    return MetricsCollector(config)


def create_trace_manager(
    config: Optional[ObservabilityConfig] = None,
) -> TraceManager:
    """Create a trace manager."""
    return TraceManager(config)


def create_health_monitor(
    config: Optional[ObservabilityConfig] = None,
) -> HealthMonitor:
    """Create a health monitor."""
    return HealthMonitor(config)


def create_alert_manager(
    config: Optional[ObservabilityConfig] = None,
) -> AlertManager:
    """Create an alert manager."""
    return AlertManager(config)


def create_slo_definition(
    name: str,
    target: float,
    window_days: int = 30,
    metric_name: str = "",
    **kwargs: Any,
) -> SLODefinition:
    """Create an SLO definition."""
    return SLODefinition(
        slo_id=str(uuid.uuid4()),
        name=name,
        target=target,
        window_days=window_days,
        metric_name=metric_name,
        **kwargs,
    )


def create_observable_provider(
    base_provider: VisionProvider,
    hub: Optional[ObservabilityHub] = None,
) -> ObservableVisionProvider:
    """Create an observable vision provider."""
    if hub is None:
        hub = create_observability_hub()
    return ObservableVisionProvider(base_provider, hub)
