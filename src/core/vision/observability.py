"""Comprehensive observability for Vision Provider system.

This module provides observability features including:
- Metrics collection and aggregation
- Structured logging
- Alerting and notifications
- Health checks and dashboards
- SLI/SLO monitoring
"""

import asyncio
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Set, TypeVar, Union

from .base import VisionDescription, VisionProvider
from src.utils.safe_eval import safe_eval


class MetricType(Enum):
    """Type of metric."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class LogLevel(Enum):
    """Log level."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """Alert severity level."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""

    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"


class HealthStatus(Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class MetricValue:
    """A metric value."""

    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "labels": dict(self.labels),
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
        }


@dataclass
class HistogramBucket:
    """Histogram bucket."""

    le: float  # Less than or equal
    count: int = 0


@dataclass
class HistogramValue:
    """Histogram metric value."""

    name: str
    buckets: List[HistogramBucket] = field(default_factory=list)
    sum_value: float = 0.0
    count: int = 0
    labels: Dict[str, str] = field(default_factory=dict)

    def observe(self, value: float) -> None:
        """Observe a value."""
        self.sum_value += value
        self.count += 1
        for bucket in self.buckets:
            if value <= bucket.le:
                bucket.count += 1


@dataclass
class LogEntry:
    """A log entry."""

    level: LogLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    logger_name: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "logger": self.logger_name,
            "context": dict(self.context),
            "trace_id": self.trace_id,
            "span_id": self.span_id,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class Alert:
    """An alert."""

    name: str
    severity: AlertSeverity
    message: str
    status: AlertStatus = AlertStatus.PENDING
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "status": self.status.value,
            "labels": dict(self.labels),
            "annotations": dict(self.annotations),
            "started_at": self.started_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "value": self.value,
        }


@dataclass
class AlertRule:
    """Alert rule definition."""

    name: str
    condition: str  # Expression to evaluate
    severity: AlertSeverity
    message_template: str
    for_duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Health check result."""

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "details": dict(self.details),
        }


@dataclass
class SLI:
    """Service Level Indicator."""

    name: str
    description: str
    current_value: float = 0.0
    target_value: float = 0.0
    unit: str = ""
    window: timedelta = field(default_factory=lambda: timedelta(days=30))

    def compliance_ratio(self) -> float:
        """Calculate compliance ratio."""
        if self.target_value == 0:
            return 1.0
        return min(1.0, self.current_value / self.target_value)


@dataclass
class SLO:
    """Service Level Objective."""

    name: str
    sli: SLI
    target: float  # Target percentage (0-100)
    window: timedelta = field(default_factory=lambda: timedelta(days=30))
    error_budget: float = 0.0  # Remaining error budget

    def is_met(self) -> bool:
        """Check if SLO is being met."""
        return self.sli.compliance_ratio() * 100 >= self.target

    def remaining_budget(self) -> float:
        """Calculate remaining error budget."""
        budget_total = 100 - self.target
        used = max(0, 100 - self.sli.compliance_ratio() * 100)
        return max(0, budget_total - used)


class MetricsCollector:
    """Collects and stores metrics."""

    def __init__(self, max_history: int = 10000) -> None:
        """Initialize collector.

        Args:
            max_history: Maximum history entries per metric
        """
        self._metrics: Dict[str, List[MetricValue]] = {}
        self._histograms: Dict[str, HistogramValue] = {}
        self._max_history = max_history
        self._lock = threading.Lock()

    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> None:
        """Increment a counter.

        Args:
            name: Metric name
            value: Value to add
            labels: Metric labels
            description: Metric description
        """
        metric = MetricValue(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            labels=labels or {},
            description=description,
        )
        self._store_metric(metric)

    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> None:
        """Set a gauge value.

        Args:
            name: Metric name
            value: Gauge value
            labels: Metric labels
            description: Metric description
        """
        metric = MetricValue(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            labels=labels or {},
            description=description,
        )
        self._store_metric(metric)

    def histogram(
        self,
        name: str,
        value: float,
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Observe a histogram value.

        Args:
            name: Metric name
            value: Observed value
            buckets: Bucket boundaries
            labels: Metric labels
        """
        key = self._metric_key(name, labels or {})

        with self._lock:
            if key not in self._histograms:
                bucket_list = buckets or [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
                self._histograms[key] = HistogramValue(
                    name=name,
                    buckets=[HistogramBucket(le=b) for b in bucket_list]
                    + [HistogramBucket(le=float("inf"))],
                    labels=labels or {},
                )

            self._histograms[key].observe(value)

    def _store_metric(self, metric: MetricValue) -> None:
        """Store a metric value."""
        key = self._metric_key(metric.name, metric.labels)

        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = []

            self._metrics[key].append(metric)

            if len(self._metrics[key]) > self._max_history:
                self._metrics[key] = self._metrics[key][-self._max_history :]

    def _metric_key(self, name: str, labels: Dict[str, str]) -> str:
        """Generate metric key."""
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_metric(self, name: str, labels: Optional[Dict[str, str]] = None) -> List[MetricValue]:
        """Get metric values.

        Args:
            name: Metric name
            labels: Metric labels

        Returns:
            List of metric values
        """
        key = self._metric_key(name, labels or {})
        with self._lock:
            return list(self._metrics.get(key, []))

    def get_histogram(
        self, name: str, labels: Optional[Dict[str, str]] = None
    ) -> Optional[HistogramValue]:
        """Get histogram.

        Args:
            name: Metric name
            labels: Metric labels

        Returns:
            Histogram value or None
        """
        key = self._metric_key(name, labels or {})
        with self._lock:
            return self._histograms.get(key)

    def get_all_metrics(self) -> Dict[str, List[MetricValue]]:
        """Get all metrics."""
        with self._lock:
            return {k: list(v) for k, v in self._metrics.items()}

    def clear(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._metrics.clear()
            self._histograms.clear()


class StructuredLogger:
    """Structured logging with context."""

    def __init__(
        self,
        name: str = "",
        default_context: Optional[Dict[str, Any]] = None,
        handlers: Optional[List[Callable[[LogEntry], None]]] = None,
    ) -> None:
        """Initialize logger.

        Args:
            name: Logger name
            default_context: Default context for all entries
            handlers: Log handlers
        """
        self._name = name
        self._default_context = default_context or {}
        self._handlers = handlers or []
        self._entries: List[LogEntry] = []
        self._lock = threading.Lock()
        self._trace_id: Optional[str] = None
        self._span_id: Optional[str] = None

    def with_context(self, **context: Any) -> "StructuredLogger":
        """Create logger with additional context.

        Args:
            **context: Additional context

        Returns:
            New logger with context
        """
        new_context = {**self._default_context, **context}
        logger = StructuredLogger(
            name=self._name,
            default_context=new_context,
            handlers=self._handlers,
        )
        logger._trace_id = self._trace_id
        logger._span_id = self._span_id
        return logger

    def with_trace(self, trace_id: str, span_id: Optional[str] = None) -> "StructuredLogger":
        """Create logger with trace context.

        Args:
            trace_id: Trace ID
            span_id: Span ID

        Returns:
            New logger with trace context
        """
        logger = self.with_context()
        logger._trace_id = trace_id
        logger._span_id = span_id
        return logger

    def _log(self, level: LogLevel, message: str, **context: Any) -> None:
        """Log a message.

        Args:
            level: Log level
            message: Log message
            **context: Additional context
        """
        entry = LogEntry(
            level=level,
            message=message,
            logger_name=self._name,
            context={**self._default_context, **context},
            trace_id=self._trace_id,
            span_id=self._span_id,
        )

        with self._lock:
            self._entries.append(entry)

        for handler in self._handlers:
            try:
                handler(entry)
            except Exception:
                pass

    def debug(self, message: str, **context: Any) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **context)

    def info(self, message: str, **context: Any) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, message, **context)

    def warning(self, message: str, **context: Any) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **context)

    def error(self, message: str, **context: Any) -> None:
        """Log error message."""
        self._log(LogLevel.ERROR, message, **context)

    def critical(self, message: str, **context: Any) -> None:
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, **context)

    def get_entries(
        self,
        level: Optional[LogLevel] = None,
        since: Optional[datetime] = None,
    ) -> List[LogEntry]:
        """Get log entries.

        Args:
            level: Filter by level
            since: Filter by timestamp

        Returns:
            List of log entries
        """
        with self._lock:
            entries = list(self._entries)

        if level:
            entries = [e for e in entries if e.level == level]
        if since:
            entries = [e for e in entries if e.timestamp >= since]

        return entries


class AlertManager:
    """Manages alerts."""

    def __init__(self) -> None:
        """Initialize manager."""
        self._rules: Dict[str, AlertRule] = {}
        self._alerts: Dict[str, Alert] = {}
        self._handlers: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()

    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule.

        Args:
            rule: Alert rule
        """
        with self._lock:
            self._rules[rule.name] = rule

    def remove_rule(self, name: str) -> None:
        """Remove alert rule.

        Args:
            name: Rule name
        """
        with self._lock:
            self._rules.pop(name, None)

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add alert handler.

        Args:
            handler: Alert handler function
        """
        self._handlers.append(handler)

    def fire_alert(
        self,
        name: str,
        severity: AlertSeverity,
        message: str,
        value: Optional[float] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> Alert:
        """Fire an alert.

        Args:
            name: Alert name
            severity: Alert severity
            message: Alert message
            value: Current value
            labels: Alert labels

        Returns:
            Alert object
        """
        alert = Alert(
            name=name,
            severity=severity,
            message=message,
            status=AlertStatus.FIRING,
            labels=labels or {},
            value=value,
        )

        with self._lock:
            self._alerts[name] = alert

        for handler in self._handlers:
            try:
                handler(alert)
            except Exception:
                pass

        return alert

    def resolve_alert(self, name: str) -> Optional[Alert]:
        """Resolve an alert.

        Args:
            name: Alert name

        Returns:
            Resolved alert or None
        """
        with self._lock:
            if name in self._alerts:
                alert = self._alerts[name]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                return alert
        return None

    def get_alerts(
        self,
        status: Optional[AlertStatus] = None,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """Get alerts.

        Args:
            status: Filter by status
            severity: Filter by severity

        Returns:
            List of alerts
        """
        with self._lock:
            alerts = list(self._alerts.values())

        if status:
            alerts = [a for a in alerts if a.status == status]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    def evaluate_condition(
        self,
        condition: str,
        metrics: Dict[str, float],
    ) -> bool:
        """Evaluate alert condition.

        Args:
            condition: Condition expression
            metrics: Current metrics

        Returns:
            True if condition is met
        """
        try:
            # Simple expression evaluation
            # In production, use a proper expression parser
            result = safe_eval(condition, metrics)
            return bool(result)
        except Exception:
            return False


class HealthChecker:
    """Health check manager."""

    def __init__(self) -> None:
        """Initialize checker."""
        self._checks: Dict[str, Callable[[], HealthCheck]] = {}
        self._results: Dict[str, HealthCheck] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        check: Callable[[], HealthCheck],
    ) -> None:
        """Register health check.

        Args:
            name: Check name
            check: Check function
        """
        with self._lock:
            self._checks[name] = check

    def unregister(self, name: str) -> None:
        """Unregister health check.

        Args:
            name: Check name
        """
        with self._lock:
            self._checks.pop(name, None)
            self._results.pop(name, None)

    def run_check(self, name: str) -> Optional[HealthCheck]:
        """Run a specific health check.

        Args:
            name: Check name

        Returns:
            Health check result or None
        """
        with self._lock:
            check = self._checks.get(name)

        if not check:
            return None

        start = time.time()
        try:
            result = check()
            result.latency_ms = (time.time() - start) * 1000
        except Exception as e:
            result = HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=(time.time() - start) * 1000,
            )

        with self._lock:
            self._results[name] = result

        return result

    def run_all(self) -> Dict[str, HealthCheck]:
        """Run all health checks.

        Returns:
            Dictionary of check results
        """
        with self._lock:
            names = list(self._checks.keys())

        results = {}
        for name in names:
            result = self.run_check(name)
            if result:
                results[name] = result

        return results

    def get_overall_status(self) -> HealthStatus:
        """Get overall health status.

        Returns:
            Overall health status
        """
        results = self.run_all()

        if not results:
            return HealthStatus.UNKNOWN

        statuses = [r.status for r in results.values()]

        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY

        return HealthStatus.UNKNOWN

    def get_results(self) -> Dict[str, HealthCheck]:
        """Get cached results.

        Returns:
            Dictionary of cached results
        """
        with self._lock:
            return dict(self._results)


class SLOMonitor:
    """SLO monitoring."""

    def __init__(self, metrics: MetricsCollector) -> None:
        """Initialize monitor.

        Args:
            metrics: Metrics collector
        """
        self._metrics = metrics
        self._slos: Dict[str, SLO] = {}
        self._lock = threading.Lock()

    def register_slo(self, slo: SLO) -> None:
        """Register an SLO.

        Args:
            slo: SLO to register
        """
        with self._lock:
            self._slos[slo.name] = slo

    def unregister_slo(self, name: str) -> None:
        """Unregister an SLO.

        Args:
            name: SLO name
        """
        with self._lock:
            self._slos.pop(name, None)

    def update_sli(self, slo_name: str, value: float) -> None:
        """Update SLI value for an SLO.

        Args:
            slo_name: SLO name
            value: New SLI value
        """
        with self._lock:
            if slo_name in self._slos:
                self._slos[slo_name].sli.current_value = value

    def get_slo_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get SLO status.

        Args:
            name: SLO name

        Returns:
            SLO status or None
        """
        with self._lock:
            slo = self._slos.get(name)

        if not slo:
            return None

        return {
            "name": slo.name,
            "target": slo.target,
            "current": slo.sli.compliance_ratio() * 100,
            "is_met": slo.is_met(),
            "remaining_budget": slo.remaining_budget(),
            "sli": {
                "name": slo.sli.name,
                "current_value": slo.sli.current_value,
                "target_value": slo.sli.target_value,
                "unit": slo.sli.unit,
            },
        }

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get all SLO statuses.

        Returns:
            Dictionary of SLO statuses
        """
        with self._lock:
            names = list(self._slos.keys())

        return {name: status for name in names if (status := self.get_slo_status(name)) is not None}


class ObservabilityContext:
    """Context for observability."""

    def __init__(
        self,
        metrics: MetricsCollector,
        logger: StructuredLogger,
        alerts: AlertManager,
        health: HealthChecker,
        slo_monitor: Optional[SLOMonitor] = None,
    ) -> None:
        """Initialize context.

        Args:
            metrics: Metrics collector
            logger: Structured logger
            alerts: Alert manager
            health: Health checker
            slo_monitor: SLO monitor
        """
        self.metrics = metrics
        self.logger = logger
        self.alerts = alerts
        self.health = health
        self.slo_monitor = slo_monitor or SLOMonitor(metrics)


class ObservableVisionProvider(VisionProvider):
    """Vision provider with full observability."""

    def __init__(
        self,
        provider: VisionProvider,
        context: ObservabilityContext,
        service_name: str = "vision",
    ) -> None:
        """Initialize provider.

        Args:
            provider: Underlying provider
            context: Observability context
            service_name: Service name for metrics
        """
        self._provider = provider
        self._context = context
        self._service_name = service_name
        self._request_count = 0
        self._error_count = 0

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"observable_{self._provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with observability.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        labels = {
            "provider": self._provider.provider_name,
            "service": self._service_name,
        }

        # Record request
        self._request_count += 1
        self._context.metrics.counter(
            "vision_requests_total",
            labels=labels,
        )
        self._context.logger.info(
            "Starting image analysis",
            provider=self._provider.provider_name,
            image_size=len(image_data),
        )

        start_time = time.time()

        try:
            result = await self._provider.analyze_image(image_data, include_description)

            latency_ms = (time.time() - start_time) * 1000

            # Record success metrics
            self._context.metrics.counter(
                "vision_requests_success",
                labels=labels,
            )
            self._context.metrics.histogram(
                "vision_request_latency_ms",
                latency_ms,
                labels=labels,
            )
            self._context.metrics.gauge(
                "vision_confidence",
                result.confidence,
                labels=labels,
            )

            self._context.logger.info(
                "Image analysis completed",
                provider=self._provider.provider_name,
                latency_ms=latency_ms,
                confidence=result.confidence,
            )

            # Check for low confidence alert
            if result.confidence < 0.5:
                self._context.alerts.fire_alert(
                    name="low_confidence",
                    severity=AlertSeverity.WARNING,
                    message=f"Low confidence result: {result.confidence}",
                    value=result.confidence,
                    labels=labels,
                )

            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._error_count += 1

            # Record error metrics
            self._context.metrics.counter(
                "vision_requests_errors",
                labels={**labels, "error_type": type(e).__name__},
            )
            self._context.metrics.histogram(
                "vision_request_latency_ms",
                latency_ms,
                labels=labels,
            )

            self._context.logger.error(
                "Image analysis failed",
                provider=self._provider.provider_name,
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=latency_ms,
            )

            # Fire error alert
            self._context.alerts.fire_alert(
                name="vision_error",
                severity=AlertSeverity.ERROR,
                message=f"Vision analysis error: {e}",
                labels={**labels, "error_type": type(e).__name__},
            )

            raise

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary.

        Returns:
            Metrics summary
        """
        error_rate = self._error_count / self._request_count if self._request_count > 0 else 0.0

        return {
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "error_rate": error_rate,
            "provider": self._provider.provider_name,
        }


def create_observable_provider(
    provider: VisionProvider,
    service_name: str = "vision",
) -> ObservableVisionProvider:
    """Create an observable vision provider.

    Args:
        provider: Provider to wrap
        service_name: Service name

    Returns:
        Observable provider
    """
    metrics = MetricsCollector()
    logger = StructuredLogger(name=service_name)
    alerts = AlertManager()
    health = HealthChecker()

    # Register provider health check
    def provider_health() -> HealthCheck:
        return HealthCheck(
            name=f"{provider.provider_name}_health",
            status=HealthStatus.HEALTHY,
            message="Provider available",
        )

    health.register(f"{provider.provider_name}_health", provider_health)

    context = ObservabilityContext(
        metrics=metrics,
        logger=logger,
        alerts=alerts,
        health=health,
    )

    return ObservableVisionProvider(
        provider=provider,
        context=context,
        service_name=service_name,
    )
