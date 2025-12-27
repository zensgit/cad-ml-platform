"""
Tests for Phase 21: Advanced Observability & Telemetry.

Tests observability hub module components including metrics, tracing,
health monitoring, alerting, SLO tracking, and anomaly detection.
"""

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.vision.base import VisionDescription
from src.core.vision.observability_hub import (  # Enums; Dataclasses; Classes; Factory functions
    Alert,
    AlertManager,
    AlertSeverity,
    AlertState,
    Anomaly,
    AnomalyDetector,
    AnomalyType,
    HealthCheck,
    HealthMonitor,
    HealthStatus,
    MetricPoint,
    MetricsCollector,
    MetricSeries,
    MetricType,
    ObservabilityConfig,
    ObservabilityHub,
    ObservableVisionProvider,
    SLODefinition,
    SLOResult,
    SLOStatus,
    SLOTracker,
    Span,
    SpanContext,
    TraceManager,
    TraceStatus,
    create_alert_manager,
    create_health_monitor,
    create_metrics_collector,
    create_observability_hub,
    create_observable_provider,
    create_slo_definition,
    create_trace_manager,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def observability_config():
    """Create a test observability configuration."""
    return ObservabilityConfig(
        metrics_retention_hours=12,
        trace_sample_rate=1.0,  # 100% sample rate for testing
        health_check_interval_seconds=10,
        alert_evaluation_interval_seconds=30,
        anomaly_detection_enabled=True,
        slo_tracking_enabled=True,
    )


@pytest.fixture
def metrics_collector(observability_config):
    """Create a metrics collector."""
    return MetricsCollector(observability_config)


@pytest.fixture
def trace_manager(observability_config):
    """Create a trace manager."""
    return TraceManager(observability_config)


@pytest.fixture
def health_monitor(observability_config):
    """Create a health monitor."""
    return HealthMonitor(observability_config)


@pytest.fixture
def alert_manager(observability_config):
    """Create an alert manager."""
    return AlertManager(observability_config)


@pytest.fixture
def observability_hub(observability_config):
    """Create an observability hub."""
    return ObservabilityHub(observability_config)


@pytest.fixture
def mock_provider():
    """Create a mock vision provider."""
    provider = MagicMock()
    provider.provider_name = "mock_provider"
    provider.analyze_image = AsyncMock(
        return_value=VisionDescription(
            summary="Test summary",
            details=["Detail 1", "Detail 2"],
            confidence=0.95,
        )
    )
    return provider


# ============================================================================
# Test Enums
# ============================================================================


class TestObservabilityEnums:
    """Tests for observability-related enums."""

    def test_metric_type_values(self):
        """Test MetricType enum values."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"
        assert MetricType.TIMER.value == "timer"

    def test_health_status_values(self):
        """Test HealthStatus enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"

    def test_alert_severity_values(self):
        """Test AlertSeverity enum values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_alert_state_values(self):
        """Test AlertState enum values."""
        assert AlertState.PENDING.value == "pending"
        assert AlertState.FIRING.value == "firing"
        assert AlertState.RESOLVED.value == "resolved"
        assert AlertState.SUPPRESSED.value == "suppressed"

    def test_trace_status_values(self):
        """Test TraceStatus enum values."""
        assert TraceStatus.UNSET.value == "unset"
        assert TraceStatus.OK.value == "ok"
        assert TraceStatus.ERROR.value == "error"

    def test_slo_status_enum_values(self):
        """Test SLOStatus enum values."""
        assert SLOStatus.MET.value == "met"
        assert SLOStatus.AT_RISK.value == "at_risk"
        assert SLOStatus.BREACHED.value == "breached"

    def test_anomaly_type_values(self):
        """Test AnomalyType enum values."""
        assert AnomalyType.SPIKE.value == "spike"
        assert AnomalyType.DROP.value == "drop"
        assert AnomalyType.TREND.value == "trend"
        assert AnomalyType.SEASONALITY.value == "seasonality"
        assert AnomalyType.OUTLIER.value == "outlier"


# ============================================================================
# Test Dataclasses
# ============================================================================


class TestObservabilityDataclasses:
    """Tests for observability dataclasses."""

    def test_metric_point_creation(self):
        """Test MetricPoint creation."""
        point = MetricPoint(
            name="test_metric",
            value=42.5,
            metric_type=MetricType.GAUGE,
            labels={"env": "test"},
            unit="ms",
        )
        assert point.name == "test_metric"
        assert point.value == 42.5
        assert point.metric_type == MetricType.GAUGE
        assert point.labels == {"env": "test"}
        assert point.unit == "ms"
        assert isinstance(point.timestamp, datetime)

    def test_metric_series_add_point(self):
        """Test MetricSeries add_point method."""
        series = MetricSeries(
            name="test_series",
            metric_type=MetricType.COUNTER,
        )
        assert len(series.points) == 0

        series.add_point(10.0)
        series.add_point(20.0)

        assert len(series.points) == 2
        assert series.points[0].value == 10.0
        assert series.points[1].value == 20.0

    def test_metric_series_get_latest(self):
        """Test MetricSeries get_latest method."""
        series = MetricSeries(
            name="test_series",
            metric_type=MetricType.GAUGE,
        )
        assert series.get_latest() is None

        series.add_point(5.0)
        series.add_point(15.0)

        latest = series.get_latest()
        assert latest is not None
        assert latest.value == 15.0

    def test_span_context_creation(self):
        """Test SpanContext creation."""
        context = SpanContext(
            trace_id="trace-123",
            span_id="span-456",
            parent_span_id="parent-789",
            baggage={"user_id": "user-1"},
            sampled=True,
        )
        assert context.trace_id == "trace-123"
        assert context.span_id == "span-456"
        assert context.parent_span_id == "parent-789"
        assert context.baggage == {"user_id": "user-1"}
        assert context.sampled is True

    def test_span_creation_and_finish(self):
        """Test Span creation and finish method."""
        context = SpanContext(trace_id="t1", span_id="s1")
        span = Span(
            context=context,
            operation_name="test_op",
            service_name="test_service",
            tags={"key": "value"},
        )

        assert span.operation_name == "test_op"
        assert span.service_name == "test_service"
        assert span.status == TraceStatus.UNSET
        assert span.end_time is None

        span.finish(TraceStatus.OK)

        assert span.status == TraceStatus.OK
        assert span.end_time is not None
        assert span.duration_ms >= 0

    def test_span_log(self):
        """Test Span log method."""
        context = SpanContext(trace_id="t1", span_id="s1")
        span = Span(
            context=context,
            operation_name="test_op",
            service_name="test_service",
        )

        span.log("Test message", key="value")

        assert len(span.logs) == 1
        assert span.logs[0]["message"] == "Test message"
        assert span.logs[0]["key"] == "value"
        assert "timestamp" in span.logs[0]

    def test_health_check_creation(self):
        """Test HealthCheck creation."""
        check = HealthCheck(
            name="db_check",
            status=HealthStatus.HEALTHY,
            message="Database is responsive",
            duration_ms=15.5,
            details={"connections": 10},
        )
        assert check.name == "db_check"
        assert check.status == HealthStatus.HEALTHY
        assert check.message == "Database is responsive"
        assert check.duration_ms == 15.5
        assert check.details == {"connections": 10}

    def test_alert_creation(self):
        """Test Alert creation."""
        alert = Alert(
            alert_id="alert-1",
            name="HighLatency",
            severity=AlertSeverity.WARNING,
            state=AlertState.FIRING,
            message="Latency exceeded threshold",
            source="api-gateway",
            labels={"service": "api"},
            value=250.0,
        )
        assert alert.alert_id == "alert-1"
        assert alert.name == "HighLatency"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.state == AlertState.FIRING
        assert alert.value == 250.0

    def test_slo_definition_creation(self):
        """Test SLODefinition creation."""
        slo = SLODefinition(
            slo_id="slo-1",
            name="API Availability",
            target=99.9,
            window_days=30,
            metric_name="success_rate",
            description="API availability SLO",
        )
        assert slo.slo_id == "slo-1"
        assert slo.name == "API Availability"
        assert slo.target == 99.9
        assert slo.window_days == 30

    def test_anomaly_creation(self):
        """Test Anomaly creation."""
        anomaly = Anomaly(
            anomaly_id="anomaly-1",
            anomaly_type=AnomalyType.SPIKE,
            metric_name="request_latency",
            expected_value=100.0,
            actual_value=500.0,
            deviation=400.0,
            confidence=0.95,
        )
        assert anomaly.anomaly_id == "anomaly-1"
        assert anomaly.anomaly_type == AnomalyType.SPIKE
        assert anomaly.deviation == 400.0
        assert anomaly.confidence == 0.95

    def test_observability_config_defaults(self):
        """Test ObservabilityConfig default values."""
        config = ObservabilityConfig()
        assert config.metrics_retention_hours == 24
        assert config.trace_sample_rate == 0.1
        assert config.health_check_interval_seconds == 30
        assert config.alert_evaluation_interval_seconds == 60
        assert config.anomaly_detection_enabled is True
        assert config.slo_tracking_enabled is True


# ============================================================================
# Test MetricsCollector
# ============================================================================


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_increment_counter(self, metrics_collector):
        """Test counter increment."""
        metrics_collector.increment("requests_total")
        assert metrics_collector.get_counter("requests_total") == 1.0

        metrics_collector.increment("requests_total", 5.0)
        assert metrics_collector.get_counter("requests_total") == 6.0

    def test_increment_with_labels(self, metrics_collector):
        """Test counter increment with labels."""
        labels = {"method": "GET", "path": "/api"}
        metrics_collector.increment("http_requests", labels=labels)
        metrics_collector.increment("http_requests", 2.0, labels=labels)

        assert metrics_collector.get_counter("http_requests", labels=labels) == 3.0
        assert metrics_collector.get_counter("http_requests", labels={"method": "POST"}) == 0

    def test_gauge_set(self, metrics_collector):
        """Test gauge setting."""
        metrics_collector.gauge("active_connections", 100)
        assert metrics_collector.get_gauge("active_connections") == 100

        metrics_collector.gauge("active_connections", 150)
        assert metrics_collector.get_gauge("active_connections") == 150

    def test_histogram_record(self, metrics_collector):
        """Test histogram recording."""
        for value in [10, 20, 30, 40, 50]:
            metrics_collector.histogram("request_duration", value)

        stats = metrics_collector.get_histogram_stats("request_duration")
        assert stats["count"] == 5
        assert stats["min"] == 10
        assert stats["max"] == 50
        assert stats["mean"] == 30

    def test_timer_record(self, metrics_collector):
        """Test timer recording."""
        metrics_collector.timer("operation_time", 100.5)
        metrics_collector.timer("operation_time", 200.5)

        stats = metrics_collector.get_histogram_stats("operation_time")
        assert stats["count"] == 2

    def test_get_metric_series(self, metrics_collector):
        """Test getting metric series."""
        metrics_collector.increment("counter_test")
        series = metrics_collector.get_metric("counter_test")

        assert series is not None
        assert series.name == "counter_test"
        assert series.metric_type == MetricType.COUNTER

    def test_list_metrics(self, metrics_collector):
        """Test listing all metrics."""
        metrics_collector.increment("metric_a")
        metrics_collector.gauge("metric_b", 10)
        metrics_collector.histogram("metric_c", 5)

        metrics = metrics_collector.list_metrics()
        assert "metric_a" in metrics
        assert "metric_b" in metrics
        assert "metric_c" in metrics

    def test_export_metrics(self, metrics_collector):
        """Test exporting metrics."""
        metrics_collector.increment("export_test_1")
        metrics_collector.gauge("export_test_2", 42)

        exported = metrics_collector.export()
        assert len(exported) >= 2
        names = [p.name for p in exported]
        assert "export_test_1" in names
        assert "export_test_2" in names


# ============================================================================
# Test TraceManager
# ============================================================================


class TestTraceManager:
    """Tests for TraceManager class."""

    def test_start_span(self, trace_manager):
        """Test starting a new span."""
        span = trace_manager.start_span(
            operation_name="test_operation",
            service_name="test_service",
        )

        assert span.operation_name == "test_operation"
        assert span.service_name == "test_service"
        assert span.context.trace_id is not None
        assert span.context.span_id is not None
        assert span.context.parent_span_id is None

    def test_start_span_with_parent(self, trace_manager):
        """Test starting a span with parent context."""
        parent = trace_manager.start_span(
            operation_name="parent_op",
            service_name="service",
        )

        child = trace_manager.start_span(
            operation_name="child_op",
            service_name="service",
            parent_context=parent.context,
        )

        assert child.context.trace_id == parent.context.trace_id
        assert child.context.parent_span_id == parent.context.span_id

    def test_finish_span(self, trace_manager):
        """Test finishing a span."""
        span = trace_manager.start_span(
            operation_name="finish_test",
            service_name="service",
        )

        trace_manager.finish_span(span, TraceStatus.OK)

        assert span.status == TraceStatus.OK
        assert span.end_time is not None

    def test_get_trace(self, trace_manager):
        """Test getting trace spans."""
        span1 = trace_manager.start_span("op1", "service")
        span2 = trace_manager.start_span("op2", "service", parent_context=span1.context)

        trace_spans = trace_manager.get_trace(span1.context.trace_id)
        assert len(trace_spans) == 2

    def test_get_active_spans(self, trace_manager):
        """Test getting active spans."""
        span1 = trace_manager.start_span("active1", "service")
        span2 = trace_manager.start_span("active2", "service")

        active = trace_manager.get_active_spans()
        assert len(active) == 2

        trace_manager.finish_span(span1)
        active = trace_manager.get_active_spans()
        assert len(active) == 1

    def test_inject_context(self, trace_manager):
        """Test injecting trace context into headers."""
        span = trace_manager.start_span("inject_test", "service")
        headers = trace_manager.inject_context(span)

        assert "X-Trace-ID" in headers
        assert "X-Span-ID" in headers
        assert headers["X-Trace-ID"] == span.context.trace_id

    def test_extract_context(self, trace_manager):
        """Test extracting trace context from headers."""
        headers = {
            "X-Trace-ID": "trace-abc",
            "X-Span-ID": "span-xyz",
            "X-Sampled": "true",
        }

        context = trace_manager.extract_context(headers)
        assert context is not None
        assert context.trace_id == "trace-abc"
        assert context.span_id == "span-xyz"
        assert context.sampled is True

    def test_extract_context_missing_headers(self, trace_manager):
        """Test extracting context with missing headers."""
        headers = {"X-Trace-ID": "trace-abc"}
        context = trace_manager.extract_context(headers)
        assert context is None


# ============================================================================
# Test HealthMonitor
# ============================================================================


class TestHealthMonitor:
    """Tests for HealthMonitor class."""

    def test_register_check(self, health_monitor):
        """Test registering a health check."""

        def check_fn():
            return HealthCheck(name="test", status=HealthStatus.HEALTHY)

        health_monitor.register_check("test_check", check_fn)
        assert "test_check" in health_monitor.list_checks()

    def test_unregister_check(self, health_monitor):
        """Test unregistering a health check."""

        def check_fn():
            return HealthCheck(name="test", status=HealthStatus.HEALTHY)

        health_monitor.register_check("to_remove", check_fn)
        assert health_monitor.unregister_check("to_remove") is True
        assert "to_remove" not in health_monitor.list_checks()
        assert health_monitor.unregister_check("nonexistent") is False

    def test_run_check(self, health_monitor):
        """Test running a health check."""

        def healthy_check():
            return HealthCheck(name="healthy", status=HealthStatus.HEALTHY)

        health_monitor.register_check("healthy_check", healthy_check)
        result = health_monitor.run_check("healthy_check")

        assert result is not None
        assert result.status == HealthStatus.HEALTHY
        assert result.duration_ms >= 0

    def test_run_check_with_exception(self, health_monitor):
        """Test running a check that throws an exception."""

        def failing_check():
            raise RuntimeError("Check failed")

        health_monitor.register_check("failing_check", failing_check)
        result = health_monitor.run_check("failing_check")

        assert result is not None
        assert result.status == HealthStatus.UNHEALTHY
        assert "Check failed" in result.message

    def test_run_all_checks(self, health_monitor):
        """Test running all health checks."""

        def check1():
            return HealthCheck(name="check1", status=HealthStatus.HEALTHY)

        def check2():
            return HealthCheck(name="check2", status=HealthStatus.DEGRADED)

        health_monitor.register_check("check1", check1)
        health_monitor.register_check("check2", check2)

        results = health_monitor.run_all_checks()
        assert len(results) == 2
        assert "check1" in results
        assert "check2" in results

    def test_get_overall_status_healthy(self, health_monitor):
        """Test overall status when all checks are healthy."""

        def healthy():
            return HealthCheck(name="h", status=HealthStatus.HEALTHY)

        health_monitor.register_check("h1", healthy)
        health_monitor.register_check("h2", healthy)
        health_monitor.run_all_checks()

        assert health_monitor.get_overall_status() == HealthStatus.HEALTHY

    def test_get_overall_status_degraded(self, health_monitor):
        """Test overall status when some checks are degraded."""

        def healthy():
            return HealthCheck(name="h", status=HealthStatus.HEALTHY)

        def degraded():
            return HealthCheck(name="d", status=HealthStatus.DEGRADED)

        health_monitor.register_check("h1", healthy)
        health_monitor.register_check("d1", degraded)
        health_monitor.run_all_checks()

        assert health_monitor.get_overall_status() == HealthStatus.DEGRADED

    def test_get_overall_status_unhealthy(self, health_monitor):
        """Test overall status when any check is unhealthy."""

        def healthy():
            return HealthCheck(name="h", status=HealthStatus.HEALTHY)

        def unhealthy():
            return HealthCheck(name="u", status=HealthStatus.UNHEALTHY)

        health_monitor.register_check("h1", healthy)
        health_monitor.register_check("u1", unhealthy)
        health_monitor.run_all_checks()

        assert health_monitor.get_overall_status() == HealthStatus.UNHEALTHY

    def test_get_overall_status_unknown(self, health_monitor):
        """Test overall status when no checks have run."""
        assert health_monitor.get_overall_status() == HealthStatus.UNKNOWN


# ============================================================================
# Test AlertManager
# ============================================================================


class TestAlertManager:
    """Tests for AlertManager class."""

    def test_register_rule(self, alert_manager):
        """Test registering an alert rule."""

        def rule(ctx):
            if ctx.get("error_rate", 0) > 0.1:
                return Alert(
                    alert_id="high-error",
                    name="HighErrorRate",
                    severity=AlertSeverity.WARNING,
                )
            return None

        alert_manager.register_rule("error_rate_rule", rule)
        # Rule is registered (no direct way to check, but evaluate_rules will use it)

    def test_unregister_rule(self, alert_manager):
        """Test unregistering an alert rule."""

        def rule(ctx):
            return None

        alert_manager.register_rule("temp_rule", rule)
        assert alert_manager.unregister_rule("temp_rule") is True
        assert alert_manager.unregister_rule("nonexistent") is False

    def test_fire_alert(self, alert_manager):
        """Test firing an alert."""
        alert = Alert(
            alert_id="fire-test",
            name="TestAlert",
            severity=AlertSeverity.ERROR,
        )

        alert_manager.fire_alert(alert)

        assert alert.state == AlertState.FIRING
        assert alert.firing_since is not None
        assert alert_manager.get_alert("fire-test") is not None

    def test_fire_alert_with_suppression(self, alert_manager):
        """Test firing an alert with suppression."""
        # Add suppression rule for low severity alerts
        alert_manager.add_suppression_rule(lambda a: a.severity == AlertSeverity.INFO)

        alert = Alert(
            alert_id="suppress-test",
            name="InfoAlert",
            severity=AlertSeverity.INFO,
        )

        alert_manager.fire_alert(alert)
        assert alert.state == AlertState.SUPPRESSED

    def test_resolve_alert(self, alert_manager):
        """Test resolving an alert."""
        alert = Alert(
            alert_id="resolve-test",
            name="TestAlert",
            severity=AlertSeverity.WARNING,
        )
        alert_manager.fire_alert(alert)

        resolved = alert_manager.resolve_alert("resolve-test")
        assert resolved is not None
        assert resolved.state == AlertState.RESOLVED
        assert resolved.resolved_at is not None

    def test_get_firing_alerts(self, alert_manager):
        """Test getting firing alerts."""
        for i in range(3):
            alert = Alert(
                alert_id=f"firing-{i}",
                name=f"Alert{i}",
                severity=AlertSeverity.WARNING,
            )
            alert_manager.fire_alert(alert)

        firing = alert_manager.get_firing_alerts()
        assert len(firing) == 3

    def test_get_alerts_by_severity(self, alert_manager):
        """Test getting alerts by severity."""
        alert_manager.fire_alert(
            Alert(alert_id="crit-1", name="Critical1", severity=AlertSeverity.CRITICAL)
        )
        alert_manager.fire_alert(
            Alert(alert_id="warn-1", name="Warning1", severity=AlertSeverity.WARNING)
        )
        alert_manager.fire_alert(
            Alert(alert_id="crit-2", name="Critical2", severity=AlertSeverity.CRITICAL)
        )

        critical = alert_manager.get_alerts_by_severity(AlertSeverity.CRITICAL)
        assert len(critical) == 2

    def test_add_handler(self, alert_manager):
        """Test adding an alert handler."""
        handled_alerts = []

        def handler(alert):
            handled_alerts.append(alert)

        alert_manager.add_handler(handler)

        alert = Alert(
            alert_id="handler-test",
            name="HandlerTest",
            severity=AlertSeverity.ERROR,
        )
        alert_manager.fire_alert(alert)

        assert len(handled_alerts) == 1
        assert handled_alerts[0].alert_id == "handler-test"

    def test_evaluate_rules(self, alert_manager):
        """Test evaluating all rules."""

        def high_latency_rule(ctx):
            if ctx.get("latency_ms", 0) > 1000:
                return Alert(
                    alert_id="high-latency",
                    name="HighLatency",
                    severity=AlertSeverity.WARNING,
                    value=ctx.get("latency_ms", 0),
                )
            return None

        alert_manager.register_rule("latency_rule", high_latency_rule)

        # Should not fire
        fired = alert_manager.evaluate_rules({"latency_ms": 500})
        assert len(fired) == 0

        # Should fire
        fired = alert_manager.evaluate_rules({"latency_ms": 1500})
        assert len(fired) == 1
        assert fired[0].name == "HighLatency"


# ============================================================================
# Test SLOTracker
# ============================================================================


class TestSLOTracker:
    """Tests for SLOTracker class."""

    def test_define_slo(self, metrics_collector):
        """Test defining an SLO."""
        tracker = SLOTracker(metrics_collector)
        slo = SLODefinition(
            slo_id="slo-1",
            name="API Availability",
            target=99.9,
        )

        tracker.define_slo(slo)
        slos = tracker.list_slos()
        assert len(slos) == 1
        assert slos[0].name == "API Availability"

    def test_record_event(self, metrics_collector):
        """Test recording SLO events."""
        tracker = SLOTracker(metrics_collector)
        slo = SLODefinition(slo_id="slo-events", name="Test", target=99.0)
        tracker.define_slo(slo)

        # Record 9 good events and 1 bad event
        for _ in range(9):
            tracker.record_event("slo-events", is_good=True)
        tracker.record_event("slo-events", is_good=False)

        status = tracker.get_slo_status("slo-events")
        assert status is not None
        assert status.current_value == 90.0  # 9/10 = 90%

    def test_get_slo_status_met(self, metrics_collector):
        """Test SLO status when target is met."""
        tracker = SLOTracker(metrics_collector)
        slo = SLODefinition(slo_id="met-slo", name="Met", target=95.0)
        tracker.define_slo(slo)

        # Record 100% success
        for _ in range(100):
            tracker.record_event("met-slo", is_good=True)

        status = tracker.get_slo_status("met-slo")
        assert status.status == "met"
        assert status.current_value == 100.0

    def test_get_slo_status_above_target(self, metrics_collector):
        """Test SLO status when above target but not perfect."""
        tracker = SLOTracker(metrics_collector)
        # Use a target of 95% which gives 5% error budget
        slo = SLODefinition(slo_id="above-slo", name="AboveTarget", target=95.0)
        tracker.define_slo(slo)

        # Record 96% success - above target (95%), so it's "met"
        for _ in range(96):
            tracker.record_event("above-slo", is_good=True)
        for _ in range(4):
            tracker.record_event("above-slo", is_good=False)

        status = tracker.get_slo_status("above-slo")
        # 96% >= 95% target, so it's "met"
        assert status.status == "met"
        assert status.current_value == 96.0

    def test_get_slo_status_breached(self, metrics_collector):
        """Test SLO status when breached."""
        tracker = SLOTracker(metrics_collector)
        slo = SLODefinition(slo_id="breach-slo", name="Breached", target=99.0)
        tracker.define_slo(slo)

        # Record 95% success (breached at 99% target)
        for _ in range(95):
            tracker.record_event("breach-slo", is_good=True)
        for _ in range(5):
            tracker.record_event("breach-slo", is_good=False)

        status = tracker.get_slo_status("breach-slo")
        assert status.status == "breached"

    def test_get_slo_status_nonexistent(self, metrics_collector):
        """Test getting status for nonexistent SLO."""
        tracker = SLOTracker(metrics_collector)
        status = tracker.get_slo_status("nonexistent")
        assert status is None


# ============================================================================
# Test AnomalyDetector
# ============================================================================


class TestAnomalyDetector:
    """Tests for AnomalyDetector class."""

    def test_calculate_baseline(self, metrics_collector):
        """Test calculating baseline for a metric."""
        detector = AnomalyDetector(metrics_collector)

        # Record some baseline data
        for value in [100, 102, 98, 101, 99, 100, 103, 97, 100, 101]:
            metrics_collector.histogram("baseline_metric", value)

        baseline = detector.calculate_baseline("baseline_metric")
        assert baseline is not None
        assert "mean" in baseline
        assert "stdev" in baseline
        assert baseline["mean"] == pytest.approx(100.1, rel=0.1)

    def test_calculate_baseline_insufficient_data(self, metrics_collector):
        """Test calculating baseline with insufficient data."""
        detector = AnomalyDetector(metrics_collector)

        # Only record 5 points (need at least 10)
        for value in [100, 101, 102, 103, 104]:
            metrics_collector.histogram("small_metric", value)

        baseline = detector.calculate_baseline("small_metric")
        assert baseline is None

    def test_detect_spike(self, metrics_collector):
        """Test detecting a spike anomaly."""
        detector = AnomalyDetector(metrics_collector, sensitivity=2.0)

        # Build baseline
        for value in [100, 102, 98, 101, 99, 100, 103, 97, 100, 101]:
            metrics_collector.histogram("spike_metric", value)
        detector.calculate_baseline("spike_metric")

        # Detect spike
        anomaly = detector.detect("spike_metric", 150.0)  # Much higher than baseline
        assert anomaly is not None
        assert anomaly.anomaly_type == AnomalyType.SPIKE
        assert anomaly.actual_value == 150.0

    def test_detect_drop(self, metrics_collector):
        """Test detecting a drop anomaly."""
        detector = AnomalyDetector(metrics_collector, sensitivity=2.0)

        # Build baseline
        for value in [100, 102, 98, 101, 99, 100, 103, 97, 100, 101]:
            metrics_collector.histogram("drop_metric", value)
        detector.calculate_baseline("drop_metric")

        # Detect drop
        anomaly = detector.detect("drop_metric", 50.0)  # Much lower than baseline
        assert anomaly is not None
        assert anomaly.anomaly_type == AnomalyType.DROP

    def test_detect_normal(self, metrics_collector):
        """Test normal values don't trigger anomaly."""
        detector = AnomalyDetector(metrics_collector, sensitivity=2.0)

        # Build baseline
        for value in [100, 102, 98, 101, 99, 100, 103, 97, 100, 101]:
            metrics_collector.histogram("normal_metric", value)
        detector.calculate_baseline("normal_metric")

        # Should not detect anomaly for normal value
        anomaly = detector.detect("normal_metric", 101.0)
        assert anomaly is None

    def test_get_detected_anomalies(self, metrics_collector):
        """Test getting detected anomalies."""
        detector = AnomalyDetector(metrics_collector, sensitivity=2.0)

        # Build baseline
        for value in [100, 102, 98, 101, 99, 100, 103, 97, 100, 101]:
            metrics_collector.histogram("anomaly_metric", value)
        detector.calculate_baseline("anomaly_metric")

        # Trigger some anomalies
        detector.detect("anomaly_metric", 150.0)
        detector.detect("anomaly_metric", 50.0)

        anomalies = detector.get_detected_anomalies()
        assert len(anomalies) == 2


# ============================================================================
# Test ObservabilityHub
# ============================================================================


class TestObservabilityHub:
    """Tests for ObservabilityHub class."""

    def test_hub_initialization(self, observability_hub):
        """Test ObservabilityHub initialization."""
        assert observability_hub.metrics is not None
        assert observability_hub.traces is not None
        assert observability_hub.health is not None
        assert observability_hub.alerts is not None
        assert observability_hub.slo_tracker is not None
        assert observability_hub.anomaly_detector is not None

    def test_hub_start_stop(self, observability_hub):
        """Test starting and stopping the hub."""
        observability_hub.start()
        assert observability_hub._running is True

        observability_hub.stop()
        assert observability_hub._running is False

    def test_get_dashboard_data(self, observability_hub):
        """Test getting dashboard data."""
        # Add some data
        observability_hub.metrics.increment("test_counter")
        observability_hub.health.register_check(
            "test_check", lambda: HealthCheck(name="test", status=HealthStatus.HEALTHY)
        )
        observability_hub.health.run_all_checks()

        data = observability_hub.get_dashboard_data()

        assert "health" in data
        assert "metrics" in data
        assert "traces" in data
        assert "alerts" in data
        assert "slos" in data
        assert "anomalies" in data
        assert "timestamp" in data

    def test_hub_component_integration(self, observability_hub):
        """Test integration between hub components."""
        # Record metrics
        observability_hub.metrics.increment("requests", labels={"service": "api"})

        # Start a trace
        span = observability_hub.traces.start_span("request", "api")

        # Register health check
        observability_hub.health.register_check(
            "api_health", lambda: HealthCheck(name="api", status=HealthStatus.HEALTHY)
        )

        # Fire an alert
        alert = Alert(
            alert_id="test-alert",
            name="TestAlert",
            severity=AlertSeverity.WARNING,
        )
        observability_hub.alerts.fire_alert(alert)

        # Verify all components have data
        assert observability_hub.metrics.get_counter("requests", labels={"service": "api"}) == 1
        assert len(observability_hub.traces.get_active_spans()) == 1
        assert len(observability_hub.health.list_checks()) == 1
        assert len(observability_hub.alerts.get_firing_alerts()) == 1


# ============================================================================
# Test ObservableVisionProvider
# ============================================================================


class TestObservableVisionProvider:
    """Tests for ObservableVisionProvider class."""

    def test_provider_name(self, mock_provider, observability_hub):
        """Test provider name property."""
        provider = ObservableVisionProvider(mock_provider, observability_hub)
        assert provider.provider_name == "observable_mock_provider"

    def test_health_check_registration(self, mock_provider, observability_hub):
        """Test health check is registered."""
        provider = ObservableVisionProvider(mock_provider, observability_hub)
        checks = observability_hub.health.list_checks()
        assert "provider_mock_provider" in checks

    @pytest.mark.asyncio
    async def test_analyze_image_success(self, mock_provider, observability_hub):
        """Test successful image analysis with observability."""
        provider = ObservableVisionProvider(mock_provider, observability_hub)

        result = await provider.analyze_image(b"test_image_data")

        assert result.summary == "Test summary"
        assert result.confidence == 0.95

        # Check metrics were recorded
        requests = observability_hub.metrics.get_counter(
            "vision_requests_total", labels={"provider": "mock_provider"}
        )
        assert requests == 1

        success = observability_hub.metrics.get_counter(
            "vision_requests_success", labels={"provider": "mock_provider"}
        )
        assert success == 1

    @pytest.mark.asyncio
    async def test_analyze_image_error(self, mock_provider, observability_hub):
        """Test image analysis error with observability."""
        mock_provider.analyze_image = AsyncMock(side_effect=RuntimeError("API Error"))
        provider = ObservableVisionProvider(mock_provider, observability_hub)

        with pytest.raises(RuntimeError):
            await provider.analyze_image(b"test_image_data")

        # Check error metrics were recorded
        errors = observability_hub.metrics.get_counter(
            "vision_requests_error", labels={"provider": "mock_provider"}
        )
        assert errors == 1

        # Check alert was fired
        firing = observability_hub.alerts.get_firing_alerts()
        assert len(firing) == 1
        assert firing[0].name == "VisionProviderError"

    @pytest.mark.asyncio
    async def test_tracing_integration(self, mock_provider, observability_hub):
        """Test that traces are created during analysis."""
        provider = ObservableVisionProvider(mock_provider, observability_hub)

        await provider.analyze_image(b"test_image_data")

        # There should be no active spans after completion
        active_spans = observability_hub.traces.get_active_spans()
        assert len(active_spans) == 0


# ============================================================================
# Test Factory Functions
# ============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_observability_hub(self):
        """Test create_observability_hub factory."""
        hub = create_observability_hub(
            metrics_retention_hours=48,
            trace_sample_rate=0.5,
        )

        assert hub is not None
        assert hub._config.metrics_retention_hours == 48
        assert hub._config.trace_sample_rate == 0.5

    def test_create_metrics_collector(self):
        """Test create_metrics_collector factory."""
        collector = create_metrics_collector()
        assert collector is not None
        assert isinstance(collector, MetricsCollector)

    def test_create_trace_manager(self):
        """Test create_trace_manager factory."""
        manager = create_trace_manager()
        assert manager is not None
        assert isinstance(manager, TraceManager)

    def test_create_health_monitor(self):
        """Test create_health_monitor factory."""
        monitor = create_health_monitor()
        assert monitor is not None
        assert isinstance(monitor, HealthMonitor)

    def test_create_alert_manager(self):
        """Test create_alert_manager factory."""
        manager = create_alert_manager()
        assert manager is not None
        assert isinstance(manager, AlertManager)

    def test_create_slo_definition(self):
        """Test create_slo_definition factory."""
        slo = create_slo_definition(
            name="Test SLO",
            target=99.5,
            window_days=7,
        )

        assert slo is not None
        assert slo.name == "Test SLO"
        assert slo.target == 99.5
        assert slo.window_days == 7
        assert slo.slo_id is not None

    def test_create_observable_provider(self, mock_provider):
        """Test create_observable_provider factory."""
        provider = create_observable_provider(mock_provider)
        assert provider is not None
        assert isinstance(provider, ObservableVisionProvider)

    def test_create_observable_provider_with_hub(self, mock_provider, observability_hub):
        """Test create_observable_provider with existing hub."""
        provider = create_observable_provider(mock_provider, observability_hub)
        assert provider is not None
        assert provider._hub is observability_hub


# ============================================================================
# Integration Tests
# ============================================================================


class TestObservabilityIntegration:
    """Integration tests for the observability system."""

    @pytest.mark.asyncio
    async def test_full_observability_pipeline(self, mock_provider):
        """Test complete observability pipeline."""
        # Create hub with full configuration
        hub = create_observability_hub(
            trace_sample_rate=1.0,
            anomaly_detection_enabled=True,
            slo_tracking_enabled=True,
        )

        # Define an SLO
        slo = create_slo_definition(
            name="Vision API Availability",
            target=99.0,
        )
        hub.slo_tracker.define_slo(slo)

        # Create observable provider
        provider = create_observable_provider(mock_provider, hub)

        # Register alert handler
        fired_alerts = []
        hub.alerts.add_handler(lambda a: fired_alerts.append(a))

        # Run multiple requests
        for _ in range(10):
            result = await provider.analyze_image(b"test_data")
            assert result is not None

        # Verify metrics
        total_requests = hub.metrics.get_counter(
            "vision_requests_total", labels={"provider": "mock_provider"}
        )
        assert total_requests == 10

        # Verify SLO tracking
        slo_status = hub.slo_tracker.get_slo_status(slo.slo_id)
        assert slo_status is not None

        # Get dashboard data
        dashboard = hub.get_dashboard_data()
        assert dashboard["metrics"]["count"] > 0

    def test_multi_component_health_aggregation(self):
        """Test health status aggregation across components."""
        hub = create_observability_hub()

        # Register multiple health checks
        hub.health.register_check(
            "database", lambda: HealthCheck(name="db", status=HealthStatus.HEALTHY)
        )
        hub.health.register_check(
            "cache", lambda: HealthCheck(name="cache", status=HealthStatus.HEALTHY)
        )
        hub.health.register_check(
            "api", lambda: HealthCheck(name="api", status=HealthStatus.DEGRADED)
        )

        hub.health.run_all_checks()
        overall = hub.health.get_overall_status()

        assert overall == HealthStatus.DEGRADED

    def test_alert_correlation(self):
        """Test alert correlation and management."""
        hub = create_observability_hub()

        # Register alert rules
        hub.alerts.register_rule(
            "high_error_rate",
            lambda ctx: Alert(
                alert_id="err-1",
                name="HighErrorRate",
                severity=AlertSeverity.ERROR,
            )
            if ctx.get("error_rate", 0) > 0.05
            else None,
        )

        hub.alerts.register_rule(
            "high_latency",
            lambda ctx: Alert(
                alert_id="lat-1",
                name="HighLatency",
                severity=AlertSeverity.WARNING,
            )
            if ctx.get("latency_ms", 0) > 500
            else None,
        )

        # Evaluate with context triggering both
        context = {"error_rate": 0.1, "latency_ms": 1000}
        fired = hub.alerts.evaluate_rules(context)

        assert len(fired) == 2
        alert_names = [a.name for a in fired]
        assert "HighErrorRate" in alert_names
        assert "HighLatency" in alert_names

    def test_trace_context_propagation(self):
        """Test trace context propagation across services."""
        hub = create_observability_hub()

        # Start parent span
        parent_span = hub.traces.start_span(
            operation_name="api_request",
            service_name="api-gateway",
        )

        # Inject context into headers
        headers = hub.traces.inject_context(parent_span)

        # Extract context in downstream service
        extracted = hub.traces.extract_context(headers)

        # Start child span with extracted context
        child_span = hub.traces.start_span(
            operation_name="db_query",
            service_name="database",
            parent_context=extracted,
        )

        # Verify trace relationship
        assert child_span.context.trace_id == parent_span.context.trace_id
        assert child_span.context.parent_span_id == parent_span.context.span_id

        # Finish spans
        hub.traces.finish_span(child_span, TraceStatus.OK)
        hub.traces.finish_span(parent_span, TraceStatus.OK)

        # Verify trace has both spans
        trace_spans = hub.traces.get_trace(parent_span.context.trace_id)
        assert len(trace_spans) == 2
