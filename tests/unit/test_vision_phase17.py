"""Tests for Phase 17: Advanced Observability & Monitoring.

This module tests:
- Metrics Dashboard (metrics_dashboard.py)
- Alert Manager (alert_manager.py)
- Log Aggregator (log_aggregator.py)
- APM Integration (apm_integration.py)
- SLA Monitor (sla_monitor.py)
"""

from datetime import datetime, timedelta

import pytest

# =============================================================================
# Metrics Dashboard Tests
# =============================================================================


class TestMetricsDashboard:
    """Tests for metrics dashboard functionality."""

    def test_metric_type_enum(self) -> None:
        """Test MetricType enum values."""
        from src.core.vision.metrics_dashboard import MetricType

        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"
        assert MetricType.TIMER.value == "timer"

    def test_aggregation_type_enum(self) -> None:
        """Test AggregationType enum values."""
        from src.core.vision.metrics_dashboard import AggregationType

        assert AggregationType.SUM.value == "sum"
        assert AggregationType.AVG.value == "avg"
        assert AggregationType.MIN.value == "min"
        assert AggregationType.MAX.value == "max"
        assert AggregationType.COUNT.value == "count"

    def test_widget_type_enum(self) -> None:
        """Test WidgetType enum values."""
        from src.core.vision.metrics_dashboard import WidgetType

        assert WidgetType.LINE_CHART.value == "line_chart"
        assert WidgetType.BAR_CHART.value == "bar_chart"
        assert WidgetType.GAUGE.value == "gauge"
        assert WidgetType.TABLE.value == "table"
        assert WidgetType.HEATMAP.value == "heatmap"

    def test_time_range_enum(self) -> None:
        """Test TimeRange enum values."""
        from src.core.vision.metrics_dashboard import TimeRange

        assert TimeRange.LAST_1_HOUR.value == "1h"
        assert TimeRange.LAST_24_HOURS.value == "24h"
        assert TimeRange.LAST_7_DAYS.value == "7d"

    def test_refresh_interval_enum(self) -> None:
        """Test RefreshInterval enum values."""
        from src.core.vision.metrics_dashboard import RefreshInterval

        assert RefreshInterval.REAL_TIME.value == "1s"
        assert RefreshInterval.NORMAL.value == "30s"

    def test_metric_definition_creation(self) -> None:
        """Test MetricDefinition dataclass."""
        from src.core.vision.metrics_dashboard import MetricDefinition, MetricType

        metric = MetricDefinition(
            name="test_metric",
            metric_type=MetricType.COUNTER,
            description="Test metric",
            unit="requests",
            labels=["service", "endpoint"],
        )

        assert metric.name == "test_metric"
        assert metric.metric_type == MetricType.COUNTER
        assert metric.description == "Test metric"
        assert metric.unit == "requests"
        assert "service" in metric.labels

    def test_metric_value_creation(self) -> None:
        """Test MetricValue dataclass."""
        from src.core.vision.metrics_dashboard import MetricValue

        now = datetime.now()
        value = MetricValue(
            value=100.5,
            timestamp=now,
            labels={"service": "api"},
        )

        assert value.timestamp == now
        assert value.value == 100.5
        assert value.labels["service"] == "api"

    def test_metric_series_creation(self) -> None:
        """Test MetricSeries dataclass."""
        from src.core.vision.metrics_dashboard import MetricSeries

        series = MetricSeries(
            metric_name="test",
            labels={"env": "prod"},
        )

        assert series.metric_name == "test"
        assert series.labels["env"] == "prod"

    def test_metric_series_add_value(self) -> None:
        """Test adding values to MetricSeries."""
        from src.core.vision.metrics_dashboard import MetricSeries

        series = MetricSeries(
            metric_name="test",
            labels={"env": "prod"},
        )
        series.add_value(100.0)
        series.add_value(200.0)

        assert len(series.values) == 2

    def test_create_metric_collector(self) -> None:
        """Test metric collector factory function."""
        from src.core.vision.metrics_dashboard import create_metric_collector

        collector = create_metric_collector()
        assert collector is not None

    def test_metric_collector_with_definition(self) -> None:
        """Test registering metrics with collector using MetricDefinition."""
        from src.core.vision.metrics_dashboard import (
            MetricDefinition,
            MetricType,
            create_metric_collector,
        )

        collector = create_metric_collector()
        definition = MetricDefinition(
            name="requests_total",
            metric_type=MetricType.COUNTER,
            description="Total requests",
        )
        result = collector.register_metric(definition)
        assert result is True

    def test_create_dashboard_manager(self) -> None:
        """Test dashboard manager factory function."""
        from src.core.vision.metrics_dashboard import (
            create_dashboard_manager,
            create_metric_collector,
        )

        collector = create_metric_collector()
        manager = create_dashboard_manager(collector)
        assert manager is not None

    def test_create_metric_streamer(self) -> None:
        """Test metric streamer factory function."""
        from src.core.vision.metrics_dashboard import (
            create_metric_collector,
            create_metric_streamer,
        )

        collector = create_metric_collector()
        streamer = create_metric_streamer(collector)
        assert streamer is not None

    def test_create_custom_metric_builder(self) -> None:
        """Test custom metric builder factory function."""
        from src.core.vision.metrics_dashboard import create_custom_metric_builder

        builder = create_custom_metric_builder()
        assert builder is not None


# =============================================================================
# Alert Manager Tests
# =============================================================================


class TestAlertManager:
    """Tests for alert manager functionality."""

    def test_alert_severity_enum(self) -> None:
        """Test AlertSeverity enum values."""
        from src.core.vision.alert_manager import AlertSeverity

        assert AlertSeverity.CRITICAL.value == "critical"
        assert AlertSeverity.HIGH.value == "high"
        assert AlertSeverity.MEDIUM.value == "medium"
        assert AlertSeverity.LOW.value == "low"
        assert AlertSeverity.INFO.value == "info"

    def test_alert_state_enum(self) -> None:
        """Test AlertState enum values."""
        from src.core.vision.alert_manager import AlertState

        assert AlertState.PENDING.value == "pending"
        assert AlertState.FIRING.value == "firing"
        assert AlertState.RESOLVED.value == "resolved"
        assert AlertState.SILENCED.value == "silenced"
        assert AlertState.INHIBITED.value == "inhibited"

    def test_notification_channel_enum(self) -> None:
        """Test NotificationChannel enum values."""
        from src.core.vision.alert_manager import NotificationChannel

        assert NotificationChannel.EMAIL.value == "email"
        assert NotificationChannel.SLACK.value == "slack"
        assert NotificationChannel.WEBHOOK.value == "webhook"
        assert NotificationChannel.PAGERDUTY.value == "pagerduty"

    def test_comparison_operator_enum(self) -> None:
        """Test ComparisonOperator enum values."""
        from src.core.vision.alert_manager import ComparisonOperator

        assert ComparisonOperator.GREATER_THAN.value == "gt"
        assert ComparisonOperator.LESS_THAN.value == "lt"
        assert ComparisonOperator.GREATER_THAN_OR_EQUAL.value == "gte"

    def test_escalation_action_enum(self) -> None:
        """Test EscalationAction enum values."""
        from src.core.vision.alert_manager import EscalationAction

        assert EscalationAction.NOTIFY.value == "notify"
        assert EscalationAction.ESCALATE.value == "escalate"
        assert EscalationAction.AUTO_RESOLVE.value == "auto_resolve"

    def test_alert_condition_creation(self) -> None:
        """Test AlertCondition dataclass."""
        from src.core.vision.alert_manager import AlertCondition, ComparisonOperator

        condition = AlertCondition(
            metric="cpu_usage",
            operator=ComparisonOperator.GREATER_THAN,
            threshold=80.0,
        )

        assert condition.metric == "cpu_usage"
        assert condition.threshold == 80.0

    def test_create_alert_manager(self) -> None:
        """Test alert manager factory function."""
        from src.core.vision.alert_manager import create_alert_manager

        manager = create_alert_manager()
        assert manager is not None


# =============================================================================
# Log Aggregator Tests
# =============================================================================


class TestLogAggregator:
    """Tests for log aggregator functionality."""

    def test_log_level_enum(self) -> None:
        """Test LogLevel enum values."""
        from src.core.vision.log_aggregator import LogLevel

        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARN.value == "warn"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.FATAL.value == "fatal"

    def test_log_source_enum(self) -> None:
        """Test LogSource enum values."""
        from src.core.vision.log_aggregator import LogSource

        assert LogSource.APPLICATION.value == "application"
        assert LogSource.SYSTEM.value == "system"
        assert LogSource.CONTAINER.value == "container"
        assert LogSource.NETWORK.value == "network"

    def test_parser_type_enum(self) -> None:
        """Test ParserType enum values."""
        from src.core.vision.log_aggregator import ParserType

        assert ParserType.JSON.value == "json"
        assert ParserType.REGEX.value == "regex"
        assert ParserType.SYSLOG.value == "syslog"
        assert ParserType.GROK.value == "grok"

    def test_aggregation_mode_enum(self) -> None:
        """Test AggregationMode enum values."""
        from src.core.vision.log_aggregator import AggregationMode

        assert AggregationMode.COUNT.value == "count"
        assert AggregationMode.RATE.value == "rate"

    def test_log_entry_creation(self) -> None:
        """Test LogEntry dataclass."""
        from src.core.vision.log_aggregator import LogEntry, LogLevel, LogSource

        entry = LogEntry(
            log_id="log-123",
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test log message",
            source=LogSource.APPLICATION,
            service="api-server",
            trace_id="trace-123",
            fields={"user_id": "user-1"},
        )

        assert entry.level == LogLevel.INFO
        assert entry.message == "Test log message"
        assert entry.trace_id == "trace-123"

    def test_create_log_aggregator(self) -> None:
        """Test log aggregator factory function."""
        from src.core.vision.log_aggregator import create_log_aggregator

        aggregator = create_log_aggregator()
        assert aggregator is not None


# =============================================================================
# APM Integration Tests
# =============================================================================


class TestAPMIntegration:
    """Tests for APM integration functionality."""

    def test_transaction_type_enum(self) -> None:
        """Test TransactionType enum values."""
        from src.core.vision.apm_integration import TransactionType

        assert TransactionType.REQUEST.value == "request"
        assert TransactionType.BACKGROUND.value == "background"
        assert TransactionType.MESSAGE.value == "message"
        assert TransactionType.SCHEDULED.value == "scheduled"

    def test_span_kind_enum(self) -> None:
        """Test SpanKind enum values."""
        from src.core.vision.apm_integration import SpanKind

        assert SpanKind.INTERNAL.value == "internal"
        assert SpanKind.SERVER.value == "server"
        assert SpanKind.CLIENT.value == "client"
        assert SpanKind.PRODUCER.value == "producer"
        assert SpanKind.CONSUMER.value == "consumer"

    def test_span_status_enum(self) -> None:
        """Test SpanStatus enum values."""
        from src.core.vision.apm_integration import SpanStatus

        assert SpanStatus.OK.value == "ok"
        assert SpanStatus.ERROR.value == "error"
        assert SpanStatus.TIMEOUT.value == "timeout"

    def test_apm_provider_enum(self) -> None:
        """Test APMProvider enum values."""
        from src.core.vision.apm_integration import APMProvider

        assert APMProvider.DATADOG.value == "datadog"
        assert APMProvider.NEW_RELIC.value == "newrelic"
        assert APMProvider.JAEGER.value == "jaeger"

    def test_profile_type_enum(self) -> None:
        """Test ProfileType enum values."""
        from src.core.vision.apm_integration import ProfileType

        assert ProfileType.CPU.value == "cpu"
        assert ProfileType.MEMORY.value == "memory"
        assert ProfileType.IO.value == "io"

    def test_transaction_creation(self) -> None:
        """Test Transaction dataclass."""
        from src.core.vision.apm_integration import Transaction, TransactionType

        tx = Transaction(
            transaction_id="tx-123",
            name="GET /api/users",
            transaction_type=TransactionType.REQUEST,
            start_time=datetime.now(),
        )

        assert tx.name == "GET /api/users"
        assert tx.transaction_type == TransactionType.REQUEST

    def test_span_creation(self) -> None:
        """Test Span dataclass."""
        from src.core.vision.apm_integration import Span, SpanKind

        span = Span(
            span_id="span-123",
            trace_id="trace-123",
            name="db.query",
            kind=SpanKind.CLIENT,
            start_time=datetime.now(),
        )

        assert span.name == "db.query"
        assert span.kind == SpanKind.CLIENT

    def test_create_apm_manager(self) -> None:
        """Test APM manager factory function."""
        from src.core.vision.apm_integration import create_apm_manager

        manager = create_apm_manager()
        assert manager is not None

    def test_create_apm_config(self) -> None:
        """Test APM config creation."""
        from src.core.vision.apm_integration import create_apm_config

        config = create_apm_config(
            service_name="my-service",
            environment="production",
            sample_rate=0.5,
        )

        assert config.service_name == "my-service"
        assert config.sample_rate == 0.5


# =============================================================================
# SLA Monitor Tests
# =============================================================================


class TestSLAMonitor:
    """Tests for SLA monitor functionality."""

    def test_sla_type_enum(self) -> None:
        """Test SLAType enum values."""
        from src.core.vision.sla_monitor import SLAType

        assert SLAType.AVAILABILITY.value == "availability"
        assert SLAType.LATENCY.value == "latency"
        assert SLAType.ERROR_RATE.value == "error_rate"
        assert SLAType.THROUGHPUT.value == "throughput"

    def test_sla_status_enum(self) -> None:
        """Test SLAStatus enum values."""
        from src.core.vision.sla_monitor import SLAStatus

        assert SLAStatus.COMPLIANT.value == "compliant"
        assert SLAStatus.AT_RISK.value == "at_risk"
        assert SLAStatus.BREACHED.value == "breached"

    def test_uptime_status_enum(self) -> None:
        """Test UptimeStatus enum values."""
        from src.core.vision.sla_monitor import UptimeStatus

        assert UptimeStatus.UP.value == "up"
        assert UptimeStatus.DOWN.value == "down"
        assert UptimeStatus.DEGRADED.value == "degraded"

    def test_report_period_enum(self) -> None:
        """Test ReportPeriod enum values."""
        from src.core.vision.sla_monitor import ReportPeriod

        assert ReportPeriod.DAILY.value == "daily"
        assert ReportPeriod.WEEKLY.value == "weekly"
        assert ReportPeriod.MONTHLY.value == "monthly"
        assert ReportPeriod.QUARTERLY.value == "quarterly"

    def test_check_type_enum(self) -> None:
        """Test CheckType enum values."""
        from src.core.vision.sla_monitor import CheckType

        assert CheckType.HTTP.value == "http"
        assert CheckType.TCP.value == "tcp"
        assert CheckType.PING.value == "ping"
        assert CheckType.CUSTOM.value == "custom"

    def test_sla_definition_creation(self) -> None:
        """Test SLADefinition dataclass."""
        from src.core.vision.sla_monitor import SLADefinition, SLAType

        definition = SLADefinition(
            sla_id="sla-1",
            name="API Availability",
            sla_type=SLAType.AVAILABILITY,
            target_value=99.9,
            warning_threshold=99.5,
            description="99.9% uptime",
            service="api",
        )

        assert definition.name == "API Availability"
        assert definition.target_value == 99.9

    def test_uptime_check_creation(self) -> None:
        """Test UptimeCheck dataclass."""
        from src.core.vision.sla_monitor import CheckType, UptimeCheck

        check = UptimeCheck(
            check_id="check-1",
            name="API Health",
            check_type=CheckType.HTTP,
            target="https://api.example.com/health",
        )

        assert check.name == "API Health"
        assert check.check_type == CheckType.HTTP

    def test_create_sla_monitor(self) -> None:
        """Test SLA monitor factory function."""
        from src.core.vision.sla_monitor import create_sla_monitor

        monitor = create_sla_monitor()
        assert monitor is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhase17Integration:
    """Integration tests for Phase 17 modules."""

    def test_all_phase17_exports_available(self) -> None:
        """Test that all Phase 17 exports are available."""
        from src.core.vision import (  # Metrics Dashboard; Alert Manager; Log Aggregator; APM Integration; SLA Monitor
            AggregationType,
            AggregatorLogLevel,
            AggregatorLogStore,
            AlertEvaluator,
            AlertManagerVisionProvider,
            AlertState,
            APMManager,
            APMSpanKind,
            APMSpanStatus,
            APMVisionProvider,
            CheckType,
            ComparisonOperator,
            CustomMetricBuilder,
            DashboardMetricType,
            DependencyTracker,
            ErrorTracker,
            EscalationAction,
            LogAggregator,
            LogAggregatorVisionProvider,
            LogCorrelator,
            LogSource,
            MetricCollector,
            MetricsDashboardManager,
            MetricsDashboardVisionProvider,
            MetricStreamer,
            NotificationChannel,
            ParserType,
            PatternDetector,
            PerformanceAnalyzer,
            Phase17AlertManager,
            Phase17AlertSeverity,
            RefreshInterval,
            ReportPeriod,
            SLAMonitor,
            SLAMonitorVisionProvider,
            SLAReporter,
            SLAStatus,
            SLATracker,
            SLAType,
            SpanTracker,
            TimeRange,
            TransactionTracker,
            TransactionType,
            UptimeStatus,
            UptimeTracker,
            WidgetType,
        )

        # All imports should succeed without errors
        assert MetricCollector is not None
        assert Phase17AlertManager is not None
        assert LogAggregator is not None
        assert APMManager is not None
        assert SLAMonitor is not None

    def test_factory_functions_available(self) -> None:
        """Test that all factory functions are available."""
        from src.core.vision import (
            create_apm_config,
            create_apm_manager,
            create_custom_metric_builder,
            create_log_aggregator,
            create_metric_collector,
            create_metric_streamer,
            create_metrics_dashboard_manager,
            create_phase17_alert_manager,
            create_sla_monitor,
        )

        assert create_metric_collector is not None
        assert create_metrics_dashboard_manager is not None
        assert create_metric_streamer is not None
        assert create_custom_metric_builder is not None
        assert create_phase17_alert_manager is not None
        assert create_log_aggregator is not None
        assert create_apm_manager is not None
        assert create_apm_config is not None
        assert create_sla_monitor is not None

    def test_dataclasses_available(self) -> None:
        """Test that all dataclasses are available."""
        from src.core.vision import (
            AggregatorLogEntry,
            AlertCondition,
            AlertInstance,
            APMConfig,
            APMSpan,
            CorrelationResult,
            LogFilter,
            MetricDataPoint,
            MetricDefinition,
            Phase17AlertRule,
            SLADefinition,
            SLAMeasurement,
            TimeSeriesData,
            Transaction,
            UptimeCheck,
        )

        assert MetricDefinition is not None
        assert MetricDataPoint is not None
        assert TimeSeriesData is not None
        assert Phase17AlertRule is not None
        assert AlertCondition is not None
        assert AlertInstance is not None
        assert AggregatorLogEntry is not None
        assert LogFilter is not None
        assert CorrelationResult is not None
        assert Transaction is not None
        assert APMSpan is not None
        assert APMConfig is not None
        assert SLADefinition is not None
        assert SLAMeasurement is not None
        assert UptimeCheck is not None
