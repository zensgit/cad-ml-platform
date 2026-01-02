"""Tests for Vision Provider Phase 9 features.

This module tests:
- Observability (metrics, logging, alerting, health checks, SLO)
- Deployment strategies (blue-green, canary, rolling)
- Multi-tenancy (tenant isolation, quotas, usage tracking)
- Compliance (audit logging, PII detection, retention policies)
- Chaos engineering (fault injection, experiments)
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.vision.base import VisionDescription, VisionProvider

# Phase 9 imports - Chaos Engineering
from src.core.vision.chaos_engineering import (
    ChaosExperiment,
    ChaosManager,
    ChaosVisionProvider,
    ErrorInjector,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    FaultConfig,
    FaultInjector,
    FaultType,
    InjectionStrategy,
    LatencyInjector,
    TargetType,
    create_chaos_provider,
    create_experiment,
)

# Phase 9 imports - Compliance
from src.core.vision.compliance import AuditEvent, AuditEventType, AuditLogger
from src.core.vision.compliance import AuditSeverity as ComplianceAuditSeverity
from src.core.vision.compliance import (
    ComplianceManager,
    ComplianceRequirement,
    ComplianceStandard,
    ComplianceVisionProvider,
    DataClassification,
    DataRetentionConfig,
    DataRetentionManager,
    PIIDetectionResult,
    PIIDetector,
    RetentionPolicy,
    create_compliant_provider,
)

# Phase 9 imports - Deployment
from src.core.vision.deployment import (
    BlueGreenDeployment,
    CanaryRelease,
    DeploymentConfig,
    DeploymentManager,
    DeploymentMetrics,
    DeploymentPhase,
    DeploymentState,
    DeploymentStrategy,
    DeploymentVersion,
    DeploymentVisionProvider,
    EnvironmentType,
    RollingUpdate,
    TrafficRouter,
    TrafficSplitMethod,
    create_blue_green_provider,
    create_canary_provider,
)

# Phase 9 imports - Multi-tenancy
from src.core.vision.multi_tenancy import (
    InMemoryTenantStore,
    IsolationLevel,
    MultiTenantVisionProvider,
    QuotaManager,
    QuotaType,
    ResourceQuota,
    Tenant,
    TenantConfig,
    TenantContext,
    TenantManager,
    TenantStatus,
    TenantStore,
    TenantTier,
    TenantUsage,
    UsageTracker,
    create_multi_tenant_provider,
)

# Phase 9 imports - Observability
from src.core.vision.observability import (
    SLI,
    SLO,
    Alert,
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    HealthCheck,
    HealthChecker,
    HealthStatus,
    HistogramBucket,
    HistogramValue,
    LogEntry,
    LogLevel,
    MetricsCollector,
    MetricType,
    MetricValue,
    ObservabilityContext,
    ObservableVisionProvider,
    SLOMonitor,
    StructuredLogger,
    create_observable_provider,
)


class MockVisionProvider(VisionProvider):
    """Mock vision provider for testing."""

    def __init__(self, name: str = "mock", fail: bool = False) -> None:
        self._name = name
        self._fail = fail
        self._call_count = 0

    @property
    def provider_name(self) -> str:
        return self._name

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        self._call_count += 1
        if self._fail:
            raise RuntimeError("Mock provider failure")
        return VisionDescription(
            summary=f"Mock analysis by {self._name}",
            details=["Detailed mock description"],
            confidence=0.85,
        )


# =============================================================================
# Observability Tests
# =============================================================================


class TestObservability:
    """Tests for observability module."""

    def test_metric_type_enum(self) -> None:
        """Test metric type enumeration."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"

    def test_log_level_enum(self) -> None:
        """Test log level enumeration."""
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.CRITICAL.value == "critical"

    def test_alert_severity_enum(self) -> None:
        """Test alert severity enumeration."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_metric_value_creation(self) -> None:
        """Test metric value creation."""
        metric = MetricValue(
            name="test_counter",
            value=42.0,
            metric_type=MetricType.COUNTER,
            labels={"service": "vision"},
        )
        assert metric.name == "test_counter"
        assert metric.value == 42.0
        assert metric.metric_type == MetricType.COUNTER
        assert metric.labels["service"] == "vision"

    def test_metric_value_to_dict(self) -> None:
        """Test metric value to dictionary conversion."""
        metric = MetricValue(
            name="test_gauge",
            value=0.85,
            metric_type=MetricType.GAUGE,
        )
        data = metric.to_dict()
        assert data["name"] == "test_gauge"
        assert data["value"] == 0.85
        assert data["type"] == "gauge"

    def test_histogram_value_observe(self) -> None:
        """Test histogram value observation."""
        histogram = HistogramValue(
            name="latency",
            buckets=[
                HistogramBucket(le=0.1),
                HistogramBucket(le=0.5),
                HistogramBucket(le=1.0),
            ],
        )
        histogram.observe(0.3)
        histogram.observe(0.7)

        assert histogram.count == 2
        assert histogram.sum_value == 1.0

    def test_log_entry_creation(self) -> None:
        """Test log entry creation."""
        entry = LogEntry(
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test_logger",
            context={"key": "value"},
        )
        assert entry.level == LogLevel.INFO
        assert entry.message == "Test message"
        assert entry.context["key"] == "value"

    def test_log_entry_to_json(self) -> None:
        """Test log entry to JSON conversion."""
        entry = LogEntry(
            level=LogLevel.ERROR,
            message="Error occurred",
        )
        json_str = entry.to_json()
        assert "error" in json_str
        assert "Error occurred" in json_str

    def test_alert_creation(self) -> None:
        """Test alert creation."""
        alert = Alert(
            name="high_latency",
            severity=AlertSeverity.WARNING,
            message="Latency exceeded threshold",
            value=500.0,
        )
        assert alert.name == "high_latency"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.status == AlertStatus.PENDING

    def test_sli_compliance_ratio(self) -> None:
        """Test SLI compliance ratio calculation."""
        sli = SLI(
            name="availability",
            description="Service availability",
            current_value=99.5,
            target_value=99.9,
            unit="percent",
        )
        ratio = sli.compliance_ratio()
        assert 0.99 < ratio < 1.0

    def test_slo_is_met(self) -> None:
        """Test SLO met check."""
        sli = SLI(
            name="latency",
            description="Response latency",
            current_value=95.0,
            target_value=100.0,
        )
        slo = SLO(
            name="latency_slo",
            sli=sli,
            target=95.0,
        )
        assert slo.is_met()

    def test_metrics_collector_counter(self) -> None:
        """Test metrics collector counter."""
        collector = MetricsCollector()
        collector.counter("requests", 1.0, {"service": "vision"})
        collector.counter("requests", 1.0, {"service": "vision"})

        metrics = collector.get_metric("requests", {"service": "vision"})
        assert len(metrics) == 2

    def test_metrics_collector_gauge(self) -> None:
        """Test metrics collector gauge."""
        collector = MetricsCollector()
        collector.gauge("memory_usage", 512.0, {"host": "server1"})

        metrics = collector.get_metric("memory_usage", {"host": "server1"})
        assert len(metrics) == 1
        assert metrics[0].value == 512.0

    def test_metrics_collector_histogram(self) -> None:
        """Test metrics collector histogram."""
        collector = MetricsCollector()
        collector.histogram("latency", 0.5, labels={"service": "vision"})
        collector.histogram("latency", 1.5, labels={"service": "vision"})

        histogram = collector.get_histogram("latency", {"service": "vision"})
        assert histogram is not None
        assert histogram.count == 2

    def test_structured_logger_info(self) -> None:
        """Test structured logger info."""
        logger = StructuredLogger(name="test")
        logger.info("Test message", key="value")

        entries = logger.get_entries()
        assert len(entries) == 1
        assert entries[0].level == LogLevel.INFO

    def test_structured_logger_with_context(self) -> None:
        """Test structured logger with context."""
        logger = StructuredLogger(name="test")
        contextual = logger.with_context(request_id="123")
        contextual.info("Test message")

        entries = contextual.get_entries()
        assert entries[0].context["request_id"] == "123"

    def test_structured_logger_with_trace(self) -> None:
        """Test structured logger with trace."""
        logger = StructuredLogger(name="test")
        traced = logger.with_trace("trace-123", "span-456")
        traced.info("Traced message")

        entries = traced.get_entries()
        assert entries[0].trace_id == "trace-123"
        assert entries[0].span_id == "span-456"

    def test_alert_manager_fire_alert(self) -> None:
        """Test alert manager fire alert."""
        manager = AlertManager()
        alert = manager.fire_alert(
            name="test_alert",
            severity=AlertSeverity.ERROR,
            message="Test alert",
        )
        assert alert.status == AlertStatus.FIRING

    def test_alert_manager_resolve_alert(self) -> None:
        """Test alert manager resolve alert."""
        manager = AlertManager()
        manager.fire_alert(
            name="test_alert",
            severity=AlertSeverity.WARNING,
            message="Test alert",
        )
        resolved = manager.resolve_alert("test_alert")
        assert resolved is not None
        assert resolved.status == AlertStatus.RESOLVED

    def test_health_checker_register(self) -> None:
        """Test health checker registration."""
        checker = HealthChecker()

        def health_check() -> HealthCheck:
            return HealthCheck(
                name="test",
                status=HealthStatus.HEALTHY,
                message="OK",
            )

        checker.register("test", health_check)
        result = checker.run_check("test")

        assert result is not None
        assert result.status == HealthStatus.HEALTHY

    def test_health_checker_overall_status(self) -> None:
        """Test health checker overall status."""
        checker = HealthChecker()

        checker.register(
            "service1", lambda: HealthCheck(name="service1", status=HealthStatus.HEALTHY)
        )
        checker.register(
            "service2", lambda: HealthCheck(name="service2", status=HealthStatus.HEALTHY)
        )

        assert checker.get_overall_status() == HealthStatus.HEALTHY

    def test_slo_monitor_register(self) -> None:
        """Test SLO monitor registration."""
        metrics = MetricsCollector()
        monitor = SLOMonitor(metrics)

        sli = SLI(name="availability", description="Uptime", current_value=99.9, target_value=99.9)
        slo = SLO(name="availability_slo", sli=sli, target=99.9)

        monitor.register_slo(slo)
        status = monitor.get_slo_status("availability_slo")

        assert status is not None
        assert status["is_met"]

    @pytest.mark.asyncio
    async def test_observable_vision_provider(self) -> None:
        """Test observable vision provider."""
        mock_provider = MockVisionProvider()
        observable = create_observable_provider(mock_provider)

        result = await observable.analyze_image(b"test_image")

        assert result.summary is not None
        summary = observable.get_metrics_summary()
        assert summary["total_requests"] == 1


# =============================================================================
# Deployment Tests
# =============================================================================


class TestDeployment:
    """Tests for deployment module."""

    def test_deployment_strategy_enum(self) -> None:
        """Test deployment strategy enumeration."""
        assert DeploymentStrategy.BLUE_GREEN.value == "blue_green"
        assert DeploymentStrategy.CANARY.value == "canary"
        assert DeploymentStrategy.ROLLING.value == "rolling"

    def test_deployment_phase_enum(self) -> None:
        """Test deployment phase enumeration."""
        assert DeploymentPhase.PENDING.value == "pending"
        assert DeploymentPhase.IN_PROGRESS.value == "in_progress"
        assert DeploymentPhase.COMPLETED.value == "completed"

    def test_environment_type_enum(self) -> None:
        """Test environment type enumeration."""
        assert EnvironmentType.BLUE.value == "blue"
        assert EnvironmentType.GREEN.value == "green"
        assert EnvironmentType.CANARY.value == "canary"

    def test_deployment_version_creation(self) -> None:
        """Test deployment version creation."""
        provider = MockVisionProvider()
        version = DeploymentVersion(
            version="1.0.0",
            provider=provider,
            metadata={"tag": "stable"},
        )
        assert version.version == "1.0.0"
        assert version.provider == provider

    def test_deployment_metrics_record(self) -> None:
        """Test deployment metrics recording."""
        metrics = DeploymentMetrics()
        metrics.record_request("v1", True, 100.0)
        metrics.record_request("v1", False, 200.0)

        assert metrics.total_requests == 2
        assert metrics.successful_requests == 1
        assert metrics.failure_rate == 0.5

    def test_deployment_config_defaults(self) -> None:
        """Test deployment config defaults."""
        config = DeploymentConfig(strategy=DeploymentStrategy.CANARY)
        assert config.canary_percentage == 10.0
        assert config.auto_rollback is True

    def test_traffic_router_add_version(self) -> None:
        """Test traffic router add version."""
        router = TrafficRouter()
        provider = MockVisionProvider()
        version = DeploymentVersion(version="1.0.0", provider=provider)

        router.add_version(version, 100.0)
        weights = router.get_weights()

        assert "1.0.0" in weights
        assert weights["1.0.0"] == 100.0

    def test_traffic_router_route(self) -> None:
        """Test traffic router routing."""
        router = TrafficRouter()
        provider = MockVisionProvider()
        version = DeploymentVersion(version="1.0.0", provider=provider)

        router.add_version(version, 100.0)
        selected = router.route()

        assert selected is not None
        assert selected.version == "1.0.0"

    def test_deployment_manager_register(self) -> None:
        """Test deployment manager register version."""
        manager = DeploymentManager()
        provider = MockVisionProvider()

        version = manager.register_version("1.0.0", provider)

        assert version.version == "1.0.0"

    def test_deployment_manager_start_canary(self) -> None:
        """Test deployment manager start canary deployment."""
        config = DeploymentConfig(
            strategy=DeploymentStrategy.CANARY,
            canary_percentage=10.0,
        )
        manager = DeploymentManager(config)

        manager.register_version("1.0.0", MockVisionProvider("v1"))
        manager.register_version("2.0.0", MockVisionProvider("v2"))

        result = manager.start_deployment("2.0.0")
        state = manager.get_state()

        assert result is True
        assert state.phase == DeploymentPhase.IN_PROGRESS
        assert state.traffic_percentage == 10.0

    def test_deployment_manager_advance(self) -> None:
        """Test deployment manager advance deployment."""
        config = DeploymentConfig(
            strategy=DeploymentStrategy.CANARY,
            rollout_steps=[10, 50, 100],
        )
        manager = DeploymentManager(config)

        manager.register_version("1.0.0", MockVisionProvider("v1"))
        manager.register_version("2.0.0", MockVisionProvider("v2"))
        manager.start_deployment("2.0.0")

        manager.advance_deployment()
        state = manager.get_state()

        assert state.traffic_percentage == 50.0

    def test_deployment_manager_complete(self) -> None:
        """Test deployment manager complete deployment."""
        manager = DeploymentManager()
        manager.register_version("1.0.0", MockVisionProvider("v1"))
        manager.register_version("2.0.0", MockVisionProvider("v2"))

        manager.start_deployment("2.0.0", DeploymentStrategy.BLUE_GREEN)
        manager.complete_deployment()

        state = manager.get_state()
        assert state.phase == DeploymentPhase.COMPLETED

    def test_deployment_manager_rollback(self) -> None:
        """Test deployment manager rollback."""
        manager = DeploymentManager()
        manager.register_version("1.0.0", MockVisionProvider("v1"))
        manager.register_version("2.0.0", MockVisionProvider("v2"))

        manager.start_deployment("2.0.0")
        manager.rollback()

        state = manager.get_state()
        assert state.phase == DeploymentPhase.ROLLING_BACK

    def test_blue_green_deployment(self) -> None:
        """Test blue-green deployment manager."""
        blue_green = BlueGreenDeployment()

        blue = DeploymentVersion(version="blue", provider=MockVisionProvider("blue"))
        green = DeploymentVersion(version="green", provider=MockVisionProvider("green"))

        blue_green.set_blue(blue)
        blue_green.set_green(green)

        assert blue_green.get_active_environment() == EnvironmentType.BLUE
        assert blue_green.get_active() == blue

        blue_green.switch()
        assert blue_green.get_active_environment() == EnvironmentType.GREEN
        assert blue_green.get_active() == green

    def test_canary_release(self) -> None:
        """Test canary release manager."""
        canary = CanaryRelease(
            initial_percentage=5.0,
            step_percentage=10.0,
        )

        stable = DeploymentVersion(version="stable", provider=MockVisionProvider("stable"))
        new_version = DeploymentVersion(version="canary", provider=MockVisionProvider("canary"))

        canary.set_stable(stable)
        canary.start_canary(new_version)

        assert canary.get_canary_percentage() == 5.0

        canary.increase_traffic()
        assert canary.get_canary_percentage() == 15.0

    def test_rolling_update(self) -> None:
        """Test rolling update manager."""
        rolling = RollingUpdate(batch_size=2)

        instances = [
            DeploymentVersion(version=f"v{i}", provider=MockVisionProvider(f"v{i}"))
            for i in range(5)
        ]
        rolling.set_instances(instances)

        batch = rolling.get_next_batch()
        assert len(batch) == 2

        for idx in batch:
            rolling.mark_updated(idx)

        assert rolling.progress() == 40.0

    @pytest.mark.asyncio
    async def test_deployment_vision_provider(self) -> None:
        """Test deployment vision provider."""
        manager = DeploymentManager()
        manager.register_version("1.0.0", MockVisionProvider())

        provider = DeploymentVisionProvider(manager)
        result = await provider.analyze_image(b"test")

        assert result.summary is not None


# =============================================================================
# Multi-Tenancy Tests
# =============================================================================


class TestMultiTenancy:
    """Tests for multi-tenancy module."""

    def test_tenant_status_enum(self) -> None:
        """Test tenant status enumeration."""
        assert TenantStatus.ACTIVE.value == "active"
        assert TenantStatus.SUSPENDED.value == "suspended"
        assert TenantStatus.DELETED.value == "deleted"

    def test_tenant_tier_enum(self) -> None:
        """Test tenant tier enumeration."""
        assert TenantTier.FREE.value == "free"
        assert TenantTier.BASIC.value == "basic"
        assert TenantTier.PROFESSIONAL.value == "professional"
        assert TenantTier.ENTERPRISE.value == "enterprise"

    def test_isolation_level_enum(self) -> None:
        """Test isolation level enumeration."""
        assert IsolationLevel.SHARED.value == "shared"
        assert IsolationLevel.DEDICATED.value == "dedicated"
        assert IsolationLevel.ISOLATED.value == "isolated"

    def test_quota_type_enum(self) -> None:
        """Test quota type enumeration."""
        assert QuotaType.REQUESTS_PER_MINUTE.value == "requests_per_minute"
        assert QuotaType.REQUESTS_PER_DAY.value == "requests_per_day"

    def test_resource_quota_creation(self) -> None:
        """Test resource quota creation."""
        quota = ResourceQuota(
            quota_type=QuotaType.REQUESTS_PER_MINUTE,
            limit=100.0,
        )
        assert quota.limit == 100.0
        assert quota.remaining() == 100.0

    def test_resource_quota_use(self) -> None:
        """Test resource quota use."""
        quota = ResourceQuota(
            quota_type=QuotaType.REQUESTS_PER_MINUTE,
            limit=10.0,
        )
        assert quota.use(5.0)
        assert quota.remaining() == 5.0
        assert not quota.use(10.0)  # Exceeds limit

    def test_tenant_config_creation(self) -> None:
        """Test tenant config creation."""
        config = TenantConfig(
            tenant_id="tenant1",
            tier=TenantTier.PROFESSIONAL,
            max_concurrent_requests=50,
        )
        assert config.tenant_id == "tenant1"
        assert config.tier == TenantTier.PROFESSIONAL

    def test_tenant_creation(self) -> None:
        """Test tenant creation."""
        tenant = Tenant(
            tenant_id="tenant1",
            name="Test Tenant",
        )
        assert tenant.tenant_id == "tenant1"
        assert tenant.status == TenantStatus.ACTIVE

    def test_tenant_usage_record(self) -> None:
        """Test tenant usage recording."""
        usage = TenantUsage(tenant_id="tenant1")
        usage.record_request(True, 100.0, 1.0)
        usage.record_request(False, 200.0)

        assert usage.total_requests == 2
        assert usage.success_rate == 0.5
        assert usage.data_processed_mb == 1.0

    def test_in_memory_tenant_store(self) -> None:
        """Test in-memory tenant store."""
        store = InMemoryTenantStore()
        tenant = Tenant(tenant_id="tenant1", name="Test")

        store.save(tenant)
        retrieved = store.get("tenant1")

        assert retrieved is not None
        assert retrieved.name == "Test"

    def test_quota_manager_set_and_check(self) -> None:
        """Test quota manager set and check."""
        manager = QuotaManager()

        quota = ResourceQuota(
            quota_type=QuotaType.REQUESTS_PER_MINUTE,
            limit=100.0,
        )
        manager.set_quota("tenant1", quota)

        assert manager.check_quota("tenant1", QuotaType.REQUESTS_PER_MINUTE)

    def test_quota_manager_use_quota(self) -> None:
        """Test quota manager use quota."""
        manager = QuotaManager()

        quota = ResourceQuota(
            quota_type=QuotaType.REQUESTS_PER_MINUTE,
            limit=10.0,
        )
        manager.set_quota("tenant1", quota)

        for _ in range(10):
            manager.use_quota("tenant1", QuotaType.REQUESTS_PER_MINUTE)

        assert not manager.check_quota("tenant1", QuotaType.REQUESTS_PER_MINUTE, 1.0)

    def test_usage_tracker(self) -> None:
        """Test usage tracker."""
        tracker = UsageTracker()
        tracker.record("tenant1", True, 100.0)
        tracker.record("tenant1", True, 150.0)

        usage = tracker.get_usage("tenant1")
        assert usage.total_requests == 2
        assert usage.avg_latency_ms == 125.0

    def test_tenant_context(self) -> None:
        """Test tenant context."""
        TenantContext.set_tenant("tenant1")
        assert TenantContext.get_tenant() == "tenant1"

        TenantContext.clear()
        assert TenantContext.get_tenant() is None

    def test_tenant_manager_create(self) -> None:
        """Test tenant manager create tenant."""
        manager = TenantManager()
        tenant = manager.create_tenant(
            tenant_id="tenant1",
            name="Test Tenant",
            tier=TenantTier.BASIC,
        )

        assert tenant.tenant_id == "tenant1"
        retrieved = manager.get_tenant("tenant1")
        assert retrieved is not None

    def test_tenant_manager_suspend(self) -> None:
        """Test tenant manager suspend."""
        manager = TenantManager()
        manager.create_tenant("tenant1", "Test")

        manager.suspend_tenant("tenant1")
        tenant = manager.get_tenant("tenant1")

        assert tenant is not None
        assert tenant.status == TenantStatus.SUSPENDED

    def test_tenant_manager_check_quota(self) -> None:
        """Test tenant manager check quota."""
        manager = TenantManager()
        manager.create_tenant("tenant1", "Test", TenantTier.FREE)

        # Free tier has limited quota
        assert manager.check_quota("tenant1", QuotaType.REQUESTS_PER_MINUTE)

    @pytest.mark.asyncio
    async def test_multi_tenant_vision_provider(self) -> None:
        """Test multi-tenant vision provider."""
        manager = TenantManager()
        manager.create_tenant("tenant1", "Test Tenant")
        manager.set_default_provider(MockVisionProvider())

        provider = MultiTenantVisionProvider(manager)

        result = await provider.analyze_for_tenant("tenant1", b"test")
        assert result.summary is not None


# =============================================================================
# Compliance Tests
# =============================================================================


class TestCompliance:
    """Tests for compliance module."""

    def test_audit_event_type_enum(self) -> None:
        """Test audit event type enumeration."""
        assert AuditEventType.REQUEST.value == "request"
        assert AuditEventType.RESPONSE.value == "response"
        assert AuditEventType.ERROR.value == "error"

    def test_compliance_standard_enum(self) -> None:
        """Test compliance standard enumeration."""
        assert ComplianceStandard.GDPR.value == "gdpr"
        assert ComplianceStandard.HIPAA.value == "hipaa"
        assert ComplianceStandard.SOC2.value == "soc2"

    def test_data_classification_enum(self) -> None:
        """Test data classification enumeration."""
        assert DataClassification.PUBLIC.value == "public"
        assert DataClassification.CONFIDENTIAL.value == "confidential"
        assert DataClassification.PII.value == "pii"

    def test_retention_policy_enum(self) -> None:
        """Test retention policy enumeration."""
        assert RetentionPolicy.SHORT_TERM.value == "short_term"
        assert RetentionPolicy.LONG_TERM.value == "long_term"
        assert RetentionPolicy.PERMANENT.value == "permanent"

    def test_audit_event_creation(self) -> None:
        """Test audit event creation."""
        event = AuditEvent(
            event_id="evt_001",
            event_type=AuditEventType.REQUEST,
            actor="user1",
            resource="image:123",
            action="analyze",
        )
        assert event.event_id == "evt_001"
        assert event.actor == "user1"

    def test_audit_event_to_json(self) -> None:
        """Test audit event to JSON."""
        event = AuditEvent(
            event_id="evt_001",
            event_type=AuditEventType.REQUEST,
        )
        json_str = event.to_json()
        assert "evt_001" in json_str

    def test_data_retention_config(self) -> None:
        """Test data retention config."""
        config = DataRetentionConfig(
            policy=RetentionPolicy.SHORT_TERM,
            retention_days=30,
        )
        created = datetime.now()
        expires = config.get_expiration_date(created)

        assert expires is not None
        assert expires > created

    def test_pii_detector_email(self) -> None:
        """Test PII detector email detection."""
        detector = PIIDetector()
        result = detector.detect("Contact me at test@example.com")

        assert result.detected
        assert "email" in result.pii_types

    def test_pii_detector_phone(self) -> None:
        """Test PII detector phone detection."""
        detector = PIIDetector()
        result = detector.detect("Call me at +15551234567")

        assert result.detected
        assert "phone" in result.pii_types

    def test_pii_detector_redact(self) -> None:
        """Test PII detector redact."""
        detector = PIIDetector()
        redacted = detector.redact("Email: test@example.com")

        assert "test@example.com" not in redacted
        assert "[REDACTED]" in redacted

    def test_audit_logger_log_request(self) -> None:
        """Test audit logger log request."""
        logger = AuditLogger()
        event = logger.log_request(
            resource="image:123",
            actor="user1",
        )

        assert event.event_type == AuditEventType.REQUEST

    def test_audit_logger_log_error(self) -> None:
        """Test audit logger log error."""
        logger = AuditLogger()
        event = logger.log_error(
            resource="image:123",
            error="Processing failed",
        )

        assert event.severity == ComplianceAuditSeverity.ERROR

    def test_audit_logger_get_events(self) -> None:
        """Test audit logger get events."""
        logger = AuditLogger()
        logger.log_request("resource1", "user1")
        logger.log_response("resource1", "user1")

        events = logger.get_events()
        assert len(events) == 2

    def test_compliance_manager_enable_standard(self) -> None:
        """Test compliance manager enable standard."""
        manager = ComplianceManager()
        manager.enable_standard(ComplianceStandard.GDPR)

        requirements = manager.get_requirements(ComplianceStandard.GDPR)
        assert len(requirements) > 0

    def test_compliance_manager_check_compliance(self) -> None:
        """Test compliance manager check compliance."""
        manager = ComplianceManager()
        manager.enable_standard(ComplianceStandard.GDPR)

        status = manager.check_compliance()
        assert "compliance_percentage" in status

    def test_data_retention_manager(self) -> None:
        """Test data retention manager."""
        manager = DataRetentionManager()

        config = DataRetentionConfig(
            policy=RetentionPolicy.SHORT_TERM,
            retention_days=30,
        )
        manager.set_retention("images", config)

        retrieved = manager.get_retention("images")
        assert retrieved.policy == RetentionPolicy.SHORT_TERM

    @pytest.mark.asyncio
    async def test_compliance_vision_provider(self) -> None:
        """Test compliance vision provider."""
        mock_provider = MockVisionProvider()
        compliance = ComplianceManager()

        provider = ComplianceVisionProvider(
            provider=mock_provider,
            compliance=compliance,
        )

        result = await provider.analyze_image(b"test")
        assert result.summary is not None

        # Check audit log
        events = compliance.get_audit_logger().get_events()
        assert len(events) >= 2  # Request and response


# =============================================================================
# Chaos Engineering Tests
# =============================================================================


class TestChaosEngineering:
    """Tests for chaos engineering module."""

    def test_fault_type_enum(self) -> None:
        """Test fault type enumeration."""
        assert FaultType.LATENCY.value == "latency"
        assert FaultType.ERROR.value == "error"
        assert FaultType.TIMEOUT.value == "timeout"

    def test_experiment_status_enum(self) -> None:
        """Test experiment status enumeration."""
        assert ExperimentStatus.PENDING.value == "pending"
        assert ExperimentStatus.RUNNING.value == "running"
        assert ExperimentStatus.COMPLETED.value == "completed"

    def test_target_type_enum(self) -> None:
        """Test target type enumeration."""
        assert TargetType.PROVIDER.value == "provider"
        assert TargetType.REQUEST.value == "request"

    def test_fault_config_creation(self) -> None:
        """Test fault config creation."""
        config = FaultConfig(
            fault_type=FaultType.LATENCY,
            probability=0.5,
            duration_ms=1000,
        )
        assert config.probability == 0.5
        assert config.duration_ms == 1000

    def test_fault_config_should_inject(self) -> None:
        """Test fault config should inject."""
        config = FaultConfig(
            fault_type=FaultType.ERROR,
            probability=1.0,  # Always inject
        )
        assert config.should_inject()

        config.enabled = False
        assert not config.should_inject()

    def test_experiment_config_creation(self) -> None:
        """Test experiment config creation."""
        config = ExperimentConfig(
            name="test_experiment",
            description="Test chaos experiment",
            faults=[FaultConfig(fault_type=FaultType.LATENCY, probability=0.1)],
        )
        assert config.name == "test_experiment"
        assert len(config.faults) == 1

    def test_experiment_result_impact(self) -> None:
        """Test experiment result impact calculation."""
        result = ExperimentResult(
            experiment_name="test",
            status=ExperimentStatus.RUNNING,
            total_requests=100,
            affected_requests=25,
        )
        assert result.impact_percentage == 25.0

    def test_fault_injector_add_fault(self) -> None:
        """Test fault injector add fault."""
        injector = FaultInjector()

        fault = FaultConfig(fault_type=FaultType.LATENCY, probability=0.1)
        injector.add_fault(fault)
        injector.enable()

        assert injector.is_enabled()

    def test_fault_injector_remove_fault(self) -> None:
        """Test fault injector remove fault."""
        injector = FaultInjector()

        fault = FaultConfig(fault_type=FaultType.LATENCY)
        injector.add_fault(fault)
        injector.remove_fault(FaultType.LATENCY)

    def test_latency_injector(self) -> None:
        """Test latency injector."""
        injector = LatencyInjector(
            min_latency_ms=100,
            max_latency_ms=500,
            probability=1.0,
        )
        injector.enable()

        stats = injector.get_stats()
        assert stats["total_injected"] == 0

    def test_error_injector(self) -> None:
        """Test error injector."""
        injector = ErrorInjector(probability=1.0)
        injector.enable()

        error = injector.maybe_inject()
        assert error is not None
        assert isinstance(error, Exception)

    def test_chaos_experiment_lifecycle(self) -> None:
        """Test chaos experiment lifecycle."""
        config = ExperimentConfig(
            name="test_exp",
            faults=[FaultConfig(fault_type=FaultType.LATENCY, probability=0.0)],
        )
        experiment = ChaosExperiment(config)

        experiment.start()
        assert experiment.is_running()

        result = experiment.stop()
        assert result.status == ExperimentStatus.COMPLETED

    def test_chaos_experiment_abort(self) -> None:
        """Test chaos experiment abort."""
        config = ExperimentConfig(name="test_exp")
        experiment = ChaosExperiment(config)

        experiment.start()
        result = experiment.abort("Manual abort")

        assert result.status == ExperimentStatus.ABORTED
        assert result.error == "Manual abort"

    def test_chaos_manager_create_experiment(self) -> None:
        """Test chaos manager create experiment."""
        manager = ChaosManager()

        config = ExperimentConfig(name="test_exp")
        experiment = manager.create_experiment(config)

        assert experiment is not None
        assert "test_exp" in manager.list_experiments()

    def test_chaos_manager_start_stop(self) -> None:
        """Test chaos manager start and stop experiment."""
        manager = ChaosManager()

        config = ExperimentConfig(name="test_exp")
        manager.create_experiment(config)

        assert manager.start_experiment("test_exp")
        assert manager.get_active_experiment() is not None

        result = manager.stop_experiment()
        assert result is not None
        assert result.status == ExperimentStatus.COMPLETED

    def test_create_experiment_helper(self) -> None:
        """Test create experiment helper function."""
        config = create_experiment(
            name="latency_test",
            faults=[FaultConfig(fault_type=FaultType.LATENCY)],
            duration_minutes=10,
            max_impact=30.0,
        )

        assert config.name == "latency_test"
        assert config.max_impact_percentage == 30.0

    @pytest.mark.asyncio
    async def test_chaos_vision_provider(self) -> None:
        """Test chaos vision provider."""
        mock_provider = MockVisionProvider()
        chaos_provider = ChaosVisionProvider(mock_provider)

        # Without any injection enabled
        result = await chaos_provider.analyze_image(b"test")
        assert result.summary is not None

    @pytest.mark.asyncio
    async def test_chaos_vision_provider_with_latency(self) -> None:
        """Test chaos vision provider with latency injection."""
        mock_provider = MockVisionProvider()
        chaos_provider = create_chaos_provider(mock_provider)

        # Enable latency injection with 0% probability (no delay)
        chaos_provider.enable_latency_injection(
            min_ms=1,
            max_ms=2,
            probability=0.0,
        )

        result = await chaos_provider.analyze_image(b"test")
        assert result.summary is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhase9Integration:
    """Integration tests for Phase 9 features."""

    @pytest.mark.asyncio
    async def test_observable_with_deployment(self) -> None:
        """Test observable provider with deployment."""
        # Create deployment manager with observable providers
        manager = DeploymentManager()

        v1_provider = MockVisionProvider("v1")
        observable_v1 = create_observable_provider(v1_provider)

        manager.register_version("1.0.0", observable_v1)

        deployment_provider = DeploymentVisionProvider(manager)
        result = await deployment_provider.analyze_image(b"test")

        assert result.summary is not None

    @pytest.mark.asyncio
    async def test_multi_tenant_with_compliance(self) -> None:
        """Test multi-tenant provider with compliance."""
        tenant_manager = TenantManager()
        tenant_manager.create_tenant("tenant1", "Test Tenant")

        compliance = ComplianceManager()
        compliance.enable_standard(ComplianceStandard.GDPR)

        mock_provider = MockVisionProvider()
        compliant_provider = ComplianceVisionProvider(
            provider=mock_provider,
            compliance=compliance,
        )

        tenant_manager.set_default_provider(compliant_provider)

        mt_provider = MultiTenantVisionProvider(tenant_manager)
        result = await mt_provider.analyze_for_tenant("tenant1", b"test")

        assert result.summary is not None

        # Verify audit trail
        events = compliance.get_audit_logger().get_events()
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_chaos_with_observability(self) -> None:
        """Test chaos provider with observability."""
        mock_provider = MockVisionProvider()
        observable = create_observable_provider(mock_provider)
        chaos = ChaosVisionProvider(observable)

        result = await chaos.analyze_image(b"test")
        assert result.summary is not None

    @pytest.mark.asyncio
    async def test_full_phase9_stack(self) -> None:
        """Test full Phase 9 feature stack."""
        # Create base provider
        base_provider = MockVisionProvider("base")

        # Wrap with observability
        observable = create_observable_provider(base_provider)

        # Create compliance manager
        compliance = ComplianceManager()
        compliance.enable_standard(ComplianceStandard.GDPR)

        # Wrap with compliance
        compliant = ComplianceVisionProvider(
            provider=observable,
            compliance=compliance,
        )

        # Create tenant manager
        tenant_manager = TenantManager()
        tenant_manager.create_tenant(
            "enterprise_tenant",
            "Enterprise Customer",
            TenantTier.ENTERPRISE,
        )
        tenant_manager.set_default_provider(compliant)

        # Create multi-tenant provider
        mt_provider = MultiTenantVisionProvider(tenant_manager)

        # Wrap with chaos engineering
        chaos_provider = ChaosVisionProvider(mt_provider)

        # Execute
        TenantContext.set_tenant("enterprise_tenant")
        try:
            result = await chaos_provider.analyze_image(b"test_image_data")
            assert result.summary is not None
            assert result.confidence > 0
        finally:
            TenantContext.clear()

        # Verify audit
        events = compliance.get_audit_logger().get_events()
        assert len(events) > 0

        # Verify tenant usage
        usage = tenant_manager.get_usage("enterprise_tenant")
        assert usage.total_requests > 0
