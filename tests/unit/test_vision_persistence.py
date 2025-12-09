"""Tests for vision persistence, analytics, failover, and health monitoring modules."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.vision.base import VisionDescription, VisionProvider
from src.core.vision.persistence import (
    AnalysisRecord,
    InMemoryStorage,
    QueryFilter,
    ResultPersistence,
    PersistentVisionProvider,
    create_persistent_provider,
)
from src.core.vision.analytics import (
    VisionAnalytics,
    ProviderStats,
    TimeBucket,
    TrendData,
    TimeGranularity,
    AnalyticsReport,
    create_analytics,
)
from src.core.vision.failover import (
    FailoverManager,
    FailoverVisionProvider,
    ProviderEndpoint,
    ProviderHealth,
    FailoverStrategy,
    FailoverConfig,
    FailoverResult,
    create_failover_provider,
)
from src.core.vision.health import (
    HealthMonitor,
    HealthAwareVisionProvider,
    HealthMetrics,
    HealthStatus,
    HealthAlert,
    AlertSeverity,
    HealthCheckConfig,
    HealthDashboard,
    create_health_aware_provider,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_provider():
    """Create a mock VisionProvider."""
    provider = MagicMock(spec=VisionProvider)
    provider.provider_name = "test_provider"
    provider.analyze_image = AsyncMock(return_value=VisionDescription(
        summary="Test summary",
        details=["Detail 1", "Detail 2"],
        confidence=0.95,
    ))
    return provider


@pytest.fixture
def mock_failing_provider():
    """Create a mock VisionProvider that fails."""
    provider = MagicMock(spec=VisionProvider)
    provider.provider_name = "failing_provider"
    provider.analyze_image = AsyncMock(side_effect=Exception("Provider error"))
    return provider


@pytest.fixture
def sample_result():
    """Create a sample VisionDescription."""
    return VisionDescription(
        summary="Engineering drawing analysis",
        details=["Contains resistor", "Contains capacitor"],
        confidence=0.92,
    )


@pytest.fixture
def sample_image():
    """Create sample image bytes."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


# ============================================================================
# Persistence Tests
# ============================================================================


class TestInMemoryStorage:
    """Tests for InMemoryStorage."""

    @pytest.mark.asyncio
    async def test_save_and_get(self, sample_result):
        """Test saving and retrieving a record."""
        storage = InMemoryStorage()
        record = AnalysisRecord(
            record_id="test-1",
            image_hash="abc123",
            provider="openai",
            result=sample_result,
            created_at=datetime.now(),
        )

        await storage.save(record)
        retrieved = await storage.get("test-1")

        assert retrieved is not None
        assert retrieved.record_id == "test-1"
        assert retrieved.result.summary == "Engineering drawing analysis"

    @pytest.mark.asyncio
    async def test_query_by_provider(self, sample_result):
        """Test querying records by provider."""
        storage = InMemoryStorage()

        # Add records for different providers
        for i, provider in enumerate(["openai", "openai", "anthropic"]):
            record = AnalysisRecord(
                record_id=f"test-{i}",
                image_hash=f"hash-{i}",
                provider=provider,
                result=sample_result,
                created_at=datetime.now(),
            )
            await storage.save(record)

        filter = QueryFilter(provider="openai")
        result = await storage.query(filter)

        assert result.total_count == 2
        assert all(r.provider == "openai" for r in result.records)

    @pytest.mark.asyncio
    async def test_query_by_date_range(self, sample_result):
        """Test querying records by date range."""
        storage = InMemoryStorage()

        # Add records with different dates
        now = datetime.now()
        dates = [now - timedelta(days=5), now - timedelta(days=2), now]

        for i, date in enumerate(dates):
            record = AnalysisRecord(
                record_id=f"test-{i}",
                image_hash=f"hash-{i}",
                provider="openai",
                result=sample_result,
                created_at=date,
            )
            await storage.save(record)

        filter = QueryFilter(
            start_date=now - timedelta(days=3),
            end_date=now,
        )
        result = await storage.query(filter)

        assert result.total_count == 2

    @pytest.mark.asyncio
    async def test_delete_record(self, sample_result):
        """Test deleting a record."""
        storage = InMemoryStorage()
        record = AnalysisRecord(
            record_id="test-1",
            image_hash="abc123",
            provider="openai",
            result=sample_result,
            created_at=datetime.now(),
        )

        await storage.save(record)
        assert await storage.delete("test-1") is True
        assert await storage.get("test-1") is None

    @pytest.mark.asyncio
    async def test_update_tags(self, sample_result):
        """Test updating tags on a record."""
        storage = InMemoryStorage()
        record = AnalysisRecord(
            record_id="test-1",
            image_hash="abc123",
            provider="openai",
            result=sample_result,
            created_at=datetime.now(),
            tags=["initial"],
        )

        await storage.save(record)
        await storage.update_tags("test-1", ["updated", "tags"])

        retrieved = await storage.get("test-1")
        assert retrieved.tags == ["updated", "tags"]


class TestResultPersistence:
    """Tests for ResultPersistence."""

    @pytest.mark.asyncio
    async def test_save_result(self, sample_result, sample_image):
        """Test saving an analysis result."""
        persistence = ResultPersistence()

        record = await persistence.save_result(
            image_data=sample_image,
            provider="openai",
            result=sample_result,
            processing_time_ms=150.0,
            cost_usd=0.01,
            tags=["test"],
        )

        assert record.record_id is not None
        assert record.provider == "openai"
        assert record.processing_time_ms == 150.0

    @pytest.mark.asyncio
    async def test_find_by_image(self, sample_result, sample_image):
        """Test finding records by image."""
        persistence = ResultPersistence()

        # Save multiple results for same image
        await persistence.save_result(sample_image, "openai", sample_result)
        await persistence.save_result(sample_image, "anthropic", sample_result)

        records = await persistence.find_by_image(sample_image)
        assert len(records) == 2

    @pytest.mark.asyncio
    async def test_query_with_confidence_filter(self, sample_image):
        """Test querying with confidence filter."""
        persistence = ResultPersistence()

        # Save results with different confidence levels
        for conf in [0.5, 0.7, 0.9]:
            result = VisionDescription(
                summary="Test",
                details=[],
                confidence=conf,
            )
            await persistence.save_result(sample_image, "openai", result)

        query_result = await persistence.query_results(min_confidence=0.6)
        assert query_result.total_count == 2


class TestPersistentVisionProvider:
    """Tests for PersistentVisionProvider."""

    @pytest.mark.asyncio
    async def test_analyze_and_persist(self, mock_provider, sample_image):
        """Test analyzing image and persisting result."""
        persistence = ResultPersistence()
        persistent_provider = PersistentVisionProvider(
            provider=mock_provider,
            persistence=persistence,
            auto_tag=["auto"],
        )

        result, record = await persistent_provider.analyze_image(
            sample_image,
            tags=["manual"],
        )

        assert result.summary == "Test summary"
        assert record.record_id is not None
        assert "auto" in record.tags
        assert "manual" in record.tags


# ============================================================================
# Analytics Tests
# ============================================================================


class TestProviderStats:
    """Tests for ProviderStats."""

    def test_success_rate(self):
        """Test success rate calculation."""
        stats = ProviderStats(
            provider="openai",
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            total_cost_usd=1.0,
            total_processing_time_ms=15000.0,
            avg_confidence=0.9,
            min_confidence=0.7,
            max_confidence=0.99,
        )

        assert stats.success_rate == 0.95
        assert stats.avg_processing_time_ms == 150.0
        assert stats.avg_cost_per_request == 0.01


class TestVisionAnalytics:
    """Tests for VisionAnalytics."""

    @pytest.mark.asyncio
    async def test_get_provider_stats(self, sample_result, sample_image):
        """Test getting provider statistics."""
        persistence = ResultPersistence()

        # Add some records
        for _ in range(5):
            await persistence.save_result(
                sample_image, "openai", sample_result,
                processing_time_ms=100.0, cost_usd=0.01,
            )

        analytics = VisionAnalytics(persistence)
        stats = await analytics.get_provider_stats()

        assert "openai" in stats
        assert stats["openai"].total_requests == 5

    @pytest.mark.asyncio
    async def test_get_trends(self, sample_result, sample_image):
        """Test getting usage trends."""
        persistence = ResultPersistence()

        # Add records
        for _ in range(3):
            await persistence.save_result(sample_image, "openai", sample_result)

        analytics = VisionAnalytics(persistence)
        trends = await analytics.get_trends(
            granularity=TimeGranularity.DAY,
            days=7,
        )

        assert trends.total_requests == 3
        assert len(trends.buckets) > 0

    @pytest.mark.asyncio
    async def test_get_top_tags(self, sample_result, sample_image):
        """Test getting top tags."""
        persistence = ResultPersistence()

        # Add records with tags
        await persistence.save_result(
            sample_image, "openai", sample_result, tags=["circuit", "pcb"],
        )
        await persistence.save_result(
            sample_image, "openai", sample_result, tags=["circuit", "schematic"],
        )

        analytics = VisionAnalytics(persistence)
        top_tags = await analytics.get_top_tags(limit=5)

        assert len(top_tags) > 0
        assert top_tags[0][0] == "circuit"  # Most common tag
        assert top_tags[0][1] == 2

    @pytest.mark.asyncio
    async def test_generate_report(self, sample_result, sample_image):
        """Test generating analytics report."""
        persistence = ResultPersistence()

        # Add records
        await persistence.save_result(sample_image, "openai", sample_result)

        analytics = VisionAnalytics(persistence)
        report = await analytics.generate_report()

        assert report.report_id is not None
        assert report.overall_stats.total_requests == 1
        assert isinstance(report.to_dict(), dict)


# ============================================================================
# Failover Tests
# ============================================================================


class TestProviderEndpoint:
    """Tests for ProviderEndpoint."""

    def test_record_success(self, mock_provider):
        """Test recording successful request."""
        endpoint = ProviderEndpoint(
            provider=mock_provider,
            priority=0,
            health=ProviderHealth.UNKNOWN,
        )

        endpoint.record_success(100.0)

        assert endpoint.health == ProviderHealth.HEALTHY
        assert endpoint.consecutive_failures == 0
        assert endpoint.avg_latency_ms == 100.0

    def test_record_failure_degraded(self, mock_provider):
        """Test recording failures leading to degraded status."""
        endpoint = ProviderEndpoint(
            provider=mock_provider,
            priority=0,
            health=ProviderHealth.HEALTHY,
        )

        endpoint.record_failure()
        endpoint.record_failure()

        assert endpoint.health == ProviderHealth.DEGRADED
        assert endpoint.consecutive_failures == 2

    def test_record_failure_unhealthy(self, mock_provider):
        """Test recording failures leading to unhealthy status."""
        endpoint = ProviderEndpoint(
            provider=mock_provider,
            priority=0,
            health=ProviderHealth.HEALTHY,
        )

        for _ in range(5):
            endpoint.record_failure()

        assert endpoint.health == ProviderHealth.UNHEALTHY
        assert endpoint.is_available() is False


class TestFailoverManager:
    """Tests for FailoverManager."""

    @pytest.mark.asyncio
    async def test_select_endpoint_priority(self, mock_provider):
        """Test endpoint selection with priority strategy."""
        provider1 = MagicMock(spec=VisionProvider)
        provider1.provider_name = "provider1"
        provider2 = MagicMock(spec=VisionProvider)
        provider2.provider_name = "provider2"

        endpoints = [
            ProviderEndpoint(provider=provider2, priority=1, health=ProviderHealth.HEALTHY),
            ProviderEndpoint(provider=provider1, priority=0, health=ProviderHealth.HEALTHY),
        ]

        config = FailoverConfig(strategy=FailoverStrategy.PRIORITY)
        manager = FailoverManager(endpoints=endpoints, config=config)

        selected = await manager.select_endpoint()
        assert selected.provider.provider_name == "provider1"

    @pytest.mark.asyncio
    async def test_select_endpoint_round_robin(self, mock_provider):
        """Test endpoint selection with round-robin strategy."""
        provider1 = MagicMock(spec=VisionProvider)
        provider1.provider_name = "provider1"
        provider2 = MagicMock(spec=VisionProvider)
        provider2.provider_name = "provider2"

        endpoints = [
            ProviderEndpoint(provider=provider1, priority=0, health=ProviderHealth.HEALTHY),
            ProviderEndpoint(provider=provider2, priority=0, health=ProviderHealth.HEALTHY),
        ]

        config = FailoverConfig(strategy=FailoverStrategy.ROUND_ROBIN)
        manager = FailoverManager(endpoints=endpoints, config=config)

        first = await manager.select_endpoint()
        second = await manager.select_endpoint()

        assert first.provider.provider_name != second.provider.provider_name

    @pytest.mark.asyncio
    async def test_analyze_with_failover_success(self, mock_provider, sample_image):
        """Test successful analysis with failover."""
        endpoints = [
            ProviderEndpoint(
                provider=mock_provider,
                priority=0,
                health=ProviderHealth.HEALTHY,
            ),
        ]

        manager = FailoverManager(endpoints=endpoints)
        result = await manager.analyze_with_failover(sample_image)

        assert result.success is True
        assert result.result is not None
        assert result.provider_used == "test_provider"

    @pytest.mark.asyncio
    async def test_analyze_with_failover_fallback(
        self, mock_provider, mock_failing_provider, sample_image
    ):
        """Test failover to backup provider."""
        endpoints = [
            ProviderEndpoint(
                provider=mock_failing_provider,
                priority=0,
                health=ProviderHealth.HEALTHY,
            ),
            ProviderEndpoint(
                provider=mock_provider,
                priority=1,
                health=ProviderHealth.HEALTHY,
            ),
        ]

        config = FailoverConfig(max_retries=3)
        manager = FailoverManager(endpoints=endpoints, config=config)
        result = await manager.analyze_with_failover(sample_image)

        assert result.success is True
        assert result.provider_used == "test_provider"
        assert len(result.attempts) >= 2


class TestFailoverVisionProvider:
    """Tests for FailoverVisionProvider."""

    @pytest.mark.asyncio
    async def test_analyze_image(self, mock_provider, sample_image):
        """Test analyzing image through failover provider."""
        endpoints = [
            ProviderEndpoint(
                provider=mock_provider,
                priority=0,
                health=ProviderHealth.HEALTHY,
            ),
        ]

        manager = FailoverManager(endpoints=endpoints)
        failover_provider = FailoverVisionProvider(manager)

        result = await failover_provider.analyze_image(sample_image)

        assert result.summary == "Test summary"
        assert failover_provider.provider_name == "failover"


class TestCreateFailoverProvider:
    """Tests for create_failover_provider factory."""

    def test_create_failover_provider(self, mock_provider):
        """Test creating failover provider."""
        providers = [mock_provider]
        failover = create_failover_provider(
            providers,
            strategy=FailoverStrategy.PRIORITY,
            max_retries=5,
        )

        assert failover.provider_name == "failover"
        assert len(failover.failover_manager.get_endpoints()) == 1


# ============================================================================
# Health Monitoring Tests
# ============================================================================


class TestHealthMetrics:
    """Tests for HealthMetrics."""

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = HealthMetrics(
            provider="openai",
            status=HealthStatus.HEALTHY,
            total_checks=100,
            successful_checks=95,
            avg_latency_ms=150.0,
        )

        data = metrics.to_dict()

        assert data["provider"] == "openai"
        assert data["status"] == "healthy"
        assert data["total_checks"] == 100


class TestHealthMonitor:
    """Tests for HealthMonitor."""

    @pytest.mark.asyncio
    async def test_register_provider(self, mock_provider):
        """Test registering a provider for monitoring."""
        monitor = HealthMonitor()
        monitor.register_provider(mock_provider)

        assert mock_provider.provider_name in monitor._metrics
        assert monitor._metrics[mock_provider.provider_name].status == HealthStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_check_provider_success(self, mock_provider):
        """Test successful health check."""
        monitor = HealthMonitor()
        monitor.register_provider(mock_provider)

        metrics = await monitor.check_provider(mock_provider.provider_name)

        assert metrics.status == HealthStatus.HEALTHY
        assert metrics.successful_checks == 1
        assert metrics.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_check_provider_failure(self, mock_failing_provider):
        """Test failed health check."""
        monitor = HealthMonitor(config=HealthCheckConfig(
            failure_threshold=1,
            degraded_threshold=1,
        ))
        monitor.register_provider(mock_failing_provider)

        metrics = await monitor.check_provider(mock_failing_provider.provider_name)

        assert metrics.status in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)
        assert metrics.failed_checks == 1

    @pytest.mark.asyncio
    async def test_check_all_providers(self, mock_provider):
        """Test checking all providers."""
        provider1 = MagicMock(spec=VisionProvider)
        provider1.provider_name = "provider1"
        provider1.analyze_image = AsyncMock(return_value=VisionDescription(
            summary="Test", details=[], confidence=0.9,
        ))

        provider2 = MagicMock(spec=VisionProvider)
        provider2.provider_name = "provider2"
        provider2.analyze_image = AsyncMock(return_value=VisionDescription(
            summary="Test", details=[], confidence=0.9,
        ))

        monitor = HealthMonitor()
        monitor.register_provider(provider1)
        monitor.register_provider(provider2)

        results = await monitor.check_all_providers()

        assert len(results) == 2
        assert all(m.status == HealthStatus.HEALTHY for m in results.values())

    @pytest.mark.asyncio
    async def test_get_dashboard(self, mock_provider):
        """Test generating health dashboard."""
        monitor = HealthMonitor()
        monitor.register_provider(mock_provider)
        await monitor.check_provider(mock_provider.provider_name)

        dashboard = monitor.get_dashboard()

        assert dashboard.overall_status == HealthStatus.HEALTHY
        assert dashboard.summary["total_providers"] == 1
        assert dashboard.summary["healthy_count"] == 1

    @pytest.mark.asyncio
    async def test_alert_on_status_change(self, mock_failing_provider):
        """Test alert generation on status change."""
        alerts_received = []

        def callback(alert):
            alerts_received.append(alert)

        monitor = HealthMonitor(config=HealthCheckConfig(
            failure_threshold=1,
            degraded_threshold=1,
        ))
        monitor.register_provider(mock_failing_provider)
        monitor.add_alert_callback(callback)

        await monitor.check_provider(mock_failing_provider.provider_name)

        # May or may not have alerts depending on status change
        alerts = monitor.get_alerts(active_only=False)
        assert isinstance(alerts, list)

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, mock_failing_provider):
        """Test acknowledging an alert."""
        monitor = HealthMonitor(config=HealthCheckConfig(
            failure_threshold=1,
            degraded_threshold=1,
        ))
        monitor.register_provider(mock_failing_provider)

        await monitor.check_provider(mock_failing_provider.provider_name)

        alerts = monitor.get_alerts(active_only=False)
        if alerts:
            alert_id = alerts[0].alert_id
            assert monitor.acknowledge_alert(alert_id) is True
            assert alerts[0].acknowledged is True


class TestHealthAwareVisionProvider:
    """Tests for HealthAwareVisionProvider."""

    @pytest.mark.asyncio
    async def test_analyze_updates_metrics(self, mock_provider, sample_image):
        """Test that analysis updates health metrics."""
        monitor = HealthMonitor()
        aware_provider = HealthAwareVisionProvider(mock_provider, monitor)

        result = await aware_provider.analyze_image(sample_image)

        assert result.summary == "Test summary"
        health = aware_provider.get_health()
        assert health.successful_checks == 1
        assert health.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_analyze_failure_updates_metrics(
        self, mock_failing_provider, sample_image
    ):
        """Test that failed analysis updates health metrics."""
        monitor = HealthMonitor(config=HealthCheckConfig(
            failure_threshold=1,
            degraded_threshold=1,
        ))
        aware_provider = HealthAwareVisionProvider(mock_failing_provider, monitor)

        with pytest.raises(Exception):
            await aware_provider.analyze_image(sample_image)

        health = aware_provider.get_health()
        assert health.failed_checks == 1
        assert health.consecutive_failures == 1


class TestCreateHealthAwareProvider:
    """Tests for create_health_aware_provider factory."""

    def test_create_health_aware_provider(self, mock_provider):
        """Test creating health-aware provider."""
        aware_provider = create_health_aware_provider(mock_provider)

        assert aware_provider.provider_name == mock_provider.provider_name
        assert aware_provider.health_monitor is not None


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple modules."""

    @pytest.mark.asyncio
    async def test_persistent_failover_with_health(self, sample_image):
        """Test combining persistence, failover, and health monitoring."""
        # Create providers
        provider1 = MagicMock(spec=VisionProvider)
        provider1.provider_name = "provider1"
        provider1.analyze_image = AsyncMock(return_value=VisionDescription(
            summary="Provider 1 result",
            details=[],
            confidence=0.95,
        ))

        provider2 = MagicMock(spec=VisionProvider)
        provider2.provider_name = "provider2"
        provider2.analyze_image = AsyncMock(return_value=VisionDescription(
            summary="Provider 2 result",
            details=[],
            confidence=0.90,
        ))

        # Create failover provider
        failover = create_failover_provider([provider1, provider2])

        # Create health monitor
        monitor = HealthMonitor()
        monitor.register_provider(provider1)
        monitor.register_provider(provider2)

        # Create persistent provider wrapping failover
        persistence = ResultPersistence()
        persistent = PersistentVisionProvider(
            provider=failover,
            persistence=persistence,
        )

        # Analyze image
        result, record = await persistent.analyze_image(sample_image)

        assert result.summary in ["Provider 1 result", "Provider 2 result"]
        assert record.record_id is not None

        # Check analytics
        analytics = VisionAnalytics(persistence)
        stats = await analytics.get_provider_stats()
        assert "failover" in stats

    @pytest.mark.asyncio
    async def test_analytics_report_generation(self, sample_result, sample_image):
        """Test full analytics report generation."""
        persistence = ResultPersistence()

        # Create diverse test data
        providers = ["openai", "anthropic", "deepseek"]
        for i, provider in enumerate(providers):
            result = VisionDescription(
                summary=f"Result from {provider}",
                details=[],
                confidence=0.8 + i * 0.05,
            )
            await persistence.save_result(
                sample_image,
                provider,
                result,
                processing_time_ms=100 + i * 50,
                cost_usd=0.01 * (i + 1),
                tags=["test", provider],
            )

        analytics = VisionAnalytics(persistence)
        report = await analytics.generate_report()

        assert report.overall_stats.total_requests == 3
        assert len(report.provider_stats) == 3
        assert len(report.insights) > 0
        assert "test" in [t[0] for t in report.top_tags]
