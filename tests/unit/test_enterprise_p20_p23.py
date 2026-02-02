"""Unit tests for P20-P23 enterprise modules."""

import asyncio
import time
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

# P20: Gateway tests
from src.core.gateway.rate_limiter import (
    RateLimitConfig,
    RateLimitStrategy,
    FixedWindowLimiter,
    SlidingWindowLimiter,
    TokenBucketLimiter,
    create_rate_limiter,
)
from src.core.gateway.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreakerError,
    get_circuit_breaker,
)
from src.core.gateway.api_key import (
    APIKey,
    APIKeyScope,
    APIKeyManager,
    SCOPE_HIERARCHY,
)

# P21: Audit tests
from src.core.audit.logger import (
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    AuditContext,
    AuditLogger,
    MemoryAuditBackend,
)
from src.core.audit.compliance import (
    ComplianceFramework,
    DataCategory,
    AccessPurpose,
    RetentionPolicy,
    ComplianceTracker,
)

# P22: Cache tests
from src.core.cache.client import (
    CacheConfig,
    CacheEntry,
    MemoryCache,
)
from src.core.cache.strategies import (
    CacheAside,
)

# P23: Health tests
from src.core.health.checker import (
    HealthChecker,
    HealthStatus,
    DependencyHealth,
    DiskSpaceHealthCheck,
    CustomHealthCheck,
)
from src.core.health.probes import (
    LivenessProbe,
    ReadinessProbe,
    StartupProbe,
    ProbeStatus,
)
from src.core.health.self_healing import (
    SelfHealer,
    HealingAction,
    HealingActionType,
    CacheClearStrategy,
)


# =============================================================================
# P20: Gateway Tests
# =============================================================================

class TestRateLimiter:
    """Test rate limiting implementations."""

    @pytest.mark.asyncio
    async def test_fixed_window_allows_requests(self):
        """Test fixed window allows requests under limit."""
        config = RateLimitConfig(requests_per_minute=10)
        limiter = FixedWindowLimiter(config)

        for i in range(5):
            result = await limiter.check(f"user1")
            assert result.allowed is True

    @pytest.mark.asyncio
    async def test_fixed_window_blocks_excess(self):
        """Test fixed window blocks requests over limit."""
        config = RateLimitConfig(requests_per_minute=5)
        limiter = FixedWindowLimiter(config)

        # Use up the limit
        for i in range(5):
            await limiter.check("user1")

        # Next request should be blocked
        result = await limiter.check("user1")
        assert result.allowed is False
        assert result.retry_after is not None

    @pytest.mark.asyncio
    async def test_sliding_window_allows_requests(self):
        """Test sliding window allows requests under limit."""
        config = RateLimitConfig(requests_per_minute=10)
        limiter = SlidingWindowLimiter(config)

        for i in range(5):
            result = await limiter.check("user1")
            assert result.allowed is True

    @pytest.mark.asyncio
    async def test_token_bucket_allows_burst(self):
        """Test token bucket allows burst up to bucket size."""
        config = RateLimitConfig(burst_size=5, requests_per_second=1)
        limiter = TokenBucketLimiter(config)

        # Burst should succeed
        for i in range(5):
            result = await limiter.check("user1")
            assert result.allowed is True

    @pytest.mark.asyncio
    async def test_rate_limit_reset(self):
        """Test rate limit reset."""
        config = RateLimitConfig(requests_per_minute=5)
        limiter = SlidingWindowLimiter(config)

        # Use up limit
        for i in range(5):
            await limiter.check("user1")

        # Reset
        await limiter.reset("user1")

        # Should allow again
        result = await limiter.check("user1")
        assert result.allowed is True


class TestCircuitBreaker:
    """Test circuit breaker implementation."""

    @pytest.mark.asyncio
    async def test_circuit_starts_closed(self):
        """Test circuit starts in closed state."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_opens_on_failures(self):
        """Test circuit opens after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)

        async def failing_func():
            raise Exception("Failure")

        # Cause failures
        for i in range(3):
            try:
                await cb.call(failing_func)
            except Exception:
                pass

        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_rejects_when_open(self):
        """Test circuit rejects calls when open."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        cb.force_open()

        with pytest.raises(CircuitBreakerError):
            await cb.call(lambda: "test")

    @pytest.mark.asyncio
    async def test_circuit_closes_on_success(self):
        """Test circuit closes after success in half-open."""
        config = CircuitBreakerConfig(success_threshold=1, timeout=0.1)
        cb = CircuitBreaker("test", config)
        cb.force_open()

        # Wait for timeout
        await asyncio.sleep(0.2)

        # Should be half-open now
        assert cb.state == CircuitState.HALF_OPEN

        # Success should close
        await cb.call(lambda: "success")
        assert cb.state == CircuitState.CLOSED


class TestAPIKey:
    """Test API key management."""

    def test_api_key_scope_check(self):
        """Test API key scope checking."""
        key = APIKey(
            key_id="key1",
            key_hash="hash",
            name="Test Key",
            owner_id="user1",
            scopes={APIKeyScope.READ},
        )

        assert key.has_scope(APIKeyScope.READ) is True
        assert key.has_scope(APIKeyScope.READ_DOCUMENTS) is True  # Inherited
        assert key.has_scope(APIKeyScope.WRITE) is False

    def test_api_key_expiration(self):
        """Test API key expiration."""
        # Non-expired key
        key1 = APIKey(
            key_id="key1",
            key_hash="hash",
            name="Test Key",
            owner_id="user1",
            expires_at=datetime.utcnow() + timedelta(days=1),
        )
        assert key1.is_expired is False
        assert key1.is_valid is True

        # Expired key
        key2 = APIKey(
            key_id="key2",
            key_hash="hash",
            name="Test Key",
            owner_id="user1",
            expires_at=datetime.utcnow() - timedelta(days=1),
        )
        assert key2.is_expired is True
        assert key2.is_valid is False

    @pytest.mark.asyncio
    async def test_api_key_manager_create(self):
        """Test API key creation."""
        manager = APIKeyManager(secret_key="test-secret")

        result = await manager.create_key(
            name="Test Key",
            owner_id="user1",
            scopes={APIKeyScope.READ, APIKeyScope.WRITE},
        )

        assert result.raw_key.startswith("cad_")
        assert result.key.name == "Test Key"
        assert result.key.owner_id == "user1"

    @pytest.mark.asyncio
    async def test_api_key_manager_validate(self):
        """Test API key validation."""
        manager = APIKeyManager(secret_key="test-secret")

        result = await manager.create_key(
            name="Test Key",
            owner_id="user1",
            scopes={APIKeyScope.READ},
        )

        # Valid key
        validated = await manager.validate_key(result.raw_key)
        assert validated is not None
        assert validated.key_id == result.key.key_id

        # Invalid key
        invalid = await manager.validate_key("invalid-key")
        assert invalid is None

    @pytest.mark.asyncio
    async def test_api_key_manager_revoke(self):
        """Test API key revocation."""
        manager = APIKeyManager(secret_key="test-secret")

        result = await manager.create_key(
            name="Test Key",
            owner_id="user1",
        )

        # Revoke
        await manager.revoke_key(result.key.key_id, reason="Test revocation")

        # Should no longer validate
        validated = await manager.validate_key(result.raw_key)
        assert validated is None


# =============================================================================
# P21: Audit Tests
# =============================================================================

class TestAuditLogger:
    """Test audit logging."""

    @pytest.mark.asyncio
    async def test_audit_event_creation(self):
        """Test audit event creation."""
        backend = MemoryAuditBackend()
        logger = AuditLogger(backends=[backend])

        event = await logger.log(
            event_type=AuditEventType.DATA_READ,
            action="Read document",
            resource_type="document",
            resource_id="doc123",
        )

        assert event.event_type == AuditEventType.DATA_READ
        assert event.action == "Read document"
        assert event.checksum is not None

    @pytest.mark.asyncio
    async def test_audit_event_query(self):
        """Test audit event querying."""
        backend = MemoryAuditBackend()
        logger = AuditLogger(backends=[backend])

        # Create events
        await logger.log(AuditEventType.DATA_READ, "Read 1")
        await logger.log(AuditEventType.DATA_CREATE, "Create 1")
        await logger.log(AuditEventType.DATA_READ, "Read 2")

        # Query
        results = await logger.query(event_types=[AuditEventType.DATA_READ])
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_audit_context(self):
        """Test audit context management."""
        backend = MemoryAuditBackend()
        logger = AuditLogger(backends=[backend])

        context = AuditContext(user_id="user1", tenant_id="tenant1")
        logger.push_context(context)

        event = await logger.log(AuditEventType.DATA_READ, "Read")
        assert event.context.user_id == "user1"

        logger.pop_context()


class TestComplianceTracker:
    """Test compliance tracking."""

    @pytest.mark.asyncio
    async def test_log_data_access(self):
        """Test data access logging."""
        tracker = ComplianceTracker()

        log = await tracker.log_data_access(
            user_id="user1",
            data_category=DataCategory.PERSONAL,
            resource_type="user_profile",
            resource_id="profile123",
            access_type="read",
            purpose=AccessPurpose.USER_REQUEST,
        )

        assert log.user_id == "user1"
        assert log.data_category == DataCategory.PERSONAL

    @pytest.mark.asyncio
    async def test_consent_management(self):
        """Test consent recording and checking."""
        tracker = ComplianceTracker()

        # Record consent
        consent = await tracker.record_consent(
            user_id="user1",
            purpose="marketing",
            scope={"email", "analytics"},
        )

        assert consent.is_valid is True

        # Check consent
        found = await tracker.check_consent("user1", "marketing")
        assert found is not None

        # Revoke consent
        await tracker.revoke_consent(consent.consent_id)
        found = await tracker.check_consent("user1", "marketing")
        assert found is None

    @pytest.mark.asyncio
    async def test_dsr_submission(self):
        """Test data subject request submission."""
        tracker = ComplianceTracker()

        request = await tracker.submit_dsr(
            user_id="user1",
            request_type="access",
        )

        assert request.status == "pending"
        assert request.is_overdue is False

        # Complete
        await tracker.complete_dsr(request.request_id, {"data": "user_data"})
        assert tracker._dsr_requests[request.request_id].status == "completed"


# =============================================================================
# P22: Cache Tests
# =============================================================================

class TestMemoryCache:
    """Test memory cache."""

    @pytest.mark.asyncio
    async def test_cache_set_get(self):
        """Test basic set and get."""
        cache = MemoryCache()

        await cache.set("key1", "value1")
        result = await cache.get("key1")

        assert result == "value1"

    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test cache expiration."""
        config = CacheConfig(memory_cache_ttl_seconds=1)
        cache = MemoryCache(config)

        await cache.set("key1", "value1", ttl=1)

        # Should exist
        assert await cache.exists("key1") is True

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired
        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self):
        """Test LRU eviction."""
        config = CacheConfig(memory_cache_size=3)
        cache = MemoryCache(config)

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Access key1 to make it recent
        await cache.get("key1")

        # Add key4, should evict key2 (least recently used)
        await cache.set("key4", "value4")

        assert await cache.exists("key1") is True
        assert await cache.exists("key2") is False
        assert await cache.exists("key3") is True
        assert await cache.exists("key4") is True

    @pytest.mark.asyncio
    async def test_cache_delete_by_tags(self):
        """Test deletion by tags."""
        cache = MemoryCache()

        await cache.set("key1", "value1", tags=["tag1"])
        await cache.set("key2", "value2", tags=["tag1", "tag2"])
        await cache.set("key3", "value3", tags=["tag2"])

        count = await cache.delete_by_tags(["tag1"])
        assert count == 2

        assert await cache.exists("key1") is False
        assert await cache.exists("key2") is False
        assert await cache.exists("key3") is True


class TestCacheStrategies:
    """Test cache strategies."""

    @pytest.mark.asyncio
    async def test_cache_aside(self):
        """Test cache-aside strategy."""
        cache = MemoryCache()
        strategy = CacheAside(cache=cache)

        call_count = 0

        def loader():
            nonlocal call_count
            call_count += 1
            return "loaded_value"

        # First call - cache miss
        result1 = await strategy.get("key1", loader)
        assert result1 == "loaded_value"
        assert call_count == 1

        # Second call - cache hit
        result2 = await strategy.get("key1", loader)
        assert result2 == "loaded_value"
        assert call_count == 1  # Loader not called again


# =============================================================================
# P23: Health Tests
# =============================================================================

class TestHealthChecker:
    """Test health checker."""

    @pytest.mark.asyncio
    async def test_healthy_status(self):
        """Test healthy status with all checks passing."""
        checker = HealthChecker()

        # Add a passing check
        checker.register(CustomHealthCheck(
            name="test",
            check_func=lambda: DependencyHealth(
                name="test",
                status=HealthStatus.HEALTHY,
            ),
        ))

        result = await checker.check_all()
        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_degraded_status(self):
        """Test degraded status with non-critical failure."""
        checker = HealthChecker()

        # Add a failing non-critical check
        checker.register(CustomHealthCheck(
            name="test",
            check_func=lambda: DependencyHealth(
                name="test",
                status=HealthStatus.UNHEALTHY,
            ),
            critical=False,
        ))

        result = await checker.check_all()
        assert result.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_unhealthy_status(self):
        """Test unhealthy status with critical failure."""
        checker = HealthChecker()

        # Add a failing critical check
        checker.register(CustomHealthCheck(
            name="test",
            check_func=lambda: DependencyHealth(
                name="test",
                status=HealthStatus.UNHEALTHY,
            ),
            critical=True,
        ))

        result = await checker.check_all()
        assert result.status == HealthStatus.UNHEALTHY


class TestProbes:
    """Test Kubernetes probes."""

    @pytest.mark.asyncio
    async def test_liveness_probe(self):
        """Test liveness probe."""
        probe = LivenessProbe()
        result = await probe.execute()

        assert result.is_success is True

    @pytest.mark.asyncio
    async def test_startup_probe(self):
        """Test startup probe."""
        probe = StartupProbe()

        # First execution
        result = await probe.execute()
        assert result.is_success is True
        assert probe.is_started is True

        # Subsequent executions always succeed
        result2 = await probe.execute()
        assert result2.is_success is True

    @pytest.mark.asyncio
    async def test_readiness_with_failing_check(self):
        """Test readiness probe with failing check."""
        probe = ReadinessProbe()

        def failing_check():
            return False

        probe.add_check(failing_check)
        result = await probe.execute()

        assert result.is_success is False


class TestSelfHealing:
    """Test self-healing mechanisms."""

    @pytest.mark.asyncio
    async def test_cache_clear_healing(self):
        """Test cache clearing healing action."""
        cache = MemoryCache()
        await cache.set("key1", "value1")

        strategy = CacheClearStrategy()
        strategy.register_cache("test_cache", cache)

        action = HealingAction(
            action_type=HealingActionType.CLEAR_CACHE,
            target="test_cache",
        )

        result = await strategy.heal(action)
        assert result.success is True
        assert await cache.exists("key1") is False

    @pytest.mark.asyncio
    async def test_self_healer_stats(self):
        """Test self-healer statistics."""
        healer = SelfHealer()

        strategy = CacheClearStrategy()
        cache = MemoryCache()
        strategy.register_cache("test", cache)

        healer.register_strategy(HealingActionType.CLEAR_CACHE, strategy)

        action = HealingAction(
            action_type=HealingActionType.CLEAR_CACHE,
            target="test",
        )

        await healer.heal(action)

        stats = healer.get_stats()
        assert stats["total_actions"] == 1
        assert stats["successful_actions"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
