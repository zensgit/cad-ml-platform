"""Tests for Phase 7 Vision Provider features.

Phase 7 includes:
- Middleware chain pattern
- Advanced circuit breaker
- Configurable retry policies
- Request context propagation
- Provider connection pooling
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timedelta

from src.core.vision.base import VisionDescription, VisionProvider


# Helper functions
def create_test_description(
    summary: str = "Test summary",
    details: list = None,
    confidence: float = 0.85,
) -> VisionDescription:
    """Create a test VisionDescription."""
    return VisionDescription(
        summary=summary,
        details=details or ["Detail 1", "Detail 2"],
        confidence=confidence,
    )


def create_mock_provider(name: str = "mock") -> MagicMock:
    """Create a mock vision provider."""
    provider = MagicMock()
    provider.provider_name = name
    provider.analyze_image = AsyncMock(return_value=create_test_description())
    return provider


# ==================== Middleware Chain Tests ====================


class TestMiddlewareChain:
    """Tests for middleware chain module."""

    def test_middleware_phase_enum(self):
        """Test MiddlewarePhase enum values."""
        from src.core.vision.middleware import MiddlewarePhase

        assert MiddlewarePhase.BEFORE.value == "before"
        assert MiddlewarePhase.AFTER.value == "after"
        assert MiddlewarePhase.ERROR.value == "error"
        assert MiddlewarePhase.FINALLY.value == "finally"

    def test_middleware_priority_enum(self):
        """Test MiddlewarePriority enum values."""
        from src.core.vision.middleware import MiddlewarePriority

        assert MiddlewarePriority.HIGHEST.value == 0
        assert MiddlewarePriority.NORMAL.value == 50
        assert MiddlewarePriority.LOWEST.value == 100

    def test_middleware_context_creation(self):
        """Test MiddlewareContext creation."""
        from src.core.vision.middleware import MiddlewareContext

        context = MiddlewareContext(
            request_id="test-123",
            image_data=b"test_image",
            provider_name="test_provider",
        )

        assert context.request_id == "test-123"
        assert context.image_data == b"test_image"
        assert context.skip_remaining is False

    def test_middleware_context_skip(self):
        """Test MiddlewareContext skip functionality."""
        from src.core.vision.middleware import MiddlewareContext

        context = MiddlewareContext(
            request_id="test",
            image_data=b"data",
        )

        context.skip()
        assert context.skip_remaining is True

    def test_middleware_chain_creation(self):
        """Test MiddlewareChain creation."""
        from src.core.vision.middleware import MiddlewareChain

        chain = MiddlewareChain(name="test_chain")

        assert chain.name == "test_chain"
        assert chain.stats.total_executions == 0

    def test_middleware_chain_add_remove(self):
        """Test adding and removing middleware."""
        from src.core.vision.middleware import (
            MiddlewareChain,
            LambdaMiddleware,
        )

        chain = MiddlewareChain()

        middleware = LambdaMiddleware(
            name="test",
            before_fn=lambda ctx: None,
        )

        chain.add(middleware)
        assert chain.get("test") is not None

        removed = chain.remove("test")
        assert removed is True
        assert chain.get("test") is None

    def test_lambda_middleware(self):
        """Test LambdaMiddleware."""
        from src.core.vision.middleware import (
            LambdaMiddleware,
            MiddlewareContext,
        )

        calls = []

        def before_fn(ctx):
            calls.append("before")

        def after_fn(ctx):
            calls.append("after")

        middleware = LambdaMiddleware(
            name="test",
            before_fn=before_fn,
            after_fn=after_fn,
        )

        context = MiddlewareContext(
            request_id="test",
            image_data=b"data",
        )

        middleware.before(context)
        middleware.after(context)

        assert "before" in calls
        assert "after" in calls

    def test_timing_middleware(self):
        """Test TimingMiddleware."""
        from src.core.vision.middleware import (
            TimingMiddleware,
            MiddlewareContext,
        )
        import time

        middleware = TimingMiddleware()
        context = MiddlewareContext(
            request_id="test",
            image_data=b"data",
        )

        middleware.before(context)
        time.sleep(0.01)
        middleware.after(context)

        assert "timing_ms" in context.metadata
        assert context.metadata["timing_ms"] > 0

    def test_middleware_chain_stats(self):
        """Test MiddlewareChainStats tracking."""
        from src.core.vision.middleware import MiddlewareChainStats

        stats = MiddlewareChainStats()
        stats.record_execution("mw1", True, 10.0)
        stats.record_execution("mw1", True, 15.0)
        stats.record_execution("mw2", False, 5.0)

        assert stats.total_executions == 3
        assert stats.successful_executions == 2
        assert stats.failed_executions == 1
        assert stats.middleware_durations["mw1"] == 25.0

    @pytest.mark.asyncio
    async def test_middleware_vision_provider(self):
        """Test MiddlewareVisionProvider."""
        from src.core.vision.middleware import (
            MiddlewareVisionProvider,
            MiddlewareChain,
            TimingMiddleware,
        )

        mock_provider = create_mock_provider()
        chain = MiddlewareChain()
        chain.add(TimingMiddleware())

        provider = MiddlewareVisionProvider(mock_provider, chain)

        result = await provider.analyze_image(b"test_image")

        assert result is not None
        mock_provider.analyze_image.assert_called_once()

    def test_create_middleware_provider(self):
        """Test create_middleware_provider factory."""
        from src.core.vision.middleware import create_middleware_provider

        mock_provider = create_mock_provider()
        provider = create_middleware_provider(mock_provider)

        assert provider.provider_name == "middleware_mock"


# ==================== Advanced Circuit Breaker Tests ====================


class TestCircuitBreaker:
    """Tests for advanced circuit breaker module."""

    def test_circuit_state_enum(self):
        """Test CircuitState enum values."""
        from src.core.vision.circuit_breaker import CircuitState

        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_failure_type_enum(self):
        """Test FailureType enum values."""
        from src.core.vision.circuit_breaker import FailureType

        assert FailureType.EXCEPTION.value == "exception"
        assert FailureType.TIMEOUT.value == "timeout"

    def test_recovery_strategy_enum(self):
        """Test RecoveryStrategy enum values."""
        from src.core.vision.circuit_breaker import RecoveryStrategy

        assert RecoveryStrategy.FIXED.value == "fixed"
        assert RecoveryStrategy.EXPONENTIAL.value == "exponential"

    def test_circuit_breaker_config(self):
        """Test CircuitBreakerConfig dataclass."""
        from src.core.vision.circuit_breaker import CircuitBreakerConfig

        config = CircuitBreakerConfig(
            failure_threshold=10,
            timeout_seconds=60.0,
        )

        assert config.failure_threshold == 10
        assert config.timeout_seconds == 60.0

    def test_sliding_window(self):
        """Test SlidingWindow."""
        from src.core.vision.circuit_breaker import SlidingWindow, CallRecord

        window = SlidingWindow(size=5)

        # Add successful calls
        for _ in range(3):
            window.add(CallRecord(timestamp=datetime.now(), success=True, duration_ms=100))

        # Add failed calls
        for _ in range(2):
            window.add(CallRecord(timestamp=datetime.now(), success=False, duration_ms=100))

        assert window.size == 5
        assert window.get_failure_rate() == pytest.approx(0.4)

    def test_circuit_breaker_creation(self):
        """Test CircuitBreaker creation."""
        from src.core.vision.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(name="test_cb")

        assert cb.name == "test_cb"
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after failures."""
        from src.core.vision.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(name="test", config=config)

        # Record failures
        for _ in range(3):
            cb.record_failure(100.0)

        assert cb.state == CircuitState.OPEN

    def test_circuit_breaker_stats(self):
        """Test CircuitBreakerStats."""
        from src.core.vision.circuit_breaker import CircuitBreakerStats

        stats = CircuitBreakerStats()
        stats.successful_calls = 8
        stats.failed_calls = 2

        assert stats.failure_rate == pytest.approx(0.2)

    def test_circuit_breaker_reset(self):
        """Test circuit breaker reset."""
        from src.core.vision.circuit_breaker import (
            CircuitBreaker,
            CircuitState,
        )

        cb = CircuitBreaker(name="test")

        # Open circuit
        for _ in range(5):
            cb.record_failure(100.0)

        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_vision_provider(self):
        """Test CircuitBreakerVisionProvider."""
        from src.core.vision.circuit_breaker import (
            CircuitBreakerVisionProvider,
            CircuitBreaker,
        )

        mock_provider = create_mock_provider()
        cb = CircuitBreaker(name="test")

        provider = CircuitBreakerVisionProvider(mock_provider, cb)

        result = await provider.analyze_image(b"test_image")

        assert result is not None
        assert cb.stats.successful_calls == 1


# ==================== Retry Policy Tests ====================


class TestRetryPolicy:
    """Tests for retry policy module."""

    def test_backoff_strategy_enum(self):
        """Test BackoffStrategy enum values."""
        from src.core.vision.retry_policy import BackoffStrategy

        assert BackoffStrategy.FIXED.value == "fixed"
        assert BackoffStrategy.EXPONENTIAL.value == "exponential"
        assert BackoffStrategy.LINEAR.value == "linear"
        assert BackoffStrategy.FIBONACCI.value == "fibonacci"

    def test_jitter_type_enum(self):
        """Test JitterType enum values."""
        from src.core.vision.retry_policy import JitterType

        assert JitterType.NONE.value == "none"
        assert JitterType.FULL.value == "full"
        assert JitterType.EQUAL.value == "equal"

    def test_retry_policy_config(self):
        """Test RetryPolicyConfig dataclass."""
        from src.core.vision.retry_policy import RetryPolicyConfig

        config = RetryPolicyConfig(
            max_retries=5,
            initial_delay_seconds=2.0,
        )

        assert config.max_retries == 5
        assert config.initial_delay_seconds == 2.0

    def test_fixed_backoff(self):
        """Test FixedBackoff calculator."""
        from src.core.vision.retry_policy import FixedBackoff, RetryPolicyConfig

        config = RetryPolicyConfig(initial_delay_seconds=1.0)
        backoff = FixedBackoff()

        assert backoff.calculate_delay(0, config) == 1.0
        assert backoff.calculate_delay(1, config) == 1.0
        assert backoff.calculate_delay(5, config) == 1.0

    def test_exponential_backoff(self):
        """Test ExponentialBackoff calculator."""
        from src.core.vision.retry_policy import ExponentialBackoff, RetryPolicyConfig

        config = RetryPolicyConfig(
            initial_delay_seconds=1.0,
            backoff_multiplier=2.0,
            max_delay_seconds=60.0,
        )
        backoff = ExponentialBackoff()

        assert backoff.calculate_delay(0, config) == 1.0
        assert backoff.calculate_delay(1, config) == 2.0
        assert backoff.calculate_delay(2, config) == 4.0
        assert backoff.calculate_delay(3, config) == 8.0

    def test_linear_backoff(self):
        """Test LinearBackoff calculator."""
        from src.core.vision.retry_policy import LinearBackoff, RetryPolicyConfig

        config = RetryPolicyConfig(
            initial_delay_seconds=1.0,
            max_delay_seconds=60.0,
        )
        backoff = LinearBackoff()

        assert backoff.calculate_delay(0, config) == 1.0
        assert backoff.calculate_delay(1, config) == 2.0
        assert backoff.calculate_delay(2, config) == 3.0

    def test_fibonacci_backoff(self):
        """Test FibonacciBackoff calculator."""
        from src.core.vision.retry_policy import FibonacciBackoff, RetryPolicyConfig

        config = RetryPolicyConfig(
            initial_delay_seconds=1.0,
            max_delay_seconds=60.0,
        )
        backoff = FibonacciBackoff()

        # Fibonacci: 1, 1, 2, 3, 5, 8...
        assert backoff.calculate_delay(0, config) == 1.0
        assert backoff.calculate_delay(1, config) == 1.0
        assert backoff.calculate_delay(2, config) == 2.0
        assert backoff.calculate_delay(3, config) == 3.0
        assert backoff.calculate_delay(4, config) == 5.0

    def test_retry_budget(self):
        """Test RetryBudget."""
        from src.core.vision.retry_policy import RetryBudget

        budget = RetryBudget(
            max_retries_per_window=5,
            window_seconds=60.0,
        )

        assert budget.can_retry() is True

        for _ in range(5):
            budget.record_retry()

        assert budget.can_retry() is False

    def test_retry_stats(self):
        """Test RetryStats."""
        from src.core.vision.retry_policy import RetryStats

        stats = RetryStats()
        stats.total_operations = 10
        stats.successful_first_attempt = 7
        stats.successful_after_retry = 2
        stats.failed_after_retries = 1

        assert stats.success_rate == pytest.approx(0.9)
        assert stats.first_attempt_success_rate == pytest.approx(0.7)

    def test_retry_policy_should_retry(self):
        """Test RetryPolicy should_retry."""
        from src.core.vision.retry_policy import RetryPolicy, RetryPolicyConfig

        config = RetryPolicyConfig(max_retries=3)
        policy = RetryPolicy(config)

        assert policy.should_retry(0, ValueError("test")) is True
        assert policy.should_retry(2, ValueError("test")) is True
        assert policy.should_retry(3, ValueError("test")) is False

    @pytest.mark.asyncio
    async def test_retry_vision_provider(self):
        """Test RetryVisionProvider."""
        from src.core.vision.retry_policy import (
            RetryVisionProvider,
            RetryPolicy,
            RetryPolicyConfig,
        )

        mock_provider = create_mock_provider()
        config = RetryPolicyConfig(max_retries=3)
        policy = RetryPolicy(config)

        provider = RetryVisionProvider(mock_provider, policy)

        result = await provider.analyze_image(b"test_image")

        assert result is not None
        assert policy.stats.successful_first_attempt == 1


# ==================== Request Context Tests ====================


class TestRequestContext:
    """Tests for request context module."""

    def test_context_scope_enum(self):
        """Test ContextScope enum values."""
        from src.core.vision.request_context import ContextScope

        assert ContextScope.REQUEST.value == "request"
        assert ContextScope.SESSION.value == "session"
        assert ContextScope.GLOBAL.value == "global"

    def test_baggage_item(self):
        """Test BaggageItem dataclass."""
        from src.core.vision.request_context import BaggageItem

        item = BaggageItem(key="user_id", value="12345", propagate=True)

        assert item.key == "user_id"
        assert item.value == "12345"
        assert item.propagate is True

    def test_request_context_creation(self):
        """Test RequestContext creation."""
        from src.core.vision.request_context import RequestContext

        context = RequestContext()

        assert context.request_id is not None
        assert len(context.request_id) > 0

    def test_request_context_baggage(self):
        """Test RequestContext baggage operations."""
        from src.core.vision.request_context import RequestContext

        context = RequestContext()

        context.set_baggage("key1", "value1")
        assert context.get_baggage("key1") == "value1"
        assert context.get_baggage("nonexistent") is None

        removed = context.remove_baggage("key1")
        assert removed is True
        assert context.get_baggage("key1") is None

    def test_request_context_attributes(self):
        """Test RequestContext attribute operations."""
        from src.core.vision.request_context import RequestContext

        context = RequestContext()

        context.set_attribute("key1", "value1")
        context.set_attribute("key2", 123)

        assert context.get_attribute("key1") == "value1"
        assert context.get_attribute("key2") == 123
        assert context.get_attribute("missing", "default") == "default"

    def test_request_context_spans(self):
        """Test RequestContext span operations."""
        from src.core.vision.request_context import RequestContext

        context = RequestContext()

        span = context.start_span("operation1")
        assert span.operation_name == "operation1"
        assert context.current_span == span

        span.add_event("event1", {"key": "value"})
        assert len(span.events) == 1

        ended = context.end_span()
        assert ended == span
        assert span.end_time is not None

    def test_request_context_child(self):
        """Test RequestContext child creation."""
        from src.core.vision.request_context import RequestContext

        parent = RequestContext()
        parent.set_baggage("propagate_me", "value", propagate=True)
        parent.set_baggage("dont_propagate", "value", propagate=False)

        child = parent.create_child()

        assert child.parent_context_id == parent.request_id
        assert child.get_baggage("propagate_me") == "value"
        assert child.get_baggage("dont_propagate") is None

    def test_context_manager_create_get(self):
        """Test ContextManager create and get."""
        from src.core.vision.request_context import ContextManager

        manager = ContextManager()

        context = manager.create_context(correlation_id="corr-123")
        assert context.correlation_id == "corr-123"

        retrieved = manager.get_context(context.request_id)
        assert retrieved == context

    def test_context_scope_context_manager(self):
        """Test context_scope context manager."""
        from src.core.vision.request_context import (
            context_scope,
            get_current_context,
            RequestContext,
        )

        context = RequestContext()

        with context_scope(context) as ctx:
            assert ctx == context
            assert get_current_context() == context

        # Context should be reset after exiting

    def test_context_to_dict(self):
        """Test RequestContext to_dict."""
        from src.core.vision.request_context import RequestContext

        context = RequestContext()
        context.set_baggage("key", "value")
        context.set_attribute("attr", "value")

        d = context.to_dict()

        assert "request_id" in d
        assert "baggage" in d
        assert "attributes" in d

    @pytest.mark.asyncio
    async def test_context_aware_vision_provider(self):
        """Test ContextAwareVisionProvider."""
        from src.core.vision.request_context import (
            ContextAwareVisionProvider,
            ContextManager,
        )

        mock_provider = create_mock_provider()
        manager = ContextManager()

        provider = ContextAwareVisionProvider(
            mock_provider, manager, auto_create_context=True
        )

        result = await provider.analyze_image(b"test_image")

        assert result is not None


# ==================== Provider Pool Tests ====================


class TestProviderPool:
    """Tests for provider pool module."""

    def test_pool_state_enum(self):
        """Test PoolState enum values."""
        from src.core.vision.provider_pool import PoolState

        assert PoolState.RUNNING.value == "running"
        assert PoolState.PAUSED.value == "paused"
        assert PoolState.DRAINING.value == "draining"
        assert PoolState.STOPPED.value == "stopped"

    def test_connection_state_enum(self):
        """Test ConnectionState enum values."""
        from src.core.vision.provider_pool import ConnectionState

        assert ConnectionState.IDLE.value == "idle"
        assert ConnectionState.IN_USE.value == "in_use"
        assert ConnectionState.UNHEALTHY.value == "unhealthy"
        assert ConnectionState.CLOSED.value == "closed"

    def test_pool_config(self):
        """Test PoolConfig dataclass."""
        from src.core.vision.provider_pool import PoolConfig

        config = PoolConfig(
            min_size=2,
            max_size=20,
            acquire_timeout_seconds=10.0,
        )

        assert config.min_size == 2
        assert config.max_size == 20
        assert config.acquire_timeout_seconds == 10.0

    def test_pooled_connection(self):
        """Test PooledConnection."""
        from src.core.vision.provider_pool import PooledConnection, ConnectionState

        mock_provider = create_mock_provider()
        conn = PooledConnection(
            connection_id="conn_1",
            provider=mock_provider,
        )

        assert conn.state == ConnectionState.IDLE
        assert conn.use_count == 0

        conn.mark_in_use()
        assert conn.state == ConnectionState.IN_USE
        assert conn.use_count == 1

        conn.mark_idle(100.0)
        assert conn.state == ConnectionState.IDLE
        assert conn.total_duration_ms == 100.0

    def test_pooled_connection_expiration(self):
        """Test PooledConnection expiration."""
        from src.core.vision.provider_pool import PooledConnection
        from datetime import timedelta

        mock_provider = create_mock_provider()
        conn = PooledConnection(
            connection_id="conn_1",
            provider=mock_provider,
        )

        # Not expired with long lifetime
        assert conn.is_expired(3600) is False

        # Would be expired with 0 lifetime
        assert conn.is_expired(0) is True

    def test_pool_stats(self):
        """Test PoolStats."""
        from src.core.vision.provider_pool import PoolStats

        stats = PoolStats()
        stats.total_acquisitions = 100
        stats.successful_acquisitions = 95
        stats.total_wait_time_ms = 5000.0

        assert stats.average_wait_time_ms == pytest.approx(50.0)

    def test_simple_provider_factory(self):
        """Test SimpleProviderFactory."""
        from src.core.vision.provider_pool import SimpleProviderFactory

        def create_fn():
            return create_mock_provider()

        factory = SimpleProviderFactory(create_fn)

        provider = factory.create()
        assert provider is not None

    @pytest.mark.asyncio
    async def test_provider_pool_creation(self):
        """Test ProviderPool creation."""
        from src.core.vision.provider_pool import (
            ProviderPool,
            SimpleProviderFactory,
            PoolConfig,
        )

        factory = SimpleProviderFactory(create_fn=create_mock_provider)
        config = PoolConfig(min_size=2, max_size=10)
        pool = ProviderPool(factory, config)

        await pool.initialize()

        assert pool.size == 2
        assert pool.stats.connections_created == 2

    @pytest.mark.asyncio
    async def test_provider_pool_acquire_release(self):
        """Test ProviderPool acquire and release."""
        from src.core.vision.provider_pool import (
            ProviderPool,
            SimpleProviderFactory,
            PoolConfig,
        )

        factory = SimpleProviderFactory(create_fn=create_mock_provider)
        config = PoolConfig(min_size=1, max_size=5, validation_on_acquire=False)
        pool = ProviderPool(factory, config)

        await pool.initialize()

        conn = await pool.acquire()
        assert conn is not None
        assert pool.stats.active_connections == 1

        await pool.release(conn, 100.0)
        assert pool.stats.idle_connections == 1

    @pytest.mark.asyncio
    async def test_pooled_vision_provider(self):
        """Test PooledVisionProvider."""
        from src.core.vision.provider_pool import (
            PooledVisionProvider,
            ProviderPool,
            SimpleProviderFactory,
            PoolConfig,
        )

        factory = SimpleProviderFactory(create_fn=create_mock_provider)
        config = PoolConfig(min_size=1, max_size=5, validation_on_acquire=False)
        pool = ProviderPool(factory, config)

        await pool.initialize()

        provider = PooledVisionProvider(pool)

        result = await provider.analyze_image(b"test_image")

        assert result is not None
        assert pool.stats.successful_acquisitions == 1


# ==================== Integration Tests ====================


class TestPhase7Integration:
    """Integration tests for Phase 7 features."""

    @pytest.mark.asyncio
    async def test_middleware_with_circuit_breaker(self):
        """Test middleware with circuit breaker."""
        from src.core.vision.middleware import (
            MiddlewareVisionProvider,
            MiddlewareChain,
            TimingMiddleware,
        )
        from src.core.vision.circuit_breaker import (
            CircuitBreakerVisionProvider,
            CircuitBreaker,
        )

        mock_provider = create_mock_provider()

        # Add circuit breaker
        cb = CircuitBreaker(name="test")
        cb_provider = CircuitBreakerVisionProvider(mock_provider, cb)

        # Add middleware
        chain = MiddlewareChain()
        chain.add(TimingMiddleware())
        mw_provider = MiddlewareVisionProvider(cb_provider, chain)

        result = await mw_provider.analyze_image(b"test_image")

        assert result is not None
        assert cb.stats.successful_calls == 1

    @pytest.mark.asyncio
    async def test_retry_with_context(self):
        """Test retry policy with request context."""
        from src.core.vision.retry_policy import (
            RetryVisionProvider,
            RetryPolicy,
        )
        from src.core.vision.request_context import (
            ContextAwareVisionProvider,
            ContextManager,
        )

        mock_provider = create_mock_provider()

        # Add context awareness
        manager = ContextManager()
        ctx_provider = ContextAwareVisionProvider(mock_provider, manager)

        # Add retry policy
        policy = RetryPolicy()
        retry_provider = RetryVisionProvider(ctx_provider, policy)

        result = await retry_provider.analyze_image(b"test_image")

        assert result is not None
        assert policy.stats.successful_first_attempt == 1

    @pytest.mark.asyncio
    async def test_pool_with_middleware(self):
        """Test provider pool with middleware."""
        from src.core.vision.provider_pool import (
            ProviderPool,
            SimpleProviderFactory,
            PoolConfig,
        )
        from src.core.vision.middleware import (
            MiddlewareVisionProvider,
            MiddlewareChain,
            TimingMiddleware,
        )

        # Create pool
        factory = SimpleProviderFactory(create_fn=create_mock_provider)
        config = PoolConfig(min_size=1, max_size=5, validation_on_acquire=False)
        pool = ProviderPool(factory, config)
        await pool.initialize()

        # Acquire and wrap with middleware
        conn = await pool.acquire()

        chain = MiddlewareChain()
        chain.add(TimingMiddleware())
        mw_provider = MiddlewareVisionProvider(conn.provider, chain)

        result = await mw_provider.analyze_image(b"test_image")
        assert result is not None

        await pool.release(conn, 100.0)
