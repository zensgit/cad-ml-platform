"""
Unit tests for Tenacity-based retry module.

Tests cover:
- RetryConfig configuration
- with_retry decorator behavior
- Pre-configured retry decorators
- Fallback behavior when tenacity unavailable
- RetryContext context manager
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import time

from src.core.resilience.retry import (
    RetryConfig,
    with_retry,
    provider_retry,
    database_retry,
    network_retry,
    quick_retry,
    no_retry,
    RetryContext,
    RetryError,
    TENACITY_AVAILABLE,
)


class TestRetryConfig:
    """Test RetryConfig configuration class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.min_wait == 1.0
        assert config.max_wait == 60.0
        assert config.multiplier == 2.0
        assert config.retry_exceptions == (Exception,)
        assert config.reraise is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = RetryConfig(
            max_attempts=5,
            min_wait=0.5,
            max_wait=30.0,
            multiplier=1.5,
            retry_exceptions=(ConnectionError, TimeoutError),
            reraise=False,
        )

        assert config.max_attempts == 5
        assert config.min_wait == 0.5
        assert config.max_wait == 30.0
        assert config.multiplier == 1.5
        assert config.retry_exceptions == (ConnectionError, TimeoutError)
        assert config.reraise is False


class TestWithRetryDecorator:
    """Test the configurable with_retry decorator."""

    def test_sync_function_success(self):
        """Test successful sync function execution."""
        call_count = 0

        @with_retry(max_attempts=3)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()

        assert result == "success"
        assert call_count == 1

    def test_sync_function_retry_then_success(self):
        """Test sync function that fails then succeeds."""
        call_count = 0

        @with_retry(max_attempts=3, min_wait=0.01, max_wait=0.01)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient error")
            return "success"

        result = flaky_func()

        assert result == "success"
        assert call_count == 3

    def test_sync_function_exhausted_retries(self):
        """Test sync function that exhausts all retries."""
        call_count = 0

        @with_retry(max_attempts=2, min_wait=0.01, max_wait=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Persistent error")

        with pytest.raises((ConnectionError, RetryError)):
            always_fails()

        assert call_count == 2

    def test_sync_function_specific_exceptions(self):
        """Test retry only on specific exceptions."""
        call_count = 0

        @with_retry(
            max_attempts=3,
            min_wait=0.01,
            retry_exceptions=(ConnectionError,),
        )
        def selective_retry():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            selective_retry()

        # Should only be called once since ValueError is not retryable
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_function_success(self):
        """Test successful async function execution."""
        call_count = 0

        @with_retry(max_attempts=3)
        async def async_successful():
            nonlocal call_count
            call_count += 1
            return "async success"

        result = await async_successful()

        assert result == "async success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_function_retry_then_success(self):
        """Test async function that fails then succeeds."""
        call_count = 0

        @with_retry(max_attempts=3, min_wait=0.01, max_wait=0.01)
        async def async_flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Transient timeout")
            return "recovered"

        result = await async_flaky()

        assert result == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_function_exhausted_retries(self):
        """Test async function that exhausts all retries."""
        call_count = 0

        @with_retry(max_attempts=2, min_wait=0.01, max_wait=0.01)
        async def async_always_fails():
            nonlocal call_count
            call_count += 1
            raise OSError("Persistent error")

        with pytest.raises((OSError, RetryError)):
            await async_always_fails()

        assert call_count == 2


class TestPreConfiguredDecorators:
    """Test pre-configured retry decorators."""

    @pytest.mark.asyncio
    async def test_provider_retry(self):
        """Test provider_retry decorator."""
        call_count = 0

        @provider_retry
        async def call_provider():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Provider unavailable")
            return "provider response"

        result = await call_provider()
        assert result == "provider response"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_database_retry(self):
        """Test database_retry decorator."""
        call_count = 0

        @database_retry
        async def db_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("DB connection lost")
            return "db result"

        result = await db_operation()
        assert result == "db result"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_network_retry(self):
        """Test network_retry decorator."""
        call_count = 0

        @network_retry
        async def network_call():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("Request timed out")
            return "network response"

        result = await network_call()
        assert result == "network response"
        assert call_count == 2

    def test_quick_retry(self):
        """Test quick_retry decorator."""
        call_count = 0

        @quick_retry
        def quick_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Quick failure")
            return "quick result"

        result = quick_operation()
        assert result == "quick result"
        assert call_count == 2


class TestNoRetryDecorator:
    """Test the no_retry decorator."""

    def test_no_retry_marks_function(self):
        """Test that no_retry marks function as non-retryable."""

        @no_retry
        def critical_operation():
            return "critical"

        assert hasattr(critical_operation, "_no_retry")
        assert critical_operation._no_retry is True

    def test_no_retry_preserves_function(self):
        """Test that no_retry doesn't change function behavior."""

        @no_retry
        def normal_func(x, y):
            return x + y

        assert normal_func(1, 2) == 3


class TestRetryContext:
    """Test RetryContext context manager."""

    def test_iteration(self):
        """Test basic iteration through retry attempts."""
        ctx = RetryContext(max_attempts=3)
        attempts = list(ctx)

        assert attempts == [0, 1, 2]

    def test_should_retry_within_limit(self):
        """Test should_retry within attempt limit."""
        ctx = RetryContext(max_attempts=3)
        ctx._attempt = 1

        assert ctx.should_retry() is True

    def test_should_retry_at_limit(self):
        """Test should_retry at attempt limit."""
        ctx = RetryContext(max_attempts=3)
        ctx._attempt = 3

        assert ctx.should_retry() is False

    def test_wait_sync(self):
        """Test synchronous wait with backoff."""
        ctx = RetryContext(max_attempts=3, min_wait=0.01, max_wait=0.1, multiplier=2.0)

        start = time.time()
        ctx.wait_sync()
        elapsed = time.time() - start

        assert elapsed >= 0.01
        assert ctx._wait_time == 0.02  # min_wait * multiplier

    @pytest.mark.asyncio
    async def test_wait_async(self):
        """Test asynchronous wait with backoff."""
        ctx = RetryContext(max_attempts=3, min_wait=0.01, max_wait=0.1, multiplier=2.0)

        start = time.time()
        await ctx.wait()
        elapsed = time.time() - start

        assert elapsed >= 0.01
        assert ctx._wait_time == 0.02

    def test_sync_context_manager(self):
        """Test synchronous context manager."""
        with RetryContext(max_attempts=2) as ctx:
            assert ctx.max_attempts == 2

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test asynchronous context manager."""
        async with RetryContext(max_attempts=2) as ctx:
            assert ctx.max_attempts == 2

    def test_retry_loop_pattern(self):
        """Test common retry loop pattern."""
        results = []
        errors = []

        ctx = RetryContext(max_attempts=3, min_wait=0.01)

        for attempt in ctx:
            try:
                if attempt < 2:
                    raise ValueError(f"Attempt {attempt} failed")
                results.append(f"Success on attempt {attempt}")
                break
            except ValueError as e:
                errors.append(str(e))
                if ctx.should_retry():
                    ctx.wait_sync()

        assert len(errors) == 2
        assert len(results) == 1
        assert results[0] == "Success on attempt 2"


class TestFallbackBehavior:
    """Test fallback behavior when tenacity is not available."""

    def test_fallback_sync_retry(self):
        """Test fallback sync retry implementation."""
        from src.core.resilience.retry import _fallback_retry

        call_count = 0

        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Transient")
            return "success"

        wrapped = _fallback_retry(
            flaky_func,
            max_attempts=3,
            min_wait=0.01,
            max_wait=0.1,
            multiplier=2.0,
            retry_exceptions=(ConnectionError,),
        )

        result = wrapped()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_fallback_async_retry(self):
        """Test fallback async retry implementation."""
        from src.core.resilience.retry import _fallback_retry

        call_count = 0

        async def async_flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Transient")
            return "async success"

        wrapped = _fallback_retry(
            async_flaky,
            max_attempts=3,
            min_wait=0.01,
            max_wait=0.1,
            multiplier=2.0,
            retry_exceptions=(TimeoutError,),
        )

        result = await wrapped()
        assert result == "async success"
        assert call_count == 2


class TestRetryModuleExports:
    """Test module exports and availability."""

    def test_all_exports_available(self):
        """Test that all expected exports are available."""
        from src.core.resilience import retry

        assert hasattr(retry, "with_retry")
        assert hasattr(retry, "provider_retry")
        assert hasattr(retry, "database_retry")
        assert hasattr(retry, "network_retry")
        assert hasattr(retry, "quick_retry")
        assert hasattr(retry, "no_retry")
        assert hasattr(retry, "RetryConfig")
        assert hasattr(retry, "RetryError")
        assert hasattr(retry, "RetryContext")

    def test_tenacity_availability_flag(self):
        """Test TENACITY_AVAILABLE flag is set."""
        from src.core.resilience.retry import TENACITY_AVAILABLE

        # Should be True since tenacity is in requirements.txt
        assert isinstance(TENACITY_AVAILABLE, bool)
