"""Tests for src/core/resilience/retry_policy.py to improve coverage.

Covers:
- RetryStats dataclass
- RetryStrategy subclasses (FixedDelay, LinearBackoff, ExponentialBackoff, FibonacciBackoff)
- RetryPolicy execute, _should_retry, get_stats, reset_stats, get_health, _emit_metrics
- retry decorator
- AdaptiveRetry execute and _adjust_parameters
- RetryError exception
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.core.resilience.retry_policy import (
    RetryError,
    RetryStats,
    RetryStrategy,
    FixedDelay,
    LinearBackoff,
    ExponentialBackoff,
    FibonacciBackoff,
    RetryPolicy,
    retry,
    AdaptiveRetry,
)


class TestRetryStats:
    """Tests for RetryStats dataclass."""

    def test_default_values(self):
        """Test RetryStats default values."""
        stats = RetryStats()

        assert stats.total_attempts == 0
        assert stats.successful_retries == 0
        assert stats.failed_retries == 0
        assert stats.total_delay_time == 0.0
        assert stats.last_retry_time is None
        assert stats.error_distribution == {}

    def test_post_init_error_distribution(self):
        """Test __post_init__ initializes error_distribution."""
        stats = RetryStats(total_attempts=5)

        assert stats.error_distribution == {}

    def test_custom_values(self):
        """Test RetryStats with custom values."""
        stats = RetryStats(
            total_attempts=10,
            successful_retries=8,
            failed_retries=2,
            total_delay_time=15.5,
            last_retry_time=datetime.now(),
            error_distribution={"TimeoutError": 2}
        )

        assert stats.total_attempts == 10
        assert stats.successful_retries == 8
        assert stats.error_distribution["TimeoutError"] == 2


class TestFixedDelay:
    """Tests for FixedDelay strategy."""

    def test_fixed_delay_constant(self):
        """Test FixedDelay returns constant delay."""
        strategy = FixedDelay(delay=5.0)

        assert strategy.get_delay(1) == 5.0
        assert strategy.get_delay(2) == 5.0
        assert strategy.get_delay(10) == 5.0

    def test_fixed_delay_zero(self):
        """Test FixedDelay with zero delay."""
        strategy = FixedDelay(delay=0.0)

        assert strategy.get_delay(1) == 0.0


class TestLinearBackoff:
    """Tests for LinearBackoff strategy."""

    def test_linear_backoff_calculation(self):
        """Test LinearBackoff calculates correctly."""
        strategy = LinearBackoff(initial_delay=1.0, increment=2.0)

        assert strategy.get_delay(1) == 1.0
        assert strategy.get_delay(2) == 3.0
        assert strategy.get_delay(3) == 5.0

    def test_linear_backoff_max_delay(self):
        """Test LinearBackoff respects max_delay."""
        strategy = LinearBackoff(initial_delay=1.0, increment=10.0, max_delay=15.0)

        assert strategy.get_delay(1) == 1.0
        assert strategy.get_delay(2) == 11.0
        assert strategy.get_delay(3) == 15.0  # Capped at max
        assert strategy.get_delay(100) == 15.0  # Still capped


class TestExponentialBackoff:
    """Tests for ExponentialBackoff strategy."""

    def test_exponential_backoff_calculation(self):
        """Test ExponentialBackoff calculates correctly."""
        strategy = ExponentialBackoff(base_delay=1.0, exponential_base=2.0, jitter=False)

        assert strategy.get_delay(1) == 1.0
        assert strategy.get_delay(2) == 2.0
        assert strategy.get_delay(3) == 4.0
        assert strategy.get_delay(4) == 8.0

    def test_exponential_backoff_max_delay(self):
        """Test ExponentialBackoff respects max_delay."""
        strategy = ExponentialBackoff(base_delay=1.0, exponential_base=2.0, max_delay=5.0, jitter=False)

        assert strategy.get_delay(1) == 1.0
        assert strategy.get_delay(2) == 2.0
        assert strategy.get_delay(3) == 4.0
        assert strategy.get_delay(4) == 5.0  # Capped at max

    def test_exponential_backoff_with_jitter(self):
        """Test ExponentialBackoff with jitter adds randomness."""
        strategy = ExponentialBackoff(base_delay=1.0, exponential_base=2.0, jitter=True)

        # With jitter, delay should be in range [0.5*delay, 1.5*delay]
        delays = [strategy.get_delay(2) for _ in range(10)]

        # All delays should be in valid range for attempt 2 (base delay 2.0)
        for d in delays:
            assert 1.0 <= d <= 3.0


class TestFibonacciBackoff:
    """Tests for FibonacciBackoff strategy."""

    def test_fibonacci_backoff_calculation(self):
        """Test FibonacciBackoff uses Fibonacci sequence."""
        strategy = FibonacciBackoff(initial_delay=1.0)

        # Fibonacci: 0, 1, 1, 2, 3, 5, 8, 13...
        assert strategy.get_delay(1) == 1.0  # fib(1) = 1
        assert strategy.get_delay(2) == 1.0  # fib(2) = 1
        assert strategy.get_delay(3) == 2.0  # fib(3) = 2
        assert strategy.get_delay(4) == 3.0  # fib(4) = 3
        assert strategy.get_delay(5) == 5.0  # fib(5) = 5

    def test_fibonacci_backoff_max_delay(self):
        """Test FibonacciBackoff respects max_delay."""
        strategy = FibonacciBackoff(initial_delay=1.0, max_delay=5.0)

        assert strategy.get_delay(5) == 5.0  # fib(5) = 5, at max
        assert strategy.get_delay(6) == 5.0  # fib(6) = 8, capped at 5
        assert strategy.get_delay(10) == 5.0  # Still capped

    def test_fibonacci_caching(self):
        """Test Fibonacci values are cached."""
        strategy = FibonacciBackoff(initial_delay=1.0)

        # First call computes and caches
        strategy.get_delay(10)

        # Check cache is populated
        assert 10 in strategy._fib_cache


class TestRetryPolicy:
    """Tests for RetryPolicy class."""

    def test_init_defaults(self):
        """Test RetryPolicy initialization with defaults."""
        policy = RetryPolicy(name="test")

        assert policy.name == "test"
        assert policy.max_attempts == 3
        assert isinstance(policy.strategy, ExponentialBackoff)
        assert policy.retryable_exceptions == [Exception]
        assert policy.non_retryable_exceptions == []

    def test_init_custom_strategy(self):
        """Test RetryPolicy with custom strategy."""
        strategy = FixedDelay(delay=1.0)
        policy = RetryPolicy(name="test", strategy=strategy)

        assert policy.strategy == strategy

    def test_execute_success_first_try(self):
        """Test execute succeeds on first try."""
        policy = RetryPolicy(name="test", max_attempts=3)

        def success_func():
            return "result"

        result = policy.execute(success_func)

        assert result == "result"
        assert policy._stats.successful_retries == 0  # No retries needed

    def test_execute_success_after_retry(self):
        """Test execute succeeds after retry."""
        policy = RetryPolicy(
            name="test",
            max_attempts=3,
            strategy=FixedDelay(delay=0.01)
        )

        call_count = 0

        def sometimes_fails():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "success"

        result = policy.execute(sometimes_fails)

        assert result == "success"
        assert call_count == 2
        assert policy._stats.successful_retries == 1

    def test_execute_max_retries_exhausted(self):
        """Test execute raises RetryError after max retries."""
        policy = RetryPolicy(
            name="test",
            max_attempts=3,
            strategy=FixedDelay(delay=0.01)
        )

        def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(RetryError) as exc_info:
            policy.execute(always_fails)

        assert "3 attempts" in str(exc_info.value)
        assert policy._stats.failed_retries == 1

    def test_should_retry_retryable_exception(self):
        """Test _should_retry returns True for retryable exceptions."""
        policy = RetryPolicy(
            name="test",
            retryable_exceptions=[ValueError, TypeError]
        )

        assert policy._should_retry(ValueError("test")) is True
        assert policy._should_retry(TypeError("test")) is True

    def test_should_retry_non_retryable_exception(self):
        """Test _should_retry returns False for non-retryable exceptions."""
        policy = RetryPolicy(
            name="test",
            retryable_exceptions=[Exception],
            non_retryable_exceptions=[KeyError]
        )

        assert policy._should_retry(KeyError("test")) is False

    def test_should_retry_unknown_exception(self):
        """Test _should_retry returns False for unknown exceptions."""
        policy = RetryPolicy(
            name="test",
            retryable_exceptions=[ValueError]
        )

        assert policy._should_retry(TypeError("test")) is False

    def test_non_retryable_exception_raised_immediately(self):
        """Test non-retryable exceptions are raised immediately."""
        policy = RetryPolicy(
            name="test",
            max_attempts=3,
            non_retryable_exceptions=[KeyError],
            strategy=FixedDelay(delay=0.01)
        )

        def raises_key_error():
            raise KeyError("Non-retryable")

        with pytest.raises(KeyError):
            policy.execute(raises_key_error)

        # Should only try once
        assert policy._stats.total_attempts == 1

    def test_get_stats(self):
        """Test get_stats returns stats object."""
        policy = RetryPolicy(name="test")

        stats = policy.get_stats()

        assert isinstance(stats, RetryStats)

    def test_reset_stats(self):
        """Test reset_stats clears statistics."""
        policy = RetryPolicy(name="test")
        policy._stats.total_attempts = 10
        policy._stats.successful_retries = 5

        policy.reset_stats()

        assert policy._stats.total_attempts == 0
        assert policy._stats.successful_retries == 0

    def test_get_health(self):
        """Test get_health returns health dictionary."""
        policy = RetryPolicy(name="test_policy", max_attempts=5)
        policy._stats.total_attempts = 10
        policy._stats.successful_retries = 8
        policy._stats.failed_retries = 2
        policy._stats.total_delay_time = 15.5

        health = policy.get_health()

        assert health["name"] == "test_policy"
        assert health["max_attempts"] == 5
        assert health["total_attempts"] == 10
        assert health["successful_retries"] == 8
        assert health["success_rate"] == 0.8
        assert health["total_delay_time"] == 15.5
        assert health["avg_delay_time"] == 1.55

    def test_get_health_zero_attempts(self):
        """Test get_health with zero attempts."""
        policy = RetryPolicy(name="test")

        health = policy.get_health()

        assert health["success_rate"] == 0
        assert health["avg_delay_time"] == 0

    def test_get_health_with_last_retry_time(self):
        """Test get_health includes last_retry_time."""
        policy = RetryPolicy(name="test")
        policy._stats.last_retry_time = datetime(2024, 1, 1, 12, 0, 0)

        health = policy.get_health()

        assert health["last_retry"] is not None
        assert "2024-01-01" in health["last_retry"]

    def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        callback_calls = []

        def on_retry_callback(exc, attempt):
            callback_calls.append((str(exc), attempt))

        policy = RetryPolicy(
            name="test",
            max_attempts=3,
            strategy=FixedDelay(delay=0.01),
            on_retry=on_retry_callback
        )

        call_count = 0

        def sometimes_fails():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Fail")
            return "success"

        policy.execute(sometimes_fails)

        assert len(callback_calls) == 2
        assert callback_calls[0][1] == 1
        assert callback_calls[1][1] == 2

    def test_metrics_callback(self):
        """Test metrics_callback is called."""
        metrics_calls = []

        def metrics_callback(metrics):
            metrics_calls.append(metrics)

        policy = RetryPolicy(
            name="test",
            max_attempts=3,
            strategy=FixedDelay(delay=0.01),
            metrics_callback=metrics_callback
        )

        def always_succeeds():
            return "success"

        policy.execute(always_succeeds)

        assert len(metrics_calls) >= 1
        assert metrics_calls[0]["event"] == "success"

    def test_error_distribution_tracking(self):
        """Test error distribution is tracked."""
        policy = RetryPolicy(
            name="test",
            max_attempts=5,
            strategy=FixedDelay(delay=0.01)
        )

        call_count = 0

        def varies_errors():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Val error")
            elif call_count == 2:
                raise TypeError("Type error")
            elif call_count == 3:
                raise ValueError("Val error 2")
            return "success"

        policy.execute(varies_errors)

        assert policy._stats.error_distribution.get("ValueError", 0) == 2
        assert policy._stats.error_distribution.get("TypeError", 0) == 1


class TestRetryDecorator:
    """Tests for retry decorator."""

    def test_retry_decorator_success(self):
        """Test retry decorator allows success."""
        @retry(max_attempts=3, strategy=FixedDelay(delay=0.01))
        def decorated_func():
            return "result"

        result = decorated_func()

        assert result == "result"

    def test_retry_decorator_with_retry(self):
        """Test retry decorator retries on failure."""
        call_count = 0

        @retry(max_attempts=3, strategy=FixedDelay(delay=0.01))
        def decorated_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Fail")
            return "success"

        result = decorated_func()

        assert result == "success"
        assert call_count == 2

    def test_retry_decorator_exhausted(self):
        """Test retry decorator raises after exhaustion."""
        @retry(max_attempts=2, strategy=FixedDelay(delay=0.01))
        def decorated_func():
            raise ValueError("Always fails")

        with pytest.raises(RetryError):
            decorated_func()

    def test_retry_decorator_policy_accessible(self):
        """Test decorated function has retry_policy attribute."""
        @retry(max_attempts=5)
        def decorated_func():
            pass

        assert hasattr(decorated_func, "retry_policy")
        assert decorated_func.retry_policy.max_attempts == 5


class TestAdaptiveRetry:
    """Tests for AdaptiveRetry class."""

    def test_adaptive_retry_init(self):
        """Test AdaptiveRetry initialization."""
        adaptive = AdaptiveRetry(
            name="adaptive_test",
            initial_max_attempts=3,
            min_success_rate=0.5,
            adjustment_window=100
        )

        assert adaptive.max_attempts == 3
        assert adaptive.min_success_rate == 0.5
        assert adaptive.adjustment_window == 100

    def test_adaptive_retry_execute(self):
        """Test AdaptiveRetry execute works like parent."""
        adaptive = AdaptiveRetry(
            name="adaptive_test",
            initial_max_attempts=3,
            strategy=FixedDelay(delay=0.01)
        )

        def success_func():
            return "result"

        result = adaptive.execute(success_func)

        assert result == "result"

    def test_adaptive_retry_adjust_increase(self):
        """Test adaptive adjustment increases attempts on low success rate."""
        adaptive = AdaptiveRetry(
            name="adaptive_test",
            initial_max_attempts=3,
            min_success_rate=0.5,
            adjustment_window=1,  # Adjust after every call
            strategy=FixedDelay(delay=0.01)
        )

        # Simulate low success rate scenario
        adaptive._stats.total_attempts = 10
        adaptive._stats.successful_retries = 3  # 30% success rate

        adaptive._adjust_parameters()

        assert adaptive.max_attempts == 4  # Increased from 3

    def test_adaptive_retry_adjust_decrease(self):
        """Test adaptive adjustment decreases attempts on high success rate."""
        adaptive = AdaptiveRetry(
            name="adaptive_test",
            initial_max_attempts=5,
            min_success_rate=0.5,
            adjustment_window=1,
            strategy=FixedDelay(delay=0.01)
        )

        # Simulate high success rate scenario
        adaptive._stats.total_attempts = 10
        adaptive._stats.successful_retries = 9  # 90% success rate

        adaptive._adjust_parameters()

        assert adaptive.max_attempts == 4  # Decreased from 5

    def test_adaptive_retry_adjust_no_change(self):
        """Test adaptive adjustment no change for moderate success rate."""
        adaptive = AdaptiveRetry(
            name="adaptive_test",
            initial_max_attempts=3,
            min_success_rate=0.5,
            adjustment_window=1,
            strategy=FixedDelay(delay=0.01)
        )

        # Simulate moderate success rate
        adaptive._stats.total_attempts = 10
        adaptive._stats.successful_retries = 7  # 70% success rate

        adaptive._adjust_parameters()

        # No change for moderate rate between min_success_rate and 0.8
        assert adaptive.max_attempts == 3

    def test_adaptive_retry_max_attempts_cap(self):
        """Test adaptive adjustment respects max cap of 10."""
        adaptive = AdaptiveRetry(
            name="adaptive_test",
            initial_max_attempts=10,
            min_success_rate=0.5,
            adjustment_window=1,
            strategy=FixedDelay(delay=0.01)
        )

        adaptive._stats.total_attempts = 10
        adaptive._stats.successful_retries = 2  # 20% success rate

        adaptive._adjust_parameters()

        assert adaptive.max_attempts == 10  # Capped at 10

    def test_adaptive_retry_min_attempts_cap(self):
        """Test adaptive adjustment respects min cap of 1."""
        adaptive = AdaptiveRetry(
            name="adaptive_test",
            initial_max_attempts=1,
            min_success_rate=0.5,
            adjustment_window=1,
            strategy=FixedDelay(delay=0.01)
        )

        adaptive._stats.total_attempts = 10
        adaptive._stats.successful_retries = 9  # 90% success rate

        adaptive._adjust_parameters()

        assert adaptive.max_attempts == 1  # Capped at 1

    def test_adaptive_retry_no_adjust_zero_attempts(self):
        """Test adaptive adjustment does nothing with zero attempts."""
        adaptive = AdaptiveRetry(
            name="adaptive_test",
            initial_max_attempts=3,
            min_success_rate=0.5,
            adjustment_window=1,
            strategy=FixedDelay(delay=0.01)
        )

        initial_attempts = adaptive.max_attempts
        adaptive._adjust_parameters()

        assert adaptive.max_attempts == initial_attempts


class TestRetryError:
    """Tests for RetryError exception."""

    def test_retry_error_message(self):
        """Test RetryError stores message."""
        error = RetryError("Custom message")

        assert str(error) == "Custom message"

    def test_retry_error_inheritance(self):
        """Test RetryError is an Exception."""
        error = RetryError("Test")

        assert isinstance(error, Exception)
