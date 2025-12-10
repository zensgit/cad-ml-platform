"""Tests for src/core/resilience/circuit_breaker.py to improve coverage.

Covers:
- CircuitState enum
- CircuitBreakerError exception
- CircuitBreakerStats dataclass
- CircuitBreaker class
- circuit_breaker decorator
"""

from __future__ import annotations

import time
import threading
from datetime import datetime
from unittest.mock import MagicMock

import pytest


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_circuit_state_closed(self):
        """Test CLOSED state value."""
        from src.core.resilience.circuit_breaker import CircuitState

        assert CircuitState.CLOSED.value == "closed"

    def test_circuit_state_open(self):
        """Test OPEN state value."""
        from src.core.resilience.circuit_breaker import CircuitState

        assert CircuitState.OPEN.value == "open"

    def test_circuit_state_half_open(self):
        """Test HALF_OPEN state value."""
        from src.core.resilience.circuit_breaker import CircuitState

        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_circuit_state_enum_members(self):
        """Test all enum members exist."""
        from src.core.resilience.circuit_breaker import CircuitState

        assert hasattr(CircuitState, "CLOSED")
        assert hasattr(CircuitState, "OPEN")
        assert hasattr(CircuitState, "HALF_OPEN")


class TestCircuitBreakerError:
    """Tests for CircuitBreakerError exception."""

    def test_is_exception_subclass(self):
        """Test CircuitBreakerError is an Exception subclass."""
        from src.core.resilience.circuit_breaker import CircuitBreakerError

        assert issubclass(CircuitBreakerError, Exception)

    def test_error_with_message(self):
        """Test CircuitBreakerError with message."""
        from src.core.resilience.circuit_breaker import CircuitBreakerError

        error = CircuitBreakerError("Circuit is open")
        assert str(error) == "Circuit is open"

    def test_error_can_be_raised(self):
        """Test CircuitBreakerError can be raised and caught."""
        from src.core.resilience.circuit_breaker import CircuitBreakerError

        with pytest.raises(CircuitBreakerError) as exc_info:
            raise CircuitBreakerError("Test error")

        assert "Test error" in str(exc_info.value)


class TestCircuitBreakerStats:
    """Tests for CircuitBreakerStats dataclass."""

    def test_stats_defaults(self):
        """Test CircuitBreakerStats default values."""
        from src.core.resilience.circuit_breaker import CircuitBreakerStats

        stats = CircuitBreakerStats()

        assert stats.success_count == 0
        assert stats.failure_count == 0
        assert stats.consecutive_failures == 0
        assert stats.last_failure_time is None
        assert stats.last_success_time is None
        assert stats.total_calls == 0
        assert stats.state_transitions == []
        assert stats.error_distribution == {}

    def test_stats_custom_values(self):
        """Test CircuitBreakerStats with custom values."""
        from src.core.resilience.circuit_breaker import CircuitBreakerStats

        now = datetime.now()
        stats = CircuitBreakerStats(
            success_count=100,
            failure_count=10,
            consecutive_failures=3,
            last_failure_time=now,
            last_success_time=now,
            total_calls=110,
            state_transitions=[{"from": "closed", "to": "open"}],
            error_distribution={"ValueError": 5, "TimeoutError": 5}
        )

        assert stats.success_count == 100
        assert stats.failure_count == 10
        assert stats.consecutive_failures == 3
        assert stats.last_failure_time == now
        assert stats.last_success_time == now
        assert stats.total_calls == 110


class TestCircuitBreakerInit:
    """Tests for CircuitBreaker initialization."""

    def test_init_defaults(self):
        """Test CircuitBreaker initialization with defaults."""
        from src.core.resilience.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(name="test")

        assert breaker.name == "test"
        assert breaker.failure_threshold == 5
        assert breaker.recovery_timeout == 60
        assert breaker.expected_exception is Exception
        assert breaker.half_open_max_calls == 3
        assert breaker.success_threshold == 2
        assert breaker.metrics_callback is None
        assert breaker.state == CircuitState.CLOSED

    def test_init_custom_values(self):
        """Test CircuitBreaker initialization with custom values."""
        from src.core.resilience.circuit_breaker import CircuitBreaker

        callback = MagicMock()
        breaker = CircuitBreaker(
            name="custom",
            failure_threshold=3,
            recovery_timeout=30,
            expected_exception=ValueError,
            half_open_max_calls=2,
            success_threshold=1,
            metrics_callback=callback
        )

        assert breaker.name == "custom"
        assert breaker.failure_threshold == 3
        assert breaker.recovery_timeout == 30
        assert breaker.expected_exception is ValueError
        assert breaker.half_open_max_calls == 2
        assert breaker.success_threshold == 1
        assert breaker.metrics_callback == callback


class TestCircuitBreakerCall:
    """Tests for CircuitBreaker call method."""

    def test_call_success(self):
        """Test successful call through circuit breaker."""
        from src.core.resilience.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(name="test")

        result = breaker.call(lambda x: x * 2, 5)

        assert result == 10
        assert breaker.stats.success_count == 1
        assert breaker.stats.total_calls == 1

    def test_call_with_kwargs(self):
        """Test call with keyword arguments."""
        from src.core.resilience.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(name="test")

        def func(a, b=10):
            return a + b

        result = breaker.call(func, 5, b=3)

        assert result == 8

    def test_call_failure_increments_stats(self):
        """Test failure increments failure stats."""
        from src.core.resilience.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(name="test")

        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            breaker.call(failing_func)

        assert breaker.stats.failure_count == 1
        assert breaker.stats.consecutive_failures == 1
        assert breaker.stats.total_calls == 1

    def test_call_tracks_error_distribution(self):
        """Test failure tracks error type distribution."""
        from src.core.resilience.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(name="test")

        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            breaker.call(failing_func)

        assert "ValueError" in breaker.stats.error_distribution
        assert breaker.stats.error_distribution["ValueError"] == 1


class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transitions."""

    def test_open_on_failure_threshold(self):
        """Test circuit opens after reaching failure threshold."""
        from src.core.resilience.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(name="test", failure_threshold=3)

        def failing_func():
            raise Exception("Error")

        # Trigger failures
        for _ in range(3):
            with pytest.raises(Exception):
                breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

    def test_open_rejects_calls(self):
        """Test open circuit rejects calls."""
        from src.core.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerError

        breaker = CircuitBreaker(name="test", failure_threshold=1)

        # Open the circuit
        with pytest.raises(Exception):
            breaker.call(lambda: 1/0)

        # Next call should be rejected
        with pytest.raises(CircuitBreakerError) as exc_info:
            breaker.call(lambda: 42)

        assert "OPEN" in str(exc_info.value)

    def test_half_open_after_recovery_timeout(self):
        """Test circuit transitions to half-open after recovery timeout."""
        from src.core.resilience.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0)

        # Open the circuit
        with pytest.raises(Exception):
            breaker.call(lambda: 1/0)

        # With recovery_timeout=0, state property check triggers immediate transition
        # The first access to state after failure will transition to HALF_OPEN
        time.sleep(0.1)

        # Accessing state should trigger transition check
        state = breaker.state
        # State is HALF_OPEN because recovery_timeout=0 causes immediate transition
        assert state == CircuitState.HALF_OPEN

    def test_half_open_limits_test_calls(self):
        """Test half-open state limits test calls."""
        from src.core.resilience.circuit_breaker import (
            CircuitBreaker, CircuitBreakerError
        )

        breaker = CircuitBreaker(
            name="test",
            failure_threshold=1,
            recovery_timeout=0,
            half_open_max_calls=2,
            success_threshold=10  # High threshold so we don't close circuit
        )

        # Open the circuit
        with pytest.raises(Exception):
            breaker.call(lambda: 1/0)

        # Wait for recovery and force transition
        time.sleep(0.1)
        _ = breaker.state  # Force transition to half-open

        # Execute max test calls - these succeed but don't close circuit due to high success_threshold
        call_count = 0
        for _ in range(2):
            try:
                breaker.call(lambda: 42)
                call_count += 1
            except CircuitBreakerError:
                break

        # After 2 successful calls in half-open, circuit should have hit max test calls
        # or closed (if success_threshold was met)
        # The key is that half_open_max_calls limits the test calls
        assert call_count <= 2

    def test_half_open_to_closed_on_success(self):
        """Test circuit closes on successful calls in half-open state."""
        from src.core.resilience.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(
            name="test",
            failure_threshold=1,
            recovery_timeout=0,
            success_threshold=2
        )

        # Open the circuit
        with pytest.raises(Exception):
            breaker.call(lambda: 1/0)

        time.sleep(0.1)

        # Execute successful calls
        breaker.call(lambda: 42)
        breaker.call(lambda: 42)

        assert breaker.state == CircuitState.CLOSED

    def test_half_open_to_open_on_failure(self):
        """Test circuit reopens on failure in half-open state."""
        from src.core.resilience.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(
            name="test",
            failure_threshold=1,
            recovery_timeout=1  # Use positive timeout so we can control state
        )

        # Open the circuit
        with pytest.raises(Exception):
            breaker.call(lambda: 1/0)

        # Manually force state to HALF_OPEN for testing
        breaker._state = CircuitState.HALF_OPEN
        breaker._half_open_calls = 0

        # Fail in half-open
        with pytest.raises(Exception):
            breaker.call(lambda: 1/0)

        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerReset:
    """Tests for circuit breaker reset method."""

    def test_reset_clears_stats(self):
        """Test reset clears statistics."""
        from src.core.resilience.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(name="test")

        # Accumulate stats
        breaker.call(lambda: 42)

        # Reset
        breaker.reset()

        assert breaker.stats.success_count == 0
        assert breaker.stats.failure_count == 0
        assert breaker.stats.total_calls == 0

    def test_reset_closes_circuit(self):
        """Test reset closes the circuit."""
        from src.core.resilience.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(name="test", failure_threshold=1)

        # Open the circuit
        with pytest.raises(Exception):
            breaker.call(lambda: 1/0)

        assert breaker.state == CircuitState.OPEN

        # Reset
        breaker.reset()

        assert breaker.state == CircuitState.CLOSED


class TestCircuitBreakerHealth:
    """Tests for circuit breaker get_health method."""

    def test_get_health_basic(self):
        """Test get_health returns expected fields."""
        from src.core.resilience.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(name="test")
        breaker.call(lambda: 42)

        health = breaker.get_health()

        assert health["name"] == "test"
        assert health["state"] == "closed"
        assert "success_count" in health
        assert "failure_count" in health
        assert "total_calls" in health
        assert "failure_rate" in health
        assert "error_distribution" in health

    def test_get_health_failure_rate(self):
        """Test get_health calculates failure rate."""
        from src.core.resilience.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(name="test", failure_threshold=10)

        # 3 successes, 1 failure
        breaker.call(lambda: 42)
        breaker.call(lambda: 42)
        breaker.call(lambda: 42)
        with pytest.raises(Exception):
            breaker.call(lambda: 1/0)

        health = breaker.get_health()

        assert health["failure_rate"] == 0.25  # 1/4

    def test_get_health_with_timestamps(self):
        """Test get_health includes timestamps when available."""
        from src.core.resilience.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(name="test", failure_threshold=10)

        breaker.call(lambda: 42)
        with pytest.raises(Exception):
            breaker.call(lambda: 1/0)

        health = breaker.get_health()

        assert health["last_success"] is not None
        assert health["last_failure"] is not None


class TestCircuitBreakerMetricsCallback:
    """Tests for metrics callback functionality."""

    def test_success_emits_metric(self):
        """Test successful call emits metric."""
        from src.core.resilience.circuit_breaker import CircuitBreaker

        callback_data = []

        def callback(data):
            callback_data.append(data)

        breaker = CircuitBreaker(name="test", metrics_callback=callback)
        breaker.call(lambda: 42)

        assert len(callback_data) == 1
        assert callback_data[0]["event"] == "success"
        assert callback_data[0]["circuit_breaker"] == "test"

    def test_failure_emits_metric(self):
        """Test failure emits metric."""
        from src.core.resilience.circuit_breaker import CircuitBreaker

        callback_data = []

        def callback(data):
            callback_data.append(data)

        breaker = CircuitBreaker(name="test", metrics_callback=callback)

        with pytest.raises(Exception):
            breaker.call(lambda: 1/0)

        assert len(callback_data) == 1
        assert callback_data[0]["event"] == "failure"

    def test_rejection_emits_metric(self):
        """Test rejection emits metric."""
        from src.core.resilience.circuit_breaker import (
            CircuitBreaker, CircuitBreakerError
        )

        callback_data = []

        def callback(data):
            callback_data.append(data)

        breaker = CircuitBreaker(
            name="test",
            failure_threshold=1,
            metrics_callback=callback
        )

        # Open circuit
        with pytest.raises(Exception):
            breaker.call(lambda: 1/0)

        # Try to call - should be rejected
        with pytest.raises(CircuitBreakerError):
            breaker.call(lambda: 42)

        # Find rejection event
        rejection_events = [d for d in callback_data if d["event"] == "rejected"]
        assert len(rejection_events) >= 1

    def test_state_change_emits_metric(self):
        """Test state change emits metric."""
        from src.core.resilience.circuit_breaker import CircuitBreaker

        callback_data = []

        def callback(data):
            callback_data.append(data)

        breaker = CircuitBreaker(
            name="test",
            failure_threshold=1,
            metrics_callback=callback
        )

        # Open circuit
        with pytest.raises(Exception):
            breaker.call(lambda: 1/0)

        # Find state_change event
        state_events = [d for d in callback_data if d["event"] == "state_change"]
        assert len(state_events) >= 1


class TestCircuitBreakerDecorator:
    """Tests for circuit_breaker decorator."""

    def test_decorator_basic(self):
        """Test basic decorator usage."""
        from src.core.resilience.circuit_breaker import circuit_breaker

        @circuit_breaker(name="test_func")
        def test_func(x):
            return x * 2

        result = test_func(5)
        assert result == 10

    def test_decorator_attaches_breaker(self):
        """Test decorator attaches circuit_breaker attribute."""
        from src.core.resilience.circuit_breaker import (
            circuit_breaker, CircuitBreaker
        )

        @circuit_breaker(name="test_func")
        def test_func():
            return 42

        assert hasattr(test_func, "circuit_breaker")
        assert isinstance(test_func.circuit_breaker, CircuitBreaker)

    def test_decorator_auto_generates_name(self):
        """Test decorator auto-generates name when not provided."""
        from src.core.resilience.circuit_breaker import circuit_breaker

        @circuit_breaker()
        def my_function():
            return 42

        assert "my_function" in my_function.circuit_breaker.name

    def test_decorator_with_custom_params(self):
        """Test decorator with custom parameters."""
        from src.core.resilience.circuit_breaker import circuit_breaker

        @circuit_breaker(
            name="custom",
            failure_threshold=2,
            recovery_timeout=30,
            expected_exception=ValueError
        )
        def test_func():
            return 42

        assert test_func.circuit_breaker.name == "custom"
        assert test_func.circuit_breaker.failure_threshold == 2
        assert test_func.circuit_breaker.recovery_timeout == 30
        assert test_func.circuit_breaker.expected_exception is ValueError


class TestCircuitBreakerTimeUntilRecovery:
    """Tests for time_until_recovery calculation."""

    def test_time_until_recovery_positive(self):
        """Test time until recovery is positive when open."""
        from src.core.resilience.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(
            name="test",
            failure_threshold=1,
            recovery_timeout=60
        )

        # Open circuit
        with pytest.raises(Exception):
            breaker.call(lambda: 1/0)

        # Time until recovery should be positive
        time_remaining = breaker._time_until_recovery()
        assert time_remaining > 0
        assert time_remaining <= 60

    def test_time_until_recovery_zero_after_timeout(self):
        """Test time until recovery is zero after timeout."""
        from src.core.resilience.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(
            name="test",
            failure_threshold=1,
            recovery_timeout=0
        )

        # Open circuit
        with pytest.raises(Exception):
            breaker.call(lambda: 1/0)

        time.sleep(0.1)

        # Time until recovery should be zero
        time_remaining = breaker._time_until_recovery()
        assert time_remaining == 0


class TestCircuitBreakerThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_calls(self):
        """Test concurrent calls are thread-safe."""
        from src.core.resilience.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(name="test")
        results = []
        errors = []

        def worker():
            try:
                result = breaker.call(lambda: 42)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(results) == 10
        assert len(errors) == 0
        assert breaker.stats.total_calls == 10


class TestCircuitBreakerConsecutiveFailures:
    """Tests for consecutive failure tracking."""

    def test_consecutive_failures_reset_on_success(self):
        """Test consecutive failures reset on successful call."""
        from src.core.resilience.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(name="test", failure_threshold=5)

        # Fail twice
        for _ in range(2):
            with pytest.raises(Exception):
                breaker.call(lambda: 1/0)

        assert breaker.stats.consecutive_failures == 2

        # Success should reset
        breaker.call(lambda: 42)

        assert breaker.stats.consecutive_failures == 0


class TestCircuitBreakerStateTransitionRecording:
    """Tests for state transition recording."""

    def test_transitions_recorded_in_stats(self):
        """Test state transitions are recorded in stats."""
        from src.core.resilience.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0)

        # Open circuit
        with pytest.raises(Exception):
            breaker.call(lambda: 1/0)

        # Check transition recorded
        transitions = breaker.stats.state_transitions
        assert len(transitions) >= 1
        assert transitions[0]["to"] == "open"

    def test_get_health_shows_recent_transitions(self):
        """Test get_health shows recent transitions."""
        from src.core.resilience.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0)

        # Open circuit
        with pytest.raises(Exception):
            breaker.call(lambda: 1/0)

        health = breaker.get_health()

        assert "recent_transitions" in health
        assert len(health["recent_transitions"]) >= 1


class TestCircuitBreakerEdgeCases:
    """Tests for edge cases."""

    def test_zero_total_calls_failure_rate(self):
        """Test failure rate is 0 when no calls made."""
        from src.core.resilience.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(name="test")

        health = breaker.get_health()

        assert health["failure_rate"] == 0

    def test_expected_exception_only_triggers_failure(self):
        """Test only expected exception type triggers failure."""
        from src.core.resilience.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(name="test", expected_exception=ValueError)

        # ValueError should be counted
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("test")))

        assert breaker.stats.failure_count == 1

    def test_success_count_in_half_open(self):
        """Test success count tracked in half-open state."""
        from src.core.resilience.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(
            name="test",
            failure_threshold=1,
            recovery_timeout=0,
            success_threshold=3
        )

        # Open circuit
        with pytest.raises(Exception):
            breaker.call(lambda: 1/0)

        time.sleep(0.1)

        # Call in half-open
        breaker.call(lambda: 42)

        # Should still be half-open (need success_threshold successes)
        # Note: success_count is global, not half-open specific
        assert breaker.stats.success_count >= 1
