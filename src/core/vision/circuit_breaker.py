"""Advanced circuit breaker module for Vision Provider system.

This module provides circuit breaker capabilities including:
- Multiple circuit breaker states (closed, open, half-open)
- Configurable failure thresholds
- Sliding window failure tracking
- Recovery strategies
- Circuit breaker events and callbacks
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from .base import VisionDescription, VisionProvider


class CircuitState(Enum):
    """State of the circuit breaker."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class FailureType(Enum):
    """Type of failure."""

    EXCEPTION = "exception"
    TIMEOUT = "timeout"
    REJECTION = "rejection"
    CUSTOM = "custom"


class RecoveryStrategy(Enum):
    """Strategy for circuit recovery."""

    FIXED = "fixed"  # Fixed timeout before half-open
    EXPONENTIAL = "exponential"  # Exponential backoff
    LINEAR = "linear"  # Linear increase
    ADAPTIVE = "adaptive"  # Based on success rate


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures to open circuit
    success_threshold: int = 3  # Successes to close circuit
    timeout_seconds: float = 30.0  # Time before half-open
    half_open_max_calls: int = 3  # Max calls in half-open
    sliding_window_size: int = 10  # Window for failure calculation
    failure_rate_threshold: float = 0.5  # Rate to open circuit
    slow_call_threshold_seconds: float = 5.0  # Slow call threshold
    slow_call_rate_threshold: float = 0.5  # Slow call rate to open
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.FIXED
    max_timeout_seconds: float = 300.0  # Max timeout for backoff


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    slow_calls: int = 0
    state_transitions: int = 0
    current_state: CircuitState = CircuitState.CLOSED
    last_state_change: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        total = self.successful_calls + self.failed_calls
        if total == 0:
            return 0.0
        return self.failed_calls / total

    @property
    def slow_call_rate(self) -> float:
        """Calculate slow call rate."""
        total = self.successful_calls + self.failed_calls
        if total == 0:
            return 0.0
        return self.slow_calls / total


@dataclass
class FailureRecord:
    """Record of a failure."""

    timestamp: datetime
    failure_type: FailureType
    error_message: str
    duration_ms: float


@dataclass
class CallRecord:
    """Record of a call."""

    timestamp: datetime
    success: bool
    duration_ms: float
    slow: bool = False


class SlidingWindow:
    """Sliding window for tracking calls."""

    def __init__(self, size: int = 10) -> None:
        """Initialize sliding window.

        Args:
            size: Window size
        """
        self._size = size
        self._records: List[CallRecord] = []
        self._lock = threading.Lock()

    def add(self, record: CallRecord) -> None:
        """Add a record to the window."""
        with self._lock:
            self._records.append(record)
            if len(self._records) > self._size:
                self._records.pop(0)

    def get_failure_rate(self) -> float:
        """Calculate failure rate in window."""
        with self._lock:
            if not self._records:
                return 0.0
            failures = sum(1 for r in self._records if not r.success)
            return failures / len(self._records)

    def get_slow_call_rate(self) -> float:
        """Calculate slow call rate in window."""
        with self._lock:
            if not self._records:
                return 0.0
            slow = sum(1 for r in self._records if r.slow)
            return slow / len(self._records)

    def clear(self) -> None:
        """Clear the window."""
        with self._lock:
            self._records.clear()

    @property
    def size(self) -> int:
        """Return current size."""
        with self._lock:
            return len(self._records)


class CircuitBreaker:
    """Advanced circuit breaker implementation."""

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            name: Circuit breaker name
            config: Configuration
        """
        self._name = name
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._window = SlidingWindow(self._config.sliding_window_size)
        self._half_open_calls = 0
        self._open_time: Optional[datetime] = None
        self._current_timeout = self._config.timeout_seconds
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[CircuitState, CircuitState], None]] = []

    @property
    def name(self) -> str:
        """Return circuit breaker name."""
        return self._name

    @property
    def state(self) -> CircuitState:
        """Return current state."""
        return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Return statistics."""
        return self._stats

    @property
    def config(self) -> CircuitBreakerConfig:
        """Return configuration."""
        return self._config

    def add_callback(
        self,
        callback: Callable[[CircuitState, CircuitState], None],
    ) -> None:
        """Add state change callback."""
        self._callbacks.append(callback)

    def can_execute(self) -> bool:
        """Check if execution is allowed.

        Returns:
            True if execution is allowed
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                if self._should_try_recovery():
                    self._transition_to(CircuitState.HALF_OPEN)
                    return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self._config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    def record_success(self, duration_ms: float) -> None:
        """Record a successful call.

        Args:
            duration_ms: Call duration in milliseconds
        """
        with self._lock:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success = datetime.now()

            slow = duration_ms > (self._config.slow_call_threshold_seconds * 1000)
            if slow:
                self._stats.slow_calls += 1

            self._window.add(
                CallRecord(
                    timestamp=datetime.now(),
                    success=True,
                    duration_ms=duration_ms,
                    slow=slow,
                )
            )

            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self._config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    self._reset_timeout()

    def record_failure(
        self,
        duration_ms: float,
        failure_type: FailureType = FailureType.EXCEPTION,
        error_message: str = "",
    ) -> None:
        """Record a failed call.

        Args:
            duration_ms: Call duration in milliseconds
            failure_type: Type of failure
            error_message: Error message
        """
        with self._lock:
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure = datetime.now()

            self._window.add(
                CallRecord(
                    timestamp=datetime.now(),
                    success=False,
                    duration_ms=duration_ms,
                )
            )

            if self._state == CircuitState.CLOSED:
                if self._should_open():
                    self._transition_to(CircuitState.OPEN)

            elif self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
                self._increase_timeout()

    def record_rejection(self) -> None:
        """Record a rejected call."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.rejected_calls += 1

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._stats = CircuitBreakerStats()
            self._window.clear()
            self._half_open_calls = 0
            self._open_time = None
            self._current_timeout = self._config.timeout_seconds

            if old_state != CircuitState.CLOSED:
                self._notify_callbacks(old_state, CircuitState.CLOSED)

    def force_open(self) -> None:
        """Force circuit to open state."""
        with self._lock:
            self._transition_to(CircuitState.OPEN)

    def force_closed(self) -> None:
        """Force circuit to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)

    def _should_open(self) -> bool:
        """Check if circuit should open."""
        if self._stats.consecutive_failures >= self._config.failure_threshold:
            return True

        if self._window.size >= self._config.sliding_window_size:
            if self._window.get_failure_rate() >= self._config.failure_rate_threshold:
                return True

            if self._window.get_slow_call_rate() >= self._config.slow_call_rate_threshold:
                return True

        return False

    def _should_try_recovery(self) -> bool:
        """Check if recovery should be attempted."""
        if not self._open_time:
            return True

        elapsed = (datetime.now() - self._open_time).total_seconds()
        return elapsed >= self._current_timeout

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state."""
        if self._state == new_state:
            return

        old_state = self._state
        self._state = new_state
        self._stats.current_state = new_state
        self._stats.state_transitions += 1
        self._stats.last_state_change = datetime.now()

        if new_state == CircuitState.OPEN:
            self._open_time = datetime.now()
            self._half_open_calls = 0

        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0

        elif new_state == CircuitState.CLOSED:
            self._window.clear()
            self._stats.consecutive_failures = 0

        self._notify_callbacks(old_state, new_state)

    def _notify_callbacks(
        self,
        old_state: CircuitState,
        new_state: CircuitState,
    ) -> None:
        """Notify callbacks of state change."""
        for callback in self._callbacks:
            try:
                callback(old_state, new_state)
            except Exception:
                pass

    def _increase_timeout(self) -> None:
        """Increase timeout based on recovery strategy."""
        strategy = self._config.recovery_strategy

        if strategy == RecoveryStrategy.EXPONENTIAL:
            self._current_timeout = min(
                self._current_timeout * 2,
                self._config.max_timeout_seconds,
            )
        elif strategy == RecoveryStrategy.LINEAR:
            self._current_timeout = min(
                self._current_timeout + self._config.timeout_seconds,
                self._config.max_timeout_seconds,
            )
        # FIXED and ADAPTIVE don't increase

    def _reset_timeout(self) -> None:
        """Reset timeout to initial value."""
        self._current_timeout = self._config.timeout_seconds


class CircuitBreakerError(Exception):
    """Error raised when circuit is open."""

    def __init__(self, circuit_name: str, state: CircuitState) -> None:
        """Initialize error."""
        self.circuit_name = circuit_name
        self.state = state
        super().__init__(f"Circuit breaker '{circuit_name}' is {state.value}")


class CircuitBreakerVisionProvider(VisionProvider):
    """Vision provider with circuit breaker protection."""

    def __init__(
        self,
        provider: VisionProvider,
        circuit_breaker: CircuitBreaker,
        fallback: Optional[Callable[[], VisionDescription]] = None,
    ) -> None:
        """Initialize circuit breaker provider.

        Args:
            provider: Underlying vision provider
            circuit_breaker: Circuit breaker
            fallback: Optional fallback function
        """
        self._provider = provider
        self._circuit_breaker = circuit_breaker
        self._fallback = fallback

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"circuit_breaker_{self._provider.provider_name}"

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Return circuit breaker."""
        return self._circuit_breaker

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with circuit breaker protection.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        if not self._circuit_breaker.can_execute():
            self._circuit_breaker.record_rejection()

            if self._fallback:
                return self._fallback()

            raise CircuitBreakerError(
                self._circuit_breaker.name,
                self._circuit_breaker.state,
            )

        start_time = time.time()

        try:
            result = await self._provider.analyze_image(
                image_data, include_description
            )
            duration_ms = (time.time() - start_time) * 1000
            self._circuit_breaker.record_success(duration_ms)
            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._circuit_breaker.record_failure(
                duration_ms,
                FailureType.EXCEPTION,
                str(e),
            )
            raise


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker.

    Args:
        name: Circuit breaker name
        config: Optional configuration

    Returns:
        CircuitBreaker instance
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def create_circuit_breaker_provider(
    provider: VisionProvider,
    name: Optional[str] = None,
    config: Optional[CircuitBreakerConfig] = None,
    fallback: Optional[Callable[[], VisionDescription]] = None,
) -> CircuitBreakerVisionProvider:
    """Create a circuit breaker provider.

    Args:
        provider: Underlying vision provider
        name: Circuit breaker name
        config: Circuit breaker configuration
        fallback: Optional fallback function

    Returns:
        CircuitBreakerVisionProvider instance
    """
    cb_name = name or f"cb_{provider.provider_name}"
    circuit_breaker = get_circuit_breaker(cb_name, config)

    return CircuitBreakerVisionProvider(
        provider=provider,
        circuit_breaker=circuit_breaker,
        fallback=fallback,
    )
