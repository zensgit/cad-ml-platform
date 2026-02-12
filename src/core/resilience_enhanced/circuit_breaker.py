"""Enhanced Circuit Breaker.

Provides advanced circuit breaker patterns:
- State machine with half-open state
- Failure counting strategies
- Configurable thresholds
- Health monitoring
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, rejecting calls
    HALF_OPEN = "half_open" # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    # Failure threshold
    failure_threshold: int = 5
    failure_rate_threshold: float = 0.5  # 50%

    # Recovery
    recovery_timeout: float = 60.0  # seconds
    half_open_max_calls: int = 3

    # Counting window
    sliding_window_size: int = 10
    sliding_window_type: str = "count"  # "count" or "time"
    sliding_window_time: float = 60.0  # seconds if type is "time"

    # Exceptions to count
    record_exceptions: Set[Type[Exception]] = field(default_factory=lambda: {Exception})
    ignore_exceptions: Set[Type[Exception]] = field(default_factory=set)

    # Slow calls
    slow_call_duration: float = 5.0  # seconds
    slow_call_rate_threshold: float = 0.5


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker state."""
    state: CircuitState
    failure_count: int
    success_count: int
    slow_call_count: int
    total_calls: int
    failure_rate: float
    slow_call_rate: float
    last_failure_time: Optional[float]
    last_state_change: float
    consecutive_successes: int


class FailureCounter(ABC):
    """Base class for failure counting strategies."""

    @abstractmethod
    def record_success(self, duration: float) -> None:
        """Record a successful call."""
        pass

    @abstractmethod
    def record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        pass

    @abstractmethod
    def get_failure_rate(self) -> float:
        """Get current failure rate (0.0 to 1.0)."""
        pass

    @abstractmethod
    def get_slow_call_rate(self) -> float:
        """Get current slow call rate."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset counters."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        pass


class CountBasedFailureCounter(FailureCounter):
    """Count-based sliding window failure counter."""

    def __init__(self, window_size: int, slow_call_threshold: float):
        self.window_size = window_size
        self.slow_call_threshold = slow_call_threshold
        self._results: List[Dict[str, Any]] = []
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def record_success(self, duration: float) -> None:
        self._results.append({
            "success": True,
            "slow": duration >= self.slow_call_threshold,
            "time": time.time(),
        })
        if len(self._results) > self.window_size:
            self._results.pop(0)

    def record_failure(self, exception: Exception) -> None:
        self._results.append({
            "success": False,
            "slow": False,
            "exception": type(exception).__name__,
            "time": time.time(),
        })
        if len(self._results) > self.window_size:
            self._results.pop(0)

    def get_failure_rate(self) -> float:
        if not self._results:
            return 0.0
        failures = sum(1 for r in self._results if not r["success"])
        return failures / len(self._results)

    def get_slow_call_rate(self) -> float:
        if not self._results:
            return 0.0
        slow = sum(1 for r in self._results if r.get("slow", False))
        return slow / len(self._results)

    def reset(self) -> None:
        self._results.clear()

    def get_metrics(self) -> Dict[str, Any]:
        total = len(self._results)
        if total == 0:
            return {
                "total_calls": 0,
                "success_count": 0,
                "failure_count": 0,
                "slow_call_count": 0,
                "failure_rate": 0.0,
                "slow_call_rate": 0.0,
            }

        success = sum(1 for r in self._results if r["success"])
        slow = sum(1 for r in self._results if r.get("slow", False))
        failures = total - success

        return {
            "total_calls": total,
            "success_count": success,
            "failure_count": failures,
            "slow_call_count": slow,
            "failure_rate": failures / total,
            "slow_call_rate": slow / total,
        }


class TimeBasedFailureCounter(FailureCounter):
    """Time-based sliding window failure counter."""

    def __init__(self, window_seconds: float, slow_call_threshold: float):
        self.window_seconds = window_seconds
        self.slow_call_threshold = slow_call_threshold
        self._results: List[Dict[str, Any]] = []

    def _cleanup(self) -> None:
        """Remove expired entries."""
        cutoff = time.time() - self.window_seconds
        self._results = [r for r in self._results if r["time"] > cutoff]

    def record_success(self, duration: float) -> None:
        self._cleanup()
        self._results.append({
            "success": True,
            "slow": duration >= self.slow_call_threshold,
            "time": time.time(),
        })

    def record_failure(self, exception: Exception) -> None:
        self._cleanup()
        self._results.append({
            "success": False,
            "slow": False,
            "exception": type(exception).__name__,
            "time": time.time(),
        })

    def get_failure_rate(self) -> float:
        self._cleanup()
        if not self._results:
            return 0.0
        failures = sum(1 for r in self._results if not r["success"])
        return failures / len(self._results)

    def get_slow_call_rate(self) -> float:
        self._cleanup()
        if not self._results:
            return 0.0
        slow = sum(1 for r in self._results if r.get("slow", False))
        return slow / len(self._results)

    def reset(self) -> None:
        self._results.clear()

    def get_metrics(self) -> Dict[str, Any]:
        self._cleanup()
        total = len(self._results)
        if total == 0:
            return {
                "total_calls": 0,
                "success_count": 0,
                "failure_count": 0,
                "slow_call_count": 0,
                "failure_rate": 0.0,
                "slow_call_rate": 0.0,
            }

        success = sum(1 for r in self._results if r["success"])
        slow = sum(1 for r in self._results if r.get("slow", False))
        failures = total - success

        return {
            "total_calls": total,
            "success_count": success,
            "failure_count": failures,
            "slow_call_count": slow,
            "failure_rate": failures / total,
            "slow_call_rate": slow / total,
        }


class CircuitBreakerError(Exception):
    """Raised when circuit breaker rejects a call."""

    def __init__(self, state: CircuitState, message: str = ""):
        self.state = state
        self.message = message or f"Circuit breaker is {state.value}"
        super().__init__(self.message)


class CircuitBreaker:
    """Enhanced circuit breaker with advanced features."""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._state_changed_at = time.time()
        self._half_open_calls = 0
        self._consecutive_successes = 0
        self._last_failure_time: Optional[float] = None

        # Create failure counter
        if self.config.sliding_window_type == "time":
            self._counter: FailureCounter = TimeBasedFailureCounter(
                self.config.sliding_window_time,
                self.config.slow_call_duration,
            )
        else:
            self._counter = CountBasedFailureCounter(
                self.config.sliding_window_size,
                self.config.slow_call_duration,
            )

        # Listeners
        self._listeners: List[Callable[[CircuitState, CircuitState], None]] = []

    @property
    def state(self) -> CircuitState:
        """Get current state, transitioning if needed."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._state_changed_at >= self.config.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        self._state_changed_at = time.time()

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._consecutive_successes = 0
        elif new_state == CircuitState.CLOSED:
            self._counter.reset()

        logger.info(f"Circuit breaker '{self.name}' transitioned from {old_state.value} to {new_state.value}")

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(old_state, new_state)
            except Exception as e:
                logger.error(f"Listener error: {e}")

    def _should_record_exception(self, exception: Exception) -> bool:
        """Check if exception should be recorded as failure."""
        # Check ignore list first
        for exc_type in self.config.ignore_exceptions:
            if isinstance(exception, exc_type):
                return False

        # Check record list
        for exc_type in self.config.record_exceptions:
            if isinstance(exception, exc_type):
                return True

        return False

    async def __aenter__(self) -> "CircuitBreaker":
        """Async context manager entry."""
        if not self.allow_request():
            raise CircuitBreakerError(self.state)
        self._start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Async context manager exit."""
        duration = time.time() - self._start_time

        if exc_val is None:
            self.record_success(duration)
        elif self._should_record_exception(exc_val):
            self.record_failure(exc_val)
        # Don't suppress the exception
        return False

    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        state = self.state  # This may trigger state transition

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.OPEN:
            return False

        if state == CircuitState.HALF_OPEN:
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

        return False

    def record_success(self, duration: float = 0.0) -> None:
        """Record a successful call."""
        state = self.state

        if state == CircuitState.HALF_OPEN:
            self._consecutive_successes += 1
            if self._consecutive_successes >= self.config.half_open_max_calls:
                self._transition_to(CircuitState.CLOSED)
        elif state == CircuitState.CLOSED:
            self._counter.record_success(duration)
            self._check_thresholds()

    def record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        self._last_failure_time = time.time()
        state = self.state

        if state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
        elif state == CircuitState.CLOSED:
            self._counter.record_failure(exception)
            self._check_thresholds()

    def _check_thresholds(self) -> None:
        """Check if thresholds are exceeded."""
        metrics = self._counter.get_metrics()

        # Check minimum calls
        if metrics["total_calls"] < self.config.sliding_window_size:
            return

        # Check failure rate
        if metrics["failure_rate"] >= self.config.failure_rate_threshold:
            logger.warning(
                f"Circuit breaker '{self.name}' opening: "
                f"failure rate {metrics['failure_rate']:.2%} >= {self.config.failure_rate_threshold:.2%}"
            )
            self._transition_to(CircuitState.OPEN)
            return

        # Check slow call rate
        if metrics["slow_call_rate"] >= self.config.slow_call_rate_threshold:
            logger.warning(
                f"Circuit breaker '{self.name}' opening: "
                f"slow call rate {metrics['slow_call_rate']:.2%} >= {self.config.slow_call_rate_threshold:.2%}"
            )
            self._transition_to(CircuitState.OPEN)

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get current metrics."""
        counter_metrics = self._counter.get_metrics()

        return CircuitBreakerMetrics(
            state=self.state,
            failure_count=counter_metrics["failure_count"],
            success_count=counter_metrics["success_count"],
            slow_call_count=counter_metrics["slow_call_count"],
            total_calls=counter_metrics["total_calls"],
            failure_rate=counter_metrics["failure_rate"],
            slow_call_rate=counter_metrics["slow_call_rate"],
            last_failure_time=self._last_failure_time,
            last_state_change=self._state_changed_at,
            consecutive_successes=self._consecutive_successes,
        )

    def add_listener(self, listener: Callable[[CircuitState, CircuitState], None]) -> None:
        """Add state change listener."""
        self._listeners.append(listener)

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._transition_to(CircuitState.CLOSED)
        self._counter.reset()
        self._consecutive_successes = 0
        self._last_failure_time = None


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._default_config = CircuitBreakerConfig()

    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name,
                config or self._default_config,
            )
        return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._breakers.get(name)

    def get_all_metrics(self) -> Dict[str, CircuitBreakerMetrics]:
        """Get metrics for all circuit breakers."""
        return {
            name: breaker.get_metrics()
            for name, breaker in self._breakers.items()
        }

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()


# Global registry
_registry: Optional[CircuitBreakerRegistry] = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    global _registry
    if _registry is None:
        _registry = CircuitBreakerRegistry()
    return _registry
