"""Circuit Breaker Module.

Provides fault tolerance:
- Circuit breaker pattern
- Fallback strategies
- Health tracking
- Automatic recovery
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5  # Failures to open circuit
    success_threshold: int = 3  # Successes to close from half-open
    timeout_seconds: float = 30.0  # Time before trying half-open
    half_open_max_calls: int = 3  # Max calls in half-open state

    # Failure detection
    failure_rate_threshold: float = 0.5  # 50% failure rate
    slow_call_threshold_seconds: float = 5.0  # Slow call threshold
    slow_call_rate_threshold: float = 0.5  # 50% slow call rate

    # Sliding window
    window_size: int = 10  # Number of calls to track
    min_calls: int = 5  # Minimum calls before calculating rate


@dataclass
class CircuitStats:
    """Circuit breaker statistics."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    slow_calls: int = 0

    consecutive_successes: int = 0
    consecutive_failures: int = 0

    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_state_change: Optional[datetime] = None

    state_history: List[tuple] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls

    @property
    def slow_call_rate(self) -> float:
        """Calculate slow call rate."""
        if self.total_calls == 0:
            return 0.0
        return self.slow_calls / self.total_calls

    def record_success(self, duration: float, slow_threshold: float) -> None:
        """Record successful call."""
        self.total_calls += 1
        self.successful_calls += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = datetime.utcnow()

        if duration >= slow_threshold:
            self.slow_calls += 1

    def record_failure(self) -> None:
        """Record failed call."""
        self.total_calls += 1
        self.failed_calls += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = datetime.utcnow()

    def record_rejection(self) -> None:
        """Record rejected call."""
        self.rejected_calls += 1

    def record_state_change(self, old_state: CircuitState, new_state: CircuitState) -> None:
        """Record state change."""
        self.last_state_change = datetime.utcnow()
        self.state_history.append((datetime.utcnow(), old_state, new_state))

        # Keep only last 100 state changes
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]

    def reset(self) -> None:
        """Reset statistics."""
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.slow_calls = 0
        self.consecutive_successes = 0
        self.consecutive_failures = 0


class SlidingWindow:
    """Sliding window for tracking call outcomes."""

    def __init__(self, size: int):
        self._size = size
        self._calls: List[tuple] = []  # (timestamp, success, duration)

    def record(self, success: bool, duration: float) -> None:
        """Record a call outcome."""
        self._calls.append((time.time(), success, duration))

        # Remove old entries
        if len(self._calls) > self._size:
            self._calls = self._calls[-self._size:]

    def get_stats(self, slow_threshold: float) -> tuple:
        """Get current statistics (total, failures, slow_calls)."""
        total = len(self._calls)
        failures = sum(1 for _, success, _ in self._calls if not success)
        slow = sum(1 for _, _, duration in self._calls if duration >= slow_threshold)
        return total, failures, slow

    def clear(self) -> None:
        """Clear the window."""
        self._calls.clear()


class FallbackStrategy(ABC):
    """Abstract fallback strategy."""

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Execute fallback."""
        pass


class DefaultValueFallback(FallbackStrategy):
    """Return a default value."""

    def __init__(self, default: Any):
        self._default = default

    async def execute(self, *args, **kwargs) -> Any:
        return self._default


class FunctionFallback(FallbackStrategy):
    """Execute a fallback function."""

    def __init__(self, func: Callable):
        self._func = func

    async def execute(self, *args, **kwargs) -> Any:
        result = self._func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result


class CacheFallback(FallbackStrategy):
    """Return cached value."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def cache_result(self, key: str, value: Any) -> None:
        """Cache a result."""
        self._cache[key] = value

    async def execute(self, *args, **kwargs) -> Any:
        key = str(args) + str(sorted(kwargs.items()))
        if key in self._cache:
            return self._cache[key]
        raise CircuitBreakerError("No cached value available")


class CircuitBreaker:
    """Circuit breaker implementation."""

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback: Optional[FallbackStrategy] = None,
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None,
    ):
        self._name = name
        self._config = config or CircuitBreakerConfig()
        self._fallback = fallback
        self._on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._window = SlidingWindow(self._config.window_size)
        self._half_open_calls = 0
        self._opened_at: Optional[float] = None
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def stats(self) -> CircuitStats:
        return self._stats

    async def call(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute function through circuit breaker."""
        async with self._lock:
            # Check if we should allow the call
            allowed, reason = self._should_allow_call()

            if not allowed:
                self._stats.record_rejection()

                if self._fallback:
                    logger.debug(f"Circuit {self._name}: using fallback ({reason})")
                    return await self._fallback.execute(*args, **kwargs)

                raise CircuitBreakerOpen(self._name, reason)

        # Execute the call
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result

            duration = time.time() - start_time
            await self._on_success(duration)

            return result

        except Exception as e:
            await self._on_failure(e)
            raise

    def _should_allow_call(self) -> tuple[bool, str]:
        """Check if call should be allowed."""
        if self._state == CircuitState.CLOSED:
            return True, ""

        elif self._state == CircuitState.OPEN:
            # Check if timeout has passed
            if self._opened_at and time.time() - self._opened_at >= self._config.timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)
                return True, ""
            return False, "circuit is open"

        elif self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls < self._config.half_open_max_calls:
                self._half_open_calls += 1
                return True, ""
            return False, "half-open call limit reached"

        return False, "unknown state"

    async def _on_success(self, duration: float) -> None:
        """Handle successful call."""
        async with self._lock:
            self._stats.record_success(duration, self._config.slow_call_threshold_seconds)
            self._window.record(True, duration)

            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self._config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    async def _on_failure(self, error: Exception) -> None:
        """Handle failed call."""
        async with self._lock:
            self._stats.record_failure()
            self._window.record(False, 0)

            if self._state == CircuitState.CLOSED:
                # Check if we should open
                if self._should_open():
                    self._transition_to(CircuitState.OPEN)

            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                self._transition_to(CircuitState.OPEN)

    def _should_open(self) -> bool:
        """Check if circuit should open."""
        # Check consecutive failures
        if self._stats.consecutive_failures >= self._config.failure_threshold:
            return True

        # Check failure rate
        total, failures, slow = self._window.get_stats(
            self._config.slow_call_threshold_seconds
        )

        if total >= self._config.min_calls:
            failure_rate = failures / total if total > 0 else 0
            slow_rate = slow / total if total > 0 else 0

            if failure_rate >= self._config.failure_rate_threshold:
                return True

            if slow_rate >= self._config.slow_call_rate_threshold:
                return True

        return False

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state."""
        old_state = self._state

        if old_state == new_state:
            return

        self._state = new_state
        self._stats.record_state_change(old_state, new_state)

        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()
            logger.warning(f"Circuit {self._name}: OPEN (was {old_state.value})")

        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            logger.info(f"Circuit {self._name}: HALF_OPEN (testing recovery)")

        elif new_state == CircuitState.CLOSED:
            self._opened_at = None
            self._half_open_calls = 0
            self._window.clear()
            self._stats.reset()
            logger.info(f"Circuit {self._name}: CLOSED (recovered)")

        if self._on_state_change:
            try:
                self._on_state_change(old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._transition_to(CircuitState.CLOSED)

    def force_open(self) -> None:
        """Manually open the circuit."""
        self._transition_to(CircuitState.OPEN)


class CircuitBreakerRegistry:
    """Registry for circuit breakers."""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._default_config = CircuitBreakerConfig()

    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback: Optional[FallbackStrategy] = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                config=config or self._default_config,
                fallback=fallback,
            )
        return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)

    def list_breakers(self) -> List[str]:
        """List all circuit breaker names."""
        return list(self._breakers.keys())

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all circuit breakers."""
        return {
            name: {
                "state": breaker.state.value,
                "total_calls": breaker.stats.total_calls,
                "failed_calls": breaker.stats.failed_calls,
                "rejected_calls": breaker.stats.rejected_calls,
                "failure_rate": breaker.stats.failure_rate,
            }
            for name, breaker in self._breakers.items()
        }

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()


def circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
    fallback: Optional[Union[Any, Callable]] = None,
    registry: Optional[CircuitBreakerRegistry] = None,
):
    """Decorator to wrap function with circuit breaker."""
    _registry = registry or _default_registry

    # Convert fallback to strategy
    fallback_strategy = None
    if fallback is not None:
        if callable(fallback):
            fallback_strategy = FunctionFallback(fallback)
        else:
            fallback_strategy = DefaultValueFallback(fallback)

    breaker = _registry.get_or_create(name, config, fallback_strategy)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        async def async_wrapper(*args, **kwargs) -> T:
            return await breaker.call(func, *args, **kwargs)

        def sync_wrapper(*args, **kwargs) -> T:
            return asyncio.run(breaker.call(func, *args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class CircuitBreakerError(Exception):
    """Base circuit breaker error."""
    pass


class CircuitBreakerOpen(CircuitBreakerError):
    """Raised when circuit is open."""

    def __init__(self, name: str, reason: str = ""):
        self.name = name
        self.reason = reason
        super().__init__(f"Circuit breaker '{name}' is open: {reason}")


# Default registry
_default_registry = CircuitBreakerRegistry()


def get_default_registry() -> CircuitBreakerRegistry:
    """Get default circuit breaker registry."""
    return _default_registry


__all__ = [
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitStats",
    "SlidingWindow",
    "FallbackStrategy",
    "DefaultValueFallback",
    "FunctionFallback",
    "CacheFallback",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "circuit_breaker",
    "CircuitBreakerError",
    "CircuitBreakerOpen",
    "get_default_registry",
]
