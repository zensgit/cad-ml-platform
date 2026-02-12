"""Circuit Breaker Pattern Implementation.

Prevents cascading failures by failing fast when a service is unhealthy.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, Type

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes to close from half-open
    timeout: float = 30.0  # Seconds before trying half-open
    half_open_max_calls: int = 3  # Max calls in half-open state

    # Failure detection
    failure_rate_threshold: float = 0.5  # Failure rate to open
    minimum_calls: int = 10  # Minimum calls before rate calculation

    # Exceptions to count as failures
    failure_exceptions: tuple = field(default_factory=lambda: (Exception,))

    # Exceptions to ignore (not counted as success or failure)
    ignored_exceptions: tuple = field(default_factory=tuple)


@dataclass
class CircuitStats:
    """Circuit breaker statistics."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls

    def record_success(self) -> None:
        """Record a successful call."""
        self.total_calls += 1
        self.successful_calls += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = time.time()

    def record_failure(self) -> None:
        """Record a failed call."""
        self.total_calls += 1
        self.failed_calls += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = time.time()

    def record_rejection(self) -> None:
        """Record a rejected call (circuit open)."""
        self.rejected_calls += 1

    def reset(self) -> None:
        """Reset statistics."""
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0


class CircuitBreakerError(Exception):
    """Raised when circuit is open."""

    def __init__(self, circuit_name: str, retry_after: float):
        self.circuit_name = circuit_name
        self.retry_after = retry_after
        super().__init__(f"Circuit '{circuit_name}' is open. Retry after {retry_after:.1f}s")


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures."""

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._opened_at: Optional[float] = None
        self._half_open_calls = 0
        self._lock: Optional[asyncio.Lock] = None
        self._listeners: list = []

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for automatic transitions."""
        if self._state == CircuitState.OPEN:
            if self._should_try_half_open():
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state

    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""
        return self._stats

    def _should_try_half_open(self) -> bool:
        """Check if we should transition to half-open."""
        if self._opened_at is None:
            return False
        return time.time() - self._opened_at >= self.config.timeout

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()
            logger.warning(f"Circuit '{self.name}' opened")
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            logger.info(f"Circuit '{self.name}' half-open, testing...")
        elif new_state == CircuitState.CLOSED:
            self._opened_at = None
            self._stats.reset()
            logger.info(f"Circuit '{self.name}' closed")

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(self.name, old_state, new_state)
            except Exception as e:
                logger.error(f"Circuit listener error: {e}")

    def _should_open(self) -> bool:
        """Check if circuit should open."""
        # Check consecutive failures
        if self._stats.consecutive_failures >= self.config.failure_threshold:
            return True

        # Check failure rate
        if self._stats.total_calls >= self.config.minimum_calls:
            if self._stats.failure_rate >= self.config.failure_rate_threshold:
                return True

        return False

    def _should_close(self) -> bool:
        """Check if circuit should close from half-open."""
        return self._stats.consecutive_successes >= self.config.success_threshold

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function through circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        async with self._get_lock():
            state = self.state

            # Check if circuit is open
            if state == CircuitState.OPEN:
                self._stats.record_rejection()
                retry_after = self.config.timeout - (time.time() - (self._opened_at or 0))
                raise CircuitBreakerError(self.name, max(0, retry_after))

            # Check half-open call limit
            if state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self._stats.record_rejection()
                    raise CircuitBreakerError(self.name, 1.0)
                self._half_open_calls += 1

        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Record success
            async with self._get_lock():
                self._stats.record_success()

                if self._state == CircuitState.HALF_OPEN and self._should_close():
                    self._transition_to(CircuitState.CLOSED)

            return result

        except self.config.ignored_exceptions:
            # Don't count ignored exceptions
            raise

        except self.config.failure_exceptions as e:
            # Record failure
            async with self._get_lock():
                self._stats.record_failure()

                if self._state == CircuitState.HALF_OPEN:
                    # Any failure in half-open goes back to open
                    self._transition_to(CircuitState.OPEN)
                elif self._state == CircuitState.CLOSED and self._should_open():
                    self._transition_to(CircuitState.OPEN)

            raise

    def add_listener(self, listener: Callable[[str, CircuitState, CircuitState], None]) -> None:
        """Add state change listener.

        Args:
            listener: Callback(circuit_name, old_state, new_state)
        """
        self._listeners.append(listener)

    def force_open(self) -> None:
        """Manually open the circuit."""
        self._transition_to(CircuitState.OPEN)

    def force_close(self) -> None:
        """Manually close the circuit."""
        self._transition_to(CircuitState.CLOSED)

    def reset(self) -> None:
        """Reset circuit to initial state."""
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._opened_at = None
        self._half_open_calls = 0


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create a circuit breaker by name."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def circuit_breaker(
    name: Optional[str] = None,
    config: Optional[CircuitBreakerConfig] = None,
    fallback: Optional[Callable[..., Any]] = None,
) -> Callable[[F], F]:
    """Decorator for circuit breaker protection.

    Args:
        name: Circuit name (defaults to function name)
        config: Circuit configuration
        fallback: Fallback function if circuit is open

    Example:
        @circuit_breaker(name="external_api")
        async def call_external_api():
            ...

        @circuit_breaker(fallback=lambda: {"status": "degraded"})
        async def get_data():
            ...
    """
    def decorator(func: F) -> F:
        circuit_name = name or func.__name__
        cb = get_circuit_breaker(circuit_name, config)

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await cb.call(func, *args, **kwargs)
            except CircuitBreakerError:
                if fallback:
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback(*args, **kwargs)
                    return fallback(*args, **kwargs)
                raise

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # For sync functions, we need to handle differently
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(cb.call(func, *args, **kwargs))
            except CircuitBreakerError:
                if fallback:
                    return fallback(*args, **kwargs)
                raise
            finally:
                loop.close()

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._global_listeners: list = []

    def register(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Register a new circuit breaker."""
        breaker = CircuitBreaker(name, config)

        # Add global listeners
        for listener in self._global_listeners:
            breaker.add_listener(listener)

        self._breakers[name] = breaker
        return breaker

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._breakers.get(name)

    def get_or_create(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        if name not in self._breakers:
            return self.register(name, config)
        return self._breakers[name]

    def list_all(self) -> Dict[str, CircuitBreaker]:
        """List all registered circuit breakers."""
        return dict(self._breakers)

    def add_global_listener(self, listener: Callable[[str, CircuitState, CircuitState], None]) -> None:
        """Add listener to all circuit breakers."""
        self._global_listeners.append(listener)
        for breaker in self._breakers.values():
            breaker.add_listener(listener)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {
            name: {
                "state": breaker.state.value,
                "total_calls": breaker.stats.total_calls,
                "successful_calls": breaker.stats.successful_calls,
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


# Global registry
_registry: Optional[CircuitBreakerRegistry] = None


def get_circuit_registry() -> CircuitBreakerRegistry:
    """Get global circuit breaker registry."""
    global _registry
    if _registry is None:
        _registry = CircuitBreakerRegistry()
    return _registry
