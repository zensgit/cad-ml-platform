"""Fallback Strategies.

Provides fallback mechanisms:
- Static fallback
- Cached fallback
- Fallback chain
- Graceful degradation
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class FallbackMetrics:
    """Metrics for fallback usage."""
    total_calls: int
    primary_successes: int
    fallback_invocations: int
    fallback_successes: int
    fallback_failures: int


class FallbackStrategy(ABC, Generic[T]):
    """Abstract base class for fallback strategies."""

    @abstractmethod
    async def execute(self, exception: Exception) -> T:
        """Execute fallback and return result."""
        pass

    @abstractmethod
    def get_metrics(self) -> FallbackMetrics:
        """Get fallback metrics."""
        pass


class StaticFallback(FallbackStrategy[T]):
    """Return a static value on failure."""

    def __init__(self, value: T):
        self.value = value
        self._invocations = 0

    async def execute(self, exception: Exception) -> T:
        self._invocations += 1
        logger.debug(f"Static fallback invoked, returning: {self.value}")
        return self.value

    def get_metrics(self) -> FallbackMetrics:
        return FallbackMetrics(
            total_calls=self._invocations,
            primary_successes=0,
            fallback_invocations=self._invocations,
            fallback_successes=self._invocations,
            fallback_failures=0,
        )


class FunctionFallback(FallbackStrategy[T]):
    """Execute a function to generate fallback value."""

    def __init__(
        self,
        fallback_fn: Callable[[Exception], T],
        pass_args: bool = False,
    ):
        self.fallback_fn = fallback_fn
        self.pass_args = pass_args
        self._invocations = 0
        self._failures = 0

    async def execute(self, exception: Exception) -> T:
        self._invocations += 1
        try:
            if asyncio.iscoroutinefunction(self.fallback_fn):
                result = await self.fallback_fn(exception)
            else:
                result = self.fallback_fn(exception)
            return result
        except Exception as e:
            self._failures += 1
            logger.error(f"Fallback function failed: {e}")
            raise

    def get_metrics(self) -> FallbackMetrics:
        return FallbackMetrics(
            total_calls=self._invocations,
            primary_successes=0,
            fallback_invocations=self._invocations,
            fallback_successes=self._invocations - self._failures,
            fallback_failures=self._failures,
        )


class CachedFallback(FallbackStrategy[T]):
    """Return the last successful result on failure."""

    def __init__(
        self,
        max_age: Optional[float] = None,
        default_value: Optional[T] = None,
    ):
        self.max_age = max_age
        self.default_value = default_value
        self._cached_value: Optional[T] = None
        self._cached_at: Optional[float] = None
        self._invocations = 0
        self._cache_hits = 0
        self._cache_misses = 0

    def cache_result(self, value: T) -> None:
        """Cache a successful result."""
        self._cached_value = value
        self._cached_at = time.time()

    async def execute(self, exception: Exception) -> T:
        self._invocations += 1

        # Check if cache is valid
        if self._cached_value is not None:
            if self.max_age is None or (
                self._cached_at and time.time() - self._cached_at < self.max_age
            ):
                self._cache_hits += 1
                logger.debug("Cache fallback: returning cached value")
                return self._cached_value

        self._cache_misses += 1

        if self.default_value is not None:
            logger.debug("Cache fallback: returning default value")
            return self.default_value

        raise exception

    def get_metrics(self) -> FallbackMetrics:
        return FallbackMetrics(
            total_calls=self._invocations,
            primary_successes=0,
            fallback_invocations=self._invocations,
            fallback_successes=self._cache_hits,
            fallback_failures=self._cache_misses,
        )


class FallbackChain(FallbackStrategy[T]):
    """Try multiple fallback strategies in order."""

    def __init__(self, strategies: List[FallbackStrategy[T]]):
        self.strategies = strategies
        self._invocations = 0
        self._strategy_successes: Dict[int, int] = {i: 0 for i in range(len(strategies))}
        self._all_failed = 0

    async def execute(self, exception: Exception) -> T:
        self._invocations += 1
        last_exception = exception

        for i, strategy in enumerate(self.strategies):
            try:
                result = await strategy.execute(last_exception)
                self._strategy_successes[i] += 1
                logger.debug(f"Fallback chain: strategy {i} succeeded")
                return result
            except Exception as e:
                last_exception = e
                continue

        self._all_failed += 1
        logger.error("All fallback strategies failed")
        raise last_exception

    def get_metrics(self) -> FallbackMetrics:
        total_successes = sum(self._strategy_successes.values())
        return FallbackMetrics(
            total_calls=self._invocations,
            primary_successes=0,
            fallback_invocations=self._invocations,
            fallback_successes=total_successes,
            fallback_failures=self._all_failed,
        )


class GracefulDegradation(FallbackStrategy[T]):
    """Provide degraded functionality based on error type."""

    def __init__(self):
        self._handlers: Dict[type, Callable[[Exception], T]] = {}
        self._default_handler: Optional[Callable[[Exception], T]] = None
        self._invocations = 0
        self._successes = 0
        self._failures = 0

    def on_exception(
        self,
        exception_type: type,
        handler: Callable[[Exception], T],
    ) -> "GracefulDegradation[T]":
        """Register handler for specific exception type."""
        self._handlers[exception_type] = handler
        return self

    def default(self, handler: Callable[[Exception], T]) -> "GracefulDegradation[T]":
        """Set default handler for unmatched exceptions."""
        self._default_handler = handler
        return self

    async def execute(self, exception: Exception) -> T:
        self._invocations += 1

        # Find matching handler
        handler = None
        for exc_type, h in self._handlers.items():
            if isinstance(exception, exc_type):
                handler = h
                break

        if handler is None:
            handler = self._default_handler

        if handler is None:
            self._failures += 1
            raise exception

        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(exception)
            else:
                result = handler(exception)
            self._successes += 1
            return result
        except Exception as e:
            self._failures += 1
            raise

    def get_metrics(self) -> FallbackMetrics:
        return FallbackMetrics(
            total_calls=self._invocations,
            primary_successes=0,
            fallback_invocations=self._invocations,
            fallback_successes=self._successes,
            fallback_failures=self._failures,
        )


class FallbackDecorator:
    """Decorator for adding fallback to functions."""

    def __init__(
        self,
        fallback: Union[FallbackStrategy, Callable, Any],
        exceptions: tuple = (Exception,),
    ):
        if isinstance(fallback, FallbackStrategy):
            self._fallback = fallback
        elif callable(fallback):
            self._fallback = FunctionFallback(fallback)
        else:
            self._fallback = StaticFallback(fallback)

        self._exceptions = exceptions
        self._primary_successes = 0

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        import functools

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                result = await func(*args, **kwargs)
                self._primary_successes += 1

                # Cache result if using CachedFallback
                if isinstance(self._fallback, CachedFallback):
                    self._fallback.cache_result(result)

                return result
            except self._exceptions as e:
                return await self._fallback.execute(e)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                result = func(*args, **kwargs)
                self._primary_successes += 1

                if isinstance(self._fallback, CachedFallback):
                    self._fallback.cache_result(result)

                return result
            except self._exceptions as e:
                # Run fallback synchronously
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(self._fallback.execute(e))
                finally:
                    loop.close()

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore


def with_fallback(
    fallback: Union[FallbackStrategy, Callable, Any],
    exceptions: tuple = (Exception,),
) -> Callable:
    """Decorator factory for adding fallback."""
    return FallbackDecorator(fallback, exceptions)
