"""Unified retry logic using Tenacity.

This module provides declarative retry decorators to replace manual
try-except-sleep loops scattered throughout the codebase.

Benefits over manual retry:
- Declarative configuration
- Automatic exponential backoff
- Built-in logging hooks
- Easy to test (can disable retries)
- Consistent behavior across modules

Example:
    >>> from src.core.resilience.retry import with_retry, provider_retry
    >>>
    >>> @provider_retry
    >>> async def call_vision_api(image: bytes) -> dict:
    ...     return await api_client.analyze(image)
    >>>
    >>> @with_retry(max_attempts=5, min_wait=1, max_wait=30)
    >>> async def custom_operation():
    ...     ...
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

# Conditional import for tenacity
try:
    from tenacity import (
        AsyncRetrying,
        RetryError,
        Retrying,
        before_sleep_log,
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        stop_after_delay,
        wait_exponential,
        wait_fixed,
        wait_random,
    )

    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

    # Fallback implementations
    class RetryError(Exception):
        pass


__all__ = [
    "with_retry",
    "provider_retry",
    "database_retry",
    "network_retry",
    "quick_retry",
    "RetryConfig",
    "RetryError",
    "no_retry",
]

T = TypeVar("T")


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        min_wait: float = 1.0,
        max_wait: float = 60.0,
        multiplier: float = 2.0,
        retry_exceptions: tuple[type[Exception], ...] = (Exception,),
        reraise: bool = True,
        log_level: int = logging.WARNING,
    ):
        """Initialize retry configuration.

        Args:
            max_attempts: Maximum number of retry attempts.
            min_wait: Minimum wait time between retries (seconds).
            max_wait: Maximum wait time between retries (seconds).
            multiplier: Multiplier for exponential backoff.
            retry_exceptions: Exception types to retry on.
            reraise: Whether to re-raise the final exception.
            log_level: Log level for retry attempts.
        """
        self.max_attempts = max_attempts
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.multiplier = multiplier
        self.retry_exceptions = retry_exceptions
        self.reraise = reraise
        self.log_level = log_level


def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    multiplier: float = 2.0,
    retry_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    reraise: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Create a configurable retry decorator.

    This replaces manual retry loops like:
        for attempt in range(max_retries):
            try:
                return await operation()
            except Exception:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2

    Args:
        max_attempts: Maximum number of attempts.
        min_wait: Minimum wait between retries (seconds).
        max_wait: Maximum wait between retries (seconds).
        multiplier: Exponential backoff multiplier.
        retry_exceptions: Exception types to retry on (default: all).
        reraise: Re-raise the final exception if all retries fail.

    Returns:
        Decorator function.

    Example:
        >>> @with_retry(max_attempts=5, min_wait=2, max_wait=30)
        >>> async def flaky_operation():
        ...     ...
    """
    if retry_exceptions is None:
        retry_exceptions = (Exception,)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if not TENACITY_AVAILABLE:
            # Fallback: simple retry without tenacity
            return _fallback_retry(
                func, max_attempts, min_wait, max_wait, multiplier, retry_exceptions
            )

        # Check if async
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(max_attempts),
                    wait=wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait),
                    retry=retry_if_exception_type(retry_exceptions),
                    before_sleep=before_sleep_log(logger, logging.WARNING),
                    reraise=reraise,
                ):
                    with attempt:
                        return await func(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                for attempt in Retrying(
                    stop=stop_after_attempt(max_attempts),
                    wait=wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait),
                    retry=retry_if_exception_type(retry_exceptions),
                    before_sleep=before_sleep_log(logger, logging.WARNING),
                    reraise=reraise,
                ):
                    with attempt:
                        return func(*args, **kwargs)

            return sync_wrapper

    return decorator


def _fallback_retry(
    func: Callable[..., T],
    max_attempts: int,
    min_wait: float,
    max_wait: float,
    multiplier: float,
    retry_exceptions: tuple[type[Exception], ...],
) -> Callable[..., T]:
    """Fallback retry implementation when tenacity is not available."""
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_fallback(*args: Any, **kwargs: Any) -> T:
            wait_time = min_wait
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retry_exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__}: {e}"
                        )
                        await asyncio.sleep(wait_time)
                        wait_time = min(wait_time * multiplier, max_wait)

            raise last_exception

        return async_fallback
    else:
        import time

        @functools.wraps(func)
        def sync_fallback(*args: Any, **kwargs: Any) -> T:
            wait_time = min_wait
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__}: {e}"
                        )
                        time.sleep(wait_time)
                        wait_time = min(wait_time * multiplier, max_wait)

            raise last_exception

        return sync_fallback


def no_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to explicitly mark a function as non-retryable.

    Useful for documentation and testing purposes.

    Example:
        >>> @no_retry
        >>> def critical_operation():
        ...     # This should never be retried
        ...     ...
    """
    func._no_retry = True
    return func


# ============================================================================
# Pre-configured retry decorators for common scenarios
# ============================================================================


def provider_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Retry decorator for external provider calls (Vision APIs, etc.).

    Configuration:
    - 3 attempts
    - 2-30 second exponential backoff
    - Retries on TimeoutError, ConnectionError

    Example:
        >>> @provider_retry
        >>> async def call_openai_vision(image: bytes) -> dict:
        ...     return await openai_client.analyze(image)
    """
    return with_retry(
        max_attempts=3,
        min_wait=2.0,
        max_wait=30.0,
        multiplier=2.0,
        retry_exceptions=(TimeoutError, ConnectionError, OSError),
    )(func)


def database_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Retry decorator for database operations.

    Configuration:
    - 5 attempts
    - 1-10 second exponential backoff
    - Retries on connection errors

    Example:
        >>> @database_retry
        >>> async def save_document(doc: Document) -> str:
        ...     return await db.insert(doc)
    """
    return with_retry(
        max_attempts=5,
        min_wait=1.0,
        max_wait=10.0,
        multiplier=1.5,
        retry_exceptions=(ConnectionError, TimeoutError, OSError),
    )(func)


def network_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Retry decorator for network operations (HTTP, Redis, etc.).

    Configuration:
    - 4 attempts
    - 1-20 second exponential backoff
    - Retries on network errors

    Example:
        >>> @network_retry
        >>> async def fetch_remote_config() -> dict:
        ...     return await http_client.get("/config")
    """
    return with_retry(
        max_attempts=4,
        min_wait=1.0,
        max_wait=20.0,
        multiplier=2.0,
        retry_exceptions=(ConnectionError, TimeoutError, OSError),
    )(func)


def quick_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Retry decorator for quick operations that may transiently fail.

    Configuration:
    - 2 attempts
    - Fixed 0.5 second wait
    - Retries on any exception

    Example:
        >>> @quick_retry
        >>> def parse_config(data: str) -> dict:
        ...     return json.loads(data)
    """
    return with_retry(
        max_attempts=2,
        min_wait=0.5,
        max_wait=0.5,
        multiplier=1.0,
        retry_exceptions=(Exception,),
    )(func)


# ============================================================================
# Context manager for retry
# ============================================================================


class RetryContext:
    """Context manager for retry logic.

    Useful when you need more control over the retry loop.

    Example:
        >>> async with RetryContext(max_attempts=3) as ctx:
        ...     for attempt in ctx:
        ...         try:
        ...             result = await risky_operation()
        ...             break
        ...         except TransientError:
        ...             if not ctx.should_retry():
        ...                 raise
    """

    def __init__(
        self,
        max_attempts: int = 3,
        min_wait: float = 1.0,
        max_wait: float = 60.0,
        multiplier: float = 2.0,
    ):
        self.max_attempts = max_attempts
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.multiplier = multiplier
        self._attempt = 0
        self._wait_time = min_wait

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self._attempt >= self.max_attempts:
            raise StopIteration
        attempt = self._attempt
        self._attempt += 1
        return attempt

    def should_retry(self) -> bool:
        """Check if we should retry."""
        return self._attempt < self.max_attempts

    async def wait(self) -> None:
        """Wait before next retry (async)."""
        await asyncio.sleep(self._wait_time)
        self._wait_time = min(self._wait_time * self.multiplier, self.max_wait)

    def wait_sync(self) -> None:
        """Wait before next retry (sync)."""
        import time

        time.sleep(self._wait_time)
        self._wait_time = min(self._wait_time * self.multiplier, self.max_wait)

    async def __aenter__(self) -> "RetryContext":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def __enter__(self) -> "RetryContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False
