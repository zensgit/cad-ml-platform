"""Vision provider resilience patterns.

Provides:
- Retry with exponential backoff
- Circuit breaker pattern
- Provider health tracking
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, TypeVar

from .base import VisionDescription, VisionProvider, VisionProviderError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RetryConfig:
    """Retry configuration."""

    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 30.0  # seconds
    exponential_base: float = 2.0
    retryable_exceptions: tuple = (
        VisionProviderError,
        asyncio.TimeoutError,
        ConnectionError,
    )


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout: float = 60.0  # Seconds before trying half-open
    half_open_max_calls: int = 1  # Max concurrent calls in half-open


@dataclass
class CircuitBreakerState:
    """Circuit breaker runtime state."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0
    half_open_calls: int = 0


@dataclass
class ProviderMetrics:
    """Metrics for a vision provider."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0
    last_error: Optional[str] = None
    last_error_time: Optional[float] = None
    circuit_opens: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests


class ResilientVisionProvider(VisionProvider):
    """Wrapper that adds retry and circuit breaker to any VisionProvider."""

    def __init__(
        self,
        provider: VisionProvider,
        retry_config: Optional[RetryConfig] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
    ):
        """
        Initialize resilient provider wrapper.

        Args:
            provider: The underlying vision provider
            retry_config: Retry configuration
            circuit_config: Circuit breaker configuration
        """
        self._provider = provider
        self._retry_config = retry_config or RetryConfig()
        self._circuit_config = circuit_config or CircuitBreakerConfig()
        self._circuit_state = CircuitBreakerState()
        self._metrics = ProviderMetrics()
        # Lazy-initialize asyncio primitives to support sync construction
        # in tests and call sites where no event loop is running yet.
        self._lock: Optional[asyncio.Lock] = None

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True
    ) -> VisionDescription:
        """
        Analyze image with retry and circuit breaker protection.

        Args:
            image_data: Raw image bytes
            include_description: Whether to generate description

        Returns:
            VisionDescription with analysis results

        Raises:
            VisionProviderError: On provider errors after retries
            CircuitOpenError: When circuit breaker is open
        """
        # Check circuit breaker
        await self._check_circuit()

        start_time = time.time()
        self._metrics.total_requests += 1

        last_exception = None
        for attempt in range(self._retry_config.max_retries + 1):
            try:
                result = await self._provider.analyze_image(image_data, include_description)

                # Success - update metrics and circuit
                latency_ms = (time.time() - start_time) * 1000
                await self._on_success(latency_ms)
                return result

            except self._retry_config.retryable_exceptions as e:
                last_exception = e
                await self._on_failure(str(e))

                if attempt < self._retry_config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"[{self.provider_name}] Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"[{self.provider_name}] All {self._retry_config.max_retries + 1} "
                        f"attempts failed: {e}"
                    )

            except Exception as e:
                # Non-retryable exception
                await self._on_failure(str(e))
                raise

        # All retries exhausted
        raise VisionProviderError(
            self.provider_name,
            f"Failed after {self._retry_config.max_retries + 1} attempts: " f"{last_exception}",
        )

    @property
    def provider_name(self) -> str:
        """Return wrapped provider name with resilient suffix."""
        return f"{self._provider.provider_name}"

    @property
    def metrics(self) -> ProviderMetrics:
        """Get provider metrics."""
        return self._metrics

    @property
    def circuit_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self._circuit_state.state

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        import random

        delay = self._retry_config.base_delay * (self._retry_config.exponential_base**attempt)
        delay = min(delay, self._retry_config.max_delay)
        # Add jitter (±20%)
        jitter = delay * 0.2 * (2 * random.random() - 1)
        return delay + jitter

    def _get_lock(self) -> asyncio.Lock:
        """Get or create the internal asyncio lock."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def _check_circuit(self) -> None:
        """Check and update circuit breaker state."""
        async with self._get_lock():
            state = self._circuit_state

            if state.state == CircuitState.OPEN:
                # Check if timeout has passed
                if time.time() - state.last_failure_time >= self._circuit_config.timeout:
                    state.state = CircuitState.HALF_OPEN
                    state.half_open_calls = 0
                    logger.info(f"[{self.provider_name}] Circuit half-open, testing...")
                else:
                    remaining = self._circuit_config.timeout - (
                        time.time() - state.last_failure_time
                    )
                    raise CircuitOpenError(
                        self.provider_name,
                        f"Circuit breaker open, retry after {remaining:.0f}s",
                    )

            elif state.state == CircuitState.HALF_OPEN:
                if state.half_open_calls >= self._circuit_config.half_open_max_calls:
                    raise CircuitOpenError(
                        self.provider_name,
                        "Circuit half-open, max test calls reached",
                    )
                state.half_open_calls += 1

    async def _on_success(self, latency_ms: float) -> None:
        """Handle successful request."""
        async with self._get_lock():
            self._metrics.successful_requests += 1
            self._metrics.total_latency_ms += latency_ms

            state = self._circuit_state
            if state.state == CircuitState.HALF_OPEN:
                state.success_count += 1
                if state.success_count >= self._circuit_config.success_threshold:
                    state.state = CircuitState.CLOSED
                    state.failure_count = 0
                    state.success_count = 0
                    logger.info(f"[{self.provider_name}] Circuit closed, recovered")
            elif state.state == CircuitState.CLOSED:
                # Reset failure count on success
                state.failure_count = 0

    async def _on_failure(self, error: str) -> None:
        """Handle failed request."""
        async with self._get_lock():
            self._metrics.failed_requests += 1
            self._metrics.last_error = error
            self._metrics.last_error_time = time.time()

            state = self._circuit_state
            state.failure_count += 1
            state.last_failure_time = time.time()

            if state.state == CircuitState.HALF_OPEN:
                # Failed during test, reopen circuit
                state.state = CircuitState.OPEN
                state.success_count = 0
                self._metrics.circuit_opens += 1
                logger.warning(f"[{self.provider_name}] Circuit reopened after test failure")

            elif state.state == CircuitState.CLOSED:
                if state.failure_count >= self._circuit_config.failure_threshold:
                    state.state = CircuitState.OPEN
                    self._metrics.circuit_opens += 1
                    logger.warning(
                        f"[{self.provider_name}] Circuit opened after "
                        f"{state.failure_count} failures"
                    )

    def reset_circuit(self) -> None:
        """Manually reset circuit breaker to closed state."""
        self._circuit_state = CircuitBreakerState()
        logger.info(f"[{self.provider_name}] Circuit manually reset")

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self._metrics = ProviderMetrics()


class CircuitOpenError(VisionProviderError):
    """Raised when circuit breaker is open."""

    pass


def create_resilient_provider(
    provider: VisionProvider,
    max_retries: int = 3,
    circuit_failure_threshold: int = 5,
    circuit_timeout: float = 60.0,
) -> ResilientVisionProvider:
    """
    Factory to create a resilient provider with common defaults.

    Args:
        provider: The underlying vision provider
        max_retries: Maximum retry attempts
        circuit_failure_threshold: Failures before opening circuit
        circuit_timeout: Seconds before testing half-open

    Returns:
        ResilientVisionProvider wrapping the original provider

    Example:
        >>> from src.core.vision import create_vision_provider
        >>> provider = create_vision_provider("openai")
        >>> resilient = create_resilient_provider(provider)
        >>> result = await resilient.analyze_image(image_bytes)
    """
    return ResilientVisionProvider(
        provider=provider,
        retry_config=RetryConfig(max_retries=max_retries),
        circuit_config=CircuitBreakerConfig(
            failure_threshold=circuit_failure_threshold,
            timeout=circuit_timeout,
        ),
    )
