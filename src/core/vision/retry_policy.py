"""Configurable retry policies for Vision Provider system.

This module provides retry capabilities including:
- Multiple retry strategies (fixed, exponential, linear)
- Configurable backoff policies
- Retry conditions and filters
- Jitter support
- Retry budgets
"""

import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from .base import VisionDescription, VisionProvider


class BackoffStrategy(Enum):
    """Backoff strategy for retries."""

    FIXED = "fixed"  # Fixed delay between retries
    EXPONENTIAL = "exponential"  # Exponential backoff
    LINEAR = "linear"  # Linear increase
    FIBONACCI = "fibonacci"  # Fibonacci sequence
    DECORRELATED = "decorrelated"  # Decorrelated jitter


class JitterType(Enum):
    """Type of jitter to apply."""

    NONE = "none"
    FULL = "full"  # Random between 0 and delay
    EQUAL = "equal"  # Random between delay/2 and delay
    DECORRELATED = "decorrelated"  # AWS-style decorrelated


@dataclass
class RetryPolicyConfig:
    """Configuration for retry policy."""

    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    backoff_multiplier: float = 2.0
    jitter_type: JitterType = JitterType.FULL
    jitter_factor: float = 0.1
    retryable_exceptions: Optional[Set[Type[Exception]]] = None
    non_retryable_exceptions: Optional[Set[Type[Exception]]] = None
    retry_on_result: Optional[Callable[[Any], bool]] = None


@dataclass
class RetryAttempt:
    """Record of a retry attempt."""

    attempt_number: int
    timestamp: datetime = field(default_factory=datetime.now)
    delay_seconds: float = 0.0
    success: bool = False
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class RetryStats:
    """Statistics for retry operations."""

    total_operations: int = 0
    successful_first_attempt: int = 0
    successful_after_retry: int = 0
    failed_after_retries: int = 0
    total_retries: int = 0
    total_delay_seconds: float = 0.0
    attempts_histogram: Dict[int, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_operations == 0:
            return 0.0
        successful = self.successful_first_attempt + self.successful_after_retry
        return successful / self.total_operations

    @property
    def first_attempt_success_rate(self) -> float:
        """Calculate first attempt success rate."""
        if self.total_operations == 0:
            return 0.0
        return self.successful_first_attempt / self.total_operations


class BackoffCalculator(ABC):
    """Abstract base class for backoff calculators."""

    @abstractmethod
    def calculate_delay(
        self,
        attempt: int,
        config: RetryPolicyConfig,
    ) -> float:
        """Calculate delay for attempt.

        Args:
            attempt: Attempt number (0-based)
            config: Retry configuration

        Returns:
            Delay in seconds
        """
        pass


class FixedBackoff(BackoffCalculator):
    """Fixed delay backoff."""

    def calculate_delay(
        self,
        attempt: int,
        config: RetryPolicyConfig,
    ) -> float:
        """Return fixed delay."""
        return config.initial_delay_seconds


class ExponentialBackoff(BackoffCalculator):
    """Exponential backoff."""

    def calculate_delay(
        self,
        attempt: int,
        config: RetryPolicyConfig,
    ) -> float:
        """Calculate exponential delay."""
        delay = config.initial_delay_seconds * (config.backoff_multiplier ** attempt)
        return min(delay, config.max_delay_seconds)


class LinearBackoff(BackoffCalculator):
    """Linear backoff."""

    def calculate_delay(
        self,
        attempt: int,
        config: RetryPolicyConfig,
    ) -> float:
        """Calculate linear delay."""
        delay = config.initial_delay_seconds * (attempt + 1)
        return min(delay, config.max_delay_seconds)


class FibonacciBackoff(BackoffCalculator):
    """Fibonacci backoff."""

    def __init__(self) -> None:
        """Initialize fibonacci calculator."""
        self._cache: Dict[int, int] = {0: 0, 1: 1}

    def _fibonacci(self, n: int) -> int:
        """Calculate fibonacci number."""
        if n in self._cache:
            return self._cache[n]

        result = self._fibonacci(n - 1) + self._fibonacci(n - 2)
        self._cache[n] = result
        return result

    def calculate_delay(
        self,
        attempt: int,
        config: RetryPolicyConfig,
    ) -> float:
        """Calculate fibonacci delay."""
        fib = self._fibonacci(attempt + 1)
        delay = config.initial_delay_seconds * fib
        return min(delay, config.max_delay_seconds)


class DecorrelatedBackoff(BackoffCalculator):
    """Decorrelated jitter backoff (AWS style)."""

    def __init__(self) -> None:
        """Initialize decorrelated calculator."""
        self._last_delay: Optional[float] = None

    def calculate_delay(
        self,
        attempt: int,
        config: RetryPolicyConfig,
    ) -> float:
        """Calculate decorrelated delay."""
        if self._last_delay is None:
            self._last_delay = config.initial_delay_seconds

        delay = random.uniform(
            config.initial_delay_seconds,
            self._last_delay * 3,
        )
        delay = min(delay, config.max_delay_seconds)
        self._last_delay = delay
        return delay


def get_backoff_calculator(strategy: BackoffStrategy) -> BackoffCalculator:
    """Get backoff calculator for strategy."""
    calculators = {
        BackoffStrategy.FIXED: FixedBackoff,
        BackoffStrategy.EXPONENTIAL: ExponentialBackoff,
        BackoffStrategy.LINEAR: LinearBackoff,
        BackoffStrategy.FIBONACCI: FibonacciBackoff,
        BackoffStrategy.DECORRELATED: DecorrelatedBackoff,
    }
    return calculators[strategy]()


def apply_jitter(
    delay: float,
    jitter_type: JitterType,
    jitter_factor: float,
) -> float:
    """Apply jitter to delay.

    Args:
        delay: Base delay
        jitter_type: Type of jitter
        jitter_factor: Jitter factor

    Returns:
        Delay with jitter applied
    """
    if jitter_type == JitterType.NONE:
        return delay

    if jitter_type == JitterType.FULL:
        return random.uniform(0, delay)

    if jitter_type == JitterType.EQUAL:
        half = delay / 2
        return half + random.uniform(0, half)

    if jitter_type == JitterType.DECORRELATED:
        return random.uniform(0, delay * (1 + jitter_factor))

    return delay


@dataclass
class RetryBudget:
    """Budget for retry attempts."""

    max_retries_per_window: int = 100
    window_seconds: float = 60.0
    min_retries_available: int = 10

    _attempts: List[datetime] = field(default_factory=list)

    def can_retry(self) -> bool:
        """Check if retry is allowed within budget."""
        self._cleanup_old_attempts()
        return len(self._attempts) < self.max_retries_per_window

    def record_retry(self) -> None:
        """Record a retry attempt."""
        self._attempts.append(datetime.now())

    def _cleanup_old_attempts(self) -> None:
        """Remove old attempts outside window."""
        cutoff = datetime.now() - timedelta(seconds=self.window_seconds)
        self._attempts = [a for a in self._attempts if a > cutoff]

    @property
    def remaining_retries(self) -> int:
        """Get remaining retries in window."""
        self._cleanup_old_attempts()
        return max(0, self.max_retries_per_window - len(self._attempts))


class RetryPolicy:
    """Configurable retry policy."""

    def __init__(
        self,
        config: Optional[RetryPolicyConfig] = None,
        budget: Optional[RetryBudget] = None,
    ) -> None:
        """Initialize retry policy.

        Args:
            config: Retry configuration
            budget: Optional retry budget
        """
        self._config = config or RetryPolicyConfig()
        self._budget = budget
        self._backoff = get_backoff_calculator(self._config.backoff_strategy)
        self._stats = RetryStats()

    @property
    def config(self) -> RetryPolicyConfig:
        """Return configuration."""
        return self._config

    @property
    def stats(self) -> RetryStats:
        """Return statistics."""
        return self._stats

    def should_retry(
        self,
        attempt: int,
        exception: Optional[Exception] = None,
        result: Any = None,
    ) -> bool:
        """Check if operation should be retried.

        Args:
            attempt: Current attempt number
            exception: Exception if any
            result: Result if any

        Returns:
            True if should retry
        """
        if attempt >= self._config.max_retries:
            return False

        if self._budget and not self._budget.can_retry():
            return False

        if exception:
            if self._config.non_retryable_exceptions:
                if type(exception) in self._config.non_retryable_exceptions:
                    return False

            if self._config.retryable_exceptions:
                return type(exception) in self._config.retryable_exceptions

            return True

        if result is not None and self._config.retry_on_result:
            return self._config.retry_on_result(result)

        return False

    def get_delay(self, attempt: int) -> float:
        """Get delay for attempt.

        Args:
            attempt: Attempt number

        Returns:
            Delay in seconds
        """
        delay = self._backoff.calculate_delay(attempt, self._config)
        return apply_jitter(
            delay,
            self._config.jitter_type,
            self._config.jitter_factor,
        )

    async def execute_with_retry(
        self,
        operation: Callable[[], Any],
    ) -> Any:
        """Execute operation with retry.

        Args:
            operation: Async operation to execute

        Returns:
            Operation result
        """
        self._stats.total_operations += 1
        attempts: List[RetryAttempt] = []
        last_exception: Optional[Exception] = None

        for attempt in range(self._config.max_retries + 1):
            start_time = time.time()

            try:
                result = await operation()
                duration_ms = (time.time() - start_time) * 1000

                attempts.append(
                    RetryAttempt(
                        attempt_number=attempt,
                        success=True,
                        duration_ms=duration_ms,
                    )
                )

                if attempt == 0:
                    self._stats.successful_first_attempt += 1
                else:
                    self._stats.successful_after_retry += 1

                self._record_attempts(attempt)
                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                last_exception = e

                attempts.append(
                    RetryAttempt(
                        attempt_number=attempt,
                        success=False,
                        error=str(e),
                        duration_ms=duration_ms,
                    )
                )

                if not self.should_retry(attempt, e):
                    break

                delay = self.get_delay(attempt)
                self._stats.total_retries += 1
                self._stats.total_delay_seconds += delay

                if self._budget:
                    self._budget.record_retry()

                await asyncio.sleep(delay)

        self._stats.failed_after_retries += 1
        self._record_attempts(len(attempts) - 1)

        if last_exception:
            raise last_exception

        raise RuntimeError("Retry exhausted without exception")

    def _record_attempts(self, final_attempt: int) -> None:
        """Record attempt count in histogram."""
        self._stats.attempts_histogram[final_attempt] = (
            self._stats.attempts_histogram.get(final_attempt, 0) + 1
        )


class RetryVisionProvider(VisionProvider):
    """Vision provider with retry policy."""

    def __init__(
        self,
        provider: VisionProvider,
        policy: RetryPolicy,
    ) -> None:
        """Initialize retry provider.

        Args:
            provider: Underlying vision provider
            policy: Retry policy
        """
        self._provider = provider
        self._policy = policy

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"retry_{self._provider.provider_name}"

    @property
    def policy(self) -> RetryPolicy:
        """Return retry policy."""
        return self._policy

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with retry policy.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        async def operation() -> VisionDescription:
            return await self._provider.analyze_image(
                image_data, include_description
            )

        return await self._policy.execute_with_retry(operation)


def create_retry_provider(
    provider: VisionProvider,
    config: Optional[RetryPolicyConfig] = None,
    budget: Optional[RetryBudget] = None,
) -> RetryVisionProvider:
    """Create a retry vision provider.

    Args:
        provider: Underlying vision provider
        config: Retry configuration
        budget: Optional retry budget

    Returns:
        RetryVisionProvider instance
    """
    policy = RetryPolicy(config, budget)
    return RetryVisionProvider(provider=provider, policy=policy)
