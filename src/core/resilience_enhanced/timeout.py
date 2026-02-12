"""Timeout Policies.

Provides timeout management:
- Simple timeout
- Adaptive timeout
- Hedged requests
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


class TimeoutError(Exception):
    """Raised when operation times out."""

    def __init__(self, timeout: float, message: str = ""):
        self.timeout = timeout
        self.message = message or f"Operation timed out after {timeout}s"
        super().__init__(self.message)


@dataclass
class TimeoutConfig:
    """Timeout configuration."""
    timeout: float = 30.0  # seconds
    cancel_on_timeout: bool = True


@dataclass
class TimeoutMetrics:
    """Metrics for timeout policy."""
    total_calls: int
    successful_calls: int
    timed_out_calls: int
    average_duration: float
    p50_duration: float
    p95_duration: float
    p99_duration: float
    current_timeout: float


class TimeoutPolicy(ABC):
    """Abstract base class for timeout policies."""

    @abstractmethod
    async def execute(
        self,
        operation: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute operation with timeout."""
        pass

    @abstractmethod
    def get_metrics(self) -> TimeoutMetrics:
        """Get current metrics."""
        pass


class SimpleTimeout(TimeoutPolicy):
    """Simple fixed timeout policy."""

    def __init__(self, config: Optional[TimeoutConfig] = None):
        self.config = config or TimeoutConfig()
        self._durations: List[float] = []
        self._timed_out = 0
        self._successful = 0

    async def execute(
        self,
        operation: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute operation with fixed timeout."""
        start = time.time()

        try:
            if asyncio.iscoroutinefunction(operation):
                result = await asyncio.wait_for(
                    operation(*args, **kwargs),
                    timeout=self.config.timeout,
                )
            else:
                # Wrap sync function in executor
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: operation(*args, **kwargs)),
                    timeout=self.config.timeout,
                )

            duration = time.time() - start
            self._durations.append(duration)
            if len(self._durations) > 1000:
                self._durations = self._durations[-1000:]
            self._successful += 1
            return result

        except asyncio.TimeoutError:
            self._timed_out += 1
            raise TimeoutError(self.config.timeout)

    def get_metrics(self) -> TimeoutMetrics:
        """Get current metrics."""
        if not self._durations:
            return TimeoutMetrics(
                total_calls=0,
                successful_calls=0,
                timed_out_calls=0,
                average_duration=0.0,
                p50_duration=0.0,
                p95_duration=0.0,
                p99_duration=0.0,
                current_timeout=self.config.timeout,
            )

        sorted_durations = sorted(self._durations)
        n = len(sorted_durations)

        return TimeoutMetrics(
            total_calls=self._successful + self._timed_out,
            successful_calls=self._successful,
            timed_out_calls=self._timed_out,
            average_duration=statistics.mean(sorted_durations),
            p50_duration=sorted_durations[int(n * 0.5)],
            p95_duration=sorted_durations[int(n * 0.95)] if n >= 20 else sorted_durations[-1],
            p99_duration=sorted_durations[int(n * 0.99)] if n >= 100 else sorted_durations[-1],
            current_timeout=self.config.timeout,
        )


class AdaptiveTimeout(TimeoutPolicy):
    """Adaptive timeout that adjusts based on response times."""

    def __init__(
        self,
        initial_timeout: float = 30.0,
        min_timeout: float = 1.0,
        max_timeout: float = 120.0,
        percentile: float = 0.99,
        multiplier: float = 2.0,
        window_size: int = 100,
    ):
        self.initial_timeout = initial_timeout
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.percentile = percentile
        self.multiplier = multiplier
        self.window_size = window_size

        self._current_timeout = initial_timeout
        self._durations: List[float] = []
        self._timed_out = 0
        self._successful = 0

    def _update_timeout(self) -> None:
        """Update timeout based on recent durations."""
        if len(self._durations) < 10:
            return

        sorted_durations = sorted(self._durations)
        percentile_index = int(len(sorted_durations) * self.percentile)
        percentile_value = sorted_durations[percentile_index]

        new_timeout = percentile_value * self.multiplier
        self._current_timeout = max(
            self.min_timeout,
            min(self.max_timeout, new_timeout)
        )

    async def execute(
        self,
        operation: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute operation with adaptive timeout."""
        start = time.time()

        try:
            if asyncio.iscoroutinefunction(operation):
                result = await asyncio.wait_for(
                    operation(*args, **kwargs),
                    timeout=self._current_timeout,
                )
            else:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: operation(*args, **kwargs)),
                    timeout=self._current_timeout,
                )

            duration = time.time() - start
            self._durations.append(duration)
            if len(self._durations) > self.window_size:
                self._durations = self._durations[-self.window_size:]
            self._successful += 1
            self._update_timeout()
            return result

        except asyncio.TimeoutError:
            self._timed_out += 1
            raise TimeoutError(self._current_timeout)

    def get_metrics(self) -> TimeoutMetrics:
        """Get current metrics."""
        if not self._durations:
            return TimeoutMetrics(
                total_calls=0,
                successful_calls=0,
                timed_out_calls=0,
                average_duration=0.0,
                p50_duration=0.0,
                p95_duration=0.0,
                p99_duration=0.0,
                current_timeout=self._current_timeout,
            )

        sorted_durations = sorted(self._durations)
        n = len(sorted_durations)

        return TimeoutMetrics(
            total_calls=self._successful + self._timed_out,
            successful_calls=self._successful,
            timed_out_calls=self._timed_out,
            average_duration=statistics.mean(sorted_durations),
            p50_duration=sorted_durations[int(n * 0.5)],
            p95_duration=sorted_durations[int(n * 0.95)] if n >= 20 else sorted_durations[-1],
            p99_duration=sorted_durations[int(n * 0.99)] if n >= 100 else sorted_durations[-1],
            current_timeout=self._current_timeout,
        )


class HedgedRequest(TimeoutPolicy):
    """Hedged requests - send backup request if primary is slow."""

    def __init__(
        self,
        primary_timeout: float = 5.0,
        hedge_delay: float = 2.0,
        max_hedges: int = 2,
    ):
        """Initialize hedged request policy.

        Args:
            primary_timeout: Total timeout for request
            hedge_delay: Delay before sending hedge request
            max_hedges: Maximum number of hedge requests
        """
        self.primary_timeout = primary_timeout
        self.hedge_delay = hedge_delay
        self.max_hedges = max_hedges

        self._durations: List[float] = []
        self._primary_wins = 0
        self._hedge_wins = 0
        self._timed_out = 0

    async def execute(
        self,
        operation: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute with hedging."""
        start = time.time()

        async def run_operation() -> Any:
            if asyncio.iscoroutinefunction(operation):
                return await operation(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: operation(*args, **kwargs)
                )

        # Create primary task
        primary_task = asyncio.create_task(run_operation())
        hedge_tasks: List[asyncio.Task] = []
        all_tasks = [primary_task]

        try:
            # Set up hedge timer
            done = False
            result = None

            while not done:
                remaining = self.primary_timeout - (time.time() - start)
                if remaining <= 0:
                    break

                # Wait for completion or hedge delay
                wait_time = min(self.hedge_delay, remaining)

                done_tasks, pending = await asyncio.wait(
                    all_tasks,
                    timeout=wait_time,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if done_tasks:
                    # Got a result
                    for task in done_tasks:
                        try:
                            result = task.result()
                            done = True

                            # Track which request won
                            if task == primary_task:
                                self._primary_wins += 1
                            else:
                                self._hedge_wins += 1

                            break
                        except Exception:
                            # This task failed, continue waiting for others
                            pass

                if not done and len(hedge_tasks) < self.max_hedges:
                    # Start a hedge request
                    hedge_task = asyncio.create_task(run_operation())
                    hedge_tasks.append(hedge_task)
                    all_tasks.append(hedge_task)

            if result is None:
                self._timed_out += 1
                raise TimeoutError(self.primary_timeout)

            duration = time.time() - start
            self._durations.append(duration)
            if len(self._durations) > 1000:
                self._durations = self._durations[-1000:]

            return result

        finally:
            # Cancel all pending tasks
            for task in all_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    def get_metrics(self) -> TimeoutMetrics:
        """Get current metrics."""
        total = self._primary_wins + self._hedge_wins + self._timed_out

        if not self._durations:
            return TimeoutMetrics(
                total_calls=total,
                successful_calls=self._primary_wins + self._hedge_wins,
                timed_out_calls=self._timed_out,
                average_duration=0.0,
                p50_duration=0.0,
                p95_duration=0.0,
                p99_duration=0.0,
                current_timeout=self.primary_timeout,
            )

        sorted_durations = sorted(self._durations)
        n = len(sorted_durations)

        return TimeoutMetrics(
            total_calls=total,
            successful_calls=self._primary_wins + self._hedge_wins,
            timed_out_calls=self._timed_out,
            average_duration=statistics.mean(sorted_durations),
            p50_duration=sorted_durations[int(n * 0.5)],
            p95_duration=sorted_durations[int(n * 0.95)] if n >= 20 else sorted_durations[-1],
            p99_duration=sorted_durations[int(n * 0.99)] if n >= 100 else sorted_durations[-1],
            current_timeout=self.primary_timeout,
        )

    def get_hedge_stats(self) -> Dict[str, int]:
        """Get statistics on hedge effectiveness."""
        return {
            "primary_wins": self._primary_wins,
            "hedge_wins": self._hedge_wins,
            "timed_out": self._timed_out,
            "hedge_win_rate": (
                self._hedge_wins / (self._primary_wins + self._hedge_wins)
                if (self._primary_wins + self._hedge_wins) > 0
                else 0.0
            ),
        }
