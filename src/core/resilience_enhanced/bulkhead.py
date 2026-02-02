"""Bulkhead Pattern Implementation.

Provides isolation patterns:
- Semaphore-based bulkhead
- Thread pool bulkhead
- Async bulkhead
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BulkheadError(Exception):
    """Raised when bulkhead rejects a call."""

    def __init__(self, message: str = "Bulkhead is full"):
        self.message = message
        super().__init__(message)


@dataclass
class BulkheadConfig:
    """Bulkhead configuration."""
    max_concurrent_calls: int = 10
    max_wait_time: float = 0.0  # 0 means no waiting
    name: str = "default"


@dataclass
class BulkheadMetrics:
    """Metrics for bulkhead state."""
    available_permits: int
    max_concurrent_calls: int
    active_calls: int
    waiting_calls: int
    rejected_calls: int
    successful_calls: int
    failed_calls: int


class Bulkhead(ABC):
    """Abstract base class for bulkhead implementations."""

    @abstractmethod
    async def acquire(self) -> bool:
        """Acquire a permit. Returns True if acquired."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Release a permit."""
        pass

    @abstractmethod
    def get_metrics(self) -> BulkheadMetrics:
        """Get current metrics."""
        pass


class SemaphoreBulkhead(Bulkhead):
    """Semaphore-based bulkhead for async operations."""

    def __init__(self, config: Optional[BulkheadConfig] = None):
        self.config = config or BulkheadConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_calls)
        self._active_calls = 0
        self._waiting_calls = 0
        self._rejected_calls = 0
        self._successful_calls = 0
        self._failed_calls = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire a permit with optional timeout."""
        async with self._lock:
            self._waiting_calls += 1

        try:
            if self.config.max_wait_time > 0:
                try:
                    acquired = await asyncio.wait_for(
                        self._semaphore.acquire(),
                        timeout=self.config.max_wait_time,
                    )
                except asyncio.TimeoutError:
                    async with self._lock:
                        self._waiting_calls -= 1
                        self._rejected_calls += 1
                    return False
            else:
                # Non-blocking acquire
                acquired = self._semaphore.locked()
                if not acquired:
                    await self._semaphore.acquire()
                else:
                    async with self._lock:
                        self._waiting_calls -= 1
                        self._rejected_calls += 1
                    return False

            async with self._lock:
                self._waiting_calls -= 1
                self._active_calls += 1
            return True

        except Exception:
            async with self._lock:
                self._waiting_calls -= 1
                self._rejected_calls += 1
            raise

    def release(self) -> None:
        """Release a permit."""
        self._semaphore.release()
        # Note: _active_calls is decremented in record_success/record_failure

    async def record_success(self) -> None:
        """Record a successful call completion."""
        async with self._lock:
            self._active_calls -= 1
            self._successful_calls += 1
        self._semaphore.release()

    async def record_failure(self) -> None:
        """Record a failed call completion."""
        async with self._lock:
            self._active_calls -= 1
            self._failed_calls += 1
        self._semaphore.release()

    def get_metrics(self) -> BulkheadMetrics:
        """Get current metrics."""
        return BulkheadMetrics(
            available_permits=self.config.max_concurrent_calls - self._active_calls,
            max_concurrent_calls=self.config.max_concurrent_calls,
            active_calls=self._active_calls,
            waiting_calls=self._waiting_calls,
            rejected_calls=self._rejected_calls,
            successful_calls=self._successful_calls,
            failed_calls=self._failed_calls,
        )

    async def __aenter__(self) -> "SemaphoreBulkhead":
        """Async context manager entry."""
        if not await self.acquire():
            raise BulkheadError(
                f"Bulkhead '{self.config.name}' is full "
                f"({self._active_calls}/{self.config.max_concurrent_calls} active)"
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Async context manager exit."""
        if exc_val is None:
            await self.record_success()
        else:
            await self.record_failure()
        return False


class ThreadPoolBulkhead:
    """Thread pool-based bulkhead for blocking operations."""

    def __init__(self, config: Optional[BulkheadConfig] = None):
        self.config = config or BulkheadConfig()
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_calls,
            thread_name_prefix=f"bulkhead-{self.config.name}",
        )
        self._active_calls = 0
        self._queue_size = 0
        self._rejected_calls = 0
        self._successful_calls = 0
        self._failed_calls = 0
        self._lock = threading.Lock()

    def submit(
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> Future[T]:
        """Submit a task to the bulkhead."""
        with self._lock:
            if self._active_calls >= self.config.max_concurrent_calls:
                self._rejected_calls += 1
                raise BulkheadError(
                    f"Bulkhead '{self.config.name}' is full"
                )
            self._active_calls += 1

        def wrapped() -> T:
            try:
                result = fn(*args, **kwargs)
                with self._lock:
                    self._active_calls -= 1
                    self._successful_calls += 1
                return result
            except Exception as e:
                with self._lock:
                    self._active_calls -= 1
                    self._failed_calls += 1
                raise

        return self._executor.submit(wrapped)

    def get_metrics(self) -> BulkheadMetrics:
        """Get current metrics."""
        with self._lock:
            return BulkheadMetrics(
                available_permits=self.config.max_concurrent_calls - self._active_calls,
                max_concurrent_calls=self.config.max_concurrent_calls,
                active_calls=self._active_calls,
                waiting_calls=self._queue_size,
                rejected_calls=self._rejected_calls,
                successful_calls=self._successful_calls,
                failed_calls=self._failed_calls,
            )

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=wait)


class AdaptiveBulkhead(Bulkhead):
    """Adaptive bulkhead that adjusts limits based on load."""

    def __init__(
        self,
        config: Optional[BulkheadConfig] = None,
        min_permits: int = 1,
        max_permits: int = 100,
        growth_rate: float = 1.5,
        shrink_rate: float = 0.5,
        success_threshold: float = 0.9,
        failure_threshold: float = 0.5,
        window_size: int = 100,
    ):
        base_config = config or BulkheadConfig()
        self.config = base_config
        self.min_permits = min_permits
        self.max_permits = max_permits
        self.growth_rate = growth_rate
        self.shrink_rate = shrink_rate
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold
        self.window_size = window_size

        self._current_permits = base_config.max_concurrent_calls
        self._semaphore = asyncio.Semaphore(self._current_permits)
        self._results: List[bool] = []  # True = success, False = failure
        self._active_calls = 0
        self._lock = asyncio.Lock()

        # Metrics
        self._rejected_calls = 0
        self._successful_calls = 0
        self._failed_calls = 0

    async def acquire(self) -> bool:
        """Acquire a permit."""
        if self._semaphore.locked():
            async with self._lock:
                self._rejected_calls += 1
            return False

        await self._semaphore.acquire()
        async with self._lock:
            self._active_calls += 1
        return True

    def release(self) -> None:
        """Release a permit."""
        self._semaphore.release()

    async def record_result(self, success: bool) -> None:
        """Record a call result and potentially adjust limits."""
        async with self._lock:
            self._active_calls -= 1
            if success:
                self._successful_calls += 1
            else:
                self._failed_calls += 1

            self._results.append(success)
            if len(self._results) > self.window_size:
                self._results.pop(0)

            # Check if we should adjust
            if len(self._results) >= self.window_size:
                await self._adjust_permits()

        self._semaphore.release()

    async def _adjust_permits(self) -> None:
        """Adjust permit count based on success rate."""
        success_rate = sum(self._results) / len(self._results)

        if success_rate >= self.success_threshold:
            # Growing - increase permits
            new_permits = min(
                self.max_permits,
                int(self._current_permits * self.growth_rate)
            )
            if new_permits > self._current_permits:
                # Add permits to semaphore
                for _ in range(new_permits - self._current_permits):
                    self._semaphore.release()
                self._current_permits = new_permits
                logger.info(f"Bulkhead '{self.config.name}' grew to {new_permits} permits")

        elif success_rate <= self.failure_threshold:
            # Shrinking - decrease permits
            new_permits = max(
                self.min_permits,
                int(self._current_permits * self.shrink_rate)
            )
            if new_permits < self._current_permits:
                # We can't easily remove permits from semaphore
                # Instead, we track the new limit and reject excess
                self._current_permits = new_permits
                logger.info(f"Bulkhead '{self.config.name}' shrunk to {new_permits} permits")

    def get_metrics(self) -> BulkheadMetrics:
        """Get current metrics."""
        return BulkheadMetrics(
            available_permits=self._current_permits - self._active_calls,
            max_concurrent_calls=self._current_permits,
            active_calls=self._active_calls,
            waiting_calls=0,
            rejected_calls=self._rejected_calls,
            successful_calls=self._successful_calls,
            failed_calls=self._failed_calls,
        )

    async def __aenter__(self) -> "AdaptiveBulkhead":
        """Async context manager entry."""
        if not await self.acquire():
            raise BulkheadError(f"Bulkhead '{self.config.name}' is full")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Async context manager exit."""
        await self.record_result(exc_val is None)
        return False


class BulkheadRegistry:
    """Registry for managing multiple bulkheads."""

    def __init__(self):
        self._bulkheads: Dict[str, Bulkhead] = {}

    def get_or_create(
        self,
        name: str,
        config: Optional[BulkheadConfig] = None,
        bulkhead_type: str = "semaphore",
    ) -> Bulkhead:
        """Get or create a bulkhead."""
        if name not in self._bulkheads:
            cfg = config or BulkheadConfig(name=name)
            if bulkhead_type == "adaptive":
                self._bulkheads[name] = AdaptiveBulkhead(cfg)
            else:
                self._bulkheads[name] = SemaphoreBulkhead(cfg)
        return self._bulkheads[name]

    def get(self, name: str) -> Optional[Bulkhead]:
        """Get a bulkhead by name."""
        return self._bulkheads.get(name)

    def get_all_metrics(self) -> Dict[str, BulkheadMetrics]:
        """Get metrics for all bulkheads."""
        return {
            name: bulkhead.get_metrics()
            for name, bulkhead in self._bulkheads.items()
        }
