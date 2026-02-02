"""
Async inference queue for high-throughput serving.

Provides:
- Asynchronous request processing
- Priority queuing
- Concurrent execution
- Backpressure handling
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

from src.ml.serving.request import (
    InferenceRequest,
    InferenceResponse,
    RequestPriority,
    RequestStatus,
)

logger = logging.getLogger(__name__)


class QueueState(str, Enum):
    """Queue state."""
    RUNNING = "running"
    PAUSED = "paused"
    DRAINING = "draining"
    STOPPED = "stopped"


@dataclass
class QueueConfig:
    """Configuration for async queue."""
    max_queue_size: int = 1000
    max_concurrent: int = 4
    worker_threads: int = 2
    timeout_seconds: float = 30.0
    enable_priority: bool = True
    drain_timeout: float = 60.0
    batch_wait_ms: float = 10.0  # Wait time for batch accumulation


@dataclass
class QueueStats:
    """Statistics for async queue."""
    total_enqueued: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_timeout: int = 0
    total_rejected: int = 0
    current_queue_size: int = 0
    current_processing: int = 0
    avg_queue_time_ms: float = 0.0
    avg_processing_time_ms: float = 0.0
    max_queue_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_enqueued": self.total_enqueued,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "total_timeout": self.total_timeout,
            "total_rejected": self.total_rejected,
            "current_queue_size": self.current_queue_size,
            "current_processing": self.current_processing,
            "avg_queue_time_ms": round(self.avg_queue_time_ms, 2),
            "avg_processing_time_ms": round(self.avg_processing_time_ms, 2),
            "max_queue_time_ms": round(self.max_queue_time_ms, 2),
            "max_processing_time_ms": round(self.max_processing_time_ms, 2),
            "throughput": self.total_completed / max(1, self.avg_processing_time_ms / 1000),
        }


@dataclass
class QueuedRequest:
    """A request in the queue."""
    request: InferenceRequest
    future: asyncio.Future
    enqueue_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None

    @property
    def queue_time_ms(self) -> float:
        if self.start_time is None:
            return (time.time() - self.enqueue_time) * 1000
        return (self.start_time - self.enqueue_time) * 1000

    @property
    def priority_value(self) -> int:
        """Get numeric priority value for sorting."""
        priority_map = {
            RequestPriority.LOW: 3,
            RequestPriority.NORMAL: 2,
            RequestPriority.HIGH: 1,
            RequestPriority.CRITICAL: 0,
        }
        return priority_map.get(self.request.priority, 2)


class AsyncInferenceQueue:
    """
    Async queue for inference requests.

    Provides:
    - Priority-based queuing
    - Concurrent request processing
    - Backpressure when queue is full
    - Graceful shutdown with draining
    """

    def __init__(
        self,
        process_fn: Callable[[InferenceRequest], InferenceResponse],
        config: Optional[QueueConfig] = None,
    ):
        """
        Initialize async inference queue.

        Args:
            process_fn: Function to process requests
            config: Queue configuration
        """
        self._process_fn = process_fn
        self._config = config or QueueConfig()

        # Queue storage - separate queues for each priority
        self._queues: Dict[RequestPriority, deque] = {
            RequestPriority.CRITICAL: deque(),
            RequestPriority.HIGH: deque(),
            RequestPriority.NORMAL: deque(),
            RequestPriority.LOW: deque(),
        }

        self._state = QueueState.STOPPED
        self._stats = QueueStats()
        self._lock = threading.Lock()
        self._processing_count = 0

        # Worker components
        self._executor: Optional[ThreadPoolExecutor] = None
        self._process_task: Optional[asyncio.Task] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue_event = asyncio.Event()

    @property
    def state(self) -> QueueState:
        return self._state

    @property
    def stats(self) -> QueueStats:
        return self._stats

    @property
    def queue_size(self) -> int:
        """Get total queue size across all priorities."""
        return sum(len(q) for q in self._queues.values())

    async def start(self) -> None:
        """Start the queue processor."""
        if self._state == QueueState.RUNNING:
            return

        self._event_loop = asyncio.get_event_loop()
        self._executor = ThreadPoolExecutor(max_workers=self._config.worker_threads)
        self._state = QueueState.RUNNING
        self._process_task = asyncio.create_task(self._process_loop())

        logger.info(f"AsyncInferenceQueue started with {self._config.worker_threads} workers")

    async def stop(self, drain: bool = True) -> None:
        """
        Stop the queue processor.

        Args:
            drain: Whether to process remaining requests before stopping
        """
        if self._state == QueueState.STOPPED:
            return

        if drain and self.queue_size > 0:
            self._state = QueueState.DRAINING
            logger.info(f"Draining {self.queue_size} remaining requests...")

            # Wait for queue to drain
            start = time.time()
            while self.queue_size > 0 and (time.time() - start) < self._config.drain_timeout:
                await asyncio.sleep(0.1)

        self._state = QueueState.STOPPED

        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        if self._executor:
            self._executor.shutdown(wait=True)

        logger.info("AsyncInferenceQueue stopped")

    def pause(self) -> None:
        """Pause queue processing."""
        if self._state == QueueState.RUNNING:
            self._state = QueueState.PAUSED
            logger.info("AsyncInferenceQueue paused")

    def resume(self) -> None:
        """Resume queue processing."""
        if self._state == QueueState.PAUSED:
            self._state = QueueState.RUNNING
            self._queue_event.set()
            logger.info("AsyncInferenceQueue resumed")

    async def submit(self, request: InferenceRequest) -> InferenceResponse:
        """
        Submit a request to the queue.

        Args:
            request: Inference request

        Returns:
            InferenceResponse when complete

        Raises:
            asyncio.TimeoutError: If request times out
            RuntimeError: If queue is full or stopped
        """
        if self._state == QueueState.STOPPED:
            raise RuntimeError("Queue is stopped")

        if self.queue_size >= self._config.max_queue_size:
            self._stats.total_rejected += 1
            raise RuntimeError("Queue is full")

        # Create future for result
        future = self._event_loop.create_future() if self._event_loop else asyncio.get_event_loop().create_future()

        # Create queued request
        queued = QueuedRequest(
            request=request,
            future=future,
        )

        # Add to appropriate priority queue
        with self._lock:
            self._queues[request.priority].append(queued)
            self._stats.total_enqueued += 1
            self._stats.current_queue_size = self.queue_size

        # Signal processor
        self._queue_event.set()

        # Wait for result with timeout
        try:
            return await asyncio.wait_for(future, timeout=request.timeout)
        except asyncio.TimeoutError:
            self._stats.total_timeout += 1
            return InferenceResponse.error_response(
                request.request_id,
                request.model_name,
                "Request timeout",
            )

    def submit_sync(self, request: InferenceRequest) -> Future:
        """
        Submit request synchronously (returns Future).

        Args:
            request: Inference request

        Returns:
            Future that will contain InferenceResponse
        """
        if self._executor is None:
            raise RuntimeError("Queue not started")

        return self._executor.submit(self._process_sync, request)

    def _process_sync(self, request: InferenceRequest) -> InferenceResponse:
        """Process request synchronously."""
        try:
            return self._process_fn(request)
        except Exception as e:
            return InferenceResponse.error_response(
                request.request_id,
                request.model_name,
                str(e),
            )

    async def _process_loop(self) -> None:
        """Main processing loop."""
        while self._state in (QueueState.RUNNING, QueueState.DRAINING, QueueState.PAUSED):
            if self._state == QueueState.PAUSED:
                await self._queue_event.wait()
                self._queue_event.clear()
                continue

            # Get next request from highest priority queue
            queued = self._get_next_request()

            if queued is None:
                if self._state == QueueState.DRAINING:
                    break
                # Wait for new requests
                try:
                    await asyncio.wait_for(self._queue_event.wait(), timeout=0.1)
                    self._queue_event.clear()
                except asyncio.TimeoutError:
                    pass
                continue

            # Check concurrent limit
            while self._processing_count >= self._config.max_concurrent:
                await asyncio.sleep(0.001)

            # Process request
            asyncio.create_task(self._process_request(queued))

    def _get_next_request(self) -> Optional[QueuedRequest]:
        """Get next request from queues (priority order)."""
        with self._lock:
            for priority in [RequestPriority.CRITICAL, RequestPriority.HIGH,
                             RequestPriority.NORMAL, RequestPriority.LOW]:
                if self._queues[priority]:
                    queued = self._queues[priority].popleft()
                    self._stats.current_queue_size = self.queue_size
                    return queued
        return None

    async def _process_request(self, queued: QueuedRequest) -> None:
        """Process a single request."""
        queued.start_time = time.time()
        self._processing_count += 1
        self._stats.current_processing = self._processing_count

        try:
            # Run inference in thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._process_fn,
                queued.request,
            )

            # Update stats
            process_time = (time.time() - queued.start_time) * 1000
            queue_time = queued.queue_time_ms

            self._update_stats(queue_time, process_time, response.success)

            # Set result
            if not queued.future.done():
                queued.future.set_result(response)

        except Exception as e:
            logger.error(f"Request processing error: {e}")
            self._stats.total_failed += 1

            if not queued.future.done():
                queued.future.set_result(InferenceResponse.error_response(
                    queued.request.request_id,
                    queued.request.model_name,
                    str(e),
                ))

        finally:
            self._processing_count -= 1
            self._stats.current_processing = self._processing_count

    def _update_stats(self, queue_time: float, process_time: float, success: bool) -> None:
        """Update queue statistics."""
        if success:
            self._stats.total_completed += 1
        else:
            self._stats.total_failed += 1

        # Update averages (exponential moving average)
        alpha = 0.1
        self._stats.avg_queue_time_ms = (
            alpha * queue_time + (1 - alpha) * self._stats.avg_queue_time_ms
        )
        self._stats.avg_processing_time_ms = (
            alpha * process_time + (1 - alpha) * self._stats.avg_processing_time_ms
        )

        # Update max values
        self._stats.max_queue_time_ms = max(self._stats.max_queue_time_ms, queue_time)
        self._stats.max_processing_time_ms = max(self._stats.max_processing_time_ms, process_time)

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "state": self._state.value,
            "queue_sizes": {
                priority.value: len(self._queues[priority])
                for priority in RequestPriority
            },
            "stats": self._stats.to_dict(),
        }


class BatchAccumulator:
    """
    Accumulates requests into batches for efficient processing.

    Waits for either:
    - Batch size threshold
    - Time threshold
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_ms: float = 50.0,
    ):
        """
        Initialize batch accumulator.

        Args:
            max_batch_size: Maximum batch size
            max_wait_ms: Maximum wait time in milliseconds
        """
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._buffer: Dict[str, List[QueuedRequest]] = {}  # model_name -> requests
        self._lock = threading.Lock()
        self._last_batch_time: Dict[str, float] = {}

    def add(self, queued: QueuedRequest) -> Optional[List[QueuedRequest]]:
        """
        Add request to accumulator.

        Returns batch if ready, None otherwise.
        """
        model_name = queued.request.model_name

        with self._lock:
            if model_name not in self._buffer:
                self._buffer[model_name] = []
                self._last_batch_time[model_name] = time.time()

            self._buffer[model_name].append(queued)

            # Check if batch is ready
            if len(self._buffer[model_name]) >= self._max_batch_size:
                return self._extract_batch(model_name)

            # Check time threshold
            elapsed = (time.time() - self._last_batch_time[model_name]) * 1000
            if elapsed >= self._max_wait_ms and self._buffer[model_name]:
                return self._extract_batch(model_name)

        return None

    def _extract_batch(self, model_name: str) -> List[QueuedRequest]:
        """Extract batch from buffer."""
        batch = self._buffer[model_name][:self._max_batch_size]
        self._buffer[model_name] = self._buffer[model_name][self._max_batch_size:]
        self._last_batch_time[model_name] = time.time()
        return batch

    def get_ready_batches(self) -> Dict[str, List[QueuedRequest]]:
        """Get all batches that are ready (time threshold exceeded)."""
        ready = {}
        current_time = time.time()

        with self._lock:
            for model_name, buffer in list(self._buffer.items()):
                if not buffer:
                    continue

                elapsed = (current_time - self._last_batch_time.get(model_name, current_time)) * 1000
                if elapsed >= self._max_wait_ms:
                    ready[model_name] = self._extract_batch(model_name)

        return ready

    def flush(self, model_name: Optional[str] = None) -> Dict[str, List[QueuedRequest]]:
        """Flush all pending requests."""
        flushed = {}

        with self._lock:
            if model_name:
                if model_name in self._buffer and self._buffer[model_name]:
                    flushed[model_name] = self._buffer.pop(model_name)
            else:
                for name, buffer in list(self._buffer.items()):
                    if buffer:
                        flushed[name] = buffer
                self._buffer.clear()
                self._last_batch_time.clear()

        return flushed

    def get_pending_count(self, model_name: Optional[str] = None) -> int:
        """Get number of pending requests."""
        with self._lock:
            if model_name:
                return len(self._buffer.get(model_name, []))
            return sum(len(b) for b in self._buffer.values())
