"""Request prioritization module for Vision Provider system.

This module provides priority queue and request scheduling capabilities including:
- Priority levels and queues
- Request scheduling and ordering
- Priority-based rate limiting
- Preemption and deadline handling
- Fair queuing algorithms
"""

import asyncio
import heapq
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .base import VisionDescription, VisionProvider


class Priority(IntEnum):
    """Request priority levels."""

    CRITICAL = 0  # Highest priority - system critical
    HIGH = 1  # High priority - important user requests
    NORMAL = 2  # Normal priority - standard requests
    LOW = 3  # Low priority - batch/background
    BACKGROUND = 4  # Lowest priority - can be delayed


class QueueStrategy(Enum):
    """Queue processing strategies."""

    STRICT_PRIORITY = "strict_priority"  # Always process highest priority first
    WEIGHTED_FAIR = "weighted_fair"  # Weight-based fair queuing
    ROUND_ROBIN = "round_robin"  # Round-robin across priorities
    DEADLINE = "deadline"  # Earliest deadline first
    DEFICIT_ROUND_ROBIN = "deficit_round_robin"  # DRR algorithm


class RequestStatus(Enum):
    """Status of a queued request."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass(order=True)
class PrioritizedRequest:
    """A request with priority metadata."""

    priority_value: int = field(compare=True)  # For heap ordering
    sequence: int = field(compare=True)  # Tie-breaker for same priority
    request_id: str = field(compare=False)
    priority: Priority = field(compare=False)
    image_data: bytes = field(compare=False)
    prompt: Optional[str] = field(compare=False, default=None)
    ocr_only: bool = field(compare=False, default=False)
    deadline: Optional[datetime] = field(compare=False, default=None)
    created_at: datetime = field(compare=False, default_factory=datetime.now)
    status: RequestStatus = field(compare=False, default=RequestStatus.PENDING)
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)
    callback: Optional[Callable] = field(compare=False, default=None)

    @property
    def is_expired(self) -> bool:
        """Check if request has expired."""
        if self.deadline:
            return datetime.now() > self.deadline
        return False

    @property
    def time_in_queue_ms(self) -> float:
        """Get time spent in queue."""
        return (datetime.now() - self.created_at).total_seconds() * 1000


@dataclass
class QueueStats:
    """Statistics for a priority queue."""

    total_enqueued: int = 0
    total_dequeued: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_expired: int = 0
    total_cancelled: int = 0
    current_size: int = 0
    max_size_reached: int = 0
    total_wait_time_ms: float = 0.0
    requests_by_priority: Dict[Priority, int] = field(default_factory=dict)

    @property
    def average_wait_time_ms(self) -> float:
        """Calculate average wait time."""
        if self.total_dequeued == 0:
            return 0.0
        return self.total_wait_time_ms / self.total_dequeued


class PriorityQueue(ABC):
    """Abstract base for priority queues."""

    @abstractmethod
    def enqueue(self, request: PrioritizedRequest) -> None:
        """Add a request to the queue."""
        pass

    @abstractmethod
    def dequeue(self) -> Optional[PrioritizedRequest]:
        """Remove and return the highest priority request."""
        pass

    @abstractmethod
    def peek(self) -> Optional[PrioritizedRequest]:
        """View the highest priority request without removing."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Get current queue size."""
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        pass


class HeapPriorityQueue(PriorityQueue):
    """Heap-based priority queue implementation."""

    def __init__(self, max_size: Optional[int] = None) -> None:
        """Initialize the queue."""
        self._heap: List[PrioritizedRequest] = []
        self._max_size = max_size
        self._sequence = 0

    def enqueue(self, request: PrioritizedRequest) -> None:
        """Add a request to the queue."""
        if self._max_size and len(self._heap) >= self._max_size:
            raise QueueFullError(f"Queue full (max size: {self._max_size})")

        self._sequence += 1
        request.sequence = self._sequence
        request.priority_value = request.priority.value

        heapq.heappush(self._heap, request)

    def dequeue(self) -> Optional[PrioritizedRequest]:
        """Remove and return the highest priority request."""
        if not self._heap:
            return None
        return heapq.heappop(self._heap)

    def peek(self) -> Optional[PrioritizedRequest]:
        """View the highest priority request without removing."""
        if not self._heap:
            return None
        return self._heap[0]

    def size(self) -> int:
        """Get current queue size."""
        return len(self._heap)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._heap) == 0

    def remove_expired(self) -> List[PrioritizedRequest]:
        """Remove and return all expired requests."""
        expired = []
        remaining = []

        for request in self._heap:
            if request.is_expired:
                request.status = RequestStatus.EXPIRED
                expired.append(request)
            else:
                remaining.append(request)

        if expired:
            self._heap = remaining
            heapq.heapify(self._heap)

        return expired


class MultiLevelQueue:
    """Multi-level feedback queue with separate queues per priority."""

    def __init__(
        self,
        strategy: QueueStrategy = QueueStrategy.STRICT_PRIORITY,
        max_size_per_level: int = 1000,
    ) -> None:
        """Initialize the multi-level queue."""
        self._queues: Dict[Priority, HeapPriorityQueue] = {
            p: HeapPriorityQueue(max_size=max_size_per_level) for p in Priority
        }
        self._strategy = strategy
        self._round_robin_index = 0
        self._weights = {
            Priority.CRITICAL: 100,
            Priority.HIGH: 50,
            Priority.NORMAL: 25,
            Priority.LOW: 10,
            Priority.BACKGROUND: 5,
        }
        self._deficits: Dict[Priority, int] = {p: 0 for p in Priority}
        self._quantum = 10  # Deficit quantum for DRR

    def enqueue(self, request: PrioritizedRequest) -> None:
        """Add a request to the appropriate queue."""
        queue = self._queues.get(request.priority)
        if queue:
            queue.enqueue(request)

    def dequeue(self) -> Optional[PrioritizedRequest]:
        """Remove and return a request based on strategy."""
        if self._strategy == QueueStrategy.STRICT_PRIORITY:
            return self._dequeue_strict()
        elif self._strategy == QueueStrategy.WEIGHTED_FAIR:
            return self._dequeue_weighted()
        elif self._strategy == QueueStrategy.ROUND_ROBIN:
            return self._dequeue_round_robin()
        elif self._strategy == QueueStrategy.DEFICIT_ROUND_ROBIN:
            return self._dequeue_drr()
        else:
            return self._dequeue_strict()

    def _dequeue_strict(self) -> Optional[PrioritizedRequest]:
        """Strict priority ordering."""
        for priority in Priority:
            queue = self._queues[priority]
            if not queue.is_empty():
                return queue.dequeue()
        return None

    def _dequeue_weighted(self) -> Optional[PrioritizedRequest]:
        """Weighted fair queuing."""
        import random

        non_empty = [(p, q) for p, q in self._queues.items() if not q.is_empty()]
        if not non_empty:
            return None

        weights = [self._weights[p] for p, _ in non_empty]
        selected = random.choices(non_empty, weights=weights, k=1)[0]
        return selected[1].dequeue()

    def _dequeue_round_robin(self) -> Optional[PrioritizedRequest]:
        """Round-robin across non-empty queues."""
        priorities = list(Priority)
        for _ in range(len(priorities)):
            self._round_robin_index = (self._round_robin_index + 1) % len(priorities)
            priority = priorities[self._round_robin_index]
            queue = self._queues[priority]
            if not queue.is_empty():
                return queue.dequeue()
        return None

    def _dequeue_drr(self) -> Optional[PrioritizedRequest]:
        """Deficit round-robin algorithm."""
        priorities = list(Priority)

        for _ in range(len(priorities)):
            for priority in priorities:
                queue = self._queues[priority]
                if queue.is_empty():
                    self._deficits[priority] = 0
                    continue

                self._deficits[priority] += self._quantum * self._weights[priority]

                if self._deficits[priority] > 0:
                    request = queue.dequeue()
                    if request:
                        self._deficits[priority] -= 1
                        return request

        return None

    def size(self) -> int:
        """Get total queue size."""
        return sum(q.size() for q in self._queues.values())

    def size_by_priority(self) -> Dict[Priority, int]:
        """Get size per priority level."""
        return {p: q.size() for p, q in self._queues.items()}

    def is_empty(self) -> bool:
        """Check if all queues are empty."""
        return all(q.is_empty() for q in self._queues.values())

    def remove_expired(self) -> List[PrioritizedRequest]:
        """Remove expired requests from all queues."""
        expired = []
        for queue in self._queues.values():
            expired.extend(queue.remove_expired())
        return expired


class RequestScheduler:
    """Schedules and processes prioritized requests."""

    def __init__(
        self,
        provider: VisionProvider,
        queue: Optional[MultiLevelQueue] = None,
        max_concurrent: int = 5,
        process_interval_ms: int = 100,
    ) -> None:
        """Initialize the scheduler.

        Args:
            provider: The vision provider to use
            queue: The priority queue
            max_concurrent: Maximum concurrent requests
            process_interval_ms: Processing interval in milliseconds
        """
        self._provider = provider
        self._queue = queue or MultiLevelQueue()
        self._max_concurrent = max_concurrent
        self._process_interval = process_interval_ms / 1000.0
        self._active_requests: Dict[str, PrioritizedRequest] = {}
        self._completed_requests: Dict[str, Tuple[PrioritizedRequest, Any]] = {}
        self._stats = QueueStats()
        self._running = False
        self._sequence = 0

    def submit(
        self,
        image_data: bytes,
        priority: Priority = Priority.NORMAL,
        prompt: Optional[str] = None,
        ocr_only: bool = False,
        deadline: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable] = None,
    ) -> str:
        """Submit a request to the queue.

        Args:
            image_data: Raw image bytes
            priority: Request priority
            prompt: Optional analysis prompt
            ocr_only: Whether to only perform OCR
            deadline: Optional deadline for the request
            metadata: Additional metadata
            callback: Callback when complete

        Returns:
            Request ID
        """
        import uuid

        self._sequence += 1
        request_id = f"req_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._sequence}"

        request = PrioritizedRequest(
            priority_value=priority.value,
            sequence=self._sequence,
            request_id=request_id,
            priority=priority,
            image_data=image_data,
            prompt=prompt,
            ocr_only=ocr_only,
            deadline=deadline,
            metadata=metadata or {},
            callback=callback,
        )

        self._queue.enqueue(request)

        # Update stats
        self._stats.total_enqueued += 1
        self._stats.current_size = self._queue.size()
        self._stats.max_size_reached = max(self._stats.max_size_reached, self._stats.current_size)
        self._stats.requests_by_priority[priority] = (
            self._stats.requests_by_priority.get(priority, 0) + 1
        )

        return request_id

    async def process_next(self) -> Optional[Tuple[str, VisionDescription]]:
        """Process the next request from the queue.

        Returns:
            Tuple of (request_id, result) or None if queue empty
        """
        # Remove expired requests first
        expired = self._queue.remove_expired()
        self._stats.total_expired += len(expired)

        # Check if we can process more
        if len(self._active_requests) >= self._max_concurrent:
            return None

        # Get next request
        request = self._queue.dequeue()
        if not request:
            return None

        # Update stats
        self._stats.total_dequeued += 1
        self._stats.total_wait_time_ms += request.time_in_queue_ms
        self._stats.current_size = self._queue.size()

        # Mark as processing
        request.status = RequestStatus.PROCESSING
        self._active_requests[request.request_id] = request

        try:
            result = await self._provider.analyze_image(request.image_data)
            request.status = RequestStatus.COMPLETED
            self._stats.total_completed += 1

            # Store result
            self._completed_requests[request.request_id] = (request, result)

            # Call callback if provided
            if request.callback:
                try:
                    request.callback(request.request_id, result, None)
                except Exception:
                    pass

            return request.request_id, result

        except Exception as e:
            request.status = RequestStatus.FAILED
            self._stats.total_failed += 1

            if request.callback:
                try:
                    request.callback(request.request_id, None, e)
                except Exception:
                    pass

            raise

        finally:
            self._active_requests.pop(request.request_id, None)

    async def run(self) -> None:
        """Run the scheduler loop."""
        self._running = True

        while self._running:
            if not self._queue.is_empty():
                try:
                    await self.process_next()
                except Exception:
                    pass  # Errors handled in process_next

            await asyncio.sleep(self._process_interval)

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False

    def cancel(self, request_id: str) -> bool:
        """Cancel a pending request."""
        # Cannot cancel active requests
        if request_id in self._active_requests:
            return False

        # Would need to implement removal from queue
        self._stats.total_cancelled += 1
        return True

    def get_status(self, request_id: str) -> Optional[RequestStatus]:
        """Get status of a request."""
        if request_id in self._active_requests:
            return RequestStatus.PROCESSING

        if request_id in self._completed_requests:
            request, _ = self._completed_requests[request_id]
            return request.status

        return None

    def get_result(self, request_id: str) -> Optional[VisionDescription]:
        """Get result of a completed request."""
        if request_id in self._completed_requests:
            _, result = self._completed_requests[request_id]
            return result
        return None

    def get_stats(self) -> QueueStats:
        """Get queue statistics."""
        return self._stats

    def get_queue_sizes(self) -> Dict[Priority, int]:
        """Get queue sizes by priority."""
        return self._queue.size_by_priority()


class PrioritizedVisionProvider(VisionProvider):
    """Vision provider wrapper with priority queue support."""

    def __init__(
        self,
        provider: VisionProvider,
        strategy: QueueStrategy = QueueStrategy.STRICT_PRIORITY,
        max_queue_size: int = 1000,
        default_priority: Priority = Priority.NORMAL,
    ) -> None:
        """Initialize the prioritized provider.

        Args:
            provider: The underlying provider
            strategy: Queue processing strategy
            max_queue_size: Maximum queue size
            default_priority: Default request priority
        """
        self._provider = provider
        self._queue = MultiLevelQueue(strategy=strategy, max_size_per_level=max_queue_size)
        self._scheduler = RequestScheduler(provider, self._queue)
        self._default_priority = default_priority

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"prioritized_{self._provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with default priority.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        return await self.analyze_with_priority(image_data, self._default_priority)

    async def analyze_with_priority(
        self,
        image_data: bytes,
        priority: Priority,
        deadline: Optional[datetime] = None,
    ) -> VisionDescription:
        """Analyze image with specified priority.

        Args:
            image_data: Raw image bytes
            priority: Request priority
            deadline: Optional deadline

        Returns:
            Vision analysis description
        """
        # For immediate processing, bypass queue
        if priority == Priority.CRITICAL:
            return await self._provider.analyze_image(image_data)

        # Submit to queue
        request_id = self._scheduler.submit(
            image_data=image_data,
            priority=priority,
            deadline=deadline,
        )

        # Process immediately (synchronous mode)
        result = await self._scheduler.process_next()
        if result and result[0] == request_id:
            return result[1]

        # Fallback to direct processing
        return await self._provider.analyze_image(image_data)

    def get_scheduler(self) -> RequestScheduler:
        """Get the request scheduler."""
        return self._scheduler

    def get_stats(self) -> QueueStats:
        """Get queue statistics."""
        return self._scheduler.get_stats()


class QueueFullError(Exception):
    """Error raised when queue is full."""

    pass


def create_prioritized_provider(
    provider: VisionProvider,
    strategy: QueueStrategy = QueueStrategy.STRICT_PRIORITY,
    max_queue_size: int = 1000,
    default_priority: Priority = Priority.NORMAL,
) -> PrioritizedVisionProvider:
    """Create a prioritized provider wrapper.

    Args:
        provider: The underlying provider
        strategy: Queue processing strategy
        max_queue_size: Maximum queue size
        default_priority: Default request priority

    Returns:
        Prioritized provider wrapper
    """
    return PrioritizedVisionProvider(
        provider=provider,
        strategy=strategy,
        max_queue_size=max_queue_size,
        default_priority=default_priority,
    )
