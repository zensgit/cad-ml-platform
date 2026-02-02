"""Stream Processing.

Provides stream processing capabilities:
- Event streams
- Windowed processing
- Stream operators
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
)

from src.core.data_pipeline.transform import Transformer

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class StreamEvent(Generic[T]):
    """Event in a stream."""
    data: T
    timestamp: datetime = field(default_factory=datetime.utcnow)
    key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    offset: Optional[int] = None
    partition: Optional[int] = None


class WindowType(Enum):
    """Types of windows."""
    TUMBLING = "tumbling"
    SLIDING = "sliding"
    SESSION = "session"
    COUNT = "count"


@dataclass
class WindowConfig:
    """Window configuration."""
    window_type: WindowType
    size: timedelta  # For time-based windows
    slide: Optional[timedelta] = None  # For sliding windows
    gap: Optional[timedelta] = None  # For session windows
    count: Optional[int] = None  # For count-based windows


@dataclass
class Window(Generic[T]):
    """A processing window."""
    start: datetime
    end: datetime
    events: List[StreamEvent[T]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.events)

    def add(self, event: StreamEvent[T]) -> None:
        self.events.append(event)


class StreamSource(ABC, Generic[T]):
    """Abstract source of stream events."""

    @abstractmethod
    async def read(self) -> AsyncIterator[StreamEvent[T]]:
        """Read events from source."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the source."""
        pass


class StreamSink(ABC, Generic[T]):
    """Abstract sink for stream events."""

    @abstractmethod
    async def write(self, event: StreamEvent[T]) -> None:
        """Write event to sink."""
        pass

    @abstractmethod
    async def write_batch(self, events: List[StreamEvent[T]]) -> None:
        """Write batch of events."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the sink."""
        pass


class InMemorySource(StreamSource[T]):
    """In-memory event source."""

    def __init__(self, events: List[T], delay_ms: int = 0):
        self._events = events
        self._delay = delay_ms / 1000
        self._closed = False

    async def read(self) -> AsyncIterator[StreamEvent[T]]:
        for i, data in enumerate(self._events):
            if self._closed:
                break
            if self._delay:
                await asyncio.sleep(self._delay)
            yield StreamEvent(data=data, offset=i)

    async def close(self) -> None:
        self._closed = True


class InMemorySink(StreamSink[T]):
    """In-memory event sink."""

    def __init__(self):
        self.events: List[StreamEvent[T]] = []
        self._closed = False

    async def write(self, event: StreamEvent[T]) -> None:
        if not self._closed:
            self.events.append(event)

    async def write_batch(self, events: List[StreamEvent[T]]) -> None:
        if not self._closed:
            self.events.extend(events)

    async def close(self) -> None:
        self._closed = True


class BufferedSink(StreamSink[T]):
    """Buffered sink with flush capabilities."""

    def __init__(
        self,
        inner_sink: StreamSink[T],
        buffer_size: int = 100,
        flush_interval_seconds: float = 1.0,
    ):
        self._inner = inner_sink
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval_seconds
        self._buffer: List[StreamEvent[T]] = []
        self._last_flush = time.time()
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def write(self, event: StreamEvent[T]) -> None:
        async with self._get_lock():
            self._buffer.append(event)
            await self._maybe_flush()

    async def write_batch(self, events: List[StreamEvent[T]]) -> None:
        async with self._get_lock():
            self._buffer.extend(events)
            await self._maybe_flush()

    async def _maybe_flush(self) -> None:
        should_flush = (
            len(self._buffer) >= self._buffer_size
            or time.time() - self._last_flush >= self._flush_interval
        )
        if should_flush and self._buffer:
            await self._flush()

    async def _flush(self) -> None:
        if self._buffer:
            await self._inner.write_batch(self._buffer)
            self._buffer = []
            self._last_flush = time.time()

    async def close(self) -> None:
        async with self._get_lock():
            await self._flush()
            await self._inner.close()


# Stream Operators

class StreamOperator(ABC, Generic[T, R]):
    """Base class for stream operators."""

    @abstractmethod
    async def process(self, event: StreamEvent[T]) -> Optional[StreamEvent[R]]:
        """Process a single event."""
        pass


class MapOperator(StreamOperator[T, R]):
    """Map transformation operator."""

    def __init__(self, func: Callable[[T], R]):
        self.func = func

    async def process(self, event: StreamEvent[T]) -> Optional[StreamEvent[R]]:
        try:
            result = self.func(event.data)
            return StreamEvent(
                data=result,
                timestamp=event.timestamp,
                key=event.key,
                headers=event.headers,
            )
        except Exception as e:
            logger.error(f"Map error: {e}")
            return None


class FilterOperator(StreamOperator[T, T]):
    """Filter operator."""

    def __init__(self, predicate: Callable[[T], bool]):
        self.predicate = predicate

    async def process(self, event: StreamEvent[T]) -> Optional[StreamEvent[T]]:
        if self.predicate(event.data):
            return event
        return None


class FlatMapOperator(StreamOperator[T, R]):
    """FlatMap operator that can emit multiple events."""

    def __init__(self, func: Callable[[T], List[R]]):
        self.func = func
        self._pending: Deque[StreamEvent[R]] = deque()

    async def process(self, event: StreamEvent[T]) -> Optional[StreamEvent[R]]:
        if self._pending:
            return self._pending.popleft()

        try:
            results = self.func(event.data)
            for result in results:
                self._pending.append(StreamEvent(
                    data=result,
                    timestamp=event.timestamp,
                    key=event.key,
                    headers=event.headers,
                ))

            if self._pending:
                return self._pending.popleft()
        except Exception as e:
            logger.error(f"FlatMap error: {e}")

        return None


class TransformerOperator(StreamOperator[Dict[str, Any], Dict[str, Any]]):
    """Operator using Transformer."""

    def __init__(self, transformer: Transformer[Dict[str, Any], Dict[str, Any]]):
        self.transformer = transformer

    async def process(
        self,
        event: StreamEvent[Dict[str, Any]],
    ) -> Optional[StreamEvent[Dict[str, Any]]]:
        try:
            result = self.transformer.transform(event.data)
            if result is None:
                return None
            return StreamEvent(
                data=result,
                timestamp=event.timestamp,
                key=event.key,
                headers=event.headers,
            )
        except Exception as e:
            logger.error(f"Transformer error: {e}")
            return None


# Window Processors

class WindowProcessor(Generic[T, R]):
    """Process events in windows."""

    def __init__(
        self,
        config: WindowConfig,
        aggregator: Callable[[List[T]], R],
    ):
        self.config = config
        self.aggregator = aggregator
        self._windows: Dict[str, Window[T]] = {}
        self._current_window_start: Optional[datetime] = None

    async def process(
        self,
        event: StreamEvent[T],
    ) -> Optional[StreamEvent[R]]:
        """Process event and return result if window closes."""
        window_key = self._get_window_key(event)

        if window_key not in self._windows:
            self._windows[window_key] = self._create_window(event)

        window = self._windows[window_key]
        window.add(event)

        # Check if window should close
        if self._should_close_window(window, event):
            result = self._close_window(window_key)
            return result

        return None

    def _get_window_key(self, event: StreamEvent[T]) -> str:
        """Get window key based on config."""
        if self.config.window_type == WindowType.TUMBLING:
            window_start = self._align_to_window(event.timestamp)
            return f"tumbling_{window_start.isoformat()}"

        elif self.config.window_type == WindowType.SLIDING:
            # Sliding windows are more complex - simplified here
            window_start = self._align_to_window(event.timestamp)
            return f"sliding_{window_start.isoformat()}"

        elif self.config.window_type == WindowType.COUNT:
            return "count_current"

        return "default"

    def _align_to_window(self, timestamp: datetime) -> datetime:
        """Align timestamp to window boundary."""
        size_seconds = self.config.size.total_seconds()
        epoch = timestamp.timestamp()
        aligned_epoch = (epoch // size_seconds) * size_seconds
        return datetime.fromtimestamp(aligned_epoch)

    def _create_window(self, event: StreamEvent[T]) -> Window[T]:
        """Create new window."""
        if self.config.window_type in (WindowType.TUMBLING, WindowType.SLIDING):
            start = self._align_to_window(event.timestamp)
            end = start + self.config.size
        else:
            start = event.timestamp
            end = start + self.config.size

        return Window(start=start, end=end)

    def _should_close_window(
        self,
        window: Window[T],
        event: StreamEvent[T],
    ) -> bool:
        """Check if window should close."""
        if self.config.window_type == WindowType.TUMBLING:
            return event.timestamp >= window.end

        elif self.config.window_type == WindowType.COUNT:
            count = self.config.count or 10
            return window.count >= count

        return False

    def _close_window(self, window_key: str) -> Optional[StreamEvent[R]]:
        """Close window and emit result."""
        window = self._windows.pop(window_key, None)
        if not window or not window.events:
            return None

        data_list = [e.data for e in window.events]
        result = self.aggregator(data_list)

        return StreamEvent(
            data=result,
            timestamp=window.end,
            key=window_key,
        )

    async def flush(self) -> List[StreamEvent[R]]:
        """Flush all open windows."""
        results = []
        for key in list(self._windows.keys()):
            result = self._close_window(key)
            if result:
                results.append(result)
        return results


# Stream Pipeline

class StreamPipeline(Generic[T, R]):
    """Stream processing pipeline."""

    def __init__(
        self,
        source: StreamSource[T],
        operators: Optional[List[StreamOperator]] = None,
        sink: Optional[StreamSink[R]] = None,
    ):
        self.source = source
        self.operators = operators or []
        self.sink = sink
        self._running = False
        self._processed_count = 0
        self._error_count = 0

    def map(self, func: Callable) -> "StreamPipeline":
        """Add map operator."""
        self.operators.append(MapOperator(func))
        return self

    def filter(self, predicate: Callable) -> "StreamPipeline":
        """Add filter operator."""
        self.operators.append(FilterOperator(predicate))
        return self

    def transform(self, transformer: Transformer) -> "StreamPipeline":
        """Add transformer operator."""
        self.operators.append(TransformerOperator(transformer))
        return self

    def to(self, sink: StreamSink) -> "StreamPipeline":
        """Set output sink."""
        self.sink = sink
        return self

    async def run(self) -> None:
        """Run the pipeline."""
        self._running = True
        logger.info("Starting stream pipeline")

        try:
            async for event in self.source.read():
                if not self._running:
                    break

                result = await self._process_event(event)
                if result and self.sink:
                    await self.sink.write(result)

                self._processed_count += 1

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self._error_count += 1
            raise
        finally:
            self._running = False
            await self._cleanup()

    async def _process_event(self, event: StreamEvent) -> Optional[StreamEvent]:
        """Process event through operators."""
        current: Any = event

        for operator in self.operators:
            if current is None:
                return None

            try:
                current = await operator.process(current)
            except Exception as e:
                logger.error(f"Operator error: {e}")
                self._error_count += 1
                return None

        return current

    async def _cleanup(self) -> None:
        """Cleanup resources."""
        await self.source.close()
        if self.sink:
            await self.sink.close()

    def stop(self) -> None:
        """Stop the pipeline."""
        self._running = False

    @property
    def stats(self) -> Dict[str, int]:
        """Get pipeline statistics."""
        return {
            "processed": self._processed_count,
            "errors": self._error_count,
        }


# Parallel Processing

class ParallelStreamPipeline(Generic[T, R]):
    """Parallel stream processing."""

    def __init__(
        self,
        source: StreamSource[T],
        operators: List[StreamOperator],
        sink: StreamSink[R],
        parallelism: int = 4,
    ):
        self.source = source
        self.operators = operators
        self.sink = sink
        self.parallelism = parallelism
        self._running = False
        self._queue: asyncio.Queue[Optional[StreamEvent[T]]] = asyncio.Queue()

    async def run(self) -> None:
        """Run parallel pipeline."""
        self._running = True

        # Start workers
        workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.parallelism)
        ]

        # Read and enqueue
        try:
            async for event in self.source.read():
                if not self._running:
                    break
                await self._queue.put(event)

            # Signal completion
            for _ in range(self.parallelism):
                await self._queue.put(None)

            # Wait for workers
            await asyncio.gather(*workers)

        finally:
            self._running = False
            await self.source.close()
            await self.sink.close()

    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine."""
        while self._running:
            event = await self._queue.get()
            if event is None:
                break

            result = await self._process(event)
            if result:
                await self.sink.write(result)

    async def _process(self, event: StreamEvent[T]) -> Optional[StreamEvent[R]]:
        """Process event through operators."""
        current: Any = event
        for op in self.operators:
            if current is None:
                return None
            current = await op.process(current)
        return current

    def stop(self) -> None:
        self._running = False
