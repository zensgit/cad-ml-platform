"""Stream processing for Vision Provider system.

This module provides stream processing features including:
- Real-time event streaming
- Window computations
- Stream aggregations
- Event time processing
- Stream joins
"""

import asyncio
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Deque, Dict, Generic, Iterator, List, Optional, Set, Tuple, Type, TypeVar, Union

from .base import VisionDescription, VisionProvider


class WindowType(Enum):
    """Window types."""

    TUMBLING = "tumbling"
    SLIDING = "sliding"
    SESSION = "session"
    GLOBAL = "global"


class TriggerType(Enum):
    """Trigger types."""

    PROCESSING_TIME = "processing_time"
    EVENT_TIME = "event_time"
    COUNT = "count"
    CONTINUOUS = "continuous"


class StreamState(Enum):
    """Stream state."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


@dataclass
class StreamEvent(Generic[T]):
    """Stream event."""

    event_id: str
    data: T
    timestamp: datetime = field(default_factory=datetime.now)
    event_time: Optional[datetime] = None
    key: Optional[str] = None
    partition: int = 0
    headers: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "event_time": self.event_time.isoformat() if self.event_time else None,
            "key": self.key,
            "partition": self.partition,
            "headers": dict(self.headers),
        }


@dataclass
class WindowResult(Generic[T]):
    """Window computation result."""

    window_id: str
    window_type: WindowType
    start_time: datetime
    end_time: datetime
    result: T
    event_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class Window(ABC, Generic[T]):
    """Abstract window."""

    @abstractmethod
    def add(self, event: StreamEvent[T]) -> None:
        """Add event to window."""
        pass

    @abstractmethod
    def get_events(self) -> List[StreamEvent[T]]:
        """Get events in window."""
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        """Check if window is closed."""
        pass

    @abstractmethod
    def trigger(self) -> bool:
        """Check if window should trigger."""
        pass


class TumblingWindow(Window[T]):
    """Tumbling window - non-overlapping fixed-size windows."""

    def __init__(self, size: timedelta) -> None:
        """Initialize window."""
        self._size = size
        self._events: List[StreamEvent[T]] = []
        self._start_time: Optional[datetime] = None
        self._closed = False

    def add(self, event: StreamEvent[T]) -> None:
        """Add event to window."""
        if self._closed:
            return

        if self._start_time is None:
            self._start_time = event.timestamp

        # Check if event belongs to this window
        window_end = self._start_time + self._size
        if event.timestamp < window_end:
            self._events.append(event)
        else:
            self._closed = True

    def get_events(self) -> List[StreamEvent[T]]:
        """Get events in window."""
        return list(self._events)

    def is_closed(self) -> bool:
        """Check if window is closed."""
        if self._start_time is None:
            return False
        return datetime.now() >= self._start_time + self._size

    def trigger(self) -> bool:
        """Check if window should trigger."""
        return self.is_closed()

    @property
    def start_time(self) -> Optional[datetime]:
        """Get window start time."""
        return self._start_time

    @property
    def end_time(self) -> Optional[datetime]:
        """Get window end time."""
        if self._start_time is None:
            return None
        return self._start_time + self._size


class SlidingWindow(Window[T]):
    """Sliding window - overlapping windows."""

    def __init__(self, size: timedelta, slide: timedelta) -> None:
        """Initialize window."""
        self._size = size
        self._slide = slide
        self._events: Deque[StreamEvent[T]] = deque()
        self._last_trigger: Optional[datetime] = None

    def add(self, event: StreamEvent[T]) -> None:
        """Add event to window."""
        self._events.append(event)

        # Remove events outside window
        cutoff = event.timestamp - self._size
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()

    def get_events(self) -> List[StreamEvent[T]]:
        """Get events in window."""
        return list(self._events)

    def is_closed(self) -> bool:
        """Sliding windows never close."""
        return False

    def trigger(self) -> bool:
        """Check if window should trigger."""
        now = datetime.now()
        if self._last_trigger is None:
            self._last_trigger = now
            return True
        if now >= self._last_trigger + self._slide:
            self._last_trigger = now
            return True
        return False


class SessionWindow(Window[T]):
    """Session window - gap-based windows."""

    def __init__(self, gap: timedelta) -> None:
        """Initialize window."""
        self._gap = gap
        self._events: List[StreamEvent[T]] = []
        self._last_event_time: Optional[datetime] = None
        self._closed = False

    def add(self, event: StreamEvent[T]) -> None:
        """Add event to window."""
        if self._closed:
            return

        now = event.timestamp
        if self._last_event_time is not None:
            if now - self._last_event_time > self._gap:
                self._closed = True
                return

        self._events.append(event)
        self._last_event_time = now

    def get_events(self) -> List[StreamEvent[T]]:
        """Get events in window."""
        return list(self._events)

    def is_closed(self) -> bool:
        """Check if window is closed."""
        if self._last_event_time is None:
            return False
        return datetime.now() - self._last_event_time > self._gap

    def trigger(self) -> bool:
        """Check if window should trigger."""
        return self.is_closed()


class CountWindow(Window[T]):
    """Count-based window."""

    def __init__(self, count: int) -> None:
        """Initialize window."""
        self._count = count
        self._events: List[StreamEvent[T]] = []

    def add(self, event: StreamEvent[T]) -> None:
        """Add event to window."""
        if len(self._events) < self._count:
            self._events.append(event)

    def get_events(self) -> List[StreamEvent[T]]:
        """Get events in window."""
        return list(self._events)

    def is_closed(self) -> bool:
        """Check if window is closed."""
        return len(self._events) >= self._count

    def trigger(self) -> bool:
        """Check if window should trigger."""
        return self.is_closed()


class StreamOperator(ABC, Generic[T]):
    """Abstract stream operator."""

    @abstractmethod
    def process(self, event: StreamEvent[T]) -> Optional[StreamEvent[Any]]:
        """Process event."""
        pass


class MapOperator(StreamOperator[T]):
    """Map operator."""

    def __init__(self, func: Callable[[T], Any]) -> None:
        """Initialize operator."""
        self._func = func

    def process(self, event: StreamEvent[T]) -> Optional[StreamEvent[Any]]:
        """Process event."""
        try:
            result = self._func(event.data)
            return StreamEvent(
                event_id=event.event_id,
                data=result,
                timestamp=event.timestamp,
                event_time=event.event_time,
                key=event.key,
                partition=event.partition,
                headers=event.headers,
            )
        except Exception:
            return None


class FilterOperator(StreamOperator[T]):
    """Filter operator."""

    def __init__(self, predicate: Callable[[T], bool]) -> None:
        """Initialize operator."""
        self._predicate = predicate

    def process(self, event: StreamEvent[T]) -> Optional[StreamEvent[T]]:
        """Process event."""
        if self._predicate(event.data):
            return event
        return None


class FlatMapOperator(StreamOperator[T]):
    """Flat map operator."""

    def __init__(self, func: Callable[[T], List[Any]]) -> None:
        """Initialize operator."""
        self._func = func
        self._pending: List[StreamEvent[Any]] = []

    def process(self, event: StreamEvent[T]) -> Optional[StreamEvent[Any]]:
        """Process event."""
        if self._pending:
            return self._pending.pop(0)

        try:
            results = self._func(event.data)
            for result in results:
                self._pending.append(StreamEvent(
                    event_id=str(uuid.uuid4()),
                    data=result,
                    timestamp=event.timestamp,
                    key=event.key,
                ))
            if self._pending:
                return self._pending.pop(0)
        except Exception:
            pass
        return None


class KeyByOperator(StreamOperator[T]):
    """Key by operator."""

    def __init__(self, key_selector: Callable[[T], str]) -> None:
        """Initialize operator."""
        self._key_selector = key_selector

    def process(self, event: StreamEvent[T]) -> Optional[StreamEvent[T]]:
        """Process event."""
        key = self._key_selector(event.data)
        return StreamEvent(
            event_id=event.event_id,
            data=event.data,
            timestamp=event.timestamp,
            event_time=event.event_time,
            key=key,
            partition=hash(key) % 16,
            headers=event.headers,
        )


class Aggregator(ABC, Generic[T, V]):
    """Abstract aggregator."""

    @abstractmethod
    def add(self, value: T) -> None:
        """Add value to aggregation."""
        pass

    @abstractmethod
    def get_result(self) -> V:
        """Get aggregation result."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset aggregation."""
        pass


class SumAggregator(Aggregator[float, float]):
    """Sum aggregator."""

    def __init__(self) -> None:
        """Initialize aggregator."""
        self._sum = 0.0

    def add(self, value: float) -> None:
        """Add value."""
        self._sum += value

    def get_result(self) -> float:
        """Get sum."""
        return self._sum

    def reset(self) -> None:
        """Reset sum."""
        self._sum = 0.0


class CountAggregator(Aggregator[Any, int]):
    """Count aggregator."""

    def __init__(self) -> None:
        """Initialize aggregator."""
        self._count = 0

    def add(self, value: Any) -> None:
        """Add value."""
        self._count += 1

    def get_result(self) -> int:
        """Get count."""
        return self._count

    def reset(self) -> None:
        """Reset count."""
        self._count = 0


class AverageAggregator(Aggregator[float, float]):
    """Average aggregator."""

    def __init__(self) -> None:
        """Initialize aggregator."""
        self._sum = 0.0
        self._count = 0

    def add(self, value: float) -> None:
        """Add value."""
        self._sum += value
        self._count += 1

    def get_result(self) -> float:
        """Get average."""
        if self._count == 0:
            return 0.0
        return self._sum / self._count

    def reset(self) -> None:
        """Reset average."""
        self._sum = 0.0
        self._count = 0


class MinAggregator(Aggregator[float, float]):
    """Min aggregator."""

    def __init__(self) -> None:
        """Initialize aggregator."""
        self._min: Optional[float] = None

    def add(self, value: float) -> None:
        """Add value."""
        if self._min is None or value < self._min:
            self._min = value

    def get_result(self) -> float:
        """Get min."""
        return self._min if self._min is not None else 0.0

    def reset(self) -> None:
        """Reset min."""
        self._min = None


class MaxAggregator(Aggregator[float, float]):
    """Max aggregator."""

    def __init__(self) -> None:
        """Initialize aggregator."""
        self._max: Optional[float] = None

    def add(self, value: float) -> None:
        """Add value."""
        if self._max is None or value > self._max:
            self._max = value

    def get_result(self) -> float:
        """Get max."""
        return self._max if self._max is not None else 0.0

    def reset(self) -> None:
        """Reset max."""
        self._max = None


class Stream(Generic[T]):
    """Stream processing pipeline."""

    def __init__(self) -> None:
        """Initialize stream."""
        self._operators: List[StreamOperator[Any]] = []
        self._sinks: List[Callable[[StreamEvent[Any]], None]] = []
        self._state = StreamState.CREATED
        self._event_queue: queue.Queue[StreamEvent[T]] = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def map(self, func: Callable[[T], Any]) -> "Stream[Any]":
        """Add map operator."""
        self._operators.append(MapOperator(func))
        return self  # type: ignore

    def filter(self, predicate: Callable[[T], bool]) -> "Stream[T]":
        """Add filter operator."""
        self._operators.append(FilterOperator(predicate))
        return self

    def flat_map(self, func: Callable[[T], List[Any]]) -> "Stream[Any]":
        """Add flat map operator."""
        self._operators.append(FlatMapOperator(func))
        return self  # type: ignore

    def key_by(self, key_selector: Callable[[T], str]) -> "Stream[T]":
        """Add key by operator."""
        self._operators.append(KeyByOperator(key_selector))
        return self

    def add_sink(self, sink: Callable[[StreamEvent[Any]], None]) -> "Stream[T]":
        """Add sink."""
        self._sinks.append(sink)
        return self

    def process_event(self, event: StreamEvent[T]) -> List[StreamEvent[Any]]:
        """Process event through operators."""
        current: Optional[StreamEvent[Any]] = event
        results: List[StreamEvent[Any]] = []

        for operator in self._operators:
            if current is None:
                break
            current = operator.process(current)

        if current is not None:
            results.append(current)
            for sink in self._sinks:
                sink(current)

        return results

    def emit(self, data: T, key: Optional[str] = None) -> None:
        """Emit event to stream."""
        event = StreamEvent(
            event_id=str(uuid.uuid4()),
            data=data,
            key=key,
        )
        self._event_queue.put(event)

    def start(self) -> None:
        """Start stream processing."""
        if self._running:
            return

        self._running = True
        self._state = StreamState.RUNNING
        self._thread = threading.Thread(target=self._process_loop)
        self._thread.daemon = True
        self._thread.start()

    def stop(self) -> None:
        """Stop stream processing."""
        self._running = False
        self._state = StreamState.STOPPED
        if self._thread:
            self._thread.join(timeout=1.0)

    def _process_loop(self) -> None:
        """Processing loop."""
        while self._running:
            try:
                event = self._event_queue.get(timeout=0.1)
                self.process_event(event)
            except queue.Empty:
                continue
            except Exception:
                self._state = StreamState.ERROR

    @property
    def state(self) -> StreamState:
        """Get stream state."""
        return self._state


class WindowedStream(Generic[T]):
    """Windowed stream."""

    def __init__(
        self,
        stream: Stream[T],
        window_type: WindowType,
        window_size: Optional[timedelta] = None,
        window_slide: Optional[timedelta] = None,
        window_gap: Optional[timedelta] = None,
        window_count: Optional[int] = None,
    ) -> None:
        """Initialize windowed stream."""
        self._stream = stream
        self._window_type = window_type
        self._window_size = window_size
        self._window_slide = window_slide
        self._window_gap = window_gap
        self._window_count = window_count
        self._windows: Dict[str, Window[T]] = {}
        self._aggregator: Optional[Aggregator[Any, Any]] = None
        self._value_extractor: Optional[Callable[[T], Any]] = None

    def aggregate(
        self,
        aggregator: Aggregator[Any, Any],
        value_extractor: Optional[Callable[[T], Any]] = None,
    ) -> "WindowedStream[T]":
        """Set aggregation."""
        self._aggregator = aggregator
        self._value_extractor = value_extractor
        return self

    def _create_window(self) -> Window[T]:
        """Create new window."""
        if self._window_type == WindowType.TUMBLING and self._window_size:
            return TumblingWindow(self._window_size)
        elif self._window_type == WindowType.SLIDING and self._window_size and self._window_slide:
            return SlidingWindow(self._window_size, self._window_slide)
        elif self._window_type == WindowType.SESSION and self._window_gap:
            return SessionWindow(self._window_gap)
        elif self._window_count:
            return CountWindow(self._window_count)
        else:
            raise ValueError("Invalid window configuration")

    def process_event(self, event: StreamEvent[T]) -> Optional[WindowResult[Any]]:
        """Process event through window."""
        key = event.key or "default"

        if key not in self._windows or self._windows[key].is_closed():
            self._windows[key] = self._create_window()

        window = self._windows[key]
        window.add(event)

        if window.trigger() and self._aggregator:
            events = window.get_events()
            self._aggregator.reset()

            for e in events:
                value = self._value_extractor(e.data) if self._value_extractor else e.data
                self._aggregator.add(value)

            result = WindowResult(
                window_id=str(uuid.uuid4()),
                window_type=self._window_type,
                start_time=events[0].timestamp if events else datetime.now(),
                end_time=events[-1].timestamp if events else datetime.now(),
                result=self._aggregator.get_result(),
                event_count=len(events),
            )

            # Create new window for next batch
            self._windows[key] = self._create_window()

            return result

        return None


class StreamBuilder:
    """Stream builder."""

    def __init__(self) -> None:
        """Initialize builder."""
        self._stream: Stream[Any] = Stream()

    def from_source(self, source: Callable[[], Iterator[Any]]) -> "StreamBuilder":
        """Set source."""
        # Source will be connected when stream starts
        return self

    def map(self, func: Callable[[Any], Any]) -> "StreamBuilder":
        """Add map operator."""
        self._stream.map(func)
        return self

    def filter(self, predicate: Callable[[Any], bool]) -> "StreamBuilder":
        """Add filter operator."""
        self._stream.filter(predicate)
        return self

    def key_by(self, key_selector: Callable[[Any], str]) -> "StreamBuilder":
        """Add key by operator."""
        self._stream.key_by(key_selector)
        return self

    def window_tumbling(self, size: timedelta) -> WindowedStream[Any]:
        """Create tumbling window."""
        return WindowedStream(
            self._stream,
            WindowType.TUMBLING,
            window_size=size,
        )

    def window_sliding(self, size: timedelta, slide: timedelta) -> WindowedStream[Any]:
        """Create sliding window."""
        return WindowedStream(
            self._stream,
            WindowType.SLIDING,
            window_size=size,
            window_slide=slide,
        )

    def window_session(self, gap: timedelta) -> WindowedStream[Any]:
        """Create session window."""
        return WindowedStream(
            self._stream,
            WindowType.SESSION,
            window_gap=gap,
        )

    def window_count(self, count: int) -> WindowedStream[Any]:
        """Create count window."""
        return WindowedStream(
            self._stream,
            WindowType.TUMBLING,
            window_count=count,
        )

    def to_sink(self, sink: Callable[[StreamEvent[Any]], None]) -> "StreamBuilder":
        """Add sink."""
        self._stream.add_sink(sink)
        return self

    def build(self) -> Stream[Any]:
        """Build stream."""
        return self._stream


class StreamingVisionProvider(VisionProvider):
    """Vision provider with stream processing."""

    def __init__(
        self,
        provider: VisionProvider,
        stream: Optional[Stream[Dict[str, Any]]] = None,
    ) -> None:
        """Initialize provider."""
        self._provider = provider
        self._stream = stream or Stream()
        self._results: List[Dict[str, Any]] = []

        # Add sink to collect results
        self._stream.add_sink(lambda e: self._results.append(e.data))

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"streaming_{self._provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image and emit to stream."""
        result = await self._provider.analyze_image(image_data, include_description)

        # Emit to stream
        self._stream.emit({
            "summary": result.summary,
            "details": result.details,
            "confidence": result.confidence,
            "timestamp": datetime.now().isoformat(),
        })

        return result

    def get_stream(self) -> Stream[Dict[str, Any]]:
        """Get stream."""
        return self._stream

    def get_results(self) -> List[Dict[str, Any]]:
        """Get collected results."""
        return list(self._results)

    def start_streaming(self) -> None:
        """Start stream processing."""
        self._stream.start()

    def stop_streaming(self) -> None:
        """Stop stream processing."""
        self._stream.stop()


def create_stream() -> StreamBuilder:
    """Create stream builder.

    Returns:
        Stream builder
    """
    return StreamBuilder()


def create_streaming_provider(
    provider: VisionProvider,
) -> StreamingVisionProvider:
    """Create streaming vision provider.

    Args:
        provider: Provider to wrap

    Returns:
        Streaming provider
    """
    return StreamingVisionProvider(provider)
