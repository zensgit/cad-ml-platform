"""OpenTelemetry Span Management.

Provides span creation and management:
- Span lifecycle
- Span attributes and events
- Span status and errors
"""

from __future__ import annotations

import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, TypeVar, Union

from src.core.tracing.context import (
    SpanContext,
    SpanId,
    TraceId,
    attach_context,
    detach_context,
    get_current_context,
)

logger = logging.getLogger(__name__)


class SpanKind(Enum):
    """Type of span."""
    INTERNAL = 0  # Default internal operation
    SERVER = 1    # Server-side handling of a synchronous request
    CLIENT = 2    # Client-side of a synchronous request
    PRODUCER = 3  # Initiator of an asynchronous request
    CONSUMER = 4  # Handler of an asynchronous request


class StatusCode(Enum):
    """Span status code."""
    UNSET = 0
    OK = 1
    ERROR = 2


@dataclass
class Status:
    """Span status."""
    code: StatusCode = StatusCode.UNSET
    description: str = ""

    @classmethod
    def ok(cls) -> "Status":
        return cls(code=StatusCode.OK)

    @classmethod
    def error(cls, description: str = "") -> "Status":
        return cls(code=StatusCode.ERROR, description=description)


@dataclass
class SpanEvent:
    """An event that occurred during a span."""
    name: str
    timestamp: float  # Unix timestamp in seconds
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanLink:
    """A link to another span."""
    context: SpanContext
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """A trace span representing a unit of work."""
    name: str
    context: SpanContext
    parent_id: Optional[SpanId] = None
    kind: SpanKind = SpanKind.INTERNAL
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: Status = field(default_factory=Status)
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    links: List[SpanLink] = field(default_factory=list)
    resource: Dict[str, Any] = field(default_factory=dict)

    # Internal state
    _is_recording: bool = field(default=True, repr=False)
    _is_ended: bool = field(default=False, repr=False)

    def is_recording(self) -> bool:
        """Check if span is still recording."""
        return self._is_recording and not self._is_ended

    def set_attribute(self, key: str, value: Any) -> "Span":
        """Set a single attribute."""
        if self.is_recording():
            self.attributes[key] = value
        return self

    def set_attributes(self, attributes: Dict[str, Any]) -> "Span":
        """Set multiple attributes."""
        if self.is_recording():
            self.attributes.update(attributes)
        return self

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> "Span":
        """Add an event to the span."""
        if self.is_recording():
            self.events.append(SpanEvent(
                name=name,
                timestamp=timestamp or time.time(),
                attributes=attributes or {},
            ))
        return self

    def add_link(
        self,
        context: SpanContext,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "Span":
        """Add a link to another span."""
        if self.is_recording():
            self.links.append(SpanLink(
                context=context,
                attributes=attributes or {},
            ))
        return self

    def set_status(self, code: StatusCode, description: str = "") -> "Span":
        """Set span status."""
        if self.is_recording():
            # Only allow setting ERROR if not already OK
            if self.status.code != StatusCode.OK or code == StatusCode.OK:
                self.status = Status(code=code, description=description)
        return self

    def record_exception(
        self,
        exception: BaseException,
        attributes: Optional[Dict[str, Any]] = None,
        escaped: bool = False,
    ) -> "Span":
        """Record an exception as an event."""
        if self.is_recording():
            exc_attributes = {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
                "exception.stacktrace": "".join(traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )),
                "exception.escaped": escaped,
            }
            if attributes:
                exc_attributes.update(attributes)

            self.add_event("exception", exc_attributes)
            self.set_status(StatusCode.ERROR, str(exception))
        return self

    def end(self, end_time: Optional[float] = None) -> None:
        """End the span."""
        if not self._is_ended:
            self.end_time = end_time or time.time()
            self._is_ended = True
            self._is_recording = False

    @property
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for export."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id.to_hex(),
            "span_id": self.context.span_id.to_hex(),
            "parent_span_id": self.parent_id.to_hex() if self.parent_id else None,
            "kind": self.kind.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": {
                "code": self.status.code.name,
                "description": self.status.description,
            },
            "attributes": self.attributes,
            "events": [
                {
                    "name": e.name,
                    "timestamp": e.timestamp,
                    "attributes": e.attributes,
                }
                for e in self.events
            ],
            "links": [
                {
                    "trace_id": l.context.trace_id.to_hex(),
                    "span_id": l.context.span_id.to_hex(),
                    "attributes": l.attributes,
                }
                for l in self.links
            ],
            "resource": self.resource,
        }


class Tracer:
    """Creates and manages spans."""

    def __init__(
        self,
        name: str,
        version: str = "",
        resource: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.version = version
        self.resource = resource or {}
        self._spans: List[Span] = []
        self._processors: List["SpanProcessor"] = []

    def add_processor(self, processor: "SpanProcessor") -> None:
        """Add a span processor."""
        self._processors.append(processor)

    def start_span(
        self,
        name: str,
        context: Optional[SpanContext] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[List[SpanLink]] = None,
        start_time: Optional[float] = None,
    ) -> Span:
        """Start a new span."""
        parent_context = context or get_current_context()

        if parent_context:
            span_context = parent_context.create_child()
            parent_id = parent_context.span_id
        else:
            span_context = SpanContext.create_root()
            parent_id = None

        span = Span(
            name=name,
            context=span_context,
            parent_id=parent_id,
            kind=kind,
            start_time=start_time or time.time(),
            attributes=attributes or {},
            links=links or [],
            resource={
                "service.name": self.name,
                "service.version": self.version,
                **self.resource,
            },
        )

        # Notify processors
        for processor in self._processors:
            processor.on_start(span)

        self._spans.append(span)
        return span

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[List[SpanLink]] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Generator[Span, None, None]:
        """Context manager that creates a span and sets it as current."""
        span = self.start_span(name, kind=kind, attributes=attributes, links=links)
        token = attach_context(span.context)

        try:
            yield span
        except Exception as e:
            if record_exception:
                span.record_exception(e, escaped=True)
            if set_status_on_exception:
                span.set_status(StatusCode.ERROR, str(e))
            raise
        finally:
            if end_on_exit:
                span.end()
                # Notify processors
                for processor in self._processors:
                    processor.on_end(span)
            detach_context(token)

    def get_spans(self) -> List[Span]:
        """Get all spans created by this tracer."""
        return self._spans.copy()

    def clear_spans(self) -> None:
        """Clear all spans."""
        self._spans.clear()


class SpanProcessor:
    """Base class for span processors."""

    def on_start(self, span: Span) -> None:
        """Called when a span is started."""
        pass

    def on_end(self, span: Span) -> None:
        """Called when a span is ended."""
        pass

    def shutdown(self) -> None:
        """Shutdown the processor."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any pending spans."""
        return True


class SimpleSpanProcessor(SpanProcessor):
    """Processor that exports spans immediately."""

    def __init__(self, exporter: "SpanExporter"):
        self.exporter = exporter

    def on_end(self, span: Span) -> None:
        if span.context.is_sampled():
            self.exporter.export([span])

    def shutdown(self) -> None:
        self.exporter.shutdown()


class BatchSpanProcessor(SpanProcessor):
    """Processor that batches spans before export."""

    def __init__(
        self,
        exporter: "SpanExporter",
        max_queue_size: int = 2048,
        max_export_batch_size: int = 512,
        schedule_delay_millis: int = 5000,
    ):
        self.exporter = exporter
        self.max_queue_size = max_queue_size
        self.max_export_batch_size = max_export_batch_size
        self.schedule_delay_millis = schedule_delay_millis
        self._queue: List[Span] = []
        self._last_export: float = time.time()

    def on_end(self, span: Span) -> None:
        if not span.context.is_sampled():
            return

        if len(self._queue) >= self.max_queue_size:
            # Drop oldest spans
            self._queue = self._queue[-(self.max_queue_size - 1):]

        self._queue.append(span)

        # Export if batch is full or enough time has passed
        if (
            len(self._queue) >= self.max_export_batch_size or
            (time.time() - self._last_export) * 1000 >= self.schedule_delay_millis
        ):
            self.force_flush()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        if not self._queue:
            return True

        batch = self._queue[:self.max_export_batch_size]
        self._queue = self._queue[self.max_export_batch_size:]

        try:
            self.exporter.export(batch)
            self._last_export = time.time()
            return True
        except Exception as e:
            logger.error(f"Failed to export spans: {e}")
            return False

    def shutdown(self) -> None:
        self.force_flush()
        self.exporter.shutdown()


class SpanExporter:
    """Base class for span exporters."""

    def export(self, spans: Sequence[Span]) -> None:
        """Export spans."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass


class ConsoleSpanExporter(SpanExporter):
    """Exports spans to console for debugging."""

    def __init__(self, pretty: bool = True):
        self.pretty = pretty

    def export(self, spans: Sequence[Span]) -> None:
        for span in spans:
            data = span.to_dict()
            if self.pretty:
                import json
                print(json.dumps(data, indent=2, default=str))
            else:
                print(data)


class InMemorySpanExporter(SpanExporter):
    """Exports spans to memory for testing."""

    def __init__(self):
        self._spans: List[Span] = []

    def export(self, spans: Sequence[Span]) -> None:
        self._spans.extend(spans)

    def get_spans(self) -> List[Span]:
        return self._spans.copy()

    def clear(self) -> None:
        self._spans.clear()


# Tracer provider
class TracerProvider:
    """Provides tracers for different components."""

    def __init__(self, resource: Optional[Dict[str, Any]] = None):
        self.resource = resource or {}
        self._tracers: Dict[str, Tracer] = {}
        self._processors: List[SpanProcessor] = []

    def add_span_processor(self, processor: SpanProcessor) -> None:
        """Add a span processor to all tracers."""
        self._processors.append(processor)
        for tracer in self._tracers.values():
            tracer.add_processor(processor)

    def get_tracer(
        self,
        name: str,
        version: str = "",
    ) -> Tracer:
        """Get or create a tracer."""
        key = f"{name}:{version}"
        if key not in self._tracers:
            tracer = Tracer(name, version, self.resource)
            for processor in self._processors:
                tracer.add_processor(processor)
            self._tracers[key] = tracer
        return self._tracers[key]

    def shutdown(self) -> None:
        """Shutdown all processors."""
        for processor in self._processors:
            processor.shutdown()


# Global tracer provider
_tracer_provider: Optional[TracerProvider] = None


def get_tracer_provider() -> TracerProvider:
    """Get the global tracer provider."""
    global _tracer_provider
    if _tracer_provider is None:
        _tracer_provider = TracerProvider()
    return _tracer_provider


def set_tracer_provider(provider: TracerProvider) -> None:
    """Set the global tracer provider."""
    global _tracer_provider
    _tracer_provider = provider


def get_tracer(name: str, version: str = "") -> Tracer:
    """Get a tracer from the global provider."""
    return get_tracer_provider().get_tracer(name, version)


# Decorator for tracing functions
F = TypeVar('F', bound=Callable[..., Any])


def trace(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """Decorator to trace a function."""
    def decorator(func: F) -> F:
        import functools

        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer(func.__module__)
            with tracer.start_as_current_span(span_name, kind=kind, attributes=attributes):
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer(func.__module__)
            with tracer.start_as_current_span(span_name, kind=kind, attributes=attributes):
                return await func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper  # type: ignore

    return decorator
