"""Distributed tracing for Vision Provider system.

This module provides distributed tracing capabilities including:
- OpenTelemetry-compatible tracing
- Span propagation across services
- Trace context management
- Sampling strategies
- Trace exporters
"""

import asyncio
import random
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar

from .base import VisionDescription, VisionProvider


class SpanKind(Enum):
    """Kind of span in trace."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(Enum):
    """Status of a span."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class SamplingDecision(Enum):
    """Sampling decision for trace."""

    DROP = "drop"
    RECORD_ONLY = "record_only"
    RECORD_AND_SAMPLE = "record_and_sample"


@dataclass
class TraceId:
    """Trace identifier."""

    value: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __str__(self) -> str:
        """Return string representation."""
        return self.value

    def __hash__(self) -> int:
        """Return hash."""
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if isinstance(other, TraceId):
            return self.value == other.value
        return False


@dataclass
class SpanId:
    """Span identifier."""

    value: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def __str__(self) -> str:
        """Return string representation."""
        return self.value

    def __hash__(self) -> int:
        """Return hash."""
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if isinstance(other, SpanId):
            return self.value == other.value
        return False


@dataclass
class SpanContext:
    """Context for span propagation."""

    trace_id: TraceId
    span_id: SpanId
    trace_flags: int = 1  # 1 = sampled
    trace_state: Dict[str, str] = field(default_factory=dict)
    is_remote: bool = False

    @property
    def is_valid(self) -> bool:
        """Check if context is valid."""
        return bool(self.trace_id.value and self.span_id.value)

    @property
    def is_sampled(self) -> bool:
        """Check if trace is sampled."""
        return (self.trace_flags & 1) == 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id.value,
            "span_id": self.span_id.value,
            "trace_flags": self.trace_flags,
            "trace_state": dict(self.trace_state),
            "is_remote": self.is_remote,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpanContext":
        """Create from dictionary."""
        return cls(
            trace_id=TraceId(value=data.get("trace_id", "")),
            span_id=SpanId(value=data.get("span_id", "")),
            trace_flags=data.get("trace_flags", 1),
            trace_state=data.get("trace_state", {}),
            is_remote=data.get("is_remote", False),
        )


@dataclass
class SpanEvent:
    """Event within a span."""

    name: str
    timestamp: datetime = field(default_factory=datetime.now)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "attributes": dict(self.attributes),
        }


@dataclass
class SpanLink:
    """Link to another span."""

    context: SpanContext
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "context": self.context.to_dict(),
            "attributes": dict(self.attributes),
        }


@dataclass
class TracingSpan:
    """Span representing an operation in a trace."""

    name: str
    context: SpanContext
    parent_context: Optional[SpanContext] = None
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.UNSET
    status_message: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    links: List[SpanLink] = field(default_factory=list)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute."""
        self.attributes[key] = value

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        """Set multiple attributes."""
        self.attributes.update(attributes)

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add event to span."""
        self.events.append(
            SpanEvent(
                name=name,
                attributes=attributes or {},
            )
        )

    def add_link(
        self,
        context: SpanContext,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add link to another span."""
        self.links.append(
            SpanLink(
                context=context,
                attributes=attributes or {},
            )
        )

    def set_status(
        self,
        status: SpanStatus,
        message: str = "",
    ) -> None:
        """Set span status."""
        self.status = status
        self.status_message = message

    def end(self, end_time: Optional[datetime] = None) -> None:
        """End the span."""
        self.end_time = end_time or datetime.now()

    @property
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "context": self.context.to_dict(),
            "parent_context": self.parent_context.to_dict() if self.parent_context else None,
            "kind": self.kind.value,
            "status": self.status.value,
            "status_message": self.status_message,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "attributes": dict(self.attributes),
            "events": [e.to_dict() for e in self.events],
            "links": [ln.to_dict() for ln in self.links],
        }


class Sampler(ABC):
    """Abstract base class for samplers."""

    @abstractmethod
    def should_sample(
        self,
        parent_context: Optional[SpanContext],
        trace_id: TraceId,
        name: str,
        attributes: Dict[str, Any],
    ) -> SamplingDecision:
        """Determine if trace should be sampled.

        Args:
            parent_context: Parent span context
            trace_id: Trace ID
            name: Span name
            attributes: Span attributes

        Returns:
            Sampling decision
        """
        pass


class AlwaysOnSampler(Sampler):
    """Sampler that always samples."""

    def should_sample(
        self,
        parent_context: Optional[SpanContext],
        trace_id: TraceId,
        name: str,
        attributes: Dict[str, Any],
    ) -> SamplingDecision:
        """Always sample."""
        return SamplingDecision.RECORD_AND_SAMPLE


class AlwaysOffSampler(Sampler):
    """Sampler that never samples."""

    def should_sample(
        self,
        parent_context: Optional[SpanContext],
        trace_id: TraceId,
        name: str,
        attributes: Dict[str, Any],
    ) -> SamplingDecision:
        """Never sample."""
        return SamplingDecision.DROP


class TraceIdRatioSampler(Sampler):
    """Sampler based on trace ID ratio."""

    def __init__(self, ratio: float = 0.1) -> None:
        """Initialize sampler.

        Args:
            ratio: Sampling ratio (0.0 to 1.0)
        """
        self._ratio = max(0.0, min(1.0, ratio))

    @property
    def ratio(self) -> float:
        """Return sampling ratio."""
        return self._ratio

    def should_sample(
        self,
        parent_context: Optional[SpanContext],
        trace_id: TraceId,
        name: str,
        attributes: Dict[str, Any],
    ) -> SamplingDecision:
        """Sample based on trace ID ratio."""
        # Deterministic based on trace ID
        hash_value = int(trace_id.value[:8], 16) / 0xFFFFFFFF
        if hash_value < self._ratio:
            return SamplingDecision.RECORD_AND_SAMPLE
        return SamplingDecision.DROP


class ParentBasedSampler(Sampler):
    """Sampler that follows parent decision."""

    def __init__(
        self,
        root_sampler: Sampler,
        remote_parent_sampled: Optional[Sampler] = None,
        remote_parent_not_sampled: Optional[Sampler] = None,
        local_parent_sampled: Optional[Sampler] = None,
        local_parent_not_sampled: Optional[Sampler] = None,
    ) -> None:
        """Initialize sampler.

        Args:
            root_sampler: Sampler for root spans
            remote_parent_sampled: Sampler for remote sampled parent
            remote_parent_not_sampled: Sampler for remote not-sampled parent
            local_parent_sampled: Sampler for local sampled parent
            local_parent_not_sampled: Sampler for local not-sampled parent
        """
        self._root = root_sampler
        self._remote_sampled = remote_parent_sampled or AlwaysOnSampler()
        self._remote_not_sampled = remote_parent_not_sampled or AlwaysOffSampler()
        self._local_sampled = local_parent_sampled or AlwaysOnSampler()
        self._local_not_sampled = local_parent_not_sampled or AlwaysOffSampler()

    def should_sample(
        self,
        parent_context: Optional[SpanContext],
        trace_id: TraceId,
        name: str,
        attributes: Dict[str, Any],
    ) -> SamplingDecision:
        """Sample based on parent."""
        if parent_context is None or not parent_context.is_valid:
            return self._root.should_sample(parent_context, trace_id, name, attributes)

        if parent_context.is_remote:
            if parent_context.is_sampled:
                return self._remote_sampled.should_sample(
                    parent_context, trace_id, name, attributes
                )
            else:
                return self._remote_not_sampled.should_sample(
                    parent_context, trace_id, name, attributes
                )
        else:
            if parent_context.is_sampled:
                return self._local_sampled.should_sample(parent_context, trace_id, name, attributes)
            else:
                return self._local_not_sampled.should_sample(
                    parent_context, trace_id, name, attributes
                )


class SpanExporter(ABC):
    """Abstract base class for span exporters."""

    @abstractmethod
    def export(self, spans: List[TracingSpan]) -> bool:
        """Export spans.

        Args:
            spans: Spans to export

        Returns:
            True if export succeeded
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown exporter."""
        pass


class InMemorySpanExporter(SpanExporter):
    """In-memory span exporter for testing."""

    def __init__(self) -> None:
        """Initialize exporter."""
        self._spans: List[TracingSpan] = []
        self._lock = threading.Lock()

    def export(self, spans: List[TracingSpan]) -> bool:
        """Export spans to memory."""
        with self._lock:
            self._spans.extend(spans)
        return True

    def shutdown(self) -> None:
        """Shutdown exporter."""
        pass

    def get_spans(self) -> List[TracingSpan]:
        """Get exported spans."""
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        """Clear exported spans."""
        with self._lock:
            self._spans.clear()


class ConsoleSpanExporter(SpanExporter):
    """Console span exporter for debugging."""

    def export(self, spans: List[TracingSpan]) -> bool:
        """Export spans to console."""
        for span in spans:
            print(f"[SPAN] {span.name}")
            print(f"  Trace ID: {span.context.trace_id}")
            print(f"  Span ID: {span.context.span_id}")
            print(
                f"  Duration: {span.duration_ms:.2f}ms" if span.duration_ms else "  Duration: N/A"
            )
            print(f"  Status: {span.status.value}")
            if span.attributes:
                print(f"  Attributes: {span.attributes}")
        return True

    def shutdown(self) -> None:
        """Shutdown exporter."""
        pass


class BatchSpanProcessor:
    """Batch processor for spans."""

    def __init__(
        self,
        exporter: SpanExporter,
        max_queue_size: int = 2048,
        batch_size: int = 512,
        export_timeout_ms: float = 30000,
    ) -> None:
        """Initialize processor.

        Args:
            exporter: Span exporter
            max_queue_size: Maximum queue size
            batch_size: Batch size for export
            export_timeout_ms: Export timeout
        """
        self._exporter = exporter
        self._max_queue_size = max_queue_size
        self._batch_size = batch_size
        self._export_timeout_ms = export_timeout_ms
        self._queue: List[TracingSpan] = []
        self._lock = threading.Lock()
        self._running = True

    def on_end(self, span: TracingSpan) -> None:
        """Process span when it ends.

        Args:
            span: Ended span
        """
        with self._lock:
            if len(self._queue) < self._max_queue_size:
                self._queue.append(span)

            if len(self._queue) >= self._batch_size:
                self._export_batch()

    def _export_batch(self) -> None:
        """Export a batch of spans."""
        if not self._queue:
            return

        batch = self._queue[: self._batch_size]
        self._queue = self._queue[self._batch_size :]

        try:
            self._exporter.export(batch)
        except Exception:
            pass

    def force_flush(self) -> bool:
        """Force flush all spans.

        Returns:
            True if flush succeeded
        """
        with self._lock:
            while self._queue:
                self._export_batch()
        return True

    def shutdown(self) -> None:
        """Shutdown processor."""
        self._running = False
        self.force_flush()
        self._exporter.shutdown()


@dataclass
class TracerConfig:
    """Configuration for tracer."""

    service_name: str = "vision-service"
    service_version: str = "1.0.0"
    deployment_environment: str = "development"
    sampler: Optional[Sampler] = None
    max_attributes: int = 128
    max_events: int = 128
    max_links: int = 128


class TracerProvider:
    """Provider for tracers."""

    def __init__(
        self,
        config: Optional[TracerConfig] = None,
    ) -> None:
        """Initialize provider.

        Args:
            config: Tracer configuration
        """
        self._config = config or TracerConfig()
        self._sampler = self._config.sampler or AlwaysOnSampler()
        self._processors: List[BatchSpanProcessor] = []
        self._tracers: Dict[str, "DistributedTracer"] = {}
        self._lock = threading.Lock()

    @property
    def config(self) -> TracerConfig:
        """Return tracer configuration."""
        return self._config

    def add_processor(self, processor: BatchSpanProcessor) -> None:
        """Add span processor.

        Args:
            processor: Span processor
        """
        self._processors.append(processor)

    def get_tracer(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> "DistributedTracer":
        """Get or create tracer.

        Args:
            name: Tracer name
            version: Tracer version

        Returns:
            Tracer instance
        """
        key = f"{name}:{version or ''}"

        with self._lock:
            if key not in self._tracers:
                self._tracers[key] = DistributedTracer(
                    name=name,
                    version=version,
                    provider=self,
                )
            return self._tracers[key]

    def on_span_end(self, span: TracingSpan) -> None:
        """Handle span end.

        Args:
            span: Ended span
        """
        for processor in self._processors:
            processor.on_end(span)

    def should_sample(
        self,
        parent_context: Optional[SpanContext],
        trace_id: TraceId,
        name: str,
        attributes: Dict[str, Any],
    ) -> SamplingDecision:
        """Determine sampling decision.

        Args:
            parent_context: Parent context
            trace_id: Trace ID
            name: Span name
            attributes: Span attributes

        Returns:
            Sampling decision
        """
        return self._sampler.should_sample(parent_context, trace_id, name, attributes)

    def shutdown(self) -> None:
        """Shutdown provider."""
        for processor in self._processors:
            processor.shutdown()


class DistributedTracer:
    """Distributed tracer for creating spans."""

    def __init__(
        self,
        name: str,
        version: Optional[str],
        provider: TracerProvider,
    ) -> None:
        """Initialize tracer.

        Args:
            name: Tracer name
            version: Tracer version
            provider: Tracer provider
        """
        self._name = name
        self._version = version
        self._provider = provider

    @property
    def name(self) -> str:
        """Return tracer name."""
        return self._name

    def start_span(
        self,
        name: str,
        parent: Optional[SpanContext] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[List[SpanLink]] = None,
    ) -> TracingSpan:
        """Start a new span.

        Args:
            name: Span name
            parent: Parent span context
            kind: Span kind
            attributes: Span attributes
            links: Span links

        Returns:
            New span
        """
        trace_id = parent.trace_id if parent else TraceId()
        span_id = SpanId()

        # Check sampling
        decision = self._provider.should_sample(parent, trace_id, name, attributes or {})

        trace_flags = 1 if decision == SamplingDecision.RECORD_AND_SAMPLE else 0

        context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            trace_flags=trace_flags,
        )

        span = TracingSpan(
            name=name,
            context=context,
            parent_context=parent,
            kind=kind,
            attributes=attributes or {},
            links=links or [],
        )

        return span

    def end_span(self, span: TracingSpan) -> None:
        """End a span.

        Args:
            span: Span to end
        """
        span.end()
        self._provider.on_span_end(span)


class TraceContext:
    """Context for trace propagation."""

    _current: Optional[SpanContext] = None
    _lock = threading.Lock()

    @classmethod
    def get_current(cls) -> Optional[SpanContext]:
        """Get current span context."""
        return cls._current

    @classmethod
    def set_current(cls, context: Optional[SpanContext]) -> Optional[SpanContext]:
        """Set current span context.

        Args:
            context: New context

        Returns:
            Previous context
        """
        with cls._lock:
            previous = cls._current
            cls._current = context
            return previous

    @classmethod
    def clear(cls) -> None:
        """Clear current context."""
        with cls._lock:
            cls._current = None


class trace_span:
    """Context manager for tracing spans."""

    def __init__(
        self,
        tracer: DistributedTracer,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize context manager.

        Args:
            tracer: Tracer instance
            name: Span name
            kind: Span kind
            attributes: Span attributes
        """
        self._tracer = tracer
        self._name = name
        self._kind = kind
        self._attributes = attributes
        self._span: Optional[TracingSpan] = None
        self._previous_context: Optional[SpanContext] = None

    def __enter__(self) -> TracingSpan:
        """Enter span context."""
        parent = TraceContext.get_current()
        self._span = self._tracer.start_span(
            name=self._name,
            parent=parent,
            kind=self._kind,
            attributes=self._attributes,
        )
        self._previous_context = TraceContext.set_current(self._span.context)
        return self._span

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit span context."""
        if self._span:
            if exc_type:
                self._span.set_status(SpanStatus.ERROR, str(exc_val))
                self._span.add_event(
                    "exception",
                    {
                        "type": exc_type.__name__ if exc_type else None,
                        "message": str(exc_val),
                    },
                )
            else:
                self._span.set_status(SpanStatus.OK)

            self._tracer.end_span(self._span)

        TraceContext.set_current(self._previous_context)


class Propagator(ABC):
    """Abstract base class for context propagators."""

    @abstractmethod
    def inject(
        self,
        context: SpanContext,
        carrier: Dict[str, str],
    ) -> None:
        """Inject context into carrier.

        Args:
            context: Span context
            carrier: Carrier to inject into
        """
        pass

    @abstractmethod
    def extract(self, carrier: Dict[str, str]) -> Optional[SpanContext]:
        """Extract context from carrier.

        Args:
            carrier: Carrier to extract from

        Returns:
            Span context or None
        """
        pass


class W3CTraceContextPropagator(Propagator):
    """W3C Trace Context propagator."""

    TRACEPARENT_HEADER = "traceparent"
    TRACESTATE_HEADER = "tracestate"

    def inject(
        self,
        context: SpanContext,
        carrier: Dict[str, str],
    ) -> None:
        """Inject context using W3C format."""
        traceparent = (
            f"00-{context.trace_id.value}-{context.span_id.value}-{context.trace_flags:02x}"
        )
        carrier[self.TRACEPARENT_HEADER] = traceparent

        if context.trace_state:
            tracestate = ",".join(f"{k}={v}" for k, v in context.trace_state.items())
            carrier[self.TRACESTATE_HEADER] = tracestate

    def extract(self, carrier: Dict[str, str]) -> Optional[SpanContext]:
        """Extract context using W3C format."""
        traceparent = carrier.get(self.TRACEPARENT_HEADER)
        if not traceparent:
            return None

        try:
            parts = traceparent.split("-")
            if len(parts) != 4:
                return None

            version, trace_id, span_id, flags = parts

            context = SpanContext(
                trace_id=TraceId(value=trace_id),
                span_id=SpanId(value=span_id),
                trace_flags=int(flags, 16),
                is_remote=True,
            )

            tracestate = carrier.get(self.TRACESTATE_HEADER)
            if tracestate:
                for item in tracestate.split(","):
                    if "=" in item:
                        key, value = item.split("=", 1)
                        context.trace_state[key.strip()] = value.strip()

            return context

        except Exception:
            return None


class B3Propagator(Propagator):
    """B3 format propagator."""

    TRACE_ID_HEADER = "X-B3-TraceId"
    SPAN_ID_HEADER = "X-B3-SpanId"
    SAMPLED_HEADER = "X-B3-Sampled"
    PARENT_SPAN_ID_HEADER = "X-B3-ParentSpanId"

    def inject(
        self,
        context: SpanContext,
        carrier: Dict[str, str],
    ) -> None:
        """Inject context using B3 format."""
        carrier[self.TRACE_ID_HEADER] = context.trace_id.value
        carrier[self.SPAN_ID_HEADER] = context.span_id.value
        carrier[self.SAMPLED_HEADER] = "1" if context.is_sampled else "0"

    def extract(self, carrier: Dict[str, str]) -> Optional[SpanContext]:
        """Extract context using B3 format."""
        trace_id = carrier.get(self.TRACE_ID_HEADER)
        span_id = carrier.get(self.SPAN_ID_HEADER)

        if not trace_id or not span_id:
            return None

        sampled = carrier.get(self.SAMPLED_HEADER, "1")

        return SpanContext(
            trace_id=TraceId(value=trace_id),
            span_id=SpanId(value=span_id),
            trace_flags=1 if sampled == "1" else 0,
            is_remote=True,
        )


@dataclass
class TracingStats:
    """Statistics for distributed tracing."""

    total_spans: int = 0
    sampled_spans: int = 0
    dropped_spans: int = 0
    error_spans: int = 0
    total_duration_ms: float = 0.0

    @property
    def sample_rate(self) -> float:
        """Calculate sample rate."""
        if self.total_spans == 0:
            return 0.0
        return self.sampled_spans / self.total_spans

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.sampled_spans == 0:
            return 0.0
        return self.error_spans / self.sampled_spans


class TracingVisionProvider(VisionProvider):
    """Vision provider with distributed tracing."""

    def __init__(
        self,
        provider: VisionProvider,
        tracer: DistributedTracer,
        propagator: Optional[Propagator] = None,
    ) -> None:
        """Initialize tracing provider.

        Args:
            provider: Underlying vision provider
            tracer: Distributed tracer
            propagator: Context propagator
        """
        self._provider = provider
        self._tracer = tracer
        self._propagator = propagator or W3CTraceContextPropagator()
        self._stats = TracingStats()

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"tracing_{self._provider.provider_name}"

    @property
    def tracer(self) -> DistributedTracer:
        """Return tracer."""
        return self._tracer

    @property
    def stats(self) -> TracingStats:
        """Return tracing statistics."""
        return self._stats

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with distributed tracing.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        self._stats.total_spans += 1

        parent = TraceContext.get_current()
        span = self._tracer.start_span(
            name="vision.analyze_image",
            parent=parent,
            kind=SpanKind.CLIENT,
            attributes={
                "vision.provider": self._provider.provider_name,
                "vision.image_size": len(image_data),
                "vision.include_description": include_description,
            },
        )

        if span.context.is_sampled:
            self._stats.sampled_spans += 1
        else:
            self._stats.dropped_spans += 1

        previous = TraceContext.set_current(span.context)

        try:
            span.add_event("request_started")

            result = await self._provider.analyze_image(image_data, include_description)

            span.set_attributes(
                {
                    "vision.confidence": result.confidence,
                    "vision.summary_length": len(result.summary),
                }
            )
            span.add_event("request_completed")
            span.set_status(SpanStatus.OK)

            return result

        except Exception as e:
            self._stats.error_spans += 1
            span.set_status(SpanStatus.ERROR, str(e))
            span.add_event(
                "exception",
                {
                    "type": type(e).__name__,
                    "message": str(e),
                },
            )
            raise

        finally:
            span.end()
            if span.duration_ms:
                self._stats.total_duration_ms += span.duration_ms

            self._tracer.end_span(span)
            TraceContext.set_current(previous)


def create_tracing_provider(
    provider: VisionProvider,
    tracer_provider: Optional[TracerProvider] = None,
    service_name: str = "vision",
) -> TracingVisionProvider:
    """Create a tracing vision provider.

    Args:
        provider: Underlying vision provider
        tracer_provider: Optional tracer provider
        service_name: Service name for tracing

    Returns:
        TracingVisionProvider instance
    """
    if tracer_provider is None:
        tracer_provider = TracerProvider(config=TracerConfig(service_name=service_name))
        exporter = InMemorySpanExporter()
        processor = BatchSpanProcessor(exporter)
        tracer_provider.add_processor(processor)

    tracer = tracer_provider.get_tracer(service_name)

    return TracingVisionProvider(
        provider=provider,
        tracer=tracer,
    )
