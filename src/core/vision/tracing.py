"""Request context and distributed tracing for vision providers.

Provides:
- Request context propagation
- Distributed tracing support
- Span management
- Correlation ID tracking
"""

from __future__ import annotations

import contextvars
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import VisionDescription, VisionProvider

logger = logging.getLogger(__name__)

# Context variable for current request context
_current_context: contextvars.ContextVar[Optional["RequestContext"]] = contextvars.ContextVar(
    "vision_request_context", default=None
)


class SpanStatus(Enum):
    """Status of a tracing span."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class SpanKind(Enum):
    """Kind of span."""

    INTERNAL = "internal"
    CLIENT = "client"
    SERVER = "server"
    PRODUCER = "producer"
    CONSUMER = "consumer"


@dataclass
class SpanAttribute:
    """An attribute on a span."""

    key: str
    value: Any


@dataclass
class SpanEvent:
    """An event that occurred during a span."""

    name: str
    timestamp: datetime
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """A tracing span representing a unit of work."""

    trace_id: str
    span_id: str
    name: str
    kind: SpanKind = SpanKind.INTERNAL
    parent_span_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: SpanStatus = SpanStatus.UNSET
    status_message: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the span."""
        self.attributes[key] = value

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an event to the span."""
        event = SpanEvent(
            name=name,
            timestamp=datetime.now(),
            attributes=attributes or {},
        )
        self.events.append(event)

    def set_status(self, status: SpanStatus, message: Optional[str] = None) -> None:
        """Set the span status."""
        self.status = status
        self.status_message = message

    def end(self) -> None:
        """End the span."""
        if self.end_time is None:
            self.end_time = datetime.now()

    @property
    def duration_ms(self) -> float:
        """Calculate span duration in milliseconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": [
                {
                    "name": e.name,
                    "timestamp": e.timestamp.isoformat(),
                    "attributes": e.attributes,
                }
                for e in self.events
            ],
        }


@dataclass
class RequestContext:
    """Context for a vision analysis request."""

    # Identifiers
    request_id: str
    trace_id: str
    correlation_id: Optional[str] = None

    # Request metadata
    provider: Optional[str] = None
    operation: str = "analyze_image"
    start_time: Optional[datetime] = None

    # User context
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None

    # Request details
    image_size_bytes: int = 0
    include_description: bool = True

    # Custom attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    baggage: Dict[str, str] = field(default_factory=dict)

    # Tracing
    spans: List[Span] = field(default_factory=list)
    _current_span: Optional[Span] = field(default=None, repr=False)

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """Start a new span."""
        span = Span(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4())[:16],
            name=name,
            kind=kind,
            parent_span_id=self._current_span.span_id if self._current_span else None,
            start_time=datetime.now(),
            attributes=attributes or {},
        )
        self.spans.append(span)
        self._current_span = span
        return span

    def end_span(self, status: SpanStatus = SpanStatus.OK) -> None:
        """End the current span."""
        if self._current_span:
            self._current_span.set_status(status)
            self._current_span.end()

            # Find parent span
            parent_id = self._current_span.parent_span_id
            if parent_id:
                for span in self.spans:
                    if span.span_id == parent_id:
                        self._current_span = span
                        return
            self._current_span = None

    @property
    def current_span(self) -> Optional[Span]:
        """Get the current span."""
        return self._current_span

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a context attribute."""
        self.attributes[key] = value

    def set_baggage(self, key: str, value: str) -> None:
        """Set baggage item for propagation."""
        self.baggage[key] = value

    def get_baggage(self, key: str) -> Optional[str]:
        """Get baggage item."""
        return self.baggage.get(key)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "correlation_id": self.correlation_id,
            "provider": self.provider,
            "operation": self.operation,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "image_size_bytes": self.image_size_bytes,
            "include_description": self.include_description,
            "attributes": self.attributes,
            "baggage": self.baggage,
            "spans": [s.to_dict() for s in self.spans],
        }

    @classmethod
    def create(
        cls,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> "RequestContext":
        """Create a new request context."""
        return cls(
            request_id=request_id or str(uuid.uuid4())[:8],
            trace_id=trace_id or str(uuid.uuid4()).replace("-", ""),
            correlation_id=correlation_id,
            start_time=datetime.now(),
            **kwargs,
        )


def get_current_context() -> Optional[RequestContext]:
    """Get the current request context."""
    return _current_context.get()


def set_current_context(context: Optional[RequestContext]) -> None:
    """Set the current request context."""
    _current_context.set(context)


class ContextManager:
    """Context manager for request context."""

    def __init__(self, context: RequestContext):
        self._context = context
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> RequestContext:
        self._token = _current_context.set(self._context)
        return self._context

    def __exit__(self, *args: Any) -> None:
        if self._token is not None:
            _current_context.reset(self._token)


def with_context(context: RequestContext) -> ContextManager:
    """Create a context manager for a request context."""
    return ContextManager(context)


@dataclass
class TracerConfig:
    """Configuration for the tracer."""

    enabled: bool = True
    sample_rate: float = 1.0  # 0.0 to 1.0
    max_spans_per_trace: int = 100
    propagate_baggage: bool = True
    export_spans: bool = False
    exporter_endpoint: Optional[str] = None


class SpanExporter:
    """Base class for span exporters."""

    def export(self, spans: List[Span]) -> bool:
        """Export spans. Override in subclass."""
        return True


class ConsoleSpanExporter(SpanExporter):
    """Export spans to console for debugging."""

    def export(self, spans: List[Span]) -> bool:
        """Export spans to console."""
        for span in spans:
            logger.info(
                f"[TRACE] {span.trace_id[:8]} {span.name} "
                f"({span.duration_ms:.2f}ms) [{span.status.value}]"
            )
        return True


class Tracer:
    """
    Distributed tracer for vision operations.

    Features:
    - Request context management
    - Span creation and management
    - Context propagation
    - Span export
    """

    def __init__(
        self,
        config: Optional[TracerConfig] = None,
        exporter: Optional[SpanExporter] = None,
    ):
        """
        Initialize tracer.

        Args:
            config: Tracer configuration
            exporter: Span exporter for external systems
        """
        self._config = config or TracerConfig()
        self._exporter = exporter or ConsoleSpanExporter()
        self._pending_spans: List[Span] = []

    def create_context(
        self,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> RequestContext:
        """Create a new request context."""
        return RequestContext.create(
            request_id=request_id,
            trace_id=trace_id,
            correlation_id=correlation_id,
            **kwargs,
        )

    def start_trace(
        self,
        name: str = "vision_analysis",
        **kwargs: Any,
    ) -> RequestContext:
        """Start a new trace."""
        context = self.create_context(**kwargs)
        context.start_span(name, kind=SpanKind.SERVER)
        set_current_context(context)
        return context

    def end_trace(self, status: SpanStatus = SpanStatus.OK) -> None:
        """End the current trace."""
        context = get_current_context()
        if context:
            # End all open spans
            while context.current_span:
                context.end_span(status)

            # Export spans if configured
            if self._config.export_spans and context.spans:
                self._exporter.export(context.spans)

            set_current_context(None)

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Optional[Span]:
        """Start a span in the current context."""
        context = get_current_context()
        if context and self._config.enabled:
            if len(context.spans) < self._config.max_spans_per_trace:
                return context.start_span(name, kind, attributes)
        return None

    def end_span(self, status: SpanStatus = SpanStatus.OK) -> None:
        """End the current span."""
        context = get_current_context()
        if context:
            context.end_span(status)

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add event to current span."""
        context = get_current_context()
        if context and context.current_span:
            context.current_span.add_event(name, attributes)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set attribute on current span."""
        context = get_current_context()
        if context and context.current_span:
            context.current_span.set_attribute(key, value)

    def extract_context(
        self,
        headers: Dict[str, str],
    ) -> Optional[RequestContext]:
        """
        Extract context from headers (for distributed tracing).

        Args:
            headers: HTTP headers containing trace context

        Returns:
            RequestContext if found, None otherwise
        """
        trace_id = (
            headers.get("x-trace-id") or headers.get("traceparent", "").split("-")[1]
            if "-" in headers.get("traceparent", "")
            else None
        )
        request_id = headers.get("x-request-id")
        correlation_id = headers.get("x-correlation-id")

        if trace_id:
            context = self.create_context(
                request_id=request_id,
                trace_id=trace_id,
                correlation_id=correlation_id,
            )

            # Extract baggage
            if self._config.propagate_baggage:
                baggage_header = headers.get("baggage", "")
                for item in baggage_header.split(","):
                    if "=" in item:
                        key, value = item.strip().split("=", 1)
                        context.set_baggage(key, value)

            return context
        return None

    def inject_context(
        self,
        headers: Dict[str, str],
        context: Optional[RequestContext] = None,
    ) -> Dict[str, str]:
        """
        Inject context into headers for propagation.

        Args:
            headers: Headers to inject into
            context: Context to inject (uses current if not provided)

        Returns:
            Updated headers
        """
        ctx = context or get_current_context()
        if ctx:
            headers["x-trace-id"] = ctx.trace_id
            headers["x-request-id"] = ctx.request_id
            if ctx.correlation_id:
                headers["x-correlation-id"] = ctx.correlation_id

            # Inject baggage
            if self._config.propagate_baggage and ctx.baggage:
                baggage_items = [f"{k}={v}" for k, v in ctx.baggage.items()]
                headers["baggage"] = ",".join(baggage_items)

        return headers


class TracingVisionProvider:
    """
    Wrapper that adds tracing to any VisionProvider.

    Automatically creates spans for vision operations.
    """

    def __init__(
        self,
        provider: VisionProvider,
        tracer: Tracer,
        span_name: Optional[str] = None,
    ):
        """
        Initialize tracing provider.

        Args:
            provider: The underlying vision provider
            tracer: Tracer instance
            span_name: Optional custom span name
        """
        self._provider = provider
        self._tracer = tracer
        self._span_name = span_name or f"vision.{provider.provider_name}.analyze"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
        context: Optional[RequestContext] = None,
    ) -> VisionDescription:
        """
        Analyze image with tracing.

        Args:
            image_data: Raw image bytes
            include_description: Whether to generate description
            context: Optional request context

        Returns:
            VisionDescription with analysis results
        """
        # Use provided context or current
        ctx = context or get_current_context()
        own_context = False

        if not ctx:
            # Create new context if none exists
            ctx = self._tracer.start_trace(
                name=self._span_name,
                provider=self._provider.provider_name,
                image_size_bytes=len(image_data),
                include_description=include_description,
            )
            own_context = True
        else:
            # Start child span
            self._tracer.start_span(
                name=self._span_name,
                kind=SpanKind.CLIENT,
                attributes={
                    "provider": self._provider.provider_name,
                    "image_size_bytes": len(image_data),
                },
            )

        try:
            # Add event for request start
            self._tracer.add_event(
                "request_started",
                {"provider": self._provider.provider_name},
            )

            start_time = time.time()
            result = await self._provider.analyze_image(image_data, include_description)
            duration_ms = (time.time() - start_time) * 1000

            # Add response attributes
            self._tracer.set_attribute("response_time_ms", duration_ms)
            self._tracer.set_attribute("confidence", result.confidence)
            self._tracer.add_event(
                "request_completed",
                {"duration_ms": duration_ms, "confidence": result.confidence},
            )

            if own_context:
                self._tracer.end_trace(SpanStatus.OK)
            else:
                self._tracer.end_span(SpanStatus.OK)

            return result

        except Exception as e:
            self._tracer.set_attribute("error", True)
            self._tracer.set_attribute("error.type", type(e).__name__)
            self._tracer.set_attribute("error.message", str(e))
            self._tracer.add_event(
                "error",
                {"type": type(e).__name__, "message": str(e)},
            )

            if own_context:
                self._tracer.end_trace(SpanStatus.ERROR)
            else:
                self._tracer.end_span(SpanStatus.ERROR)

            raise

    @property
    def provider_name(self) -> str:
        """Return wrapped provider name."""
        return self._provider.provider_name

    @property
    def tracer(self) -> Tracer:
        """Get the tracer."""
        return self._tracer


# Global tracer instance
_global_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """
    Get the global tracer instance.

    Returns:
        Tracer singleton
    """
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer()
    return _global_tracer


def create_tracing_provider(
    provider: VisionProvider,
    tracer: Optional[Tracer] = None,
    config: Optional[TracerConfig] = None,
    span_name: Optional[str] = None,
) -> TracingVisionProvider:
    """
    Factory to create a tracing provider wrapper.

    Args:
        provider: The underlying vision provider
        tracer: Optional tracer instance
        config: Optional tracer configuration
        span_name: Optional custom span name

    Returns:
        TracingVisionProvider wrapping the original

    Example:
        >>> provider = create_vision_provider("openai")
        >>> traced = create_tracing_provider(provider)
        >>> result = await traced.analyze_image(image_bytes)
    """
    if tracer is None:
        tracer = Tracer(config=config)

    return TracingVisionProvider(
        provider=provider,
        tracer=tracer,
        span_name=span_name,
    )
