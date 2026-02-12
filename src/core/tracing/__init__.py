"""OpenTelemetry Tracing Module.

Provides distributed tracing capabilities:
- Trace context propagation
- Span management
- Sampling strategies
- Multiple export formats
"""

from src.core.tracing.context import (
    TraceFlags,
    TraceId,
    SpanId,
    SpanContext,
    get_current_context,
    set_current_context,
    attach_context,
    detach_context,
    ContextToken,
    W3CTraceContextPropagator,
    B3Propagator,
    CompositePropagator,
    get_propagator,
    set_propagator,
)
from src.core.tracing.span import (
    SpanKind,
    StatusCode,
    Status,
    SpanEvent,
    SpanLink,
    Span,
    Tracer,
    SpanProcessor,
    SimpleSpanProcessor,
    BatchSpanProcessor,
    SpanExporter,
    ConsoleSpanExporter,
    InMemorySpanExporter,
    TracerProvider,
    get_tracer_provider,
    set_tracer_provider,
    get_tracer,
    trace,
)
from src.core.tracing.sampler import (
    SamplingResult,
    Sampler,
    AlwaysOnSampler,
    AlwaysOffSampler,
    TraceIdRatioSampler,
    ParentBasedSampler,
    RateLimitingSampler,
    RuleBasedSampler,
    CompositeSampler,
)

__all__ = [
    # Context
    "TraceFlags",
    "TraceId",
    "SpanId",
    "SpanContext",
    "get_current_context",
    "set_current_context",
    "attach_context",
    "detach_context",
    "ContextToken",
    "W3CTraceContextPropagator",
    "B3Propagator",
    "CompositePropagator",
    "get_propagator",
    "set_propagator",
    # Span
    "SpanKind",
    "StatusCode",
    "Status",
    "SpanEvent",
    "SpanLink",
    "Span",
    "Tracer",
    "SpanProcessor",
    "SimpleSpanProcessor",
    "BatchSpanProcessor",
    "SpanExporter",
    "ConsoleSpanExporter",
    "InMemorySpanExporter",
    "TracerProvider",
    "get_tracer_provider",
    "set_tracer_provider",
    "get_tracer",
    "trace",
    # Sampler
    "SamplingResult",
    "Sampler",
    "AlwaysOnSampler",
    "AlwaysOffSampler",
    "TraceIdRatioSampler",
    "ParentBasedSampler",
    "RateLimitingSampler",
    "RuleBasedSampler",
    "CompositeSampler",
]
