"""OpenTelemetry Tracing Module."""

from __future__ import annotations

import functools
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, Optional, TypeVar

logger = logging.getLogger(__name__)

# Conditional imports
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import Span, Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # type: ignore
    Span = None  # type: ignore
    StatusCode = None  # type: ignore

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False
    OTLPSpanExporter = None  # type: ignore

try:
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    JAEGER_AVAILABLE = True
except ImportError:
    JAEGER_AVAILABLE = False
    JaegerExporter = None  # type: ignore


F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class TracingConfig:
    """Configuration for distributed tracing."""

    service_name: str = "cad-ml-platform"
    otlp_endpoint: Optional[str] = None
    jaeger_host: Optional[str] = None
    jaeger_port: int = 6831
    console_export: bool = False
    sample_rate: float = 1.0

    @classmethod
    def from_env(cls) -> "TracingConfig":
        """Create config from environment variables."""
        return cls(
            service_name=os.getenv("OTEL_SERVICE_NAME", "cad-ml-platform"),
            otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            jaeger_host=os.getenv("JAEGER_AGENT_HOST"),
            jaeger_port=int(os.getenv("JAEGER_AGENT_PORT", "6831")),
            console_export=os.getenv("OTEL_CONSOLE_EXPORT", "false").lower() == "true",
            sample_rate=float(os.getenv("OTEL_SAMPLE_RATE", "1.0")),
        )


_tracer: Optional[Any] = None
_initialized: bool = False


def init_tracing(config: Optional[TracingConfig] = None) -> Any:
    """Initialize OpenTelemetry tracing.

    Args:
        config: Tracing configuration

    Returns:
        Tracer instance
    """
    global _tracer, _initialized

    if _initialized:
        return _tracer

    if not OTEL_AVAILABLE:
        logger.warning("OpenTelemetry not available, using no-op tracer")
        _tracer = _NoOpTracer()
        _initialized = True
        return _tracer

    config = config or TracingConfig.from_env()

    # Create resource
    resource = Resource.create({
        SERVICE_NAME: config.service_name,
        "service.version": os.getenv("APP_VERSION", "1.0.0"),
        "deployment.environment": os.getenv("ENVIRONMENT", "development"),
    })

    # Create provider
    provider = TracerProvider(resource=resource)

    # Add exporters
    if config.otlp_endpoint and OTLP_AVAILABLE:
        exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        logger.info(f"OTLP exporter configured: {config.otlp_endpoint}")

    elif config.jaeger_host and JAEGER_AVAILABLE:
        exporter = JaegerExporter(
            agent_host_name=config.jaeger_host,
            agent_port=config.jaeger_port,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        logger.info(f"Jaeger exporter configured: {config.jaeger_host}:{config.jaeger_port}")

    if config.console_export:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("Console exporter enabled")

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(config.service_name)
    _initialized = True

    logger.info(f"Tracing initialized for service: {config.service_name}")
    return _tracer


def get_tracer(name: Optional[str] = None) -> Any:
    """Get tracer instance.

    Args:
        name: Optional tracer name

    Returns:
        Tracer instance
    """
    global _tracer

    if _tracer is None:
        init_tracing()

    return _tracer


def trace_span(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """Decorator to trace a function.

    Args:
        name: Span name (defaults to function name)
        attributes: Static attributes to add

    Returns:
        Decorated function

    Example:
        @trace_span("process_document")
        async def process(doc_id: str) -> dict:
            return await do_processing(doc_id)
    """
    def decorator(func: F) -> F:
        span_name = name or func.__name__

        if not OTEL_AVAILABLE:
            return func

        import asyncio

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer()
                with tracer.start_as_current_span(span_name) as span:
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, value)
                    try:
                        result = await func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise
            return async_wrapper  # type: ignore
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer()
                with tracer.start_as_current_span(span_name) as span:
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, value)
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise
            return sync_wrapper  # type: ignore

    return decorator


def add_span_attributes(attributes: Dict[str, Any]) -> None:
    """Add attributes to the current span.

    Args:
        attributes: Key-value pairs to add
    """
    if not OTEL_AVAILABLE:
        return

    span = trace.get_current_span()
    if span:
        for key, value in attributes.items():
            span.set_attribute(key, value)


@contextmanager
def span_context(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> Generator[Any, None, None]:
    """Context manager for creating a span.

    Args:
        name: Span name
        attributes: Optional attributes

    Yields:
        Span instance
    """
    if not OTEL_AVAILABLE:
        yield _NoOpSpan()
        return

    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


# No-op implementations
class _NoOpSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def start_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()
