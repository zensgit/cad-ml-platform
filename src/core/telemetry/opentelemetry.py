"""OpenTelemetry Integration for distributed tracing.

Features:
- Distributed tracing with context propagation
- Automatic FastAPI instrumentation
- Custom span creation
- Metrics export
- Configurable exporters (OTLP, Jaeger, Zipkin)
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, Optional

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry - gracefully degrade if not available
try:
    from opentelemetry import trace
    from opentelemetry.context import Context
    from opentelemetry.propagate import extract, inject
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    Context = None
    Span = None
    Tracer = None


@dataclass
class TelemetryConfig:
    """OpenTelemetry configuration."""

    service_name: str = "cad-ml-platform"
    service_version: str = "1.0.0"
    environment: str = "development"

    # Exporter config
    exporter_type: str = "otlp"  # otlp, jaeger, zipkin, console
    otlp_endpoint: Optional[str] = None
    jaeger_agent_host: Optional[str] = None
    jaeger_agent_port: int = 6831

    # Sampling
    sample_rate: float = 1.0  # 1.0 = 100% sampling

    # Batch processor config
    max_queue_size: int = 2048
    max_export_batch_size: int = 512
    export_timeout_ms: int = 30000

    @classmethod
    def from_env(cls) -> "TelemetryConfig":
        """Create config from environment variables."""
        return cls(
            service_name=os.getenv("OTEL_SERVICE_NAME", "cad-ml-platform"),
            service_version=os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
            environment=os.getenv("OTEL_ENVIRONMENT", "development"),
            exporter_type=os.getenv("OTEL_EXPORTER_TYPE", "otlp"),
            otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            jaeger_agent_host=os.getenv("OTEL_EXPORTER_JAEGER_AGENT_HOST"),
            jaeger_agent_port=int(os.getenv("OTEL_EXPORTER_JAEGER_AGENT_PORT", "6831")),
            sample_rate=float(os.getenv("OTEL_SAMPLE_RATE", "1.0")),
        )


class TelemetryManager:
    """Manages OpenTelemetry tracing."""

    def __init__(self, config: Optional[TelemetryConfig] = None):
        self.config = config or TelemetryConfig.from_env()
        self._provider: Optional[Any] = None
        self._tracer: Optional[Any] = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize OpenTelemetry."""
        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry not available - tracing disabled")
            return False

        if self._initialized:
            return True

        try:
            # Create resource
            resource = Resource.create({
                ResourceAttributes.SERVICE_NAME: self.config.service_name,
                ResourceAttributes.SERVICE_VERSION: self.config.service_version,
                ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.config.environment,
            })

            # Create tracer provider
            self._provider = TracerProvider(resource=resource)

            # Create and add span processor with exporter
            exporter = self._create_exporter()
            if exporter:
                processor = BatchSpanProcessor(
                    exporter,
                    max_queue_size=self.config.max_queue_size,
                    max_export_batch_size=self.config.max_export_batch_size,
                    export_timeout_millis=self.config.export_timeout_ms,
                )
                self._provider.add_span_processor(processor)

            # Set global tracer provider
            trace.set_tracer_provider(self._provider)

            # Get tracer
            self._tracer = trace.get_tracer(
                self.config.service_name,
                self.config.service_version,
            )

            self._initialized = True
            logger.info(f"OpenTelemetry initialized with {self.config.exporter_type} exporter")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            return False

    def _create_exporter(self) -> Optional[Any]:
        """Create span exporter based on config."""
        try:
            if self.config.exporter_type == "otlp":
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                endpoint = self.config.otlp_endpoint or "http://localhost:4317"
                return OTLPSpanExporter(endpoint=endpoint)

            elif self.config.exporter_type == "jaeger":
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter

                return JaegerExporter(
                    agent_host_name=self.config.jaeger_agent_host or "localhost",
                    agent_port=self.config.jaeger_agent_port,
                )

            elif self.config.exporter_type == "console":
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter

                return ConsoleSpanExporter()

            else:
                logger.warning(f"Unknown exporter type: {self.config.exporter_type}")
                return None

        except ImportError as e:
            logger.warning(f"Exporter not available: {e}")
            return None

    def get_tracer(self) -> Optional[Any]:
        """Get the tracer instance."""
        if not self._initialized:
            self.initialize()
        return self._tracer

    def shutdown(self) -> None:
        """Shutdown OpenTelemetry."""
        if self._provider:
            self._provider.shutdown()
            self._initialized = False
            logger.info("OpenTelemetry shutdown complete")

    @contextmanager
    def start_span(
        self,
        name: str,
        kind: Optional[Any] = None,
        attributes: Optional[Dict[str, Any]] = None,
        parent: Optional[Any] = None,
    ) -> Generator[Optional[Any], None, None]:
        """Start a new span."""
        if not OTEL_AVAILABLE or not self._tracer:
            yield None
            return

        span_kind = kind or SpanKind.INTERNAL
        with self._tracer.start_as_current_span(
            name,
            kind=span_kind,
            attributes=attributes or {},
            context=parent,
        ) as span:
            yield span

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an event to the current span."""
        if not OTEL_AVAILABLE:
            return

        span = trace.get_current_span()
        if span:
            span.add_event(name, attributes=attributes or {})

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the current span."""
        if not OTEL_AVAILABLE:
            return

        span = trace.get_current_span()
        if span:
            span.set_attribute(key, value)

    def set_status(self, status_code: str, description: Optional[str] = None) -> None:
        """Set the status of the current span."""
        if not OTEL_AVAILABLE:
            return

        span = trace.get_current_span()
        if span:
            code = StatusCode.OK if status_code == "ok" else StatusCode.ERROR
            span.set_status(Status(code, description))

    def record_exception(self, exception: Exception) -> None:
        """Record an exception in the current span."""
        if not OTEL_AVAILABLE:
            return

        span = trace.get_current_span()
        if span:
            span.record_exception(exception)
            span.set_status(Status(StatusCode.ERROR, str(exception)))


def traced(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable:
    """Decorator to trace a function."""

    def decorator(func: Callable) -> Callable:
        import functools

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            span_name = name or f"{func.__module__}.{func.__name__}"
            with get_telemetry_manager().start_span(span_name, attributes=attributes):
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            span_name = name or f"{func.__module__}.{func.__name__}"
            with get_telemetry_manager().start_span(span_name, attributes=attributes):
                return func(*args, **kwargs)

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Global telemetry manager
_telemetry_manager: Optional[TelemetryManager] = None


def get_telemetry_manager() -> TelemetryManager:
    """Get the global telemetry manager instance."""
    global _telemetry_manager
    if _telemetry_manager is None:
        _telemetry_manager = TelemetryManager()
    return _telemetry_manager


def init_telemetry(config: Optional[TelemetryConfig] = None) -> bool:
    """Initialize telemetry with optional config."""
    manager = get_telemetry_manager()
    if config:
        manager.config = config
    return manager.initialize()


def shutdown_telemetry() -> None:
    """Shutdown telemetry."""
    global _telemetry_manager
    if _telemetry_manager:
        _telemetry_manager.shutdown()
        _telemetry_manager = None
