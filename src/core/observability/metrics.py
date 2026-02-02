"""OpenTelemetry Metrics Module."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Conditional imports
try:
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource

    OTEL_METRICS_AVAILABLE = True
except ImportError:
    OTEL_METRICS_AVAILABLE = False
    metrics = None  # type: ignore

try:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    OTLP_METRICS_AVAILABLE = True
except ImportError:
    OTLP_METRICS_AVAILABLE = False
    OTLPMetricExporter = None  # type: ignore

try:
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    PrometheusMetricReader = None  # type: ignore


@dataclass
class MetricsConfig:
    """Configuration for OpenTelemetry metrics."""

    service_name: str = "cad-ml-platform"
    otlp_endpoint: Optional[str] = None
    prometheus_port: Optional[int] = None
    console_export: bool = False
    export_interval_ms: int = 60000

    @classmethod
    def from_env(cls) -> "MetricsConfig":
        """Create config from environment variables."""
        return cls(
            service_name=os.getenv("OTEL_SERVICE_NAME", "cad-ml-platform"),
            otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"),
            prometheus_port=int(os.getenv("PROMETHEUS_METRICS_PORT", "0")) or None,
            console_export=os.getenv("OTEL_METRICS_CONSOLE_EXPORT", "false").lower() == "true",
            export_interval_ms=int(os.getenv("OTEL_METRICS_EXPORT_INTERVAL_MS", "60000")),
        )


_meter: Optional[Any] = None
_initialized: bool = False


def init_metrics(config: Optional[MetricsConfig] = None) -> Any:
    """Initialize OpenTelemetry metrics.

    Args:
        config: Metrics configuration

    Returns:
        Meter instance
    """
    global _meter, _initialized

    if _initialized:
        return _meter

    if not OTEL_METRICS_AVAILABLE:
        logger.warning("OpenTelemetry metrics not available, using no-op meter")
        _meter = _NoOpMeter()
        _initialized = True
        return _meter

    config = config or MetricsConfig.from_env()

    # Create resource
    resource = Resource.create({
        SERVICE_NAME: config.service_name,
        "service.version": os.getenv("APP_VERSION", "1.0.0"),
        "deployment.environment": os.getenv("ENVIRONMENT", "development"),
    })

    readers = []

    # Add OTLP exporter
    if config.otlp_endpoint and OTLP_METRICS_AVAILABLE:
        exporter = OTLPMetricExporter(endpoint=config.otlp_endpoint, insecure=True)
        readers.append(
            PeriodicExportingMetricReader(
                exporter,
                export_interval_millis=config.export_interval_ms,
            )
        )
        logger.info(f"OTLP metrics exporter configured: {config.otlp_endpoint}")

    # Add Prometheus exporter
    if config.prometheus_port and PROMETHEUS_AVAILABLE:
        readers.append(PrometheusMetricReader())
        logger.info(f"Prometheus metrics reader configured on port {config.prometheus_port}")

    # Add console exporter
    if config.console_export:
        readers.append(
            PeriodicExportingMetricReader(
                ConsoleMetricExporter(),
                export_interval_millis=config.export_interval_ms,
            )
        )
        logger.info("Console metrics exporter enabled")

    # Create provider
    if readers:
        provider = MeterProvider(resource=resource, metric_readers=readers)
        metrics.set_meter_provider(provider)
    else:
        metrics.set_meter_provider(MeterProvider(resource=resource))

    _meter = metrics.get_meter(config.service_name)
    _initialized = True

    logger.info(f"Metrics initialized for service: {config.service_name}")
    return _meter


def get_meter(name: Optional[str] = None) -> Any:
    """Get meter instance.

    Args:
        name: Optional meter name

    Returns:
        Meter instance
    """
    global _meter

    if _meter is None:
        init_metrics()

    return _meter


def create_counter(
    name: str,
    description: str = "",
    unit: str = "",
) -> Any:
    """Create a counter metric.

    Args:
        name: Metric name
        description: Metric description
        unit: Unit of measurement

    Returns:
        Counter instance
    """
    meter = get_meter()
    if isinstance(meter, _NoOpMeter):
        return _NoOpCounter()
    return meter.create_counter(name, description=description, unit=unit)


def create_histogram(
    name: str,
    description: str = "",
    unit: str = "",
) -> Any:
    """Create a histogram metric.

    Args:
        name: Metric name
        description: Metric description
        unit: Unit of measurement

    Returns:
        Histogram instance
    """
    meter = get_meter()
    if isinstance(meter, _NoOpMeter):
        return _NoOpHistogram()
    return meter.create_histogram(name, description=description, unit=unit)


def create_gauge(
    name: str,
    description: str = "",
    unit: str = "",
) -> Any:
    """Create a gauge metric (observable).

    Args:
        name: Metric name
        description: Metric description
        unit: Unit of measurement

    Returns:
        Gauge callback registration function
    """
    meter = get_meter()
    if isinstance(meter, _NoOpMeter):
        return lambda callback: None
    return lambda callback: meter.create_observable_gauge(
        name,
        callbacks=[callback],
        description=description,
        unit=unit,
    )


# No-op implementations
class _NoOpMeter:
    def create_counter(self, name: str, **kwargs: Any) -> "_NoOpCounter":
        return _NoOpCounter()

    def create_histogram(self, name: str, **kwargs: Any) -> "_NoOpHistogram":
        return _NoOpHistogram()

    def create_observable_gauge(self, name: str, **kwargs: Any) -> None:
        pass


class _NoOpCounter:
    def add(self, value: int, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass


class _NoOpHistogram:
    def record(self, value: float, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass
