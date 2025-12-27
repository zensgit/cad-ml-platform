"""
Unit tests for OpenTelemetry observability module.

Tests cover:
- TelemetryConfig configuration
- Tracer and meter initialization
- traced decorator
- No-op implementations when OTEL unavailable
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.observability import (
    OTEL_AVAILABLE,
    TelemetryConfig,
    get_meter,
    get_tracer,
    setup_telemetry,
    trace_operation,
    traced,
)


class TestTelemetryConfig:
    """Test TelemetryConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TelemetryConfig()

        assert config.service_name == "cad-ml-platform"
        assert config.otlp_endpoint is None
        assert config.jaeger_host is None
        assert config.jaeger_port == 6831
        assert config.enable_auto_instrumentation is True
        assert config.enable_console_export is False
        assert config.sample_rate == 1.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = TelemetryConfig(
            service_name="custom-service",
            otlp_endpoint="localhost:4317",
            jaeger_host="jaeger.local",
            jaeger_port=6832,
            enable_auto_instrumentation=False,
            enable_console_export=True,
            sample_rate=0.5,
        )

        assert config.service_name == "custom-service"
        assert config.otlp_endpoint == "localhost:4317"
        assert config.jaeger_host == "jaeger.local"
        assert config.jaeger_port == 6832
        assert config.enable_auto_instrumentation is False
        assert config.enable_console_export is True
        assert config.sample_rate == 0.5

    def test_from_env(self):
        """Test configuration from environment variables."""
        env_vars = {
            "OTEL_SERVICE_NAME": "env-service",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "otel-collector:4317",
            "JAEGER_AGENT_HOST": "jaeger-agent",
            "JAEGER_AGENT_PORT": "6833",
            "OTEL_AUTO_INSTRUMENT": "false",
            "OTEL_CONSOLE_EXPORT": "true",
            "OTEL_SAMPLE_RATE": "0.25",
        }

        with patch.dict("os.environ", env_vars):
            config = TelemetryConfig.from_env()

            assert config.service_name == "env-service"
            assert config.otlp_endpoint == "otel-collector:4317"
            assert config.jaeger_host == "jaeger-agent"
            assert config.jaeger_port == 6833
            assert config.enable_auto_instrumentation is False
            assert config.enable_console_export is True
            assert config.sample_rate == 0.25


class TestNoOpImplementations:
    """Test no-op implementations for when OTEL is unavailable."""

    def test_noop_tracer(self):
        """Test no-op tracer."""
        from src.core.observability import _NoOpSpan, _NoOpTracer

        tracer = _NoOpTracer()
        span = tracer.start_as_current_span("test")

        assert isinstance(span, _NoOpSpan)

    def test_noop_span_operations(self):
        """Test no-op span operations don't raise."""
        from src.core.observability import _NoOpSpan

        span = _NoOpSpan()
        span.set_attribute("key", "value")
        span.set_status(None)
        span.record_exception(Exception("test"))

    def test_noop_span_context_manager(self):
        """Test no-op span as context manager."""
        from src.core.observability import _NoOpSpan

        with _NoOpSpan() as span:
            span.set_attribute("key", "value")

    def test_noop_meter(self):
        """Test no-op meter."""
        from src.core.observability import _NoOpMeter

        meter = _NoOpMeter()
        counter = meter.create_counter("test_counter")
        histogram = meter.create_histogram("test_histogram")

        # Operations should not raise
        counter.add(1)
        histogram.record(0.5)


class TestGetTracer:
    """Test get_tracer function."""

    def test_get_tracer_returns_tracer(self):
        """Test that get_tracer returns a tracer."""
        tracer = get_tracer("test")
        assert tracer is not None

    def test_get_tracer_default_name(self):
        """Test get_tracer with default name."""
        tracer = get_tracer()
        assert tracer is not None


class TestGetMeter:
    """Test get_meter function."""

    def test_get_meter_returns_meter(self):
        """Test that get_meter returns a meter."""
        meter = get_meter("test")
        assert meter is not None


class TestTracedDecorator:
    """Test traced decorator."""

    def test_traced_sync_function(self):
        """Test traced decorator on sync function."""

        @traced("test_operation")
        def sync_func(x, y):
            return x + y

        result = sync_func(1, 2)
        assert result == 3

    @pytest.mark.asyncio
    async def test_traced_async_function(self):
        """Test traced decorator on async function."""

        @traced("async_operation")
        async def async_func(x, y):
            return x * y

        result = await async_func(3, 4)
        assert result == 12

    def test_traced_with_attributes(self):
        """Test traced decorator with static attributes."""

        @traced("operation", attributes={"version": "1.0"})
        def func_with_attrs():
            return "done"

        result = func_with_attrs()
        assert result == "done"

    def test_traced_preserves_function_name(self):
        """Test that traced preserves function metadata."""

        @traced()
        def my_named_function():
            """My docstring."""
            return True

        assert my_named_function.__name__ == "my_named_function"
        assert "docstring" in my_named_function.__doc__

    def test_traced_exception_handling(self):
        """Test traced handles exceptions correctly."""

        @traced("failing_op")
        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_func()

    @pytest.mark.asyncio
    async def test_traced_async_exception_handling(self):
        """Test traced handles async exceptions correctly."""

        @traced("async_failing")
        async def async_failing():
            raise RuntimeError("async error")

        with pytest.raises(RuntimeError, match="async error"):
            await async_failing()


class TestTraceOperation:
    """Test trace_operation context manager."""

    def test_trace_operation_basic(self):
        """Test basic trace_operation usage."""
        with trace_operation("test_block") as span:
            result = 1 + 1

        assert result == 2

    def test_trace_operation_with_attributes(self):
        """Test trace_operation with attributes."""
        with trace_operation("test_block", {"key": "value"}) as span:
            pass

    def test_trace_operation_exception(self):
        """Test trace_operation handles exceptions."""
        with pytest.raises(ValueError):
            with trace_operation("failing_block"):
                raise ValueError("test")


@pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
class TestSetupTelemetry:
    """Test setup_telemetry function when OTEL is available."""

    def test_setup_with_console_export(self):
        """Test setup with console export enabled."""
        config = TelemetryConfig(
            service_name="test-service",
            enable_console_export=True,
            enable_auto_instrumentation=False,
        )

        tracer = setup_telemetry(config=config)
        assert tracer is not None

    def test_setup_without_app(self):
        """Test setup without FastAPI app."""
        config = TelemetryConfig(
            service_name="test-service",
            enable_auto_instrumentation=False,
        )

        tracer = setup_telemetry(config=config)
        assert tracer is not None


class TestModuleExports:
    """Test module exports and availability."""

    def test_all_exports_available(self):
        """Test that all expected exports are available."""
        from src.core import observability

        assert hasattr(observability, "setup_telemetry")
        assert hasattr(observability, "get_tracer")
        assert hasattr(observability, "get_meter")
        assert hasattr(observability, "traced")
        assert hasattr(observability, "trace_operation")
        assert hasattr(observability, "TelemetryConfig")
        assert hasattr(observability, "OTEL_AVAILABLE")

    def test_otel_availability_flag(self):
        """Test OTEL_AVAILABLE flag is set."""
        from src.core.observability import OTEL_AVAILABLE

        assert isinstance(OTEL_AVAILABLE, bool)
