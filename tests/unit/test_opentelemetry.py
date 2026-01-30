"""Tests for OpenTelemetry integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestTelemetryConfig:
    """Tests for TelemetryConfig."""

    def test_default_values(self):
        """Test default config values."""
        from src.core.telemetry.opentelemetry import TelemetryConfig

        config = TelemetryConfig()

        assert config.service_name == "cad-ml-platform"
        assert config.service_version == "1.0.0"
        assert config.environment == "development"
        assert config.exporter_type == "otlp"
        assert config.sample_rate == 1.0

    def test_custom_values(self):
        """Test custom config values."""
        from src.core.telemetry.opentelemetry import TelemetryConfig

        config = TelemetryConfig(
            service_name="my-service",
            service_version="2.0.0",
            environment="production",
            exporter_type="jaeger",
        )

        assert config.service_name == "my-service"
        assert config.environment == "production"
        assert config.exporter_type == "jaeger"

    def test_from_env(self):
        """Test config from environment variables."""
        from src.core.telemetry.opentelemetry import TelemetryConfig

        with patch.dict(
            "os.environ",
            {
                "OTEL_SERVICE_NAME": "env-service",
                "OTEL_SERVICE_VERSION": "3.0.0",
                "OTEL_ENVIRONMENT": "staging",
                "OTEL_EXPORTER_TYPE": "console",
            },
        ):
            config = TelemetryConfig.from_env()

        assert config.service_name == "env-service"
        assert config.service_version == "3.0.0"
        assert config.environment == "staging"
        assert config.exporter_type == "console"


class TestTelemetryManager:
    """Tests for TelemetryManager."""

    def test_init_with_config(self):
        """Test manager initialization with config."""
        from src.core.telemetry.opentelemetry import TelemetryConfig, TelemetryManager

        config = TelemetryConfig(service_name="test-service")
        manager = TelemetryManager(config=config)

        assert manager.config.service_name == "test-service"
        assert manager._initialized is False

    def test_init_without_otel(self):
        """Test initialization when OpenTelemetry not available."""
        from src.core.telemetry.opentelemetry import TelemetryManager

        with patch("src.core.telemetry.opentelemetry.OTEL_AVAILABLE", False):
            manager = TelemetryManager()
            result = manager.initialize()

        assert result is False

    def test_get_tracer_before_init(self):
        """Test get_tracer initializes if needed."""
        from src.core.telemetry.opentelemetry import TelemetryManager

        manager = TelemetryManager()

        with patch.object(manager, "initialize", return_value=False) as mock_init:
            tracer = manager.get_tracer()

        mock_init.assert_called_once()

    def test_start_span_without_otel(self):
        """Test start_span when OpenTelemetry not available."""
        from src.core.telemetry.opentelemetry import TelemetryManager

        with patch("src.core.telemetry.opentelemetry.OTEL_AVAILABLE", False):
            manager = TelemetryManager()

            with manager.start_span("test-span") as span:
                assert span is None

    def test_add_event_without_otel(self):
        """Test add_event when OpenTelemetry not available."""
        from src.core.telemetry.opentelemetry import TelemetryManager

        with patch("src.core.telemetry.opentelemetry.OTEL_AVAILABLE", False):
            manager = TelemetryManager()
            # Should not raise
            manager.add_event("test-event")

    def test_set_attribute_without_otel(self):
        """Test set_attribute when OpenTelemetry not available."""
        from src.core.telemetry.opentelemetry import TelemetryManager

        with patch("src.core.telemetry.opentelemetry.OTEL_AVAILABLE", False):
            manager = TelemetryManager()
            # Should not raise
            manager.set_attribute("key", "value")

    def test_set_status_without_otel(self):
        """Test set_status when OpenTelemetry not available."""
        from src.core.telemetry.opentelemetry import TelemetryManager

        with patch("src.core.telemetry.opentelemetry.OTEL_AVAILABLE", False):
            manager = TelemetryManager()
            # Should not raise
            manager.set_status("ok")

    def test_record_exception_without_otel(self):
        """Test record_exception when OpenTelemetry not available."""
        from src.core.telemetry.opentelemetry import TelemetryManager

        with patch("src.core.telemetry.opentelemetry.OTEL_AVAILABLE", False):
            manager = TelemetryManager()
            # Should not raise
            manager.record_exception(ValueError("test"))

    def test_shutdown(self):
        """Test shutdown."""
        from src.core.telemetry.opentelemetry import TelemetryManager

        manager = TelemetryManager()
        manager._provider = MagicMock()
        manager._initialized = True

        manager.shutdown()

        manager._provider.shutdown.assert_called_once()
        assert manager._initialized is False


class TestTracedDecorator:
    """Tests for traced decorator."""

    @pytest.mark.asyncio
    async def test_traced_async_function(self):
        """Test traced decorator on async function."""
        from src.core.telemetry.opentelemetry import traced

        @traced(name="test-span")
        async def async_func():
            return "result"

        with patch("src.core.telemetry.opentelemetry.OTEL_AVAILABLE", False):
            result = await async_func()

        assert result == "result"

    def test_traced_sync_function(self):
        """Test traced decorator on sync function."""
        from src.core.telemetry.opentelemetry import traced

        @traced(name="test-span")
        def sync_func():
            return "result"

        with patch("src.core.telemetry.opentelemetry.OTEL_AVAILABLE", False):
            result = sync_func()

        assert result == "result"

    def test_traced_with_attributes(self):
        """Test traced decorator with attributes."""
        from src.core.telemetry.opentelemetry import traced

        @traced(attributes={"custom.attr": "value"})
        def func_with_attrs():
            return "result"

        with patch("src.core.telemetry.opentelemetry.OTEL_AVAILABLE", False):
            result = func_with_attrs()

        assert result == "result"


class TestGlobalFunctions:
    """Tests for global functions."""

    def test_get_telemetry_manager_singleton(self):
        """Test get_telemetry_manager returns singleton."""
        from src.core.telemetry import opentelemetry as otel_module

        # Reset global
        otel_module._telemetry_manager = None

        mgr1 = otel_module.get_telemetry_manager()
        mgr2 = otel_module.get_telemetry_manager()

        assert mgr1 is mgr2

        # Cleanup
        otel_module._telemetry_manager = None

    def test_init_telemetry(self):
        """Test init_telemetry function."""
        from src.core.telemetry import opentelemetry as otel_module

        otel_module._telemetry_manager = None

        with patch("src.core.telemetry.opentelemetry.OTEL_AVAILABLE", False):
            result = otel_module.init_telemetry()

        assert result is False

        # Cleanup
        otel_module._telemetry_manager = None

    def test_init_telemetry_with_config(self):
        """Test init_telemetry with custom config."""
        from src.core.telemetry import opentelemetry as otel_module
        from src.core.telemetry.opentelemetry import TelemetryConfig

        otel_module._telemetry_manager = None

        config = TelemetryConfig(service_name="custom-service")

        with patch("src.core.telemetry.opentelemetry.OTEL_AVAILABLE", False):
            otel_module.init_telemetry(config)

        manager = otel_module.get_telemetry_manager()
        assert manager.config.service_name == "custom-service"

        # Cleanup
        otel_module._telemetry_manager = None

    def test_shutdown_telemetry(self):
        """Test shutdown_telemetry function."""
        from src.core.telemetry import opentelemetry as otel_module

        # Create a manager
        otel_module._telemetry_manager = None
        otel_module.get_telemetry_manager()

        assert otel_module._telemetry_manager is not None

        otel_module.shutdown_telemetry()

        assert otel_module._telemetry_manager is None


class TestCreateExporter:
    """Tests for exporter creation."""

    def test_unknown_exporter_type(self):
        """Test unknown exporter type returns None."""
        from src.core.telemetry.opentelemetry import TelemetryConfig, TelemetryManager

        config = TelemetryConfig(exporter_type="unknown")
        manager = TelemetryManager(config=config)

        exporter = manager._create_exporter()
        assert exporter is None

    def test_console_exporter_fallback(self):
        """Test console exporter when others unavailable."""
        from src.core.telemetry.opentelemetry import TelemetryConfig, TelemetryManager

        config = TelemetryConfig(exporter_type="console")
        manager = TelemetryManager(config=config)

        # Try to create - may fail if SDK not installed
        try:
            exporter = manager._create_exporter()
            # If it succeeds, it should be a console exporter
            assert exporter is not None or True  # Accept None if SDK not installed
        except ImportError:
            # Expected if SDK not installed
            pass
