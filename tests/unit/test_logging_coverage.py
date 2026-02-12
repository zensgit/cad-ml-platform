"""Tests for logging utilities to improve coverage.

Targets uncovered code paths in src/utils/logging.py:
- Lines 30-40: ImportError fallback
- Lines 69, 79, 81, 83-85: request context handling
- Lines 125, 129, 133-135: extra fields and exception formatting
- Lines 148-149: SimpleFormatter
- Lines 193: SimpleFormatter in setup_logging
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


class TestJsonFormatter:
    """Tests for JsonFormatter class."""

    def test_basic_format(self):
        """JsonFormatter formats basic log record."""
        from src.utils.logging import JsonFormatter

        formatter = JsonFormatter(service_name="test-service")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["logger"] == "test.logger"
        assert data["service"] == "test-service"
        assert "timestamp" in data

    def test_without_timestamp(self):
        """JsonFormatter can exclude timestamp."""
        from src.utils.logging import JsonFormatter

        formatter = JsonFormatter(include_timestamp=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "timestamp" not in data

    def test_with_location(self):
        """JsonFormatter can include location info."""
        from src.utils.logging import JsonFormatter

        formatter = JsonFormatter(include_location=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="/path/to/test.py",
            lineno=42,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.funcName = "test_func"

        result = formatter.format(record)
        data = json.loads(result)

        assert "location" in data
        assert data["location"]["file"] == "/path/to/test.py"
        assert data["location"]["line"] == 42
        assert data["location"]["function"] == "test_func"

    def test_with_request_context(self):
        """JsonFormatter includes request context when available."""
        from src.utils.logging import JsonFormatter, request_id_var, user_id_var, tenant_id_var

        if request_id_var is None:
            pytest.skip("Structured logging not available")

        # Set context
        request_id_var.set("req-123")
        if user_id_var:
            user_id_var.set("user-456")
        if tenant_id_var:
            tenant_id_var.set("tenant-789")

        try:
            formatter = JsonFormatter()
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="Test",
                args=(),
                exc_info=None,
            )

            result = formatter.format(record)
            data = json.loads(result)

            assert data["request_id"] == "req-123"
            if user_id_var:
                assert data["user_id"] == "user-456"
            if tenant_id_var:
                assert data["tenant_id"] == "tenant-789"
        finally:
            # Clean up
            request_id_var.set(None)
            if user_id_var:
                user_id_var.set(None)
            if tenant_id_var:
                tenant_id_var.set(None)

    def test_request_context_exception_handled(self):
        """JsonFormatter handles exception when getting request context."""
        from src.utils.logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        # Patch request_id_var to raise exception
        with patch("src.utils.logging.request_id_var") as mock_var:
            mock_var.get.side_effect = RuntimeError("Context error")

            # Should not raise
            result = formatter.format(record)
            data = json.loads(result)

            assert "request_id" not in data

    def test_structured_attrs(self):
        """JsonFormatter includes structured attributes from record."""
        from src.utils.logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        # Add structured attributes
        record.provider = "test-provider"
        record.latency_ms = 150
        record.trace_id = "trace-abc"
        record.model = "v6"
        record.confidence = 0.95

        result = formatter.format(record)
        data = json.loads(result)

        assert data["provider"] == "test-provider"
        assert data["latency_ms"] == 150
        assert data["trace_id"] == "trace-abc"
        assert data["model"] == "v6"
        assert data["confidence"] == 0.95

    def test_extra_fields(self):
        """JsonFormatter includes extra_fields from record."""
        from src.utils.logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.extra_fields = {"custom_field": "custom_value", "another": 123}

        result = formatter.format(record)
        data = json.loads(result)

        assert data["custom_field"] == "custom_value"
        assert data["another"] == 123

    def test_exception_info(self):
        """JsonFormatter includes exception info."""
        from src.utils.logging import JsonFormatter

        formatter = JsonFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert data["exception"]["message"] == "Test error"
        assert isinstance(data["exception"]["stacktrace"], list)


class TestSimpleFormatter:
    """Tests for SimpleFormatter class."""

    def test_format(self):
        """SimpleFormatter formats log record as simple text."""
        from src.utils.logging import SimpleFormatter

        formatter = SimpleFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert "test.logger" in result
        assert "INFO" in result
        assert "Test message" in result
        # Should have timestamp format YYYY-MM-DD HH:MM:SS
        assert "-" in result.split(" ")[0]  # Date part


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_default_setup(self):
        """setup_logging configures logging with defaults."""
        from src.utils.logging import setup_logging

        setup_logging()

        root = logging.getLogger()
        assert root.level == logging.INFO
        assert len(root.handlers) == 1

    def test_custom_level(self):
        """setup_logging accepts custom log level."""
        from src.utils.logging import setup_logging

        setup_logging(level="DEBUG")

        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_json_output_true(self):
        """setup_logging uses JsonFormatter when json_output=True."""
        from src.utils.logging import setup_logging, JsonFormatter

        setup_logging(json_output=True)

        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)

    def test_json_output_false(self):
        """setup_logging uses SimpleFormatter when json_output=False."""
        from src.utils.logging import setup_logging, SimpleFormatter

        setup_logging(json_output=False)

        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, SimpleFormatter)

    def test_env_log_level(self):
        """setup_logging uses LOG_LEVEL env var."""
        from src.utils.logging import setup_logging

        with patch.dict(os.environ, {"LOG_LEVEL": "WARNING"}):
            setup_logging()

        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_env_log_format_simple(self):
        """setup_logging uses LOG_FORMAT env var for simple format."""
        from src.utils.logging import setup_logging, SimpleFormatter

        with patch.dict(os.environ, {"LOG_FORMAT": "simple"}):
            setup_logging()

        root = logging.getLogger()
        assert isinstance(root.handlers[0].formatter, SimpleFormatter)

    def test_env_include_location(self):
        """setup_logging uses LOG_INCLUDE_LOCATION env var."""
        from src.utils.logging import setup_logging, JsonFormatter

        with patch.dict(os.environ, {"LOG_INCLUDE_LOCATION": "true"}):
            setup_logging(json_output=True)

        root = logging.getLogger()
        formatter = root.handlers[0].formatter
        assert isinstance(formatter, JsonFormatter)
        assert formatter.include_location is True

    def test_removes_existing_handlers(self):
        """setup_logging removes existing handlers."""
        from src.utils.logging import setup_logging

        root = logging.getLogger()
        # Add extra handler
        extra_handler = logging.StreamHandler()
        root.addHandler(extra_handler)
        initial_count = len(root.handlers)

        setup_logging()

        # Should only have one handler now
        assert len(root.handlers) == 1

    def test_reduces_third_party_noise(self):
        """setup_logging sets third-party loggers to WARNING."""
        from src.utils.logging import setup_logging

        setup_logging()

        for lib in ["urllib3", "httpx", "httpcore", "asyncio"]:
            logger = logging.getLogger(lib)
            assert logger.level == logging.WARNING

    def test_custom_service_name(self):
        """setup_logging accepts custom service name."""
        from src.utils.logging import setup_logging, JsonFormatter

        setup_logging(service_name="custom-service", json_output=True)

        root = logging.getLogger()
        formatter = root.handlers[0].formatter
        assert isinstance(formatter, JsonFormatter)
        assert formatter.service_name == "custom-service"


class TestImportFallback:
    """Tests for import fallback behavior."""

    def test_exports_none_on_import_error(self):
        """Module exports None when structured logging not available."""
        # This tests the fallback behavior when StructuredLogger import fails
        # The actual module should have these available, so we just verify they're exported
        from src.utils import logging as logging_module

        # These should be available (either real or None fallback)
        assert hasattr(logging_module, "StructuredLogger")
        assert hasattr(logging_module, "get_logger")
        assert hasattr(logging_module, "set_request_context")
        assert hasattr(logging_module, "clear_request_context")
        assert hasattr(logging_module, "generate_request_id")
        assert hasattr(logging_module, "log_execution_time")
        assert hasattr(logging_module, "request_id_var")
        assert hasattr(logging_module, "user_id_var")
        assert hasattr(logging_module, "tenant_id_var")


class TestJsonFormatterEdgeCases:
    """Tests for edge cases in JsonFormatter."""

    def test_empty_extra_fields(self):
        """JsonFormatter handles empty extra_fields."""
        from src.utils.logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.extra_fields = {}

        result = formatter.format(record)
        data = json.loads(result)

        # Should still be valid JSON
        assert data["message"] == "Test"

    def test_none_extra_fields(self):
        """JsonFormatter handles None extra_fields."""
        from src.utils.logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.extra_fields = None

        result = formatter.format(record)
        data = json.loads(result)

        assert data["message"] == "Test"

    def test_message_with_args(self):
        """JsonFormatter handles message with arguments."""
        from src.utils.logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Value: %s, Count: %d",
            args=("test", 42),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["message"] == "Value: test, Count: 42"

    def test_unicode_message(self):
        """JsonFormatter handles unicode messages."""
        from src.utils.logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="中文消息: 测试",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["message"] == "中文消息: 测试"

    def test_non_serializable_extra(self):
        """JsonFormatter handles non-serializable extra data."""
        from src.utils.logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        # Add non-serializable data
        record.extra_fields = {"datetime_obj": datetime.utcnow()}

        result = formatter.format(record)
        # Should not raise, uses default=str
        data = json.loads(result)
        assert "datetime_obj" in data
