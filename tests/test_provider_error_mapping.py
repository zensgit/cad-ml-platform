"""Test provider error mapping to ErrorCode."""

import asyncio
from unittest.mock import Mock, patch

import pytest

from src.core.errors import ErrorCode
from src.core.ocr.providers.error_map import (
    handle_inference_error,
    handle_init_error,
    handle_load_error,
    handle_parse_error,
    log_and_map_exception,
    map_exception_to_error_code,
)


class TestExceptionMapping:
    """Test exception to ErrorCode mapping."""

    def test_memory_error_maps_to_resource_exhausted(self):
        """MemoryError should map to RESOURCE_EXHAUSTED."""
        exc = MemoryError("Out of memory")
        code = map_exception_to_error_code(exc)
        assert code == ErrorCode.RESOURCE_EXHAUSTED

    def test_timeout_errors_map_to_provider_timeout(self):
        """TimeoutError and asyncio.TimeoutError should map to PROVIDER_TIMEOUT."""
        exc1 = TimeoutError("Operation timed out")
        assert map_exception_to_error_code(exc1) == ErrorCode.PROVIDER_TIMEOUT

        exc2 = asyncio.TimeoutError("Async timeout")
        assert map_exception_to_error_code(exc2) == ErrorCode.PROVIDER_TIMEOUT

    def test_parse_value_error_maps_to_parse_failed(self):
        """ValueError with parse-related message should map to PARSE_FAILED."""
        exc1 = ValueError("Failed to parse JSON")
        assert map_exception_to_error_code(exc1) == ErrorCode.PARSE_FAILED

        exc2 = ValueError("Invalid format in response")
        assert map_exception_to_error_code(exc2) == ErrorCode.PARSE_FAILED

        exc3 = ValueError("Malformed data")
        assert map_exception_to_error_code(exc3) == ErrorCode.PARSE_FAILED

    def test_non_parse_value_error_maps_to_input_error(self):
        """ValueError without parse context should map to INPUT_ERROR."""
        exc = ValueError("Invalid input value")
        assert map_exception_to_error_code(exc) == ErrorCode.INPUT_ERROR

    def test_connection_errors_map_to_network_error(self):
        """Connection errors should map to NETWORK_ERROR."""
        exc1 = ConnectionError("Connection failed")
        assert map_exception_to_error_code(exc1) == ErrorCode.NETWORK_ERROR

        exc2 = ConnectionRefusedError("Connection refused")
        assert map_exception_to_error_code(exc2) == ErrorCode.NETWORK_ERROR

        exc3 = ConnectionAbortedError("Connection aborted")
        assert map_exception_to_error_code(exc3) == ErrorCode.NETWORK_ERROR

    def test_io_permission_error_maps_to_auth_failed(self):
        """IOError with permission message should map to AUTH_FAILED."""
        exc = IOError("Permission denied")
        assert map_exception_to_error_code(exc) == ErrorCode.AUTH_FAILED

    def test_generic_io_error_maps_to_network_error(self):
        """Generic IOError should map to NETWORK_ERROR."""
        exc = IOError("File not accessible")
        assert map_exception_to_error_code(exc) == ErrorCode.NETWORK_ERROR

    def test_message_pattern_detection(self):
        """Test error message pattern detection."""
        # Network patterns
        exc1 = RuntimeError("Network connection lost")
        assert map_exception_to_error_code(exc1) == ErrorCode.NETWORK_ERROR

        exc2 = RuntimeError("DNS resolution failed")
        assert map_exception_to_error_code(exc2) == ErrorCode.NETWORK_ERROR

        # Timeout patterns
        exc3 = RuntimeError("Request timed out after 30s")
        assert map_exception_to_error_code(exc3) == ErrorCode.PROVIDER_TIMEOUT

        # Resource patterns
        exc4 = RuntimeError("OOM: Out of memory")
        assert map_exception_to_error_code(exc4) == ErrorCode.RESOURCE_EXHAUSTED

        # Auth patterns
        exc5 = RuntimeError("Authentication required")
        assert map_exception_to_error_code(exc5) == ErrorCode.AUTH_FAILED

        exc6 = RuntimeError("403 Forbidden")
        assert map_exception_to_error_code(exc6) == ErrorCode.AUTH_FAILED

        # Model patterns
        exc7 = RuntimeError("Failed to load model weights")
        assert map_exception_to_error_code(exc7) == ErrorCode.MODEL_LOAD_ERROR

    def test_unknown_error_maps_to_internal_error(self):
        """Unknown exceptions should map to INTERNAL_ERROR."""
        exc1 = RuntimeError("Something went wrong")
        assert map_exception_to_error_code(exc1) == ErrorCode.INTERNAL_ERROR

        exc2 = Exception("Generic error")
        assert map_exception_to_error_code(exc2) == ErrorCode.INTERNAL_ERROR

        exc3 = TypeError("Type mismatch")
        assert map_exception_to_error_code(exc3) == ErrorCode.INTERNAL_ERROR


class TestLogAndMap:
    """Test logging and mapping functions."""

    @patch("src.core.ocr.providers.error_map.logger")
    def test_log_and_map_logs_at_correct_level(self, mock_logger):
        """Test that exceptions are logged at appropriate levels."""
        # Resource exhausted - error level
        exc1 = MemoryError("Out of memory")
        code1 = log_and_map_exception(exc1, "test_provider", "infer")
        assert code1 == ErrorCode.RESOURCE_EXHAUSTED
        assert mock_logger.error.called

        # Network error - warning level
        mock_logger.reset_mock()
        exc2 = ConnectionError("Network down")
        code2 = log_and_map_exception(exc2, "test_provider", "infer")
        assert code2 == ErrorCode.NETWORK_ERROR
        assert mock_logger.warning.called

        # Generic error - info level
        mock_logger.reset_mock()
        exc3 = RuntimeError("Some error")
        code3 = log_and_map_exception(exc3, "test_provider", "parse")
        assert code3 == ErrorCode.INTERNAL_ERROR
        assert mock_logger.info.called

    @patch("src.core.ocr.providers.error_map.logger")
    def test_log_includes_context(self, mock_logger):
        """Test that context is included in log messages."""
        exc = RuntimeError("Test error")
        log_and_map_exception(exc, "provider", "stage", context="extra info")

        # Check that debug log was called with context
        debug_calls = mock_logger.debug.call_args_list
        assert len(debug_calls) > 0
        assert "extra info" in str(debug_calls[0])


class TestConvenienceFunctions:
    """Test convenience error handling functions."""

    def test_handle_inference_error(self):
        """Test inference error handling."""
        exc = TimeoutError("Inference timeout")
        with patch("src.core.ocr.providers.error_map.log_and_map_exception") as mock_log:
            mock_log.return_value = ErrorCode.PROVIDER_TIMEOUT
            code = handle_inference_error(exc, "test_provider")
            mock_log.assert_called_once_with(exc, "test_provider", "infer")
            assert code == ErrorCode.PROVIDER_TIMEOUT

    def test_handle_parse_error_always_parse_failed(self):
        """Parse errors should always return PARSE_FAILED for common exceptions."""
        exc1 = ValueError("Parse error")
        assert handle_parse_error(exc1, "provider") == ErrorCode.PARSE_FAILED

        exc2 = TypeError("Type error")
        assert handle_parse_error(exc2, "provider") == ErrorCode.PARSE_FAILED

        exc3 = KeyError("Missing key")
        assert handle_parse_error(exc3, "provider") == ErrorCode.PARSE_FAILED

        exc4 = AttributeError("Missing attr")
        assert handle_parse_error(exc4, "provider") == ErrorCode.PARSE_FAILED

    def test_handle_init_error(self):
        """Test init error handling."""
        exc = RuntimeError("Init failed")
        with patch("src.core.ocr.providers.error_map.log_and_map_exception") as mock_log:
            mock_log.return_value = ErrorCode.INTERNAL_ERROR
            code = handle_init_error(exc, "test_provider")
            mock_log.assert_called_once_with(exc, "test_provider", "init")
            assert code == ErrorCode.INTERNAL_ERROR

    def test_handle_load_error_memory(self):
        """Load errors should detect memory issues."""
        exc1 = MemoryError("OOM during load")
        assert handle_load_error(exc1, "provider") == ErrorCode.RESOURCE_EXHAUSTED

        exc2 = RuntimeError("Model load failed")
        assert handle_load_error(exc2, "provider") == ErrorCode.MODEL_LOAD_ERROR


class TestIntegrationWithProviders:
    """Test integration with actual provider code patterns."""

    @pytest.mark.asyncio
    async def test_provider_uses_error_mapping(self):
        """Verify providers can use the error mapping."""
        from src.core.ocr.providers.error_map import map_exception_to_error_code

        # Simulate provider exception handling
        try:
            # Simulate a timeout
            raise asyncio.TimeoutError("Provider timeout")
        except Exception as e:
            code = map_exception_to_error_code(e)
            assert code == ErrorCode.PROVIDER_TIMEOUT

        # Simulate memory error
        try:
            raise MemoryError("Out of memory")
        except Exception as e:
            code = map_exception_to_error_code(e)
            assert code == ErrorCode.RESOURCE_EXHAUSTED

    def test_metrics_label_generation(self):
        """Test that error codes can be used as metric labels."""
        from src.core.errors import ErrorCode

        # All ErrorCodes should have string values suitable for labels
        for code in ErrorCode:
            assert isinstance(code.value, str)
            assert code.value.replace("_", "").isalpha()  # Valid label format


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
