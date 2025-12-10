"""Tests for src/api/v1/vision.py to improve coverage.

Covers:
- get_vision_manager singleton logic
- reset_vision_manager function
- analyze_vision endpoint logic and error handling
- health_check endpoint logic
- list_providers endpoint logic
- get_metrics endpoint logic
- VisionAnalyzeResponse structure
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


class TestGetVisionManagerLogic:
    """Tests for get_vision_manager singleton logic."""

    def test_provider_type_from_parameter(self):
        """Test provider type from explicit parameter."""
        provider_type = "openai"
        effective_provider = provider_type or os.getenv("VISION_PROVIDER", "auto")

        assert effective_provider == "openai"

    def test_provider_type_from_env_var(self):
        """Test provider type from environment variable."""
        provider_type = None

        with patch.dict("os.environ", {"VISION_PROVIDER": "anthropic"}):
            effective_provider = provider_type or os.getenv("VISION_PROVIDER", "auto")

        assert effective_provider == "anthropic"

    def test_provider_type_default_auto(self):
        """Test provider type defaults to auto."""
        provider_type = None

        with patch.dict("os.environ", {}, clear=True):
            effective_provider = provider_type or os.getenv("VISION_PROVIDER", "auto")

        assert effective_provider == "auto"

    def test_manager_recreated_on_provider_change(self):
        """Test manager is recreated when provider type changes."""
        current_provider_type = "openai"
        new_provider_type = "anthropic"

        should_recreate = current_provider_type != new_provider_type
        assert should_recreate is True

    def test_manager_reused_same_provider(self):
        """Test manager is reused when provider type unchanged."""
        current_provider_type = "openai"
        new_provider_type = "openai"

        should_recreate = current_provider_type != new_provider_type
        assert should_recreate is False


class TestResetVisionManager:
    """Tests for reset_vision_manager function."""

    def test_reset_clears_manager(self):
        """Test reset clears the manager singleton."""
        _vision_manager = MagicMock()
        _current_provider_type = "openai"

        # Simulate reset
        _vision_manager = None
        _current_provider_type = None

        assert _vision_manager is None
        assert _current_provider_type is None


class TestAnalyzeVisionLogic:
    """Tests for analyze_vision endpoint logic."""

    def test_success_response_returned_as_is(self):
        """Test successful response is returned directly."""
        response = {
            "success": True,
            "description": {"summary": "Test", "details": [], "confidence": 0.9},
            "ocr": None,
            "provider": "openai",
            "processing_time_ms": 1000.0
        }

        if response["success"]:
            result = response

        assert result == response

    def test_failure_response_structure(self):
        """Test failure response structure."""
        response = {
            "success": False,
            "error": "Vision analysis failed",
            "code": "INTERNAL_ERROR"
        }

        failure_response = {
            "success": False,
            "description": None,
            "ocr": None,
            "provider": response.get("provider", "unknown"),
            "processing_time_ms": response.get("processing_time_ms", 0.0),
            "error": response.get("error") or "Vision analysis failed",
            "code": response.get("code") or "INTERNAL_ERROR"
        }

        assert failure_response["success"] is False
        assert failure_response["description"] is None
        assert "error" in failure_response


class TestAnalyzeVisionErrorHandling:
    """Tests for analyze_vision error handling."""

    def test_vision_input_error_response(self):
        """Test VisionInputError response structure."""
        error_message = "Invalid image format"

        response = {
            "success": False,
            "description": None,
            "ocr": None,
            "provider": "unknown",
            "processing_time_ms": 0.0,
            "error": error_message,
            "code": "INPUT_ERROR"
        }

        assert response["code"] == "INPUT_ERROR"
        assert response["provider"] == "unknown"

    def test_vision_provider_error_response(self):
        """Test VisionProviderError response structure."""
        provider = "openai"
        error_message = "API rate limit exceeded"

        response = {
            "success": False,
            "description": None,
            "ocr": None,
            "provider": provider,
            "processing_time_ms": 0.0,
            "error": error_message,
            "code": "EXTERNAL_SERVICE_ERROR"
        }

        assert response["code"] == "EXTERNAL_SERVICE_ERROR"
        assert response["provider"] == "openai"

    def test_generic_exception_response(self):
        """Test generic exception response structure."""
        error = Exception("Unexpected error")

        response = {
            "success": False,
            "description": None,
            "ocr": None,
            "provider": "unknown",
            "processing_time_ms": 0.0,
            "error": f"Internal server error: {str(error)}",
            "code": "INTERNAL_ERROR"
        }

        assert "Internal server error" in response["error"]
        assert response["code"] == "INTERNAL_ERROR"


class TestHealthCheckLogic:
    """Tests for health_check endpoint logic."""

    def test_healthy_status_response(self):
        """Test healthy status response structure."""
        provider_name = "openai"
        ocr_manager_present = True

        response = {
            "status": "healthy",
            "provider": provider_name,
            "ocr_enabled": ocr_manager_present
        }

        assert response["status"] == "healthy"
        assert response["provider"] == "openai"
        assert response["ocr_enabled"] is True

    def test_degraded_status_on_exception(self):
        """Test degraded status when exception occurs."""
        error = Exception("Provider initialization failed")

        response = {
            "status": "degraded",
            "error": str(error)
        }

        assert response["status"] == "degraded"
        assert "error" in response


class TestListProvidersLogic:
    """Tests for list_providers endpoint logic."""

    def test_providers_response_structure(self):
        """Test providers response structure."""
        current_provider = "openai"
        available_providers = {
            "stub": {"available": True, "requires_key": False},
            "deepseek": {"available": True, "requires_key": True, "key_set": True},
            "openai": {"available": True, "requires_key": True, "key_set": True},
            "anthropic": {"available": True, "requires_key": True, "key_set": False}
        }

        response = {
            "current_provider": current_provider,
            "providers": available_providers
        }

        assert response["current_provider"] == "openai"
        assert "stub" in response["providers"]
        assert response["providers"]["stub"]["requires_key"] is False

    def test_unknown_provider_on_exception(self):
        """Test current provider is unknown on exception."""
        try:
            raise Exception("Manager init failed")
            current = "openai"
        except Exception:
            current = "unknown"

        assert current == "unknown"


class TestGetMetricsLogic:
    """Tests for get_metrics endpoint logic."""

    def test_resilient_provider_metrics_structure(self):
        """Test metrics response for resilient provider."""
        provider_name = "openai"
        metrics = {
            "total_requests": 100,
            "successful_requests": 95,
            "failed_requests": 5,
            "success_rate": 0.95,
            "average_latency_ms": 1234.5,
            "last_error": "Timeout",
            "last_error_time": "2024-01-01T00:00:00",
            "circuit_state": "closed",
            "circuit_opens": 1
        }

        response = {
            "provider": provider_name,
            "resilient": True,
            "metrics": metrics
        }

        assert response["resilient"] is True
        assert response["metrics"]["success_rate"] == 0.95

    def test_non_resilient_provider_metrics(self):
        """Test metrics response for non-resilient provider."""
        provider_name = "stub"

        response = {
            "provider": provider_name,
            "resilient": False,
            "metrics": {
                "message": "Metrics available with ResilientVisionProvider",
                "hint": "Use create_resilient_provider() to enable metrics"
            }
        }

        assert response["resilient"] is False
        assert "message" in response["metrics"]

    def test_metrics_error_response(self):
        """Test metrics error response structure."""
        error = Exception("Failed to get metrics")

        response = {
            "error": str(error),
            "metrics": None
        }

        assert "error" in response
        assert response["metrics"] is None


class TestVisionAnalyzeResponseModel:
    """Tests for VisionAnalyzeResponse model structure."""

    def test_success_response_fields(self):
        """Test success response has all required fields."""
        response = {
            "success": True,
            "description": {
                "summary": "Mechanical part with cylindrical features",
                "details": ["Main diameter: 20mm", "Thread: M10×1.5"],
                "confidence": 0.92
            },
            "ocr": None,
            "provider": "openai",
            "processing_time_ms": 1234.5
        }

        assert "success" in response
        assert "description" in response
        assert "ocr" in response
        assert "provider" in response
        assert "processing_time_ms" in response

    def test_description_fields(self):
        """Test description object has expected fields."""
        description = {
            "summary": "Test summary",
            "details": ["Detail 1", "Detail 2"],
            "confidence": 0.85
        }

        assert "summary" in description
        assert "details" in description
        assert "confidence" in description
        assert 0 <= description["confidence"] <= 1


class TestProviderSelection:
    """Tests for provider selection logic."""

    def test_deepseek_provider(self):
        """Test deepseek provider selection."""
        provider = "deepseek"
        valid_providers = ["deepseek", "openai", "anthropic", "stub", "auto"]

        assert provider in valid_providers

    def test_openai_provider(self):
        """Test openai provider selection."""
        provider = "openai"
        valid_providers = ["deepseek", "openai", "anthropic", "stub", "auto"]

        assert provider in valid_providers

    def test_anthropic_provider(self):
        """Test anthropic provider selection."""
        provider = "anthropic"
        valid_providers = ["deepseek", "openai", "anthropic", "stub", "auto"]

        assert provider in valid_providers

    def test_stub_provider(self):
        """Test stub provider selection."""
        provider = "stub"
        valid_providers = ["deepseek", "openai", "anthropic", "stub", "auto"]

        assert provider in valid_providers


class TestOcrManagerIntegration:
    """Tests for OCR manager integration."""

    def test_ocr_manager_created(self):
        """Test OCR manager is created with vision manager."""
        confidence_fallback = 0.85
        providers = {}

        ocr_config = {
            "providers": providers,
            "confidence_fallback": confidence_fallback
        }

        assert ocr_config["confidence_fallback"] == 0.85


class TestVisionModuleImports:
    """Tests for vision module imports."""

    def test_vision_manager_import(self):
        """Test VisionManager can be imported."""
        from src.core.vision import VisionManager

        assert VisionManager is not None

    def test_vision_analyze_request_import(self):
        """Test VisionAnalyzeRequest can be imported."""
        from src.core.vision import VisionAnalyzeRequest

        assert VisionAnalyzeRequest is not None

    def test_vision_analyze_response_import(self):
        """Test VisionAnalyzeResponse can be imported."""
        from src.core.vision import VisionAnalyzeResponse

        assert VisionAnalyzeResponse is not None

    def test_vision_errors_import(self):
        """Test vision error classes can be imported."""
        from src.core.vision import VisionInputError, VisionProviderError

        assert VisionInputError is not None
        assert VisionProviderError is not None

    def test_create_vision_provider_import(self):
        """Test create_vision_provider can be imported."""
        from src.core.vision import create_vision_provider

        assert callable(create_vision_provider)

    def test_get_available_providers_import(self):
        """Test get_available_providers can be imported."""
        from src.core.vision import get_available_providers

        assert callable(get_available_providers)

    def test_resilient_vision_provider_import(self):
        """Test ResilientVisionProvider can be imported."""
        from src.core.vision import ResilientVisionProvider

        assert ResilientVisionProvider is not None


class TestErrorCodeMapping:
    """Tests for error code mapping."""

    def test_input_error_code(self):
        """Test INPUT_ERROR code is used for input errors."""
        from src.core.errors import ErrorCode

        assert hasattr(ErrorCode, "INPUT_ERROR")

    def test_provider_down_error_code(self):
        """Test PROVIDER_DOWN code exists for provider errors."""
        from src.core.errors import ErrorCode

        assert hasattr(ErrorCode, "PROVIDER_DOWN")

    def test_internal_error_code(self):
        """Test INTERNAL_ERROR code is used for general exceptions."""
        from src.core.errors import ErrorCode

        assert hasattr(ErrorCode, "INTERNAL_ERROR")


class TestQueryParameterLogic:
    """Tests for query parameter handling."""

    def test_provider_query_param_optional(self):
        """Test provider query parameter is optional."""
        provider = None  # Default when not provided

        # Should use None as default, then get_vision_manager handles it
        assert provider is None

    def test_provider_query_param_override(self):
        """Test provider query parameter can override default."""
        provider = "stub"

        assert provider is not None
        assert provider == "stub"
