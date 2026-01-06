"""Tests for src/api/v1/vision.py to improve coverage.

Covers:
- get_vision_manager function
- reset_vision_manager function
- analyze_vision endpoint
- health_check endpoint
- list_providers endpoint
- get_metrics endpoint
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestGetVisionManager:
    """Tests for get_vision_manager function."""

    def test_creates_manager_first_time(self):
        """Test get_vision_manager creates manager on first call."""
        from src.api.v1 import vision

        # Reset singleton
        vision._vision_manager = None
        vision._current_provider_type = None

        with patch("src.api.v1.vision.create_vision_provider") as mock_create:
            with patch("src.core.ocr.manager.OcrManager"):
                mock_provider = MagicMock()
                mock_create.return_value = mock_provider

                manager = vision.get_vision_manager()

                assert manager is not None
                mock_create.assert_called_once()

    def test_returns_existing_manager(self):
        """Test get_vision_manager returns existing manager."""
        from src.api.v1 import vision

        # Reset and create manager
        vision._vision_manager = None
        vision._current_provider_type = None

        with patch("src.api.v1.vision.create_vision_provider") as mock_create:
            with patch("src.core.ocr.manager.OcrManager"):
                mock_provider = MagicMock()
                mock_create.return_value = mock_provider

                manager1 = vision.get_vision_manager()
                manager2 = vision.get_vision_manager()

                # Same provider type should return same manager
                assert manager1 is manager2
                # Should only create provider once
                assert mock_create.call_count == 1

    def test_recreates_manager_on_provider_change(self):
        """Test get_vision_manager recreates manager when provider changes."""
        from src.api.v1 import vision

        # Reset singleton
        vision._vision_manager = None
        vision._current_provider_type = None

        with patch("src.api.v1.vision.create_vision_provider") as mock_create:
            with patch("src.core.ocr.manager.OcrManager"):
                mock_provider = MagicMock()
                mock_create.return_value = mock_provider

                manager1 = vision.get_vision_manager(provider_type="stub")
                manager2 = vision.get_vision_manager(provider_type="deepseek")

                # Different provider types should create new managers
                assert mock_create.call_count == 2

    def test_uses_env_var_for_provider(self):
        """Test get_vision_manager uses VISION_PROVIDER env var."""
        from src.api.v1 import vision

        vision._vision_manager = None
        vision._current_provider_type = None

        with patch("src.api.v1.vision.create_vision_provider") as mock_create:
            with patch("src.core.ocr.manager.OcrManager"):
                with patch.dict("os.environ", {"VISION_PROVIDER": "openai"}):
                    mock_provider = MagicMock()
                    mock_create.return_value = mock_provider

                    vision.get_vision_manager()

                    mock_create.assert_called_once()
                    call_kwargs = mock_create.call_args[1]
                    assert call_kwargs["provider_type"] == "openai"


class TestResetVisionManager:
    """Tests for reset_vision_manager function."""

    def test_resets_singleton(self):
        """Test reset_vision_manager clears singleton."""
        from src.api.v1 import vision

        # Set some values
        vision._vision_manager = MagicMock()
        vision._current_provider_type = "test"

        vision.reset_vision_manager()

        assert vision._vision_manager is None
        assert vision._current_provider_type is None


class TestAnalyzeVisionEndpoint:
    """Tests for analyze_vision endpoint."""

    @pytest.mark.asyncio
    async def test_successful_analysis(self):
        """Test analyze_vision returns successful response."""
        from src.api.v1.vision import analyze_vision
        from src.core.vision import VisionAnalyzeRequest, VisionAnalyzeResponse

        request = VisionAnalyzeRequest(
            image_base64="dGVzdGltYWdl",  # base64 "testimage"
            include_description=True,
            include_ocr=False,
        )

        mock_response = VisionAnalyzeResponse(
            success=True,
            description={"summary": "Test image", "details": [], "confidence": 0.9},
            ocr=None,
            provider="stub",
            processing_time_ms=100.0,
        )

        with patch("src.api.v1.vision.get_vision_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.analyze = AsyncMock(return_value=mock_response)
            mock_get_manager.return_value = mock_manager

            result = await analyze_vision(request, provider=None)

            assert result.success is True
            assert result.provider == "stub"

    @pytest.mark.asyncio
    async def test_passes_cad_feature_thresholds(self):
        """Test analyze_vision passes CAD threshold overrides to the manager."""
        from src.api.v1.vision import analyze_vision
        from src.core.vision import VisionAnalyzeRequest, VisionAnalyzeResponse

        request = VisionAnalyzeRequest(
            image_base64="dGVzdGltYWdl",
            include_description=True,
            cad_feature_thresholds={"line_aspect": 5.0, "arc_fill_min": 0.08},
        )

        mock_response = VisionAnalyzeResponse(
            success=True,
            description={"summary": "Test image", "details": [], "confidence": 0.9},
            ocr=None,
            provider="stub",
            processing_time_ms=100.0,
        )

        with patch("src.api.v1.vision.get_vision_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.analyze = AsyncMock(return_value=mock_response)
            mock_get_manager.return_value = mock_manager

            result = await analyze_vision(request, provider=None)

            assert result.success is True
            passed_request = mock_manager.analyze.call_args[0][0]
            assert passed_request.cad_feature_thresholds == {
                "line_aspect": 5.0,
                "arc_fill_min": 0.08,
            }

    @pytest.mark.asyncio
    async def test_failed_analysis_returns_error_response(self):
        """Test analyze_vision returns error response on failure."""
        from src.api.v1.vision import analyze_vision
        from src.core.errors import ErrorCode
        from src.core.vision import VisionAnalyzeRequest, VisionAnalyzeResponse

        request = VisionAnalyzeRequest(
            image_base64="dGVzdGltYWdl",
            include_description=True,
        )

        mock_response = VisionAnalyzeResponse(
            success=False,
            description=None,
            ocr=None,
            provider="stub",
            processing_time_ms=50.0,
            error="Analysis failed",
            code=ErrorCode.INTERNAL_ERROR,
        )

        with patch("src.api.v1.vision.get_vision_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.analyze = AsyncMock(return_value=mock_response)
            mock_get_manager.return_value = mock_manager

            result = await analyze_vision(request, provider=None)

            assert result.success is False
            assert "failed" in (result.error or "").lower() or result.error is not None

    @pytest.mark.asyncio
    async def test_vision_input_error(self):
        """Test analyze_vision handles VisionInputError."""
        from src.api.v1.vision import analyze_vision
        from src.core.errors import ErrorCode
        from src.core.vision import VisionAnalyzeRequest, VisionInputError

        request = VisionAnalyzeRequest(
            image_base64="invalid",
            include_description=True,
        )

        with patch("src.api.v1.vision.get_vision_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.analyze = AsyncMock(side_effect=VisionInputError("Invalid image"))
            mock_get_manager.return_value = mock_manager

            result = await analyze_vision(request, provider=None)

            assert result.success is False
            assert result.code == ErrorCode.INPUT_ERROR
            assert "Invalid image" in (result.error or "")

    @pytest.mark.asyncio
    async def test_vision_provider_error(self):
        """Test analyze_vision handles VisionProviderError."""
        from src.api.v1.vision import analyze_vision
        from src.core.errors import ErrorCode
        from src.core.vision import VisionAnalyzeRequest

        # Create a custom exception class that matches VisionProviderError interface
        class MockVisionProviderError(Exception):
            def __init__(self, message, provider):
                self.message = message
                self.provider = provider
                super().__init__(message)

        request = VisionAnalyzeRequest(
            image_base64="dGVzdGltYWdl",
            include_description=True,
        )

        # Patch VisionProviderError to be our mock class
        with patch("src.api.v1.vision.VisionProviderError", MockVisionProviderError):
            with patch("src.api.v1.vision.get_vision_manager") as mock_get_manager:
                mock_manager = MagicMock()
                error = MockVisionProviderError(message="Provider timeout", provider="openai")
                mock_manager.analyze = AsyncMock(side_effect=error)
                mock_get_manager.return_value = mock_manager

                result = await analyze_vision(request, provider=None)

                assert result.success is False
                assert result.code == ErrorCode.EXTERNAL_SERVICE_ERROR
                assert result.provider == "openai"

    @pytest.mark.asyncio
    async def test_generic_exception(self):
        """Test analyze_vision handles generic exceptions."""
        from src.api.v1.vision import analyze_vision
        from src.core.errors import ErrorCode
        from src.core.vision import VisionAnalyzeRequest

        request = VisionAnalyzeRequest(
            image_base64="dGVzdGltYWdl",
            include_description=True,
        )

        with patch("src.api.v1.vision.get_vision_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.analyze = AsyncMock(side_effect=Exception("Unexpected error"))
            mock_get_manager.return_value = mock_manager

            result = await analyze_vision(request, provider=None)

            assert result.success is False
            assert result.code == ErrorCode.INTERNAL_ERROR
            assert "Unexpected error" in (result.error or "")


class TestHealthCheckEndpoint:
    """Tests for health_check endpoint."""

    @pytest.mark.asyncio
    async def test_healthy_status(self):
        """Test health_check returns healthy status."""
        from src.api.v1.vision import health_check

        with patch("src.api.v1.vision.get_vision_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.vision_provider.provider_name = "stub"
            mock_manager.ocr_manager = MagicMock()
            mock_get_manager.return_value = mock_manager

            result = await health_check()

            assert result["status"] == "healthy"
            assert result["provider"] == "stub"
            assert result["ocr_enabled"] is True

    @pytest.mark.asyncio
    async def test_degraded_status_on_error(self):
        """Test health_check returns degraded status on error."""
        from src.api.v1.vision import health_check

        with patch("src.api.v1.vision.get_vision_manager") as mock_get_manager:
            mock_get_manager.side_effect = Exception("Manager initialization failed")

            result = await health_check()

            assert result["status"] == "degraded"
            assert "error" in result


class TestListProvidersEndpoint:
    """Tests for list_providers endpoint."""

    @pytest.mark.asyncio
    async def test_list_providers_success(self):
        """Test list_providers returns provider information."""
        from src.api.v1.vision import list_providers

        with patch("src.api.v1.vision.get_vision_manager") as mock_get_manager:
            with patch("src.api.v1.vision.get_available_providers") as mock_get_providers:
                mock_manager = MagicMock()
                mock_manager.vision_provider.provider_name = "openai"
                mock_get_manager.return_value = mock_manager
                mock_get_providers.return_value = {
                    "stub": {"available": True},
                    "openai": {"available": True},
                }

                result = await list_providers()

                assert result["current_provider"] == "openai"
                assert "providers" in result
                assert "stub" in result["providers"]

    @pytest.mark.asyncio
    async def test_list_providers_manager_error(self):
        """Test list_providers handles manager error gracefully."""
        from src.api.v1.vision import list_providers

        with patch("src.api.v1.vision.get_vision_manager") as mock_get_manager:
            with patch("src.api.v1.vision.get_available_providers") as mock_get_providers:
                mock_get_manager.side_effect = Exception("Manager error")
                mock_get_providers.return_value = {}

                result = await list_providers()

                assert result["current_provider"] == "unknown"


class TestGetMetricsEndpoint:
    """Tests for get_metrics endpoint."""

    @pytest.mark.asyncio
    async def test_metrics_with_resilient_provider(self):
        """Test get_metrics returns detailed metrics for resilient provider."""
        from src.api.v1.vision import get_metrics
        from src.core.vision import ResilientVisionProvider

        with patch("src.api.v1.vision.get_vision_manager") as mock_get_manager:
            mock_provider = MagicMock(spec=ResilientVisionProvider)
            mock_provider.provider_name = "openai"
            mock_provider.circuit_state.value = "closed"
            mock_provider.metrics.total_requests = 100
            mock_provider.metrics.successful_requests = 95
            mock_provider.metrics.failed_requests = 5
            mock_provider.metrics.success_rate = 0.95
            mock_provider.metrics.average_latency_ms = 500.0
            mock_provider.metrics.last_error = None
            mock_provider.metrics.last_error_time = None
            mock_provider.metrics.circuit_opens = 1

            mock_manager = MagicMock()
            mock_manager.vision_provider = mock_provider
            mock_get_manager.return_value = mock_manager

            result = await get_metrics()

            assert result["provider"] == "openai"
            assert result["resilient"] is True
            assert result["metrics"]["total_requests"] == 100
            assert result["metrics"]["success_rate"] == 0.95

    @pytest.mark.asyncio
    async def test_metrics_without_resilient_provider(self):
        """Test get_metrics returns limited metrics for non-resilient provider."""
        from src.api.v1.vision import get_metrics

        with patch("src.api.v1.vision.get_vision_manager") as mock_get_manager:
            mock_provider = MagicMock()
            mock_provider.provider_name = "stub"
            # Not a ResilientVisionProvider instance

            mock_manager = MagicMock()
            mock_manager.vision_provider = mock_provider
            mock_get_manager.return_value = mock_manager

            result = await get_metrics()

            assert result["provider"] == "stub"
            assert result["resilient"] is False
            assert "message" in result["metrics"]

    @pytest.mark.asyncio
    async def test_metrics_error(self):
        """Test get_metrics handles errors gracefully."""
        from src.api.v1.vision import get_metrics

        with patch("src.api.v1.vision.get_vision_manager") as mock_get_manager:
            mock_get_manager.side_effect = Exception("Metrics error")

            result = await get_metrics()

            assert "error" in result
            assert result["metrics"] is None


class TestVisionManagerWithOcr:
    """Tests for vision manager OCR integration."""

    def test_manager_created_with_ocr_manager(self):
        """Test vision manager is created with OCR manager."""
        from src.api.v1 import vision

        vision._vision_manager = None
        vision._current_provider_type = None

        with patch("src.api.v1.vision.create_vision_provider") as mock_create:
            with patch("src.core.ocr.manager.OcrManager") as mock_ocr_class:
                mock_provider = MagicMock()
                mock_create.return_value = mock_provider
                mock_ocr = MagicMock()
                mock_ocr_class.return_value = mock_ocr

                manager = vision.get_vision_manager()

                mock_ocr_class.assert_called_once()
                assert manager.ocr_manager is mock_ocr
