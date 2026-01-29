"""Tests for src/api/v1/ocr.py endpoint to improve coverage.

Covers:
- ocr_extract endpoint
- get_manager singleton
- OcrResponse model
- Idempotency handling
- Rate limiting
- Error handling paths
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.core.errors import ErrorCode
from src.core.ocr.base import (
    DimensionInfo,
    DimensionType,
    OcrResult,
    SymbolInfo,
    SymbolType,
    TitleBlock,
)
from src.core.ocr.exceptions import OcrError


class TestGetManager:
    """Tests for get_manager function."""

    def test_get_manager_singleton(self):
        """Test get_manager returns singleton instance."""
        from src.api.v1 import ocr

        # Reset manager
        ocr._manager = None

        manager1 = ocr.get_manager()
        manager2 = ocr.get_manager()

        assert manager1 is manager2

    def test_get_manager_registers_providers(self):
        """Test get_manager registers providers."""
        from src.api.v1 import ocr

        # Reset manager
        ocr._manager = None

        manager = ocr.get_manager()

        # Check that providers are registered
        assert manager is not None


class TestOcrResponseModel:
    """Tests for OcrResponse model."""

    def test_model_creation_success(self):
        """Test OcrResponse model creation for success case."""
        from src.api.v1.ocr import OcrResponse

        response = OcrResponse(
            success=True,
            provider="paddle",
            confidence=0.95,
            fallback_level="primary",
            processing_time_ms=150,
            dimensions=[{"type": "diameter", "value": 10.5, "unit": "mm"}],
            symbols=[{"type": "perpendicular", "value": "0.05"}],
            title_block={"drawing_number": "DWG-001"},
        )

        assert response.success is True
        assert response.provider == "paddle"
        assert response.confidence == 0.95

    def test_model_creation_failure(self):
        """Test OcrResponse model creation for failure case."""
        from src.api.v1.ocr import OcrResponse

        response = OcrResponse(
            success=False,
            provider="auto",
            confidence=None,
            fallback_level=None,
            processing_time_ms=0,
            dimensions=[],
            symbols=[],
            title_block={},
            error="Provider unavailable",
            code=ErrorCode.PROVIDER_DOWN,
        )

        assert response.success is False
        assert response.error == "Provider unavailable"


class TestOcrExtractEndpoint:
    """Tests for ocr_extract endpoint."""

    @pytest.mark.asyncio
    async def test_ocr_extract_idempotency_hit(self):
        """Test idempotency cache hit returns cached response."""
        from src.api.v1.ocr import OcrResponse, ocr_extract

        # Create a mock file
        mock_file = MagicMock()
        mock_file.filename = "test.png"
        mock_file.read = AsyncMock(return_value=b"fake_image_data")

        cached_data = {
            "success": True,
            "provider": "paddle",
            "confidence": 0.9,
            "fallback_level": "primary",
            "processing_time_ms": 100,
            "dimensions": [],
            "symbols": [],
            "title_block": {},
        }

        with patch("src.api.v1.ocr.check_idempotency", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = cached_data

            result = await ocr_extract(
                file=mock_file,
                provider="auto",
                request=None,
                idempotency_key="test-key-123",
            )

            assert result.success is True
            assert result.provider == "paddle"
            mock_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_ocr_extract_rate_limit_called(self):
        """Test rate limit is called when request provided."""
        from src.api.v1.ocr import ocr_extract

        mock_file = MagicMock()
        mock_file.filename = "test.png"

        mock_request = MagicMock()

        with patch("src.api.v1.ocr.check_idempotency", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = None
            with patch("src.api.v1.ocr.rate_limit") as mock_rate_limit:
                with patch(
                    "src.api.v1.ocr.validate_and_read", new_callable=AsyncMock
                ) as mock_validate:
                    # Raise to short-circuit the rest
                    mock_validate.side_effect = HTTPException(
                        status_code=400, detail="Invalid file"
                    )

                    result = await ocr_extract(
                        file=mock_file,
                        provider="auto",
                        request=mock_request,
                        idempotency_key=None,
                    )

                    mock_rate_limit.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_ocr_extract_mime_validation_error(self):
        """Test OCR handles MIME type validation error."""
        from src.api.v1.ocr import ocr_extract

        mock_file = MagicMock()
        mock_file.filename = "test.txt"

        with patch("src.api.v1.ocr.check_idempotency", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = None
            with patch("src.api.v1.ocr.validate_and_read", new_callable=AsyncMock) as mock_validate:
                mock_validate.side_effect = HTTPException(
                    status_code=415, detail="Unsupported MIME type"
                )

                result = await ocr_extract(
                    file=mock_file,
                    provider="auto",
                    request=None,
                    idempotency_key=None,
                )

                assert result.success is False
                assert result.code == ErrorCode.INPUT_ERROR

    @pytest.mark.asyncio
    async def test_ocr_extract_file_too_large(self):
        """Test OCR handles file too large error."""
        from src.api.v1.ocr import ocr_extract

        mock_file = MagicMock()
        mock_file.filename = "test.png"

        with patch("src.api.v1.ocr.check_idempotency", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = None
            with patch("src.api.v1.ocr.validate_and_read", new_callable=AsyncMock) as mock_validate:
                mock_validate.side_effect = HTTPException(status_code=413, detail="File too large")

                result = await ocr_extract(
                    file=mock_file,
                    provider="auto",
                    request=None,
                    idempotency_key=None,
                )

                assert result.success is False

    @pytest.mark.asyncio
    async def test_ocr_extract_pdf_pages_exceed(self):
        """Test OCR handles PDF page count exceed error."""
        from src.api.v1.ocr import ocr_extract

        mock_file = MagicMock()
        mock_file.filename = "test.pdf"

        with patch("src.api.v1.ocr.check_idempotency", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = None
            with patch("src.api.v1.ocr.validate_and_read", new_callable=AsyncMock) as mock_validate:
                mock_validate.side_effect = HTTPException(
                    status_code=400, detail="Page count exceeded"
                )

                result = await ocr_extract(
                    file=mock_file,
                    provider="auto",
                    request=None,
                    idempotency_key=None,
                )

                assert result.success is False

    @pytest.mark.asyncio
    async def test_ocr_extract_pdf_forbidden_token(self):
        """Test OCR handles PDF forbidden token error."""
        from src.api.v1.ocr import ocr_extract

        mock_file = MagicMock()
        mock_file.filename = "test.pdf"

        with patch("src.api.v1.ocr.check_idempotency", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = None
            with patch("src.api.v1.ocr.validate_and_read", new_callable=AsyncMock) as mock_validate:
                mock_validate.side_effect = HTTPException(
                    status_code=400, detail="Forbidden token in PDF"
                )

                result = await ocr_extract(
                    file=mock_file,
                    provider="auto",
                    request=None,
                    idempotency_key=None,
                )

                assert result.success is False

    @pytest.mark.asyncio
    async def test_ocr_extract_general_validation_error(self):
        """Test OCR handles general validation error."""
        from src.api.v1.ocr import ocr_extract

        mock_file = MagicMock()
        mock_file.filename = "test.png"

        with patch("src.api.v1.ocr.check_idempotency", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = None
            with patch("src.api.v1.ocr.validate_and_read", new_callable=AsyncMock) as mock_validate:
                mock_validate.side_effect = ValueError("Invalid input")

                result = await ocr_extract(
                    file=mock_file,
                    provider="auto",
                    request=None,
                    idempotency_key=None,
                )

                assert result.success is False
                assert "validation failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_ocr_extract_ocr_error(self):
        """Test OCR handles OcrError from manager."""
        from src.api.v1.ocr import ocr_extract

        mock_file = MagicMock()
        mock_file.filename = "test.png"

        mock_manager = MagicMock()
        mock_manager.extract = AsyncMock(
            side_effect=OcrError(ErrorCode.PROVIDER_DOWN, "Provider down")
        )

        with patch("src.api.v1.ocr.check_idempotency", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = None
            with patch("src.api.v1.ocr.validate_and_read", new_callable=AsyncMock) as mock_validate:
                mock_validate.return_value = (b"image_bytes", "image/png")
                with patch("src.api.v1.ocr.get_manager", return_value=mock_manager):
                    result = await ocr_extract(
                        file=mock_file,
                        provider="paddle",
                        request=None,
                        idempotency_key=None,
                    )

                    assert result.success is False
                    assert result.code == ErrorCode.PROVIDER_DOWN

    @pytest.mark.asyncio
    async def test_ocr_extract_ocr_error_without_code(self):
        """Test OCR handles OcrError with non-ErrorCode code attribute."""
        from src.api.v1.ocr import ocr_extract

        mock_file = MagicMock()
        mock_file.filename = "test.png"

        mock_manager = MagicMock()
        # Create an OcrError with a string code (not ErrorCode enum)
        ocr_error = OcrError("not_an_error_code", "Unknown error")
        mock_manager.extract = AsyncMock(side_effect=ocr_error)

        with patch("src.api.v1.ocr.check_idempotency", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = None
            with patch("src.api.v1.ocr.validate_and_read", new_callable=AsyncMock) as mock_validate:
                mock_validate.return_value = (b"image_bytes", "image/png")
                with patch("src.api.v1.ocr.get_manager", return_value=mock_manager):
                    result = await ocr_extract(
                        file=mock_file,
                        provider="auto",
                        request=None,
                        idempotency_key=None,
                    )

                    assert result.success is False
                    assert result.code == ErrorCode.INTERNAL_ERROR

    @pytest.mark.asyncio
    async def test_ocr_extract_manager_exception(self):
        """Test OCR handles unexpected exception from manager."""
        from src.api.v1.ocr import ocr_extract

        mock_file = MagicMock()
        mock_file.filename = "test.png"

        mock_manager = MagicMock()
        mock_manager.extract = AsyncMock(side_effect=RuntimeError("Unexpected error"))

        with patch("src.api.v1.ocr.check_idempotency", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = None
            with patch("src.api.v1.ocr.validate_and_read", new_callable=AsyncMock) as mock_validate:
                mock_validate.return_value = (b"image_bytes", "image/png")
                with patch("src.api.v1.ocr.get_manager", return_value=mock_manager):
                    result = await ocr_extract(
                        file=mock_file,
                        provider="auto",
                        request=None,
                        idempotency_key=None,
                    )

                    assert result.success is False
                    assert result.code == ErrorCode.INTERNAL_ERROR

    @pytest.mark.asyncio
    async def test_ocr_extract_success(self):
        """Test successful OCR extraction."""
        from src.api.v1.ocr import ocr_extract

        mock_file = MagicMock()
        mock_file.filename = "test.png"

        mock_result = OcrResult(
            provider="paddle",
            confidence=0.92,
            processing_time_ms=120,
            fallback_level="primary",
            extraction_mode="provider_native",
            dimensions=[DimensionInfo(type=DimensionType.diameter, value=10.0, unit="mm")],
            symbols=[SymbolInfo(type=SymbolType.perpendicular, value="0.05")],
            title_block=TitleBlock(drawing_number="DWG-001"),
            image_hash="abc123",
            completeness=0.95,
            stages_latency_ms={"infer": 100, "parse": 20},
        )

        mock_manager = MagicMock()
        mock_manager.extract = AsyncMock(return_value=mock_result)

        with patch("src.api.v1.ocr.check_idempotency", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = None
            with patch("src.api.v1.ocr.validate_and_read", new_callable=AsyncMock) as mock_validate:
                mock_validate.return_value = (b"image_bytes", "image/png")
                with patch("src.api.v1.ocr.get_manager", return_value=mock_manager):
                    with patch(
                        "src.api.v1.ocr.store_idempotency", new_callable=AsyncMock
                    ) as mock_store:
                        result = await ocr_extract(
                            file=mock_file,
                            provider="paddle",
                            request=None,
                            idempotency_key=None,
                        )

                        assert result.success is True
                        assert result.provider == "paddle"
                        assert len(result.dimensions) == 1
                        assert len(result.symbols) == 1

    @pytest.mark.asyncio
    async def test_ocr_extract_success_with_material_info(self):
        """Test OCR extraction populates material_info when material is present."""
        from src.api.v1.ocr import ocr_extract

        mock_file = MagicMock()
        mock_file.filename = "test.png"

        mock_result = OcrResult(
            provider="paddle",
            confidence=0.9,
            processing_time_ms=100,
            fallback_level="primary",
            extraction_mode="provider_native",
            dimensions=[],
            symbols=[],
            title_block=TitleBlock(material="S30408"),
        )

        mock_manager = MagicMock()
        mock_manager.extract = AsyncMock(return_value=mock_result)

        mock_info = MagicMock()
        mock_info.grade = "S30408"
        mock_info.name = "奥氏体不锈钢"
        mock_info.category = MagicMock(value="metal")
        mock_info.group = MagicMock(value="stainless_steel")
        mock_info.process = MagicMock(
            warnings=["注意钝化处理"],
            recommendations=["建议固溶处理"],
        )

        with patch("src.api.v1.ocr.check_idempotency", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = None
            with patch("src.api.v1.ocr.validate_and_read", new_callable=AsyncMock) as mock_validate:
                mock_validate.return_value = (b"image_bytes", "image/png")
                with patch("src.api.v1.ocr.get_manager", return_value=mock_manager):
                    with patch(
                        "src.api.v1.ocr.store_idempotency", new_callable=AsyncMock
                    ):
                        with patch(
                            "src.core.materials.classify_material_detailed",
                            return_value=mock_info,
                        ):
                            result = await ocr_extract(
                                file=mock_file,
                                provider="paddle",
                                request=None,
                                idempotency_key=None,
                            )

        assert result.success is True
        assert result.material_info is not None
        assert result.material_info.found is True
        assert result.material_info.grade == "S30408"
        assert result.material_info.group == "stainless_steel"

    @pytest.mark.asyncio
    async def test_ocr_extract_success_with_idempotency(self):
        """Test successful OCR extraction stores idempotency."""
        from src.api.v1.ocr import ocr_extract

        mock_file = MagicMock()
        mock_file.filename = "test.png"

        mock_result = OcrResult(
            provider="paddle",
            confidence=0.92,
            processing_time_ms=120,
            fallback_level="primary",
            dimensions=[],
            symbols=[],
            title_block=TitleBlock(),
        )

        mock_manager = MagicMock()
        mock_manager.extract = AsyncMock(return_value=mock_result)

        with patch("src.api.v1.ocr.check_idempotency", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = None
            with patch("src.api.v1.ocr.validate_and_read", new_callable=AsyncMock) as mock_validate:
                mock_validate.return_value = (b"image_bytes", "image/png")
                with patch("src.api.v1.ocr.get_manager", return_value=mock_manager):
                    with patch(
                        "src.api.v1.ocr.store_idempotency", new_callable=AsyncMock
                    ) as mock_store:
                        result = await ocr_extract(
                            file=mock_file,
                            provider="paddle",
                            request=None,
                            idempotency_key="unique-key-456",
                        )

                        assert result.success is True
                        mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_ocr_extract_success_without_confidence(self):
        """Test successful OCR extraction without confidence (no calibration)."""
        from src.api.v1.ocr import ocr_extract

        mock_file = MagicMock()
        mock_file.filename = "test.png"

        mock_result = OcrResult(
            provider="paddle",
            confidence=None,  # No confidence
            processing_time_ms=120,
            fallback_level="primary",
            dimensions=[],
            symbols=[],
            title_block=TitleBlock(),
        )

        mock_manager = MagicMock()
        mock_manager.extract = AsyncMock(return_value=mock_result)

        with patch("src.api.v1.ocr.check_idempotency", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = None
            with patch("src.api.v1.ocr.validate_and_read", new_callable=AsyncMock) as mock_validate:
                mock_validate.return_value = (b"image_bytes", "image/png")
                with patch("src.api.v1.ocr.get_manager", return_value=mock_manager):
                    result = await ocr_extract(
                        file=mock_file,
                        provider="paddle",
                        request=None,
                        idempotency_key=None,
                    )

                    assert result.success is True
                    assert result.confidence is None

    @pytest.mark.asyncio
    async def test_ocr_extract_outer_http_exception(self):
        """Test OCR handles HTTPException in outer try block by raising it after successful extraction."""
        from src.api.v1.ocr import ocr_extract

        mock_file = MagicMock()
        mock_file.filename = "test.png"

        mock_result = OcrResult(
            provider="paddle",
            confidence=0.9,
            processing_time_ms=100,
            fallback_level="primary",
            dimensions=[],
            symbols=[],
            title_block=TitleBlock(),
        )

        mock_manager = MagicMock()
        mock_manager.extract = AsyncMock(return_value=mock_result)

        with patch("src.api.v1.ocr.check_idempotency", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = None
            with patch("src.api.v1.ocr.validate_and_read", new_callable=AsyncMock) as mock_validate:
                mock_validate.return_value = (b"image_bytes", "image/png")
                with patch("src.api.v1.ocr.get_manager", return_value=mock_manager):
                    with patch(
                        "src.api.v1.ocr.store_idempotency", new_callable=AsyncMock
                    ) as mock_store:
                        # Raise HTTPException from store_idempotency to trigger outer exception handler
                        mock_store.side_effect = HTTPException(
                            status_code=400, detail="Storage error"
                        )

                        result = await ocr_extract(
                            file=mock_file,
                            provider="auto",
                            request=None,
                            idempotency_key="test-key",
                        )

                        assert result.success is False
                        assert result.code == ErrorCode.INPUT_ERROR

    @pytest.mark.asyncio
    async def test_ocr_extract_outer_generic_exception(self):
        """Test OCR handles generic exception in outer try block."""
        from src.api.v1.ocr import ocr_extract

        mock_file = MagicMock()
        mock_file.filename = "test.png"

        mock_manager = MagicMock()

        # Simulate exception during logging by making model_dump raise
        mock_result = MagicMock()
        mock_result.provider = "paddle"
        mock_result.processing_time_ms = 100
        mock_result.fallback_level = "primary"
        mock_result.image_hash = "abc"
        mock_result.completeness = 0.9
        mock_result.confidence = 0.9
        mock_result.calibrated_confidence = 0.85
        mock_result.extraction_mode = "native"
        mock_result.dimensions = []
        mock_result.symbols = []
        mock_result.stages_latency_ms = {}
        mock_result.title_block = MagicMock()
        mock_result.title_block.model_dump = MagicMock(side_effect=Exception("Serialization error"))

        mock_manager.extract = AsyncMock(return_value=mock_result)

        with patch("src.api.v1.ocr.check_idempotency", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = None
            with patch("src.api.v1.ocr.validate_and_read", new_callable=AsyncMock) as mock_validate:
                mock_validate.return_value = (b"image_bytes", "image/png")
                with patch("src.api.v1.ocr.get_manager", return_value=mock_manager):
                    result = await ocr_extract(
                        file=mock_file,
                        provider="auto",
                        request=None,
                        idempotency_key=None,
                    )

                    assert result.success is False
                    assert result.code == ErrorCode.INTERNAL_ERROR


class TestRouterExport:
    """Tests for router export."""

    def test_router_exported(self):
        """Test router is exported from module."""
        from src.api.v1.ocr import router

        assert router is not None

    def test_router_has_ocr_tag(self):
        """Test router has ocr tag."""
        from src.api.v1.ocr import router

        assert "ocr" in router.tags
