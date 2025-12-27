"""Vision + OCR integration tests.

Tests the integration between VisionManager and OCRManager:
1. include_ocr=True returns Vision + OCR results
2. OCR exceptions don't break Vision description (graceful degradation)
3. include_ocr=False doesn't trigger OCR call
"""

import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.ocr.base import DimensionInfo, DimensionType
from src.core.ocr.base import OcrResult as OcrCoreResult
from src.core.ocr.base import SymbolInfo, SymbolType, TitleBlock
from src.core.ocr.exceptions import OcrError
from src.core.vision import VisionAnalyzeRequest, VisionManager, create_stub_provider

# ========== Test Fixtures ==========


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Return sample image bytes (1x1 PNG)."""
    # Minimal 1x1 PNG (black pixel)
    png_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00:~\x9bU\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    return png_bytes


@pytest.fixture
def sample_image_base64(sample_image_bytes) -> str:
    """Return sample image as base64 string."""
    return base64.b64encode(sample_image_bytes).decode('utf-8')


@pytest.fixture
def mock_ocr_result() -> OcrCoreResult:
    """Return mock OCR result with sample dimensions and symbols."""
    return OcrCoreResult(
        text="Sample OCR text",
        dimensions=[
            DimensionInfo(
                type=DimensionType.diameter,
                value=20.0,
                unit="mm",
                tolerance=0.02,
                confidence=0.95
            ),
            DimensionInfo(
                type=DimensionType.thread,
                value=10.0,
                unit="mm",
                pitch=1.5,
                confidence=0.92
            )
        ],
        symbols=[
            SymbolInfo(
                type=SymbolType.surface_roughness,
                value="Ra 3.2",
                normalized_form="surface_roughness_3.2",
                confidence=0.88
            )
        ],
        title_block=TitleBlock(
            drawing_number="CAD-2025-001",
            material="Aluminum 6061",
            part_name="Test Part"
        ),
        confidence=0.93,
        calibrated_confidence=0.91
    )


@pytest.fixture
def mock_ocr_manager_success(mock_ocr_result):
    """Return mock OCRManager that returns successful OCR result."""
    mock_manager = MagicMock()
    mock_manager.extract = AsyncMock(return_value=mock_ocr_result)
    return mock_manager


@pytest.fixture
def mock_ocr_manager_failure():
    """Return mock OCRManager that raises OcrError."""
    mock_manager = MagicMock()
    mock_manager.extract = AsyncMock(side_effect=OcrError(
        code="PROVIDER_DOWN",
        message="OCR provider unavailable",
        provider="deepseek_hf",
        stage="infer"
    ))
    return mock_manager


# ========== Integration Tests ==========


@pytest.mark.asyncio
async def test_vision_ocr_integration_success(sample_image_base64, mock_ocr_manager_success):
    """
    Test: include_ocr=True returns Vision + OCR results.

    Expected behavior:
    - Vision description from stub provider
    - OCR results from mock OCRManager
    - Both present in response
    - Success=True
    """
    # Create VisionManager with stub provider and mock OCRManager
    vision_provider = create_stub_provider(simulate_latency_ms=10)
    manager = VisionManager(
        vision_provider=vision_provider,
        ocr_manager=mock_ocr_manager_success
    )

    # Create request with include_ocr=True
    request = VisionAnalyzeRequest(
        image_base64=sample_image_base64,
        include_description=True,
        include_ocr=True,
        ocr_provider="auto"
    )

    # Execute analysis
    response = await manager.analyze(request)

    # Assertions
    assert response.success is True, "Analysis should succeed"

    # Vision description should be present
    assert response.description is not None, "Vision description should be present"
    assert "cylindrical part" in response.description.summary.lower()
    assert response.description.confidence > 0.0

    # OCR results should be present
    assert response.ocr is not None, "OCR results should be present"
    assert len(response.ocr.dimensions) == 2, "Should have 2 dimensions"
    assert len(response.ocr.symbols) == 1, "Should have 1 symbol"

    # Check dimension conversion (DimensionInfo -> Dict)
    dim1 = response.ocr.dimensions[0]
    assert dim1["type"] == "diameter"
    assert dim1["value"] == 20.0
    assert dim1["tolerance"] == 0.02

    dim2 = response.ocr.dimensions[1]
    assert dim2["type"] == "thread"
    assert dim2["pitch"] == 1.5

    # Check symbol conversion
    sym1 = response.ocr.symbols[0]
    assert sym1["type"] == "surface_roughness"
    assert sym1["value"] == "Ra 3.2"

    # Check title block conversion
    assert response.ocr.title_block["drawing_number"] == "CAD-2025-001"
    assert response.ocr.title_block["material"] == "Aluminum 6061"

    # Check confidence (should use calibrated_confidence if available)
    assert response.ocr.confidence == 0.91  # calibrated_confidence

    # Verify OCRManager.extract was called
    mock_ocr_manager_success.extract.assert_called_once()


@pytest.mark.asyncio
async def test_vision_ocr_integration_degradation(sample_image_base64, mock_ocr_manager_failure):
    """
    Test: OCR exceptions don't break Vision description (graceful degradation).

    Expected behavior:
    - Vision description still returns
    - OCR is None (due to exception)
    - Success=True (vision succeeded, OCR failed gracefully)
    """
    # Create VisionManager with stub provider and failing OCRManager
    vision_provider = create_stub_provider(simulate_latency_ms=10)
    manager = VisionManager(
        vision_provider=vision_provider,
        ocr_manager=mock_ocr_manager_failure
    )

    # Create request with include_ocr=True
    request = VisionAnalyzeRequest(
        image_base64=sample_image_base64,
        include_description=True,
        include_ocr=True,
        ocr_provider="auto"
    )

    # Execute analysis
    response = await manager.analyze(request)

    # Assertions
    assert response.success is True, "Analysis should succeed (vision works, OCR failed gracefully)"

    # Vision description should still be present
    assert response.description is not None, "Vision description should be present despite OCR failure"
    assert "cylindrical part" in response.description.summary.lower()
    assert response.description.confidence > 0.0

    # OCR results should be None (graceful degradation)
    assert response.ocr is None, "OCR should be None due to exception"

    # Verify OCRManager.extract was called (but failed)
    mock_ocr_manager_failure.extract.assert_called_once()


@pytest.mark.asyncio
async def test_vision_ocr_integration_skip_ocr(sample_image_base64, mock_ocr_manager_success):
    """
    Test: include_ocr=False doesn't trigger OCR call.

    Expected behavior:
    - Vision description returns normally
    - OCR is None
    - OCRManager.extract NOT called
    """
    # Create VisionManager with stub provider and mock OCRManager
    vision_provider = create_stub_provider(simulate_latency_ms=10)
    manager = VisionManager(
        vision_provider=vision_provider,
        ocr_manager=mock_ocr_manager_success
    )

    # Create request with include_ocr=False
    request = VisionAnalyzeRequest(
        image_base64=sample_image_base64,
        include_description=True,
        include_ocr=False,  # Explicitly skip OCR
        ocr_provider="auto"
    )

    # Execute analysis
    response = await manager.analyze(request)

    # Assertions
    assert response.success is True, "Analysis should succeed"

    # Vision description should be present
    assert response.description is not None, "Vision description should be present"
    assert "cylindrical part" in response.description.summary.lower()

    # OCR should be None (not called)
    assert response.ocr is None, "OCR should be None when include_ocr=False"

    # Verify OCRManager.extract was NOT called
    mock_ocr_manager_success.extract.assert_not_called()


@pytest.mark.asyncio
async def test_vision_ocr_integration_no_manager(sample_image_base64):
    """
    Test: VisionManager with ocr_manager=None gracefully handles include_ocr=True.

    Expected behavior:
    - Vision description returns normally
    - OCR is None (no manager available)
    - No exceptions raised
    """
    # Create VisionManager without OCRManager
    vision_provider = create_stub_provider(simulate_latency_ms=10)
    manager = VisionManager(
        vision_provider=vision_provider,
        ocr_manager=None  # No OCR manager
    )

    # Create request with include_ocr=True (but no manager available)
    request = VisionAnalyzeRequest(
        image_base64=sample_image_base64,
        include_description=True,
        include_ocr=True,  # Request OCR but no manager available
        ocr_provider="auto"
    )

    # Execute analysis
    response = await manager.analyze(request)

    # Assertions
    assert response.success is True, "Analysis should succeed"

    # Vision description should be present
    assert response.description is not None, "Vision description should be present"

    # OCR should be None (no manager)
    assert response.ocr is None, "OCR should be None when ocr_manager=None"
