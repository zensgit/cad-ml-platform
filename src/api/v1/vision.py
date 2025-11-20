"""Vision API endpoints.

Provides:
- POST /api/v1/vision/analyze - End-to-end vision + OCR analysis
"""

from typing import Optional

from fastapi import APIRouter

from src.core.vision import (
    VisionAnalyzeRequest,
    VisionAnalyzeResponse,
    VisionInputError,
    VisionManager,
    create_stub_provider,
)
from src.core.errors import ErrorCode

router = APIRouter(tags=["vision"])


# ========== Global Manager (Singleton Pattern) ==========
# TODO: Move to dependency injection for production

_vision_manager: Optional[VisionManager] = None


def get_vision_manager() -> VisionManager:
    """
    Get or create VisionManager singleton.

    Returns:
        VisionManager instance with stub provider and OCR integration

    Note:
        In production, this should be dependency injection with proper lifecycle management.
        For MVP, we use a simple singleton.
    """
    global _vision_manager

    if _vision_manager is None:
        from src.core.ocr.manager import OcrManager

        # Create stub provider for vision description
        vision_provider = create_stub_provider(simulate_latency_ms=50.0)

        # Create OCRManager (simplified for Phase 2 - no providers yet, will fail gracefully)
        # In Phase 3, we'll inject real providers (paddle, deepseek_hf)
        ocr_manager = OcrManager(
            providers={},  # Empty for now - OCR will gracefully return None if no providers
            confidence_fallback=0.85,
        )

        # Create manager with both Vision and OCR
        _vision_manager = VisionManager(vision_provider=vision_provider, ocr_manager=ocr_manager)

    return _vision_manager


# ========== Endpoints ==========


@router.post("/analyze", response_model=VisionAnalyzeResponse)
async def analyze_vision(request: VisionAnalyzeRequest) -> VisionAnalyzeResponse:
    """
    Analyze engineering drawing with vision + OCR.

    **Workflow:**
    1. Vision description (natural language summary)
    2. OCR extraction (dimensions, symbols, title block)
    3. Aggregated response

    **MVP Status:**
    - ✅ Vision description (stub provider with fixed response)
    - ⚠️ OCR integration (not yet connected, returns None)

    **Example Request:**
    ```json
    {
        "image_base64": "iVBORw0KGgoAAAANS...",
        "include_description": true,
        "include_ocr": true,
        "ocr_provider": "auto"
    }
    ```

    **Example Response:**
    ```json
    {
        "success": true,
        "description": {
            "summary": "Mechanical part with cylindrical features",
            "details": ["Main diameter: 20mm...", "Thread: M10×1.5"],
            "confidence": 0.92
        },
        "ocr": null,
        "provider": "deepseek_stub",
        "processing_time_ms": 52.3
    }
    ```
    """
    try:
        manager = get_vision_manager()
        response = await manager.analyze(request)
        if response.success:
            return response
        return VisionAnalyzeResponse(
            success=False,
            description=None,
            ocr=None,
            provider=response.provider,
            processing_time_ms=response.processing_time_ms,
            error=response.error or "Vision analysis failed",
            code=response.code or ErrorCode.INTERNAL_ERROR,
        )
    except VisionInputError as e:
        return VisionAnalyzeResponse(
            success=False,
            description=None,
            ocr=None,
            provider="deepseek_stub",
            processing_time_ms=0.0,
            error=str(e),
            code=ErrorCode.INPUT_ERROR,
        )
    except Exception as e:
        return VisionAnalyzeResponse(
            success=False,
            description=None,
            ocr=None,
            provider="deepseek_stub",
            processing_time_ms=0.0,
            error=f"Internal server error: {str(e)}",
            code=ErrorCode.INTERNAL_ERROR,
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint for vision service.

    Returns:
        Service status and provider information
    """
    manager = get_vision_manager()

    return {
        "status": "healthy",
        "provider": manager.vision_provider.provider_name,
        "ocr_enabled": manager.ocr_manager is not None,
    }
