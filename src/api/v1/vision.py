"""Vision API endpoints.

Provides:
- POST /api/v1/vision/analyze - End-to-end vision + OCR analysis
- GET /api/v1/vision/health - Service health check
- GET /api/v1/vision/providers - List available providers
- GET /api/v1/vision/metrics - Provider performance metrics
"""

import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query

from src.core.errors import ErrorCode
from src.core.vision import (
    ResilientVisionProvider,
    VisionAnalyzeRequest,
    VisionAnalyzeResponse,
    VisionInputError,
    VisionManager,
    VisionProviderError,
    create_vision_provider,
    get_available_providers,
)
from src.core.providers import ProviderRegistry, bootstrap_core_provider_registry

router = APIRouter(tags=["vision"])


# ========== Global Manager (Singleton Pattern) ==========

_vision_manager: Optional[VisionManager] = None
_current_provider_type: Optional[str] = None


def get_vision_manager(provider_type: Optional[str] = None) -> VisionManager:
    """
    Get or create VisionManager singleton.

    Args:
        provider_type: Optional provider override (deepseek, openai, anthropic, stub)

    Returns:
        VisionManager instance with configured provider and OCR integration

    Note:
        The provider is determined by:
        1. Explicit provider_type parameter
        2. VISION_PROVIDER environment variable
        3. Auto-detection based on available API keys
        4. Fallback to stub provider
    """
    global _vision_manager, _current_provider_type

    # Determine effective provider type
    effective_provider = provider_type or os.getenv("VISION_PROVIDER", "auto")

    # Recreate manager if provider type changed
    if _vision_manager is None or _current_provider_type != effective_provider:
        from src.core.ocr.manager import OcrManager

        # Create vision provider using factory
        vision_provider = create_vision_provider(
            provider_type=effective_provider,
            fallback_to_stub=True,
        )

        # Create OCRManager via core provider registry (best-effort).
        # Keep this behavior additive: if providers cannot be registered, OCR will degrade
        # gracefully (VisionManager returns description-only results).
        bootstrap_core_provider_registry()
        ocr_manager = OcrManager(confidence_fallback=0.85)
        try:
            ocr_manager.register_provider(
                "paddle", ProviderRegistry.get("ocr", "paddle")
            )
        except Exception:
            pass
        try:
            ocr_manager.register_provider(
                "deepseek_hf", ProviderRegistry.get("ocr", "deepseek_hf")
            )
        except Exception:
            pass

        # Create manager with both Vision and OCR
        _vision_manager = VisionManager(
            vision_provider=vision_provider,
            ocr_manager=ocr_manager,
        )
        _current_provider_type = effective_provider

    return _vision_manager


def reset_vision_manager() -> None:
    """Reset the vision manager singleton (useful for testing)."""
    global _vision_manager, _current_provider_type
    _vision_manager = None
    _current_provider_type = None


# ========== Endpoints ==========


@router.post("/analyze", response_model=VisionAnalyzeResponse)
async def analyze_vision(
    request: VisionAnalyzeRequest,
    provider: Optional[str] = Query(
        None,
        description="Vision provider override: deepseek, openai, anthropic, stub",
    ),
) -> VisionAnalyzeResponse:
    """
    Analyze engineering drawing with vision + OCR.

    **Supported Providers:**
    - `deepseek` - DeepSeek VL API (requires DEEPSEEK_API_KEY)
    - `openai` - OpenAI GPT-4o (requires OPENAI_API_KEY)
    - `anthropic` - Claude 3 (requires ANTHROPIC_API_KEY)
    - `stub` - Testing stub (no API key required)
    - `auto` - Auto-detect based on available API keys

    **Workflow:**
    1. Vision description (natural language summary)
    2. OCR extraction (dimensions, symbols, title block)
    3. Aggregated response

    **Example Request:**
    ```json
    {
        "image_base64": "iVBORw0KGgoAAAANS...",
        "include_description": true,
        "include_ocr": true,
        "ocr_provider": "auto",
        "include_cad_stats": true,
        "cad_feature_thresholds": {"line_aspect": 5.0, "arc_fill_min": 0.08}
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
        "cad_feature_stats": {
            "line_count": 1,
            "circle_count": 0,
            "arc_count": 0,
            "line_angle_bins": {
                "0-30": 1,
                "30-60": 0,
                "60-90": 0,
                "90-120": 0,
                "120-150": 0,
                "150-180": 0
            },
            "line_angle_avg": 5.0,
            "arc_sweep_avg": null,
            "arc_sweep_bins": {"0-90": 0, "90-180": 0, "180-270": 0, "270-360": 0}
        },
        "provider": "openai",
        "processing_time_ms": 1234.5
    }
    ```
    """
    try:
        manager = get_vision_manager(provider_type=provider)
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
            provider="unknown",
            processing_time_ms=0.0,
            error=str(e),
            code=ErrorCode.INPUT_ERROR,
        )

    except VisionProviderError as e:
        return VisionAnalyzeResponse(
            success=False,
            description=None,
            ocr=None,
            provider=e.provider,
            processing_time_ms=0.0,
            error=e.message,
            code=ErrorCode.EXTERNAL_SERVICE_ERROR,
        )

    except Exception as e:
        return VisionAnalyzeResponse(
            success=False,
            description=None,
            ocr=None,
            provider="unknown",
            processing_time_ms=0.0,
            error=f"Internal server error: {str(e)}",
            code=ErrorCode.INTERNAL_ERROR,
        )


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for vision service.

    Returns:
        Service status and provider information
    """
    try:
        manager = get_vision_manager()
        return {
            "status": "healthy",
            "provider": manager.vision_provider.provider_name,
            "ocr_enabled": manager.ocr_manager is not None,
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
        }


@router.get("/providers")
async def list_providers() -> Dict[str, Any]:
    """
    List available vision providers and their status.

    Returns:
        Dictionary of providers with availability information

    **Example Response:**
    ```json
    {
        "current_provider": "openai",
        "providers": {
            "stub": {"available": true, "requires_key": false},
            "deepseek": {"available": true, "requires_key": true, "key_set": true},
            "openai": {"available": true, "requires_key": true, "key_set": true},
            "anthropic": {"available": true, "requires_key": true, "key_set": false}
        }
    }
    ```
    """
    try:
        manager = get_vision_manager()
        current = manager.vision_provider.provider_name
    except Exception:
        current = "unknown"

    return {
        "current_provider": current,
        "providers": get_available_providers(),
    }


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Get performance metrics for vision providers.

    Returns metrics including:
    - Total requests
    - Success/failure counts
    - Average latency
    - Circuit breaker state (if resilient provider is used)

    **Example Response:**
    ```json
    {
        "provider": "openai",
        "metrics": {
            "total_requests": 100,
            "successful_requests": 95,
            "failed_requests": 5,
            "success_rate": 0.95,
            "average_latency_ms": 1234.5,
            "last_error": "Timeout after 60s",
            "circuit_state": "closed",
            "circuit_opens": 1
        }
    }
    ```
    """
    try:
        manager = get_vision_manager()
        provider = manager.vision_provider
        provider_name = provider.provider_name

        # Check if provider is resilient (has metrics)
        if isinstance(provider, ResilientVisionProvider):
            metrics = provider.metrics
            return {
                "provider": provider_name,
                "resilient": True,
                "metrics": {
                    "total_requests": metrics.total_requests,
                    "successful_requests": metrics.successful_requests,
                    "failed_requests": metrics.failed_requests,
                    "success_rate": round(metrics.success_rate, 4),
                    "average_latency_ms": round(metrics.average_latency_ms, 2),
                    "last_error": metrics.last_error,
                    "last_error_time": metrics.last_error_time,
                    "circuit_state": provider.circuit_state.value,
                    "circuit_opens": metrics.circuit_opens,
                },
            }
        else:
            return {
                "provider": provider_name,
                "resilient": False,
                "metrics": {
                    "message": "Metrics available with ResilientVisionProvider",
                    "hint": "Use create_resilient_provider() to enable metrics",
                },
            }
    except Exception as e:
        return {
            "error": str(e),
            "metrics": None,
        }
