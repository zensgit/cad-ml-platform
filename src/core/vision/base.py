"""Vision module base classes and models.

Provides:
- Pydantic models for vision requests/responses
- VisionProvider abstract base class
- Integration with OCR module for end-to-end analysis
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from typing import Literal

# ========== Request/Response Models ==========


class VisionAnalyzeRequest(BaseModel):
    """Request model for vision analysis."""

    image_url: Optional[str] = Field(None, description="URL of the image to analyze")
    image_base64: Optional[str] = Field(None, description="Base64-encoded image data")

    # Vision-specific options
    include_description: bool = Field(True, description="Include natural language description")
    include_ocr: bool = Field(True, description="Include OCR dimension/symbol extraction")

    # OCR provider routing (passed to OCRManager if include_ocr=True)
    ocr_provider: str = Field("auto", description="OCR provider: auto|paddle|deepseek")

    model_config = {
        "json_schema_extra": {
            "example": {
                "image_url": "https://example.com/drawing.png",
                "include_description": True,
                "include_ocr": True,
                "ocr_provider": "auto",
            }
        }
    }


class VisionDescription(BaseModel):
    """Natural language description of the image."""

    summary: str = Field(..., description="High-level summary of the drawing")
    details: List[str] = Field(default_factory=list, description="Detailed observations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")


class OcrResult(BaseModel):
    """OCR extraction results (dimensions, symbols, title block)."""

    dimensions: List[Dict[str, Any]] = Field(default_factory=list)
    symbols: List[Dict[str, Any]] = Field(default_factory=list)
    title_block: Dict[str, Any] = Field(default_factory=dict)
    fallback_level: Optional[str] = Field(None, description="json_strict|markdown_fence|text_regex")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class VisionAnalyzeResponse(BaseModel):
    """Response model for vision analysis."""

    success: bool = Field(..., description="Whether analysis succeeded")

    # Vision outputs
    description: Optional[VisionDescription] = Field(
        None, description="Natural language description"
    )

    # OCR outputs
    ocr: Optional[OcrResult] = Field(None, description="OCR extraction results")

    # Metadata
    provider: str = Field(..., description="Vision provider used (e.g., deepseek_stub)")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")

    # Error handling
    error: Optional[str] = Field(None, description="Error message if success=False")
    code: Optional[Literal["INPUT_ERROR", "INTERNAL_ERROR"]] = Field(
        None, description="Machine-readable error code if success=False"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "description": {
                    "summary": "Mechanical part with cylindrical features",
                    "details": [
                        "Main diameter: 20mm with ±0.02mm tolerance",
                        "Thread specification: M10×1.5",
                        "Surface roughness: Ra 3.2",
                    ],
                    "confidence": 0.92,
                },
                "ocr": {
                    "dimensions": [
                        {"type": "diameter", "value": 20, "tolerance": 0.02, "unit": "mm"}
                    ],
                    "symbols": [{"type": "surface_roughness", "value": "3.2"}],
                    "title_block": {"drawing_number": "CAD-2025-001", "material": "Aluminum 6061"},
                    "fallback_level": "json_strict",
                    "confidence": 0.95,
                },
                "provider": "deepseek_stub",
                "processing_time_ms": 234.5,
            }
        }
    }


# ========== Provider Abstract Base Class ==========


class VisionProvider(ABC):
    """Abstract base class for vision providers (DeepSeek-VL, future alternatives)."""

    @abstractmethod
    async def analyze_image(
        self, image_data: bytes, include_description: bool = True
    ) -> VisionDescription:
        """
        Analyze image and return natural language description.

        Args:
            image_data: Raw image bytes
            include_description: Whether to generate description (future: could skip for OCR-only)

        Returns:
            VisionDescription with summary, details, and confidence

        Raises:
            VisionProviderError: On provider-specific errors
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider identifier (e.g., 'deepseek_vl', 'deepseek_stub')."""
        pass


# ========== Exceptions ==========


class VisionError(Exception):
    """Base exception for vision module."""

    pass


class VisionProviderError(VisionError):
    """Exception raised by vision providers."""

    def __init__(self, provider: str, message: str):
        self.provider = provider
        self.message = message
        super().__init__(f"[{provider}] {message}")


class VisionInputError(VisionError):
    """Exception for invalid input (missing image_url/image_base64, etc.)."""

    pass
