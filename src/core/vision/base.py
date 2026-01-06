"""Vision module base classes and models.

Provides:
- Pydantic models for vision requests/responses
- VisionProvider abstract base class
- Integration with OCR module for end-to-end analysis
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator

from src.core.errors import ErrorCode

# ========== Request/Response Models ==========


class VisionAnalyzeRequest(BaseModel):
    """Request model for vision analysis."""

    image_url: Optional[str] = Field(None, description="URL of the image to analyze")
    image_base64: Optional[str] = Field(None, description="Base64-encoded image data")

    # Vision-specific options
    include_description: bool = Field(True, description="Include natural language description")
    include_ocr: bool = Field(True, description="Include OCR dimension/symbol extraction")
    include_cad_stats: bool = Field(
        False, description="Include CAD feature heuristic summary stats"
    )

    # OCR provider routing (passed to OCRManager if include_ocr=True)
    ocr_provider: str = Field("auto", description="OCR provider: auto|paddle|deepseek")
    cad_feature_thresholds: Optional[Dict[str, float]] = Field(
        None, description="Overrides for CAD feature heuristic thresholds"
    )

    @field_validator("cad_feature_thresholds")
    @classmethod
    def _validate_cad_feature_thresholds(
        cls, value: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("cad_feature_thresholds must be a dict of numeric values")
        allowed: Set[str] = {
            "max_dim",
            "ink_threshold",
            "min_area",
            "line_aspect",
            "line_elongation",
            "circle_aspect",
            "circle_fill_min",
            "arc_aspect",
            "arc_fill_min",
            "arc_fill_max",
        }
        unknown = set(value.keys()) - allowed
        if unknown:
            raise ValueError(f"cad_feature_thresholds has unsupported keys: {sorted(unknown)}")
        for key, val in value.items():
            if not isinstance(val, (int, float)):
                raise ValueError(f"cad_feature_thresholds[{key}] must be numeric")
            if val <= 0:
                raise ValueError(f"cad_feature_thresholds[{key}] must be > 0")
        return value

    model_config = {
        "json_schema_extra": {
            "example": {
                "image_url": "https://example.com/drawing.png",
                "include_description": True,
                "include_ocr": True,
                "ocr_provider": "auto",
                "include_cad_stats": False,
                "cad_feature_thresholds": {"line_aspect": 5.0, "arc_fill_min": 0.08},
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


class CadFeatureStats(BaseModel):
    """Summary stats for heuristic CAD feature extraction."""

    line_count: int = Field(..., ge=0, description="Number of detected line segments")
    circle_count: int = Field(..., ge=0, description="Number of detected circles")
    arc_count: int = Field(..., ge=0, description="Number of detected arcs")
    line_angle_bins: Dict[str, int] = Field(..., description="Line angle histogram buckets")
    line_angle_avg: Optional[float] = Field(None, description="Average line angle in degrees")
    arc_sweep_avg: Optional[float] = Field(None, description="Average arc sweep in degrees")
    arc_sweep_bins: Dict[str, int] = Field(..., description="Arc sweep histogram buckets")

    model_config = {"extra": "allow"}


class VisionAnalyzeResponse(BaseModel):
    """Response model for vision analysis."""

    success: bool = Field(..., description="Whether analysis succeeded")

    # Vision outputs
    description: Optional[VisionDescription] = Field(
        None, description="Natural language description"
    )

    # OCR outputs
    ocr: Optional[OcrResult] = Field(None, description="OCR extraction results")

    # CAD feature stats
    cad_feature_stats: Optional[CadFeatureStats] = Field(
        None, description="CAD feature heuristic summary stats"
    )

    # Metadata
    provider: str = Field(..., description="Vision provider used (e.g., deepseek_stub)")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")

    # Error handling
    error: Optional[str] = Field(None, description="Error message if success=False")
    code: Optional[ErrorCode] = Field(
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
                "cad_feature_stats": {
                    "line_count": 2,
                    "circle_count": 1,
                    "arc_count": 0,
                    "line_angle_bins": {"0-30": 2, "30-60": 0, "60-90": 0, "90-120": 0, "120-150": 0, "150-180": 0},
                    "line_angle_avg": 12.5,
                    "arc_sweep_avg": None,
                    "arc_sweep_bins": {"0-90": 0, "90-180": 0, "180-270": 0, "270-360": 0},
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
