"""Vision module for CAD ML Platform.

Provides vision-based analysis of engineering drawings with OCR integration.
"""

from .base import (
    OcrResult,
    VisionAnalyzeRequest,
    VisionAnalyzeResponse,
    VisionDescription,
    VisionError,
    VisionInputError,
    VisionProvider,
    VisionProviderError,
)
from .manager import VisionManager
from .providers import DeepSeekStubProvider, create_stub_provider

__all__ = [
    # Models
    "VisionAnalyzeRequest",
    "VisionAnalyzeResponse",
    "VisionDescription",
    "OcrResult",
    # Base classes
    "VisionProvider",
    # Manager
    "VisionManager",
    # Providers
    "DeepSeekStubProvider",
    "create_stub_provider",
    # Exceptions
    "VisionError",
    "VisionProviderError",
    "VisionInputError",
]
