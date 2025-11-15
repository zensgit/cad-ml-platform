"""Vision module for CAD ML Platform.

Provides vision-based analysis of engineering drawings with OCR integration.
"""

from .base import (
    VisionAnalyzeRequest,
    VisionAnalyzeResponse,
    VisionDescription,
    OcrResult,
    VisionProvider,
    VisionError,
    VisionProviderError,
    VisionInputError,
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
