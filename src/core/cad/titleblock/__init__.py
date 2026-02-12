"""
Titleblock Parsing Module.

Provides intelligent titleblock parsing for CAD drawings:
- Automatic region detection
- Template matching
- Field extraction
- OCR integration
"""

from __future__ import annotations

from src.core.cad.titleblock.parser import (
    TitleblockParser,
    TitleblockMetadata,
    ParserConfig,
)
from src.core.cad.titleblock.region_detector import (
    RegionDetector,
    TitleblockRegion,
    DetectionMethod,
)
from src.core.cad.titleblock.template_library import (
    TemplateLibrary,
    TitleblockTemplate,
    FieldDefinition,
    TemplateMatch,
)
from src.core.cad.titleblock.field_extractor import (
    FieldExtractor,
    ExtractedField,
    ExtractionResult,
)

__all__ = [
    # Parser
    "TitleblockParser",
    "TitleblockMetadata",
    "ParserConfig",
    # Region Detection
    "RegionDetector",
    "TitleblockRegion",
    "DetectionMethod",
    # Templates
    "TemplateLibrary",
    "TitleblockTemplate",
    "FieldDefinition",
    "TemplateMatch",
    # Field Extraction
    "FieldExtractor",
    "ExtractedField",
    "ExtractionResult",
]
