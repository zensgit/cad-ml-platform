"""
DWG Native Support Module (C1).

Provides native DWG file support through:
- ODA File Converter integration
- Direct DWG parsing (limited)
- Batch conversion utilities
"""

from src.ml.augmentation.cad import (
    CADAugmentation,
    LayerShuffle,
    EntityDropout,
)

from src.core.cad.dwg.converter import (
    DWGConverter,
    ConverterConfig,
    ConversionResult,
    convert_dwg_to_dxf,
    batch_convert,
)
from src.core.cad.dwg.parser import (
    DWGParser,
    DWGHeader,
    DWGEntity,
    parse_dwg_header,
)
from src.core.cad.dwg.manager import (
    DWGManager,
    DWGFile,
    get_dwg_manager,
)

__all__ = [
    # Converter
    "DWGConverter",
    "ConverterConfig",
    "ConversionResult",
    "convert_dwg_to_dxf",
    "batch_convert",
    # Parser
    "DWGParser",
    "DWGHeader",
    "DWGEntity",
    "parse_dwg_header",
    # Manager
    "DWGManager",
    "DWGFile",
    "get_dwg_manager",
]
