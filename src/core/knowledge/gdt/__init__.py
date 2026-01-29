"""
GD&T (Geometric Dimensioning and Tolerancing) Knowledge Module.

Provides geometric tolerance specifications, datum reference frames,
and tolerance zone definitions per ISO GPS and ASME Y14.5 standards.

Reference Standards:
- ISO 1101:2017 - Geometrical tolerancing
- ISO 5459:2011 - Datums and datum systems
- ASME Y14.5-2018 - Dimensioning and Tolerancing
- GB/T 1182-2018 - Chinese national standard (equivalent to ISO 1101)
"""

from .symbols import (
    GDTCharacteristic,
    GDTCategory,
    ToleranceModifier,
    DatumModifier,
    get_gdt_symbol,
    get_all_symbols,
    GDT_SYMBOLS,
)
from .tolerances import (
    GeometricTolerance,
    ToleranceZone,
    get_tolerance_zone,
    get_recommended_tolerance,
    calculate_bonus_tolerance,
    TOLERANCE_RECOMMENDATIONS,
)
from .datums import (
    DatumFeature,
    DatumFeatureType,
    DatumPriority,
    DatumReferenceFrame,
    get_datum_priority,
    create_datum_reference_frame,
    DATUM_FEATURE_TYPES,
)
from .application import (
    GDTApplication,
    get_gdt_for_feature,
    get_inspection_method,
    interpret_feature_control_frame,
    COMMON_APPLICATIONS,
)

__all__ = [
    # Symbols
    "GDTCharacteristic",
    "GDTCategory",
    "ToleranceModifier",
    "DatumModifier",
    "get_gdt_symbol",
    "get_all_symbols",
    "GDT_SYMBOLS",
    # Tolerances
    "GeometricTolerance",
    "ToleranceZone",
    "get_tolerance_zone",
    "get_recommended_tolerance",
    "calculate_bonus_tolerance",
    "TOLERANCE_RECOMMENDATIONS",
    # Datums
    "DatumFeature",
    "DatumFeatureType",
    "DatumPriority",
    "DatumReferenceFrame",
    "get_datum_priority",
    "create_datum_reference_frame",
    "DATUM_FEATURE_TYPES",
    # Application
    "GDTApplication",
    "get_gdt_for_feature",
    "get_inspection_method",
    "interpret_feature_control_frame",
    "COMMON_APPLICATIONS",
]
