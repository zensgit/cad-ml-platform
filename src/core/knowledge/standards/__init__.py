"""
Standard Parts Knowledge Module.

Provides specifications for common standard parts including:
- Metric threads (ISO 261/262)
- Rolling bearings (ISO 15)
- O-rings and seals (ISO 3601)

Reference Standards:
- ISO 261:1998 - ISO general purpose metric screw threads
- ISO 262:1998 - ISO general purpose metric screw threads - Selected sizes
- ISO 15:2017 - Rolling bearings - Radial bearings - Boundary dimensions
- ISO 3601-1:2012 - Fluid power systems - O-rings
"""

from .threads import (
    ThreadType,
    ThreadClass,
    MetricThread,
    get_thread_spec,
    get_thread_series,
    list_metric_threads,
    get_tap_drill_size,
    METRIC_THREADS,
)
from .bearings import (
    BearingType,
    BearingSeries,
    BearingSpec,
    get_bearing_spec,
    get_bearing_by_bore,
    list_bearings,
    suggest_bearing_for_shaft,
    BEARING_DATABASE,
)
from .seals import (
    SealType,
    ORingSpec,
    ORingMaterial,
    get_oring_spec,
    get_oring_by_id,
    list_orings,
    suggest_oring_material,
    ORING_DATABASE,
)

__all__ = [
    # Threads
    "ThreadType",
    "ThreadClass",
    "MetricThread",
    "get_thread_spec",
    "get_thread_series",
    "list_metric_threads",
    "get_tap_drill_size",
    "METRIC_THREADS",
    # Bearings
    "BearingType",
    "BearingSeries",
    "BearingSpec",
    "get_bearing_spec",
    "get_bearing_by_bore",
    "list_bearings",
    "suggest_bearing_for_shaft",
    "BEARING_DATABASE",
    # Seals
    "SealType",
    "ORingSpec",
    "ORingMaterial",
    "get_oring_spec",
    "get_oring_by_id",
    "list_orings",
    "suggest_oring_material",
    "ORING_DATABASE",
]
