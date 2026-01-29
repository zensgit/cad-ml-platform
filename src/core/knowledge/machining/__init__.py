"""
Machining Parameters Knowledge Module.

Provides cutting parameters, tool selection guidance, and machining recommendations
for common manufacturing operations.

Reference Standards:
- ISO 3685:1993 - Tool-life testing with single-point turning tools
- Machinery's Handbook - Cutting speeds and feeds
- Tool manufacturer recommendations
"""

from .cutting import (
    MachiningOperation,
    CuttingParameters,
    get_cutting_parameters,
    calculate_spindle_speed,
    calculate_feed_rate,
    calculate_metal_removal_rate,
    CUTTING_SPEED_TABLE,
)
from .tooling import (
    ToolMaterial,
    ToolType,
    ToolGeometry,
    ToolRecommendation,
    get_tool_recommendation,
    select_tool_for_material,
    TOOL_DATABASE,
)
from .materials import (
    MachinabilityClass,
    WorkpieceMaterial,
    get_machinability,
    get_material_cutting_data,
    MACHINABILITY_DATABASE,
)

__all__ = [
    # Cutting parameters
    "MachiningOperation",
    "CuttingParameters",
    "get_cutting_parameters",
    "calculate_spindle_speed",
    "calculate_feed_rate",
    "calculate_metal_removal_rate",
    "CUTTING_SPEED_TABLE",
    # Tooling
    "ToolMaterial",
    "ToolType",
    "ToolGeometry",
    "ToolRecommendation",
    "get_tool_recommendation",
    "select_tool_for_material",
    "TOOL_DATABASE",
    # Materials
    "MachinabilityClass",
    "WorkpieceMaterial",
    "get_machinability",
    "get_material_cutting_data",
    "MACHINABILITY_DATABASE",
]
