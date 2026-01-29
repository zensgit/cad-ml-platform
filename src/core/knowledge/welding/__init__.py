"""
Welding Parameters Knowledge Module.

Provides welding parameters, joint design guidance, and process recommendations
for common welding operations in manufacturing.

Reference Standards:
- AWS D1.1 - Structural Welding Code - Steel
- ISO 4063 - Welding and allied processes - Nomenclature of processes
- ISO 15614 - Specification and qualification of welding procedures
- GB/T 985.1 - Recommended joint preparation for gas welding
"""

from .parameters import (
    WeldingProcess,
    WeldingPosition,
    JointType,
    WeldingParameters,
    get_welding_parameters,
    get_filler_material,
    calculate_heat_input,
    WELDING_PROCESS_DATABASE,
)
from .joints import (
    JointDesign,
    GrooveType,
    get_joint_design,
    recommend_joint_for_thickness,
    get_minimum_fillet_size,
    JOINT_DESIGN_DATABASE,
)
from .materials import (
    WeldabilityClass,
    BaseMaterial,
    get_weldability,
    get_preheat_temperature,
    calculate_carbon_equivalent,
    check_material_compatibility,
    WELDABILITY_DATABASE,
)

__all__ = [
    # Welding parameters
    "WeldingProcess",
    "WeldingPosition",
    "JointType",
    "WeldingParameters",
    "get_welding_parameters",
    "get_filler_material",
    "calculate_heat_input",
    "WELDING_PROCESS_DATABASE",
    # Joint design
    "JointDesign",
    "GrooveType",
    "get_joint_design",
    "recommend_joint_for_thickness",
    "get_minimum_fillet_size",
    "JOINT_DESIGN_DATABASE",
    # Weldability
    "WeldabilityClass",
    "BaseMaterial",
    "get_weldability",
    "get_preheat_temperature",
    "calculate_carbon_equivalent",
    "check_material_compatibility",
    "WELDABILITY_DATABASE",
]
