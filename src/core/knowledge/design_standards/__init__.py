"""
Design Standards Knowledge Module.

Provides common mechanical design standards index and guidelines including:
- Surface finish standards (Ra values)
- Shaft/hole diameter series
- Chamfer and fillet standards
- General tolerances for linear and angular dimensions

Reference Standards:
- ISO 1302:2002 - Surface texture
- ISO 2768-1:1989 - General tolerances - Linear and angular
- ISO 2768-2:1989 - General tolerances - Geometrical
- GB/T 1804-2000 (Chinese equivalent)
"""

from .surface_finish import (
    SurfaceFinishGrade,
    get_ra_value,
    get_surface_finish_for_application,
    SURFACE_FINISH_TABLE,
)
from .general_tolerances import (
    GeneralToleranceClass,
    get_linear_tolerance,
    get_angular_tolerance,
    get_general_tolerance_table,
    LINEAR_TOLERANCE_TABLE,
)
from .design_features import (
    get_standard_chamfer,
    get_standard_fillet,
    get_preferred_diameter,
    PREFERRED_DIAMETERS,
    STANDARD_CHAMFERS,
    STANDARD_FILLETS,
)

__all__ = [
    # Surface finish
    "SurfaceFinishGrade",
    "get_ra_value",
    "get_surface_finish_for_application",
    "SURFACE_FINISH_TABLE",
    # General tolerances
    "GeneralToleranceClass",
    "get_linear_tolerance",
    "get_angular_tolerance",
    "get_general_tolerance_table",
    "LINEAR_TOLERANCE_TABLE",
    # Design features
    "get_standard_chamfer",
    "get_standard_fillet",
    "get_preferred_diameter",
    "PREFERRED_DIAMETERS",
    "STANDARD_CHAMFERS",
    "STANDARD_FILLETS",
]
