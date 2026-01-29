"""
Surface Treatment Knowledge Module.

Provides surface treatment process parameters, coating specifications,
and finish recommendations for manufacturing.

Reference Standards:
- ISO 2081 - Electroplated coatings of zinc
- ISO 10683 - Fasteners - Non-electrolytically applied zinc flake coatings
- MIL-A-8625 - Anodic coatings for aluminum
- ASTM B117 - Salt spray testing
"""

from .electroplating import (
    PlatingType,
    PlatingParameters,
    get_plating_parameters,
    get_plating_thickness,
    recommend_plating_for_application,
    PLATING_DATABASE,
)
from .anodizing import (
    AnodizingType,
    AnodizingParameters,
    get_anodizing_parameters,
    get_anodizing_colors,
    recommend_anodizing_for_application,
    ANODIZING_DATABASE,
)
from .coating import (
    CoatingType,
    CoatingParameters,
    get_coating_parameters,
    get_coating_for_environment,
    calculate_coating_life,
    COATING_DATABASE,
)

__all__ = [
    # Electroplating
    "PlatingType",
    "PlatingParameters",
    "get_plating_parameters",
    "get_plating_thickness",
    "recommend_plating_for_application",
    "PLATING_DATABASE",
    # Anodizing
    "AnodizingType",
    "AnodizingParameters",
    "get_anodizing_parameters",
    "get_anodizing_colors",
    "recommend_anodizing_for_application",
    "ANODIZING_DATABASE",
    # Coating
    "CoatingType",
    "CoatingParameters",
    "get_coating_parameters",
    "get_coating_for_environment",
    "calculate_coating_life",
    "COATING_DATABASE",
]
