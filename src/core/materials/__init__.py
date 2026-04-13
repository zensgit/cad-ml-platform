"""Materials module for material classification and properties.

This module re-exports all public APIs from the sub-modules so that
external code can continue to import from ``src.core.materials`` or
``src.core.materials.classifier`` without changes.
"""

# --- data models (enums, dataclasses, databases) ---
from src.core.materials.data_models import (
    MATERIAL_DATABASE,
    MATERIAL_EQUIVALENCE,
    MaterialCategory,
    MaterialGroup,
    MaterialInfo,
    MaterialProperties,
    MaterialSubCategory,
    ProcessRecommendation,
)

# --- classification & search ---
from src.core.materials.classify import (
    classify_material_detailed,
    classify_material_simple,
    search_materials,
)

# --- properties ---
from src.core.materials.properties import (
    get_material_info,
    search_by_properties,
)

# --- processing / recommendations ---
from src.core.materials.processing import (
    get_alternative_materials,
    get_material_recommendations,
    get_process_recommendations,
    list_applications,
)

# --- equivalence ---
from src.core.materials.equivalence import (
    find_equivalent_material,
    get_material_equivalence,
    list_material_standards,
)

# --- cost ---
from src.core.materials.cost import (
    MATERIAL_COST_DATA,
    compare_material_costs,
    get_cost_tier_info,
    get_material_cost,
    search_by_cost,
)

# --- compatibility ---
from src.core.materials.compatibility import (
    check_full_compatibility,
    check_galvanic_corrosion,
    check_heat_treatment_compatibility,
    check_weld_compatibility,
)

# --- export ---
from src.core.materials.export import (
    export_equivalence_csv,
    export_materials_csv,
)

__all__ = [
    "MATERIAL_COST_DATA",
    "MATERIAL_DATABASE",
    "MATERIAL_EQUIVALENCE",
    "MaterialCategory",
    "MaterialGroup",
    "MaterialInfo",
    "MaterialProperties",
    "MaterialSubCategory",
    "ProcessRecommendation",
    "check_full_compatibility",
    "check_galvanic_corrosion",
    "check_heat_treatment_compatibility",
    "check_weld_compatibility",
    "classify_material_detailed",
    "classify_material_simple",
    "compare_material_costs",
    "export_equivalence_csv",
    "export_materials_csv",
    "find_equivalent_material",
    "get_alternative_materials",
    "get_cost_tier_info",
    "get_material_cost",
    "get_material_equivalence",
    "get_material_info",
    "get_material_recommendations",
    "get_process_recommendations",
    "list_applications",
    "list_material_standards",
    "search_by_cost",
    "search_by_properties",
    "search_materials",
]
