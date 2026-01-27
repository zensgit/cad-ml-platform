"""Materials module for material classification and properties."""

from src.core.materials.classifier import (
    MATERIAL_DATABASE,
    MATERIAL_EQUIVALENCE,
    MaterialCategory,
    MaterialGroup,
    MaterialInfo,
    MaterialSubCategory,
    classify_material_detailed,
    export_equivalence_csv,
    export_materials_csv,
    find_equivalent_material,
    get_material_equivalence,
    get_material_info,
    get_process_recommendations,
    list_material_standards,
)

__all__ = [
    "MATERIAL_DATABASE",
    "MATERIAL_EQUIVALENCE",
    "MaterialCategory",
    "MaterialGroup",
    "MaterialInfo",
    "MaterialSubCategory",
    "classify_material_detailed",
    "export_equivalence_csv",
    "export_materials_csv",
    "find_equivalent_material",
    "get_material_equivalence",
    "get_material_info",
    "get_process_recommendations",
    "list_material_standards",
]
