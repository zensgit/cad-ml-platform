"""Materials module for material classification and properties."""

from src.core.materials.classifier import (
    MATERIAL_DATABASE,
    MaterialCategory,
    MaterialInfo,
    MaterialSubCategory,
    classify_material_detailed,
    get_material_info,
    get_process_recommendations,
)

__all__ = [
    "MATERIAL_DATABASE",
    "MaterialCategory",
    "MaterialInfo",
    "MaterialSubCategory",
    "classify_material_detailed",
    "get_material_info",
    "get_process_recommendations",
]
