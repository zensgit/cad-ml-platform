"""
材料属性查询功能

提供材料信息获取和属性搜索功能。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.core.materials.classify import classify_material_detailed
from src.core.materials.data_models import (
    MATERIAL_DATABASE,
    MaterialInfo,
    ProcessRecommendation,
)

logger = logging.getLogger(__name__)


def get_material_info(material: Optional[str]) -> Dict[str, Any]:
    """
    获取材料信息字典

    Args:
        material: 材料名称

    Returns:
        材料信息字典
    """
    info = classify_material_detailed(material)
    if info:
        return info.to_dict()
    return {
        "grade": material,
        "name": material,
        "category": None,
        "sub_category": None,
        "group": None,
        "found": False,
    }


def search_by_properties(
    density_range: Optional[Tuple[float, float]] = None,
    tensile_strength_range: Optional[Tuple[float, float]] = None,
    hardness_contains: Optional[str] = None,
    machinability: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    按属性搜索材料

    Args:
        density_range: 密度范围 (min, max) g/cm³
        tensile_strength_range: 抗拉强度范围 (min, max) MPa
        hardness_contains: 硬度包含的字符串 (如 "HRC", "HB")
        machinability: 可加工性 (excellent/good/fair/poor)
        category: 材料类别
        limit: 返回数量限制

    Returns:
        匹配的材料列表
    """
    results = []

    for grade, info in MATERIAL_DATABASE.items():
        # 类别过滤
        if category and info.category.value != category:
            continue

        props = info.properties

        # 密度过滤
        if density_range:
            if props.density is None:
                continue
            if not (density_range[0] <= props.density <= density_range[1]):
                continue

        # 抗拉强度过滤
        if tensile_strength_range:
            if props.tensile_strength is None:
                continue
            if not (tensile_strength_range[0] <= props.tensile_strength <= tensile_strength_range[1]):
                continue

        # 硬度过滤
        if hardness_contains:
            if props.hardness is None:
                continue
            if hardness_contains.upper() not in props.hardness.upper():
                continue

        # 可加工性过滤
        if machinability:
            if props.machinability != machinability:
                continue

        results.append({
            "grade": grade,
            "name": info.name,
            "category": info.category.value,
            "group": info.group.value,
            "density": props.density,
            "tensile_strength": props.tensile_strength,
            "hardness": props.hardness,
            "machinability": props.machinability,
        })

    # 按牌号排序
    results.sort(key=lambda x: str(x.get("grade") or ""))

    return results[:limit]
