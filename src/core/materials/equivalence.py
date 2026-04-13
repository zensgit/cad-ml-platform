"""
材料等价表系统

提供中外标准材料对照和等价查找功能。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.core.materials.classify import classify_material_detailed
from src.core.materials.data_models import MATERIAL_EQUIVALENCE

logger = logging.getLogger(__name__)


def get_material_equivalence(material: str) -> Optional[Dict[str, str]]:
    """
    获取材料等价表

    Args:
        material: 材料名称或牌号

    Returns:
        等价表字典 {标准体系: 等价牌号} 或 None
    """
    # 先尝试分类获取标准牌号
    info = classify_material_detailed(material)
    if info:
        grade = info.grade
        if grade in MATERIAL_EQUIVALENCE:
            return MATERIAL_EQUIVALENCE[grade]

    # 直接查找
    if material in MATERIAL_EQUIVALENCE:
        return MATERIAL_EQUIVALENCE[material]

    # 反向查找（输入的可能是其他标准的牌号）
    material_upper = material.upper().replace("-", "").replace(" ", "")
    for grade, equiv in MATERIAL_EQUIVALENCE.items():
        for std, val in equiv.items():
            if std != "name":
                val_clean = val.upper().replace("-", "").replace(" ", "")
                if val_clean == material_upper:
                    return equiv

    return None


def find_equivalent_material(material: str, target_standard: str = "CN") -> Optional[str]:
    """
    查找等价材料牌号

    Args:
        material: 材料名称或牌号
        target_standard: 目标标准体系 (CN/US/JP/DE/UNS)

    Returns:
        等价牌号 或 None
    """
    equiv = get_material_equivalence(material)
    if equiv and target_standard in equiv:
        return equiv[target_standard]
    return None


def list_material_standards(material: str) -> List[Tuple[str, str]]:
    """
    列出材料的所有标准牌号

    Args:
        material: 材料名称或牌号

    Returns:
        [(标准体系, 牌号), ...] 列表
    """
    equiv = get_material_equivalence(material)
    if equiv:
        return [(std, val) for std, val in equiv.items() if std != "name"]
    return []
