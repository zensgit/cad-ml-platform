"""
材料工艺推荐系统

提供工艺推荐和材料推荐功能。
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


def get_process_recommendations(material: Optional[str]) -> ProcessRecommendation:
    """
    获取材料工艺推荐

    Args:
        material: 材料名称

    Returns:
        ProcessRecommendation
    """
    info = classify_material_detailed(material)
    if info:
        return info.process
    return ProcessRecommendation()


# ============================================================================
# 材料推荐系统
# ============================================================================

# 用途到材料组的映射
APPLICATION_MAP: Dict[str, Dict[str, Any]] = {
    # 结构件
    "structural": {
        "name": "结构件",
        "groups": ["carbon_steel", "alloy_steel", "aluminum"],
        "priorities": {"strength": 0.4, "cost": 0.3, "machinability": 0.3},
    },
    "load_bearing": {
        "name": "承载件",
        "groups": ["alloy_steel", "carbon_steel"],
        "priorities": {"strength": 0.5, "toughness": 0.3, "cost": 0.2},
    },
    # 耐腐蚀
    "corrosion_resistant": {
        "name": "耐腐蚀件",
        "groups": ["stainless_steel", "corrosion_resistant", "titanium"],
        "priorities": {"corrosion": 0.5, "strength": 0.3, "cost": 0.2},
    },
    "seawater": {
        "name": "海水环境",
        "groups": ["corrosion_resistant", "titanium", "copper"],
        "priorities": {"corrosion": 0.6, "strength": 0.2, "cost": 0.2},
    },
    "chemical": {
        "name": "化工环境",
        "groups": ["corrosion_resistant", "fluoropolymer"],
        "priorities": {"corrosion": 0.6, "temperature": 0.2, "cost": 0.2},
    },
    # 耐磨
    "wear_resistant": {
        "name": "耐磨件",
        "groups": ["alloy_steel", "cast_iron", "copper"],
        "priorities": {"hardness": 0.5, "toughness": 0.3, "cost": 0.2},
    },
    "bearing": {
        "name": "轴承/轴瓦",
        "groups": ["alloy_steel", "copper"],
        "priorities": {"hardness": 0.4, "wear": 0.4, "machinability": 0.2},
    },
    # 导电导热
    "electrical": {
        "name": "导电件",
        "groups": ["copper", "aluminum"],
        "priorities": {"conductivity": 0.6, "cost": 0.2, "machinability": 0.2},
    },
    "thermal": {
        "name": "导热件",
        "groups": ["copper", "aluminum"],
        "priorities": {"thermal": 0.5, "cost": 0.3, "machinability": 0.2},
    },
    # 弹性件
    "spring": {
        "name": "弹簧/弹性件",
        "groups": ["alloy_steel", "copper"],
        "priorities": {"elasticity": 0.5, "fatigue": 0.3, "corrosion": 0.2},
    },
    # 轻量化
    "lightweight": {
        "name": "轻量化",
        "groups": ["aluminum", "titanium", "engineering_plastic"],
        "priorities": {"weight": 0.5, "strength": 0.3, "cost": 0.2},
    },
    # 高温
    "high_temperature": {
        "name": "高温环境",
        "groups": ["corrosion_resistant", "titanium"],
        "priorities": {"temperature": 0.5, "strength": 0.3, "oxidation": 0.2},
    },
    # 食品医疗
    "food_grade": {
        "name": "食品级",
        "groups": ["stainless_steel", "engineering_plastic"],
        "priorities": {"safety": 0.5, "corrosion": 0.3, "machinability": 0.2},
    },
    "medical": {
        "name": "医疗器械",
        "groups": ["stainless_steel", "titanium"],
        "priorities": {"biocompatibility": 0.5, "corrosion": 0.3, "strength": 0.2},
    },
    # 精密加工
    "precision": {
        "name": "精密零件",
        "groups": ["copper", "aluminum", "alloy_steel"],
        "priorities": {"machinability": 0.5, "stability": 0.3, "cost": 0.2},
    },
}


# 材料替代关系 (材料 -> 可替代材料列表，按优先级排序)
MATERIAL_ALTERNATIVES: Dict[str, List[Dict[str, Any]]] = {
    # 不锈钢替代
    "S30408": [
        {"grade": "S31603", "reason": "更好的耐腐蚀性", "cost_factor": 1.2},
        {"grade": "S30403", "reason": "低碳版本，焊接性更好", "cost_factor": 1.1},
    ],
    "S31603": [
        {"grade": "S30408", "reason": "成本更低", "cost_factor": 0.8},
        {"grade": "C276", "reason": "极端耐腐蚀环境", "cost_factor": 3.0},
    ],
    # 碳钢替代
    "Q235B": [
        {"grade": "Q345R", "reason": "更高强度", "cost_factor": 1.1},
        {"grade": "20", "reason": "需要渗碳时", "cost_factor": 1.0},
    ],
    "45": [
        {"grade": "40Cr", "reason": "更高强度和韧性", "cost_factor": 1.3},
        {"grade": "42CrMo", "reason": "高强度高韧性", "cost_factor": 1.5},
    ],
    # 合金钢替代
    "40Cr": [
        {"grade": "42CrMo", "reason": "更高强度", "cost_factor": 1.2},
        {"grade": "45", "reason": "成本更低", "cost_factor": 0.8},
    ],
    "42CrMo": [
        {"grade": "40Cr", "reason": "成本更低", "cost_factor": 0.8},
        {"grade": "GCr15", "reason": "需要高硬度时", "cost_factor": 1.1},
    ],
    # 铝合金替代
    "6061": [
        {"grade": "7075", "reason": "更高强度", "cost_factor": 1.5},
        {"grade": "5052", "reason": "更好的耐腐蚀性", "cost_factor": 0.9},
    ],
    "7075": [
        {"grade": "6061", "reason": "成本更低，易加工", "cost_factor": 0.7},
        {"grade": "TC4", "reason": "极端强度要求", "cost_factor": 5.0},
    ],
    # 铜合金替代
    "H62": [
        {"grade": "H68", "reason": "更好的冷加工性", "cost_factor": 1.1},
        {"grade": "HPb59-1", "reason": "更好的切削性", "cost_factor": 1.0},
    ],
    "QBe2": [
        {"grade": "QSn6.5-0.1", "reason": "无毒替代（弹性稍差）", "cost_factor": 0.5},
        {"grade": "C17510", "reason": "低铍含量", "cost_factor": 0.9},
    ],
    "Cu65": [
        {"grade": "H62", "reason": "强度更高，成本更低", "cost_factor": 0.8},
        {"grade": "6061", "reason": "轻量化替代", "cost_factor": 0.6},
    ],
    # 钛合金替代
    "TC4": [
        {"grade": "TA2", "reason": "成本更低（强度较低）", "cost_factor": 0.6},
        {"grade": "Inconel718", "reason": "高温环境", "cost_factor": 1.5},
    ],
    "TA2": [
        {"grade": "S31603", "reason": "成本更低", "cost_factor": 0.3},
        {"grade": "TC4", "reason": "需要更高强度", "cost_factor": 1.7},
    ],
    # 耐蚀合金替代
    "C276": [
        {"grade": "C22", "reason": "类似性能，略低成本", "cost_factor": 0.9},
        {"grade": "Inconel625", "reason": "高温性能更好", "cost_factor": 1.1},
    ],
    "Inconel625": [
        {"grade": "Inconel718", "reason": "更高强度", "cost_factor": 1.2},
        {"grade": "C276", "reason": "更好的耐腐蚀性", "cost_factor": 0.9},
    ],
}


def get_material_recommendations(
    application: str,
    requirements: Optional[Dict[str, Any]] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    根据用途推荐材料

    Args:
        application: 用途代码 (structural, corrosion_resistant, electrical, etc.)
        requirements: 额外要求 {
            "min_strength": float,  # 最小抗拉强度
            "max_density": float,   # 最大密度
            "machinability": str,   # 可加工性要求
            "exclude_groups": list, # 排除的材料组
        }
        limit: 返回数量限制

    Returns:
        推荐的材料列表，包含匹配分数和推荐理由
    """
    if application not in APPLICATION_MAP:
        return []

    app_config = APPLICATION_MAP[application]
    target_groups = app_config["groups"]
    requirements = requirements or {}

    results: List[Tuple[float, str, MaterialInfo, str]] = []

    for grade, info in MATERIAL_DATABASE.items():
        # 检查是否在目标材料组
        if info.group.value not in target_groups:
            continue

        # 检查排除组
        if info.group.value in requirements.get("exclude_groups", []):
            continue

        # 检查最小强度
        min_strength = requirements.get("min_strength")
        if min_strength and (info.properties.tensile_strength or 0) < min_strength:
            continue

        # 检查最大密度
        max_density = requirements.get("max_density")
        if max_density and (info.properties.density or 100) > max_density:
            continue

        # 检查可加工性
        req_machinability = requirements.get("machinability")
        if req_machinability:
            mach_order = {"excellent": 4, "good": 3, "fair": 2, "poor": 1}
            req_level = mach_order.get(req_machinability, 0)
            mat_level = mach_order.get(info.properties.machinability or "fair", 2)
            if mat_level < req_level:
                continue

        # 计算匹配分数
        score = 0.0
        reasons = []

        # 材料组匹配度（越靠前越匹配）
        group_idx = target_groups.index(info.group.value)
        group_score = 1.0 - (group_idx * 0.2)
        score += group_score * 0.3
        reasons.append(f"适合{app_config['name']}")

        # 属性评分
        props = info.properties
        if props.tensile_strength:
            if props.tensile_strength >= 800:
                score += 0.2
                reasons.append("高强度")
            elif props.tensile_strength >= 400:
                score += 0.1

        if props.machinability == "excellent":
            score += 0.15
            reasons.append("易加工")
        elif props.machinability == "good":
            score += 0.1

        if props.density and props.density < 5:
            score += 0.1
            reasons.append("轻量化")

        # 特殊工艺加分
        if info.process.special_tooling:
            score -= 0.1  # 需要特殊刀具减分

        if len(info.process.warnings) > 2:
            score -= 0.05  # 警告多减分

        results.append((score, grade, info, "；".join(reasons[:3])))

    # 按分数排序
    results.sort(key=lambda x: (-x[0], x[1]))

    # 格式化返回
    formatted = []
    for score, grade, info, reason in results[:limit]:
        formatted.append({
            "grade": grade,
            "name": info.name,
            "group": info.group.value,
            "score": round(score, 2),
            "reason": reason,
            "properties": {
                "density": info.properties.density,
                "tensile_strength": info.properties.tensile_strength,
                "machinability": info.properties.machinability,
            },
        })

    return formatted


def get_alternative_materials(
    grade: str,
    preference: str = "similar",
) -> List[Dict[str, Any]]:
    """
    获取替代材料建议

    Args:
        grade: 当前材料牌号
        preference: 替代偏好
            - "similar": 性能相近的替代
            - "cheaper": 成本更低的替代
            - "better": 性能更好的替代

    Returns:
        替代材料列表
    """
    # 先尝试获取材料信息
    info = classify_material_detailed(grade)
    if not info:
        return []

    # 使用规范牌号
    grade = info.grade

    # 获取预定义的替代关系
    predefined = MATERIAL_ALTERNATIVES.get(grade, [])

    results = []

    # 添加预定义替代
    for alt in predefined:
        alt_info = MATERIAL_DATABASE.get(alt["grade"])
        if not alt_info:
            continue

        # 根据偏好过滤
        if preference == "cheaper" and alt["cost_factor"] >= 1.0:
            continue
        if preference == "better" and alt["cost_factor"] <= 1.0:
            continue

        results.append({
            "grade": alt["grade"],
            "name": alt_info.name,
            "group": alt_info.group.value,
            "reason": alt["reason"],
            "cost_factor": alt["cost_factor"],
            "source": "predefined",
        })

    # 如果预定义不足，自动寻找同组材料
    if len(results) < 3:
        for g, alt_info in MATERIAL_DATABASE.items():
            if g == grade:
                continue
            if g in [r["grade"] for r in results]:
                continue
            if alt_info.group != info.group:
                continue

            # 简单的成本估算（基于密度和强度）
            cost_factor = 1.0
            if alt_info.properties.tensile_strength and info.properties.tensile_strength:
                strength_ratio = alt_info.properties.tensile_strength / info.properties.tensile_strength
                cost_factor *= strength_ratio ** 0.5

            # 根据偏好过滤
            if preference == "cheaper" and cost_factor >= 1.0:
                continue
            if preference == "better" and cost_factor <= 1.0:
                continue

            reason = f"同组材料（{info.group.value}）"
            if cost_factor < 1.0:
                reason = "成本较低的同组材料"
            elif cost_factor > 1.0:
                reason = "性能更好的同组材料"

            results.append({
                "grade": g,
                "name": alt_info.name,
                "group": alt_info.group.value,
                "reason": reason,
                "cost_factor": round(cost_factor, 2),
                "source": "auto",
            })

            if len(results) >= 5:
                break

    return results


def list_applications() -> List[Dict[str, str]]:
    """
    列出所有支持的用途

    Returns:
        用途列表
    """
    return [
        {"code": code, "name": config["name"]}
        for code, config in APPLICATION_MAP.items()
    ]
