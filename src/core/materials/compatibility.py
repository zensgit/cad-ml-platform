"""
材料兼容性检查

提供焊接兼容性、电偶腐蚀和热处理兼容性检查功能。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.core.materials.classify import classify_material_detailed
from src.core.materials.data_models import MaterialCategory

logger = logging.getLogger(__name__)


# ============================================================================
# 材料兼容性检查
# ============================================================================

# 焊接兼容性矩阵
# 等级: "excellent", "good", "fair", "poor", "not_recommended"
WELD_COMPATIBILITY: Dict[str, Dict[str, Dict[str, Any]]] = {
    # 碳钢焊接
    "carbon_steel": {
        "carbon_steel": {"rating": "excellent", "method": "普通焊条/CO2保护焊", "notes": "同组材料焊接性好"},
        "alloy_steel": {"rating": "good", "method": "低氢焊条", "notes": "需预热"},
        "stainless_steel": {"rating": "fair", "method": "309L/309Mo焊材", "notes": "异种钢焊接，需选择合适焊材"},
        "cast_iron": {"rating": "fair", "method": "镍基焊条", "notes": "需预热和缓冷"},
        "aluminum": {"rating": "not_recommended", "method": "-", "notes": "不建议直接焊接"},
        "copper": {"rating": "poor", "method": "铜镍焊材", "notes": "困难，不推荐"},
        "titanium": {"rating": "not_recommended", "method": "-", "notes": "不可焊接"},
    },
    # 不锈钢焊接
    "stainless_steel": {
        "stainless_steel": {"rating": "excellent", "method": "同材质焊材/TIG", "notes": "注意防止敏化"},
        "carbon_steel": {"rating": "fair", "method": "309L焊材", "notes": "异种钢焊接"},
        "alloy_steel": {"rating": "fair", "method": "309L/309Mo焊材", "notes": "需预热"},
        "corrosion_resistant": {"rating": "good", "method": "镍基焊材", "notes": "选择合适的镍基焊材"},
        "titanium": {"rating": "not_recommended", "method": "-", "notes": "不可焊接"},
        "aluminum": {"rating": "not_recommended", "method": "-", "notes": "不建议直接焊接"},
    },
    # 铝合金焊接
    "aluminum": {
        "aluminum": {"rating": "excellent", "method": "TIG/MIG-4043/5356", "notes": "需清洁表面氧化膜"},
        "carbon_steel": {"rating": "not_recommended", "method": "-", "notes": "不建议直接焊接"},
        "stainless_steel": {"rating": "not_recommended", "method": "-", "notes": "不建议直接焊接"},
        "copper": {"rating": "poor", "method": "钎焊", "notes": "只能钎焊"},
    },
    # 铜合金焊接
    "copper": {
        "copper": {"rating": "good", "method": "TIG/氧乙炔", "notes": "需预热，导热快"},
        "carbon_steel": {"rating": "poor", "method": "铜镍焊材", "notes": "困难"},
        "stainless_steel": {"rating": "fair", "method": "镍基焊材", "notes": "可行但困难"},
        "aluminum": {"rating": "poor", "method": "钎焊", "notes": "只能钎焊"},
    },
    # 钛合金焊接
    "titanium": {
        "titanium": {"rating": "good", "method": "TIG/真空电子束", "notes": "需惰性气体全保护"},
        "stainless_steel": {"rating": "not_recommended", "method": "-", "notes": "形成脆性金属间化合物"},
        "carbon_steel": {"rating": "not_recommended", "method": "-", "notes": "不可焊接"},
        "aluminum": {"rating": "not_recommended", "method": "-", "notes": "不可焊接"},
    },
    # 耐蚀合金焊接
    "corrosion_resistant": {
        "corrosion_resistant": {"rating": "good", "method": "同材质焊材/TIG", "notes": "需选择匹配焊材"},
        "stainless_steel": {"rating": "good", "method": "镍基焊材", "notes": "ERNiCrMo-3等"},
        "carbon_steel": {"rating": "fair", "method": "镍基焊材", "notes": "需预热"},
    },
    # 镁合金焊接
    "magnesium": {
        "magnesium": {"rating": "good", "method": "TIG/MIG", "notes": "需惰性气体保护，注意防火"},
        "aluminum": {"rating": "poor", "method": "钎焊/搅拌摩擦", "notes": "困难，需特殊工艺"},
        "carbon_steel": {"rating": "not_recommended", "method": "-", "notes": "不可焊接"},
        "stainless_steel": {"rating": "not_recommended", "method": "-", "notes": "不可焊接"},
    },
    # 硬质合金焊接
    "cemented_carbide": {
        "cemented_carbide": {"rating": "poor", "method": "钎焊/扩散焊", "notes": "只能钎焊"},
        "carbon_steel": {"rating": "fair", "method": "银基钎料钎焊", "notes": "用于刀具镶嵌"},
        "alloy_steel": {"rating": "fair", "method": "银基钎料钎焊", "notes": "用于刀具/模具"},
    },
    # 工具钢焊接
    "tool_steel": {
        "tool_steel": {"rating": "poor", "method": "TIG/预热+缓冷", "notes": "容易开裂，需严格控温"},
        "carbon_steel": {"rating": "fair", "method": "低氢焊条", "notes": "需预热300℃以上"},
        "alloy_steel": {"rating": "fair", "method": "低氢焊条", "notes": "需预热300℃以上"},
    },
}

# 电偶腐蚀序列 (galvanic series in seawater)
# 数值越小越活泼（阳极），越大越惰性（阴极）
GALVANIC_SERIES: Dict[str, float] = {
    # 活泼端 (阳极)
    "magnesium": -1.6,
    "zinc": -1.0,
    "aluminum": -0.8,
    "carbon_steel": -0.6,
    "cast_iron": -0.5,
    "alloy_steel": -0.5,
    "tool_steel": -0.45,  # 工具钢略惰性于普通合金钢
    "stainless_steel_active": -0.4,  # 活化态不锈钢
    "copper": -0.2,
    "stainless_steel": 0.0,  # 钝化态不锈钢
    "titanium": 0.1,
    "corrosion_resistant": 0.15,  # 镍基合金
    "cemented_carbide": 0.2,  # 硬质合金（惰性）
    # 惰性端 (阴极)
}

# 电偶腐蚀风险阈值
GALVANIC_RISK_THRESHOLDS = {
    "safe": 0.15,      # 电位差 < 0.15V: 安全
    "low": 0.25,       # 电位差 0.15-0.25V: 低风险
    "medium": 0.4,     # 电位差 0.25-0.4V: 中风险
    "high": 0.6,       # 电位差 0.4-0.6V: 高风险
    # > 0.6V: 严重风险
}


def check_weld_compatibility(
    material1: str,
    material2: str,
) -> Dict[str, Any]:
    """
    检查两种材料的焊接兼容性

    Args:
        material1: 第一种材料牌号
        material2: 第二种材料牌号

    Returns:
        焊接兼容性信息
    """
    # 获取材料信息
    info1 = classify_material_detailed(material1)
    info2 = classify_material_detailed(material2)

    if not info1 or not info2:
        return {
            "compatible": False,
            "error": "材料未找到",
            "material1": material1,
            "material2": material2,
        }

    group1 = info1.group.value
    group2 = info2.group.value

    # 查找兼容性数据
    compat_data = None

    # 先查 group1 -> group2
    if group1 in WELD_COMPATIBILITY:
        if group2 in WELD_COMPATIBILITY[group1]:
            compat_data = WELD_COMPATIBILITY[group1][group2]

    # 再查 group2 -> group1 (对称)
    if not compat_data and group2 in WELD_COMPATIBILITY:
        if group1 in WELD_COMPATIBILITY[group2]:
            compat_data = WELD_COMPATIBILITY[group2][group1]

    # 如果没有数据，返回默认
    if not compat_data:
        # 同组默认可焊
        if group1 == group2:
            compat_data = {
                "rating": "good",
                "method": "同材质焊材",
                "notes": "同组材料，通常可焊",
            }
        else:
            compat_data = {
                "rating": "unknown",
                "method": "需工艺评定",
                "notes": "无现成数据，建议工艺评定",
            }

    rating = compat_data["rating"]
    compatible = rating in ["excellent", "good", "fair"]

    return {
        "compatible": compatible,
        "rating": rating,
        "rating_cn": {
            "excellent": "优秀",
            "good": "良好",
            "fair": "一般",
            "poor": "困难",
            "not_recommended": "不推荐",
            "unknown": "未知",
        }.get(rating, rating),
        "method": compat_data.get("method", ""),
        "notes": compat_data.get("notes", ""),
        "material1": {
            "grade": info1.grade,
            "name": info1.name,
            "group": group1,
        },
        "material2": {
            "grade": info2.grade,
            "name": info2.name,
            "group": group2,
        },
    }


def check_galvanic_corrosion(
    material1: str,
    material2: str,
) -> Dict[str, Any]:
    """
    检查两种材料的电偶腐蚀风险

    Args:
        material1: 第一种材料牌号
        material2: 第二种材料牌号

    Returns:
        电偶腐蚀风险信息
    """
    # 获取材料信息
    info1 = classify_material_detailed(material1)
    info2 = classify_material_detailed(material2)

    if not info1 or not info2:
        return {
            "risk": "unknown",
            "error": "材料未找到",
        }

    group1 = info1.group.value
    group2 = info2.group.value

    # 非金属不参与电偶腐蚀
    if info1.category != MaterialCategory.METAL or info2.category != MaterialCategory.METAL:
        return {
            "risk": "none",
            "risk_cn": "无",
            "notes": "非金属材料不参与电偶腐蚀",
            "material1": {"grade": info1.grade, "name": info1.name},
            "material2": {"grade": info2.grade, "name": info2.name},
        }

    # 获取电偶序列位置
    potential1 = GALVANIC_SERIES.get(group1)
    potential2 = GALVANIC_SERIES.get(group2)

    if potential1 is None or potential2 is None:
        return {
            "risk": "unknown",
            "risk_cn": "未知",
            "notes": "缺少电偶序列数据",
            "material1": {"grade": info1.grade, "name": info1.name, "group": group1},
            "material2": {"grade": info2.grade, "name": info2.name, "group": group2},
        }

    # 计算电位差
    potential_diff = abs(potential1 - potential2)

    # 判断阳极/阴极
    if potential1 < potential2:
        anode = info1
        cathode = info2
    else:
        anode = info2
        cathode = info1

    # 评估风险
    if potential_diff < GALVANIC_RISK_THRESHOLDS["safe"]:
        risk = "safe"
        risk_cn = "安全"
        recommendation = "可直接接触使用"
    elif potential_diff < GALVANIC_RISK_THRESHOLDS["low"]:
        risk = "low"
        risk_cn = "低风险"
        recommendation = "干燥环境可用，潮湿环境需注意"
    elif potential_diff < GALVANIC_RISK_THRESHOLDS["medium"]:
        risk = "moderate"
        risk_cn = "中风险"
        recommendation = "建议绝缘隔离或表面处理"
    elif potential_diff < GALVANIC_RISK_THRESHOLDS["high"]:
        risk = "high"
        risk_cn = "高风险"
        recommendation = "必须绝缘隔离，避免电解质环境"
    else:
        risk = "severe"
        risk_cn = "严重"
        recommendation = "禁止直接接触，必须完全隔离"

    return {
        "risk": risk,
        "risk_cn": risk_cn,
        "potential_difference": round(potential_diff, 2),
        "recommendation": recommendation,
        "anode": {
            "grade": anode.grade,
            "name": anode.name,
            "role": "阳极（被腐蚀）",
        },
        "cathode": {
            "grade": cathode.grade,
            "name": cathode.name,
            "role": "阴极（受保护）",
        },
        "material1": {"grade": info1.grade, "name": info1.name, "group": group1},
        "material2": {"grade": info2.grade, "name": info2.name, "group": group2},
    }


def check_heat_treatment_compatibility(
    material: str,
    treatment: str,
) -> Dict[str, Any]:
    """
    检查材料与热处理工艺的兼容性

    Args:
        material: 材料牌号
        treatment: 热处理工艺名称

    Returns:
        兼容性信息
    """
    info = classify_material_detailed(material)

    if not info:
        return {
            "compatible": False,
            "error": "材料未找到",
        }

    # 检查推荐热处理
    recommended = info.process.heat_treatments
    forbidden = info.process.forbidden_heat_treatments

    # 标准化处理名称
    treatment_normalized = treatment.strip()

    # 检查是否被禁止
    for f in forbidden:
        if treatment_normalized in f or f in treatment_normalized:
            return {
                "compatible": False,
                "status": "forbidden",
                "status_cn": "禁止",
                "grade": info.grade,
                "name": info.name,
                "treatment": treatment,
                "reason": f"该材料禁止进行{f}处理",
                "recommended_treatments": recommended,
            }

    # 检查是否推荐
    for r in recommended:
        if treatment_normalized in r or r in treatment_normalized:
            return {
                "compatible": True,
                "status": "recommended",
                "status_cn": "推荐",
                "grade": info.grade,
                "name": info.name,
                "treatment": treatment,
                "reason": f"该材料推荐进行{r}处理",
                "recommended_treatments": recommended,
            }

    # 不在推荐列表也不在禁止列表
    return {
        "compatible": True,
        "status": "allowed",
        "status_cn": "可行",
        "grade": info.grade,
        "name": info.name,
        "treatment": treatment,
        "reason": "该热处理不在推荐/禁止列表中，需工艺验证",
        "recommended_treatments": recommended,
        "forbidden_treatments": forbidden,
    }


def check_full_compatibility(
    material1: str,
    material2: str,
) -> Dict[str, Any]:
    """
    全面检查两种材料的兼容性

    Args:
        material1: 第一种材料牌号
        material2: 第二种材料牌号

    Returns:
        完整兼容性报告
    """
    weld = check_weld_compatibility(material1, material2)
    galvanic = check_galvanic_corrosion(material1, material2)

    # 综合评估
    issues = []
    recommendations = []

    # 焊接问题
    if not weld.get("compatible", False):
        issues.append(f"焊接兼容性差: {weld.get('rating_cn', '未知')}")
        if weld.get("notes"):
            recommendations.append(f"焊接: {weld['notes']}")

    # 电偶腐蚀问题
    galvanic_risk = galvanic.get("risk", "unknown")
    if galvanic_risk in ["medium", "high", "severe"]:
        issues.append(f"电偶腐蚀风险: {galvanic.get('risk_cn', '未知')}")
        if galvanic.get("recommendation"):
            recommendations.append(f"防腐蚀: {galvanic['recommendation']}")

    # 总体评估
    if not issues:
        overall = "compatible"
        overall_cn = "兼容"
    elif len(issues) == 1:
        overall = "caution"
        overall_cn = "需注意"
    else:
        overall = "incompatible"
        overall_cn = "不兼容"

    return {
        "overall": overall,
        "overall_cn": overall_cn,
        "issues": issues,
        "recommendations": recommendations,
        "weld_compatibility": weld,
        "galvanic_corrosion": galvanic,
    }
