"""
材料分类与搜索功能

提供材料分类、识别和搜索功能。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.core.materials.data_models import (
    MATERIAL_DATABASE,
    MATERIAL_MATCH_PATTERNS,
    MaterialCategory,
    MaterialGroup,
    MaterialInfo,
    MaterialSubCategory,
)

logger = logging.getLogger(__name__)


# 拼音映射表 (常用材料名称)
PINYIN_MAP: Dict[str, List[str]] = {
    # 材料类别
    "butixiugang": ["不锈钢", "S30408", "S31603"],
    "buxiugang": ["不锈钢", "S30408", "S31603"],
    "tansugang": ["碳素钢", "Q235B", "45", "20"],
    "hejingang": ["合金钢", "40Cr", "42CrMo", "GCr15"],
    "zhutie": ["铸铁", "HT200", "QT400"],
    "lvhejin": ["铝合金", "6061", "7075"],
    "tonghejin": ["铜合金", "H62", "QBe2"],
    "taihejin": ["钛合金", "TA2", "TC4", "TA1", "TC11", "TB6", "TC21"],
    "nieji": ["镍基", "C276", "Inconel625"],
    "meihejin": ["镁合金", "AZ31B", "AZ91D", "ZK60"],
    "yingzhihejin": ["硬质合金", "YG8", "YT15", "YG6"],
    "wugang": ["钨钢", "YG8", "YG6"],

    # 具体材料
    "huangtong": ["黄铜", "H62", "H68"],
    "qingtong": ["青铜", "QSn4-3", "QAl9-4"],
    "baitong": ["白铜", "CuNi10Fe1Mn"],
    "zitong": ["紫铜", "Cu65"],
    "chungang": ["纯钢", "Q235B"],
    "digang": ["低碳钢", "Q235B", "20"],
    "zhonggang": ["中碳钢", "45"],
    "gaogang": ["高碳钢", "GCr15"],
    "moju": ["模具钢", "Cr12MoV", "H13"],
    "danjia": ["弹簧", "QSn6.5-0.1", "QBe2"],
    "naifu": ["耐腐蚀", "C276", "S31603"],
    "naimo": ["耐磨", "GCr15", "Stellite6"],
    "daodian": ["导电", "Cu65", "H62"],
    "daore": ["导热", "Cu65", "6061"],

    # 拼音缩写
    "bxg": ["不锈钢", "S30408", "S31603"],
    "tsg": ["碳素钢", "Q235B", "45"],
    "hjg": ["合金钢", "40Cr", "42CrMo"],
    "lhj": ["铝合金", "6061", "7075"],
    "thj": ["铜合金", "H62", "QBe2"],
    "zt": ["铸铁", "HT200"],
}


def _calculate_similarity(s1: str, s2: str) -> float:
    """
    计算两个字符串的相似度 (0.0-1.0)
    使用简化的编辑距离算法
    """
    s1_lower = s1.lower()
    s2_lower = s2.lower()

    # 完全匹配
    if s1_lower == s2_lower:
        return 1.0

    # 包含关系
    if s1_lower in s2_lower or s2_lower in s1_lower:
        return 0.8

    # 前缀匹配
    min_len = min(len(s1_lower), len(s2_lower))
    if min_len > 0:
        prefix_match = 0
        for i in range(min_len):
            if s1_lower[i] == s2_lower[i]:
                prefix_match += 1
            else:
                break
        if prefix_match >= 2:
            return 0.5 + 0.3 * (prefix_match / min_len)

    # 字符重叠
    set1 = set(s1_lower)
    set2 = set(s2_lower)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if union > 0:
        return 0.3 * (intersection / union)

    return 0.0


def classify_material_detailed(material: Optional[str]) -> Optional[MaterialInfo]:
    """
    详细材料分类

    Args:
        material: 材料名称字符串

    Returns:
        MaterialInfo 或 None
    """
    if not material:
        return None

    material_clean = material.strip()

    # 1. 精确匹配
    if material_clean in MATERIAL_DATABASE:
        return MATERIAL_DATABASE[material_clean]

    # 2. 大小写不敏感匹配
    material_upper = material_clean.upper()
    for grade, info in MATERIAL_DATABASE.items():
        if grade.upper() == material_upper:
            return info
        if material_upper in [a.upper() for a in info.aliases]:
            return info

    # 3. 正则模式匹配
    for pattern, grade in MATERIAL_MATCH_PATTERNS:
        if re.search(pattern, material_clean, re.IGNORECASE):
            if grade in MATERIAL_DATABASE:
                return MATERIAL_DATABASE[grade]

    # 4. 通用不锈钢识别
    if "不锈钢" in material_clean:
        return MATERIAL_DATABASE.get("S30408")

    logger.debug("Material not found in database: %s", material)
    return None


# 简化分类函数（兼容旧接口）
def classify_material_simple(material: Optional[str]) -> Optional[str]:
    """
    简化材料分类（返回材料组）

    Args:
        material: 材料名称

    Returns:
        材料组名称 或 None
    """
    info = classify_material_detailed(material)
    if info:
        return info.group.value
    return None


def search_materials(
    query: str,
    limit: int = 10,
    category: Optional[str] = None,
    group: Optional[str] = None,
    min_score: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    搜索材料（支持模糊搜索和拼音）

    Args:
        query: 搜索关键词（支持中文、英文、拼音）
        limit: 返回结果数量限制
        category: 限定材料类别 (metal/non_metal/composite)
        group: 限定材料组
        min_score: 最小匹配分数 (0.0-1.0)

    Returns:
        匹配的材料列表，按相关度排序
    """
    if not query or not query.strip():
        return []

    query = query.strip()
    query_lower = query.lower()
    results: List[Tuple[float, str, MaterialInfo, str]] = []

    # 1. 先尝试精确匹配
    exact_match = classify_material_detailed(query)
    if exact_match:
        if category and exact_match.category.value != category:
            pass
        elif group and exact_match.group.value != group:
            pass
        else:
            return [{
                "grade": exact_match.grade,
                "name": exact_match.name,
                "category": exact_match.category.value,
                "group": exact_match.group.value,
                "score": 1.0,
                "match_type": "exact",
            }]

    # 2. 检查拼音映射
    pinyin_matches: List[str] = []
    for pinyin, materials in PINYIN_MAP.items():
        if query_lower == pinyin or query_lower in pinyin or pinyin in query_lower:
            pinyin_matches.extend(materials[1:])  # 跳过描述词

    # 3. 遍历材料数据库进行模糊匹配
    for grade, info in MATERIAL_DATABASE.items():
        # 应用过滤器
        if category and info.category.value != category:
            continue
        if group and info.group.value != group:
            continue

        # 计算匹配分数
        score = 0.0
        match_type = "fuzzy"

        # 拼音匹配加分
        if grade in pinyin_matches:
            score = max(score, 0.85)
            match_type = "pinyin"

        # 牌号匹配
        grade_score = _calculate_similarity(query, grade)
        if grade_score > score:
            score = grade_score
            match_type = "grade"

        # 名称匹配
        name_score = _calculate_similarity(query, info.name)
        if name_score > score:
            score = name_score
            match_type = "name"

        # 别名匹配
        for alias in info.aliases:
            alias_score = _calculate_similarity(query, alias)
            if alias_score > score:
                score = alias_score
                match_type = "alias"

        # 描述匹配
        if info.description and query_lower in info.description.lower():
            score = max(score, 0.6)
            if match_type == "fuzzy":
                match_type = "description"

        # 标准匹配
        for std in info.standards:
            if query_lower in std.lower():
                score = max(score, 0.5)
                if match_type == "fuzzy":
                    match_type = "standard"

        if score >= min_score:
            results.append((score, grade, info, match_type))

    # 按分数排序
    results.sort(key=lambda x: (-x[0], x[1]))

    # 格式化返回结果
    formatted_results = []
    for score, grade, info, match_type in results[:limit]:
        formatted_results.append({
            "grade": grade,
            "name": info.name,
            "category": info.category.value,
            "group": info.group.value,
            "score": round(score, 2),
            "match_type": match_type,
        })

    return formatted_results
