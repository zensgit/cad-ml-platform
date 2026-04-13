"""
材料数据导出功能

提供材料数据库和等价表的 CSV 导出功能。
"""

from __future__ import annotations

import logging
from typing import Optional

from src.core.materials.data_models import MATERIAL_DATABASE, MATERIAL_EQUIVALENCE

logger = logging.getLogger(__name__)


def export_materials_csv(filepath: Optional[str] = None) -> str:
    """
    导出材料数据库为 CSV 格式

    Args:
        filepath: 可选的文件路径，如果提供则写入文件

    Returns:
        CSV 格式字符串
    """
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # 写入表头
    headers = [
        "牌号", "名称", "别名", "类别", "子类", "材料组",
        "密度(g/cm³)", "抗拉强度(MPa)", "屈服强度(MPa)", "硬度",
        "可加工性", "可焊性", "毛坯形式", "需要专用刀具", "需要冷却液",
        "推荐热处理", "禁止热处理", "推荐表面处理", "切削速度(m/min)",
        "警告", "建议", "描述"
    ]
    writer.writerow(headers)

    # 按牌号排序
    for grade in sorted(MATERIAL_DATABASE.keys()):
        info = MATERIAL_DATABASE[grade]
        props = info.properties
        proc = info.process

        row = [
            info.grade,
            info.name,
            "|".join(info.aliases) if info.aliases else "",
            info.category.value,
            info.sub_category.value,
            info.group.value,
            props.density if props.density else "",
            props.tensile_strength if props.tensile_strength else "",
            props.yield_strength if props.yield_strength else "",
            props.hardness if props.hardness else "",
            props.machinability if props.machinability else "",
            props.weldability if props.weldability else "",
            "|".join(proc.blank_forms) if proc.blank_forms else "",
            "是" if proc.special_tooling else "否",
            "是" if proc.coolant_required else "否",
            "|".join(proc.heat_treatments) if proc.heat_treatments else "",
            "|".join(proc.forbidden_heat_treatments) if proc.forbidden_heat_treatments else "",
            "|".join(proc.surface_treatments) if proc.surface_treatments else "",
            f"{proc.cutting_speed_range[0]}-{proc.cutting_speed_range[1]}" if proc.cutting_speed_range else "",
            "|".join(proc.warnings) if proc.warnings else "",
            "|".join(proc.recommendations) if proc.recommendations else "",
            info.description,
        ]
        writer.writerow(row)

    csv_content = output.getvalue()
    output.close()

    # 如果提供了文件路径，写入文件
    if filepath:
        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            f.write(csv_content)
        logger.info("Materials exported to %s", filepath)

    return csv_content


def export_equivalence_csv(filepath: Optional[str] = None) -> str:
    """
    导出材料等价表为 CSV 格式

    Args:
        filepath: 可选的文件路径，如果提供则写入文件

    Returns:
        CSV 格式字符串
    """
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # 写入表头
    headers = ["牌号", "名称", "中国(CN)", "美国(US)", "日本(JP)", "德国(DE)", "UNS"]
    writer.writerow(headers)

    # 按牌号排序
    for grade in sorted(MATERIAL_EQUIVALENCE.keys()):
        equiv = MATERIAL_EQUIVALENCE[grade]
        row = [
            grade,
            equiv.get("name", ""),
            equiv.get("CN", ""),
            equiv.get("US", ""),
            equiv.get("JP", ""),
            equiv.get("DE", ""),
            equiv.get("UNS", ""),
        ]
        writer.writerow(row)

    csv_content = output.getvalue()
    output.close()

    # 如果提供了文件路径，写入文件
    if filepath:
        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            f.write(csv_content)
        logger.info("Material equivalence exported to %s", filepath)

    return csv_content
