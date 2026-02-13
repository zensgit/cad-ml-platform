"""Label normalization utilities.

This module centralizes the mapping from fine-grained part names to coarse
"bucket" labels used by some Graph2D training/evaluation flows.
"""

from __future__ import annotations

from typing import Optional

# Fine label -> coarse bucket label.
DXF_LABEL_BUCKET_MAP: dict[str, str] = {
    "罐体部分": "罐体",
    "上封头组件": "罐体",
    "下锥体组件": "罐体",
    "上筒体组件": "罐体",
    "下筒体组件": "罐体",
    "再沸器": "设备",
    "汽水分离器": "设备",
    "自动进料装置": "设备",
    "电加热箱": "设备",
    "真空组件": "设备",
    "出料正压隔离器": "设备",
    "拖车": "设备",
    "管束": "设备",
    "阀体": "设备",
    "蜗轮蜗杆传动出料机构": "传动件",
    "旋转组件": "传动件",
    "轴头组件": "传动件",
    "搅拌桨组件": "传动件",
    "搅拌轴组件": "传动件",
    "搅拌器组件": "传动件",
    "手轮组件": "传动件",
    "拖轮组件": "传动件",
    "液压开盖组件": "传动件",
    "侧推料组件": "传动件",
    "轴向定位轴承": "轴承件",
    "轴承座": "轴承件",
    "下轴承支架组件": "轴承件",
    "短轴承座(盖)": "轴承件",
    "支承座": "轴承件",
    "超声波法兰": "法兰",
    "出料凸缘": "法兰",
    "对接法兰": "法兰",
    "人孔法兰": "法兰",
    "连接法兰(大)": "法兰",
    "保护罩组件": "罩盖件",
    "搅拌减速机机罩": "罩盖件",
    "防爆视灯组件": "罩盖件",
    "下封板": "罩盖件",
    "过滤托架": "过滤组件",
    "过滤芯组件": "过滤组件",
    "捕集器组件": "过滤组件",
    "捕集口": "开孔件",
    "人孔": "开孔件",
    "罐体支腿": "支撑件",
    "底板": "支撑件",
    "调节螺栓": "紧固件",
    "扭转弹簧": "弹簧",
}


def normalize_dxf_label(label: str, *, default: Optional[str] = None) -> str:
    """Normalize a DXF label to a coarse bucket label when applicable.

    Args:
        label: Input label (Chinese part name).
        default: If provided and ``label`` is not in the mapping, return this
            fallback value instead of the original label.

    Returns:
        Normalized label (bucket), the original cleaned label, or ``default``.
    """
    cleaned = str(label or "").strip()
    if not cleaned:
        return str(default or "")

    mapped = DXF_LABEL_BUCKET_MAP.get(cleaned)
    if mapped is not None:
        return mapped

    if default is not None:
        return str(default)

    return cleaned
