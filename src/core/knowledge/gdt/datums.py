"""
Datum Reference Frames for GD&T.

Provides datum feature definitions, datum reference frame construction,
and datum priority rules per ISO 5459 and ASME Y14.5.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class DatumFeatureType(str, Enum):
    """Types of datum features."""

    PLANE = "plane"  # 平面
    AXIS = "axis"  # 轴线
    CENTER_PLANE = "center_plane"  # 中心平面
    POINT = "point"  # 点


class DatumPriority(str, Enum):
    """Datum priority in reference frame."""

    PRIMARY = "primary"  # 主基准 (A)
    SECONDARY = "secondary"  # 次基准 (B)
    TERTIARY = "tertiary"  # 第三基准 (C)


@dataclass
class DatumFeature:
    """Definition of a datum feature."""

    label: str  # A, B, C, etc.
    feature_type: DatumFeatureType
    priority: Optional[DatumPriority] = None
    modifier: str = ""  # M, L, or empty
    description_zh: str = ""
    description_en: str = ""
    degrees_of_freedom_constrained: int = 0


@dataclass
class DatumReferenceFrame:
    """
    Complete datum reference frame (DRF).

    A DRF constrains all 6 degrees of freedom:
    - 3 translations (X, Y, Z)
    - 3 rotations (Rx, Ry, Rz)
    """

    primary: Optional[DatumFeature] = None
    secondary: Optional[DatumFeature] = None
    tertiary: Optional[DatumFeature] = None
    total_dof_constrained: int = 0
    is_fully_constrained: bool = False
    notes: List[str] = field(default_factory=list)


# Datum feature type properties
DATUM_FEATURE_TYPES: Dict[DatumFeatureType, Dict] = {
    DatumFeatureType.PLANE: {
        "name_zh": "平面基准",
        "name_en": "Planar Datum",
        "dof_as_primary": 3,  # 1 translation + 2 rotations
        "dof_as_secondary": 2,  # 1 translation + 1 rotation
        "dof_as_tertiary": 1,  # 1 translation
        "typical_features": ["端面", "底面", "侧面", "法兰面"],
        "establishment": "三点接触确定平面",
    },
    DatumFeatureType.AXIS: {
        "name_zh": "轴线基准",
        "name_en": "Axis Datum",
        "dof_as_primary": 4,  # 2 translations + 2 rotations
        "dof_as_secondary": 2,  # 1 translation + 1 rotation
        "dof_as_tertiary": 1,  # 1 rotation
        "typical_features": ["轴外圆", "孔内圆", "锥面轴线"],
        "establishment": "圆柱面的理想轴线",
    },
    DatumFeatureType.CENTER_PLANE: {
        "name_zh": "中心平面基准",
        "name_en": "Center Plane Datum",
        "dof_as_primary": 3,
        "dof_as_secondary": 2,
        "dof_as_tertiary": 1,
        "typical_features": ["键槽", "对称槽", "矩形凸台"],
        "establishment": "两对称面的中心平面",
    },
    DatumFeatureType.POINT: {
        "name_zh": "点基准",
        "name_en": "Point Datum",
        "dof_as_primary": 3,  # Only translations
        "dof_as_secondary": 2,
        "dof_as_tertiary": 1,
        "typical_features": ["球心", "锥顶点", "交点"],
        "establishment": "球面或锥面的理论中心点",
    },
}


def get_datum_priority(
    feature_type: DatumFeatureType,
    priority: DatumPriority,
) -> Dict[str, Any]:
    """
    Get degrees of freedom constrained by a datum feature at given priority.

    Args:
        feature_type: Type of datum feature
        priority: Priority level (primary, secondary, tertiary)

    Returns:
        Dict with DOF information
    """
    type_data = DATUM_FEATURE_TYPES.get(feature_type)
    if not type_data:
        return {"dof": 0, "error": "Unknown feature type"}

    dof_key = f"dof_as_{priority.value}"
    dof = type_data.get(dof_key, 0)

    return {
        "feature_type": feature_type.value,
        "priority": priority.value,
        "dof_constrained": dof,
        "name_zh": type_data["name_zh"],
        "establishment": type_data["establishment"],
    }


def create_datum_reference_frame(
    primary: Tuple[str, DatumFeatureType],
    secondary: Optional[Tuple[str, DatumFeatureType]] = None,
    tertiary: Optional[Tuple[str, DatumFeatureType]] = None,
) -> DatumReferenceFrame:
    """
    Create a datum reference frame from datum features.

    Args:
        primary: (label, feature_type) for primary datum
        secondary: Optional (label, feature_type) for secondary datum
        tertiary: Optional (label, feature_type) for tertiary datum

    Returns:
        DatumReferenceFrame with DOF analysis

    Example:
        >>> drf = create_datum_reference_frame(
        ...     primary=("A", DatumFeatureType.PLANE),
        ...     secondary=("B", DatumFeatureType.AXIS),
        ...     tertiary=("C", DatumFeatureType.PLANE),
        ... )
        >>> print(f"Fully constrained: {drf.is_fully_constrained}")
    """
    notes = []
    total_dof = 0

    # Primary datum
    primary_type = DATUM_FEATURE_TYPES.get(primary[1])
    primary_dof = primary_type["dof_as_primary"] if primary_type else 0
    total_dof += primary_dof

    primary_datum = DatumFeature(
        label=primary[0],
        feature_type=primary[1],
        priority=DatumPriority.PRIMARY,
        degrees_of_freedom_constrained=primary_dof,
        description_zh=f"主基准 {primary[0]}: {primary_type['name_zh'] if primary_type else ''}",
    )
    notes.append(f"主基准 {primary[0]} 约束 {primary_dof} 个自由度")

    # Secondary datum
    secondary_datum = None
    if secondary:
        secondary_type = DATUM_FEATURE_TYPES.get(secondary[1])
        secondary_dof = secondary_type["dof_as_secondary"] if secondary_type else 0
        total_dof += secondary_dof

        secondary_datum = DatumFeature(
            label=secondary[0],
            feature_type=secondary[1],
            priority=DatumPriority.SECONDARY,
            degrees_of_freedom_constrained=secondary_dof,
            description_zh=f"次基准 {secondary[0]}: {secondary_type['name_zh'] if secondary_type else ''}",
        )
        notes.append(f"次基准 {secondary[0]} 约束 {secondary_dof} 个自由度")

    # Tertiary datum
    tertiary_datum = None
    if tertiary:
        tertiary_type = DATUM_FEATURE_TYPES.get(tertiary[1])
        tertiary_dof = tertiary_type["dof_as_tertiary"] if tertiary_type else 0
        total_dof += tertiary_dof

        tertiary_datum = DatumFeature(
            label=tertiary[0],
            feature_type=tertiary[1],
            priority=DatumPriority.TERTIARY,
            degrees_of_freedom_constrained=tertiary_dof,
            description_zh=f"第三基准 {tertiary[0]}: {tertiary_type['name_zh'] if tertiary_type else ''}",
        )
        notes.append(f"第三基准 {tertiary[0]} 约束 {tertiary_dof} 个自由度")

    is_fully_constrained = total_dof >= 6
    if is_fully_constrained:
        notes.append("基准系统完全约束 (6个自由度)")
    else:
        notes.append(f"基准系统约束 {total_dof} 个自由度，剩余 {6 - total_dof} 个自由度未约束")

    return DatumReferenceFrame(
        primary=primary_datum,
        secondary=secondary_datum,
        tertiary=tertiary_datum,
        total_dof_constrained=total_dof,
        is_fully_constrained=is_fully_constrained,
        notes=notes,
    )


def get_common_datum_schemes() -> List[Dict[str, Any]]:
    """
    Get common datum reference frame schemes.

    Returns:
        List of common DRF configurations
    """
    return [
        {
            "name": "3-2-1定位",
            "description": "平面-轴线-平面组合，最常用",
            "primary": ("A", DatumFeatureType.PLANE),
            "secondary": ("B", DatumFeatureType.AXIS),
            "tertiary": ("C", DatumFeatureType.PLANE),
            "applications": ["箱体类零件", "支架类零件"],
        },
        {
            "name": "轴类定位",
            "description": "轴线-平面组合，适用于回转体",
            "primary": ("A", DatumFeatureType.AXIS),
            "secondary": ("B", DatumFeatureType.PLANE),
            "tertiary": None,
            "applications": ["轴类零件", "套筒类零件"],
        },
        {
            "name": "平面定位",
            "description": "三个正交平面，板类零件",
            "primary": ("A", DatumFeatureType.PLANE),
            "secondary": ("B", DatumFeatureType.PLANE),
            "tertiary": ("C", DatumFeatureType.PLANE),
            "applications": ["板类零件", "法兰"],
        },
    ]


def validate_datum_sequence(datums: List[str]) -> Dict[str, Any]:
    """
    Validate datum sequence in a feature control frame.

    Args:
        datums: List of datum labels in order, e.g., ["A", "B", "C"]

    Returns:
        Validation result
    """
    result = {
        "is_valid": True,
        "datums": datums,
        "issues": [],
    }

    # Check for duplicates
    if len(datums) != len(set(datums)):
        result["is_valid"] = False
        result["issues"].append("基准标签重复")

    # Check order (should be alphabetical or meaningful)
    if len(datums) > 1:
        for i in range(len(datums) - 1):
            if ord(datums[i]) > ord(datums[i + 1]):
                result["issues"].append(f"基准顺序可能不符合惯例: {datums[i]} 在 {datums[i + 1]} 之前")

    return result
