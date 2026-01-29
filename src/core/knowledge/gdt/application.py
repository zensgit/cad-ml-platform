"""
GD&T Application and Interpretation.

Provides guidance on applying GD&T to features and interpreting
feature control frames.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .symbols import GDTCharacteristic, GDTCategory, ToleranceModifier
from .tolerances import ToleranceZone, get_tolerance_zone


class FeatureType(str, Enum):
    """Common feature types for GD&T application."""

    HOLE = "hole"  # 孔
    SHAFT = "shaft"  # 轴
    FLAT_SURFACE = "flat_surface"  # 平面
    CYLINDRICAL_SURFACE = "cylindrical_surface"  # 圆柱面
    SLOT = "slot"  # 槽
    TAB = "tab"  # 凸台
    PATTERN = "pattern"  # 孔组/特征组
    THREAD = "thread"  # 螺纹
    EDGE = "edge"  # 边缘


class InspectionMethod(str, Enum):
    """Inspection methods for geometric tolerances."""

    CMM = "cmm"  # 三坐标测量机
    OPTICAL_COMPARATOR = "optical_comparator"  # 光学投影仪
    GAGE = "gage"  # 检具
    DIAL_INDICATOR = "dial_indicator"  # 百分表
    ROUNDNESS_TESTER = "roundness_tester"  # 圆度仪
    SURFACE_PLATE = "surface_plate"  # 平板
    PROFILOMETER = "profilometer"  # 轮廓仪


@dataclass
class FeatureControlFrame:
    """Parsed feature control frame."""

    characteristic: GDTCharacteristic
    tolerance_value: float
    tolerance_modifier: Optional[ToleranceModifier] = None
    primary_datum: Optional[str] = None
    primary_datum_modifier: Optional[str] = None
    secondary_datum: Optional[str] = None
    secondary_datum_modifier: Optional[str] = None
    tertiary_datum: Optional[str] = None
    tertiary_datum_modifier: Optional[str] = None
    composite: bool = False
    notes: List[str] = None


@dataclass
class GDTApplication:
    """GD&T application recommendation for a feature."""

    feature_type: FeatureType
    recommended_characteristics: List[GDTCharacteristic]
    typical_tolerances: Dict[GDTCharacteristic, Tuple[float, float]]  # (min, max)
    inspection_methods: List[InspectionMethod]
    notes_zh: str = ""
    notes_en: str = ""


# Common GD&T applications by feature type
COMMON_APPLICATIONS: Dict[FeatureType, Dict] = {
    FeatureType.HOLE: {
        "characteristics": [
            GDTCharacteristic.POSITION,
            GDTCharacteristic.PERPENDICULARITY,
            GDTCharacteristic.CYLINDRICITY,
        ],
        "typical_tolerances": {
            GDTCharacteristic.POSITION: (0.1, 0.5),
            GDTCharacteristic.PERPENDICULARITY: (0.02, 0.1),
            GDTCharacteristic.CYLINDRICITY: (0.01, 0.05),
        },
        "inspection": [InspectionMethod.CMM, InspectionMethod.GAGE],
        "notes_zh": "孔通常需要位置度控制，配合MMC修正符可获得额外公差",
        "notes_en": "Holes typically need position control; MMC modifier provides bonus tolerance",
    },
    FeatureType.SHAFT: {
        "characteristics": [
            GDTCharacteristic.CYLINDRICITY,
            GDTCharacteristic.CIRCULAR_RUNOUT,
            GDTCharacteristic.TOTAL_RUNOUT,
            GDTCharacteristic.STRAIGHTNESS,
        ],
        "typical_tolerances": {
            GDTCharacteristic.CYLINDRICITY: (0.01, 0.05),
            GDTCharacteristic.CIRCULAR_RUNOUT: (0.02, 0.1),
            GDTCharacteristic.TOTAL_RUNOUT: (0.03, 0.15),
            GDTCharacteristic.STRAIGHTNESS: (0.02, 0.1),
        },
        "inspection": [InspectionMethod.CMM, InspectionMethod.ROUNDNESS_TESTER, InspectionMethod.DIAL_INDICATOR],
        "notes_zh": "轴类零件重点控制圆度、圆柱度和跳动",
        "notes_en": "Shafts focus on cylindricity and runout control",
    },
    FeatureType.FLAT_SURFACE: {
        "characteristics": [
            GDTCharacteristic.FLATNESS,
            GDTCharacteristic.PARALLELISM,
            GDTCharacteristic.PERPENDICULARITY,
        ],
        "typical_tolerances": {
            GDTCharacteristic.FLATNESS: (0.02, 0.2),
            GDTCharacteristic.PARALLELISM: (0.03, 0.3),
            GDTCharacteristic.PERPENDICULARITY: (0.03, 0.3),
        },
        "inspection": [InspectionMethod.CMM, InspectionMethod.SURFACE_PLATE, InspectionMethod.DIAL_INDICATOR],
        "notes_zh": "平面作为装配基准面时需控制平面度",
        "notes_en": "Flat surfaces used as datum require flatness control",
    },
    FeatureType.PATTERN: {
        "characteristics": [
            GDTCharacteristic.POSITION,
        ],
        "typical_tolerances": {
            GDTCharacteristic.POSITION: (0.15, 1.0),
        },
        "inspection": [InspectionMethod.CMM, InspectionMethod.GAGE],
        "notes_zh": "孔组使用复合位置度控制孔间距和整体位置",
        "notes_en": "Hole patterns use composite position for pattern and feature location",
    },
    FeatureType.CYLINDRICAL_SURFACE: {
        "characteristics": [
            GDTCharacteristic.CIRCULARITY,
            GDTCharacteristic.CYLINDRICITY,
            GDTCharacteristic.CONCENTRICITY,
        ],
        "typical_tolerances": {
            GDTCharacteristic.CIRCULARITY: (0.01, 0.05),
            GDTCharacteristic.CYLINDRICITY: (0.02, 0.08),
            GDTCharacteristic.CONCENTRICITY: (0.02, 0.1),
        },
        "inspection": [InspectionMethod.ROUNDNESS_TESTER, InspectionMethod.CMM],
        "notes_zh": "圆柱面的形状控制，同心度用于质量平衡要求",
        "notes_en": "Cylindrical surface form control; concentricity for mass balance",
    },
    FeatureType.SLOT: {
        "characteristics": [
            GDTCharacteristic.POSITION,
            GDTCharacteristic.SYMMETRY,
            GDTCharacteristic.PARALLELISM,
        ],
        "typical_tolerances": {
            GDTCharacteristic.POSITION: (0.1, 0.5),
            GDTCharacteristic.SYMMETRY: (0.05, 0.2),
            GDTCharacteristic.PARALLELISM: (0.03, 0.15),
        },
        "inspection": [InspectionMethod.CMM, InspectionMethod.GAGE],
        "notes_zh": "槽类特征使用对称度或位置度控制",
        "notes_en": "Slots use symmetry or position control",
    },
}


def get_gdt_for_feature(feature_type: FeatureType) -> GDTApplication:
    """
    Get GD&T recommendations for a feature type.

    Args:
        feature_type: Type of feature

    Returns:
        GDTApplication with recommendations
    """
    data = COMMON_APPLICATIONS.get(feature_type)
    if not data:
        return GDTApplication(
            feature_type=feature_type,
            recommended_characteristics=[],
            typical_tolerances={},
            inspection_methods=[InspectionMethod.CMM],
        )

    return GDTApplication(
        feature_type=feature_type,
        recommended_characteristics=data["characteristics"],
        typical_tolerances=data["typical_tolerances"],
        inspection_methods=data["inspection"],
        notes_zh=data.get("notes_zh", ""),
        notes_en=data.get("notes_en", ""),
    )


def get_inspection_method(
    characteristic: GDTCharacteristic,
    tolerance_value: float,
) -> List[Dict]:
    """
    Get recommended inspection methods for a geometric tolerance.

    Args:
        characteristic: Geometric characteristic
        tolerance_value: Tolerance value in mm

    Returns:
        List of inspection method recommendations
    """
    from .symbols import GDT_SYMBOLS

    symbol_data = GDT_SYMBOLS.get(characteristic, {})
    inspections = symbol_data.get("inspection", ["三坐标测量机"])

    methods = []
    for insp in inspections:
        if tolerance_value < 0.02:
            # Tight tolerance - need precise equipment
            if insp in ["三坐标测量机", "圆度仪"]:
                methods.append({
                    "method": insp,
                    "suitability": "推荐",
                    "reason": "高精度要求",
                })
            else:
                methods.append({
                    "method": insp,
                    "suitability": "不推荐",
                    "reason": "精度可能不足",
                })
        elif tolerance_value < 0.1:
            methods.append({
                "method": insp,
                "suitability": "推荐",
                "reason": "适合中等精度",
            })
        else:
            methods.append({
                "method": insp,
                "suitability": "可用",
                "reason": "粗糙公差",
            })

    return methods


def interpret_feature_control_frame(
    frame_string: str,
) -> Optional[FeatureControlFrame]:
    """
    Parse and interpret a feature control frame string.

    Args:
        frame_string: Feature control frame notation
                     e.g., "⌖ Ø0.2 M A B C" or "位置度 0.2 M A B C"

    Returns:
        Parsed FeatureControlFrame or None

    Example:
        >>> fcf = interpret_feature_control_frame("位置度 0.2 M A B C")
        >>> print(f"Tolerance: {fcf.tolerance_value}mm with datums {fcf.primary_datum}-{fcf.secondary_datum}-{fcf.tertiary_datum}")
    """
    import re

    # Map Chinese names to characteristics
    name_map = {
        "直线度": GDTCharacteristic.STRAIGHTNESS,
        "平面度": GDTCharacteristic.FLATNESS,
        "圆度": GDTCharacteristic.CIRCULARITY,
        "圆柱度": GDTCharacteristic.CYLINDRICITY,
        "垂直度": GDTCharacteristic.PERPENDICULARITY,
        "平行度": GDTCharacteristic.PARALLELISM,
        "倾斜度": GDTCharacteristic.ANGULARITY,
        "位置度": GDTCharacteristic.POSITION,
        "同心度": GDTCharacteristic.CONCENTRICITY,
        "对称度": GDTCharacteristic.SYMMETRY,
        "圆跳动": GDTCharacteristic.CIRCULAR_RUNOUT,
        "全跳动": GDTCharacteristic.TOTAL_RUNOUT,
        "线轮廓度": GDTCharacteristic.PROFILE_LINE,
        "面轮廓度": GDTCharacteristic.PROFILE_SURFACE,
    }

    # Try to match characteristic
    characteristic = None
    for name, char in name_map.items():
        if name in frame_string:
            characteristic = char
            break

    if not characteristic:
        return None

    # Extract tolerance value
    value_match = re.search(r"[Ø⌀]?(\d+\.?\d*)", frame_string)
    tolerance_value = float(value_match.group(1)) if value_match else 0.0

    # Extract modifier
    tolerance_modifier = None
    if "M" in frame_string.upper():
        tolerance_modifier = ToleranceModifier.MMC
    elif "L" in frame_string.upper():
        tolerance_modifier = ToleranceModifier.LMC

    # Extract datums
    datum_match = re.findall(r"\b([A-Z])\b", frame_string)
    datums = [d for d in datum_match if d not in ["M", "L", "S", "P"]]

    return FeatureControlFrame(
        characteristic=characteristic,
        tolerance_value=tolerance_value,
        tolerance_modifier=tolerance_modifier,
        primary_datum=datums[0] if len(datums) > 0 else None,
        secondary_datum=datums[1] if len(datums) > 1 else None,
        tertiary_datum=datums[2] if len(datums) > 2 else None,
        notes=[],
    )


def generate_feature_control_frame(
    characteristic: GDTCharacteristic,
    tolerance: float,
    datums: List[str] = None,
    modifier: Optional[ToleranceModifier] = None,
    diameter_zone: bool = False,
) -> str:
    """
    Generate a feature control frame string.

    Args:
        characteristic: Geometric characteristic
        tolerance: Tolerance value
        datums: List of datum labels
        modifier: Tolerance modifier
        diameter_zone: Whether to use diameter symbol

    Returns:
        Feature control frame string
    """
    from .symbols import GDT_SYMBOLS

    symbol_data = GDT_SYMBOLS.get(characteristic, {})
    symbol = symbol_data.get("symbol", "?")

    parts = [symbol]

    if diameter_zone:
        parts.append(f"Ø{tolerance}")
    else:
        parts.append(str(tolerance))

    if modifier:
        parts.append(modifier.value)

    if datums:
        parts.extend(datums)

    return " ".join(parts)


def get_gdt_rule_one_guidance() -> Dict:
    """
    Get guidance on Rule #1 (Envelope Principle / Taylor Principle).

    Returns:
        Explanation of Rule #1
    """
    return {
        "rule_number": 1,
        "name_en": "Envelope Principle (Rule #1)",
        "name_zh": "包容原则 (规则一)",
        "standard": "ASME Y14.5-2018",
        "principle": "个体特征在最大实体条件时必须有完美形状",
        "explanation_zh": (
            "当尺寸公差应用于单个规则特征（如圆柱或平行平面）时，"
            "该特征在最大实体条件下的形状误差不得超出最大实体边界。"
            "即：MMC尺寸 = 完美形状边界"
        ),
        "explanation_en": (
            "Where a tolerance of size is applied to an individual regular feature of size, "
            "the surface of the feature shall not extend beyond the envelope of perfect form "
            "at Maximum Material Condition (MMC)."
        ),
        "implications": [
            "形状公差隐含在尺寸公差内",
            "无需额外标注形状公差控制MMC边界",
            "LMC时形状可以偏离完美形状",
        ],
        "exceptions": [
            "标注独立原则符号时不适用",
            "非刚性零件（自由状态）",
            "螺纹、花键等不适用",
        ],
    }
