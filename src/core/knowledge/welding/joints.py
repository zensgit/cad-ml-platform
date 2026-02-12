"""
Welding Joint Design Knowledge Base.

Provides joint preparation geometry, groove dimensions, and design
recommendations for common welding joint types.

Reference:
- AWS D1.1 - Structural Welding Code
- ISO 9692-1 - Welding joint preparation
- GB/T 985.1 - Recommended joint preparation
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class GrooveType(str, Enum):
    """Groove weld types."""

    SQUARE = "square"  # 方形坡口 (I型)
    SINGLE_V = "single_v"  # 单边V形坡口
    DOUBLE_V = "double_v"  # 双边V形坡口 (X型)
    SINGLE_BEVEL = "single_bevel"  # 单边斜坡口
    DOUBLE_BEVEL = "double_bevel"  # 双边斜坡口 (K型)
    SINGLE_U = "single_u"  # 单边U形坡口
    DOUBLE_U = "double_u"  # 双边U形坡口
    SINGLE_J = "single_j"  # 单边J形坡口
    DOUBLE_J = "double_j"  # 双边J形坡口


@dataclass
class JointDesign:
    """Weld joint design parameters."""

    groove_type: GrooveType
    thickness_range: Tuple[float, float]  # mm

    # Groove geometry
    groove_angle: Optional[float] = None  # degrees (total)
    root_face: Optional[float] = None  # mm
    root_gap: Optional[float] = None  # mm
    bevel_angle: Optional[float] = None  # degrees (per side)

    # For U/J grooves
    root_radius: Optional[float] = None  # mm

    # Backing requirements
    backing_required: bool = False
    backing_type: Optional[str] = None  # "ceramic", "steel", "copper"

    # Accessibility
    single_side_access: bool = True

    # Reference
    aws_designation: Optional[str] = None
    iso_designation: Optional[str] = None

    notes_zh: str = ""
    notes_en: str = ""


# Joint design database
# Format: {(groove_type, thickness_range): design_params}
JOINT_DESIGN_DATABASE: Dict[Tuple[GrooveType, Tuple[int, int]], Dict[str, Any]] = {
    # Square butt (I-joint)
    (GrooveType.SQUARE, (1, 3)): {
        "root_gap": 0,
        "root_face": None,
        "backing_required": False,
        "notes_zh": "薄板对接，无需开坡口",
        "notes_en": "Thin plate butt joint, no groove required",
        "aws_designation": "B-P1a",
    },
    (GrooveType.SQUARE, (3, 6)): {
        "root_gap": (0, 2),
        "root_face": None,
        "backing_required": False,
        "notes_zh": "中厚板对接，需适当间隙确保熔透",
        "notes_en": "Medium plate butt joint, gap needed for penetration",
        "aws_designation": "B-P1b",
    },
    # Single-V groove
    (GrooveType.SINGLE_V, (6, 12)): {
        "groove_angle": 60,
        "bevel_angle": 30,
        "root_gap": (2, 3),
        "root_face": (1, 2),
        "backing_required": False,
        "notes_zh": "单边V形坡口，适用于单面焊接",
        "notes_en": "Single-V groove for single-side welding",
        "aws_designation": "B-U2",
    },
    (GrooveType.SINGLE_V, (12, 20)): {
        "groove_angle": 60,
        "bevel_angle": 30,
        "root_gap": (2, 4),
        "root_face": (2, 3),
        "backing_required": True,
        "backing_type": "ceramic or steel",
        "notes_zh": "厚板单边V形坡口，建议使用垫板",
        "notes_en": "Single-V groove with backing for thick plates",
        "aws_designation": "B-U2a",
    },
    # Double-V groove (X-joint)
    (GrooveType.DOUBLE_V, (12, 25)): {
        "groove_angle": 60,
        "bevel_angle": 30,
        "root_gap": (0, 2),
        "root_face": (2, 4),
        "backing_required": False,
        "notes_zh": "双边V形坡口(X型)，双面可焊时优选",
        "notes_en": "Double-V groove (X-type), preferred for two-side access",
        "aws_designation": "B-U3",
    },
    (GrooveType.DOUBLE_V, (25, 50)): {
        "groove_angle": 60,
        "bevel_angle": 30,
        "root_gap": (0, 3),
        "root_face": (3, 5),
        "backing_required": False,
        "notes_zh": "厚板双边V形坡口，减少焊接变形",
        "notes_en": "Double-V for thick plates, reduces distortion",
        "aws_designation": "B-U3a",
    },
    # Single bevel
    (GrooveType.SINGLE_BEVEL, (6, 15)): {
        "groove_angle": 45,
        "bevel_angle": 45,
        "root_gap": (2, 3),
        "root_face": (1, 2),
        "backing_required": False,
        "notes_zh": "单边斜坡口，常用于T形接头",
        "notes_en": "Single bevel, commonly used for T-joints",
        "aws_designation": "TC-U4a",
    },
    # Double bevel (K-joint)
    (GrooveType.DOUBLE_BEVEL, (12, 30)): {
        "groove_angle": 45,
        "bevel_angle": 22.5,
        "root_gap": (0, 2),
        "root_face": (2, 3),
        "backing_required": False,
        "notes_zh": "双边斜坡口(K型)，T形接头双面焊接",
        "notes_en": "Double bevel (K-type) for T-joint two-side welding",
        "aws_designation": "TC-U4b",
    },
    # Single-U groove
    (GrooveType.SINGLE_U, (20, 40)): {
        "groove_angle": 20,
        "bevel_angle": 10,
        "root_gap": (0, 2),
        "root_face": (2, 3),
        "root_radius": 6,
        "backing_required": False,
        "notes_zh": "U形坡口，焊接量比V形坡口少",
        "notes_en": "U-groove requires less weld metal than V-groove",
        "aws_designation": "B-U6",
    },
    # Double-U groove
    (GrooveType.DOUBLE_U, (30, 60)): {
        "groove_angle": 20,
        "bevel_angle": 10,
        "root_gap": (0, 2),
        "root_face": (3, 5),
        "root_radius": 6,
        "backing_required": False,
        "notes_zh": "双U形坡口，特厚板焊接",
        "notes_en": "Double-U groove for very thick plates",
        "aws_designation": "B-U7",
    },
    # Single-J groove
    (GrooveType.SINGLE_J, (15, 30)): {
        "groove_angle": 35,
        "bevel_angle": 35,
        "root_gap": (0, 2),
        "root_face": (2, 3),
        "root_radius": 8,
        "backing_required": False,
        "notes_zh": "J形坡口，T形接头厚板",
        "notes_en": "J-groove for thick T-joints",
        "aws_designation": "TC-U5",
    },
}

# Fillet weld design
FILLET_WELD_SIZES: Dict[str, Dict[str, Any]] = {
    # Minimum fillet weld size based on thicker plate (AWS D1.1)
    "min_size": {
        (0, 6): 3,
        (6, 13): 5,
        (13, 19): 6,
        (19, 38): 8,
        (38, 57): 10,
        (57, 150): 13,
    },
    # Maximum fillet weld size
    "max_size_rules": {
        "at_edge": "thickness - 1.5mm",
        "not_at_edge": "1.4 * throat required",
    },
}


def get_joint_design(
    groove_type: Union[str, GrooveType],
    thickness: float,
) -> Optional[JointDesign]:
    """
    Get joint design parameters.

    Args:
        groove_type: Groove type
        thickness: Material thickness (mm)

    Returns:
        JointDesign or None

    Example:
        >>> design = get_joint_design("single_v", 10)
        >>> print(f"Groove angle: {design.groove_angle}°")
    """
    if isinstance(groove_type, str):
        try:
            groove_type = GrooveType(groove_type.lower())
        except ValueError:
            return None

    for (g_type, thickness_range), data in JOINT_DESIGN_DATABASE.items():
        if g_type == groove_type:
            if thickness_range[0] <= thickness <= thickness_range[1]:
                root_gap = data.get("root_gap")
                root_face = data.get("root_face")

                # Handle tuple values (take middle)
                if isinstance(root_gap, tuple):
                    root_gap = sum(root_gap) / 2
                if isinstance(root_face, tuple):
                    root_face = sum(root_face) / 2

                return JointDesign(
                    groove_type=groove_type,
                    thickness_range=thickness_range,
                    groove_angle=data.get("groove_angle"),
                    bevel_angle=data.get("bevel_angle"),
                    root_gap=root_gap,
                    root_face=root_face,
                    root_radius=data.get("root_radius"),
                    backing_required=data.get("backing_required", False),
                    backing_type=data.get("backing_type"),
                    aws_designation=data.get("aws_designation"),
                    notes_zh=data.get("notes_zh", ""),
                    notes_en=data.get("notes_en", ""),
                )

    return None


def recommend_joint_for_thickness(
    thickness: float,
    access_both_sides: bool = True,
    minimize_distortion: bool = False,
) -> List[Tuple[GrooveType, str]]:
    """
    Recommend suitable joint types for given thickness.

    Args:
        thickness: Material thickness (mm)
        access_both_sides: Whether both sides are accessible
        minimize_distortion: Prioritize joints that minimize distortion

    Returns:
        List of (groove_type, reason) tuples, best first
    """
    recommendations = []

    if thickness <= 3:
        recommendations.append((GrooveType.SQUARE, "薄板，无需开坡口"))
    elif thickness <= 6:
        recommendations.append((GrooveType.SQUARE, "可用间隙确保熔透"))
        recommendations.append((GrooveType.SINGLE_V, "可选V形坡口"))
    elif thickness <= 12:
        if access_both_sides and minimize_distortion:
            recommendations.append((GrooveType.DOUBLE_V, "双面焊减少变形"))
        recommendations.append((GrooveType.SINGLE_V, "单面焊接适用"))
    elif thickness <= 25:
        if access_both_sides:
            recommendations.append((GrooveType.DOUBLE_V, "双面焊减少变形和焊接量"))
            if minimize_distortion:
                recommendations.append((GrooveType.SINGLE_U, "U形坡口焊接量更少"))
        else:
            recommendations.append((GrooveType.SINGLE_V, "单面可达时使用"))
    elif thickness <= 50:
        if access_both_sides:
            recommendations.append((GrooveType.DOUBLE_U, "特厚板优选双U形"))
            recommendations.append((GrooveType.DOUBLE_V, "次选双V形"))
        else:
            recommendations.append((GrooveType.SINGLE_U, "单面U形坡口"))
    else:
        if access_both_sides:
            recommendations.append((GrooveType.DOUBLE_U, "超厚板必须双U形"))
        else:
            recommendations.append((GrooveType.SINGLE_U, "单面需特殊焊接工艺"))

    return recommendations


def get_minimum_fillet_size(
    thickness: float,
) -> float:
    """
    Get minimum fillet weld size per AWS D1.1.

    Args:
        thickness: Thickness of thicker plate (mm)

    Returns:
        Minimum fillet leg size (mm)
    """
    for (t_min, t_max), size in FILLET_WELD_SIZES["min_size"].items():
        if t_min < thickness <= t_max:
            return size
    return 13  # Maximum for very thick plates


def calculate_groove_volume(
    thickness: float,
    groove_type: GrooveType,
    weld_length: float = 1000,
) -> float:
    """
    Calculate approximate groove volume for weld metal estimation.

    Args:
        thickness: Material thickness (mm)
        groove_type: Groove type
        weld_length: Length of weld (mm)

    Returns:
        Volume in cm³
    """
    design = get_joint_design(groove_type, thickness)
    if not design:
        return 0.0

    # Simplified cross-section area calculations
    if groove_type == GrooveType.SQUARE:
        gap = design.root_gap or 0
        area = thickness * gap
    elif groove_type in [GrooveType.SINGLE_V, GrooveType.SINGLE_BEVEL]:
        angle_rad = (design.groove_angle or 60) * 3.14159 / 180
        root_face = design.root_face or 0
        height = thickness - root_face
        area = height * height * 0.5 * 2 * (angle_rad / 2)  # Approximate triangle
    elif groove_type in [GrooveType.DOUBLE_V, GrooveType.DOUBLE_BEVEL]:
        angle_rad = (design.groove_angle or 60) * 3.14159 / 180
        root_face = design.root_face or 0
        height = (thickness - root_face) / 2
        area = 2 * (height * height * 0.5 * 2 * (angle_rad / 2))
    else:
        # U/J grooves - rough approximation
        area = thickness * 5  # Very rough

    volume = area * weld_length / 1000  # cm³
    return round(volume, 1)
