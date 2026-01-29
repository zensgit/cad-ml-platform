"""
Geometric Tolerance Values and Zones.

Provides tolerance value recommendations and zone calculations
per ISO 2768-2 and common engineering practices.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from .symbols import GDTCharacteristic, GDTCategory


class ToleranceGrade(str, Enum):
    """Geometric tolerance grades (similar to ISO 2768-2)."""

    H = "H"  # Fine 精密级
    K = "K"  # Medium 中等级
    L = "L"  # Coarse 粗糙级


class ToleranceZoneShape(str, Enum):
    """Shapes of tolerance zones."""

    CYLINDRICAL = "cylindrical"  # 圆柱形
    SPHERICAL = "spherical"  # 球形
    BETWEEN_PLANES = "between_planes"  # 两平行平面之间
    BETWEEN_LINES = "between_lines"  # 两平行直线之间
    ANNULAR = "annular"  # 圆环形
    BETWEEN_COAXIAL_CYLINDERS = "between_coaxial_cylinders"  # 两同轴圆柱面之间


@dataclass
class ToleranceZone:
    """Definition of a geometric tolerance zone."""

    shape: ToleranceZoneShape
    value: float  # mm
    diameter_symbol: bool = False  # Whether preceded by Ø
    description_zh: str = ""
    description_en: str = ""


@dataclass
class GeometricTolerance:
    """Complete geometric tolerance specification."""

    characteristic: GDTCharacteristic
    value: float  # mm
    zone: ToleranceZone
    modifier: Optional[str] = None  # M, L, etc.
    datums: List[str] = None  # e.g., ["A", "B", "C"]
    datum_modifiers: Dict[str, str] = None  # e.g., {"A": "M"}


# Recommended tolerance values based on nominal size
# Format: {grade: {(min_size, max_size): tolerance_mm}}
TOLERANCE_RECOMMENDATIONS: Dict[GDTCharacteristic, Dict] = {
    GDTCharacteristic.FLATNESS: {
        ToleranceGrade.H: {
            (0, 10): 0.02,
            (10, 30): 0.05,
            (30, 100): 0.1,
            (100, 300): 0.2,
            (300, 1000): 0.3,
            (1000, 3000): 0.5,
        },
        ToleranceGrade.K: {
            (0, 10): 0.05,
            (10, 30): 0.1,
            (30, 100): 0.2,
            (100, 300): 0.4,
            (300, 1000): 0.6,
            (1000, 3000): 1.0,
        },
        ToleranceGrade.L: {
            (0, 10): 0.1,
            (10, 30): 0.2,
            (30, 100): 0.4,
            (100, 300): 0.8,
            (300, 1000): 1.2,
            (1000, 3000): 2.0,
        },
    },
    GDTCharacteristic.STRAIGHTNESS: {
        ToleranceGrade.H: {
            (0, 10): 0.02,
            (10, 30): 0.05,
            (30, 100): 0.1,
            (100, 300): 0.15,
            (300, 1000): 0.2,
            (1000, 3000): 0.3,
        },
        ToleranceGrade.K: {
            (0, 10): 0.05,
            (10, 30): 0.1,
            (30, 100): 0.2,
            (100, 300): 0.3,
            (300, 1000): 0.4,
            (1000, 3000): 0.6,
        },
        ToleranceGrade.L: {
            (0, 10): 0.1,
            (10, 30): 0.2,
            (30, 100): 0.4,
            (100, 300): 0.6,
            (300, 1000): 0.8,
            (1000, 3000): 1.2,
        },
    },
    GDTCharacteristic.CIRCULARITY: {
        ToleranceGrade.H: {
            (0, 10): 0.02,
            (10, 30): 0.03,
            (30, 100): 0.04,
            (100, 300): 0.05,
            (300, 1000): 0.06,
        },
        ToleranceGrade.K: {
            (0, 10): 0.04,
            (10, 30): 0.06,
            (30, 100): 0.08,
            (100, 300): 0.1,
            (300, 1000): 0.12,
        },
        ToleranceGrade.L: {
            (0, 10): 0.1,
            (10, 30): 0.12,
            (30, 100): 0.15,
            (100, 300): 0.2,
            (300, 1000): 0.25,
        },
    },
    GDTCharacteristic.CYLINDRICITY: {
        ToleranceGrade.H: {
            (0, 10): 0.025,
            (10, 30): 0.04,
            (30, 100): 0.05,
            (100, 300): 0.06,
            (300, 1000): 0.08,
        },
        ToleranceGrade.K: {
            (0, 10): 0.05,
            (10, 30): 0.08,
            (30, 100): 0.1,
            (100, 300): 0.12,
            (300, 1000): 0.16,
        },
        ToleranceGrade.L: {
            (0, 10): 0.12,
            (10, 30): 0.16,
            (30, 100): 0.2,
            (100, 300): 0.25,
            (300, 1000): 0.32,
        },
    },
    GDTCharacteristic.PERPENDICULARITY: {
        ToleranceGrade.H: {
            (0, 100): 0.05,
            (100, 300): 0.1,
            (300, 1000): 0.15,
            (1000, 3000): 0.2,
        },
        ToleranceGrade.K: {
            (0, 100): 0.1,
            (100, 300): 0.2,
            (300, 1000): 0.3,
            (1000, 3000): 0.4,
        },
        ToleranceGrade.L: {
            (0, 100): 0.2,
            (100, 300): 0.4,
            (300, 1000): 0.6,
            (1000, 3000): 0.8,
        },
    },
    GDTCharacteristic.PARALLELISM: {
        ToleranceGrade.H: {
            (0, 100): 0.05,
            (100, 300): 0.1,
            (300, 1000): 0.15,
            (1000, 3000): 0.2,
        },
        ToleranceGrade.K: {
            (0, 100): 0.1,
            (100, 300): 0.2,
            (300, 1000): 0.3,
            (1000, 3000): 0.4,
        },
        ToleranceGrade.L: {
            (0, 100): 0.2,
            (100, 300): 0.4,
            (300, 1000): 0.6,
            (1000, 3000): 0.8,
        },
    },
    GDTCharacteristic.CIRCULAR_RUNOUT: {
        ToleranceGrade.H: {
            (0, 10): 0.02,
            (10, 30): 0.03,
            (30, 100): 0.04,
            (100, 300): 0.05,
        },
        ToleranceGrade.K: {
            (0, 10): 0.05,
            (10, 30): 0.06,
            (30, 100): 0.08,
            (100, 300): 0.1,
        },
        ToleranceGrade.L: {
            (0, 10): 0.1,
            (10, 30): 0.12,
            (30, 100): 0.16,
            (100, 300): 0.2,
        },
    },
    GDTCharacteristic.TOTAL_RUNOUT: {
        ToleranceGrade.H: {
            (0, 10): 0.03,
            (10, 30): 0.04,
            (30, 100): 0.05,
            (100, 300): 0.06,
        },
        ToleranceGrade.K: {
            (0, 10): 0.06,
            (10, 30): 0.08,
            (30, 100): 0.1,
            (100, 300): 0.12,
        },
        ToleranceGrade.L: {
            (0, 10): 0.12,
            (10, 30): 0.16,
            (30, 100): 0.2,
            (100, 300): 0.25,
        },
    },
}

# Tolerance zone definitions
TOLERANCE_ZONE_SHAPES: Dict[GDTCharacteristic, ToleranceZoneShape] = {
    GDTCharacteristic.STRAIGHTNESS: ToleranceZoneShape.BETWEEN_PLANES,
    GDTCharacteristic.FLATNESS: ToleranceZoneShape.BETWEEN_PLANES,
    GDTCharacteristic.CIRCULARITY: ToleranceZoneShape.ANNULAR,
    GDTCharacteristic.CYLINDRICITY: ToleranceZoneShape.BETWEEN_COAXIAL_CYLINDERS,
    GDTCharacteristic.PERPENDICULARITY: ToleranceZoneShape.BETWEEN_PLANES,
    GDTCharacteristic.PARALLELISM: ToleranceZoneShape.BETWEEN_PLANES,
    GDTCharacteristic.ANGULARITY: ToleranceZoneShape.BETWEEN_PLANES,
    GDTCharacteristic.POSITION: ToleranceZoneShape.CYLINDRICAL,
    GDTCharacteristic.CONCENTRICITY: ToleranceZoneShape.CYLINDRICAL,
    GDTCharacteristic.SYMMETRY: ToleranceZoneShape.BETWEEN_PLANES,
    GDTCharacteristic.CIRCULAR_RUNOUT: ToleranceZoneShape.ANNULAR,
    GDTCharacteristic.TOTAL_RUNOUT: ToleranceZoneShape.BETWEEN_COAXIAL_CYLINDERS,
    GDTCharacteristic.PROFILE_LINE: ToleranceZoneShape.BETWEEN_LINES,
    GDTCharacteristic.PROFILE_SURFACE: ToleranceZoneShape.BETWEEN_PLANES,
}


def get_tolerance_zone(
    characteristic: GDTCharacteristic,
    value: float,
) -> ToleranceZone:
    """
    Get the tolerance zone definition for a characteristic.

    Args:
        characteristic: Geometric characteristic
        value: Tolerance value in mm

    Returns:
        ToleranceZone definition
    """
    shape = TOLERANCE_ZONE_SHAPES.get(
        characteristic, ToleranceZoneShape.BETWEEN_PLANES
    )

    zone_descriptions = {
        ToleranceZoneShape.CYLINDRICAL: (
            f"直径为{value}mm的圆柱形公差带",
            f"Cylindrical tolerance zone of diameter {value}mm",
        ),
        ToleranceZoneShape.SPHERICAL: (
            f"直径为{value}mm的球形公差带",
            f"Spherical tolerance zone of diameter {value}mm",
        ),
        ToleranceZoneShape.BETWEEN_PLANES: (
            f"相距{value}mm的两平行平面之间",
            f"Between two parallel planes {value}mm apart",
        ),
        ToleranceZoneShape.BETWEEN_LINES: (
            f"相距{value}mm的两平行直线之间",
            f"Between two parallel lines {value}mm apart",
        ),
        ToleranceZoneShape.ANNULAR: (
            f"半径差为{value}mm的两同心圆之间",
            f"Between two concentric circles with radii differing by {value}mm",
        ),
        ToleranceZoneShape.BETWEEN_COAXIAL_CYLINDERS: (
            f"半径差为{value}mm的两同轴圆柱面之间",
            f"Between two coaxial cylinders with radii differing by {value}mm",
        ),
    }

    desc_zh, desc_en = zone_descriptions.get(
        shape,
        (f"公差值{value}mm", f"Tolerance value {value}mm"),
    )

    return ToleranceZone(
        shape=shape,
        value=value,
        diameter_symbol=(shape == ToleranceZoneShape.CYLINDRICAL),
        description_zh=desc_zh,
        description_en=desc_en,
    )


def get_recommended_tolerance(
    characteristic: GDTCharacteristic,
    nominal_size: float,
    grade: ToleranceGrade = ToleranceGrade.K,
) -> Optional[float]:
    """
    Get recommended tolerance value based on size and grade.

    Args:
        characteristic: Geometric characteristic
        nominal_size: Nominal dimension in mm
        grade: Tolerance grade (H=fine, K=medium, L=coarse)

    Returns:
        Recommended tolerance in mm, or None if not available

    Example:
        >>> tol = get_recommended_tolerance(GDTCharacteristic.FLATNESS, 50, ToleranceGrade.K)
        >>> print(f"Recommended flatness: {tol}mm")
    """
    char_data = TOLERANCE_RECOMMENDATIONS.get(characteristic)
    if not char_data:
        return None

    grade_data = char_data.get(grade)
    if not grade_data:
        return None

    for (min_size, max_size), value in grade_data.items():
        if min_size <= nominal_size < max_size:
            return value

    return None


def calculate_bonus_tolerance(
    geometric_tolerance: float,
    actual_size: float,
    mmc_size: float,
    modifier: str = "M",
) -> float:
    """
    Calculate bonus tolerance when MMC or LMC modifier is applied.

    Args:
        geometric_tolerance: Specified geometric tolerance
        actual_size: Actual measured size
        mmc_size: Maximum Material Condition size
        modifier: "M" for MMC, "L" for LMC

    Returns:
        Total allowable tolerance (geometric + bonus)

    Example:
        >>> # Hole: Ø10 +0.1/0 with position tolerance Ø0.2 (M)
        >>> total = calculate_bonus_tolerance(0.2, 10.08, 10.0, "M")
        >>> print(f"Total tolerance: Ø{total}mm")  # 0.2 + 0.08 = 0.28
    """
    if modifier == "M":
        # MMC: bonus = departure from MMC
        bonus = abs(actual_size - mmc_size)
    elif modifier == "L":
        # LMC: bonus = departure from LMC
        bonus = abs(actual_size - mmc_size)
    else:
        bonus = 0.0

    return geometric_tolerance + bonus


def get_tolerance_relationship(
    form_tolerance: float,
    orientation_tolerance: float,
    location_tolerance: float,
) -> Dict[str, any]:
    """
    Check if tolerance values follow the proper hierarchy.

    Rule: Form ≤ Orientation ≤ Location

    Args:
        form_tolerance: Form tolerance value
        orientation_tolerance: Orientation tolerance value
        location_tolerance: Location tolerance value

    Returns:
        Dict with validation result and recommendations
    """
    is_valid = form_tolerance <= orientation_tolerance <= location_tolerance

    result = {
        "is_valid": is_valid,
        "form": form_tolerance,
        "orientation": orientation_tolerance,
        "location": location_tolerance,
        "rule": "形状公差 ≤ 方向公差 ≤ 位置公差",
    }

    if not is_valid:
        issues = []
        if form_tolerance > orientation_tolerance:
            issues.append("形状公差大于方向公差")
        if orientation_tolerance > location_tolerance:
            issues.append("方向公差大于位置公差")
        result["issues"] = issues
        result["recommendation"] = "调整公差值以满足层级关系"

    return result
