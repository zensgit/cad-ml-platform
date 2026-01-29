"""
General Tolerances Knowledge Base.

Provides general tolerances for linear and angular dimensions per ISO 2768.

Reference:
- ISO 2768-1:1989 - General tolerances for linear and angular dimensions
- ISO 2768-2:1989 - General tolerances for geometrical tolerances
- GB/T 1804-2000 (Chinese equivalent)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class GeneralToleranceClass(str, Enum):
    """General tolerance class per ISO 2768-1."""

    F = "f"  # Fine 精密级
    M = "m"  # Medium 中等级
    C = "c"  # Coarse 粗糙级
    V = "v"  # Very coarse 最粗级


# Linear tolerance table per ISO 2768-1
# Format: (min_size, max_size): {class: tolerance_mm}
LINEAR_TOLERANCE_TABLE: Dict[Tuple[float, float], Dict[str, float]] = {
    (0.5, 3): {"f": 0.05, "m": 0.1, "c": 0.2, "v": None},
    (3, 6): {"f": 0.05, "m": 0.1, "c": 0.3, "v": 0.5},
    (6, 30): {"f": 0.1, "m": 0.2, "c": 0.5, "v": 1.0},
    (30, 120): {"f": 0.15, "m": 0.3, "c": 0.8, "v": 1.5},
    (120, 400): {"f": 0.2, "m": 0.5, "c": 1.2, "v": 2.5},
    (400, 1000): {"f": 0.3, "m": 0.8, "c": 2.0, "v": 4.0},
    (1000, 2000): {"f": 0.5, "m": 1.2, "c": 3.0, "v": 6.0},
    (2000, 4000): {"f": None, "m": 2.0, "c": 4.0, "v": 8.0},
    (4000, 8000): {"f": None, "m": 3.0, "c": 5.0, "v": 10.0},
    (8000, 12000): {"f": None, "m": 4.0, "c": 6.0, "v": 12.0},
    (12000, 16000): {"f": None, "m": 5.0, "c": 7.0, "v": 14.0},
    (16000, 20000): {"f": None, "m": 6.0, "c": 8.0, "v": 16.0},
}

# Angular tolerance table per ISO 2768-1
# Format: (min_length, max_length): {class: tolerance_deg_min}
ANGULAR_TOLERANCE_TABLE: Dict[Tuple[float, float], Dict[str, str]] = {
    (0, 10): {"f": "±1°", "m": "±1°", "c": "±1°30'", "v": "±3°"},
    (10, 50): {"f": "±0°30'", "m": "±0°30'", "c": "±1°", "v": "±2°"},
    (50, 120): {"f": "±0°20'", "m": "±0°20'", "c": "±0°30'", "v": "±1°"},
    (120, 400): {"f": "±0°10'", "m": "±0°10'", "c": "±0°15'", "v": "±0°30'"},
    (400, float('inf')): {"f": "±0°5'", "m": "±0°5'", "c": "±0°10'", "v": "±0°20'"},
}

# Chamfer and fillet tolerances (external radius and chamfer height)
CHAMFER_FILLET_TOLERANCE: Dict[Tuple[float, float], Dict[str, float]] = {
    (0.5, 3): {"f": 0.2, "m": 0.2, "c": 0.4, "v": 0.4},
    (3, 6): {"f": 0.5, "m": 0.5, "c": 1.0, "v": 1.0},
    (6, 30): {"f": 1.0, "m": 1.0, "c": 2.0, "v": 2.0},
    (30, float('inf')): {"f": 2.0, "m": 2.0, "c": 4.0, "v": 4.0},
}


def get_linear_tolerance(
    dimension: float,
    tolerance_class: GeneralToleranceClass = GeneralToleranceClass.M,
) -> Optional[float]:
    """
    Get general linear tolerance for a dimension.

    Args:
        dimension: Nominal dimension in mm
        tolerance_class: Tolerance class (f, m, c, v)

    Returns:
        Tolerance value in mm (±), or None if not applicable

    Example:
        >>> get_linear_tolerance(50, GeneralToleranceClass.M)
        0.3
    """
    class_key = tolerance_class.value

    for (min_size, max_size), tolerances in LINEAR_TOLERANCE_TABLE.items():
        if min_size < dimension <= max_size:
            return tolerances.get(class_key)

    return None


def get_angular_tolerance(
    length: float,
    tolerance_class: GeneralToleranceClass = GeneralToleranceClass.M,
) -> Optional[str]:
    """
    Get general angular tolerance based on shorter side length.

    Args:
        length: Length of shorter side of angle in mm
        tolerance_class: Tolerance class (f, m, c, v)

    Returns:
        Angular tolerance string (e.g., "±0°30'"), or None if not found

    Example:
        >>> get_angular_tolerance(80, GeneralToleranceClass.M)
        '±0°20''
    """
    class_key = tolerance_class.value

    for (min_len, max_len), tolerances in ANGULAR_TOLERANCE_TABLE.items():
        if min_len < length <= max_len:
            return tolerances.get(class_key)

    return None


def get_chamfer_fillet_tolerance(
    size: float,
    tolerance_class: GeneralToleranceClass = GeneralToleranceClass.M,
) -> Optional[float]:
    """
    Get tolerance for chamfer/fillet dimensions.

    Args:
        size: Chamfer or fillet nominal size in mm
        tolerance_class: Tolerance class

    Returns:
        Tolerance value in mm (±)
    """
    class_key = tolerance_class.value

    for (min_size, max_size), tolerances in CHAMFER_FILLET_TOLERANCE.items():
        if min_size < size <= max_size:
            return tolerances.get(class_key)

    return None


def get_general_tolerance_table(
    tolerance_class: GeneralToleranceClass = GeneralToleranceClass.M,
) -> List[Dict]:
    """
    Get full general tolerance table for a class.

    Args:
        tolerance_class: Tolerance class

    Returns:
        List of dictionaries with size ranges and tolerances
    """
    class_key = tolerance_class.value
    results = []

    for (min_size, max_size), tolerances in LINEAR_TOLERANCE_TABLE.items():
        tol = tolerances.get(class_key)
        if tol is not None:
            results.append({
                "range_min": min_size,
                "range_max": max_size,
                "range_str": f"{min_size}-{max_size}",
                "tolerance_mm": tol,
                "tolerance_str": f"±{tol}",
            })

    return results


@dataclass
class GeneralToleranceSpec:
    """General tolerance specification for a drawing."""

    linear_class: GeneralToleranceClass
    angular_class: GeneralToleranceClass
    designation: str  # e.g., "ISO 2768-mK"

    @classmethod
    def from_designation(cls, designation: str) -> Optional['GeneralToleranceSpec']:
        """Parse ISO 2768 designation string."""
        # Format: ISO 2768-mK (m=linear, K=geometrical)
        designation = designation.upper().replace(" ", "")

        if "2768" not in designation:
            return None

        # Extract class letters
        parts = designation.split("-")
        if len(parts) < 2:
            return None

        class_str = parts[-1]
        if not class_str:
            return None

        linear_char = class_str[0].lower()
        if linear_char not in ["f", "m", "c", "v"]:
            return None

        linear_class = GeneralToleranceClass(linear_char)

        return cls(
            linear_class=linear_class,
            angular_class=linear_class,  # Same class for angular
            designation=designation,
        )


def suggest_tolerance_class(
    application: str = "general",
    precision: str = "medium",
) -> GeneralToleranceClass:
    """
    Suggest appropriate general tolerance class.

    Args:
        application: "precision", "general", "structural", "rough"
        precision: "high", "medium", "low"

    Returns:
        Recommended GeneralToleranceClass
    """
    suggestions = {
        ("precision", "high"): GeneralToleranceClass.F,
        ("precision", "medium"): GeneralToleranceClass.F,
        ("precision", "low"): GeneralToleranceClass.M,
        ("general", "high"): GeneralToleranceClass.M,
        ("general", "medium"): GeneralToleranceClass.M,
        ("general", "low"): GeneralToleranceClass.C,
        ("structural", "high"): GeneralToleranceClass.M,
        ("structural", "medium"): GeneralToleranceClass.C,
        ("structural", "low"): GeneralToleranceClass.V,
        ("rough", "high"): GeneralToleranceClass.C,
        ("rough", "medium"): GeneralToleranceClass.V,
        ("rough", "low"): GeneralToleranceClass.V,
    }

    return suggestions.get((application.lower(), precision.lower()), GeneralToleranceClass.M)
