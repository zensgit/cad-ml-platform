"""
Design Features Standards Knowledge Base.

Provides standard design features including preferred sizes, chamfers, and fillets.

Reference:
- ISO 497:1973 - Guide to the choice of series of preferred numbers
- ISO 286-1:2010 - Preferred numbers
- Various national standards for chamfers and fillets
"""

from typing import Dict, List, Optional, Tuple


# Preferred diameter series per ISO/R 497 (R10, R20, R40 series combined)
# Common shaft and hole diameters in mm
PREFERRED_DIAMETERS: List[float] = [
    # R10 series (main values)
    1.0, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0,
    10, 12, 16, 20, 25, 32, 40, 50, 63, 80,
    100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
    1000,
    # R20 intermediate values
    1.1, 1.4, 1.8, 2.2, 2.8, 3.5, 4.5, 5.5, 7.0, 9.0,
    11, 14, 18, 22, 28, 35, 45, 55, 70, 90,
    110, 140, 180, 220, 280, 355, 450, 560, 710, 900,
    # Common additional sizes
    3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,
    15, 17, 19, 24, 26, 30, 34, 36, 38, 42, 48, 52, 58, 62, 65, 68, 72, 75, 78, 85, 95,
]

# Sort and remove duplicates
PREFERRED_DIAMETERS = sorted(set(PREFERRED_DIAMETERS))


# Standard chamfer sizes (C x 45° format)
# Format: nominal_size_mm: (c_min, c_max, typical_c)
STANDARD_CHAMFERS: Dict[str, Dict] = {
    # Small chamfers
    "C0.2": {"size": 0.2, "range": (0.15, 0.25), "application_zh": "微小边缘", "application_en": "Micro edges"},
    "C0.3": {"size": 0.3, "range": (0.25, 0.35), "application_zh": "小零件边缘", "application_en": "Small part edges"},
    "C0.5": {"size": 0.5, "range": (0.4, 0.6), "application_zh": "一般小边缘", "application_en": "General small edges"},
    # Standard chamfers
    "C1": {"size": 1.0, "range": (0.8, 1.2), "application_zh": "一般边缘倒角", "application_en": "General edge chamfer"},
    "C1.5": {"size": 1.5, "range": (1.2, 1.8), "application_zh": "中等倒角", "application_en": "Medium chamfer"},
    "C2": {"size": 2.0, "range": (1.6, 2.4), "application_zh": "标准倒角", "application_en": "Standard chamfer"},
    "C2.5": {"size": 2.5, "range": (2.0, 3.0), "application_zh": "标准倒角", "application_en": "Standard chamfer"},
    "C3": {"size": 3.0, "range": (2.5, 3.5), "application_zh": "较大倒角", "application_en": "Larger chamfer"},
    # Large chamfers
    "C4": {"size": 4.0, "range": (3.5, 4.5), "application_zh": "大倒角", "application_en": "Large chamfer"},
    "C5": {"size": 5.0, "range": (4.5, 5.5), "application_zh": "大倒角", "application_en": "Large chamfer"},
    "C6": {"size": 6.0, "range": (5.5, 6.5), "application_zh": "特大倒角", "application_en": "Extra large chamfer"},
    "C8": {"size": 8.0, "range": (7.0, 9.0), "application_zh": "特大倒角", "application_en": "Extra large chamfer"},
    "C10": {"size": 10.0, "range": (9.0, 11.0), "application_zh": "超大倒角", "application_en": "Very large chamfer"},
}


# Standard fillet radii
# Format: "Rn": {size, range, application}
STANDARD_FILLETS: Dict[str, Dict] = {
    # Small fillets
    "R0.2": {"size": 0.2, "range": (0.15, 0.25), "application_zh": "微小圆角", "application_en": "Micro fillet"},
    "R0.3": {"size": 0.3, "range": (0.25, 0.35), "application_zh": "小圆角", "application_en": "Small fillet"},
    "R0.5": {"size": 0.5, "range": (0.4, 0.6), "application_zh": "一般小圆角", "application_en": "General small fillet"},
    # Standard fillets
    "R1": {"size": 1.0, "range": (0.8, 1.2), "application_zh": "标准圆角", "application_en": "Standard fillet"},
    "R1.5": {"size": 1.5, "range": (1.2, 1.8), "application_zh": "中等圆角", "application_en": "Medium fillet"},
    "R2": {"size": 2.0, "range": (1.6, 2.4), "application_zh": "标准圆角", "application_en": "Standard fillet"},
    "R2.5": {"size": 2.5, "range": (2.0, 3.0), "application_zh": "标准圆角", "application_en": "Standard fillet"},
    "R3": {"size": 3.0, "range": (2.5, 3.5), "application_zh": "较大圆角", "application_en": "Larger fillet"},
    # Large fillets
    "R4": {"size": 4.0, "range": (3.5, 4.5), "application_zh": "大圆角", "application_en": "Large fillet"},
    "R5": {"size": 5.0, "range": (4.5, 5.5), "application_zh": "大圆角", "application_en": "Large fillet"},
    "R6": {"size": 6.0, "range": (5.5, 6.5), "application_zh": "特大圆角", "application_en": "Extra large fillet"},
    "R8": {"size": 8.0, "range": (7.0, 9.0), "application_zh": "特大圆角", "application_en": "Extra large fillet"},
    "R10": {"size": 10.0, "range": (9.0, 11.0), "application_zh": "超大圆角", "application_en": "Very large fillet"},
    "R12": {"size": 12.0, "range": (11.0, 13.0), "application_zh": "超大圆角", "application_en": "Very large fillet"},
    "R15": {"size": 15.0, "range": (14.0, 16.0), "application_zh": "超大圆角", "application_en": "Very large fillet"},
    "R20": {"size": 20.0, "range": (18.0, 22.0), "application_zh": "超大圆角", "application_en": "Very large fillet"},
}


def get_preferred_diameter(target: float, direction: str = "nearest") -> float:
    """
    Get nearest preferred diameter.

    Args:
        target: Target diameter in mm
        direction: "nearest", "up", or "down"

    Returns:
        Preferred diameter in mm

    Example:
        >>> get_preferred_diameter(23)
        22  # Nearest preferred size
        >>> get_preferred_diameter(23, "up")
        24
    """
    if direction == "up":
        for d in PREFERRED_DIAMETERS:
            if d >= target:
                return d
        return PREFERRED_DIAMETERS[-1]

    elif direction == "down":
        for d in reversed(PREFERRED_DIAMETERS):
            if d <= target:
                return d
        return PREFERRED_DIAMETERS[0]

    else:  # nearest
        closest = PREFERRED_DIAMETERS[0]
        min_diff = abs(target - closest)

        for d in PREFERRED_DIAMETERS:
            diff = abs(target - d)
            if diff < min_diff:
                min_diff = diff
                closest = d

        return closest


def get_standard_chamfer(target_size: float) -> Optional[Dict]:
    """
    Get standard chamfer closest to target size.

    Args:
        target_size: Target chamfer size in mm

    Returns:
        Dictionary with chamfer specification

    Example:
        >>> result = get_standard_chamfer(1.8)
        >>> print(result['designation'])  # 'C2'
    """
    closest = None
    min_diff = float('inf')

    for designation, data in STANDARD_CHAMFERS.items():
        diff = abs(data["size"] - target_size)
        if diff < min_diff:
            min_diff = diff
            closest = {
                "designation": designation,
                "size": data["size"],
                "range": data["range"],
                "application_zh": data["application_zh"],
                "application_en": data["application_en"],
            }

    return closest


def get_standard_fillet(target_radius: float) -> Optional[Dict]:
    """
    Get standard fillet closest to target radius.

    Args:
        target_radius: Target fillet radius in mm

    Returns:
        Dictionary with fillet specification

    Example:
        >>> result = get_standard_fillet(2.3)
        >>> print(result['designation'])  # 'R2.5'
    """
    closest = None
    min_diff = float('inf')

    for designation, data in STANDARD_FILLETS.items():
        diff = abs(data["size"] - target_radius)
        if diff < min_diff:
            min_diff = diff
            closest = {
                "designation": designation,
                "size": data["size"],
                "range": data["range"],
                "application_zh": data["application_zh"],
                "application_en": data["application_en"],
            }

    return closest


def suggest_chamfer_for_thread(thread_diameter: float) -> Dict:
    """
    Suggest chamfer size for thread entry.

    Args:
        thread_diameter: Thread nominal diameter in mm

    Returns:
        Recommended chamfer specification
    """
    # Rule of thumb: chamfer ≈ pitch or 0.1*D, whichever is larger
    suggested_size = max(thread_diameter * 0.1, 0.5)

    return get_standard_chamfer(suggested_size)


def suggest_fillet_for_shaft(
    shaft_diameter: float,
    stress_concentration: str = "medium",
) -> Dict:
    """
    Suggest fillet radius for shaft shoulder.

    Args:
        shaft_diameter: Shaft diameter in mm
        stress_concentration: "low", "medium", or "high" concern

    Returns:
        Recommended fillet specification
    """
    # Fillet radius recommendations based on stress concentration
    factors = {
        "low": 0.02,  # R ≈ 2% of D
        "medium": 0.05,  # R ≈ 5% of D
        "high": 0.1,  # R ≈ 10% of D
    }

    factor = factors.get(stress_concentration.lower(), 0.05)
    suggested_radius = shaft_diameter * factor

    # Minimum practical fillet
    suggested_radius = max(suggested_radius, 0.5)

    return get_standard_fillet(suggested_radius)


def list_preferred_diameters(min_d: float = 0, max_d: float = 200) -> List[float]:
    """
    List preferred diameters within a range.

    Args:
        min_d: Minimum diameter (mm)
        max_d: Maximum diameter (mm)

    Returns:
        List of preferred diameters
    """
    return [d for d in PREFERRED_DIAMETERS if min_d <= d <= max_d]
