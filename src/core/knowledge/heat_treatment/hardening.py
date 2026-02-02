"""
Steel Hardening Knowledge Base.

Provides hardenability data, Jominy curves, and tempering
parameter recommendations for common steels.

Reference:
- ASM Handbook Volume 4 - Heat Treating
- ASTM A255 - Hardenability test methods
- ISO 642 - Hardenability test (Jominy)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import math


class HardenabilityClass(str, Enum):
    """Hardenability classification."""

    VERY_LOW = "very_low"  # 极低淬透性
    LOW = "low"  # 低淬透性
    MEDIUM = "medium"  # 中等淬透性
    HIGH = "high"  # 高淬透性
    VERY_HIGH = "very_high"  # 极高淬透性 (air hardening)


@dataclass
class HardenabilityData:
    """Hardenability data for a steel."""

    material_id: str
    hardenability_class: HardenabilityClass

    # Critical diameter (DI) in mm
    # Diameter that can be through-hardened in given medium
    critical_diameter_water: Optional[float] = None
    critical_diameter_oil: Optional[float] = None
    critical_diameter_air: Optional[float] = None

    # Ideal critical diameter (DI) - theoretical in ideal quench
    ideal_critical_diameter: Optional[float] = None

    # Jominy distance for 50% martensite (J50% in mm)
    jominy_50_martensite: Optional[float] = None

    # As-quenched hardness
    as_quenched_hardness_max: Optional[float] = None  # HRC

    notes_zh: str = ""
    notes_en: str = ""


# Hardenability database
HARDENING_DATABASE: Dict[str, HardenabilityData] = {
    "45": HardenabilityData(
        material_id="45",
        hardenability_class=HardenabilityClass.LOW,
        critical_diameter_water=25,
        critical_diameter_oil=10,
        ideal_critical_diameter=22,
        jominy_50_martensite=8,
        as_quenched_hardness_max=60,
        notes_zh="45钢淬透性低，适合小截面零件",
        notes_en="Low hardenability, suitable for small sections",
    ),
    "40Cr": HardenabilityData(
        material_id="40Cr",
        hardenability_class=HardenabilityClass.MEDIUM,
        critical_diameter_water=40,
        critical_diameter_oil=25,
        ideal_critical_diameter=45,
        jominy_50_martensite=20,
        as_quenched_hardness_max=58,
        notes_zh="40Cr淬透性中等，常用调质钢",
        notes_en="Medium hardenability, common Q&T steel",
    ),
    "42CrMo": HardenabilityData(
        material_id="42CrMo",
        hardenability_class=HardenabilityClass.HIGH,
        critical_diameter_water=70,
        critical_diameter_oil=50,
        ideal_critical_diameter=85,
        jominy_50_martensite=40,
        as_quenched_hardness_max=57,
        notes_zh="42CrMo淬透性好，大型锻件常用",
        notes_en="Good hardenability, used for large forgings",
    ),
    "Cr12MoV": HardenabilityData(
        material_id="Cr12MoV",
        hardenability_class=HardenabilityClass.VERY_HIGH,
        critical_diameter_oil=150,
        critical_diameter_air=80,
        ideal_critical_diameter=200,
        jominy_50_martensite=100,
        as_quenched_hardness_max=64,
        notes_zh="D2钢空淬硬化，极高淬透性",
        notes_en="Air hardening, very high hardenability",
    ),
    "W18Cr4V": HardenabilityData(
        material_id="W18Cr4V",
        hardenability_class=HardenabilityClass.VERY_HIGH,
        critical_diameter_oil=150,
        critical_diameter_air=100,
        as_quenched_hardness_max=65,
        notes_zh="高速钢，空淬或油淬",
        notes_en="HSS, air or oil hardening",
    ),
    "GCr15": HardenabilityData(
        material_id="GCr15",
        hardenability_class=HardenabilityClass.MEDIUM,
        critical_diameter_water=35,
        critical_diameter_oil=20,
        ideal_critical_diameter=40,
        jominy_50_martensite=15,
        as_quenched_hardness_max=65,
        notes_zh="轴承钢，需马氏体淬火后低温回火",
        notes_en="Bearing steel, requires martensite quench + low temp temper",
    ),
    "65Mn": HardenabilityData(
        material_id="65Mn",
        hardenability_class=HardenabilityClass.LOW,
        critical_diameter_water=18,
        critical_diameter_oil=8,
        ideal_critical_diameter=20,
        jominy_50_martensite=6,
        as_quenched_hardness_max=62,
        notes_zh="弹簧钢，淬透性较低",
        notes_en="Spring steel, low hardenability",
    ),
    "T10": HardenabilityData(
        material_id="T10",
        hardenability_class=HardenabilityClass.VERY_LOW,
        critical_diameter_water=15,
        critical_diameter_oil=6,
        ideal_critical_diameter=12,
        jominy_50_martensite=4,
        as_quenched_hardness_max=66,
        notes_zh="碳素工具钢，淬透性极低",
        notes_en="Carbon tool steel, very low hardenability",
    ),
}

# Tempering temperature vs hardness curves (approximate)
# Format: {material: [(temp, HRC), ...]}
TEMPERING_CURVES: Dict[str, List[Tuple[int, int]]] = {
    "45": [
        (150, 58),
        (200, 56),
        (250, 54),
        (300, 50),
        (350, 46),
        (400, 42),
        (450, 38),
        (500, 34),
        (550, 30),
        (600, 26),
        (650, 22),
    ],
    "40Cr": [
        (150, 56),
        (200, 54),
        (250, 52),
        (300, 49),
        (350, 46),
        (400, 43),
        (450, 40),
        (500, 36),
        (550, 32),
        (600, 28),
        (650, 24),
    ],
    "42CrMo": [
        (150, 55),
        (200, 53),
        (250, 51),
        (300, 49),
        (350, 46),
        (400, 44),
        (450, 42),
        (500, 38),
        (550, 34),
        (600, 30),
        (650, 26),
    ],
    "Cr12MoV": [
        (150, 62),
        (200, 61),
        (250, 60),
        (300, 58),
        (350, 56),
        (400, 54),
        (450, 52),
        (500, 54),  # Secondary hardening peak
        (520, 58),  # Secondary hardening peak
        (550, 56),
        (600, 52),
    ],
    "GCr15": [
        (150, 64),
        (175, 62),
        (200, 60),
        (225, 58),
        (250, 56),
        (300, 52),
        (350, 48),
        (400, 44),
    ],
}


def get_hardenability(material_id: str) -> Optional[HardenabilityData]:
    """
    Get hardenability data for a material.

    Args:
        material_id: Material identifier

    Returns:
        HardenabilityData or None

    Example:
        >>> data = get_hardenability("40Cr")
        >>> print(f"Critical diameter (oil): {data.critical_diameter_oil}mm")
    """
    return HARDENING_DATABASE.get(material_id)


def can_through_harden(
    material_id: str,
    section_size: float,
    quench_media: str = "oil",
) -> Dict[str, Any]:
    """
    Check if a section can be through-hardened.

    Args:
        material_id: Material identifier
        section_size: Maximum section size (mm)
        quench_media: "water", "oil", or "air"

    Returns:
        Dict with assessment results
    """
    data = HARDENING_DATABASE.get(material_id)
    if not data:
        return {"success": False, "reason": "Material not found"}

    # Get critical diameter for media
    critical_diameters = {
        "water": data.critical_diameter_water,
        "oil": data.critical_diameter_oil,
        "air": data.critical_diameter_air,
    }

    critical_d = critical_diameters.get(quench_media)
    if critical_d is None:
        return {
            "success": False,
            "reason": f"{material_id} cannot be hardened with {quench_media}",
        }

    can_harden = section_size <= critical_d
    margin = critical_d - section_size

    return {
        "success": can_harden,
        "critical_diameter": critical_d,
        "section_size": section_size,
        "margin": margin,
        "recommendation": "可完全淬透" if can_harden else "截面过大，无法完全淬透",
        "alternative": None if can_harden else suggest_alternative_quench(material_id, section_size),
    }


def suggest_alternative_quench(
    material_id: str,
    section_size: float,
) -> Optional[str]:
    """Suggest alternative quench media if current cannot through-harden."""
    data = HARDENING_DATABASE.get(material_id)
    if not data:
        return None

    if data.critical_diameter_water and section_size <= data.critical_diameter_water:
        return "water"
    if data.critical_diameter_oil and section_size <= data.critical_diameter_oil:
        return "oil"
    if data.critical_diameter_air and section_size <= data.critical_diameter_air:
        return "air"

    return None


def get_tempering_temperature(
    material_id: str,
    target_hardness: float,
) -> Optional[Tuple[int, int]]:
    """
    Get tempering temperature to achieve target hardness.

    Args:
        material_id: Material identifier
        target_hardness: Target hardness (HRC)

    Returns:
        Tuple of (min_temp, max_temp) or None

    Example:
        >>> temp = get_tempering_temperature("45", 40)
        >>> print(f"Temper at: {temp[0]}-{temp[1]}°C")
    """
    curve = TEMPERING_CURVES.get(material_id)
    if not curve:
        return None

    # Find temperature range that gives target hardness
    for i, (temp, hrc) in enumerate(curve):
        if hrc <= target_hardness:
            if i == 0:
                return (temp - 25, temp)
            prev_temp, prev_hrc = curve[i - 1]
            return (prev_temp, temp)

    # Target too soft - use highest temp
    return (curve[-1][0], curve[-1][0] + 50)


def calculate_hardness_after_tempering(
    material_id: str,
    tempering_temperature: float,
) -> Optional[float]:
    """
    Calculate expected hardness after tempering.

    Args:
        material_id: Material identifier
        tempering_temperature: Tempering temperature (°C)

    Returns:
        Expected hardness (HRC) or None

    Example:
        >>> hrc = calculate_hardness_after_tempering("45", 500)
        >>> print(f"Expected hardness: {hrc} HRC")
    """
    curve = TEMPERING_CURVES.get(material_id)
    if not curve:
        return None

    # Linear interpolation
    for i, (temp, hrc) in enumerate(curve):
        if temp >= tempering_temperature:
            if i == 0:
                return float(hrc)
            prev_temp, prev_hrc = curve[i - 1]
            # Interpolate
            ratio = (tempering_temperature - prev_temp) / (temp - prev_temp)
            return round(prev_hrc + ratio * (hrc - prev_hrc), 1)

    # Beyond curve - extrapolate
    return float(curve[-1][1])


def estimate_jominy_hardness(
    material_id: str,
    distance_from_quenched_end: float,
) -> Optional[float]:
    """
    Estimate hardness at distance from quenched end (Jominy test).

    Args:
        material_id: Material identifier
        distance_from_quenched_end: Distance J (mm)

    Returns:
        Estimated hardness (HRC) or None
    """
    data = HARDENING_DATABASE.get(material_id)
    if not data or not data.as_quenched_hardness_max:
        return None

    j50 = data.jominy_50_martensite
    if not j50:
        return None

    h_max = data.as_quenched_hardness_max
    h_min = 20  # Approximate minimum (pearlite)

    # Exponential decay model (simplified)
    if distance_from_quenched_end <= 1.5:
        return h_max

    decay_rate = math.log(2) / j50
    fraction = math.exp(-decay_rate * distance_from_quenched_end)
    hardness = h_min + (h_max - h_min) * fraction

    return round(max(h_min, hardness), 1)


def recommend_quench_for_section(
    material_id: str,
    section_size: float,
    required_core_hardness: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Recommend quenching approach for a given section size.

    Args:
        material_id: Material identifier
        section_size: Maximum section size (mm)
        required_core_hardness: Required core hardness (HRC)

    Returns:
        Dict with recommendations
    """
    data = HARDENING_DATABASE.get(material_id)
    if not data:
        return {"error": "Material not found"}

    result = {
        "material_id": material_id,
        "section_size": section_size,
        "hardenability_class": data.hardenability_class.value,
        "recommendations": [],
    }

    # Check each quench media
    if data.critical_diameter_water and section_size <= data.critical_diameter_water:
        result["recommendations"].append({
            "quench_media": "water",
            "can_through_harden": True,
            "notes_zh": "水淬可完全淬透，注意开裂风险",
        })
    elif data.critical_diameter_water:
        result["recommendations"].append({
            "quench_media": "water",
            "can_through_harden": False,
            "notes_zh": f"水淬临界直径{data.critical_diameter_water}mm，截面过大",
        })

    if data.critical_diameter_oil and section_size <= data.critical_diameter_oil:
        result["recommendations"].append({
            "quench_media": "oil",
            "can_through_harden": True,
            "notes_zh": "油淬可完全淬透，变形开裂风险较小",
        })

    if data.critical_diameter_air and section_size <= data.critical_diameter_air:
        result["recommendations"].append({
            "quench_media": "air",
            "can_through_harden": True,
            "notes_zh": "空冷硬化，变形最小",
        })

    # Best recommendation
    if result["recommendations"]:
        # Prefer oil over water, air over oil (for distortion control)
        preference = {"air": 3, "oil": 2, "water": 1}
        best = max(
            [r for r in result["recommendations"] if r["can_through_harden"]],
            key=lambda x: preference.get(x["quench_media"], 0),
            default=None,
        )
        if best:
            result["best_recommendation"] = best["quench_media"]
        else:
            result["best_recommendation"] = None
            result["notes"] = "截面过大，无法完全淬透，建议选用更高淬透性钢种"

    return result
