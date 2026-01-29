"""
Weldability and Material Compatibility Knowledge Base.

Provides weldability classifications, preheat requirements, and
material compatibility information for welding.

Reference:
- AWS D1.1 - Structural Welding Code
- ASME Section IX - Welding and Brazing Qualifications
- ISO/TR 15608 - Welding - Guidelines for material grouping
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import math


class WeldabilityClass(str, Enum):
    """Weldability classification."""

    EXCELLENT = "excellent"  # 优良可焊性
    GOOD = "good"  # 良好可焊性
    FAIR = "fair"  # 一般可焊性
    POOR = "poor"  # 较差可焊性
    DIFFICULT = "difficult"  # 困难焊接


@dataclass
class BaseMaterial:
    """Base material properties for welding."""

    material_id: str
    material_name_zh: str
    material_name_en: str

    # Weldability
    weldability: WeldabilityClass
    carbon_equivalent: Optional[float] = None  # CE(IIW)

    # Preheat requirements
    preheat_required: bool = False
    preheat_temp_min: Optional[float] = None  # °C
    preheat_temp_max: Optional[float] = None  # °C

    # Post-weld heat treatment
    pwht_required: bool = False
    pwht_temp_range: Optional[Tuple[float, float]] = None  # °C
    pwht_time_per_inch: Optional[float] = None  # hours/inch

    # Interpass temperature
    interpass_temp_max: Optional[float] = None  # °C

    # Special considerations
    special_requirements: List[str] = None

    notes_zh: str = ""
    notes_en: str = ""


# Weldability database
WELDABILITY_DATABASE: Dict[str, BaseMaterial] = {
    # Low carbon steel
    "Q235": BaseMaterial(
        material_id="Q235",
        material_name_zh="Q235碳素结构钢",
        material_name_en="Q235 Carbon Structural Steel",
        weldability=WeldabilityClass.EXCELLENT,
        carbon_equivalent=0.25,
        preheat_required=False,
        interpass_temp_max=250,
        notes_zh="低碳钢，焊接性能优良，一般不需预热",
        notes_en="Low carbon steel, excellent weldability, no preheat normally required",
    ),
    "Q345": BaseMaterial(
        material_id="Q345",
        material_name_zh="Q345低合金高强度结构钢",
        material_name_en="Q345 Low Alloy High Strength Steel",
        weldability=WeldabilityClass.GOOD,
        carbon_equivalent=0.40,
        preheat_required=True,
        preheat_temp_min=50,
        preheat_temp_max=150,
        interpass_temp_max=250,
        notes_zh="低合金钢，厚度>25mm时需预热",
        notes_en="Low alloy steel, preheat required for thickness >25mm",
    ),
    "45": BaseMaterial(
        material_id="45",
        material_name_zh="45号碳素结构钢",
        material_name_en="AISI 1045 Medium Carbon Steel",
        weldability=WeldabilityClass.FAIR,
        carbon_equivalent=0.60,
        preheat_required=True,
        preheat_temp_min=150,
        preheat_temp_max=250,
        pwht_required=True,
        pwht_temp_range=(580, 620),
        interpass_temp_max=300,
        special_requirements=["使用低氢焊条", "控制层间温度"],
        notes_zh="中碳钢，必须预热，焊后需消除应力处理",
        notes_en="Medium carbon steel, must preheat, PWHT required",
    ),
    "40Cr": BaseMaterial(
        material_id="40Cr",
        material_name_zh="40Cr合金结构钢",
        material_name_en="AISI 5140 Alloy Steel",
        weldability=WeldabilityClass.POOR,
        carbon_equivalent=0.72,
        preheat_required=True,
        preheat_temp_min=200,
        preheat_temp_max=300,
        pwht_required=True,
        pwht_temp_range=(550, 650),
        interpass_temp_max=350,
        special_requirements=["使用低氢焊条", "焊后立即保温", "缓冷处理"],
        notes_zh="合金钢，焊接性较差，必须严格预热和后热处理",
        notes_en="Alloy steel, poor weldability, strict preheat and PWHT required",
    ),
    # Stainless steels
    "304": BaseMaterial(
        material_id="304",
        material_name_zh="304奥氏体不锈钢",
        material_name_en="304 Austenitic Stainless Steel",
        weldability=WeldabilityClass.GOOD,
        preheat_required=False,
        interpass_temp_max=150,
        special_requirements=["控制热输入", "防止敏化", "背面保护"],
        notes_zh="奥氏体不锈钢，焊接性良好，注意热输入控制",
        notes_en="Austenitic SS, good weldability, control heat input",
    ),
    "316L": BaseMaterial(
        material_id="316L",
        material_name_zh="316L奥氏体不锈钢",
        material_name_en="316L Austenitic Stainless Steel",
        weldability=WeldabilityClass.GOOD,
        preheat_required=False,
        interpass_temp_max=150,
        special_requirements=["低碳型抗敏化", "控制热输入"],
        notes_zh="低碳316不锈钢，抗晶间腐蚀性能好",
        notes_en="Low carbon 316L SS, good intergranular corrosion resistance",
    ),
    "2205": BaseMaterial(
        material_id="2205",
        material_name_zh="2205双相不锈钢",
        material_name_en="2205 Duplex Stainless Steel",
        weldability=WeldabilityClass.FAIR,
        preheat_required=False,
        interpass_temp_max=100,
        special_requirements=["严格控制热输入", "控制铁素体/奥氏体比例", "快速冷却"],
        notes_zh="双相不锈钢，需严格控制热输入和层间温度",
        notes_en="Duplex SS, strict heat input and interpass control required",
    ),
    # Aluminum alloys
    "6061": BaseMaterial(
        material_id="6061",
        material_name_zh="6061铝合金",
        material_name_en="6061 Aluminum Alloy",
        weldability=WeldabilityClass.GOOD,
        preheat_required=False,
        preheat_temp_min=0,
        preheat_temp_max=150,
        interpass_temp_max=120,
        special_requirements=["清除氧化膜", "使用高纯氩气", "焊后可时效处理"],
        notes_zh="热处理型铝合金，焊后热影响区强度下降",
        notes_en="Heat treatable aluminum, HAZ softening occurs",
    ),
    "5052": BaseMaterial(
        material_id="5052",
        material_name_zh="5052铝合金",
        material_name_en="5052 Aluminum Alloy",
        weldability=WeldabilityClass.EXCELLENT,
        preheat_required=False,
        interpass_temp_max=120,
        special_requirements=["清除氧化膜"],
        notes_zh="非热处理型铝合金，焊接性优良",
        notes_en="Non-heat treatable aluminum, excellent weldability",
    ),
    "7075": BaseMaterial(
        material_id="7075",
        material_name_zh="7075高强铝合金",
        material_name_en="7075 High Strength Aluminum",
        weldability=WeldabilityClass.DIFFICULT,
        preheat_required=True,
        preheat_temp_min=150,
        preheat_temp_max=200,
        special_requirements=["易产生热裂纹", "不推荐熔焊", "建议机械连接或搅拌摩擦焊"],
        notes_zh="高强度铝合金，常规熔焊困难，易热裂",
        notes_en="High strength aluminum, fusion welding difficult, prone to hot cracking",
    ),
    # Tool steels
    "Cr12MoV": BaseMaterial(
        material_id="Cr12MoV",
        material_name_zh="Cr12MoV冷作模具钢",
        material_name_en="D2 Tool Steel",
        weldability=WeldabilityClass.DIFFICULT,
        preheat_required=True,
        preheat_temp_min=300,
        preheat_temp_max=400,
        pwht_required=True,
        pwht_temp_range=(200, 250),
        special_requirements=["必须高温预热", "焊后缓冷", "建议堆焊修复"],
        notes_zh="高碳高铬工具钢，焊接性极差",
        notes_en="High carbon high chromium tool steel, very poor weldability",
    ),
}

# Carbon equivalent formula coefficients
CE_FORMULA_IIW = {
    "C": 1.0,
    "Mn": 1 / 6,
    "Cr": 1 / 5,
    "Mo": 1 / 5,
    "V": 1 / 5,
    "Ni": 1 / 15,
    "Cu": 1 / 15,
}

# Preheat temperature based on CE and thickness
PREHEAT_CHART: Dict[Tuple[float, float], Dict[Tuple[float, float], int]] = {
    # CE range: {thickness range: preheat temp}
    (0.0, 0.30): {
        (0, 20): 0,
        (20, 40): 0,
        (40, 60): 50,
        (60, 100): 75,
    },
    (0.30, 0.40): {
        (0, 20): 0,
        (20, 40): 50,
        (40, 60): 100,
        (60, 100): 150,
    },
    (0.40, 0.50): {
        (0, 20): 50,
        (20, 40): 100,
        (40, 60): 150,
        (60, 100): 200,
    },
    (0.50, 0.60): {
        (0, 20): 100,
        (20, 40): 150,
        (40, 60): 200,
        (60, 100): 250,
    },
    (0.60, 1.0): {
        (0, 20): 150,
        (20, 40): 200,
        (40, 60): 250,
        (60, 100): 300,
    },
}


def get_weldability(material_id: str) -> Optional[BaseMaterial]:
    """
    Get weldability information for a material.

    Args:
        material_id: Material identifier

    Returns:
        BaseMaterial or None

    Example:
        >>> mat = get_weldability("Q345")
        >>> print(f"Weldability: {mat.weldability.value}")
    """
    return WELDABILITY_DATABASE.get(material_id)


def calculate_carbon_equivalent(
    composition: Dict[str, float],
    formula: str = "IIW",
) -> float:
    """
    Calculate carbon equivalent from chemical composition.

    Args:
        composition: Dict of element percentages (e.g., {"C": 0.20, "Mn": 1.2})
        formula: Formula to use ("IIW" or "Pcm")

    Returns:
        Carbon equivalent value

    Example:
        >>> ce = calculate_carbon_equivalent({"C": 0.18, "Mn": 1.4, "Cr": 0.2})
        >>> print(f"CE(IIW) = {ce:.3f}")
    """
    if formula.upper() == "IIW":
        # CE(IIW) = C + Mn/6 + (Cr+Mo+V)/5 + (Ni+Cu)/15
        C = composition.get("C", 0)
        Mn = composition.get("Mn", 0)
        Cr = composition.get("Cr", 0)
        Mo = composition.get("Mo", 0)
        V = composition.get("V", 0)
        Ni = composition.get("Ni", 0)
        Cu = composition.get("Cu", 0)

        ce = C + Mn / 6 + (Cr + Mo + V) / 5 + (Ni + Cu) / 15
    elif formula.upper() == "PCM":
        # Pcm = C + Si/30 + (Mn+Cu+Cr)/20 + Ni/60 + Mo/15 + V/10 + 5B
        C = composition.get("C", 0)
        Si = composition.get("Si", 0)
        Mn = composition.get("Mn", 0)
        Cu = composition.get("Cu", 0)
        Cr = composition.get("Cr", 0)
        Ni = composition.get("Ni", 0)
        Mo = composition.get("Mo", 0)
        V = composition.get("V", 0)
        B = composition.get("B", 0)

        ce = C + Si / 30 + (Mn + Cu + Cr) / 20 + Ni / 60 + Mo / 15 + V / 10 + 5 * B
    else:
        ce = composition.get("C", 0)

    return round(ce, 3)


def get_preheat_temperature(
    material_id: Optional[str] = None,
    carbon_equivalent: Optional[float] = None,
    thickness: float = 20,
    hydrogen_level: str = "low",
) -> Tuple[int, int]:
    """
    Get recommended preheat temperature.

    Args:
        material_id: Material identifier (optional if CE provided)
        carbon_equivalent: Carbon equivalent value (optional if material_id provided)
        thickness: Material thickness (mm)
        hydrogen_level: "low", "medium", or "high"

    Returns:
        Tuple of (min_temp, max_temp) in °C

    Example:
        >>> temp = get_preheat_temperature("Q345", thickness=30)
        >>> print(f"Preheat: {temp[0]}-{temp[1]}°C")
    """
    ce = carbon_equivalent

    # Get CE from material database if not provided
    if ce is None and material_id:
        mat = WELDABILITY_DATABASE.get(material_id)
        if mat and mat.carbon_equivalent:
            ce = mat.carbon_equivalent
        elif mat and mat.preheat_temp_min is not None:
            return (int(mat.preheat_temp_min), int(mat.preheat_temp_max or mat.preheat_temp_min + 50))

    if ce is None:
        return (0, 0)

    # Find preheat from chart
    preheat = 0
    for (ce_min, ce_max), thickness_temps in PREHEAT_CHART.items():
        if ce_min <= ce < ce_max:
            for (t_min, t_max), temp in thickness_temps.items():
                if t_min < thickness <= t_max:
                    preheat = temp
                    break
            break

    # Adjust for hydrogen level
    hydrogen_adjustments = {
        "low": 0,  # H4 (≤4 ml/100g)
        "medium": 25,  # H8
        "high": 50,  # H16
    }
    adjustment = hydrogen_adjustments.get(hydrogen_level, 0)
    preheat += adjustment

    # Return range
    return (preheat, min(preheat + 50, 400))


def check_material_compatibility(
    material_1: str,
    material_2: str,
) -> Dict:
    """
    Check compatibility of two materials for welding.

    Args:
        material_1: First material identifier
        material_2: Second material identifier

    Returns:
        Dict with compatibility assessment
    """
    mat1 = WELDABILITY_DATABASE.get(material_1)
    mat2 = WELDABILITY_DATABASE.get(material_2)

    result = {
        "compatible": True,
        "filler_recommendation": None,
        "preheat_required": False,
        "preheat_temp": 0,
        "special_requirements": [],
        "notes": [],
    }

    if not mat1 or not mat2:
        result["notes"].append("材料信息不完整")
        return result

    # Same material - straightforward
    if material_1 == material_2:
        result["notes"].append("同种材料焊接")
        result["preheat_required"] = mat1.preheat_required
        if mat1.preheat_temp_min:
            result["preheat_temp"] = mat1.preheat_temp_min
        return result

    # Check for incompatible combinations
    incompatible_pairs = [
        ("aluminum", "steel"),  # Generic
        ("7075", "steel"),  # Specific
    ]

    for pair in incompatible_pairs:
        if (material_1 in pair and material_2 in pair):
            result["compatible"] = False
            result["notes"].append("这两种材料不能直接熔焊")
            return result

    # Dissimilar metal welding
    result["notes"].append("异种金属焊接")

    # Use higher preheat requirement
    preheat1 = mat1.preheat_temp_min or 0
    preheat2 = mat2.preheat_temp_min or 0
    if preheat1 > 0 or preheat2 > 0:
        result["preheat_required"] = True
        result["preheat_temp"] = max(preheat1, preheat2)

    # Combine special requirements
    reqs = []
    if mat1.special_requirements:
        reqs.extend(mat1.special_requirements)
    if mat2.special_requirements:
        reqs.extend(mat2.special_requirements)
    result["special_requirements"] = list(set(reqs))

    # Filler recommendation for common dissimilar combinations
    dissimilar_fillers = {
        ("304", "carbon_steel"): "ER309L",
        ("316L", "carbon_steel"): "ER309LMo",
        ("Q235", "Q345"): "ER70S-6",
    }

    pair_key = (material_1, material_2)
    reverse_key = (material_2, material_1)
    filler = dissimilar_fillers.get(pair_key) or dissimilar_fillers.get(reverse_key)
    if filler:
        result["filler_recommendation"] = filler

    return result


def get_pwht_parameters(material_id: str) -> Optional[Dict]:
    """
    Get post-weld heat treatment parameters.

    Args:
        material_id: Material identifier

    Returns:
        Dict with PWHT parameters or None
    """
    mat = WELDABILITY_DATABASE.get(material_id)
    if not mat or not mat.pwht_required:
        return None

    return {
        "required": True,
        "temperature_range": mat.pwht_temp_range,
        "time_per_inch": mat.pwht_time_per_inch or 1.0,
        "heating_rate_max": 220,  # °C/hour typical
        "cooling_rate_max": 280,  # °C/hour typical
        "notes_zh": f"{mat.material_name_zh}焊后热处理",
    }
