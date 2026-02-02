"""
Anodizing Process Knowledge Base.

Provides anodizing parameters, color specifications, and process
recommendations for aluminum parts.

Reference:
- MIL-A-8625 - Anodic coatings for aluminum
- ISO 7599 - Anodizing of aluminum
- AMS 2468 - Hard anodic coating
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class AnodizingType(str, Enum):
    """Anodizing process types."""

    TYPE_I = "type_i"  # 铬酸阳极氧化 (Chromic acid)
    TYPE_II = "type_ii"  # 硫酸阳极氧化 (Sulfuric acid) - decorative
    TYPE_III = "type_iii"  # 硬质阳极氧化 (Hard anodize)

    # Sub-types
    TYPE_IB = "type_ib"  # Thin chromic acid
    TYPE_IC = "type_ic"  # Non-chromic acid (替代Type I)
    TYPE_IIB = "type_iib"  # Thin sulfuric acid

    # Special
    PHOSPHORIC = "phosphoric"  # 磷酸阳极氧化 (adhesive bonding prep)


class AnodizeClass(str, Enum):
    """Anodize classification per MIL-A-8625."""

    CLASS_1 = "class_1"  # Non-dyed (natural)
    CLASS_2 = "class_2"  # Dyed


@dataclass
class AnodizingParameters:
    """Anodizing process parameters."""

    anodize_type: AnodizingType
    anodize_class: AnodizeClass

    # Thickness
    thickness_min: float  # μm
    thickness_max: float  # μm
    thickness_typical: float  # μm

    # Process parameters
    voltage_min: float  # V
    voltage_max: float  # V
    voltage_typical: float  # V

    current_density_min: float  # A/dm²
    current_density_max: float  # A/dm²
    current_density_typical: float  # A/dm²

    temperature_min: float  # °C
    temperature_max: float  # °C
    temperature_typical: float  # °C

    process_time_min: float  # minutes
    process_time_max: float  # minutes

    # Solution
    acid_concentration: Tuple[float, float]  # g/L or %
    acid_type: str

    # Properties
    hardness_hv: Optional[Tuple[int, int]] = None
    wear_resistance: Optional[str] = None
    corrosion_resistance: Optional[str] = None

    # Sealing
    sealing_required: bool = True
    sealing_methods: List[str] = None

    notes_zh: str = ""
    notes_en: str = ""


# Anodizing database
ANODIZING_DATABASE: Dict[AnodizingType, Dict] = {
    AnodizingType.TYPE_I: {
        "thickness_range": (0.5, 7),
        "thickness_typical": 2.5,
        "voltage": (20, 50, 40),
        "current_density": (0.3, 1.0, 0.5),
        "temperature": (32, 43, 38),
        "process_time": (30, 60),
        "acid_type": "铬酸",
        "acid_concentration": (30, 100),  # g/L
        "sealing_required": True,
        "sealing_methods": ["热水封孔", "铬酸封孔"],
        "notes_zh": "Type I铬酸阳极氧化，薄层，适用于疲劳敏感零件",
        "notes_en": "Type I chromic acid anodize, thin film, fatigue-sensitive parts",
    },
    AnodizingType.TYPE_II: {
        "thickness_range": (5, 25),
        "thickness_typical": 15,
        "voltage": (12, 22, 18),
        "current_density": (1.2, 2.0, 1.5),
        "temperature": (18, 24, 21),
        "process_time": (20, 60),
        "acid_type": "硫酸",
        "acid_concentration": (150, 200),  # g/L
        "hardness_hv": (200, 400),
        "sealing_required": True,
        "sealing_methods": ["热水封孔", "镍盐封孔", "冷封孔"],
        "notes_zh": "Type II硫酸阳极氧化，最常用，可染色",
        "notes_en": "Type II sulfuric acid anodize, most common, dyeable",
    },
    AnodizingType.TYPE_IIB: {
        "thickness_range": (2.5, 7.6),
        "thickness_typical": 5,
        "voltage": (12, 22, 15),
        "current_density": (1.0, 1.8, 1.2),
        "temperature": (18, 24, 21),
        "process_time": (10, 30),
        "acid_type": "硫酸",
        "acid_concentration": (150, 200),
        "sealing_required": True,
        "notes_zh": "Type IIB薄层硫酸阳极氧化，尺寸敏感零件",
        "notes_en": "Type IIB thin sulfuric acid anodize, dimension-sensitive parts",
    },
    AnodizingType.TYPE_III: {
        "thickness_range": (25, 125),
        "thickness_typical": 50,
        "voltage": (30, 100, 60),
        "current_density": (2.0, 4.0, 3.0),
        "temperature": (-5, 5, 0),  # Near freezing
        "process_time": (60, 180),
        "acid_type": "硫酸 (低温)",
        "acid_concentration": (180, 250),
        "hardness_hv": (400, 700),
        "wear_resistance": "优秀",
        "sealing_required": False,  # Often left unsealed for wear
        "sealing_methods": ["热水封孔", "PTFE封孔"],
        "notes_zh": "Type III硬质阳极氧化，耐磨耐腐蚀",
        "notes_en": "Type III hard anodize, wear and corrosion resistant",
    },
    AnodizingType.TYPE_IC: {
        "thickness_range": (0.5, 7),
        "thickness_typical": 2.5,
        "voltage": (15, 25, 20),
        "current_density": (0.5, 1.5, 1.0),
        "temperature": (20, 30, 25),
        "process_time": (20, 40),
        "acid_type": "磷酸/硼酸/硫酸混合",
        "acid_concentration": None,
        "sealing_required": True,
        "notes_zh": "Type IC非铬酸，环保替代Type I",
        "notes_en": "Type IC non-chromic, environmentally friendly alternative to Type I",
    },
    AnodizingType.PHOSPHORIC: {
        "thickness_range": (3, 5),
        "thickness_typical": 4,
        "voltage": (10, 20, 15),
        "current_density": (0.5, 1.5, 1.0),
        "temperature": (20, 30, 25),
        "process_time": (20, 30),
        "acid_type": "磷酸",
        "acid_concentration": (100, 150),
        "sealing_required": False,
        "notes_zh": "磷酸阳极氧化，胶接前处理",
        "notes_en": "Phosphoric acid anodize, adhesive bonding preparation",
    },
}

# Available anodize colors
ANODIZE_COLORS: Dict[str, Dict[str, Any]] = {
    "natural": {
        "name_zh": "本色/银白色",
        "name_en": "Natural/Clear",
        "dye_required": False,
        "compatible_types": [AnodizingType.TYPE_I, AnodizingType.TYPE_II, AnodizingType.TYPE_III],
    },
    "black": {
        "name_zh": "黑色",
        "name_en": "Black",
        "dye_required": True,
        "dye_type": "有机染料或无机盐",
        "compatible_types": [AnodizingType.TYPE_II],
        "notes": "最常见的装饰色，遮盖性好",
    },
    "red": {
        "name_zh": "红色",
        "name_en": "Red",
        "dye_required": True,
        "dye_type": "有机染料",
        "compatible_types": [AnodizingType.TYPE_II],
    },
    "blue": {
        "name_zh": "蓝色",
        "name_en": "Blue",
        "dye_required": True,
        "dye_type": "有机染料",
        "compatible_types": [AnodizingType.TYPE_II],
    },
    "gold": {
        "name_zh": "金色",
        "name_en": "Gold",
        "dye_required": True,
        "dye_type": "有机染料或电解着色",
        "compatible_types": [AnodizingType.TYPE_II],
    },
    "bronze": {
        "name_zh": "古铜色",
        "name_en": "Bronze",
        "dye_required": True,
        "dye_type": "电解着色（锡盐）",
        "compatible_types": [AnodizingType.TYPE_II],
        "notes": "电解着色耐候性优于有机染料",
    },
    "hard_gray": {
        "name_zh": "硬质灰色",
        "name_en": "Hard Gray",
        "dye_required": False,
        "compatible_types": [AnodizingType.TYPE_III],
        "notes": "硬质阳极氧化自然色，取决于合金成分",
    },
}


def get_anodizing_parameters(
    anodize_type: Union[str, AnodizingType],
    anodize_class: AnodizeClass = AnodizeClass.CLASS_1,
) -> Optional[AnodizingParameters]:
    """
    Get anodizing parameters.

    Args:
        anodize_type: Type of anodizing
        anodize_class: Class (natural or dyed)

    Returns:
        AnodizingParameters or None

    Example:
        >>> params = get_anodizing_parameters("type_ii")
        >>> print(f"Thickness: {params.thickness_typical}μm")
    """
    if isinstance(anodize_type, str):
        try:
            anodize_type = AnodizingType(anodize_type.lower())
        except ValueError:
            return None

    data = ANODIZING_DATABASE.get(anodize_type)
    if not data:
        return None

    thickness_range = data.get("thickness_range", (5, 25))
    voltage = data.get("voltage", (15, 25, 20))
    current_density = data.get("current_density", (1, 2, 1.5))
    temperature = data.get("temperature", (18, 24, 21))
    process_time = data.get("process_time", (20, 60))

    return AnodizingParameters(
        anodize_type=anodize_type,
        anodize_class=anodize_class,
        thickness_min=thickness_range[0],
        thickness_max=thickness_range[1],
        thickness_typical=data.get("thickness_typical", sum(thickness_range) / 2),
        voltage_min=voltage[0],
        voltage_max=voltage[1],
        voltage_typical=voltage[2],
        current_density_min=current_density[0],
        current_density_max=current_density[1],
        current_density_typical=current_density[2],
        temperature_min=temperature[0],
        temperature_max=temperature[1],
        temperature_typical=temperature[2],
        process_time_min=process_time[0],
        process_time_max=process_time[1],
        acid_concentration=data.get("acid_concentration", (0, 0)),
        acid_type=data.get("acid_type", ""),
        hardness_hv=data.get("hardness_hv"),
        wear_resistance=data.get("wear_resistance"),
        sealing_required=data.get("sealing_required", True),
        sealing_methods=data.get("sealing_methods"),
        notes_zh=data.get("notes_zh", ""),
        notes_en=data.get("notes_en", ""),
    )


def get_anodizing_colors(
    anodize_type: Union[str, AnodizingType] = AnodizingType.TYPE_II,
) -> List[Dict[str, Any]]:
    """
    Get available colors for an anodize type.

    Args:
        anodize_type: Type of anodizing

    Returns:
        List of available colors
    """
    if isinstance(anodize_type, str):
        try:
            anodize_type = AnodizingType(anodize_type.lower())
        except ValueError:
            return []

    available = []
    for color_id, color_data in ANODIZE_COLORS.items():
        if anodize_type in color_data.get("compatible_types", []):
            available.append({
                "color_id": color_id,
                "name_zh": color_data["name_zh"],
                "name_en": color_data["name_en"],
                "dye_required": color_data.get("dye_required", False),
            })

    return available


def recommend_anodizing_for_application(
    application: str,
    requirements: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Recommend anodizing type for an application.

    Args:
        application: Application description
        requirements: List of requirements

    Returns:
        List of recommendations
    """
    requirements = requirements or []
    recommendations = []

    if "wear" in requirements or "耐磨" in requirements:
        recommendations.append({
            "type": AnodizingType.TYPE_III,
            "reason": "Type III硬质阳极氧化，硬度400-700HV",
            "priority": 1,
        })

    if "fatigue" in requirements or "疲劳" in requirements:
        recommendations.append({
            "type": AnodizingType.TYPE_I,
            "reason": "Type I薄层，对疲劳性能影响最小",
            "priority": 1,
        })
        recommendations.append({
            "type": AnodizingType.TYPE_IC,
            "reason": "Type IC非铬酸，环保且适用于疲劳敏感零件",
            "priority": 2,
        })

    if "color" in requirements or "颜色" in requirements or "装饰" in requirements:
        recommendations.append({
            "type": AnodizingType.TYPE_II,
            "reason": "Type II可染色，装饰性最佳",
            "priority": 1,
        })

    if "bonding" in requirements or "胶接" in requirements:
        recommendations.append({
            "type": AnodizingType.PHOSPHORIC,
            "reason": "磷酸阳极氧化，胶接前处理最佳",
            "priority": 1,
        })

    if "dimension" in requirements or "尺寸" in requirements:
        recommendations.append({
            "type": AnodizingType.TYPE_IIB,
            "reason": "Type IIB薄层，尺寸变化最小",
            "priority": 1,
        })

    # Default
    if not recommendations:
        recommendations.append({
            "type": AnodizingType.TYPE_II,
            "reason": "Type II最通用，平衡性能和成本",
            "priority": 1,
        })

    recommendations.sort(key=lambda x: x["priority"])
    return recommendations


def calculate_dimension_change(
    thickness: float,
    anodize_type: AnodizingType = AnodizingType.TYPE_II,
) -> Dict[str, Any]:
    """
    Calculate dimension change from anodizing.

    Anodizing grows approximately 50% into the base material
    and 50% outward for Type II. Type III grows more outward.

    Args:
        thickness: Anodize thickness (μm)
        anodize_type: Type of anodizing

    Returns:
        Dict with dimension change
    """
    # Growth ratios (outward growth / total thickness)
    growth_ratios = {
        AnodizingType.TYPE_I: 0.33,  # 1/3 out, 2/3 in
        AnodizingType.TYPE_II: 0.50,  # 1/2 out, 1/2 in
        AnodizingType.TYPE_IIB: 0.50,
        AnodizingType.TYPE_III: 0.67,  # 2/3 out, 1/3 in
    }

    ratio = growth_ratios.get(anodize_type, 0.5)
    outward_growth = thickness * ratio

    return {
        "anodize_thickness_um": thickness,
        "outward_growth_um": round(outward_growth, 1),
        "inward_penetration_um": round(thickness - outward_growth, 1),
        "diameter_increase_um": round(outward_growth * 2, 1),  # Both sides
        "notes": f"外径增加约 {round(outward_growth * 2, 1)}μm",
    }
