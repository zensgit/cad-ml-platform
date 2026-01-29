"""
Electroplating Process Knowledge Base.

Provides electroplating parameters, thickness specifications, and
process recommendations for common plating types.

Reference:
- ISO 2081 - Electroplated coatings of zinc
- ISO 1456 - Electroplated coatings of nickel plus chromium
- ASTM B633 - Electrodeposited coatings of zinc
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union


class PlatingType(str, Enum):
    """Electroplating types."""

    # Zinc plating
    ZINC_CLEAR = "zinc_clear"  # 镀锌白钝化
    ZINC_BLUE = "zinc_blue"  # 镀锌蓝白钝化
    ZINC_YELLOW = "zinc_yellow"  # 镀锌彩钝化
    ZINC_BLACK = "zinc_black"  # 镀锌黑钝化
    ZINC_NICKEL = "zinc_nickel"  # 锌镍合金镀

    # Nickel plating
    NICKEL_BRIGHT = "nickel_bright"  # 光亮镍
    NICKEL_SATIN = "nickel_satin"  # 缎面镍
    NICKEL_SULFAMATE = "nickel_sulfamate"  # 氨基磺酸镍

    # Chrome plating
    CHROME_DECORATIVE = "chrome_decorative"  # 装饰性镀铬
    CHROME_HARD = "chrome_hard"  # 硬铬

    # Other
    COPPER = "copper"  # 镀铜
    TIN = "tin"  # 镀锡
    SILVER = "silver"  # 镀银
    GOLD = "gold"  # 镀金
    CADMIUM = "cadmium"  # 镀镉

    # Conversion coating
    PHOSPHATE = "phosphate"  # 磷化
    CHROMATE = "chromate"  # 铬酸盐处理


class SubstrateType(str, Enum):
    """Base material types for plating."""

    STEEL = "steel"
    STAINLESS_STEEL = "stainless_steel"
    ALUMINUM = "aluminum"
    COPPER = "copper"
    BRASS = "brass"
    ZINC_DIE_CAST = "zinc_die_cast"


@dataclass
class PlatingParameters:
    """Electroplating process parameters."""

    plating_type: PlatingType
    substrate: SubstrateType

    # Thickness
    thickness_min: float  # μm
    thickness_max: float  # μm
    thickness_typical: float  # μm

    # Process parameters
    current_density_min: float  # A/dm²
    current_density_max: float  # A/dm²
    current_density_typical: float  # A/dm²

    temperature_min: float  # °C
    temperature_max: float  # °C
    temperature_typical: float  # °C

    plating_time_per_um: float  # minutes per μm

    # Solution
    bath_type: str
    ph_range: Optional[Tuple[float, float]] = None

    # Properties
    hardness_hv: Optional[Tuple[int, int]] = None
    corrosion_resistance_hours: Optional[int] = None  # Salt spray hours

    notes_zh: str = ""
    notes_en: str = ""


# Plating database
PLATING_DATABASE: Dict[PlatingType, Dict] = {
    # Zinc plating
    PlatingType.ZINC_CLEAR: {
        "thickness_range": (5, 25),
        "thickness_typical": 8,
        "current_density": (1, 4, 2),
        "temperature": (20, 35, 25),
        "plating_rate": 2.5,  # min/μm
        "bath_type": "酸性氯化物",
        "ph_range": (4.5, 5.5),
        "salt_spray_hours": 24,
        "notes_zh": "镀锌白钝化，经济型防腐",
        "notes_en": "Zinc clear chromate, economical corrosion protection",
    },
    PlatingType.ZINC_BLUE: {
        "thickness_range": (5, 25),
        "thickness_typical": 10,
        "current_density": (1, 4, 2),
        "temperature": (20, 35, 25),
        "plating_rate": 2.5,
        "bath_type": "酸性氯化物",
        "ph_range": (4.5, 5.5),
        "salt_spray_hours": 48,
        "notes_zh": "镀锌蓝白钝化，RoHS兼容",
        "notes_en": "Zinc blue chromate, RoHS compliant",
    },
    PlatingType.ZINC_YELLOW: {
        "thickness_range": (8, 25),
        "thickness_typical": 12,
        "current_density": (1, 4, 2),
        "temperature": (20, 35, 25),
        "plating_rate": 2.5,
        "bath_type": "酸性氯化物",
        "ph_range": (4.5, 5.5),
        "salt_spray_hours": 96,
        "notes_zh": "镀锌彩钝化，较好防腐性能",
        "notes_en": "Zinc yellow chromate, better corrosion protection",
    },
    PlatingType.ZINC_NICKEL: {
        "thickness_range": (8, 20),
        "thickness_typical": 12,
        "current_density": (1, 5, 3),
        "temperature": (25, 35, 30),
        "plating_rate": 3.0,
        "bath_type": "碱性或酸性",
        "nickel_content": (12, 16),  # %
        "salt_spray_hours": 720,
        "hardness_hv": (350, 450),
        "notes_zh": "锌镍合金，汽车工业首选",
        "notes_en": "Zinc-nickel alloy, automotive industry preferred",
    },
    # Nickel plating
    PlatingType.NICKEL_BRIGHT: {
        "thickness_range": (5, 40),
        "thickness_typical": 15,
        "current_density": (2, 8, 4),
        "temperature": (50, 60, 55),
        "plating_rate": 1.5,
        "bath_type": "瓦特镍",
        "ph_range": (3.8, 4.6),
        "hardness_hv": (400, 600),
        "notes_zh": "光亮镀镍，装饰和耐磨",
        "notes_en": "Bright nickel, decorative and wear resistant",
    },
    PlatingType.NICKEL_SULFAMATE: {
        "thickness_range": (25, 500),
        "thickness_typical": 100,
        "current_density": (2, 15, 8),
        "temperature": (40, 55, 50),
        "plating_rate": 0.8,
        "bath_type": "氨基磺酸镍",
        "ph_range": (3.5, 4.5),
        "hardness_hv": (200, 350),
        "notes_zh": "氨基磺酸镍，低应力厚镀层",
        "notes_en": "Sulfamate nickel, low stress thick deposits",
    },
    # Chrome plating
    PlatingType.CHROME_DECORATIVE: {
        "thickness_range": (0.25, 1.0),
        "thickness_typical": 0.5,
        "current_density": (10, 30, 20),
        "temperature": (40, 55, 45),
        "plating_rate": 15,
        "bath_type": "六价铬",
        "notes_zh": "装饰性镀铬，薄镀层外观",
        "notes_en": "Decorative chrome, thin layer for appearance",
    },
    PlatingType.CHROME_HARD: {
        "thickness_range": (20, 500),
        "thickness_typical": 50,
        "current_density": (30, 60, 45),
        "temperature": (50, 65, 55),
        "plating_rate": 0.5,
        "bath_type": "六价铬",
        "hardness_hv": (800, 1000),
        "notes_zh": "硬铬，耐磨和修复",
        "notes_en": "Hard chrome, wear resistance and repair",
    },
    # Copper plating
    PlatingType.COPPER: {
        "thickness_range": (10, 50),
        "thickness_typical": 25,
        "current_density": (2, 6, 4),
        "temperature": (20, 40, 30),
        "plating_rate": 2.0,
        "bath_type": "酸性硫酸铜",
        "ph_range": (0, 1),
        "notes_zh": "镀铜，常作底层或导电层",
        "notes_en": "Copper plating, often as undercoat or conductive layer",
    },
    # Tin plating
    PlatingType.TIN: {
        "thickness_range": (5, 25),
        "thickness_typical": 10,
        "current_density": (1, 3, 2),
        "temperature": (20, 35, 25),
        "plating_rate": 3.0,
        "bath_type": "酸性或碱性",
        "notes_zh": "镀锡，食品级和可焊性",
        "notes_en": "Tin plating, food-safe and solderability",
    },
    # Phosphate
    PlatingType.PHOSPHATE: {
        "thickness_range": (3, 25),
        "thickness_typical": 10,
        "current_density": None,  # Chemical process
        "temperature": (50, 95, 70),
        "plating_rate": 5.0,
        "bath_type": "锌系/锰系/铁系磷化",
        "notes_zh": "磷化处理，涂装底层或防锈",
        "notes_en": "Phosphating, paint adhesion or rust prevention",
    },
}

# Thickness specifications per ISO 2081 / ASTM B633
THICKNESS_SPECIFICATIONS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "Fe/Zn": {  # ISO 2081 zinc on steel
        "Fe/Zn 5": (5, None),
        "Fe/Zn 8": (8, None),
        "Fe/Zn 12": (12, None),
        "Fe/Zn 25": (25, None),
    },
    "ASTM_B633": {  # ASTM B633 classification
        "Fe/Zn 5 (SC1)": (5, None),  # Mild indoor
        "Fe/Zn 8 (SC2)": (8, None),  # Moderate indoor
        "Fe/Zn 12 (SC3)": (12, None),  # Severe indoor
        "Fe/Zn 25 (SC4)": (25, None),  # Outdoor
    },
}


def get_plating_parameters(
    plating_type: Union[str, PlatingType],
    substrate: Union[str, SubstrateType] = SubstrateType.STEEL,
) -> Optional[PlatingParameters]:
    """
    Get electroplating parameters.

    Args:
        plating_type: Type of plating
        substrate: Base material

    Returns:
        PlatingParameters or None

    Example:
        >>> params = get_plating_parameters("zinc_yellow")
        >>> print(f"Thickness: {params.thickness_typical}μm")
    """
    if isinstance(plating_type, str):
        try:
            plating_type = PlatingType(plating_type.lower())
        except ValueError:
            return None

    if isinstance(substrate, str):
        try:
            substrate = SubstrateType(substrate.lower())
        except ValueError:
            substrate = SubstrateType.STEEL

    data = PLATING_DATABASE.get(plating_type)
    if not data:
        return None

    thickness_range = data.get("thickness_range", (5, 25))
    current_density = data.get("current_density", (1, 5, 3))
    temperature = data.get("temperature", (20, 40, 30))

    return PlatingParameters(
        plating_type=plating_type,
        substrate=substrate,
        thickness_min=thickness_range[0],
        thickness_max=thickness_range[1],
        thickness_typical=data.get("thickness_typical", sum(thickness_range) / 2),
        current_density_min=current_density[0] if current_density else 0,
        current_density_max=current_density[1] if current_density else 0,
        current_density_typical=current_density[2] if current_density else 0,
        temperature_min=temperature[0],
        temperature_max=temperature[1],
        temperature_typical=temperature[2],
        plating_time_per_um=data.get("plating_rate", 2.0),
        bath_type=data.get("bath_type", ""),
        ph_range=data.get("ph_range"),
        hardness_hv=data.get("hardness_hv"),
        corrosion_resistance_hours=data.get("salt_spray_hours"),
        notes_zh=data.get("notes_zh", ""),
        notes_en=data.get("notes_en", ""),
    )


def get_plating_thickness(
    application: str,
    environment: str = "indoor",
) -> Dict:
    """
    Get recommended plating thickness for application.

    Args:
        application: Application type (fastener, bracket, housing, etc.)
        environment: Environment (indoor, outdoor, marine, etc.)

    Returns:
        Dict with thickness recommendations
    """
    # Application-based recommendations
    recommendations = {
        "fastener": {
            "indoor": ("Fe/Zn 5", 5, "室内紧固件"),
            "outdoor": ("Fe/Zn 12", 12, "室外紧固件"),
            "marine": ("Zn-Ni", 15, "海洋环境紧固件"),
        },
        "bracket": {
            "indoor": ("Fe/Zn 8", 8, "室内支架"),
            "outdoor": ("Fe/Zn 25", 25, "室外支架"),
            "marine": ("Zn-Ni + sealant", 15, "海洋环境支架"),
        },
        "housing": {
            "indoor": ("Ni 15", 15, "室内壳体"),
            "outdoor": ("Ni 25 + Cr", 25, "室外壳体"),
        },
        "electrical": {
            "default": ("Sn 10", 10, "电气连接镀锡"),
            "high_reliability": ("Au 1-3", 2, "高可靠性镀金"),
        },
    }

    app_rec = recommendations.get(application, {})
    env_rec = app_rec.get(environment, app_rec.get("default"))

    if env_rec:
        return {
            "specification": env_rec[0],
            "thickness_um": env_rec[1],
            "description": env_rec[2],
        }

    return {
        "specification": "Fe/Zn 8",
        "thickness_um": 8,
        "description": "标准通用镀层",
    }


def recommend_plating_for_application(
    application: str,
    requirements: List[str] = None,
) -> List[Dict]:
    """
    Recommend plating types for an application.

    Args:
        application: Application description
        requirements: List of requirements (corrosion, wear, appearance, etc.)

    Returns:
        List of recommendations with rationale
    """
    requirements = requirements or []
    recommendations = []

    # Requirement-based recommendations
    if "corrosion" in requirements or "防腐" in requirements:
        recommendations.append({
            "plating": PlatingType.ZINC_NICKEL,
            "reason": "锌镍合金提供最佳防腐性能，盐雾试验>720小时",
            "priority": 1,
        })
        recommendations.append({
            "plating": PlatingType.ZINC_YELLOW,
            "reason": "镀锌彩钝化，经济的防腐方案",
            "priority": 2,
        })

    if "wear" in requirements or "耐磨" in requirements:
        recommendations.append({
            "plating": PlatingType.CHROME_HARD,
            "reason": "硬铬镀层，硬度800-1000HV",
            "priority": 1,
        })
        recommendations.append({
            "plating": PlatingType.NICKEL_SULFAMATE,
            "reason": "氨基磺酸镍，中等耐磨性",
            "priority": 2,
        })

    if "appearance" in requirements or "外观" in requirements:
        recommendations.append({
            "plating": PlatingType.CHROME_DECORATIVE,
            "reason": "装饰性镀铬，光亮外观",
            "priority": 1,
        })
        recommendations.append({
            "plating": PlatingType.NICKEL_BRIGHT,
            "reason": "光亮镀镍，良好外观",
            "priority": 2,
        })

    if "solderable" in requirements or "可焊" in requirements:
        recommendations.append({
            "plating": PlatingType.TIN,
            "reason": "镀锡，优秀可焊性",
            "priority": 1,
        })

    if "conductive" in requirements or "导电" in requirements:
        recommendations.append({
            "plating": PlatingType.SILVER,
            "reason": "镀银，最佳导电性",
            "priority": 1,
        })
        recommendations.append({
            "plating": PlatingType.COPPER,
            "reason": "镀铜，良好导电性和成本效益",
            "priority": 2,
        })

    # Default recommendation
    if not recommendations:
        recommendations.append({
            "plating": PlatingType.ZINC_BLUE,
            "reason": "镀锌蓝白钝化，RoHS兼容通用方案",
            "priority": 1,
        })

    # Sort by priority
    recommendations.sort(key=lambda x: x["priority"])
    return recommendations


def calculate_plating_time(
    thickness: float,
    plating_type: Union[str, PlatingType],
    area: float = 1.0,
) -> Dict:
    """
    Calculate plating time.

    Args:
        thickness: Target thickness (μm)
        plating_type: Type of plating
        area: Surface area (dm²)

    Returns:
        Dict with time estimate
    """
    params = get_plating_parameters(plating_type)
    if not params:
        return {"error": "Unknown plating type"}

    base_time = thickness * params.plating_time_per_um

    # Adjust for area (larger area = slightly longer time due to current distribution)
    area_factor = 1 + (area - 1) * 0.1 if area > 1 else 1

    total_time = base_time * area_factor

    return {
        "plating_type": plating_type if isinstance(plating_type, str) else plating_type.value,
        "thickness_um": thickness,
        "area_dm2": area,
        "plating_time_minutes": round(total_time, 1),
        "plating_time_hours": round(total_time / 60, 2),
    }
