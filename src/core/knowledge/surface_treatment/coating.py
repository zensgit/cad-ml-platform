"""
Surface Coating Knowledge Base.

Provides coating parameters, specifications, and recommendations
for paints, powder coatings, and other surface treatments.

Reference:
- ISO 12944 - Corrosion protection by protective paint systems
- ASTM D3359 - Adhesion testing
- ISO 2808 - Paint film thickness measurement
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class CoatingType(str, Enum):
    """Surface coating types."""

    # Liquid paints
    EPOXY = "epoxy"  # 环氧漆
    POLYURETHANE = "polyurethane"  # 聚氨酯漆
    ALKYD = "alkyd"  # 醇酸漆
    ACRYLIC = "acrylic"  # 丙烯酸漆
    ZINC_RICH = "zinc_rich"  # 富锌底漆

    # Powder coatings
    POWDER_EPOXY = "powder_epoxy"  # 环氧粉末
    POWDER_POLYESTER = "powder_polyester"  # 聚酯粉末
    POWDER_HYBRID = "powder_hybrid"  # 环氧聚酯混合
    POWDER_POLYURETHANE = "powder_polyurethane"  # 聚氨酯粉末
    POWDER_NYLON = "powder_nylon"  # 尼龙粉末

    # Specialty
    PTFE = "ptfe"  # 特氟龙涂层
    CERAMIC = "ceramic"  # 陶瓷涂层
    ZINC_FLAKE = "zinc_flake"  # 锌铝涂层 (达克罗)
    E_COAT = "e_coat"  # 电泳涂装


class EnvironmentClass(str, Enum):
    """Corrosivity categories per ISO 12944."""

    C1 = "C1"  # Very low - heated interiors
    C2 = "C2"  # Low - unheated interiors
    C3 = "C3"  # Medium - urban/industrial atmospheres
    C4 = "C4"  # High - industrial/coastal
    C5_I = "C5-I"  # Very high industrial
    C5_M = "C5-M"  # Very high marine
    CX = "CX"  # Extreme - offshore


@dataclass
class CoatingParameters:
    """Coating process parameters."""

    coating_type: CoatingType

    # Thickness
    dft_min: float  # Dry film thickness (μm)
    dft_max: float  # Dry film thickness (μm)
    dft_recommended: float  # μm

    # Application
    application_method: List[str]  # spray, brush, dip, etc.
    coats_required: int
    recoat_interval_min: float  # hours
    recoat_interval_max: float  # hours

    # Curing
    cure_temperature: Optional[Tuple[float, float]] = None  # °C
    cure_time: Optional[Tuple[float, float]] = None  # minutes
    air_dry_time: Optional[float] = None  # hours for handling

    # Properties
    hardness: Optional[str] = None  # pencil hardness or similar
    gloss_range: Optional[Tuple[int, int]] = None  # gloss units
    salt_spray_hours: Optional[int] = None

    # Substrate prep
    surface_prep_required: str = ""

    notes_zh: str = ""
    notes_en: str = ""


# Coating database
COATING_DATABASE: Dict[CoatingType, Dict] = {
    # Liquid paints
    CoatingType.EPOXY: {
        "dft_range": (40, 125),
        "dft_recommended": 80,
        "application_methods": ["喷涂", "刷涂", "滚涂"],
        "coats": 2,
        "recoat_interval": (4, 48),
        "air_dry_time": 8,
        "hardness": "2H-4H",
        "salt_spray_hours": 500,
        "surface_prep": "Sa2.5喷砂或磷化",
        "notes_zh": "环氧漆，优秀的附着力和耐化学性",
        "notes_en": "Epoxy paint, excellent adhesion and chemical resistance",
    },
    CoatingType.POLYURETHANE: {
        "dft_range": (30, 75),
        "dft_recommended": 50,
        "application_methods": ["喷涂"],
        "coats": 2,
        "recoat_interval": (2, 24),
        "air_dry_time": 4,
        "hardness": "H-2H",
        "gloss_range": (80, 95),
        "salt_spray_hours": 750,
        "surface_prep": "环氧底漆",
        "notes_zh": "聚氨酯面漆，优秀的耐候性和光泽保持",
        "notes_en": "Polyurethane topcoat, excellent weathering and gloss retention",
    },
    CoatingType.ZINC_RICH: {
        "dft_range": (60, 100),
        "dft_recommended": 75,
        "application_methods": ["喷涂"],
        "coats": 1,
        "recoat_interval": (24, 72),
        "air_dry_time": 24,
        "zinc_content": (85, 95),  # % in dry film
        "salt_spray_hours": 1000,
        "surface_prep": "Sa2.5喷砂",
        "notes_zh": "富锌底漆，阴极保护",
        "notes_en": "Zinc-rich primer, cathodic protection",
    },
    # Powder coatings
    CoatingType.POWDER_EPOXY: {
        "dft_range": (60, 100),
        "dft_recommended": 75,
        "application_methods": ["静电喷涂", "流化床"],
        "coats": 1,
        "cure_temperature": (180, 200),
        "cure_time": (10, 20),
        "hardness": "2H-4H",
        "salt_spray_hours": 500,
        "surface_prep": "磷化或喷砂",
        "notes_zh": "环氧粉末，室内用，耐化学性好",
        "notes_en": "Epoxy powder, indoor use, good chemical resistance",
    },
    CoatingType.POWDER_POLYESTER: {
        "dft_range": (60, 100),
        "dft_recommended": 70,
        "application_methods": ["静电喷涂"],
        "coats": 1,
        "cure_temperature": (180, 200),
        "cure_time": (10, 15),
        "hardness": "H-2H",
        "gloss_range": (30, 90),
        "salt_spray_hours": 750,
        "surface_prep": "磷化",
        "notes_zh": "聚酯粉末，室外用，耐候性好",
        "notes_en": "Polyester powder, outdoor use, good weathering",
    },
    CoatingType.POWDER_HYBRID: {
        "dft_range": (50, 80),
        "dft_recommended": 65,
        "application_methods": ["静电喷涂"],
        "coats": 1,
        "cure_temperature": (180, 200),
        "cure_time": (10, 15),
        "hardness": "H-2H",
        "salt_spray_hours": 400,
        "surface_prep": "磷化",
        "notes_zh": "环氧聚酯混合，室内装饰用",
        "notes_en": "Epoxy-polyester hybrid, indoor decorative",
    },
    # Specialty
    CoatingType.ZINC_FLAKE: {
        "dft_range": (8, 15),
        "dft_recommended": 10,
        "application_methods": ["浸涂+旋转", "喷涂"],
        "coats": 2,
        "cure_temperature": (300, 340),
        "cure_time": (20, 30),
        "salt_spray_hours": 720,
        "surface_prep": "喷砂或碱洗",
        "notes_zh": "锌铝涂层(达克罗)，紧固件防腐",
        "notes_en": "Zinc flake coating (Dacromet/Geomet), fastener corrosion protection",
    },
    CoatingType.E_COAT: {
        "dft_range": (15, 35),
        "dft_recommended": 25,
        "application_methods": ["电泳"],
        "coats": 1,
        "cure_temperature": (170, 190),
        "cure_time": (20, 30),
        "salt_spray_hours": 500,
        "surface_prep": "磷化",
        "notes_zh": "电泳涂装，汽车零件底漆",
        "notes_en": "E-coat, automotive primer",
    },
    CoatingType.PTFE: {
        "dft_range": (15, 50),
        "dft_recommended": 25,
        "application_methods": ["喷涂"],
        "coats": 2,
        "cure_temperature": (370, 400),
        "cure_time": (10, 20),
        "friction_coefficient": 0.05,
        "max_service_temp": 260,
        "surface_prep": "喷砂+底漆",
        "notes_zh": "特氟龙涂层，不粘和低摩擦",
        "notes_en": "PTFE coating, non-stick and low friction",
    },
}

# Environment to coating system recommendations (ISO 12944)
CORROSIVITY_RECOMMENDATIONS: Dict[EnvironmentClass, Dict] = {
    EnvironmentClass.C1: {
        "min_dft": 80,
        "system": "一道环氧底漆",
        "expected_life": "15年以上",
        "notes_zh": "室内加热环境，最低要求",
    },
    EnvironmentClass.C2: {
        "min_dft": 120,
        "system": "环氧底漆+醇酸面漆",
        "expected_life": "15年以上",
        "notes_zh": "室内非加热环境",
    },
    EnvironmentClass.C3: {
        "min_dft": 200,
        "system": "环氧底漆+聚氨酯面漆",
        "expected_life": "15年以上",
        "notes_zh": "城市/轻工业环境",
    },
    EnvironmentClass.C4: {
        "min_dft": 280,
        "system": "富锌底漆+环氧中间漆+聚氨酯面漆",
        "expected_life": "15年以上",
        "notes_zh": "工业/沿海环境",
    },
    EnvironmentClass.C5_I: {
        "min_dft": 320,
        "system": "富锌底漆+环氧中间漆+聚氨酯面漆",
        "expected_life": "15年以上",
        "notes_zh": "重工业环境",
    },
    EnvironmentClass.C5_M: {
        "min_dft": 320,
        "system": "富锌底漆+环氧中间漆+聚氨酯面漆",
        "expected_life": "15年以上",
        "notes_zh": "海洋环境",
    },
    EnvironmentClass.CX: {
        "min_dft": 400,
        "system": "富锌底漆+厚浆环氧+聚氨酯面漆",
        "expected_life": "15年以上",
        "notes_zh": "海上平台等极端环境",
    },
}


def get_coating_parameters(
    coating_type: Union[str, CoatingType],
) -> Optional[CoatingParameters]:
    """
    Get coating parameters.

    Args:
        coating_type: Type of coating

    Returns:
        CoatingParameters or None

    Example:
        >>> params = get_coating_parameters("powder_polyester")
        >>> print(f"DFT: {params.dft_recommended}μm")
    """
    if isinstance(coating_type, str):
        try:
            coating_type = CoatingType(coating_type.lower())
        except ValueError:
            return None

    data = COATING_DATABASE.get(coating_type)
    if not data:
        return None

    dft_range = data.get("dft_range", (50, 100))
    recoat = data.get("recoat_interval", (4, 24))

    return CoatingParameters(
        coating_type=coating_type,
        dft_min=dft_range[0],
        dft_max=dft_range[1],
        dft_recommended=data.get("dft_recommended", sum(dft_range) / 2),
        application_method=data.get("application_methods", ["喷涂"]),
        coats_required=data.get("coats", 1),
        recoat_interval_min=recoat[0],
        recoat_interval_max=recoat[1],
        cure_temperature=data.get("cure_temperature"),
        cure_time=data.get("cure_time"),
        air_dry_time=data.get("air_dry_time"),
        hardness=data.get("hardness"),
        gloss_range=data.get("gloss_range"),
        salt_spray_hours=data.get("salt_spray_hours"),
        surface_prep_required=data.get("surface_prep", ""),
        notes_zh=data.get("notes_zh", ""),
        notes_en=data.get("notes_en", ""),
    )


def get_coating_for_environment(
    environment: Union[str, EnvironmentClass],
    design_life: str = "medium",
) -> Dict[str, Any]:
    """
    Get coating system recommendation for environment.

    Args:
        environment: Corrosivity category (C1-CX)
        design_life: "low" (<7yr), "medium" (7-15yr), "high" (>15yr)

    Returns:
        Dict with coating system recommendation
    """
    if isinstance(environment, str):
        try:
            environment = EnvironmentClass(environment.upper())
        except ValueError:
            environment = EnvironmentClass.C3

    rec = CORROSIVITY_RECOMMENDATIONS.get(environment, {})

    # Adjust DFT for design life
    life_factors = {
        "low": 0.7,
        "medium": 1.0,
        "high": 1.2,
    }
    factor = life_factors.get(design_life, 1.0)

    return {
        "environment": environment.value,
        "design_life": design_life,
        "min_dft_um": round(rec.get("min_dft", 200) * factor),
        "recommended_system": rec.get("system", ""),
        "expected_life": rec.get("expected_life", ""),
        "notes_zh": rec.get("notes_zh", ""),
    }


def calculate_coating_life(
    coating_type: Union[str, CoatingType],
    environment: Union[str, EnvironmentClass],
    dft: float,
) -> Dict[str, Any]:
    """
    Estimate coating service life.

    Args:
        coating_type: Type of coating
        environment: Corrosivity category
        dft: Dry film thickness (μm)

    Returns:
        Dict with life estimate
    """
    params = get_coating_parameters(coating_type)
    if not params:
        return {"error": "Unknown coating type"}

    if isinstance(environment, str):
        try:
            environment = EnvironmentClass(environment.upper())
        except ValueError:
            environment = EnvironmentClass.C3

    # Base life estimation (simplified model)
    base_life_years = {
        EnvironmentClass.C1: 25,
        EnvironmentClass.C2: 20,
        EnvironmentClass.C3: 15,
        EnvironmentClass.C4: 10,
        EnvironmentClass.C5_I: 7,
        EnvironmentClass.C5_M: 5,
        EnvironmentClass.CX: 3,
    }

    base = base_life_years.get(environment, 10)

    # Adjust for DFT (more thickness = longer life)
    dft_factor = dft / params.dft_recommended

    # Adjust for coating type
    type_factors = {
        CoatingType.POLYURETHANE: 1.2,
        CoatingType.POWDER_POLYESTER: 1.1,
        CoatingType.ZINC_RICH: 1.3,
        CoatingType.ZINC_FLAKE: 1.2,
        CoatingType.EPOXY: 0.9,  # Good but chalks outdoors
    }
    type_factor = type_factors.get(params.coating_type, 1.0)

    estimated_life = base * dft_factor * type_factor

    return {
        "coating_type": params.coating_type.value,
        "environment": environment.value,
        "dft_um": dft,
        "estimated_life_years": round(estimated_life, 1),
        "notes": "估算值，实际寿命取决于施工质量和维护",
    }


def recommend_coating_system(
    substrate: str,
    environment: str,
    requirements: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Recommend complete coating system.

    Args:
        substrate: Base material (steel, aluminum, etc.)
        environment: Operating environment
        requirements: Special requirements (color, wear, chemical, etc.)

    Returns:
        List of coating system recommendations
    """
    requirements = requirements or []
    recommendations = []

    # Steel substrates
    if substrate.lower() in ["steel", "钢"]:
        if "outdoor" in environment or "室外" in environment:
            recommendations.append({
                "system": [
                    {"layer": "底漆", "type": CoatingType.ZINC_RICH, "dft": 75},
                    {"layer": "中间漆", "type": CoatingType.EPOXY, "dft": 100},
                    {"layer": "面漆", "type": CoatingType.POLYURETHANE, "dft": 50},
                ],
                "total_dft": 225,
                "reason": "室外钢结构标准防腐系统",
            })
        else:
            recommendations.append({
                "system": [
                    {"layer": "底漆", "type": CoatingType.EPOXY, "dft": 40},
                    {"layer": "面漆", "type": CoatingType.POWDER_HYBRID, "dft": 65},
                ],
                "total_dft": 105,
                "reason": "室内钢件标准系统",
            })

    # Aluminum substrates
    elif substrate.lower() in ["aluminum", "铝"]:
        recommendations.append({
            "system": [
                {"layer": "预处理", "type": "铬酸盐/无铬转化膜", "dft": 0},
                {"layer": "粉末涂层", "type": CoatingType.POWDER_POLYESTER, "dft": 70},
            ],
            "total_dft": 70,
            "reason": "铝件粉末涂装标准系统",
        })

    # Fasteners
    if "fastener" in requirements or "紧固件" in requirements:
        recommendations.append({
            "system": [
                {"layer": "锌铝涂层", "type": CoatingType.ZINC_FLAKE, "dft": 10},
            ],
            "total_dft": 10,
            "reason": "紧固件达克罗涂层",
        })

    # Default
    if not recommendations:
        recommendations.append({
            "system": [
                {"layer": "粉末涂层", "type": CoatingType.POWDER_HYBRID, "dft": 65},
            ],
            "total_dft": 65,
            "reason": "通用粉末涂装",
        })

    return recommendations
