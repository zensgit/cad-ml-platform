"""
Workpiece Material Machinability Knowledge Base.

Provides machinability ratings and cutting characteristics for common
engineering materials to guide parameter selection.

Reference:
- ISO 3685:1993 - Tool-life testing
- Machinery's Handbook - Machinability ratings
- AISI machinability ratings (B1112 = 100%)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class MachinabilityClass(str, Enum):
    """Material machinability classification."""

    EXCELLENT = "excellent"  # 优良 (>100%)
    GOOD = "good"  # 良好 (70-100%)
    FAIR = "fair"  # 一般 (40-70%)
    POOR = "poor"  # 较差 (20-40%)
    DIFFICULT = "difficult"  # 困难 (<20%)


@dataclass
class WorkpieceMaterial:
    """Workpiece material cutting characteristics."""

    material_group: str  # e.g., "carbon_steel", "stainless_steel"
    name_zh: str
    name_en: str

    # Machinability
    machinability_rating: float  # Percentage (AISI B1112 = 100%)
    machinability_class: MachinabilityClass

    # Hardness range
    hardness_min: int  # HB
    hardness_max: int  # HB

    # Cutting characteristics
    chip_formation: str  # "continuous", "segmented", "discontinuous"
    built_up_edge_tendency: str  # "low", "medium", "high"
    work_hardening: str  # "low", "medium", "high"
    abrasiveness: str  # "low", "medium", "high"

    # Recommended cutting conditions
    cutting_speed_factor: float  # Multiplier for base cutting speed
    feed_factor: float  # Multiplier for base feed rate

    # Coolant recommendation
    coolant_required: bool
    coolant_type: str  # "flood", "mist", "dry", "MQL"


# Machinability database organized by ISO material groups
# Based on ISO 513 material classification
MACHINABILITY_DATABASE: Dict[str, WorkpieceMaterial] = {
    # P - Steel (ISO P)
    "low_carbon_steel": WorkpieceMaterial(
        material_group="P",
        name_zh="低碳钢",
        name_en="Low carbon steel",
        machinability_rating=70,
        machinability_class=MachinabilityClass.GOOD,
        hardness_min=120,
        hardness_max=180,
        chip_formation="continuous",
        built_up_edge_tendency="high",
        work_hardening="low",
        abrasiveness="low",
        cutting_speed_factor=1.0,
        feed_factor=1.0,
        coolant_required=True,
        coolant_type="flood",
    ),
    "medium_carbon_steel": WorkpieceMaterial(
        material_group="P",
        name_zh="中碳钢",
        name_en="Medium carbon steel",
        machinability_rating=65,
        machinability_class=MachinabilityClass.FAIR,
        hardness_min=180,
        hardness_max=250,
        chip_formation="continuous",
        built_up_edge_tendency="medium",
        work_hardening="low",
        abrasiveness="low",
        cutting_speed_factor=0.9,
        feed_factor=1.0,
        coolant_required=True,
        coolant_type="flood",
    ),
    "high_carbon_steel": WorkpieceMaterial(
        material_group="P",
        name_zh="高碳钢",
        name_en="High carbon steel",
        machinability_rating=50,
        machinability_class=MachinabilityClass.FAIR,
        hardness_min=200,
        hardness_max=300,
        chip_formation="segmented",
        built_up_edge_tendency="low",
        work_hardening="low",
        abrasiveness="medium",
        cutting_speed_factor=0.7,
        feed_factor=0.9,
        coolant_required=True,
        coolant_type="flood",
    ),
    "alloy_steel": WorkpieceMaterial(
        material_group="P",
        name_zh="合金钢",
        name_en="Alloy steel",
        machinability_rating=55,
        machinability_class=MachinabilityClass.FAIR,
        hardness_min=200,
        hardness_max=350,
        chip_formation="continuous",
        built_up_edge_tendency="medium",
        work_hardening="medium",
        abrasiveness="medium",
        cutting_speed_factor=0.75,
        feed_factor=0.9,
        coolant_required=True,
        coolant_type="flood",
    ),
    "free_machining_steel": WorkpieceMaterial(
        material_group="P",
        name_zh="易切削钢",
        name_en="Free machining steel",
        machinability_rating=100,
        machinability_class=MachinabilityClass.EXCELLENT,
        hardness_min=150,
        hardness_max=220,
        chip_formation="discontinuous",
        built_up_edge_tendency="low",
        work_hardening="low",
        abrasiveness="low",
        cutting_speed_factor=1.3,
        feed_factor=1.2,
        coolant_required=False,
        coolant_type="dry",
    ),
    "tool_steel_annealed": WorkpieceMaterial(
        material_group="P",
        name_zh="工具钢(退火态)",
        name_en="Tool steel (annealed)",
        machinability_rating=40,
        machinability_class=MachinabilityClass.FAIR,
        hardness_min=180,
        hardness_max=250,
        chip_formation="continuous",
        built_up_edge_tendency="medium",
        work_hardening="medium",
        abrasiveness="medium",
        cutting_speed_factor=0.6,
        feed_factor=0.8,
        coolant_required=True,
        coolant_type="flood",
    ),
    "tool_steel_hardened": WorkpieceMaterial(
        material_group="H",
        name_zh="工具钢(淬硬态)",
        name_en="Tool steel (hardened)",
        machinability_rating=15,
        machinability_class=MachinabilityClass.DIFFICULT,
        hardness_min=450,
        hardness_max=650,
        chip_formation="segmented",
        built_up_edge_tendency="low",
        work_hardening="low",
        abrasiveness="high",
        cutting_speed_factor=0.2,
        feed_factor=0.5,
        coolant_required=True,
        coolant_type="flood",
    ),

    # M - Stainless Steel (ISO M)
    "austenitic_stainless": WorkpieceMaterial(
        material_group="M",
        name_zh="奥氏体不锈钢",
        name_en="Austenitic stainless steel",
        machinability_rating=45,
        machinability_class=MachinabilityClass.FAIR,
        hardness_min=150,
        hardness_max=250,
        chip_formation="continuous",
        built_up_edge_tendency="high",
        work_hardening="high",
        abrasiveness="medium",
        cutting_speed_factor=0.5,
        feed_factor=0.8,
        coolant_required=True,
        coolant_type="flood",
    ),
    "ferritic_stainless": WorkpieceMaterial(
        material_group="M",
        name_zh="铁素体不锈钢",
        name_en="Ferritic stainless steel",
        machinability_rating=55,
        machinability_class=MachinabilityClass.FAIR,
        hardness_min=150,
        hardness_max=220,
        chip_formation="continuous",
        built_up_edge_tendency="medium",
        work_hardening="low",
        abrasiveness="low",
        cutting_speed_factor=0.65,
        feed_factor=0.9,
        coolant_required=True,
        coolant_type="flood",
    ),
    "martensitic_stainless": WorkpieceMaterial(
        material_group="M",
        name_zh="马氏体不锈钢",
        name_en="Martensitic stainless steel",
        machinability_rating=50,
        machinability_class=MachinabilityClass.FAIR,
        hardness_min=200,
        hardness_max=350,
        chip_formation="segmented",
        built_up_edge_tendency="medium",
        work_hardening="medium",
        abrasiveness="medium",
        cutting_speed_factor=0.55,
        feed_factor=0.85,
        coolant_required=True,
        coolant_type="flood",
    ),
    "duplex_stainless": WorkpieceMaterial(
        material_group="M",
        name_zh="双相不锈钢",
        name_en="Duplex stainless steel",
        machinability_rating=35,
        machinability_class=MachinabilityClass.POOR,
        hardness_min=250,
        hardness_max=350,
        chip_formation="segmented",
        built_up_edge_tendency="high",
        work_hardening="high",
        abrasiveness="medium",
        cutting_speed_factor=0.4,
        feed_factor=0.7,
        coolant_required=True,
        coolant_type="flood",
    ),

    # K - Cast Iron (ISO K)
    "gray_cast_iron": WorkpieceMaterial(
        material_group="K",
        name_zh="灰铸铁",
        name_en="Gray cast iron",
        machinability_rating=80,
        machinability_class=MachinabilityClass.GOOD,
        hardness_min=150,
        hardness_max=250,
        chip_formation="discontinuous",
        built_up_edge_tendency="low",
        work_hardening="low",
        abrasiveness="high",
        cutting_speed_factor=1.1,
        feed_factor=1.1,
        coolant_required=False,
        coolant_type="dry",
    ),
    "ductile_cast_iron": WorkpieceMaterial(
        material_group="K",
        name_zh="球墨铸铁",
        name_en="Ductile cast iron",
        machinability_rating=60,
        machinability_class=MachinabilityClass.FAIR,
        hardness_min=150,
        hardness_max=300,
        chip_formation="segmented",
        built_up_edge_tendency="low",
        work_hardening="low",
        abrasiveness="high",
        cutting_speed_factor=0.85,
        feed_factor=1.0,
        coolant_required=False,
        coolant_type="dry",
    ),

    # N - Non-ferrous (ISO N)
    "aluminum_wrought": WorkpieceMaterial(
        material_group="N",
        name_zh="变形铝合金",
        name_en="Wrought aluminum alloy",
        machinability_rating=300,
        machinability_class=MachinabilityClass.EXCELLENT,
        hardness_min=30,
        hardness_max=150,
        chip_formation="continuous",
        built_up_edge_tendency="high",
        work_hardening="low",
        abrasiveness="low",
        cutting_speed_factor=3.0,
        feed_factor=1.5,
        coolant_required=True,
        coolant_type="flood",
    ),
    "aluminum_cast": WorkpieceMaterial(
        material_group="N",
        name_zh="铸造铝合金",
        name_en="Cast aluminum alloy",
        machinability_rating=250,
        machinability_class=MachinabilityClass.EXCELLENT,
        hardness_min=50,
        hardness_max=130,
        chip_formation="discontinuous",
        built_up_edge_tendency="medium",
        work_hardening="low",
        abrasiveness="medium",
        cutting_speed_factor=2.5,
        feed_factor=1.3,
        coolant_required=True,
        coolant_type="flood",
    ),
    "copper_alloy": WorkpieceMaterial(
        material_group="N",
        name_zh="铜合金",
        name_en="Copper alloy",
        machinability_rating=90,
        machinability_class=MachinabilityClass.GOOD,
        hardness_min=50,
        hardness_max=200,
        chip_formation="continuous",
        built_up_edge_tendency="high",
        work_hardening="medium",
        abrasiveness="low",
        cutting_speed_factor=1.2,
        feed_factor=1.1,
        coolant_required=True,
        coolant_type="flood",
    ),
    "brass": WorkpieceMaterial(
        material_group="N",
        name_zh="黄铜",
        name_en="Brass",
        machinability_rating=120,
        machinability_class=MachinabilityClass.EXCELLENT,
        hardness_min=60,
        hardness_max=150,
        chip_formation="discontinuous",
        built_up_edge_tendency="low",
        work_hardening="low",
        abrasiveness="low",
        cutting_speed_factor=1.5,
        feed_factor=1.2,
        coolant_required=False,
        coolant_type="dry",
    ),
    "bronze": WorkpieceMaterial(
        material_group="N",
        name_zh="青铜",
        name_en="Bronze",
        machinability_rating=80,
        machinability_class=MachinabilityClass.GOOD,
        hardness_min=80,
        hardness_max=200,
        chip_formation="discontinuous",
        built_up_edge_tendency="medium",
        work_hardening="low",
        abrasiveness="medium",
        cutting_speed_factor=1.0,
        feed_factor=1.0,
        coolant_required=True,
        coolant_type="flood",
    ),

    # S - Heat Resistant Alloys (ISO S)
    "titanium_alloy": WorkpieceMaterial(
        material_group="S",
        name_zh="钛合金",
        name_en="Titanium alloy",
        machinability_rating=25,
        machinability_class=MachinabilityClass.POOR,
        hardness_min=300,
        hardness_max=400,
        chip_formation="segmented",
        built_up_edge_tendency="high",
        work_hardening="high",
        abrasiveness="high",
        cutting_speed_factor=0.25,
        feed_factor=0.6,
        coolant_required=True,
        coolant_type="flood",
    ),
    "nickel_alloy": WorkpieceMaterial(
        material_group="S",
        name_zh="镍基合金",
        name_en="Nickel-based alloy",
        machinability_rating=20,
        machinability_class=MachinabilityClass.DIFFICULT,
        hardness_min=250,
        hardness_max=450,
        chip_formation="continuous",
        built_up_edge_tendency="high",
        work_hardening="high",
        abrasiveness="high",
        cutting_speed_factor=0.2,
        feed_factor=0.5,
        coolant_required=True,
        coolant_type="flood",
    ),
    "cobalt_alloy": WorkpieceMaterial(
        material_group="S",
        name_zh="钴基合金",
        name_en="Cobalt-based alloy",
        machinability_rating=18,
        machinability_class=MachinabilityClass.DIFFICULT,
        hardness_min=300,
        hardness_max=500,
        chip_formation="segmented",
        built_up_edge_tendency="high",
        work_hardening="high",
        abrasiveness="high",
        cutting_speed_factor=0.18,
        feed_factor=0.45,
        coolant_required=True,
        coolant_type="flood",
    ),
}


def get_machinability(material_key: str) -> Optional[WorkpieceMaterial]:
    """
    Get machinability data for a material.

    Args:
        material_key: Material identifier (e.g., "austenitic_stainless")

    Returns:
        WorkpieceMaterial data or None if not found

    Example:
        >>> mat = get_machinability("aluminum_wrought")
        >>> print(f"Rating: {mat.machinability_rating}%")
    """
    return MACHINABILITY_DATABASE.get(material_key.lower())


def get_material_cutting_data(
    material_key: str,
) -> Optional[Dict]:
    """
    Get cutting-relevant data for a material.

    Args:
        material_key: Material identifier

    Returns:
        Dictionary with cutting factors and recommendations
    """
    mat = get_machinability(material_key)
    if mat is None:
        return None

    return {
        "material": material_key,
        "iso_group": mat.material_group,
        "machinability_rating": mat.machinability_rating,
        "machinability_class": mat.machinability_class.value,
        "cutting_speed_factor": mat.cutting_speed_factor,
        "feed_factor": mat.feed_factor,
        "coolant": {
            "required": mat.coolant_required,
            "type": mat.coolant_type,
        },
        "characteristics": {
            "chip_formation": mat.chip_formation,
            "built_up_edge": mat.built_up_edge_tendency,
            "work_hardening": mat.work_hardening,
            "abrasiveness": mat.abrasiveness,
        },
    }


def list_materials_by_group(iso_group: str) -> List[str]:
    """
    List all materials in an ISO material group.

    Args:
        iso_group: ISO 513 material group (P, M, K, N, S, H)

    Returns:
        List of material keys in the group
    """
    return [
        key for key, mat in MACHINABILITY_DATABASE.items()
        if mat.material_group == iso_group.upper()
    ]


def get_materials_by_machinability(
    min_rating: float = 0,
    max_rating: float = 500,
) -> List[str]:
    """
    Find materials within a machinability rating range.

    Args:
        min_rating: Minimum machinability rating (%)
        max_rating: Maximum machinability rating (%)

    Returns:
        List of material keys sorted by rating
    """
    results = [
        (key, mat.machinability_rating)
        for key, mat in MACHINABILITY_DATABASE.items()
        if min_rating <= mat.machinability_rating <= max_rating
    ]
    return [key for key, _ in sorted(results, key=lambda x: -x[1])]
