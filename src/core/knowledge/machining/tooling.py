"""
Cutting Tool Knowledge Base.

Provides tool material recommendations, geometry guidelines, and tool selection
assistance for common machining operations.

Reference:
- ISO 513:2012 - Classification of tool materials
- Tool manufacturer catalogs (Sandvik, Kennametal, etc.)
- Machinery's Handbook
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class ToolMaterial(str, Enum):
    """Cutting tool material classification per ISO 513."""

    # Carbide grades
    CARBIDE_P = "carbide_P"  # For steel (P group)
    CARBIDE_M = "carbide_M"  # For stainless (M group)
    CARBIDE_K = "carbide_K"  # For cast iron (K group)
    CARBIDE_N = "carbide_N"  # For non-ferrous (N group)
    CARBIDE_S = "carbide_S"  # For superalloys (S group)
    CARBIDE_H = "carbide_H"  # For hardened steel (H group)

    # Coated carbide
    COATED_CVD = "coated_CVD"  # CVD coated carbide
    COATED_PVD = "coated_PVD"  # PVD coated carbide

    # Cermet
    CERMET = "cermet"

    # Ceramic
    CERAMIC_OXIDE = "ceramic_oxide"  # Al2O3 based
    CERAMIC_NITRIDE = "ceramic_nitride"  # Si3N4 based
    CERAMIC_MIXED = "ceramic_mixed"  # Mixed ceramic

    # Superhard
    CBN = "CBN"  # Cubic boron nitride
    PCD = "PCD"  # Polycrystalline diamond

    # HSS
    HSS = "HSS"  # High speed steel
    HSS_COBALT = "HSS_cobalt"  # Cobalt HSS (M35, M42)
    HSS_PM = "HSS_PM"  # Powder metallurgy HSS


class ToolType(str, Enum):
    """Cutting tool type."""

    # Turning tools
    TURNING_EXTERNAL = "turning_external"
    TURNING_INTERNAL = "turning_internal"
    TURNING_FACING = "turning_facing"
    TURNING_GROOVING = "turning_grooving"
    TURNING_THREADING = "turning_threading"
    TURNING_PARTING = "turning_parting"

    # Milling tools
    MILLING_FACE = "milling_face"
    MILLING_END = "milling_end"
    MILLING_BALL = "milling_ball"
    MILLING_SLOT = "milling_slot"
    MILLING_SHOULDER = "milling_shoulder"

    # Drilling tools
    DRILL_TWIST = "drill_twist"
    DRILL_INDEXABLE = "drill_indexable"
    DRILL_GUN = "drill_gun"
    DRILL_CENTER = "drill_center"

    # Boring tools
    BORING_ROUGH = "boring_rough"
    BORING_FINISH = "boring_finish"

    # Reaming
    REAMER = "reamer"

    # Threading
    TAP = "tap"
    THREAD_MILL = "thread_mill"


@dataclass
class ToolGeometry:
    """Recommended tool geometry parameters."""

    rake_angle: float  # Rake angle (degrees)
    relief_angle: float  # Relief/clearance angle (degrees)
    nose_radius: float  # Nose radius (mm)
    edge_preparation: str  # "sharp", "honed", "chamfered", "rounded"

    # For indexable inserts
    insert_shape: Optional[str] = None  # "C", "D", "R", "S", "T", "V", "W"
    insert_size: Optional[str] = None  # IC size in mm


@dataclass
class ToolRecommendation:
    """Tool recommendation for a specific application."""

    tool_material: ToolMaterial
    tool_type: ToolType
    geometry: ToolGeometry

    # Operating parameters
    cutting_speed_range: tuple  # (min, max) m/min
    feed_range: tuple  # (min, max) mm/rev or mm/tooth
    depth_of_cut_range: tuple  # (min, max) mm

    # Additional recommendations
    coolant: str  # "flood", "mist", "dry", "MQL"
    notes_zh: str
    notes_en: str

    suitability: float  # 0.0 to 1.0


# Tool material recommendations by workpiece material group
# Key: (workpiece_group, operation_type): [ToolMaterial priority list]
TOOL_MATERIAL_MATRIX: Dict[tuple, List[ToolMaterial]] = {
    # Steel (P)
    ("P", "roughing"): [ToolMaterial.COATED_CVD, ToolMaterial.CARBIDE_P, ToolMaterial.CERMET],
    ("P", "finishing"): [ToolMaterial.CERMET, ToolMaterial.COATED_PVD, ToolMaterial.CARBIDE_P],
    ("P", "drilling"): [ToolMaterial.COATED_PVD, ToolMaterial.HSS_COBALT, ToolMaterial.CARBIDE_P],

    # Stainless (M)
    ("M", "roughing"): [ToolMaterial.COATED_PVD, ToolMaterial.CARBIDE_M, ToolMaterial.CERMET],
    ("M", "finishing"): [ToolMaterial.COATED_PVD, ToolMaterial.CERMET, ToolMaterial.CARBIDE_M],
    ("M", "drilling"): [ToolMaterial.COATED_PVD, ToolMaterial.HSS_COBALT, ToolMaterial.CARBIDE_M],

    # Cast iron (K)
    ("K", "roughing"): [ToolMaterial.CERAMIC_NITRIDE, ToolMaterial.COATED_CVD, ToolMaterial.CARBIDE_K],
    ("K", "finishing"): [ToolMaterial.CBN, ToolMaterial.CERAMIC_MIXED, ToolMaterial.COATED_CVD],
    ("K", "drilling"): [ToolMaterial.COATED_CVD, ToolMaterial.CARBIDE_K, ToolMaterial.HSS],

    # Non-ferrous (N)
    ("N", "roughing"): [ToolMaterial.PCD, ToolMaterial.CARBIDE_N, ToolMaterial.COATED_PVD],
    ("N", "finishing"): [ToolMaterial.PCD, ToolMaterial.CARBIDE_N, ToolMaterial.COATED_PVD],
    ("N", "drilling"): [ToolMaterial.CARBIDE_N, ToolMaterial.HSS, ToolMaterial.COATED_PVD],

    # Superalloys (S)
    ("S", "roughing"): [ToolMaterial.CERAMIC_NITRIDE, ToolMaterial.COATED_PVD, ToolMaterial.CARBIDE_S],
    ("S", "finishing"): [ToolMaterial.CBN, ToolMaterial.CERAMIC_MIXED, ToolMaterial.COATED_PVD],
    ("S", "drilling"): [ToolMaterial.COATED_PVD, ToolMaterial.CARBIDE_S, ToolMaterial.HSS_COBALT],

    # Hardened steel (H)
    ("H", "roughing"): [ToolMaterial.CBN, ToolMaterial.CERAMIC_MIXED, ToolMaterial.CERAMIC_OXIDE],
    ("H", "finishing"): [ToolMaterial.CBN, ToolMaterial.CERAMIC_MIXED, ToolMaterial.CERAMIC_OXIDE],
    ("H", "drilling"): [ToolMaterial.CARBIDE_H, ToolMaterial.HSS_PM, ToolMaterial.COATED_PVD],
}

# Tool geometry recommendations by material group
GEOMETRY_RECOMMENDATIONS: Dict[str, ToolGeometry] = {
    "P_general": ToolGeometry(
        rake_angle=6,
        relief_angle=7,
        nose_radius=0.8,
        edge_preparation="honed",
        insert_shape="C",
        insert_size="12",
    ),
    "M_general": ToolGeometry(
        rake_angle=10,
        relief_angle=8,
        nose_radius=0.8,
        edge_preparation="sharp",
        insert_shape="C",
        insert_size="12",
    ),
    "K_general": ToolGeometry(
        rake_angle=0,
        relief_angle=6,
        nose_radius=1.2,
        edge_preparation="chamfered",
        insert_shape="S",
        insert_size="12",
    ),
    "N_general": ToolGeometry(
        rake_angle=15,
        relief_angle=10,
        nose_radius=0.4,
        edge_preparation="sharp",
        insert_shape="D",
        insert_size="09",
    ),
    "S_general": ToolGeometry(
        rake_angle=6,
        relief_angle=8,
        nose_radius=0.8,
        edge_preparation="honed",
        insert_shape="R",
        insert_size="12",
    ),
    "H_general": ToolGeometry(
        rake_angle=-6,
        relief_angle=6,
        nose_radius=0.8,
        edge_preparation="chamfered",
        insert_shape="C",
        insert_size="12",
    ),
}

# Comprehensive tool database
TOOL_DATABASE: Dict[str, Dict] = {
    # Turning inserts
    "CNMG120408": {
        "type": ToolType.TURNING_EXTERNAL,
        "shape": "C",
        "size_ic": 12.7,
        "thickness": 4.76,
        "nose_radius": 0.8,
        "suitable_materials": ["P", "M", "K"],
        "name_zh": "80度菱形外圆车刀片",
        "name_en": "80° rhombic turning insert",
    },
    "DNMG150608": {
        "type": ToolType.TURNING_EXTERNAL,
        "shape": "D",
        "size_ic": 15,
        "thickness": 6,
        "nose_radius": 0.8,
        "suitable_materials": ["P", "M"],
        "name_zh": "55度菱形外圆车刀片",
        "name_en": "55° rhombic turning insert",
    },
    "WNMG080408": {
        "type": ToolType.TURNING_EXTERNAL,
        "shape": "W",
        "size_ic": 8,
        "thickness": 4,
        "nose_radius": 0.8,
        "suitable_materials": ["P", "M", "S"],
        "name_zh": "80度三角形车刀片",
        "name_en": "Trigon turning insert",
    },
    "RCMT1204MO": {
        "type": ToolType.TURNING_EXTERNAL,
        "shape": "R",
        "size_ic": 12,
        "thickness": 4,
        "nose_radius": 6.0,
        "suitable_materials": ["S", "M"],
        "name_zh": "圆形车刀片",
        "name_en": "Round turning insert",
    },

    # End mills
    "END_MILL_4F_D10": {
        "type": ToolType.MILLING_END,
        "diameter": 10,
        "flutes": 4,
        "helix_angle": 35,
        "suitable_materials": ["P", "M", "K"],
        "name_zh": "4刃立铣刀 Φ10",
        "name_en": "4-flute end mill D10",
    },
    "END_MILL_2F_D8": {
        "type": ToolType.MILLING_END,
        "diameter": 8,
        "flutes": 2,
        "helix_angle": 30,
        "suitable_materials": ["N"],
        "name_zh": "2刃立铣刀 Φ8",
        "name_en": "2-flute end mill D8",
    },
    "BALL_MILL_2F_D6": {
        "type": ToolType.MILLING_BALL,
        "diameter": 6,
        "flutes": 2,
        "ball_radius": 3,
        "suitable_materials": ["P", "M", "H"],
        "name_zh": "2刃球头铣刀 R3",
        "name_en": "2-flute ball mill R3",
    },

    # Drills
    "DRILL_CARBIDE_D10": {
        "type": ToolType.DRILL_TWIST,
        "diameter": 10,
        "point_angle": 140,
        "material": ToolMaterial.COATED_PVD,
        "suitable_materials": ["P", "M", "K"],
        "name_zh": "硬质合金钻头 Φ10",
        "name_en": "Carbide drill D10",
    },
    "DRILL_HSS_D8": {
        "type": ToolType.DRILL_TWIST,
        "diameter": 8,
        "point_angle": 118,
        "material": ToolMaterial.HSS_COBALT,
        "suitable_materials": ["P", "M", "N"],
        "name_zh": "高速钢钻头 Φ8",
        "name_en": "HSS drill D8",
    },
}


def get_tool_recommendation(
    workpiece_group: str,
    operation: str,
    finish_quality: str = "medium",
) -> Optional[ToolRecommendation]:
    """
    Get tool recommendation for a specific application.

    Args:
        workpiece_group: ISO material group (P, M, K, N, S, H)
        operation: "roughing", "finishing", or "drilling"
        finish_quality: "rough", "medium", or "fine"

    Returns:
        ToolRecommendation or None

    Example:
        >>> rec = get_tool_recommendation("P", "finishing", "fine")
        >>> print(f"Tool: {rec.tool_material.value}")
    """
    key = (workpiece_group.upper(), operation.lower())
    materials = TOOL_MATERIAL_MATRIX.get(key)

    if not materials:
        return None

    # Select best material
    tool_material = materials[0]

    # Get geometry
    geom_key = f"{workpiece_group.upper()}_general"
    geometry = GEOMETRY_RECOMMENDATIONS.get(geom_key)

    if geometry is None:
        geometry = GEOMETRY_RECOMMENDATIONS["P_general"]

    # Determine tool type
    if operation == "drilling":
        tool_type = ToolType.DRILL_TWIST
    elif operation == "roughing":
        tool_type = ToolType.TURNING_EXTERNAL
    else:
        tool_type = ToolType.TURNING_EXTERNAL

    # Base cutting parameters (will be refined by cutting.py)
    speed_range = (80, 250) if workpiece_group in ["P", "K"] else (30, 150)
    feed_range = (0.15, 0.4) if operation == "roughing" else (0.05, 0.2)
    depth_range = (2.0, 6.0) if operation == "roughing" else (0.2, 1.0)

    # Coolant recommendation
    if workpiece_group in ["K"]:
        coolant = "dry"
    elif workpiece_group in ["S", "H"]:
        coolant = "flood"
    else:
        coolant = "flood" if operation == "roughing" else "MQL"

    return ToolRecommendation(
        tool_material=tool_material,
        tool_type=tool_type,
        geometry=geometry,
        cutting_speed_range=speed_range,
        feed_range=feed_range,
        depth_of_cut_range=depth_range,
        coolant=coolant,
        notes_zh=f"适用于ISO {workpiece_group}组材料的{operation}加工",
        notes_en=f"Suitable for {operation} of ISO {workpiece_group} materials",
        suitability=0.9,
    )


def select_tool_for_material(
    material_key: str,
    operation: str = "general",
) -> List[Dict]:
    """
    Select suitable tools for a workpiece material.

    Args:
        material_key: Material identifier from materials.py
        operation: Operation type

    Returns:
        List of tool recommendations with suitability scores
    """
    from .materials import get_machinability

    mat = get_machinability(material_key)
    if mat is None:
        return []

    group = mat.material_group
    results = []

    for tool_id, tool_data in TOOL_DATABASE.items():
        if group in tool_data.get("suitable_materials", []):
            results.append({
                "tool_id": tool_id,
                "name_zh": tool_data["name_zh"],
                "name_en": tool_data["name_en"],
                "type": tool_data["type"].value,
                "suitability": 0.8 if group in tool_data["suitable_materials"][:2] else 0.6,
            })

    return sorted(results, key=lambda x: -x["suitability"])


def get_insert_recommendation(
    workpiece_group: str,
    operation: str,
) -> Optional[str]:
    """
    Get recommended insert designation for turning.

    Args:
        workpiece_group: ISO material group
        operation: "roughing" or "finishing"

    Returns:
        Insert designation (e.g., "CNMG120408")
    """
    # Priority by material and operation
    recommendations = {
        ("P", "roughing"): "CNMG120408",
        ("P", "finishing"): "DNMG150608",
        ("M", "roughing"): "CNMG120408",
        ("M", "finishing"): "DNMG150608",
        ("K", "roughing"): "CNMG120408",
        ("K", "finishing"): "CNMG120408",
        ("N", "roughing"): "DNMG150608",
        ("N", "finishing"): "DNMG150608",
        ("S", "roughing"): "RCMT1204MO",
        ("S", "finishing"): "RCMT1204MO",
        ("H", "roughing"): "CNMG120408",
        ("H", "finishing"): "WNMG080408",
    }

    return recommendations.get((workpiece_group.upper(), operation.lower()))
