"""
Cutting Parameters Knowledge Base.

Provides cutting speed, feed rate, and depth of cut recommendations
for common machining operations based on material and tool combinations.

Reference:
- Machinery's Handbook - Cutting speeds and feeds
- Tool manufacturer recommendations (Sandvik, Kennametal)
- ISO 3685:1993 - Tool-life testing
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union
import math


class MachiningOperation(str, Enum):
    """Machining operation types."""

    # Turning
    TURNING_ROUGH = "turning_rough"
    TURNING_FINISH = "turning_finish"
    TURNING_FACE = "turning_face"
    TURNING_BORE = "turning_bore"
    TURNING_THREAD = "turning_thread"
    TURNING_GROOVE = "turning_groove"
    TURNING_PARTING = "turning_parting"

    # Milling
    MILLING_FACE = "milling_face"
    MILLING_SLOT = "milling_slot"
    MILLING_SHOULDER = "milling_shoulder"
    MILLING_POCKET = "milling_pocket"
    MILLING_PROFILE = "milling_profile"
    MILLING_3D = "milling_3d"

    # Drilling
    DRILLING = "drilling"
    DRILLING_DEEP = "drilling_deep"
    DRILLING_PECK = "drilling_peck"

    # Other
    REAMING = "reaming"
    TAPPING = "tapping"
    BORING = "boring"


@dataclass
class CuttingParameters:
    """Recommended cutting parameters for an operation."""

    operation: MachiningOperation

    # Speed
    cutting_speed_min: float  # Vc min (m/min)
    cutting_speed_max: float  # Vc max (m/min)
    cutting_speed_recommended: float  # Vc recommended (m/min)

    # Feed
    feed_min: float  # f min (mm/rev for turning, mm/tooth for milling)
    feed_max: float  # f max
    feed_recommended: float  # f recommended

    # Depth of cut
    depth_min: float  # ap min (mm)
    depth_max: float  # ap max (mm)
    depth_recommended: float  # ap recommended

    # For milling: width of cut
    width_min: Optional[float] = None  # ae min (mm)
    width_max: Optional[float] = None  # ae max (mm)
    width_recommended: Optional[float] = None  # ae recommended

    # Notes
    notes_zh: str = ""
    notes_en: str = ""


# Base cutting speeds by material group and tool material (m/min)
# Format: {(workpiece_group, tool_material): (Vc_min, Vc_max, Vc_rec)}
CUTTING_SPEED_TABLE: Dict[tuple, tuple] = {
    # Steel (P) with various tools
    ("P", "carbide"): (120, 280, 200),
    ("P", "coated_carbide"): (150, 350, 250),
    ("P", "cermet"): (180, 400, 280),
    ("P", "ceramic"): (300, 800, 500),
    ("P", "HSS"): (25, 45, 35),

    # Stainless (M)
    ("M", "carbide"): (80, 180, 130),
    ("M", "coated_carbide"): (100, 220, 160),
    ("M", "cermet"): (120, 250, 180),
    ("M", "ceramic"): (200, 500, 350),
    ("M", "HSS"): (15, 30, 22),

    # Cast iron (K)
    ("K", "carbide"): (100, 250, 180),
    ("K", "coated_carbide"): (120, 300, 220),
    ("K", "ceramic"): (300, 1000, 600),
    ("K", "CBN"): (500, 1500, 900),
    ("K", "HSS"): (20, 40, 30),

    # Non-ferrous (N)
    ("N", "carbide"): (300, 1000, 600),
    ("N", "coated_carbide"): (350, 1200, 700),
    ("N", "PCD"): (500, 3000, 1500),
    ("N", "HSS"): (80, 200, 150),

    # Superalloys (S)
    ("S", "carbide"): (25, 60, 40),
    ("S", "coated_carbide"): (30, 80, 50),
    ("S", "ceramic"): (150, 400, 250),
    ("S", "CBN"): (200, 500, 300),
    ("S", "HSS"): (8, 18, 12),

    # Hardened steel (H)
    ("H", "carbide"): (50, 120, 80),
    ("H", "coated_carbide"): (60, 150, 100),
    ("H", "ceramic"): (100, 300, 180),
    ("H", "CBN"): (150, 400, 250),
}

# Feed rate recommendations by operation (mm/rev for turning, mm/tooth for milling)
FEED_TABLE: Dict[MachiningOperation, tuple] = {
    # Turning operations (mm/rev)
    MachiningOperation.TURNING_ROUGH: (0.20, 0.60, 0.35),
    MachiningOperation.TURNING_FINISH: (0.05, 0.20, 0.12),
    MachiningOperation.TURNING_FACE: (0.15, 0.50, 0.30),
    MachiningOperation.TURNING_BORE: (0.10, 0.35, 0.20),
    MachiningOperation.TURNING_THREAD: (0.50, 3.00, 1.50),  # = pitch
    MachiningOperation.TURNING_GROOVE: (0.05, 0.15, 0.10),
    MachiningOperation.TURNING_PARTING: (0.05, 0.15, 0.08),

    # Milling operations (mm/tooth)
    MachiningOperation.MILLING_FACE: (0.10, 0.30, 0.18),
    MachiningOperation.MILLING_SLOT: (0.05, 0.15, 0.10),
    MachiningOperation.MILLING_SHOULDER: (0.08, 0.20, 0.12),
    MachiningOperation.MILLING_POCKET: (0.06, 0.18, 0.10),
    MachiningOperation.MILLING_PROFILE: (0.05, 0.15, 0.08),
    MachiningOperation.MILLING_3D: (0.03, 0.12, 0.06),

    # Drilling (mm/rev)
    MachiningOperation.DRILLING: (0.10, 0.35, 0.20),
    MachiningOperation.DRILLING_DEEP: (0.08, 0.25, 0.15),
    MachiningOperation.DRILLING_PECK: (0.10, 0.30, 0.18),

    # Other (mm/rev)
    MachiningOperation.REAMING: (0.20, 0.80, 0.50),
    MachiningOperation.TAPPING: (1.0, 3.0, 1.5),  # = pitch
    MachiningOperation.BORING: (0.08, 0.25, 0.15),
}

# Depth of cut recommendations by operation (mm)
DEPTH_TABLE: Dict[MachiningOperation, tuple] = {
    MachiningOperation.TURNING_ROUGH: (2.0, 8.0, 4.0),
    MachiningOperation.TURNING_FINISH: (0.2, 1.0, 0.5),
    MachiningOperation.TURNING_FACE: (1.0, 5.0, 2.5),
    MachiningOperation.TURNING_BORE: (0.5, 3.0, 1.5),
    MachiningOperation.TURNING_GROOVE: (0.5, 4.0, 2.0),
    MachiningOperation.TURNING_PARTING: (2.0, 6.0, 3.0),

    MachiningOperation.MILLING_FACE: (1.0, 5.0, 2.5),
    MachiningOperation.MILLING_SLOT: (0.5, 3.0, 1.5),
    MachiningOperation.MILLING_SHOULDER: (1.0, 8.0, 4.0),
    MachiningOperation.MILLING_POCKET: (0.5, 3.0, 1.5),
    MachiningOperation.MILLING_PROFILE: (0.3, 2.0, 1.0),
    MachiningOperation.MILLING_3D: (0.1, 1.0, 0.3),

    MachiningOperation.BORING: (0.3, 2.0, 1.0),
    MachiningOperation.REAMING: (0.1, 0.3, 0.15),
}


def get_cutting_parameters(
    operation: Union[str, MachiningOperation],
    workpiece_group: str,
    tool_material: str = "coated_carbide",
) -> Optional[CuttingParameters]:
    """
    Get recommended cutting parameters for an operation.

    Args:
        operation: Machining operation type
        workpiece_group: ISO material group (P, M, K, N, S, H)
        tool_material: Tool material type

    Returns:
        CuttingParameters or None

    Example:
        >>> params = get_cutting_parameters("turning_rough", "P", "coated_carbide")
        >>> print(f"Vc: {params.cutting_speed_recommended} m/min")
    """
    # Normalize operation
    if isinstance(operation, str):
        try:
            operation = MachiningOperation(operation.lower())
        except ValueError:
            return None

    # Get base cutting speed
    speed_key = (workpiece_group.upper(), tool_material.lower())
    speeds = CUTTING_SPEED_TABLE.get(speed_key)

    if speeds is None:
        # Try generic carbide
        speed_key = (workpiece_group.upper(), "carbide")
        speeds = CUTTING_SPEED_TABLE.get(speed_key)

    if speeds is None:
        return None

    vc_min, vc_max, vc_rec = speeds

    # Adjust for operation type
    operation_factors = {
        MachiningOperation.TURNING_ROUGH: 0.85,
        MachiningOperation.TURNING_FINISH: 1.15,
        MachiningOperation.TURNING_BORE: 0.80,
        MachiningOperation.TURNING_THREAD: 0.60,
        MachiningOperation.TURNING_GROOVE: 0.70,
        MachiningOperation.TURNING_PARTING: 0.60,
        MachiningOperation.MILLING_FACE: 1.00,
        MachiningOperation.MILLING_SLOT: 0.75,
        MachiningOperation.MILLING_SHOULDER: 0.90,
        MachiningOperation.MILLING_POCKET: 0.70,
        MachiningOperation.MILLING_PROFILE: 0.85,
        MachiningOperation.MILLING_3D: 0.80,
        MachiningOperation.DRILLING: 0.70,
        MachiningOperation.DRILLING_DEEP: 0.50,
        MachiningOperation.DRILLING_PECK: 0.60,
        MachiningOperation.REAMING: 0.30,
        MachiningOperation.TAPPING: 0.25,
        MachiningOperation.BORING: 0.75,
    }

    factor = operation_factors.get(operation, 1.0)
    vc_min = vc_min * factor
    vc_max = vc_max * factor
    vc_rec = vc_rec * factor

    # Get feed
    feeds = FEED_TABLE.get(operation, (0.1, 0.3, 0.2))
    f_min, f_max, f_rec = feeds

    # Get depth
    depths = DEPTH_TABLE.get(operation, (0.5, 3.0, 1.5))
    ap_min, ap_max, ap_rec = depths

    # Width for milling
    width_min, width_max, width_rec = None, None, None
    if "milling" in operation.value:
        width_min = 0.3  # 30% of cutter diameter typical
        width_max = 1.0  # Full width
        width_rec = 0.6

    return CuttingParameters(
        operation=operation,
        cutting_speed_min=round(vc_min, 1),
        cutting_speed_max=round(vc_max, 1),
        cutting_speed_recommended=round(vc_rec, 1),
        feed_min=f_min,
        feed_max=f_max,
        feed_recommended=f_rec,
        depth_min=ap_min,
        depth_max=ap_max,
        depth_recommended=ap_rec,
        width_min=width_min,
        width_max=width_max,
        width_recommended=width_rec,
        notes_zh=f"ISO {workpiece_group}组材料 {tool_material}刀具参数",
        notes_en=f"Parameters for ISO {workpiece_group} with {tool_material}",
    )


def calculate_spindle_speed(
    cutting_speed: float,
    diameter: float,
) -> int:
    """
    Calculate spindle speed from cutting speed and diameter.

    Args:
        cutting_speed: Cutting speed Vc (m/min)
        diameter: Workpiece or tool diameter (mm)

    Returns:
        Spindle speed n (rpm)

    Formula: n = (1000 * Vc) / (π * D)

    Example:
        >>> calculate_spindle_speed(200, 50)
        1273
    """
    if diameter <= 0:
        return 0

    n = (1000 * cutting_speed) / (math.pi * diameter)
    return round(n)


def calculate_feed_rate(
    spindle_speed: int,
    feed_per_rev: float,
    num_teeth: int = 1,
) -> float:
    """
    Calculate table feed rate.

    Args:
        spindle_speed: Spindle speed n (rpm)
        feed_per_rev: Feed per revolution f (mm/rev) for turning
                     or feed per tooth fz (mm/tooth) for milling
        num_teeth: Number of teeth (1 for turning, >1 for milling)

    Returns:
        Table feed rate vf (mm/min)

    Formula: vf = n * f (turning) or vf = n * fz * z (milling)

    Example:
        >>> calculate_feed_rate(1000, 0.2, 1)  # Turning
        200.0
        >>> calculate_feed_rate(1000, 0.1, 4)  # Milling
        400.0
    """
    vf = spindle_speed * feed_per_rev * num_teeth
    return round(vf, 1)


def calculate_metal_removal_rate(
    cutting_speed: float,
    feed: float,
    depth: float,
    width: Optional[float] = None,
) -> float:
    """
    Calculate metal removal rate (MRR).

    Args:
        cutting_speed: Cutting speed Vc (m/min)
        feed: Feed f (mm/rev) or fz*z (mm/rev equivalent)
        depth: Depth of cut ap (mm)
        width: Width of cut ae (mm) - for milling only

    Returns:
        Metal removal rate Q (cm³/min)

    Formula:
        Turning: Q = Vc * f * ap
        Milling: Q = Vc * f * ap * ae / 1000

    Example:
        >>> calculate_metal_removal_rate(200, 0.3, 4)  # Turning
        240.0
    """
    if width is not None:
        # Milling
        Q = cutting_speed * feed * depth * width / 1000
    else:
        # Turning
        Q = cutting_speed * feed * depth

    return round(Q, 1)


def suggest_parameters_for_surface_finish(
    target_ra: float,
    nose_radius: float = 0.8,
) -> Dict[str, float]:
    """
    Suggest feed rate to achieve target surface roughness.

    Args:
        target_ra: Target surface roughness Ra (μm)
        nose_radius: Tool nose radius r (mm)

    Returns:
        Dictionary with suggested feed and achievable Ra

    Formula: Ra ≈ f² / (32 * r)

    Example:
        >>> suggest_parameters_for_surface_finish(1.6, 0.8)
        {'feed_max': 0.20, 'achievable_ra': 1.56}
    """
    # f = sqrt(Ra * 32 * r)
    f_max = math.sqrt(target_ra * 32 * nose_radius / 1000)  # Convert to mm
    f_max = min(f_max, 0.5)  # Practical limit

    # Calculate achievable Ra with suggested feed
    achievable_ra = (f_max ** 2 * 1000) / (32 * nose_radius)

    return {
        "feed_max": round(f_max, 3),
        "achievable_ra": round(achievable_ra, 2),
        "nose_radius_used": nose_radius,
    }


def get_drilling_parameters(
    drill_diameter: float,
    workpiece_group: str,
    hole_depth: Optional[float] = None,
    tool_material: str = "coated_carbide",
) -> Dict:
    """
    Get drilling-specific parameters.

    Args:
        drill_diameter: Drill diameter (mm)
        workpiece_group: ISO material group
        hole_depth: Hole depth (mm) - for deep hole determination
        tool_material: Tool material type

    Returns:
        Dictionary with drilling parameters
    """
    # Determine if deep hole drilling
    is_deep_hole = hole_depth is not None and hole_depth > 3 * drill_diameter

    operation = MachiningOperation.DRILLING_DEEP if is_deep_hole else MachiningOperation.DRILLING
    params = get_cutting_parameters(operation, workpiece_group, tool_material)

    if params is None:
        return {}

    spindle_speed = calculate_spindle_speed(params.cutting_speed_recommended, drill_diameter)

    result = {
        "diameter": drill_diameter,
        "cutting_speed": params.cutting_speed_recommended,
        "spindle_speed": spindle_speed,
        "feed_per_rev": params.feed_recommended,
        "feed_rate": calculate_feed_rate(spindle_speed, params.feed_recommended),
        "is_deep_hole": is_deep_hole,
    }

    if is_deep_hole:
        result["peck_depth"] = round(drill_diameter * 1.5, 1)
        result["retract_amount"] = round(drill_diameter * 0.5, 1)

    return result
