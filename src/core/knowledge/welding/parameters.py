"""
Welding Process Parameters Knowledge Base.

Provides welding current, voltage, speed, and other parameters for common
welding processes and material combinations.

Reference:
- AWS Welding Handbook
- Lincoln Electric Procedure Handbook
- ISO 15614 - Welding procedure specifications
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import math


class WeldingProcess(str, Enum):
    """Welding process types per ISO 4063."""

    # Arc welding
    SMAW = "SMAW"  # 111 - Shielded Metal Arc Welding (手工电弧焊)
    GMAW = "GMAW"  # 131/135 - Gas Metal Arc Welding (MIG/MAG)
    GTAW = "GTAW"  # 141 - Gas Tungsten Arc Welding (TIG)
    FCAW = "FCAW"  # 136/138 - Flux Cored Arc Welding
    SAW = "SAW"  # 121 - Submerged Arc Welding (埋弧焊)

    # Resistance welding
    RSW = "RSW"  # 21 - Resistance Spot Welding (点焊)
    RSEW = "RSEW"  # 22 - Resistance Seam Welding (缝焊)

    # Other
    LBW = "LBW"  # 52 - Laser Beam Welding (激光焊)
    EBW = "EBW"  # 51 - Electron Beam Welding (电子束焊)
    PAW = "PAW"  # 15 - Plasma Arc Welding (等离子焊)


class WeldingPosition(str, Enum):
    """Welding positions per AWS/ISO standards."""

    FLAT = "1G/1F"  # 平焊
    HORIZONTAL = "2G/2F"  # 横焊
    VERTICAL_UP = "3G/3F_up"  # 立焊上行
    VERTICAL_DOWN = "3G/3F_down"  # 立焊下行
    OVERHEAD = "4G/4F"  # 仰焊
    PIPE_ROLL = "1G_pipe"  # 管道转动平焊
    PIPE_FIXED = "5G/6G"  # 管道固定位置焊


class JointType(str, Enum):
    """Basic joint types."""

    BUTT = "butt"  # 对接接头
    LAP = "lap"  # 搭接接头
    TEE = "tee"  # T形接头
    CORNER = "corner"  # 角接头
    EDGE = "edge"  # 边缘接头


@dataclass
class WeldingParameters:
    """Recommended welding parameters for a process/material combination."""

    process: WeldingProcess
    base_material: str
    thickness_range: Tuple[float, float]  # mm

    # Current parameters
    current_min: float  # A
    current_max: float  # A
    current_recommended: float  # A
    current_type: str  # "DC+" "DC-" "AC"

    # Voltage parameters
    voltage_min: float  # V
    voltage_max: float  # V
    voltage_recommended: float  # V

    # Travel speed
    speed_min: float  # mm/min or cm/min
    speed_max: float  # mm/min
    speed_recommended: float  # mm/min

    # Wire/electrode
    electrode_diameter: float  # mm
    wire_feed_speed: Optional[float] = None  # m/min (for GMAW/FCAW)

    # Shielding gas
    shielding_gas: Optional[str] = None
    gas_flow_rate: Optional[float] = None  # L/min

    # Other
    interpass_temp_max: Optional[float] = None  # °C
    preheat_temp: Optional[float] = None  # °C

    notes_zh: str = ""
    notes_en: str = ""


# Welding parameters database
# Format: {(process, material_group, thickness_range): parameters}
WELDING_PROCESS_DATABASE: Dict[tuple, Dict] = {
    # SMAW (手工电弧焊) - Carbon Steel
    (WeldingProcess.SMAW, "carbon_steel", (1.5, 6)): {
        "electrode_diameter": 3.2,
        "current_range": (90, 150),
        "current_recommended": 120,
        "current_type": "DC+",
        "voltage_range": (22, 28),
        "voltage_recommended": 24,
        "speed_range": (100, 200),
        "speed_recommended": 150,
        "electrode_types": ["E6010", "E6011", "E7018"],
        "notes_zh": "碳钢手工电弧焊，E7018低氢型焊条适用于重要结构",
    },
    (WeldingProcess.SMAW, "carbon_steel", (6, 12)): {
        "electrode_diameter": 4.0,
        "current_range": (140, 200),
        "current_recommended": 170,
        "current_type": "DC+",
        "voltage_range": (24, 30),
        "voltage_recommended": 26,
        "speed_range": (80, 160),
        "speed_recommended": 120,
        "electrode_types": ["E7018", "E7024"],
        "notes_zh": "中厚板碳钢焊接，建议多道焊",
    },
    (WeldingProcess.SMAW, "carbon_steel", (12, 25)): {
        "electrode_diameter": 5.0,
        "current_range": (180, 250),
        "current_recommended": 210,
        "current_type": "DC+",
        "voltage_range": (26, 32),
        "voltage_recommended": 28,
        "speed_range": (60, 120),
        "speed_recommended": 90,
        "electrode_types": ["E7018", "E7024"],
        "notes_zh": "厚板碳钢焊接，需预热和层间温度控制",
    },
    # SMAW - Stainless Steel
    (WeldingProcess.SMAW, "stainless_steel", (1.5, 6)): {
        "electrode_diameter": 2.5,
        "current_range": (50, 90),
        "current_recommended": 70,
        "current_type": "DC+",
        "voltage_range": (20, 26),
        "voltage_recommended": 22,
        "speed_range": (80, 150),
        "speed_recommended": 100,
        "electrode_types": ["E308L-16", "E316L-16"],
        "notes_zh": "不锈钢焊接电流比碳钢低30%，防止过热",
    },
    # GMAW (MIG/MAG) - Carbon Steel
    (WeldingProcess.GMAW, "carbon_steel", (1.0, 3)): {
        "wire_diameter": 0.8,
        "current_range": (80, 140),
        "current_recommended": 110,
        "current_type": "DC+",
        "voltage_range": (17, 22),
        "voltage_recommended": 19,
        "speed_range": (300, 600),
        "speed_recommended": 450,
        "wire_feed_speed": (4, 8),
        "shielding_gas": "CO2 or Ar+CO2(80/20)",
        "gas_flow_rate": (15, 20),
        "notes_zh": "薄板气体保护焊，短路过渡模式",
    },
    (WeldingProcess.GMAW, "carbon_steel", (3, 10)): {
        "wire_diameter": 1.2,
        "current_range": (180, 280),
        "current_recommended": 230,
        "current_type": "DC+",
        "voltage_range": (24, 32),
        "voltage_recommended": 28,
        "speed_range": (250, 450),
        "speed_recommended": 350,
        "wire_feed_speed": (8, 14),
        "shielding_gas": "Ar+CO2(80/20) or CO2",
        "gas_flow_rate": (18, 25),
        "notes_zh": "中厚板气体保护焊，喷射过渡或脉冲模式",
    },
    # GMAW - Aluminum
    (WeldingProcess.GMAW, "aluminum", (2, 6)): {
        "wire_diameter": 1.2,
        "current_range": (120, 200),
        "current_recommended": 160,
        "current_type": "DC+",
        "voltage_range": (22, 28),
        "voltage_recommended": 25,
        "speed_range": (400, 700),
        "speed_recommended": 550,
        "wire_feed_speed": (8, 12),
        "shielding_gas": "Ar (99.99%)",
        "gas_flow_rate": (18, 25),
        "notes_zh": "铝合金MIG焊，使用推拉式送丝系统",
    },
    # GTAW (TIG) - Stainless Steel
    (WeldingProcess.GTAW, "stainless_steel", (1, 3)): {
        "tungsten_diameter": 2.4,
        "current_range": (50, 120),
        "current_recommended": 80,
        "current_type": "DC-",
        "voltage_range": (10, 14),
        "voltage_recommended": 12,
        "speed_range": (80, 150),
        "speed_recommended": 100,
        "filler_diameter": (1.6, 2.4),
        "shielding_gas": "Ar (99.99%)",
        "gas_flow_rate": (8, 12),
        "notes_zh": "不锈钢TIG焊，背面需氩气保护",
    },
    (WeldingProcess.GTAW, "stainless_steel", (3, 8)): {
        "tungsten_diameter": 3.2,
        "current_range": (100, 200),
        "current_recommended": 150,
        "current_type": "DC-",
        "voltage_range": (12, 16),
        "voltage_recommended": 14,
        "speed_range": (60, 120),
        "speed_recommended": 80,
        "filler_diameter": (2.4, 3.2),
        "shielding_gas": "Ar (99.99%)",
        "gas_flow_rate": (10, 15),
        "notes_zh": "中厚板不锈钢TIG焊，多道焊接",
    },
    # GTAW - Aluminum
    (WeldingProcess.GTAW, "aluminum", (1, 4)): {
        "tungsten_diameter": 2.4,
        "current_range": (60, 150),
        "current_recommended": 100,
        "current_type": "AC",
        "voltage_range": (12, 18),
        "voltage_recommended": 15,
        "speed_range": (100, 200),
        "speed_recommended": 150,
        "filler_diameter": (2.4, 3.2),
        "shielding_gas": "Ar (99.99%)",
        "gas_flow_rate": (12, 18),
        "notes_zh": "铝合金TIG焊，使用交流电源清除氧化膜",
    },
    # SAW (埋弧焊) - Carbon Steel
    (WeldingProcess.SAW, "carbon_steel", (8, 25)): {
        "wire_diameter": 4.0,
        "current_range": (400, 700),
        "current_recommended": 550,
        "current_type": "DC+",
        "voltage_range": (28, 36),
        "voltage_recommended": 32,
        "speed_range": (300, 600),
        "speed_recommended": 450,
        "flux_type": "F7A2-EM12K",
        "notes_zh": "埋弧焊适用于长直焊缝和厚板焊接",
    },
    # FCAW - Carbon Steel
    (WeldingProcess.FCAW, "carbon_steel", (3, 12)): {
        "wire_diameter": 1.2,
        "current_range": (150, 280),
        "current_recommended": 220,
        "current_type": "DC+",
        "voltage_range": (24, 32),
        "voltage_recommended": 28,
        "speed_range": (200, 400),
        "speed_recommended": 300,
        "wire_feed_speed": (6, 12),
        "shielding_gas": "CO2 or Ar+CO2(75/25)",
        "gas_flow_rate": (18, 25),
        "notes_zh": "药芯焊丝焊接，熔敷效率高",
    },
}

# Filler material recommendations
FILLER_MATERIAL_DATABASE: Dict[str, Dict[str, Any]] = {
    # Carbon steel fillers
    "carbon_steel": {
        "SMAW": {
            "standard": ["E7018", "E6013"],
            "high_strength": ["E8018", "E9018"],
            "weathering": ["E7018-W1"],
        },
        "GMAW": {
            "solid_wire": ["ER70S-6", "ER70S-3"],
            "flux_cored": ["E71T-1", "E71T-8"],
        },
        "GTAW": {
            "filler_rod": ["ER70S-2", "ER70S-6"],
        },
        "SAW": {
            "wire_flux": ["EM12K + F7A2", "EA2 + F7A4"],
        },
    },
    # Stainless steel fillers
    "304_stainless": {
        "SMAW": ["E308L-16", "E308L-17"],
        "GMAW": ["ER308L", "ER308LSi"],
        "GTAW": ["ER308L", "ER308LSi"],
    },
    "316_stainless": {
        "SMAW": ["E316L-16", "E316L-17"],
        "GMAW": ["ER316L", "ER316LSi"],
        "GTAW": ["ER316L", "ER316LSi"],
    },
    "duplex_stainless": {
        "SMAW": ["E2209-16"],
        "GMAW": ["ER2209"],
        "GTAW": ["ER2209"],
    },
    # Aluminum fillers
    "aluminum_6xxx": {
        "GMAW": ["ER4043", "ER5356"],
        "GTAW": ["ER4043", "ER5356"],
    },
    "aluminum_5xxx": {
        "GMAW": ["ER5356", "ER5183"],
        "GTAW": ["ER5356", "ER5183"],
    },
    # Low alloy steel
    "low_alloy_steel": {
        "SMAW": ["E8018-B2", "E8018-C3"],
        "GMAW": ["ER80S-D2", "ER80S-Ni1"],
        "GTAW": ["ER80S-D2", "ER80S-Ni1"],
    },
}


def get_welding_parameters(
    process: Union[str, WeldingProcess],
    base_material: str,
    thickness: float,
    position: WeldingPosition = WeldingPosition.FLAT,
) -> Optional[WeldingParameters]:
    """
    Get recommended welding parameters.

    Args:
        process: Welding process type
        base_material: Base material type
        thickness: Material thickness (mm)
        position: Welding position

    Returns:
        WeldingParameters or None

    Example:
        >>> params = get_welding_parameters("GMAW", "carbon_steel", 6)
        >>> print(f"Current: {params.current_recommended}A")
    """
    if isinstance(process, str):
        try:
            process = WeldingProcess(process.upper())
        except ValueError:
            return None

    # Find matching thickness range
    for (proc, material, thickness_range), data in WELDING_PROCESS_DATABASE.items():
        if proc == process and material == base_material:
            if thickness_range[0] <= thickness <= thickness_range[1]:
                # Position adjustment factors
                position_factors = {
                    WeldingPosition.FLAT: 1.0,
                    WeldingPosition.HORIZONTAL: 0.9,
                    WeldingPosition.VERTICAL_UP: 0.75,
                    WeldingPosition.VERTICAL_DOWN: 0.85,
                    WeldingPosition.OVERHEAD: 0.8,
                    WeldingPosition.PIPE_ROLL: 1.0,
                    WeldingPosition.PIPE_FIXED: 0.85,
                }
                factor = position_factors.get(position, 1.0)

                current_range = data.get("current_range", (100, 200))
                voltage_range = data.get("voltage_range", (20, 30))
                speed_range = data.get("speed_range", (100, 300))

                return WeldingParameters(
                    process=process,
                    base_material=base_material,
                    thickness_range=thickness_range,
                    current_min=current_range[0] * factor,
                    current_max=current_range[1] * factor,
                    current_recommended=data.get("current_recommended", sum(current_range) / 2) * factor,
                    current_type=data.get("current_type", "DC+"),
                    voltage_min=voltage_range[0],
                    voltage_max=voltage_range[1],
                    voltage_recommended=data.get("voltage_recommended", sum(voltage_range) / 2),
                    speed_min=speed_range[0],
                    speed_max=speed_range[1],
                    speed_recommended=data.get("speed_recommended", sum(speed_range) / 2),
                    electrode_diameter=data.get("electrode_diameter", data.get("wire_diameter", 1.2)),
                    wire_feed_speed=data.get("wire_feed_speed", (None, None))[0] if isinstance(data.get("wire_feed_speed"), tuple) else None,
                    shielding_gas=data.get("shielding_gas"),
                    gas_flow_rate=data.get("gas_flow_rate", (None, None))[0] if isinstance(data.get("gas_flow_rate"), tuple) else None,
                    notes_zh=data.get("notes_zh", ""),
                    notes_en=data.get("notes_en", ""),
                )

    return None


def get_filler_material(
    base_material: str,
    process: Union[str, WeldingProcess],
    application: str = "standard",
) -> List[str]:
    """
    Get recommended filler materials.

    Args:
        base_material: Base material type
        process: Welding process
        application: Application type (standard, high_strength, etc.)

    Returns:
        List of recommended filler materials
    """
    if isinstance(process, WeldingProcess):
        process = process.value

    material_data = FILLER_MATERIAL_DATABASE.get(base_material, {})
    process_data = material_data.get(process, {})

    if isinstance(process_data, list):
        return process_data

    return process_data.get(application, process_data.get("standard", []))


def calculate_heat_input(
    voltage: float,
    current: float,
    travel_speed: float,
    efficiency: float = 0.8,
) -> float:
    """
    Calculate welding heat input.

    Args:
        voltage: Arc voltage (V)
        current: Welding current (A)
        travel_speed: Travel speed (mm/min)
        efficiency: Process efficiency (default 0.8 for GMAW)

    Returns:
        Heat input (kJ/mm)

    Formula: H = (η * V * I * 60) / (S * 1000)

    Example:
        >>> calculate_heat_input(28, 230, 350)
        1.10
    """
    if travel_speed <= 0:
        return 0.0

    heat_input = (efficiency * voltage * current * 60) / (travel_speed * 1000)
    return round(heat_input, 2)


def estimate_weld_passes(
    thickness: float,
    joint_type: JointType,
    process: WeldingProcess,
) -> int:
    """
    Estimate number of weld passes required.

    Args:
        thickness: Material thickness (mm)
        joint_type: Joint type
        process: Welding process

    Returns:
        Estimated number of passes
    """
    # Deposition per pass varies by process
    deposition_per_pass = {
        WeldingProcess.SMAW: 3.0,
        WeldingProcess.GMAW: 4.0,
        WeldingProcess.GTAW: 2.0,
        WeldingProcess.FCAW: 5.0,
        WeldingProcess.SAW: 8.0,
    }

    # Joint type factor
    joint_factors = {
        JointType.BUTT: 1.0,
        JointType.TEE: 0.7,
        JointType.LAP: 0.5,
        JointType.CORNER: 0.8,
        JointType.EDGE: 0.3,
    }

    deposition = deposition_per_pass.get(process, 3.0)
    factor = joint_factors.get(joint_type, 1.0)

    # Weld volume approximation
    passes = math.ceil((thickness * factor) / deposition)
    return max(1, passes)


def get_interpass_temperature(
    base_material: str,
    thickness: float,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Get interpass temperature limits.

    Args:
        base_material: Base material type
        thickness: Material thickness (mm)

    Returns:
        Tuple of (min_temp, max_temp) in °C, None if not applicable
    """
    interpass_temps = {
        "carbon_steel": (None, 250),
        "low_alloy_steel": (150, 250),
        "stainless_steel": (None, 150),
        "duplex_stainless": (None, 100),
        "aluminum": (None, 120),
    }

    return interpass_temps.get(base_material, (None, None))
