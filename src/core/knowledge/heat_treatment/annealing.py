"""
Annealing Process Knowledge Base.

Provides annealing temperature, time, and cooling parameters
for stress relief and microstructure improvement.

Reference:
- ASM Handbook Volume 4 - Heat Treating
- ISO 683 - Heat-treatable steels
- GB/T 16924 - Steel annealing
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union


class AnnealingType(str, Enum):
    """Annealing process types."""

    FULL = "full"  # 完全退火
    ISOTHERMAL = "isothermal"  # 等温退火
    SPHEROIDIZING = "spheroidizing"  # 球化退火
    STRESS_RELIEF = "stress_relief"  # 去应力退火
    RECRYSTALLIZATION = "recrystallization"  # 再结晶退火
    PROCESS = "process"  # 工序间退火
    BRIGHT = "bright"  # 光亮退火


@dataclass
class AnnealingParameters:
    """Annealing process parameters."""

    annealing_type: AnnealingType
    material_id: str

    # Temperature parameters
    temperature_min: float  # °C
    temperature_max: float  # °C
    temperature_recommended: float  # °C

    # Holding time
    holding_time_factor: float  # min/mm
    holding_time_min: float  # minutes
    holding_time_max: float  # minutes

    # Heating rate
    heating_rate_max: Optional[float] = None  # °C/hour

    # Cooling
    cooling_method: str = "furnace"  # "furnace", "air", "controlled"
    cooling_rate_max: Optional[float] = None  # °C/hour (for controlled cooling)
    cooling_to_temperature: Optional[float] = None  # °C (then air cool)

    # Atmosphere
    atmosphere: Optional[str] = None

    # Expected results
    hardness_after: Optional[Tuple[float, float]] = None  # HB range

    notes_zh: str = ""
    notes_en: str = ""


# Annealing database
ANNEALING_DATABASE: Dict[str, Dict[AnnealingType, Dict]] = {
    # 45 steel
    "45": {
        AnnealingType.FULL: {
            "temperature_range": (820, 850),
            "temperature_recommended": 830,
            "holding_time_factor": 2.0,
            "cooling_method": "furnace",
            "cooling_to": 500,
            "hardness_after": (156, 197),
            "notes_zh": "完全退火获得铁素体+珠光体组织",
        },
        AnnealingType.ISOTHERMAL: {
            "temperature_range": (820, 850),
            "austenitizing_time": 60,
            "isothermal_temp": 680,
            "isothermal_time": 120,
            "cooling_method": "air",
            "hardness_after": (156, 197),
            "notes_zh": "等温退火缩短处理时间",
        },
        AnnealingType.STRESS_RELIEF: {
            "temperature_range": (550, 650),
            "temperature_recommended": 600,
            "holding_time_factor": 2.5,
            "cooling_method": "furnace",
            "cooling_to": 300,
            "notes_zh": "去应力退火，不改变组织",
        },
        AnnealingType.SPHEROIDIZING: {
            "temperature_range": (740, 780),
            "temperature_recommended": 760,
            "holding_time": 4,  # hours
            "cooling_method": "furnace",
            "cooling_rate": 20,  # °C/hour
            "cooling_to": 600,
            "hardness_after": (156, 187),
            "notes_zh": "球化退火改善切削加工性",
        },
    },
    # 40Cr steel
    "40Cr": {
        AnnealingType.FULL: {
            "temperature_range": (830, 850),
            "temperature_recommended": 840,
            "holding_time_factor": 2.0,
            "cooling_method": "furnace",
            "cooling_to": 500,
            "hardness_after": (179, 217),
            "notes_zh": "完全退火",
        },
        AnnealingType.ISOTHERMAL: {
            "temperature_range": (830, 850),
            "austenitizing_time": 60,
            "isothermal_temp": 700,
            "isothermal_time": 150,
            "cooling_method": "air",
            "hardness_after": (179, 217),
        },
        AnnealingType.STRESS_RELIEF: {
            "temperature_range": (550, 650),
            "temperature_recommended": 600,
            "holding_time_factor": 2.5,
            "cooling_method": "furnace",
            "cooling_to": 300,
        },
        AnnealingType.SPHEROIDIZING: {
            "temperature_range": (750, 780),
            "temperature_recommended": 760,
            "holding_time": 6,
            "cooling_method": "furnace",
            "cooling_rate": 15,
            "cooling_to": 600,
            "hardness_after": (179, 207),
        },
    },
    # Cr12MoV tool steel
    "Cr12MoV": {
        AnnealingType.SPHEROIDIZING: {
            "temperature_range": (850, 870),
            "temperature_recommended": 860,
            "holding_time": 4,
            "cooling_method": "furnace",
            "cooling_rate": 20,
            "cooling_to": 700,
            "then": "furnace cool to 500°C, air cool",
            "hardness_after": (217, 255),
            "notes_zh": "D2钢球化退火，改善切削性",
        },
        AnnealingType.STRESS_RELIEF: {
            "temperature_range": (650, 700),
            "temperature_recommended": 680,
            "holding_time_factor": 2.0,
            "cooling_method": "furnace",
            "notes_zh": "去应力退火，加工间使用",
        },
    },
    # 304 stainless steel
    "304": {
        AnnealingType.FULL: {
            "temperature_range": (1010, 1120),
            "temperature_recommended": 1050,
            "holding_time_factor": 2.5,
            "cooling_method": "water or air",
            "hardness_after": (130, 180),
            "notes_zh": "固溶退火恢复耐腐蚀性",
        },
        AnnealingType.STRESS_RELIEF: {
            "temperature_range": (400, 500),
            "temperature_recommended": 450,
            "holding_time_factor": 2.0,
            "cooling_method": "air",
            "notes_zh": "低温去应力，不影响耐腐蚀性",
        },
    },
    # GCr15 bearing steel
    "GCr15": {
        AnnealingType.SPHEROIDIZING: {
            "temperature_range": (780, 800),
            "temperature_recommended": 790,
            "holding_time": 6,
            "cooling_method": "furnace",
            "cooling_rate": 10,
            "cooling_to": 650,
            "hardness_after": (179, 207),
            "notes_zh": "轴承钢球化退火，要求球化率>90%",
        },
        AnnealingType.ISOTHERMAL: {
            "temperature_range": (780, 800),
            "austenitizing_time": 120,
            "isothermal_temp": 720,
            "isothermal_time": 180,
            "cooling_method": "air",
            "hardness_after": (179, 207),
        },
    },
    # 6061 aluminum
    "6061": {
        AnnealingType.FULL: {
            "temperature_range": (410, 430),
            "temperature_recommended": 415,
            "holding_time": 2,  # hours
            "cooling_method": "furnace",
            "cooling_rate": 30,
            "cooling_to": 260,
            "hardness_after": (30, 40),  # HB
            "notes_zh": "完全退火得到O态",
        },
        AnnealingType.STRESS_RELIEF: {
            "temperature_range": (230, 290),
            "temperature_recommended": 260,
            "holding_time": 1,
            "cooling_method": "air",
            "notes_zh": "去应力退火，不显著降低强度",
        },
    },
}

# Stress relief temperature guidelines by material type
STRESS_RELIEF_GUIDELINES: Dict[str, Dict] = {
    "carbon_steel": {
        "temperature_range": (550, 650),
        "holding_time_factor": 2.5,
        "cooling_to": 300,
        "notes_zh": "碳钢去应力温度不超过Ac1",
    },
    "low_alloy_steel": {
        "temperature_range": (580, 680),
        "holding_time_factor": 2.5,
        "cooling_to": 300,
        "notes_zh": "低合金钢去应力",
    },
    "stainless_steel_austenitic": {
        "temperature_range": (400, 500),
        "holding_time_factor": 2.0,
        "cooling_method": "air",
        "notes_zh": "奥氏体不锈钢低温去应力",
    },
    "stainless_steel_martensitic": {
        "temperature_range": (550, 650),
        "holding_time_factor": 2.0,
        "notes_zh": "马氏体不锈钢去应力",
    },
    "aluminum": {
        "temperature_range": (200, 290),
        "holding_time_factor": 1.0,
        "notes_zh": "铝合金去应力温度较低",
    },
    "copper": {
        "temperature_range": (150, 200),
        "holding_time_factor": 1.0,
        "notes_zh": "铜合金低温去应力",
    },
    "cast_iron": {
        "temperature_range": (500, 600),
        "holding_time_factor": 3.0,
        "cooling_to": 200,
        "notes_zh": "铸铁去应力退火",
    },
}


def get_annealing_parameters(
    material_id: str,
    annealing_type: Union[str, AnnealingType],
) -> Optional[AnnealingParameters]:
    """
    Get annealing parameters for a material.

    Args:
        material_id: Material identifier
        annealing_type: Type of annealing

    Returns:
        AnnealingParameters or None

    Example:
        >>> params = get_annealing_parameters("45", "full")
        >>> print(f"Temperature: {params.temperature_recommended}°C")
    """
    if isinstance(annealing_type, str):
        try:
            annealing_type = AnnealingType(annealing_type.lower())
        except ValueError:
            return None

    material_data = ANNEALING_DATABASE.get(material_id)
    if not material_data:
        return None

    process_data = material_data.get(annealing_type)
    if not process_data:
        return None

    temp_range = process_data["temperature_range"]

    return AnnealingParameters(
        annealing_type=annealing_type,
        material_id=material_id,
        temperature_min=temp_range[0],
        temperature_max=temp_range[1],
        temperature_recommended=process_data.get("temperature_recommended", sum(temp_range) / 2),
        holding_time_factor=process_data.get("holding_time_factor", 2.0),
        holding_time_min=process_data.get("holding_time", 60),
        holding_time_max=process_data.get("holding_time", 60) * 2,
        heating_rate_max=process_data.get("heating_rate_max"),
        cooling_method=process_data.get("cooling_method", "furnace"),
        cooling_rate_max=process_data.get("cooling_rate"),
        cooling_to_temperature=process_data.get("cooling_to"),
        atmosphere=process_data.get("atmosphere"),
        hardness_after=process_data.get("hardness_after"),
        notes_zh=process_data.get("notes_zh", ""),
        notes_en=process_data.get("notes_en", ""),
    )


def get_stress_relief_parameters(
    material_type: str,
    thickness: float,
) -> Dict:
    """
    Get stress relief annealing parameters.

    Args:
        material_type: Material type category
        thickness: Section thickness (mm)

    Returns:
        Dict with stress relief parameters

    Example:
        >>> params = get_stress_relief_parameters("carbon_steel", 50)
        >>> print(f"Temperature: {params['temperature_recommended']}°C")
    """
    guidelines = STRESS_RELIEF_GUIDELINES.get(material_type)
    if not guidelines:
        return {"error": f"Material type '{material_type}' not found"}

    temp_range = guidelines["temperature_range"]
    time_factor = guidelines.get("holding_time_factor", 2.0)

    # Calculate holding time
    base_time = thickness * time_factor
    holding_time = max(30, base_time)  # Minimum 30 minutes

    return {
        "material_type": material_type,
        "temperature_range": temp_range,
        "temperature_recommended": sum(temp_range) // 2,
        "holding_time_minutes": round(holding_time),
        "cooling_method": guidelines.get("cooling_method", "furnace"),
        "cooling_to": guidelines.get("cooling_to"),
        "notes_zh": guidelines.get("notes_zh", ""),
    }


def calculate_annealing_cycle_time(
    temperature: float,
    section_thickness: float,
    annealing_type: AnnealingType,
    furnace_type: str = "box",
) -> Dict:
    """
    Calculate total annealing cycle time.

    Args:
        temperature: Annealing temperature (°C)
        section_thickness: Maximum section thickness (mm)
        annealing_type: Type of annealing
        furnace_type: "box", "pit", or "continuous"

    Returns:
        Dict with cycle time breakdown
    """
    # Heating rate depends on furnace type and material thickness
    heating_rates = {
        "box": 100,  # °C/hour typical
        "pit": 80,
        "continuous": 150,
    }

    heating_rate = heating_rates.get(furnace_type, 100)

    # Adjust for thickness (slower for thick sections)
    if section_thickness > 50:
        heating_rate *= 0.7
    elif section_thickness > 100:
        heating_rate *= 0.5

    # Time calculations
    # Assume starting from room temperature (20°C)
    heating_time = (temperature - 20) / heating_rate  # hours

    # Holding time
    holding_time_factor = {
        AnnealingType.FULL: 2.0,
        AnnealingType.ISOTHERMAL: 3.0,
        AnnealingType.SPHEROIDIZING: 4.0,
        AnnealingType.STRESS_RELIEF: 2.5,
        AnnealingType.RECRYSTALLIZATION: 1.5,
    }
    factor = holding_time_factor.get(annealing_type, 2.0)
    holding_time = section_thickness * factor / 60  # Convert to hours

    # Cooling time (furnace cooling is slow)
    cooling_rate = 30  # °C/hour typical furnace cooling
    cooling_to = 300  # Typical temperature before air cooling
    cooling_time = (temperature - cooling_to) / cooling_rate

    total_time = heating_time + holding_time + cooling_time

    return {
        "heating_time_hours": round(heating_time, 1),
        "holding_time_hours": round(holding_time, 1),
        "cooling_time_hours": round(cooling_time, 1),
        "total_time_hours": round(total_time, 1),
        "heating_rate": heating_rate,
        "notes": f"基于{section_thickness}mm截面厚度计算",
    }


def recommend_annealing_for_purpose(
    material_id: str,
    purpose: str,
) -> Optional[Dict]:
    """
    Recommend annealing type based on purpose.

    Args:
        material_id: Material identifier
        purpose: Purpose of annealing ("machinability", "stress", "softening", "structure")

    Returns:
        Dict with recommendation or None
    """
    purpose_mapping = {
        "machinability": [AnnealingType.SPHEROIDIZING, AnnealingType.FULL],
        "stress": [AnnealingType.STRESS_RELIEF],
        "softening": [AnnealingType.FULL, AnnealingType.ISOTHERMAL],
        "structure": [AnnealingType.FULL, AnnealingType.ISOTHERMAL],
        "cold_working": [AnnealingType.RECRYSTALLIZATION, AnnealingType.PROCESS],
    }

    recommended_types = purpose_mapping.get(purpose)
    if not recommended_types:
        return None

    material_data = ANNEALING_DATABASE.get(material_id)
    if not material_data:
        return None

    # Find best available type
    for ann_type in recommended_types:
        if ann_type in material_data:
            params = get_annealing_parameters(material_id, ann_type)
            return {
                "material_id": material_id,
                "purpose": purpose,
                "recommended_type": ann_type.value,
                "parameters": params,
            }

    return None
