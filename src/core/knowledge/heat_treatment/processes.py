"""
Heat Treatment Process Parameters Knowledge Base.

Provides temperature, time, and cooling parameters for common
heat treatment processes.

Reference:
- ASM Handbook Volume 4 - Heat Treating
- Tool Steel Handbook
- ISO 683 - Heat-treatable steels
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import math


class HeatTreatmentProcess(str, Enum):
    """Heat treatment process types."""

    # Hardening processes
    QUENCH_HARDENING = "quench_hardening"  # 淬火
    TEMPERING = "tempering"  # 回火
    QUENCH_TEMPER = "quench_temper"  # 调质 (淬火+回火)

    # Annealing processes
    FULL_ANNEALING = "full_annealing"  # 完全退火
    SPHEROIDIZING = "spheroidizing"  # 球化退火
    STRESS_RELIEF = "stress_relief"  # 去应力退火
    RECRYSTALLIZATION = "recrystallization"  # 再结晶退火

    # Normalizing
    NORMALIZING = "normalizing"  # 正火

    # Surface hardening
    CARBURIZING = "carburizing"  # 渗碳
    NITRIDING = "nitriding"  # 氮化
    CARBONITRIDING = "carbonitriding"  # 碳氮共渗
    INDUCTION_HARDENING = "induction_hardening"  # 感应淬火

    # Solution treatment
    SOLUTION_TREATMENT = "solution_treatment"  # 固溶处理
    AGING = "aging"  # 时效处理


class QuenchMedia(str, Enum):
    """Quenching media types."""

    WATER = "water"  # 水淬
    OIL = "oil"  # 油淬
    POLYMER = "polymer"  # 聚合物淬火液
    SALT = "salt"  # 盐浴
    AIR = "air"  # 空冷
    VACUUM = "vacuum"  # 真空


@dataclass
class HeatTreatmentParameters:
    """Heat treatment process parameters."""

    process: HeatTreatmentProcess
    material_id: str

    # Temperature parameters
    temperature_min: float  # °C
    temperature_max: float  # °C
    temperature_recommended: float  # °C

    # Holding time
    holding_time_factor: float  # min/mm or min/inch
    holding_time_min: float  # minutes
    holding_time_max: float  # minutes

    # Heating rate
    heating_rate_max: Optional[float] = None  # °C/hour

    # Cooling
    quench_media: Optional[QuenchMedia] = None
    cooling_rate_min: Optional[float] = None  # °C/second
    cooling_rate_max: Optional[float] = None  # °C/second

    # Atmosphere
    atmosphere: Optional[str] = None  # "air", "nitrogen", "vacuum", etc.

    # Expected results
    hardness_range: Optional[Tuple[float, float]] = None  # HRC or HB

    notes_zh: str = ""
    notes_en: str = ""


# Heat treatment database
HEAT_TREATMENT_DATABASE: Dict[str, Dict[str, Dict]] = {
    # 45 steel (AISI 1045)
    "45": {
        HeatTreatmentProcess.QUENCH_HARDENING: {
            "temperature_range": (820, 860),
            "temperature_recommended": 840,
            "holding_time_factor": 1.5,  # min/mm
            "quench_media": QuenchMedia.WATER,
            "hardness_expected": (55, 60),  # HRC
            "notes_zh": "45钢水淬，临界直径约25mm",
            "notes_en": "Water quench, critical diameter ~25mm",
        },
        HeatTreatmentProcess.TEMPERING: {
            "low": {
                "temperature_range": (150, 250),
                "hardness_expected": (54, 58),
                "notes_zh": "低温回火，保持高硬度",
            },
            "medium": {
                "temperature_range": (350, 500),
                "hardness_expected": (40, 50),
                "notes_zh": "中温回火，提高韧性",
            },
            "high": {
                "temperature_range": (500, 650),
                "hardness_expected": (25, 35),
                "notes_zh": "高温回火(调质)，综合性能",
            },
        },
        HeatTreatmentProcess.NORMALIZING: {
            "temperature_range": (850, 880),
            "temperature_recommended": 860,
            "holding_time_factor": 1.0,
            "quench_media": QuenchMedia.AIR,
            "hardness_expected": (170, 220),  # HB
            "notes_zh": "正火细化晶粒，改善切削性能",
        },
        HeatTreatmentProcess.FULL_ANNEALING: {
            "temperature_range": (820, 850),
            "temperature_recommended": 830,
            "holding_time_factor": 2.0,
            "cooling_rate": "furnace",
            "hardness_expected": (156, 197),  # HB
            "notes_zh": "完全退火，消除加工硬化",
        },
        HeatTreatmentProcess.INDUCTION_HARDENING: {
            "temperature_range": (850, 950),
            "temperature_recommended": 900,
            "quench_media": QuenchMedia.WATER,
            "hardness_expected": (55, 62),  # HRC
            "case_depth": (1.0, 3.0),  # mm
            "notes_zh": "感应淬火，表面硬化",
        },
    },
    # 40Cr steel (AISI 5140)
    "40Cr": {
        HeatTreatmentProcess.QUENCH_HARDENING: {
            "temperature_range": (830, 860),
            "temperature_recommended": 850,
            "holding_time_factor": 1.5,
            "quench_media": QuenchMedia.OIL,
            "hardness_expected": (55, 60),
            "notes_zh": "40Cr油淬，淬透性好",
        },
        HeatTreatmentProcess.TEMPERING: {
            "low": {
                "temperature_range": (150, 250),
                "hardness_expected": (54, 58),
            },
            "medium": {
                "temperature_range": (400, 500),
                "hardness_expected": (45, 52),
            },
            "high": {
                "temperature_range": (500, 650),
                "hardness_expected": (28, 35),
                "notes_zh": "调质处理，HB 220-280",
            },
        },
        HeatTreatmentProcess.NORMALIZING: {
            "temperature_range": (860, 880),
            "temperature_recommended": 870,
            "holding_time_factor": 1.0,
            "quench_media": QuenchMedia.AIR,
            "hardness_expected": (179, 229),
        },
        HeatTreatmentProcess.CARBURIZING: {
            "temperature_range": (900, 950),
            "temperature_recommended": 920,
            "carburizing_time": 4,  # hours per 0.5mm case depth
            "case_depth": (0.5, 2.0),
            "carbon_potential": 1.0,
            "notes_zh": "气体渗碳处理",
        },
    },
    # Cr12MoV (D2 tool steel)
    "Cr12MoV": {
        HeatTreatmentProcess.QUENCH_HARDENING: {
            "temperature_range": (1010, 1040),
            "temperature_recommended": 1020,
            "holding_time_factor": 0.5,
            "preheat_stages": [(650, 15), (850, 10)],
            "quench_media": QuenchMedia.OIL,
            "hardness_expected": (60, 64),
            "notes_zh": "D2钢需分级预热，油淬或气冷",
        },
        HeatTreatmentProcess.TEMPERING: {
            "standard": {
                "temperature_range": (200, 250),
                "cycles": 2,
                "hardness_expected": (58, 62),
                "notes_zh": "二次回火，每次2小时以上",
            },
            "high_toughness": {
                "temperature_range": (480, 520),
                "cycles": 2,
                "hardness_expected": (56, 60),
                "notes_zh": "高温回火获得更好韧性",
            },
        },
        HeatTreatmentProcess.SPHEROIDIZING: {
            "temperature_range": (850, 870),
            "holding_time": 4,  # hours
            "cooling_rate": "furnace to 700°C, then air",
            "hardness_expected": (217, 255),
            "notes_zh": "球化退火改善可加工性",
        },
    },
    # 304 stainless steel
    "304": {
        HeatTreatmentProcess.SOLUTION_TREATMENT: {
            "temperature_range": (1010, 1150),
            "temperature_recommended": 1050,
            "holding_time_factor": 2.5,
            "quench_media": QuenchMedia.WATER,
            "hardness_expected": (130, 180),  # HB
            "notes_zh": "固溶处理恢复耐腐蚀性能",
        },
        HeatTreatmentProcess.STRESS_RELIEF: {
            "temperature_range": (400, 500),
            "temperature_recommended": 450,
            "holding_time_factor": 2.0,
            "quench_media": QuenchMedia.AIR,
            "notes_zh": "消除加工应力，不影响耐腐蚀性",
        },
    },
    # 6061 aluminum
    "6061": {
        HeatTreatmentProcess.SOLUTION_TREATMENT: {
            "temperature_range": (510, 550),
            "temperature_recommended": 530,
            "holding_time_factor": 3.0,
            "quench_media": QuenchMedia.WATER,
            "notes_zh": "固溶处理后需时效",
        },
        HeatTreatmentProcess.AGING: {
            "T6": {
                "temperature_range": (160, 180),
                "temperature_recommended": 175,
                "holding_time": 8,  # hours
                "hardness_expected": (95, 105),  # HB
                "tensile_strength": 310,  # MPa
                "notes_zh": "T6人工时效，最高强度状态",
            },
            "T4": {
                "temperature_range": (20, 25),
                "holding_time": 96,  # hours (natural aging)
                "notes_zh": "T4自然时效",
            },
        },
    },
    # 7075 aluminum
    "7075": {
        HeatTreatmentProcess.SOLUTION_TREATMENT: {
            "temperature_range": (465, 480),
            "temperature_recommended": 470,
            "holding_time_factor": 3.0,
            "quench_media": QuenchMedia.WATER,
            "notes_zh": "固溶温度控制严格，防止过烧",
        },
        HeatTreatmentProcess.AGING: {
            "T6": {
                "temperature_range": (115, 125),
                "temperature_recommended": 120,
                "holding_time": 24,
                "hardness_expected": (150, 160),
                "tensile_strength": 570,
                "notes_zh": "T6时效，最高强度",
            },
            "T73": {
                "stages": [(105, 8), (175, 8)],
                "hardness_expected": (135, 145),
                "notes_zh": "T73过时效，提高抗应力腐蚀性",
            },
        },
    },
}

# Quench media characteristics
QUENCH_MEDIA_DATA: Dict[QuenchMedia, Dict] = {
    QuenchMedia.WATER: {
        "cooling_rate": (200, 300),  # °C/s at 300°C
        "temperature_range": (15, 40),
        "agitation": "required",
        "notes_zh": "冷却速度最快，适用于碳钢",
        "notes_en": "Fastest cooling, suitable for carbon steels",
    },
    QuenchMedia.OIL: {
        "cooling_rate": (50, 80),
        "temperature_range": (40, 80),
        "agitation": "recommended",
        "notes_zh": "冷却较缓，减少变形开裂风险",
        "notes_en": "Moderate cooling, reduces distortion and cracking",
    },
    QuenchMedia.POLYMER: {
        "cooling_rate": (80, 150),
        "temperature_range": (20, 40),
        "concentration": (5, 25),  # %
        "notes_zh": "可调节冷却速度的水溶性淬火液",
    },
    QuenchMedia.SALT: {
        "cooling_rate": (100, 150),
        "temperature_range": (150, 550),
        "notes_zh": "分级淬火用盐浴，马氏体等温淬火",
    },
    QuenchMedia.AIR: {
        "cooling_rate": (5, 15),
        "notes_zh": "空冷，适用于高淬透性钢",
    },
    QuenchMedia.VACUUM: {
        "cooling_rate": (20, 50),
        "notes_zh": "真空淬火，表面质量优良",
    },
}


def get_heat_treatment_parameters(
    material_id: str,
    process: Union[str, HeatTreatmentProcess],
    variant: Optional[str] = None,
) -> Optional[HeatTreatmentParameters]:
    """
    Get heat treatment parameters for a material and process.

    Args:
        material_id: Material identifier
        process: Heat treatment process
        variant: Process variant (e.g., "low", "high", "T6")

    Returns:
        HeatTreatmentParameters or None

    Example:
        >>> params = get_heat_treatment_parameters("45", "quench_hardening")
        >>> print(f"Temperature: {params.temperature_recommended}°C")
    """
    if isinstance(process, str):
        try:
            process = HeatTreatmentProcess(process.lower())
        except ValueError:
            return None

    material_data = HEAT_TREATMENT_DATABASE.get(material_id)
    if not material_data:
        return None

    process_data = material_data.get(process)
    if not process_data:
        return None

    # Handle variant selection
    if variant and isinstance(process_data, dict):
        if variant in process_data:
            process_data = process_data[variant]
        elif "temperature_range" not in process_data:
            # Process data is a dict of variants, select first if no variant specified
            first_key = list(process_data.keys())[0]
            process_data = process_data[first_key]

    if "temperature_range" not in process_data:
        return None

    temp_range = process_data["temperature_range"]

    return HeatTreatmentParameters(
        process=process,
        material_id=material_id,
        temperature_min=temp_range[0],
        temperature_max=temp_range[1],
        temperature_recommended=process_data.get("temperature_recommended", sum(temp_range) / 2),
        holding_time_factor=process_data.get("holding_time_factor", 1.0),
        holding_time_min=process_data.get("holding_time", 30),
        holding_time_max=process_data.get("holding_time", 60) * 2,
        heating_rate_max=process_data.get("heating_rate_max"),
        quench_media=process_data.get("quench_media"),
        hardness_range=process_data.get("hardness_expected"),
        atmosphere=process_data.get("atmosphere"),
        notes_zh=process_data.get("notes_zh", ""),
        notes_en=process_data.get("notes_en", ""),
    )


def get_quench_media(media: Union[str, QuenchMedia]) -> Optional[Dict]:
    """
    Get quench media characteristics.

    Args:
        media: Quench media type

    Returns:
        Dict with media characteristics or None
    """
    if isinstance(media, str):
        try:
            media = QuenchMedia(media.lower())
        except ValueError:
            return None

    return QUENCH_MEDIA_DATA.get(media)


def calculate_holding_time(
    thickness: float,
    time_factor: float = 1.0,
    geometry: str = "solid",
) -> Tuple[float, float]:
    """
    Calculate heat treatment holding time based on section thickness.

    Args:
        thickness: Maximum section thickness (mm)
        time_factor: Time factor (min/mm)
        geometry: "solid", "hollow", or "complex"

    Returns:
        Tuple of (min_time, max_time) in minutes

    Example:
        >>> time = calculate_holding_time(25, time_factor=1.5)
        >>> print(f"Hold time: {time[0]}-{time[1]} minutes")
    """
    # Geometry factors
    geometry_factors = {
        "solid": 1.0,
        "hollow": 0.8,
        "complex": 1.2,
    }
    geo_factor = geometry_factors.get(geometry, 1.0)

    base_time = thickness * time_factor * geo_factor

    # Minimum practical time
    min_time = max(15, base_time * 0.8)
    max_time = base_time * 1.5

    return (round(min_time), round(max_time))


def recommend_process_for_hardness(
    material_id: str,
    target_hardness: float,
    hardness_scale: str = "HRC",
) -> Optional[Dict]:
    """
    Recommend heat treatment process to achieve target hardness.

    Args:
        material_id: Material identifier
        target_hardness: Desired hardness value
        hardness_scale: "HRC", "HB", or "HV"

    Returns:
        Dict with recommended process and parameters
    """
    material_data = HEAT_TREATMENT_DATABASE.get(material_id)
    if not material_data:
        return None

    recommendations = []

    for process, data in material_data.items():
        if isinstance(data, dict):
            if "hardness_expected" in data:
                h_min, h_max = data["hardness_expected"]
                if h_min <= target_hardness <= h_max:
                    recommendations.append({
                        "process": process,
                        "variant": None,
                        "hardness_range": (h_min, h_max),
                        "notes": data.get("notes_zh", ""),
                    })
            else:
                # Check variants
                for variant, variant_data in data.items():
                    if isinstance(variant_data, dict) and "hardness_expected" in variant_data:
                        h_min, h_max = variant_data["hardness_expected"]
                        if h_min <= target_hardness <= h_max:
                            recommendations.append({
                                "process": process,
                                "variant": variant,
                                "hardness_range": (h_min, h_max),
                                "notes": variant_data.get("notes_zh", ""),
                            })

    if not recommendations:
        return None

    return {
        "material_id": material_id,
        "target_hardness": target_hardness,
        "recommendations": recommendations,
    }
