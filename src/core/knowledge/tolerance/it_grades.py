"""
ISO Tolerance Grades (IT Grades) Knowledge Base.

Provides standard tolerance values according to ISO 286-1:2010.
Tolerance grades range from IT01 (most precise) to IT18 (least precise).

Reference:
- ISO 286-1:2010 Table 1 - Standard tolerance grades
- GB/T 1800.1-2020 (Chinese equivalent)
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union


class ITGrade(str, Enum):
    """ISO Standard Tolerance Grades."""

    IT01 = "IT01"
    IT0 = "IT0"
    IT1 = "IT1"
    IT2 = "IT2"
    IT3 = "IT3"
    IT4 = "IT4"
    IT5 = "IT5"
    IT6 = "IT6"
    IT7 = "IT7"
    IT8 = "IT8"
    IT9 = "IT9"
    IT10 = "IT10"
    IT11 = "IT11"
    IT12 = "IT12"
    IT13 = "IT13"
    IT14 = "IT14"
    IT15 = "IT15"
    IT16 = "IT16"
    IT17 = "IT17"
    IT18 = "IT18"


# Basic size ranges in mm (lower bound, upper bound)
# Per ISO 286-1:2010 Table 1
SIZE_RANGES: List[Tuple[float, float]] = [
    (0, 3),
    (3, 6),
    (6, 10),
    (10, 18),
    (18, 30),
    (30, 50),
    (50, 80),
    (80, 120),
    (120, 180),
    (180, 250),
    (250, 315),
    (315, 400),
    (400, 500),
    (500, 630),
    (630, 800),
    (800, 1000),
    (1000, 1250),
    (1250, 1600),
    (1600, 2000),
    (2000, 2500),
    (2500, 3150),
]


# Tolerance values in micrometers (μm)
# ISO 286-1:2010 Table 1 - Standard tolerance values
# Format: {size_range_index: {grade: tolerance_um}}
TOLERANCE_GRADES: Dict[int, Dict[str, float]] = {
    # 0-3 mm
    0: {
        "IT01": 0.3, "IT0": 0.5, "IT1": 0.8, "IT2": 1.2, "IT3": 2,
        "IT4": 3, "IT5": 4, "IT6": 6, "IT7": 10, "IT8": 14,
        "IT9": 25, "IT10": 40, "IT11": 60, "IT12": 100, "IT13": 140,
        "IT14": 250, "IT15": 400, "IT16": 600, "IT17": 1000, "IT18": 1400,
    },
    # 3-6 mm
    1: {
        "IT01": 0.4, "IT0": 0.6, "IT1": 1, "IT2": 1.5, "IT3": 2.5,
        "IT4": 4, "IT5": 5, "IT6": 8, "IT7": 12, "IT8": 18,
        "IT9": 30, "IT10": 48, "IT11": 75, "IT12": 120, "IT13": 180,
        "IT14": 300, "IT15": 480, "IT16": 750, "IT17": 1200, "IT18": 1800,
    },
    # 6-10 mm
    2: {
        "IT01": 0.4, "IT0": 0.6, "IT1": 1, "IT2": 1.5, "IT3": 2.5,
        "IT4": 4, "IT5": 6, "IT6": 9, "IT7": 15, "IT8": 22,
        "IT9": 36, "IT10": 58, "IT11": 90, "IT12": 150, "IT13": 220,
        "IT14": 360, "IT15": 580, "IT16": 900, "IT17": 1500, "IT18": 2200,
    },
    # 10-18 mm
    3: {
        "IT01": 0.5, "IT0": 0.8, "IT1": 1.2, "IT2": 2, "IT3": 3,
        "IT4": 5, "IT5": 8, "IT6": 11, "IT7": 18, "IT8": 27,
        "IT9": 43, "IT10": 70, "IT11": 110, "IT12": 180, "IT13": 270,
        "IT14": 430, "IT15": 700, "IT16": 1100, "IT17": 1800, "IT18": 2700,
    },
    # 18-30 mm
    4: {
        "IT01": 0.6, "IT0": 1, "IT1": 1.5, "IT2": 2.5, "IT3": 4,
        "IT4": 6, "IT5": 9, "IT6": 13, "IT7": 21, "IT8": 33,
        "IT9": 52, "IT10": 84, "IT11": 130, "IT12": 210, "IT13": 330,
        "IT14": 520, "IT15": 840, "IT16": 1300, "IT17": 2100, "IT18": 3300,
    },
    # 30-50 mm
    5: {
        "IT01": 0.6, "IT0": 1, "IT1": 1.5, "IT2": 2.5, "IT3": 4,
        "IT4": 7, "IT5": 11, "IT6": 16, "IT7": 25, "IT8": 39,
        "IT9": 62, "IT10": 100, "IT11": 160, "IT12": 250, "IT13": 390,
        "IT14": 620, "IT15": 1000, "IT16": 1600, "IT17": 2500, "IT18": 3900,
    },
    # 50-80 mm
    6: {
        "IT01": 0.8, "IT0": 1.2, "IT1": 2, "IT2": 3, "IT3": 5,
        "IT4": 8, "IT5": 13, "IT6": 19, "IT7": 30, "IT8": 46,
        "IT9": 74, "IT10": 120, "IT11": 190, "IT12": 300, "IT13": 460,
        "IT14": 740, "IT15": 1200, "IT16": 1900, "IT17": 3000, "IT18": 4600,
    },
    # 80-120 mm
    7: {
        "IT01": 1, "IT0": 1.5, "IT1": 2.5, "IT2": 4, "IT3": 6,
        "IT4": 10, "IT5": 15, "IT6": 22, "IT7": 35, "IT8": 54,
        "IT9": 87, "IT10": 140, "IT11": 220, "IT12": 350, "IT13": 540,
        "IT14": 870, "IT15": 1400, "IT16": 2200, "IT17": 3500, "IT18": 5400,
    },
    # 120-180 mm
    8: {
        "IT01": 1.2, "IT0": 2, "IT1": 3.5, "IT2": 5, "IT3": 8,
        "IT4": 12, "IT5": 18, "IT6": 25, "IT7": 40, "IT8": 63,
        "IT9": 100, "IT10": 160, "IT11": 250, "IT12": 400, "IT13": 630,
        "IT14": 1000, "IT15": 1600, "IT16": 2500, "IT17": 4000, "IT18": 6300,
    },
    # 180-250 mm
    9: {
        "IT01": 2, "IT0": 3, "IT1": 4.5, "IT2": 7, "IT3": 10,
        "IT4": 14, "IT5": 20, "IT6": 29, "IT7": 46, "IT8": 72,
        "IT9": 115, "IT10": 185, "IT11": 290, "IT12": 460, "IT13": 720,
        "IT14": 1150, "IT15": 1850, "IT16": 2900, "IT17": 4600, "IT18": 7200,
    },
    # 250-315 mm
    10: {
        "IT01": 2.5, "IT0": 4, "IT1": 6, "IT2": 8, "IT3": 12,
        "IT4": 16, "IT5": 23, "IT6": 32, "IT7": 52, "IT8": 81,
        "IT9": 130, "IT10": 210, "IT11": 320, "IT12": 520, "IT13": 810,
        "IT14": 1300, "IT15": 2100, "IT16": 3200, "IT17": 5200, "IT18": 8100,
    },
    # 315-400 mm
    11: {
        "IT01": 3, "IT0": 5, "IT1": 7, "IT2": 9, "IT3": 13,
        "IT4": 18, "IT5": 25, "IT6": 36, "IT7": 57, "IT8": 89,
        "IT9": 140, "IT10": 230, "IT11": 360, "IT12": 570, "IT13": 890,
        "IT14": 1400, "IT15": 2300, "IT16": 3600, "IT17": 5700, "IT18": 8900,
    },
    # 400-500 mm
    12: {
        "IT01": 4, "IT0": 6, "IT1": 8, "IT2": 10, "IT3": 15,
        "IT4": 20, "IT5": 27, "IT6": 40, "IT7": 63, "IT8": 97,
        "IT9": 155, "IT10": 250, "IT11": 400, "IT12": 630, "IT13": 970,
        "IT14": 1550, "IT15": 2500, "IT16": 4000, "IT17": 6300, "IT18": 9700,
    },
    # 500-630 mm
    13: {
        "IT01": 4.5, "IT0": 7, "IT1": 9, "IT2": 11, "IT3": 16,
        "IT4": 22, "IT5": 32, "IT6": 44, "IT7": 70, "IT8": 110,
        "IT9": 175, "IT10": 280, "IT11": 440, "IT12": 700, "IT13": 1100,
        "IT14": 1750, "IT15": 2800, "IT16": 4400, "IT17": 7000, "IT18": 11000,
    },
    # 630-800 mm
    14: {
        "IT01": 5, "IT0": 8, "IT1": 10, "IT2": 13, "IT3": 18,
        "IT4": 25, "IT5": 36, "IT6": 50, "IT7": 80, "IT8": 125,
        "IT9": 200, "IT10": 320, "IT11": 500, "IT12": 800, "IT13": 1250,
        "IT14": 2000, "IT15": 3200, "IT16": 5000, "IT17": 8000, "IT18": 12500,
    },
    # 800-1000 mm
    15: {
        "IT01": 5.5, "IT0": 9, "IT1": 11, "IT2": 15, "IT3": 21,
        "IT4": 28, "IT5": 40, "IT6": 56, "IT7": 90, "IT8": 140,
        "IT9": 230, "IT10": 360, "IT11": 560, "IT12": 900, "IT13": 1400,
        "IT14": 2300, "IT15": 3600, "IT16": 5600, "IT17": 9000, "IT18": 14000,
    },
    # 1000-1250 mm
    16: {
        "IT01": 6.5, "IT0": 10, "IT1": 13, "IT2": 18, "IT3": 24,
        "IT4": 33, "IT5": 47, "IT6": 66, "IT7": 105, "IT8": 165,
        "IT9": 260, "IT10": 420, "IT11": 660, "IT12": 1050, "IT13": 1650,
        "IT14": 2600, "IT15": 4200, "IT16": 6600, "IT17": 10500, "IT18": 16500,
    },
    # 1250-1600 mm
    17: {
        "IT01": 8, "IT0": 12, "IT1": 15, "IT2": 21, "IT3": 29,
        "IT4": 39, "IT5": 55, "IT6": 78, "IT7": 125, "IT8": 195,
        "IT9": 310, "IT10": 500, "IT11": 780, "IT12": 1250, "IT13": 1950,
        "IT14": 3100, "IT15": 5000, "IT16": 7800, "IT17": 12500, "IT18": 19500,
    },
    # 1600-2000 mm
    18: {
        "IT01": 9, "IT0": 14, "IT1": 18, "IT2": 25, "IT3": 35,
        "IT4": 46, "IT5": 65, "IT6": 92, "IT7": 150, "IT8": 230,
        "IT9": 370, "IT10": 600, "IT11": 920, "IT12": 1500, "IT13": 2300,
        "IT14": 3700, "IT15": 6000, "IT16": 9200, "IT17": 15000, "IT18": 23000,
    },
    # 2000-2500 mm
    19: {
        "IT01": 10.5, "IT0": 16, "IT1": 21, "IT2": 30, "IT3": 41,
        "IT4": 55, "IT5": 78, "IT6": 110, "IT7": 175, "IT8": 280,
        "IT9": 440, "IT10": 700, "IT11": 1100, "IT12": 1750, "IT13": 2800,
        "IT14": 4400, "IT15": 7000, "IT16": 11000, "IT17": 17500, "IT18": 28000,
    },
    # 2500-3150 mm
    20: {
        "IT01": 12.5, "IT0": 19, "IT1": 25, "IT2": 36, "IT3": 50,
        "IT4": 68, "IT5": 96, "IT6": 135, "IT7": 210, "IT8": 330,
        "IT9": 540, "IT10": 860, "IT11": 1350, "IT12": 2100, "IT13": 3300,
        "IT14": 5400, "IT15": 8600, "IT16": 13500, "IT17": 21000, "IT18": 33000,
    },
}


# Grade application guidance
GRADE_APPLICATIONS: Dict[str, Dict[str, str]] = {
    "IT01": {
        "name_zh": "超精密级",
        "name_en": "Ultra-precision",
        "typical_use": "量块、精密量具",
        "typical_use_en": "Gauge blocks, precision measuring instruments",
        "process": "研磨、超精磨",
        "process_en": "Lapping, super-finishing",
    },
    "IT0": {
        "name_zh": "超精密级",
        "name_en": "Ultra-precision",
        "typical_use": "量块、精密量具",
        "typical_use_en": "Gauge blocks, precision measuring instruments",
        "process": "研磨、超精磨",
        "process_en": "Lapping, super-finishing",
    },
    "IT1": {
        "name_zh": "超精密级",
        "name_en": "Ultra-precision",
        "typical_use": "精密量具、高精度轴承",
        "typical_use_en": "Precision gauges, high-precision bearings",
        "process": "研磨、精密磨削",
        "process_en": "Lapping, precision grinding",
    },
    "IT2": {
        "name_zh": "超精密级",
        "name_en": "Ultra-precision",
        "typical_use": "精密量具、精密轴承",
        "typical_use_en": "Precision gauges, precision bearings",
        "process": "精密磨削",
        "process_en": "Precision grinding",
    },
    "IT3": {
        "name_zh": "超精密级",
        "name_en": "Ultra-precision",
        "typical_use": "精密轴承、高精度机床主轴",
        "typical_use_en": "Precision bearings, high-precision spindles",
        "process": "精密磨削",
        "process_en": "Precision grinding",
    },
    "IT4": {
        "name_zh": "精密级",
        "name_en": "Precision",
        "typical_use": "精密轴承、精密丝杠",
        "typical_use_en": "Precision bearings, precision lead screws",
        "process": "精密磨削",
        "process_en": "Precision grinding",
    },
    "IT5": {
        "name_zh": "精密级",
        "name_en": "Precision",
        "typical_use": "精密配合、高精度机械",
        "typical_use_en": "Precision fits, high-precision machinery",
        "process": "磨削、精密车削",
        "process_en": "Grinding, precision turning",
    },
    "IT6": {
        "name_zh": "精密级",
        "name_en": "Precision",
        "typical_use": "配合孔轴、精密机械",
        "typical_use_en": "Fit holes/shafts, precision machinery",
        "process": "磨削、精密车削、铰孔",
        "process_en": "Grinding, precision turning, reaming",
    },
    "IT7": {
        "name_zh": "较精密级",
        "name_en": "Semi-precision",
        "typical_use": "一般配合、机床主轴",
        "typical_use_en": "General fits, machine spindles",
        "process": "精车、精铣、铰孔",
        "process_en": "Fine turning, fine milling, reaming",
    },
    "IT8": {
        "name_zh": "中等精度",
        "name_en": "Medium precision",
        "typical_use": "一般配合、轴承座",
        "typical_use_en": "General fits, bearing housings",
        "process": "精车、精铣、镗孔",
        "process_en": "Fine turning, fine milling, boring",
    },
    "IT9": {
        "name_zh": "中等精度",
        "name_en": "Medium precision",
        "typical_use": "非配合尺寸、支架",
        "typical_use_en": "Non-fit dimensions, brackets",
        "process": "车削、铣削",
        "process_en": "Turning, milling",
    },
    "IT10": {
        "name_zh": "中等精度",
        "name_en": "Medium precision",
        "typical_use": "非配合尺寸、机架",
        "typical_use_en": "Non-fit dimensions, frames",
        "process": "车削、铣削",
        "process_en": "Turning, milling",
    },
    "IT11": {
        "name_zh": "粗糙级",
        "name_en": "Coarse",
        "typical_use": "非重要尺寸、粗加工",
        "typical_use_en": "Non-critical dimensions, rough machining",
        "process": "粗车、粗铣",
        "process_en": "Rough turning, rough milling",
    },
    "IT12": {
        "name_zh": "粗糙级",
        "name_en": "Coarse",
        "typical_use": "自由尺寸、锻件",
        "typical_use_en": "Free dimensions, forgings",
        "process": "粗加工、锻造",
        "process_en": "Rough machining, forging",
    },
    "IT13": {
        "name_zh": "粗糙级",
        "name_en": "Coarse",
        "typical_use": "铸件毛坯、锻件毛坯",
        "typical_use_en": "Casting blanks, forging blanks",
        "process": "铸造、锻造",
        "process_en": "Casting, forging",
    },
    "IT14": {
        "name_zh": "粗糙级",
        "name_en": "Coarse",
        "typical_use": "铸件、毛坯件",
        "typical_use_en": "Castings, blanks",
        "process": "铸造、冲压",
        "process_en": "Casting, stamping",
    },
    "IT15": {
        "name_zh": "毛坯级",
        "name_en": "Blank",
        "typical_use": "大型铸件、焊接件",
        "typical_use_en": "Large castings, welded structures",
        "process": "铸造、焊接",
        "process_en": "Casting, welding",
    },
    "IT16": {
        "name_zh": "毛坯级",
        "name_en": "Blank",
        "typical_use": "大型铸件、焊接件",
        "typical_use_en": "Large castings, welded structures",
        "process": "铸造、焊接",
        "process_en": "Casting, welding",
    },
    "IT17": {
        "name_zh": "毛坯级",
        "name_en": "Blank",
        "typical_use": "大型焊接件、结构件",
        "typical_use_en": "Large welded structures, structural parts",
        "process": "焊接、热轧",
        "process_en": "Welding, hot rolling",
    },
    "IT18": {
        "name_zh": "毛坯级",
        "name_en": "Blank",
        "typical_use": "大型结构件、热轧件",
        "typical_use_en": "Large structural parts, hot rolled parts",
        "process": "热轧、锻造",
        "process_en": "Hot rolling, forging",
    },
}


def _get_size_range_index(size_mm: float) -> Optional[int]:
    """Find the size range index for a given nominal size."""
    for i, (lower, upper) in enumerate(SIZE_RANGES):
        if lower < size_mm <= upper:
            return i
        if i == 0 and size_mm > 0 and size_mm <= upper:
            return i
    return None


def get_tolerance_value(
    nominal_size_mm: float,
    grade: Union[str, ITGrade],
) -> Optional[float]:
    """
    Get the tolerance value for a given size and IT grade.

    Args:
        nominal_size_mm: Nominal dimension in millimeters (0 < size <= 3150)
        grade: IT grade (e.g., "IT7" or ITGrade.IT7)

    Returns:
        Tolerance value in micrometers (μm), or None if out of range

    Example:
        >>> get_tolerance_value(25, "IT7")
        21.0
        >>> get_tolerance_value(25, ITGrade.IT7)
        21.0
    """
    if nominal_size_mm <= 0 or nominal_size_mm > 3150:
        return None

    grade_str = grade.value if isinstance(grade, ITGrade) else grade

    range_idx = _get_size_range_index(nominal_size_mm)
    if range_idx is None:
        return None

    grade_data = TOLERANCE_GRADES.get(range_idx)
    if grade_data is None:
        return None

    return grade_data.get(grade_str)


def get_tolerance_table(
    nominal_size_mm: float,
    grades: Optional[List[Union[str, ITGrade]]] = None,
) -> Dict[str, float]:
    """
    Get tolerance values for multiple grades at a given size.

    Args:
        nominal_size_mm: Nominal dimension in millimeters
        grades: List of grades to include (default: IT5-IT14)

    Returns:
        Dictionary of {grade: tolerance_um}

    Example:
        >>> get_tolerance_table(25)
        {'IT5': 9.0, 'IT6': 13.0, 'IT7': 21.0, ...}
    """
    if grades is None:
        grades = [f"IT{i}" for i in range(5, 15)]

    result = {}
    for grade in grades:
        grade_str = grade.value if isinstance(grade, ITGrade) else grade
        value = get_tolerance_value(nominal_size_mm, grade_str)
        if value is not None:
            result[grade_str] = value

    return result


def get_grade_info(grade: Union[str, ITGrade]) -> Optional[Dict[str, str]]:
    """
    Get application information for a tolerance grade.

    Args:
        grade: IT grade (e.g., "IT7" or ITGrade.IT7)

    Returns:
        Dictionary with name, typical_use, and process information
    """
    grade_str = grade.value if isinstance(grade, ITGrade) else grade
    return GRADE_APPLICATIONS.get(grade_str)


def suggest_grade_for_application(
    application: str,
    precision_level: str = "medium",
) -> List[str]:
    """
    Suggest appropriate IT grades for a given application.

    Args:
        application: Application type (e.g., "bearing", "fit", "general")
        precision_level: "high", "medium", or "low"

    Returns:
        List of recommended IT grades
    """
    suggestions = {
        "bearing": {
            "high": ["IT4", "IT5"],
            "medium": ["IT5", "IT6", "IT7"],
            "low": ["IT7", "IT8"],
        },
        "fit": {
            "high": ["IT5", "IT6"],
            "medium": ["IT6", "IT7", "IT8"],
            "low": ["IT8", "IT9", "IT10"],
        },
        "spindle": {
            "high": ["IT3", "IT4", "IT5"],
            "medium": ["IT5", "IT6"],
            "low": ["IT6", "IT7"],
        },
        "general": {
            "high": ["IT6", "IT7"],
            "medium": ["IT8", "IT9", "IT10"],
            "low": ["IT11", "IT12"],
        },
        "casting": {
            "high": ["IT12", "IT13"],
            "medium": ["IT13", "IT14"],
            "low": ["IT14", "IT15", "IT16"],
        },
    }

    app_key = application.lower()
    if app_key not in suggestions:
        app_key = "general"

    return suggestions[app_key].get(precision_level, suggestions[app_key]["medium"])
