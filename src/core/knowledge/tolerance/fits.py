"""
ISO Fit Systems Knowledge Base.

Provides standard hole-basis and shaft-basis fit systems according to ISO 286-2:2010.
Includes fundamental deviations for shafts (a-zc) and holes (A-ZC).

Reference:
- ISO 286-2:2010 - Tables of standard tolerance classes and limit deviations
- GB/T 1800.2-2020 (Chinese equivalent)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .it_grades import ITGrade, get_tolerance_value


class FitType(str, Enum):
    """Classification of fits by clearance/interference."""

    CLEARANCE = "clearance"  # 间隙配合
    TRANSITION = "transition"  # 过渡配合
    INTERFERENCE = "interference"  # 过盈配合


class FitClass(str, Enum):
    """Common fit classifications."""

    # Clearance fits (间隙配合)
    LOOSE_RUNNING = "loose_running"  # 松动配合 H11/c11
    FREE_RUNNING = "free_running"  # 自由转动配合 H9/d9
    EASY_RUNNING = "easy_running"  # 易转配合 H8/f7
    NORMAL_RUNNING = "normal_running"  # 正常转动配合 H7/g6
    SLIDING = "sliding"  # 滑动配合 H7/h6
    CLOSE_RUNNING = "close_running"  # 紧密转动配合 H7/h6

    # Transition fits (过渡配合)
    LOCATION_CLEARANCE = "location_clearance"  # 定位间隙配合 H7/js6
    LOCATION_TRANSITION = "location_transition"  # 定位过渡配合 H7/k6
    LOCATION_INTERFERENCE = "location_interference"  # 定位过盈配合 H7/n6

    # Interference fits (过盈配合)
    LIGHT_PRESS = "light_press"  # 轻压配合 H7/p6
    MEDIUM_PRESS = "medium_press"  # 中压配合 H7/r6
    HEAVY_PRESS = "heavy_press"  # 重压配合 H7/s6
    FORCE = "force"  # 强力配合 H7/u6


@dataclass
class ShaftDeviation:
    """Shaft fundamental deviation data."""

    symbol: str  # e.g., "g", "h", "k"
    name_zh: str
    name_en: str
    fit_type: FitType
    deviation_formula: str  # Description of deviation calculation


@dataclass
class HoleDeviation:
    """Hole fundamental deviation data."""

    symbol: str  # e.g., "H", "G", "K"
    name_zh: str
    name_en: str
    fit_type: FitType
    deviation_formula: str


# Fundamental deviations for shafts (lowercase letters)
# Values in μm, based on ISO 286-2:2010
# Format: {symbol: {size_range: (upper_deviation, lower_deviation)}}
# For each shaft symbol, we store the fundamental deviation

SHAFT_SYMBOLS: Dict[str, ShaftDeviation] = {
    "a": ShaftDeviation("a", "最大间隙", "Maximum clearance", FitType.CLEARANCE, "Large negative"),
    "b": ShaftDeviation("b", "大间隙", "Large clearance", FitType.CLEARANCE, "Large negative"),
    "c": ShaftDeviation("c", "间隙", "Clearance", FitType.CLEARANCE, "Negative"),
    "d": ShaftDeviation("d", "间隙", "Clearance", FitType.CLEARANCE, "Negative"),
    "e": ShaftDeviation("e", "小间隙", "Small clearance", FitType.CLEARANCE, "Small negative"),
    "f": ShaftDeviation("f", "小间隙", "Small clearance", FitType.CLEARANCE, "Small negative"),
    "g": ShaftDeviation("g", "微间隙", "Minimal clearance", FitType.CLEARANCE, "Very small negative"),
    "h": ShaftDeviation("h", "基准轴", "Basic shaft", FitType.CLEARANCE, "Zero upper deviation"),
    "js": ShaftDeviation("js", "对称", "Symmetrical", FitType.TRANSITION, "Symmetrical ±IT/2"),
    "j": ShaftDeviation("j", "微过渡", "Slight transition", FitType.TRANSITION, "Small positive/negative"),
    "k": ShaftDeviation("k", "过渡", "Transition", FitType.TRANSITION, "Small positive"),
    "m": ShaftDeviation("m", "过渡", "Transition", FitType.TRANSITION, "Positive"),
    "n": ShaftDeviation("n", "过渡/过盈", "Transition/interference", FitType.TRANSITION, "Positive"),
    "p": ShaftDeviation("p", "轻过盈", "Light interference", FitType.INTERFERENCE, "Positive"),
    "r": ShaftDeviation("r", "过盈", "Interference", FitType.INTERFERENCE, "Positive"),
    "s": ShaftDeviation("s", "中过盈", "Medium interference", FitType.INTERFERENCE, "Larger positive"),
    "t": ShaftDeviation("t", "中过盈", "Medium interference", FitType.INTERFERENCE, "Larger positive"),
    "u": ShaftDeviation("u", "重过盈", "Heavy interference", FitType.INTERFERENCE, "Large positive"),
    "v": ShaftDeviation("v", "重过盈", "Heavy interference", FitType.INTERFERENCE, "Large positive"),
    "x": ShaftDeviation("x", "重过盈", "Heavy interference", FitType.INTERFERENCE, "Very large positive"),
    "y": ShaftDeviation("y", "重过盈", "Heavy interference", FitType.INTERFERENCE, "Very large positive"),
    "z": ShaftDeviation("z", "最大过盈", "Maximum interference", FitType.INTERFERENCE, "Maximum positive"),
    "za": ShaftDeviation("za", "最大过盈", "Maximum interference", FitType.INTERFERENCE, "Maximum positive"),
    "zb": ShaftDeviation("zb", "最大过盈", "Maximum interference", FitType.INTERFERENCE, "Maximum positive"),
    "zc": ShaftDeviation("zc", "最大过盈", "Maximum interference", FitType.INTERFERENCE, "Maximum positive"),
}

HOLE_SYMBOLS: Dict[str, HoleDeviation] = {
    "A": HoleDeviation("A", "最大间隙", "Maximum clearance", FitType.CLEARANCE, "Large positive"),
    "B": HoleDeviation("B", "大间隙", "Large clearance", FitType.CLEARANCE, "Large positive"),
    "C": HoleDeviation("C", "间隙", "Clearance", FitType.CLEARANCE, "Positive"),
    "D": HoleDeviation("D", "间隙", "Clearance", FitType.CLEARANCE, "Positive"),
    "E": HoleDeviation("E", "小间隙", "Small clearance", FitType.CLEARANCE, "Small positive"),
    "F": HoleDeviation("F", "小间隙", "Small clearance", FitType.CLEARANCE, "Small positive"),
    "G": HoleDeviation("G", "微间隙", "Minimal clearance", FitType.CLEARANCE, "Very small positive"),
    "H": HoleDeviation("H", "基准孔", "Basic hole", FitType.CLEARANCE, "Zero lower deviation"),
    "JS": HoleDeviation("JS", "对称", "Symmetrical", FitType.TRANSITION, "Symmetrical ±IT/2"),
    "J": HoleDeviation("J", "微过渡", "Slight transition", FitType.TRANSITION, "Small positive/negative"),
    "K": HoleDeviation("K", "过渡", "Transition", FitType.TRANSITION, "Small negative"),
    "M": HoleDeviation("M", "过渡", "Transition", FitType.TRANSITION, "Negative"),
    "N": HoleDeviation("N", "过渡/过盈", "Transition/interference", FitType.TRANSITION, "Negative"),
    "P": HoleDeviation("P", "轻过盈", "Light interference", FitType.INTERFERENCE, "Negative"),
    "R": HoleDeviation("R", "过盈", "Interference", FitType.INTERFERENCE, "Negative"),
    "S": HoleDeviation("S", "中过盈", "Medium interference", FitType.INTERFERENCE, "Larger negative"),
    "T": HoleDeviation("T", "中过盈", "Medium interference", FitType.INTERFERENCE, "Larger negative"),
    "U": HoleDeviation("U", "重过盈", "Heavy interference", FitType.INTERFERENCE, "Large negative"),
    "V": HoleDeviation("V", "重过盈", "Heavy interference", FitType.INTERFERENCE, "Large negative"),
    "X": HoleDeviation("X", "重过盈", "Heavy interference", FitType.INTERFERENCE, "Very large negative"),
    "Y": HoleDeviation("Y", "重过盈", "Heavy interference", FitType.INTERFERENCE, "Very large negative"),
    "Z": HoleDeviation("Z", "最大过盈", "Maximum interference", FitType.INTERFERENCE, "Maximum negative"),
    "ZA": HoleDeviation("ZA", "最大过盈", "Maximum interference", FitType.INTERFERENCE, "Maximum negative"),
    "ZB": HoleDeviation("ZB", "最大过盈", "Maximum interference", FitType.INTERFERENCE, "Maximum negative"),
    "ZC": HoleDeviation("ZC", "最大过盈", "Maximum interference", FitType.INTERFERENCE, "Maximum negative"),
}


# Fundamental deviation values for shafts (μm)
# ISO 286-2:2010 Table 2
# Format: {symbol: [(size_upper_bound, deviation_um), ...]}
# Deviation is the upper deviation for clearance fits, lower deviation for interference fits

SHAFT_FUNDAMENTAL_DEVIATIONS: Dict[str, List[Tuple[float, float]]] = {
    # g - minimal clearance shaft
    "g": [
        (3, -2), (6, -4), (10, -5), (18, -6), (30, -7),
        (50, -9), (80, -10), (120, -12), (180, -14), (250, -15),
        (315, -17), (400, -18), (500, -20),
    ],
    # h - basic shaft (upper deviation = 0)
    "h": [
        (3, 0), (6, 0), (10, 0), (18, 0), (30, 0),
        (50, 0), (80, 0), (120, 0), (180, 0), (250, 0),
        (315, 0), (400, 0), (500, 0),
    ],
    # k - transition shaft (small positive lower deviation)
    "k": [
        (3, 0), (6, 1), (10, 1), (18, 1), (30, 2),
        (50, 2), (80, 2), (120, 3), (180, 3), (250, 4),
        (315, 4), (400, 4), (500, 4),
    ],
    # m - transition shaft
    "m": [
        (3, 2), (6, 4), (10, 6), (18, 7), (30, 8),
        (50, 9), (80, 11), (120, 13), (180, 15), (250, 17),
        (315, 20), (400, 21), (500, 23),
    ],
    # n - transition/interference shaft
    "n": [
        (3, 4), (6, 8), (10, 10), (18, 12), (30, 15),
        (50, 17), (80, 20), (120, 23), (180, 27), (250, 31),
        (315, 34), (400, 37), (500, 40),
    ],
    # p - light interference shaft
    "p": [
        (3, 6), (6, 12), (10, 15), (18, 18), (30, 22),
        (50, 26), (80, 32), (120, 37), (180, 43), (250, 50),
        (315, 56), (400, 62), (500, 68),
    ],
    # r - interference shaft
    "r": [
        (3, 10), (6, 15), (10, 19), (18, 23), (30, 28),
        (50, 34), (80, 41), (120, 48), (180, 55), (250, 63),
        (315, 72), (400, 78), (500, 86),
    ],
    # s - medium interference shaft
    "s": [
        (3, 14), (6, 19), (10, 23), (18, 28), (30, 35),
        (50, 43), (80, 53), (120, 63), (180, 73), (250, 84),
        (315, 96), (400, 108), (500, 120),
    ],
    # f - small clearance shaft
    "f": [
        (3, -6), (6, -10), (10, -13), (18, -16), (30, -20),
        (50, -25), (80, -30), (120, -36), (180, -43), (250, -50),
        (315, -56), (400, -62), (500, -68),
    ],
    # e - small clearance shaft
    "e": [
        (3, -14), (6, -20), (10, -25), (18, -32), (30, -40),
        (50, -50), (80, -60), (120, -72), (180, -85), (250, -100),
        (315, -110), (400, -125), (500, -135),
    ],
    # d - clearance shaft
    "d": [
        (3, -20), (6, -30), (10, -40), (18, -50), (30, -65),
        (50, -80), (80, -100), (120, -120), (180, -145), (250, -170),
        (315, -190), (400, -210), (500, -230),
    ],
    # c - clearance shaft
    "c": [
        (3, -60), (6, -70), (10, -80), (18, -95), (30, -110),
        (50, -120), (80, -140), (120, -160), (180, -185), (250, -210),
        (315, -240), (400, -265), (500, -290),
    ],
}


# Common standard fits - Hole basis system (基孔制)
# Format: "HoleShaft": (hole_symbol, hole_grade, shaft_symbol, shaft_grade)
COMMON_FITS: Dict[str, Dict] = {
    # Clearance fits (间隙配合)
    "H11/c11": {
        "hole": ("H", 11),
        "shaft": ("c", 11),
        "type": FitType.CLEARANCE,
        "class": FitClass.LOOSE_RUNNING,
        "name_zh": "松动配合",
        "name_en": "Loose running fit",
        "application_zh": "轴与套在非常松动的情况下运转，如皮带轮与轴",
        "application_en": "Loose running fit for belt pulleys on shafts",
    },
    "H9/d9": {
        "hole": ("H", 9),
        "shaft": ("d", 9),
        "type": FitType.CLEARANCE,
        "class": FitClass.FREE_RUNNING,
        "name_zh": "自由转动配合",
        "name_en": "Free running fit",
        "application_zh": "轴承、滑轮等在较高速度下运转",
        "application_en": "Bearings and pulleys at high speeds",
    },
    "H8/f7": {
        "hole": ("H", 8),
        "shaft": ("f", 7),
        "type": FitType.CLEARANCE,
        "class": FitClass.EASY_RUNNING,
        "name_zh": "易转配合",
        "name_en": "Easy running fit",
        "application_zh": "一般用途的滑动轴承",
        "application_en": "General purpose sliding bearings",
    },
    "H7/g6": {
        "hole": ("H", 7),
        "shaft": ("g", 6),
        "type": FitType.CLEARANCE,
        "class": FitClass.NORMAL_RUNNING,
        "name_zh": "正常转动配合",
        "name_en": "Normal running fit",
        "application_zh": "精密滑动配合，可准确定位",
        "application_en": "Precision sliding fit with accurate location",
    },
    "H7/h6": {
        "hole": ("H", 7),
        "shaft": ("h", 6),
        "type": FitType.CLEARANCE,
        "class": FitClass.SLIDING,
        "name_zh": "滑动配合",
        "name_en": "Sliding fit",
        "application_zh": "需要经常拆装的定位配合",
        "application_en": "Locational fit for parts frequently dismantled",
    },
    "H8/h7": {
        "hole": ("H", 8),
        "shaft": ("h", 7),
        "type": FitType.CLEARANCE,
        "class": FitClass.CLOSE_RUNNING,
        "name_zh": "紧密滑动配合",
        "name_en": "Close running fit",
        "application_zh": "准确定位配合",
        "application_en": "Accurate location fit",
    },
    # Transition fits (过渡配合)
    "H7/js6": {
        "hole": ("H", 7),
        "shaft": ("js", 6),
        "type": FitType.TRANSITION,
        "class": FitClass.LOCATION_CLEARANCE,
        "name_zh": "定位间隙配合",
        "name_en": "Location clearance fit",
        "application_zh": "可手动拆装的定位配合",
        "application_en": "Location fit that can be dismantled by hand",
    },
    "H7/k6": {
        "hole": ("H", 7),
        "shaft": ("k", 6),
        "type": FitType.TRANSITION,
        "class": FitClass.LOCATION_TRANSITION,
        "name_zh": "定位过渡配合",
        "name_en": "Location transition fit",
        "application_zh": "需用软锤或压力机装配",
        "application_en": "Requires soft hammer or press for assembly",
    },
    "H7/m6": {
        "hole": ("H", 7),
        "shaft": ("m", 6),
        "type": FitType.TRANSITION,
        "class": FitClass.LOCATION_TRANSITION,
        "name_zh": "定位过渡配合",
        "name_en": "Location transition fit",
        "application_zh": "精确定位，用于齿轮、联轴器",
        "application_en": "Precise location for gears, couplings",
    },
    "H7/n6": {
        "hole": ("H", 7),
        "shaft": ("n", 6),
        "type": FitType.TRANSITION,
        "class": FitClass.LOCATION_INTERFERENCE,
        "name_zh": "定位过盈配合",
        "name_en": "Location interference fit",
        "application_zh": "紧固定位，用于定位销",
        "application_en": "Tight location for locating pins",
    },
    # Interference fits (过盈配合)
    "H7/p6": {
        "hole": ("H", 7),
        "shaft": ("p", 6),
        "type": FitType.INTERFERENCE,
        "class": FitClass.LIGHT_PRESS,
        "name_zh": "轻压配合",
        "name_en": "Light press fit",
        "application_zh": "薄壁零件的永久装配",
        "application_en": "Permanent assembly of thin-walled parts",
    },
    "H7/r6": {
        "hole": ("H", 7),
        "shaft": ("r", 6),
        "type": FitType.INTERFERENCE,
        "class": FitClass.MEDIUM_PRESS,
        "name_zh": "中压配合",
        "name_en": "Medium press fit",
        "application_zh": "传递载荷的永久装配",
        "application_en": "Permanent assembly for load transmission",
    },
    "H7/s6": {
        "hole": ("H", 7),
        "shaft": ("s", 6),
        "type": FitType.INTERFERENCE,
        "class": FitClass.HEAVY_PRESS,
        "name_zh": "重压配合",
        "name_en": "Heavy press fit",
        "application_zh": "传递大扭矩的压入配合",
        "application_en": "Press fit for high torque transmission",
    },
    "H7/u6": {
        "hole": ("H", 7),
        "shaft": ("u", 6),
        "type": FitType.INTERFERENCE,
        "class": FitClass.FORCE,
        "name_zh": "强力配合",
        "name_en": "Force fit",
        "application_zh": "需加热或冷却装配的过盈配合",
        "application_en": "Interference fit requiring heat/cold assembly",
    },
}


def _get_fundamental_deviation(
    symbol: str,
    nominal_size_mm: float,
) -> Optional[float]:
    """Get fundamental deviation for a shaft symbol at given size."""
    deviations = SHAFT_FUNDAMENTAL_DEVIATIONS.get(symbol.lower())
    if deviations is None:
        return None

    for size_upper, deviation in deviations:
        if nominal_size_mm <= size_upper:
            return deviation

    # Return last value for sizes beyond range
    if deviations:
        return deviations[-1][1]

    return None


@dataclass
class FitDeviations:
    """Complete deviation data for a fit."""

    fit_code: str  # e.g., "H7/g6"
    nominal_size_mm: float
    hole_upper_deviation_um: float
    hole_lower_deviation_um: float
    shaft_upper_deviation_um: float
    shaft_lower_deviation_um: float
    max_clearance_um: float  # or max interference (negative)
    min_clearance_um: float  # or min interference (negative)
    fit_type: FitType


def get_fit_deviations(
    fit_code: str,
    nominal_size_mm: float,
) -> Optional[FitDeviations]:
    """
    Calculate deviations for a standard fit.

    Args:
        fit_code: Standard fit code (e.g., "H7/g6", "H7/h6")
        nominal_size_mm: Nominal dimension in mm

    Returns:
        FitDeviations object with all deviation values, or None if invalid

    Example:
        >>> result = get_fit_deviations("H7/g6", 25)
        >>> print(f"Max clearance: {result.max_clearance_um} μm")
    """
    fit_data = COMMON_FITS.get(fit_code)
    if fit_data is None:
        return None

    hole_symbol, hole_grade = fit_data["hole"]
    shaft_symbol, shaft_grade = fit_data["shaft"]

    # Get tolerance values
    hole_tolerance = get_tolerance_value(nominal_size_mm, f"IT{hole_grade}")
    shaft_tolerance = get_tolerance_value(nominal_size_mm, f"IT{shaft_grade}")

    if hole_tolerance is None or shaft_tolerance is None:
        return None

    # For H hole (basic hole system): lower deviation = 0
    hole_lower = 0
    hole_upper = hole_tolerance

    # Get shaft fundamental deviation
    shaft_fund_dev = _get_fundamental_deviation(shaft_symbol, nominal_size_mm)
    if shaft_fund_dev is None:
        # Handle special cases
        if shaft_symbol == "js":
            # Symmetrical tolerance
            shaft_upper = shaft_tolerance / 2
            shaft_lower = -shaft_tolerance / 2
        else:
            return None
    else:
        # For clearance shafts (g, f, e, d, c, h): fundamental deviation is upper
        if shaft_symbol in ["g", "f", "e", "d", "c", "h", "a", "b"]:
            shaft_upper = shaft_fund_dev
            shaft_lower = shaft_fund_dev - shaft_tolerance
        else:
            # For interference shafts (k, m, n, p, r, s, etc.): fundamental deviation is lower
            shaft_lower = shaft_fund_dev
            shaft_upper = shaft_fund_dev + shaft_tolerance

    # Calculate clearances/interferences
    max_clearance = hole_upper - shaft_lower  # Maximum material condition
    min_clearance = hole_lower - shaft_upper  # Minimum material condition

    return FitDeviations(
        fit_code=fit_code,
        nominal_size_mm=nominal_size_mm,
        hole_upper_deviation_um=hole_upper,
        hole_lower_deviation_um=hole_lower,
        shaft_upper_deviation_um=shaft_upper,
        shaft_lower_deviation_um=shaft_lower,
        max_clearance_um=max_clearance,
        min_clearance_um=min_clearance,
        fit_type=fit_data["type"],
    )


def get_common_fits(
    fit_type: Optional[FitType] = None,
) -> Dict[str, Dict]:
    """
    Get common standard fits, optionally filtered by type.

    Args:
        fit_type: Optional filter by FitType (CLEARANCE, TRANSITION, INTERFERENCE)

    Returns:
        Dictionary of fit codes and their properties
    """
    if fit_type is None:
        return COMMON_FITS

    return {
        code: data
        for code, data in COMMON_FITS.items()
        if data["type"] == fit_type
    }


def get_fit_info(fit_code: str) -> Optional[Dict]:
    """Get detailed information about a standard fit."""
    return COMMON_FITS.get(fit_code)


def list_fit_codes_by_class(fit_class: FitClass) -> List[str]:
    """List all fit codes belonging to a specific fit class."""
    return [
        code
        for code, data in COMMON_FITS.items()
        if data.get("class") == fit_class
    ]
