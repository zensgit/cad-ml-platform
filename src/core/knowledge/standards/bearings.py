"""
Rolling Bearing Specifications Knowledge Base.

Provides ISO standard bearing dimensions and specifications according to ISO 15.
Includes deep groove ball bearings, angular contact bearings, and cylindrical roller bearings.

Reference:
- ISO 15:2017 - Rolling bearings - Radial bearings - Boundary dimensions
- ISO 492:2014 - Rolling bearings - Radial bearings - Geometrical product specifications
- GB/T 276-2013 (Chinese equivalent)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class BearingType(str, Enum):
    """Bearing type classification."""

    DEEP_GROOVE_BALL = "deep_groove_ball"  # 深沟球轴承
    ANGULAR_CONTACT_BALL = "angular_contact_ball"  # 角接触球轴承
    SELF_ALIGNING_BALL = "self_aligning_ball"  # 调心球轴承
    CYLINDRICAL_ROLLER = "cylindrical_roller"  # 圆柱滚子轴承
    TAPERED_ROLLER = "tapered_roller"  # 圆锥滚子轴承
    SPHERICAL_ROLLER = "spherical_roller"  # 调心滚子轴承
    THRUST_BALL = "thrust_ball"  # 推力球轴承
    NEEDLE_ROLLER = "needle_roller"  # 滚针轴承


class BearingSeries(str, Enum):
    """Bearing dimension series."""

    # Deep groove ball bearings
    SERIES_60 = "60"  # Extra light series
    SERIES_62 = "62"  # Light series
    SERIES_63 = "63"  # Medium series
    SERIES_64 = "64"  # Heavy series

    # Angular contact ball bearings
    SERIES_72 = "72"  # Light series
    SERIES_73 = "73"  # Medium series

    # Cylindrical roller bearings
    SERIES_NU10 = "NU10"  # Extra light series
    SERIES_NU2 = "NU2"  # Light series
    SERIES_NU3 = "NU3"  # Medium series
    SERIES_NU4 = "NU4"  # Heavy series


@dataclass
class BearingSpec:
    """Bearing specification."""

    designation: str  # e.g., "6205", "6205-2RS"
    bearing_type: BearingType
    series: BearingSeries

    # Basic dimensions (mm)
    bore_d: float  # Inner diameter d
    outer_d: float  # Outer diameter D
    width_b: float  # Width B

    # Load ratings (kN)
    dynamic_load_c: float  # Basic dynamic load rating C
    static_load_c0: float  # Basic static load rating C0

    # Speed ratings (rpm)
    limiting_speed_grease: int  # Limiting speed with grease
    limiting_speed_oil: int  # Limiting speed with oil

    # Weight (kg)
    weight: float

    # Additional info
    name_zh: str
    name_en: str


# Deep Groove Ball Bearings - 60 Series (Extra Light)
# Format: bore_code: (d, D, B, C_kN, C0_kN, n_grease, n_oil, weight_kg)
SERIES_60_DATA: Dict[str, tuple] = {
    "6000": (10, 26, 8, 4.62, 1.96, 30000, 36000, 0.024),
    "6001": (12, 28, 8, 5.07, 2.24, 28000, 34000, 0.027),
    "6002": (15, 32, 9, 5.59, 2.55, 26000, 32000, 0.035),
    "6003": (17, 35, 10, 6.05, 2.85, 24000, 30000, 0.042),
    "6004": (20, 42, 12, 9.36, 4.5, 20000, 26000, 0.070),
    "6005": (25, 47, 12, 10.1, 5.0, 18000, 22000, 0.078),
    "6006": (30, 55, 13, 13.3, 6.8, 15000, 19000, 0.110),
    "6007": (35, 62, 14, 15.9, 8.3, 13000, 17000, 0.150),
    "6008": (40, 68, 15, 17.8, 9.5, 12000, 15000, 0.180),
    "6009": (45, 75, 16, 21.2, 11.4, 11000, 14000, 0.230),
    "6010": (50, 80, 16, 22.0, 12.2, 10000, 13000, 0.250),
    "6011": (55, 90, 18, 28.3, 15.6, 9000, 12000, 0.350),
    "6012": (60, 95, 18, 29.6, 16.5, 8500, 11000, 0.380),
    "6013": (65, 100, 18, 30.7, 17.4, 8000, 10000, 0.400),
    "6014": (70, 110, 20, 37.1, 21.2, 7500, 9500, 0.520),
    "6015": (75, 115, 20, 38.0, 22.0, 7000, 9000, 0.550),
    "6016": (80, 125, 22, 44.9, 26.0, 6500, 8500, 0.710),
    "6017": (85, 130, 22, 46.2, 27.0, 6000, 8000, 0.750),
    "6018": (90, 140, 24, 53.0, 31.0, 5600, 7500, 0.920),
    "6019": (95, 145, 24, 54.0, 32.0, 5300, 7000, 0.960),
    "6020": (100, 150, 24, 55.3, 33.5, 5000, 6700, 1.000),
}

# Deep Groove Ball Bearings - 62 Series (Light)
SERIES_62_DATA: Dict[str, tuple] = {
    "6200": (10, 30, 9, 5.07, 2.24, 28000, 34000, 0.034),
    "6201": (12, 32, 10, 6.89, 3.05, 26000, 32000, 0.044),
    "6202": (15, 35, 11, 7.8, 3.55, 24000, 30000, 0.053),
    "6203": (17, 40, 12, 9.56, 4.5, 20000, 26000, 0.075),
    "6204": (20, 47, 14, 12.7, 6.2, 17000, 22000, 0.110),
    "6205": (25, 52, 15, 14.0, 6.95, 15000, 19000, 0.130),
    "6206": (30, 62, 16, 19.5, 10.0, 13000, 17000, 0.200),
    "6207": (35, 72, 17, 25.5, 13.2, 11000, 14000, 0.290),
    "6208": (40, 80, 18, 29.1, 15.3, 10000, 13000, 0.370),
    "6209": (45, 85, 19, 32.5, 17.4, 9000, 12000, 0.420),
    "6210": (50, 90, 20, 35.1, 19.0, 8500, 11000, 0.470),
    "6211": (55, 100, 21, 43.6, 24.0, 7500, 10000, 0.600),
    "6212": (60, 110, 22, 52.0, 29.0, 7000, 9000, 0.780),
    "6213": (65, 120, 23, 57.2, 32.5, 6300, 8500, 0.960),
    "6214": (70, 125, 24, 61.8, 35.5, 6000, 8000, 1.040),
    "6215": (75, 130, 25, 66.3, 38.0, 5600, 7500, 1.120),
    "6216": (80, 140, 26, 72.8, 42.5, 5300, 7000, 1.350),
    "6217": (85, 150, 28, 83.2, 49.0, 5000, 6700, 1.650),
    "6218": (90, 160, 30, 95.6, 56.0, 4800, 6300, 2.000),
    "6219": (95, 170, 32, 108, 64.0, 4500, 6000, 2.400),
    "6220": (100, 180, 34, 124, 73.5, 4300, 5600, 2.850),
}

# Deep Groove Ball Bearings - 63 Series (Medium)
SERIES_63_DATA: Dict[str, tuple] = {
    "6300": (10, 35, 11, 8.06, 3.4, 22000, 28000, 0.055),
    "6301": (12, 37, 12, 9.75, 4.15, 20000, 26000, 0.068),
    "6302": (15, 42, 13, 11.4, 5.0, 18000, 24000, 0.090),
    "6303": (17, 47, 14, 13.5, 6.1, 16000, 20000, 0.120),
    "6304": (20, 52, 15, 15.9, 7.35, 14000, 18000, 0.150),
    "6305": (25, 62, 17, 22.5, 10.8, 12000, 15000, 0.240),
    "6306": (30, 72, 19, 28.1, 13.7, 10000, 13000, 0.350),
    "6307": (35, 80, 21, 33.2, 16.6, 9000, 12000, 0.450),
    "6308": (40, 90, 23, 41.0, 20.4, 8000, 10000, 0.600),
    "6309": (45, 100, 25, 52.7, 27.0, 7000, 9000, 0.800),
    "6310": (50, 110, 27, 61.8, 32.0, 6300, 8500, 1.000),
    "6311": (55, 120, 29, 71.5, 38.0, 6000, 8000, 1.250),
    "6312": (60, 130, 31, 81.9, 44.0, 5600, 7500, 1.500),
    "6313": (65, 140, 33, 92.3, 50.0, 5000, 6700, 1.800),
    "6314": (70, 150, 35, 104, 57.0, 4800, 6300, 2.150),
    "6315": (75, 160, 37, 112, 62.0, 4500, 6000, 2.550),
    "6316": (80, 170, 39, 124, 69.5, 4300, 5600, 3.000),
    "6317": (85, 180, 41, 133, 76.5, 4000, 5300, 3.500),
    "6318": (90, 190, 43, 143, 83.0, 3800, 5000, 4.050),
    "6319": (95, 200, 45, 153, 90.0, 3600, 4800, 4.650),
    "6320": (100, 215, 47, 174, 104, 3400, 4500, 5.600),
}


def _create_bearing_spec(
    designation: str,
    series: BearingSeries,
    data: tuple,
) -> BearingSpec:
    """Create a BearingSpec from raw data."""
    d, D, B, C, C0, n_grease, n_oil, weight = data

    # Determine bearing type from series
    if series.value.startswith("6"):
        bearing_type = BearingType.DEEP_GROOVE_BALL
    elif series.value.startswith("7"):
        bearing_type = BearingType.ANGULAR_CONTACT_BALL
    elif series.value.startswith("NU"):
        bearing_type = BearingType.CYLINDRICAL_ROLLER
    else:
        bearing_type = BearingType.DEEP_GROOVE_BALL

    # Names
    type_names = {
        BearingType.DEEP_GROOVE_BALL: ("深沟球轴承", "Deep groove ball bearing"),
        BearingType.ANGULAR_CONTACT_BALL: ("角接触球轴承", "Angular contact ball bearing"),
        BearingType.CYLINDRICAL_ROLLER: ("圆柱滚子轴承", "Cylindrical roller bearing"),
    }
    name_zh, name_en = type_names.get(bearing_type, ("轴承", "Bearing"))

    return BearingSpec(
        designation=designation,
        bearing_type=bearing_type,
        series=series,
        bore_d=d,
        outer_d=D,
        width_b=B,
        dynamic_load_c=C,
        static_load_c0=C0,
        limiting_speed_grease=n_grease,
        limiting_speed_oil=n_oil,
        weight=weight,
        name_zh=f"{name_zh} {designation}",
        name_en=f"{name_en} {designation}",
    )


# Build bearing database
BEARING_DATABASE: Dict[str, BearingSpec] = {}

# Add 60 series
for desig, data in SERIES_60_DATA.items():
    spec = _create_bearing_spec(desig, BearingSeries.SERIES_60, data)
    BEARING_DATABASE[desig] = spec

# Add 62 series
for desig, data in SERIES_62_DATA.items():
    spec = _create_bearing_spec(desig, BearingSeries.SERIES_62, data)
    BEARING_DATABASE[desig] = spec

# Add 63 series
for desig, data in SERIES_63_DATA.items():
    spec = _create_bearing_spec(desig, BearingSeries.SERIES_63, data)
    BEARING_DATABASE[desig] = spec


def get_bearing_spec(designation: str) -> Optional[BearingSpec]:
    """
    Get bearing specification by designation.

    Args:
        designation: Bearing designation (e.g., "6205", "6205-2RS")

    Returns:
        BearingSpec or None if not found

    Example:
        >>> spec = get_bearing_spec("6205")
        >>> print(f"Bore: {spec.bore_d}mm, OD: {spec.outer_d}mm")
    """
    # Remove suffix like -2RS, -ZZ, etc.
    base_desig = designation.split("-")[0].split("/")[0].upper()

    if base_desig in BEARING_DATABASE:
        return BEARING_DATABASE[base_desig]

    return None


def get_bearing_by_bore(
    bore_diameter: float,
    bearing_type: Optional[BearingType] = None,
    series: Optional[BearingSeries] = None,
) -> List[BearingSpec]:
    """
    Find bearings by bore diameter.

    Args:
        bore_diameter: Inner diameter d (mm)
        bearing_type: Filter by bearing type
        series: Filter by dimension series

    Returns:
        List of matching BearingSpec objects
    """
    results = []

    for spec in BEARING_DATABASE.values():
        if spec.bore_d != bore_diameter:
            continue
        if bearing_type and spec.bearing_type != bearing_type:
            continue
        if series and spec.series != series:
            continue
        results.append(spec)

    return sorted(results, key=lambda x: (x.series.value, x.outer_d))


def list_bearings(
    series: Optional[BearingSeries] = None,
    min_bore: float = 0,
    max_bore: float = 1000,
) -> List[BearingSpec]:
    """
    List bearings filtered by series and bore range.

    Args:
        series: Filter by dimension series
        min_bore: Minimum bore diameter (mm)
        max_bore: Maximum bore diameter (mm)

    Returns:
        List of matching BearingSpec objects
    """
    results = []

    for spec in BEARING_DATABASE.values():
        if series and spec.series != series:
            continue
        if spec.bore_d < min_bore or spec.bore_d > max_bore:
            continue
        results.append(spec)

    return sorted(results, key=lambda x: (x.series.value, x.bore_d))


def calculate_bearing_life(
    designation: str,
    radial_load_kn: float,
    axial_load_kn: float = 0,
    speed_rpm: int = 1000,
) -> Optional[Dict[str, float]]:
    """
    Calculate basic bearing life (L10).

    Args:
        designation: Bearing designation
        radial_load_kn: Radial load Fr (kN)
        axial_load_kn: Axial load Fa (kN)
        speed_rpm: Rotational speed (rpm)

    Returns:
        Dictionary with L10 life in hours and revolutions
    """
    spec = get_bearing_spec(designation)
    if spec is None:
        return None

    # Equivalent dynamic load (simplified for deep groove ball bearings)
    # P = Fr when Fa/Fr < e (typically 0.25)
    if axial_load_kn / radial_load_kn < 0.25 if radial_load_kn > 0 else True:
        P = radial_load_kn
    else:
        # Approximate X=0.56, Y=1.4 for deep groove ball bearings
        P = 0.56 * radial_load_kn + 1.4 * axial_load_kn

    if P <= 0:
        return None

    # Basic rating life L10 (million revolutions)
    # Ball bearings: p = 3
    L10_rev = (spec.dynamic_load_c / P) ** 3

    # Life in hours
    L10_hours = (L10_rev * 1e6) / (60 * speed_rpm)

    return {
        "L10_million_rev": L10_rev,
        "L10_hours": L10_hours,
        "equivalent_load_kn": P,
    }


def suggest_bearing_for_shaft(
    shaft_diameter: float,
    load_type: str = "medium",
    speed: str = "medium",
) -> List[str]:
    """
    Suggest suitable bearings for a shaft diameter.

    Args:
        shaft_diameter: Shaft diameter (mm)
        load_type: "light", "medium", or "heavy"
        speed: "low", "medium", or "high"

    Returns:
        List of suggested bearing designations
    """
    # Find bearings matching the shaft diameter
    bearings = get_bearing_by_bore(shaft_diameter)

    if not bearings:
        return []

    # Sort by suitability
    series_preference = {
        "light": [BearingSeries.SERIES_60, BearingSeries.SERIES_62, BearingSeries.SERIES_63],
        "medium": [BearingSeries.SERIES_62, BearingSeries.SERIES_63, BearingSeries.SERIES_60],
        "heavy": [BearingSeries.SERIES_63, BearingSeries.SERIES_62, BearingSeries.SERIES_60],
    }

    preference = series_preference.get(load_type, series_preference["medium"])

    def sort_key(b: BearingSpec) -> int:
        try:
            return preference.index(b.series)
        except ValueError:
            return 99

    bearings.sort(key=sort_key)

    return [b.designation for b in bearings[:3]]
