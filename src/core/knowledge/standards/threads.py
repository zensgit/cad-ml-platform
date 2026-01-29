"""
Metric Thread Specifications Knowledge Base.

Provides ISO metric thread specifications according to ISO 261/262.
Includes coarse and fine pitch series with dimensions and tolerances.

Reference:
- ISO 261:1998 - ISO general purpose metric screw threads - General plan
- ISO 262:1998 - ISO general purpose metric screw threads - Selected sizes
- ISO 965-1:1998 - ISO general purpose metric screw threads - Tolerances
- GB/T 196-2003 (Chinese equivalent)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ThreadType(str, Enum):
    """Thread type classification."""

    METRIC_COARSE = "metric_coarse"  # 粗牙
    METRIC_FINE = "metric_fine"  # 细牙
    METRIC_SUPERFINE = "metric_superfine"  # 超细牙


class ThreadClass(str, Enum):
    """Thread tolerance class."""

    # External thread (bolt/screw)
    EXT_6g = "6g"  # Medium fit, general purpose
    EXT_6h = "6h"  # Medium fit, no allowance
    EXT_4g6g = "4g6g"  # Close fit

    # Internal thread (nut/hole)
    INT_6H = "6H"  # Medium fit, general purpose
    INT_5H = "5H"  # Close fit
    INT_6G = "6G"  # Medium fit, with allowance


@dataclass
class MetricThread:
    """Metric thread specification."""

    designation: str  # e.g., "M10", "M10x1"
    nominal_diameter: float  # Nominal diameter D (mm)
    pitch: float  # Thread pitch P (mm)
    thread_type: ThreadType

    # Derived dimensions (mm)
    pitch_diameter: float  # d2/D2
    minor_diameter_ext: float  # d3 (external thread)
    minor_diameter_int: float  # D1 (internal thread)

    # Tap drill recommendations
    tap_drill_size: float  # Recommended tap drill diameter (mm)
    tap_drill_depth_factor: float  # Multiply by pitch for min thread depth

    # Thread engagement
    thread_depth: float  # H1 = 0.5413 * P (fundamental triangle height)
    engagement_75: float  # 75% thread engagement depth

    # Additional info
    name_zh: str
    name_en: str
    application: str


# ISO Metric Coarse Thread Series (ISO 262)
# Format: nominal_d: (pitch, pitch_d, minor_d_ext, minor_d_int, tap_drill)
METRIC_COARSE_DATA: Dict[float, Tuple[float, float, float, float, float]] = {
    1.0: (0.25, 0.838, 0.693, 0.729, 0.75),
    1.2: (0.25, 1.038, 0.893, 0.929, 0.95),
    1.4: (0.3, 1.205, 1.032, 1.075, 1.1),
    1.6: (0.35, 1.373, 1.171, 1.221, 1.25),
    1.8: (0.35, 1.573, 1.371, 1.421, 1.45),
    2.0: (0.4, 1.740, 1.509, 1.567, 1.6),
    2.2: (0.45, 1.908, 1.648, 1.713, 1.75),
    2.5: (0.45, 2.208, 1.948, 2.013, 2.05),
    3.0: (0.5, 2.675, 2.387, 2.459, 2.5),
    3.5: (0.6, 3.110, 2.764, 2.850, 2.9),
    4.0: (0.7, 3.545, 3.141, 3.242, 3.3),
    4.5: (0.75, 4.013, 3.580, 3.688, 3.7),
    5.0: (0.8, 4.480, 4.019, 4.134, 4.2),
    6.0: (1.0, 5.350, 4.773, 4.917, 5.0),
    7.0: (1.0, 6.350, 5.773, 5.917, 6.0),
    8.0: (1.25, 7.188, 6.466, 6.647, 6.8),
    10.0: (1.5, 9.026, 8.160, 8.376, 8.5),
    12.0: (1.75, 10.863, 9.853, 10.106, 10.2),
    14.0: (2.0, 12.701, 11.546, 11.835, 12.0),
    16.0: (2.0, 14.701, 13.546, 13.835, 14.0),
    18.0: (2.5, 16.376, 14.933, 15.294, 15.5),
    20.0: (2.5, 18.376, 16.933, 17.294, 17.5),
    22.0: (2.5, 20.376, 18.933, 19.294, 19.5),
    24.0: (3.0, 22.051, 20.319, 20.752, 21.0),
    27.0: (3.0, 25.051, 23.319, 23.752, 24.0),
    30.0: (3.5, 27.727, 25.706, 26.211, 26.5),
    33.0: (3.5, 30.727, 28.706, 29.211, 29.5),
    36.0: (4.0, 33.402, 31.093, 31.670, 32.0),
    39.0: (4.0, 36.402, 34.093, 34.670, 35.0),
    42.0: (4.5, 39.077, 36.479, 37.129, 37.5),
    45.0: (4.5, 42.077, 39.479, 40.129, 40.5),
    48.0: (5.0, 44.752, 41.866, 42.587, 43.0),
    52.0: (5.0, 48.752, 45.866, 46.587, 47.0),
    56.0: (5.5, 52.428, 49.252, 50.046, 50.5),
    60.0: (5.5, 56.428, 53.252, 54.046, 54.5),
    64.0: (6.0, 60.103, 56.639, 57.505, 58.0),
    68.0: (6.0, 64.103, 60.639, 61.505, 62.0),
}

# ISO Metric Fine Thread Series (selected sizes)
# Format: (nominal_d, pitch): (pitch_d, minor_d_ext, minor_d_int, tap_drill)
METRIC_FINE_DATA: Dict[Tuple[float, float], Tuple[float, float, float, float]] = {
    # M8 fine pitches
    (8.0, 1.0): (7.350, 6.773, 6.917, 7.0),
    (8.0, 0.75): (7.513, 7.080, 7.188, 7.25),
    # M10 fine pitches
    (10.0, 1.25): (9.188, 8.466, 8.647, 8.8),
    (10.0, 1.0): (9.350, 8.773, 8.917, 9.0),
    (10.0, 0.75): (9.513, 9.080, 9.188, 9.25),
    # M12 fine pitches
    (12.0, 1.5): (11.026, 10.160, 10.376, 10.5),
    (12.0, 1.25): (11.188, 10.466, 10.647, 10.8),
    (12.0, 1.0): (11.350, 10.773, 10.917, 11.0),
    # M14 fine pitches
    (14.0, 1.5): (13.026, 12.160, 12.376, 12.5),
    (14.0, 1.25): (13.188, 12.466, 12.647, 12.8),
    (14.0, 1.0): (13.350, 12.773, 12.917, 13.0),
    # M16 fine pitches
    (16.0, 1.5): (15.026, 14.160, 14.376, 14.5),
    (16.0, 1.0): (15.350, 14.773, 14.917, 15.0),
    # M18 fine pitches
    (18.0, 2.0): (16.701, 15.546, 15.835, 16.0),
    (18.0, 1.5): (17.026, 16.160, 16.376, 16.5),
    (18.0, 1.0): (17.350, 16.773, 16.917, 17.0),
    # M20 fine pitches
    (20.0, 2.0): (18.701, 17.546, 17.835, 18.0),
    (20.0, 1.5): (19.026, 18.160, 18.376, 18.5),
    (20.0, 1.0): (19.350, 18.773, 18.917, 19.0),
    # M22 fine pitches
    (22.0, 2.0): (20.701, 19.546, 19.835, 20.0),
    (22.0, 1.5): (21.026, 20.160, 20.376, 20.5),
    # M24 fine pitches
    (24.0, 2.0): (22.701, 21.546, 21.835, 22.0),
    (24.0, 1.5): (23.026, 22.160, 22.376, 22.5),
    # M27 fine pitches
    (27.0, 2.0): (25.701, 24.546, 24.835, 25.0),
    (27.0, 1.5): (26.026, 25.160, 25.376, 25.5),
    # M30 fine pitches
    (30.0, 3.0): (28.051, 26.319, 26.752, 27.0),
    (30.0, 2.0): (28.701, 27.546, 27.835, 28.0),
    (30.0, 1.5): (29.026, 28.160, 28.376, 28.5),
    # M33 fine pitches
    (33.0, 3.0): (31.051, 29.319, 29.752, 30.0),
    (33.0, 2.0): (31.701, 30.546, 30.835, 31.0),
    # M36 fine pitches
    (36.0, 3.0): (34.051, 32.319, 32.752, 33.0),
    (36.0, 2.0): (34.701, 33.546, 33.835, 34.0),
    (36.0, 1.5): (35.026, 34.160, 34.376, 34.5),
    # M39 fine pitches
    (39.0, 3.0): (37.051, 35.319, 35.752, 36.0),
    (39.0, 2.0): (37.701, 36.546, 36.835, 37.0),
    # M42 fine pitches
    (42.0, 4.0): (39.402, 37.093, 37.670, 38.0),
    (42.0, 3.0): (40.051, 38.319, 38.752, 39.0),
    (42.0, 2.0): (40.701, 39.546, 39.835, 40.0),
    # M45 fine pitches
    (45.0, 4.0): (42.402, 40.093, 40.670, 41.0),
    (45.0, 3.0): (43.051, 41.319, 41.752, 42.0),
    # M48 fine pitches
    (48.0, 4.0): (45.402, 43.093, 43.670, 44.0),
    (48.0, 3.0): (46.051, 44.319, 44.752, 45.0),
    (48.0, 2.0): (46.701, 45.546, 45.835, 46.0),
}


def _create_thread_spec(
    nominal_d: float,
    pitch: float,
    pitch_d: float,
    minor_d_ext: float,
    minor_d_int: float,
    tap_drill: float,
    thread_type: ThreadType,
) -> MetricThread:
    """Create a MetricThread specification from raw data."""
    # Calculate thread depth (fundamental triangle height)
    thread_depth = 0.5413 * pitch
    engagement_75 = 0.75 * thread_depth * 2  # 75% engagement

    # Designation
    if thread_type == ThreadType.METRIC_COARSE:
        designation = f"M{nominal_d:.0f}" if nominal_d == int(nominal_d) else f"M{nominal_d}"
    else:
        designation = f"M{nominal_d:.0f}x{pitch}" if nominal_d == int(nominal_d) else f"M{nominal_d}x{pitch}"

    # Names
    if thread_type == ThreadType.METRIC_COARSE:
        name_zh = f"公制粗牙螺纹 {designation}"
        name_en = f"Metric coarse thread {designation}"
    else:
        name_zh = f"公制细牙螺纹 {designation}"
        name_en = f"Metric fine thread {designation}"

    # Application guidance
    if thread_type == ThreadType.METRIC_COARSE:
        application = "General purpose fastening, structural connections"
    else:
        application = "Fine adjustment, thin-walled parts, vibration resistance"

    return MetricThread(
        designation=designation,
        nominal_diameter=nominal_d,
        pitch=pitch,
        thread_type=thread_type,
        pitch_diameter=pitch_d,
        minor_diameter_ext=minor_d_ext,
        minor_diameter_int=minor_d_int,
        tap_drill_size=tap_drill,
        tap_drill_depth_factor=1.5,  # Min thread depth = 1.5 * pitch
        thread_depth=thread_depth,
        engagement_75=engagement_75,
        name_zh=name_zh,
        name_en=name_en,
        application=application,
    )


# Build thread database
METRIC_THREADS: Dict[str, MetricThread] = {}

# Add coarse threads
for nominal_d, (pitch, pitch_d, minor_ext, minor_int, tap_drill) in METRIC_COARSE_DATA.items():
    spec = _create_thread_spec(
        nominal_d, pitch, pitch_d, minor_ext, minor_int, tap_drill,
        ThreadType.METRIC_COARSE
    )
    METRIC_THREADS[spec.designation] = spec

# Add fine threads
for (nominal_d, pitch), (pitch_d, minor_ext, minor_int, tap_drill) in METRIC_FINE_DATA.items():
    spec = _create_thread_spec(
        nominal_d, pitch, pitch_d, minor_ext, minor_int, tap_drill,
        ThreadType.METRIC_FINE
    )
    METRIC_THREADS[spec.designation] = spec


def get_thread_spec(designation: str) -> Optional[MetricThread]:
    """
    Get thread specification by designation.

    Args:
        designation: Thread designation (e.g., "M10", "M10x1.25", "M10x1")

    Returns:
        MetricThread specification or None if not found

    Example:
        >>> spec = get_thread_spec("M10")
        >>> print(f"Pitch: {spec.pitch}mm, Tap drill: {spec.tap_drill_size}mm")
    """
    # Normalize designation
    designation = designation.upper().replace(" ", "")

    # Try exact match
    if designation in METRIC_THREADS:
        return METRIC_THREADS[designation]

    # Try case-insensitive match
    for key, spec in METRIC_THREADS.items():
        if key.upper() == designation:
            return spec

    # Try matching with .0 suffix for fine threads (M10x1 -> M10x1.0)
    if "X" in designation and "." not in designation.split("X")[1]:
        designation_with_decimal = designation + ".0"
        for key, spec in METRIC_THREADS.items():
            if key.upper() == designation_with_decimal:
                return spec

    return None


def get_thread_series(
    thread_type: ThreadType = ThreadType.METRIC_COARSE,
    min_diameter: float = 0,
    max_diameter: float = 100,
) -> List[MetricThread]:
    """
    Get a series of threads filtered by type and diameter range.

    Args:
        thread_type: Filter by thread type
        min_diameter: Minimum nominal diameter (mm)
        max_diameter: Maximum nominal diameter (mm)

    Returns:
        List of matching MetricThread specifications
    """
    results = []
    for spec in METRIC_THREADS.values():
        if spec.thread_type != thread_type:
            continue
        if spec.nominal_diameter < min_diameter:
            continue
        if spec.nominal_diameter > max_diameter:
            continue
        results.append(spec)

    return sorted(results, key=lambda x: (x.nominal_diameter, x.pitch))


def list_metric_threads(
    nominal_diameter: Optional[float] = None,
) -> List[MetricThread]:
    """
    List all metric threads, optionally filtered by nominal diameter.

    Args:
        nominal_diameter: Filter to specific diameter (shows all pitches)

    Returns:
        List of MetricThread specifications
    """
    if nominal_diameter is None:
        return sorted(METRIC_THREADS.values(), key=lambda x: (x.nominal_diameter, -x.pitch))

    results = []
    for spec in METRIC_THREADS.values():
        if spec.nominal_diameter == nominal_diameter:
            results.append(spec)

    return sorted(results, key=lambda x: -x.pitch)  # Coarse first


def get_tap_drill_size(designation: str) -> Optional[float]:
    """
    Get recommended tap drill size for a thread.

    Args:
        designation: Thread designation (e.g., "M10", "M10x1.25")

    Returns:
        Tap drill diameter in mm, or None if not found

    Example:
        >>> get_tap_drill_size("M10")
        8.5
    """
    spec = get_thread_spec(designation)
    if spec is None:
        return None
    return spec.tap_drill_size


def calculate_thread_engagement(
    designation: str,
    material_strength: str = "medium",
) -> Optional[Dict[str, float]]:
    """
    Calculate recommended thread engagement lengths.

    Args:
        designation: Thread designation
        material_strength: "soft", "medium", or "hard"

    Returns:
        Dictionary with min/recommended engagement lengths
    """
    spec = get_thread_spec(designation)
    if spec is None:
        return None

    # Engagement factors based on material
    factors = {
        "soft": {"min": 1.5, "recommended": 2.0},  # Aluminum, plastic
        "medium": {"min": 1.0, "recommended": 1.5},  # Steel
        "hard": {"min": 0.8, "recommended": 1.0},  # Hardened steel
    }

    factor = factors.get(material_strength, factors["medium"])

    return {
        "min_engagement_mm": factor["min"] * spec.nominal_diameter,
        "recommended_engagement_mm": factor["recommended"] * spec.nominal_diameter,
        "min_tapped_depth_mm": factor["recommended"] * spec.nominal_diameter + spec.pitch * 2,
    }


def get_clearance_hole_size(
    designation: str,
    fit: str = "medium",
) -> Optional[float]:
    """
    Get clearance hole diameter for a bolt/screw.

    Args:
        designation: Thread designation
        fit: "close", "medium", or "free"

    Returns:
        Clearance hole diameter in mm
    """
    spec = get_thread_spec(designation)
    if spec is None:
        return None

    d = spec.nominal_diameter

    # Clearance hole sizes per ISO 273
    clearance_table = {
        "close": {
            1.6: 1.7, 2: 2.2, 2.5: 2.7, 3: 3.2, 4: 4.3, 5: 5.3,
            6: 6.4, 8: 8.4, 10: 10.5, 12: 13, 14: 15, 16: 17,
            18: 19, 20: 21, 22: 23, 24: 25, 27: 28, 30: 31,
        },
        "medium": {
            1.6: 1.8, 2: 2.4, 2.5: 2.9, 3: 3.4, 4: 4.5, 5: 5.5,
            6: 6.6, 8: 9, 10: 11, 12: 13.5, 14: 15.5, 16: 17.5,
            18: 20, 20: 22, 22: 24, 24: 26, 27: 30, 30: 33,
        },
        "free": {
            1.6: 2, 2: 2.6, 2.5: 3.1, 3: 3.6, 4: 4.8, 5: 5.8,
            6: 7, 8: 10, 10: 12, 12: 14.5, 14: 16.5, 16: 18.5,
            18: 21, 20: 24, 22: 26, 24: 28, 27: 32, 30: 35,
        },
    }

    fit_table = clearance_table.get(fit, clearance_table["medium"])

    # Find nearest standard size
    if d in fit_table:
        return fit_table[d]

    # Interpolate for non-standard sizes
    for std_d, hole in sorted(fit_table.items()):
        if std_d >= d:
            return hole

    return None
