"""
O-Ring and Seal Specifications Knowledge Base.

Provides ISO standard O-ring dimensions and specifications according to ISO 3601.
Includes metric O-ring series and material recommendations.

Reference:
- ISO 3601-1:2012 - Fluid power systems - O-rings - Inside diameters, cross-sections
- ISO 3601-2:2016 - Fluid power systems - O-rings - Housing dimensions
- GB/T 3452.1-2005 (Chinese equivalent)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class SealType(str, Enum):
    """Seal type classification."""

    ORING = "oring"  # O形圈
    ORING_STATIC = "oring_static"  # 静密封O形圈
    ORING_DYNAMIC = "oring_dynamic"  # 动密封O形圈
    BACKUP_RING = "backup_ring"  # 挡圈
    QUAD_RING = "quad_ring"  # X形圈
    V_RING = "v_ring"  # V形圈


class ORingMaterial(str, Enum):
    """O-ring material classification."""

    NBR = "NBR"  # 丁腈橡胶 (-30 to +100°C)
    FKM = "FKM"  # 氟橡胶 (-20 to +200°C)
    EPDM = "EPDM"  # 三元乙丙橡胶 (-50 to +150°C)
    SILICONE = "silicone"  # 硅橡胶 (-60 to +200°C)
    PTFE = "PTFE"  # 聚四氟乙烯 (-200 to +260°C)
    NEOPRENE = "neoprene"  # 氯丁橡胶 (-40 to +100°C)


@dataclass
class ORingSpec:
    """O-ring specification."""

    designation: str  # e.g., "10x2", "OR-10"
    inner_diameter: float  # ID (mm)
    cross_section: float  # CS diameter (mm)

    # Tolerances
    id_tolerance_plus: float
    id_tolerance_minus: float
    cs_tolerance_plus: float
    cs_tolerance_minus: float

    # Housing dimensions (for reference)
    groove_width_static: float  # Static seal groove width
    groove_width_dynamic: float  # Dynamic seal groove width
    groove_depth_static: float  # Static seal groove depth
    groove_depth_dynamic: float  # Dynamic seal groove depth

    # Additional info
    standard: str  # ISO, JIS, AS568, etc.
    name_zh: str
    name_en: str


# ISO 3601-1 Standard O-ring Cross-sections
STANDARD_CROSS_SECTIONS = [1.0, 1.5, 1.78, 2.0, 2.62, 3.0, 3.53, 4.0, 5.0, 5.33, 6.0, 7.0, 8.0]

# ISO 3601-1 O-Ring Data
# Format: (ID, CS): (id_tol+, id_tol-, cs_tol+, cs_tol-)
# Tolerances based on ISO 3601-1 Grade N
ORING_TOLERANCES: Dict[str, Dict[str, float]] = {
    # Cross-section 1.78mm tolerances by ID range
    "1.78": {
        "id_small": (0.15, 0.15),  # ID < 10
        "id_medium": (0.20, 0.20),  # 10 <= ID < 30
        "id_large": (0.25, 0.25),  # ID >= 30
        "cs": (0.08, 0.08),
    },
    "2.62": {
        "id_small": (0.20, 0.20),
        "id_medium": (0.25, 0.25),
        "id_large": (0.30, 0.30),
        "cs": (0.09, 0.09),
    },
    "3.53": {
        "id_small": (0.25, 0.25),
        "id_medium": (0.30, 0.30),
        "id_large": (0.35, 0.35),
        "cs": (0.10, 0.10),
    },
    "5.33": {
        "id_small": (0.30, 0.30),
        "id_medium": (0.35, 0.35),
        "id_large": (0.40, 0.40),
        "cs": (0.13, 0.13),
    },
}

# Common metric O-ring sizes (ID x CS in mm)
# Format: "IDxCS": (groove_width_static, groove_width_dynamic, groove_depth_static, groove_depth_dynamic)
METRIC_ORING_DATA: Dict[str, tuple] = {
    # CS = 1.5mm series
    "3x1.5": (1.9, 2.1, 1.2, 1.25),
    "4x1.5": (1.9, 2.1, 1.2, 1.25),
    "5x1.5": (1.9, 2.1, 1.2, 1.25),
    "6x1.5": (1.9, 2.1, 1.2, 1.25),
    "7x1.5": (1.9, 2.1, 1.2, 1.25),
    "8x1.5": (1.9, 2.1, 1.2, 1.25),
    "9x1.5": (1.9, 2.1, 1.2, 1.25),
    "10x1.5": (1.9, 2.1, 1.2, 1.25),
    "12x1.5": (1.9, 2.1, 1.2, 1.25),
    "14x1.5": (1.9, 2.1, 1.2, 1.25),
    "15x1.5": (1.9, 2.1, 1.2, 1.25),
    "16x1.5": (1.9, 2.1, 1.2, 1.25),
    "18x1.5": (1.9, 2.1, 1.2, 1.25),
    "20x1.5": (1.9, 2.1, 1.2, 1.25),

    # CS = 2.0mm series
    "5x2": (2.5, 2.8, 1.6, 1.7),
    "6x2": (2.5, 2.8, 1.6, 1.7),
    "7x2": (2.5, 2.8, 1.6, 1.7),
    "8x2": (2.5, 2.8, 1.6, 1.7),
    "9x2": (2.5, 2.8, 1.6, 1.7),
    "10x2": (2.5, 2.8, 1.6, 1.7),
    "11x2": (2.5, 2.8, 1.6, 1.7),
    "12x2": (2.5, 2.8, 1.6, 1.7),
    "14x2": (2.5, 2.8, 1.6, 1.7),
    "15x2": (2.5, 2.8, 1.6, 1.7),
    "16x2": (2.5, 2.8, 1.6, 1.7),
    "18x2": (2.5, 2.8, 1.6, 1.7),
    "20x2": (2.5, 2.8, 1.6, 1.7),
    "22x2": (2.5, 2.8, 1.6, 1.7),
    "24x2": (2.5, 2.8, 1.6, 1.7),
    "25x2": (2.5, 2.8, 1.6, 1.7),
    "26x2": (2.5, 2.8, 1.6, 1.7),
    "28x2": (2.5, 2.8, 1.6, 1.7),
    "30x2": (2.5, 2.8, 1.6, 1.7),
    "32x2": (2.5, 2.8, 1.6, 1.7),
    "35x2": (2.5, 2.8, 1.6, 1.7),
    "38x2": (2.5, 2.8, 1.6, 1.7),
    "40x2": (2.5, 2.8, 1.6, 1.7),

    # CS = 2.5mm series
    "10x2.5": (3.2, 3.5, 2.0, 2.1),
    "12x2.5": (3.2, 3.5, 2.0, 2.1),
    "14x2.5": (3.2, 3.5, 2.0, 2.1),
    "15x2.5": (3.2, 3.5, 2.0, 2.1),
    "16x2.5": (3.2, 3.5, 2.0, 2.1),
    "18x2.5": (3.2, 3.5, 2.0, 2.1),
    "20x2.5": (3.2, 3.5, 2.0, 2.1),
    "22x2.5": (3.2, 3.5, 2.0, 2.1),
    "24x2.5": (3.2, 3.5, 2.0, 2.1),
    "25x2.5": (3.2, 3.5, 2.0, 2.1),
    "28x2.5": (3.2, 3.5, 2.0, 2.1),
    "30x2.5": (3.2, 3.5, 2.0, 2.1),
    "32x2.5": (3.2, 3.5, 2.0, 2.1),
    "35x2.5": (3.2, 3.5, 2.0, 2.1),
    "38x2.5": (3.2, 3.5, 2.0, 2.1),
    "40x2.5": (3.2, 3.5, 2.0, 2.1),
    "45x2.5": (3.2, 3.5, 2.0, 2.1),
    "50x2.5": (3.2, 3.5, 2.0, 2.1),

    # CS = 3.0mm series
    "15x3": (3.8, 4.2, 2.4, 2.55),
    "16x3": (3.8, 4.2, 2.4, 2.55),
    "18x3": (3.8, 4.2, 2.4, 2.55),
    "20x3": (3.8, 4.2, 2.4, 2.55),
    "22x3": (3.8, 4.2, 2.4, 2.55),
    "24x3": (3.8, 4.2, 2.4, 2.55),
    "25x3": (3.8, 4.2, 2.4, 2.55),
    "26x3": (3.8, 4.2, 2.4, 2.55),
    "28x3": (3.8, 4.2, 2.4, 2.55),
    "30x3": (3.8, 4.2, 2.4, 2.55),
    "32x3": (3.8, 4.2, 2.4, 2.55),
    "34x3": (3.8, 4.2, 2.4, 2.55),
    "35x3": (3.8, 4.2, 2.4, 2.55),
    "36x3": (3.8, 4.2, 2.4, 2.55),
    "38x3": (3.8, 4.2, 2.4, 2.55),
    "40x3": (3.8, 4.2, 2.4, 2.55),
    "42x3": (3.8, 4.2, 2.4, 2.55),
    "45x3": (3.8, 4.2, 2.4, 2.55),
    "48x3": (3.8, 4.2, 2.4, 2.55),
    "50x3": (3.8, 4.2, 2.4, 2.55),
    "55x3": (3.8, 4.2, 2.4, 2.55),
    "60x3": (3.8, 4.2, 2.4, 2.55),
    "65x3": (3.8, 4.2, 2.4, 2.55),
    "70x3": (3.8, 4.2, 2.4, 2.55),

    # CS = 4.0mm series
    "20x4": (5.0, 5.5, 3.2, 3.4),
    "22x4": (5.0, 5.5, 3.2, 3.4),
    "25x4": (5.0, 5.5, 3.2, 3.4),
    "28x4": (5.0, 5.5, 3.2, 3.4),
    "30x4": (5.0, 5.5, 3.2, 3.4),
    "32x4": (5.0, 5.5, 3.2, 3.4),
    "35x4": (5.0, 5.5, 3.2, 3.4),
    "38x4": (5.0, 5.5, 3.2, 3.4),
    "40x4": (5.0, 5.5, 3.2, 3.4),
    "42x4": (5.0, 5.5, 3.2, 3.4),
    "45x4": (5.0, 5.5, 3.2, 3.4),
    "48x4": (5.0, 5.5, 3.2, 3.4),
    "50x4": (5.0, 5.5, 3.2, 3.4),
    "55x4": (5.0, 5.5, 3.2, 3.4),
    "60x4": (5.0, 5.5, 3.2, 3.4),
    "65x4": (5.0, 5.5, 3.2, 3.4),
    "70x4": (5.0, 5.5, 3.2, 3.4),
    "75x4": (5.0, 5.5, 3.2, 3.4),
    "80x4": (5.0, 5.5, 3.2, 3.4),
    "85x4": (5.0, 5.5, 3.2, 3.4),
    "90x4": (5.0, 5.5, 3.2, 3.4),
    "95x4": (5.0, 5.5, 3.2, 3.4),
    "100x4": (5.0, 5.5, 3.2, 3.4),

    # CS = 5.0mm series
    "30x5": (6.3, 6.9, 4.0, 4.25),
    "35x5": (6.3, 6.9, 4.0, 4.25),
    "40x5": (6.3, 6.9, 4.0, 4.25),
    "45x5": (6.3, 6.9, 4.0, 4.25),
    "50x5": (6.3, 6.9, 4.0, 4.25),
    "55x5": (6.3, 6.9, 4.0, 4.25),
    "60x5": (6.3, 6.9, 4.0, 4.25),
    "65x5": (6.3, 6.9, 4.0, 4.25),
    "70x5": (6.3, 6.9, 4.0, 4.25),
    "75x5": (6.3, 6.9, 4.0, 4.25),
    "80x5": (6.3, 6.9, 4.0, 4.25),
    "85x5": (6.3, 6.9, 4.0, 4.25),
    "90x5": (6.3, 6.9, 4.0, 4.25),
    "95x5": (6.3, 6.9, 4.0, 4.25),
    "100x5": (6.3, 6.9, 4.0, 4.25),
    "110x5": (6.3, 6.9, 4.0, 4.25),
    "120x5": (6.3, 6.9, 4.0, 4.25),
}


def _get_tolerances(id_mm: float, cs_mm: float) -> tuple:
    """Get tolerances for given ID and CS."""
    # Default tolerances (ISO 3601-1 Grade N approximation)
    if cs_mm <= 2.0:
        cs_tol = 0.08
    elif cs_mm <= 3.0:
        cs_tol = 0.09
    elif cs_mm <= 4.0:
        cs_tol = 0.10
    else:
        cs_tol = 0.13

    if id_mm < 10:
        id_tol = 0.15
    elif id_mm < 30:
        id_tol = 0.20
    elif id_mm < 50:
        id_tol = 0.25
    else:
        id_tol = 0.30

    return (id_tol, id_tol, cs_tol, cs_tol)


def _create_oring_spec(designation: str, data: tuple) -> ORingSpec:
    """Create an ORingSpec from raw data."""
    # Parse designation "IDxCS"
    parts = designation.lower().split("x")
    id_mm = float(parts[0])
    cs_mm = float(parts[1])

    groove_w_static, groove_w_dyn, groove_d_static, groove_d_dyn = data
    id_tol_p, id_tol_m, cs_tol_p, cs_tol_m = _get_tolerances(id_mm, cs_mm)

    return ORingSpec(
        designation=designation,
        inner_diameter=id_mm,
        cross_section=cs_mm,
        id_tolerance_plus=id_tol_p,
        id_tolerance_minus=id_tol_m,
        cs_tolerance_plus=cs_tol_p,
        cs_tolerance_minus=cs_tol_m,
        groove_width_static=groove_w_static,
        groove_width_dynamic=groove_w_dyn,
        groove_depth_static=groove_d_static,
        groove_depth_dynamic=groove_d_dyn,
        standard="ISO 3601",
        name_zh=f"O形圈 {designation}",
        name_en=f"O-ring {designation}",
    )


# Build O-ring database
ORING_DATABASE: Dict[str, ORingSpec] = {}

for desig, data in METRIC_ORING_DATA.items():
    spec = _create_oring_spec(desig, data)
    ORING_DATABASE[desig] = spec


def get_oring_spec(designation: str) -> Optional[ORingSpec]:
    """
    Get O-ring specification by designation.

    Args:
        designation: O-ring designation (e.g., "10x2", "20x3")

    Returns:
        ORingSpec or None if not found

    Example:
        >>> spec = get_oring_spec("20x3")
        >>> print(f"ID: {spec.inner_diameter}mm, CS: {spec.cross_section}mm")
    """
    # Normalize designation
    normalized = designation.lower().replace(" ", "").replace("×", "x")

    if normalized in ORING_DATABASE:
        return ORING_DATABASE[normalized]

    # Try without decimal points
    for key, spec in ORING_DATABASE.items():
        if key.replace(".0", "") == normalized.replace(".0", ""):
            return spec

    return None


def get_oring_by_id(
    inner_diameter: float,
    cross_section: Optional[float] = None,
) -> List[ORingSpec]:
    """
    Find O-rings by inner diameter.

    Args:
        inner_diameter: Inner diameter ID (mm)
        cross_section: Optional cross-section filter (mm)

    Returns:
        List of matching ORingSpec objects
    """
    results = []

    for spec in ORING_DATABASE.values():
        if spec.inner_diameter != inner_diameter:
            continue
        if cross_section and spec.cross_section != cross_section:
            continue
        results.append(spec)

    return sorted(results, key=lambda x: x.cross_section)


def list_orings(
    min_id: float = 0,
    max_id: float = 1000,
    cross_section: Optional[float] = None,
) -> List[ORingSpec]:
    """
    List O-rings filtered by ID range and cross-section.

    Args:
        min_id: Minimum inner diameter (mm)
        max_id: Maximum inner diameter (mm)
        cross_section: Filter by specific cross-section (mm)

    Returns:
        List of matching ORingSpec objects
    """
    results = []

    for spec in ORING_DATABASE.values():
        if spec.inner_diameter < min_id or spec.inner_diameter > max_id:
            continue
        if cross_section and spec.cross_section != cross_section:
            continue
        results.append(spec)

    return sorted(results, key=lambda x: (x.cross_section, x.inner_diameter))


def suggest_oring_material(
    temperature_min: float,
    temperature_max: float,
    medium: str = "oil",
) -> List[ORingMaterial]:
    """
    Suggest suitable O-ring materials based on conditions.

    Args:
        temperature_min: Minimum operating temperature (°C)
        temperature_max: Maximum operating temperature (°C)
        medium: Contact medium ("oil", "water", "air", "fuel", "chemical")

    Returns:
        List of suitable materials sorted by preference
    """
    # Material temperature ranges and medium compatibility
    materials = {
        ORingMaterial.NBR: {
            "temp": (-30, 100),
            "media": ["oil", "fuel", "water", "air"],
        },
        ORingMaterial.FKM: {
            "temp": (-20, 200),
            "media": ["oil", "fuel", "chemical", "air"],
        },
        ORingMaterial.EPDM: {
            "temp": (-50, 150),
            "media": ["water", "steam", "air"],
        },
        ORingMaterial.SILICONE: {
            "temp": (-60, 200),
            "media": ["air", "water"],
        },
        ORingMaterial.PTFE: {
            "temp": (-200, 260),
            "media": ["oil", "fuel", "water", "air", "chemical"],
        },
        ORingMaterial.NEOPRENE: {
            "temp": (-40, 100),
            "media": ["oil", "water", "air"],
        },
    }

    suitable = []

    for mat, props in materials.items():
        temp_min, temp_max = props["temp"]

        # Check temperature range
        if temperature_min < temp_min or temperature_max > temp_max:
            continue

        # Check medium compatibility
        if medium.lower() not in props["media"]:
            continue

        suitable.append(mat)

    # Sort by preference (NBR is most common, then FKM, etc.)
    preference = [ORingMaterial.NBR, ORingMaterial.FKM, ORingMaterial.EPDM,
                  ORingMaterial.SILICONE, ORingMaterial.NEOPRENE, ORingMaterial.PTFE]

    return sorted(suitable, key=lambda x: preference.index(x) if x in preference else 99)


def calculate_compression(
    oring_cs: float,
    groove_depth: float,
) -> Dict[str, float]:
    """
    Calculate O-ring compression percentage.

    Args:
        oring_cs: O-ring cross-section diameter (mm)
        groove_depth: Groove depth (mm)

    Returns:
        Dictionary with compression data
    """
    compression = (oring_cs - groove_depth) / oring_cs * 100

    # Recommended compression ranges
    if compression < 10:
        status = "too_low"
        recommendation = "Increase compression to 15-25%"
    elif compression > 30:
        status = "too_high"
        recommendation = "Reduce compression to 15-25%"
    else:
        status = "ok"
        recommendation = "Compression within acceptable range"

    return {
        "compression_percent": round(compression, 1),
        "status": status,
        "recommendation": recommendation,
        "recommended_min": 15.0,
        "recommended_max": 25.0,
    }
