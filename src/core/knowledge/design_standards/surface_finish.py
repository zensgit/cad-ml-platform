"""
Surface Finish Standards Knowledge Base.

Provides surface roughness (Ra) values and selection guidance per ISO 1302.

Reference:
- ISO 1302:2002 - Geometrical Product Specifications (GPS) - Surface texture
- ISO 4287:1997 - Surface texture: Profile method
- GB/T 1031-2009 (Chinese equivalent)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class SurfaceFinishGrade(str, Enum):
    """Surface finish grade classification."""

    N1 = "N1"  # Ra 0.025 μm - Mirror finish
    N2 = "N2"  # Ra 0.05 μm
    N3 = "N3"  # Ra 0.1 μm - Super finish
    N4 = "N4"  # Ra 0.2 μm - Lapping
    N5 = "N5"  # Ra 0.4 μm - Honing
    N6 = "N6"  # Ra 0.8 μm - Fine grinding
    N7 = "N7"  # Ra 1.6 μm - Grinding
    N8 = "N8"  # Ra 3.2 μm - Fine turning/milling
    N9 = "N9"  # Ra 6.3 μm - Turning/milling
    N10 = "N10"  # Ra 12.5 μm - Rough machining
    N11 = "N11"  # Ra 25 μm - Coarse machining
    N12 = "N12"  # Ra 50 μm - Very coarse


# Surface finish table: Grade -> (Ra_μm, Rz_μm_approx, typical_process_zh, typical_process_en)
SURFACE_FINISH_TABLE: Dict[SurfaceFinishGrade, Tuple[float, float, str, str]] = {
    SurfaceFinishGrade.N1: (0.025, 0.1, "超精研磨", "Super-finishing, lapping"),
    SurfaceFinishGrade.N2: (0.05, 0.2, "精密研磨", "Precision lapping"),
    SurfaceFinishGrade.N3: (0.1, 0.4, "研磨、珩磨", "Lapping, honing"),
    SurfaceFinishGrade.N4: (0.2, 0.8, "精密珩磨", "Precision honing"),
    SurfaceFinishGrade.N5: (0.4, 1.6, "珩磨、精密磨削", "Honing, precision grinding"),
    SurfaceFinishGrade.N6: (0.8, 3.2, "精密磨削", "Fine grinding"),
    SurfaceFinishGrade.N7: (1.6, 6.3, "磨削、精车", "Grinding, fine turning"),
    SurfaceFinishGrade.N8: (3.2, 12.5, "精车、精铣", "Fine turning, fine milling"),
    SurfaceFinishGrade.N9: (6.3, 25, "车削、铣削", "Turning, milling"),
    SurfaceFinishGrade.N10: (12.5, 50, "粗车、粗铣", "Rough turning, rough milling"),
    SurfaceFinishGrade.N11: (25, 100, "粗加工", "Rough machining"),
    SurfaceFinishGrade.N12: (50, 200, "铸造、锻造表面", "As-cast, as-forged"),
}


# Application-based surface finish recommendations
# Format: application -> (recommended_grade, Ra_range, description)
SURFACE_FINISH_APPLICATIONS: Dict[str, Dict] = {
    # Precision fits
    "bearing_journal": {
        "grade": SurfaceFinishGrade.N6,
        "ra_range": (0.4, 1.6),
        "description_zh": "轴承配合表面",
        "description_en": "Bearing journal surface",
    },
    "bearing_housing": {
        "grade": SurfaceFinishGrade.N7,
        "ra_range": (0.8, 3.2),
        "description_zh": "轴承座孔",
        "description_en": "Bearing housing bore",
    },
    "piston_bore": {
        "grade": SurfaceFinishGrade.N5,
        "ra_range": (0.2, 0.8),
        "description_zh": "活塞缸筒",
        "description_en": "Cylinder bore",
    },
    "piston_surface": {
        "grade": SurfaceFinishGrade.N6,
        "ra_range": (0.4, 1.6),
        "description_zh": "活塞表面",
        "description_en": "Piston surface",
    },

    # Sealing surfaces
    "oring_groove": {
        "grade": SurfaceFinishGrade.N7,
        "ra_range": (1.6, 3.2),
        "description_zh": "O形圈沟槽",
        "description_en": "O-ring groove",
    },
    "gasket_surface": {
        "grade": SurfaceFinishGrade.N8,
        "ra_range": (1.6, 6.3),
        "description_zh": "密封垫片接触面",
        "description_en": "Gasket contact surface",
    },
    "lip_seal_shaft": {
        "grade": SurfaceFinishGrade.N6,
        "ra_range": (0.2, 0.8),
        "description_zh": "油封轴颈",
        "description_en": "Lip seal shaft",
    },

    # Thread surfaces
    "thread_fit": {
        "grade": SurfaceFinishGrade.N8,
        "ra_range": (1.6, 6.3),
        "description_zh": "螺纹配合面",
        "description_en": "Thread surface",
    },

    # General fits
    "clearance_fit": {
        "grade": SurfaceFinishGrade.N8,
        "ra_range": (1.6, 6.3),
        "description_zh": "间隙配合表面",
        "description_en": "Clearance fit surface",
    },
    "transition_fit": {
        "grade": SurfaceFinishGrade.N7,
        "ra_range": (0.8, 3.2),
        "description_zh": "过渡配合表面",
        "description_en": "Transition fit surface",
    },
    "interference_fit": {
        "grade": SurfaceFinishGrade.N6,
        "ra_range": (0.4, 1.6),
        "description_zh": "过盈配合表面",
        "description_en": "Interference fit surface",
    },

    # Non-critical surfaces
    "free_surface": {
        "grade": SurfaceFinishGrade.N9,
        "ra_range": (3.2, 12.5),
        "description_zh": "自由表面",
        "description_en": "Free surface",
    },
    "non_functional": {
        "grade": SurfaceFinishGrade.N10,
        "ra_range": (6.3, 25),
        "description_zh": "非功能表面",
        "description_en": "Non-functional surface",
    },
}


def get_ra_value(grade: SurfaceFinishGrade) -> float:
    """
    Get Ra value for a surface finish grade.

    Args:
        grade: Surface finish grade (N1-N12)

    Returns:
        Ra value in micrometers

    Example:
        >>> get_ra_value(SurfaceFinishGrade.N7)
        1.6
    """
    data = SURFACE_FINISH_TABLE.get(grade)
    if data:
        return data[0]
    return 0.0


def get_surface_finish_for_application(application: str) -> Optional[Dict]:
    """
    Get recommended surface finish for an application.

    Args:
        application: Application key (e.g., "bearing_journal", "oring_groove")

    Returns:
        Dictionary with grade, Ra range, and description

    Example:
        >>> result = get_surface_finish_for_application("bearing_journal")
        >>> print(f"Ra: {result['ra_range']}")
    """
    app_data = SURFACE_FINISH_APPLICATIONS.get(application.lower())
    if app_data is None:
        return None

    grade = app_data["grade"]
    grade_data = SURFACE_FINISH_TABLE.get(grade)

    return {
        "application": application,
        "grade": grade.value,
        "ra_value": grade_data[0] if grade_data else None,
        "ra_range": app_data["ra_range"],
        "process_zh": grade_data[2] if grade_data else "",
        "process_en": grade_data[3] if grade_data else "",
        "description_zh": app_data["description_zh"],
        "description_en": app_data["description_en"],
    }


def suggest_surface_finish(target_ra: float) -> SurfaceFinishGrade:
    """
    Suggest appropriate surface finish grade for target Ra.

    Args:
        target_ra: Target Ra value in micrometers

    Returns:
        Closest standard surface finish grade
    """
    # Find closest grade
    closest_grade = SurfaceFinishGrade.N12
    closest_diff = float('inf')

    for grade, (ra, _, _, _) in SURFACE_FINISH_TABLE.items():
        diff = abs(ra - target_ra)
        if diff < closest_diff:
            closest_diff = diff
            closest_grade = grade

    return closest_grade


def list_surface_finishes() -> List[Dict]:
    """
    List all surface finish grades with details.

    Returns:
        List of dictionaries with grade details
    """
    results = []
    for grade, (ra, rz, process_zh, process_en) in SURFACE_FINISH_TABLE.items():
        results.append({
            "grade": grade.value,
            "ra_um": ra,
            "rz_um": rz,
            "process_zh": process_zh,
            "process_en": process_en,
        })
    return results
