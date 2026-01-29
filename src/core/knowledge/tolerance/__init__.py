"""
Tolerance and Fits Knowledge Module.

Provides ISO tolerance grades (IT01-IT18), standard fit systems (hole/shaft basis),
and selection guidance for mechanical design applications.

Reference Standards:
- ISO 286-1:2010 - Geometrical product specifications (GPS) - ISO code system
- ISO 286-2:2010 - Tables of standard tolerance classes and limit deviations
- GB/T 1800.1-2020 - Chinese national standard (equivalent to ISO 286)
"""

from .it_grades import (
    ITGrade,
    get_tolerance_value,
    get_tolerance_table,
    TOLERANCE_GRADES,
)
from .fits import (
    FitType,
    FitClass,
    ShaftDeviation,
    HoleDeviation,
    get_fit_deviations,
    get_common_fits,
    COMMON_FITS,
)
from .selection import (
    FitApplication,
    select_fit_for_application,
    get_fit_recommendations,
)

__all__ = [
    # IT Grades
    "ITGrade",
    "get_tolerance_value",
    "get_tolerance_table",
    "TOLERANCE_GRADES",
    # Fits
    "FitType",
    "FitClass",
    "ShaftDeviation",
    "HoleDeviation",
    "get_fit_deviations",
    "get_common_fits",
    "COMMON_FITS",
    # Selection
    "FitApplication",
    "select_fit_for_application",
    "get_fit_recommendations",
]
