"""
Fit Selection Guidance Knowledge Base.

Provides recommendations for selecting appropriate fits based on
application requirements, operating conditions, and design constraints.

Reference:
- ISO 286-1:2010 Annex B - Guide to the selection of fits
- Engineering design handbooks
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .fits import FitType, FitClass, COMMON_FITS


class FitApplication(str, Enum):
    """Common application categories for fit selection."""

    # Rotating assemblies
    BEARING_ROTATING = "bearing_rotating"  # 转动轴承
    BEARING_STATIONARY = "bearing_stationary"  # 固定轴承座
    SPINDLE = "spindle"  # 主轴
    PULLEY = "pulley"  # 皮带轮
    GEAR_HUB = "gear_hub"  # 齿轮轮毂
    COUPLING = "coupling"  # 联轴器
    FLYWHEEL = "flywheel"  # 飞轮

    # Sliding assemblies
    SLIDE_GUIDE = "slide_guide"  # 滑动导轨
    PISTON = "piston"  # 活塞
    VALVE_GUIDE = "valve_guide"  # 阀门导向

    # Location/positioning
    LOCATING_PIN = "locating_pin"  # 定位销
    DOWEL_PIN = "dowel_pin"  # 销钉
    KEY_KEYWAY = "key_keyway"  # 键与键槽

    # Press fits
    PRESS_FIT_BUSH = "press_fit_bush"  # 压入衬套
    PRESS_FIT_RING = "press_fit_ring"  # 压入环
    SHRINK_FIT = "shrink_fit"  # 热装配合

    # General
    ASSEMBLY_FREQUENT = "assembly_frequent"  # 频繁拆装
    ASSEMBLY_PERMANENT = "assembly_permanent"  # 永久装配
    ADJUSTABLE = "adjustable"  # 可调节


@dataclass
class FitRecommendation:
    """Fit recommendation with rationale."""

    fit_code: str
    fit_type: FitType
    suitability: float  # 0.0-1.0 suitability score
    rationale_zh: str
    rationale_en: str
    conditions: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)


# Application-based fit selection rules
FIT_SELECTION_RULES: Dict[FitApplication, Dict] = {
    FitApplication.BEARING_ROTATING: {
        "primary": ["H7/g6", "H7/h6"],
        "heavy_load": ["H7/k6", "H7/m6"],
        "light_load": ["H8/f7", "H9/d9"],
        "description_zh": "转动轴承配合",
        "description_en": "Rotating bearing fits",
        "factors": ["转速", "载荷", "润滑条件"],
        "factors_en": ["Speed", "Load", "Lubrication"],
    },
    FitApplication.BEARING_STATIONARY: {
        "primary": ["H7/k6", "H7/m6"],
        "heavy_load": ["H7/n6", "H7/p6"],
        "light_load": ["H7/js6", "H7/h6"],
        "description_zh": "固定轴承座配合",
        "description_en": "Stationary bearing housing fits",
        "factors": ["轴承类型", "载荷方向", "拆装需求"],
        "factors_en": ["Bearing type", "Load direction", "Dismantling needs"],
    },
    FitApplication.SPINDLE: {
        "primary": ["H7/g6"],
        "precision": ["H6/g5", "H6/h5"],
        "general": ["H7/h6"],
        "description_zh": "主轴配合",
        "description_en": "Spindle fits",
        "factors": ["精度等级", "转速", "刚度要求"],
        "factors_en": ["Precision class", "Speed", "Stiffness requirement"],
    },
    FitApplication.PULLEY: {
        "primary": ["H7/k6", "H7/m6"],
        "keyed": ["H8/h7"],
        "loose": ["H11/c11"],
        "description_zh": "皮带轮配合",
        "description_en": "Pulley fits",
        "factors": ["传递扭矩", "键连接", "安装方式"],
        "factors_en": ["Torque transmission", "Key connection", "Mounting method"],
    },
    FitApplication.GEAR_HUB: {
        "primary": ["H7/k6", "H7/m6"],
        "high_torque": ["H7/p6", "H7/r6"],
        "with_key": ["H7/js6", "H7/h6"],
        "description_zh": "齿轮轮毂配合",
        "description_en": "Gear hub fits",
        "factors": ["传递功率", "冲击载荷", "键连接"],
        "factors_en": ["Power transmission", "Shock loads", "Key connection"],
    },
    FitApplication.COUPLING: {
        "primary": ["H7/k6"],
        "precise": ["H7/js6"],
        "heavy_duty": ["H7/m6", "H7/n6"],
        "description_zh": "联轴器配合",
        "description_en": "Coupling fits",
        "factors": ["同心度要求", "扭矩大小", "拆装频率"],
        "factors_en": ["Concentricity", "Torque level", "Assembly frequency"],
    },
    FitApplication.FLYWHEEL: {
        "primary": ["H7/r6", "H7/s6"],
        "shrink_fit": ["H7/u6"],
        "keyed": ["H7/n6"],
        "description_zh": "飞轮配合",
        "description_en": "Flywheel fits",
        "factors": ["离心力", "扭矩波动", "键连接"],
        "factors_en": ["Centrifugal force", "Torque fluctuation", "Key connection"],
    },
    FitApplication.SLIDE_GUIDE: {
        "primary": ["H7/g6", "H7/h6"],
        "precision": ["H6/g5"],
        "general": ["H8/f7"],
        "description_zh": "滑动导轨配合",
        "description_en": "Slide guide fits",
        "factors": ["滑动精度", "润滑", "磨损补偿"],
        "factors_en": ["Sliding precision", "Lubrication", "Wear compensation"],
    },
    FitApplication.PISTON: {
        "primary": ["H8/f7"],
        "precision": ["H7/f7"],
        "general": ["H9/e8"],
        "description_zh": "活塞配合",
        "description_en": "Piston fits",
        "factors": ["工作温度", "密封要求", "热膨胀"],
        "factors_en": ["Operating temperature", "Sealing", "Thermal expansion"],
    },
    FitApplication.LOCATING_PIN: {
        "primary": ["H7/m6", "H7/n6"],
        "removable": ["H7/k6"],
        "permanent": ["H7/p6"],
        "description_zh": "定位销配合",
        "description_en": "Locating pin fits",
        "factors": ["定位精度", "拆装需求", "载荷"],
        "factors_en": ["Location accuracy", "Dismantling needs", "Load"],
    },
    FitApplication.DOWEL_PIN: {
        "primary": ["H7/m6"],
        "precision": ["H7/n6"],
        "general": ["H7/k6"],
        "description_zh": "销钉配合",
        "description_en": "Dowel pin fits",
        "factors": ["定位功能", "剪切载荷", "材料"],
        "factors_en": ["Location function", "Shear load", "Material"],
    },
    FitApplication.KEY_KEYWAY: {
        "hub_fit": ["H9/d9", "H9/e9"],
        "shaft_fit": ["N9/h9", "P9/h9"],
        "sliding": ["D10/h9"],
        "description_zh": "键与键槽配合",
        "description_en": "Key and keyway fits",
        "factors": ["键类型", "扭矩大小", "轴向滑动"],
        "factors_en": ["Key type", "Torque level", "Axial sliding"],
    },
    FitApplication.PRESS_FIT_BUSH: {
        "primary": ["H7/p6", "H7/r6"],
        "light": ["H7/n6"],
        "heavy": ["H7/s6"],
        "description_zh": "压入衬套配合",
        "description_en": "Press-fit bushing",
        "factors": ["衬套材料", "壁厚", "过盈量"],
        "factors_en": ["Bushing material", "Wall thickness", "Interference"],
    },
    FitApplication.SHRINK_FIT: {
        "primary": ["H7/s6", "H7/u6"],
        "light": ["H7/r6"],
        "heavy": ["H7/x6"],
        "description_zh": "热装配合",
        "description_en": "Shrink fit",
        "factors": ["温度差", "材料膨胀系数", "装配应力"],
        "factors_en": ["Temperature difference", "Expansion coefficient", "Assembly stress"],
    },
    FitApplication.ASSEMBLY_FREQUENT: {
        "primary": ["H7/h6", "H7/g6"],
        "loose": ["H8/f7", "H9/d9"],
        "snug": ["H7/js6"],
        "description_zh": "频繁拆装配合",
        "description_en": "Frequently dismantled assemblies",
        "factors": ["拆装频率", "定位精度", "工具"],
        "factors_en": ["Assembly frequency", "Location accuracy", "Tools"],
    },
    FitApplication.ASSEMBLY_PERMANENT: {
        "primary": ["H7/p6", "H7/r6"],
        "light": ["H7/n6"],
        "heavy": ["H7/s6", "H7/u6"],
        "description_zh": "永久装配",
        "description_en": "Permanent assembly",
        "factors": ["传递载荷", "振动", "安全系数"],
        "factors_en": ["Load transmission", "Vibration", "Safety factor"],
    },
}


def select_fit_for_application(
    application: FitApplication,
    load_level: str = "normal",
    precision_level: str = "normal",
) -> List[FitRecommendation]:
    """
    Select appropriate fits for a given application.

    Args:
        application: Application type from FitApplication enum
        load_level: "light", "normal", or "heavy"
        precision_level: "general", "normal", or "precision"

    Returns:
        List of FitRecommendation objects sorted by suitability

    Example:
        >>> recs = select_fit_for_application(FitApplication.GEAR_HUB, "heavy")
        >>> print(recs[0].fit_code)
        'H7/p6'
    """
    rules = FIT_SELECTION_RULES.get(application)
    if rules is None:
        return []

    recommendations = []

    # Determine which fit category to use
    if load_level == "heavy":
        fit_key = "heavy_load" if "heavy_load" in rules else "heavy"
        if fit_key not in rules:
            fit_key = "primary"
    elif load_level == "light":
        fit_key = "light_load" if "light_load" in rules else "light"
        if fit_key not in rules:
            fit_key = "primary"
    else:
        if precision_level == "precision":
            fit_key = "precision" if "precision" in rules else "primary"
        elif precision_level == "general":
            fit_key = "general" if "general" in rules else "primary"
        else:
            fit_key = "primary"

    primary_fits = rules.get(fit_key, rules.get("primary", []))
    all_fits = rules.get("primary", [])

    # Build recommendations
    for i, fit_code in enumerate(primary_fits):
        fit_data = COMMON_FITS.get(fit_code)
        if fit_data is None:
            continue

        suitability = 1.0 - (i * 0.1)  # First choice is most suitable

        rec = FitRecommendation(
            fit_code=fit_code,
            fit_type=fit_data["type"],
            suitability=min(1.0, max(0.0, suitability)),
            rationale_zh=f"{rules['description_zh']}的推荐配合",
            rationale_en=f"Recommended fit for {rules['description_en'].lower()}",
            conditions=rules.get("factors_en", []),
            alternatives=[f for f in all_fits if f != fit_code],
        )
        recommendations.append(rec)

    return sorted(recommendations, key=lambda x: x.suitability, reverse=True)


def get_fit_recommendations(
    fit_type: Optional[FitType] = None,
    fit_class: Optional[FitClass] = None,
) -> List[Dict]:
    """
    Get fit recommendations filtered by type or class.

    Args:
        fit_type: Filter by FitType (CLEARANCE, TRANSITION, INTERFERENCE)
        fit_class: Filter by FitClass

    Returns:
        List of fit information dictionaries
    """
    results = []

    for code, data in COMMON_FITS.items():
        if fit_type and data["type"] != fit_type:
            continue
        if fit_class and data.get("class") != fit_class:
            continue

        results.append({
            "code": code,
            "type": data["type"].value,
            "class": data.get("class", FitClass.SLIDING).value if data.get("class") else None,
            "name_zh": data["name_zh"],
            "name_en": data["name_en"],
            "application_zh": data["application_zh"],
            "application_en": data["application_en"],
        })

    return results


def suggest_fit_by_clearance(
    min_clearance_um: float,
    max_clearance_um: float,
    nominal_size_mm: float = 25.0,
) -> List[str]:
    """
    Suggest fits based on desired clearance range.

    Args:
        min_clearance_um: Minimum clearance in micrometers (negative for interference)
        max_clearance_um: Maximum clearance in micrometers (negative for interference)
        nominal_size_mm: Reference nominal size for comparison

    Returns:
        List of fit codes that may achieve the desired clearance
    """
    from .fits import get_fit_deviations

    suggestions = []

    for code in COMMON_FITS:
        deviations = get_fit_deviations(code, nominal_size_mm)
        if deviations is None:
            continue

        # Check if the fit's clearance range overlaps with desired range
        fit_min = deviations.min_clearance_um
        fit_max = deviations.max_clearance_um

        # Check for overlap
        if fit_min <= max_clearance_um and fit_max >= min_clearance_um:
            suggestions.append(code)

    return suggestions


def get_application_guide(application: FitApplication) -> Optional[Dict]:
    """
    Get detailed guidance for a specific application.

    Args:
        application: Application type

    Returns:
        Dictionary with selection rules and factors
    """
    rules = FIT_SELECTION_RULES.get(application)
    if rules is None:
        return None

    return {
        "application": application.value,
        "description_zh": rules.get("description_zh"),
        "description_en": rules.get("description_en"),
        "primary_fits": rules.get("primary", []),
        "selection_factors_zh": rules.get("factors", []),
        "selection_factors_en": rules.get("factors_en", []),
        "all_options": {
            k: v for k, v in rules.items()
            if k not in ["description_zh", "description_en", "factors", "factors_en"]
        },
    }


def list_applications_for_fit(fit_code: str) -> List[FitApplication]:
    """
    Find applications where a specific fit is recommended.

    Args:
        fit_code: Standard fit code (e.g., "H7/g6")

    Returns:
        List of FitApplication enums where this fit is recommended
    """
    applications = []

    for app, rules in FIT_SELECTION_RULES.items():
        for category, fits in rules.items():
            if isinstance(fits, list) and fit_code in fits:
                applications.append(app)
                break

    return applications
