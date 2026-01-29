"""
GD&T Symbols and Characteristics.

Defines geometric characteristic symbols, modifiers, and their
meanings per ISO 1101 and ASME Y14.5.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class GDTCategory(str, Enum):
    """Categories of geometric characteristics."""

    FORM = "form"  # 形状公差
    ORIENTATION = "orientation"  # 方向公差
    LOCATION = "location"  # 位置公差
    RUNOUT = "runout"  # 跳动公差


class GDTCharacteristic(str, Enum):
    """Geometric characteristic symbols per ISO 1101."""

    # Form tolerances (形状公差) - no datum required
    STRAIGHTNESS = "straightness"  # 直线度 ⏤
    FLATNESS = "flatness"  # 平面度 ⏥
    CIRCULARITY = "circularity"  # 圆度 ○
    CYLINDRICITY = "cylindricity"  # 圆柱度 ⌭

    # Orientation tolerances (方向公差) - datum required
    PERPENDICULARITY = "perpendicularity"  # 垂直度 ⊥
    PARALLELISM = "parallelism"  # 平行度 ∥
    ANGULARITY = "angularity"  # 倾斜度 ∠

    # Location tolerances (位置公差) - datum required
    POSITION = "position"  # 位置度 ⌖
    CONCENTRICITY = "concentricity"  # 同心度 ◎
    SYMMETRY = "symmetry"  # 对称度 ⌯

    # Profile tolerances (轮廓度)
    PROFILE_LINE = "profile_line"  # 线轮廓度 ⌒
    PROFILE_SURFACE = "profile_surface"  # 面轮廓度 ⌓

    # Runout tolerances (跳动公差) - datum required
    CIRCULAR_RUNOUT = "circular_runout"  # 圆跳动 ↗
    TOTAL_RUNOUT = "total_runout"  # 全跳动 ⇗


class ToleranceModifier(str, Enum):
    """Tolerance zone modifiers."""

    MMC = "M"  # Maximum Material Condition 最大实体条件
    LMC = "L"  # Least Material Condition 最小实体条件
    RFS = "S"  # Regardless of Feature Size (default in ISO)
    PROJECTED = "P"  # Projected tolerance zone 延伸公差带
    FREE_STATE = "F"  # Free state condition
    TANGENT_PLANE = "T"  # Tangent plane
    UNEQUAL = "U"  # Unequally disposed tolerance zone


class DatumModifier(str, Enum):
    """Datum feature modifiers."""

    MMB = "M"  # Maximum Material Boundary
    LMB = "L"  # Least Material Boundary
    RMB = ""  # Regardless of Material Boundary (default)


@dataclass
class GDTSymbolInfo:
    """Information about a GD&T symbol."""

    characteristic: GDTCharacteristic
    category: GDTCategory
    symbol_unicode: str
    name_en: str
    name_zh: str
    requires_datum: bool
    description_en: str
    description_zh: str
    typical_applications: List[str]


# GD&T Symbol Database
GDT_SYMBOLS: Dict[GDTCharacteristic, Dict] = {
    # Form tolerances
    GDTCharacteristic.STRAIGHTNESS: {
        "category": GDTCategory.FORM,
        "symbol": "⏤",
        "name_en": "Straightness",
        "name_zh": "直线度",
        "requires_datum": False,
        "description_en": "Controls how straight a line element must be",
        "description_zh": "控制线要素的直线程度",
        "applications": ["轴类零件的母线", "边缘直线度", "中心线直线度"],
        "tolerance_zone": "两平行直线或两平行平面之间",
        "inspection": ["直尺", "塞尺", "光学投影仪"],
    },
    GDTCharacteristic.FLATNESS: {
        "category": GDTCategory.FORM,
        "symbol": "⏥",
        "name_en": "Flatness",
        "name_zh": "平面度",
        "requires_datum": False,
        "description_en": "Controls how flat a surface must be",
        "description_zh": "控制平面的平整程度",
        "applications": ["密封面", "安装基准面", "导轨面"],
        "tolerance_zone": "两平行平面之间",
        "inspection": ["平板", "百分表", "三坐标测量机"],
    },
    GDTCharacteristic.CIRCULARITY: {
        "category": GDTCategory.FORM,
        "symbol": "○",
        "name_en": "Circularity (Roundness)",
        "name_zh": "圆度",
        "requires_datum": False,
        "description_en": "Controls how circular a cross-section must be",
        "description_zh": "控制横截面的圆形程度",
        "applications": ["轴颈", "活塞", "轴承座孔"],
        "tolerance_zone": "同一截面上两同心圆之间的区域",
        "inspection": ["圆度仪", "V形块+百分表", "三坐标测量机"],
    },
    GDTCharacteristic.CYLINDRICITY: {
        "category": GDTCategory.FORM,
        "symbol": "⌭",
        "name_en": "Cylindricity",
        "name_zh": "圆柱度",
        "requires_datum": False,
        "description_en": "Controls how closely a surface conforms to a perfect cylinder",
        "description_zh": "控制圆柱面的圆柱形程度，综合圆度、直线度、平行度",
        "applications": ["精密轴", "液压缸筒", "精密轴承座"],
        "tolerance_zone": "两同轴圆柱面之间的区域",
        "inspection": ["圆度仪", "三坐标测量机"],
    },

    # Orientation tolerances
    GDTCharacteristic.PERPENDICULARITY: {
        "category": GDTCategory.ORIENTATION,
        "symbol": "⊥",
        "name_en": "Perpendicularity",
        "name_zh": "垂直度",
        "requires_datum": True,
        "description_en": "Controls how perpendicular a feature is to a datum",
        "description_zh": "控制被测要素相对于基准的垂直程度",
        "applications": ["法兰端面", "T型槽侧面", "立柱与底板"],
        "tolerance_zone": "垂直于基准的两平行平面之间",
        "inspection": ["直角尺", "百分表", "三坐标测量机"],
    },
    GDTCharacteristic.PARALLELISM: {
        "category": GDTCategory.ORIENTATION,
        "symbol": "∥",
        "name_en": "Parallelism",
        "name_zh": "平行度",
        "requires_datum": True,
        "description_en": "Controls how parallel a feature is to a datum",
        "description_zh": "控制被测要素相对于基准的平行程度",
        "applications": ["导轨面", "轴肩端面", "键槽底面"],
        "tolerance_zone": "平行于基准的两平行平面之间",
        "inspection": ["平板+百分表", "三坐标测量机"],
    },
    GDTCharacteristic.ANGULARITY: {
        "category": GDTCategory.ORIENTATION,
        "symbol": "∠",
        "name_en": "Angularity",
        "name_zh": "倾斜度",
        "requires_datum": True,
        "description_en": "Controls the orientation of a feature at a specified angle to a datum",
        "description_zh": "控制被测要素相对于基准成一定角度的程度",
        "applications": ["斜面", "锥面", "V形块"],
        "tolerance_zone": "与基准成指定角度的两平行平面之间",
        "inspection": ["角度块+百分表", "三坐标测量机", "万能角度尺"],
    },

    # Location tolerances
    GDTCharacteristic.POSITION: {
        "category": GDTCategory.LOCATION,
        "symbol": "⌖",
        "name_en": "Position",
        "name_zh": "位置度",
        "requires_datum": True,
        "description_en": "Controls the location of a feature from datums",
        "description_zh": "控制被测要素相对于基准的位置",
        "applications": ["螺栓孔组", "定位销孔", "装配孔"],
        "tolerance_zone": "以理论正确位置为中心的圆柱或球形区域",
        "inspection": ["三坐标测量机", "位置度检具", "功能检具"],
    },
    GDTCharacteristic.CONCENTRICITY: {
        "category": GDTCategory.LOCATION,
        "symbol": "◎",
        "name_en": "Concentricity",
        "name_zh": "同心度",
        "requires_datum": True,
        "description_en": "Controls how well the center points of a feature align with a datum axis",
        "description_zh": "控制被测轴线相对于基准轴线的同轴程度",
        "applications": ["多级轴", "套筒内外圆", "齿轮轴"],
        "tolerance_zone": "以基准轴线为轴线的圆柱区域",
        "inspection": ["三坐标测量机", "圆度仪"],
    },
    GDTCharacteristic.SYMMETRY: {
        "category": GDTCategory.LOCATION,
        "symbol": "⌯",
        "name_en": "Symmetry",
        "name_zh": "对称度",
        "requires_datum": True,
        "description_en": "Controls how well the median plane of a feature aligns with a datum",
        "description_zh": "控制被测中心平面相对于基准中心平面的对称程度",
        "applications": ["键槽", "T形槽", "对称凸台"],
        "tolerance_zone": "关于基准中心平面对称的两平行平面之间",
        "inspection": ["三坐标测量机", "专用检具"],
    },

    # Profile tolerances
    GDTCharacteristic.PROFILE_LINE: {
        "category": GDTCategory.FORM,
        "symbol": "⌒",
        "name_en": "Profile of a Line",
        "name_zh": "线轮廓度",
        "requires_datum": False,  # Can be with or without datum
        "description_en": "Controls the form of a line element on a surface",
        "description_zh": "控制曲线轮廓的形状，可带或不带基准",
        "applications": ["凸轮轮廓", "涡轮叶片截面", "曲面零件截面"],
        "tolerance_zone": "由理论正确轮廓线等距分布的两包络线之间",
        "inspection": ["轮廓仪", "光学投影仪", "三坐标测量机"],
    },
    GDTCharacteristic.PROFILE_SURFACE: {
        "category": GDTCategory.FORM,
        "symbol": "⌓",
        "name_en": "Profile of a Surface",
        "name_zh": "面轮廓度",
        "requires_datum": False,  # Can be with or without datum
        "description_en": "Controls the form of a surface",
        "description_zh": "控制曲面的形状，可带或不带基准",
        "applications": ["整体曲面", "模具型腔", "流线型外壳"],
        "tolerance_zone": "由理论正确曲面等距分布的两包络面之间",
        "inspection": ["三坐标测量机", "激光扫描仪"],
    },

    # Runout tolerances
    GDTCharacteristic.CIRCULAR_RUNOUT: {
        "category": GDTCategory.RUNOUT,
        "symbol": "↗",
        "name_en": "Circular Runout",
        "name_zh": "圆跳动",
        "requires_datum": True,
        "description_en": "Controls circular elements of a surface relative to a datum axis",
        "description_zh": "控制被测要素绕基准轴线旋转一周时的跳动量",
        "applications": ["轴颈", "法兰端面", "轴肩"],
        "tolerance_zone": "在任一测量截面上，以基准轴线为圆心的两同心圆之间",
        "inspection": ["V形块+百分表", "顶尖+百分表", "圆度仪"],
    },
    GDTCharacteristic.TOTAL_RUNOUT: {
        "category": GDTCategory.RUNOUT,
        "symbol": "⇗",
        "name_en": "Total Runout",
        "name_zh": "全跳动",
        "requires_datum": True,
        "description_en": "Controls all surface elements relative to a datum axis",
        "description_zh": "控制被测要素绕基准轴线连续旋转时的全部跳动量",
        "applications": ["精密轴外圆", "精密端面", "凸轮外圆"],
        "tolerance_zone": "以基准轴线为轴线的两同轴圆柱面之间（径向）或两平行平面之间（轴向）",
        "inspection": ["顶尖+百分表轴向移动", "圆度仪+轴向扫描"],
    },
}


def get_gdt_symbol(characteristic: GDTCharacteristic) -> Optional[GDTSymbolInfo]:
    """
    Get detailed information about a GD&T symbol.

    Args:
        characteristic: The geometric characteristic

    Returns:
        GDTSymbolInfo or None

    Example:
        >>> info = get_gdt_symbol(GDTCharacteristic.FLATNESS)
        >>> print(f"{info.name_zh}: {info.symbol_unicode}")
    """
    data = GDT_SYMBOLS.get(characteristic)
    if not data:
        return None

    return GDTSymbolInfo(
        characteristic=characteristic,
        category=data["category"],
        symbol_unicode=data["symbol"],
        name_en=data["name_en"],
        name_zh=data["name_zh"],
        requires_datum=data["requires_datum"],
        description_en=data["description_en"],
        description_zh=data["description_zh"],
        typical_applications=data["applications"],
    )


def get_all_symbols(category: Optional[GDTCategory] = None) -> List[GDTSymbolInfo]:
    """
    Get all GD&T symbols, optionally filtered by category.

    Args:
        category: Optional category filter

    Returns:
        List of GDTSymbolInfo
    """
    results = []
    for char, data in GDT_SYMBOLS.items():
        if category is None or data["category"] == category:
            results.append(get_gdt_symbol(char))
    return [r for r in results if r is not None]


def get_symbols_by_datum_requirement(requires_datum: bool) -> List[GDTSymbolInfo]:
    """
    Get symbols by datum requirement.

    Args:
        requires_datum: Whether datum is required

    Returns:
        List of matching symbols
    """
    results = []
    for char, data in GDT_SYMBOLS.items():
        if data["requires_datum"] == requires_datum:
            results.append(get_gdt_symbol(char))
    return [r for r in results if r is not None]
