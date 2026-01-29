"""
材料分类与属性系统

提供详细的材料分类、属性查询和工艺推荐功能。

分类层次：
1. 大类 (Category): 金属/非金属/组合件
2. 子类 (SubCategory): 钢铁/有色金属/塑料等
3. 材料组 (Group): 碳素钢/不锈钢/铝合金等
4. 牌号 (Grade): 具体材料牌号

每种材料包含：
- 分类信息
- 物理属性（密度、熔点等）
- 机械属性（强度、硬度等）
- 工艺建议（热处理、表面处理、加工参数等）
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MaterialCategory(str, Enum):
    """材料大类"""
    METAL = "metal"  # 金属
    NON_METAL = "non_metal"  # 非金属
    COMPOSITE = "composite"  # 组合件


class MaterialSubCategory(str, Enum):
    """材料子类"""
    # 金属子类
    FERROUS = "ferrous"  # 钢铁类
    NON_FERROUS = "non_ferrous"  # 有色金属

    # 非金属子类
    POLYMER = "polymer"  # 塑料/橡胶
    GLASS = "glass"  # 玻璃
    CERAMIC = "ceramic"  # 陶瓷/纤维
    ALUMINA_CERAMIC = "alumina_ceramic"  # 氧化铝陶瓷
    NITRIDE_CERAMIC = "nitride_ceramic"  # 氮化物陶瓷
    ZIRCONIA_CERAMIC = "zirconia_ceramic"  # 氧化锆陶瓷

    # 组合件
    ASSEMBLY = "assembly"  # 组合件


class MaterialGroup(str, Enum):
    """材料组"""
    # 钢铁类
    CARBON_STEEL = "carbon_steel"  # 碳素钢
    ALLOY_STEEL = "alloy_steel"  # 合金钢
    TOOL_STEEL = "tool_steel"  # 工具钢
    STAINLESS_STEEL = "stainless_steel"  # 不锈钢
    CORROSION_RESISTANT = "corrosion_resistant"  # 耐蚀合金
    CAST_IRON = "cast_iron"  # 铸铁

    # 有色金属
    ALUMINUM = "aluminum"  # 铝合金
    COPPER = "copper"  # 铜合金
    TITANIUM = "titanium"  # 钛合金
    NICKEL = "nickel"  # 镍合金
    MAGNESIUM = "magnesium"  # 镁合金
    CEMENTED_CARBIDE = "cemented_carbide"  # 硬质合金
    PRECISION_ALLOY = "precision_alloy"  # 精密合金
    ELECTRICAL_STEEL = "electrical_steel"  # 电工钢
    WELDING_MATERIAL = "welding_material"  # 焊接材料
    COMPOSITE = "composite"  # 复合材料
    POWDER_METALLURGY = "powder_metallurgy"  # 粉末冶金
    FREE_CUTTING_STEEL = "free_cutting_steel"  # 易切削钢
    WEAR_RESISTANT_STEEL = "wear_resistant_steel"  # 耐磨钢
    BEARING_STEEL = "bearing_steel"  # 轴承钢
    SPRING_STEEL = "spring_steel"  # 弹簧钢
    ELECTRICAL_CONTACT = "electrical_contact"  # 电接触材料
    GEAR_STEEL = "gear_steel"  # 齿轮钢
    VALVE_STEEL = "valve_steel"  # 气门钢
    CHAIN_STEEL = "chain_steel"  # 链条钢
    SUPERALLOY = "superalloy"  # 高温合金
    CAST_ALUMINUM = "cast_aluminum"  # 铸造铝合金
    ZINC_ALLOY = "zinc_alloy"  # 锌合金
    TIN_BRONZE = "tin_bronze"  # 锡青铜
    SILICON_BRASS = "silicon_brass"  # 硅黄铜
    WEAR_RESISTANT_IRON = "wear_resistant_iron"  # 耐磨铸铁
    VERMICULAR_IRON = "vermicular_iron"  # 蠕墨铸铁
    MALLEABLE_IRON = "malleable_iron"  # 可锻铸铁
    CAST_MAGNESIUM = "cast_magnesium"  # 铸造镁合金
    STRUCTURAL_CERAMIC = "structural_ceramic"  # 结构陶瓷
    REFRACTORY_METAL = "refractory_metal"  # 难熔金属
    ALUMINUM_BRONZE = "aluminum_bronze"  # 铝青铜
    BERYLLIUM_COPPER = "beryllium_copper"  # 铍铜
    SOLDER = "solder"  # 焊锡
    BRAZING_ALLOY = "brazing_alloy"  # 钎焊合金
    SHAPE_MEMORY_ALLOY = "shape_memory_alloy"  # 形状记忆合金
    BEARING_ALLOY = "bearing_alloy"  # 轴承合金
    THERMOCOUPLE_ALLOY = "thermocouple_alloy"  # 热电偶合金
    PERMANENT_MAGNET = "permanent_magnet"  # 永磁材料
    RESISTANCE_ALLOY = "resistance_alloy"  # 电阻合金
    LOW_EXPANSION_ALLOY = "low_expansion_alloy"  # 低膨胀合金
    SUPERCONDUCTOR = "superconductor"  # 超导材料
    NUCLEAR_MATERIAL = "nuclear_material"  # 核工业材料
    MEDICAL_ALLOY = "medical_alloy"  # 医用合金
    OPTICAL_MATERIAL = "optical_material"  # 光学材料
    BATTERY_MATERIAL = "battery_material"  # 电池材料
    SEMICONDUCTOR = "semiconductor"  # 半导体材料
    THERMAL_INTERFACE = "thermal_interface"  # 热界面材料
    ADDITIVE_MANUFACTURING = "additive_manufacturing"  # 增材制造材料
    HARD_ALLOY = "hard_alloy"  # 硬质合金
    THERMAL_BARRIER = "thermal_barrier"  # 热障涂层材料
    EM_SHIELDING = "em_shielding"  # 电磁屏蔽材料

    # 塑料/橡胶
    FLUOROPOLYMER = "fluoropolymer"  # 氟塑料
    ENGINEERING_PLASTIC = "engineering_plastic"  # 工程塑料
    RUBBER = "rubber"  # 橡胶
    POLYURETHANE = "polyurethane"  # 聚氨酯

    # 玻璃
    BOROSILICATE = "borosilicate"  # 硼硅玻璃
    TEMPERED = "tempered"  # 钢化玻璃

    # 陶瓷
    ALUMINA_SILICATE = "alumina_silicate"  # 硅酸铝

    # 组合件
    WELDED_ASSEMBLY = "welded_assembly"  # 组焊件
    MECHANICAL_ASSEMBLY = "mechanical_assembly"  # 组合件


@dataclass
class MaterialProperties:
    """材料物理和机械属性"""
    # 物理属性
    density: Optional[float] = None  # 密度 g/cm³
    melting_point: Optional[float] = None  # 熔点 ℃
    thermal_conductivity: Optional[float] = None  # 导热系数 W/(m·K)
    conductivity: Optional[float] = None  # 电导率 %IACS

    # 机械属性
    tensile_strength: Optional[float] = None  # 抗拉强度 MPa
    yield_strength: Optional[float] = None  # 屈服强度 MPa
    hardness: Optional[str] = None  # 硬度 (如 HB200, HRC45)
    elongation: Optional[float] = None  # 延伸率 %

    # 加工属性
    machinability: Optional[str] = None  # 可加工性 (excellent/good/fair/poor)
    weldability: Optional[str] = None  # 可焊性 (excellent/good/fair/poor)

    # 成本属性
    cost_tier: Optional[int] = None  # 成本等级 1-5 (1=最低, 5=最高)
    cost_index: Optional[float] = None  # 相对成本指数 (Q235B=1.0)


@dataclass
class ProcessRecommendation:
    """工艺推荐"""
    # 毛坯
    blank_forms: List[str] = field(default_factory=list)  # 毛坯形式
    blank_hint: str = ""  # 毛坯描述

    # 热处理
    heat_treatments: List[str] = field(default_factory=list)  # 推荐热处理
    heat_treatment_notes: List[str] = field(default_factory=list)  # 热处理注意事项
    forbidden_heat_treatments: List[str] = field(default_factory=list)  # 禁止热处理

    # 表面处理
    surface_treatments: List[str] = field(default_factory=list)  # 推荐表面处理
    surface_treatment_notes: List[str] = field(default_factory=list)  # 表面处理注意事项
    forbidden_surface_treatments: List[str] = field(default_factory=list)  # 禁止表面处理

    # 加工参数
    cutting_speed_range: Optional[Tuple[float, float]] = None  # 切削速度范围 m/min
    feed_rate_range: Optional[Tuple[float, float]] = None  # 进给量范围 mm/r
    coolant_required: bool = True  # 是否需要冷却液
    special_tooling: bool = False  # 是否需要特殊刀具

    # 警告和建议
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class MaterialInfo:
    """材料完整信息"""
    # 标识
    grade: str  # 牌号
    name: str  # 名称
    aliases: List[str] = field(default_factory=list)  # 别名

    # 分类
    category: MaterialCategory = MaterialCategory.METAL
    sub_category: MaterialSubCategory = MaterialSubCategory.FERROUS
    group: MaterialGroup = MaterialGroup.CARBON_STEEL

    # 标准
    standards: List[str] = field(default_factory=list)  # 相关标准

    # 属性
    properties: MaterialProperties = field(default_factory=MaterialProperties)

    # 工艺
    process: ProcessRecommendation = field(default_factory=ProcessRecommendation)

    # 备注
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "grade": self.grade,
            "name": self.name,
            "aliases": self.aliases,
            "category": self.category.value,
            "sub_category": self.sub_category.value,
            "group": self.group.value,
            "standards": self.standards,
            "description": self.description,
            "properties": {
                "density": self.properties.density,
                "melting_point": self.properties.melting_point,
                "tensile_strength": self.properties.tensile_strength,
                "yield_strength": self.properties.yield_strength,
                "hardness": self.properties.hardness,
                "machinability": self.properties.machinability,
                "weldability": self.properties.weldability,
            },
            "process": {
                "blank_forms": self.process.blank_forms,
                "blank_hint": self.process.blank_hint,
                "heat_treatments": self.process.heat_treatments,
                "surface_treatments": self.process.surface_treatments,
                "warnings": self.process.warnings,
                "recommendations": self.process.recommendations,
            },
        }


# ============================================================================
# 材料数据库
# ============================================================================

MATERIAL_DATABASE: Dict[str, MaterialInfo] = {
    # -------------------------------------------------------------------------
    # 碳素钢 (Carbon Steel)
    # -------------------------------------------------------------------------
    "Q235B": MaterialInfo(
        grade="Q235B",
        name="普通碳素结构钢",
        aliases=["Q235", "A3钢", "A3"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CARBON_STEEL,
        standards=["GB/T 700-2006"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1500,
            thermal_conductivity=50,
            tensile_strength=370,
            yield_strength=235,
            elongation=26,  # %
            hardness="HB120-140",
            machinability="good",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "型材", "棒材", "管材"],
            blank_hint="热轧板材/型材",
            heat_treatments=["正火", "退火"],
            surface_treatments=["镀锌", "喷漆", "发黑", "磷化"],
            cutting_speed_range=(80, 150),
            recommendations=["适合焊接结构件", "可镀锌防腐"],
        ),
        description="通用结构钢，焊接性好，价格低廉",
    ),

    "Q345R": MaterialInfo(
        grade="Q345R",
        name="压力容器用钢",
        aliases=["16MnR", "Q345"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CARBON_STEEL,
        standards=["GB/T 713-2014"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1500,
            thermal_conductivity=48,
            tensile_strength=510,
            yield_strength=345,
            elongation=21,  # %
            hardness="HB150-180",
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "锻件"],
            blank_hint="正火态板材",
            heat_treatments=["正火", "正火+回火", "调质"],
            surface_treatments=["喷漆", "防腐涂层"],
            warnings=["焊接后需消除应力"],
            recommendations=["压力容器必须正火态交货", "焊接需预热"],
        ),
        description="压力容器专用钢，韧性好",
    ),

    "45": MaterialInfo(
        grade="45",
        name="优质碳素结构钢",
        aliases=["45#", "45钢", "S45C", "C45"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CARBON_STEEL,
        standards=["GB/T 699-2015"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1495,
            thermal_conductivity=48,
            tensile_strength=600,
            yield_strength=355,
            elongation=16,  # %
            hardness="HB170-220",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件"],
            blank_hint="锻造/圆钢",
            heat_treatments=["调质", "淬火+回火", "正火", "表面淬火"],
            surface_treatments=["镀铬", "发黑", "氮化"],
            heat_treatment_notes=["调质后硬度HB220-250", "表面淬火可达HRC50-55"],
            cutting_speed_range=(60, 120),
            recommendations=["轴类零件首选", "调质后加工性好"],
        ),
        description="中碳钢，综合力学性能好，轴类零件常用",
    ),

    "20": MaterialInfo(
        grade="20",
        name="低碳钢",
        aliases=["20#", "20钢"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CARBON_STEEL,
        standards=["GB/T 699-2015"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1510,
            thermal_conductivity=51.9,
            tensile_strength=410,
            yield_strength=245,
            elongation=25,  # %
            hardness="HB120-160",
            machinability="excellent",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "管材", "板材"],
            blank_hint="棒材/管材",
            heat_treatments=["渗碳淬火", "正火", "退火"],
            surface_treatments=["渗碳", "镀锌", "发黑"],
            heat_treatment_notes=["渗碳层深0.5-1.5mm", "渗碳淬火表面硬度HRC58-62"],
            recommendations=["适合渗碳件", "小模数齿轮常用"],
        ),
        description="低碳钢，适合渗碳处理",
    ),

    "10": MaterialInfo(
        grade="10",
        name="低碳钢",
        aliases=["10#", "10钢", "C10"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CARBON_STEEL,
        standards=["GB/T 699-2015"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1525,
            thermal_conductivity=54.6,
            tensile_strength=335,
            yield_strength=205,
            elongation=31,  # %
            hardness="HB100-130",
            machinability="excellent",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "管材", "板材", "线材"],
            blank_hint="冷拔棒材/管材",
            heat_treatments=["渗碳淬火", "正火", "退火"],
            surface_treatments=["渗碳", "镀锌", "发黑"],
            heat_treatment_notes=["渗碳层深0.3-1.0mm", "渗碳淬火后表面硬度HRC56-60"],
            recommendations=["适合冷镦件", "适合渗碳件", "适合焊接件"],
        ),
        description="低碳钢，塑性和焊接性极好",
    ),

    "15": MaterialInfo(
        grade="15",
        name="低碳钢",
        aliases=["15#", "15钢", "C15"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CARBON_STEEL,
        standards=["GB/T 699-2015"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1520,
            thermal_conductivity=52.7,
            tensile_strength=375,
            yield_strength=225,
            elongation=27,  # %
            hardness="HB110-140",
            machinability="excellent",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "管材", "板材"],
            blank_hint="棒材/管材",
            heat_treatments=["渗碳淬火", "正火", "退火"],
            surface_treatments=["渗碳", "镀锌", "发黑", "氰化"],
            heat_treatment_notes=["渗碳层深0.5-1.2mm", "渗碳淬火后表面硬度HRC58-62"],
            recommendations=["适合渗碳件", "轻载齿轮常用", "销轴类零件"],
        ),
        description="低碳钢，强度略高于10钢",
    ),

    "35": MaterialInfo(
        grade="35",
        name="中碳钢",
        aliases=["35#", "35钢", "S35C", "C35"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CARBON_STEEL,
        standards=["GB/T 699-2015"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1495,
            thermal_conductivity=49,
            tensile_strength=530,
            yield_strength=315,
            elongation=20,  # %
            hardness="HB150-190",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件"],
            blank_hint="棒材/锻件",
            heat_treatments=["调质", "正火", "淬火+回火"],
            surface_treatments=["发黑", "镀铬", "氮化"],
            heat_treatment_notes=["调质后硬度HB200-230", "表面淬火可达HRC45-50"],
            cutting_speed_range=(60, 120),
            recommendations=["受力不大的轴类", "连杆、螺栓等"],
        ),
        description="中碳钢，强度中等，塑性良好",
    ),

    "50": MaterialInfo(
        grade="50",
        name="中碳钢",
        aliases=["50#", "50钢", "S50C", "C50"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CARBON_STEEL,
        standards=["GB/T 699-2015"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1480,
            thermal_conductivity=47,
            tensile_strength=630,
            yield_strength=375,
            elongation=14,  # %
            hardness="HB180-230",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件"],
            blank_hint="棒材/锻件",
            heat_treatments=["调质", "淬火+回火", "表面淬火"],
            surface_treatments=["发黑", "镀铬", "氮化"],
            heat_treatment_notes=["调质后硬度HB230-260", "表面淬火可达HRC52-58"],
            cutting_speed_range=(50, 100),
            warnings=["焊接性较差，需预热"],
            recommendations=["高强度轴类", "齿轮、联轴器等"],
        ),
        description="中碳钢，强度较高，韧性中等",
    ),

    "65Mn": MaterialInfo(
        grade="65Mn",
        name="弹簧钢",
        aliases=["65锰", "弹簧钢"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CARBON_STEEL,
        standards=["GB/T 1222-2016"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1470,
            thermal_conductivity=44,
            tensile_strength=980,
            yield_strength=785,
            hardness="HRC42-50",
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["带材", "棒材", "线材"],
            blank_hint="冷轧带材/线材",
            heat_treatments=["淬火+中温回火", "等温淬火"],
            surface_treatments=["发黑", "磷化", "喷丸"],
            heat_treatment_notes=["淬火温度820-840℃", "回火温度400-500℃"],
            warnings=["焊接性差", "淬火变形大"],
            recommendations=["弹簧件专用", "需要喷丸强化"],
        ),
        description="弹簧钢，弹性极限高",
    ),

    # -------------------------------------------------------------------------
    # 合金钢 (Alloy Steel)
    # -------------------------------------------------------------------------
    "40Cr": MaterialInfo(
        grade="40Cr",
        name="铬钢",
        aliases=["40铬", "SCr440", "5140"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["GB/T 3077-2015"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1480,
            thermal_conductivity=42,
            tensile_strength=980,
            yield_strength=785,
            elongation=9,  # %
            hardness="HB207-255",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件"],
            blank_hint="锻造",
            heat_treatments=["调质", "淬火+回火", "表面淬火", "氮化"],
            surface_treatments=["镀铬", "氮化", "发黑"],
            heat_treatment_notes=["调质后HB260-300", "表面淬火HRC50-55"],
            cutting_speed_range=(50, 100),
            recommendations=["中等载荷轴类", "调质后使用"],
        ),
        description="通用调质钢，淬透性好",
    ),

    "42CrMo": MaterialInfo(
        grade="42CrMo",
        name="铬钼钢",
        aliases=["42CrMo4", "SCM440", "4140"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["GB/T 3077-2015"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1470,
            thermal_conductivity=40,
            tensile_strength=1080,
            yield_strength=930,
            elongation=12,  # %
            hardness="HB217-269",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["锻件", "棒材"],
            blank_hint="锻造",
            heat_treatments=["调质", "淬火+回火", "渗氮"],
            surface_treatments=["渗氮", "镀铬", "发黑"],
            heat_treatment_notes=["调质后HB280-320", "淬透性优于40Cr"],
            warnings=["焊接需预热200℃以上"],
            recommendations=["高强度轴类", "大截面零件"],
        ),
        description="高强度调质钢，大截面淬透性好",
    ),

    "GCr15": MaterialInfo(
        grade="GCr15",
        name="轴承钢",
        aliases=["SUJ2", "52100", "100Cr6"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["GB/T 18254-2016"],
        properties=MaterialProperties(
            density=7.81,
            melting_point=1450,
            thermal_conductivity=46.6,
            tensile_strength=None,
            yield_strength=None,
            hardness="HRC60-65",
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "管材"],
            blank_hint="球化退火态棒材",
            heat_treatments=["淬火+低温回火", "球化退火"],
            surface_treatments=["抛光", "超精研"],
            heat_treatment_notes=["淬火温度840℃", "回火温度150-180℃"],
            warnings=["不可焊接", "需要超精加工"],
            special_tooling=True,
            recommendations=["轴承套圈专用", "需要高精度磨削"],
        ),
        description="轴承钢，高硬度高耐磨",
    ),

    "GCr15SiMn": MaterialInfo(
        grade="GCr15SiMn",
        name="硅锰轴承钢",
        aliases=["SUJ3", "52100 Mod", "100CrMnSi6-4"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["GB/T 18254-2016"],
        properties=MaterialProperties(
            density=7.80,
            melting_point=1450,
            thermal_conductivity=40,
            tensile_strength=None,
            yield_strength=None,
            hardness="HRC60-65",
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "管材", "锻件"],
            blank_hint="球化退火态棒材",
            heat_treatments=["淬火+低温回火", "球化退火", "贝氏体等温淬火"],
            heat_treatment_notes=["淬火温度850-870℃", "回火温度160-180℃"],
            surface_treatments=["抛光", "超精研", "氮化"],
            special_tooling=True,
            warnings=["不可焊接", "需要超精加工", "淬透性优于GCr15"],
            recommendations=["大型轴承", "特大型轴承套圈", "重载轴承"],
        ),
        description="高淬透性轴承钢，大截面轴承用",
    ),

    "GCr18Mo": MaterialInfo(
        grade="GCr18Mo",
        name="铬钼轴承钢",
        aliases=["SUJ5", "100CrMo7-3", "A485 Gr.3"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["GB/T 18254-2016", "ASTM A485"],
        properties=MaterialProperties(
            density=7.83,
            melting_point=1450,
            thermal_conductivity=38,
            tensile_strength=2250,  # MPa, 淬火回火后
            yield_strength=1900,  # MPa, 淬火回火后
            hardness="HRC60-65",
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "管材", "锻件"],
            blank_hint="球化退火态棒材/锻件",
            heat_treatments=["淬火+低温回火", "球化退火", "深冷处理"],
            heat_treatment_notes=["淬火温度850-880℃", "回火温度180-200℃", "深冷-70℃"],
            surface_treatments=["抛光", "超精研"],
            special_tooling=True,
            warnings=["不可焊接", "高温尺寸稳定性好", "成本较高"],
            recommendations=["航空轴承", "高温轴承", "精密主轴轴承"],
        ),
        description="航空级轴承钢，高温尺寸稳定性好",
    ),

    "20CrMnTi": MaterialInfo(
        grade="20CrMnTi",
        name="渗碳钢",
        aliases=["20CrMnTiH", "SCM420H"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["GB/T 3077-2015"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1470,
            thermal_conductivity=37,
            tensile_strength=1080,
            yield_strength=835,
            elongation=10,  # %
            hardness="HRC58-62",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件"],
            blank_hint="锻件/棒材",
            heat_treatments=["渗碳淬火+低温回火", "正火"],
            surface_treatments=["渗碳", "喷丸"],
            heat_treatment_notes=["渗碳温度920℃", "渗碳层深0.8-1.2mm", "淬火850℃油冷"],
            recommendations=["齿轮首选", "轴类零件", "渗碳件"],
        ),
        description="渗碳钢，齿轮常用材料",
    ),

    "20Cr": MaterialInfo(
        grade="20Cr",
        name="渗碳钢",
        aliases=["5120", "SCr420", "20Cr4"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["GB/T 3077-2015"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1470,
            thermal_conductivity=38,
            tensile_strength=835,
            yield_strength=540,
            elongation=15,  # %
            hardness="HRC56-62",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件"],
            blank_hint="棒材/锻件",
            heat_treatments=["渗碳淬火+低温回火", "正火"],
            surface_treatments=["渗碳", "氰化"],
            heat_treatment_notes=["渗碳温度900-920℃", "渗碳层深0.5-1.0mm"],
            recommendations=["小模数齿轮", "销轴", "活塞销"],
        ),
        description="渗碳钢，小型渗碳件常用",
    ),

    "38CrMoAl": MaterialInfo(
        grade="38CrMoAl",
        name="氮化钢",
        aliases=["38CrMoAlA", "SACM645"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["GB/T 3077-2015"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1480,
            thermal_conductivity=33,
            tensile_strength=980,
            yield_strength=835,
            elongation=10,  # %
            hardness="HV850-1000",
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件"],
            blank_hint="调质态棒材",
            heat_treatments=["调质", "气体氮化", "离子氮化"],
            surface_treatments=["氮化", "软氮化"],
            heat_treatment_notes=["调质硬度HB260-300", "氮化温度500-560℃", "氮化层深0.3-0.6mm"],
            warnings=["氮化前必须调质", "氮化时间长"],
            recommendations=["丝杠", "主轴", "精密量具", "耐磨件"],
        ),
        description="氮化钢，表面硬度极高",
    ),

    "30CrMnSi": MaterialInfo(
        grade="30CrMnSi",
        name="高强度结构钢",
        aliases=["30CrMnSiA", "AISI 8630"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["GB/T 3077-2015"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1485,
            thermal_conductivity=35,
            tensile_strength=1080,
            yield_strength=885,
            elongation=11,  # %
            hardness="HB270-320",
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "锻件"],
            blank_hint="锻件/棒材",
            heat_treatments=["调质", "淬火+中温回火"],
            surface_treatments=["发黑", "镀铬"],
            heat_treatment_notes=["淬火温度880℃油冷", "回火温度500-550℃"],
            warnings=["焊接需预热200-300℃", "焊后需热处理"],
            recommendations=["航空结构件", "高强度螺栓", "重载轴"],
        ),
        description="高强度结构钢，航空航天常用",
    ),

    # 弹簧钢
    "65Mn": MaterialInfo(
        grade="65Mn",
        name="锰钢弹簧钢",
        aliases=["65Mn弹簧钢"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["GB/T 1222-2016"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1470,
            thermal_conductivity=44,
            tensile_strength=980,
            yield_strength=785,
            hardness="HRC42-50",
            elongation=8,
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "带材", "线材"],
            blank_hint="带材/线材",
            heat_treatments=["淬火+中温回火", "等温淬火"],
            heat_treatment_notes=["淬火830℃油冷", "回火400-500℃"],
            surface_treatments=["发蓝", "磷化", "喷丸"],
            warnings=["淬透性有限", "大截面不适用"],
            recommendations=["汽车板簧", "卷簧", "垫圈"],
        ),
        description="常用弹簧钢，经济实用",
    ),

    "60Si2Mn": MaterialInfo(
        grade="60Si2Mn",
        name="硅锰弹簧钢",
        aliases=["60Si2MnA", "SUP7"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["GB/T 1222-2016"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1470,
            thermal_conductivity=30,
            tensile_strength=1470,
            yield_strength=1275,
            hardness="HRC48-52",
            elongation=6,
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "扁钢"],
            blank_hint="扁钢/棒材",
            heat_treatments=["淬火+中温回火"],
            heat_treatment_notes=["淬火860℃油冷", "回火450-500℃"],
            surface_treatments=["喷丸强化", "发蓝"],
            warnings=["脱碳敏感", "需控制气氛"],
            recommendations=["重载弹簧", "汽车悬架弹簧", "铁路弹簧"],
        ),
        description="高强度弹簧钢，疲劳性能好",
    ),

    "50CrVA": MaterialInfo(
        grade="50CrVA",
        name="铬钒弹簧钢",
        aliases=["50CrV4", "SUP10"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["GB/T 1222-2016"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1475,
            thermal_conductivity=34,
            tensile_strength=1470,
            yield_strength=1320,
            hardness="HRC48-52",
            elongation=7,
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "线材"],
            blank_hint="棒材/线材",
            heat_treatments=["淬火+中温回火"],
            heat_treatment_notes=["淬火850℃油冷", "回火450℃"],
            surface_treatments=["喷丸强化", "发蓝", "磷化"],
            recommendations=["高应力弹簧", "阀门弹簧", "离合器弹簧"],
        ),
        description="高级弹簧钢，抗疲劳性优异",
    ),

    # 工具钢
    "Cr12MoV": MaterialInfo(
        grade="Cr12MoV",
        name="冷作模具钢",
        aliases=["D2", "SKD11", "1.2379"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.TOOL_STEEL,
        standards=["GB/T 1299-2014"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1420,
            thermal_conductivity=20,
            tensile_strength=1900,  # MPa, 淬火回火后
            yield_strength=1600,  # MPa, 淬火回火后
            hardness="HRC58-62",
            machinability="poor",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "锻件"],
            blank_hint="锻件/模块",
            heat_treatments=["淬火+低温回火", "真空淬火"],
            heat_treatment_notes=["淬火1020-1050℃", "回火200-250℃"],
            surface_treatments=["TD处理", "PVD涂层", "氮化"],
            special_tooling=True,
            warnings=["脆性大", "需预热加工", "禁止快速加热"],
            recommendations=["冲裁模", "冷挤压模", "拉丝模"],
        ),
        description="高耐磨冷作模具钢",
    ),

    "H13": MaterialInfo(
        grade="H13",
        name="热作模具钢",
        aliases=["SKD61", "4Cr5MoSiV1", "1.2344"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.TOOL_STEEL,
        standards=["GB/T 1299-2014"],
        properties=MaterialProperties(
            density=7.80,
            melting_point=1430,
            thermal_conductivity=24.6,
            tensile_strength=1960,
            yield_strength=1520,
            elongation=10,  # %
            hardness="HRC48-52",
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件"],
            blank_hint="锻件",
            heat_treatments=["淬火+高温回火"],
            heat_treatment_notes=["淬火1020℃气冷", "回火550-600℃两次"],
            surface_treatments=["氮化", "TD处理", "PVD涂层"],
            special_tooling=True,
            warnings=["需预热加工"],
            recommendations=["压铸模", "热锻模", "挤压模"],
        ),
        description="通用热作模具钢，热疲劳性能好",
    ),

    "W18Cr4V": MaterialInfo(
        grade="W18Cr4V",
        name="高速钢",
        aliases=["T1", "SKH2", "1.3355"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.TOOL_STEEL,
        standards=["GB/T 9943-2008"],
        properties=MaterialProperties(
            density=8.70,
            melting_point=1350,
            thermal_conductivity=24,
            tensile_strength=2200,  # MPa, 淬火三次回火后
            yield_strength=1900,  # MPa, 淬火三次回火后
            hardness="HRC63-66",
            machinability="poor",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材"],
            blank_hint="棒材",
            heat_treatments=["淬火+三次回火"],
            heat_treatment_notes=["淬火1260-1280℃", "回火560℃三次"],
            surface_treatments=["TiN涂层", "氮化"],
            special_tooling=True,
            warnings=["淬火温度高", "需盐浴加热"],
            recommendations=["车刀", "钻头", "铣刀", "齿轮刀具"],
        ),
        description="经典高速钢，红硬性好",
    ),

    "W6Mo5Cr4V2": MaterialInfo(
        grade="W6Mo5Cr4V2",
        name="高性能高速钢",
        aliases=["M2", "SKH51", "1.3343"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.TOOL_STEEL,
        standards=["GB/T 9943-2008"],
        properties=MaterialProperties(
            density=8.16,
            melting_point=1430,
            thermal_conductivity=19,
            tensile_strength=2300,  # MPa, 淬火三次回火后
            yield_strength=2000,  # MPa, 淬火三次回火后
            hardness="HRC63-66",
            machinability="poor",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "线材"],
            blank_hint="棒材",
            heat_treatments=["淬火+三次回火"],
            heat_treatment_notes=["淬火1210-1230℃", "回火560℃三次"],
            surface_treatments=["TiN涂层", "TiAlN涂层", "氮化"],
            special_tooling=True,
            recommendations=["钻头", "丝锥", "铰刀", "拉刀"],
        ),
        description="通用高速钢，综合性能好",
    ),

    # 特殊钢材 (Special Steels)
    "9Cr18": MaterialInfo(
        grade="9Cr18",
        name="高碳马氏体不锈钢",
        aliases=["440C", "SUS440C", "1.4125", "9Cr18Mo", "9Cr18MoV"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.TOOL_STEEL,
        standards=["GB/T 1220-2007", "ASTM A276"],
        properties=MaterialProperties(
            density=7.75,
            melting_point=1480,
            thermal_conductivity=24.2,
            tensile_strength=760,
            yield_strength=450,
            elongation=2,  # %
            hardness="HRC56-60",
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "锻件"],
            blank_hint="棒材/锻件",
            heat_treatments=["淬火+低温回火"],
            heat_treatment_notes=["淬火1040-1070℃油冷", "回火150-200℃"],
            surface_treatments=["抛光", "镜面抛光", "钝化"],
            special_tooling=True,
            warnings=["高碳易脱碳", "需保护气氛热处理", "焊接性差"],
            recommendations=["高档刀具", "轴承", "喷嘴", "阀座"],
        ),
        description="高碳马氏体不锈钢，高硬度耐磨耐蚀，刀具钢",
    ),

    "12Cr1MoV": MaterialInfo(
        grade="12Cr1MoV",
        name="耐热钢",
        aliases=["12Cr1MoVG", "15CrMo", "13CrMo44"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["GB/T 5310-2017", "GB/T 3077-2015"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1500,
            thermal_conductivity=35,
            tensile_strength=490,
            yield_strength=275,
            elongation=20,  # %
            hardness="HB156-207",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["管材", "板材", "锻件"],
            blank_hint="无缝管/板材",
            heat_treatments=["正火+回火", "淬火+回火"],
            heat_treatment_notes=["正火980℃", "回火720-750℃"],
            surface_treatments=["除锈涂漆", "热喷涂"],
            warnings=["焊接需预热", "焊后需热处理"],
            recommendations=["锅炉过热器管", "高温高压容器", "汽轮机部件"],
        ),
        description="珠光体耐热钢，550℃以下长期使用",
    ),

    "Mn13": MaterialInfo(
        grade="Mn13",
        name="高锰耐磨钢",
        aliases=["ZGMn13", "Mn13Cr2", "Hadfield钢", "X120Mn12"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["GB/T 5680-2010"],
        properties=MaterialProperties(
            density=7.90,
            melting_point=1350,
            thermal_conductivity=14.5,
            tensile_strength=830,
            yield_strength=350,
            elongation=45,  # %
            hardness="HB200-250 (加工硬化后HRC50-55)",
            machinability="poor",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件"],
            blank_hint="铸件",
            heat_treatments=["水韧处理"],
            heat_treatment_notes=["1050-1100℃保温后快速水冷"],
            surface_treatments=["无"],
            special_tooling=True,
            warnings=["只能铸造不能锻造", "切削加工极困难", "冲击下表面硬化"],
            recommendations=["破碎机锤头", "衬板", "挖掘机齿", "铁路道岔"],
        ),
        description="奥氏体高锰钢，冲击磨损下自硬化，耐磨性极好",
    ),

    # -------------------------------------------------------------------------
    # 不锈钢 (Stainless Steel)
    # -------------------------------------------------------------------------
    "S30408": MaterialInfo(
        grade="S30408",
        name="奥氏体不锈钢",
        aliases=["304", "0Cr18Ni9", "1.4301", "SUS304"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.STAINLESS_STEEL,
        standards=["GB/T 24511-2017"],
        properties=MaterialProperties(
            density=7.93,
            melting_point=1450,
            thermal_conductivity=16,
            tensile_strength=520,
            yield_strength=205,
            elongation=40,  # %
            hardness="HB187",
            machinability="fair",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材", "锻件"],
            blank_hint="板材/棒材",
            heat_treatments=["固溶处理"],
            forbidden_heat_treatments=["淬火", "渗碳"],
            surface_treatments=["钝化", "酸洗", "抛光", "电解抛光"],
            surface_treatment_notes=["必须钝化处理提高耐蚀性"],
            cutting_speed_range=(40, 80),
            coolant_required=True,
            warnings=["切削时易加工硬化", "不可渗碳"],
            recommendations=["焊接后需固溶或钝化", "切削需要充分冷却"],
        ),
        description="通用奥氏体不锈钢，耐蚀性好",
    ),

    "S31603": MaterialInfo(
        grade="S31603",
        name="低碳奥氏体不锈钢",
        aliases=["316L", "00Cr17Ni14Mo2", "1.4404", "SUS316L"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.STAINLESS_STEEL,
        standards=["GB/T 24511-2017"],
        properties=MaterialProperties(
            density=7.98,
            melting_point=1440,
            thermal_conductivity=16,
            tensile_strength=480,
            yield_strength=170,
            elongation=40,  # %
            hardness="HB187",
            machinability="fair",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材", "锻件"],
            blank_hint="板材/棒材",
            heat_treatments=["固溶处理"],
            forbidden_heat_treatments=["淬火", "渗碳"],
            surface_treatments=["钝化", "酸洗", "抛光", "电解抛光"],
            cutting_speed_range=(35, 70),
            coolant_required=True,
            warnings=["切削时易加工硬化"],
            recommendations=["耐氯化物腐蚀", "医疗/食品行业首选"],
        ),
        description="耐蚀性优于304，含钼，低碳抗晶间腐蚀",
    ),

    "2Cr13": MaterialInfo(
        grade="2Cr13",
        name="马氏体不锈钢",
        aliases=["420", "SUS420J1", "1.4021"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.STAINLESS_STEEL,
        standards=["GB/T 1220-2007"],
        properties=MaterialProperties(
            density=7.75,
            melting_point=1530,
            thermal_conductivity=24.9,
            tensile_strength=640,
            yield_strength=440,
            elongation=15,  # %
            hardness="HRC45-52",
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材"],
            blank_hint="退火态棒材",
            heat_treatments=["淬火+回火", "退火"],
            surface_treatments=["抛光", "镀铬", "钝化"],
            heat_treatment_notes=["淬火温度980-1050℃", "回火200-600℃"],
            warnings=["焊接性差", "淬火前需预热"],
            recommendations=["刀具/阀门零件", "需要硬度和耐蚀性兼顾"],
        ),
        description="马氏体不锈钢，可淬火强化",
    ),

    "20Cr13": MaterialInfo(
        grade="20Cr13",
        name="马氏体不锈钢",
        aliases=["420J2", "SUS420J2"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.STAINLESS_STEEL,
        standards=["GB/T 1220-2007"],
        properties=MaterialProperties(
            density=7.75,
            melting_point=1530,
            thermal_conductivity=25.2,
            tensile_strength=690,
            yield_strength=490,
            elongation=15,  # %
            hardness="HRC50-55",
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材"],
            blank_hint="退火态棒材",
            heat_treatments=["淬火+回火"],
            surface_treatments=["抛光", "钝化"],
            heat_treatment_notes=["淬火温度1000-1050℃"],
            warnings=["焊接性差"],
            recommendations=["刀具/阀芯", "比2Cr13硬度更高"],
        ),
        description="高碳马氏体不锈钢，硬度更高",
    ),

    "321": MaterialInfo(
        grade="321",
        name="钛稳定奥氏体不锈钢",
        aliases=["S32100", "SUS321", "1.4541", "0Cr18Ni10Ti", "1Cr18Ni9Ti"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.STAINLESS_STEEL,
        standards=["GB/T 1220-2007", "ASTM A240"],
        properties=MaterialProperties(
            density=7.9,
            melting_point=1400,
            thermal_conductivity=16.1,
            tensile_strength=520,
            yield_strength=205,
            elongation=40,  # %
            hardness="HB150-190",
            machinability="fair",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材"],
            blank_hint="固溶态板材/管材",
            heat_treatments=["固溶处理", "稳定化处理"],
            forbidden_heat_treatments=["淬火", "渗碳"],
            surface_treatments=["钝化", "酸洗", "抛光"],
            heat_treatment_notes=["固溶温度1000-1100℃", "稳定化处理850-930℃"],
            cutting_speed_range=(35, 70),
            coolant_required=True,
            warnings=["切削时易加工硬化"],
            recommendations=["高温抗晶间腐蚀", "焊接结构首选", "430-900℃长期使用"],
        ),
        description="钛稳定化奥氏体不锈钢，抗晶间腐蚀性优异",
    ),

    "347": MaterialInfo(
        grade="347",
        name="铌稳定奥氏体不锈钢",
        aliases=["S34700", "SUS347", "1.4550", "0Cr18Ni11Nb"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.STAINLESS_STEEL,
        standards=["GB/T 1220-2007", "ASTM A240"],
        properties=MaterialProperties(
            density=7.9,
            melting_point=1400,
            thermal_conductivity=16.2,
            tensile_strength=515,
            yield_strength=205,
            elongation=40,  # %
            hardness="HB150-190",
            machinability="fair",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材"],
            blank_hint="固溶态板材/管材",
            heat_treatments=["固溶处理", "稳定化处理"],
            forbidden_heat_treatments=["淬火", "渗碳"],
            surface_treatments=["钝化", "酸洗", "抛光"],
            heat_treatment_notes=["固溶温度1000-1100℃", "稳定化处理850-950℃"],
            cutting_speed_range=(35, 70),
            coolant_required=True,
            warnings=["切削时易加工硬化"],
            recommendations=["高温抗晶间腐蚀", "比321高温强度更好", "航空/核工业"],
        ),
        description="铌稳定化奥氏体不锈钢，高温性能优于321",
    ),

    "430": MaterialInfo(
        grade="430",
        name="铁素体不锈钢",
        aliases=["S43000", "SUS430", "1.4016", "1Cr17"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.STAINLESS_STEEL,
        standards=["GB/T 1220-2007", "ASTM A240"],
        properties=MaterialProperties(
            density=7.7,
            melting_point=1425,
            thermal_conductivity=26.1,
            tensile_strength=450,
            yield_strength=205,
            elongation=22,  # %
            hardness="HB150-180",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材"],
            blank_hint="退火态板材",
            heat_treatments=["退火"],
            forbidden_heat_treatments=["淬火"],
            surface_treatments=["抛光", "钝化", "拉丝"],
            heat_treatment_notes=["退火温度760-830℃", "空冷或缓冷"],
            cutting_speed_range=(50, 90),
            warnings=["焊接后需退火", "低温韧性差", "475℃脆性"],
            recommendations=["家电面板", "厨具", "建筑装饰", "成本低于304"],
        ),
        description="铁素体不锈钢，耐蚀性中等，成本低",
    ),

    "410": MaterialInfo(
        grade="410",
        name="马氏体不锈钢",
        aliases=["S41000", "SUS410", "1.4006", "1Cr13"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.STAINLESS_STEEL,
        standards=["GB/T 1220-2007", "ASTM A240"],
        properties=MaterialProperties(
            density=7.75,
            melting_point=1480,
            thermal_conductivity=24.9,
            tensile_strength=550,
            yield_strength=345,
            elongation=20,  # %
            hardness="HRC38-45",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "锻件"],
            blank_hint="退火态棒材",
            heat_treatments=["淬火+回火", "退火", "正火"],
            surface_treatments=["抛光", "镀铬", "钝化"],
            heat_treatment_notes=["淬火温度925-1010℃", "回火150-370℃"],
            cutting_speed_range=(45, 85),
            warnings=["焊接后需热处理", "预热和缓冷"],
            recommendations=["阀门", "泵轴", "紧固件", "刀具"],
        ),
        description="通用马氏体不锈钢，可淬火强化",
    ),

    "17-4PH": MaterialInfo(
        grade="17-4PH",
        name="沉淀硬化不锈钢",
        aliases=["S17400", "SUS630", "1.4542", "0Cr17Ni4Cu4Nb", "630"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.STAINLESS_STEEL,
        standards=["ASTM A564", "GB/T 1220-2007"],
        properties=MaterialProperties(
            density=7.78,
            melting_point=1400,
            thermal_conductivity=18.3,
            tensile_strength=1310,
            yield_strength=1170,
            elongation=10,  # %
            hardness="HRC40-45",
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "锻件"],
            blank_hint="固溶态棒材",
            heat_treatments=["固溶处理", "时效处理", "H900", "H1025", "H1150"],
            surface_treatments=["钝化", "抛光"],
            heat_treatment_notes=["固溶温度1040℃", "H900时效480℃/1h", "H1025时效550℃/4h"],
            cutting_speed_range=(30, 60),
            coolant_required=True,
            warnings=["加工硬化倾向大", "时效后精加工"],
            recommendations=["航空结构件", "阀门阀杆", "高强度紧固件", "模具"],
        ),
        description="沉淀硬化不锈钢，高强度高硬度，耐蚀性好",
    ),

    "904L": MaterialInfo(
        grade="904L",
        name="超级奥氏体不锈钢",
        aliases=["N08904", "SUS890L", "1.4539", "00Cr20Ni25Mo4.5Cu"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.STAINLESS_STEEL,
        standards=["ASTM B625", "GB/T 1220-2007"],
        properties=MaterialProperties(
            density=8.0,
            melting_point=1350,
            thermal_conductivity=12.5,
            tensile_strength=490,
            yield_strength=220,
            elongation=35,  # %
            hardness="HB150-180",
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材"],
            blank_hint="固溶态板材/管材",
            heat_treatments=["固溶处理"],
            forbidden_heat_treatments=["淬火"],
            surface_treatments=["钝化", "酸洗", "电解抛光"],
            heat_treatment_notes=["固溶温度1100-1150℃", "水冷"],
            cutting_speed_range=(25, 50),
            coolant_required=True,
            warnings=["切削时易加工硬化", "价格较高"],
            recommendations=["硫酸环境", "磷酸环境", "海水淡化", "造纸漂白"],
        ),
        description="超级奥氏体不锈钢，耐硫酸腐蚀性极强",
    ),

    "254SMO": MaterialInfo(
        grade="254SMO",
        name="超级奥氏体不锈钢",
        aliases=["S31254", "1.4547", "00Cr20Ni18Mo6CuN"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.STAINLESS_STEEL,
        standards=["ASTM A240"],
        properties=MaterialProperties(
            density=8.0,
            melting_point=1340,
            thermal_conductivity=13.5,
            tensile_strength=650,
            yield_strength=310,
            elongation=40,  # %
            hardness="HB180-220",
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材"],
            blank_hint="固溶态板材/管材",
            heat_treatments=["固溶处理"],
            forbidden_heat_treatments=["淬火"],
            surface_treatments=["钝化", "酸洗"],
            heat_treatment_notes=["固溶温度1150-1200℃", "快冷"],
            cutting_speed_range=(20, 45),
            coolant_required=True,
            warnings=["加工硬化严重", "需要锋利刀具"],
            recommendations=["海水系统", "脱硫设备", "氯化物环境", "6%Mo耐点蚀"],
        ),
        description="6钼超级奥氏体不锈钢，PREN>40，海水级",
    ),

    "316Ti": MaterialInfo(
        grade="316Ti",
        name="钛稳定奥氏体不锈钢",
        aliases=["S31635", "SUS316Ti", "1.4571", "0Cr17Ni12Mo2Ti"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.STAINLESS_STEEL,
        standards=["GB/T 1220-2007", "EN 10088"],
        properties=MaterialProperties(
            density=8.0,
            melting_point=1400,
            thermal_conductivity=14.6,
            tensile_strength=520,
            yield_strength=220,
            elongation=35,  # %
            hardness="HB150-190",
            machinability="fair",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材"],
            blank_hint="固溶态板材/管材",
            heat_treatments=["固溶处理", "稳定化处理"],
            forbidden_heat_treatments=["淬火"],
            surface_treatments=["钝化", "酸洗", "抛光"],
            heat_treatment_notes=["固溶温度1020-1100℃", "稳定化处理850-930℃"],
            cutting_speed_range=(30, 60),
            coolant_required=True,
            warnings=["切削易加工硬化"],
            recommendations=["高温焊接结构", "化工设备", "比316L更好的高温性能"],
        ),
        description="钛稳定316型不锈钢，高温抗晶间腐蚀",
    ),

    # -------------------------------------------------------------------------
    # 耐蚀合金 (Corrosion Resistant Alloy)
    # -------------------------------------------------------------------------
    "C276": MaterialInfo(
        grade="C276",
        name="哈氏合金C-276",
        aliases=["Hastelloy C-276", "N10276", "C276Ⅱ"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CORROSION_RESISTANT,
        standards=["ASTM B575"],
        properties=MaterialProperties(
            density=8.89,
            melting_point=1370,
            thermal_conductivity=10.2,
            tensile_strength=690,
            yield_strength=310,
            elongation=60,  # %
            hardness="HB200",
            machinability="poor",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材", "锻件"],
            blank_hint="固溶态板材/锻件",
            heat_treatments=["固溶处理"],
            surface_treatments=["酸洗", "钝化"],
            cutting_speed_range=(15, 30),
            special_tooling=True,
            coolant_required=True,
            warnings=["切削难度大", "刀具磨损快", "需要低速高进给"],
            recommendations=["强酸/强碱环境", "氯化物环境首选"],
        ),
        description="镍基耐蚀合金，耐强酸强碱",
    ),

    "C22": MaterialInfo(
        grade="C22",
        name="哈氏合金C-22",
        aliases=["Hastelloy C-22", "N06022", "C22Ⅱ"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CORROSION_RESISTANT,
        standards=["ASTM B575"],
        properties=MaterialProperties(
            density=8.69,
            melting_point=1357,
            thermal_conductivity=10.1,
            tensile_strength=690,
            yield_strength=310,
            elongation=55,  # %
            hardness="HB200",
            machinability="poor",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材"],
            blank_hint="固溶态板材",
            heat_treatments=["固溶处理"],
            surface_treatments=["酸洗", "钝化"],
            cutting_speed_range=(15, 30),
            special_tooling=True,
            coolant_required=True,
            warnings=["切削难度大"],
            recommendations=["比C276更好的抗氧化性"],
        ),
        description="镍基耐蚀合金，综合耐蚀性优异",
    ),

    "Inconel625": MaterialInfo(
        grade="Inconel625",
        name="因科镍625",
        aliases=["Inconel 625", "N06625", "625", "NCF625"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CORROSION_RESISTANT,
        standards=["ASTM B446"],
        properties=MaterialProperties(
            density=8.44,
            melting_point=1350,
            thermal_conductivity=9.8,
            tensile_strength=830,
            yield_strength=415,
            elongation=50,  # %
            hardness="HB200",
            machinability="poor",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材", "锻件"],
            blank_hint="固溶态锻件/板材",
            heat_treatments=["固溶处理", "时效"],
            surface_treatments=["酸洗", "钝化"],
            cutting_speed_range=(12, 25),
            special_tooling=True,
            coolant_required=True,
            warnings=["切削难度大", "加工硬化严重", "刀具磨损快"],
            recommendations=["海洋环境首选", "高温耐蚀环境"],
        ),
        description="镍基高温合金，耐蚀性和焊接性优异",
    ),

    "Inconel718": MaterialInfo(
        grade="Inconel718",
        name="因科镍718",
        aliases=["Inconel 718", "N07718", "718", "GH4169"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CORROSION_RESISTANT,
        standards=["ASTM B637"],
        properties=MaterialProperties(
            density=8.19,
            melting_point=1336,
            thermal_conductivity=11.4,
            tensile_strength=1240,
            yield_strength=1030,
            elongation=20,  # %
            hardness="HRC36-40",
            machinability="poor",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["锻件", "棒材"],
            blank_hint="固溶+时效态锻件",
            heat_treatments=["固溶处理", "时效强化"],
            surface_treatments=["酸洗"],
            cutting_speed_range=(10, 20),
            special_tooling=True,
            coolant_required=True,
            warnings=["极难加工", "需要专用刀具和参数", "切削热大"],
            recommendations=["航空发动机零件", "高温高强度场合"],
        ),
        description="镍基高温合金，高温强度极高",
    ),

    "Monel400": MaterialInfo(
        grade="Monel400",
        name="蒙乃尔400",
        aliases=["Monel 400", "N04400", "NCu30", "2.4360"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CORROSION_RESISTANT,
        standards=["ASTM B127", "ASTM B164"],
        properties=MaterialProperties(
            density=8.80,
            melting_point=1350,
            thermal_conductivity=21.8,
            tensile_strength=550,
            yield_strength=240,
            elongation=35,  # %
            hardness="HB110-150",
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材", "锻件"],
            blank_hint="热轧/冷拔棒材",
            heat_treatments=["退火"],
            surface_treatments=["酸洗"],
            cutting_speed_range=(15, 30),
            special_tooling=True,
            coolant_required=True,
            warnings=["加工硬化倾向", "需要锋利刀具"],
            recommendations=["海水设备", "阀门", "泵轴"],
        ),
        description="镍铜合金，优异的耐海水腐蚀性",
    ),

    "MonelK500": MaterialInfo(
        grade="MonelK500",
        name="蒙乃尔K500",
        aliases=["Monel K-500", "Monel K500", "N05500", "2.4375"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CORROSION_RESISTANT,
        standards=["ASTM B865"],
        properties=MaterialProperties(
            density=8.44,
            melting_point=1350,
            thermal_conductivity=17.5,
            tensile_strength=1100,
            yield_strength=790,
            elongation=20,  # %
            hardness="HRC28-35",
            machinability="poor",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件"],
            blank_hint="时效硬化态棒材",
            heat_treatments=["时效硬化"],
            surface_treatments=["酸洗"],
            cutting_speed_range=(8, 15),
            special_tooling=True,
            coolant_required=True,
            warnings=["极难加工", "时效态更难加工", "需要专用刀具"],
            recommendations=["海洋紧固件", "泵轴", "弹簧"],
        ),
        description="时效硬化镍铜合金，高强度耐海水腐蚀",
    ),

    "HastelloyB3": MaterialInfo(
        grade="HastelloyB3",
        name="哈氏合金B3",
        aliases=["Hastelloy B-3", "Hastelloy B3", "N10675", "2.4600"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CORROSION_RESISTANT,
        standards=["ASTM B333"],
        properties=MaterialProperties(
            density=9.22,
            melting_point=1370,
            thermal_conductivity=10,
            tensile_strength=760,
            yield_strength=380,
            elongation=40,  # %
            hardness="HB200",
            machinability="poor",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材"],
            blank_hint="固溶态板材",
            heat_treatments=["固溶处理"],
            surface_treatments=["酸洗"],
            cutting_speed_range=(8, 15),
            special_tooling=True,
            coolant_required=True,
            warnings=["极难加工", "对盐酸耐蚀但对氧化性介质敏感"],
            recommendations=["盐酸环境", "还原性酸环境"],
        ),
        description="镍钼合金，耐盐酸和还原性酸腐蚀",
    ),

    "Stellite6": MaterialInfo(
        grade="Stellite6",
        name="司太立6",
        aliases=["Stellite 6", "钴基6号", "CoCr-A"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.CORROSION_RESISTANT,
        standards=["AWS A5.21"],
        properties=MaterialProperties(
            density=8.40,
            melting_point=1285,
            thermal_conductivity=14.6,
            tensile_strength=900,
            yield_strength=700,
            elongation=1,  # %
            hardness="HRC38-44",
            machinability="poor",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件", "堆焊层", "粉末冶金件"],
            blank_hint="铸造或堆焊成型",
            heat_treatments=[],
            surface_treatments=["研磨"],
            cutting_speed_range=(5, 10),
            special_tooling=True,
            coolant_required=True,
            warnings=["极难加工", "只能用CBN或金刚石刀具", "通常只做研磨"],
            recommendations=["阀座", "密封面", "高温耐磨件"],
        ),
        description="钴基硬质合金，极高的硬度和耐磨性",
    ),

    "Incoloy825": MaterialInfo(
        grade="Incoloy825",
        name="因科洛伊825",
        aliases=["Incoloy 825", "N08825", "GH2825", "2.4858"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CORROSION_RESISTANT,
        standards=["ASTM B424", "ASTM B425"],
        properties=MaterialProperties(
            density=8.14,
            melting_point=1370,
            thermal_conductivity=11.1,
            tensile_strength=690,
            yield_strength=310,
            elongation=30,  # %
            hardness="HB120-200",
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材", "锻件"],
            blank_hint="固溶态板材/管材",
            heat_treatments=["固溶处理", "稳定化处理"],
            surface_treatments=["酸洗", "钝化"],
            cutting_speed_range=(20, 35),
            special_tooling=True,
            coolant_required=True,
            warnings=["加工硬化倾向", "需要锋利刀具和充分冷却"],
            recommendations=["硫酸环境", "磷酸环境", "油气井下管材"],
        ),
        description="镍铁铬合金，优异的耐酸腐蚀性能",
    ),

    "2205": MaterialInfo(
        grade="2205",
        name="双相不锈钢",
        aliases=["S31803", "S32205", "SAF2205", "1.4462"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.STAINLESS_STEEL,
        standards=["ASTM A240"],
        properties=MaterialProperties(
            density=7.8,
            melting_point=1385,
            thermal_conductivity=19,
            tensile_strength=620,
            yield_strength=450,
            elongation=25,  # %
            hardness="HB290",
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材", "锻件"],
            blank_hint="固溶态板材/锻件",
            heat_treatments=["固溶处理"],
            forbidden_heat_treatments=["淬火"],
            surface_treatments=["钝化", "酸洗"],
            cutting_speed_range=(30, 60),
            coolant_required=True,
            warnings=["比奥氏体不锈钢难加工", "需要更大切削力"],
            recommendations=["海水/氯化物环境", "强度要求高的耐蚀场合"],
        ),
        description="双相不锈钢，强度和耐蚀性兼顾",
    ),

    "2507": MaterialInfo(
        grade="2507",
        name="超级双相不锈钢",
        aliases=["S32750", "SAF2507", "1.4410"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.STAINLESS_STEEL,
        standards=["ASTM A240"],
        properties=MaterialProperties(
            density=7.8,
            melting_point=1350,
            thermal_conductivity=14,
            tensile_strength=800,
            yield_strength=550,
            elongation=15,  # %
            hardness="HB310",
            machinability="poor",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "锻件"],
            blank_hint="固溶态板材/锻件",
            heat_treatments=["固溶处理"],
            forbidden_heat_treatments=["淬火"],
            surface_treatments=["钝化", "酸洗"],
            cutting_speed_range=(25, 50),
            special_tooling=True,
            coolant_required=True,
            warnings=["难加工", "焊接需严格控制热输入"],
            recommendations=["海水淡化", "化工高压管道"],
        ),
        description="超级双相不锈钢，耐蚀性极强",
    ),

    # -------------------------------------------------------------------------
    # 耐热钢/高温合金 (Heat Resistant Steel / Superalloy)
    # -------------------------------------------------------------------------
    "310S": MaterialInfo(
        grade="310S",
        name="耐热不锈钢",
        aliases=["S31008", "SUS310S", "1.4845", "0Cr25Ni20"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CORROSION_RESISTANT,
        standards=["GB/T 1220-2007", "ASTM A240"],
        properties=MaterialProperties(
            density=7.98,
            melting_point=1400,
            thermal_conductivity=14.2,
            tensile_strength=520,
            yield_strength=205,
            elongation=40,  # %
            hardness="HB150-190",
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材"],
            blank_hint="固溶态板材",
            heat_treatments=["固溶处理"],
            forbidden_heat_treatments=["淬火"],
            surface_treatments=["酸洗", "钝化"],
            heat_treatment_notes=["固溶温度1030-1150℃", "快冷"],
            cutting_speed_range=(30, 60),
            coolant_required=True,
            warnings=["高温强度下降需注意", "焊接后需固溶"],
            recommendations=["炉膛部件", "热处理夹具", "1000℃以下长期使用"],
        ),
        description="耐热不锈钢，高温抗氧化性优异",
    ),

    "GH3030": MaterialInfo(
        grade="GH3030",
        name="镍基高温合金",
        aliases=["Nimonic 75", "N06075", "GH30"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CORROSION_RESISTANT,
        standards=["GB/T 14992-2005"],
        properties=MaterialProperties(
            density=8.4,
            melting_point=1370,
            thermal_conductivity=11.7,
            tensile_strength=735,
            yield_strength=275,
            elongation=40,  # %
            hardness="HB150-200",
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材", "丝材"],
            blank_hint="固溶态板材/棒材",
            heat_treatments=["固溶处理"],
            surface_treatments=["酸洗", "机械抛光"],
            heat_treatment_notes=["固溶温度1080-1120℃", "空冷或水冷"],
            cutting_speed_range=(15, 30),
            special_tooling=True,
            coolant_required=True,
            warnings=["加工硬化严重", "切削力大"],
            recommendations=["燃烧室部件", "涡轮导向叶片", "850℃以下长期使用"],
        ),
        description="固溶强化镍基高温合金",
    ),

    "GH4169": MaterialInfo(
        grade="GH4169",
        name="镍基高温合金",
        aliases=["Inconel 718", "N07718", "GH169"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CORROSION_RESISTANT,
        standards=["GB/T 14992-2005"],
        properties=MaterialProperties(
            density=8.19,
            melting_point=1336,
            thermal_conductivity=11.4,
            tensile_strength=1240,
            yield_strength=1030,
            elongation=12,  # %
            hardness="HRC38-44",
            machinability="poor",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["锻件", "棒材", "环形件"],
            blank_hint="固溶+时效态锻件",
            heat_treatments=["固溶处理", "时效强化", "直接时效"],
            surface_treatments=["酸洗", "喷丸"],
            heat_treatment_notes=["固溶980℃/1h", "时效720℃/8h+620℃/8h"],
            cutting_speed_range=(10, 25),
            special_tooling=True,
            coolant_required=True,
            warnings=["难加工材料", "刀具磨损严重", "加工变形大"],
            recommendations=["航空发动机盘", "涡轮叶片", "650℃以下长期使用"],
        ),
        description="时效强化镍基高温合金，航空发动机关键材料",
    ),

    "GH4099": MaterialInfo(
        grade="GH4099",
        name="镍基高温合金",
        aliases=["Waspaloy", "N07001"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CORROSION_RESISTANT,
        standards=["GB/T 14992-2005"],
        properties=MaterialProperties(
            density=8.19,
            melting_point=1330,
            thermal_conductivity=10.7,
            tensile_strength=1280,
            yield_strength=795,
            elongation=10,  # %
            hardness="HRC38-44",
            machinability="poor",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["锻件", "棒材"],
            blank_hint="固溶+时效态锻件",
            heat_treatments=["固溶处理", "时效强化"],
            surface_treatments=["酸洗"],
            heat_treatment_notes=["固溶1010℃/4h", "稳定化处理845℃/4h", "时效760℃/16h"],
            cutting_speed_range=(8, 18),
            special_tooling=True,
            coolant_required=True,
            warnings=["极难加工", "焊接易裂纹"],
            recommendations=["航空发动机涡轮盘", "高温紧固件", "850℃以下使用"],
        ),
        description="高强度镍基高温合金，涡轮盘专用",
    ),

    # 高温合金补充 (Superalloy Supplement)
    "GH2132": MaterialInfo(
        grade="GH2132",
        name="铁基高温合金",
        aliases=["A-286", "SUH660", "1.4980", "Incoloy A-286"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.SUPERALLOY,
        standards=["GB/T 14992-2005", "AMS 5525"],
        properties=MaterialProperties(
            density=7.94,
            melting_point=1400,
            thermal_conductivity=12.8,
            tensile_strength=980,
            yield_strength=690,
            elongation=20,  # %
            hardness="HRC28-35",
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["锻件", "棒材", "板材"],
            blank_hint="固溶+时效态",
            heat_treatments=["固溶处理", "时效强化"],
            surface_treatments=["酸洗", "钝化"],
            heat_treatment_notes=["固溶980℃/1h水冷", "时效720℃/16h空冷"],
            cutting_speed_range=(15, 30),
            special_tooling=True,
            coolant_required=True,
            warnings=["加工硬化严重", "需锋利刀具"],
            recommendations=["航空紧固件", "涡轮盘", "650℃以下长期使用"],
        ),
        description="铁基高温合金，性价比高的航空材料",
    ),

    "K403": MaterialInfo(
        grade="K403",
        name="镍基铸造高温合金",
        aliases=["Mar-M246", "IN-100", "铸造高温合金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.SUPERALLOY,
        standards=["GB/T 14993-2005"],
        properties=MaterialProperties(
            density=8.52,
            melting_point=1330,
            thermal_conductivity=8.4,
            tensile_strength=900,
            yield_strength=750,
            elongation=8,  # %
            hardness="HRC35-42",
            machinability="poor",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["精密铸件"],
            blank_hint="熔模精密铸造",
            heat_treatments=["固溶处理", "时效强化"],
            surface_treatments=["酸洗", "喷丸"],
            heat_treatment_notes=["固溶1200℃/4h", "时效870℃/20h"],
            cutting_speed_range=(5, 12),
            special_tooling=True,
            coolant_required=True,
            warnings=["极难加工", "铸造缺陷敏感", "不可焊接"],
            recommendations=["航空发动机叶片", "导向叶片", "900℃工作温度"],
        ),
        description="镍基铸造高温合金，涡轮叶片专用",
    ),

    "K418": MaterialInfo(
        grade="K418",
        name="镍基铸造高温合金",
        aliases=["IN-738", "铸造涡轮材料"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.SUPERALLOY,
        standards=["GB/T 14993-2005"],
        properties=MaterialProperties(
            density=8.25,
            melting_point=1290,
            thermal_conductivity=8.9,
            tensile_strength=980,
            yield_strength=820,
            elongation=5,  # %
            hardness="HRC38-45",
            machinability="poor",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["精密铸件"],
            blank_hint="定向凝固或单晶铸造",
            heat_treatments=["固溶处理", "两段时效"],
            surface_treatments=["酸洗", "热障涂层"],
            heat_treatment_notes=["固溶1120℃/2h", "一次时效1080℃/4h", "二次时效845℃/24h"],
            cutting_speed_range=(4, 10),
            special_tooling=True,
            coolant_required=True,
            warnings=["极难加工", "成本极高", "不可焊接"],
            recommendations=["航空发动机一级涡轮叶片", "燃气轮机叶片", "950℃工作温度"],
        ),
        description="高性能镍基铸造高温合金，一级涡轮叶片专用",
    ),

    # -------------------------------------------------------------------------
    # 铸铁 (Cast Iron)
    # -------------------------------------------------------------------------
    "HT200": MaterialInfo(
        grade="HT200",
        name="灰铸铁",
        aliases=["灰铁", "HT20", "FC200"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CAST_IRON,
        standards=["GB/T 9439-2010"],
        properties=MaterialProperties(
            density=7.2,
            melting_point=1200,
            thermal_conductivity=48,
            tensile_strength=200,
            yield_strength=None,  # 灰铸铁脆性材料，无明确屈服点
            elongation=0.5,  # % 脆性材料
            hardness="HB170-240",
            machinability="excellent",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件"],
            blank_hint="铸造毛坯",
            heat_treatments=["时效处理", "去应力退火"],
            surface_treatments=["发黑", "喷漆", "镀锌"],
            heat_treatment_notes=["时效处理消除铸造应力"],
            warnings=["焊接性差", "脆性材料"],
            recommendations=["机床床身/箱体", "需要时效消除应力"],
        ),
        description="灰铸铁，减振性好，适合箱体类零件",
    ),

    "QT400": MaterialInfo(
        grade="QT400",
        name="球墨铸铁",
        aliases=["球铁", "QT400-15", "FCD400"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CAST_IRON,
        standards=["GB/T 1348-2009"],
        properties=MaterialProperties(
            density=7.1,
            melting_point=1150,
            thermal_conductivity=36,
            tensile_strength=400,
            yield_strength=250,
            elongation=15,  # %
            hardness="HB130-180",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件"],
            blank_hint="铸造毛坯",
            heat_treatments=["正火", "退火", "等温淬火"],
            surface_treatments=["发黑", "喷漆", "镀层"],
            recommendations=["曲轴/齿轮", "可部分替代锻钢"],
        ),
        description="球墨铸铁，强度和韧性好于灰铸铁",
    ),

    "HT250": MaterialInfo(
        grade="HT250",
        name="灰铸铁",
        aliases=["HT25", "FC250", "灰铁250"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CAST_IRON,
        standards=["GB/T 9439-2010"],
        properties=MaterialProperties(
            density=7.25,
            melting_point=1200,
            thermal_conductivity=46,
            tensile_strength=250,
            yield_strength=None,  # 灰铸铁脆性材料，无明确屈服点
            hardness="HB180-250",
            elongation=0.5,  # %
            machinability="excellent",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件"],
            blank_hint="铸造毛坯",
            heat_treatments=["时效处理", "去应力退火"],
            surface_treatments=["发黑", "喷漆", "镀锌"],
            heat_treatment_notes=["时效处理消除铸造应力"],
            warnings=["焊接性差", "脆性材料"],
            recommendations=["机床床身/箱体", "承载要求较高的结构件"],
        ),
        description="灰铸铁，强度高于HT200，减振性好",
    ),

    "HT300": MaterialInfo(
        grade="HT300",
        name="高强度灰铸铁",
        aliases=["HT30", "FC300", "灰铁300"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CAST_IRON,
        standards=["GB/T 9439-2010"],
        properties=MaterialProperties(
            density=7.3,
            melting_point=1200,
            thermal_conductivity=44,
            tensile_strength=300,
            yield_strength=None,  # 灰铸铁脆性材料，无明确屈服点
            hardness="HB200-280",
            elongation=0.5,  # %
            machinability="good",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件"],
            blank_hint="铸造毛坯",
            heat_treatments=["时效处理", "去应力退火", "正火"],
            surface_treatments=["发黑", "喷漆"],
            heat_treatment_notes=["时效处理消除铸造应力", "可正火提高硬度"],
            warnings=["焊接性差", "脆性材料", "铸造难度较大"],
            recommendations=["高承载机床床身", "液压缸体", "重载齿轮箱"],
        ),
        description="高强度灰铸铁，用于承载要求高的场合",
    ),

    "QT500-7": MaterialInfo(
        grade="QT500-7",
        name="球墨铸铁",
        aliases=["QT500", "FCD500", "球铁500"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CAST_IRON,
        standards=["GB/T 1348-2009"],
        properties=MaterialProperties(
            density=7.1,
            melting_point=1150,
            thermal_conductivity=34,
            tensile_strength=500,
            yield_strength=320,
            elongation=7,  # %
            hardness="HB170-230",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件"],
            blank_hint="铸造毛坯",
            heat_treatments=["正火", "退火", "调质", "等温淬火"],
            surface_treatments=["发黑", "喷漆", "镀层", "氮化"],
            heat_treatment_notes=["正火可提高强度", "等温淬火可获得贝氏体组织"],
            recommendations=["曲轴/连杆", "齿轮/凸轮轴", "替代部分锻钢件"],
        ),
        description="中强度球墨铸铁，综合性能好",
    ),

    "QT600-3": MaterialInfo(
        grade="QT600-3",
        name="高强度球墨铸铁",
        aliases=["QT600", "FCD600", "球铁600"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CAST_IRON,
        standards=["GB/T 1348-2009"],
        properties=MaterialProperties(
            density=7.1,
            melting_point=1150,
            thermal_conductivity=32,
            tensile_strength=600,
            yield_strength=370,
            elongation=3,  # %
            hardness="HB190-270",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件"],
            blank_hint="铸造毛坯",
            heat_treatments=["正火", "调质", "等温淬火"],
            surface_treatments=["发黑", "喷漆", "镀层", "氮化"],
            heat_treatment_notes=["正火后硬度HB220-280", "等温淬火可获高强高韧"],
            warnings=["韧性低于QT500-7"],
            recommendations=["高强度曲轴", "重载齿轮", "液压件"],
        ),
        description="高强度球墨铸铁，强度高但韧性稍低",
    ),

    "QT700-2": MaterialInfo(
        grade="QT700-2",
        name="高强度球墨铸铁",
        aliases=["QT700", "FCD700", "球铁700"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CAST_IRON,
        standards=["GB/T 1348-2009"],
        properties=MaterialProperties(
            density=7.1,
            melting_point=1150,
            thermal_conductivity=30,
            tensile_strength=700,
            yield_strength=420,
            elongation=2,  # %
            hardness="HB225-305",
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件"],
            blank_hint="铸造毛坯",
            heat_treatments=["正火+回火", "调质", "等温淬火"],
            surface_treatments=["发黑", "喷漆"],
            heat_treatment_notes=["正火后硬度HB260-320"],
            warnings=["韧性较低", "铸造工艺要求高"],
            recommendations=["高强度齿轮", "重载结构件", "替代锻钢件"],
        ),
        description="高强度球墨铸铁，强度最高的球铁之一",
    ),

    # -------------------------------------------------------------------------
    # 铝合金 (Aluminum Alloy)
    # -------------------------------------------------------------------------
    "6061": MaterialInfo(
        grade="6061",
        name="铝镁硅合金",
        aliases=["6061-T6", "AlMg1SiCu"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ALUMINUM,
        standards=["GB/T 3190-2020"],
        properties=MaterialProperties(
            density=2.70,
            melting_point=580,
            thermal_conductivity=167,
            tensile_strength=310,
            yield_strength=276,
            elongation=12,  # %
            hardness="HB95",
            machinability="excellent",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "型材", "棒材", "管材"],
            blank_hint="挤压型材/板材",
            heat_treatments=["固溶+时效(T6)"],
            forbidden_heat_treatments=["淬火"],
            surface_treatments=["阳极氧化", "喷漆", "电泳"],
            forbidden_surface_treatments=["电镀"],
            cutting_speed_range=(200, 500),
            coolant_required=False,
            warnings=["不可电镀", "焊接后强度降低"],
            recommendations=["结构件首选", "阳极氧化后耐蚀性好"],
        ),
        description="通用铝合金，可热处理强化",
    ),

    "7075": MaterialInfo(
        grade="7075",
        name="超硬铝合金",
        aliases=["7075-T6", "AlZnMgCu"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ALUMINUM,
        standards=["GB/T 3190-2020"],
        properties=MaterialProperties(
            density=2.81,
            melting_point=475,
            thermal_conductivity=130,
            tensile_strength=572,
            yield_strength=503,
            elongation=11,  # %
            hardness="HB150",
            machinability="good",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "锻件"],
            blank_hint="板材/锻件",
            heat_treatments=["固溶+时效(T6)", "T651"],
            surface_treatments=["阳极氧化", "化学氧化"],
            warnings=["焊接性差", "应力腐蚀敏感"],
            recommendations=["航空结构件", "高强度要求场合"],
        ),
        description="超硬铝，强度最高的铝合金之一",
    ),

    "2024": MaterialInfo(
        grade="2024",
        name="硬铝合金",
        aliases=["2024-T4", "2024-T351", "AlCuMg1"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ALUMINUM,
        standards=["GB/T 3190-2020"],
        properties=MaterialProperties(
            density=2.78,
            melting_point=502,
            thermal_conductivity=121,
            tensile_strength=470,
            yield_strength=325,
            elongation=10,  # %
            hardness="HB120",
            machinability="good",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "型材"],
            blank_hint="板材/棒材",
            heat_treatments=["固溶(T4)", "自然时效(T351)"],
            forbidden_heat_treatments=["熔焊"],
            surface_treatments=["阳极氧化", "化学氧化", "喷漆"],
            cutting_speed_range=(150, 400),
            warnings=["焊接性差", "应力腐蚀敏感"],
            recommendations=["航空结构件", "机翼蒙皮", "铆钉"],
        ),
        description="经典航空铝合金，高强度高韧性",
    ),

    "5052": MaterialInfo(
        grade="5052",
        name="防锈铝合金",
        aliases=["5052-H32", "AlMg2.5"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ALUMINUM,
        standards=["GB/T 3190-2020"],
        properties=MaterialProperties(
            density=2.68,
            melting_point=607,
            thermal_conductivity=138,
            tensile_strength=230,
            yield_strength=195,
            elongation=12,  # %
            hardness="HB60",
            machinability="good",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "带材", "管材"],
            blank_hint="板材/带材",
            heat_treatments=["退火(O)", "加工硬化(H32)"],
            surface_treatments=["阳极氧化", "化学氧化", "喷涂"],
            cutting_speed_range=(200, 500),
            recommendations=["船舶", "汽车油箱", "仪表面板"],
        ),
        description="耐蚀性好，焊接性优良",
    ),

    "5083": MaterialInfo(
        grade="5083",
        name="船用铝合金",
        aliases=["5083-H116", "AlMg4.5Mn"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ALUMINUM,
        standards=["GB/T 3190-2020"],
        properties=MaterialProperties(
            density=2.66,
            melting_point=591,
            thermal_conductivity=121,
            tensile_strength=315,
            yield_strength=230,
            elongation=14,  # %
            hardness="HB75",
            machinability="good",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "型材"],
            blank_hint="中厚板/型材",
            heat_treatments=["退火(O)", "稳定化处理(H116)"],
            surface_treatments=["阳极氧化", "喷涂"],
            cutting_speed_range=(180, 450),
            recommendations=["船体", "LNG储罐", "压力容器"],
        ),
        description="海洋环境铝合金，耐海水腐蚀",
    ),

    "2A12": MaterialInfo(
        grade="2A12",
        name="硬铝合金",
        aliases=["LY12", "2A12-T4"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ALUMINUM,
        standards=["GB/T 3190-2020"],
        properties=MaterialProperties(
            density=2.78,
            melting_point=502,
            thermal_conductivity=121,
            tensile_strength=470,
            yield_strength=325,
            elongation=12,  # %
            hardness="HB105",
            machinability="good",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "型材", "锻件"],
            blank_hint="板材/锻件",
            heat_treatments=["固溶(T4)", "人工时效(T6)"],
            forbidden_heat_treatments=["熔焊"],
            surface_treatments=["阳极氧化", "化学氧化"],
            cutting_speed_range=(150, 400),
            warnings=["焊接性差", "需包铝板防腐"],
            recommendations=["航空骨架", "机翼", "承力构件"],
        ),
        description="中国航空主力铝合金，等同于2024",
    ),

    "6063": MaterialInfo(
        grade="6063",
        name="建筑铝合金",
        aliases=["6063-T5", "6063-T6", "AlMgSi"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ALUMINUM,
        standards=["GB/T 3190-2020"],
        properties=MaterialProperties(
            density=2.69,
            melting_point=615,
            thermal_conductivity=200,
            tensile_strength=205,
            yield_strength=170,
            elongation=12,  # %
            hardness="HB73",
            machinability="excellent",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["型材", "管材"],
            blank_hint="挤压型材",
            heat_treatments=["固溶+时效(T5/T6)"],
            surface_treatments=["阳极氧化", "电泳涂装", "粉末喷涂"],
            cutting_speed_range=(200, 600),
            recommendations=["门窗型材", "散热器", "装饰件"],
        ),
        description="挤压性能优异，表面质量好",
    ),

    "A356": MaterialInfo(
        grade="A356",
        name="铸造铝硅合金",
        aliases=["ZL101", "AlSi7Mg", "AC4C"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ALUMINUM,
        standards=["ASTM B108", "GB/T 1173-2013"],
        properties=MaterialProperties(
            density=2.68,
            melting_point=615,
            thermal_conductivity=151,
            tensile_strength=230,
            yield_strength=165,
            elongation=3,  # %
            hardness="HB75-90",
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件"],
            blank_hint="低压铸造/重力铸造",
            heat_treatments=["T6 (固溶+时效)", "T5"],
            surface_treatments=["阳极氧化", "喷涂", "化学镀"],
            heat_treatment_notes=["固溶540℃/8h", "时效155℃/6h"],
            cutting_speed_range=(200, 500),
            recommendations=["汽车轮毂", "航空铸件", "泵壳体", "结构件"],
        ),
        description="优质铸造铝合金，力学性能好，铸造性优",
    ),

    "ZL102": MaterialInfo(
        grade="ZL102",
        name="铸造铝硅合金",
        aliases=["A413", "AlSi12", "ADC1"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ALUMINUM,
        standards=["GB/T 1173-2013"],
        properties=MaterialProperties(
            density=2.65,
            melting_point=577,
            thermal_conductivity=151,
            tensile_strength=150,
            yield_strength=70,
            elongation=2,  # %
            hardness="HB50-65",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件"],
            blank_hint="压铸/低压铸造",
            heat_treatments=["T2 (退火)"],
            surface_treatments=["阳极氧化", "喷涂"],
            cutting_speed_range=(250, 600),
            recommendations=["薄壁复杂件", "壳体", "罩盖", "铸造性极佳"],
        ),
        description="高硅铝合金，流动性极好，适合复杂薄壁件",
    ),

    "ADC12": MaterialInfo(
        grade="ADC12",
        name="压铸铝合金",
        aliases=["A383", "AlSi11Cu3", "YL113"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ALUMINUM,
        standards=["JIS H5302", "GB/T 15115-2009"],
        properties=MaterialProperties(
            density=2.74,
            melting_point=580,
            thermal_conductivity=92,
            tensile_strength=310,
            yield_strength=150,
            elongation=1,  # %
            hardness="HB80-95",
            machinability="good",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["压铸件"],
            blank_hint="压铸成型",
            heat_treatments=["T5 (人工时效)"],
            surface_treatments=["喷涂", "电镀", "阳极氧化"],
            cutting_speed_range=(200, 500),
            warnings=["含铜不宜阳极氧化", "焊接性差"],
            recommendations=["汽车变速箱壳", "发动机缸体", "电子壳体"],
        ),
        description="通用压铸铝合金，综合性能好，用量最大",
    ),

    # 铸造铝合金补充 (Cast Aluminum Supplement)
    "ZL101": MaterialInfo(
        grade="ZL101",
        name="铸造铝硅镁合金",
        aliases=["A356.2", "AlSi7Mg", "AC4CH", "高纯ZL101"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.CAST_ALUMINUM,
        standards=["GB/T 1173-2013", "ASTM B108"],
        properties=MaterialProperties(
            density=2.68,
            melting_point=615,
            thermal_conductivity=151,
            tensile_strength=290,
            yield_strength=210,
            elongation=2,  # %
            hardness="HB85-100",
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件", "低压铸件"],
            blank_hint="低压铸造/重力铸造",
            heat_treatments=["T6 (固溶+时效)", "T5 (人工时效)"],
            surface_treatments=["阳极氧化", "微弧氧化", "喷涂"],
            heat_treatment_notes=["固溶535℃/8h水淬", "时效155℃/6h"],
            cutting_speed_range=(200, 450),
            recommendations=["汽车轮毂", "底盘件", "航空结构件", "高气密性要求"],
        ),
        description="优质铸造铝合金，力学性能优异，气密性好",
    ),

    "ZL104": MaterialInfo(
        grade="ZL104",
        name="铸造铝硅铜合金",
        aliases=["A319", "AlSi5Cu1Mg", "AC2B"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.CAST_ALUMINUM,
        standards=["GB/T 1173-2013"],
        properties=MaterialProperties(
            density=2.75,
            melting_point=600,
            thermal_conductivity=109,
            tensile_strength=240,
            yield_strength=180,
            elongation=1,  # %
            hardness="HB90-105",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件", "压铸件"],
            blank_hint="砂型铸造/金属型铸造",
            heat_treatments=["T6", "T5", "T7 (过时效)"],
            surface_treatments=["喷涂", "电镀", "化学氧化"],
            heat_treatment_notes=["固溶505℃/6h水淬", "时效175℃/8h"],
            cutting_speed_range=(180, 400),
            warnings=["含铜不宜阳极氧化"],
            recommendations=["发动机缸盖", "变速箱壳", "泵体", "耐热性好"],
        ),
        description="铝硅铜合金，耐热性好，适合发动机铸件",
    ),

    "ZL201": MaterialInfo(
        grade="ZL201",
        name="铸造铝铜合金",
        aliases=["A201", "AlCu4MgTi", "高强铸铝"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.CAST_ALUMINUM,
        standards=["GB/T 1173-2013", "AMS 4235"],
        properties=MaterialProperties(
            density=2.80,
            melting_point=650,
            thermal_conductivity=121,
            tensile_strength=450,
            yield_strength=380,
            elongation=3,  # %
            hardness="HB130-145",
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件"],
            blank_hint="砂型铸造/熔模铸造",
            heat_treatments=["T4 (固溶+自然时效)", "T6", "T7"],
            surface_treatments=["化学氧化", "喷涂"],
            heat_treatment_notes=["固溶530℃/12h水淬", "自然时效5天以上"],
            cutting_speed_range=(100, 250),
            warnings=["铸造性差", "热裂倾向", "不宜阳极氧化"],
            recommendations=["航空承力件", "高强度结构件", "需要高强度的铸件"],
        ),
        description="高强度铸造铝合金，航空结构件专用",
    ),

    # 锌合金 (Zinc Alloy)
    "Zamak3": MaterialInfo(
        grade="Zamak3",
        name="锌铝合金3号",
        aliases=["ZA-3", "ZAMAK-3", "ZnAl4", "锌合金3号"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ZINC_ALLOY,
        standards=["GB/T 1175-2018", "ASTM B86"],
        properties=MaterialProperties(
            density=6.6,
            melting_point=387,
            thermal_conductivity=113,
            tensile_strength=283,
            yield_strength=221,
            elongation=10,  # %
            hardness="HB82",
            machinability="excellent",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["压铸件"],
            blank_hint="热室压铸",
            heat_treatments=["稳定化处理"],
            surface_treatments=["电镀", "喷涂", "钝化"],
            heat_treatment_notes=["100℃/3-6h稳定尺寸"],
            cutting_speed_range=(150, 350),
            warnings=["不耐高温", "蠕变倾向"],
            recommendations=["五金配件", "锁具", "玩具", "汽车小件"],
        ),
        description="通用锌合金，压铸性能优异，用量最大",
    ),

    "Zamak5": MaterialInfo(
        grade="Zamak5",
        name="锌铝合金5号",
        aliases=["ZA-5", "ZAMAK-5", "ZnAl4Cu1", "锌合金5号"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ZINC_ALLOY,
        standards=["GB/T 1175-2018", "ASTM B86"],
        properties=MaterialProperties(
            density=6.7,
            melting_point=386,
            thermal_conductivity=109,
            tensile_strength=328,
            yield_strength=269,
            elongation=7,  # %
            hardness="HB91",
            machinability="excellent",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["压铸件"],
            blank_hint="热室压铸",
            heat_treatments=["稳定化处理"],
            surface_treatments=["电镀", "喷涂", "钝化"],
            heat_treatment_notes=["100℃/3-6h稳定尺寸"],
            cutting_speed_range=(150, 350),
            warnings=["不耐高温", "尺寸稳定性略差于Zamak3"],
            recommendations=["汽车零件", "电器配件", "高强度要求件"],
        ),
        description="高强度锌合金，强度优于Zamak3，耐磨性好",
    ),

    "ZA-8": MaterialInfo(
        grade="ZA-8",
        name="锌铝合金ZA-8",
        aliases=["ZnAl8Cu1", "超强锌合金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ZINC_ALLOY,
        standards=["GB/T 1175-2018", "ASTM B669"],
        properties=MaterialProperties(
            density=6.3,
            melting_point=404,
            thermal_conductivity=115,
            tensile_strength=374,
            yield_strength=290,
            elongation=8,  # %
            hardness="HB103",
            machinability="excellent",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["压铸件", "重力铸件"],
            blank_hint="热室压铸/重力铸造",
            heat_treatments=["T5 (稳定化)"],
            surface_treatments=["电镀", "喷涂", "阳极氧化"],
            heat_treatment_notes=["95℃/10h或150℃/5h稳定化"],
            cutting_speed_range=(150, 350),
            warnings=["铝含量高需注意氧化"],
            recommendations=["轴承座", "齿轮", "替代青铜件", "耐磨件"],
        ),
        description="高铝锌合金，强度高，可替代部分铜合金",
    ),

    # -------------------------------------------------------------------------
    # 铜合金 (Copper Alloy)
    # -------------------------------------------------------------------------
    "H62": MaterialInfo(
        grade="H62",
        name="普通黄铜",
        aliases=["黄铜", "CuZn40"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.COPPER,
        standards=["GB/T 5231-2012"],
        properties=MaterialProperties(
            density=8.43,
            melting_point=900,
            thermal_conductivity=109,
            tensile_strength=380,
            yield_strength=150,
            elongation=15,  # %
            hardness="HB80-120",
            machinability="excellent",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "管材"],
            blank_hint="棒材/板材",
            heat_treatments=["退火"],
            surface_treatments=["抛光", "镀层", "钝化"],
            cutting_speed_range=(100, 300),
            recommendations=["导电/导热件", "装饰件"],
        ),
        description="普通黄铜，加工性好",
    ),

    "QSn4-3": MaterialInfo(
        grade="QSn4-3",
        name="锡青铜",
        aliases=["青铜", "锡磷青铜"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.COPPER,
        standards=["GB/T 5231-2012"],
        properties=MaterialProperties(
            density=8.8,
            melting_point=1000,
            thermal_conductivity=62,
            tensile_strength=350,
            yield_strength=200,
            elongation=15,  # %
            hardness="HB70-90",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "管材", "铸件"],
            blank_hint="棒材/铸件",
            heat_treatments=["退火"],
            surface_treatments=["抛光"],
            recommendations=["轴瓦/衬套", "耐磨件"],
        ),
        description="锡青铜，耐磨性好，适合轴瓦",
    ),

    "QAl9-4": MaterialInfo(
        grade="QAl9-4",
        name="铝青铜",
        aliases=["QA19-4", "铝青铜"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.COPPER,
        standards=["GB/T 5231-2012"],
        properties=MaterialProperties(
            density=7.6,
            melting_point=1040,
            thermal_conductivity=58,
            tensile_strength=600,
            yield_strength=250,
            elongation=10,  # %
            hardness="HB160-200",
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "铸件"],
            blank_hint="棒材/铸件",
            heat_treatments=["退火", "淬火+回火"],
            surface_treatments=["抛光"],
            warnings=["加工时易粘刀"],
            recommendations=["耐磨蜗轮", "海水环境零件"],
        ),
        description="铝青铜，高强度耐蚀耐磨",
    ),

    "Cu65": MaterialInfo(
        grade="Cu65",
        name="紫铜",
        aliases=["紫铜", "T2", "纯铜"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.COPPER,
        standards=["GB/T 5231-2012"],
        properties=MaterialProperties(
            density=8.9,
            melting_point=1083,
            thermal_conductivity=391,
            tensile_strength=220,
            yield_strength=70,
            elongation=35,  # %
            hardness="HB35-45",
            machinability="fair",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "管材", "线材"],
            blank_hint="棒材/板材",
            heat_treatments=["退火"],
            surface_treatments=["抛光", "镀层"],
            warnings=["软，易变形", "需要锋利刀具"],
            recommendations=["导电件", "导热件"],
        ),
        description="纯铜，导电导热性最好",
    ),

    "QBe2": MaterialInfo(
        grade="QBe2",
        name="铍青铜",
        aliases=["铍铜", "C17200", "BeCu", "CuBe2"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.COPPER,
        standards=["GB/T 5231-2012", "ASTM C17200"],
        properties=MaterialProperties(
            density=8.3,
            melting_point=870,
            thermal_conductivity=115,
            tensile_strength=1250,
            yield_strength=1100,
            elongation=3,  # %
            hardness="HRC38-45",
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "线材"],
            blank_hint="棒材/板材",
            heat_treatments=["固溶处理", "时效硬化"],
            forbidden_heat_treatments=["淬火"],
            surface_treatments=["抛光", "镀层"],
            warnings=["铍有毒，加工需通风", "粉尘有害", "禁止干磨削"],
            recommendations=["弹簧", "导电弹片", "耐磨模具"],
        ),
        description="铍青铜，高强度高弹性，导电性好",
    ),

    "QAl10-3-1.5": MaterialInfo(
        grade="QAl10-3-1.5",
        name="铝青铜",
        aliases=["QAl10-3-1.5", "C63000", "铝铁镍青铜"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.COPPER,
        standards=["GB/T 5231-2012", "ASTM C63000"],
        properties=MaterialProperties(
            density=7.5,
            melting_point=1050,
            thermal_conductivity=50,
            tensile_strength=700,
            yield_strength=350,
            elongation=10,  # %
            hardness="HB180-230",
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "铸件", "锻件"],
            blank_hint="棒材/铸件",
            heat_treatments=["退火", "淬火+回火"],
            surface_treatments=["抛光"],
            warnings=["加工时易粘刀", "需使用冷却液"],
            recommendations=["蜗轮", "阀座", "海水泵件"],
        ),
        description="高强度铝青铜，耐海水腐蚀",
    ),

    "H68": MaterialInfo(
        grade="H68",
        name="高精黄铜",
        aliases=["CuZn33", "C26800", "黄铜68"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.COPPER,
        standards=["GB/T 5231-2012"],
        properties=MaterialProperties(
            density=8.5,
            melting_point=915,
            thermal_conductivity=121,
            tensile_strength=350,
            yield_strength=130,
            elongation=25,  # %
            hardness="HB55-85",
            machinability="excellent",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "带材", "管材"],
            blank_hint="板材/带材",
            heat_treatments=["退火"],
            surface_treatments=["抛光", "镀层", "钝化"],
            recommendations=["精密零件", "弹壳", "散热器"],
        ),
        description="高精黄铜，冷加工性优异",
    ),

    "HPb59-1": MaterialInfo(
        grade="HPb59-1",
        name="铅黄铜",
        aliases=["C38500", "易切削黄铜", "快削黄铜"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.COPPER,
        standards=["GB/T 5231-2012", "ASTM C38500"],
        properties=MaterialProperties(
            density=8.5,
            melting_point=900,
            thermal_conductivity=109,
            tensile_strength=400,
            yield_strength=160,
            elongation=10,  # %
            hardness="HB80-110",
            machinability="excellent",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "型材"],
            blank_hint="棒材",
            heat_treatments=["退火"],
            surface_treatments=["抛光", "镀层"],
            warnings=["含铅，不适合食品接触"],
            recommendations=["钟表零件", "精密仪器件", "自动车床件"],
        ),
        description="铅黄铜，切削性最好的铜合金",
    ),

    "QSn6.5-0.1": MaterialInfo(
        grade="QSn6.5-0.1",
        name="磷青铜",
        aliases=["C51900", "磷铜", "弹簧磷青铜"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.COPPER,
        standards=["GB/T 5231-2012", "ASTM C51900"],
        properties=MaterialProperties(
            density=8.8,
            melting_point=1000,
            thermal_conductivity=62,
            tensile_strength=550,
            yield_strength=450,
            elongation=3,  # %
            hardness="HB100-150",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["带材", "线材", "板材"],
            blank_hint="带材/板材",
            heat_treatments=["退火", "去应力退火"],
            surface_treatments=["抛光", "镀锡"],
            recommendations=["弹簧", "导电弹片", "接插件"],
        ),
        description="磷青铜，高弹性耐疲劳",
    ),

    "CuNi10Fe1Mn": MaterialInfo(
        grade="CuNi10Fe1Mn",
        name="白铜",
        aliases=["B10", "CN102", "C70600", "90/10白铜"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.COPPER,
        standards=["GB/T 5231-2012", "ASTM C70600"],
        properties=MaterialProperties(
            density=8.9,
            melting_point=1150,
            thermal_conductivity=45,
            tensile_strength=350,
            yield_strength=140,
            elongation=15,  # %
            hardness="HB70-100",
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["管材", "板材", "棒材"],
            blank_hint="管材/板材",
            heat_treatments=["退火"],
            surface_treatments=["抛光", "钝化"],
            recommendations=["海水冷凝管", "热交换器", "船用管件"],
        ),
        description="白铜，优异的耐海水腐蚀性",
    ),

    # 锡青铜补充 (Tin Bronze Supplement)
    "QSn7-0.2": MaterialInfo(
        grade="QSn7-0.2",
        name="高锡磷青铜",
        aliases=["C52100", "CuSn8", "高弹性磷铜"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.TIN_BRONZE,
        standards=["GB/T 5231-2012", "ASTM C52100"],
        properties=MaterialProperties(
            density=8.8,
            melting_point=1000,
            thermal_conductivity=50,
            tensile_strength=700,
            yield_strength=600,
            elongation=3,  # %
            hardness="HB150-200",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["带材", "线材", "板材"],
            blank_hint="带材/板材",
            heat_treatments=["退火", "去应力退火"],
            surface_treatments=["抛光", "镀锡", "镀镍"],
            cutting_speed_range=(50, 120),
            recommendations=["高负荷弹簧", "电气开关弹片", "精密仪器弹性件"],
        ),
        description="高锡磷青铜，弹性极限高，耐疲劳",
    ),

    "QSn4-0.3": MaterialInfo(
        grade="QSn4-0.3",
        name="低锡磷青铜",
        aliases=["C51000", "CuSn4", "导电磷青铜"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.TIN_BRONZE,
        standards=["GB/T 5231-2012", "ASTM C51000"],
        properties=MaterialProperties(
            density=8.9,
            melting_point=1040,
            thermal_conductivity=72,
            tensile_strength=450,
            yield_strength=350,
            elongation=10,  # %
            hardness="HB90-130",
            machinability="good",
            weldability="good",
            conductivity=15.0,  # 15% IACS
        ),
        process=ProcessRecommendation(
            blank_forms=["带材", "线材", "板材"],
            blank_hint="带材/板材",
            heat_treatments=["退火", "去应力退火"],
            surface_treatments=["抛光", "镀锡", "镀金"],
            cutting_speed_range=(60, 150),
            recommendations=["电器接插件", "导电弹片", "端子"],
        ),
        description="低锡磷青铜，导电性与弹性兼顾",
    ),

    "ZCuSn10P1": MaterialInfo(
        grade="ZCuSn10P1",
        name="铸造锡磷青铜",
        aliases=["C90700", "PBC2A", "高力青铜"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.TIN_BRONZE,
        standards=["GB/T 1176-2013", "ASTM C90700"],
        properties=MaterialProperties(
            density=8.8,
            melting_point=1000,
            thermal_conductivity=50,
            tensile_strength=280,
            yield_strength=140,
            elongation=5,  # %
            hardness="HB80-100",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件"],
            blank_hint="砂型铸造/离心铸造",
            heat_treatments=["去应力退火"],
            surface_treatments=["抛光"],
            cutting_speed_range=(40, 100),
            recommendations=["重载轴瓦", "蜗轮", "耐磨衬套", "泵体"],
        ),
        description="铸造锡磷青铜，耐磨性极好，重载轴瓦专用",
    ),

    # 硅黄铜 (Silicon Brass)
    "HSi80-3": MaterialInfo(
        grade="HSi80-3",
        name="硅黄铜",
        aliases=["C87500", "CuZn16Si3", "耐蚀黄铜"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.SILICON_BRASS,
        standards=["GB/T 5231-2012", "ASTM C87500"],
        properties=MaterialProperties(
            density=8.3,
            melting_point=950,
            thermal_conductivity=38,
            tensile_strength=480,
            yield_strength=200,
            elongation=25,  # %
            hardness="HB100-140",
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "管材", "型材"],
            blank_hint="棒材/型材",
            heat_treatments=["退火"],
            surface_treatments=["抛光", "钝化"],
            cutting_speed_range=(60, 150),
            recommendations=["船用配件", "化工管道", "耐蚀零件", "热水器配件"],
        ),
        description="硅黄铜，耐蚀性优于普通黄铜，可焊性好",
    ),

    # 铝黄铜 (Aluminum Brass)
    "HAl77-2": MaterialInfo(
        grade="HAl77-2",
        name="铝黄铜",
        aliases=["C68700", "CuZn22Al2", "海军黄铜"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.COPPER,
        standards=["GB/T 5231-2012", "ASTM C68700"],
        properties=MaterialProperties(
            density=8.4,
            melting_point=920,
            thermal_conductivity=100,
            tensile_strength=440,
            yield_strength=180,
            elongation=10,  # %
            hardness="HB80-120",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["管材", "棒材", "板材"],
            blank_hint="管材/棒材",
            heat_treatments=["退火", "去应力退火"],
            surface_treatments=["抛光", "钝化"],
            cutting_speed_range=(50, 130),
            warnings=["需防止脱锌腐蚀"],
            recommendations=["海水冷凝管", "船用配件", "热交换器管"],
        ),
        description="铝黄铜，优异的耐海水腐蚀性能",
    ),

    # 耐磨铸铁 (Wear-Resistant Cast Iron)
    "NiHard1": MaterialInfo(
        grade="NiHard1",
        name="镍硬铸铁1型",
        aliases=["Ni-Hard1", "NiCr4", "高铬镍铸铁"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.WEAR_RESISTANT_IRON,
        standards=["ASTM A532", "GB/T 8263-2010"],
        properties=MaterialProperties(
            density=7.6,
            melting_point=1170,
            thermal_conductivity=21,
            tensile_strength=350,
            yield_strength=None,  # 脆性耐磨铸铁，无明确屈服点
            hardness="HRC55-62",
            elongation=0.5,  # %
            machinability="poor",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件"],
            blank_hint="砂型铸造",
            heat_treatments=["消除应力退火", "空冷淬火"],
            heat_treatment_notes=["铸后空冷至室温", "去应力退火300℃"],
            surface_treatments=["喷丸"],
            cutting_speed_range=(5, 15),
            special_tooling=True,
            coolant_required=True,
            warnings=["极难加工", "脆性材料", "不可焊接修复"],
            recommendations=["球磨机衬板", "破碎机锤头", "渣浆泵叶轮"],
        ),
        description="镍硬铸铁，高硬度高耐磨，矿山设备专用",
    ),

    "NiHard4": MaterialInfo(
        grade="NiHard4",
        name="镍硬铸铁4型",
        aliases=["Ni-Hard4", "NiCrMo", "高铬钼镍铸铁"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.WEAR_RESISTANT_IRON,
        standards=["ASTM A532", "GB/T 8263-2010"],
        properties=MaterialProperties(
            density=7.7,
            melting_point=1170,
            thermal_conductivity=19,
            tensile_strength=450,
            yield_strength=None,  # 脆性耐磨铸铁，无明确屈服点
            hardness="HRC58-65",
            elongation=0.5,  # %
            machinability="poor",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件"],
            blank_hint="砂型铸造/消失模铸造",
            heat_treatments=["消除应力退火", "二次硬化"],
            heat_treatment_notes=["铸后空冷", "去应力300℃/4h", "可二次硬化"],
            surface_treatments=["喷丸"],
            cutting_speed_range=(3, 10),
            special_tooling=True,
            coolant_required=True,
            warnings=["极难加工", "高脆性", "不可焊接"],
            recommendations=["轧辊", "高耐磨衬板", "大型破碎机部件"],
        ),
        description="镍硬铸铁4型，硬度更高，适合严苛磨损工况",
    ),

    "Cr26": MaterialInfo(
        grade="Cr26",
        name="高铬铸铁",
        aliases=["Cr26Mo", "KmTBCr26", "26%铬铸铁"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.WEAR_RESISTANT_IRON,
        standards=["GB/T 8263-2010", "ASTM A532"],
        properties=MaterialProperties(
            density=7.5,
            melting_point=1350,
            thermal_conductivity=15,
            tensile_strength=500,
            yield_strength=None,  # 脆性高铬铸铁，无明确屈服点
            hardness="HRC60-68",
            elongation=0.5,  # %
            machinability="poor",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件"],
            blank_hint="砂型铸造",
            heat_treatments=["淬火+回火", "亚临界处理"],
            heat_treatment_notes=["淬火950-1050℃", "回火200-250℃", "可亚临界处理提高韧性"],
            surface_treatments=["喷丸"],
            cutting_speed_range=(3, 8),
            special_tooling=True,
            coolant_required=True,
            warnings=["极难加工", "脆性材料", "需要CBN/陶瓷刀具"],
            recommendations=["矿山磨机衬板", "大型破碎机锤头", "耐磨管道"],
        ),
        description="高铬铸铁，最高硬度耐磨铸铁，矿山重载设备专用",
    ),

    # 蠕墨铸铁 (Vermicular/Compacted Graphite Iron)
    "RuT300": MaterialInfo(
        grade="RuT300",
        name="蠕墨铸铁",
        aliases=["CGI300", "GJV-300", "蠕铁300"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.VERMICULAR_IRON,
        standards=["GB/T 26655-2011", "ISO 16112"],
        properties=MaterialProperties(
            density=7.1,
            melting_point=1180,
            thermal_conductivity=38,
            tensile_strength=300,
            yield_strength=210,
            elongation=3,  # %
            hardness="HB140-200",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件"],
            blank_hint="砂型铸造",
            heat_treatments=["时效处理", "退火"],
            surface_treatments=["发黑", "喷漆"],
            heat_treatment_notes=["时效消除铸造应力"],
            cutting_speed_range=(80, 150),
            recommendations=["发动机缸体", "缸盖", "制动盘", "排气歧管"],
        ),
        description="蠕墨铸铁，综合灰铁和球铁优点，发动机缸体首选",
    ),

    "RuT350": MaterialInfo(
        grade="RuT350",
        name="蠕墨铸铁",
        aliases=["CGI350", "GJV-350", "蠕铁350"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.VERMICULAR_IRON,
        standards=["GB/T 26655-2011", "ISO 16112"],
        properties=MaterialProperties(
            density=7.15,
            melting_point=1180,
            thermal_conductivity=36,
            tensile_strength=350,
            yield_strength=245,
            elongation=2,  # %
            hardness="HB160-220",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件"],
            blank_hint="砂型铸造",
            heat_treatments=["时效处理", "退火", "正火"],
            surface_treatments=["发黑", "喷漆"],
            heat_treatment_notes=["正火可提高强度"],
            cutting_speed_range=(70, 140),
            recommendations=["柴油机缸体", "缸盖", "高强度制动盘"],
        ),
        description="中高强度蠕墨铸铁，柴油机缸体专用",
    ),

    "RuT400": MaterialInfo(
        grade="RuT400",
        name="高强度蠕墨铸铁",
        aliases=["CGI400", "GJV-400", "蠕铁400"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.VERMICULAR_IRON,
        standards=["GB/T 26655-2011", "ISO 16112"],
        properties=MaterialProperties(
            density=7.2,
            melting_point=1180,
            thermal_conductivity=34,
            tensile_strength=400,
            yield_strength=280,
            elongation=1,  # %
            hardness="HB180-250",
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件"],
            blank_hint="砂型铸造",
            heat_treatments=["时效处理", "正火", "等温淬火"],
            surface_treatments=["发黑", "喷漆", "镀层"],
            heat_treatment_notes=["正火可提高强度", "等温淬火可进一步强化"],
            cutting_speed_range=(60, 120),
            warnings=["加工时刀具磨损较快"],
            recommendations=["重型柴油机缸体", "高性能制动盘", "大型机床床身"],
        ),
        description="高强度蠕墨铸铁，重型柴油机和高性能应用",
    ),

    # 可锻铸铁 (Malleable Cast Iron)
    "KTH300-06": MaterialInfo(
        grade="KTH300-06",
        name="黑心可锻铸铁",
        aliases=["B300-06", "黑心铸铁", "可锻铁"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.MALLEABLE_IRON,
        standards=["GB/T 9440-2010", "ISO 5922"],
        properties=MaterialProperties(
            density=7.3,
            melting_point=1150,
            thermal_conductivity=50,
            tensile_strength=300,
            yield_strength=180,
            elongation=6,  # %
            hardness="HB140-180",
            machinability="excellent",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件"],
            blank_hint="砂型铸造+石墨化退火",
            heat_treatments=["石墨化退火"],
            heat_treatment_notes=["石墨化退火900-950℃保温后缓冷"],
            surface_treatments=["发黑", "镀锌", "喷漆"],
            cutting_speed_range=(100, 200),
            recommendations=["管件接头", "阀门", "农机零件", "汽车零件"],
        ),
        description="黑心可锻铸铁，韧性好，适合薄壁复杂件",
    ),

    "KTZ450-06": MaterialInfo(
        grade="KTZ450-06",
        name="珠光体可锻铸铁",
        aliases=["P450-06", "珠光体可锻铁"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.MALLEABLE_IRON,
        standards=["GB/T 9440-2010", "ISO 5922"],
        properties=MaterialProperties(
            density=7.35,
            melting_point=1150,
            thermal_conductivity=45,
            tensile_strength=450,
            yield_strength=270,
            hardness="HB180-230",
            elongation=6.0,
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件"],
            blank_hint="砂型铸造+特殊热处理",
            heat_treatments=["珠光体化退火", "正火"],
            heat_treatment_notes=["石墨化后快冷得珠光体基体"],
            surface_treatments=["发黑", "镀锌", "喷漆"],
            cutting_speed_range=(80, 160),
            recommendations=["连杆", "曲轴", "齿轮", "高强度管件"],
        ),
        description="珠光体可锻铸铁，强度高于黑心可锻铸铁",
    ),

    "KTZ550-04": MaterialInfo(
        grade="KTZ550-04",
        name="高强度珠光体可锻铸铁",
        aliases=["P550-04", "高强可锻铁"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.MALLEABLE_IRON,
        standards=["GB/T 9440-2010", "ISO 5922"],
        properties=MaterialProperties(
            density=7.4,
            melting_point=1150,
            thermal_conductivity=42,
            tensile_strength=550,
            yield_strength=340,
            hardness="HB220-280",
            elongation=4.0,
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件"],
            blank_hint="砂型铸造+特殊热处理",
            heat_treatments=["珠光体化退火", "正火+回火"],
            heat_treatment_notes=["正火后回火提高韧性"],
            surface_treatments=["发黑", "喷漆"],
            cutting_speed_range=(60, 120),
            warnings=["加工时需注意刀具选择"],
            recommendations=["高强度连杆", "重载齿轮", "承力结构件"],
        ),
        description="高强度珠光体可锻铸铁，用于承力结构件",
    ),

    # 铸造镁合金 (Cast Magnesium Alloy)
    "ZM5": MaterialInfo(
        grade="ZM5",
        name="铸造镁铝锌合金",
        aliases=["AZ91D", "MgAl9Zn1", "压铸镁合金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.CAST_MAGNESIUM,
        standards=["GB/T 1177-2018", "ASTM B94"],
        properties=MaterialProperties(
            density=1.81,
            melting_point=595,
            thermal_conductivity=72,
            tensile_strength=230,
            yield_strength=150,
            elongation=6,  # %
            hardness="HB60-75",
            machinability="excellent",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["压铸件", "铸件"],
            blank_hint="压铸成型",
            heat_treatments=["T4 (固溶)", "T6 (固溶+时效)"],
            heat_treatment_notes=["固溶415℃/16h", "时效175℃/16h"],
            surface_treatments=["化学氧化", "微弧氧化", "喷涂"],
            cutting_speed_range=(200, 600),
            warnings=["镁屑易燃", "需要专用切削液", "禁止水基冷却液"],
            recommendations=["汽车变速箱壳", "笔记本外壳", "电子设备壳体"],
        ),
        description="最常用铸造镁合金，轻量化首选",
    ),

    "AM60B": MaterialInfo(
        grade="AM60B",
        name="铸造镁铝锰合金",
        aliases=["MgAl6Mn", "高韧镁合金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.CAST_MAGNESIUM,
        standards=["GB/T 1177-2018", "ASTM B94"],
        properties=MaterialProperties(
            density=1.80,
            melting_point=615,
            thermal_conductivity=62,
            tensile_strength=220,
            yield_strength=130,
            hardness="HB55-70",
            elongation=8.0,
            machinability="excellent",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["压铸件"],
            blank_hint="高压压铸",
            heat_treatments=["F (铸态)"],
            surface_treatments=["化学氧化", "微弧氧化", "喷涂"],
            cutting_speed_range=(200, 600),
            warnings=["镁屑易燃", "需要专用切削液"],
            recommendations=["汽车仪表盘支架", "座椅框架", "方向盘骨架"],
        ),
        description="高韧性铸造镁合金，汽车安全件专用",
    ),

    "AZ63": MaterialInfo(
        grade="AZ63",
        name="铸造镁铝锌合金",
        aliases=["MgAl6Zn3", "砂型镁合金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.CAST_MAGNESIUM,
        standards=["GB/T 1177-2018"],
        properties=MaterialProperties(
            density=1.84,
            melting_point=600,
            thermal_conductivity=83,
            tensile_strength=275,
            yield_strength=130,
            elongation=8,  # %
            hardness="HB55-70",
            machinability="excellent",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件"],
            blank_hint="砂型铸造/金属型铸造",
            heat_treatments=["T4 (固溶)", "T6 (固溶+时效)"],
            heat_treatment_notes=["固溶385℃/10h", "时效220℃/5h"],
            surface_treatments=["化学氧化", "阳极氧化", "喷涂"],
            cutting_speed_range=(200, 600),
            warnings=["镁屑易燃", "热处理需惰性气氛保护"],
            recommendations=["航空机匣", "大型结构件", "工装夹具"],
        ),
        description="砂型铸造镁合金，航空结构件专用",
    ),

    # -------------------------------------------------------------------------
    # 粉末冶金材料 (Powder Metallurgy Materials)
    # -------------------------------------------------------------------------
    "Fe-Cu-C": MaterialInfo(
        grade="Fe-Cu-C",
        name="铁铜碳粉末冶金材料",
        aliases=["FC-0205", "SINT-C11", "铁基粉末冶金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.POWDER_METALLURGY,
        standards=["GB/T 5163-2006", "MPIF Standard 35"],
        properties=MaterialProperties(
            density=6.6,
            melting_point=1150,
            thermal_conductivity=35,
            tensile_strength=350,
            yield_strength=280,
            elongation=2,  # %
            hardness="HRB60-80",
            machinability="good",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["粉末冶金件"],
            blank_hint="压制烧结成型",
            heat_treatments=["渗碳淬火", "蒸汽处理"],
            heat_treatment_notes=["蒸汽处理可提高耐蚀性和硬度"],
            surface_treatments=["浸油", "镀层", "蒸汽处理"],
            cutting_speed_range=(80, 150),
            warnings=["多孔结构需特殊考虑", "避免重切削"],
            recommendations=["齿轮", "含油轴承", "凸轮", "结构件"],
        ),
        description="最常用铁基粉末冶金材料，用于齿轮和结构件",
    ),

    "Fe-Ni-Cu": MaterialInfo(
        grade="Fe-Ni-Cu",
        name="铁镍铜粉末冶金材料",
        aliases=["FN-0205", "SINT-D11", "高强度粉末冶金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.POWDER_METALLURGY,
        standards=["GB/T 5163-2006", "MPIF Standard 35"],
        properties=MaterialProperties(
            density=7.0,
            melting_point=1200,
            thermal_conductivity=32,
            tensile_strength=550,
            yield_strength=400,
            elongation=3,  # %
            hardness="HRB80-95",
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["粉末冶金件"],
            blank_hint="压制烧结成型",
            heat_treatments=["渗碳淬火", "碳氮共渗"],
            heat_treatment_notes=["渗碳后可达HRC58-62"],
            surface_treatments=["浸油", "镀层"],
            cutting_speed_range=(60, 120),
            warnings=["高密度件加工注意刀具磨损"],
            recommendations=["高强度齿轮", "同步器齿套", "传动零件"],
        ),
        description="高强度铁基粉末冶金材料，用于汽车传动件",
    ),

    "316L-PM": MaterialInfo(
        grade="316L-PM",
        name="316L不锈钢粉末冶金",
        aliases=["SS-316L", "MIM-316L", "不锈钢粉末冶金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.POWDER_METALLURGY,
        standards=["GB/T 5163-2006", "MPIF Standard 35"],
        properties=MaterialProperties(
            density=7.6,
            melting_point=1400,
            thermal_conductivity=14,
            tensile_strength=480,
            yield_strength=170,
            elongation=15,  # %
            hardness="HRB70-85",
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["MIM件", "粉末冶金件"],
            blank_hint="金属注射成型(MIM)或压制烧结",
            heat_treatments=["固溶处理"],
            heat_treatment_notes=["烧结后固溶处理提高耐蚀性"],
            surface_treatments=["电解抛光", "钝化"],
            cutting_speed_range=(50, 100),
            warnings=["MIM件尺寸收缩约15-20%", "注意孔隙率控制"],
            recommendations=["医疗器械", "表壳", "手机零件", "精密零件"],
        ),
        description="不锈钢粉末冶金材料，适用于医疗和精密零件",
    ),

    # -------------------------------------------------------------------------
    # 硬质合金 (Cemented Carbide)
    # -------------------------------------------------------------------------
    "YG8": MaterialInfo(
        grade="YG8",
        name="钨钴硬质合金",
        aliases=["K30", "WC-8Co", "ISO K30"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.CEMENTED_CARBIDE,
        standards=["GB/T 5242-2006", "ISO 513"],
        properties=MaterialProperties(
            density=14.7,
            melting_point=2870,
            thermal_conductivity=75,
            hardness="HRA88-89",
            tensile_strength=1500,
            yield_strength=1500,  # 脆性材料，屈服强度≈抗拉强度
            machinability="very_poor",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["烧结坯料", "棒材"],
            blank_hint="粉末冶金烧结",
            heat_treatments=[],
            forbidden_heat_treatments=["淬火", "回火"],
            surface_treatments=["CVD涂层", "PVD涂层"],
            surface_treatment_notes=["涂层可显著提高耐磨性"],
            cutting_speed_range=(5, 20),
            special_tooling=True,
            coolant_required=True,
            warnings=["需金刚石砂轮磨削", "电火花加工", "禁止常规切削"],
            recommendations=["铸铁加工刀具", "冲模", "拉丝模", "耐磨零件"],
        ),
        description="通用型硬质合金，适合铸铁和有色金属加工",
    ),

    "YT15": MaterialInfo(
        grade="YT15",
        name="钨钛钴硬质合金",
        aliases=["P15", "WC-TiC-15Co", "ISO P15"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.CEMENTED_CARBIDE,
        standards=["GB/T 5242-2006", "ISO 513"],
        properties=MaterialProperties(
            density=11.5,
            melting_point=2720,
            thermal_conductivity=42,
            hardness="HRA91-92",
            tensile_strength=1100,
            yield_strength=1100,  # 脆性材料，屈服强度≈抗拉强度
            machinability="very_poor",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["烧结坯料", "刀片"],
            blank_hint="粉末冶金烧结",
            heat_treatments=[],
            forbidden_heat_treatments=["淬火", "回火"],
            surface_treatments=["CVD涂层", "PVD涂层"],
            cutting_speed_range=(5, 15),
            special_tooling=True,
            coolant_required=True,
            warnings=["需金刚石砂轮磨削", "脆性大避免冲击"],
            recommendations=["钢材精加工刀具", "车刀", "镗刀"],
        ),
        description="钢材加工专用硬质合金，抗月牙洼磨损",
    ),

    "YW1": MaterialInfo(
        grade="YW1",
        name="钨钛钽钴硬质合金",
        aliases=["M10", "WC-TiC-TaC-Co", "ISO M10"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.CEMENTED_CARBIDE,
        standards=["GB/T 5242-2006", "ISO 513"],
        properties=MaterialProperties(
            density=12.9,
            melting_point=2780,
            thermal_conductivity=50,
            hardness="HRA90-91",
            tensile_strength=1300,
            yield_strength=1300,  # 脆性材料，屈服强度≈抗拉强度
            machinability="very_poor",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["烧结坯料", "刀片"],
            blank_hint="粉末冶金烧结",
            heat_treatments=[],
            forbidden_heat_treatments=["淬火", "回火"],
            surface_treatments=["CVD涂层", "PVD涂层"],
            cutting_speed_range=(5, 18),
            special_tooling=True,
            coolant_required=True,
            warnings=["需金刚石砂轮磨削", "通用型适用范围广"],
            recommendations=["通用加工刀具", "铣刀", "不锈钢加工"],
        ),
        description="通用型硬质合金，钢和铸铁均可加工",
    ),

    # -------------------------------------------------------------------------
    # 结构陶瓷 (Structural Ceramics)
    # -------------------------------------------------------------------------
    "Al2O3-99": MaterialInfo(
        grade="Al2O3-99",
        name="99氧化铝陶瓷",
        aliases=["99瓷", "高纯氧化铝", "刚玉陶瓷"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.ALUMINA_CERAMIC,
        group=MaterialGroup.STRUCTURAL_CERAMIC,
        standards=["GB/T 5593-2015", "ASTM C799"],
        properties=MaterialProperties(
            density=3.9,
            melting_point=2050,
            thermal_conductivity=30,
            hardness="HV1800-2000",
            tensile_strength=300,
            yield_strength=300,  # 脆性陶瓷，屈服强度≈抗拉强度
            machinability="very_poor",
            weldability="none",
        ),
        process=ProcessRecommendation(
            blank_forms=["烧结坯料", "注浆成型件"],
            blank_hint="干压/注浆成型后烧结",
            heat_treatments=[],
            forbidden_heat_treatments=["淬火", "回火", "退火"],
            surface_treatments=["研磨抛光", "金属化"],
            surface_treatment_notes=["金属化后可钎焊"],
            cutting_speed_range=(5, 15),
            special_tooling=True,
            coolant_required=True,
            warnings=["只能金刚石磨削", "脆性大防止崩边", "热冲击敏感"],
            recommendations=["电子绝缘件", "耐磨衬板", "密封环", "刀具基体"],
        ),
        description="高纯氧化铝陶瓷，绝缘耐磨耐高温",
    ),

    "Si3N4": MaterialInfo(
        grade="Si3N4",
        name="氮化硅陶瓷",
        aliases=["氮化硅", "SRBSN", "GPS-Si3N4"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.NITRIDE_CERAMIC,
        group=MaterialGroup.STRUCTURAL_CERAMIC,
        standards=["GB/T 21944-2008"],
        properties=MaterialProperties(
            density=3.2,
            melting_point=1900,
            thermal_conductivity=20,
            hardness="HV1400-1600",
            tensile_strength=700,
            yield_strength=700,  # 脆性陶瓷，屈服强度≈抗拉强度
            machinability="very_poor",
            weldability="none",
        ),
        process=ProcessRecommendation(
            blank_forms=["烧结坯料", "热压件"],
            blank_hint="热压烧结或气压烧结",
            heat_treatments=[],
            forbidden_heat_treatments=["淬火", "回火"],
            surface_treatments=["研磨抛光"],
            cutting_speed_range=(3, 10),
            special_tooling=True,
            coolant_required=True,
            warnings=["只能金刚石磨削", "成本高", "形状受限"],
            recommendations=["轴承滚珠", "涡轮增压器转子", "切削刀具", "热机部件"],
        ),
        description="氮化硅陶瓷，高强度耐热冲击",
    ),

    "ZrO2-3Y": MaterialInfo(
        grade="ZrO2-3Y",
        name="钇稳定氧化锆陶瓷",
        aliases=["3Y-TZP", "氧化锆陶瓷", "TZP"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.ZIRCONIA_CERAMIC,
        group=MaterialGroup.STRUCTURAL_CERAMIC,
        standards=["GB/T 23806-2009", "ISO 13356"],
        properties=MaterialProperties(
            density=6.05,
            melting_point=2700,
            thermal_conductivity=2.5,
            hardness="HV1200-1300",
            tensile_strength=1000,
            yield_strength=1000,  # 脆性陶瓷，屈服强度≈抗拉强度
            machinability="poor",
            weldability="none",
        ),
        process=ProcessRecommendation(
            blank_forms=["烧结坯料", "注射成型件"],
            blank_hint="CIM注射成型后烧结",
            heat_treatments=[],
            forbidden_heat_treatments=["淬火", "回火"],
            surface_treatments=["研磨抛光", "着色"],
            surface_treatment_notes=["可抛光至镜面"],
            cutting_speed_range=(5, 15),
            special_tooling=True,
            coolant_required=True,
            warnings=["金刚石磨削", "低温相变注意", "成本较高"],
            recommendations=["陶瓷刀", "手表表壳", "牙科材料", "光纤连接器"],
        ),
        description="氧化锆陶瓷，高强度高韧性，可抛光",
    ),

    # -------------------------------------------------------------------------
    # 难熔金属 (Refractory Metals)
    # -------------------------------------------------------------------------
    "Mo-1": MaterialInfo(
        grade="Mo-1",
        name="纯钼",
        aliases=["Mo", "纯钼板", "钼棒", "ASTM B386"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.REFRACTORY_METAL,
        standards=["GB/T 3462-2007", "ASTM B386"],
        properties=MaterialProperties(
            density=10.2,
            melting_point=2620,
            thermal_conductivity=138,
            tensile_strength=550,
            yield_strength=400,
            elongation=15,  # %
            hardness="HV200-250",
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "丝材"],
            blank_hint="粉末冶金/电弧熔炼",
            heat_treatments=["去应力退火", "再结晶退火"],
            heat_treatment_notes=["退火900-1100℃", "惰性气氛或真空保护"],
            surface_treatments=["电镀", "化学镀"],
            cutting_speed_range=(15, 40),
            special_tooling=True,
            coolant_required=True,
            warnings=["高温氧化", "需惰性气氛保护", "脆性温度区间300-800℃"],
            recommendations=["电真空器件", "高温炉发热体", "玻璃熔炼电极"],
        ),
        description="纯钼，高温强度好，电真空器件首选",
    ),

    "TZM": MaterialInfo(
        grade="TZM",
        name="钛锆钼合金",
        aliases=["Mo-TZM", "TZM钼合金", "高温钼合金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.REFRACTORY_METAL,
        standards=["GB/T 3462-2007", "ASTM B387"],
        properties=MaterialProperties(
            density=10.2,
            melting_point=2620,
            thermal_conductivity=126,
            tensile_strength=900,
            yield_strength=700,
            elongation=10,  # %
            hardness="HV280-350",
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "锻件"],
            blank_hint="粉末冶金后锻造",
            heat_treatments=["去应力退火", "再结晶退火"],
            heat_treatment_notes=["退火1200-1400℃", "惰性气氛保护"],
            surface_treatments=["电镀", "涂层"],
            cutting_speed_range=(10, 30),
            special_tooling=True,
            coolant_required=True,
            warnings=["高温氧化严重", "加工需惰性气氛", "成本高"],
            recommendations=["热等静压模具", "高温结构件", "航天发动机部件"],
        ),
        description="高强度钼合金，高温模具和航天用",
    ),

    "W-1": MaterialInfo(
        grade="W-1",
        name="纯钨",
        aliases=["W", "纯钨棒", "钨板", "ASTM B760"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.REFRACTORY_METAL,
        standards=["GB/T 3459-2006", "ASTM B760"],
        properties=MaterialProperties(
            density=19.3,
            melting_point=3410,
            thermal_conductivity=173,
            tensile_strength=800,
            yield_strength=600,
            elongation=2,  # %
            hardness="HV350-450",
            machinability="poor",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "丝材"],
            blank_hint="粉末冶金烧结",
            heat_treatments=["去应力退火"],
            heat_treatment_notes=["退火1000-1200℃", "真空或氢气保护"],
            surface_treatments=["电镀", "化学镀镍"],
            cutting_speed_range=(5, 15),
            special_tooling=True,
            coolant_required=True,
            warnings=["极高硬度", "脆性大", "需金刚石或CBN刀具"],
            recommendations=["电极", "配重块", "辐射屏蔽", "高温炉部件"],
        ),
        description="纯钨，熔点最高金属，密度大",
    ),

    "Ta-1": MaterialInfo(
        grade="Ta-1",
        name="纯钽",
        aliases=["Ta", "钽板", "钽棒", "ASTM B708"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.REFRACTORY_METAL,
        standards=["GB/T 26282-2010", "ASTM B708"],
        properties=MaterialProperties(
            density=16.6,
            melting_point=2996,
            thermal_conductivity=57,
            tensile_strength=400,
            yield_strength=200,
            elongation=25,  # %
            hardness="HV100-150",
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "丝材", "管材"],
            blank_hint="电子束熔炼/粉末冶金",
            heat_treatments=["退火", "去应力退火"],
            heat_treatment_notes=["退火1000-1200℃", "真空保护"],
            surface_treatments=["阳极氧化", "电解抛光"],
            cutting_speed_range=(30, 80),
            coolant_required=True,
            warnings=["成本极高", "氢脆敏感", "需避免氢气污染"],
            recommendations=["电容器", "化工设备", "医疗植入物", "高温炉具"],
        ),
        description="纯钽，耐腐蚀性极佳，医疗和化工用",
    ),

    # -------------------------------------------------------------------------
    # 铝青铜 (Aluminum Bronze)
    # -------------------------------------------------------------------------
    "QAl9-4": MaterialInfo(
        grade="QAl9-4",
        name="铝青铜",
        aliases=["CuAl9Fe4", "C62300", "AB2", "铝铁青铜"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ALUMINUM_BRONZE,
        standards=["GB/T 5231-2012", "ASTM B150"],
        properties=MaterialProperties(
            density=7.6,
            melting_point=1040,
            thermal_conductivity=59,
            tensile_strength=600,
            yield_strength=250,
            elongation=0.5,  # %
            hardness="HB150-180",
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "管材", "铸件"],
            blank_hint="热轧/铸造",
            heat_treatments=["淬火+时效", "退火"],
            heat_treatment_notes=["固溶900℃水淬", "时效400-500℃"],
            surface_treatments=["抛光", "钝化"],
            cutting_speed_range=(40, 100),
            coolant_required=True,
            warnings=["高温时易氧化", "需注意铝青铜专用焊材"],
            recommendations=["船用螺旋桨", "耐蚀阀门", "轴套", "蜗轮"],
        ),
        description="铝青铜，高强度耐磨耐蚀，船舶和阀门用",
    ),

    "QAl10-4-4": MaterialInfo(
        grade="QAl10-4-4",
        name="铝镍铁青铜",
        aliases=["CuAl10Ni5Fe5", "C63000", "NAB", "镍铝青铜"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ALUMINUM_BRONZE,
        standards=["GB/T 5231-2012", "ASTM B150"],
        properties=MaterialProperties(
            density=7.5,
            melting_point=1060,
            thermal_conductivity=42,
            tensile_strength=700,
            yield_strength=320,
            elongation=8,  # %
            hardness="HB180-220",
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "铸件", "锻件"],
            blank_hint="热加工/铸造",
            heat_treatments=["淬火+时效", "去应力退火"],
            heat_treatment_notes=["固溶900℃水淬", "时效500-600℃"],
            surface_treatments=["抛光", "喷涂"],
            cutting_speed_range=(30, 80),
            coolant_required=True,
            warnings=["强度高加工难度大", "需专用焊材"],
            recommendations=["海水泵叶轮", "船舶螺旋桨", "阀座", "海洋平台"],
        ),
        description="镍铝青铜，最强耐海水腐蚀，船舶工业首选",
    ),

    # -------------------------------------------------------------------------
    # 铍铜 (Beryllium Copper)
    # -------------------------------------------------------------------------
    "QBe2": MaterialInfo(
        grade="QBe2",
        name="铍青铜",
        aliases=["CuBe2", "C17200", "铍铜", "BeCu"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.BERYLLIUM_COPPER,
        standards=["GB/T 5231-2012", "ASTM B194"],
        properties=MaterialProperties(
            density=8.3,
            melting_point=870,
            thermal_conductivity=115,
            tensile_strength=1250,
            yield_strength=1100,
            elongation=0.5,  # %
            hardness="HRC38-42",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "带材", "棒材", "丝材"],
            blank_hint="热轧+冷轧",
            heat_treatments=["固溶处理", "时效硬化"],
            heat_treatment_notes=["固溶780-800℃水淬", "时效315-345℃/2-3h"],
            surface_treatments=["电镀", "钝化", "化学镀镍"],
            cutting_speed_range=(60, 150),
            coolant_required=True,
            warnings=["铍粉有毒需防护", "加工需局部排风", "禁止干磨"],
            recommendations=["弹簧接触件", "防爆工具", "模具镶块", "电子连接器"],
        ),
        description="铍铜，强度最高的铜合金，弹性和导电性优良",
    ),

    "QBe1.9": MaterialInfo(
        grade="QBe1.9",
        name="低铍铜",
        aliases=["CuBe1.9", "C17000", "低铍青铜"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.BERYLLIUM_COPPER,
        standards=["GB/T 5231-2012", "ASTM B194"],
        properties=MaterialProperties(
            density=8.3,
            melting_point=870,
            thermal_conductivity=106,
            tensile_strength=1000,
            yield_strength=850,
            elongation=4,  # %
            hardness="HRC32-38",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "带材", "棒材"],
            blank_hint="热轧+冷轧",
            heat_treatments=["固溶处理", "时效硬化"],
            heat_treatment_notes=["固溶760-780℃水淬", "时效300-330℃/2-3h"],
            surface_treatments=["电镀", "钝化"],
            cutting_speed_range=(60, 150),
            coolant_required=True,
            warnings=["铍粉有毒需防护", "加工需局部排风"],
            recommendations=["弹性元件", "导电弹簧", "插座端子"],
        ),
        description="低铍铜，成本低于QBe2，中等强度弹性应用",
    ),

    "CuNi2Si": MaterialInfo(
        grade="CuNi2Si",
        name="镍硅铜",
        aliases=["C70250", "铜镍硅", "无铍铜"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.BERYLLIUM_COPPER,
        standards=["GB/T 5231-2012", "ASTM B422"],
        properties=MaterialProperties(
            density=8.8,
            melting_point=1080,
            thermal_conductivity=190,
            tensile_strength=700,
            yield_strength=600,
            elongation=12,  # %
            hardness="HV200-250",
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "带材", "棒材"],
            blank_hint="热轧+冷轧",
            heat_treatments=["固溶处理", "时效硬化"],
            heat_treatment_notes=["固溶800-850℃水淬", "时效450-500℃"],
            surface_treatments=["电镀", "钝化", "镀锡"],
            cutting_speed_range=(80, 180),
            coolant_required=False,
            warnings=["强度低于铍铜", "无毒替代方案"],
            recommendations=["替代铍铜", "引线框架", "连接器端子", "继电器弹片"],
        ),
        description="镍硅铜，铍铜无毒替代品，电子连接器用",
    ),

    # -------------------------------------------------------------------------
    # 焊锡/无铅焊料 (Lead-Free Solder)
    # -------------------------------------------------------------------------
    "SAC305": MaterialInfo(
        grade="SAC305",
        name="无铅焊锡",
        aliases=["Sn96.5Ag3.0Cu0.5", "SAC 305", "Sn-3.0Ag-0.5Cu"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.SOLDER,
        standards=["IPC J-STD-006", "JIS Z 3282"],
        properties=MaterialProperties(
            density=7.37,
            melting_point=217,
            tensile_strength=45,
            yield_strength=35,
            elongation=45,  # %
            hardness="HB15",
            machinability="excellent",
            weldability="excellent",
            thermal_conductivity=58,
        ),
        process=ProcessRecommendation(
            blank_forms=["丝材", "条材", "球状", "膏状"],
            blank_hint="丝材/膏状",
            heat_treatments=["无"],
            surface_treatments=["无"],
            cutting_speed_range=(100, 300),
            coolant_required=False,
            warnings=["熔点比有铅焊锡高", "润湿性稍差", "需要助焊剂"],
            recommendations=["电子元器件焊接", "SMT回流焊", "波峰焊", "手工焊接"],
        ),
        description="无铅焊锡，RoHS合规，电子行业主流焊料",
    ),
    "SAC387": MaterialInfo(
        grade="SAC387",
        name="高银无铅焊锡",
        aliases=["Sn95.5Ag3.8Cu0.7", "SAC 387", "Sn-3.8Ag-0.7Cu"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.SOLDER,
        standards=["IPC J-STD-006", "JIS Z 3282"],
        properties=MaterialProperties(
            density=7.40,
            melting_point=217,
            tensile_strength=50,
            yield_strength=40,
            elongation=40,  # %
            hardness="HB16",
            machinability="excellent",
            weldability="excellent",
            thermal_conductivity=60,
        ),
        process=ProcessRecommendation(
            blank_forms=["丝材", "条材", "球状", "膏状"],
            blank_hint="丝材/膏状",
            heat_treatments=["无"],
            surface_treatments=["无"],
            cutting_speed_range=(100, 300),
            coolant_required=False,
            warnings=["银含量较高成本高", "熔点比有铅焊锡高"],
            recommendations=["高可靠性焊接", "BGA封装", "汽车电子", "航空航天电子"],
        ),
        description="高银无铅焊锡，可靠性高，BGA和高端电子用",
    ),
    "Sn99.3Cu0.7": MaterialInfo(
        grade="Sn99.3Cu0.7",
        name="无银无铅焊锡",
        aliases=["Sn-0.7Cu", "SN100C", "SnCu0.7"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.SOLDER,
        standards=["IPC J-STD-006", "JIS Z 3282"],
        properties=MaterialProperties(
            density=7.30,
            melting_point=227,
            tensile_strength=35,
            yield_strength=28,
            elongation=50,  # %
            hardness="HB12",
            machinability="excellent",
            weldability="excellent",
            thermal_conductivity=55,
        ),
        process=ProcessRecommendation(
            blank_forms=["丝材", "条材", "球状"],
            blank_hint="丝材/条材",
            heat_treatments=["无"],
            surface_treatments=["无"],
            cutting_speed_range=(100, 300),
            coolant_required=False,
            warnings=["熔点最高", "润湿性较差", "需要更高焊接温度"],
            recommendations=["波峰焊", "通孔元器件焊接", "成本敏感应用"],
        ),
        description="无银无铅焊锡，成本低，波峰焊和通孔焊接用",
    ),

    # -------------------------------------------------------------------------
    # 钎焊合金 (Brazing Alloy)
    # -------------------------------------------------------------------------
    "BAg-5": MaterialInfo(
        grade="BAg-5",
        name="银钎料",
        aliases=["HL302", "45%银钎料", "AWS A5.8 BAg-5"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.BRAZING_ALLOY,
        standards=["GB/T 10046", "AWS A5.8"],
        properties=MaterialProperties(
            density=9.30,
            melting_point=690,
            thermal_conductivity=72,
            tensile_strength=350,
            yield_strength=200,
            elongation=25,  # %
            hardness="HB90",
            machinability="good",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["丝材", "片材", "环材", "膏状"],
            blank_hint="丝材/片材",
            heat_treatments=["无"],
            surface_treatments=["无"],
            cutting_speed_range=(80, 200),
            coolant_required=False,
            warnings=["含镉注意安全", "需要助焊剂", "加热时注意通风"],
            recommendations=["铜合金钎焊", "钢铁钎焊", "硬质合金钎焊", "工具制造"],
        ),
        description="45%银钎料，流动性好，工具和硬质合金钎焊用",
    ),
    "BCu-1": MaterialInfo(
        grade="BCu-1",
        name="纯铜钎料",
        aliases=["HL101", "99.9%Cu钎料", "AWS A5.8 BCu-1"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.BRAZING_ALLOY,
        standards=["GB/T 10046", "AWS A5.8"],
        properties=MaterialProperties(
            density=8.94,
            melting_point=1083,
            thermal_conductivity=391,
            tensile_strength=220,
            yield_strength=70,
            elongation=35,  # %
            hardness="HB45",
            machinability="excellent",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["丝材", "片材", "粉末"],
            blank_hint="丝材/片材",
            heat_treatments=["无"],
            surface_treatments=["无"],
            cutting_speed_range=(100, 250),
            coolant_required=False,
            warnings=["需要还原性或惰性气氛", "熔点高", "不适合铜钎焊"],
            recommendations=["钢铁钎焊", "真空钎焊", "保护气氛钎焊", "炉中钎焊"],
        ),
        description="纯铜钎料，用于钢铁在保护气氛或真空中钎焊",
    ),
    "BNi-2": MaterialInfo(
        grade="BNi-2",
        name="镍基钎料",
        aliases=["HL401", "Ni-7Cr-3B-4.5Si-3Fe", "AWS A5.8 BNi-2"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.BRAZING_ALLOY,
        standards=["GB/T 10046", "AWS A5.8"],
        properties=MaterialProperties(
            density=7.90,
            melting_point=1000,
            thermal_conductivity=15,
            tensile_strength=450,
            yield_strength=280,
            elongation=3,  # %
            hardness="HRC35",
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["粉末", "膏状", "片材", "箔材"],
            blank_hint="粉末/膏状",
            heat_treatments=["无"],
            surface_treatments=["无"],
            cutting_speed_range=(30, 80),
            coolant_required=True,
            warnings=["需要真空或惰性气氛", "钎焊温度高", "脆性大"],
            recommendations=["不锈钢钎焊", "高温合金钎焊", "航空航天钎焊", "真空钎焊"],
        ),
        description="镍基钎料，耐高温耐蚀，航空航天和不锈钢钎焊用",
    ),

    # -------------------------------------------------------------------------
    # 形状记忆合金 (Shape Memory Alloy)
    # -------------------------------------------------------------------------
    "NiTi": MaterialInfo(
        grade="NiTi",
        name="镍钛记忆合金",
        aliases=["Nitinol", "NiTi-55", "TiNi", "形状记忆合金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.SHAPE_MEMORY_ALLOY,
        standards=["ASTM F2063", "GB/T 24627"],
        properties=MaterialProperties(
            density=6.45,
            melting_point=1310,
            thermal_conductivity=18,
            tensile_strength=900,
            yield_strength=200,
            elongation=10,  # %
            hardness="HRC40",
            machinability="poor",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["丝材", "棒材", "管材", "板材"],
            blank_hint="丝材/棒材",
            heat_treatments=["时效处理", "形状记忆训练"],
            surface_treatments=["电解抛光", "钝化处理"],
            cutting_speed_range=(10, 30),
            special_tooling=True,
            coolant_required=True,
            warnings=["加工硬化严重", "切削力大", "需要专用刀具", "热处理影响性能"],
            recommendations=["医疗器械", "眼镜框", "驱动器", "航空航天紧固件"],
        ),
        description="镍钛形状记忆合金，超弹性和形状记忆效应，医疗和航空用",
    ),
    "CuZnAl": MaterialInfo(
        grade="CuZnAl",
        name="铜锌铝记忆合金",
        aliases=["Cu-Zn-Al", "CuZnAl-SMA", "铜基记忆合金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.SHAPE_MEMORY_ALLOY,
        standards=["企业标准"],
        properties=MaterialProperties(
            density=7.80,
            melting_point=950,
            thermal_conductivity=120,
            tensile_strength=500,
            yield_strength=150,
            elongation=5,  # %
            hardness="HB180",
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["丝材", "棒材", "板材"],
            blank_hint="丝材/板材",
            heat_treatments=["固溶处理", "时效处理", "形状记忆训练"],
            surface_treatments=["钝化", "镀层"],
            cutting_speed_range=(30, 80),
            coolant_required=True,
            warnings=["相变温度范围宽", "疲劳寿命有限", "时效不稳定"],
            recommendations=["管道接头", "温控阀门", "电气连接器", "消费电子"],
        ),
        description="铜锌铝记忆合金，成本低于NiTi，民用记忆合金应用",
    ),
    "CuAlNi": MaterialInfo(
        grade="CuAlNi",
        name="铜铝镍记忆合金",
        aliases=["Cu-Al-Ni", "CuAlNi-SMA", "高温铜基记忆合金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.SHAPE_MEMORY_ALLOY,
        standards=["企业标准"],
        properties=MaterialProperties(
            density=7.20,
            melting_point=1050,
            thermal_conductivity=75,
            tensile_strength=600,
            yield_strength=180,
            elongation=4,  # %
            hardness="HB200",
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["丝材", "棒材", "板材"],
            blank_hint="棒材/板材",
            heat_treatments=["固溶处理", "时效处理", "形状记忆训练"],
            surface_treatments=["钝化", "镀层"],
            cutting_speed_range=(25, 70),
            coolant_required=True,
            warnings=["脆性较大", "加工需小心", "相变温度高"],
            recommendations=["高温驱动器", "工业阀门", "高温传感器", "航空航天应用"],
        ),
        description="铜铝镍记忆合金，相变温度高，高温记忆合金应用",
    ),

    # -------------------------------------------------------------------------
    # 电触头材料 (Electrical Contact Material)
    # -------------------------------------------------------------------------
    "AgCdO": MaterialInfo(
        grade="AgCdO",
        name="银氧化镉触点",
        aliases=["Ag/CdO", "银镉触点", "AgCdO-12"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ELECTRICAL_CONTACT,
        standards=["GB/T 5235", "IEC 60056"],
        properties=MaterialProperties(
            density=10.30,
            melting_point=960,
            thermal_conductivity=360,
            tensile_strength=300,
            yield_strength=200,
            elongation=0.5,  # %
            hardness="HB70",
            machinability="good",
            weldability="fair",
            conductivity=85,
        ),
        process=ProcessRecommendation(
            blank_forms=["铆钉", "片材", "触点"],
            blank_hint="铆钉/触点",
            heat_treatments=["内氧化处理"],
            surface_treatments=["无"],
            cutting_speed_range=(80, 200),
            coolant_required=False,
            warnings=["含镉有毒", "需要防护", "逐步被替代"],
            recommendations=["低压电器触点", "继电器触点", "接触器触点"],
        ),
        description="银氧化镉触点，导电性好抗电弧，低压电器用（含镉有毒）",
    ),
    "AgSnO2": MaterialInfo(
        grade="AgSnO2",
        name="银氧化锡触点",
        aliases=["Ag/SnO2", "银锡触点", "AgSnO2-12", "无镉触点"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ELECTRICAL_CONTACT,
        standards=["GB/T 5235", "IEC 60056"],
        properties=MaterialProperties(
            density=10.10,
            melting_point=960,
            thermal_conductivity=340,
            tensile_strength=280,
            yield_strength=180,
            elongation=0.5,  # %
            hardness="HB75",
            machinability="good",
            weldability="fair",
            conductivity=80,
        ),
        process=ProcessRecommendation(
            blank_forms=["铆钉", "片材", "触点"],
            blank_hint="铆钉/触点",
            heat_treatments=["内氧化处理"],
            surface_treatments=["无"],
            cutting_speed_range=(80, 200),
            coolant_required=False,
            warnings=["抗熔焊性略差", "需要优化设计"],
            recommendations=["环保替代AgCdO", "低压电器触点", "汽车继电器"],
        ),
        description="银氧化锡触点，环保无镉，替代AgCdO的主流触点材料",
    ),
    "CuW": MaterialInfo(
        grade="CuW",
        name="钨铜合金",
        aliases=["Cu-W", "WCu", "CuW70", "W70Cu30"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ELECTRICAL_CONTACT,
        standards=["GB/T 8320", "ASTM B702"],
        properties=MaterialProperties(
            density=14.50,
            melting_point=1083,
            thermal_conductivity=180,
            tensile_strength=500,
            yield_strength=350,
            elongation=2,  # %
            hardness="HB200",
            machinability="fair",
            weldability="poor",
            conductivity=45,
        ),
        process=ProcessRecommendation(
            blank_forms=["块材", "棒材", "板材"],
            blank_hint="块材/棒材",
            heat_treatments=["无"],
            surface_treatments=["无"],
            cutting_speed_range=(20, 60),
            special_tooling=True,
            coolant_required=True,
            warnings=["硬度高难加工", "粉末冶金制备", "成本较高"],
            recommendations=["高压开关触头", "电火花电极", "热沉材料", "电阻焊电极"],
        ),
        description="钨铜合金，耐电弧耐磨，高压开关和电火花加工电极用",
    ),

    # -------------------------------------------------------------------------
    # 轴承合金 (Bearing Alloy)
    # -------------------------------------------------------------------------
    "ZChSnSb11-6": MaterialInfo(
        grade="ZChSnSb11-6",
        name="锡基巴氏合金",
        aliases=["Babbitt", "巴氏合金", "SnSb11Cu6", "白合金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.BEARING_ALLOY,
        standards=["GB/T 1174", "ASTM B23"],
        properties=MaterialProperties(
            density=7.38,
            melting_point=240,
            thermal_conductivity=55,
            tensile_strength=90,
            yield_strength=60,
            elongation=5,  # %
            hardness="HB25",
            machinability="excellent",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件", "复合带材"],
            blank_hint="铸造/复合",
            heat_treatments=["无"],
            surface_treatments=["无"],
            cutting_speed_range=(100, 300),
            coolant_required=False,
            warnings=["承载能力有限", "需要良好润滑", "不耐高温"],
            recommendations=["汽轮机轴承", "压缩机轴承", "大型电机轴承", "低速重载轴承"],
        ),
        description="锡基巴氏合金，自润滑减摩，大型旋转机械滑动轴承用",
    ),
    "ZChPbSb16-16-2": MaterialInfo(
        grade="ZChPbSb16-16-2",
        name="铅基巴氏合金",
        aliases=["PbSb16Sn16Cu2", "铅基轴承合金", "16-16-2"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.BEARING_ALLOY,
        standards=["GB/T 1174", "ASTM B23"],
        properties=MaterialProperties(
            density=9.50,
            melting_point=280,
            thermal_conductivity=23,
            tensile_strength=75,
            yield_strength=50,
            elongation=3,  # %
            hardness="HB20",
            machinability="excellent",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件", "复合带材"],
            blank_hint="铸造/复合",
            heat_treatments=["无"],
            surface_treatments=["无"],
            cutting_speed_range=(100, 300),
            coolant_required=False,
            warnings=["含铅有毒", "承载能力低于锡基", "需要防护"],
            recommendations=["低速轴承", "一般机械轴承", "成本敏感应用"],
        ),
        description="铅基巴氏合金，成本低于锡基，一般机械滑动轴承用",
    ),
    "CuPb24Sn4": MaterialInfo(
        grade="CuPb24Sn4",
        name="铜铅合金轴承",
        aliases=["铅青铜轴承", "CuPb24Sn", "SAE49"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.BEARING_ALLOY,
        standards=["GB/T 1176", "SAE J460"],
        properties=MaterialProperties(
            density=9.70,
            melting_point=950,
            thermal_conductivity=42,
            tensile_strength=180,
            yield_strength=100,
            elongation=8,  # %
            hardness="HB45",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["双金属带材", "轴瓦"],
            blank_hint="双金属/轴瓦",
            heat_treatments=["无"],
            surface_treatments=["镀层"],
            cutting_speed_range=(60, 150),
            coolant_required=True,
            warnings=["含铅有毒", "需要钢背衬", "需要良好润滑"],
            recommendations=["发动机轴瓦", "高速重载轴承", "连杆轴瓦", "曲轴轴承"],
        ),
        description="铜铅合金轴承，高速重载，发动机主轴瓦和连杆轴瓦用",
    ),

    # -------------------------------------------------------------------------
    # 热电偶合金 (Thermocouple Alloy)
    # -------------------------------------------------------------------------
    "Chromel": MaterialInfo(
        grade="Chromel",
        name="镍铬合金",
        aliases=["NiCr10", "Chromel-P", "K型热电偶正极"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.THERMOCOUPLE_ALLOY,
        standards=["IEC 60584", "GB/T 4994"],
        properties=MaterialProperties(
            density=8.50,
            melting_point=1400,
            thermal_conductivity=19,
            tensile_strength=650,
            yield_strength=300,
            elongation=20,  # %
            hardness="HB170",
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["丝材", "带材"],
            blank_hint="丝材",
            heat_treatments=["退火"],
            surface_treatments=["无"],
            cutting_speed_range=(30, 80),
            coolant_required=True,
            warnings=["高温氧化", "需要保护管", "配对使用"],
            recommendations=["K型热电偶正极", "高温测温", "工业炉温度控制"],
        ),
        description="镍铬合金，K型热电偶正极材料，工业高温测温用",
    ),
    "Alumel": MaterialInfo(
        grade="Alumel",
        name="镍铝合金",
        aliases=["NiAl3", "Alumel", "K型热电偶负极"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.THERMOCOUPLE_ALLOY,
        standards=["IEC 60584", "GB/T 4994"],
        properties=MaterialProperties(
            density=8.60,
            melting_point=1400,
            thermal_conductivity=30,
            tensile_strength=600,
            yield_strength=280,
            elongation=25,  # %
            hardness="HB160",
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["丝材", "带材"],
            blank_hint="丝材",
            heat_treatments=["退火"],
            surface_treatments=["无"],
            cutting_speed_range=(30, 80),
            coolant_required=True,
            warnings=["需要与Chromel配对", "避免还原气氛"],
            recommendations=["K型热电偶负极", "高温测温", "工业炉温度控制"],
        ),
        description="镍铝合金，K型热电偶负极材料，与Chromel配对使用",
    ),
    "Constantan": MaterialInfo(
        grade="Constantan",
        name="康铜",
        aliases=["CuNi44", "Constantan", "6J40", "J/T型热电偶负极"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.THERMOCOUPLE_ALLOY,
        standards=["IEC 60584", "GB/T 4994"],
        properties=MaterialProperties(
            density=8.90,
            melting_point=1260,
            thermal_conductivity=22,
            tensile_strength=500,
            yield_strength=200,
            elongation=45,  # %
            hardness="HB130",
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["丝材", "带材", "电阻片"],
            blank_hint="丝材/带材",
            heat_treatments=["退火"],
            surface_treatments=["无"],
            cutting_speed_range=(40, 100),
            coolant_required=True,
            warnings=["电阻温度系数极低", "耐蚀性好"],
            recommendations=["J/T型热电偶负极", "精密电阻", "应变片", "补偿导线"],
        ),
        description="康铜（铜镍合金），电阻温度系数极低，热电偶和精密电阻用",
    ),

    # -------------------------------------------------------------------------
    # 永磁材料 (Permanent Magnet)
    # -------------------------------------------------------------------------
    "NdFeB": MaterialInfo(
        grade="NdFeB",
        name="钕铁硼永磁",
        aliases=["N35", "N42", "N52", "稀土永磁", "钕磁铁"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.PERMANENT_MAGNET,
        standards=["GB/T 13560", "IEC 60404-8-1"],
        properties=MaterialProperties(
            density=7.50,
            melting_point=1150,
            thermal_conductivity=9,
            tensile_strength=80,
            yield_strength=None,  # 脆性永磁材料，无屈服点
            elongation=0.5,  # %
            hardness="HV600",
            machinability="poor",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["烧结块", "粘结块", "异形件"],
            blank_hint="烧结/粘结",
            heat_treatments=["无"],
            surface_treatments=["镀镍", "镀锌", "环氧涂层"],
            cutting_speed_range=(5, 20),
            special_tooling=True,
            coolant_required=True,
            warnings=["脆性大易碎", "易氧化腐蚀", "需要防护涂层", "高温退磁"],
            recommendations=["电机转子", "风力发电机", "电动汽车", "音响扬声器"],
        ),
        description="钕铁硼永磁体，磁性最强，电机和电子器件用",
    ),
    "SmCo": MaterialInfo(
        grade="SmCo",
        name="钐钴永磁",
        aliases=["SmCo5", "Sm2Co17", "钐钴磁铁", "1:5钐钴", "2:17钐钴"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.PERMANENT_MAGNET,
        standards=["GB/T 13560", "IEC 60404-8-1"],
        properties=MaterialProperties(
            density=8.40,
            melting_point=1300,
            thermal_conductivity=11,
            tensile_strength=50,
            yield_strength=None,  # 脆性永磁材料，无屈服点
            hardness="HV550",
            elongation=0.5,  # %
            machinability="poor",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["烧结块", "异形件"],
            blank_hint="烧结",
            heat_treatments=["无"],
            surface_treatments=["无需涂层"],
            cutting_speed_range=(5, 15),
            special_tooling=True,
            coolant_required=True,
            warnings=["极脆易碎", "成本高", "钐易燃"],
            recommendations=["高温电机", "航空航天", "军工电子", "高温传感器"],
        ),
        description="钐钴永磁体，耐高温耐腐蚀，航空航天和高温应用",
    ),
    "Alnico": MaterialInfo(
        grade="Alnico",
        name="铝镍钴永磁",
        aliases=["AlNiCo5", "AlNiCo8", "LNG52", "铝镍钴磁铁"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.PERMANENT_MAGNET,
        standards=["GB/T 13560", "IEC 60404-8-1"],
        properties=MaterialProperties(
            density=7.30,
            melting_point=1350,
            thermal_conductivity=10,
            tensile_strength=40,
            yield_strength=None,  # 脆性永磁材料，无屈服点
            hardness="HRC45",
            elongation=0.5,  # %
            machinability="poor",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸造", "烧结"],
            blank_hint="铸造/烧结",
            heat_treatments=["磁场热处理"],
            surface_treatments=["无"],
            cutting_speed_range=(10, 30),
            special_tooling=True,
            coolant_required=True,
            warnings=["脆性大", "易退磁", "矫顽力低"],
            recommendations=["仪表电机", "传感器", "教学磁铁", "高温简单应用"],
        ),
        description="铝镍钴永磁体，温度稳定性好，仪表和传感器用",
    ),

    # -------------------------------------------------------------------------
    # 电阻合金 (Resistance Alloy)
    # -------------------------------------------------------------------------
    "Cr20Ni80": MaterialInfo(
        grade="Cr20Ni80",
        name="镍铬电热合金",
        aliases=["Nichrome", "Ni80Cr20", "电炉丝", "OCr20Ni80"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.RESISTANCE_ALLOY,
        standards=["GB/T 1234", "IEC 60404-8-5"],
        properties=MaterialProperties(
            density=8.40,
            melting_point=1400,
            thermal_conductivity=14,
            tensile_strength=650,
            yield_strength=300,
            elongation=20,  # %
            hardness="HB200",
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["丝材", "带材", "棒材"],
            blank_hint="丝材/带材",
            heat_treatments=["退火"],
            surface_treatments=["无"],
            cutting_speed_range=(30, 80),
            coolant_required=True,
            warnings=["高温氧化", "需要通风", "电阻随温度变化"],
            recommendations=["电热元件", "电炉丝", "电阻器", "工业加热"],
        ),
        description="镍铬电热合金，耐高温氧化，电加热元件和电炉用",
    ),
    "Manganin": MaterialInfo(
        grade="Manganin",
        name="锰铜合金",
        aliases=["6J13", "CuMn12Ni", "锰铜电阻"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.RESISTANCE_ALLOY,
        standards=["GB/T 5235", "IEC 60404-8-5"],
        properties=MaterialProperties(
            density=8.40,
            melting_point=1020,
            thermal_conductivity=22,
            tensile_strength=450,
            yield_strength=200,
            elongation=40,  # %
            hardness="HB120",
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["丝材", "带材", "电阻片"],
            blank_hint="丝材/带材",
            heat_treatments=["稳定化退火"],
            surface_treatments=["无"],
            cutting_speed_range=(40, 100),
            coolant_required=True,
            warnings=["温度系数极低", "需要老化稳定"],
            recommendations=["精密电阻", "标准电阻", "分流器", "电桥"],
        ),
        description="锰铜合金，电阻温度系数极低，精密电阻和标准电阻用",
    ),
    "Karma": MaterialInfo(
        grade="Karma",
        name="卡玛合金",
        aliases=["Ni80Cr20Al", "电阻应变合金", "Karma合金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.RESISTANCE_ALLOY,
        standards=["企业标准"],
        properties=MaterialProperties(
            density=8.10,
            melting_point=1350,
            thermal_conductivity=13,
            tensile_strength=700,
            yield_strength=350,
            elongation=3,  # %
            hardness="HB220",
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["箔材", "丝材"],
            blank_hint="箔材",
            heat_treatments=["稳定化退火"],
            surface_treatments=["无"],
            cutting_speed_range=(30, 70),
            coolant_required=True,
            warnings=["加工精度要求高", "需要严格热处理"],
            recommendations=["应变片", "力传感器", "称重传感器", "精密测量"],
        ),
        description="卡玛合金，应变灵敏系数稳定，高精度应变片用",
    ),

    # -------------------------------------------------------------------------
    # 低膨胀合金 (Low Expansion Alloy)
    # -------------------------------------------------------------------------
    "Invar": MaterialInfo(
        grade="Invar",
        name="因瓦合金",
        aliases=["4J36", "Fe-Ni36", "殷钢", "低膨胀合金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.LOW_EXPANSION_ALLOY,
        standards=["GB/T 15018", "ASTM F1684"],
        properties=MaterialProperties(
            density=8.10,
            melting_point=1430,
            thermal_conductivity=13,
            tensile_strength=500,
            yield_strength=280,
            elongation=35,  # %
            hardness="HB160",
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "带材", "丝材"],
            blank_hint="棒材/板材",
            heat_treatments=["稳定化退火", "消除应力"],
            surface_treatments=["镀镍", "钝化"],
            cutting_speed_range=(30, 80),
            coolant_required=True,
            warnings=["热膨胀系数极低", "需要稳定化处理", "磁性"],
            recommendations=["精密仪器", "激光器结构", "标准尺", "卫星天线"],
        ),
        description="因瓦合金，热膨胀系数极低，精密仪器和光学器件用",
    ),
    "Kovar": MaterialInfo(
        grade="Kovar",
        name="可伐合金",
        aliases=["4J29", "Fe-Ni-Co", "玻封合金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.LOW_EXPANSION_ALLOY,
        standards=["GB/T 15018", "ASTM F15"],
        properties=MaterialProperties(
            density=8.30,
            melting_point=1450,
            thermal_conductivity=17,
            tensile_strength=520,
            yield_strength=300,
            elongation=25,  # %
            hardness="HB170",
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "带材", "管材", "丝材"],
            blank_hint="带材/管材",
            heat_treatments=["退火", "氧化处理"],
            surface_treatments=["氧化", "镀金"],
            cutting_speed_range=(30, 80),
            coolant_required=True,
            warnings=["膨胀系数与玻璃匹配", "需要氧化处理", "磁性"],
            recommendations=["电子管封接", "集成电路封装", "真空器件", "传感器壳体"],
        ),
        description="可伐合金，热膨胀与玻璃/陶瓷匹配，电子封装用",
    ),
    "4J32": MaterialInfo(
        grade="4J32",
        name="超因瓦合金",
        aliases=["Super Invar", "Fe-Ni-Co32", "超低膨胀合金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.LOW_EXPANSION_ALLOY,
        standards=["GB/T 15018", "ASTM F1684"],
        properties=MaterialProperties(
            density=8.15,
            melting_point=1425,
            thermal_conductivity=12,
            tensile_strength=480,
            yield_strength=260,
            elongation=30,  # %
            hardness="HB150",
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "环材"],
            blank_hint="棒材/板材",
            heat_treatments=["稳定化退火", "深冷处理"],
            surface_treatments=["镀镍", "钝化"],
            cutting_speed_range=(30, 75),
            coolant_required=True,
            warnings=["膨胀系数比Invar更低", "需要严格稳定化", "成本高"],
            recommendations=["激光陀螺仪", "航天结构", "精密光学", "天文望远镜"],
        ),
        description="超因瓦合金，热膨胀系数比Invar更低，航天和精密光学用",
    ),

    # -------------------------------------------------------------------------
    # 超导材料 (Superconductor)
    # -------------------------------------------------------------------------
    "NbTi": MaterialInfo(
        grade="NbTi",
        name="铌钛超导合金",
        aliases=["Nb-Ti", "NbTi47", "低温超导"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.SUPERCONDUCTOR,
        standards=["ASTM B884", "企业标准"],
        properties=MaterialProperties(
            density=6.50,
            melting_point=1950,
            thermal_conductivity=8,
            tensile_strength=800,
            yield_strength=500,
            elongation=20,  # %
            hardness="HV250",
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["丝材", "棒材", "复合线材"],
            blank_hint="复合线材",
            heat_treatments=["退火", "热处理优化"],
            surface_treatments=["无"],
            cutting_speed_range=(20, 60),
            coolant_required=True,
            warnings=["需要液氦冷却", "Tc约9K", "复合加工"],
            recommendations=["MRI磁体", "粒子加速器", "磁悬浮", "科研磁体"],
        ),
        description="铌钛超导合金，低温超导体，MRI和粒子加速器用",
    ),
    "Nb3Sn": MaterialInfo(
        grade="Nb3Sn",
        name="铌三锡超导体",
        aliases=["Nb-3Sn", "A15超导体", "高场超导"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.SUPERCONDUCTOR,
        standards=["ITER标准", "企业标准"],
        properties=MaterialProperties(
            density=8.90,
            melting_point=2130,
            thermal_conductivity=3,
            tensile_strength=200,
            yield_strength=None,  # 脆性金属间化合物超导体，无屈服点
            elongation=0.5,  # %
            hardness="HV800",
            machinability="poor",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["复合线材", "带材"],
            blank_hint="复合线材",
            heat_treatments=["反应热处理650-700℃"],
            surface_treatments=["无"],
            cutting_speed_range=(10, 30),
            special_tooling=True,
            coolant_required=True,
            warnings=["脆性大", "需要液氦冷却", "Tc约18K", "复杂制备"],
            recommendations=["高场磁体", "ITER聚变堆", "NMR磁体", "高能物理"],
        ),
        description="铌三锡超导体，高临界磁场，核聚变和高场磁体用",
    ),
    "YBCO": MaterialInfo(
        grade="YBCO",
        name="钇钡铜氧超导体",
        aliases=["YBa2Cu3O7", "高温超导", "REBCO", "二代高温超导"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.CERAMIC,
        group=MaterialGroup.SUPERCONDUCTOR,
        standards=["企业标准"],
        properties=MaterialProperties(
            density=6.30,
            melting_point=1015,
            thermal_conductivity=4,
            tensile_strength=150,
            yield_strength=None,  # 脆性陶瓷超导体，无屈服点
            elongation=0.5,  # %
            hardness="HV600",
            machinability="poor",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["涂层导体", "薄膜", "块材"],
            blank_hint="涂层导体",
            heat_treatments=["高温烧结", "氧退火"],
            surface_treatments=["无"],
            cutting_speed_range=(5, 20),
            special_tooling=True,
            coolant_required=True,
            warnings=["陶瓷脆性", "需要液氮冷却", "Tc约93K", "各向异性"],
            recommendations=["高温超导电缆", "超导电机", "磁悬浮列车", "医疗设备"],
        ),
        description="钇钡铜氧高温超导体，液氮温区工作，超导电力和交通用",
    ),

    # -------------------------------------------------------------------------
    # 核工业材料 (Nuclear Material)
    # -------------------------------------------------------------------------
    "Zircaloy-4": MaterialInfo(
        grade="Zircaloy-4",
        name="锆合金-4",
        aliases=["Zr-4", "R60804", "核级锆合金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.NUCLEAR_MATERIAL,
        standards=["ASTM B350", "RCC-M"],
        properties=MaterialProperties(
            density=6.56,
            melting_point=1850,
            thermal_conductivity=13,
            tensile_strength=480,
            yield_strength=380,
            elongation=18,  # %
            hardness="HB200",
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["管材", "板材", "棒材"],
            blank_hint="管材",
            heat_treatments=["退火", "β淬火"],
            surface_treatments=["阳极氧化", "自钝化"],
            cutting_speed_range=(30, 80),
            coolant_required=True,
            warnings=["核级质量控制", "避免氢脆", "高温水腐蚀"],
            recommendations=["核燃料包壳", "反应堆结构件", "核级管道"],
        ),
        description="锆合金-4，低中子吸收截面，核反应堆燃料包壳用",
    ),
    "Hafnium": MaterialInfo(
        grade="Hafnium",
        name="铪",
        aliases=["Hf", "核级铪", "控制棒材料"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.NUCLEAR_MATERIAL,
        standards=["ASTM B776", "RCC-M"],
        properties=MaterialProperties(
            density=13.31,
            melting_point=2233,
            thermal_conductivity=23,
            tensile_strength=450,
            yield_strength=250,
            elongation=25,  # %
            hardness="HV175",
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "管材"],
            blank_hint="棒材",
            heat_treatments=["退火"],
            surface_treatments=["钝化"],
            cutting_speed_range=(20, 50),
            coolant_required=True,
            warnings=["高中子吸收截面", "与锆分离困难", "价格昂贵"],
            recommendations=["核反应堆控制棒", "中子吸收体", "等离子切割电极"],
        ),
        description="铪金属，高中子吸收截面，核反应堆控制棒用",
    ),
    "B4C": MaterialInfo(
        grade="B4C",
        name="碳化硼",
        aliases=["硼碳化物", "B4C陶瓷", "中子吸收材料"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.CERAMIC,
        group=MaterialGroup.NUCLEAR_MATERIAL,
        standards=["ASTM C750", "企业标准"],
        properties=MaterialProperties(
            density=2.52,
            melting_point=2450,
            thermal_conductivity=30,
            tensile_strength=350,
            yield_strength=350,  # 脆性陶瓷，屈服强度≈抗拉强度
            hardness="HV3000",
            elongation=0.5,  # %
            machinability="very_poor",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["烧结块", "颗粒", "涂层"],
            blank_hint="烧结块",
            heat_treatments=["无"],
            surface_treatments=["无"],
            cutting_speed_range=(2, 10),
            special_tooling=True,
            coolant_required=True,
            warnings=["极硬难加工", "金刚石工具", "脆性大"],
            recommendations=["核反应堆屏蔽", "中子吸收体", "防弹装甲", "耐磨件"],
        ),
        description="碳化硼陶瓷，高硬度高中子吸收，核屏蔽和防弹装甲用",
    ),

    # -------------------------------------------------------------------------
    # 医用合金 (Medical Alloy)
    # -------------------------------------------------------------------------
    "CoCrMo": MaterialInfo(
        grade="CoCrMo",
        name="钴铬钼合金",
        aliases=["Co-Cr-Mo", "ASTM F75", "Vitallium", "医用钴铬合金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.MEDICAL_ALLOY,
        standards=["ASTM F75", "ISO 5832-4"],
        properties=MaterialProperties(
            density=8.30,
            melting_point=1350,
            thermal_conductivity=14.8,
            tensile_strength=900,
            yield_strength=600,
            elongation=8,  # %
            hardness="HRC35",
            machinability="poor",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件", "锻件", "粉末冶金"],
            blank_hint="铸造/锻造",
            heat_treatments=["固溶处理", "时效处理"],
            surface_treatments=["抛光", "钝化"],
            cutting_speed_range=(15, 40),
            special_tooling=True,
            coolant_required=True,
            warnings=["加工硬化", "需要专用刀具", "生物相容性测试"],
            recommendations=["人工关节", "牙科修复", "心脏支架", "骨固定器"],
        ),
        description="钴铬钼医用合金，耐磨耐蚀，人工关节和牙科修复用",
    ),
    "Ti6Al4V-ELI": MaterialInfo(
        grade="Ti6Al4V-ELI",
        name="医用钛合金",
        aliases=["TC4-ELI", "Grade 23", "ASTM F136", "低间隙钛合金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.MEDICAL_ALLOY,
        standards=["ASTM F136", "ISO 5832-3"],
        properties=MaterialProperties(
            density=4.43,
            melting_point=1660,
            thermal_conductivity=6.7,
            tensile_strength=860,
            yield_strength=795,
            elongation=10,  # %
            hardness="HRC36",
            machinability="poor",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "丝材", "粉末"],
            blank_hint="棒材/板材",
            heat_treatments=["退火", "固溶时效"],
            surface_treatments=["阳极氧化", "钝化", "表面改性"],
            cutting_speed_range=(20, 50),
            special_tooling=True,
            coolant_required=True,
            warnings=["低间隙控制", "疲劳性能优异", "需要专用刀具"],
            recommendations=["骨科植入物", "脊柱固定", "骨板骨钉", "3D打印医疗"],
        ),
        description="医用低间隙钛合金，生物相容性优异，骨科植入物用",
    ),
    "316L-Medical": MaterialInfo(
        grade="316L-Medical",
        name="医用不锈钢",
        aliases=["316LVM", "ASTM F138", "医用316L", "手术器械钢"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.MEDICAL_ALLOY,
        standards=["ASTM F138", "ISO 5832-1"],
        properties=MaterialProperties(
            density=8.00,
            melting_point=1400,
            thermal_conductivity=14,
            tensile_strength=580,
            yield_strength=290,
            elongation=40,  # %
            hardness="HB200",
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "管材", "丝材"],
            blank_hint="棒材/板材",
            heat_treatments=["固溶处理"],
            surface_treatments=["电解抛光", "钝化"],
            cutting_speed_range=(40, 100),
            coolant_required=True,
            warnings=["真空熔炼", "低夹杂物", "表面光洁度要求高"],
            recommendations=["手术器械", "骨固定器", "临时植入物", "医疗器械"],
        ),
        description="医用级316L不锈钢，真空熔炼低夹杂，手术器械和临时植入物用",
    ),

    # -------------------------------------------------------------------------
    # 光学材料 (Optical Material)
    # -------------------------------------------------------------------------
    "Fused-Silica": MaterialInfo(
        grade="Fused-Silica",
        name="熔融石英",
        aliases=["石英玻璃", "Quartz Glass", "SiO2", "光学石英"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.GLASS,
        group=MaterialGroup.OPTICAL_MATERIAL,
        standards=["JGS1", "JGS2", "Corning 7980"],
        properties=MaterialProperties(
            density=2.20,
            melting_point=1700,
            thermal_conductivity=1.4,
            tensile_strength=50,
            yield_strength=50,  # 脆性玻璃，屈服强度≈抗拉强度
            hardness="HV500",
            elongation=0.5,  # %
            machinability="poor",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材", "预制件"],
            blank_hint="光学级板材/棒材",
            heat_treatments=["退火"],
            surface_treatments=["光学抛光", "镀膜"],
            cutting_speed_range=(5, 20),
            coolant_required=True,
            warnings=["脆性材料", "热冲击敏感", "需要专用工具"],
            recommendations=["光学镜片", "激光窗口", "紫外光学", "半导体"],
        ),
        description="高纯度熔融石英，优异透光性和热稳定性，光学和半导体用",
    ),
    "Sapphire": MaterialInfo(
        grade="Sapphire",
        name="蓝宝石",
        aliases=["Al2O3单晶", "人造蓝宝石", "刚玉单晶", "Sapphire Crystal"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.CERAMIC,
        group=MaterialGroup.OPTICAL_MATERIAL,
        standards=["MIL-PRF-13830", "ASTM E112"],
        properties=MaterialProperties(
            density=3.98,
            melting_point=2050,
            thermal_conductivity=40,
            tensile_strength=400,
            yield_strength=400,  # 脆性单晶，屈服强度≈抗拉强度
            hardness="HV2000",
            elongation=0.5,  # %
            machinability="very_poor",
            weldability="none",
        ),
        process=ProcessRecommendation(
            blank_forms=["晶体", "晶棒", "晶片", "窗片"],
            blank_hint="单晶晶体/晶片",
            heat_treatments=["退火"],
            surface_treatments=["研磨", "抛光", "镀膜"],
            cutting_speed_range=(1, 5),
            coolant_required=True,
            warnings=["极硬脆性", "金刚石工具", "加工成本高"],
            recommendations=["手表镜面", "LED衬底", "红外窗口", "耐磨部件"],
        ),
        description="单晶蓝宝石，硬度仅次于金刚石，光学和耐磨应用",
    ),
    "Germanium": MaterialInfo(
        grade="Germanium",
        name="锗",
        aliases=["Ge", "红外锗", "光学锗", "锗单晶"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.CERAMIC,
        group=MaterialGroup.OPTICAL_MATERIAL,
        standards=["ASTM F1244", "MIL-G-25883"],
        properties=MaterialProperties(
            density=5.32,
            melting_point=938,
            thermal_conductivity=60,
            tensile_strength=100,
            yield_strength=100,  # 脆性半导体晶体
            hardness="HK780",
            elongation=0.5,  # %
            machinability="fair",
            weldability="none",
        ),
        process=ProcessRecommendation(
            blank_forms=["晶体", "晶片", "窗片", "透镜毛坯"],
            blank_hint="光学级晶片/毛坯",
            heat_treatments=["退火"],
            surface_treatments=["抛光", "镀膜"],
            cutting_speed_range=(10, 30),
            coolant_required=True,
            warnings=["红外透射材料", "对划痕敏感", "需要保护气氛"],
            recommendations=["红外光学", "热像仪", "夜视仪", "红外窗口"],
        ),
        description="光学级锗，8-12μm红外透明，热像仪和红外光学用",
    ),

    # -------------------------------------------------------------------------
    # 电池材料 (Battery Material)
    # -------------------------------------------------------------------------
    "LiFePO4": MaterialInfo(
        grade="LiFePO4",
        name="磷酸铁锂",
        aliases=["LFP", "磷酸铁锂正极", "铁锂", "Iron Phosphate"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.CERAMIC,
        group=MaterialGroup.BATTERY_MATERIAL,
        standards=["GB/T 36276", "IEC 62660"],
        properties=MaterialProperties(
            density=3.60,
            melting_point=600,
            thermal_conductivity=1.1,
            tensile_strength=50,
            yield_strength=None,  # 粉末/颗粒材料，无屈服强度概念
            elongation=0.5,  # %
            hardness="HV300",
            machinability="fair",
            weldability="none",
        ),
        process=ProcessRecommendation(
            blank_forms=["粉末", "浆料", "极片"],
            blank_hint="正极粉末/极片",
            heat_treatments=["烧结", "煅烧"],
            surface_treatments=["碳包覆", "表面改性"],
            cutting_speed_range=(10, 50),
            coolant_required=False,
            warnings=["吸湿敏感", "需要干燥环境", "粉尘防护"],
            recommendations=["动力电池", "储能电池", "电动汽车", "UPS"],
        ),
        description="磷酸铁锂正极材料，安全性好循环寿命长，动力电池用",
    ),
    "NMC": MaterialInfo(
        grade="NMC",
        name="三元正极材料",
        aliases=["NCM", "镍钴锰", "LiNiMnCoO2", "三元锂"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.CERAMIC,
        group=MaterialGroup.BATTERY_MATERIAL,
        standards=["GB/T 36276", "IEC 62660"],
        properties=MaterialProperties(
            density=4.70,
            melting_point=500,
            thermal_conductivity=3.5,
            tensile_strength=40,
            yield_strength=None,  # 粉末/颗粒材料，无屈服强度概念
            elongation=0.5,  # %
            hardness="HV250",
            machinability="fair",
            weldability="none",
        ),
        process=ProcessRecommendation(
            blank_forms=["粉末", "浆料", "极片"],
            blank_hint="正极粉末/极片",
            heat_treatments=["烧结", "煅烧"],
            surface_treatments=["包覆", "掺杂"],
            cutting_speed_range=(10, 50),
            coolant_required=False,
            warnings=["热稳定性差", "钴资源限制", "需要保护气氛"],
            recommendations=["高能量电池", "电动汽车", "3C电子", "储能"],
        ),
        description="镍钴锰三元正极材料，高能量密度，电动汽车主流",
    ),
    "Graphite-Battery": MaterialInfo(
        grade="Graphite-Battery",
        name="电池负极石墨",
        aliases=["负极石墨", "人造石墨", "天然石墨负极", "Anode Graphite"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.CERAMIC,
        group=MaterialGroup.BATTERY_MATERIAL,
        standards=["GB/T 24533", "IEC 62660"],
        properties=MaterialProperties(
            density=2.25,
            melting_point=3650,
            thermal_conductivity=150,
            tensile_strength=30,
            yield_strength=None,  # 粉末/颗粒材料，无屈服强度概念
            elongation=0.5,  # %
            hardness="HV10",
            machinability="good",
            weldability="none",
        ),
        process=ProcessRecommendation(
            blank_forms=["粉末", "浆料", "极片"],
            blank_hint="负极粉末/极片",
            heat_treatments=["石墨化", "包覆"],
            surface_treatments=["碳包覆", "表面氧化"],
            cutting_speed_range=(20, 80),
            coolant_required=False,
            warnings=["粉尘易燃", "静电敏感", "需要干燥环境"],
            recommendations=["锂电池负极", "储能电池", "动力电池", "3C电池"],
        ),
        description="电池级石墨负极材料，层状结构可逆嵌锂，锂电池核心材料",
    ),

    # -------------------------------------------------------------------------
    # 半导体材料 (Semiconductor Material)
    # -------------------------------------------------------------------------
    "Silicon-Wafer": MaterialInfo(
        grade="Silicon-Wafer",
        name="硅晶圆",
        aliases=["单晶硅", "Si Wafer", "硅片", "Monocrystalline Si"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.CERAMIC,
        group=MaterialGroup.SEMICONDUCTOR,
        standards=["SEMI M1", "ASTM F1241"],
        properties=MaterialProperties(
            density=2.33,
            melting_point=1414,
            thermal_conductivity=150,
            tensile_strength=120,
            yield_strength=120,  # 脆性单晶，屈服强度≈抗拉强度
            hardness="HV1000",
            elongation=0.5,  # %
            machinability="fair",
            weldability="none",
        ),
        process=ProcessRecommendation(
            blank_forms=["晶圆", "晶棒", "晶片"],
            blank_hint="抛光晶圆",
            heat_treatments=["退火", "氧化"],
            surface_treatments=["化学机械抛光", "外延生长"],
            cutting_speed_range=(5, 20),
            coolant_required=True,
            warnings=["超净环境", "静电敏感", "颗粒污染"],
            recommendations=["集成电路", "太阳能电池", "MEMS", "功率器件"],
        ),
        description="高纯度单晶硅晶圆，集成电路和太阳能电池基础材料",
    ),
    "GaAs": MaterialInfo(
        grade="GaAs",
        name="砷化镓",
        aliases=["Gallium Arsenide", "化合物半导体", "III-V族"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.CERAMIC,
        group=MaterialGroup.SEMICONDUCTOR,
        standards=["SEMI M15", "MIL-PRF-19500"],
        properties=MaterialProperties(
            density=5.32,
            melting_point=1238,
            thermal_conductivity=55,
            tensile_strength=80,
            yield_strength=80,  # 脆性半导体晶体
            hardness="HK750",
            elongation=0.5,  # %
            machinability="poor",
            weldability="none",
        ),
        process=ProcessRecommendation(
            blank_forms=["晶圆", "晶片", "外延片"],
            blank_hint="半绝缘/N型晶圆",
            heat_treatments=["退火"],
            surface_treatments=["外延生长", "抛光"],
            cutting_speed_range=(3, 15),
            coolant_required=True,
            warnings=["砷毒性", "脆性材料", "特殊废弃处理"],
            recommendations=["射频器件", "LED", "激光器", "太阳能电池"],
        ),
        description="砷化镓化合物半导体，高频和光电器件用",
    ),
    "SiC-Semiconductor": MaterialInfo(
        grade="SiC-Semiconductor",
        name="碳化硅半导体",
        aliases=["Silicon Carbide", "宽禁带半导体", "第三代半导体"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.CERAMIC,
        group=MaterialGroup.SEMICONDUCTOR,
        standards=["SEMI M55", "JIS H8680"],
        properties=MaterialProperties(
            density=3.21,
            melting_point=2730,
            thermal_conductivity=490,
            tensile_strength=500,
            yield_strength=500,  # 脆性陶瓷半导体，屈服强度≈抗拉强度
            hardness="HV2800",
            elongation=0.5,  # %
            machinability="very_poor",
            weldability="none",
        ),
        process=ProcessRecommendation(
            blank_forms=["晶圆", "衬底", "外延片"],
            blank_hint="4H-SiC/6H-SiC晶圆",
            heat_treatments=["高温退火"],
            surface_treatments=["CMP抛光", "外延生长"],
            cutting_speed_range=(1, 8),
            coolant_required=True,
            warnings=["极硬材料", "加工困难", "成本高"],
            recommendations=["电动汽车逆变器", "5G基站", "充电桩", "光伏逆变器"],
        ),
        description="宽禁带碳化硅半导体，高温高压高频功率器件用",
    ),

    # -------------------------------------------------------------------------
    # 热界面材料 (Thermal Interface Material)
    # -------------------------------------------------------------------------
    "Thermal-Paste": MaterialInfo(
        grade="Thermal-Paste",
        name="导热硅脂",
        aliases=["硅脂", "导热膏", "Thermal Grease", "TIM"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.POLYMER,
        group=MaterialGroup.THERMAL_INTERFACE,
        standards=["ASTM D5470", "ISO 22007"],
        properties=MaterialProperties(
            density=2.50,
            melting_point=200,  # 硅油基体软化/分解温度
            thermal_conductivity=5,
            hardness="粘稠体",
            machinability="N/A",
            weldability="none",
        ),
        process=ProcessRecommendation(
            blank_forms=["膏状", "注射器", "桶装"],
            blank_hint="注射器/管装",
            heat_treatments=["无需"],
            surface_treatments=["无需"],
            cutting_speed_range=(0, 0),
            coolant_required=False,
            warnings=["涂抹均匀", "避免气泡", "定期更换"],
            recommendations=["CPU散热", "GPU散热", "LED散热", "电子散热"],
        ),
        description="导热硅脂，填充散热器与芯片间隙，电子散热用",
    ),
    "Thermal-Pad": MaterialInfo(
        grade="Thermal-Pad",
        name="导热垫片",
        aliases=["导热硅胶垫", "Thermal Pad", "Gap Filler", "导热片"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.POLYMER,
        group=MaterialGroup.THERMAL_INTERFACE,
        standards=["ASTM D5470", "ISO 22007"],
        properties=MaterialProperties(
            density=2.80,
            melting_point=200,  # 硅胶基体软化温度
            thermal_conductivity=8,
            tensile_strength=0.5,
            yield_strength=None,  # 柔性弹性体，无屈服点
            elongation=0.5,  # %
            hardness="Shore 00-50",
            machinability="good",
            weldability="none",
        ),
        process=ProcessRecommendation(
            blank_forms=["片材", "卷材", "模切件"],
            blank_hint="片材/模切件",
            heat_treatments=["无需"],
            surface_treatments=["无需"],
            cutting_speed_range=(50, 200),
            coolant_required=False,
            warnings=["压缩量控制", "避免过压", "选择合适厚度"],
            recommendations=["电源模块", "存储器", "电池包", "汽车电子"],
        ),
        description="柔性导热垫片，公差大场合的热界面材料",
    ),
    "Graphene-TIM": MaterialInfo(
        grade="Graphene-TIM",
        name="石墨烯导热材料",
        aliases=["石墨烯散热膜", "Graphene Film", "石墨烯TIM"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.CERAMIC,
        group=MaterialGroup.THERMAL_INTERFACE,
        standards=["GB/T 30544", "ISO/TS 80004"],
        properties=MaterialProperties(
            density=2.10,
            melting_point=3650,  # 石墨升华温度
            thermal_conductivity=1500,
            tensile_strength=130,
            yield_strength=130,  # 石墨烯薄膜，屈服强度≈抗拉强度
            hardness="柔性",
            elongation=1,  # %
            machinability="good",
            weldability="none",
        ),
        process=ProcessRecommendation(
            blank_forms=["薄膜", "片材", "卷材"],
            blank_hint="石墨烯薄膜/片材",
            heat_treatments=["无需"],
            surface_treatments=["层压", "复合"],
            cutting_speed_range=(20, 100),
            coolant_required=False,
            warnings=["面内导热优异", "层间导热一般", "防静电"],
            recommendations=["手机散热", "平板散热", "笔记本", "LED"],
        ),
        description="石墨烯基导热材料，超高面内导热率，消费电子散热用",
    ),

    # -------------------------------------------------------------------------
    # 增材制造材料 (Additive Manufacturing Material)
    # -------------------------------------------------------------------------
    "AlSi10Mg-AM": MaterialInfo(
        grade="AlSi10Mg-AM",
        name="3D打印铝合金",
        aliases=["AlSi10Mg", "SLM铝合金", "DMLS铝合金", "增材铝合金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ADDITIVE_MANUFACTURING,
        standards=["ASTM F3318", "ISO/ASTM 52904"],
        properties=MaterialProperties(
            density=2.67,
            melting_point=570,
            thermal_conductivity=130,
            tensile_strength=450,
            yield_strength=270,
            elongation=6,  # %
            hardness="HB120",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["粉末", "打印件"],
            blank_hint="球形粉末/近净成形件",
            heat_treatments=["T6热处理", "去应力退火"],
            surface_treatments=["喷砂", "阳极氧化", "机加工"],
            cutting_speed_range=(200, 500),
            coolant_required=True,
            warnings=["粉末防爆", "惰性气氛", "后处理必需"],
            recommendations=["航空结构件", "汽车零件", "热交换器", "轻量化部件"],
        ),
        description="SLM/DMLS用AlSi10Mg粉末，轻量化增材制造首选材料",
    ),
    "IN718-AM": MaterialInfo(
        grade="IN718-AM",
        name="3D打印镍基高温合金",
        aliases=["Inconel 718 AM", "SLM镍合金", "增材IN718"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ADDITIVE_MANUFACTURING,
        standards=["ASTM F3055", "AMS 5662"],
        properties=MaterialProperties(
            density=8.19,
            melting_point=1336,
            thermal_conductivity=11.4,
            tensile_strength=1380,
            yield_strength=1100,
            elongation=15,  # %
            hardness="HRC45",
            machinability="poor",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["粉末", "打印件"],
            blank_hint="球形粉末/近净成形件",
            heat_treatments=["固溶+时效", "HIP处理"],
            surface_treatments=["喷砂", "电解抛光", "机加工"],
            cutting_speed_range=(20, 50),
            coolant_required=True,
            warnings=["高温打印", "残余应力大", "需要HIP致密化"],
            recommendations=["航空发动机", "燃气轮机", "火箭喷嘴", "核工业"],
        ),
        description="增材制造用IN718高温合金，航空发动机复杂零件首选",
    ),
    "Ti64-AM": MaterialInfo(
        grade="Ti64-AM",
        name="3D打印钛合金",
        aliases=["Ti-6Al-4V AM", "SLM钛合金", "EBM钛合金", "增材TC4"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ADDITIVE_MANUFACTURING,
        standards=["ASTM F2924", "ASTM F3001"],
        properties=MaterialProperties(
            density=4.43,
            melting_point=1660,
            thermal_conductivity=6.7,
            tensile_strength=1100,
            yield_strength=1000,
            elongation=8,  # %
            hardness="HRC36",
            machinability="poor",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["粉末", "打印件"],
            blank_hint="球形粉末/近净成形件",
            heat_treatments=["去应力退火", "HIP处理"],
            surface_treatments=["喷砂", "化学抛光", "机加工"],
            cutting_speed_range=(30, 80),
            coolant_required=True,
            warnings=["惰性气氛必需", "氧含量控制", "回收粉末管理"],
            recommendations=["医用植入物", "航空结构件", "赛车部件", "定制骨科"],
        ),
        description="增材制造用Ti-6Al-4V钛合金，医疗和航空增材首选",
    ),

    # -------------------------------------------------------------------------
    # 硬质合金 (Hard Alloy / Cemented Carbide)
    # -------------------------------------------------------------------------
    "WC-Co": MaterialInfo(
        grade="WC-Co",
        name="钨钴硬质合金",
        aliases=["碳化钨", "Tungsten Carbide", "YG8", "YG15", "硬质合金"],
        category=MaterialCategory.COMPOSITE,
        sub_category=MaterialSubCategory.CERAMIC,
        group=MaterialGroup.HARD_ALLOY,
        standards=["GB/T 3488", "ISO 513", "ASTM B777"],
        properties=MaterialProperties(
            density=14.5,
            melting_point=2870,
            thermal_conductivity=80,
            tensile_strength=1500,
            yield_strength=1500,  # 脆性材料，屈服强度≈抗拉强度
            hardness="HRA89",
            elongation=0.5,  # %
            machinability="very_poor",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "刀片", "模具毛坯"],
            blank_hint="烧结毛坯/刀片",
            heat_treatments=["无需"],
            surface_treatments=["PVD涂层", "CVD涂层", "研磨"],
            cutting_speed_range=(1, 10),
            coolant_required=True,
            warnings=["金刚石工具加工", "脆性大", "钴粉有毒"],
            recommendations=["切削刀具", "模具", "矿山工具", "耐磨零件"],
        ),
        description="钨钴硬质合金，高硬度高耐磨，切削刀具和模具首选",
    ),
    "Stellite": MaterialInfo(
        grade="Stellite",
        name="司太立合金",
        aliases=["Stellite 6", "钴基耐磨合金", "Co-Cr-W", "堆焊合金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.HARD_ALLOY,
        standards=["AWS A5.13", "AMS 5387"],
        properties=MaterialProperties(
            density=8.40,
            melting_point=1285,
            thermal_conductivity=14.7,
            tensile_strength=900,
            yield_strength=550,
            elongation=0.5,  # %
            hardness="HRC45",
            machinability="poor",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "焊丝", "粉末", "铸件"],
            blank_hint="焊丝/棒材",
            heat_treatments=["固溶处理"],
            surface_treatments=["堆焊", "喷涂", "研磨"],
            cutting_speed_range=(10, 30),
            coolant_required=True,
            warnings=["加工硬化严重", "刀具磨损快", "钴粉防护"],
            recommendations=["阀门密封面", "轴承", "热作模具", "耐磨堆焊"],
        ),
        description="司太立钴基合金，高温耐磨耐蚀，阀门和轴承用",
    ),
    "CBN": MaterialInfo(
        grade="CBN",
        name="立方氮化硼",
        aliases=["Cubic Boron Nitride", "PCBN", "氮化硼刀具"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.CERAMIC,
        group=MaterialGroup.HARD_ALLOY,
        standards=["ISO 1832", "GB/T 6108"],
        properties=MaterialProperties(
            density=3.48,
            melting_point=3000,
            thermal_conductivity=740,
            tensile_strength=700,
            yield_strength=700,  # 脆性超硬材料，屈服强度≈抗拉强度
            hardness="HK4500",
            elongation=0.5,  # %
            machinability="none",
            weldability="none",
        ),
        process=ProcessRecommendation(
            blank_forms=["刀片", "砂轮", "磨料"],
            blank_hint="PCBN刀片/砂轮",
            heat_treatments=["无需"],
            surface_treatments=["无需"],
            cutting_speed_range=(0, 0),
            coolant_required=False,
            warnings=["极硬材料", "脆性", "高温敏感"],
            recommendations=["淬硬钢加工", "高速切削", "精密磨削", "铸铁加工"],
        ),
        description="立方氮化硼超硬材料，硬度仅次于金刚石，淬硬钢加工用",
    ),

    # -------------------------------------------------------------------------
    # 热障涂层材料 (Thermal Barrier Coating)
    # -------------------------------------------------------------------------
    "YSZ": MaterialInfo(
        grade="YSZ",
        name="氧化钇稳定氧化锆",
        aliases=["Yttria-Stabilized Zirconia", "8YSZ", "热障涂层", "TBC"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.CERAMIC,
        group=MaterialGroup.THERMAL_BARRIER,
        standards=["AMS 2448", "ASTM C1327"],
        properties=MaterialProperties(
            density=6.10,
            melting_point=2700,
            thermal_conductivity=2.0,
            tensile_strength=200,
            yield_strength=200,  # 脆性陶瓷涂层，屈服强度≈抗拉强度
            hardness="HV1200",
            elongation=0.5,  # %
            machinability="very_poor",
            weldability="none",
        ),
        process=ProcessRecommendation(
            blank_forms=["粉末", "喷涂层", "预制件"],
            blank_hint="喷涂粉末",
            heat_treatments=["烧结"],
            surface_treatments=["等离子喷涂", "EB-PVD"],
            cutting_speed_range=(0, 0),
            coolant_required=False,
            warnings=["热循环应力", "与基体匹配", "厚度控制"],
            recommendations=["航空发动机", "燃气轮机", "热端部件", "涡轮叶片"],
        ),
        description="氧化钇稳定氧化锆热障涂层，航空发动机热端部件保护",
    ),
    "Al2O3-TBC": MaterialInfo(
        grade="Al2O3-TBC",
        name="氧化铝热障涂层",
        aliases=["Alumina Coating", "TGO", "氧化铝涂层"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.CERAMIC,
        group=MaterialGroup.THERMAL_BARRIER,
        standards=["AMS 2447", "ASTM C633"],
        properties=MaterialProperties(
            density=3.95,
            melting_point=2072,
            thermal_conductivity=30,
            tensile_strength=300,
            yield_strength=300,  # 脆性陶瓷涂层，屈服强度≈抗拉强度
            hardness="HV2000",
            elongation=0.5,  # %
            machinability="very_poor",
            weldability="none",
        ),
        process=ProcessRecommendation(
            blank_forms=["粉末", "喷涂层"],
            blank_hint="喷涂粉末",
            heat_treatments=["无需"],
            surface_treatments=["等离子喷涂", "火焰喷涂"],
            cutting_speed_range=(0, 0),
            coolant_required=False,
            warnings=["热膨胀失配", "界面结合", "耐冲击性"],
            recommendations=["耐磨涂层", "绝缘涂层", "耐蚀涂层", "纺织导丝"],
        ),
        description="氧化铝陶瓷涂层，耐磨耐蚀绝缘，工业耐磨部件用",
    ),
    "MCrAlY": MaterialInfo(
        grade="MCrAlY",
        name="金属粘结层",
        aliases=["NiCrAlY", "CoCrAlY", "Bond Coat", "粘结涂层"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.THERMAL_BARRIER,
        standards=["AMS 4454", "PWA 1375"],
        properties=MaterialProperties(
            density=7.80,
            melting_point=1350,
            thermal_conductivity=10,
            tensile_strength=800,
            yield_strength=600,
            elongation=5,  # %
            hardness="HRC35",
            machinability="poor",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["粉末", "喷涂层"],
            blank_hint="喷涂粉末",
            heat_treatments=["扩散退火"],
            surface_treatments=["HVOF喷涂", "等离子喷涂"],
            cutting_speed_range=(0, 0),
            coolant_required=False,
            warnings=["氧化性能关键", "与基体匹配", "厚度均匀"],
            recommendations=["TBC粘结层", "抗氧化涂层", "热端部件", "燃气轮机"],
        ),
        description="MCrAlY金属粘结涂层，热障涂层系统的抗氧化过渡层",
    ),

    # -------------------------------------------------------------------------
    # 电磁屏蔽材料 (EM Shielding Material)
    # -------------------------------------------------------------------------
    "Mu-Metal": MaterialInfo(
        grade="Mu-Metal",
        name="坡莫合金",
        aliases=["μ-metal", "高导磁合金", "1J79", "磁屏蔽合金"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.EM_SHIELDING,
        standards=["ASTM A753", "GB/T 14986"],
        properties=MaterialProperties(
            density=8.70,
            melting_point=1450,
            thermal_conductivity=20,
            tensile_strength=550,
            yield_strength=150,
            elongation=35,  # %
            hardness="HB150",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "带材", "管材", "屏蔽罩"],
            blank_hint="退火态板材/带材",
            heat_treatments=["磁场退火", "真空退火"],
            surface_treatments=["电镀", "绝缘涂层"],
            cutting_speed_range=(50, 150),
            coolant_required=True,
            warnings=["机械应力降低磁导率", "需要退火恢复", "避免冷加工"],
            recommendations=["磁屏蔽罩", "传感器屏蔽", "医疗设备", "精密仪器"],
        ),
        description="高磁导率坡莫合金，低频磁场屏蔽，精密仪器防护用",
    ),
    "Permalloy": MaterialInfo(
        grade="Permalloy",
        name="铁镍软磁合金",
        aliases=["1J50", "Ni-Fe合金", "软磁合金", "45Permalloy"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.EM_SHIELDING,
        standards=["ASTM A753", "GB/T 14986"],
        properties=MaterialProperties(
            density=8.25,
            melting_point=1450,
            thermal_conductivity=15,
            tensile_strength=500,
            yield_strength=120,
            elongation=40,  # %
            hardness="HB130",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "带材", "丝材", "铁芯"],
            blank_hint="退火态带材",
            heat_treatments=["氢气退火"],
            surface_treatments=["绝缘涂层", "氧化"],
            cutting_speed_range=(60, 180),
            coolant_required=True,
            warnings=["磁导率随成分变化", "热处理敏感", "防磁污染"],
            recommendations=["变压器铁芯", "磁头", "电感", "继电器"],
        ),
        description="铁镍软磁合金，高磁导率低矫顽力，电子元件磁芯用",
    ),
    "Copper-Mesh": MaterialInfo(
        grade="Copper-Mesh",
        name="铜网屏蔽材料",
        aliases=["铜丝网", "EMI屏蔽网", "Copper Shield", "RF屏蔽"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.EM_SHIELDING,
        standards=["MIL-DTL-83528", "IEEE 299"],
        properties=MaterialProperties(
            density=8.90,
            melting_point=1083,
            thermal_conductivity=391,
            conductivity=100,
            tensile_strength=300,
            yield_strength=150,  # 铜丝网，退火态
            hardness="HB60",
            elongation=30,  # %
            machinability="excellent",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["网材", "编织布", "箔材", "泡沫铜"],
            blank_hint="编织网/针织网",
            heat_treatments=["退火"],
            surface_treatments=["镀锡", "镀镍", "导电胶"],
            cutting_speed_range=(100, 300),
            coolant_required=False,
            warnings=["接地连续性", "网孔尺寸选择", "腐蚀防护"],
            recommendations=["EMI屏蔽", "RF屏蔽室", "电子设备", "通信机房"],
        ),
        description="铜丝编织网，高频电磁屏蔽，电子设备EMI防护用",
    ),

    # -------------------------------------------------------------------------
    # 钛合金 (Titanium Alloy)
    # -------------------------------------------------------------------------
    "TA2": MaterialInfo(
        grade="TA2",
        name="工业纯钛",
        aliases=["Gr2", "TA2锻件"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.TITANIUM,
        standards=["GB/T 3620.1-2016"],
        properties=MaterialProperties(
            density=4.51,
            melting_point=1668,
            thermal_conductivity=16.4,
            tensile_strength=400,
            yield_strength=275,
            elongation=20,  # %
            hardness="HB200",
            machinability="poor",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材", "锻件"],
            blank_hint="锻造/板材",
            heat_treatments=["退火", "去应力退火"],
            surface_treatments=["阳极氧化", "酸洗"],
            cutting_speed_range=(20, 60),
            special_tooling=True,
            coolant_required=True,
            warnings=["需要专用刀具", "切削速度要低", "易着火"],
            recommendations=["耐蚀结构件", "医疗器械"],
        ),
        description="工业纯钛，耐蚀性极好",
    ),

    "TC4": MaterialInfo(
        grade="TC4",
        name="钛合金",
        aliases=["Ti-6Al-4V", "Gr5"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.TITANIUM,
        standards=["GB/T 3620.1-2016"],
        properties=MaterialProperties(
            density=4.43,
            melting_point=1650,
            thermal_conductivity=6.7,
            tensile_strength=895,
            yield_strength=825,
            elongation=10,  # %
            hardness="HB320",
            machinability="poor",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["锻件", "棒材", "板材"],
            blank_hint="锻造/板材",
            heat_treatments=["退火", "固溶+时效"],
            surface_treatments=["阳极氧化", "酸洗"],
            cutting_speed_range=(15, 40),
            special_tooling=True,
            coolant_required=True,
            warnings=["难加工材料", "需要专用刀具", "切削热大"],
            recommendations=["航空结构件", "医疗植入物"],
        ),
        description="最常用钛合金，高比强度",
    ),

    "TA1": MaterialInfo(
        grade="TA1",
        name="工业纯钛",
        aliases=["Gr1", "TA1锻件"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.TITANIUM,
        standards=["GB/T 3620.1-2016"],
        properties=MaterialProperties(
            density=4.51,
            melting_point=1668,
            thermal_conductivity=16.4,
            tensile_strength=275,
            yield_strength=170,
            elongation=24,  # %
            hardness="HB120",
            machinability="fair",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材", "箔材"],
            blank_hint="板材/管材",
            heat_treatments=["退火", "去应力退火"],
            surface_treatments=["阳极氧化", "酸洗", "抛光"],
            cutting_speed_range=(30, 80),
            special_tooling=True,
            coolant_required=True,
            warnings=["需要专用刀具", "注意切削热"],
            recommendations=["化工耐蚀管道", "换热器", "电镀挂具"],
        ),
        description="工业纯钛，耐蚀性优异，塑性好",
    ),

    "TC11": MaterialInfo(
        grade="TC11",
        name="高温钛合金",
        aliases=["Ti-6.5Al-3.5Mo-1.5Zr-0.3Si", "BT9"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.TITANIUM,
        standards=["GB/T 3620.1-2016"],
        properties=MaterialProperties(
            density=4.48,
            melting_point=1630,
            thermal_conductivity=7.5,
            tensile_strength=1030,
            yield_strength=930,
            elongation=8,  # %
            hardness="HB350",
            machinability="poor",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["锻件", "棒材"],
            blank_hint="锻造",
            heat_treatments=["退火", "固溶+时效", "双重退火"],
            surface_treatments=["阳极氧化", "酸洗"],
            cutting_speed_range=(10, 30),
            special_tooling=True,
            coolant_required=True,
            warnings=["难加工材料", "需要专用刀具", "切削温度高"],
            recommendations=["航空发动机压气机盘", "叶片", "高温结构件"],
        ),
        description="高温钛合金，500℃下仍有良好性能",
    ),

    "TB6": MaterialInfo(
        grade="TB6",
        name="β型钛合金",
        aliases=["Ti-10V-2Fe-3Al", "Ti1023"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.TITANIUM,
        standards=["GB/T 3620.1-2016"],
        properties=MaterialProperties(
            density=4.65,
            melting_point=1620,
            thermal_conductivity=7.8,
            tensile_strength=1250,
            yield_strength=1170,
            elongation=10,  # %
            hardness="HRC42",
            machinability="poor",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["锻件", "棒材"],
            blank_hint="等温锻造",
            heat_treatments=["固溶+时效", "STA处理"],
            surface_treatments=["酸洗", "喷丸强化"],
            cutting_speed_range=(8, 25),
            special_tooling=True,
            coolant_required=True,
            warnings=["极难加工材料", "需要专用硬质合金刀具", "切削力大"],
            recommendations=["飞机起落架", "大型锻件", "高强度结构件"],
        ),
        description="高强度β钛合金，可热处理强化",
    ),

    "TC21": MaterialInfo(
        grade="TC21",
        name="损伤容限型钛合金",
        aliases=["Ti-6Al-2Zr-2Sn-2Mo-1.5Cr-2Nb"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.TITANIUM,
        standards=["GB/T 3620.1-2016"],
        properties=MaterialProperties(
            density=4.52,
            melting_point=1650,
            thermal_conductivity=7.2,
            tensile_strength=1100,
            yield_strength=1000,
            elongation=6,  # %
            hardness="HB350",
            machinability="poor",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["锻件", "板材"],
            blank_hint="锻造/轧制",
            heat_treatments=["退火", "固溶+时效", "三重热处理"],
            surface_treatments=["喷丸强化", "酸洗"],
            cutting_speed_range=(10, 35),
            special_tooling=True,
            coolant_required=True,
            warnings=["难加工材料", "需要专用刀具"],
            recommendations=["战斗机主承力结构", "大型整体框", "机身蒙皮"],
        ),
        description="国产高强损伤容限钛合金，综合性能优异",
    ),

    # -------------------------------------------------------------------------
    # 镁合金 (Magnesium Alloy)
    # -------------------------------------------------------------------------
    "AZ31B": MaterialInfo(
        grade="AZ31B",
        name="变形镁合金",
        aliases=["AZ31", "MB2"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.MAGNESIUM,
        standards=["GB/T 5153-2016"],
        properties=MaterialProperties(
            density=1.77,
            melting_point=630,
            thermal_conductivity=96,
            tensile_strength=260,
            yield_strength=200,
            elongation=15,  # %
            hardness="HB49",
            machinability="excellent",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材", "型材"],
            blank_hint="轧制/挤压",
            heat_treatments=["退火", "去应力退火"],
            forbidden_heat_treatments=["淬火"],
            surface_treatments=["微弧氧化", "化学转化膜", "喷涂"],
            cutting_speed_range=(300, 1000),
            special_tooling=False,
            coolant_required=False,
            warnings=["易燃材料", "切屑需及时清理", "禁止使用水基冷却液"],
            recommendations=["电子产品外壳", "汽车零件", "轻量化结构"],
        ),
        description="最常用变形镁合金，比强度高，易加工",
    ),

    "AZ91D": MaterialInfo(
        grade="AZ91D",
        name="压铸镁合金",
        aliases=["AZ91", "MB15"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.MAGNESIUM,
        standards=["GB/T 1177-2018"],
        properties=MaterialProperties(
            density=1.81,
            melting_point=595,
            thermal_conductivity=72,
            tensile_strength=230,
            yield_strength=160,
            elongation=3,  # %
            hardness="HB63",
            machinability="excellent",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["压铸件"],
            blank_hint="压铸",
            heat_treatments=["T4固溶", "T6时效"],
            surface_treatments=["微弧氧化", "化学镀", "喷涂"],
            cutting_speed_range=(200, 800),
            special_tooling=False,
            coolant_required=False,
            warnings=["易燃材料", "切屑需及时清理", "禁止使用水基冷却液"],
            recommendations=["汽车变速箱壳", "方向盘骨架", "仪表盘支架"],
        ),
        description="最常用压铸镁合金，流动性好",
    ),

    "ZK60": MaterialInfo(
        grade="ZK60",
        name="高强镁合金",
        aliases=["ZK60A", "MB15"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.MAGNESIUM,
        standards=["GB/T 5153-2016"],
        properties=MaterialProperties(
            density=1.83,
            melting_point=620,
            thermal_conductivity=100,
            tensile_strength=340,
            yield_strength=280,
            elongation=11,  # %
            hardness="HB72",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["锻件", "挤压件"],
            blank_hint="锻造/挤压",
            heat_treatments=["T5时效", "T6时效"],
            surface_treatments=["微弧氧化", "阳极氧化"],
            cutting_speed_range=(150, 600),
            special_tooling=False,
            coolant_required=False,
            warnings=["易燃材料", "热加工温度控制严格"],
            recommendations=["航空结构件", "高强度轻量化零件"],
        ),
        description="高强度镁合金，用于航空航天",
    ),

    # -------------------------------------------------------------------------
    # 硬质合金 (Cemented Carbide)
    # -------------------------------------------------------------------------
    "YG8": MaterialInfo(
        grade="YG8",
        name="钨钴硬质合金",
        aliases=["K20", "WC-8Co"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.CEMENTED_CARBIDE,
        standards=["GB/T 3489-2003"],
        properties=MaterialProperties(
            density=14.7,
            melting_point=2800,
            thermal_conductivity=75,
            tensile_strength=1500,
            yield_strength=1500,  # 脆性材料，屈服强度≈抗拉强度
            hardness="HRA89",
            machinability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["烧结件", "棒材", "板材"],
            blank_hint="粉末冶金",
            heat_treatments=[],
            forbidden_heat_treatments=["淬火", "正火"],
            surface_treatments=["CVD涂层", "PVD涂层"],
            cutting_speed_range=(5, 20),
            special_tooling=True,
            coolant_required=True,
            warnings=["只能电火花或磨削加工", "硬度极高"],
            recommendations=["冲压模具", "拉丝模", "矿用钻头"],
        ),
        description="通用型硬质合金，韧性好",
    ),

    "YT15": MaterialInfo(
        grade="YT15",
        name="钨钛钴硬质合金",
        aliases=["P20", "WC-TiC-15Co"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.CEMENTED_CARBIDE,
        standards=["GB/T 3489-2003"],
        properties=MaterialProperties(
            density=11.5,
            melting_point=2800,
            thermal_conductivity=42,
            tensile_strength=1200,
            yield_strength=1200,  # 脆性材料，屈服强度≈抗拉强度
            hardness="HRA91",
            machinability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["烧结件", "刀片"],
            blank_hint="粉末冶金",
            heat_treatments=[],
            forbidden_heat_treatments=["淬火", "正火"],
            surface_treatments=["CVD涂层", "PVD涂层", "TiN涂层"],
            cutting_speed_range=(5, 15),
            special_tooling=True,
            coolant_required=True,
            warnings=["只能电火花或磨削加工", "硬度极高", "脆性大"],
            recommendations=["车刀", "铣刀", "钻头", "钢件精加工"],
        ),
        description="切削钢用硬质合金，耐磨性好",
    ),

    "YG6": MaterialInfo(
        grade="YG6",
        name="钨钴硬质合金",
        aliases=["K10", "WC-6Co"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.CEMENTED_CARBIDE,
        standards=["GB/T 3489-2003"],
        properties=MaterialProperties(
            density=14.9,
            melting_point=2800,
            thermal_conductivity=80,
            tensile_strength=1350,
            yield_strength=1350,  # 脆性材料，屈服强度≈抗拉强度
            hardness="HRA90",
            elongation=0.5,  # %
            machinability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["烧结件", "棒材"],
            blank_hint="粉末冶金",
            heat_treatments=[],
            forbidden_heat_treatments=["淬火", "正火"],
            surface_treatments=["CVD涂层", "PVD涂层"],
            cutting_speed_range=(5, 15),
            special_tooling=True,
            coolant_required=True,
            warnings=["只能电火花或磨削加工", "硬度极高"],
            recommendations=["铸铁加工刀具", "有色金属刀具", "耐磨零件"],
        ),
        description="高硬度硬质合金，耐磨性优异",
    ),

    # -------------------------------------------------------------------------
    # 精密合金 (Precision Alloy)
    # -------------------------------------------------------------------------
    "4J36": MaterialInfo(
        grade="4J36",
        name="因瓦合金",
        aliases=["Invar", "Invar36", "FeNi36", "1.3912"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.PRECISION_ALLOY,
        standards=["GB/T 1586-2015", "ASTM F1684"],
        properties=MaterialProperties(
            density=8.05,
            melting_point=1430,
            thermal_conductivity=10.5,
            tensile_strength=500,
            yield_strength=280,
            elongation=30,  # %
            hardness="HB160-200",
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "带材", "线材"],
            blank_hint="棒材/带材",
            heat_treatments=["退火", "稳定化处理"],
            heat_treatment_notes=["830℃退火炉冷", "稳定化处理消除加工应力"],
            surface_treatments=["钝化", "镀镍"],
            cutting_speed_range=(20, 50),
            coolant_required=True,
            warnings=["加工硬化明显", "磁性材料", "需稳定化处理"],
            recommendations=["精密仪器", "标准尺", "激光器支架", "光学平台"],
        ),
        description="超低热膨胀系数(1.2ppm/℃)，精密仪器首选",
    ),

    "4J29": MaterialInfo(
        grade="4J29",
        name="可伐合金",
        aliases=["Kovar", "FeNiCo29", "ASTM F15"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.PRECISION_ALLOY,
        standards=["GB/T 1586-2015", "ASTM F15"],
        properties=MaterialProperties(
            density=8.36,
            melting_point=1450,
            thermal_conductivity=17.3,
            tensile_strength=520,
            yield_strength=350,
            elongation=25,  # %
            hardness="HB150-200",
            machinability="fair",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "带材", "管材"],
            blank_hint="棒材/带材",
            heat_treatments=["退火", "氢气退火"],
            heat_treatment_notes=["氢气退火保证封接性能"],
            surface_treatments=["镀镍", "镀金", "氧化处理"],
            cutting_speed_range=(20, 50),
            coolant_required=True,
            warnings=["加工硬化", "封接前需氢气退火"],
            recommendations=["玻璃-金属封接", "陶瓷封接", "电真空器件", "集成电路引线框"],
        ),
        description="热膨胀系数与硬玻璃匹配，封接合金",
    ),

    "4J42": MaterialInfo(
        grade="4J42",
        name="恒弹性合金",
        aliases=["Elinvar", "FeNi42"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.PRECISION_ALLOY,
        standards=["GB/T 1586-2015"],
        properties=MaterialProperties(
            density=8.12,
            melting_point=1425,
            thermal_conductivity=14,
            tensile_strength=600,
            yield_strength=350,
            elongation=20,  # %
            hardness="HB180-220",
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["带材", "线材", "丝材"],
            blank_hint="冷轧带材",
            heat_treatments=["时效处理", "稳定化处理"],
            heat_treatment_notes=["时效处理获得恒弹性"],
            surface_treatments=["钝化"],
            cutting_speed_range=(20, 50),
            coolant_required=True,
            warnings=["冷加工变形量影响性能"],
            recommendations=["钟表游丝", "精密弹簧", "谐振器", "传感器弹性元件"],
        ),
        description="弹性模量温度系数近零，恒弹性合金",
    ),

    # -------------------------------------------------------------------------
    # 电工钢 (Electrical Steel)
    # -------------------------------------------------------------------------
    "50W470": MaterialInfo(
        grade="50W470",
        name="冷轧无取向硅钢",
        aliases=["50W470", "M470-50A", "硅钢片"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ELECTRICAL_STEEL,
        standards=["GB/T 2521.1-2016", "IEC 60404-8-4"],
        properties=MaterialProperties(
            density=7.65,
            melting_point=1500,
            thermal_conductivity=25,
            tensile_strength=450,
            yield_strength=320,  # 硅钢退火态
            hardness="HV180-220",
            elongation=3,  # %
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["卷材", "片材"],
            blank_hint="冷轧卷材",
            heat_treatments=["消除应力退火"],
            heat_treatment_notes=["冲压后需750-800℃退火"],
            surface_treatments=["绝缘涂层"],
            surface_treatment_notes=["表面有绝缘涂层"],
            cutting_speed_range=(30, 80),
            warnings=["冲压模具磨损大", "叠片需绝缘处理"],
            recommendations=["电动机铁芯", "发电机铁芯", "小型变压器"],
        ),
        description="低铁损无取向硅钢，电机铁芯材料",
    ),

    "30Q130": MaterialInfo(
        grade="30Q130",
        name="冷轧取向硅钢",
        aliases=["30Q130", "M130-30S", "取向硅钢", "高磁感取向硅钢"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ELECTRICAL_STEEL,
        standards=["GB/T 2521.2-2016", "IEC 60404-8-7"],
        properties=MaterialProperties(
            density=7.65,
            melting_point=1500,
            thermal_conductivity=18,
            tensile_strength=350,
            yield_strength=250,  # 取向硅钢退火态
            hardness="HV150-180",
            elongation=2,  # %
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["卷材", "片材"],
            blank_hint="冷轧卷材",
            heat_treatments=["消除应力退火"],
            heat_treatment_notes=["退火温度不超过820℃避免破坏取向"],
            surface_treatments=["绝缘涂层", "激光刻痕"],
            surface_treatment_notes=["激光刻痕降低铁损"],
            cutting_speed_range=(20, 50),
            special_tooling=True,
            warnings=["必须沿轧制方向使用", "避免剪切应力"],
            recommendations=["大型变压器铁芯", "配电变压器", "电抗器"],
        ),
        description="极低铁损取向硅钢，变压器铁芯材料",
    ),

    "1J79": MaterialInfo(
        grade="1J79",
        name="坡莫合金",
        aliases=["Permalloy", "Supermalloy", "NiFe80Mo5"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.PRECISION_ALLOY,
        standards=["GB/T 14986-2015"],
        properties=MaterialProperties(
            density=8.6,
            melting_point=1410,
            thermal_conductivity=17,
            tensile_strength=500,
            yield_strength=200,
            elongation=40,  # %
            hardness="HB120-150",
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["带材", "薄板", "线材"],
            blank_hint="冷轧带材",
            heat_treatments=["磁场退火", "氢气退火"],
            heat_treatment_notes=["1100-1200℃氢气退火获得高磁导率"],
            surface_treatments=["绝缘涂层"],
            cutting_speed_range=(40, 80),
            warnings=["对应力敏感", "加工后需退火恢复磁性"],
            recommendations=["磁屏蔽罩", "磁放大器", "脉冲变压器", "传感器磁芯"],
        ),
        description="超高磁导率软磁合金，磁屏蔽材料",
    ),

    # -------------------------------------------------------------------------
    # 焊接材料 (Welding Materials)
    # -------------------------------------------------------------------------
    "ER308L": MaterialInfo(
        grade="ER308L",
        name="不锈钢焊丝",
        aliases=["308L焊丝", "Y308L", "AWS A5.9 ER308L"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.WELDING_MATERIAL,
        standards=["AWS A5.9", "GB/T 17853-2018"],
        properties=MaterialProperties(
            density=7.93,
            melting_point=1400,
            thermal_conductivity=15,
            tensile_strength=550,
            yield_strength=350,
            elongation=35,  # %
            hardness="HB160-190",
            machinability="N/A",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["焊丝", "盘装"],
            blank_hint="盘装焊丝",
            heat_treatments=[],
            surface_treatments=["镀铜"],
            warnings=["存储需防潮", "配合保护气体使用"],
            recommendations=["304/304L不锈钢焊接", "TIG/MIG焊"],
        ),
        description="奥氏体不锈钢焊丝，低碳抗晶间腐蚀",
    ),

    "ER316L": MaterialInfo(
        grade="ER316L",
        name="不锈钢焊丝",
        aliases=["316L焊丝", "Y316L", "AWS A5.9 ER316L"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.WELDING_MATERIAL,
        standards=["AWS A5.9", "GB/T 17853-2018"],
        properties=MaterialProperties(
            density=7.98,
            melting_point=1400,
            thermal_conductivity=14,
            tensile_strength=530,
            yield_strength=320,
            elongation=35,  # %
            hardness="HB160-190",
            machinability="N/A",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["焊丝", "盘装"],
            blank_hint="盘装焊丝",
            heat_treatments=[],
            surface_treatments=["镀铜"],
            warnings=["存储需防潮", "配合保护气体使用"],
            recommendations=["316/316L不锈钢焊接", "耐蚀要求高的场合"],
        ),
        description="含钼奥氏体不锈钢焊丝，耐点蚀",
    ),

    "ER70S-6": MaterialInfo(
        grade="ER70S-6",
        name="碳钢焊丝",
        aliases=["70S-6", "H08Mn2SiA", "AWS A5.18 ER70S-6"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.WELDING_MATERIAL,
        standards=["AWS A5.18", "GB/T 8110-2008"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1500,
            thermal_conductivity=50,
            tensile_strength=500,
            yield_strength=420,
            elongation=22,  # %
            hardness="HB150-180",
            machinability="N/A",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["焊丝", "盘装"],
            blank_hint="盘装焊丝",
            heat_treatments=[],
            surface_treatments=["镀铜"],
            warnings=["存储需防潮"],
            recommendations=["碳钢/低合金钢焊接", "通用MIG焊丝"],
        ),
        description="通用碳钢实心焊丝，脱氧性好",
    ),

    "E7018": MaterialInfo(
        grade="E7018",
        name="低氢焊条",
        aliases=["J507", "AWS A5.1 E7018", "碱性焊条"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.WELDING_MATERIAL,
        standards=["AWS A5.1", "GB/T 5117-2012"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1500,
            thermal_conductivity=50,
            tensile_strength=490,
            yield_strength=400,
            elongation=22,  # %
            hardness="HB150-180",
            machinability="N/A",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["焊条"],
            blank_hint="包装焊条",
            heat_treatments=[],
            surface_treatments=[],
            warnings=["使用前需烘干350-400℃", "存储需防潮"],
            recommendations=["重要结构件焊接", "低合金高强钢焊接"],
        ),
        description="低氢型碱性焊条，抗裂性好",
    ),

    # -------------------------------------------------------------------------
    # 复合材料 (Composite Materials)
    # -------------------------------------------------------------------------
    "CFRP": MaterialInfo(
        grade="CFRP",
        name="碳纤维增强塑料",
        aliases=["碳纤维复合材料", "Carbon Fiber", "CF/EP", "T300/环氧"],
        category=MaterialCategory.COMPOSITE,
        sub_category=MaterialSubCategory.ASSEMBLY,
        group=MaterialGroup.COMPOSITE,
        standards=["ASTM D3039", "GB/T 3354-2014"],
        properties=MaterialProperties(
            density=1.55,
            melting_point=300,  # 树脂分解温度
            thermal_conductivity=5,
            tensile_strength=1500,
            yield_strength=None,  # 纤维增强复合材料，无明确屈服点
            elongation=0.5,  # %
            hardness="N/A",
            machinability="fair",
            weldability="N/A",
        ),
        process=ProcessRecommendation(
            blank_forms=["预浸料", "板材", "管材", "成型件"],
            blank_hint="预浸料/成型件",
            heat_treatments=["固化"],
            heat_treatment_notes=["环氧体系120-180℃固化"],
            surface_treatments=["打磨", "喷漆"],
            cutting_speed_range=(50, 200),
            special_tooling=True,
            coolant_required=False,
            warnings=["粉尘有害需防护", "各向异性材料", "需专用刀具"],
            recommendations=["航空结构件", "赛车部件", "高端运动器材"],
        ),
        description="高比强度复合材料，航空航天首选",
    ),

    "GFRP": MaterialInfo(
        grade="GFRP",
        name="玻璃纤维增强塑料",
        aliases=["玻璃钢", "玻纤复合材料", "GF/EP", "E-glass/环氧"],
        category=MaterialCategory.COMPOSITE,
        sub_category=MaterialSubCategory.ASSEMBLY,
        group=MaterialGroup.COMPOSITE,
        standards=["ASTM D3039", "GB/T 1447-2005"],
        properties=MaterialProperties(
            density=1.85,
            melting_point=300,  # 树脂分解温度
            thermal_conductivity=0.3,
            tensile_strength=450,
            yield_strength=None,  # 纤维增强复合材料，无明确屈服点
            elongation=2.5,  # %
            hardness="N/A",
            machinability="good",
            weldability="N/A",
        ),
        process=ProcessRecommendation(
            blank_forms=["预浸料", "板材", "管材", "手糊件"],
            blank_hint="板材/成型件",
            heat_treatments=["固化"],
            heat_treatment_notes=["室温或加热固化"],
            surface_treatments=["打磨", "喷漆", "胶衣"],
            cutting_speed_range=(80, 250),
            special_tooling=True,
            coolant_required=False,
            warnings=["粉尘有害需防护", "需专用刀具"],
            recommendations=["船艇外壳", "储罐", "风电叶片", "建筑装饰"],
        ),
        description="通用复合材料，性价比高",
    ),

    # -------------------------------------------------------------------------
    # 粉末冶金材料 (Powder Metallurgy)
    # -------------------------------------------------------------------------
    "FC-0208": MaterialInfo(
        grade="FC-0208",
        name="铁碳粉末冶金",
        aliases=["烧结铁", "PM铁基", "MPIF FC-0208"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.POWDER_METALLURGY,
        standards=["MPIF Standard 35", "GB/T 30063-2013"],
        properties=MaterialProperties(
            density=6.7,
            melting_point=1450,
            thermal_conductivity=25,
            tensile_strength=280,
            yield_strength=200,
            elongation=1,  # %
            hardness="HRB60-80",
            machinability="good",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["烧结件", "毛坯"],
            blank_hint="烧结件",
            heat_treatments=["渗碳淬火", "蒸汽处理"],
            heat_treatment_notes=["蒸汽处理改善耐蚀性"],
            surface_treatments=["浸油", "电镀", "发黑"],
            cutting_speed_range=(80, 150),
            warnings=["多孔结构", "不宜承受高冲击"],
            recommendations=["齿轮", "凸轮", "连杆", "轴承座"],
        ),
        description="通用烧结铁基材料，自润滑性好",
    ),

    "FN-0205": MaterialInfo(
        grade="FN-0205",
        name="铁镍粉末冶金",
        aliases=["烧结铁镍", "PM铁镍", "MPIF FN-0205"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.POWDER_METALLURGY,
        standards=["MPIF Standard 35", "GB/T 30063-2013"],
        properties=MaterialProperties(
            density=7.0,
            melting_point=1450,
            thermal_conductivity=20,
            tensile_strength=380,
            yield_strength=280,
            elongation=2,  # %
            hardness="HRB70-90",
            machinability="good",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["烧结件", "毛坯"],
            blank_hint="烧结件",
            heat_treatments=["渗碳淬火", "感应淬火"],
            heat_treatment_notes=["淬火后表面硬度可达HRC55+"],
            surface_treatments=["浸油", "电镀"],
            cutting_speed_range=(60, 120),
            warnings=["多孔结构", "热处理变形需控制"],
            recommendations=["高强度齿轮", "同步器齿环", "ABS齿圈"],
        ),
        description="高强度烧结材料，适合渗碳淬火",
    ),

    # -------------------------------------------------------------------------
    # 塑料/橡胶 (Polymer)
    # -------------------------------------------------------------------------
    "PTFE": MaterialInfo(
        grade="PTFE",
        name="聚四氟乙烯",
        aliases=["特氟龙", "Teflon", "F4"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.POLYMER,
        group=MaterialGroup.FLUOROPOLYMER,
        properties=MaterialProperties(
            density=2.2,
            melting_point=327,
            thermal_conductivity=0.25,
            tensile_strength=25,
            yield_strength=12,
            machinability="good",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "管材"],
            blank_hint="模压棒材/板材",
            heat_treatments=[],
            surface_treatments=[],
            cutting_speed_range=(100, 300),
            coolant_required=False,
            warnings=["热膨胀系数大", "冷流变形"],
            recommendations=["密封件", "耐腐蚀衬里"],
        ),
        description="化学惰性，摩擦系数极低",
    ),

    "RPTFE": MaterialInfo(
        grade="RPTFE",
        name="增强聚四氟乙烯",
        aliases=["填充PTFE", "改性四氟"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.POLYMER,
        group=MaterialGroup.FLUOROPOLYMER,
        properties=MaterialProperties(
            density=2.3,
            melting_point=327,
            thermal_conductivity=0.4,
            tensile_strength=20,
            yield_strength=10,
            machinability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材"],
            blank_hint="模压成型",
            cutting_speed_range=(100, 300),
            coolant_required=False,
            recommendations=["密封垫片", "比纯PTFE耐磨"],
        ),
        description="增强PTFE，抗蠕变性更好",
    ),

    # 工程塑料
    "PEEK": MaterialInfo(
        grade="PEEK",
        name="聚醚醚酮",
        aliases=["Peek", "聚醚醚酮树脂"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.POLYMER,
        group=MaterialGroup.ENGINEERING_PLASTIC,
        standards=["ISO 19924"],
        properties=MaterialProperties(
            density=1.30,
            melting_point=343,
            thermal_conductivity=0.25,
            tensile_strength=100,
            yield_strength=90,
            machinability="good",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "管材"],
            blank_hint="挤出棒材/板材",
            heat_treatments=[],
            surface_treatments=[],
            cutting_speed_range=(100, 300),
            coolant_required=True,
            warnings=["价格昂贵", "加工需冷却"],
            recommendations=["高温轴承", "航空密封件", "医疗器械"],
        ),
        description="超高性能工程塑料，耐高温耐化学品",
    ),

    "POM": MaterialInfo(
        grade="POM",
        name="聚甲醛",
        aliases=["Delrin", "赛钢", "聚缩醛", "POM-C", "POM-H"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.POLYMER,
        group=MaterialGroup.ENGINEERING_PLASTIC,
        standards=["ISO 9988"],
        properties=MaterialProperties(
            density=1.41,
            melting_point=175,
            thermal_conductivity=0.31,
            tensile_strength=70,
            yield_strength=65,
            machinability="excellent",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材"],
            blank_hint="挤出/注塑棒材",
            heat_treatments=[],
            surface_treatments=[],
            cutting_speed_range=(150, 400),
            coolant_required=False,
            warnings=["热变形温度低", "不耐强酸"],
            recommendations=["齿轮", "轴承", "精密零件"],
        ),
        description="自润滑性好，尺寸稳定性高",
    ),

    "PA66": MaterialInfo(
        grade="PA66",
        name="尼龙66",
        aliases=["Nylon66", "聚酰胺66", "PA6", "Nylon6", "尼龙"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.POLYMER,
        group=MaterialGroup.ENGINEERING_PLASTIC,
        standards=["ISO 1874"],
        properties=MaterialProperties(
            density=1.14,
            melting_point=260,
            thermal_conductivity=0.25,
            tensile_strength=80,
            yield_strength=75,
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "注塑件"],
            blank_hint="挤出/注塑成型",
            heat_treatments=[],
            surface_treatments=[],
            cutting_speed_range=(100, 300),
            coolant_required=True,
            warnings=["吸湿性强", "尺寸随湿度变化"],
            recommendations=["齿轮", "滑块", "结构件"],
        ),
        description="高强度尼龙，耐磨耐疲劳",
    ),

    "PC": MaterialInfo(
        grade="PC",
        name="聚碳酸酯",
        aliases=["Polycarbonate", "透明PC"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.POLYMER,
        group=MaterialGroup.ENGINEERING_PLASTIC,
        standards=["ISO 7391"],
        properties=MaterialProperties(
            density=1.20,
            melting_point=267,
            thermal_conductivity=0.2,
            tensile_strength=65,
            yield_strength=60,
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材"],
            blank_hint="挤出板材",
            heat_treatments=[],
            surface_treatments=["硬化涂层"],
            cutting_speed_range=(100, 250),
            coolant_required=True,
            warnings=["不耐碱", "易应力开裂"],
            recommendations=["透明防护罩", "视镜", "光学件"],
        ),
        description="高透明度，抗冲击性极好",
    ),

    "UHMWPE": MaterialInfo(
        grade="UHMWPE",
        name="超高分子量聚乙烯",
        aliases=["UPE", "超高分子聚乙烯", "PE-UHMW"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.POLYMER,
        group=MaterialGroup.ENGINEERING_PLASTIC,
        standards=["ISO 11542"],
        properties=MaterialProperties(
            density=0.93,
            melting_point=136,
            thermal_conductivity=0.42,
            tensile_strength=40,
            yield_strength=20,
            machinability="good",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材"],
            blank_hint="模压板材",
            heat_treatments=[],
            surface_treatments=[],
            cutting_speed_range=(100, 300),
            coolant_required=False,
            warnings=["热膨胀系数大", "不能热焊"],
            recommendations=["耐磨衬板", "导轨", "链轮"],
        ),
        description="极耐磨，自润滑，耐化学品",
    ),

    "PPS": MaterialInfo(
        grade="PPS",
        name="聚苯硫醚",
        aliases=["Ryton", "聚苯硫醚树脂"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.POLYMER,
        group=MaterialGroup.ENGINEERING_PLASTIC,
        properties=MaterialProperties(
            density=1.35,
            melting_point=280,
            thermal_conductivity=0.3,
            tensile_strength=85,
            yield_strength=75,
            elongation=2,  # %
            hardness="Rockwell R123",
            machinability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["注塑件", "挤出件", "压塑件"],
            blank_hint="注塑/压塑成型",
            heat_treatments=[],
            surface_treatments=["等离子处理"],
            cutting_speed_range=(80, 200),
            coolant_required=False,
            warnings=["高温时有轻微气味", "需要高模温"],
            recommendations=["电子元器件", "汽车部件", "化工泵阀", "200℃长期使用"],
        ),
        description="耐高温耐化学品的结晶性工程塑料",
    ),

    "PI": MaterialInfo(
        grade="PI",
        name="聚酰亚胺",
        aliases=["Vespel", "Kapton", "聚酰亚胺树脂"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.POLYMER,
        group=MaterialGroup.ENGINEERING_PLASTIC,
        properties=MaterialProperties(
            density=1.42,
            melting_point=388,
            thermal_conductivity=0.35,
            tensile_strength=100,
            yield_strength=85,
            elongation=8,  # %
            hardness="Rockwell E75",
            machinability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["烧结件", "薄膜", "层压板"],
            blank_hint="烧结/压制成型",
            heat_treatments=[],
            surface_treatments=[],
            cutting_speed_range=(30, 100),
            special_tooling=True,
            coolant_required=False,
            warnings=["价格昂贵", "加工时产生粉尘"],
            recommendations=["航空轴承", "半导体设备", "高温绝缘", "260℃长期使用"],
        ),
        description="极高温工程塑料，热稳定性优异",
    ),

    "PSU": MaterialInfo(
        grade="PSU",
        name="聚砜",
        aliases=["Polysulfone", "聚砜树脂", "Udel"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.POLYMER,
        group=MaterialGroup.ENGINEERING_PLASTIC,
        properties=MaterialProperties(
            density=1.24,
            melting_point=185,
            thermal_conductivity=0.26,
            tensile_strength=75,
            yield_strength=70,
            elongation=50,  # %
            hardness="Rockwell M69",
            machinability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["注塑件", "挤出件", "棒材", "板材"],
            blank_hint="注塑/挤出成型",
            heat_treatments=[],
            surface_treatments=["等离子处理"],
            cutting_speed_range=(60, 150),
            coolant_required=False,
            warnings=["对应力开裂敏感", "需要干燥"],
            recommendations=["医疗器械", "食品加工设备", "高温透明件", "150℃长期使用"],
        ),
        description="耐高温透明工程塑料，耐水解",
    ),

    "PEI": MaterialInfo(
        grade="PEI",
        name="聚醚酰亚胺",
        aliases=["Ultem", "聚醚酰亚胺树脂"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.POLYMER,
        group=MaterialGroup.ENGINEERING_PLASTIC,
        properties=MaterialProperties(
            density=1.27,
            melting_point=217,
            thermal_conductivity=0.22,
            tensile_strength=105,
            yield_strength=95,
            elongation=60,  # %
            hardness="Rockwell M109",
            machinability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["注塑件", "挤出件", "棒材", "板材"],
            blank_hint="注塑/挤出成型",
            heat_treatments=[],
            surface_treatments=["等离子处理", "化学蚀刻"],
            cutting_speed_range=(50, 150),
            coolant_required=False,
            warnings=["需要充分干燥", "模温要求高"],
            recommendations=["航空内饰", "电子连接器", "医疗器械", "170℃长期使用"],
        ),
        description="高性能透明工程塑料，阻燃等级高",
    ),

    "EPDM": MaterialInfo(
        grade="EPDM",
        name="三元乙丙橡胶",
        aliases=["乙丙橡胶"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.POLYMER,
        group=MaterialGroup.RUBBER,
        properties=MaterialProperties(
            density=1.0,
            melting_point=150,  # 使用温度上限
            thermal_conductivity=0.25,
            tensile_strength=15,
            yield_strength=8,
            machinability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["模压件", "挤出件"],
            blank_hint="模压/挤出成型",
            recommendations=["O型圈", "密封条", "耐热老化"],
        ),
        description="耐热耐候性好的橡胶",
    ),

    "VMQ": MaterialInfo(
        grade="VMQ",
        name="硅橡胶",
        aliases=["硅胶", "VMQ+FEP"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.POLYMER,
        group=MaterialGroup.RUBBER,
        properties=MaterialProperties(
            density=1.1,
            melting_point=250,  # 使用温度上限
            thermal_conductivity=0.2,
            tensile_strength=10,
            yield_strength=5,
            machinability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["模压件"],
            blank_hint="模压成型",
            recommendations=["食品级密封件", "耐高低温"],
        ),
        description="耐高低温，食品级",
    ),

    "聚氨酯": MaterialInfo(
        grade="聚氨酯",
        name="聚氨酯",
        aliases=["PU", "聚氨酯-3"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.POLYMER,
        group=MaterialGroup.POLYURETHANE,
        properties=MaterialProperties(
            density=1.2,
            melting_point=180,  # 使用温度上限
            thermal_conductivity=0.19,
            tensile_strength=40,
            yield_strength=25,
            machinability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["浇注件", "模压件"],
            blank_hint="浇注/模压成型",
            recommendations=["耐磨衬套", "缓冲件"],
        ),
        description="耐磨性极好的弹性体",
    ),

    # -------------------------------------------------------------------------
    # 玻璃 (Glass)
    # -------------------------------------------------------------------------
    "硼硅玻璃": MaterialInfo(
        grade="硼硅玻璃",
        name="硼硅玻璃",
        aliases=["耐热玻璃", "高硼硅"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.GLASS,
        group=MaterialGroup.BOROSILICATE,
        properties=MaterialProperties(
            density=2.23,
            melting_point=820,
            thermal_conductivity=1.14,
            machinability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["管材", "板材"],
            blank_hint="热成型",
            warnings=["脆性材料", "需要专用加工"],
            recommendations=["视镜", "化工设备"],
        ),
        description="耐热冲击性好",
    ),

    "钢化玻璃": MaterialInfo(
        grade="钢化玻璃",
        name="钢化玻璃",
        aliases=["强化玻璃"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.GLASS,
        group=MaterialGroup.TEMPERED,
        properties=MaterialProperties(
            density=2.5,
            melting_point=700,
            thermal_conductivity=1.0,
            machinability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材"],
            blank_hint="钢化成型",
            warnings=["不可再加工", "破碎成小颗粒"],
            recommendations=["安全视窗", "防护罩"],
        ),
        description="强度高，破碎安全",
    ),

    # -------------------------------------------------------------------------
    # 陶瓷/纤维 (Ceramic)
    # -------------------------------------------------------------------------
    "硅酸铝": MaterialInfo(
        grade="硅酸铝",
        name="硅酸铝纤维",
        aliases=["陶瓷纤维"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.CERAMIC,
        group=MaterialGroup.ALUMINA_SILICATE,
        properties=MaterialProperties(
            density=0.2,
            melting_point=1260,
            thermal_conductivity=0.12,
            machinability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["毡", "板", "纸"],
            blank_hint="成型制品",
            recommendations=["隔热材料", "耐高温绝缘"],
        ),
        description="高温隔热材料",
    ),

    # -------------------------------------------------------------------------
    # 高强度结构钢 (High-Strength Structural Steel)
    # -------------------------------------------------------------------------
    "Q460": MaterialInfo(
        grade="Q460",
        name="高强度结构钢",
        aliases=["Q460C", "Q460D", "S460", "SM570"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CARBON_STEEL,
        standards=["GB/T 1591-2018", "EN 10025-4"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1500,
            thermal_conductivity=45,
            tensile_strength=550,
            yield_strength=460,
            elongation=17,  # %
            hardness="HB160-200",
            
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "型材", "管材"],
            blank_hint="热轧态/TMCP态",
            heat_treatments=["正火", "TMCP"],
            heat_treatment_notes=["正火880-920℃空冷", "TMCP工艺交货"],
            surface_treatments=["喷砂", "喷漆", "热浸镀锌"],
            cutting_speed_range=(40, 80),
            coolant_required=True,
            warnings=["焊接需控制线能量", "厚板需预热"],
            recommendations=["桥梁结构", "高层建筑", "起重机械"],
        ),
        description="高强度低合金结构钢，屈服强度460MPa",
    ),

    "Q550": MaterialInfo(
        grade="Q550",
        name="高强度结构钢",
        aliases=["Q550D", "Q550E", "S550", "HY80"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["GB/T 16270-2009", "EN 10025-6"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1490,
            thermal_conductivity=42,
            tensile_strength=670,
            yield_strength=550,
            elongation=14,  # %
            hardness="HB180-220",
            
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "型材"],
            blank_hint="调质态/TMCP态",
            heat_treatments=["调质", "TMCP"],
            heat_treatment_notes=["调质淬火900℃水冷", "回火600-650℃"],
            surface_treatments=["喷砂", "喷漆"],
            cutting_speed_range=(30, 60),
            coolant_required=True,
            warnings=["焊接需预热100-150℃", "需低氢焊条"],
            recommendations=["工程机械", "矿用设备", "海工结构"],
        ),
        description="高强度低合金钢，屈服强度550MPa",
    ),

    "Q690": MaterialInfo(
        grade="Q690",
        name="超高强度结构钢",
        aliases=["Q690D", "Q690E", "S690", "HY100"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["GB/T 16270-2009", "EN 10025-6"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1480,
            thermal_conductivity=40,
            tensile_strength=770,
            yield_strength=690,
            elongation=12,  # %
            hardness="HB230-280",
            
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材"],
            blank_hint="调质态",
            heat_treatments=["调质"],
            heat_treatment_notes=["淬火880-920℃水冷", "回火550-620℃"],
            surface_treatments=["喷砂", "喷漆"],
            cutting_speed_range=(25, 50),
            special_tooling=False,
            coolant_required=True,
            warnings=["焊接需预热150-200℃", "需超低氢焊材", "焊后需消应力"],
            recommendations=["大型起重机", "海洋平台", "特种车辆"],
        ),
        description="超高强度结构钢，屈服强度690MPa",
    ),

    # -------------------------------------------------------------------------
    # 锅炉/压力容器钢 (Boiler and Pressure Vessel Steel)
    # -------------------------------------------------------------------------
    "20G": MaterialInfo(
        grade="20G",
        name="锅炉用碳素钢",
        aliases=["20g", "A106-B", "STB410"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CARBON_STEEL,
        standards=["GB/T 5310-2017", "ASTM A106"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1500,
            thermal_conductivity=50,
            tensile_strength=410,
            yield_strength=245,
            hardness="HB120-160",
            elongation=24,
            machinability="good",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["无缝管", "板材"],
            blank_hint="正火态无缝管",
            heat_treatments=["正火", "消应力退火"],
            heat_treatment_notes=["正火900-930℃空冷"],
            surface_treatments=["酸洗", "防锈漆"],
            cutting_speed_range=(60, 100),
            coolant_required=True,
            warnings=["高温长期使用注意石墨化"],
            recommendations=["中低压锅炉管", "蒸汽管道", "450℃以下使用"],
        ),
        description="锅炉用碳素钢，中低压锅炉管",
    ),

    "15CrMoG": MaterialInfo(
        grade="15CrMoG",
        name="锅炉用合金钢",
        aliases=["15CrMo", "A335-P12", "STBA22"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["GB/T 5310-2017", "ASTM A335"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1490,
            thermal_conductivity=40,
            tensile_strength=440,
            yield_strength=295,
            hardness="HB130-180",
            elongation=21,
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["无缝管", "锻件"],
            blank_hint="正火+回火态",
            heat_treatments=["正火+回火", "消应力退火"],
            heat_treatment_notes=["正火930-960℃", "回火680-720℃"],
            surface_treatments=["酸洗", "防锈漆"],
            cutting_speed_range=(50, 90),
            coolant_required=True,
            warnings=["焊后需热处理", "需控制磷硫含量"],
            recommendations=["高压锅炉管", "石化加热炉管", "550℃以下使用"],
        ),
        description="铬钼锅炉钢，高压锅炉过热器管",
    ),

    "12Cr2Mo1R": MaterialInfo(
        grade="12Cr2Mo1R",
        name="压力容器用钢",
        aliases=["SA387 Gr.22", "2.25Cr-1Mo", "10CrMo9-10"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["GB/T 713-2014", "ASTM A387"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1480,
            thermal_conductivity=36,
            tensile_strength=515,
            yield_strength=310,
            hardness="HB150-200",
            elongation=18,
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "锻件"],
            blank_hint="正火+回火态",
            heat_treatments=["正火+回火", "调质"],
            heat_treatment_notes=["正火920-950℃", "回火680-730℃"],
            surface_treatments=["喷砂", "防锈漆"],
            cutting_speed_range=(40, 70),
            coolant_required=True,
            warnings=["焊接需预热200-250℃", "焊后需PWHT", "防止回火脆性"],
            recommendations=["加氢反应器", "石化压力容器", "550℃以下高压设备"],
        ),
        description="2.25Cr-1Mo钢，石化加氢装置用",
    ),

    # -------------------------------------------------------------------------
    # 管线钢 (Pipeline Steel)
    # -------------------------------------------------------------------------
    "X52": MaterialInfo(
        grade="X52",
        name="管线钢",
        aliases=["L360", "API 5L X52", "X52M"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CARBON_STEEL,
        standards=["API 5L", "GB/T 9711-2017"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1500,
            thermal_conductivity=48,
            tensile_strength=460,
            yield_strength=360,
            hardness="HB140-180",
            elongation=21,
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["无缝管", "焊管", "板材"],
            blank_hint="TMCP态/正火态",
            heat_treatments=["正火", "TMCP"],
            heat_treatment_notes=["正火900-930℃空冷"],
            surface_treatments=["3PE防腐", "FBE涂层", "环氧涂层"],
            cutting_speed_range=(50, 90),
            coolant_required=True,
            warnings=["焊接需控制热输入", "注意氢致裂纹"],
            recommendations=["油气输送管道", "城市燃气管", "中低压管线"],
        ),
        description="中等强度管线钢，油气输送",
    ),

    "X65": MaterialInfo(
        grade="X65",
        name="管线钢",
        aliases=["L450", "API 5L X65", "X65M"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["API 5L", "GB/T 9711-2017"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1495,
            thermal_conductivity=45,
            tensile_strength=535,
            yield_strength=450,
            hardness="HB160-200",
            elongation=19,
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["无缝管", "焊管", "板材"],
            blank_hint="TMCP态",
            heat_treatments=["TMCP"],
            heat_treatment_notes=["控轧控冷工艺"],
            surface_treatments=["3PE防腐", "FBE涂层"],
            cutting_speed_range=(45, 80),
            coolant_required=True,
            warnings=["焊接需低氢焊材", "需控制碳当量"],
            recommendations=["高压油气管道", "海底管线", "长输管道"],
        ),
        description="高强度管线钢，高压输送",
    ),

    "X80": MaterialInfo(
        grade="X80",
        name="高强度管线钢",
        aliases=["L555", "API 5L X80", "X80M"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["API 5L", "GB/T 9711-2017"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1490,
            thermal_conductivity=43,
            tensile_strength=625,
            yield_strength=555,
            hardness="HB180-220",
            elongation=18,
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["焊管", "板材"],
            blank_hint="TMCP态",
            heat_treatments=["TMCP"],
            heat_treatment_notes=["严格控轧控冷", "超低碳设计"],
            surface_treatments=["3PE防腐", "FBE涂层"],
            cutting_speed_range=(35, 65),
            coolant_required=True,
            warnings=["焊接工艺要求严格", "需严格控制氢含量", "防止HAZ软化"],
            recommendations=["西气东输", "高压天然气管道", "超高压管线"],
        ),
        description="超高强度管线钢，西气东输主力钢种",
    ),

    # -------------------------------------------------------------------------
    # 模具钢补充 (Die Steel Supplement)
    # -------------------------------------------------------------------------
    "DC53": MaterialInfo(
        grade="DC53",
        name="冷作模具钢",
        aliases=["SLD-MAGIC", "8%Cr钢"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.TOOL_STEEL,
        standards=["JIS G4404", "大同特殊钢"],
        properties=MaterialProperties(
            density=7.87,
            melting_point=1420,
            thermal_conductivity=22,
            tensile_strength=2100,  # MPa, 淬火回火后
            yield_strength=1750,  # MPa, 淬火回火后
            hardness="HRC62-64",
            elongation=4,  # %, 高硬度时韧性较低
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "锻件"],
            blank_hint="退火态模块",
            heat_treatments=["真空淬火", "淬火+低温回火"],
            heat_treatment_notes=["淬火1020-1040℃", "回火180-200℃", "可达HRC62-64"],
            surface_treatments=["氮化", "PVD涂层", "TD处理"],
            special_tooling=True,
            warnings=["需预热", "淬火变形小于Cr12MoV"],
            recommendations=["精密冲裁模", "冷镦模", "高寿命冲压模"],
        ),
        description="高韧性冷作模具钢，比Cr12MoV韧性更好",
    ),

    "S136": MaterialInfo(
        grade="S136",
        name="塑料模具钢",
        aliases=["1.2083", "420MOD", "SUS420J2"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.TOOL_STEEL,
        standards=["ASSAB", "DIN 1.2083"],
        properties=MaterialProperties(
            density=7.80,
            melting_point=1450,
            thermal_conductivity=20,
            tensile_strength=1700,  # MPa, 淬火回火后
            yield_strength=1500,  # MPa, 淬火回火后
            hardness="HRC48-52",
            elongation=3,  # %, 高硬度时韧性有限
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "预硬块"],
            blank_hint="预硬态模块(HRC28-32)",
            heat_treatments=["淬火+回火", "真空淬火"],
            heat_treatment_notes=["淬火1020-1050℃油冷", "回火200-250℃"],
            surface_treatments=["镜面抛光", "镀铬", "氮化"],
            special_tooling=False,
            warnings=["抛光前需精细研磨"],
            recommendations=["透明塑料模", "光学模具", "镜面模具"],
        ),
        description="镜面塑料模具钢，耐腐蚀抛光性优异",
    ),

    "NAK80": MaterialInfo(
        grade="NAK80",
        name="预硬塑料模具钢",
        aliases=["P21", "STAVAX"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.TOOL_STEEL,
        standards=["大同特殊钢", "JIS G4404"],
        properties=MaterialProperties(
            density=7.84,
            melting_point=1460,
            thermal_conductivity=30,
            tensile_strength=1100,
            yield_strength=900,
            elongation=6,  # %
            hardness="HRC37-43",
            machinability="excellent",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "模块"],
            blank_hint="预硬态模块(HRC37-43)",
            heat_treatments=[],  # 通常无需热处理
            heat_treatment_notes=["出厂即预硬态", "可直接加工使用"],
            surface_treatments=["抛光", "蚀纹", "镀铬"],
            special_tooling=False,
            warnings=["预硬钢不宜再次淬火"],
            recommendations=["大型塑料模", "汽车内饰模", "家电模具"],
        ),
        description="预硬型塑料模具钢，可直接加工",
    ),

    # -------------------------------------------------------------------------
    # 易切削钢 (Free-Cutting Steel)
    # -------------------------------------------------------------------------
    "12L14": MaterialInfo(
        grade="12L14",
        name="含铅易切削钢",
        aliases=["Y12Pb", "SUM24L", "1215"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.FREE_CUTTING_STEEL,
        standards=["ASTM A29", "GB/T 8731-2008"],
        properties=MaterialProperties(
            density=7.87,
            melting_point=1500,
            thermal_conductivity=51,
            tensile_strength=390,
            yield_strength=230,
            hardness="HB160-200",
            elongation=22,
            machinability="excellent",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "六角棒", "盘条"],
            blank_hint="冷拔光亮棒",
            heat_treatments=["正火", "退火"],
            heat_treatment_notes=["正火850-880℃", "退火680-720℃"],
            surface_treatments=["镀锌", "镀镍", "发黑"],
            cutting_speed_range=(80, 150),
            coolant_required=True,
            warnings=["含铅有毒", "不可焊接", "废料需特殊处理"],
            recommendations=["自动车床零件", "螺钉螺母", "精密轴销"],
        ),
        description="含铅易切削钢，切削性能极佳",
    ),

    "Y15": MaterialInfo(
        grade="Y15",
        name="硫系易切削钢",
        aliases=["1215", "SUM22", "A1215"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.FREE_CUTTING_STEEL,
        standards=["GB/T 8731-2008", "AISI 1215"],
        properties=MaterialProperties(
            density=7.87,
            melting_point=1500,
            thermal_conductivity=50,
            tensile_strength=380,
            yield_strength=220,
            hardness="HB150-180",
            elongation=25,
            machinability="excellent",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "盘条"],
            blank_hint="冷拔棒材",
            heat_treatments=["正火", "退火"],
            heat_treatment_notes=["正火850-880℃空冷"],
            surface_treatments=["镀锌", "镀镍", "发黑"],
            cutting_speed_range=(70, 140),
            coolant_required=True,
            warnings=["硫含量高", "不适合焊接", "热加工性差"],
            recommendations=["自动车床零件", "标准件", "轴类零件"],
        ),
        description="硫系易切削钢，环保型替代12L14",
    ),

    "Y40Mn": MaterialInfo(
        grade="Y40Mn",
        name="易切削调质钢",
        aliases=["1140", "SUM43", "40MnS"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.FREE_CUTTING_STEEL,
        standards=["GB/T 8731-2008", "AISI 1140"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1495,
            thermal_conductivity=48,
            tensile_strength=620,
            yield_strength=370,
            hardness="HB180-220",
            elongation=18,
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件"],
            blank_hint="调质态棒材",
            heat_treatments=["调质", "正火"],
            heat_treatment_notes=["淬火840℃油冷", "回火540-600℃"],
            surface_treatments=["发黑", "镀锌", "磷化"],
            cutting_speed_range=(60, 120),
            coolant_required=True,
            warnings=["调质后强度较高"],
            recommendations=["传动轴", "连杆", "中强度零件"],
        ),
        description="易切削中碳钢，可调质使用",
    ),

    # -------------------------------------------------------------------------
    # 耐磨钢板 (Wear-Resistant Steel)
    # -------------------------------------------------------------------------
    "NM400": MaterialInfo(
        grade="NM400",
        name="耐磨钢板",
        aliases=["Hardox400", "XAR400", "DILLIDUR400"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.WEAR_RESISTANT_STEEL,
        standards=["GB/T 24186-2009", "EN 10029"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1480,
            thermal_conductivity=38,
            tensile_strength=1250,
            yield_strength=1000,
            hardness="HBW370-430",
            elongation=10,
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材"],
            blank_hint="淬火态钢板",
            heat_treatments=[],  # 出厂即淬火态
            heat_treatment_notes=["出厂淬火态", "二次加热会降低硬度"],
            surface_treatments=["喷砂", "喷漆"],
            cutting_speed_range=(15, 30),
            special_tooling=True,
            coolant_required=True,
            warnings=["硬度高切削困难", "焊接需预热150-200℃", "禁止火焰切割后立即淬火"],
            recommendations=["自卸车车厢", "挖掘机铲斗", "混凝土搅拌筒"],
        ),
        description="中硬度耐磨钢板，通用型",
    ),

    "NM500": MaterialInfo(
        grade="NM500",
        name="高硬度耐磨钢板",
        aliases=["Hardox500", "XAR500", "DILLIDUR500"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.WEAR_RESISTANT_STEEL,
        standards=["GB/T 24186-2009", "EN 10029"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1470,
            thermal_conductivity=35,
            tensile_strength=1600,
            yield_strength=1300,
            hardness="HBW470-530",
            elongation=8,
            machinability="poor",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材"],
            blank_hint="淬火态钢板",
            heat_treatments=[],
            heat_treatment_notes=["出厂淬火态", "加热超过250℃会软化"],
            surface_treatments=["喷砂", "喷漆"],
            cutting_speed_range=(10, 20),
            special_tooling=True,
            coolant_required=True,
            warnings=["极难切削", "焊接需预热200-250℃", "需低氢焊条"],
            recommendations=["高磨损工况", "破碎机衬板", "球磨机衬板"],
        ),
        description="高硬度耐磨钢板，重载工况",
    ),

    "Hardox450": MaterialInfo(
        grade="Hardox450",
        name="悍达耐磨钢板",
        aliases=["HARDOX 450", "HX450"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.WEAR_RESISTANT_STEEL,
        standards=["SSAB标准", "EN 10029"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1475,
            thermal_conductivity=36,
            tensile_strength=1400,
            yield_strength=1200,
            hardness="HBW420-475",
            elongation=10,
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材"],
            blank_hint="淬火态钢板",
            heat_treatments=[],
            heat_treatment_notes=["SSAB专利淬火工艺", "焊接性优于同级别"],
            surface_treatments=["喷砂", "喷漆", "耐磨堆焊"],
            cutting_speed_range=(12, 25),
            special_tooling=True,
            coolant_required=True,
            warnings=["焊接需预热150-200℃", "建议等离子或激光切割"],
            recommendations=["工程机械", "矿山设备", "农业机械"],
        ),
        description="SSAB悍达耐磨钢板，韧性好焊接性优",
    ),

    # -------------------------------------------------------------------------
    # 弹簧钢补充 (Spring Steel Supplement)
    # -------------------------------------------------------------------------
    "55CrSi": MaterialInfo(
        grade="55CrSi",
        name="硅铬弹簧钢",
        aliases=["55CrSiA", "SUP12", "55SiCr"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["GB/T 1222-2016", "JIS G4801"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1480,
            thermal_conductivity=38,
            tensile_strength=1570,
            yield_strength=1370,
            hardness="HRC50-55",
            elongation=5,
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "线材", "扁钢"],
            blank_hint="热轧棒材/线材",
            heat_treatments=["淬火+中温回火", "等温淬火"],
            heat_treatment_notes=["淬火870℃油冷", "回火470-520℃"],
            surface_treatments=["喷丸强化", "发蓝", "磷化"],
            warnings=["脱碳敏感", "淬透性有限"],
            recommendations=["汽车气门弹簧", "高应力弹簧", "阀门弹簧"],
        ),
        description="高强度弹簧钢，抗松弛性优异",
    ),

    # -------------------------------------------------------------------------
    # 高温合金补充 (High-Temperature Alloy Supplement)
    # -------------------------------------------------------------------------
    "A-286": MaterialInfo(
        grade="A-286",
        name="铁镍基高温合金",
        aliases=["A286", "S66286"],  # GH2132/SUH660 are now separate material
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.CORROSION_RESISTANT,
        standards=["AMS 5525", "GB/T 14992-2005"],
        properties=MaterialProperties(
            density=7.94,
            melting_point=1400,
            thermal_conductivity=13,
            tensile_strength=965,
            yield_strength=620,
            elongation=15,  # %
            hardness="HRC28-35",
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件", "板材", "紧固件"],
            blank_hint="固溶+时效态锻件/棒材",
            heat_treatments=["固溶处理", "时效强化"],
            heat_treatment_notes=["固溶980℃/1h水冷", "时效720℃/16h空冷"],
            surface_treatments=["酸洗", "钝化", "镀镍"],
            cutting_speed_range=(15, 35),
            special_tooling=True,
            coolant_required=True,
            warnings=["加工硬化", "刀具磨损较快"],
            recommendations=["燃气轮机部件", "航空紧固件", "650℃以下长期使用"],
        ),
        description="铁镍基沉淀硬化高温合金，成本相对较低",
    ),

    "Waspaloy": MaterialInfo(
        grade="Waspaloy",
        name="镍基高温合金",
        aliases=["Waspaloy®", "N07001", "2.4654", "UNS N07001"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.CORROSION_RESISTANT,
        standards=["AMS 5544", "AMS 5586"],
        properties=MaterialProperties(
            density=8.19,
            melting_point=1355,
            thermal_conductivity=11,
            tensile_strength=1275,
            yield_strength=795,
            elongation=25,  # %
            hardness="HRC35-42",
            machinability="poor",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["锻件", "棒材", "环形件"],
            blank_hint="锻件+热处理态",
            heat_treatments=["固溶处理", "稳定化处理", "时效强化"],
            heat_treatment_notes=["固溶1010℃/4h水冷", "稳定化845℃/4h空冷", "时效760℃/16h空冷"],
            surface_treatments=["酸洗", "喷丸"],
            cutting_speed_range=(8, 20),
            special_tooling=True,
            coolant_required=True,
            warnings=["难加工材料", "刀具磨损严重", "切削速度要低"],
            recommendations=["涡轮盘", "燃烧室部件", "760℃以下长期使用"],
        ),
        description="时效硬化镍基高温合金，高温强度优异",
    ),

    "Rene41": MaterialInfo(
        grade="Rene41",
        name="镍基高温合金",
        aliases=["Rene 41", "René 41", "N07041", "R-41"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.CORROSION_RESISTANT,
        standards=["AMS 5545", "AMS 5712"],
        properties=MaterialProperties(
            density=8.25,
            melting_point=1350,
            thermal_conductivity=10,
            tensile_strength=1310,
            yield_strength=860,
            elongation=14,  # %
            hardness="HRC36-44",
            machinability="poor",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["锻件", "板材", "棒材"],
            blank_hint="锻件+热处理态",
            heat_treatments=["固溶处理", "时效强化"],
            heat_treatment_notes=["固溶1065℃/4h空冷", "时效760℃/16h空冷"],
            surface_treatments=["酸洗", "机械抛光"],
            cutting_speed_range=(6, 18),
            special_tooling=True,
            coolant_required=True,
            warnings=["极难加工", "易产生裂纹", "需严格控制加热速率"],
            recommendations=["航空发动机燃烧室", "加力燃烧室", "815℃以下长期使用"],
        ),
        description="高强度镍基高温合金，用于航空发动机热端部件",
    ),

    # -------------------------------------------------------------------------
    # 铜合金补充 (Copper Alloy Supplement) - 使用国际标准UNS牌号
    # -------------------------------------------------------------------------
    # 注意: C63000对应国标QAl10-3-1.5, C95400对应国标QAl9-4
    # 已在数据库中作为QAl10-3-1.5和QAl9-4存在, 此处添加UNS别名

    # -------------------------------------------------------------------------
    # 低温钢 (Low-Temperature Steel)
    # -------------------------------------------------------------------------
    "09MnNiD": MaterialInfo(
        grade="09MnNiD",
        name="低温用钢",
        aliases=["3.5Ni钢", "A203 Gr.D"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["GB/T 3531-2014", "ASTM A203"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1490,
            thermal_conductivity=38,
            tensile_strength=490,
            yield_strength=315,
            hardness="HB140-180",
            elongation=22,
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "锻件"],
            blank_hint="正火态板材",
            heat_treatments=["正火", "正火+回火", "调质"],
            heat_treatment_notes=["正火890-920℃空冷", "回火600-650℃"],
            surface_treatments=["喷砂", "防锈漆"],
            cutting_speed_range=(50, 100),
            coolant_required=True,
            warnings=["焊后需消应力", "低温冲击韧性要求严格"],
            recommendations=["LPG储罐", "-50℃低温容器", "冷冻设备"],
        ),
        description="3.5%镍低温用钢，-60℃以上使用",
    ),

    "16MnDR": MaterialInfo(
        grade="16MnDR",
        name="低温压力容器用钢",
        aliases=["16MnD", "SA516 Gr.70"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["GB/T 3531-2014", "NB/T 47008-2017"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1495,
            thermal_conductivity=45,
            tensile_strength=490,
            yield_strength=295,
            hardness="HB130-170",
            elongation=21,
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "锻件"],
            blank_hint="正火态板材",
            heat_treatments=["正火", "消应力退火"],
            heat_treatment_notes=["正火900-930℃空冷", "消应力580-620℃"],
            surface_treatments=["喷砂", "防腐涂装"],
            cutting_speed_range=(50, 100),
            coolant_required=True,
            warnings=["控制碳当量", "焊接需预热"],
            recommendations=["低温压力容器", "-40℃低温设备", "乙烯球罐"],
        ),
        description="低温压力容器用钢，-40℃以上使用",
    ),

    "9Ni钢": MaterialInfo(
        grade="9Ni钢",
        name="9%镍钢",
        aliases=["A553 Type I", "X8Ni9", "06Ni9"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ALLOY_STEEL,
        standards=["GB/T 3531-2014", "ASTM A553"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1480,
            thermal_conductivity=26,
            tensile_strength=690,
            yield_strength=585,
            hardness="HB190-230",
            elongation=20,
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "锻件"],
            blank_hint="双重正火+回火态板材",
            heat_treatments=["双重正火+回火", "淬火+回火"],
            heat_treatment_notes=["第一次正火900℃", "第二次正火790℃", "回火580℃"],
            surface_treatments=["喷砂", "保温层"],
            cutting_speed_range=(30, 70),
            coolant_required=True,
            special_tooling=False,
            warnings=["焊接需特殊焊材", "镍基焊丝"],
            recommendations=["LNG储罐", "-196℃深冷设备", "液氮容器"],
        ),
        description="9%镍钢，用于-196℃超低温",
    ),

    # -------------------------------------------------------------------------
    # 电接触材料 (Electrical Contact Materials)
    # -------------------------------------------------------------------------
    "AgCdO": MaterialInfo(
        grade="AgCdO",
        name="银氧化镉",
        aliases=["银镉合金", "AgCdO10", "AgCdO12", "触点银"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ELECTRICAL_CONTACT,
        standards=["GB/T 24582-2009", "JIS H 2141"],
        properties=MaterialProperties(
            density=10.3,
            melting_point=960,
            thermal_conductivity=360,
            tensile_strength=300,  # MPa
            yield_strength=200,  # MPa
            conductivity=80,  # %IACS
            hardness="HV70-90",
            machinability="good",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["触点", "铆钉", "带材"],
            blank_hint="粉末冶金或内氧化法制备",
            heat_treatments=["无需热处理"],
            surface_treatments=["保持清洁"],
            warnings=["有毒材料需注意防护", "欧盟RoHS限制使用"],
            recommendations=["继电器触点", "接触器", "开关电器"],
        ),
        description="传统电接触材料，导电导热性能好",
    ),

    "AgSnO2": MaterialInfo(
        grade="AgSnO2",
        name="银氧化锡",
        aliases=["银锡合金", "AgSnO2In2O3", "环保触点", "SnO2触点"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ELECTRICAL_CONTACT,
        standards=["GB/T 24582-2009", "IEC 60068"],
        properties=MaterialProperties(
            density=10.1,
            melting_point=961,
            thermal_conductivity=340,
            tensile_strength=280,  # MPa
            yield_strength=180,  # MPa
            conductivity=75,  # %IACS
            hardness="HV80-100",
            machinability="good",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["触点", "铆钉", "复合带"],
            blank_hint="粉末冶金法制备",
            heat_treatments=["无需热处理"],
            surface_treatments=["保持清洁", "防氧化包装"],
            warnings=["耐电弧侵蚀性能优于AgCdO"],
            recommendations=["替代AgCdO", "RoHS合规产品", "交流接触器"],
        ),
        description="环保型电接触材料，替代AgCdO",
    ),

    "CuW70": MaterialInfo(
        grade="CuW70",
        name="铜钨合金",
        aliases=["钨铜合金", "W70Cu30", "CuW", "电极铜钨"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ELECTRICAL_CONTACT,
        standards=["GB/T 8320-2003", "RWMA Class 10-14"],
        properties=MaterialProperties(
            density=14.0,
            melting_point=1083,  # 铜的熔点，钨未熔化
            thermal_conductivity=180,
            tensile_strength=550,  # MPa
            yield_strength=450,  # MPa
            conductivity=50,  # %IACS
            hardness="HB200-250",
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "电极"],
            blank_hint="粉末冶金渗铜法",
            heat_treatments=["无需热处理"],
            surface_treatments=["精磨", "抛光"],
            special_tooling=True,
            warnings=["钨含量影响性能", "加工需硬质合金刀具"],
            recommendations=["电火花电极", "真空开关触头", "高压断路器"],
        ),
        description="高熔点耐电弧材料，电火花加工首选",
    ),

    # -------------------------------------------------------------------------
    # 轴承钢 (Bearing Steel)
    # -------------------------------------------------------------------------
    "GCr15": MaterialInfo(
        grade="GCr15",
        name="高碳铬轴承钢",
        aliases=["SUJ2", "52100", "100Cr6", "1.3505", "轴承钢"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.BEARING_STEEL,
        standards=["GB/T 18254-2016", "ASTM A295", "JIS G4805"],
        properties=MaterialProperties(
            density=7.81,
            melting_point=1450,
            thermal_conductivity=46.6,
            tensile_strength=2150,  # MPa, 淬火回火后
            yield_strength=1810,  # MPa, 淬火回火后
            hardness="HRC60-64",
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "管材", "线材"],
            blank_hint="球化退火态",
            heat_treatments=["球化退火", "淬火+低温回火"],
            heat_treatment_notes=["淬火830-850℃油冷", "回火150-180℃", "可达HRC60-64"],
            surface_treatments=["磨削", "超精加工"],
            cutting_speed_range=(30, 80),
            coolant_required=True,
            special_tooling=False,
            warnings=["淬火变形需预留磨量", "需严格控制夹杂物"],
            recommendations=["滚动轴承", "滚珠丝杠", "精密轴承"],
        ),
        description="最常用轴承钢，高硬度高耐磨",
    ),

    "GCr15SiMn": MaterialInfo(
        grade="GCr15SiMn",
        name="高碳铬硅锰轴承钢",
        aliases=["SUJ4", "52100改", "大截面轴承钢"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.BEARING_STEEL,
        standards=["GB/T 18254-2016", "JIS G4805"],
        properties=MaterialProperties(
            density=7.80,
            melting_point=1450,
            thermal_conductivity=42,
            tensile_strength=2200,  # MPa, 淬火回火后
            yield_strength=1850,  # MPa, 淬火回火后
            hardness="HRC60-64",
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件"],
            blank_hint="球化退火态锻件",
            heat_treatments=["球化退火", "淬火+低温回火"],
            heat_treatment_notes=["淬火温度略高于GCr15", "回火150-180℃", "淬透性更好"],
            surface_treatments=["磨削", "超精加工"],
            cutting_speed_range=(25, 70),
            coolant_required=True,
            special_tooling=False,
            warnings=["用于大截面轴承零件", "淬透性优于GCr15"],
            recommendations=["大型轴承", "铁路轴承", "重载轴承"],
        ),
        description="大截面轴承钢，淬透性好",
    ),

    "GCr4": MaterialInfo(
        grade="GCr4",
        name="渗碳轴承钢",
        aliases=["SAE 4320", "20CrNiMo", "渗碳轴承"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.BEARING_STEEL,
        standards=["GB/T 18254-2016", "SAE J404"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1460,
            thermal_conductivity=42,
            tensile_strength=980,
            yield_strength=785,
            elongation=10,  # %
            hardness="表面HRC58-62，心部HRC30-45",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件"],
            blank_hint="正火态",
            heat_treatments=["渗碳", "淬火+低温回火"],
            heat_treatment_notes=["渗碳920-930℃", "渗碳层0.8-1.5mm", "表硬心韧"],
            surface_treatments=["磨削", "喷丸强化"],
            cutting_speed_range=(50, 120),
            coolant_required=True,
            warnings=["渗碳后需淬火回火", "表面硬度高心部韧性好"],
            recommendations=["重载轴承", "航空轴承", "抗冲击轴承"],
        ),
        description="渗碳轴承钢，表硬心韧",
    ),

    # -------------------------------------------------------------------------
    # 弹簧钢补充 (Spring Steel Supplement)
    # -------------------------------------------------------------------------
    "60Si2Mn": MaterialInfo(
        grade="60Si2Mn",
        name="硅锰弹簧钢",
        aliases=["SUP6", "9260", "1.7108", "硅锰钢"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.SPRING_STEEL,
        standards=["GB/T 1222-2016", "ASTM A689", "JIS G4801"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1470,
            thermal_conductivity=30,
            tensile_strength=1470,
            yield_strength=1274,
            hardness="HRC44-52",
            elongation=5,
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "扁钢", "盘条"],
            blank_hint="退火态",
            heat_treatments=["油淬", "中温回火"],
            heat_treatment_notes=["淬火850-870℃油冷", "回火420-480℃", "回火后HRC44-52"],
            surface_treatments=["喷丸强化", "发蓝"],
            cutting_speed_range=(25, 60),
            coolant_required=True,
            warnings=["脱碳敏感性高", "热处理需保护气氛"],
            recommendations=["汽车板簧", "减震弹簧", "大型弹簧"],
        ),
        description="通用硅锰弹簧钢，成本较低",
    ),

    "60Si2CrA": MaterialInfo(
        grade="60Si2CrA",
        name="硅铬弹簧钢",
        aliases=["SUP12", "60SC7", "铬硅弹簧钢"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.SPRING_STEEL,
        standards=["GB/T 1222-2016", "JIS G4801"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1475,
            thermal_conductivity=32,
            tensile_strength=1570,
            yield_strength=1370,
            elongation=6,  # %
            hardness="HRC46-54",
            
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "扁钢", "盘条"],
            blank_hint="退火态",
            heat_treatments=["油淬", "中温回火"],
            heat_treatment_notes=["淬火850-880℃油冷", "回火400-460℃", "淬透性比60Si2Mn好"],
            surface_treatments=["喷丸强化", "磷化"],
            cutting_speed_range=(22, 55),
            coolant_required=True,
            warnings=["脱碳敏感性高", "回火脆性区400-500℃注意"],
            recommendations=["汽车悬挂弹簧", "高应力弹簧", "重载弹簧"],
        ),
        description="高性能硅铬弹簧钢，强度更高",
    ),

    "50CrVA": MaterialInfo(
        grade="50CrVA",
        name="铬钒弹簧钢",
        aliases=["SUP10", "6150", "50CrV4", "1.8159"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.SPRING_STEEL,
        standards=["GB/T 1222-2016", "ASTM A689", "DIN 17221"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1475,
            thermal_conductivity=34,
            tensile_strength=1570,
            yield_strength=1370,
            hardness="HRC47-55",
            elongation=6,
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "扁钢", "线材"],
            blank_hint="退火态",
            heat_treatments=["油淬", "中温回火"],
            heat_treatment_notes=["淬火850-870℃油冷", "回火400-440℃", "细化晶粒"],
            surface_treatments=["喷丸强化", "发黑"],
            cutting_speed_range=(25, 60),
            coolant_required=True,
            warnings=["钒细化晶粒提高韧性", "综合性能优良"],
            recommendations=["气门弹簧", "离合器弹簧", "高疲劳寿命弹簧"],
        ),
        description="铬钒弹簧钢，疲劳性能优良",
    ),

    # -------------------------------------------------------------------------
    # 耐热不锈钢补充 (Heat-Resistant Stainless Steel)
    # -------------------------------------------------------------------------
    "2Cr13": MaterialInfo(
        grade="2Cr13",
        name="马氏体不锈钢",
        aliases=["420", "SUS420J1", "1.4021", "X20Cr13"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.STAINLESS_STEEL,
        standards=["GB/T 1220-2007", "ASTM A276", "JIS G4303"],
        properties=MaterialProperties(
            density=7.75,
            melting_point=1480,
            thermal_conductivity=25,
            tensile_strength=735,
            yield_strength=540,
            hardness="HRC45-50",
            elongation=12,
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "锻件"],
            blank_hint="退火态",
            heat_treatments=["淬火+回火", "调质处理"],
            heat_treatment_notes=["淬火1000-1050℃油冷", "回火200-400℃", "可达HRC45-50"],
            surface_treatments=["抛光", "钝化", "氮化"],
            cutting_speed_range=(30, 80),
            coolant_required=True,
            warnings=["回火脆性区400-600℃避免", "耐蚀性低于奥氏体"],
            recommendations=["刀具", "阀门零件", "轴类零件"],
        ),
        description="马氏体不锈钢，可热处理强化",
    ),

    "1Cr17": MaterialInfo(
        grade="1Cr17",
        name="铁素体不锈钢",
        aliases=["430", "SUS430", "1.4016", "X6Cr17"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.STAINLESS_STEEL,
        standards=["GB/T 1220-2007", "ASTM A276", "JIS G4303"],
        properties=MaterialProperties(
            density=7.70,
            melting_point=1500,
            thermal_conductivity=26,
            tensile_strength=450,
            yield_strength=275,
            elongation=20,  # %
            hardness="HB180-220",
            
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "带材", "棒材"],
            blank_hint="退火态",
            heat_treatments=["退火"],
            heat_treatment_notes=["退火750-850℃缓冷", "不能淬火强化"],
            surface_treatments=["抛光", "拉丝", "钝化"],
            cutting_speed_range=(50, 120),
            coolant_required=True,
            warnings=["高温脆性475℃", "焊接易脆化"],
            recommendations=["厨具", "汽车装饰", "家电面板"],
        ),
        description="铁素体不锈钢，成本较低",
    ),

    "0Cr25Ni20": MaterialInfo(
        grade="0Cr25Ni20",
        name="耐热不锈钢",
        aliases=["310S", "SUS310S", "1.4845", "X8CrNi25-21"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.STAINLESS_STEEL,
        standards=["GB/T 1220-2007", "ASTM A240", "JIS G4304"],
        properties=MaterialProperties(
            density=7.98,
            melting_point=1400,
            thermal_conductivity=14,
            tensile_strength=520,
            yield_strength=205,
            elongation=35,  # %
            hardness="HB190-220",
            
            machinability="poor",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "管材", "棒材"],
            blank_hint="固溶态",
            heat_treatments=["固溶处理"],
            heat_treatment_notes=["固溶1040-1100℃水冷", "工作温度可达1000℃"],
            surface_treatments=["酸洗钝化", "喷砂"],
            cutting_speed_range=(20, 50),
            coolant_required=True,
            special_tooling=True,
            warnings=["加工硬化严重", "需低速大进给"],
            recommendations=["炉用零件", "高温设备", "热处理工装"],
        ),
        description="高镍铬耐热不锈钢，耐高温氧化",
    ),

    # -------------------------------------------------------------------------
    # 齿轮钢 (Gear Steel)
    # -------------------------------------------------------------------------
    "20CrMnTi": MaterialInfo(
        grade="20CrMnTi",
        name="渗碳齿轮钢",
        aliases=["SCM420H", "4118", "齿轮钢", "渗碳钢"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.GEAR_STEEL,
        standards=["GB/T 3077-2015", "JIS G4052", "ASTM A322"],
        properties=MaterialProperties(
            density=7.87,
            melting_point=1480,
            thermal_conductivity=45,
            tensile_strength=1080,
            yield_strength=835,
            elongation=0.5,  # %
            hardness="表面HRC58-62，心部HRC30-45",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件"],
            blank_hint="正火态",
            heat_treatments=["渗碳", "淬火+低温回火"],
            heat_treatment_notes=["渗碳920℃", "渗碳层0.8-1.2mm", "淬火后回火180-200℃"],
            surface_treatments=["喷丸强化", "磷化"],
            cutting_speed_range=(60, 120),
            coolant_required=True,
            warnings=["渗碳后需淬火回火", "钛细化晶粒"],
            recommendations=["汽车齿轮", "变速箱齿轮", "差速器齿轮"],
        ),
        description="最常用渗碳齿轮钢，钛细化晶粒",
    ),

    "20CrMo": MaterialInfo(
        grade="20CrMo",
        name="铬钼渗碳钢",
        aliases=["SCM420", "4118", "铬钼钢"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.GEAR_STEEL,
        standards=["GB/T 3077-2015", "JIS G4105", "ASTM A322"],
        properties=MaterialProperties(
            density=7.87,
            melting_point=1485,
            thermal_conductivity=42,
            tensile_strength=980,
            yield_strength=785,
            elongation=12,  # %
            hardness="表面HRC58-62，心部HRC28-42",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件", "管材"],
            blank_hint="正火态",
            heat_treatments=["渗碳", "淬火+低温回火"],
            heat_treatment_notes=["渗碳900-920℃", "渗碳层0.6-1.0mm", "回火150-180℃"],
            surface_treatments=["喷丸强化", "发黑"],
            cutting_speed_range=(65, 130),
            coolant_required=True,
            warnings=["淬透性好于20CrMnTi", "心部韧性好"],
            recommendations=["重载齿轮", "轴类零件", "高速齿轮"],
        ),
        description="铬钼渗碳钢，淬透性好",
    ),

    "20CrNiMo": MaterialInfo(
        grade="20CrNiMo",
        name="铬镍钼渗碳钢",
        aliases=["SNCM220", "8620", "4320", "高级齿轮钢"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.GEAR_STEEL,
        standards=["GB/T 3077-2015", "JIS G4103", "ASTM A322"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1475,
            thermal_conductivity=38,
            tensile_strength=1080,
            yield_strength=880,
            elongation=10,  # %
            hardness="表面HRC58-64，心部HRC35-48",
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件"],
            blank_hint="正火态或退火态",
            heat_treatments=["渗碳", "淬火+低温回火"],
            heat_treatment_notes=["渗碳900-920℃", "渗碳层0.8-1.5mm", "回火160-200℃"],
            surface_treatments=["喷丸强化", "磷化", "渗氮"],
            cutting_speed_range=(50, 100),
            coolant_required=True,
            warnings=["综合性能最好", "成本较高"],
            recommendations=["航空齿轮", "重载传动", "精密齿轮"],
        ),
        description="高级渗碳齿轮钢，综合性能优异",
    ),

    # -------------------------------------------------------------------------
    # 航空铝合金补充 (Aerospace Aluminum Supplement)
    # -------------------------------------------------------------------------
    "5A06": MaterialInfo(
        grade="5A06",
        name="防锈铝合金",
        aliases=["5456", "AlMg5", "A5456", "LF6"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ALUMINUM,
        standards=["GB/T 3190-2020", "ASTM B209"],
        properties=MaterialProperties(
            density=2.64,
            melting_point=640,
            thermal_conductivity=117,
            tensile_strength=315,
            yield_strength=155,
            elongation=10,  # %
            hardness="HB73",
            
            machinability="good",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "管材", "棒材"],
            blank_hint="退火态或H态",
            heat_treatments=["退火"],
            heat_treatment_notes=["退火340-400℃", "不能热处理强化"],
            surface_treatments=["阳极氧化", "喷涂"],
            cutting_speed_range=(200, 600),
            coolant_required=False,
            warnings=["不能热处理强化", "焊接性能优良"],
            recommendations=["船舶结构", "压力容器", "焊接结构件"],
        ),
        description="高镁防锈铝，焊接性能好",
    ),

    "2A14": MaterialInfo(
        grade="2A14",
        name="高强铝合金",
        aliases=["2014", "LD10", "AlCu4SiMg", "A2014"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ALUMINUM,
        standards=["GB/T 3190-2020", "ASTM B211"],
        properties=MaterialProperties(
            density=2.80,
            melting_point=635,
            thermal_conductivity=155,
            tensile_strength=470,
            yield_strength=415,
            elongation=8,  # %
            hardness="HB135",
            
            machinability="good",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["锻件", "棒材", "板材"],
            blank_hint="T6态锻件",
            heat_treatments=["固溶+人工时效"],
            heat_treatment_notes=["固溶500-510℃水淬", "时效165-175℃保温10-12h"],
            surface_treatments=["阳极氧化", "化学氧化"],
            cutting_speed_range=(150, 400),
            coolant_required=True,
            warnings=["耐蚀性差需防护", "可锻性好"],
            recommendations=["航空锻件", "飞机框架", "高强度结构"],
        ),
        description="高强可锻铝合金，航空结构用",
    ),

    "7A04": MaterialInfo(
        grade="7A04",
        name="超高强铝合金",
        aliases=["7050", "LC4", "AlZn6MgCu", "超硬铝"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ALUMINUM,
        standards=["GB/T 3190-2020", "ASTM B247"],
        properties=MaterialProperties(
            density=2.82,
            melting_point=630,
            thermal_conductivity=150,
            tensile_strength=560,
            yield_strength=490,
            elongation=6,  # %
            hardness="HB150",
            
            machinability="good",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["锻件", "棒材", "板材"],
            blank_hint="T6态锻件",
            heat_treatments=["固溶+人工时效"],
            heat_treatment_notes=["固溶470℃水淬", "时效120℃/24h+165℃/8h双级时效"],
            surface_treatments=["阳极氧化", "化学氧化", "喷漆"],
            cutting_speed_range=(150, 400),
            coolant_required=True,
            warnings=["应力腐蚀敏感", "需严格热处理控制"],
            recommendations=["飞机大梁", "机翼蒙皮", "高载荷结构"],
        ),
        description="超高强铝合金，航空主承力结构",
    ),

    # -------------------------------------------------------------------------
    # 气门钢 (Valve Steel)
    # -------------------------------------------------------------------------
    "4Cr10Si2Mo": MaterialInfo(
        grade="4Cr10Si2Mo",
        name="马氏体耐热钢",
        aliases=["SUH3", "X45CrSi9-3", "气门钢", "排气门钢"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.VALVE_STEEL,
        standards=["GB/T 1221-2007", "JIS G4311", "DIN 17470"],
        properties=MaterialProperties(
            density=7.70,
            melting_point=1480,
            thermal_conductivity=24,
            tensile_strength=980,
            yield_strength=785,
            elongation=0.5,  # %
            hardness="HRC38-45",
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件"],
            blank_hint="锻造毛坯",
            heat_treatments=["淬火+回火", "调质处理"],
            heat_treatment_notes=["淬火1000-1050℃油冷", "回火700-750℃", "耐温600℃"],
            surface_treatments=["渗氮", "镀铬"],
            cutting_speed_range=(20, 50),
            coolant_required=True,
            special_tooling=True,
            warnings=["高温工作环境", "需考虑热疲劳"],
            recommendations=["发动机排气门", "涡轮增压器", "高温紧固件"],
        ),
        description="马氏体耐热钢，发动机排气门首选",
    ),

    "5Cr21Mn9Ni4N": MaterialInfo(
        grade="5Cr21Mn9Ni4N",
        name="奥氏体耐热钢",
        aliases=["SUH35", "21-4N", "进气门钢", "奥氏体气门钢"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.VALVE_STEEL,
        standards=["GB/T 1221-2007", "JIS G4311", "ASTM A565"],
        properties=MaterialProperties(
            density=7.88,
            melting_point=1400,
            thermal_conductivity=15,
            tensile_strength=830,
            yield_strength=540,
            hardness="HB250-320",
            elongation=15,
            machinability="poor",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件"],
            blank_hint="固溶态锻件",
            heat_treatments=["固溶处理", "时效处理"],
            heat_treatment_notes=["固溶1100-1150℃水冷", "时效750-800℃", "耐温800℃"],
            surface_treatments=["渗氮", "硬质合金焊接"],
            cutting_speed_range=(15, 40),
            coolant_required=True,
            special_tooling=True,
            warnings=["加工硬化严重", "需低速大进给"],
            recommendations=["发动机进气门", "高温阀杆", "涡轮叶片"],
        ),
        description="奥氏体耐热钢，高温强度好",
    ),

    "4Cr14Ni14W2Mo": MaterialInfo(
        grade="4Cr14Ni14W2Mo",
        name="高温气门钢",
        aliases=["SUH38", "X40CrNiW14-14", "钨钼气门钢"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.VALVE_STEEL,
        standards=["GB/T 1221-2007", "JIS G4311"],
        properties=MaterialProperties(
            density=7.95,
            melting_point=1380,
            thermal_conductivity=12,
            tensile_strength=880,
            yield_strength=590,
            hardness="HB280-350",
            elongation=12,
            machinability="poor",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件"],
            blank_hint="固溶态锻件",
            heat_treatments=["固溶处理", "时效处理"],
            heat_treatment_notes=["固溶1100-1180℃水冷", "时效800-850℃", "耐温850℃"],
            surface_treatments=["渗氮", "硬质合金堆焊"],
            cutting_speed_range=(12, 35),
            coolant_required=True,
            special_tooling=True,
            warnings=["极难加工", "需硬质合金刀具"],
            recommendations=["重型柴油机气门", "船用发动机", "高性能发动机"],
        ),
        description="钨钼高温气门钢，极端高温用",
    ),

    # -------------------------------------------------------------------------
    # 链条钢/渗碳硼钢 (Chain Steel / Boron Steel)
    # -------------------------------------------------------------------------
    "20MnVB": MaterialInfo(
        grade="20MnVB",
        name="渗碳硼钢",
        aliases=["链条钢", "销轴钢", "硼钢"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CHAIN_STEEL,
        standards=["GB/T 5216-2014", "JIS G4052"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1490,
            thermal_conductivity=45,
            tensile_strength=980,
            yield_strength=785,
            elongation=10,  # %
            hardness="表面HRC58-62，心部HRC30-40",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "盘条"],
            blank_hint="热轧态",
            heat_treatments=["渗碳", "淬火+低温回火"],
            heat_treatment_notes=["渗碳900-920℃", "硼提高淬透性", "回火180-200℃"],
            surface_treatments=["发黑", "磷化"],
            cutting_speed_range=(60, 130),
            coolant_required=True,
            warnings=["硼含量0.0005-0.003%", "淬透性好于20CrMnTi"],
            recommendations=["链条销轴", "紧固件", "冷镦件"],
        ),
        description="渗碳硼钢，链条销轴首选",
    ),

    "15MnVB": MaterialInfo(
        grade="15MnVB",
        name="低碳硼钢",
        aliases=["冷镦钢", "螺栓钢", "细晶粒硼钢"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CHAIN_STEEL,
        standards=["GB/T 5216-2014", "JIS G4052"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1495,
            thermal_conductivity=47,
            tensile_strength=880,
            yield_strength=685,
            elongation=12,  # %
            hardness="表面HRC56-60，心部HRC28-38",
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["盘条", "棒材"],
            blank_hint="热轧球化退火态",
            heat_treatments=["渗碳", "淬火+低温回火"],
            heat_treatment_notes=["渗碳880-900℃", "淬火840℃油冷", "回火180℃"],
            surface_treatments=["磷化皂化", "发黑"],
            cutting_speed_range=(70, 150),
            coolant_required=True,
            warnings=["冷镦性能优良", "钒细化晶粒"],
            recommendations=["高强度螺栓", "链条", "冷镦紧固件"],
        ),
        description="低碳渗碳硼钢，冷镦性能好",
    ),

    "22MnB5": MaterialInfo(
        grade="22MnB5",
        name="热成形钢",
        aliases=["热冲压钢", "USIBOR", "PHS钢", "硼钢板"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.CHAIN_STEEL,
        standards=["EN 10083-3", "VDA 239-100"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1490,
            thermal_conductivity=42,
            tensile_strength=1500,
            yield_strength=1100,
            hardness="HV450-500",
            elongation=5,
            machinability="poor",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "卷材"],
            blank_hint="Al-Si涂层板",
            heat_treatments=["热冲压淬火"],
            heat_treatment_notes=["加热900-950℃", "模具内淬火", "马氏体组织"],
            surface_treatments=["Al-Si涂层", "GA涂层"],
            cutting_speed_range=(20, 60),
            coolant_required=True,
            special_tooling=True,
            warnings=["需专用热冲压生产线", "回弹小"],
            recommendations=["汽车A/B柱", "车门防撞梁", "安全结构件"],
        ),
        description="热成形硼钢，汽车安全件首选",
    ),

    # -------------------------------------------------------------------------
    # 电工硅钢补充 (Electrical Steel Supplement)
    # -------------------------------------------------------------------------
    "B50A600": MaterialInfo(
        grade="B50A600",
        name="无取向硅钢",
        aliases=["50W600", "M600-50A", "电机硅钢"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ELECTRICAL_STEEL,
        standards=["GB/T 2521.1-2016", "IEC 60404-8-4"],
        properties=MaterialProperties(
            density=7.65,
            melting_point=1500,
            thermal_conductivity=20,
            tensile_strength=450,
            yield_strength=300,
            hardness="HV150-180",
            elongation=25,
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["薄板", "卷材"],
            blank_hint="0.5mm厚度卷材",
            heat_treatments=["去应力退火"],
            heat_treatment_notes=["退火750-800℃", "氢气或真空气氛", "改善磁性能"],
            surface_treatments=["绝缘涂层", "磷化"],
            cutting_speed_range=(80, 200),
            coolant_required=False,
            warnings=["避免冲压毛刺", "叠装需绝缘"],
            recommendations=["小型电机", "压缩机", "通用电机铁芯"],
        ),
        description="无取向硅钢，通用电机级",
    ),

    "B35A230": MaterialInfo(
        grade="B35A230",
        name="高效无取向硅钢",
        aliases=["35W230", "M230-35A", "高效电机硅钢"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ELECTRICAL_STEEL,
        standards=["GB/T 2521.1-2016", "IEC 60404-8-4"],
        properties=MaterialProperties(
            density=7.60,
            melting_point=1500,
            thermal_conductivity=18,
            tensile_strength=480,
            yield_strength=340,
            hardness="HV160-190",
            elongation=20,
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["薄板", "卷材"],
            blank_hint="0.35mm厚度卷材",
            heat_treatments=["去应力退火"],
            heat_treatment_notes=["退火780-820℃", "低铁损高效率"],
            surface_treatments=["绝缘涂层", "自粘涂层"],
            cutting_speed_range=(80, 200),
            coolant_required=False,
            warnings=["高效电机专用", "成本高于普通硅钢"],
            recommendations=["变频电机", "新能源汽车电机", "高效节能电机"],
        ),
        description="高效无取向硅钢，新能源电机用",
    ),

    "B27R090": MaterialInfo(
        grade="B27R090",
        name="取向硅钢",
        aliases=["27RK090", "M090-27P", "变压器硅钢"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.ELECTRICAL_STEEL,
        standards=["GB/T 2521.2-2016", "IEC 60404-8-7"],
        properties=MaterialProperties(
            density=7.65,
            melting_point=1500,
            thermal_conductivity=15,
            tensile_strength=360,
            yield_strength=280,
            hardness="HV140-170",
            elongation=8,
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["薄板", "卷材"],
            blank_hint="0.27mm厚度卷材",
            heat_treatments=["高温退火"],
            heat_treatment_notes=["退火800-850℃", "定向晶粒组织", "沿轧向使用"],
            surface_treatments=["玻璃膜涂层", "磷酸盐涂层"],
            cutting_speed_range=(60, 150),
            coolant_required=False,
            warnings=["沿轧制方向使用", "横向磁性能差"],
            recommendations=["电力变压器", "配电变压器", "互感器铁芯"],
        ),
        description="取向硅钢，变压器铁芯专用",
    ),

    # -------------------------------------------------------------------------
    # 新增材料 - 不锈钢 (New Stainless Steels)
    # -------------------------------------------------------------------------
    "631": MaterialInfo(
        grade="631",
        name="沉淀硬化不锈钢",
        aliases=["17-7PH", "SUS631", "1.4568"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.STAINLESS_STEEL,
        standards=["ASTM A693", "GB/T 1220-2007"],
        properties=MaterialProperties(
            density=7.80,
            melting_point=1400,
            thermal_conductivity=16.4,
            tensile_strength=1310,
            yield_strength=1170,
            elongation=6,  # %
            hardness="HRC44-48",
            machinability="poor",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["带材", "板材", "棒材"],
            blank_hint="固溶处理态",
            heat_treatments=["TH1050", "RH950", "CH900"],
            surface_treatments=["钝化", "电解抛光"],
            cutting_speed_range=(15, 40),
            warnings=["热处理制度复杂", "需严格控制温度"],
            recommendations=["弹簧", "膜片", "紧固件", "航空航天"],
        ),
        description="17-7PH沉淀硬化不锈钢，高强度弹簧和膜片用",
    ),
    "15-5PH": MaterialInfo(
        grade="15-5PH",
        name="沉淀硬化不锈钢",
        aliases=["XM-12", "1.4545", "S15500"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.STAINLESS_STEEL,
        standards=["ASTM A564", "AMS 5659"],
        properties=MaterialProperties(
            density=7.80,
            melting_point=1400,
            thermal_conductivity=18.4,
            tensile_strength=1070,
            yield_strength=1000,
            elongation=10,  # %
            hardness="HRC35-42",
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件", "板材"],
            blank_hint="H1025状态",
            heat_treatments=["H900", "H1025", "H1075", "H1150"],
            surface_treatments=["钝化", "电解抛光"],
            cutting_speed_range=(20, 50),
            recommendations=["阀杆", "紧固件", "航空结构件"],
        ),
        description="15-5PH沉淀硬化不锈钢，焊接性好于17-4PH",
    ),
    "Nitronic50": MaterialInfo(
        grade="Nitronic50",
        name="高氮奥氏体不锈钢",
        aliases=["XM-19", "S20910", "22-13-5"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.STAINLESS_STEEL,
        standards=["ASTM A276", "ASTM A479"],
        properties=MaterialProperties(
            density=7.88,
            melting_point=1400,
            thermal_conductivity=12.8,
            tensile_strength=860,
            yield_strength=517,
            elongation=35,  # %
            hardness="HB269",
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "锻件"],
            blank_hint="固溶处理态",
            heat_treatments=["固溶处理"],
            surface_treatments=["钝化"],
            cutting_speed_range=(20, 50),
            recommendations=["海洋设备", "紧固件", "泵轴", "阀杆"],
        ),
        description="Nitronic 50高氮不锈钢，耐腐蚀强度高于316",
    ),
    # -------------------------------------------------------------------------
    # 新增材料 - 铝合金 (New Aluminum Alloys)
    # -------------------------------------------------------------------------
    "3003": MaterialInfo(
        grade="3003",
        name="防锈铝合金",
        aliases=["3003-H14", "AlMn1Cu"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ALUMINUM,
        standards=["GB/T 3190-2020", "ASTM B209"],
        properties=MaterialProperties(
            density=2.73,
            melting_point=655,
            thermal_conductivity=193,
            tensile_strength=130,
            yield_strength=110,
            elongation=20,  # %
            hardness="HB40",
            machinability="excellent",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "箔材", "管材"],
            blank_hint="H14状态板材",
            heat_treatments=["退火"],
            surface_treatments=["阳极氧化", "涂装"],
            cutting_speed_range=(200, 500),
            recommendations=["散热器", "食品包装", "化工设备"],
        ),
        description="3003防锈铝，焊接性和成形性优异",
    ),
    "1060": MaterialInfo(
        grade="1060",
        name="工业纯铝",
        aliases=["1060-O", "Al99.6"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ALUMINUM,
        standards=["GB/T 3190-2020", "ASTM B209"],
        properties=MaterialProperties(
            density=2.70,
            melting_point=660,
            thermal_conductivity=234,
            tensile_strength=70,
            yield_strength=30,
            elongation=43,  # %
            hardness="HB23",
            machinability="excellent",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "箔材", "带材"],
            blank_hint="O态板材",
            heat_treatments=["退火"],
            surface_treatments=["阳极氧化"],
            cutting_speed_range=(300, 600),
            recommendations=["电容器外壳", "散热片", "化工容器"],
        ),
        description="1060工业纯铝，导电导热性好",
    ),
    "LY12": MaterialInfo(
        grade="LY12",
        name="硬铝合金",
        aliases=["2A12-T4", "AA2024", "杜拉铝"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ALUMINUM,
        standards=["GB/T 3190-2020"],
        properties=MaterialProperties(
            density=2.78,
            melting_point=640,
            thermal_conductivity=121,
            tensile_strength=470,
            yield_strength=325,
            elongation=12,  # %
            hardness="HB120",
            machinability="good",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "锻件"],
            blank_hint="T4状态",
            heat_treatments=["固溶+自然时效"],
            surface_treatments=["阳极氧化", "喷涂"],
            cutting_speed_range=(100, 300),
            warnings=["耐腐蚀性差", "需要表面保护"],
            recommendations=["飞机蒙皮", "骨架", "铆钉"],
        ),
        description="LY12硬铝，航空结构首选",
    ),
    # -------------------------------------------------------------------------
    # 新增材料 - 工具钢 (New Tool Steels)
    # -------------------------------------------------------------------------
    "D2": MaterialInfo(
        grade="D2",
        name="高碳高铬冷作模具钢",
        aliases=["SKD11", "1.2379", "Cr12Mo1V1"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.TOOL_STEEL,
        standards=["ASTM A681", "GB/T 1299-2014"],
        properties=MaterialProperties(
            density=7.70,
            melting_point=1420,
            thermal_conductivity=20,
            tensile_strength=1930,
            yield_strength=1650,
            elongation=2,  # %
            hardness="HRC58-62",
            machinability="poor",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "锻件"],
            blank_hint="退火态",
            heat_treatments=["淬火+回火", "深冷处理"],
            heat_treatment_notes=["淬火1010-1030℃油冷", "回火200-250℃"],
            surface_treatments=["氮化", "PVD涂层"],
            cutting_speed_range=(10, 30),
            warnings=["淬火开裂风险", "需要预热"],
            recommendations=["冲裁模", "冷镦模", "剪切刀"],
        ),
        description="D2冷作模具钢，高耐磨高硬度",
    ),
    "O1": MaterialInfo(
        grade="O1",
        name="油淬冷作模具钢",
        aliases=["SKS3", "1.2510", "9CrWMn"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.TOOL_STEEL,
        standards=["ASTM A681", "GB/T 1299-2014"],
        properties=MaterialProperties(
            density=7.85,
            melting_point=1460,
            thermal_conductivity=30,
            tensile_strength=1800,
            yield_strength=1500,
            elongation=3,  # %
            hardness="HRC58-62",
            machinability="good",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材"],
            blank_hint="退火态",
            heat_treatments=["淬火+回火"],
            heat_treatment_notes=["淬火800-820℃油冷", "回火175-200℃"],
            surface_treatments=["发黑", "氮化"],
            cutting_speed_range=(20, 50),
            recommendations=["量具", "刃具", "木工刀具", "小型模具"],
        ),
        description="O1油淬工具钢，尺寸稳定性好",
    ),
    # -------------------------------------------------------------------------
    # 新增材料 - 高温合金 (New Superalloys)
    # -------------------------------------------------------------------------
    "Haynes230": MaterialInfo(
        grade="Haynes230",
        name="镍铬钨钼高温合金",
        aliases=["Haynes 230", "N06230", "2.4733"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.SUPERALLOY,
        standards=["ASTM B435", "AMS 5878"],
        properties=MaterialProperties(
            density=8.97,
            melting_point=1350,
            thermal_conductivity=8.9,
            tensile_strength=860,
            yield_strength=390,
            elongation=48,  # %
            hardness="HB210",
            machinability="poor",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材"],
            blank_hint="固溶态",
            heat_treatments=["固溶处理"],
            surface_treatments=["酸洗", "钝化"],
            cutting_speed_range=(10, 25),
            warnings=["加工硬化严重", "需要低速高进给"],
            recommendations=["燃气轮机", "工业炉", "热处理夹具"],
        ),
        description="Haynes 230高温合金，抗氧化性优异",
    ),
    # -------------------------------------------------------------------------
    # 新增材料 - 钛合金 (New Titanium Alloys)
    # -------------------------------------------------------------------------
    "Ti-6242": MaterialInfo(
        grade="Ti-6242",
        name="近α型钛合金",
        aliases=["Ti-6Al-2Sn-4Zr-2Mo", "IMI 829"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.TITANIUM,
        standards=["AMS 4975", "AMS 4976"],
        properties=MaterialProperties(
            density=4.54,
            melting_point=1660,
            thermal_conductivity=7.0,
            tensile_strength=1030,
            yield_strength=900,
            elongation=10,  # %
            hardness="HRC36",
            machinability="poor",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件", "板材"],
            blank_hint="退火态",
            heat_treatments=["退火", "固溶+时效"],
            surface_treatments=["阳极氧化", "酸洗"],
            cutting_speed_range=(15, 40),
            special_tooling=True,
            coolant_required=True,
            recommendations=["航空发动机", "压气机盘", "叶片"],
        ),
        description="Ti-6242近α型钛合金，高温蠕变性能好",
    ),
    # -------------------------------------------------------------------------
    # 新增材料 - 工程塑料 (New Engineering Plastics)
    # -------------------------------------------------------------------------
    "PVDF": MaterialInfo(
        grade="PVDF",
        name="聚偏氟乙烯",
        aliases=["Kynar", "Solef", "聚偏二氟乙烯"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.POLYMER,
        group=MaterialGroup.FLUOROPOLYMER,
        standards=["ASTM D3222"],
        properties=MaterialProperties(
            density=1.78,
            melting_point=170,
            thermal_conductivity=0.19,
            tensile_strength=50,
            yield_strength=45,
            elongation=50,  # %
            hardness="Shore D75",
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "板材", "管材"],
            blank_hint="挤出/注塑件",
            heat_treatments=["退火消除应力"],
            surface_treatments=["等离子处理"],
            cutting_speed_range=(100, 300),
            recommendations=["化工管道", "半导体设备", "锂电池隔膜"],
        ),
        description="PVDF氟塑料，耐化学性和机械强度兼备",
    ),
    "LCP": MaterialInfo(
        grade="LCP",
        name="液晶聚合物",
        aliases=["Vectra", "Zenite", "液晶高分子"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.POLYMER,
        group=MaterialGroup.ENGINEERING_PLASTIC,
        standards=["ASTM D4067"],
        properties=MaterialProperties(
            density=1.40,
            melting_point=280,
            thermal_conductivity=0.2,
            tensile_strength=180,
            yield_strength=170,
            elongation=3,  # %
            hardness="Shore D85",
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["注塑件", "挤出件"],
            blank_hint="注塑成型",
            heat_treatments=["无需"],
            surface_treatments=["等离子处理"],
            cutting_speed_range=(80, 200),
            warnings=["流动方向强度差异大", "需要特殊模具设计"],
            recommendations=["电子连接器", "SMT基座", "微型零件"],
        ),
        description="LCP液晶聚合物，尺寸稳定流动性好",
    ),
    # -------------------------------------------------------------------------
    # 新增材料 - 橡胶 (New Rubbers)
    # -------------------------------------------------------------------------
    "NBR": MaterialInfo(
        grade="NBR",
        name="丁腈橡胶",
        aliases=["Nitrile", "Buna-N", "丁二烯丙烯腈橡胶"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.POLYMER,
        group=MaterialGroup.RUBBER,
        standards=["ASTM D2000"],
        properties=MaterialProperties(
            density=1.00,
            melting_point=120,  # 使用温度上限
            thermal_conductivity=0.25,
            tensile_strength=20,
            yield_strength=10,
            elongation=450,  # %
            hardness="Shore A60-80",
            machinability="fair",
            weldability="none",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "O型圈", "油封"],
            blank_hint="模压/注射成型",
            heat_treatments=["硫化"],
            surface_treatments=["无需"],
            cutting_speed_range=(50, 150),
            warnings=["不耐臭氧", "不耐芳烃溶剂"],
            recommendations=["油封", "燃油管", "液压密封"],
        ),
        description="NBR丁腈橡胶，耐油性优异",
    ),
    "FKM": MaterialInfo(
        grade="FKM",
        name="氟橡胶",
        aliases=["Viton", "FPM", "氟弹性体"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.POLYMER,
        group=MaterialGroup.RUBBER,
        standards=["ASTM D2000"],
        properties=MaterialProperties(
            density=1.85,
            melting_point=200,  # 使用温度上限
            thermal_conductivity=0.2,
            tensile_strength=15,
            yield_strength=8,
            elongation=250,  # %
            hardness="Shore A75-90",
            machinability="fair",
            weldability="none",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "O型圈", "密封件"],
            blank_hint="模压成型",
            heat_treatments=["二次硫化"],
            surface_treatments=["无需"],
            cutting_speed_range=(50, 150),
            warnings=["不耐酮类和酯类", "高温分解有毒"],
            recommendations=["航空密封", "化工密封", "汽车油封"],
        ),
        description="FKM氟橡胶，耐高温耐化学品",
    ),
    # -------------------------------------------------------------------------
    # 新增材料 - 特种合金 (New Special Alloys)
    # -------------------------------------------------------------------------
    "HastelloyX": MaterialInfo(
        grade="HastelloyX",
        name="哈氏合金X",
        aliases=["Hastelloy X", "N06002", "2.4665"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.SUPERALLOY,
        standards=["ASTM B435", "AMS 5754"],
        properties=MaterialProperties(
            density=8.22,
            melting_point=1355,
            thermal_conductivity=9.1,
            tensile_strength=785,
            yield_strength=360,
            elongation=43,  # %
            hardness="HB200",
            machinability="poor",
            weldability="excellent",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材"],
            blank_hint="固溶态",
            heat_treatments=["固溶处理"],
            surface_treatments=["酸洗", "钝化"],
            cutting_speed_range=(10, 25),
            recommendations=["燃气轮机", "燃烧室", "核反应堆部件"],
        ),
        description="哈氏合金X，高温强度和抗氧化性优异",
    ),
    "MP35N": MaterialInfo(
        grade="MP35N",
        name="钴镍钼钛合金",
        aliases=["35N", "N30035", "Co-35Ni-20Cr-10Mo"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.SUPERALLOY,
        standards=["ASTM F562", "AMS 5758"],
        properties=MaterialProperties(
            density=8.43,
            melting_point=1350,
            thermal_conductivity=11.3,
            tensile_strength=1795,
            yield_strength=1585,
            elongation=10,  # %
            hardness="HRC50",
            machinability="poor",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "丝材", "带材"],
            blank_hint="冷加工+时效态",
            heat_treatments=["时效处理"],
            surface_treatments=["电解抛光", "钝化"],
            cutting_speed_range=(8, 20),
            special_tooling=True,
            warnings=["极难加工", "需要专用刀具"],
            recommendations=["心脏起搏器", "海底电缆", "航空紧固件"],
        ),
        description="MP35N超高强度合金，医疗器械和海洋工程用",
    ),
    "L605": MaterialInfo(
        grade="L605",
        name="钴基高温合金",
        aliases=["Haynes 25", "N10605", "Co-20Cr-15W-10Ni"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.SUPERALLOY,
        standards=["ASTM F90", "AMS 5759"],
        properties=MaterialProperties(
            density=9.13,
            melting_point=1380,
            thermal_conductivity=9.4,
            tensile_strength=1000,
            yield_strength=460,
            elongation=50,  # %
            hardness="HB277",
            machinability="poor",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材"],
            blank_hint="退火态",
            heat_treatments=["固溶处理"],
            surface_treatments=["酸洗", "钝化"],
            cutting_speed_range=(10, 25),
            recommendations=["心脏瓣膜", "燃气轮机", "核工业"],
        ),
        description="L605钴基合金，高温强度和延展性优异",
    ),

    # -------------------------------------------------------------------------
    # 新增材料 - 补充至300种 (2026-01-29)
    # -------------------------------------------------------------------------

    # 铜合金补充
    "C11000": MaterialInfo(
        grade="C11000",
        name="电解铜",
        aliases=["ETP Copper", "纯铜", "T2"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.COPPER,
        standards=["ASTM B152", "GB/T 5231"],
        properties=MaterialProperties(
            density=8.94,
            melting_point=1083,
            thermal_conductivity=391,
            tensile_strength=220,
            yield_strength=70,
            elongation=45,  # %
            hardness="HB40-60",
            machinability="poor",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "管材", "线材"],
            blank_hint="退火态",
            heat_treatments=["退火"],
            surface_treatments=["酸洗", "电镀"],
            cutting_speed_range=(100, 300),
            recommendations=["电气导体", "散热器", "装饰件"],
        ),
        description="高纯电解铜，导电导热性极佳",
    ),

    "C26000": MaterialInfo(
        grade="C26000",
        name="黄铜",
        aliases=["Cartridge Brass", "70-30黄铜", "H70"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.COPPER,
        standards=["ASTM B36", "GB/T 5231"],
        properties=MaterialProperties(
            density=8.53,
            melting_point=955,
            thermal_conductivity=120,
            tensile_strength=340,
            yield_strength=105,
            elongation=65,  # %
            hardness="HB65-95",
            machinability="good",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "带材", "管材"],
            blank_hint="退火态或半硬态",
            heat_treatments=["去应力退火"],
            surface_treatments=["抛光", "电镀", "钝化"],
            cutting_speed_range=(60, 150),
            recommendations=["弹壳", "散热器", "装饰件"],
        ),
        description="70/30黄铜，冷加工性能优异",
    ),

    "C52100": MaterialInfo(
        grade="C52100",
        name="磷青铜",
        aliases=["Phosphor Bronze", "QSn8-0.3", "CuSn8"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.TIN_BRONZE,
        standards=["ASTM B103", "GB/T 5231"],
        properties=MaterialProperties(
            density=8.80,
            melting_point=1000,
            thermal_conductivity=50,
            tensile_strength=550,
            yield_strength=410,
            elongation=10,  # %
            hardness="HB120-180",
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "带材", "线材"],
            blank_hint="硬态或弹性态",
            heat_treatments=["去应力退火"],
            surface_treatments=["电镀", "钝化"],
            cutting_speed_range=(40, 100),
            recommendations=["弹簧", "电接触件", "轴承"],
        ),
        description="高弹性磷青铜，耐磨耐腐蚀",
    ),

    # 镁合金补充
    "WE43": MaterialInfo(
        grade="WE43",
        name="稀土镁合金",
        aliases=["Mg-Y-RE", "稀土镁"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.MAGNESIUM,
        standards=["ASTM B403", "AMS 4427"],
        properties=MaterialProperties(
            density=1.84,
            melting_point=545,
            thermal_conductivity=51,
            tensile_strength=250,
            yield_strength=180,
            elongation=4,  # %
            hardness="HB85",
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["铸件", "板材"],
            blank_hint="T6状态",
            heat_treatments=["固溶时效"],
            surface_treatments=["阳极氧化", "化学转化"],
            cutting_speed_range=(200, 600),
            warnings=["镁屑易燃", "避免水基切削液"],
            recommendations=["航空航天", "赛车部件", "医疗植入物"],
        ),
        description="高温稀土镁合金，生物相容性好",
    ),

    "AM60": MaterialInfo(
        grade="AM60",
        name="压铸镁合金",
        aliases=["Mg-Al-Mn", "AM60B"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.CAST_MAGNESIUM,
        standards=["ASTM B94", "SAE J465"],
        properties=MaterialProperties(
            density=1.80,
            melting_point=615,
            thermal_conductivity=61,
            tensile_strength=240,
            yield_strength=130,
            elongation=8,  # %
            hardness="HB65",
            machinability="excellent",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["压铸件"],
            blank_hint="压铸态",
            heat_treatments=["去应力"],
            surface_treatments=["喷涂", "电泳", "阳极氧化"],
            cutting_speed_range=(300, 900),
            warnings=["镁屑易燃", "需专用灭火设备"],
            recommendations=["汽车方向盘", "座椅骨架", "仪表板支架"],
        ),
        description="汽车用压铸镁合金，延展性好",
    ),

    # 不锈钢补充
    "13-8Mo": MaterialInfo(
        grade="13-8Mo",
        name="沉淀硬化不锈钢",
        aliases=["XM-13", "S13800"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.STAINLESS_STEEL,
        standards=["ASTM A564", "AMS 5629"],
        properties=MaterialProperties(
            density=7.76,
            melting_point=1440,
            thermal_conductivity=14,
            tensile_strength=1520,
            yield_strength=1410,
            elongation=10,  # %
            hardness="HRC47-49",
            machinability="fair",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "锻件"],
            blank_hint="固溶态或时效态",
            heat_treatments=["固溶处理", "时效硬化"],
            surface_treatments=["钝化", "化学处理"],
            cutting_speed_range=(20, 50),
            recommendations=["航空结构件", "核工业", "高强度紧固件"],
        ),
        description="高强度沉淀硬化不锈钢，韧性好",
    ),

    "22Cr双相钢": MaterialInfo(
        grade="22Cr双相钢",
        name="双相不锈钢",
        aliases=["2205", "S31803", "SAF2205"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.STAINLESS_STEEL,
        standards=["ASTM A240", "EN 1.4462"],
        properties=MaterialProperties(
            density=7.80,
            melting_point=1450,
            thermal_conductivity=19,
            tensile_strength=620,
            yield_strength=450,
            elongation=25,  # %
            hardness="HB290",
            machinability="fair",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "管材", "棒材"],
            blank_hint="固溶态",
            heat_treatments=["固溶退火"],
            surface_treatments=["酸洗钝化", "电化学抛光"],
            cutting_speed_range=(30, 60),
            recommendations=["化工设备", "海洋工程", "压力容器"],
        ),
        description="22%铬双相钢，强度高耐蚀性好",
    ),

    # 工具钢补充
    "A2": MaterialInfo(
        grade="A2",
        name="空淬冷作模具钢",
        aliases=["AISI A2", "1.2363", "Cr5Mo1V"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.FERROUS,
        group=MaterialGroup.TOOL_STEEL,
        standards=["ASTM A681", "GB/T 1299-2014"],
        properties=MaterialProperties(
            density=7.86,
            melting_point=1425,
            thermal_conductivity=24,
            tensile_strength=1930,
            yield_strength=1590,
            elongation=1.5,  # %
            hardness="HRC57-62",
            machinability="fair",
            weldability="poor",
        ),
        process=ProcessRecommendation(
            blank_forms=["板材", "棒材", "块料"],
            blank_hint="退火态",
            heat_treatments=["淬火", "回火", "深冷处理"],
            surface_treatments=["氮化", "镀铬"],
            cutting_speed_range=(15, 35),
            recommendations=["冲裁模", "成型模", "切边模"],
        ),
        description="空气淬火模具钢，变形小韧性好",
    ),

    # 工程塑料补充
    "PPSU": MaterialInfo(
        grade="PPSU",
        name="聚苯砜",
        aliases=["Polyphenylsulfone", "Radel"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.POLYMER,
        group=MaterialGroup.ENGINEERING_PLASTIC,
        standards=["ASTM D6394"],
        properties=MaterialProperties(
            density=1.29,
            melting_point=220,  # 玻璃化转变温度
            thermal_conductivity=0.22,
            tensile_strength=70,
            yield_strength=55,
            elongation=60,  # %
            hardness="Rockwell M85",
            machinability="excellent",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["注塑件", "板材", "棒材"],
            blank_hint="注塑或机加工",
            heat_treatments=["退火去应力"],
            surface_treatments=["等离子处理", "喷涂"],
            cutting_speed_range=(100, 300),
            recommendations=["医疗器械", "婴儿用品", "航空内饰"],
        ),
        description="高性能工程塑料，耐高温耐水解",
    ),

    # 特种合金补充
    "Waspaloy": MaterialInfo(
        grade="Waspaloy",
        name="镍基高温合金",
        aliases=["UNS N07001", "W.Nr 2.4654"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.SUPERALLOY,
        standards=["AMS 5544", "AMS 5828"],
        properties=MaterialProperties(
            density=8.19,
            melting_point=1330,
            thermal_conductivity=10.7,
            tensile_strength=1280,
            yield_strength=795,
            elongation=25,  # %
            hardness="HRC35-42",
            machinability="poor",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件", "板材"],
            blank_hint="固溶或时效态",
            heat_treatments=["固溶处理", "双重时效"],
            surface_treatments=["酸洗", "热障涂层"],
            cutting_speed_range=(8, 20),
            warnings=["加工硬化严重", "需刚性夹具"],
            recommendations=["燃气涡轮", "航空发动机盘", "环件"],
        ),
        description="镍基时效硬化高温合金，760℃长期使用",
    ),

    "Rene41": MaterialInfo(
        grade="Rene41",
        name="镍基高温合金",
        aliases=["Alloy 41", "UNS N07041"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.SUPERALLOY,
        standards=["AMS 5545", "AMS 5712"],
        properties=MaterialProperties(
            density=8.25,
            melting_point=1315,
            thermal_conductivity=10.4,
            tensile_strength=1310,
            yield_strength=850,
            elongation=14,  # %
            hardness="HRC36-44",
            machinability="poor",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["棒材", "锻件", "板材"],
            blank_hint="固溶或时效态",
            heat_treatments=["固溶处理", "时效硬化"],
            surface_treatments=["酸洗", "喷丸"],
            cutting_speed_range=(6, 18),
            warnings=["刀具磨损快", "需大量冷却"],
            recommendations=["航空发动机", "燃气轮机", "加力燃烧室"],
        ),
        description="高强度镍基高温合金，980℃短时使用",
    ),

    "Elgiloy": MaterialInfo(
        grade="Elgiloy",
        name="钴铬镍钼合金",
        aliases=["UNS R30003", "Phynox"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.SUPERALLOY,
        standards=["ASTM F1058", "AMS 5833"],
        properties=MaterialProperties(
            density=8.30,
            melting_point=1340,
            thermal_conductivity=11.1,
            tensile_strength=1860,
            yield_strength=1450,
            elongation=8,  # %
            hardness="HRC50-52",
            machinability="poor",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["线材", "带材", "棒材"],
            blank_hint="加工硬化态",
            heat_treatments=["时效硬化"],
            surface_treatments=["电解抛光", "钝化"],
            cutting_speed_range=(5, 15),
            recommendations=["手表发条", "医疗器械", "弹簧"],
        ),
        description="钴基弹簧合金，耐腐蚀高弹性",
    ),

    "Phynox": MaterialInfo(
        grade="Phynox",
        name="钴铬镍钼钨合金",
        aliases=["UNS R30003", "Co-Cr-Ni-Mo-W"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.SUPERALLOY,
        standards=["ASTM F1058", "ISO 5832-7"],
        properties=MaterialProperties(
            density=8.30,
            melting_point=1350,
            thermal_conductivity=11.0,
            tensile_strength=1900,
            yield_strength=1500,
            elongation=6,  # %
            hardness="HRC51-53",
            machinability="poor",
            weldability="fair",
        ),
        process=ProcessRecommendation(
            blank_forms=["线材", "带材"],
            blank_hint="冷加工硬化态",
            heat_treatments=["时效处理"],
            surface_treatments=["电解抛光", "钝化"],
            cutting_speed_range=(5, 15),
            recommendations=["医疗植入物", "心脏起搏器", "精密弹簧"],
        ),
        description="医用钴基合金，生物相容性极佳",
    ),

    # 铝合金补充 (达到300种)
    "6005": MaterialInfo(
        grade="6005",
        name="中强度铝合金",
        aliases=["6005A", "AlSiMg"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ALUMINUM,
        standards=["ASTM B221", "GB/T 3190"],
        properties=MaterialProperties(
            density=2.70,
            melting_point=655,
            thermal_conductivity=180,
            tensile_strength=270,
            yield_strength=225,
            elongation=10,  # %
            hardness="HB95",
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["型材", "管材", "棒材"],
            blank_hint="T5或T6状态",
            heat_treatments=["固溶时效"],
            surface_treatments=["阳极氧化", "喷涂"],
            cutting_speed_range=(150, 400),
            recommendations=["轨道交通", "汽车结构", "建筑型材"],
        ),
        description="轨道交通用铝合金，焊接性好",
    ),

    "6101": MaterialInfo(
        grade="6101",
        name="导电铝合金",
        aliases=["EC6101", "AlMgSi"],
        category=MaterialCategory.METAL,
        sub_category=MaterialSubCategory.NON_FERROUS,
        group=MaterialGroup.ALUMINUM,
        standards=["ASTM B317", "GB/T 3190"],
        properties=MaterialProperties(
            density=2.70,
            melting_point=655,
            thermal_conductivity=218,
            tensile_strength=200,
            yield_strength=170,
            elongation=15,  # %
            hardness="HB71",
            machinability="good",
            weldability="good",
        ),
        process=ProcessRecommendation(
            blank_forms=["型材", "管材", "母排"],
            blank_hint="T6状态",
            heat_treatments=["固溶时效"],
            surface_treatments=["镀锡", "镀银"],
            cutting_speed_range=(150, 400),
            recommendations=["输电线路", "电气母排", "散热器"],
        ),
        description="高导电率铝合金，用于电气导体",
    ),

    # -------------------------------------------------------------------------
    # 组合件 (Assembly)
    # -------------------------------------------------------------------------
    "组焊件": MaterialInfo(
        grade="组焊件",
        name="组焊件",
        aliases=["焊接组件"],
        category=MaterialCategory.COMPOSITE,
        sub_category=MaterialSubCategory.ASSEMBLY,
        group=MaterialGroup.WELDED_ASSEMBLY,
        process=ProcessRecommendation(
            blank_hint="多零件焊接组装",
            heat_treatments=["去应力退火"],
            surface_treatments=["喷砂", "喷漆"],
            warnings=["焊后需消除应力", "注意焊接变形"],
            recommendations=["焊后时效或振动去应力"],
        ),
        description="多零件焊接组成",
    ),

    "组合件": MaterialInfo(
        grade="组合件",
        name="组合件",
        aliases=["组件", "装配件"],
        category=MaterialCategory.COMPOSITE,
        sub_category=MaterialSubCategory.ASSEMBLY,
        group=MaterialGroup.MECHANICAL_ASSEMBLY,
        process=ProcessRecommendation(
            blank_hint="多零件机械组装",
            recommendations=["按装配工艺执行"],
        ),
        description="多零件机械连接组成",
    ),
}


# ============================================================================
# 材料匹配模式（用于模糊匹配）
# ============================================================================

MATERIAL_MATCH_PATTERNS: List[Tuple[str, str]] = [
    # (正则模式, 材料牌号)
    # 不锈钢 - 奥氏体
    (r"S304\d*", "S30408"),
    (r"0?Cr18Ni9", "S30408"),
    (r"S316\d*", "S31603"),
    (r"00Cr17Ni14Mo2", "S31603"),
    (r"(?:1\.)?4301", "S30408"),
    (r"(?:1\.)?4404", "S31603"),
    (r"SUS304", "S30408"),
    (r"SUS316L?", "S31603"),
    # AISI/ASTM 标准变体
    (r"AISI[-\s]?304", "S30408"),
    (r"AISI[-\s]?316L?", "S31603"),
    (r"ASTM[-\s]?304", "S30408"),
    (r"ASTM[-\s]?316L?", "S31603"),
    # 常见英文变体
    (r"304[-\s]?(?:stainless|SS|steel)", "S30408"),
    (r"316[-\s]?L?[-\s]?(?:stainless|SS|steel)", "S31603"),
    (r"(?:stainless|SS)[-\s]?304", "S30408"),
    (r"(?:stainless|SS)[-\s]?316L?", "S31603"),
    # 带空格的变体 (316 L, 304 L)
    (r"304\s+L", "S30408"),
    (r"316\s+L", "S31603"),
    # 不锈钢 - 钛/铌稳定型
    (r"321", "321"),
    (r"S32100", "321"),
    (r"SUS321", "321"),
    (r"(?:1\.)?4541", "321"),
    (r"0?Cr18Ni10Ti", "321"),
    (r"1Cr18Ni9Ti", "321"),
    (r"AISI[-\s]?321", "321"),
    (r"347", "347"),
    (r"S34700", "347"),
    (r"SUS347", "347"),
    (r"(?:1\.)?4550", "347"),
    (r"0?Cr18Ni11Nb", "347"),
    (r"AISI[-\s]?347", "347"),
    # 不锈钢 - 铁素体
    (r"430", "430"),
    (r"S43000", "430"),
    (r"SUS430", "430"),
    (r"(?:1\.)?4016", "430"),
    (r"1Cr17", "430"),
    (r"AISI[-\s]?430", "430"),
    # 不锈钢 - 马氏体
    (r"410", "410"),
    (r"S41000", "410"),
    (r"SUS410", "410"),
    (r"(?:1\.)?4006", "410"),
    (r"1Cr13", "410"),
    (r"AISI[-\s]?410", "410"),
    # 不锈钢 - 沉淀硬化
    (r"17[-\s]?4[-\s]?PH", "17-4PH"),
    (r"S17400", "17-4PH"),
    (r"SUS630", "17-4PH"),
    (r"(?:1\.)?4542", "17-4PH"),
    (r"0?Cr17Ni4Cu4Nb", "17-4PH"),
    (r"630不?锈?钢?", "17-4PH"),
    (r"AISI[-\s]?630", "17-4PH"),
    # 不锈钢 - 超级奥氏体
    (r"904L", "904L"),
    (r"N08904", "904L"),
    (r"SUS890L", "904L"),
    (r"(?:1\.)?4539", "904L"),
    (r"00Cr20Ni25Mo4\.?5Cu", "904L"),
    (r"254[-\s]?SMO", "254SMO"),
    (r"S31254", "254SMO"),
    (r"(?:1\.)?4547", "254SMO"),
    (r"00Cr20Ni18Mo6CuN", "254SMO"),
    (r"316Ti", "316Ti"),
    (r"S31635", "316Ti"),
    (r"SUS316Ti", "316Ti"),
    (r"(?:1\.)?4571", "316Ti"),
    (r"0?Cr17Ni12Mo2Ti", "316Ti"),

    # 碳素钢
    (r"Q235[A-D]?", "Q235B"),
    (r"A3钢?(?![0-9])", "Q235B"),
    (r"Q345[A-E]?R?", "Q345R"),
    (r"16MnR?", "Q345R"),
    (r"45[#钢]?", "45"),
    (r"S45C", "45"),
    (r"C45", "45"),
    (r"AISI[-\s]?1045", "45"),
    (r"SAE[-\s]?1045", "45"),
    (r"20[#钢]", "20"),  # 需要后缀才能匹配，避免与2205/2507冲突
    (r"^20$", "20"),  # 精确匹配20
    (r"AISI[-\s]?1020", "20"),
    (r"SAE[-\s]?1020", "20"),
    (r"10[#钢]", "10"),  # 需要后缀才能匹配
    (r"^10$", "10"),  # 精确匹配10
    (r"C10", "10"),
    (r"AISI[-\s]?1010", "10"),
    (r"SAE[-\s]?1010", "10"),
    (r"15[#钢]", "15"),  # 需要后缀才能匹配
    (r"^15$", "15"),  # 精确匹配15
    (r"C15", "15"),
    (r"AISI[-\s]?1015", "15"),
    (r"SAE[-\s]?1015", "15"),
    (r"(?<![A-Z0-9])35[#钢]?(?![0-9])", "35"),  # 35钢，避免与HT350冲突
    (r"S35C", "35"),
    (r"C35(?![0-9])", "35"),
    (r"AISI[-\s]?1035", "35"),
    (r"SAE[-\s]?1035", "35"),
    (r"(?<![A-Z0-9])50[#钢](?![0-9])", "50"),  # 50钢，需后缀避免与50W冲突
    (r"^50$", "50"),  # 精确匹配50
    (r"S50C", "50"),
    (r"C50(?![0-9])", "50"),
    (r"AISI[-\s]?1050", "50"),
    (r"SAE[-\s]?1050", "50"),
    (r"65Mn", "65Mn"),

    # 合金钢
    (r"40Cr\d*", "40Cr"),
    (r"AISI[-\s]?5140", "40Cr"),
    (r"SAE[-\s]?5140", "40Cr"),
    (r"42CrMo\d*", "42CrMo"),
    (r"AISI[-\s]?4140", "42CrMo"),
    (r"SAE[-\s]?4140", "42CrMo"),
    (r"SCM440", "42CrMo"),
    # 轴承钢
    (r"GCr18Mo", "GCr18Mo"),
    (r"100CrMo7", "GCr18Mo"),
    (r"SUJ5", "GCr18Mo"),
    (r"A485", "GCr18Mo"),
    (r"航空轴承钢", "GCr18Mo"),
    (r"GCr15SiMn", "GCr15SiMn"),
    (r"100CrMnSi", "GCr15SiMn"),
    (r"SUJ3", "GCr15SiMn"),
    (r"GCr15(?!Si)", "GCr15"),
    (r"52100(?!\s*Mod)", "GCr15"),
    (r"SUJ2", "GCr15"),
    (r"100Cr6", "GCr15"),
    (r"轴承钢", "GCr15"),
    (r"20CrMnTi[H]?", "20CrMnTi"),
    (r"SCM420H?", "20CrMnTi"),
    (r"20Cr[4]?", "20Cr"),
    (r"5120", "20Cr"),
    (r"SCr420", "20Cr"),
    (r"38CrMoAl[A]?", "38CrMoAl"),
    (r"SACM645", "38CrMoAl"),
    (r"氮化钢", "38CrMoAl"),
    (r"30CrMnSi[A]?", "30CrMnSi"),
    (r"渗碳钢", "20CrMnTi"),

    # 耐热钢/高温合金
    (r"310S", "310S"),
    (r"S31008", "310S"),
    (r"SUS310S", "310S"),
    (r"(?:1\.)?4845", "310S"),
    (r"0?Cr25Ni20", "310S"),
    (r"GH3030", "GH3030"),
    (r"GH30(?!6)", "GH3030"),
    (r"Nimonic[-\s]?75", "GH3030"),
    (r"GH4169", "GH4169"),
    (r"GH169", "GH4169"),
    (r"GH4099", "GH4099"),
    (r"Waspaloy", "GH4099"),
    # 高温合金补充 (China standard grades as primary, A-286 remains separate)
    (r"GH2132", "GH2132"),
    (r"SUH660", "GH2132"),
    (r"Incoloy[-\s]?A[-\s]?286", "GH2132"),
    (r"1\.4980", "GH2132"),
    (r"铁基高温合金", "GH2132"),
    (r"K403", "K403"),
    (r"Mar[-\s]?M[-\s]?246", "K403"),
    (r"IN[-\s]?100", "K403"),
    (r"K418", "K418"),
    (r"IN[-\s]?738", "K418"),
    (r"铸造高温合金", "K403"),
    (r"涡轮叶片材料", "K418"),

    # 铸铁 - 灰铸铁 (先精确匹配)
    (r"HT[-\s]?300", "HT300"),
    (r"FC[-\s]?300", "HT300"),
    (r"HT[-\s]?250", "HT250"),
    (r"FC[-\s]?250", "HT250"),
    (r"HT[-\s]?200", "HT200"),
    (r"FC[-\s]?200", "HT200"),
    (r"HT\d+", "HT200"),  # 其他灰铁通配
    (r"灰铁\d*", "HT200"),
    (r"灰铸?铁", "HT200"),
    # 铸铁 - 球墨铸铁 (先精确匹配)
    (r"QT[-\s]?700[-\s]?2?", "QT700-2"),
    (r"FCD[-\s]?700", "QT700-2"),
    (r"QT[-\s]?600[-\s]?3?", "QT600-3"),
    (r"FCD[-\s]?600", "QT600-3"),
    (r"QT[-\s]?500[-\s]?7?", "QT500-7"),
    (r"FCD[-\s]?500", "QT500-7"),
    (r"QT[-\s]?400[-\s]?\d*", "QT400"),
    (r"FCD[-\s]?400", "QT400"),
    (r"QT\d+-?\d*", "QT400"),  # 其他球铁通配
    (r"球铁\d*", "QT400"),
    (r"球墨铸?铁", "QT400"),

    # 耐磨铸铁
    (r"Ni[-\s]?Hard[-\s]?1", "NiHard1"),
    (r"NiHard[-\s]?1", "NiHard1"),
    (r"NiCr4", "NiHard1"),
    (r"Ni[-\s]?Hard[-\s]?4", "NiHard4"),
    (r"NiHard[-\s]?4", "NiHard4"),
    (r"NiCrMo", "NiHard4"),
    (r"Cr[-\s]?26(?:Mo)?", "Cr26"),
    (r"KmTBCr26", "Cr26"),
    (r"26%?铬铸铁", "Cr26"),
    (r"高铬(?:钼)?铸铁", "Cr26"),
    (r"镍硬铸铁", "NiHard1"),
    (r"耐磨铸铁", "NiHard1"),

    # 蠕墨铸铁
    (r"RuT[-\s]?400", "RuT400"),
    (r"CGI[-\s]?400", "RuT400"),
    (r"GJV[-\s]?400", "RuT400"),
    (r"RuT[-\s]?350", "RuT350"),
    (r"CGI[-\s]?350", "RuT350"),
    (r"GJV[-\s]?350", "RuT350"),
    (r"RuT[-\s]?300", "RuT300"),
    (r"CGI[-\s]?300", "RuT300"),
    (r"GJV[-\s]?300", "RuT300"),
    (r"蠕铁\d*", "RuT300"),
    (r"蠕墨铸?铁", "RuT300"),

    # 可锻铸铁
    (r"KTZ[-\s]?550[-\s]?0?4", "KTZ550-04"),
    (r"P[-\s]?550[-\s]?0?4", "KTZ550-04"),
    (r"KTZ[-\s]?450[-\s]?0?6", "KTZ450-06"),
    (r"P[-\s]?450[-\s]?0?6", "KTZ450-06"),
    (r"KTH[-\s]?300[-\s]?0?6", "KTH300-06"),
    (r"B[-\s]?300[-\s]?0?6", "KTH300-06"),
    (r"黑心铸铁", "KTH300-06"),
    (r"黑心可锻", "KTH300-06"),
    (r"珠光体可锻", "KTZ450-06"),
    (r"可锻铸?铁", "KTH300-06"),

    # 铸造镁合金
    (r"ZM[-\s]?5", "ZM5"),
    (r"AZ91[AD]?压铸", "ZM5"),
    (r"压铸镁合金", "ZM5"),
    (r"AM60[AB]?", "AM60B"),
    (r"MgAl6Mn", "AM60B"),
    (r"高韧镁合金", "AM60B"),
    (r"AZ63", "AZ63"),
    (r"MgAl6Zn3", "AZ63"),
    (r"砂型镁合金", "AZ63"),
    (r"铸造镁合金", "ZM5"),

    # 粉末冶金材料
    (r"Fe[-\s]?Cu[-\s]?C", "Fe-Cu-C"),
    (r"FC[-\s]?0205", "Fe-Cu-C"),
    (r"SINT[-\s]?C11", "Fe-Cu-C"),
    (r"铁基粉末冶金", "Fe-Cu-C"),
    (r"Fe[-\s]?Ni[-\s]?Cu", "Fe-Ni-Cu"),
    (r"FN[-\s]?0205", "Fe-Ni-Cu"),
    (r"SINT[-\s]?D11", "Fe-Ni-Cu"),
    (r"高强度粉末冶金", "Fe-Ni-Cu"),
    (r"316L[-\s]?PM", "316L-PM"),
    (r"SS[-\s]?316L粉末", "316L-PM"),
    (r"MIM[-\s]?316L", "316L-PM"),
    (r"不锈钢粉末冶金", "316L-PM"),
    (r"粉末冶金", "Fe-Cu-C"),

    # 硬质合金
    (r"YG[-\s]?8", "YG8"),
    (r"K[-\s]?30硬质合金?", "YG8"),
    (r"WC[-\s]?8Co", "YG8"),
    (r"YT[-\s]?15", "YT15"),
    (r"P[-\s]?15硬质合金?", "YT15"),
    (r"WC[-\s]?TiC[-\s]?15Co", "YT15"),
    (r"YW[-\s]?1", "YW1"),
    (r"M[-\s]?10硬质合金?", "YW1"),
    (r"钨钴合金", "YG8"),
    (r"钨钛合金", "YT15"),
    (r"硬质合金", "YG8"),

    # 结构陶瓷
    (r"Al2O3[-\s]?99", "Al2O3-99"),
    (r"99[%]?氧化铝", "Al2O3-99"),
    (r"99瓷", "Al2O3-99"),
    (r"高纯氧化铝", "Al2O3-99"),
    (r"刚玉陶瓷", "Al2O3-99"),
    (r"Si3N4", "Si3N4"),
    (r"氮化硅", "Si3N4"),
    (r"SRBSN", "Si3N4"),
    (r"ZrO2[-\s]?3Y", "ZrO2-3Y"),
    (r"3Y[-\s]?TZP", "ZrO2-3Y"),
    (r"氧化锆陶瓷?", "ZrO2-3Y"),
    (r"TZP陶瓷?", "ZrO2-3Y"),
    (r"氧化铝陶瓷", "Al2O3-99"),
    (r"结构陶瓷", "Al2O3-99"),

    # 难熔金属
    (r"Mo[-\s]?1", "Mo-1"),
    (r"纯钼", "Mo-1"),
    (r"钼板", "Mo-1"),
    (r"钼棒", "Mo-1"),
    (r"TZM钼?合金?", "TZM"),
    (r"Mo[-\s]?TZM", "TZM"),
    (r"高温钼合金", "TZM"),
    (r"W[-\s]?1(?!Cu)", "W-1"),
    (r"纯钨", "W-1"),
    (r"钨板", "W-1"),
    (r"钨棒", "W-1"),
    (r"Ta[-\s]?1", "Ta-1"),
    (r"纯钽", "Ta-1"),
    (r"钽板", "Ta-1"),
    (r"钽棒", "Ta-1"),
    (r"难熔金属", "Mo-1"),

    # 铝青铜
    (r"QAl[-\s]?9[-\s]?4", "QAl9-4"),
    (r"CuAl9Fe4", "QAl9-4"),
    (r"C62300", "QAl9-4"),
    (r"AB[-\s]?2", "QAl9-4"),
    (r"铝铁青铜", "QAl9-4"),
    (r"QAl[-\s]?10[-\s]?4[-\s]?4", "QAl10-4-4"),
    (r"CuAl10Ni5Fe5", "QAl10-4-4"),
    (r"C63000", "QAl10-4-4"),
    (r"NAB", "QAl10-4-4"),
    (r"镍铝青铜", "QAl10-4-4"),
    (r"铝青铜", "QAl9-4"),

    # 铍铜
    (r"QBe[-\s]?2(?!\.)", "QBe2"),
    (r"CuBe[-\s]?2(?!\.)", "QBe2"),
    (r"C17200", "QBe2"),
    (r"QBe[-\s]?1\.9", "QBe1.9"),
    (r"CuBe[-\s]?1\.9", "QBe1.9"),
    (r"C17000", "QBe1.9"),
    (r"低铍铜", "QBe1.9"),
    (r"CuNi2Si", "CuNi2Si"),
    (r"C70250", "CuNi2Si"),
    (r"铜镍硅", "CuNi2Si"),
    (r"无铍铜", "CuNi2Si"),
    (r"铍铜", "QBe2"),
    (r"BeCu", "QBe2"),
    (r"铍青铜", "QBe2"),

    # 无铅焊锡 (Lead-Free Solder)
    (r"SAC[-\s]?305", "SAC305"),
    (r"Sn96\.?5Ag3\.?0Cu0\.?5", "SAC305"),
    (r"Sn[-\s]?3\.?0Ag[-\s]?0\.?5Cu", "SAC305"),
    (r"SAC[-\s]?387", "SAC387"),
    (r"Sn95\.?5Ag3\.?8Cu0\.?7", "SAC387"),
    (r"Sn[-\s]?3\.?8Ag[-\s]?0\.?7Cu", "SAC387"),
    (r"Sn99\.?3Cu0\.?7", "Sn99.3Cu0.7"),
    (r"SN100C", "Sn99.3Cu0.7"),
    (r"SnCu0\.?7", "Sn99.3Cu0.7"),
    (r"Sn[-\s]?0\.?7Cu", "Sn99.3Cu0.7"),
    (r"无铅焊锡", "SAC305"),
    (r"无铅焊料", "SAC305"),

    # 钎焊合金 (Brazing Alloy)
    (r"BAg[-\s]?5", "BAg-5"),
    (r"HL[-\s]?302", "BAg-5"),
    (r"45[%％]?银钎料", "BAg-5"),
    (r"银钎料", "BAg-5"),
    (r"BCu[-\s]?1", "BCu-1"),
    (r"HL[-\s]?101", "BCu-1"),
    (r"99\.?9[%％]?Cu钎料", "BCu-1"),
    (r"纯铜钎料", "BCu-1"),
    (r"BNi[-\s]?2", "BNi-2"),
    (r"HL[-\s]?401", "BNi-2"),
    (r"镍基钎料", "BNi-2"),

    # 形状记忆合金 (Shape Memory Alloy)
    (r"NiTi(?:[-\s]?55)?", "NiTi"),
    (r"Nitinol", "NiTi"),
    (r"TiNi", "NiTi"),
    (r"镍钛合金", "NiTi"),
    (r"镍钛记忆合金", "NiTi"),
    (r"形状记忆合金", "NiTi"),
    (r"CuZnAl(?:[-\s]?SMA)?", "CuZnAl"),
    (r"Cu[-\s]?Zn[-\s]?Al", "CuZnAl"),
    (r"铜锌铝", "CuZnAl"),
    (r"铜基记忆合金", "CuZnAl"),
    (r"CuAlNi(?:[-\s]?SMA)?", "CuAlNi"),
    (r"Cu[-\s]?Al[-\s]?Ni", "CuAlNi"),
    (r"铜铝镍", "CuAlNi"),

    # 电触头材料 (Electrical Contact Material)
    (r"AgCdO(?:[-\s]?12)?", "AgCdO"),
    (r"Ag[/]?CdO", "AgCdO"),
    (r"银镉触点", "AgCdO"),
    (r"银氧化镉", "AgCdO"),
    (r"AgSnO2(?:[-\s]?12)?", "AgSnO2"),
    (r"Ag[/]?SnO2", "AgSnO2"),
    (r"银锡触点", "AgSnO2"),
    (r"银氧化锡", "AgSnO2"),
    (r"无镉触点", "AgSnO2"),
    (r"CuW(?:70)?", "CuW"),
    (r"WCu(?:30)?", "CuW"),
    (r"Cu[-\s]?W", "CuW"),
    (r"W[-\s]?Cu", "CuW"),
    (r"钨铜", "CuW"),
    (r"电触头", "AgSnO2"),
    (r"触点材料", "AgSnO2"),

    # 轴承合金 (Bearing Alloy)
    (r"ZChSnSb11[-\s]?6", "ZChSnSb11-6"),
    (r"SnSb11Cu6", "ZChSnSb11-6"),
    (r"Babbitt", "ZChSnSb11-6"),
    (r"巴氏合金", "ZChSnSb11-6"),
    (r"白合金", "ZChSnSb11-6"),
    (r"锡基轴承合金", "ZChSnSb11-6"),
    (r"ZChPbSb16[-\s]?16[-\s]?2", "ZChPbSb16-16-2"),
    (r"PbSb16Sn16Cu2", "ZChPbSb16-16-2"),
    (r"16[-\s]?16[-\s]?2", "ZChPbSb16-16-2"),
    (r"铅基轴承合金", "ZChPbSb16-16-2"),
    (r"CuPb24Sn4?", "CuPb24Sn4"),
    (r"铅青铜轴承", "CuPb24Sn4"),
    (r"SAE[-\s]?49", "CuPb24Sn4"),
    (r"轴瓦材料", "CuPb24Sn4"),

    # 热电偶合金 (Thermocouple Alloy)
    (r"Chromel(?:[-\s]?P)?", "Chromel"),
    (r"NiCr10", "Chromel"),
    (r"K型热电偶正极", "Chromel"),
    (r"镍铬热电偶", "Chromel"),
    (r"Alumel", "Alumel"),
    (r"NiAl3", "Alumel"),
    (r"K型热电偶负极", "Alumel"),
    (r"镍铝热电偶", "Alumel"),
    (r"Constantan", "Constantan"),
    (r"CuNi44", "Constantan"),
    (r"6J40", "Constantan"),
    (r"康铜", "Constantan"),
    (r"J型热电偶", "Constantan"),
    (r"T型热电偶", "Constantan"),
    (r"热电偶丝", "Chromel"),

    # 永磁材料 (Permanent Magnet)
    (r"NdFeB", "NdFeB"),
    (r"Nd[-\s]?Fe[-\s]?B", "NdFeB"),
    (r"N[3-5][0-9][HMSH]*$", "NdFeB"),
    (r"钕铁硼", "NdFeB"),
    (r"钕磁铁", "NdFeB"),
    (r"稀土永磁", "NdFeB"),
    (r"SmCo[5]?", "SmCo"),
    (r"Sm2?Co17?", "SmCo"),
    (r"钐钴", "SmCo"),
    (r"钐钴磁铁", "SmCo"),
    (r"Alnico[0-9]*", "Alnico"),
    (r"AlNiCo[0-9]*", "Alnico"),
    (r"LNG[0-9]+", "Alnico"),
    (r"铝镍钴", "Alnico"),
    (r"永磁体", "NdFeB"),

    # 电阻合金 (Resistance Alloy)
    (r"Cr20Ni80", "Cr20Ni80"),
    (r"Ni80Cr20", "Cr20Ni80"),
    (r"Nichrome", "Cr20Ni80"),
    (r"OCr20Ni80", "Cr20Ni80"),
    (r"电炉丝", "Cr20Ni80"),
    (r"镍铬丝", "Cr20Ni80"),
    (r"Manganin", "Manganin"),
    (r"6J13", "Manganin"),
    (r"CuMn12Ni", "Manganin"),
    (r"锰铜", "Manganin"),
    (r"Karma", "Karma"),
    (r"卡玛合金", "Karma"),
    (r"电阻应变合金", "Karma"),
    (r"精密电阻", "Manganin"),

    # 低膨胀合金 (Low Expansion Alloy)
    (r"Invar", "Invar"),
    (r"4J36", "Invar"),
    (r"Fe[-\s]?Ni36", "Invar"),
    (r"殷钢", "Invar"),
    (r"因瓦合金", "Invar"),
    (r"低膨胀合金", "Invar"),
    (r"Kovar", "Kovar"),
    (r"4J29", "Kovar"),
    (r"Fe[-\s]?Ni[-\s]?Co(?!32)", "Kovar"),
    (r"可伐合金", "Kovar"),
    (r"玻封合金", "Kovar"),
    (r"4J32", "4J32"),
    (r"Super[-\s]?Invar", "4J32"),
    (r"超因瓦", "4J32"),

    # 超导材料 (Superconductor)
    (r"NbTi", "NbTi"),
    (r"Nb[-\s]?Ti", "NbTi"),
    (r"NbTi47", "NbTi"),
    (r"铌钛合金?", "NbTi"),
    (r"低温超导", "NbTi"),
    (r"Nb3Sn", "Nb3Sn"),
    (r"Nb[-\s]?3[-\s]?Sn", "Nb3Sn"),
    (r"铌三锡", "Nb3Sn"),
    (r"YBCO", "YBCO"),
    (r"YBa2Cu3O7", "YBCO"),
    (r"钇钡铜氧", "YBCO"),
    (r"高温超导", "YBCO"),

    # 核工业材料 (Nuclear Material)
    (r"Zircaloy[-\s]?4", "Zircaloy-4"),
    (r"Zr[-\s]?4", "Zircaloy-4"),
    (r"锆合金[-\s]?4", "Zircaloy-4"),
    (r"核级锆", "Zircaloy-4"),
    (r"燃料包壳", "Zircaloy-4"),
    (r"Hafnium", "Hafnium"),
    (r"Hf(?![-a-zA-Z])", "Hafnium"),
    (r"铪[金属]?", "Hafnium"),
    (r"控制棒材料", "Hafnium"),
    (r"B4C", "B4C"),
    (r"碳化硼", "B4C"),
    (r"中子吸收[材料剂]?", "B4C"),
    (r"屏蔽材料", "B4C"),

    # 医用合金 (Medical Alloy)
    (r"CoCrMo", "CoCrMo"),
    (r"Co[-\s]?Cr[-\s]?Mo", "CoCrMo"),
    (r"钴铬钼合金?", "CoCrMo"),
    (r"医用钴基", "CoCrMo"),
    (r"骨科植入", "CoCrMo"),
    (r"Ti6Al4V[-\s]?ELI", "Ti6Al4V-ELI"),
    (r"Ti[-\s]?6[-\s]?4[-\s]?ELI", "Ti6Al4V-ELI"),
    (r"TC4[-\s]?ELI", "Ti6Al4V-ELI"),
    (r"医用钛合金", "Ti6Al4V-ELI"),
    (r"骨科钛", "Ti6Al4V-ELI"),
    (r"316L[-\s]?Medical", "316L-Medical"),
    (r"医用316L?", "316L-Medical"),
    (r"手术器械钢", "316L-Medical"),
    (r"植入级不锈钢", "316L-Medical"),

    # 光学材料 (Optical Material)
    (r"Fused[-\s]?Silica", "Fused-Silica"),
    (r"熔融石英", "Fused-Silica"),
    (r"石英玻璃", "Fused-Silica"),
    (r"Quartz[-\s]?Glass", "Fused-Silica"),
    (r"光学石英", "Fused-Silica"),
    (r"JGS[12]", "Fused-Silica"),
    (r"Sapphire", "Sapphire"),
    (r"蓝宝石", "Sapphire"),
    (r"Al2O3单晶", "Sapphire"),
    (r"人造蓝宝石", "Sapphire"),
    (r"刚玉单晶", "Sapphire"),
    (r"Germanium", "Germanium"),
    (r"Ge(?!rmanium)(?![a-zA-Z])", "Germanium"),
    (r"锗[单晶]?", "Germanium"),
    (r"红外锗", "Germanium"),
    (r"光学锗", "Germanium"),

    # 电池材料 (Battery Material)
    (r"LiFePO4", "LiFePO4"),
    (r"LFP(?![a-zA-Z])", "LiFePO4"),
    (r"磷酸铁锂", "LiFePO4"),
    (r"铁锂[正极]?", "LiFePO4"),
    (r"NMC(?![a-zA-Z])", "NMC"),
    (r"NCM(?![a-zA-Z])", "NMC"),
    (r"三元[正极锂]?", "NMC"),
    (r"镍钴锰", "NMC"),
    (r"Graphite[-\s]?Battery", "Graphite-Battery"),
    (r"负极石墨", "Graphite-Battery"),
    (r"人造石墨负极?", "Graphite-Battery"),
    (r"天然石墨负极", "Graphite-Battery"),
    (r"锂电负极", "Graphite-Battery"),

    # 半导体材料 (Semiconductor Material)
    (r"Silicon[-\s]?Wafer", "Silicon-Wafer"),
    (r"Si[-\s]?Wafer", "Silicon-Wafer"),
    (r"单晶硅", "Silicon-Wafer"),
    (r"硅[晶]?片", "Silicon-Wafer"),
    (r"硅晶圆", "Silicon-Wafer"),
    (r"GaAs", "GaAs"),
    (r"砷化镓", "GaAs"),
    (r"Gallium[-\s]?Arsenide", "GaAs"),
    (r"化合物半导体", "GaAs"),
    (r"III[-\s]?V族", "GaAs"),
    (r"SiC[-\s]?Semiconductor", "SiC-Semiconductor"),
    (r"碳化硅半导体", "SiC-Semiconductor"),
    (r"宽禁带半导体", "SiC-Semiconductor"),
    (r"第三代半导体", "SiC-Semiconductor"),
    (r"4H[-\s]?SiC", "SiC-Semiconductor"),
    (r"6H[-\s]?SiC", "SiC-Semiconductor"),

    # 热界面材料 (Thermal Interface Material)
    (r"Thermal[-\s]?Paste", "Thermal-Paste"),
    (r"导热硅脂", "Thermal-Paste"),
    (r"硅脂", "Thermal-Paste"),
    (r"导热膏", "Thermal-Paste"),
    (r"Thermal[-\s]?Grease", "Thermal-Paste"),
    (r"Thermal[-\s]?Pad", "Thermal-Pad"),
    (r"导热垫片", "Thermal-Pad"),
    (r"导热硅胶垫", "Thermal-Pad"),
    (r"Gap[-\s]?Filler", "Thermal-Pad"),
    (r"导热片", "Thermal-Pad"),
    (r"Graphene[-\s]?TIM", "Graphene-TIM"),
    (r"石墨烯散热膜?", "Graphene-TIM"),
    (r"石墨烯导热", "Graphene-TIM"),
    (r"Graphene[-\s]?Film", "Graphene-TIM"),

    # 增材制造材料 (Additive Manufacturing Material)
    (r"AlSi10Mg[-\s]?AM", "AlSi10Mg-AM"),
    (r"AlSi10Mg(?![-\s]?AM)", "AlSi10Mg-AM"),
    (r"SLM铝合金", "AlSi10Mg-AM"),
    (r"DMLS铝合金", "AlSi10Mg-AM"),
    (r"增材铝合金", "AlSi10Mg-AM"),
    (r"3D打印铝", "AlSi10Mg-AM"),
    (r"IN718[-\s]?AM", "IN718-AM"),
    (r"Inconel[-\s]?718[-\s]?AM", "IN718-AM"),
    (r"SLM镍合金", "IN718-AM"),
    (r"增材IN718", "IN718-AM"),
    (r"3D打印镍基", "IN718-AM"),
    (r"Ti64[-\s]?AM", "Ti64-AM"),
    (r"Ti[-\s]?6Al[-\s]?4V[-\s]?AM", "Ti64-AM"),
    (r"SLM钛合金", "Ti64-AM"),
    (r"EBM钛合金", "Ti64-AM"),
    (r"增材TC4", "Ti64-AM"),
    (r"3D打印钛", "Ti64-AM"),

    # 硬质合金 (Hard Alloy)
    (r"WC[-\s]?Co", "WC-Co"),
    (r"碳化钨", "WC-Co"),
    (r"Tungsten[-\s]?Carbide", "WC-Co"),
    (r"YG[0-9]+", "WC-Co"),
    (r"硬质合金", "WC-Co"),
    (r"Stellite", "Stellite"),
    (r"Stellite[-\s]?6", "Stellite"),
    (r"司太立", "Stellite"),
    (r"钴基耐磨", "Stellite"),
    (r"堆焊合金", "Stellite"),
    (r"CBN(?![a-zA-Z])", "CBN"),
    (r"PCBN", "CBN"),
    (r"立方氮化硼", "CBN"),
    (r"氮化硼刀具", "CBN"),
    (r"Cubic[-\s]?Boron[-\s]?Nitride", "CBN"),

    # 热障涂层材料 (Thermal Barrier Coating)
    (r"YSZ(?![a-zA-Z])", "YSZ"),
    (r"8YSZ", "YSZ"),
    (r"氧化钇稳定氧化锆", "YSZ"),
    (r"Yttria[-\s]?Stabilized[-\s]?Zirconia", "YSZ"),
    (r"热障涂层", "YSZ"),
    (r"TBC(?![a-zA-Z])", "YSZ"),
    (r"Al2O3[-\s]?TBC", "Al2O3-TBC"),
    (r"氧化铝涂层", "Al2O3-TBC"),
    (r"Alumina[-\s]?Coating", "Al2O3-TBC"),
    (r"TGO(?![a-zA-Z])", "Al2O3-TBC"),
    (r"MCrAlY", "MCrAlY"),
    (r"NiCrAlY", "MCrAlY"),
    (r"CoCrAlY", "MCrAlY"),
    (r"Bond[-\s]?Coat", "MCrAlY"),
    (r"粘结涂层", "MCrAlY"),

    # 电磁屏蔽材料 (EM Shielding Material)
    (r"Mu[-\s]?Metal", "Mu-Metal"),
    (r"μ[-\s]?metal", "Mu-Metal"),
    (r"坡莫合金", "Mu-Metal"),
    (r"高导磁合金", "Mu-Metal"),
    (r"1J79", "Mu-Metal"),
    (r"磁屏蔽合金", "Mu-Metal"),
    (r"Permalloy", "Permalloy"),
    (r"1J50", "Permalloy"),
    (r"软磁合金", "Permalloy"),
    (r"45Permalloy", "Permalloy"),
    (r"Copper[-\s]?Mesh", "Copper-Mesh"),
    (r"铜丝网", "Copper-Mesh"),
    (r"EMI屏蔽网", "Copper-Mesh"),
    (r"Copper[-\s]?Shield", "Copper-Mesh"),
    (r"RF屏蔽", "Copper-Mesh"),

    # 弹簧钢
    (r"65Mn", "65Mn"),
    (r"60Si2Mn[A]?", "60Si2Mn"),
    (r"SUP[-\s]?7", "60Si2Mn"),
    (r"50CrV[A]?", "50CrVA"),
    (r"SUP[-\s]?10", "50CrVA"),
    (r"弹簧钢", "65Mn"),

    # 工具钢
    (r"Cr12MoV[1]?", "Cr12MoV"),
    (r"D[-\s]?2", "Cr12MoV"),
    (r"SKD[-\s]?11", "Cr12MoV"),
    (r"1\.2379", "Cr12MoV"),
    (r"H[-\s]?13", "H13"),
    (r"SKD[-\s]?61", "H13"),
    (r"4Cr5MoSiV1", "H13"),
    (r"1\.2344", "H13"),
    (r"W18Cr4V", "W18Cr4V"),
    (r"T[-\s]?1高速钢?", "W18Cr4V"),
    (r"SKH[-\s]?2", "W18Cr4V"),
    (r"W6Mo5Cr4V2", "W6Mo5Cr4V2"),
    (r"M[-\s]?2", "W6Mo5Cr4V2"),
    (r"SKH[-\s]?51", "W6Mo5Cr4V2"),
    (r"高速钢", "W6Mo5Cr4V2"),
    (r"模具钢", "H13"),

    # 特殊钢材
    (r"9Cr18Mo?V?", "9Cr18"),
    (r"440[BC]", "9Cr18"),
    (r"SUS440[BC]?", "9Cr18"),
    (r"(?:1\.)?4125", "9Cr18"),
    (r"刀具钢", "9Cr18"),
    (r"12Cr1MoV[G]?", "12Cr1MoV"),
    (r"15CrMo[G]?(?!\d)", "12Cr1MoV"),
    (r"13CrMo44", "12Cr1MoV"),
    (r"P22", "12Cr1MoV"),
    (r"耐热钢", "12Cr1MoV"),
    (r"(?:ZG)?Mn13(?:Cr2)?", "Mn13"),
    (r"X120Mn12", "Mn13"),
    (r"Hadfield", "Mn13"),
    (r"高锰钢", "Mn13"),
    (r"耐磨钢(?!板)", "Mn13"),

    # 精密合金
    (r"4J36", "4J36"),
    (r"Invar[-\s]?36?", "4J36"),
    (r"FeNi36", "4J36"),
    (r"(?:1\.)?3912", "4J36"),
    (r"因瓦合金?", "4J36"),
    (r"低膨胀合金", "4J36"),
    (r"4J29", "4J29"),
    (r"Kovar", "4J29"),
    (r"FeNiCo29", "4J29"),
    (r"可伐合金?", "4J29"),
    (r"封接合金", "4J29"),
    (r"4J42", "4J42"),
    (r"Elinvar", "4J42"),
    (r"FeNi42", "4J42"),
    (r"恒弹性合金", "4J42"),
    (r"1J79", "1J79"),
    (r"Permalloy", "1J79"),
    (r"Supermalloy", "1J79"),
    (r"坡莫合金", "1J79"),
    (r"高磁导率合金", "1J79"),

    # 电工钢
    (r"50W\d{2,3}", "50W470"),
    (r"M\d{3}-50A?", "50W470"),
    (r"无取向硅钢", "50W470"),
    (r"硅钢片", "50W470"),
    (r"30Q\d{2,3}", "30Q130"),
    (r"M\d{3}-30S?", "30Q130"),
    (r"取向硅钢", "30Q130"),
    (r"变压器硅钢", "30Q130"),
    (r"电工钢", "50W470"),

    # 焊接材料
    (r"ER308L?", "ER308L"),
    (r"308L?焊丝", "ER308L"),
    (r"Y308L?", "ER308L"),
    (r"ER316L?", "ER316L"),
    (r"316L?焊丝", "ER316L"),
    (r"Y316L?", "ER316L"),
    (r"ER70S[-\s]?6", "ER70S-6"),
    (r"70S[-\s]?6", "ER70S-6"),
    (r"H08Mn2SiA?", "ER70S-6"),
    (r"碳钢焊丝", "ER70S-6"),
    (r"E7018", "E7018"),
    (r"J507", "E7018"),
    (r"低氢焊条", "E7018"),
    (r"碱性焊条", "E7018"),
    (r"不锈钢焊丝", "ER308L"),
    (r"焊丝", "ER70S-6"),
    (r"焊条", "E7018"),

    # 复合材料
    (r"CFRP", "CFRP"),
    (r"碳纤维复合", "CFRP"),
    (r"碳纤维增强", "CFRP"),
    (r"CF[-/]?EP", "CFRP"),
    (r"T300", "CFRP"),
    (r"T700", "CFRP"),
    (r"碳纤维", "CFRP"),
    (r"GFRP", "GFRP"),
    (r"玻璃钢", "GFRP"),
    (r"玻纤复合", "GFRP"),
    (r"玻纤增强", "GFRP"),
    (r"GF[-/]?EP", "GFRP"),
    (r"E[-\s]?glass", "GFRP"),

    # 粉末冶金
    (r"FC[-\s]?0208", "FC-0208"),
    (r"烧结铁(?!镍)", "FC-0208"),
    (r"PM铁基", "FC-0208"),
    (r"FN[-\s]?0205", "FN-0205"),
    (r"烧结铁镍", "FN-0205"),
    (r"PM铁镍", "FN-0205"),
    (r"粉末冶金(?!材)", "FC-0208"),

    # 弹簧钢补充
    (r"55CrSi[A]?", "55CrSi"),
    (r"55SiCr", "55CrSi"),
    (r"SUP[-\s]?12", "55CrSi"),

    # 高温合金补充 (A-286 patterns, GH2132/SUH660 handled above)
    (r"A[-\s]?286", "A-286"),
    (r"S66286", "A-286"),
    # Note: GH2132/SUH660 now map to GH2132, removed from here
    (r"Waspaloy", "Waspaloy"),
    (r"N07001", "Waspaloy"),
    (r"2\.4654", "Waspaloy"),
    (r"Rene[-\s]?41", "Rene41"),
    (r"René[-\s]?41", "Rene41"),
    (r"N07041", "Rene41"),
    (r"R[-\s]?41", "Rene41"),

    # 铜合金补充 - 铝青铜国际牌号映射到国标
    (r"C63000", "QAl10-3-1.5"),
    (r"QAl10[-\s]?3[-\s]?1\.?5", "QAl10-3-1.5"),
    (r"CuAl10Ni5Fe4", "QAl10-3-1.5"),
    (r"Ampco[-\s]?21", "QAl10-3-1.5"),
    (r"C95400", "QAl9-4"),
    (r"NAB", "QAl9-4"),
    (r"AMPCO[-\s]?18", "QAl9-4"),
    (r"QAl9[-\s]?4", "QAl9-4"),
    (r"CuAl10Fe5Ni5", "QAl9-4"),
    (r"镍铝青铜", "QAl9-4"),
    (r"铝青铜", "QAl9-4"),

    # 高强度结构钢
    (r"Q460[C-E]?", "Q460"),
    (r"S460", "Q460"),
    (r"SM570", "Q460"),
    (r"Q550[D-E]?", "Q550"),
    (r"S550", "Q550"),
    (r"HY[-\s]?80", "Q550"),
    (r"Q690[D-E]?", "Q690"),
    (r"S690", "Q690"),
    (r"HY[-\s]?100", "Q690"),

    # 锅炉/压力容器钢
    (r"20[Gg]", "20G"),
    (r"A106[-\s]?B", "20G"),
    (r"STB410", "20G"),
    (r"锅炉管", "20G"),
    (r"15CrMo[Gg]?", "15CrMoG"),
    (r"A335[-\s]?P12", "15CrMoG"),
    (r"STBA22", "15CrMoG"),
    (r"12Cr2Mo1[Rr]?", "12Cr2Mo1R"),
    (r"SA387\s*(?:Gr\.?\s*)?22", "12Cr2Mo1R"),
    (r"2\.?25Cr[-\s]?1Mo", "12Cr2Mo1R"),
    (r"10CrMo9[-\s]?10", "12Cr2Mo1R"),

    # 管线钢
    (r"X[-\s]?52[Mm]?", "X52"),
    (r"L360", "X52"),
    (r"API\s*5L\s*X52", "X52"),
    (r"X[-\s]?65[Mm]?", "X65"),
    (r"L450", "X65"),
    (r"API\s*5L\s*X65", "X65"),
    (r"X[-\s]?80[Mm]?", "X80"),
    (r"L555", "X80"),
    (r"API\s*5L\s*X80", "X80"),
    (r"管线钢", "X65"),
    (r"西气东输", "X80"),

    # 模具钢补充
    (r"DC[-\s]?53", "DC53"),
    (r"SLD[-\s]?MAGIC", "DC53"),
    (r"8%?Cr钢", "DC53"),
    (r"S[-\s]?136", "S136"),
    (r"1\.2083", "S136"),
    (r"420[-\s]?MOD", "S136"),
    (r"SUS420J2", "S136"),
    (r"NAK[-\s]?80", "NAK80"),
    (r"P[-\s]?21(?![0-9])", "NAK80"),
    (r"STAVAX", "NAK80"),
    (r"塑料模具钢", "S136"),
    (r"预硬钢", "NAK80"),

    # 易切削钢
    (r"12L14", "12L14"),
    (r"Y12Pb", "12L14"),
    (r"SUM24L", "12L14"),
    (r"Y[-\s]?15(?![0-9])", "Y15"),
    (r"1215(?!L)", "Y15"),
    (r"SUM22", "Y15"),
    (r"A1215", "Y15"),
    (r"Y40Mn", "Y40Mn"),
    (r"1140", "Y40Mn"),
    (r"SUM43", "Y40Mn"),
    (r"40MnS", "Y40Mn"),
    (r"易切削钢", "Y15"),
    (r"快削钢", "Y15"),

    # 耐磨钢板
    (r"NM[-\s]?400", "NM400"),
    (r"Hardox[-\s]?400", "NM400"),
    (r"XAR[-\s]?400", "NM400"),
    (r"NM[-\s]?500", "NM500"),
    (r"Hardox[-\s]?500", "NM500"),
    (r"XAR[-\s]?500", "NM500"),
    (r"Hardox[-\s]?450", "Hardox450"),
    (r"HX[-\s]?450", "Hardox450"),
    (r"耐磨钢(?!板)", "NM400"),
    (r"耐磨板", "NM400"),
    (r"悍达", "Hardox450"),

    # 低温钢
    (r"09MnNiD", "09MnNiD"),
    (r"3\.?5[-\s]?Ni", "09MnNiD"),
    (r"A203\s*(?:Gr\.?\s*)?D", "09MnNiD"),
    (r"16MnD[R]?", "16MnDR"),
    (r"SA516\s*(?:Gr\.?\s*)?70", "16MnDR"),
    (r"9[-\s]?Ni钢?", "9Ni钢"),
    (r"06Ni9", "9Ni钢"),
    (r"X8Ni9", "9Ni钢"),
    (r"A553", "9Ni钢"),
    (r"LNG钢", "9Ni钢"),
    (r"低温钢", "16MnDR"),

    # 电接触材料
    (r"AgCdO[-\s]?(?:10|12)?", "AgCdO"),
    (r"银氧化镉", "AgCdO"),
    (r"银镉合金", "AgCdO"),
    (r"触点银", "AgCdO"),
    (r"AgSnO2(?:In2O3)?", "AgSnO2"),
    (r"银氧化锡", "AgSnO2"),
    (r"银锡合金", "AgSnO2"),
    (r"环保触点", "AgSnO2"),
    (r"SnO2触点", "AgSnO2"),
    (r"Cu[-\s]?W[-\s]?(?:70)?", "CuW70"),
    (r"W[-\s]?(?:70)?Cu(?:30)?", "CuW70"),
    (r"钨铜合金", "CuW70"),
    (r"铜钨合金", "CuW70"),
    (r"电极铜钨", "CuW70"),
    (r"RWMA\s*Class\s*1[0-4]", "CuW70"),
    (r"电接触材料", "AgSnO2"),

    # 轴承钢
    (r"GCr[-\s]?15(?!Si)", "GCr15"),
    (r"SUJ[-\s]?2", "GCr15"),
    (r"52100", "GCr15"),
    (r"100Cr6", "GCr15"),
    (r"1\.3505", "GCr15"),
    (r"轴承钢", "GCr15"),
    (r"GCr15SiMn", "GCr15SiMn"),
    (r"SUJ[-\s]?4", "GCr15SiMn"),
    (r"52100改", "GCr15SiMn"),
    (r"大截面轴承钢", "GCr15SiMn"),
    (r"GCr[-\s]?4(?![0-9])", "GCr4"),
    (r"SAE\s*4320", "GCr4"),
    (r"渗碳轴承", "GCr4"),

    # 弹簧钢
    (r"60Si2Mn", "60Si2Mn"),
    (r"SUP[-\s]?6", "60Si2Mn"),
    (r"9260", "60Si2Mn"),
    (r"1\.7108", "60Si2Mn"),
    (r"硅锰钢", "60Si2Mn"),
    (r"60Si2CrA?", "60Si2CrA"),
    (r"SUP[-\s]?12", "60Si2CrA"),
    (r"60SC7", "60Si2CrA"),
    (r"铬硅弹簧钢", "60Si2CrA"),
    (r"50CrV[-\s]?A?(?:4)?", "50CrVA"),
    (r"SUP[-\s]?10", "50CrVA"),
    (r"6150", "50CrVA"),
    (r"1\.8159", "50CrVA"),
    (r"铬钒弹簧钢", "50CrVA"),
    (r"弹簧钢", "60Si2Mn"),

    # 耐热不锈钢补充
    (r"2Cr13", "2Cr13"),
    (r"420(?!J2|MOD)", "2Cr13"),
    (r"SUS420J1", "2Cr13"),
    (r"1\.4021", "2Cr13"),
    (r"X20Cr13", "2Cr13"),
    (r"1Cr17(?![Ni])", "1Cr17"),
    (r"430(?![1-9])", "1Cr17"),
    (r"SUS430(?![1-9])", "1Cr17"),
    (r"1\.4016", "1Cr17"),
    (r"X6Cr17", "1Cr17"),
    (r"铁素体不锈钢", "1Cr17"),
    (r"0Cr25Ni20", "0Cr25Ni20"),
    (r"310S", "0Cr25Ni20"),
    (r"SUS310S", "0Cr25Ni20"),
    (r"1\.4845", "0Cr25Ni20"),
    (r"X8CrNi25[-\s]?21", "0Cr25Ni20"),
    (r"耐热不锈钢", "0Cr25Ni20"),

    # 齿轮钢
    (r"20CrMnTi", "20CrMnTi"),
    (r"SCM420H", "20CrMnTi"),
    (r"齿轮钢", "20CrMnTi"),
    (r"渗碳钢", "20CrMnTi"),
    (r"20CrMo(?![1-9])", "20CrMo"),
    (r"SCM420(?!H)", "20CrMo"),
    (r"4118", "20CrMo"),
    (r"铬钼钢", "20CrMo"),
    (r"20CrNiMo", "20CrNiMo"),
    (r"SNCM220", "20CrNiMo"),
    (r"8620", "20CrNiMo"),
    (r"4320", "20CrNiMo"),
    (r"高级齿轮钢", "20CrNiMo"),

    # 航空铝合金补充
    (r"5A06[-H\d]*", "5A06"),
    (r"5456[-H\d]*", "5A06"),
    (r"AlMg5(?!Si)", "5A06"),
    (r"A5456", "5A06"),
    (r"LF6", "5A06"),
    (r"防锈铝", "5A06"),
    (r"2A14[-T\d]*", "2A14"),
    (r"2014[-T\d]*", "2A14"),
    (r"LD10", "2A14"),
    (r"AlCu4SiMg", "2A14"),
    (r"A2014", "2A14"),
    (r"7A04[-T\d]*", "7A04"),
    (r"7050[-T\d]*", "7A04"),
    (r"LC4", "7A04"),
    (r"AlZn6MgCu", "7A04"),
    (r"超硬铝", "7A04"),

    # 气门钢
    (r"4Cr10Si2Mo", "4Cr10Si2Mo"),
    (r"SUH[-\s]?3(?![0-9])", "4Cr10Si2Mo"),
    (r"X45CrSi9[-\s]?3", "4Cr10Si2Mo"),
    (r"气门钢", "4Cr10Si2Mo"),
    (r"排气门钢", "4Cr10Si2Mo"),
    (r"5Cr21Mn9Ni4N", "5Cr21Mn9Ni4N"),
    (r"SUH[-\s]?35", "5Cr21Mn9Ni4N"),
    (r"21[-\s]?4[-\s]?N", "5Cr21Mn9Ni4N"),
    (r"进气门钢", "5Cr21Mn9Ni4N"),
    (r"4Cr14Ni14W2Mo", "4Cr14Ni14W2Mo"),
    (r"SUH[-\s]?38", "4Cr14Ni14W2Mo"),
    (r"X40CrNiW14[-\s]?14", "4Cr14Ni14W2Mo"),
    (r"钨钼气门钢", "4Cr14Ni14W2Mo"),

    # 链条钢/渗碳硼钢
    (r"20MnVB", "20MnVB"),
    (r"链条钢", "20MnVB"),
    (r"销轴钢", "20MnVB"),
    (r"硼钢(?!板)", "20MnVB"),
    (r"15MnVB", "15MnVB"),
    (r"冷镦钢", "15MnVB"),
    (r"螺栓钢", "15MnVB"),
    (r"22MnB5", "22MnB5"),
    (r"热成形钢", "22MnB5"),
    (r"热冲压钢", "22MnB5"),
    (r"USIBOR", "22MnB5"),
    (r"PHS钢", "22MnB5"),
    (r"硼钢板", "22MnB5"),

    # 电工硅钢补充
    (r"B50A600", "B50A600"),
    (r"50W600", "B50A600"),
    (r"M600[-\s]?50A", "B50A600"),
    (r"电机硅钢", "B50A600"),
    (r"B35A230", "B35A230"),
    (r"35W230", "B35A230"),
    (r"M230[-\s]?35A", "B35A230"),
    (r"高效电机硅钢", "B35A230"),
    (r"B27R090", "B27R090"),
    (r"27RK090", "B27R090"),
    (r"M090[-\s]?27P", "B27R090"),
    (r"变压器硅钢", "B27R090"),
    (r"取向硅钢", "B27R090"),

    # 铝合金 - 变形铝合金
    (r"6061[-T\d]*", "6061"),
    (r"AA[-\s]?6061", "6061"),
    (r"Al[-\s]?6061", "6061"),
    (r"7075[-T\d]*", "7075"),
    (r"AA[-\s]?7075", "7075"),
    (r"Al[-\s]?7075", "7075"),
    (r"2024[-T\d]*", "2024"),
    (r"AA[-\s]?2024", "2024"),
    (r"Al[-\s]?2024", "2024"),
    (r"5052[-H\d]*", "5052"),
    (r"AA[-\s]?5052", "5052"),
    (r"Al[-\s]?5052", "5052"),
    (r"5083[-H\d]*", "5083"),
    (r"AA[-\s]?5083", "5083"),
    (r"Al[-\s]?5083", "5083"),
    (r"2A12[-T\d]*", "2A12"),
    (r"LY12", "2A12"),
    (r"6063[-T\d]*", "6063"),
    (r"AA[-\s]?6063", "6063"),
    (r"Al[-\s]?6063", "6063"),
    # 铝合金 - 铸造铝合金
    (r"A356[-T\d]*", "A356"),
    (r"ZL101[-T\d]*", "A356"),
    (r"AlSi7Mg", "A356"),
    (r"AC4C", "A356"),
    (r"ZL102", "ZL102"),
    (r"A413", "ZL102"),
    (r"AlSi12(?!Cu)", "ZL102"),
    (r"ADC1(?![0-9])", "ZL102"),
    (r"ADC12", "ADC12"),
    (r"A383", "ADC12"),
    (r"AlSi11Cu3", "ADC12"),
    (r"YL113", "ADC12"),
    (r"压铸铝", "ADC12"),
    # 铸造铝合金补充
    (r"ZL101[-T\d]*", "ZL101"),
    (r"A356\.2", "ZL101"),
    (r"AC4CH", "ZL101"),
    (r"高纯铸铝", "ZL101"),
    (r"ZL104[-T\d]*", "ZL104"),
    (r"A319", "ZL104"),
    (r"AC2B", "ZL104"),
    (r"发动机铸铝", "ZL104"),
    (r"ZL201[-T\d]*", "ZL201"),
    (r"A201", "ZL201"),
    (r"高强铸铝", "ZL201"),
    (r"航空铸铝", "ZL201"),
    (r"铸造铝", "A356"),
    (r"铝合金", "6061"),

    # 锌合金
    (r"Zamak[-\s]?3", "Zamak3"),
    (r"ZAMAK[-\s]?3", "Zamak3"),
    (r"ZA[-\s]?3", "Zamak3"),
    (r"ZnAl4(?![0-9Cu])", "Zamak3"),
    (r"锌合金3号?", "Zamak3"),
    (r"Zamak[-\s]?5", "Zamak5"),
    (r"ZAMAK[-\s]?5", "Zamak5"),
    (r"ZA[-\s]?5", "Zamak5"),
    (r"ZnAl4Cu1", "Zamak5"),
    (r"锌合金5号", "Zamak5"),
    (r"ZA[-\s]?8", "ZA-8"),
    (r"ZnAl8Cu", "ZA-8"),
    (r"超强锌合金", "ZA-8"),
    (r"锌合金", "Zamak3"),  # 通用匹配

    # 铜合金 - 黄铜
    (r"H62", "H62"),
    (r"H68", "H68"),
    (r"CuZn33", "H68"),
    (r"C26800", "H68"),
    (r"HPb59-1", "HPb59-1"),
    (r"C38500", "HPb59-1"),
    (r"易切削黄铜", "HPb59-1"),
    (r"快削黄铜", "HPb59-1"),
    # 硅黄铜/铝黄铜 (must be before generic 黄铜)
    (r"HSi80-3", "HSi80-3"),
    (r"C87500", "HSi80-3"),
    (r"CuZn16Si3", "HSi80-3"),
    (r"耐蚀黄铜", "HSi80-3"),
    (r"硅黄铜", "HSi80-3"),
    (r"HAl77-2", "HAl77-2"),
    (r"C68700", "HAl77-2"),
    (r"CuZn22Al2", "HAl77-2"),
    (r"海军黄铜", "HAl77-2"),
    (r"铝黄铜", "HAl77-2"),
    (r"黄铜", "H62"),  # 通用匹配放最后

    # 铜合金 - 青铜
    (r"QBe[-\s]?2", "QBe2"),
    (r"C17200", "QBe2"),
    (r"BeCu", "QBe2"),
    (r"CuBe[-\s]?2", "QBe2"),
    (r"铍青?铜", "QBe2"),
    (r"QAl10-3-1\.?5", "QAl10-3-1.5"),
    (r"C63000", "QAl10-3-1.5"),
    (r"铝铁镍青铜", "QAl10-3-1.5"),
    (r"QSn6\.?5-0\.?1", "QSn6.5-0.1"),
    (r"C51900", "QSn6.5-0.1"),
    (r"磷青?铜", "QSn6.5-0.1"),
    (r"弹簧磷青铜", "QSn6.5-0.1"),
    (r"QSn\d+-\d+", "QSn4-3"),  # 其他锡青铜
    (r"QA[l]?\d+-\d+", "QAl9-4"),  # 其他铝青铜
    (r"青铜", "QSn4-3"),  # 通用匹配放最后

    # 铜合金 - 白铜
    (r"CuNi10Fe1Mn", "CuNi10Fe1Mn"),
    (r"B10", "CuNi10Fe1Mn"),
    (r"CN102", "CuNi10Fe1Mn"),
    (r"C70600", "CuNi10Fe1Mn"),
    (r"90/10白铜", "CuNi10Fe1Mn"),
    (r"白铜", "CuNi10Fe1Mn"),

    # 铜合金 - 锡青铜补充
    (r"QSn7-0\.?2", "QSn7-0.2"),
    (r"C52100", "QSn7-0.2"),
    (r"CuSn8", "QSn7-0.2"),
    (r"高弹性磷铜", "QSn7-0.2"),
    (r"QSn4-0\.?3", "QSn4-0.3"),
    (r"C51000", "QSn4-0.3"),
    (r"CuSn4", "QSn4-0.3"),
    (r"导电磷青铜", "QSn4-0.3"),
    (r"ZCuSn10P1", "ZCuSn10P1"),
    (r"C90700", "ZCuSn10P1"),
    (r"PBC2A", "ZCuSn10P1"),
    (r"高力青铜", "ZCuSn10P1"),
    (r"铸造青铜", "ZCuSn10P1"),

    # 铜合金 - 紫铜
    (r"Cu65", "Cu65"),
    (r"[紫纯]铜", "Cu65"),
    (r"T[12]", "Cu65"),

    # 钛合金
    (r"TA1锻?件?", "TA1"),
    (r"Gr?[-\s]?1", "TA1"),
    (r"Grade[-\s]?1[-\s]?Ti", "TA1"),
    (r"Ti[-\s]?Grade[-\s]?1", "TA1"),
    (r"TA2锻?件?", "TA2"),
    (r"Gr?[-\s]?2", "TA2"),
    (r"Grade[-\s]?2[-\s]?Ti", "TA2"),
    (r"Ti[-\s]?Grade[-\s]?2", "TA2"),
    (r"CP[-\s]?Ti", "TA2"),
    (r"TC4", "TC4"),
    (r"Ti[-\s]?6Al[-\s]?4V", "TC4"),
    (r"Gr?[-\s]?5", "TC4"),
    (r"Grade[-\s]?5[-\s]?Ti", "TC4"),
    (r"Ti[-\s]?Grade[-\s]?5", "TC4"),
    (r"Ti64", "TC4"),
    (r"TC11", "TC11"),
    (r"Ti[-\s]?6\.?5Al", "TC11"),
    (r"BT[-\s]?9", "TC11"),
    (r"TB6", "TB6"),
    (r"Ti[-\s]?10V[-\s]?2Fe[-\s]?3Al", "TB6"),
    (r"Ti[-\s]?1023", "TB6"),
    (r"TC21", "TC21"),

    # 镁合金
    (r"AZ31[AB]?", "AZ31B"),
    (r"MB[-\s]?2", "AZ31B"),
    (r"AZ91[AD]?", "AZ91D"),
    (r"ZK60[A]?", "ZK60"),
    (r"镁合金", "AZ31B"),

    # 硬质合金
    (r"YG[-\s]?8", "YG8"),
    (r"K[-\s]?20", "YG8"),
    (r"WC-?8Co", "YG8"),
    (r"YT[-\s]?15", "YT15"),
    (r"P[-\s]?20", "YT15"),
    (r"YG[-\s]?6", "YG6"),
    (r"K[-\s]?10", "YG6"),
    (r"硬质合金", "YG8"),
    (r"钨钢", "YG8"),

    # 耐蚀合金
    (r"C[-\s]?276[ⅡI]*", "C276"),
    (r"Hastelloy\s*C[-\s]?276", "C276"),
    (r"N10276", "C276"),
    (r"C[-\s]?22[ⅡI]*", "C22"),
    (r"Hastelloy\s*C[-\s]?22", "C22"),
    (r"N06022", "C22"),
    # Inconel 625
    (r"Inconel[-\s]?625", "Inconel625"),
    (r"IN[-\s]?625", "Inconel625"),
    (r"N06625", "Inconel625"),
    (r"NCF[-\s]?625", "Inconel625"),
    (r"Alloy[-\s]?625", "Inconel625"),
    # Inconel 718
    (r"Inconel[-\s]?718", "Inconel718"),
    (r"IN[-\s]?718", "Inconel718"),
    (r"N07718", "Inconel718"),
    (r"GH[-\s]?4169", "Inconel718"),
    (r"Alloy[-\s]?718", "Inconel718"),
    # Monel
    (r"Monel[-\s]?400", "Monel400"),
    (r"N04400", "Monel400"),
    (r"NCu[-\s]?30", "Monel400"),
    (r"Monel[-\s]?K[-\s]?500", "MonelK500"),
    (r"N05500", "MonelK500"),
    # Hastelloy B
    (r"Hastelloy[-\s]?B[-\s]?3", "HastelloyB3"),
    (r"N10675", "HastelloyB3"),
    # Stellite
    (r"Stellite[-\s]?6", "Stellite6"),
    (r"钴基[-\s]?6号?", "Stellite6"),
    (r"CoCr[-\s]?A", "Stellite6"),
    # Incoloy
    (r"Incoloy[-\s]?825", "Incoloy825"),
    (r"N08825", "Incoloy825"),
    (r"GH[-\s]?2825", "Incoloy825"),

    # 双相不锈钢
    (r"2205[A-Z]?", "2205"),
    (r"S318[0-9]{2}", "2205"),
    (r"S322[0-9]{2}", "2205"),
    (r"SAF[-\s]?2205", "2205"),
    (r"(?:1\.)?4462", "2205"),
    (r"2507[A-Z]?", "2507"),
    (r"S327[0-9]{2}", "2507"),
    (r"SAF[-\s]?2507", "2507"),
    (r"(?:1\.)?4410", "2507"),

    # 紧固件材料
    (r"A[24]-\d+", "S30408"),  # A2-50, A2-70, A4-70 是不锈钢紧固件
    (r"A1(?:-\d+)?$", "Q235B"),  # A1 是碳钢紧固件
    (r"A4$", "S31603"),  # A4 单独出现时是316L等级不锈钢紧固件

    # 塑料/橡胶
    (r"R?PTFE", "PTFE"),
    (r"[特铁]氟龙", "PTFE"),
    (r"Teflon", "PTFE"),
    # 工程塑料
    (r"PEEK", "PEEK"),
    (r"聚醚醚酮", "PEEK"),
    (r"POM[-CHch]?", "POM"),
    (r"Delrin", "POM"),
    (r"赛钢", "POM"),
    (r"聚甲醛", "POM"),
    (r"聚缩醛", "POM"),
    (r"PA[-]?66?", "PA66"),
    (r"Nylon[-]?66?", "PA66"),
    (r"尼龙[-\d]*", "PA66"),
    (r"聚酰胺", "PA66"),
    (r"PC", "PC"),
    (r"聚碳酸酯", "PC"),
    (r"Polycarbonate", "PC"),
    (r"UHMW[-]?PE", "UHMWPE"),
    (r"UPE", "UHMWPE"),
    (r"超高分子.*聚乙烯", "UHMWPE"),
    # 高性能工程塑料
    (r"PPS", "PPS"),
    (r"Ryton", "PPS"),
    (r"聚苯硫醚", "PPS"),
    (r"PI", "PI"),
    (r"Vespel", "PI"),
    (r"Kapton", "PI"),
    (r"聚酰亚胺", "PI"),
    (r"PSU", "PSU"),
    (r"Polysulfone", "PSU"),
    (r"Udel", "PSU"),
    (r"聚砜", "PSU"),
    (r"PEI", "PEI"),
    (r"Ultem", "PEI"),
    (r"聚醚酰亚胺", "PEI"),
    # 橡胶
    (r"EPDM", "EPDM"),
    (r"VMQ", "VMQ"),
    (r"硅橡?胶", "VMQ"),
    (r"聚氨酯[-\d]*", "聚氨酯"),
    (r"PU", "聚氨酯"),

    # 玻璃
    (r"硼硅玻璃", "硼硅玻璃"),
    (r"高硼硅", "硼硅玻璃"),
    (r"钢化玻璃", "钢化玻璃"),

    # 陶瓷
    (r"硅酸铝", "硅酸铝"),
    (r"陶瓷纤维", "硅酸铝"),

    # 组合件
    (r"组焊件", "组焊件"),
    (r"组合件", "组合件"),
    (r"组件", "组合件"),
]


# ============================================================================
# 分类函数
# ============================================================================

def classify_material_detailed(material: Optional[str]) -> Optional[MaterialInfo]:
    """
    详细材料分类

    Args:
        material: 材料名称字符串

    Returns:
        MaterialInfo 或 None
    """
    if not material:
        return None

    material_clean = material.strip()

    # 1. 精确匹配
    if material_clean in MATERIAL_DATABASE:
        return MATERIAL_DATABASE[material_clean]

    # 2. 大小写不敏感匹配
    material_upper = material_clean.upper()
    for grade, info in MATERIAL_DATABASE.items():
        if grade.upper() == material_upper:
            return info
        if material_upper in [a.upper() for a in info.aliases]:
            return info

    # 3. 正则模式匹配
    for pattern, grade in MATERIAL_MATCH_PATTERNS:
        if re.search(pattern, material_clean, re.IGNORECASE):
            if grade in MATERIAL_DATABASE:
                return MATERIAL_DATABASE[grade]

    # 4. 通用不锈钢识别
    if "不锈钢" in material_clean:
        return MATERIAL_DATABASE.get("S30408")

    logger.debug("Material not found in database: %s", material)
    return None


def get_material_info(material: Optional[str]) -> Dict[str, Any]:
    """
    获取材料信息字典

    Args:
        material: 材料名称

    Returns:
        材料信息字典
    """
    info = classify_material_detailed(material)
    if info:
        return info.to_dict()
    return {
        "grade": material,
        "name": material,
        "category": None,
        "sub_category": None,
        "group": None,
        "found": False,
    }


def get_process_recommendations(material: Optional[str]) -> ProcessRecommendation:
    """
    获取材料工艺推荐

    Args:
        material: 材料名称

    Returns:
        ProcessRecommendation
    """
    info = classify_material_detailed(material)
    if info:
        return info.process
    return ProcessRecommendation()


# 简化分类函数（兼容旧接口）
def classify_material_simple(material: Optional[str]) -> Optional[str]:
    """
    简化材料分类（返回材料组）

    Args:
        material: 材料名称

    Returns:
        材料组名称 或 None
    """
    info = classify_material_detailed(material)
    if info:
        return info.group.value
    return None


# ============================================================================
# 材料等价表（中外标准对照）
# ============================================================================

# 格式: {标准牌号: {标准体系: 等价牌号}}
MATERIAL_EQUIVALENCE: Dict[str, Dict[str, str]] = {
    # 不锈钢
    "S30408": {
        "CN": "S30408",           # 中国 GB
        "US": "304",              # 美国 ASTM/AISI
        "JP": "SUS304",           # 日本 JIS
        "DE": "1.4301",           # 德国 DIN/EN
        "name": "奥氏体不锈钢",
    },
    "S31603": {
        "CN": "S31603",
        "US": "316L",
        "JP": "SUS316L",
        "DE": "1.4404",
        "name": "低碳奥氏体不锈钢",
    },
    "2205": {
        "CN": "S31803",
        "US": "2205",
        "JP": "SUS329J3L",
        "DE": "1.4462",
        "name": "双相不锈钢",
    },
    "2507": {
        "CN": "S32750",
        "US": "2507",
        "JP": "SUS327L1",
        "DE": "1.4410",
        "name": "超级双相不锈钢",
    },
    "321": {
        "CN": "S32100",
        "US": "321",
        "JP": "SUS321",
        "DE": "1.4541",
        "name": "钛稳定奥氏体不锈钢",
    },
    "347": {
        "CN": "S34700",
        "US": "347",
        "JP": "SUS347",
        "DE": "1.4550",
        "name": "铌稳定奥氏体不锈钢",
    },
    "430": {
        "CN": "S43000",
        "US": "430",
        "JP": "SUS430",
        "DE": "1.4016",
        "name": "铁素体不锈钢",
    },
    "410": {
        "CN": "S41000",
        "US": "410",
        "JP": "SUS410",
        "DE": "1.4006",
        "name": "马氏体不锈钢",
    },
    "17-4PH": {
        "CN": "S17400",
        "US": "17-4PH",
        "JP": "SUS630",
        "DE": "1.4542",
        "name": "沉淀硬化不锈钢",
    },
    # 超级奥氏体不锈钢
    "904L": {
        "CN": "00Cr20Ni25Mo4.5Cu",
        "US": "904L",
        "JP": "SUS890L",
        "DE": "1.4539",
        "name": "超级奥氏体不锈钢",
    },
    "254SMO": {
        "CN": "00Cr20Ni18Mo6CuN",
        "US": "S31254",
        "JP": "-",
        "DE": "1.4547",
        "name": "超级奥氏体不锈钢",
    },
    "316Ti": {
        "CN": "0Cr17Ni12Mo2Ti",
        "US": "316Ti",
        "JP": "SUS316Ti",
        "DE": "1.4571",
        "name": "钛稳定奥氏体不锈钢",
    },
    # 碳素钢
    "Q235B": {
        "CN": "Q235B",
        "US": "A36",
        "JP": "SS400",
        "DE": "S235JR",
        "name": "普通碳素结构钢",
    },
    "45": {
        "CN": "45",
        "US": "1045",
        "JP": "S45C",
        "DE": "C45",
        "name": "优质碳素结构钢",
    },
    "20": {
        "CN": "20",
        "US": "1020",
        "JP": "S20C",
        "DE": "C20",
        "name": "优质碳素结构钢",
    },
    "10": {
        "CN": "10",
        "US": "1010",
        "JP": "S10C",
        "DE": "C10",
        "name": "低碳钢",
    },
    "15": {
        "CN": "15",
        "US": "1015",
        "JP": "S15C",
        "DE": "C15",
        "name": "低碳钢",
    },
    "35": {
        "CN": "35",
        "US": "1035",
        "JP": "S35C",
        "DE": "C35",
        "name": "中碳钢",
    },
    "50": {
        "CN": "50",
        "US": "1050",
        "JP": "S50C",
        "DE": "C50",
        "name": "中碳钢",
    },
    # 合金钢
    "40Cr": {
        "CN": "40Cr",
        "US": "5140",
        "JP": "SCr440",
        "DE": "41Cr4",
        "name": "合金结构钢",
    },
    "42CrMo": {
        "CN": "42CrMo",
        "US": "4140",
        "JP": "SCM440",
        "DE": "42CrMo4",
        "name": "合金结构钢",
    },
    "GCr15": {
        "CN": "GCr15",
        "US": "52100",
        "JP": "SUJ2",
        "DE": "100Cr6",
        "name": "高碳铬轴承钢",
    },
    "GCr15SiMn": {
        "CN": "GCr15SiMn",
        "US": "52100 Mod",
        "JP": "SUJ3",
        "DE": "100CrMnSi6-4",
        "name": "硅锰轴承钢",
    },
    "GCr18Mo": {
        "CN": "GCr18Mo",
        "US": "A485 Gr.3",
        "JP": "SUJ5",
        "DE": "100CrMo7-3",
        "name": "铬钼轴承钢",
    },
    "20CrMnTi": {
        "CN": "20CrMnTi",
        "US": "8620",
        "JP": "SCM420H",
        "DE": "20MnCr5",
        "name": "渗碳钢",
    },
    "20Cr": {
        "CN": "20Cr",
        "US": "5120",
        "JP": "SCr420",
        "DE": "20Cr4",
        "name": "渗碳钢",
    },
    "38CrMoAl": {
        "CN": "38CrMoAl",
        "US": "-",
        "JP": "SACM645",
        "DE": "41CrAlMo7",
        "name": "氮化钢",
    },
    "30CrMnSi": {
        "CN": "30CrMnSi",
        "US": "8630",
        "JP": "-",
        "DE": "30CrMnSi",
        "name": "高强度结构钢",
    },
    # 弹簧钢
    "65Mn": {
        "CN": "65Mn",
        "US": "1066",
        "JP": "SUP6",
        "DE": "65Mn4",
        "name": "锰钢弹簧钢",
    },
    "60Si2Mn": {
        "CN": "60Si2Mn",
        "US": "9260",
        "JP": "SUP7",
        "DE": "60SiCr7",
        "name": "硅锰弹簧钢",
    },
    "50CrVA": {
        "CN": "50CrVA",
        "US": "6150",
        "JP": "SUP10",
        "DE": "50CrV4",
        "name": "铬钒弹簧钢",
    },
    # 工具钢
    "Cr12MoV": {
        "CN": "Cr12MoV",
        "US": "D2",
        "JP": "SKD11",
        "DE": "1.2379",
        "name": "冷作模具钢",
    },
    "H13": {
        "CN": "H13",
        "US": "H13",
        "JP": "SKD61",
        "DE": "1.2344",
        "name": "热作模具钢",
    },
    "W18Cr4V": {
        "CN": "W18Cr4V",
        "US": "T1",
        "JP": "SKH2",
        "DE": "1.3355",
        "name": "高速钢",
    },
    "W6Mo5Cr4V2": {
        "CN": "W6Mo5Cr4V2",
        "US": "M2",
        "JP": "SKH51",
        "DE": "1.3343",
        "name": "高性能高速钢",
    },
    # 特殊钢材
    "9Cr18": {
        "CN": "9Cr18",
        "US": "440C",
        "JP": "SUS440C",
        "DE": "1.4125",
        "name": "高碳马氏体不锈钢",
    },
    "12Cr1MoV": {
        "CN": "12Cr1MoV",
        "US": "P22 (近似)",
        "JP": "SCMV4",
        "DE": "13CrMo44",
        "name": "珠光体耐热钢",
    },
    "Mn13": {
        "CN": "ZGMn13",
        "US": "A128 Gr.A",
        "JP": "SCMnH1",
        "DE": "X120Mn12",
        "name": "高锰耐磨钢",
    },
    # 精密合金
    "4J36": {
        "CN": "4J36",
        "US": "Invar 36",
        "JP": "-",
        "DE": "1.3912",
        "name": "因瓦合金",
    },
    "4J29": {
        "CN": "4J29",
        "US": "Kovar",
        "JP": "-",
        "DE": "1.3981",
        "name": "可伐合金",
    },
    "4J42": {
        "CN": "4J42",
        "US": "Ni42",
        "JP": "-",
        "DE": "1.3917",
        "name": "恒弹性合金",
    },
    "1J79": {
        "CN": "1J79",
        "US": "Permalloy 80",
        "JP": "PC",
        "DE": "-",
        "name": "坡莫合金",
    },
    # 电工钢
    "50W470": {
        "CN": "50W470",
        "US": "M-43",
        "JP": "50A470",
        "DE": "M470-50A",
        "name": "无取向硅钢",
    },
    "30Q130": {
        "CN": "30Q130",
        "US": "M-4",
        "JP": "30P130",
        "DE": "M130-30S",
        "name": "取向硅钢",
    },
    # 焊接材料
    "ER308L": {
        "CN": "H0Cr21Ni10",
        "US": "ER308L",
        "JP": "Y308L",
        "DE": "1.4316",
        "name": "不锈钢焊丝",
    },
    "ER316L": {
        "CN": "H0Cr19Ni12Mo2",
        "US": "ER316L",
        "JP": "Y316L",
        "DE": "1.4430",
        "name": "不锈钢焊丝",
    },
    "ER70S-6": {
        "CN": "H08Mn2SiA",
        "US": "ER70S-6",
        "JP": "YGW12",
        "DE": "G3Si1",
        "name": "碳钢焊丝",
    },
    "E7018": {
        "CN": "J507",
        "US": "E7018",
        "JP": "D4316",
        "DE": "E 46 4 B 32 H5",
        "name": "低氢焊条",
    },
    # 复合材料
    "CFRP": {
        "CN": "碳纤维增强塑料",
        "US": "CFRP",
        "JP": "CFRP",
        "DE": "CFK",
        "name": "碳纤维复合材料",
    },
    "GFRP": {
        "CN": "玻璃钢",
        "US": "GFRP",
        "JP": "GFRP",
        "DE": "GFK",
        "name": "玻璃纤维复合材料",
    },
    # 粉末冶金
    "FC-0208": {
        "CN": "Fe-0.8C",
        "US": "FC-0208",
        "JP": "SMF2",
        "DE": "Sint-D10",
        "name": "烧结铁",
    },
    "FN-0205": {
        "CN": "Fe-2Ni-0.5C",
        "US": "FN-0205",
        "JP": "SMF4",
        "DE": "Sint-D30",
        "name": "烧结铁镍",
    },
    # 耐热钢/高温合金
    "310S": {
        "CN": "S31008",
        "US": "310S",
        "JP": "SUS310S",
        "DE": "1.4845",
        "name": "耐热不锈钢",
    },
    "GH3030": {
        "CN": "GH3030",
        "US": "N06075",
        "JP": "-",
        "DE": "Nimonic 75",
        "name": "镍基高温合金",
    },
    "GH4169": {
        "CN": "GH4169",
        "US": "N07718",
        "JP": "NCF718",
        "DE": "Inconel 718",
        "name": "时效强化镍基高温合金",
    },
    "GH4099": {
        "CN": "GH4099",
        "US": "N07001",
        "JP": "-",
        "DE": "Waspaloy",
        "name": "高强度镍基高温合金",
    },
    # 新增高温合金
    "A-286": {
        "CN": "GH2132",
        "US": "S66286",
        "JP": "SUH660",
        "DE": "1.4980",
        "name": "铁镍基高温合金",
    },
    "Waspaloy": {
        "CN": "GH4099",
        "US": "N07001",
        "JP": "-",
        "DE": "2.4654",
        "name": "镍基高温合金",
    },
    "Rene41": {
        "CN": "-",
        "US": "N07041",
        "JP": "-",
        "DE": "2.4973",
        "name": "镍基高温合金",
    },
    # 高强度结构钢
    "Q460": {
        "CN": "Q460",
        "US": "-",
        "JP": "SM570",
        "DE": "S460",
        "name": "高强度结构钢",
    },
    "Q550": {
        "CN": "Q550",
        "US": "HY80",
        "JP": "-",
        "DE": "S550",
        "name": "高强度结构钢",
    },
    "Q690": {
        "CN": "Q690",
        "US": "HY100",
        "JP": "-",
        "DE": "S690",
        "name": "超高强度结构钢",
    },
    # 锅炉/压力容器钢
    "20G": {
        "CN": "20G",
        "US": "A106-B",
        "JP": "STB410",
        "DE": "St35.8",
        "name": "锅炉用碳素钢",
    },
    "15CrMoG": {
        "CN": "15CrMoG",
        "US": "A335-P12",
        "JP": "STBA22",
        "DE": "13CrMo44",
        "name": "锅炉用合金钢",
    },
    "12Cr2Mo1R": {
        "CN": "12Cr2Mo1R",
        "US": "SA387 Gr.22",
        "JP": "SCMV4",
        "DE": "10CrMo9-10",
        "name": "压力容器用钢",
    },
    # 管线钢
    "X52": {
        "CN": "L360",
        "US": "X52",
        "JP": "-",
        "DE": "L360",
        "name": "管线钢",
    },
    "X65": {
        "CN": "L450",
        "US": "X65",
        "JP": "-",
        "DE": "L450",
        "name": "高强度管线钢",
    },
    "X80": {
        "CN": "L555",
        "US": "X80",
        "JP": "-",
        "DE": "L555",
        "name": "超高强度管线钢",
    },
    # 模具钢补充
    "DC53": {
        "CN": "DC53",
        "US": "-",
        "JP": "DC53",
        "DE": "-",
        "name": "冷作模具钢",
    },
    "S136": {
        "CN": "S136",
        "US": "420MOD",
        "JP": "SUS420J2",
        "DE": "1.2083",
        "name": "塑料模具钢",
    },
    "NAK80": {
        "CN": "NAK80",
        "US": "P21",
        "JP": "NAK80",
        "DE": "-",
        "name": "预硬塑料模具钢",
    },
    # 易切削钢
    "12L14": {
        "CN": "Y12Pb",
        "US": "12L14",
        "JP": "SUM24L",
        "DE": "9SMnPb28",
        "name": "含铅易切削钢",
    },
    "Y15": {
        "CN": "Y15",
        "US": "1215",
        "JP": "SUM22",
        "DE": "9SMn28",
        "name": "硫系易切削钢",
    },
    "Y40Mn": {
        "CN": "Y40Mn",
        "US": "1140",
        "JP": "SUM43",
        "DE": "40MnS",
        "name": "易切削调质钢",
    },
    # 耐磨钢板
    "NM400": {
        "CN": "NM400",
        "US": "AR400",
        "JP": "-",
        "DE": "XAR400",
        "name": "耐磨钢板",
    },
    "NM500": {
        "CN": "NM500",
        "US": "AR500",
        "JP": "-",
        "DE": "XAR500",
        "name": "高硬度耐磨钢板",
    },
    "Hardox450": {
        "CN": "Hardox450",
        "US": "Hardox450",
        "JP": "-",
        "DE": "Hardox450",
        "name": "悍达耐磨钢板",
    },
    # 弹簧钢补充
    "55CrSi": {
        "CN": "55CrSi",
        "US": "-",
        "JP": "SUP12",
        "DE": "55SiCr6",
        "name": "硅铬弹簧钢",
    },
    # 低温钢
    "09MnNiD": {
        "CN": "09MnNiD",
        "US": "A203 Gr.D",
        "JP": "SLA325",
        "DE": "10Ni14",
        "name": "3.5%镍低温钢",
    },
    "16MnDR": {
        "CN": "16MnDR",
        "US": "SA516 Gr.70",
        "JP": "SM490",
        "DE": "P355N",
        "name": "低温压力容器钢",
    },
    "9Ni钢": {
        "CN": "06Ni9",
        "US": "A553 Type I",
        "JP": "SL9N590",
        "DE": "X8Ni9",
        "name": "9%镍钢",
    },
    # 电接触材料
    "AgCdO": {
        "CN": "AgCdO10",
        "US": "AgCdO",
        "JP": "AgCdO",
        "DE": "AgCdO10",
        "name": "银氧化镉触点",
    },
    "AgSnO2": {
        "CN": "AgSnO2",
        "US": "AgSnO2",
        "JP": "AgSnO2",
        "DE": "AgSnO2",
        "name": "银氧化锡触点",
    },
    "CuW70": {
        "CN": "CuW70",
        "US": "RWMA Class 11",
        "JP": "CuW70",
        "DE": "CuW70",
        "name": "铜钨合金",
    },
    # 轴承钢
    "GCr15": {
        "CN": "GCr15",
        "US": "52100",
        "JP": "SUJ2",
        "DE": "100Cr6",
        "name": "高碳铬轴承钢",
    },
    "GCr15SiMn": {
        "CN": "GCr15SiMn",
        "US": "52100改",
        "JP": "SUJ4",
        "DE": "-",
        "name": "高碳铬硅锰轴承钢",
    },
    "GCr4": {
        "CN": "GCr4",
        "US": "SAE 4320",
        "JP": "-",
        "DE": "-",
        "name": "渗碳轴承钢",
    },
    # 弹簧钢
    "60Si2Mn": {
        "CN": "60Si2Mn",
        "US": "9260",
        "JP": "SUP6",
        "DE": "1.7108",
        "name": "硅锰弹簧钢",
    },
    "60Si2CrA": {
        "CN": "60Si2CrA",
        "US": "-",
        "JP": "SUP12",
        "DE": "60SC7",
        "name": "硅铬弹簧钢",
    },
    "50CrVA": {
        "CN": "50CrVA",
        "US": "6150",
        "JP": "SUP10",
        "DE": "50CrV4",
        "name": "铬钒弹簧钢",
    },
    # 耐热不锈钢补充
    "2Cr13": {
        "CN": "2Cr13",
        "US": "420",
        "JP": "SUS420J1",
        "DE": "X20Cr13",
        "name": "马氏体不锈钢",
    },
    "1Cr17": {
        "CN": "1Cr17",
        "US": "430",
        "JP": "SUS430",
        "DE": "X6Cr17",
        "name": "铁素体不锈钢",
    },
    "0Cr25Ni20": {
        "CN": "0Cr25Ni20",
        "US": "310S",
        "JP": "SUS310S",
        "DE": "X8CrNi25-21",
        "name": "耐热不锈钢",
    },
    # 齿轮钢
    "20CrMnTi": {
        "CN": "20CrMnTi",
        "US": "4118",
        "JP": "SCM420H",
        "DE": "-",
        "name": "渗碳齿轮钢",
    },
    "20CrMo": {
        "CN": "20CrMo",
        "US": "4118",
        "JP": "SCM420",
        "DE": "20CrMo5",
        "name": "铬钼渗碳钢",
    },
    "20CrNiMo": {
        "CN": "20CrNiMo",
        "US": "8620",
        "JP": "SNCM220",
        "DE": "20CrNiMo5",
        "name": "铬镍钼渗碳钢",
    },
    # 航空铝合金补充
    "5A06": {
        "CN": "5A06",
        "US": "5456",
        "JP": "A5456",
        "DE": "AlMg5",
        "name": "防锈铝合金",
    },
    "2A14": {
        "CN": "2A14",
        "US": "2014",
        "JP": "A2014",
        "DE": "AlCu4SiMg",
        "name": "高强铝合金",
    },
    "7A04": {
        "CN": "7A04",
        "US": "7050",
        "JP": "A7050",
        "DE": "AlZn6MgCu",
        "name": "超高强铝合金",
    },
    # 气门钢
    "4Cr10Si2Mo": {
        "CN": "4Cr10Si2Mo",
        "US": "-",
        "JP": "SUH3",
        "DE": "X45CrSi9-3",
        "name": "马氏体耐热钢",
    },
    "5Cr21Mn9Ni4N": {
        "CN": "5Cr21Mn9Ni4N",
        "US": "-",
        "JP": "SUH35",
        "DE": "21-4N",
        "name": "奥氏体耐热钢",
    },
    "4Cr14Ni14W2Mo": {
        "CN": "4Cr14Ni14W2Mo",
        "US": "-",
        "JP": "SUH38",
        "DE": "X40CrNiW14-14",
        "name": "高温气门钢",
    },
    # 链条钢/渗碳硼钢
    "20MnVB": {
        "CN": "20MnVB",
        "US": "-",
        "JP": "-",
        "DE": "-",
        "name": "渗碳硼钢",
    },
    "15MnVB": {
        "CN": "15MnVB",
        "US": "-",
        "JP": "-",
        "DE": "-",
        "name": "低碳硼钢",
    },
    "22MnB5": {
        "CN": "22MnB5",
        "US": "USIBOR",
        "JP": "-",
        "DE": "22MnB5",
        "name": "热成形钢",
    },
    # 电工硅钢补充
    "B50A600": {
        "CN": "B50A600",
        "US": "-",
        "JP": "-",
        "DE": "M600-50A",
        "name": "无取向硅钢",
    },
    "B35A230": {
        "CN": "B35A230",
        "US": "-",
        "JP": "-",
        "DE": "M230-35A",
        "name": "高效无取向硅钢",
    },
    "B27R090": {
        "CN": "B27R090",
        "US": "-",
        "JP": "-",
        "DE": "M090-27P",
        "name": "取向硅钢",
    },
    # 高温合金补充
    "GH2132": {
        "CN": "GH2132",
        "US": "A-286",
        "JP": "SUH660",
        "DE": "1.4980",
        "name": "铁基高温合金",
    },
    "K403": {
        "CN": "K403",
        "US": "Mar-M246",
        "JP": "-",
        "DE": "-",
        "name": "镍基铸造高温合金",
    },
    "K418": {
        "CN": "K418",
        "US": "IN-738",
        "JP": "-",
        "DE": "-",
        "name": "镍基铸造高温合金",
    },
    # 铸造铝合金补充
    "ZL101": {
        "CN": "ZL101",
        "US": "A356.2",
        "JP": "AC4CH",
        "DE": "AlSi7Mg",
        "name": "铸造铝硅镁合金",
    },
    "ZL104": {
        "CN": "ZL104",
        "US": "A319",
        "JP": "AC2B",
        "DE": "AlSi5Cu1Mg",
        "name": "铸造铝硅铜合金",
    },
    "ZL201": {
        "CN": "ZL201",
        "US": "A201",
        "JP": "-",
        "DE": "AlCu4MgTi",
        "name": "高强度铸造铝铜合金",
    },
    # 锌合金
    "Zamak3": {
        "CN": "ZnAl4",
        "US": "ASTM B86-AG40A",
        "JP": "ZDC1",
        "DE": "Z400",
        "name": "锌铝合金3号",
    },
    "Zamak5": {
        "CN": "ZnAl4Cu1",
        "US": "ASTM B86-AC41A",
        "JP": "ZDC2",
        "DE": "Z410",
        "name": "锌铝合金5号",
    },
    "ZA-8": {
        "CN": "ZnAl8Cu1",
        "US": "ASTM B669",
        "JP": "-",
        "DE": "-",
        "name": "高铝锌合金",
    },
    # 锡青铜补充
    "QSn7-0.2": {
        "CN": "QSn7-0.2",
        "US": "C52100",
        "JP": "-",
        "DE": "CuSn8",
        "name": "高锡磷青铜",
    },
    "QSn4-0.3": {
        "CN": "QSn4-0.3",
        "US": "C51000",
        "JP": "-",
        "DE": "CuSn4",
        "name": "低锡磷青铜",
    },
    "ZCuSn10P1": {
        "CN": "ZCuSn10P1",
        "US": "C90700",
        "JP": "PBC2A",
        "DE": "-",
        "name": "铸造锡磷青铜",
    },
    # 硅黄铜/铝黄铜
    "HSi80-3": {
        "CN": "HSi80-3",
        "US": "C87500",
        "JP": "-",
        "DE": "CuZn16Si3",
        "name": "硅黄铜",
    },
    "HAl77-2": {
        "CN": "HAl77-2",
        "US": "C68700",
        "JP": "-",
        "DE": "CuZn22Al2",
        "name": "铝黄铜",
    },
    # 耐磨铸铁
    "NiHard1": {
        "CN": "NiHard1",
        "US": "ASTM A532 Class I Type A",
        "JP": "-",
        "DE": "-",
        "name": "镍硬铸铁1型",
    },
    "NiHard4": {
        "CN": "NiHard4",
        "US": "ASTM A532 Class I Type D",
        "JP": "-",
        "DE": "-",
        "name": "镍硬铸铁4型",
    },
    "Cr26": {
        "CN": "KmTBCr26",
        "US": "ASTM A532 Class III",
        "JP": "-",
        "DE": "-",
        "name": "高铬铸铁",
    },
    # 蠕墨铸铁
    "RuT300": {
        "CN": "RuT300",
        "US": "CGI 300",
        "JP": "-",
        "DE": "GJV-300",
        "name": "蠕墨铸铁",
    },
    "RuT350": {
        "CN": "RuT350",
        "US": "CGI 350",
        "JP": "-",
        "DE": "GJV-350",
        "name": "蠕墨铸铁",
    },
    "RuT400": {
        "CN": "RuT400",
        "US": "CGI 400",
        "JP": "-",
        "DE": "GJV-400",
        "name": "高强度蠕墨铸铁",
    },
    # 可锻铸铁
    "KTH300-06": {
        "CN": "KTH300-06",
        "US": "M3210",
        "JP": "-",
        "DE": "GJMB-300-6",
        "name": "黑心可锻铸铁",
    },
    "KTZ450-06": {
        "CN": "KTZ450-06",
        "US": "M4504",
        "JP": "-",
        "DE": "GJMP-450-6",
        "name": "珠光体可锻铸铁",
    },
    "KTZ550-04": {
        "CN": "KTZ550-04",
        "US": "M5503",
        "JP": "-",
        "DE": "GJMP-550-4",
        "name": "高强度珠光体可锻铸铁",
    },
    # 铸造镁合金
    "ZM5": {
        "CN": "ZM5",
        "US": "AZ91D",
        "JP": "MDC1D",
        "DE": "MgAl9Zn1",
        "name": "铸造镁铝锌合金",
    },
    "AM60B": {
        "CN": "AM60B",
        "US": "AM60B",
        "JP": "-",
        "DE": "MgAl6Mn",
        "name": "高韧铸造镁合金",
    },
    "AZ63": {
        "CN": "AZ63",
        "US": "AZ63A",
        "JP": "-",
        "DE": "MgAl6Zn3",
        "name": "砂型铸造镁合金",
    },
    # 粉末冶金材料
    "Fe-Cu-C": {
        "CN": "Fe-Cu-C",
        "US": "FC-0205",
        "JP": "SINT-C11",
        "DE": "Sint-C11",
        "name": "铁铜碳粉末冶金",
    },
    "Fe-Ni-Cu": {
        "CN": "Fe-Ni-Cu",
        "US": "FN-0205",
        "JP": "SINT-D11",
        "DE": "Sint-D11",
        "name": "铁镍铜粉末冶金",
    },
    "316L-PM": {
        "CN": "316L-PM",
        "US": "SS-316L",
        "JP": "SUS316L-PM",
        "DE": "X2CrNiMo17-12-2-PM",
        "name": "316L不锈钢粉末冶金",
    },
    # 硬质合金
    "YG8": {
        "CN": "YG8",
        "US": "C2/C3",
        "JP": "G3",
        "DE": "K30",
        "name": "钨钴硬质合金",
    },
    "YT15": {
        "CN": "YT15",
        "US": "C5/C6",
        "JP": "P15",
        "DE": "P15",
        "name": "钨钛钴硬质合金",
    },
    "YW1": {
        "CN": "YW1",
        "US": "C7",
        "JP": "M10",
        "DE": "M10",
        "name": "钨钛钽钴硬质合金",
    },
    # 结构陶瓷
    "Al2O3-99": {
        "CN": "Al2O3-99",
        "US": "99% Alumina",
        "JP": "A99",
        "DE": "F99.7",
        "name": "99氧化铝陶瓷",
    },
    "Si3N4": {
        "CN": "Si3N4",
        "US": "Si3N4",
        "JP": "SN-220",
        "DE": "Si3N4-GPS",
        "name": "氮化硅陶瓷",
    },
    "ZrO2-3Y": {
        "CN": "ZrO2-3Y",
        "US": "3Y-TZP",
        "JP": "TZP-3Y",
        "DE": "ZrO2-Y2O3",
        "name": "氧化锆陶瓷",
    },
    # 难熔金属
    "Mo-1": {
        "CN": "Mo-1",
        "US": "R03600",
        "JP": "Mo-1",
        "DE": "2.4617",
        "name": "纯钼",
    },
    "TZM": {
        "CN": "TZM",
        "US": "R03630",
        "JP": "TZM",
        "DE": "2.4633",
        "name": "钛锆钼合金",
    },
    "W-1": {
        "CN": "W-1",
        "US": "R07003",
        "JP": "W-1",
        "DE": "2.4701",
        "name": "纯钨",
    },
    "Ta-1": {
        "CN": "Ta-1",
        "US": "R05200",
        "JP": "Ta-1",
        "DE": "2.5001",
        "name": "纯钽",
    },
    # 铝青铜
    "QAl9-4": {
        "CN": "QAl9-4",
        "US": "C62300",
        "JP": "CAC703",
        "DE": "CuAl9Fe4",
        "name": "铝青铜",
    },
    "QAl10-4-4": {
        "CN": "QAl10-4-4",
        "US": "C63000",
        "JP": "CAC704",
        "DE": "CuAl10Ni5Fe5",
        "name": "镍铝青铜",
    },
    # 铍铜
    "QBe2": {
        "CN": "QBe2",
        "US": "C17200",
        "JP": "BeCuA",
        "DE": "CuBe2",
        "name": "铍青铜",
    },
    "QBe1.9": {
        "CN": "QBe1.9",
        "US": "C17000",
        "JP": "BeCuB",
        "DE": "CuBe1.9",
        "name": "低铍铜",
    },
    "CuNi2Si": {
        "CN": "CuNi2Si",
        "US": "C70250",
        "JP": "-",
        "DE": "CuNi2Si",
        "name": "镍硅铜",
    },
    # 无铅焊锡
    "SAC305": {
        "CN": "SAC305",
        "US": "Sn96.5Ag3.0Cu0.5",
        "JP": "SAC305",
        "DE": "SnAg3Cu0.5",
        "name": "无铅焊锡",
    },
    "SAC387": {
        "CN": "SAC387",
        "US": "Sn95.5Ag3.8Cu0.7",
        "JP": "SAC387",
        "DE": "SnAg4Cu0.5",
        "name": "高银无铅焊锡",
    },
    "Sn99.3Cu0.7": {
        "CN": "Sn99.3Cu0.7",
        "US": "SN100C",
        "JP": "Sn-0.7Cu",
        "DE": "SnCu0.7",
        "name": "无银无铅焊锡",
    },
    # 钎焊合金
    "BAg-5": {
        "CN": "HL302",
        "US": "BAg-5",
        "JP": "BAg-5",
        "DE": "L-Ag45Cd",
        "name": "银钎料",
    },
    "BCu-1": {
        "CN": "HL101",
        "US": "BCu-1",
        "JP": "BCu-1",
        "DE": "Cu99.9",
        "name": "纯铜钎料",
    },
    "BNi-2": {
        "CN": "HL401",
        "US": "BNi-2",
        "JP": "BNi-2",
        "DE": "Ni620",
        "name": "镍基钎料",
    },
    # 形状记忆合金
    "NiTi": {
        "CN": "TiNi",
        "US": "Nitinol",
        "JP": "NiTi",
        "DE": "NiTi",
        "name": "镍钛记忆合金",
    },
    "CuZnAl": {
        "CN": "CuZnAl",
        "US": "CuZnAl-SMA",
        "JP": "-",
        "DE": "-",
        "name": "铜锌铝记忆合金",
    },
    "CuAlNi": {
        "CN": "CuAlNi",
        "US": "CuAlNi-SMA",
        "JP": "-",
        "DE": "-",
        "name": "铜铝镍记忆合金",
    },
    # 电触头材料
    "AgCdO": {
        "CN": "AgCdO",
        "US": "Ag/CdO",
        "JP": "AgCdO",
        "DE": "AgCdO",
        "name": "银氧化镉触点",
    },
    "AgSnO2": {
        "CN": "AgSnO2",
        "US": "Ag/SnO2",
        "JP": "AgSnO2",
        "DE": "AgSnO2",
        "name": "银氧化锡触点",
    },
    "CuW": {
        "CN": "CuW70",
        "US": "ASTM B702",
        "JP": "CuW",
        "DE": "WCu30",
        "name": "钨铜合金",
    },
    # 轴承合金
    "ZChSnSb11-6": {
        "CN": "ZChSnSb11-6",
        "US": "ASTM B23 Gr.2",
        "JP": "WJ2",
        "DE": "LgSn89",
        "name": "锡基巴氏合金",
    },
    "ZChPbSb16-16-2": {
        "CN": "ZChPbSb16-16-2",
        "US": "ASTM B23 Gr.15",
        "JP": "-",
        "DE": "LgPb80",
        "name": "铅基巴氏合金",
    },
    "CuPb24Sn4": {
        "CN": "CuPb24Sn4",
        "US": "SAE49",
        "JP": "-",
        "DE": "CuPb24Sn4",
        "name": "铜铅合金轴承",
    },
    # 热电偶合金
    "Chromel": {
        "CN": "镍铬10",
        "US": "Chromel",
        "JP": "NiCr10",
        "DE": "NiCr10",
        "name": "镍铬合金",
    },
    "Alumel": {
        "CN": "镍铝3",
        "US": "Alumel",
        "JP": "NiAl3",
        "DE": "NiAl3",
        "name": "镍铝合金",
    },
    "Constantan": {
        "CN": "6J40",
        "US": "Constantan",
        "JP": "CuNi44",
        "DE": "CuNi44",
        "name": "康铜",
    },
    # 永磁材料
    "NdFeB": {
        "CN": "NdFeB",
        "US": "N35-N52",
        "JP": "NdFeB",
        "DE": "NdFeB",
        "name": "钕铁硼永磁",
    },
    "SmCo": {
        "CN": "SmCo",
        "US": "SmCo5/Sm2Co17",
        "JP": "SmCo",
        "DE": "SmCo",
        "name": "钐钴永磁",
    },
    "Alnico": {
        "CN": "LNG52",
        "US": "Alnico 5",
        "JP": "Alnico",
        "DE": "AlNiCo",
        "name": "铝镍钴永磁",
    },
    # 电阻合金
    "Cr20Ni80": {
        "CN": "Cr20Ni80",
        "US": "Nichrome",
        "JP": "Ni80Cr20",
        "DE": "NiCr80/20",
        "name": "镍铬电热合金",
    },
    "Manganin": {
        "CN": "6J13",
        "US": "Manganin",
        "JP": "CuMn12Ni",
        "DE": "Manganin",
        "name": "锰铜合金",
    },
    "Karma": {
        "CN": "Karma",
        "US": "Karma",
        "JP": "-",
        "DE": "-",
        "name": "卡玛合金",
    },
    # 低膨胀合金
    "Invar": {
        "CN": "4J36",
        "US": "Invar 36",
        "JP": "Invar",
        "DE": "FeNi36",
        "name": "因瓦合金",
    },
    "Kovar": {
        "CN": "4J29",
        "US": "Kovar",
        "JP": "Kovar",
        "DE": "FeNiCo29",
        "name": "可伐合金",
    },
    "4J32": {
        "CN": "4J32",
        "US": "Super Invar",
        "JP": "-",
        "DE": "FeNiCo",
        "name": "超因瓦合金",
    },
    # 超导材料
    "NbTi": {
        "CN": "NbTi",
        "US": "NbTi",
        "JP": "NbTi",
        "DE": "NbTi",
        "name": "铌钛超导合金",
    },
    "Nb3Sn": {
        "CN": "Nb3Sn",
        "US": "Nb3Sn",
        "JP": "Nb3Sn",
        "DE": "Nb3Sn",
        "name": "铌三锡超导体",
    },
    "YBCO": {
        "CN": "YBCO",
        "US": "YBCO",
        "JP": "YBCO",
        "DE": "YBCO",
        "name": "钇钡铜氧高温超导",
    },
    # 核工业材料
    "Zircaloy-4": {
        "CN": "Zr-4",
        "US": "Zircaloy-4",
        "JP": "Zry-4",
        "DE": "Zircaloy-4",
        "name": "锆合金-4",
    },
    "Hafnium": {
        "CN": "Hf",
        "US": "Hafnium",
        "JP": "Hf",
        "DE": "Hafnium",
        "name": "铪",
    },
    "B4C": {
        "CN": "B4C",
        "US": "B4C",
        "JP": "B4C",
        "DE": "B4C",
        "name": "碳化硼",
    },
    # 医用合金
    "CoCrMo": {
        "CN": "CoCrMo",
        "US": "ASTM F75",
        "JP": "CoCrMo",
        "DE": "CoCr28Mo6",
        "name": "钴铬钼医用合金",
    },
    "Ti6Al4V-ELI": {
        "CN": "TC4-ELI",
        "US": "Ti-6Al-4V ELI",
        "JP": "TAB6400ELI",
        "DE": "Ti6Al4V-ELI",
        "name": "医用钛合金",
    },
    "316L-Medical": {
        "CN": "00Cr17Ni14Mo2",
        "US": "316L Medical",
        "JP": "SUS316L-Med",
        "DE": "1.4404-Med",
        "name": "医用不锈钢",
    },
    # 光学材料
    "Fused-Silica": {
        "CN": "JGS1/JGS2",
        "US": "Fused Silica",
        "JP": "石英ガラス",
        "DE": "Quarzglas",
        "name": "熔融石英",
    },
    "Sapphire": {
        "CN": "蓝宝石",
        "US": "Sapphire",
        "JP": "サファイア",
        "DE": "Saphir",
        "name": "蓝宝石",
    },
    "Germanium": {
        "CN": "Ge",
        "US": "Germanium",
        "JP": "ゲルマニウム",
        "DE": "Germanium",
        "name": "锗",
    },
    # 电池材料
    "LiFePO4": {
        "CN": "LFP",
        "US": "LiFePO4",
        "JP": "LFP",
        "DE": "LiFePO4",
        "name": "磷酸铁锂",
    },
    "NMC": {
        "CN": "NCM",
        "US": "NMC",
        "JP": "NCM",
        "DE": "NMC",
        "name": "三元正极材料",
    },
    "Graphite-Battery": {
        "CN": "负极石墨",
        "US": "Anode Graphite",
        "JP": "負極黒鉛",
        "DE": "Anodengraphit",
        "name": "电池负极石墨",
    },
    # 半导体材料
    "Silicon-Wafer": {
        "CN": "单晶硅",
        "US": "Si Wafer",
        "JP": "シリコンウェハ",
        "DE": "Siliziumwafer",
        "name": "硅晶圆",
    },
    "GaAs": {
        "CN": "GaAs",
        "US": "GaAs",
        "JP": "GaAs",
        "DE": "GaAs",
        "name": "砷化镓",
    },
    "SiC-Semiconductor": {
        "CN": "SiC",
        "US": "SiC",
        "JP": "SiC",
        "DE": "SiC",
        "name": "碳化硅半导体",
    },
    # 热界面材料
    "Thermal-Paste": {
        "CN": "导热硅脂",
        "US": "Thermal Paste",
        "JP": "サーマルグリス",
        "DE": "Wärmeleitpaste",
        "name": "导热硅脂",
    },
    "Thermal-Pad": {
        "CN": "导热垫片",
        "US": "Thermal Pad",
        "JP": "サーマルパッド",
        "DE": "Wärmeleitpad",
        "name": "导热垫片",
    },
    "Graphene-TIM": {
        "CN": "石墨烯散热膜",
        "US": "Graphene TIM",
        "JP": "グラフェン放熱",
        "DE": "Graphen-TIM",
        "name": "石墨烯导热材料",
    },
    # 增材制造材料
    "AlSi10Mg-AM": {
        "CN": "AlSi10Mg",
        "US": "AlSi10Mg",
        "JP": "AlSi10Mg",
        "DE": "AlSi10Mg",
        "name": "3D打印铝合金",
    },
    "IN718-AM": {
        "CN": "IN718",
        "US": "Inconel 718",
        "JP": "IN718",
        "DE": "IN718",
        "name": "3D打印镍基高温合金",
    },
    "Ti64-AM": {
        "CN": "TC4",
        "US": "Ti-6Al-4V",
        "JP": "Ti64",
        "DE": "Ti6Al4V",
        "name": "3D打印钛合金",
    },
    # 硬质合金
    "WC-Co": {
        "CN": "YG8/YG15",
        "US": "Grade C2",
        "JP": "V10",
        "DE": "K10/K20",
        "name": "钨钴硬质合金",
    },
    "Stellite": {
        "CN": "钴6",
        "US": "Stellite 6",
        "JP": "Stellite",
        "DE": "Stellite 6",
        "name": "司太立合金",
    },
    "CBN": {
        "CN": "CBN",
        "US": "CBN",
        "JP": "CBN",
        "DE": "CBN",
        "name": "立方氮化硼",
    },
    # 热障涂层材料
    "YSZ": {
        "CN": "8YSZ",
        "US": "YSZ",
        "JP": "YSZ",
        "DE": "YSZ",
        "name": "氧化钇稳定氧化锆",
    },
    "Al2O3-TBC": {
        "CN": "Al2O3涂层",
        "US": "Alumina TBC",
        "JP": "Al2O3",
        "DE": "Al2O3",
        "name": "氧化铝热障涂层",
    },
    "MCrAlY": {
        "CN": "MCrAlY",
        "US": "MCrAlY",
        "JP": "MCrAlY",
        "DE": "MCrAlY",
        "name": "金属粘结层",
    },
    # 电磁屏蔽材料
    "Mu-Metal": {
        "CN": "1J79",
        "US": "Mu-Metal",
        "JP": "パーマロイ",
        "DE": "Mumetall",
        "name": "坡莫合金",
    },
    "Permalloy": {
        "CN": "1J50",
        "US": "Permalloy",
        "JP": "パーマロイ",
        "DE": "Permalloy",
        "name": "铁镍软磁合金",
    },
    "Copper-Mesh": {
        "CN": "铜网",
        "US": "Copper Mesh",
        "JP": "銅メッシュ",
        "DE": "Kupfernetz",
        "name": "铜网屏蔽材料",
    },
    # 铸铁
    "HT200": {
        "CN": "HT200",
        "US": "Class 30",
        "JP": "FC200",
        "DE": "GG20",
        "name": "灰铸铁",
    },
    "HT250": {
        "CN": "HT250",
        "US": "Class 35",
        "JP": "FC250",
        "DE": "GG25",
        "name": "灰铸铁",
    },
    "HT300": {
        "CN": "HT300",
        "US": "Class 40",
        "JP": "FC300",
        "DE": "GG30",
        "name": "高强度灰铸铁",
    },
    "QT400": {
        "CN": "QT400-15",
        "US": "60-40-18",
        "JP": "FCD400",
        "DE": "GGG40",
        "name": "球墨铸铁",
    },
    "QT500-7": {
        "CN": "QT500-7",
        "US": "80-55-06",
        "JP": "FCD500",
        "DE": "GGG50",
        "name": "球墨铸铁",
    },
    "QT600-3": {
        "CN": "QT600-3",
        "US": "100-70-03",
        "JP": "FCD600",
        "DE": "GGG60",
        "name": "高强度球墨铸铁",
    },
    "QT700-2": {
        "CN": "QT700-2",
        "US": "120-90-02",
        "JP": "FCD700",
        "DE": "GGG70",
        "name": "高强度球墨铸铁",
    },
    # 铝合金
    "6061": {
        "CN": "6061",
        "US": "6061",
        "JP": "A6061",
        "DE": "AlMg1SiCu",
        "name": "铝镁硅合金",
    },
    "7075": {
        "CN": "7075",
        "US": "7075",
        "JP": "A7075",
        "DE": "AlZn5.5MgCu",
        "name": "超硬铝合金",
    },
    "2024": {
        "CN": "2024",
        "US": "2024",
        "JP": "A2024",
        "DE": "AlCuMg1",
        "name": "硬铝合金",
    },
    "5052": {
        "CN": "5052",
        "US": "5052",
        "JP": "A5052",
        "DE": "AlMg2.5",
        "name": "防锈铝合金",
    },
    "5083": {
        "CN": "5083",
        "US": "5083",
        "JP": "A5083",
        "DE": "AlMg4.5Mn",
        "name": "船用铝合金",
    },
    "2A12": {
        "CN": "2A12",
        "US": "2024",
        "JP": "A2024",
        "DE": "AlCuMg1",
        "name": "硬铝合金",
    },
    "6063": {
        "CN": "6063",
        "US": "6063",
        "JP": "A6063",
        "DE": "AlMgSi",
        "name": "建筑铝合金",
    },
    # 铸造铝合金
    "A356": {
        "CN": "ZL101",
        "US": "A356",
        "JP": "AC4C",
        "DE": "AlSi7Mg",
        "name": "铸造铝硅合金",
    },
    "ZL102": {
        "CN": "ZL102",
        "US": "A413",
        "JP": "ADC1",
        "DE": "AlSi12",
        "name": "铸造铝硅合金",
    },
    "ADC12": {
        "CN": "YL113",
        "US": "A383",
        "JP": "ADC12",
        "DE": "AlSi11Cu3",
        "name": "压铸铝合金",
    },
    # 钛合金
    "TA1": {
        "CN": "TA1",
        "US": "Gr1",
        "JP": "TB270C",
        "DE": "3.7025",
        "name": "工业纯钛",
    },
    "TA2": {
        "CN": "TA2",
        "US": "Gr2",
        "JP": "TB340C",
        "DE": "3.7035",
        "name": "工业纯钛",
    },
    "TC4": {
        "CN": "TC4",
        "US": "Ti-6Al-4V",
        "JP": "TB480",
        "DE": "3.7165",
        "name": "钛合金",
    },
    "TC11": {
        "CN": "TC11",
        "US": "-",
        "JP": "-",
        "DE": "-",
        "name": "高温钛合金",
    },
    "TB6": {
        "CN": "TB6",
        "US": "Ti-10V-2Fe-3Al",
        "JP": "-",
        "DE": "-",
        "name": "β型钛合金",
    },
    "TC21": {
        "CN": "TC21",
        "US": "-",
        "JP": "-",
        "DE": "-",
        "name": "损伤容限型钛合金",
    },
    # 镁合金
    "AZ31B": {
        "CN": "AZ31B",
        "US": "AZ31B",
        "JP": "MC1",
        "DE": "MgAl3Zn1",
        "name": "变形镁合金",
    },
    "AZ91D": {
        "CN": "AZ91D",
        "US": "AZ91D",
        "JP": "MDC1D",
        "DE": "MgAl9Zn1",
        "name": "压铸镁合金",
    },
    "ZK60": {
        "CN": "ZK60",
        "US": "ZK60A",
        "JP": "ZK60",
        "DE": "MgZn6Zr",
        "name": "高强镁合金",
    },
    # 硬质合金
    "YG8": {
        "CN": "YG8",
        "US": "C2",
        "ISO": "K20",
        "DE": "WC-8Co",
        "name": "钨钴硬质合金",
    },
    "YT15": {
        "CN": "YT15",
        "US": "C5",
        "ISO": "P20",
        "DE": "WC-TiC-15Co",
        "name": "钨钛钴硬质合金",
    },
    "YG6": {
        "CN": "YG6",
        "US": "C1",
        "ISO": "K10",
        "DE": "WC-6Co",
        "name": "钨钴硬质合金",
    },
    # 镍基合金
    "C276": {
        "CN": "NS334",
        "US": "C-276",
        "JP": "NW0276",
        "DE": "2.4819",
        "UNS": "N10276",
        "name": "哈氏合金",
    },
    "C22": {
        "CN": "NS335",
        "US": "C-22",
        "JP": "NW0022",
        "DE": "2.4602",
        "UNS": "N06022",
        "name": "哈氏合金",
    },
    "Inconel625": {
        "CN": "GH3625",
        "US": "Inconel 625",
        "JP": "NCF625",
        "DE": "2.4856",
        "UNS": "N06625",
        "name": "因科镍合金",
    },
    "Inconel718": {
        "CN": "GH4169",
        "US": "Inconel 718",
        "JP": "NCF718",
        "DE": "2.4668",
        "UNS": "N07718",
        "name": "因科镍合金",
    },
    "Monel400": {
        "CN": "NCu30",
        "US": "Monel 400",
        "JP": "NW4400",
        "DE": "2.4360",
        "UNS": "N04400",
        "name": "蒙乃尔合金",
    },
    "MonelK500": {
        "CN": "NCu30-2.5-1.5",
        "US": "Monel K-500",
        "JP": "NW5500",
        "DE": "2.4375",
        "UNS": "N05500",
        "name": "时效硬化蒙乃尔",
    },
    "HastelloyB3": {
        "CN": "NS321",
        "US": "Hastelloy B-3",
        "DE": "2.4600",
        "UNS": "N10675",
        "name": "哈氏B3合金",
    },
    "Stellite6": {
        "CN": "钴基6号",
        "US": "Stellite 6",
        "DE": "2.4778",
        "name": "司太立合金",
    },
    "Incoloy825": {
        "CN": "GH2825",
        "US": "Incoloy 825",
        "JP": "NCF825",
        "DE": "2.4858",
        "UNS": "N08825",
        "name": "因科洛伊合金",
    },
    # 铜合金
    "H62": {
        "CN": "H62",
        "US": "C27400",
        "JP": "C2700",
        "DE": "CuZn37",
        "name": "普通黄铜",
    },
    "H68": {
        "CN": "H68",
        "US": "C26800",
        "JP": "C2680",
        "DE": "CuZn33",
        "name": "高精黄铜",
    },
    "HPb59-1": {
        "CN": "HPb59-1",
        "US": "C38500",
        "JP": "C3604",
        "DE": "CuZn39Pb3",
        "name": "铅黄铜",
    },
    "QBe2": {
        "CN": "QBe2",
        "US": "C17200",
        "JP": "C1720",
        "DE": "CuBe2",
        "UNS": "C17200",
        "name": "铍青铜",
    },
    "QAl10-3-1.5": {
        "CN": "QAl10-3-1.5",
        "US": "C63000",
        "JP": "CAC702",
        "DE": "CuAl10Fe3",
        "UNS": "C63000",
        "name": "铝青铜",
    },
    "QAl9-4": {
        "CN": "QAl9-4",
        "US": "C95400",
        "JP": "CAC702",
        "DE": "CuAl10Fe",
        "UNS": "C95400",
        "name": "铝青铜",
    },
    "QSn4-3": {
        "CN": "QSn4-3",
        "US": "C52100",
        "JP": "C5210",
        "DE": "CuSn4",
        "name": "锡青铜",
    },
    "QSn6.5-0.1": {
        "CN": "QSn6.5-0.1",
        "US": "C51900",
        "JP": "C5191",
        "DE": "CuSn6",
        "UNS": "C51900",
        "name": "磷青铜",
    },
    "Cu65": {
        "CN": "T2",
        "US": "C11000",
        "JP": "C1100",
        "DE": "Cu-ETP",
        "UNS": "C11000",
        "name": "紫铜",
    },
    "CuNi10Fe1Mn": {
        "CN": "BFe10-1-1",
        "US": "C70600",
        "JP": "C7060",
        "DE": "CuNi10Fe1Mn",
        "UNS": "C70600",
        "name": "白铜",
    },
    # 工程塑料
    "PEEK": {
        "CN": "PEEK",
        "US": "PEEK",
        "DE": "PEEK",
        "ISO": "ISO 19924",
        "name": "聚醚醚酮",
    },
    "POM": {
        "CN": "POM",
        "US": "Delrin",
        "DE": "POM-C/POM-H",
        "ISO": "ISO 9988",
        "name": "聚甲醛",
    },
    "PA66": {
        "CN": "PA66",
        "US": "Nylon 66",
        "DE": "PA66",
        "ISO": "ISO 1874",
        "name": "尼龙66",
    },
    "PC": {
        "CN": "PC",
        "US": "Polycarbonate",
        "DE": "PC",
        "ISO": "ISO 7391",
        "name": "聚碳酸酯",
    },
    "UHMWPE": {
        "CN": "UHMWPE",
        "US": "UHMW-PE",
        "DE": "PE-UHMW",
        "ISO": "ISO 11542",
        "name": "超高分子量聚乙烯",
    },
    "PTFE": {
        "CN": "PTFE",
        "US": "Teflon",
        "DE": "PTFE",
        "ISO": "ISO 12086",
        "name": "聚四氟乙烯",
    },
}


def get_material_equivalence(material: str) -> Optional[Dict[str, str]]:
    """
    获取材料等价表

    Args:
        material: 材料名称或牌号

    Returns:
        等价表字典 {标准体系: 等价牌号} 或 None
    """
    # 先尝试分类获取标准牌号
    info = classify_material_detailed(material)
    if info:
        grade = info.grade
        if grade in MATERIAL_EQUIVALENCE:
            return MATERIAL_EQUIVALENCE[grade]

    # 直接查找
    if material in MATERIAL_EQUIVALENCE:
        return MATERIAL_EQUIVALENCE[material]

    # 反向查找（输入的可能是其他标准的牌号）
    material_upper = material.upper().replace("-", "").replace(" ", "")
    for grade, equiv in MATERIAL_EQUIVALENCE.items():
        for std, val in equiv.items():
            if std != "name":
                val_clean = val.upper().replace("-", "").replace(" ", "")
                if val_clean == material_upper:
                    return equiv

    return None


def find_equivalent_material(material: str, target_standard: str = "CN") -> Optional[str]:
    """
    查找等价材料牌号

    Args:
        material: 材料名称或牌号
        target_standard: 目标标准体系 (CN/US/JP/DE/UNS)

    Returns:
        等价牌号 或 None
    """
    equiv = get_material_equivalence(material)
    if equiv and target_standard in equiv:
        return equiv[target_standard]
    return None


def list_material_standards(material: str) -> List[Tuple[str, str]]:
    """
    列出材料的所有标准牌号

    Args:
        material: 材料名称或牌号

    Returns:
        [(标准体系, 牌号), ...] 列表
    """
    equiv = get_material_equivalence(material)
    if equiv:
        return [(std, val) for std, val in equiv.items() if std != "name"]
    return []


def export_materials_csv(filepath: Optional[str] = None) -> str:
    """
    导出材料数据库为 CSV 格式

    Args:
        filepath: 可选的文件路径，如果提供则写入文件

    Returns:
        CSV 格式字符串
    """
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # 写入表头
    headers = [
        "牌号", "名称", "别名", "类别", "子类", "材料组",
        "密度(g/cm³)", "抗拉强度(MPa)", "屈服强度(MPa)", "硬度",
        "可加工性", "可焊性", "毛坯形式", "需要专用刀具", "需要冷却液",
        "推荐热处理", "禁止热处理", "推荐表面处理", "切削速度(m/min)",
        "警告", "建议", "描述"
    ]
    writer.writerow(headers)

    # 按牌号排序
    for grade in sorted(MATERIAL_DATABASE.keys()):
        info = MATERIAL_DATABASE[grade]
        props = info.properties
        proc = info.process

        row = [
            info.grade,
            info.name,
            "|".join(info.aliases) if info.aliases else "",
            info.category.value,
            info.sub_category.value,
            info.group.value,
            props.density if props.density else "",
            props.tensile_strength if props.tensile_strength else "",
            props.yield_strength if props.yield_strength else "",
            props.hardness if props.hardness else "",
            props.machinability if props.machinability else "",
            props.weldability if props.weldability else "",
            "|".join(proc.blank_forms) if proc.blank_forms else "",
            "是" if proc.special_tooling else "否",
            "是" if proc.coolant_required else "否",
            "|".join(proc.heat_treatments) if proc.heat_treatments else "",
            "|".join(proc.forbidden_heat_treatments) if proc.forbidden_heat_treatments else "",
            "|".join(proc.surface_treatments) if proc.surface_treatments else "",
            f"{proc.cutting_speed_range[0]}-{proc.cutting_speed_range[1]}" if proc.cutting_speed_range else "",
            "|".join(proc.warnings) if proc.warnings else "",
            "|".join(proc.recommendations) if proc.recommendations else "",
            info.description,
        ]
        writer.writerow(row)

    csv_content = output.getvalue()
    output.close()

    # 如果提供了文件路径，写入文件
    if filepath:
        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            f.write(csv_content)
        logger.info("Materials exported to %s", filepath)

    return csv_content


def export_equivalence_csv(filepath: Optional[str] = None) -> str:
    """
    导出材料等价表为 CSV 格式

    Args:
        filepath: 可选的文件路径，如果提供则写入文件

    Returns:
        CSV 格式字符串
    """
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # 写入表头
    headers = ["牌号", "名称", "中国(CN)", "美国(US)", "日本(JP)", "德国(DE)", "UNS"]
    writer.writerow(headers)

    # 按牌号排序
    for grade in sorted(MATERIAL_EQUIVALENCE.keys()):
        equiv = MATERIAL_EQUIVALENCE[grade]
        row = [
            grade,
            equiv.get("name", ""),
            equiv.get("CN", ""),
            equiv.get("US", ""),
            equiv.get("JP", ""),
            equiv.get("DE", ""),
            equiv.get("UNS", ""),
        ]
        writer.writerow(row)

    csv_content = output.getvalue()
    output.close()

    # 如果提供了文件路径，写入文件
    if filepath:
        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            f.write(csv_content)
        logger.info("Material equivalence exported to %s", filepath)

    return csv_content


# ============================================================================
# 材料搜索功能
# ============================================================================

# 拼音映射表 (常用材料名称)
PINYIN_MAP: Dict[str, List[str]] = {
    # 材料类别
    "butixiugang": ["不锈钢", "S30408", "S31603"],
    "buxiugang": ["不锈钢", "S30408", "S31603"],
    "tansugang": ["碳素钢", "Q235B", "45", "20"],
    "hejingang": ["合金钢", "40Cr", "42CrMo", "GCr15"],
    "zhutie": ["铸铁", "HT200", "QT400"],
    "lvhejin": ["铝合金", "6061", "7075"],
    "tonghejin": ["铜合金", "H62", "QBe2"],
    "taihejin": ["钛合金", "TA2", "TC4", "TA1", "TC11", "TB6", "TC21"],
    "nieji": ["镍基", "C276", "Inconel625"],
    "meihejin": ["镁合金", "AZ31B", "AZ91D", "ZK60"],
    "yingzhihejin": ["硬质合金", "YG8", "YT15", "YG6"],
    "wugang": ["钨钢", "YG8", "YG6"],

    # 具体材料
    "huangtong": ["黄铜", "H62", "H68"],
    "qingtong": ["青铜", "QSn4-3", "QAl9-4"],
    "baitong": ["白铜", "CuNi10Fe1Mn"],
    "zitong": ["紫铜", "Cu65"],
    "chungang": ["纯钢", "Q235B"],
    "digang": ["低碳钢", "Q235B", "20"],
    "zhonggang": ["中碳钢", "45"],
    "gaogang": ["高碳钢", "GCr15"],
    "moju": ["模具钢", "Cr12MoV", "H13"],
    "danjia": ["弹簧", "QSn6.5-0.1", "QBe2"],
    "naifu": ["耐腐蚀", "C276", "S31603"],
    "naimo": ["耐磨", "GCr15", "Stellite6"],
    "daodian": ["导电", "Cu65", "H62"],
    "daore": ["导热", "Cu65", "6061"],

    # 拼音缩写
    "bxg": ["不锈钢", "S30408", "S31603"],
    "tsg": ["碳素钢", "Q235B", "45"],
    "hjg": ["合金钢", "40Cr", "42CrMo"],
    "lhj": ["铝合金", "6061", "7075"],
    "thj": ["铜合金", "H62", "QBe2"],
    "zt": ["铸铁", "HT200"],
}


def _calculate_similarity(s1: str, s2: str) -> float:
    """
    计算两个字符串的相似度 (0.0-1.0)
    使用简化的编辑距离算法
    """
    s1_lower = s1.lower()
    s2_lower = s2.lower()

    # 完全匹配
    if s1_lower == s2_lower:
        return 1.0

    # 包含关系
    if s1_lower in s2_lower or s2_lower in s1_lower:
        return 0.8

    # 前缀匹配
    min_len = min(len(s1_lower), len(s2_lower))
    if min_len > 0:
        prefix_match = 0
        for i in range(min_len):
            if s1_lower[i] == s2_lower[i]:
                prefix_match += 1
            else:
                break
        if prefix_match >= 2:
            return 0.5 + 0.3 * (prefix_match / min_len)

    # 字符重叠
    set1 = set(s1_lower)
    set2 = set(s2_lower)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if union > 0:
        return 0.3 * (intersection / union)

    return 0.0


def search_materials(
    query: str,
    limit: int = 10,
    category: Optional[str] = None,
    group: Optional[str] = None,
    min_score: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    搜索材料（支持模糊搜索和拼音）

    Args:
        query: 搜索关键词（支持中文、英文、拼音）
        limit: 返回结果数量限制
        category: 限定材料类别 (metal/non_metal/composite)
        group: 限定材料组
        min_score: 最小匹配分数 (0.0-1.0)

    Returns:
        匹配的材料列表，按相关度排序
    """
    if not query or not query.strip():
        return []

    query = query.strip()
    query_lower = query.lower()
    results: List[Tuple[float, str, MaterialInfo, str]] = []

    # 1. 先尝试精确匹配
    exact_match = classify_material_detailed(query)
    if exact_match:
        if category and exact_match.category.value != category:
            pass
        elif group and exact_match.group.value != group:
            pass
        else:
            return [{
                "grade": exact_match.grade,
                "name": exact_match.name,
                "category": exact_match.category.value,
                "group": exact_match.group.value,
                "score": 1.0,
                "match_type": "exact",
            }]

    # 2. 检查拼音映射
    pinyin_matches: List[str] = []
    for pinyin, materials in PINYIN_MAP.items():
        if query_lower == pinyin or query_lower in pinyin or pinyin in query_lower:
            pinyin_matches.extend(materials[1:])  # 跳过描述词

    # 3. 遍历材料数据库进行模糊匹配
    for grade, info in MATERIAL_DATABASE.items():
        # 应用过滤器
        if category and info.category.value != category:
            continue
        if group and info.group.value != group:
            continue

        # 计算匹配分数
        score = 0.0
        match_type = "fuzzy"

        # 拼音匹配加分
        if grade in pinyin_matches:
            score = max(score, 0.85)
            match_type = "pinyin"

        # 牌号匹配
        grade_score = _calculate_similarity(query, grade)
        if grade_score > score:
            score = grade_score
            match_type = "grade"

        # 名称匹配
        name_score = _calculate_similarity(query, info.name)
        if name_score > score:
            score = name_score
            match_type = "name"

        # 别名匹配
        for alias in info.aliases:
            alias_score = _calculate_similarity(query, alias)
            if alias_score > score:
                score = alias_score
                match_type = "alias"

        # 描述匹配
        if info.description and query_lower in info.description.lower():
            score = max(score, 0.6)
            if match_type == "fuzzy":
                match_type = "description"

        # 标准匹配
        for std in info.standards:
            if query_lower in std.lower():
                score = max(score, 0.5)
                if match_type == "fuzzy":
                    match_type = "standard"

        if score >= min_score:
            results.append((score, grade, info, match_type))

    # 按分数排序
    results.sort(key=lambda x: (-x[0], x[1]))

    # 格式化返回结果
    formatted_results = []
    for score, grade, info, match_type in results[:limit]:
        formatted_results.append({
            "grade": grade,
            "name": info.name,
            "category": info.category.value,
            "group": info.group.value,
            "score": round(score, 2),
            "match_type": match_type,
        })

    return formatted_results


def search_by_properties(
    density_range: Optional[Tuple[float, float]] = None,
    tensile_strength_range: Optional[Tuple[float, float]] = None,
    hardness_contains: Optional[str] = None,
    machinability: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    按属性搜索材料

    Args:
        density_range: 密度范围 (min, max) g/cm³
        tensile_strength_range: 抗拉强度范围 (min, max) MPa
        hardness_contains: 硬度包含的字符串 (如 "HRC", "HB")
        machinability: 可加工性 (excellent/good/fair/poor)
        category: 材料类别
        limit: 返回数量限制

    Returns:
        匹配的材料列表
    """
    results = []

    for grade, info in MATERIAL_DATABASE.items():
        # 类别过滤
        if category and info.category.value != category:
            continue

        props = info.properties

        # 密度过滤
        if density_range:
            if props.density is None:
                continue
            if not (density_range[0] <= props.density <= density_range[1]):
                continue

        # 抗拉强度过滤
        if tensile_strength_range:
            if props.tensile_strength is None:
                continue
            if not (tensile_strength_range[0] <= props.tensile_strength <= tensile_strength_range[1]):
                continue

        # 硬度过滤
        if hardness_contains:
            if props.hardness is None:
                continue
            if hardness_contains.upper() not in props.hardness.upper():
                continue

        # 可加工性过滤
        if machinability:
            if props.machinability != machinability:
                continue

        results.append({
            "grade": grade,
            "name": info.name,
            "category": info.category.value,
            "group": info.group.value,
            "density": props.density,
            "tensile_strength": props.tensile_strength,
            "hardness": props.hardness,
            "machinability": props.machinability,
        })

    # 按牌号排序
    results.sort(key=lambda x: str(x.get("grade") or ""))

    return results[:limit]


# ============================================================================
# 材料推荐系统
# ============================================================================

# 用途到材料组的映射
APPLICATION_MAP: Dict[str, Dict[str, Any]] = {
    # 结构件
    "structural": {
        "name": "结构件",
        "groups": ["carbon_steel", "alloy_steel", "aluminum"],
        "priorities": {"strength": 0.4, "cost": 0.3, "machinability": 0.3},
    },
    "load_bearing": {
        "name": "承载件",
        "groups": ["alloy_steel", "carbon_steel"],
        "priorities": {"strength": 0.5, "toughness": 0.3, "cost": 0.2},
    },
    # 耐腐蚀
    "corrosion_resistant": {
        "name": "耐腐蚀件",
        "groups": ["stainless_steel", "corrosion_resistant", "titanium"],
        "priorities": {"corrosion": 0.5, "strength": 0.3, "cost": 0.2},
    },
    "seawater": {
        "name": "海水环境",
        "groups": ["corrosion_resistant", "titanium", "copper"],
        "priorities": {"corrosion": 0.6, "strength": 0.2, "cost": 0.2},
    },
    "chemical": {
        "name": "化工环境",
        "groups": ["corrosion_resistant", "fluoropolymer"],
        "priorities": {"corrosion": 0.6, "temperature": 0.2, "cost": 0.2},
    },
    # 耐磨
    "wear_resistant": {
        "name": "耐磨件",
        "groups": ["alloy_steel", "cast_iron", "copper"],
        "priorities": {"hardness": 0.5, "toughness": 0.3, "cost": 0.2},
    },
    "bearing": {
        "name": "轴承/轴瓦",
        "groups": ["alloy_steel", "copper"],
        "priorities": {"hardness": 0.4, "wear": 0.4, "machinability": 0.2},
    },
    # 导电导热
    "electrical": {
        "name": "导电件",
        "groups": ["copper", "aluminum"],
        "priorities": {"conductivity": 0.6, "cost": 0.2, "machinability": 0.2},
    },
    "thermal": {
        "name": "导热件",
        "groups": ["copper", "aluminum"],
        "priorities": {"thermal": 0.5, "cost": 0.3, "machinability": 0.2},
    },
    # 弹性件
    "spring": {
        "name": "弹簧/弹性件",
        "groups": ["alloy_steel", "copper"],
        "priorities": {"elasticity": 0.5, "fatigue": 0.3, "corrosion": 0.2},
    },
    # 轻量化
    "lightweight": {
        "name": "轻量化",
        "groups": ["aluminum", "titanium", "engineering_plastic"],
        "priorities": {"weight": 0.5, "strength": 0.3, "cost": 0.2},
    },
    # 高温
    "high_temperature": {
        "name": "高温环境",
        "groups": ["corrosion_resistant", "titanium"],
        "priorities": {"temperature": 0.5, "strength": 0.3, "oxidation": 0.2},
    },
    # 食品医疗
    "food_grade": {
        "name": "食品级",
        "groups": ["stainless_steel", "engineering_plastic"],
        "priorities": {"safety": 0.5, "corrosion": 0.3, "machinability": 0.2},
    },
    "medical": {
        "name": "医疗器械",
        "groups": ["stainless_steel", "titanium"],
        "priorities": {"biocompatibility": 0.5, "corrosion": 0.3, "strength": 0.2},
    },
    # 精密加工
    "precision": {
        "name": "精密零件",
        "groups": ["copper", "aluminum", "alloy_steel"],
        "priorities": {"machinability": 0.5, "stability": 0.3, "cost": 0.2},
    },
}

# 材料替代关系 (材料 -> 可替代材料列表，按优先级排序)
MATERIAL_ALTERNATIVES: Dict[str, List[Dict[str, Any]]] = {
    # 不锈钢替代
    "S30408": [
        {"grade": "S31603", "reason": "更好的耐腐蚀性", "cost_factor": 1.2},
        {"grade": "S30403", "reason": "低碳版本，焊接性更好", "cost_factor": 1.1},
    ],
    "S31603": [
        {"grade": "S30408", "reason": "成本更低", "cost_factor": 0.8},
        {"grade": "C276", "reason": "极端耐腐蚀环境", "cost_factor": 3.0},
    ],
    # 碳钢替代
    "Q235B": [
        {"grade": "Q345R", "reason": "更高强度", "cost_factor": 1.1},
        {"grade": "20", "reason": "需要渗碳时", "cost_factor": 1.0},
    ],
    "45": [
        {"grade": "40Cr", "reason": "更高强度和韧性", "cost_factor": 1.3},
        {"grade": "42CrMo", "reason": "高强度高韧性", "cost_factor": 1.5},
    ],
    # 合金钢替代
    "40Cr": [
        {"grade": "42CrMo", "reason": "更高强度", "cost_factor": 1.2},
        {"grade": "45", "reason": "成本更低", "cost_factor": 0.8},
    ],
    "42CrMo": [
        {"grade": "40Cr", "reason": "成本更低", "cost_factor": 0.8},
        {"grade": "GCr15", "reason": "需要高硬度时", "cost_factor": 1.1},
    ],
    # 铝合金替代
    "6061": [
        {"grade": "7075", "reason": "更高强度", "cost_factor": 1.5},
        {"grade": "5052", "reason": "更好的耐腐蚀性", "cost_factor": 0.9},
    ],
    "7075": [
        {"grade": "6061", "reason": "成本更低，易加工", "cost_factor": 0.7},
        {"grade": "TC4", "reason": "极端强度要求", "cost_factor": 5.0},
    ],
    # 铜合金替代
    "H62": [
        {"grade": "H68", "reason": "更好的冷加工性", "cost_factor": 1.1},
        {"grade": "HPb59-1", "reason": "更好的切削性", "cost_factor": 1.0},
    ],
    "QBe2": [
        {"grade": "QSn6.5-0.1", "reason": "无毒替代（弹性稍差）", "cost_factor": 0.5},
        {"grade": "C17510", "reason": "低铍含量", "cost_factor": 0.9},
    ],
    "Cu65": [
        {"grade": "H62", "reason": "强度更高，成本更低", "cost_factor": 0.8},
        {"grade": "6061", "reason": "轻量化替代", "cost_factor": 0.6},
    ],
    # 钛合金替代
    "TC4": [
        {"grade": "TA2", "reason": "成本更低（强度较低）", "cost_factor": 0.6},
        {"grade": "Inconel718", "reason": "高温环境", "cost_factor": 1.5},
    ],
    "TA2": [
        {"grade": "S31603", "reason": "成本更低", "cost_factor": 0.3},
        {"grade": "TC4", "reason": "需要更高强度", "cost_factor": 1.7},
    ],
    # 耐蚀合金替代
    "C276": [
        {"grade": "C22", "reason": "类似性能，略低成本", "cost_factor": 0.9},
        {"grade": "Inconel625", "reason": "高温性能更好", "cost_factor": 1.1},
    ],
    "Inconel625": [
        {"grade": "Inconel718", "reason": "更高强度", "cost_factor": 1.2},
        {"grade": "C276", "reason": "更好的耐腐蚀性", "cost_factor": 0.9},
    ],
}


def get_material_recommendations(
    application: str,
    requirements: Optional[Dict[str, Any]] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    根据用途推荐材料

    Args:
        application: 用途代码 (structural, corrosion_resistant, electrical, etc.)
        requirements: 额外要求 {
            "min_strength": float,  # 最小抗拉强度
            "max_density": float,   # 最大密度
            "machinability": str,   # 可加工性要求
            "exclude_groups": list, # 排除的材料组
        }
        limit: 返回数量限制

    Returns:
        推荐的材料列表，包含匹配分数和推荐理由
    """
    if application not in APPLICATION_MAP:
        return []

    app_config = APPLICATION_MAP[application]
    target_groups = app_config["groups"]
    requirements = requirements or {}

    results: List[Tuple[float, str, MaterialInfo, str]] = []

    for grade, info in MATERIAL_DATABASE.items():
        # 检查是否在目标材料组
        if info.group.value not in target_groups:
            continue

        # 检查排除组
        if info.group.value in requirements.get("exclude_groups", []):
            continue

        # 检查最小强度
        min_strength = requirements.get("min_strength")
        if min_strength and (info.properties.tensile_strength or 0) < min_strength:
            continue

        # 检查最大密度
        max_density = requirements.get("max_density")
        if max_density and (info.properties.density or 100) > max_density:
            continue

        # 检查可加工性
        req_machinability = requirements.get("machinability")
        if req_machinability:
            mach_order = {"excellent": 4, "good": 3, "fair": 2, "poor": 1}
            req_level = mach_order.get(req_machinability, 0)
            mat_level = mach_order.get(info.properties.machinability or "fair", 2)
            if mat_level < req_level:
                continue

        # 计算匹配分数
        score = 0.0
        reasons = []

        # 材料组匹配度（越靠前越匹配）
        group_idx = target_groups.index(info.group.value)
        group_score = 1.0 - (group_idx * 0.2)
        score += group_score * 0.3
        reasons.append(f"适合{app_config['name']}")

        # 属性评分
        props = info.properties
        if props.tensile_strength:
            if props.tensile_strength >= 800:
                score += 0.2
                reasons.append("高强度")
            elif props.tensile_strength >= 400:
                score += 0.1

        if props.machinability == "excellent":
            score += 0.15
            reasons.append("易加工")
        elif props.machinability == "good":
            score += 0.1

        if props.density and props.density < 5:
            score += 0.1
            reasons.append("轻量化")

        # 特殊工艺加分
        if info.process.special_tooling:
            score -= 0.1  # 需要特殊刀具减分

        if len(info.process.warnings) > 2:
            score -= 0.05  # 警告多减分

        results.append((score, grade, info, "；".join(reasons[:3])))

    # 按分数排序
    results.sort(key=lambda x: (-x[0], x[1]))

    # 格式化返回
    formatted = []
    for score, grade, info, reason in results[:limit]:
        formatted.append({
            "grade": grade,
            "name": info.name,
            "group": info.group.value,
            "score": round(score, 2),
            "reason": reason,
            "properties": {
                "density": info.properties.density,
                "tensile_strength": info.properties.tensile_strength,
                "machinability": info.properties.machinability,
            },
        })

    return formatted


def get_alternative_materials(
    grade: str,
    preference: str = "similar",
) -> List[Dict[str, Any]]:
    """
    获取替代材料建议

    Args:
        grade: 当前材料牌号
        preference: 替代偏好
            - "similar": 性能相近的替代
            - "cheaper": 成本更低的替代
            - "better": 性能更好的替代

    Returns:
        替代材料列表
    """
    # 先尝试获取材料信息
    info = classify_material_detailed(grade)
    if not info:
        return []

    # 使用规范牌号
    grade = info.grade

    # 获取预定义的替代关系
    predefined = MATERIAL_ALTERNATIVES.get(grade, [])

    results = []

    # 添加预定义替代
    for alt in predefined:
        alt_info = MATERIAL_DATABASE.get(alt["grade"])
        if not alt_info:
            continue

        # 根据偏好过滤
        if preference == "cheaper" and alt["cost_factor"] >= 1.0:
            continue
        if preference == "better" and alt["cost_factor"] <= 1.0:
            continue

        results.append({
            "grade": alt["grade"],
            "name": alt_info.name,
            "group": alt_info.group.value,
            "reason": alt["reason"],
            "cost_factor": alt["cost_factor"],
            "source": "predefined",
        })

    # 如果预定义不足，自动寻找同组材料
    if len(results) < 3:
        for g, alt_info in MATERIAL_DATABASE.items():
            if g == grade:
                continue
            if g in [r["grade"] for r in results]:
                continue
            if alt_info.group != info.group:
                continue

            # 简单的成本估算（基于密度和强度）
            cost_factor = 1.0
            if alt_info.properties.tensile_strength and info.properties.tensile_strength:
                strength_ratio = alt_info.properties.tensile_strength / info.properties.tensile_strength
                cost_factor *= strength_ratio ** 0.5

            # 根据偏好过滤
            if preference == "cheaper" and cost_factor >= 1.0:
                continue
            if preference == "better" and cost_factor <= 1.0:
                continue

            reason = f"同组材料（{info.group.value}）"
            if cost_factor < 1.0:
                reason = "成本较低的同组材料"
            elif cost_factor > 1.0:
                reason = "性能更好的同组材料"

            results.append({
                "grade": g,
                "name": alt_info.name,
                "group": alt_info.group.value,
                "reason": reason,
                "cost_factor": round(cost_factor, 2),
                "source": "auto",
            })

            if len(results) >= 5:
                break

    return results


def list_applications() -> List[Dict[str, str]]:
    """
    列出所有支持的用途

    Returns:
        用途列表
    """
    return [
        {"code": code, "name": config["name"]}
        for code, config in APPLICATION_MAP.items()
    ]


# ============================================================================
# 材料成本数据
# ============================================================================

# 成本等级定义
COST_TIER_DESCRIPTIONS: Dict[int, Dict[str, str]] = {
    1: {"name": "经济型", "description": "成本最低，常规应用", "range": "基准价"},
    2: {"name": "标准型", "description": "性价比高，广泛应用", "range": "1.5-3x 基准"},
    3: {"name": "中高端", "description": "特殊性能，专业应用", "range": "3-8x 基准"},
    4: {"name": "高端型", "description": "高性能材料，精密应用", "range": "8-20x 基准"},
    5: {"name": "特种型", "description": "极端性能，特殊环境", "range": "20x+ 基准"},
}

# 材料成本数据 (grade -> (cost_tier, cost_index))
# cost_tier: 1-5 等级
# cost_index: 相对成本指数，以 Q235B=1.0 为基准
MATERIAL_COST_DATA: Dict[str, Tuple[int, float]] = {
    # 碳素钢 (Tier 1-2)
    "Q235B": (1, 1.0),      # 基准材料
    "Q345R": (1, 1.2),
    "10": (1, 1.0),
    "15": (1, 1.0),
    "20": (1, 1.1),
    "35": (1, 1.2),
    "45": (1, 1.3),
    "50": (1, 1.4),

    # 合金钢 (Tier 2)
    "40Cr": (2, 1.8),
    "42CrMo": (2, 2.2),
    "GCr15": (2, 2.5),
    "GCr15SiMn": (2, 3.0),  # 高淬透性轴承钢
    "GCr18Mo": (3, 5.0),    # 航空级轴承钢
    "20Cr13": (2, 2.0),
    "2Cr13": (2, 2.0),
    "20CrMnTi": (2, 1.8),
    "20Cr": (2, 1.6),
    "38CrMoAl": (2, 2.5),
    "30CrMnSi": (2, 2.2),
    # 弹簧钢 (Tier 2)
    "65Mn": (2, 1.6),
    "60Si2Mn": (2, 1.8),
    "50CrVA": (2, 2.2),
    # 工具钢 (Tier 3)
    "Cr12MoV": (3, 6.0),
    "H13": (3, 5.5),
    "W18Cr4V": (3, 8.0),
    "W6Mo5Cr4V2": (3, 7.0),
    # 特殊钢材 (Tier 2-3)
    "9Cr18": (3, 5.0),      # 高碳马氏体不锈钢/刀具钢
    "12Cr1MoV": (2, 2.8),   # 耐热钢
    "Mn13": (2, 2.5),       # 高锰耐磨钢
    # 精密合金 (Tier 4)
    "4J36": (4, 15.0),      # 因瓦合金
    "4J29": (4, 18.0),      # 可伐合金
    "4J42": (4, 16.0),      # 恒弹性合金
    "1J79": (4, 25.0),      # 坡莫合金
    # 电工钢 (Tier 2-3)
    "50W470": (2, 2.5),     # 无取向硅钢
    "30Q130": (3, 5.0),     # 取向硅钢
    # 焊接材料 (Tier 2)
    "ER308L": (2, 4.0),     # 不锈钢焊丝
    "ER316L": (2, 5.0),     # 含钼不锈钢焊丝
    "ER70S-6": (2, 1.5),    # 碳钢焊丝
    "E7018": (1, 1.2),      # 低氢焊条
    # 复合材料 (Tier 4-5)
    "CFRP": (5, 80.0),      # 碳纤维复合材料
    "GFRP": (3, 8.0),       # 玻璃钢
    # 粉末冶金 (Tier 2)
    "FC-0208": (2, 2.0),    # 烧结铁
    "FN-0205": (2, 3.0),    # 烧结铁镍

    # 耐热钢/高温合金 (Tier 3-5)
    "310S": (3, 5.0),
    "GH3030": (4, 40.0),
    "GH4169": (5, 70.0),
    "GH4099": (5, 85.0),
    "A-286": (4, 30.0),       # 铁镍基高温合金
    "Waspaloy": (5, 85.0),    # 镍基高温合金
    "Rene41": (5, 100.0),     # 高温合金，极贵

    # 高强度结构钢 (Tier 1-2)
    "Q460": (1, 1.5),         # 高强度结构钢
    "Q550": (2, 2.0),         # 高强度结构钢
    "Q690": (2, 2.5),         # 超高强度结构钢

    # 锅炉/压力容器钢 (Tier 1-3)
    "20G": (1, 1.3),          # 锅炉用碳素钢
    "15CrMoG": (2, 2.5),      # 锅炉用合金钢
    "12Cr2Mo1R": (3, 4.0),    # 压力容器用钢

    # 管线钢 (Tier 1-2)
    "X52": (1, 1.4),          # 管线钢
    "X65": (2, 1.8),          # 高强度管线钢
    "X80": (2, 2.2),          # 超高强度管线钢

    # 模具钢补充 (Tier 2-3)
    "DC53": (3, 8.0),         # 冷作模具钢
    "S136": (3, 6.0),         # 塑料模具钢
    "NAK80": (3, 7.0),        # 预硬塑料模具钢

    # 易切削钢 (Tier 1-2)
    "12L14": (2, 1.8),        # 含铅易切削钢
    "Y15": (1, 1.5),          # 硫系易切削钢
    "Y40Mn": (2, 2.0),        # 易切削调质钢

    # 耐磨钢板 (Tier 2-3)
    "NM400": (2, 4.0),        # 耐磨钢板
    "NM500": (3, 5.5),        # 高硬度耐磨钢板
    "Hardox450": (3, 6.5),    # 悍达耐磨钢板

    # 弹簧钢补充 (Tier 2)
    "55CrSi": (2, 2.5),       # 硅铬弹簧钢

    # 低温钢 (Tier 2-4)
    "09MnNiD": (3, 4.5),      # 3.5%镍低温钢
    "16MnDR": (2, 2.8),       # 低温压力容器钢
    "9Ni钢": (4, 15.0),       # 9%镍钢，较贵

    # 电接触材料 (Tier 3-5)
    "AgCdO": (4, 25.0),       # 银氧化镉触点
    "AgSnO2": (4, 28.0),      # 银氧化锡触点，环保
    "CuW70": (4, 35.0),       # 铜钨合金

    # 轴承钢 (Tier 2-3)
    "GCr15": (2, 2.5),        # 高碳铬轴承钢
    "GCr15SiMn": (2, 3.0),    # 高碳铬硅锰轴承钢
    "GCr4": (2, 2.8),         # 渗碳轴承钢

    # 弹簧钢 (Tier 2)
    "60Si2Mn": (2, 2.0),      # 硅锰弹簧钢
    "60Si2CrA": (2, 2.5),     # 硅铬弹簧钢
    "50CrVA": (2, 3.0),       # 铬钒弹簧钢

    # 耐热不锈钢补充 (Tier 2-3)
    "2Cr13": (2, 2.8),        # 马氏体不锈钢
    "1Cr17": (2, 2.5),        # 铁素体不锈钢
    "0Cr25Ni20": (3, 5.0),    # 耐热不锈钢

    # 齿轮钢 (Tier 2)
    "20CrMnTi": (2, 2.2),     # 渗碳齿轮钢
    "20CrMo": (2, 2.5),       # 铬钼渗碳钢
    "20CrNiMo": (2, 3.5),     # 铬镍钼渗碳钢

    # 航空铝合金补充 (Tier 2-3)
    "5A06": (2, 2.2),         # 防锈铝合金
    "2A14": (3, 4.0),         # 高强铝合金
    "7A04": (3, 5.0),         # 超高强铝合金

    # 气门钢 (Tier 3-4)
    "4Cr10Si2Mo": (3, 8.0),   # 马氏体耐热钢
    "5Cr21Mn9Ni4N": (4, 15.0),  # 奥氏体耐热钢
    "4Cr14Ni14W2Mo": (4, 25.0),  # 高温气门钢

    # 链条钢/渗碳硼钢 (Tier 2)
    "20MnVB": (2, 2.0),       # 渗碳硼钢
    "15MnVB": (2, 1.8),       # 低碳硼钢
    "22MnB5": (2, 2.5),       # 热成形钢

    # 电工硅钢补充 (Tier 2-3)
    "B50A600": (2, 2.5),      # 无取向硅钢
    "B35A230": (3, 4.5),      # 高效无取向硅钢
    "B27R090": (3, 6.0),      # 取向硅钢

    # 高温合金补充 (Tier 5 - 极高成本)
    "GH2132": (4, 45.0),      # 铁基高温合金
    "K403": (5, 120.0),       # 镍基铸造高温合金
    "K418": (5, 150.0),       # 高性能铸造高温合金

    # 不锈钢 (Tier 2-3)
    "S30408": (2, 3.5),     # 304
    "S30403": (2, 3.8),     # 304L
    "S31603": (3, 4.5),     # 316L
    "S31608": (3, 4.2),     # 316
    "321": (2, 4.0),        # 321
    "347": (3, 4.5),        # 347
    "430": (2, 2.5),        # 430 铁素体
    "410": (2, 2.8),        # 410 马氏体
    "17-4PH": (3, 8.0),     # 17-4PH 沉淀硬化
    "2205": (3, 6.0),       # 双相
    "2507": (3, 8.0),       # 超级双相
    # 超级奥氏体不锈钢 (Tier 4)
    "904L": (4, 15.0),      # 904L 超级奥氏体
    "254SMO": (4, 18.0),    # 254SMO 6Mo超级奥氏体
    "316Ti": (3, 5.5),      # 316Ti 钛稳定

    # 铸铁 (Tier 1)
    "HT200": (1, 0.8),
    "HT250": (1, 0.9),
    "HT300": (1, 1.0),
    "QT400": (1, 1.0),
    "QT500-7": (1, 1.2),
    "QT600-3": (1, 1.4),
    "QT700-2": (1, 1.6),

    # 铝合金 (Tier 2)
    "6061": (2, 2.8),
    "7075": (2, 4.0),
    "2024": (2, 3.8),
    "5052": (2, 2.5),
    "5083": (2, 3.2),
    "2A12": (2, 3.5),
    "6063": (2, 2.2),
    # 铸造铝合金 (Tier 2)
    "A356": (2, 3.0),       # 铸造铝硅镁合金
    "ZL102": (2, 2.5),      # 铸造铝硅合金
    "ADC12": (2, 2.8),      # 压铸铝合金
    # 铸造铝合金补充 (Tier 2-3)
    "ZL101": (2, 3.2),      # 高纯铸造铝合金
    "ZL104": (2, 2.8),      # 铸造铝硅铜合金
    "ZL201": (3, 5.0),      # 高强度铸造铝铜合金

    # 锌合金 (Tier 1-2)
    "Zamak3": (1, 1.5),     # 通用锌合金
    "Zamak5": (1, 1.8),     # 高强度锌合金
    "ZA-8": (2, 2.2),       # 高铝锌合金

    # 铜合金 (Tier 2-3)
    "H62": (2, 5.0),
    "H68": (2, 5.5),
    "HPb59-1": (2, 5.2),
    "Cu65": (3, 7.0),       # 紫铜
    "QBe2": (4, 25.0),      # 铍铜，贵
    "QAl9-4": (3, 8.0),
    "QAl10-3-1.5": (3, 9.0),
    "QSn4-3": (3, 7.5),
    "QSn6.5-0.1": (3, 8.5),
    "CuNi10Fe1Mn": (3, 10.0),  # 白铜
    # 铜合金补充 (Tier 2-3)
    "QSn7-0.2": (3, 9.0),      # 高锡磷青铜
    "QSn4-0.3": (2, 7.0),      # 低锡磷青铜
    "ZCuSn10P1": (3, 8.0),     # 铸造锡磷青铜
    "HSi80-3": (2, 6.0),       # 硅黄铜
    "HAl77-2": (2, 6.5),       # 铝黄铜

    # 耐磨铸铁 (Tier 2-3)
    "NiHard1": (2, 3.5),       # 镍硬铸铁1型
    "NiHard4": (3, 4.5),       # 镍硬铸铁4型
    "Cr26": (3, 5.0),          # 高铬铸铁

    # 蠕墨铸铁 (Tier 1-2)
    "RuT300": (1, 1.2),        # 蠕墨铸铁
    "RuT350": (2, 1.5),        # 蠕墨铸铁
    "RuT400": (2, 1.8),        # 高强度蠕墨铸铁

    # 可锻铸铁 (Tier 1-2)
    "KTH300-06": (1, 1.3),     # 黑心可锻铸铁
    "KTZ450-06": (2, 1.6),     # 珠光体可锻铸铁
    "KTZ550-04": (2, 2.0),     # 高强度可锻铸铁

    # 铸造镁合金 (Tier 2-3)
    "ZM5": (2, 4.0),           # 压铸镁合金
    "AM60B": (2, 4.5),         # 高韧镁合金
    "AZ63": (3, 5.5),          # 航空镁合金

    # 粉末冶金材料 (Tier 2-3)
    "Fe-Cu-C": (2, 2.5),       # 铁铜碳粉末冶金
    "Fe-Ni-Cu": (2, 3.5),      # 铁镍铜粉末冶金
    "316L-PM": (3, 8.0),       # 不锈钢粉末冶金

    # 硬质合金 (Tier 4-5)
    "YW1": (4, 58.0),          # 通用型硬质合金

    # 结构陶瓷 (Tier 3-4)
    "Al2O3-99": (3, 15.0),     # 99氧化铝陶瓷
    "Si3N4": (4, 80.0),        # 氮化硅陶瓷
    "ZrO2-3Y": (4, 60.0),      # 氧化锆陶瓷

    # 难熔金属 (Tier 4-5)
    "Mo-1": (4, 40.0),         # 纯钼
    "TZM": (5, 80.0),          # 钛锆钼合金
    "W-1": (5, 60.0),          # 纯钨
    "Ta-1": (5, 200.0),        # 纯钽

    # 铝青铜 (Tier 2-3)
    "QAl9-4": (2, 5.0),        # 铝青铜
    "QAl10-4-4": (3, 8.0),     # 镍铝青铜

    # 铍铜 (Tier 3-4)
    "QBe2": (4, 25.0),         # 铍青铜
    "QBe1.9": (4, 22.0),       # 低铍铜
    "CuNi2Si": (3, 12.0),      # 镍硅铜

    # 无铅焊锡 (Tier 2)
    "SAC305": (2, 3.5),        # 无铅焊锡
    "SAC387": (2, 4.5),        # 高银无铅焊锡
    "Sn99.3Cu0.7": (2, 2.5),   # 无银无铅焊锡

    # 钎焊合金 (Tier 3-4)
    "BAg-5": (4, 35.0),        # 银钎料
    "BCu-1": (2, 6.0),         # 纯铜钎料
    "BNi-2": (3, 20.0),        # 镍基钎料

    # 形状记忆合金 (Tier 4-5)
    "NiTi": (5, 150.0),        # 镍钛记忆合金
    "CuZnAl": (3, 15.0),       # 铜锌铝记忆合金
    "CuAlNi": (3, 18.0),       # 铜铝镍记忆合金

    # 电触头材料 (Tier 3-4)
    "AgCdO": (4, 40.0),        # 银氧化镉触点
    "AgSnO2": (4, 45.0),       # 银氧化锡触点
    "CuW": (4, 50.0),          # 钨铜合金

    # 轴承合金 (Tier 2-3)
    "ZChSnSb11-6": (3, 12.0),  # 锡基巴氏合金
    "ZChPbSb16-16-2": (2, 5.0), # 铅基巴氏合金
    "CuPb24Sn4": (2, 8.0),     # 铜铅合金轴承

    # 热电偶合金 (Tier 3-4)
    "Chromel": (3, 20.0),      # 镍铬合金
    "Alumel": (3, 18.0),       # 镍铝合金
    "Constantan": (3, 15.0),   # 康铜

    # 永磁材料 (Tier 4-5)
    "NdFeB": (4, 80.0),        # 钕铁硼永磁
    "SmCo": (5, 200.0),        # 钐钴永磁
    "Alnico": (3, 30.0),       # 铝镍钴永磁

    # 电阻合金 (Tier 2-3)
    "Cr20Ni80": (3, 15.0),     # 镍铬电热合金
    "Manganin": (3, 25.0),     # 锰铜合金
    "Karma": (4, 40.0),        # 卡玛合金

    # 低膨胀合金 (Tier 3-4)
    "Invar": (3, 25.0),        # 因瓦合金
    "Kovar": (4, 35.0),        # 可伐合金
    "4J32": (4, 45.0),         # 超因瓦合金

    # 超导材料 (Tier 5)
    "NbTi": (5, 150.0),        # 铌钛超导合金
    "Nb3Sn": (5, 250.0),       # 铌三锡超导体
    "YBCO": (5, 500.0),        # 钇钡铜氧高温超导

    # 核工业材料 (Tier 4-5)
    "Zircaloy-4": (4, 80.0),   # 锆合金-4
    "Hafnium": (5, 800.0),     # 铪（稀有金属）
    "B4C": (4, 60.0),          # 碳化硼

    # 医用合金 (Tier 4-5)
    "CoCrMo": (4, 55.0),       # 钴铬钼医用合金
    "Ti6Al4V-ELI": (5, 85.0),  # 医用钛合金
    "316L-Medical": (4, 25.0), # 医用不锈钢

    # 光学材料 (Tier 4-5)
    "Fused-Silica": (4, 50.0),      # 熔融石英
    "Sapphire": (5, 200.0),         # 蓝宝石
    "Germanium": (5, 350.0),        # 锗

    # 电池材料 (Tier 3-4)
    "LiFePO4": (3, 15.0),           # 磷酸铁锂
    "NMC": (4, 25.0),               # 三元正极材料
    "Graphite-Battery": (3, 8.0),  # 电池负极石墨

    # 半导体材料 (Tier 4-5)
    "Silicon-Wafer": (4, 50.0),       # 硅晶圆
    "GaAs": (5, 300.0),               # 砷化镓
    "SiC-Semiconductor": (5, 500.0),  # 碳化硅半导体

    # 热界面材料 (Tier 2-4)
    "Thermal-Paste": (2, 5.0),        # 导热硅脂
    "Thermal-Pad": (3, 15.0),         # 导热垫片
    "Graphene-TIM": (4, 80.0),        # 石墨烯导热材料

    # 增材制造材料 (Tier 4-5)
    "AlSi10Mg-AM": (4, 60.0),         # 3D打印铝合金
    "IN718-AM": (5, 150.0),           # 3D打印镍基高温合金
    "Ti64-AM": (5, 200.0),            # 3D打印钛合金

    # 硬质合金 (Tier 4-5)
    "WC-Co": (4, 80.0),               # 钨钴硬质合金
    "Stellite": (4, 65.0),            # 司太立合金
    "CBN": (5, 500.0),                # 立方氮化硼

    # 热障涂层材料 (Tier 4-5)
    "YSZ": (4, 45.0),                 # 氧化钇稳定氧化锆
    "Al2O3-TBC": (3, 25.0),           # 氧化铝热障涂层
    "MCrAlY": (4, 70.0),              # 金属粘结层

    # 电磁屏蔽材料 (Tier 3-4)
    "Mu-Metal": (4, 50.0),            # 坡莫合金
    "Permalloy": (3, 35.0),           # 铁镍软磁合金
    "Copper-Mesh": (3, 20.0),         # 铜网屏蔽材料

    # 钛合金 (Tier 4)
    "TA1": (4, 12.0),
    "TA2": (4, 15.0),
    "TC4": (4, 25.0),
    "TC11": (4, 35.0),
    "TB6": (4, 40.0),
    "TC21": (4, 38.0),

    # 镁合金 (Tier 2-3)
    "AZ31B": (2, 3.0),
    "AZ91D": (2, 2.5),
    "ZK60": (3, 5.0),

    # 硬质合金 (Tier 4)
    "YG8": (4, 50.0),
    "YT15": (4, 55.0),
    "YG6": (4, 52.0),

    # 耐蚀合金 (Tier 4-5)
    "C276": (5, 80.0),
    "C22": (5, 75.0),
    "Inconel625": (5, 60.0),
    "Inconel718": (5, 70.0),
    "Monel400": (4, 35.0),
    "MonelK500": (4, 45.0),
    "HastelloyB3": (5, 90.0),
    "Stellite6": (5, 100.0),
    "Incoloy825": (4, 40.0),

    # 工程塑料 (Tier 1-3)
    "PTFE": (2, 3.0),
    "RPTFE": (2, 3.5),
    "PEEK": (4, 50.0),      # PEEK 很贵
    "POM": (2, 2.5),
    "PA66": (2, 2.0),
    "PC": (2, 2.2),
    "UHMWPE": (2, 3.0),
    # 高性能工程塑料 (Tier 3-5)
    "PPS": (3, 8.0),
    "PI": (5, 80.0),        # 聚酰亚胺很贵
    "PSU": (3, 12.0),
    "PEI": (4, 25.0),

    # 橡胶 (Tier 1-2)
    "EPDM": (1, 1.5),
    "VMQ": (2, 4.0),

    # 聚氨酯 (Tier 2)
    "聚氨酯": (2, 3.0),

    # 玻璃 (Tier 2)
    "硼硅玻璃": (2, 2.5),
    "钢化玻璃": (2, 1.8),

    # 陶瓷/纤维 (Tier 2)
    "硅酸铝": (2, 2.0),

    # 新增材料成本
    # 不锈钢
    "631": (3, 8.0),           # 17-7PH沉淀硬化
    "15-5PH": (3, 7.0),        # 15-5PH沉淀硬化
    "Nitronic50": (3, 10.0),   # 高氮奥氏体
    # 铝合金
    "3003": (1, 1.2),          # 防锈铝
    "1060": (1, 1.0),          # 工业纯铝
    "LY12": (2, 2.0),          # 硬铝
    "6005": (2, 1.8),          # 轨道交通铝合金
    "6101": (2, 1.6),          # 导电铝合金
    # 工具钢
    "D2": (3, 5.5),            # SKD11
    "O1": (2, 3.5),            # 油淬工具钢
    "A2": (3, 5.0),            # 空淬工具钢
    # 高温合金
    "Haynes230": (5, 120.0),   # 镍基高温合金
    "HastelloyX": (5, 100.0),  # 哈氏合金X
    "MP35N": (5, 200.0),       # 超高强度医用合金
    "L605": (5, 150.0),        # 钴基高温合金
    "Waspaloy": (5, 130.0),    # 镍基时效高温合金
    "Rene41": (5, 140.0),      # 镍基高温合金
    "Elgiloy": (5, 180.0),     # 钴铬镍钼合金
    "Phynox": (5, 190.0),      # 医用钴基合金
    # 钛合金
    "Ti-6242": (4, 60.0),      # 近α型钛合金
    # 铜合金
    "C11000": (2, 3.0),        # 电解铜
    "C26000": (2, 2.5),        # 黄铜
    "C52100": (3, 8.0),        # 磷青铜
    # 镁合金
    "WE43": (4, 40.0),         # 稀土镁合金
    "AM60": (2, 3.5),          # 压铸镁合金
    # 不锈钢
    "13-8Mo": (4, 20.0),       # 沉淀硬化不锈钢
    "22Cr双相钢": (3, 8.0),    # 双相不锈钢
    # 工程塑料
    "PVDF": (3, 15.0),         # 聚偏氟乙烯
    "LCP": (4, 25.0),          # 液晶聚合物
    "PPSU": (4, 22.0),         # 聚苯砜
    # 橡胶
    "NBR": (1, 1.5),           # 丁腈橡胶
    "FKM": (3, 12.0),          # 氟橡胶

    # 组合件 - 无固定成本
    # "组焊件" 和 "组合件" 成本取决于组成部件，不列入固定成本表
}


def get_material_cost(grade: str) -> Optional[Dict[str, Any]]:
    """
    获取材料成本信息

    Args:
        grade: 材料牌号

    Returns:
        成本信息字典，包含:
        - tier: 成本等级 (1-5)
        - tier_name: 等级名称
        - tier_description: 等级描述
        - cost_index: 相对成本指数
        - price_range: 价格区间描述
    """
    # 先尝试获取材料信息
    info = classify_material_detailed(grade)
    if not info:
        return None

    # 使用规范牌号
    grade = info.grade

    # 查找成本数据
    cost_data = MATERIAL_COST_DATA.get(grade)
    if not cost_data:
        # 尝试根据材料组估算
        group_defaults = {
            MaterialGroup.CARBON_STEEL: (1, 1.2),
            MaterialGroup.ALLOY_STEEL: (2, 2.0),
            MaterialGroup.STAINLESS_STEEL: (2, 4.0),
            MaterialGroup.CORROSION_RESISTANT: (4, 50.0),
            MaterialGroup.CAST_IRON: (1, 0.9),
            MaterialGroup.ALUMINUM: (2, 3.0),
            MaterialGroup.COPPER: (2, 6.0),
            MaterialGroup.TITANIUM: (4, 20.0),
            MaterialGroup.NICKEL: (4, 40.0),
            MaterialGroup.ENGINEERING_PLASTIC: (2, 3.0),
            MaterialGroup.FLUOROPOLYMER: (2, 3.5),
            MaterialGroup.RUBBER: (1, 2.0),
        }
        cost_data = group_defaults.get(info.group, (2, 5.0))

    tier, cost_index = cost_data
    tier_info = COST_TIER_DESCRIPTIONS.get(tier, {})

    return {
        "grade": grade,
        "name": info.name,
        "tier": tier,
        "tier_name": tier_info.get("name", "未知"),
        "tier_description": tier_info.get("description", ""),
        "cost_index": cost_index,
        "price_range": tier_info.get("range", ""),
        "group": info.group.value,
    }


def compare_material_costs(
    grades: List[str],
    include_missing: bool = False,
) -> Any:
    """
    比较多个材料的成本

    Args:
        grades: 材料牌号列表
        include_missing: 是否返回未命中的材料列表

    Returns:
        成本比较结果，按成本指数排序；
        include_missing=True 时返回 (results, missing)
    """
    results = []
    missing = []

    for grade in grades:
        cost_info = get_material_cost(grade)
        if cost_info:
            results.append(cost_info)
        else:
            missing.append(grade)

    # 按成本指数排序
    results.sort(key=lambda x: float(x.get("cost_index") or 0.0))

    # 添加相对比较
    if results:
        min_cost = results[0]["cost_index"]
        for r in results:
            r["relative_to_cheapest"] = round(r["cost_index"] / min_cost, 2)

    if include_missing:
        return results, missing
    return results


def search_by_cost(
    max_tier: Optional[int] = None,
    max_cost_index: Optional[float] = None,
    category: Optional[str] = None,
    group: Optional[str] = None,
    limit: int = 20,
    include_estimated: bool = False,
) -> List[Dict[str, Any]]:
    """
    按成本筛选材料

    Args:
        max_tier: 最大成本等级 (1-5)
        max_cost_index: 最大成本指数
        category: 限定材料类别
        group: 限定材料组
        limit: 返回数量限制

    Returns:
        符合条件的材料列表
    """
    results = []

    for grade, info in MATERIAL_DATABASE.items():
        # 类别过滤
        if category and info.category.value != category:
            continue
        if group and info.group.value != group:
            continue

        tier = None
        cost_index = None
        tier_name = "未知"

        if include_estimated:
            cost_info = get_material_cost(grade)
            if not cost_info:
                continue
            tier = cost_info["tier"]
            cost_index = cost_info["cost_index"]
            tier_name = cost_info["tier_name"]
        else:
            cost_data = MATERIAL_COST_DATA.get(grade)
            if not cost_data:
                continue
            tier, cost_index = cost_data
            tier_info = COST_TIER_DESCRIPTIONS.get(tier, {})
            tier_name = tier_info.get("name", "未知")

        # 成本等级过滤
        if max_tier and tier and tier > max_tier:
            continue

        # 成本指数过滤
        if max_cost_index and cost_index and cost_index > max_cost_index:
            continue

        results.append({
            "grade": grade,
            "name": info.name,
            "category": info.category.value,
            "group": info.group.value,
            "tier": tier,
            "tier_name": tier_name,
            "cost_index": cost_index,
        })

    # 按成本指数排序
    results.sort(key=lambda x: float(x.get("cost_index") or 0.0))

    return results[:limit]


def get_cost_tier_info() -> List[Dict[str, Any]]:
    """
    获取成本等级定义

    Returns:
        成本等级列表
    """
    return [
        {
            "tier": tier,
            "name": info["name"],
            "description": info["description"],
            "price_range": info["range"],
        }
        for tier, info in COST_TIER_DESCRIPTIONS.items()
    ]


# ============================================================================
# 材料兼容性检查
# ============================================================================

# 焊接兼容性矩阵
# 等级: "excellent", "good", "fair", "poor", "not_recommended"
WELD_COMPATIBILITY: Dict[str, Dict[str, Dict[str, Any]]] = {
    # 碳钢焊接
    "carbon_steel": {
        "carbon_steel": {"rating": "excellent", "method": "普通焊条/CO2保护焊", "notes": "同组材料焊接性好"},
        "alloy_steel": {"rating": "good", "method": "低氢焊条", "notes": "需预热"},
        "stainless_steel": {"rating": "fair", "method": "309L/309Mo焊材", "notes": "异种钢焊接，需选择合适焊材"},
        "cast_iron": {"rating": "fair", "method": "镍基焊条", "notes": "需预热和缓冷"},
        "aluminum": {"rating": "not_recommended", "method": "-", "notes": "不建议直接焊接"},
        "copper": {"rating": "poor", "method": "铜镍焊材", "notes": "困难，不推荐"},
        "titanium": {"rating": "not_recommended", "method": "-", "notes": "不可焊接"},
    },
    # 不锈钢焊接
    "stainless_steel": {
        "stainless_steel": {"rating": "excellent", "method": "同材质焊材/TIG", "notes": "注意防止敏化"},
        "carbon_steel": {"rating": "fair", "method": "309L焊材", "notes": "异种钢焊接"},
        "alloy_steel": {"rating": "fair", "method": "309L/309Mo焊材", "notes": "需预热"},
        "corrosion_resistant": {"rating": "good", "method": "镍基焊材", "notes": "选择合适的镍基焊材"},
        "titanium": {"rating": "not_recommended", "method": "-", "notes": "不可焊接"},
        "aluminum": {"rating": "not_recommended", "method": "-", "notes": "不建议直接焊接"},
    },
    # 铝合金焊接
    "aluminum": {
        "aluminum": {"rating": "excellent", "method": "TIG/MIG-4043/5356", "notes": "需清洁表面氧化膜"},
        "carbon_steel": {"rating": "not_recommended", "method": "-", "notes": "不建议直接焊接"},
        "stainless_steel": {"rating": "not_recommended", "method": "-", "notes": "不建议直接焊接"},
        "copper": {"rating": "poor", "method": "钎焊", "notes": "只能钎焊"},
    },
    # 铜合金焊接
    "copper": {
        "copper": {"rating": "good", "method": "TIG/氧乙炔", "notes": "需预热，导热快"},
        "carbon_steel": {"rating": "poor", "method": "铜镍焊材", "notes": "困难"},
        "stainless_steel": {"rating": "fair", "method": "镍基焊材", "notes": "可行但困难"},
        "aluminum": {"rating": "poor", "method": "钎焊", "notes": "只能钎焊"},
    },
    # 钛合金焊接
    "titanium": {
        "titanium": {"rating": "good", "method": "TIG/真空电子束", "notes": "需惰性气体全保护"},
        "stainless_steel": {"rating": "not_recommended", "method": "-", "notes": "形成脆性金属间化合物"},
        "carbon_steel": {"rating": "not_recommended", "method": "-", "notes": "不可焊接"},
        "aluminum": {"rating": "not_recommended", "method": "-", "notes": "不可焊接"},
    },
    # 耐蚀合金焊接
    "corrosion_resistant": {
        "corrosion_resistant": {"rating": "good", "method": "同材质焊材/TIG", "notes": "需选择匹配焊材"},
        "stainless_steel": {"rating": "good", "method": "镍基焊材", "notes": "ERNiCrMo-3等"},
        "carbon_steel": {"rating": "fair", "method": "镍基焊材", "notes": "需预热"},
    },
    # 镁合金焊接
    "magnesium": {
        "magnesium": {"rating": "good", "method": "TIG/MIG", "notes": "需惰性气体保护，注意防火"},
        "aluminum": {"rating": "poor", "method": "钎焊/搅拌摩擦", "notes": "困难，需特殊工艺"},
        "carbon_steel": {"rating": "not_recommended", "method": "-", "notes": "不可焊接"},
        "stainless_steel": {"rating": "not_recommended", "method": "-", "notes": "不可焊接"},
    },
    # 硬质合金焊接
    "cemented_carbide": {
        "cemented_carbide": {"rating": "poor", "method": "钎焊/扩散焊", "notes": "只能钎焊"},
        "carbon_steel": {"rating": "fair", "method": "银基钎料钎焊", "notes": "用于刀具镶嵌"},
        "alloy_steel": {"rating": "fair", "method": "银基钎料钎焊", "notes": "用于刀具/模具"},
    },
    # 工具钢焊接
    "tool_steel": {
        "tool_steel": {"rating": "poor", "method": "TIG/预热+缓冷", "notes": "容易开裂，需严格控温"},
        "carbon_steel": {"rating": "fair", "method": "低氢焊条", "notes": "需预热300℃以上"},
        "alloy_steel": {"rating": "fair", "method": "低氢焊条", "notes": "需预热300℃以上"},
    },
}

# 电偶腐蚀序列 (galvanic series in seawater)
# 数值越小越活泼（阳极），越大越惰性（阴极）
GALVANIC_SERIES: Dict[str, float] = {
    # 活泼端 (阳极)
    "magnesium": -1.6,
    "zinc": -1.0,
    "aluminum": -0.8,
    "carbon_steel": -0.6,
    "cast_iron": -0.5,
    "alloy_steel": -0.5,
    "tool_steel": -0.45,  # 工具钢略惰性于普通合金钢
    "stainless_steel_active": -0.4,  # 活化态不锈钢
    "copper": -0.2,
    "stainless_steel": 0.0,  # 钝化态不锈钢
    "titanium": 0.1,
    "corrosion_resistant": 0.15,  # 镍基合金
    "cemented_carbide": 0.2,  # 硬质合金（惰性）
    # 惰性端 (阴极)
}

# 电偶腐蚀风险阈值
GALVANIC_RISK_THRESHOLDS = {
    "safe": 0.15,      # 电位差 < 0.15V: 安全
    "low": 0.25,       # 电位差 0.15-0.25V: 低风险
    "medium": 0.4,     # 电位差 0.25-0.4V: 中风险
    "high": 0.6,       # 电位差 0.4-0.6V: 高风险
    # > 0.6V: 严重风险
}


def check_weld_compatibility(
    material1: str,
    material2: str,
) -> Dict[str, Any]:
    """
    检查两种材料的焊接兼容性

    Args:
        material1: 第一种材料牌号
        material2: 第二种材料牌号

    Returns:
        焊接兼容性信息
    """
    # 获取材料信息
    info1 = classify_material_detailed(material1)
    info2 = classify_material_detailed(material2)

    if not info1 or not info2:
        return {
            "compatible": False,
            "error": "材料未找到",
            "material1": material1,
            "material2": material2,
        }

    group1 = info1.group.value
    group2 = info2.group.value

    # 查找兼容性数据
    compat_data = None

    # 先查 group1 -> group2
    if group1 in WELD_COMPATIBILITY:
        if group2 in WELD_COMPATIBILITY[group1]:
            compat_data = WELD_COMPATIBILITY[group1][group2]

    # 再查 group2 -> group1 (对称)
    if not compat_data and group2 in WELD_COMPATIBILITY:
        if group1 in WELD_COMPATIBILITY[group2]:
            compat_data = WELD_COMPATIBILITY[group2][group1]

    # 如果没有数据，返回默认
    if not compat_data:
        # 同组默认可焊
        if group1 == group2:
            compat_data = {
                "rating": "good",
                "method": "同材质焊材",
                "notes": "同组材料，通常可焊",
            }
        else:
            compat_data = {
                "rating": "unknown",
                "method": "需工艺评定",
                "notes": "无现成数据，建议工艺评定",
            }

    rating = compat_data["rating"]
    compatible = rating in ["excellent", "good", "fair"]

    return {
        "compatible": compatible,
        "rating": rating,
        "rating_cn": {
            "excellent": "优秀",
            "good": "良好",
            "fair": "一般",
            "poor": "困难",
            "not_recommended": "不推荐",
            "unknown": "未知",
        }.get(rating, rating),
        "method": compat_data.get("method", ""),
        "notes": compat_data.get("notes", ""),
        "material1": {
            "grade": info1.grade,
            "name": info1.name,
            "group": group1,
        },
        "material2": {
            "grade": info2.grade,
            "name": info2.name,
            "group": group2,
        },
    }


def check_galvanic_corrosion(
    material1: str,
    material2: str,
) -> Dict[str, Any]:
    """
    检查两种材料的电偶腐蚀风险

    Args:
        material1: 第一种材料牌号
        material2: 第二种材料牌号

    Returns:
        电偶腐蚀风险信息
    """
    # 获取材料信息
    info1 = classify_material_detailed(material1)
    info2 = classify_material_detailed(material2)

    if not info1 or not info2:
        return {
            "risk": "unknown",
            "error": "材料未找到",
        }

    group1 = info1.group.value
    group2 = info2.group.value

    # 非金属不参与电偶腐蚀
    if info1.category != MaterialCategory.METAL or info2.category != MaterialCategory.METAL:
        return {
            "risk": "none",
            "risk_cn": "无",
            "notes": "非金属材料不参与电偶腐蚀",
            "material1": {"grade": info1.grade, "name": info1.name},
            "material2": {"grade": info2.grade, "name": info2.name},
        }

    # 获取电偶序列位置
    potential1 = GALVANIC_SERIES.get(group1)
    potential2 = GALVANIC_SERIES.get(group2)

    if potential1 is None or potential2 is None:
        return {
            "risk": "unknown",
            "risk_cn": "未知",
            "notes": "缺少电偶序列数据",
            "material1": {"grade": info1.grade, "name": info1.name, "group": group1},
            "material2": {"grade": info2.grade, "name": info2.name, "group": group2},
        }

    # 计算电位差
    potential_diff = abs(potential1 - potential2)

    # 判断阳极/阴极
    if potential1 < potential2:
        anode = info1
        cathode = info2
    else:
        anode = info2
        cathode = info1

    # 评估风险
    if potential_diff < GALVANIC_RISK_THRESHOLDS["safe"]:
        risk = "safe"
        risk_cn = "安全"
        recommendation = "可直接接触使用"
    elif potential_diff < GALVANIC_RISK_THRESHOLDS["low"]:
        risk = "low"
        risk_cn = "低风险"
        recommendation = "干燥环境可用，潮湿环境需注意"
    elif potential_diff < GALVANIC_RISK_THRESHOLDS["medium"]:
        risk = "moderate"
        risk_cn = "中风险"
        recommendation = "建议绝缘隔离或表面处理"
    elif potential_diff < GALVANIC_RISK_THRESHOLDS["high"]:
        risk = "high"
        risk_cn = "高风险"
        recommendation = "必须绝缘隔离，避免电解质环境"
    else:
        risk = "severe"
        risk_cn = "严重"
        recommendation = "禁止直接接触，必须完全隔离"

    return {
        "risk": risk,
        "risk_cn": risk_cn,
        "potential_difference": round(potential_diff, 2),
        "recommendation": recommendation,
        "anode": {
            "grade": anode.grade,
            "name": anode.name,
            "role": "阳极（被腐蚀）",
        },
        "cathode": {
            "grade": cathode.grade,
            "name": cathode.name,
            "role": "阴极（受保护）",
        },
        "material1": {"grade": info1.grade, "name": info1.name, "group": group1},
        "material2": {"grade": info2.grade, "name": info2.name, "group": group2},
    }


def check_heat_treatment_compatibility(
    material: str,
    treatment: str,
) -> Dict[str, Any]:
    """
    检查材料与热处理工艺的兼容性

    Args:
        material: 材料牌号
        treatment: 热处理工艺名称

    Returns:
        兼容性信息
    """
    info = classify_material_detailed(material)

    if not info:
        return {
            "compatible": False,
            "error": "材料未找到",
        }

    # 检查推荐热处理
    recommended = info.process.heat_treatments
    forbidden = info.process.forbidden_heat_treatments

    # 标准化处理名称
    treatment_normalized = treatment.strip()

    # 检查是否被禁止
    for f in forbidden:
        if treatment_normalized in f or f in treatment_normalized:
            return {
                "compatible": False,
                "status": "forbidden",
                "status_cn": "禁止",
                "grade": info.grade,
                "name": info.name,
                "treatment": treatment,
                "reason": f"该材料禁止进行{f}处理",
                "recommended_treatments": recommended,
            }

    # 检查是否推荐
    for r in recommended:
        if treatment_normalized in r or r in treatment_normalized:
            return {
                "compatible": True,
                "status": "recommended",
                "status_cn": "推荐",
                "grade": info.grade,
                "name": info.name,
                "treatment": treatment,
                "reason": f"该材料推荐进行{r}处理",
                "recommended_treatments": recommended,
            }

    # 不在推荐列表也不在禁止列表
    return {
        "compatible": True,
        "status": "allowed",
        "status_cn": "可行",
        "grade": info.grade,
        "name": info.name,
        "treatment": treatment,
        "reason": "该热处理不在推荐/禁止列表中，需工艺验证",
        "recommended_treatments": recommended,
        "forbidden_treatments": forbidden,
    }


def check_full_compatibility(
    material1: str,
    material2: str,
) -> Dict[str, Any]:
    """
    全面检查两种材料的兼容性

    Args:
        material1: 第一种材料牌号
        material2: 第二种材料牌号

    Returns:
        完整兼容性报告
    """
    weld = check_weld_compatibility(material1, material2)
    galvanic = check_galvanic_corrosion(material1, material2)

    # 综合评估
    issues = []
    recommendations = []

    # 焊接问题
    if not weld.get("compatible", False):
        issues.append(f"焊接兼容性差: {weld.get('rating_cn', '未知')}")
        if weld.get("notes"):
            recommendations.append(f"焊接: {weld['notes']}")

    # 电偶腐蚀问题
    galvanic_risk = galvanic.get("risk", "unknown")
    if galvanic_risk in ["medium", "high", "severe"]:
        issues.append(f"电偶腐蚀风险: {galvanic.get('risk_cn', '未知')}")
        if galvanic.get("recommendation"):
            recommendations.append(f"防腐蚀: {galvanic['recommendation']}")

    # 总体评估
    if not issues:
        overall = "compatible"
        overall_cn = "兼容"
    elif len(issues) == 1:
        overall = "caution"
        overall_cn = "需注意"
    else:
        overall = "incompatible"
        overall_cn = "不兼容"

    return {
        "overall": overall,
        "overall_cn": overall_cn,
        "issues": issues,
        "recommendations": recommendations,
        "weld_compatibility": weld,
        "galvanic_corrosion": galvanic,
    }
