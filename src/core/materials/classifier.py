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

    # 组合件
    ASSEMBLY = "assembly"  # 组合件


class MaterialGroup(str, Enum):
    """材料组"""
    # 钢铁类
    CARBON_STEEL = "carbon_steel"  # 碳素钢
    ALLOY_STEEL = "alloy_steel"  # 合金钢
    STAINLESS_STEEL = "stainless_steel"  # 不锈钢
    CORROSION_RESISTANT = "corrosion_resistant"  # 耐蚀合金
    CAST_IRON = "cast_iron"  # 铸铁

    # 有色金属
    ALUMINUM = "aluminum"  # 铝合金
    COPPER = "copper"  # 铜合金
    TITANIUM = "titanium"  # 钛合金
    NICKEL = "nickel"  # 镍合金

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

    # 机械属性
    tensile_strength: Optional[float] = None  # 抗拉强度 MPa
    yield_strength: Optional[float] = None  # 屈服强度 MPa
    hardness: Optional[str] = None  # 硬度 (如 HB200, HRC45)
    elongation: Optional[float] = None  # 延伸率 %

    # 加工属性
    machinability: Optional[str] = None  # 可加工性 (excellent/good/fair/poor)
    weldability: Optional[str] = None  # 可焊性 (excellent/good/fair/poor)


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
            tensile_strength=370,
            yield_strength=235,
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
            tensile_strength=510,
            yield_strength=345,
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
            tensile_strength=600,
            yield_strength=355,
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
            tensile_strength=410,
            yield_strength=245,
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
            tensile_strength=980,
            yield_strength=785,
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
            tensile_strength=1080,
            yield_strength=930,
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
            tensile_strength=520,
            yield_strength=205,
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
            tensile_strength=480,
            yield_strength=170,
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
            tensile_strength=640,
            yield_strength=440,
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
            tensile_strength=690,
            yield_strength=490,
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
            tensile_strength=690,
            yield_strength=310,
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
            tensile_strength=690,
            yield_strength=310,
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
            tensile_strength=830,
            yield_strength=415,
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
            tensile_strength=1240,
            yield_strength=1030,
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
            tensile_strength=550,
            yield_strength=240,
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
            tensile_strength=1100,
            yield_strength=790,
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
            tensile_strength=760,
            yield_strength=380,
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
            tensile_strength=900,
            yield_strength=700,
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
            tensile_strength=690,
            yield_strength=310,
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
            tensile_strength=620,
            yield_strength=450,
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
            tensile_strength=800,
            yield_strength=550,
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
            tensile_strength=200,
            yield_strength=None,
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
            tensile_strength=400,
            yield_strength=250,
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
            tensile_strength=310,
            yield_strength=276,
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
            tensile_strength=572,
            yield_strength=503,
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
            tensile_strength=380,
            yield_strength=150,
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
            tensile_strength=350,
            yield_strength=200,
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
            tensile_strength=600,
            yield_strength=250,
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
            tensile_strength=220,
            yield_strength=70,
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
            tensile_strength=400,
            yield_strength=275,
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
            tensile_strength=895,
            yield_strength=825,
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
            tensile_strength=25,
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
            tensile_strength=20,
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
            tensile_strength=100,
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
            tensile_strength=70,
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
            tensile_strength=80,
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
            tensile_strength=65,
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
            tensile_strength=40,
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

    "EPDM": MaterialInfo(
        grade="EPDM",
        name="三元乙丙橡胶",
        aliases=["乙丙橡胶"],
        category=MaterialCategory.NON_METAL,
        sub_category=MaterialSubCategory.POLYMER,
        group=MaterialGroup.RUBBER,
        properties=MaterialProperties(
            density=1.0,
            tensile_strength=15,
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
            tensile_strength=10,
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
            tensile_strength=40,
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
    # 不锈钢
    (r"S304\d*", "S30408"),
    (r"0?Cr18Ni9", "S30408"),
    (r"S316\d*", "S31603"),
    (r"00Cr17Ni14Mo2", "S31603"),
    (r"(?:1\.)?4301", "S30408"),
    (r"(?:1\.)?4404", "S31603"),
    (r"SUS304", "S30408"),
    (r"SUS316L?", "S31603"),

    # 碳素钢
    (r"Q235[A-D]?", "Q235B"),
    (r"A3钢?", "Q235B"),
    (r"Q345[A-E]?R?", "Q345R"),
    (r"16MnR?", "Q345R"),
    (r"45[#钢]?", "45"),
    (r"S45C", "45"),
    (r"C45", "45"),
    (r"20[#钢]", "20"),  # 需要后缀才能匹配，避免与2205/2507冲突
    (r"^20$", "20"),  # 精确匹配20
    (r"65Mn", "65Mn"),

    # 合金钢
    (r"40Cr\d*", "40Cr"),
    (r"42CrMo\d*", "42CrMo"),
    (r"GCr15", "GCr15"),

    # 铸铁
    (r"HT\d+", "HT200"),
    (r"QT\d+-?\d*", "QT400"),
    (r"灰铸?铁", "HT200"),
    (r"球墨铸?铁", "QT400"),

    # 铝合金
    (r"6061[-T\d]*", "6061"),
    (r"7075[-T\d]*", "7075"),
    (r"LY12", "6061"),
    (r"铝合金", "6061"),

    # 铜合金
    (r"H6[2-8]", "H62"),
    (r"黄铜", "H62"),
    (r"QSn\d+-\d+", "QSn4-3"),
    (r"QA[l]?\d+-\d+", "QAl9-4"),
    (r"[紫纯]铜", "Cu65"),
    (r"T[12]", "Cu65"),

    # 钛合金
    (r"TA2锻?件?", "TA2"),
    (r"Gr?2", "TA2"),
    (r"TC4", "TC4"),
    (r"Ti-?6Al-?4V", "TC4"),
    (r"Gr?5", "TC4"),

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
    # 钛合金
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
