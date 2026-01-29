"""
工艺路线推荐模块

根据提取的工艺要求自动生成制造工艺路线。

工艺路线生成规则：
1. 基础工序：毛坯准备 → 粗加工 → 精加工 → 检验
2. 热处理插入：根据类型决定插入位置
   - 调质/正火/退火：粗加工后
   - 淬火/渗碳/渗氮：精加工前
   - 去应力：焊接后/热处理后
3. 表面处理：精加工后、检验前
4. 焊接：根据结构件特点，通常在机加工后
5. 材料相关规则：
   - 使用详细材料数据库获取工艺建议
   - 根据材料属性自动生成警告和推荐
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.core.ocr.base import (
    HeatTreatmentType,
    ProcessRequirements,
    SurfaceTreatmentType,
    WeldingType,
)

logger = logging.getLogger(__name__)


# 导入新的材料分类系统
try:
    from src.core.materials import (
        MaterialInfo,
        classify_material_detailed,
        get_process_recommendations,
    )
    from src.core.materials.classifier import (
        MaterialGroup,
        ProcessRecommendation,
    )
    _MATERIALS_AVAILABLE = True
except ImportError:
    _MATERIALS_AVAILABLE = False
    MaterialInfo = None  # type: ignore
    MaterialGroup = None  # type: ignore
    ProcessRecommendation = None  # type: ignore


class ProcessStage(str, Enum):
    """工序阶段"""
    blank_preparation = "blank_preparation"  # 毛坯准备
    rough_machining = "rough_machining"  # 粗加工
    semi_finish_machining = "semi_finish_machining"  # 半精加工
    heat_treatment_pre = "heat_treatment_pre"  # 预热处理（调质/正火）
    finish_machining = "finish_machining"  # 精加工
    heat_treatment_post = "heat_treatment_post"  # 后热处理（淬火/渗碳）
    grinding = "grinding"  # 磨削
    welding = "welding"  # 焊接
    stress_relief = "stress_relief"  # 去应力
    surface_treatment = "surface_treatment"  # 表面处理
    inspection = "inspection"  # 检验
    packaging = "packaging"  # 包装


@dataclass
class ProcessStep:
    """工序步骤"""
    stage: ProcessStage
    name: str  # 工序名称
    description: Optional[str] = None  # 详细描述
    parameters: Dict[str, Any] = field(default_factory=dict)  # 工艺参数
    source: Optional[str] = None  # 来源（如"图纸要求"）
    sequence: int = 0  # 序号

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "source": self.source,
            "sequence": self.sequence,
        }


@dataclass
class ProcessRoute:
    """工艺路线"""
    steps: List[ProcessStep] = field(default_factory=list)
    material: Optional[str] = None
    drawing_type: Optional[str] = None
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "material": self.material,
            "drawing_type": self.drawing_type,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "total_steps": len(self.steps),
        }


# 热处理类型到工序阶段的映射
HEAT_TREATMENT_STAGE_MAP = {
    # 预热处理（粗加工后）
    HeatTreatmentType.normalizing: ProcessStage.heat_treatment_pre,
    HeatTreatmentType.annealing: ProcessStage.heat_treatment_pre,
    HeatTreatmentType.quench_temper: ProcessStage.heat_treatment_pre,
    HeatTreatmentType.aging: ProcessStage.heat_treatment_pre,
    HeatTreatmentType.solution_treatment: ProcessStage.heat_treatment_pre,
    # 后热处理（精加工前或精加工后）
    HeatTreatmentType.quenching: ProcessStage.heat_treatment_post,
    HeatTreatmentType.tempering: ProcessStage.heat_treatment_post,
    HeatTreatmentType.carburizing: ProcessStage.heat_treatment_post,
    HeatTreatmentType.nitriding: ProcessStage.heat_treatment_post,
    HeatTreatmentType.induction_hardening: ProcessStage.heat_treatment_post,
    HeatTreatmentType.flame_hardening: ProcessStage.heat_treatment_post,
    HeatTreatmentType.general_heat_treatment: ProcessStage.heat_treatment_post,
    # 去应力
    HeatTreatmentType.stress_relief: ProcessStage.stress_relief,
}

# 热处理中文名称
HEAT_TREATMENT_NAMES = {
    HeatTreatmentType.quenching: "淬火",
    HeatTreatmentType.tempering: "回火",
    HeatTreatmentType.annealing: "退火",
    HeatTreatmentType.normalizing: "正火",
    HeatTreatmentType.carburizing: "渗碳",
    HeatTreatmentType.nitriding: "渗氮",
    HeatTreatmentType.induction_hardening: "感应淬火",
    HeatTreatmentType.flame_hardening: "火焰淬火",
    HeatTreatmentType.stress_relief: "去应力退火",
    HeatTreatmentType.aging: "时效处理",
    HeatTreatmentType.quench_temper: "调质处理",
    HeatTreatmentType.solution_treatment: "固溶处理",
    HeatTreatmentType.general_heat_treatment: "热处理",
}

# 表面处理中文名称
SURFACE_TREATMENT_NAMES = {
    SurfaceTreatmentType.electroplating: "电镀",
    SurfaceTreatmentType.galvanizing: "镀锌",
    SurfaceTreatmentType.chromating: "镀铬",
    SurfaceTreatmentType.nickel_plating: "镀镍",
    SurfaceTreatmentType.anodizing: "阳极氧化",
    SurfaceTreatmentType.phosphating: "磷化",
    SurfaceTreatmentType.blackening: "发黑",
    SurfaceTreatmentType.painting: "喷漆",
    SurfaceTreatmentType.powder_coating: "粉末喷涂",
    SurfaceTreatmentType.polishing: "抛光",
    SurfaceTreatmentType.sandblasting: "喷砂",
    SurfaceTreatmentType.passivation: "钝化",
}

# 焊接中文名称
WELDING_NAMES = {
    WeldingType.arc_welding: "电弧焊",
    WeldingType.mig_welding: "MIG焊",
    WeldingType.tig_welding: "氩弧焊",
    WeldingType.spot_welding: "点焊",
    WeldingType.seam_welding: "缝焊",
    WeldingType.laser_welding: "激光焊",
    WeldingType.electron_beam: "电子束焊",
    WeldingType.brazing: "钎焊",
    WeldingType.soldering: "软钎焊",
}


# 材料分类模式
MATERIAL_PATTERNS = {
    "cast_iron": ["HT", "QT", "球墨", "灰铁", "铸铁", "KT"],
    "carbon_steel": ["Q235", "Q345", "45#", "45钢", "20钢", "20#", "A3"],
    "alloy_steel": ["40Cr", "42CrMo", "GCr15", "Cr12", "合金钢"],
    "stainless_steel": ["304", "316", "S304", "S316", "S30408", "不锈钢", "1Cr18Ni9Ti"],
    "aluminum": ["6061", "7075", "2024", "铝合金", "AL", "LY12"],
    "copper": ["黄铜", "紫铜", "H62", "H68", "铜合金", "CuZn"],
    "titanium": ["TC4", "TA2", "钛合金", "Ti"],
}


def classify_material(material: Optional[str]) -> Optional[str]:
    """
    对材料进行分类

    优先使用新的详细材料分类系统，回退到旧的模式匹配。

    Args:
        material: 材料名称字符串

    Returns:
        材料类别 或 None
    """
    if not material:
        return None

    # 优先使用新的材料分类系统
    if _MATERIALS_AVAILABLE:
        info = classify_material_detailed(material)
        if info:
            # 将 MaterialGroup 映射到旧的类别名称
            group_to_category = {
                MaterialGroup.CAST_IRON: "cast_iron",
                MaterialGroup.CARBON_STEEL: "carbon_steel",
                MaterialGroup.ALLOY_STEEL: "alloy_steel",
                MaterialGroup.STAINLESS_STEEL: "stainless_steel",
                MaterialGroup.CORROSION_RESISTANT: "stainless_steel",  # 耐蚀合金归类为不锈钢
                MaterialGroup.ALUMINUM: "aluminum",
                MaterialGroup.COPPER: "copper",
                MaterialGroup.TITANIUM: "titanium",
            }
            return group_to_category.get(info.group)

    # 回退到旧的模式匹配
    material_upper = material.upper()

    for category, patterns in MATERIAL_PATTERNS.items():
        for pattern in patterns:
            if not pattern:
                continue
            # Chinese patterns use direct substring matching
            if any("\u4e00" <= ch <= "\u9fff" for ch in pattern):
                if pattern in material:
                    return category
                continue

            token = pattern.upper()
            if len(token) <= 2:
                # Short tokens require word-ish boundaries to avoid false positives
                if re.search(rf"(?<![A-Z0-9]){re.escape(token)}(?![A-Z0-9])", material_upper):
                    return category
            else:
                if token in material_upper:
                    return category

    return None


def get_detailed_material_info(material: Optional[str]) -> Optional[MaterialInfo]:
    """
    获取详细的材料信息

    Args:
        material: 材料名称字符串

    Returns:
        MaterialInfo 或 None
    """
    if not material or not _MATERIALS_AVAILABLE:
        return None
    return classify_material_detailed(material)


# 材料相关的工艺建议
MATERIAL_PROCESS_HINTS: Dict[str, Dict[str, Any]] = {
    "cast_iron": {
        "blank_hint": "铸造毛坯",
        "aging_recommended": True,
        "aging_reason": "铸铁件建议时效处理消除铸造内应力",
        "surface_hint": "发黑/喷漆",
    },
    "carbon_steel": {
        "blank_hint": "锻造/圆钢",
        "can_carburize": True,
        "surface_hint": "镀锌/发黑",
    },
    "alloy_steel": {
        "blank_hint": "锻造",
        "quench_temper_recommended": True,
        "surface_hint": "镀铬/发黑",
    },
    "stainless_steel": {
        "blank_hint": "板材/棒材",
        "passivation_recommended": True,
        "passivation_reason": "不锈钢建议钝化处理提高耐蚀性",
        "no_carburize": True,
        "surface_hint": "抛光/钝化",
    },
    "aluminum": {
        "blank_hint": "挤压型材/板材",
        "anodize_recommended": True,
        "anodize_reason": "铝合金建议阳极氧化处理",
        "no_electroplate": True,
        "surface_hint": "阳极氧化/喷漆",
    },
    "copper": {
        "blank_hint": "棒材/板材",
        "surface_hint": "抛光/镀层",
    },
    "titanium": {
        "blank_hint": "锻造/板材",
        "surface_hint": "阳极氧化",
        "special_tooling": True,
        "special_tooling_reason": "钛合金需要专用刀具和切削参数",
    },
}


class ProcessRouteGenerator:
    """工艺路线生成器"""

    def __init__(self) -> None:
        self._base_steps = [
            (ProcessStage.blank_preparation, "毛坯准备", "下料/锻造/铸造"),
            (ProcessStage.rough_machining, "粗加工", "车/铣/刨粗加工"),
            (ProcessStage.finish_machining, "精加工", "车/铣精加工"),
            (ProcessStage.inspection, "检验", "尺寸/形位公差检验"),
        ]

    def generate(
        self,
        process_requirements: Optional[ProcessRequirements],
        material: Optional[str] = None,
    ) -> ProcessRoute:
        """
        根据工艺要求生成工艺路线

        Args:
            process_requirements: 提取的工艺要求
            material: 材料名称（可选）

        Returns:
            ProcessRoute 工艺路线
        """
        if not process_requirements:
            return self._generate_basic_route(material=material)

        route = ProcessRoute()
        route.material = material
        steps: List[ProcessStep] = []
        warnings: List[str] = []

        # 材料分类 - 优先使用详细材料系统
        material_category = classify_material(material)
        material_hints: Dict[str, Any] = (
            MATERIAL_PROCESS_HINTS.get(material_category, {})
            if material_category
            else {}
        )

        # 获取详细材料信息（如果可用）
        material_info = get_detailed_material_info(material)
        if material_info:
            # 使用详细材料数据库的工艺建议
            process_rec = material_info.process
            if process_rec.blank_hint:
                material_hints["blank_hint"] = process_rec.blank_hint
            # 添加材料数据库的警告
            for warn in process_rec.warnings:
                warnings.append(f"[{material_info.grade}] {warn}")
            # 添加材料数据库的推荐
            for rec in process_rec.recommendations:
                warnings.append(f"[工艺建议] {rec}")

        # 提取切削参数（如果有材料信息）
        cutting_params: Dict[str, Any] = {}
        if material_info and material_info.process.cutting_speed_range:
            cutting_params["cutting_speed_min"] = material_info.process.cutting_speed_range[0]
            cutting_params["cutting_speed_max"] = material_info.process.cutting_speed_range[1]
            cutting_params["cutting_speed_unit"] = "m/min"
        if material_info and material_info.process.feed_rate_range:
            cutting_params["feed_rate_min"] = material_info.process.feed_rate_range[0]
            cutting_params["feed_rate_max"] = material_info.process.feed_rate_range[1]
            cutting_params["feed_rate_unit"] = "mm/r"
        if material_info:
            cutting_params["coolant_required"] = material_info.process.coolant_required
            cutting_params["special_tooling"] = material_info.process.special_tooling

        # 1. 毛坯准备
        blank_desc = material_hints.get("blank_hint", "下料/锻造/铸造")
        steps.append(ProcessStep(
            stage=ProcessStage.blank_preparation,
            name="毛坯准备",
            description=blank_desc,
        ))

        # 2. 粗加工
        rough_params: Dict[str, Any] = {}
        if cutting_params:
            # 粗加工使用较低切速、较大进给
            if "cutting_speed_min" in cutting_params:
                rough_params["cutting_speed_recommended"] = cutting_params["cutting_speed_min"]
                rough_params["cutting_speed_unit"] = "m/min"
            rough_params["coolant_required"] = cutting_params.get("coolant_required", True)
            rough_params["special_tooling"] = cutting_params.get("special_tooling", False)
        rough_desc = "车/铣/刨粗加工，留精加工余量"
        if cutting_params.get("special_tooling"):
            rough_desc += "（需专用刀具）"
        steps.append(ProcessStep(
            stage=ProcessStage.rough_machining,
            name="粗加工",
            description=rough_desc,
            parameters=rough_params if rough_params else {},
        ))

        # 3. 预热处理（调质/正火/退火等）
        pre_heat_treatments = [
            ht for ht in (process_requirements.heat_treatments or [])
            if HEAT_TREATMENT_STAGE_MAP.get(ht.type) == ProcessStage.heat_treatment_pre
        ]
        for ht in pre_heat_treatments:
            name = HEAT_TREATMENT_NAMES.get(ht.type, str(ht.type))
            ht_params: Dict[str, Any] = {}
            if ht.hardness:
                ht_params["hardness"] = ht.hardness
            if ht.depth:
                ht_params["depth_mm"] = ht.depth
            steps.append(ProcessStep(
                stage=ProcessStage.heat_treatment_pre,
                name=name,
                description=ht.raw,
                parameters=ht_params,
                source="图纸要求",
            ))

        # 4. 半精加工（如果有后续热处理）
        post_heat_treatments = [
            ht for ht in (process_requirements.heat_treatments or [])
            if HEAT_TREATMENT_STAGE_MAP.get(ht.type) == ProcessStage.heat_treatment_post
        ]
        if post_heat_treatments:
            steps.append(ProcessStep(
                stage=ProcessStage.semi_finish_machining,
                name="半精加工",
                description="为热处理留余量",
            ))

        # 5. 后热处理（淬火/渗碳/渗氮等）
        for ht in post_heat_treatments:
            name = HEAT_TREATMENT_NAMES.get(ht.type, str(ht.type))
            post_params: Dict[str, Any] = {}
            if ht.hardness:
                post_params["hardness"] = ht.hardness
            if ht.hardness_min:
                post_params["hardness_min"] = ht.hardness_min
            if ht.hardness_max:
                post_params["hardness_max"] = ht.hardness_max
            if ht.hardness_unit:
                post_params["hardness_unit"] = ht.hardness_unit
            if ht.depth:
                post_params["depth_mm"] = ht.depth
            steps.append(ProcessStep(
                stage=ProcessStage.heat_treatment_post,
                name=name,
                description=ht.raw,
                parameters=post_params,
                source="图纸要求",
            ))

        # 6. 精加工/磨削
        if post_heat_treatments:
            # 热处理后需要磨削
            steps.append(ProcessStep(
                stage=ProcessStage.grinding,
                name="磨削",
                description="热处理后精密磨削",
            ))
        else:
            # 精加工使用较高切速
            finish_params: Dict[str, Any] = {}
            if cutting_params:
                if "cutting_speed_max" in cutting_params:
                    finish_params["cutting_speed_recommended"] = cutting_params["cutting_speed_max"]
                    finish_params["cutting_speed_unit"] = "m/min"
                finish_params["coolant_required"] = cutting_params.get("coolant_required", True)
                finish_params["special_tooling"] = cutting_params.get("special_tooling", False)
            finish_desc = "车/铣精加工至图纸尺寸"
            if cutting_params.get("special_tooling"):
                finish_desc += "（需专用刀具）"
            steps.append(ProcessStep(
                stage=ProcessStage.finish_machining,
                name="精加工",
                description=finish_desc,
                parameters=finish_params if finish_params else {},
            ))

        # 7. 焊接工序
        welding_ops = process_requirements.welding or []
        for w in welding_ops:
            name = WELDING_NAMES.get(w.type, str(w.type))
            weld_params: Dict[str, Any] = {}
            if w.filler_material:
                weld_params["filler_material"] = w.filler_material
            if w.leg_size:
                weld_params["leg_size_mm"] = w.leg_size
            steps.append(ProcessStep(
                stage=ProcessStage.welding,
                name=name,
                description=w.raw,
                parameters=weld_params,
                source="图纸要求",
            ))

        # 8. 去应力退火（焊接后或热处理后）
        stress_relief_treatments = [
            ht for ht in (process_requirements.heat_treatments or [])
            if HEAT_TREATMENT_STAGE_MAP.get(ht.type) == ProcessStage.stress_relief
        ]
        for ht in stress_relief_treatments:
            steps.append(ProcessStep(
                stage=ProcessStage.stress_relief,
                name="去应力退火",
                description=ht.raw,
                source="图纸要求",
            ))
        # 如果有焊接但没有明确去应力，添加警告
        if welding_ops and not stress_relief_treatments:
            warnings.append("焊接件建议增加去应力退火工序")

        # 9. 表面处理
        surface_ops = process_requirements.surface_treatments or []
        for st in surface_ops:
            name = SURFACE_TREATMENT_NAMES.get(st.type, str(st.type))
            surface_params: Dict[str, Any] = {}
            if st.thickness:
                surface_params["thickness_um"] = st.thickness
            steps.append(ProcessStep(
                stage=ProcessStage.surface_treatment,
                name=name,
                description=st.raw,
                parameters=surface_params,
                source="图纸要求",
            ))

        # 9.1 材料相关的工艺建议
        has_passivation = any(
            st.type == SurfaceTreatmentType.passivation for st in surface_ops
        )
        has_anodizing = any(
            st.type == SurfaceTreatmentType.anodizing for st in surface_ops
        )

        # 不锈钢钝化建议
        if material_hints.get("passivation_recommended") and not has_passivation:
            warnings.append(material_hints.get("passivation_reason", "建议增加钝化处理"))

        # 铝合金阳极氧化建议
        if material_hints.get("anodize_recommended") and not has_anodizing:
            warnings.append(material_hints.get("anodize_reason", "建议增加阳极氧化处理"))

        # 铸铁时效建议
        has_aging = any(
            ht.type == HeatTreatmentType.aging
            for ht in (process_requirements.heat_treatments or [])
        )
        if material_hints.get("aging_recommended") and not has_aging:
            warnings.append(material_hints.get("aging_reason", "建议增加时效处理"))

        # 钛合金特殊刀具提示
        if material_hints.get("special_tooling"):
            warnings.append(material_hints.get("special_tooling_reason", "需要特殊加工参数"))

        # 10. 检验
        inspection_items = []
        if post_heat_treatments:
            inspection_items.append("硬度检验")
        if surface_ops:
            inspection_items.append("镀层/涂层检验")
        if welding_ops:
            inspection_items.append("焊缝检验")
        inspection_items.append("尺寸检验")

        steps.append(ProcessStep(
            stage=ProcessStage.inspection,
            name="检验",
            description="；".join(inspection_items),
        ))

        # 设置序号
        for i, step in enumerate(steps):
            step.sequence = i + 1

        # 计算置信度
        confidence = self._calculate_confidence(process_requirements, material=material)

        route.steps = steps
        route.confidence = confidence
        route.warnings = warnings

        logger.info(
            "process_route.generated",
            extra={
                "total_steps": len(steps),
                "heat_treatments": len(process_requirements.heat_treatments or []),
                "surface_treatments": len(process_requirements.surface_treatments or []),
                "welding_ops": len(welding_ops),
                "confidence": confidence,
            },
        )

        return route

    def _generate_basic_route(self, material: Optional[str] = None) -> ProcessRoute:
        """生成基础工艺路线（无特殊工艺要求时）"""
        steps: List[ProcessStep] = []
        warnings: List[str] = []
        material_category = classify_material(material)
        material_hints: Dict[str, Any] = (
            MATERIAL_PROCESS_HINTS.get(material_category, {})
            if material_category
            else {}
        )

        # 获取详细材料信息（如果可用）
        material_info = get_detailed_material_info(material)
        if material_info:
            process_rec = material_info.process
            if process_rec.blank_hint:
                material_hints["blank_hint"] = process_rec.blank_hint
            # 添加材料数据库的警告
            for warn in process_rec.warnings:
                warnings.append(f"[{material_info.grade}] {warn}")
            # 添加材料数据库的推荐
            for rec in process_rec.recommendations:
                warnings.append(f"[工艺建议] {rec}")

        for i, (stage, name, desc) in enumerate(self._base_steps):
            # 毛坯准备根据材料调整描述
            if stage == ProcessStage.blank_preparation and material_hints.get(
                "blank_hint"
            ):
                desc = material_hints["blank_hint"]
            steps.append(ProcessStep(
                stage=stage,
                name=name,
                description=desc,
                sequence=i + 1,
            ))
        return ProcessRoute(
            steps=steps,
            material=material,
            confidence=0.3,
            warnings=warnings,
        )

    def _calculate_confidence(
        self,
        proc: ProcessRequirements,
        material: Optional[str] = None,
    ) -> float:
        """计算工艺路线置信度"""
        score = 0.4  # 基础分
        if proc.heat_treatments:
            score += 0.2
        if proc.surface_treatments:
            score += 0.15
        if proc.welding:
            score += 0.15
        if proc.general_notes:
            score += 0.1
        # 有材料信息增加置信度
        if material:
            score += 0.05
            # 如果在详细材料数据库中找到，进一步增加置信度
            if _MATERIALS_AVAILABLE and classify_material_detailed(material):
                score += 0.05
        return min(score, 1.0)


# 单例
_route_generator: Optional[ProcessRouteGenerator] = None


def get_route_generator() -> ProcessRouteGenerator:
    """获取 ProcessRouteGenerator 单例"""
    global _route_generator
    if _route_generator is None:
        _route_generator = ProcessRouteGenerator()
    return _route_generator


def generate_process_route(
    process_requirements: Optional[ProcessRequirements],
    material: Optional[str] = None,
) -> ProcessRoute:
    """
    便捷函数：生成工艺路线

    Args:
        process_requirements: 提取的工艺要求
        material: 材料名称（可选）

    Returns:
        ProcessRoute 工艺路线
    """
    return get_route_generator().generate(process_requirements, material=material)
