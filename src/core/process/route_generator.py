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
"""

from __future__ import annotations

import logging
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


class ProcessRouteGenerator:
    """工艺路线生成器"""

    def __init__(self):
        self._base_steps = [
            (ProcessStage.blank_preparation, "毛坯准备", "下料/锻造/铸造"),
            (ProcessStage.rough_machining, "粗加工", "车/铣/刨粗加工"),
            (ProcessStage.finish_machining, "精加工", "车/铣精加工"),
            (ProcessStage.inspection, "检验", "尺寸/形位公差检验"),
        ]

    def generate(self, process_requirements: Optional[ProcessRequirements]) -> ProcessRoute:
        """
        根据工艺要求生成工艺路线

        Args:
            process_requirements: 提取的工艺要求

        Returns:
            ProcessRoute 工艺路线
        """
        if not process_requirements:
            return self._generate_basic_route()

        route = ProcessRoute()
        steps: List[ProcessStep] = []
        warnings: List[str] = []

        # 1. 毛坯准备
        steps.append(ProcessStep(
            stage=ProcessStage.blank_preparation,
            name="毛坯准备",
            description="下料/锻造/铸造",
        ))

        # 2. 粗加工
        steps.append(ProcessStep(
            stage=ProcessStage.rough_machining,
            name="粗加工",
            description="车/铣/刨粗加工，留精加工余量",
        ))

        # 3. 预热处理（调质/正火/退火等）
        pre_heat_treatments = [
            ht for ht in (process_requirements.heat_treatments or [])
            if HEAT_TREATMENT_STAGE_MAP.get(ht.type) == ProcessStage.heat_treatment_pre
        ]
        for ht in pre_heat_treatments:
            name = HEAT_TREATMENT_NAMES.get(ht.type, str(ht.type))
            params = {}
            if ht.hardness:
                params["hardness"] = ht.hardness
            if ht.depth:
                params["depth_mm"] = ht.depth
            steps.append(ProcessStep(
                stage=ProcessStage.heat_treatment_pre,
                name=name,
                description=ht.raw,
                parameters=params,
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
            params = {}
            if ht.hardness:
                params["hardness"] = ht.hardness
            if ht.hardness_min:
                params["hardness_min"] = ht.hardness_min
            if ht.hardness_max:
                params["hardness_max"] = ht.hardness_max
            if ht.hardness_unit:
                params["hardness_unit"] = ht.hardness_unit
            if ht.depth:
                params["depth_mm"] = ht.depth
            steps.append(ProcessStep(
                stage=ProcessStage.heat_treatment_post,
                name=name,
                description=ht.raw,
                parameters=params,
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
            steps.append(ProcessStep(
                stage=ProcessStage.finish_machining,
                name="精加工",
                description="车/铣精加工至图纸尺寸",
            ))

        # 7. 焊接工序
        welding_ops = process_requirements.welding or []
        for w in welding_ops:
            name = WELDING_NAMES.get(w.type, str(w.type))
            params = {}
            if w.filler_material:
                params["filler_material"] = w.filler_material
            if w.leg_size:
                params["leg_size_mm"] = w.leg_size
            steps.append(ProcessStep(
                stage=ProcessStage.welding,
                name=name,
                description=w.raw,
                parameters=params,
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
            params = {}
            if st.thickness:
                params["thickness_um"] = st.thickness
            steps.append(ProcessStep(
                stage=ProcessStage.surface_treatment,
                name=name,
                description=st.raw,
                parameters=params,
                source="图纸要求",
            ))

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
        confidence = self._calculate_confidence(process_requirements)

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

    def _generate_basic_route(self) -> ProcessRoute:
        """生成基础工艺路线（无特殊工艺要求时）"""
        steps = []
        for i, (stage, name, desc) in enumerate(self._base_steps):
            steps.append(ProcessStep(
                stage=stage,
                name=name,
                description=desc,
                sequence=i + 1,
            ))
        return ProcessRoute(steps=steps, confidence=0.3)

    def _calculate_confidence(self, proc: ProcessRequirements) -> float:
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
        return min(score, 1.0)


# 单例
_route_generator: Optional[ProcessRouteGenerator] = None


def get_route_generator() -> ProcessRouteGenerator:
    """获取 ProcessRouteGenerator 单例"""
    global _route_generator
    if _route_generator is None:
        _route_generator = ProcessRouteGenerator()
    return _route_generator


def generate_process_route(process_requirements: Optional[ProcessRequirements]) -> ProcessRoute:
    """
    便捷函数：生成工艺路线

    Args:
        process_requirements: 提取的工艺要求

    Returns:
        ProcessRoute 工艺路线
    """
    return get_route_generator().generate(process_requirements)
