"""
约束降级策略
处理不支持的约束类型，提供降级方案
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ConstraintSupportLevel(str, Enum):
    """约束支持级别"""

    NATIVE = "native"  # 原生支持
    WORKAROUND = "workaround"  # 有变通方案
    UNSUPPORTED = "unsupported"  # 不支持
    PARTIAL = "partial"  # 部分支持


@dataclass
class UnsupportedConstraint:
    """不支持的约束"""

    constraint_id: str
    constraint_type: str
    target_engine: str
    support_level: ConstraintSupportLevel
    fallback_type: Optional[str] = None
    evidence: List[Dict] = None
    warning_message: Optional[str] = None
    suggested_alternative: Optional[str] = None


@dataclass
class FallbackStrategy:
    """降级策略"""

    original_type: str
    target_engine: str
    fallback_type: str
    transformation: Dict[str, Any]
    accuracy_loss: float  # 0-1, 精度损失
    explanation: str


class ConstraintFallbackManager:
    """约束降级管理器"""

    def __init__(self):
        # 定义各引擎的约束支持矩阵
        self.support_matrix = {
            "urdf": {
                "native": ["fixed", "revolute", "prismatic", "continuous", "floating", "planar"],
                "unsupported": ["gear", "belt", "chain", "cam", "screw", "rack_pinion"],
                "workaround": {
                    "gear": "coupled_revolute",
                    "belt": "coupled_revolute",
                    "chain": "coupled_revolute",
                    "cam": "position_controlled",
                    "screw": "helical_joint",
                },
            },
            "pybullet": {
                "native": [
                    "fixed",
                    "revolute",
                    "prismatic",
                    "continuous",
                    "planar",
                    "gear",
                    "custom",
                ],
                "unsupported": ["cam_contact", "belt_stretch"],
                "workaround": {
                    "belt": "constraint_with_compliance",
                    "chain": "multi_body_constraint",
                },
            },
            "chrono": {
                "native": ["all_basic", "gear", "belt", "chain", "motor", "spring", "damper"],
                "unsupported": [],  # Chrono支持大部分约束
                "workaround": {},
            },
            "mujoco": {
                "native": ["hinge", "slide", "ball", "free"],
                "unsupported": ["gear_mesh", "belt_drive"],
                "workaround": {
                    "gear": "equality_constraint",
                    "belt": "tendon",
                    "chain": "composite",
                },
            },
        }

        # 降级策略库
        self.fallback_strategies = self._init_fallback_strategies()

        # 不支持约束的存储
        self.unsupported_constraints = []

    def check_constraint_support(
        self, constraint_type: str, target_engine: str
    ) -> Tuple[ConstraintSupportLevel, Optional[str]]:
        """
        检查约束支持情况

        Returns:
            (支持级别, 降级方案)
        """
        if target_engine not in self.support_matrix:
            return ConstraintSupportLevel.UNSUPPORTED, None

        engine_support = self.support_matrix[target_engine]

        # 检查原生支持
        if constraint_type in engine_support["native"]:
            return ConstraintSupportLevel.NATIVE, None

        # 检查是否有变通方案
        if constraint_type in engine_support.get("workaround", {}):
            fallback = engine_support["workaround"][constraint_type]
            return ConstraintSupportLevel.WORKAROUND, fallback

        # 检查是否明确不支持
        if constraint_type in engine_support.get("unsupported", []):
            return ConstraintSupportLevel.UNSUPPORTED, None

        # 部分支持（需要进一步判断）
        return ConstraintSupportLevel.PARTIAL, None

    def apply_fallback(
        self,
        constraint: Dict,
        target_engine: str,
        evidence: Optional[List[Dict]] = None,
    ) -> Tuple[Dict, Optional[UnsupportedConstraint]]:
        """
        应用降级策略

        Args:
            constraint: 原始约束
            target_engine: 目标引擎
            evidence: 支撑证据

        Returns:
            (转换后的约束, 不支持约束记录)
        """
        constraint_type = constraint.get("type", "unknown")
        support_level, fallback_type = self.check_constraint_support(constraint_type, target_engine)

        # 原生支持，无需转换
        if support_level == ConstraintSupportLevel.NATIVE:
            return constraint, None

        # 有变通方案
        if support_level == ConstraintSupportLevel.WORKAROUND and fallback_type:
            strategy = self._get_fallback_strategy(constraint_type, target_engine, fallback_type)

            if strategy:
                # 应用转换
                transformed = self._transform_constraint(constraint, strategy)

                # 记录降级信息
                unsupported = UnsupportedConstraint(
                    constraint_id=constraint.get("id", "unknown"),
                    constraint_type=constraint_type,
                    target_engine=target_engine,
                    support_level=support_level,
                    fallback_type=fallback_type,
                    evidence=evidence or [],
                    warning_message=f"约束 '{constraint_type}' 已降级为 '{fallback_type}'",
                    suggested_alternative=strategy.explanation,
                )

                self.unsupported_constraints.append(unsupported)

                logger.warning(
                    f"Constraint '{constraint_type}' downgraded to '{fallback_type}' "
                    f"for {target_engine} (accuracy loss: {strategy.accuracy_loss:.1%})"
                )

                return transformed, unsupported

        # 完全不支持
        unsupported = UnsupportedConstraint(
            constraint_id=constraint.get("id", "unknown"),
            constraint_type=constraint_type,
            target_engine=target_engine,
            support_level=ConstraintSupportLevel.UNSUPPORTED,
            fallback_type=None,
            evidence=evidence or [],
            warning_message=f"约束 '{constraint_type}' 在 {target_engine} 中不支持",
            suggested_alternative=self._suggest_alternative(constraint_type, target_engine),
        )

        self.unsupported_constraints.append(unsupported)

        # 返回固定约束作为最后的降级
        fallback_constraint = {
            **constraint,
            "type": "fixed",
            "original_type": constraint_type,
            "_fallback_applied": True,
            "_warning": unsupported.warning_message,
        }

        return fallback_constraint, unsupported

    def _init_fallback_strategies(self) -> Dict:
        """初始化降级策略库"""

        strategies = {}

        # URDF齿轮降级策略
        strategies[("gear", "urdf", "coupled_revolute")] = FallbackStrategy(
            original_type="gear",
            target_engine="urdf",
            fallback_type="coupled_revolute",
            transformation={"type": "mimic", "multiplier": lambda r: r, "offset": 0},  # 传动比
            accuracy_loss=0.1,
            explanation="使用URDF mimic joint模拟齿轮传动，保持传动比但失去接触力学",
        )

        # URDF皮带降级策略
        strategies[("belt", "urdf", "coupled_revolute")] = FallbackStrategy(
            original_type="belt",
            target_engine="urdf",
            fallback_type="coupled_revolute",
            transformation={
                "type": "mimic",
                "multiplier": lambda d1, d2: d1 / d2,  # 直径比
                "offset": 0,
            },
            accuracy_loss=0.15,
            explanation="使用耦合转动副模拟皮带，忽略皮带弹性和打滑",
        )

        # MuJoCo齿轮降级策略
        strategies[("gear", "mujoco", "equality_constraint")] = FallbackStrategy(
            original_type="gear",
            target_engine="mujoco",
            fallback_type="equality",
            transformation={
                "type": "equality",
                "joint1": lambda c: c["part1"],
                "joint2": lambda c: c["part2"],
                "polycoef": lambda r: [0, -r, 0, 0, 0],  # 传动比约束
            },
            accuracy_loss=0.05,
            explanation="使用MuJoCo等式约束实现齿轮传动比",
        )

        # PyBullet皮带降级策略
        strategies[("belt", "pybullet", "constraint_with_compliance")] = FallbackStrategy(
            original_type="belt",
            target_engine="pybullet",
            fallback_type="generic_constraint",
            transformation={
                "type": "createConstraint",
                "jointType": "JOINT_GEAR",
                "jointAxis": [0, 0, 1],
                "erp": 0.1,  # 误差修正参数（模拟弹性）
                "relativePositionTarget": 0,
            },
            accuracy_loss=0.08,
            explanation="使用PyBullet通用约束模拟皮带，包含柔性参数",
        )

        return strategies

    def _get_fallback_strategy(
        self, original_type: str, target_engine: str, fallback_type: str
    ) -> Optional[FallbackStrategy]:
        """获取降级策略"""

        key = (original_type, target_engine, fallback_type)
        return self.fallback_strategies.get(key)

    def _transform_constraint(self, constraint: Dict, strategy: FallbackStrategy) -> Dict:
        """应用转换策略"""

        transformed = {
            **constraint,
            "type": strategy.fallback_type,
            "original_type": constraint["type"],
            "_transformation": strategy.transformation,
            "_accuracy_loss": strategy.accuracy_loss,
        }

        # 应用特定转换
        if strategy.fallback_type == "coupled_revolute":
            # 计算传动比
            if "transmission_ratio" in constraint:
                transformed["mimic_multiplier"] = constraint["transmission_ratio"]
            elif "gear_ratio" in constraint:
                transformed["mimic_multiplier"] = constraint["gear_ratio"]

        elif strategy.fallback_type == "equality":
            # MuJoCo等式约束
            transformed["equality_type"] = "joint"
            transformed["data"] = strategy.transformation.get("polycoef", [0, -1, 0, 0, 0])

        return transformed

    def _suggest_alternative(self, constraint_type: str, target_engine: str) -> str:
        """建议替代方案"""

        suggestions = {
            ("gear", "urdf"): "建议使用PyBullet或Chrono以获得原生齿轮支持",
            ("belt", "urdf"): "考虑使用MuJoCo的tendon系统或Chrono的belt模块",
            ("cam", "urdf"): "需要位置控制或使用Chrono的cam-follower约束",
            ("chain", "urdf"): "建议分段建模或使用Chrono的chain模块",
        }

        return suggestions.get(
            (constraint_type, target_engine), f"考虑使用支持 {constraint_type} 的其他仿真引擎"
        )

    def generate_fallback_report(self) -> Dict:
        """生成降级报告"""

        if not self.unsupported_constraints:
            return {"has_fallbacks": False, "message": "所有约束原生支持", "constraints": []}

        report = {
            "has_fallbacks": True,
            "total_unsupported": len(self.unsupported_constraints),
            "by_level": {},
            "by_engine": {},
            "constraints": [],
            "recommendations": [],
        }

        # 按级别统计
        for constraint in self.unsupported_constraints:
            level = constraint.support_level.value
            report["by_level"][level] = report["by_level"].get(level, 0) + 1

            engine = constraint.target_engine
            report["by_engine"][engine] = report["by_engine"].get(engine, 0) + 1

            report["constraints"].append(
                {
                    "id": constraint.constraint_id,
                    "type": constraint.constraint_type,
                    "engine": constraint.target_engine,
                    "level": constraint.support_level.value,
                    "fallback": constraint.fallback_type,
                    "warning": constraint.warning_message,
                    "suggestion": constraint.suggested_alternative,
                }
            )

        # 生成建议
        if report["by_level"].get("unsupported", 0) > 0:
            report["recommendations"].append("存在完全不支持的约束，建议切换仿真引擎或重新设计约束")

        if report["by_level"].get("workaround", 0) > 3:
            report["recommendations"].append("大量约束需要降级，可能影响仿真精度")

        return report

    def clear_unsupported_log(self):
        """清除不支持约束记录"""
        self.unsupported_constraints.clear()


# 使用示例
if __name__ == "__main__":
    # 创建降级管理器
    fallback_manager = ConstraintFallbackManager()

    # 测试约束
    test_constraints = [
        {"id": "c1", "type": "fixed", "part1": "p1", "part2": "p2"},
        {"id": "c2", "type": "gear", "part1": "gear1", "part2": "gear2", "transmission_ratio": 3.0},
        {"id": "c3", "type": "belt", "part1": "pulley1", "part2": "pulley2"},
        {"id": "c4", "type": "cam", "part1": "cam", "part2": "follower"},
    ]

    # 应用降级（目标：URDF）
    print("目标引擎: URDF")
    print("-" * 50)

    for constraint in test_constraints:
        transformed, unsupported = fallback_manager.apply_fallback(constraint, "urdf")

        print(f"原始: {constraint['type']} → 转换: {transformed['type']}")
        if unsupported:
            print(f"  ⚠️ {unsupported.warning_message}")
            if unsupported.suggested_alternative:
                print(f"  💡 {unsupported.suggested_alternative}")

    # 生成报告
    report = fallback_manager.generate_fallback_report()
    print("\n降级报告:")
    print(f"  总计不支持: {report['total_unsupported']}")
    print(f"  按级别: {report['by_level']}")
    print(f"  建议: {report['recommendations']}")
