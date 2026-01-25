"""
混合分类器配置模块

统一管理所有 Feature Flags 和配置项，支持一键开关各功能模块。

配置优先级: 环境变量 > 配置文件 > 默认值
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# 默认配置文件路径
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config/hybrid_classifier.yaml"


@dataclass
class FilenameClassifierConfig:
    """文件名分类器配置"""
    enabled: bool = True
    min_confidence: float = 0.8
    exact_match_conf: float = 0.95
    partial_match_conf: float = 0.7
    fuzzy_match_conf: float = 0.5
    fusion_weight: float = 0.7


@dataclass
class Graph2DConfig:
    """Graph2D 分类器配置"""
    enabled: bool = False
    min_confidence: float = 0.5
    fusion_weight: float = 0.3
    exclude_labels: str = "other"
    allow_labels: str = ""


@dataclass
class TitleBlockConfig:
    """标题栏特征配置"""
    enabled: bool = False
    region_x_ratio: float = 0.6
    region_y_ratio: float = 0.4


@dataclass
class SamplingConfig:
    """节点采样配置"""
    max_nodes: int = 200
    strategy: str = "importance"  # importance|random|hybrid
    seed: int = 42
    text_priority_ratio: float = 0.3


@dataclass
class ClassBalanceConfig:
    """类别平衡配置"""
    strategy: str = "focal"  # none|weights|focal|logit_adj
    weight_mode: str = "sqrt"  # inverse|sqrt|log
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    logit_adj_tau: float = 1.0


@dataclass
class MultimodalFusionConfig:
    """多模态融合配置"""
    enabled: bool = True
    geometry_weight: float = 0.3
    text_weight: float = 0.5
    rule_weight: float = 0.2
    gate_type: str = "weighted"  # weighted|attention|learned


@dataclass
class DistillationConfig:
    """知识蒸馏配置"""
    enabled: bool = False
    alpha: float = 0.3
    temperature: float = 3.0
    teacher_type: str = "hybrid"  # filename|titleblock|hybrid


@dataclass
class HybridClassifierConfig:
    """混合分类器总配置"""
    enabled: bool = True
    version: str = "1.0.0"

    filename: FilenameClassifierConfig = field(default_factory=FilenameClassifierConfig)
    graph2d: Graph2DConfig = field(default_factory=Graph2DConfig)
    titleblock: TitleBlockConfig = field(default_factory=TitleBlockConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    class_balance: ClassBalanceConfig = field(default_factory=ClassBalanceConfig)
    multimodal: MultimodalFusionConfig = field(default_factory=MultimodalFusionConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "enabled": self.enabled,
            "version": self.version,
            "filename": {
                "enabled": self.filename.enabled,
                "min_confidence": self.filename.min_confidence,
                "exact_match_conf": self.filename.exact_match_conf,
                "partial_match_conf": self.filename.partial_match_conf,
                "fuzzy_match_conf": self.filename.fuzzy_match_conf,
                "fusion_weight": self.filename.fusion_weight,
            },
            "graph2d": {
                "enabled": self.graph2d.enabled,
                "min_confidence": self.graph2d.min_confidence,
                "fusion_weight": self.graph2d.fusion_weight,
            },
            "titleblock": {
                "enabled": self.titleblock.enabled,
                "region_x_ratio": self.titleblock.region_x_ratio,
                "region_y_ratio": self.titleblock.region_y_ratio,
            },
            "sampling": {
                "max_nodes": self.sampling.max_nodes,
                "strategy": self.sampling.strategy,
                "seed": self.sampling.seed,
            },
            "class_balance": {
                "strategy": self.class_balance.strategy,
                "weight_mode": self.class_balance.weight_mode,
            },
            "multimodal": {
                "enabled": self.multimodal.enabled,
                "geometry_weight": self.multimodal.geometry_weight,
                "text_weight": self.multimodal.text_weight,
                "rule_weight": self.multimodal.rule_weight,
            },
            "distillation": {
                "enabled": self.distillation.enabled,
                "alpha": self.distillation.alpha,
                "temperature": self.distillation.temperature,
            },
        }

    @classmethod
    def from_env(cls) -> "HybridClassifierConfig":
        """从环境变量加载配置"""
        config = cls()

        # 总开关
        config.enabled = os.getenv("HYBRID_CLASSIFIER_ENABLED", "true").lower() == "true"

        # 文件名分类器
        config.filename.enabled = os.getenv("FILENAME_CLASSIFIER_ENABLED", "true").lower() == "true"
        config.filename.min_confidence = float(os.getenv("FILENAME_MIN_CONF", "0.8"))
        config.filename.exact_match_conf = float(os.getenv("FILENAME_EXACT_MATCH_CONF", "0.95"))
        config.filename.fusion_weight = float(os.getenv("FILENAME_FUSION_WEIGHT", "0.7"))

        # Graph2D
        config.graph2d.enabled = os.getenv("GRAPH2D_ENABLED", "false").lower() == "true"
        config.graph2d.min_confidence = float(os.getenv("GRAPH2D_MIN_CONF", "0.5"))
        config.graph2d.fusion_weight = float(os.getenv("GRAPH2D_FUSION_WEIGHT", "0.3"))

        # 标题栏
        config.titleblock.enabled = os.getenv("TITLEBLOCK_ENABLED", "false").lower() == "true"

        # 采样
        config.sampling.max_nodes = int(os.getenv("DXF_MAX_NODES", "200"))
        config.sampling.strategy = os.getenv("DXF_SAMPLING_STRATEGY", "importance")
        config.sampling.seed = int(os.getenv("DXF_SAMPLING_SEED", "42"))

        # 类别平衡
        config.class_balance.strategy = os.getenv("CLASS_BALANCE_STRATEGY", "focal")

        # 多模态融合
        config.multimodal.enabled = os.getenv("MULTIMODAL_FUSION_ENABLED", "true").lower() == "true"

        # 蒸馏
        config.distillation.enabled = os.getenv("DISTILLATION_ENABLED", "false").lower() == "true"

        return config


# 全局配置实例
_CONFIG: Optional[HybridClassifierConfig] = None


def get_config() -> HybridClassifierConfig:
    """获取全局配置"""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = HybridClassifierConfig.from_env()
    return _CONFIG


def reset_config() -> None:
    """重置配置"""
    global _CONFIG
    _CONFIG = None


def dump_config() -> str:
    """导出当前配置为 JSON"""
    return json.dumps(get_config().to_dict(), indent=2, ensure_ascii=False)
