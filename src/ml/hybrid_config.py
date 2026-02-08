"""
混合分类器配置模块。

配置优先级: 环境变量 > 配置文件 > 默认值
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except Exception:  # pragma: no cover - optional import guard
    yaml = None

logger = logging.getLogger(__name__)

# 默认配置文件路径
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config/hybrid_classifier.yaml"
)

DEFAULT_GRAPH2D_DRAWING_TYPE_LABELS = [
    "零件图",
    "机械制图",
    "装配图",
    "练习零件图",
    "原理图",
    "模板",
]


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_str(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


@dataclass
class FilenameClassifierConfig:
    """文件名分类器配置"""

    enabled: bool = True
    min_confidence: float = 0.8
    exact_match_conf: float = 0.95
    partial_match_conf: float = 0.7
    fuzzy_match_conf: float = 0.5
    fusion_weight: float = 0.7
    synonyms_path: str = ""


@dataclass
class Graph2DConfig:
    """Graph2D 分类器配置"""

    enabled: bool = False
    min_confidence: float = 0.5
    fusion_weight: float = 0.3
    exclude_labels: str = "other"
    allow_labels: str = ""
    drawing_type_labels: List[str] = field(
        default_factory=lambda: list(DEFAULT_GRAPH2D_DRAWING_TYPE_LABELS)
    )


@dataclass
class TitleBlockConfig:
    """标题栏特征配置"""

    enabled: bool = False
    region_x_ratio: float = 0.6
    region_y_ratio: float = 0.4
    min_confidence: float = 0.75
    fusion_weight: float = 0.2
    override_enabled: bool = False


@dataclass
class ProcessConfig:
    """工艺特征配置"""

    enabled: bool = True
    min_confidence: float = 0.3
    fusion_weight: float = 0.15


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
    process: ProcessConfig = field(default_factory=ProcessConfig)
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
                "synonyms_path": self.filename.synonyms_path,
            },
            "graph2d": {
                "enabled": self.graph2d.enabled,
                "min_confidence": self.graph2d.min_confidence,
                "fusion_weight": self.graph2d.fusion_weight,
                "exclude_labels": self.graph2d.exclude_labels,
                "allow_labels": self.graph2d.allow_labels,
                "drawing_type_labels": list(self.graph2d.drawing_type_labels),
            },
            "titleblock": {
                "enabled": self.titleblock.enabled,
                "region_x_ratio": self.titleblock.region_x_ratio,
                "region_y_ratio": self.titleblock.region_y_ratio,
                "min_confidence": self.titleblock.min_confidence,
                "fusion_weight": self.titleblock.fusion_weight,
                "override_enabled": self.titleblock.override_enabled,
            },
            "process": {
                "enabled": self.process.enabled,
                "min_confidence": self.process.min_confidence,
                "fusion_weight": self.process.fusion_weight,
            },
            "sampling": {
                "max_nodes": self.sampling.max_nodes,
                "strategy": self.sampling.strategy,
                "seed": self.sampling.seed,
                "text_priority_ratio": self.sampling.text_priority_ratio,
            },
            "class_balance": {
                "strategy": self.class_balance.strategy,
                "weight_mode": self.class_balance.weight_mode,
                "focal_alpha": self.class_balance.focal_alpha,
                "focal_gamma": self.class_balance.focal_gamma,
                "logit_adj_tau": self.class_balance.logit_adj_tau,
            },
            "multimodal": {
                "enabled": self.multimodal.enabled,
                "geometry_weight": self.multimodal.geometry_weight,
                "text_weight": self.multimodal.text_weight,
                "rule_weight": self.multimodal.rule_weight,
                "gate_type": self.multimodal.gate_type,
            },
            "distillation": {
                "enabled": self.distillation.enabled,
                "alpha": self.distillation.alpha,
                "temperature": self.distillation.temperature,
                "teacher_type": self.distillation.teacher_type,
            },
        }

    def apply_dict(self, payload: Dict[str, Any]) -> None:
        """应用配置字典（配置文件层）"""
        if not isinstance(payload, dict):
            return

        self.enabled = _to_bool(payload.get("enabled"), self.enabled)
        self.version = _to_str(payload.get("version"), self.version)

        filename = payload.get("filename", {})
        if isinstance(filename, dict):
            self.filename.enabled = _to_bool(
                filename.get("enabled"), self.filename.enabled
            )
            self.filename.min_confidence = _to_float(
                filename.get("min_confidence"), self.filename.min_confidence
            )
            self.filename.exact_match_conf = _to_float(
                filename.get("exact_match_conf"), self.filename.exact_match_conf
            )
            self.filename.partial_match_conf = _to_float(
                filename.get("partial_match_conf"), self.filename.partial_match_conf
            )
            self.filename.fuzzy_match_conf = _to_float(
                filename.get("fuzzy_match_conf"), self.filename.fuzzy_match_conf
            )
            self.filename.fusion_weight = _to_float(
                filename.get("fusion_weight"), self.filename.fusion_weight
            )
            self.filename.synonyms_path = _to_str(
                filename.get("synonyms_path"), self.filename.synonyms_path
            )

        graph2d = payload.get("graph2d", {})
        if isinstance(graph2d, dict):
            self.graph2d.enabled = _to_bool(
                graph2d.get("enabled"), self.graph2d.enabled
            )
            self.graph2d.min_confidence = _to_float(
                graph2d.get("min_confidence"), self.graph2d.min_confidence
            )
            self.graph2d.fusion_weight = _to_float(
                graph2d.get("fusion_weight"), self.graph2d.fusion_weight
            )
            self.graph2d.exclude_labels = _to_str(
                graph2d.get("exclude_labels"), self.graph2d.exclude_labels
            )
            self.graph2d.allow_labels = _to_str(
                graph2d.get("allow_labels"), self.graph2d.allow_labels
            )
            labels = graph2d.get("drawing_type_labels")
            if isinstance(labels, list):
                self.graph2d.drawing_type_labels = [
                    str(label).strip() for label in labels if str(label).strip()
                ]

        titleblock = payload.get("titleblock", {})
        if isinstance(titleblock, dict):
            self.titleblock.enabled = _to_bool(
                titleblock.get("enabled"), self.titleblock.enabled
            )
            self.titleblock.region_x_ratio = _to_float(
                titleblock.get("region_x_ratio"), self.titleblock.region_x_ratio
            )
            self.titleblock.region_y_ratio = _to_float(
                titleblock.get("region_y_ratio"), self.titleblock.region_y_ratio
            )
            self.titleblock.min_confidence = _to_float(
                titleblock.get("min_confidence"), self.titleblock.min_confidence
            )
            self.titleblock.fusion_weight = _to_float(
                titleblock.get("fusion_weight"), self.titleblock.fusion_weight
            )
            self.titleblock.override_enabled = _to_bool(
                titleblock.get("override_enabled"), self.titleblock.override_enabled
            )

        process = payload.get("process", {})
        if isinstance(process, dict):
            self.process.enabled = _to_bool(
                process.get("enabled"), self.process.enabled
            )
            self.process.min_confidence = _to_float(
                process.get("min_confidence"), self.process.min_confidence
            )
            self.process.fusion_weight = _to_float(
                process.get("fusion_weight"), self.process.fusion_weight
            )

        sampling = payload.get("sampling", {})
        if isinstance(sampling, dict):
            self.sampling.max_nodes = _to_int(
                sampling.get("max_nodes"), self.sampling.max_nodes
            )
            self.sampling.strategy = _to_str(
                sampling.get("strategy"), self.sampling.strategy
            )
            self.sampling.seed = _to_int(sampling.get("seed"), self.sampling.seed)
            self.sampling.text_priority_ratio = _to_float(
                sampling.get("text_priority_ratio"), self.sampling.text_priority_ratio
            )

        class_balance = payload.get("class_balance", {})
        if isinstance(class_balance, dict):
            self.class_balance.strategy = _to_str(
                class_balance.get("strategy"), self.class_balance.strategy
            )
            self.class_balance.weight_mode = _to_str(
                class_balance.get("weight_mode"), self.class_balance.weight_mode
            )
            self.class_balance.focal_alpha = _to_float(
                class_balance.get("focal_alpha"), self.class_balance.focal_alpha
            )
            self.class_balance.focal_gamma = _to_float(
                class_balance.get("focal_gamma"), self.class_balance.focal_gamma
            )
            self.class_balance.logit_adj_tau = _to_float(
                class_balance.get("logit_adj_tau"), self.class_balance.logit_adj_tau
            )

        multimodal = payload.get("multimodal", {})
        if isinstance(multimodal, dict):
            self.multimodal.enabled = _to_bool(
                multimodal.get("enabled"), self.multimodal.enabled
            )
            self.multimodal.geometry_weight = _to_float(
                multimodal.get("geometry_weight"), self.multimodal.geometry_weight
            )
            self.multimodal.text_weight = _to_float(
                multimodal.get("text_weight"), self.multimodal.text_weight
            )
            self.multimodal.rule_weight = _to_float(
                multimodal.get("rule_weight"), self.multimodal.rule_weight
            )
            self.multimodal.gate_type = _to_str(
                multimodal.get("gate_type"), self.multimodal.gate_type
            )

        distillation = payload.get("distillation", {})
        if isinstance(distillation, dict):
            self.distillation.enabled = _to_bool(
                distillation.get("enabled"), self.distillation.enabled
            )
            self.distillation.alpha = _to_float(
                distillation.get("alpha"), self.distillation.alpha
            )
            self.distillation.temperature = _to_float(
                distillation.get("temperature"), self.distillation.temperature
            )
            self.distillation.teacher_type = _to_str(
                distillation.get("teacher_type"), self.distillation.teacher_type
            )

    def apply_env(self) -> None:
        """应用环境变量覆盖"""
        self.enabled = _to_bool(os.getenv("HYBRID_CLASSIFIER_ENABLED"), self.enabled)

        self.filename.enabled = _to_bool(
            os.getenv("FILENAME_CLASSIFIER_ENABLED"), self.filename.enabled
        )
        self.filename.min_confidence = _to_float(
            os.getenv("FILENAME_MIN_CONF"), self.filename.min_confidence
        )
        self.filename.exact_match_conf = _to_float(
            os.getenv("FILENAME_EXACT_MATCH_CONF"), self.filename.exact_match_conf
        )
        self.filename.partial_match_conf = _to_float(
            os.getenv("FILENAME_PARTIAL_MATCH_CONF"), self.filename.partial_match_conf
        )
        self.filename.fuzzy_match_conf = _to_float(
            os.getenv("FILENAME_FUZZY_MATCH_CONF"), self.filename.fuzzy_match_conf
        )
        self.filename.fusion_weight = _to_float(
            os.getenv("FILENAME_FUSION_WEIGHT"), self.filename.fusion_weight
        )
        self.filename.synonyms_path = _to_str(
            os.getenv("FILENAME_SYNONYMS_PATH"), self.filename.synonyms_path
        )

        self.graph2d.enabled = _to_bool(
            os.getenv("GRAPH2D_ENABLED"), self.graph2d.enabled
        )
        self.graph2d.min_confidence = _to_float(
            os.getenv("GRAPH2D_MIN_CONF"), self.graph2d.min_confidence
        )
        self.graph2d.fusion_weight = _to_float(
            os.getenv("GRAPH2D_FUSION_WEIGHT"), self.graph2d.fusion_weight
        )
        # Backward-compatible env mapping:
        # - Prefer *_FUSION_* variables (newer naming)
        # - Fall back to GRAPH2D_EXCLUDE_LABELS / GRAPH2D_ALLOW_LABELS (legacy)
        exclude_env = os.getenv("GRAPH2D_FUSION_EXCLUDE_LABELS")
        if exclude_env is None:
            exclude_env = os.getenv("GRAPH2D_EXCLUDE_LABELS")
        allow_env = os.getenv("GRAPH2D_FUSION_ALLOW_LABELS")
        if allow_env is None:
            allow_env = os.getenv("GRAPH2D_ALLOW_LABELS")

        self.graph2d.exclude_labels = _to_str(exclude_env, self.graph2d.exclude_labels)
        self.graph2d.allow_labels = _to_str(allow_env, self.graph2d.allow_labels)
        labels_raw = os.getenv("GRAPH2D_DRAWING_TYPE_LABELS", "").strip()
        if labels_raw:
            self.graph2d.drawing_type_labels = [
                label.strip() for label in labels_raw.split(",") if label.strip()
            ]

        self.titleblock.enabled = _to_bool(
            os.getenv("TITLEBLOCK_ENABLED"), self.titleblock.enabled
        )
        self.titleblock.region_x_ratio = _to_float(
            os.getenv("TITLEBLOCK_REGION_X_RATIO"), self.titleblock.region_x_ratio
        )
        self.titleblock.region_y_ratio = _to_float(
            os.getenv("TITLEBLOCK_REGION_Y_RATIO"), self.titleblock.region_y_ratio
        )
        self.titleblock.min_confidence = _to_float(
            os.getenv("TITLEBLOCK_MIN_CONF"), self.titleblock.min_confidence
        )
        self.titleblock.fusion_weight = _to_float(
            os.getenv("TITLEBLOCK_FUSION_WEIGHT"), self.titleblock.fusion_weight
        )
        self.titleblock.override_enabled = _to_bool(
            os.getenv("TITLEBLOCK_OVERRIDE_ENABLED"), self.titleblock.override_enabled
        )

        self.process.enabled = _to_bool(
            os.getenv("PROCESS_FEATURES_ENABLED"), self.process.enabled
        )
        self.process.min_confidence = _to_float(
            os.getenv("PROCESS_MIN_CONF"), self.process.min_confidence
        )
        self.process.fusion_weight = _to_float(
            os.getenv("PROCESS_FUSION_WEIGHT"), self.process.fusion_weight
        )

        self.sampling.max_nodes = _to_int(
            os.getenv("DXF_MAX_NODES"), self.sampling.max_nodes
        )
        self.sampling.strategy = _to_str(
            os.getenv("DXF_SAMPLING_STRATEGY"), self.sampling.strategy
        )
        self.sampling.seed = _to_int(os.getenv("DXF_SAMPLING_SEED"), self.sampling.seed)
        self.sampling.text_priority_ratio = _to_float(
            os.getenv("DXF_TEXT_PRIORITY_RATIO"), self.sampling.text_priority_ratio
        )

        self.class_balance.strategy = _to_str(
            os.getenv("CLASS_BALANCE_STRATEGY"), self.class_balance.strategy
        )
        self.class_balance.weight_mode = _to_str(
            os.getenv("CLASS_WEIGHT_MODE"), self.class_balance.weight_mode
        )

        self.multimodal.enabled = _to_bool(
            os.getenv("MULTIMODAL_FUSION_ENABLED"), self.multimodal.enabled
        )

        self.distillation.enabled = _to_bool(
            os.getenv("DISTILLATION_ENABLED"), self.distillation.enabled
        )

    @classmethod
    def from_sources(
        cls, config_path: Optional[Path] = None
    ) -> "HybridClassifierConfig":
        """从配置文件与环境变量加载"""
        config = cls()
        if config_path is not None:
            if config_path.exists():
                if yaml is None:
                    logger.warning(
                        "pyyaml unavailable, skip config file: %s", config_path
                    )
                else:
                    try:
                        payload = (
                            yaml.safe_load(config_path.read_text(encoding="utf-8"))
                            or {}
                        )
                        if isinstance(payload, dict):
                            config.apply_dict(payload)
                    except Exception as exc:
                        logger.warning(
                            "Failed loading hybrid config %s: %s", config_path, exc
                        )
            else:
                logger.info(
                    "Hybrid config file not found, using defaults: %s", config_path
                )
        config.apply_env()
        return config

    @classmethod
    def from_env(cls) -> "HybridClassifierConfig":
        """兼容旧接口：仍支持仅按默认路径+环境变量加载"""
        config_path = Path(os.getenv("HYBRID_CONFIG_PATH", str(DEFAULT_CONFIG_PATH)))
        return cls.from_sources(config_path=config_path)


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
