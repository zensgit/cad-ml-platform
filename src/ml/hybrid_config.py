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
    """文件名分类器配置

    B4.4 升级说明 (2026-04-14):
      - fusion_weight: 0.7 → 0.45（Graph2D 提升后，降低文件名过度依赖）

    B5.8 升级说明 (2026-04-14):
      - fusion_weight: 0.45 → 0.50（v4 权重搜索最优：fn=0.50/g2d=0.40/txt=0.10）
    """

    enabled: bool = True
    min_confidence: float = 0.8
    exact_match_conf: float = 0.95
    partial_match_conf: float = 0.7
    fuzzy_match_conf: float = 0.5
    fusion_weight: float = 0.50  # B5.8: v4 三路融合最优（原为 0.45）
    synonyms_path: str = ""


@dataclass
class Graph2DConfig:
    """Graph2D 分类器配置

    B4.5 升级说明 (2026-04-14):
      - enabled: True（B4.4 GraphEncoderV2 模型已达 90.5% acc，正式启用）
      - fusion_weight: 0.3 → 0.50（模型准确率大幅提升，从旧 ~27% 到 90.5%）
      - min_confidence: 0.5 → 0.35（24 类分布下置信度分散，适当降低门槛）

    B5.0 升级说明 (2026-04-14):
      - v3 模型：数据增强后再训练，acc=91.0%，轴承座/阀门 recall 均达 100%
      - 模型路径通过环境变量 GRAPH2D_MODEL_PATH 配置：
          export GRAPH2D_MODEL_PATH=models/graph2d_finetuned_24class_v3.pth

    B5.1 升级说明 (2026-04-14):
      - fusion_weight: 0.50 → 0.35（三路融合权重搜索最优：fn=0.45/g2d=0.35/txt=0.10）

    B5.8 升级说明 (2026-04-14):
      - v4 模型：定向增强后再训练，acc=91.9%（+0.9pp vs v3）
      - fusion_weight: 0.35 → 0.40（v4 权重搜索最优：fn=0.50/g2d=0.40/txt=0.10 → avg=94.8%）
    """

    enabled: bool = True   # B4.4: 正式启用（原为 False）
    min_confidence: float = 0.35  # B4.4: 24类分布适当降低（原为 0.5）
    # Optional additional guardrail:
    # require a minimum margin between top-1 and top-2 probabilities.
    # This is disabled by default to avoid changing existing behavior.
    min_margin: float = 0.0
    fusion_weight: float = 0.40  # B5.8: v4 三路融合最优（原为 0.35）
    exclude_labels: str = "other"
    allow_labels: str = ""
    drawing_type_labels: List[str] = field(
        default_factory=lambda: list(DEFAULT_GRAPH2D_DRAWING_TYPE_LABELS)
    )


@dataclass
class TextContentConfig:
    """DXF 文字内容分类器配置（B5.1 新增）

    基于关键词匹配的 24 类文字融合分类器。
    当无关键词命中时，分类器主动放弃（返回空概率），不向融合引入噪声。

    B5.1 权重搜索结果 (2026-04-14):
      - 最优配置：fn=0.45, g2d=0.35, txt=0.10 → 综合 avg=94.1%
      - 文字命中覆盖：val set 仅 14.9% 样本有效命中（主类 法兰/轴类/箱体 命中率低）
      - 受益类别：换热器/罐体/过滤器/弹簧/筒体精度 100%（文字信号精准）
      - 推荐低权重（0.10）：避免稀疏文字噪声降低主类精度
    """

    enabled: bool = True
    fusion_weight: float = 0.10  # B5.1: 网格搜索最优（低权重避免噪声）
    min_text_len: int = 4        # 最短有效文字长度（字符数）


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
class HistorySequenceConfig:
    """历史命令序列特征配置（HPSketch）"""

    enabled: bool = False
    shadow_only: bool = False
    min_confidence: float = 0.55
    fusion_weight: float = 0.2
    prototypes_path: str = "data/knowledge/history_sequence_prototypes_template.json"
    model_path: str = ""
    prototype_token_weight: float = 1.0
    prototype_bigram_weight: float = 1.0


@dataclass
class RejectionConfig:
    """低置信度拒识策略"""

    enabled: bool = False
    min_confidence: float = 0.0


@dataclass
class AutoEnableConfig:
    """按证据自动启用分支"""

    titleblock_on_text: bool = True
    history_on_path: bool = True


@dataclass
class DecisionConfig:
    """融合与解释输出配置"""

    advanced_fusion_enabled: bool = True
    fusion_strategy: str = "weighted_average"
    auto_select_fusion: bool = False
    explanation_enabled: bool = True


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
    text_content: TextContentConfig = field(default_factory=TextContentConfig)
    titleblock: TitleBlockConfig = field(default_factory=TitleBlockConfig)
    process: ProcessConfig = field(default_factory=ProcessConfig)
    history_sequence: HistorySequenceConfig = field(
        default_factory=HistorySequenceConfig
    )
    rejection: RejectionConfig = field(default_factory=RejectionConfig)
    auto_enable: AutoEnableConfig = field(default_factory=AutoEnableConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
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
                "min_margin": self.graph2d.min_margin,
                "fusion_weight": self.graph2d.fusion_weight,
                "exclude_labels": self.graph2d.exclude_labels,
                "allow_labels": self.graph2d.allow_labels,
                "drawing_type_labels": list(self.graph2d.drawing_type_labels),
            },
            "text_content": {
                "enabled": self.text_content.enabled,
                "fusion_weight": self.text_content.fusion_weight,
                "min_text_len": self.text_content.min_text_len,
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
            "history_sequence": {
                "enabled": self.history_sequence.enabled,
                "shadow_only": self.history_sequence.shadow_only,
                "min_confidence": self.history_sequence.min_confidence,
                "fusion_weight": self.history_sequence.fusion_weight,
                "prototypes_path": self.history_sequence.prototypes_path,
                "model_path": self.history_sequence.model_path,
                "prototype_token_weight": self.history_sequence.prototype_token_weight,
                "prototype_bigram_weight": self.history_sequence.prototype_bigram_weight,
            },
            "rejection": {
                "enabled": self.rejection.enabled,
                "min_confidence": self.rejection.min_confidence,
            },
            "auto_enable": {
                "titleblock_on_text": self.auto_enable.titleblock_on_text,
                "history_on_path": self.auto_enable.history_on_path,
            },
            "decision": {
                "advanced_fusion_enabled": self.decision.advanced_fusion_enabled,
                "fusion_strategy": self.decision.fusion_strategy,
                "auto_select_fusion": self.decision.auto_select_fusion,
                "explanation_enabled": self.decision.explanation_enabled,
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
            self.graph2d.min_margin = _to_float(
                graph2d.get("min_margin"), self.graph2d.min_margin
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

        text_content = payload.get("text_content", {})
        if isinstance(text_content, dict):
            self.text_content.enabled = _to_bool(
                text_content.get("enabled"), self.text_content.enabled
            )
            self.text_content.fusion_weight = _to_float(
                text_content.get("fusion_weight"), self.text_content.fusion_weight
            )
            self.text_content.min_text_len = _to_int(
                text_content.get("min_text_len"), self.text_content.min_text_len
            )

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

        history_sequence = payload.get("history_sequence", {})
        if isinstance(history_sequence, dict):
            self.history_sequence.enabled = _to_bool(
                history_sequence.get("enabled"), self.history_sequence.enabled
            )
            self.history_sequence.shadow_only = _to_bool(
                history_sequence.get("shadow_only"),
                self.history_sequence.shadow_only,
            )
            self.history_sequence.min_confidence = _to_float(
                history_sequence.get("min_confidence"),
                self.history_sequence.min_confidence,
            )
            self.history_sequence.fusion_weight = _to_float(
                history_sequence.get("fusion_weight"),
                self.history_sequence.fusion_weight,
            )
            self.history_sequence.prototypes_path = _to_str(
                history_sequence.get("prototypes_path"),
                self.history_sequence.prototypes_path,
            )
            self.history_sequence.model_path = _to_str(
                history_sequence.get("model_path"),
                self.history_sequence.model_path,
            )
            self.history_sequence.prototype_token_weight = _to_float(
                history_sequence.get("prototype_token_weight"),
                self.history_sequence.prototype_token_weight,
            )
            self.history_sequence.prototype_bigram_weight = _to_float(
                history_sequence.get("prototype_bigram_weight"),
                self.history_sequence.prototype_bigram_weight,
            )

        rejection = payload.get("rejection", {})
        if isinstance(rejection, dict):
            self.rejection.enabled = _to_bool(
                rejection.get("enabled"), self.rejection.enabled
            )
            self.rejection.min_confidence = _to_float(
                rejection.get("min_confidence"),
                self.rejection.min_confidence,
            )

        auto_enable = payload.get("auto_enable", {})
        if isinstance(auto_enable, dict):
            self.auto_enable.titleblock_on_text = _to_bool(
                auto_enable.get("titleblock_on_text"),
                self.auto_enable.titleblock_on_text,
            )
            self.auto_enable.history_on_path = _to_bool(
                auto_enable.get("history_on_path"),
                self.auto_enable.history_on_path,
            )

        decision = payload.get("decision", {})
        if isinstance(decision, dict):
            self.decision.advanced_fusion_enabled = _to_bool(
                decision.get("advanced_fusion_enabled"),
                self.decision.advanced_fusion_enabled,
            )
            self.decision.fusion_strategy = _to_str(
                decision.get("fusion_strategy"),
                self.decision.fusion_strategy,
            )
            self.decision.auto_select_fusion = _to_bool(
                decision.get("auto_select_fusion"),
                self.decision.auto_select_fusion,
            )
            self.decision.explanation_enabled = _to_bool(
                decision.get("explanation_enabled"),
                self.decision.explanation_enabled,
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
        self.graph2d.min_margin = _to_float(
            os.getenv("GRAPH2D_MIN_MARGIN"), self.graph2d.min_margin
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

        self.text_content.enabled = _to_bool(
            os.getenv("TEXT_CONTENT_ENABLED"), self.text_content.enabled
        )
        self.text_content.fusion_weight = _to_float(
            os.getenv("TEXT_CONTENT_FUSION_WEIGHT"), self.text_content.fusion_weight
        )
        self.text_content.min_text_len = _to_int(
            os.getenv("TEXT_CONTENT_MIN_TEXT_LEN"), self.text_content.min_text_len
        )

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

        self.history_sequence.enabled = _to_bool(
            os.getenv("HISTORY_SEQUENCE_ENABLED"), self.history_sequence.enabled
        )
        self.history_sequence.shadow_only = _to_bool(
            os.getenv("HISTORY_SEQUENCE_SHADOW_ONLY"),
            self.history_sequence.shadow_only,
        )
        self.history_sequence.min_confidence = _to_float(
            os.getenv("HISTORY_SEQUENCE_MIN_CONF"),
            self.history_sequence.min_confidence,
        )
        self.history_sequence.fusion_weight = _to_float(
            os.getenv("HISTORY_SEQUENCE_FUSION_WEIGHT"),
            self.history_sequence.fusion_weight,
        )
        self.history_sequence.prototypes_path = _to_str(
            os.getenv("HISTORY_SEQUENCE_PROTOTYPES_PATH"),
            self.history_sequence.prototypes_path,
        )
        self.history_sequence.model_path = _to_str(
            os.getenv("HISTORY_SEQUENCE_MODEL_PATH"),
            self.history_sequence.model_path,
        )
        self.history_sequence.prototype_token_weight = _to_float(
            os.getenv("HISTORY_SEQUENCE_PROTOTYPE_TOKEN_WEIGHT"),
            self.history_sequence.prototype_token_weight,
        )
        self.history_sequence.prototype_bigram_weight = _to_float(
            os.getenv("HISTORY_SEQUENCE_PROTOTYPE_BIGRAM_WEIGHT"),
            self.history_sequence.prototype_bigram_weight,
        )

        self.rejection.enabled = _to_bool(
            os.getenv("HYBRID_REJECT_ENABLED"), self.rejection.enabled
        )
        self.rejection.min_confidence = _to_float(
            os.getenv("HYBRID_REJECT_MIN_CONFIDENCE"),
            self.rejection.min_confidence,
        )

        self.auto_enable.titleblock_on_text = _to_bool(
            os.getenv("TITLEBLOCK_AUTO_ENABLE"),
            self.auto_enable.titleblock_on_text,
        )
        self.auto_enable.history_on_path = _to_bool(
            os.getenv("HISTORY_SEQUENCE_AUTO_ENABLE"),
            self.auto_enable.history_on_path,
        )

        self.decision.advanced_fusion_enabled = _to_bool(
            os.getenv("HYBRID_ADVANCED_FUSION_ENABLED"),
            self.decision.advanced_fusion_enabled,
        )
        self.decision.fusion_strategy = _to_str(
            os.getenv("HYBRID_FUSION_STRATEGY"),
            self.decision.fusion_strategy,
        )
        self.decision.auto_select_fusion = _to_bool(
            os.getenv("HYBRID_AUTO_SELECT_FUSION"),
            self.decision.auto_select_fusion,
        )
        self.decision.explanation_enabled = _to_bool(
            os.getenv("HYBRID_EXPLANATION_ENABLED"),
            self.decision.explanation_enabled,
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
