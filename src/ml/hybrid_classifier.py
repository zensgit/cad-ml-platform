"""
HybridClassifier - 混合分类器

融合多种分类信号：
1. FilenameClassifier (文件名)
2. Graph2DClassifier (几何图神经网络)
3. TitleBlockClassifier (标题栏文本)
4. ProcessClassifier (工艺特征)
5. HistorySequenceClassifier (HPSketch 历史命令序列)

Feature Flags:
    HYBRID_CLASSIFIER_ENABLED: 是否启用混合分类 (default: true)
    FILENAME_CLASSIFIER_ENABLED: 是否启用文件名分类 (default: true)
    FILENAME_FUSION_WEIGHT: 文件名分类权重 (default: 0.7)
    GRAPH2D_FUSION_WEIGHT: Graph2D 分类权重 (default: 0.3)
    TITLEBLOCK_ENABLED: 是否启用标题栏特征 (default: false)
    TITLEBLOCK_OVERRIDE_ENABLED: 是否允许标题栏直接覆盖 (default: false)
    PROCESS_FEATURES_ENABLED: 是否启用工艺特征 (default: true)
    PROCESS_FUSION_WEIGHT: 工艺特征权重 (default: 0.15)
    HISTORY_SEQUENCE_ENABLED: 是否启用历史序列特征 (default: false)
    HISTORY_SEQUENCE_FUSION_WEIGHT: 历史序列权重 (default: 0.2)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.ml.hybrid_config import get_config

logger = logging.getLogger(__name__)

DEFAULT_GRAPH2D_DRAWING_LABELS = {
    "零件图",
    "机械制图",
    "装配图",
    "练习零件图",
    "原理图",
    "模板",
}


class DecisionSource(str, Enum):
    """决策来源"""

    FILENAME = "filename"
    GRAPH2D = "graph2d"
    TITLEBLOCK = "titleblock"
    PROCESS = "process"
    HISTORY = "history_sequence"
    FUSION = "fusion"
    FALLBACK = "fallback"


@dataclass
class ClassificationResult:
    """分类结果"""

    label: Optional[str] = None
    confidence: float = 0.0
    source: DecisionSource = DecisionSource.FALLBACK

    # 各分支预测
    filename_prediction: Optional[Dict[str, Any]] = None
    graph2d_prediction: Optional[Dict[str, Any]] = None
    titleblock_prediction: Optional[Dict[str, Any]] = None
    process_prediction: Optional[Dict[str, Any]] = None
    history_prediction: Optional[Dict[str, Any]] = None

    # 融合决策详情
    fusion_weights: Dict[str, float] = field(default_factory=dict)
    source_contributions: Dict[str, float] = field(default_factory=dict)
    fusion_metadata: Optional[Dict[str, Any]] = None
    decision_path: List[str] = field(default_factory=list)
    rejection: Optional[Dict[str, Any]] = None
    explanation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "source": self.source.value,
            "filename_prediction": self.filename_prediction,
            "graph2d_prediction": self.graph2d_prediction,
            "titleblock_prediction": self.titleblock_prediction,
            "process_prediction": self.process_prediction,
            "history_prediction": self.history_prediction,
            "fusion_weights": self.fusion_weights,
            "source_contributions": self.source_contributions,
            "fusion_metadata": self.fusion_metadata,
            "decision_path": self.decision_path,
            "rejection": self.rejection,
            "explanation": self.explanation,
        }


class HybridClassifier:
    """混合分类器"""

    def __init__(
        self,
        filename_weight: Optional[float] = None,
        graph2d_weight: Optional[float] = None,
        titleblock_weight: Optional[float] = None,
        process_weight: Optional[float] = None,
        history_weight: Optional[float] = None,
        filename_min_conf: Optional[float] = None,
        graph2d_min_conf: Optional[float] = None,
        titleblock_min_conf: Optional[float] = None,
        process_min_conf: Optional[float] = None,
        history_min_conf: Optional[float] = None,
    ):
        """
        初始化混合分类器

        Args:
            filename_weight: 文件名分类权重
            graph2d_weight: Graph2D 分类权重
            titleblock_weight: 标题栏分类权重
            process_weight: 工艺特征分类权重
            history_weight: 历史序列分类权重
            filename_min_conf: 文件名分类最低置信度（高于此值优先采用）
            graph2d_min_conf: Graph2D 分类最低置信度
            titleblock_min_conf: 标题栏分类最低置信度
            process_min_conf: 工艺特征分类最低置信度
            history_min_conf: 历史序列分类最低置信度
        """
        self._config = get_config()

        self.filename_weight = self._resolve_float(
            "FILENAME_FUSION_WEIGHT",
            explicit=filename_weight,
            default=self._config.filename.fusion_weight,
        )
        self.graph2d_weight = self._resolve_float(
            "GRAPH2D_FUSION_WEIGHT",
            explicit=graph2d_weight,
            default=self._config.graph2d.fusion_weight,
        )
        self.titleblock_weight = self._resolve_float(
            "TITLEBLOCK_FUSION_WEIGHT",
            explicit=titleblock_weight,
            default=self._config.titleblock.fusion_weight,
        )
        self.process_weight = self._resolve_float(
            "PROCESS_FUSION_WEIGHT",
            explicit=process_weight,
            default=self._config.process.fusion_weight,
        )
        self.history_weight = self._resolve_float(
            "HISTORY_SEQUENCE_FUSION_WEIGHT",
            explicit=history_weight,
            default=self._config.history_sequence.fusion_weight,
        )

        self.filename_min_conf = self._resolve_float(
            "FILENAME_MIN_CONF",
            explicit=filename_min_conf,
            default=self._config.filename.min_confidence,
        )
        self.graph2d_min_conf = self._resolve_float(
            "GRAPH2D_MIN_CONF",
            explicit=graph2d_min_conf,
            default=self._config.graph2d.min_confidence,
        )
        self.graph2d_min_margin = self._resolve_float(
            "GRAPH2D_MIN_MARGIN",
            explicit=None,
            default=getattr(self._config.graph2d, "min_margin", 0.0),
        )
        self.titleblock_min_conf = self._resolve_float(
            "TITLEBLOCK_MIN_CONF",
            explicit=titleblock_min_conf,
            default=self._config.titleblock.min_confidence,
        )
        self.process_min_conf = self._resolve_float(
            "PROCESS_MIN_CONF",
            explicit=process_min_conf,
            default=self._config.process.min_confidence,
        )
        self.history_min_conf = self._resolve_float(
            "HISTORY_SEQUENCE_MIN_CONF",
            explicit=history_min_conf,
            default=self._config.history_sequence.min_confidence,
        )
        self.reject_enabled = self._resolve_bool(
            "HYBRID_REJECT_ENABLED",
            getattr(self._config.rejection, "enabled", False),
        )
        self.reject_min_conf = self._resolve_float(
            "HYBRID_REJECT_MIN_CONFIDENCE",
            explicit=None,
            default=getattr(self._config.rejection, "min_confidence", 0.0),
        )
        self.auto_enable_titleblock = self._resolve_bool(
            "TITLEBLOCK_AUTO_ENABLE",
            getattr(self._config.auto_enable, "titleblock_on_text", True),
        )
        self.auto_enable_history = self._resolve_bool(
            "HISTORY_SEQUENCE_AUTO_ENABLE",
            getattr(self._config.auto_enable, "history_on_path", True),
        )
        self.advanced_fusion_enabled = self._resolve_bool(
            "HYBRID_ADVANCED_FUSION_ENABLED",
            getattr(self._config.decision, "advanced_fusion_enabled", True),
        )
        self.auto_select_fusion = self._resolve_bool(
            "HYBRID_AUTO_SELECT_FUSION",
            getattr(self._config.decision, "auto_select_fusion", False),
        )
        self.explanation_enabled = self._resolve_bool(
            "HYBRID_EXPLANATION_ENABLED",
            getattr(self._config.decision, "explanation_enabled", True),
        )
        self.fusion_strategy_name = self._resolve_str(
            "HYBRID_FUSION_STRATEGY",
            getattr(self._config.decision, "fusion_strategy", "weighted_average"),
        ) or "weighted_average"

        self.titleblock_override_enabled = self._resolve_bool(
            "TITLEBLOCK_OVERRIDE_ENABLED", self._config.titleblock.override_enabled
        )
        drawing_labels_raw = os.getenv("GRAPH2D_DRAWING_TYPE_LABELS", "").strip()
        if drawing_labels_raw:
            self.graph2d_drawing_labels = {
                label.strip()
                for label in drawing_labels_raw.split(",")
                if label.strip()
            }
        else:
            self.graph2d_drawing_labels = set(self._config.graph2d.drawing_type_labels)

        self.graph2d_exclude_labels = self._parse_label_set(
            self._config.graph2d.exclude_labels
        )
        self.graph2d_allow_labels = self._parse_label_set(
            self._config.graph2d.allow_labels
        )

        # 懒加载分类器
        self._filename_classifier = None
        self._graph2d_classifier = None
        self._titleblock_classifier = None
        self._process_classifier = None
        self._history_sequence_classifier = None
        self._fusion_manager = None
        self._explainer = None

        logger.info(
            "HybridClassifier initialized",
            extra={
                "filename_weight": self.filename_weight,
                "graph2d_weight": self.graph2d_weight,
                "titleblock_weight": self.titleblock_weight,
                "process_weight": self.process_weight,
                "history_weight": self.history_weight,
                "filename_min_conf": self.filename_min_conf,
                "graph2d_min_conf": self.graph2d_min_conf,
                "titleblock_min_conf": self.titleblock_min_conf,
                "process_min_conf": self.process_min_conf,
                "history_min_conf": self.history_min_conf,
                "reject_enabled": self.reject_enabled,
                "reject_min_conf": self.reject_min_conf,
                "auto_enable_titleblock": self.auto_enable_titleblock,
                "auto_enable_history": self.auto_enable_history,
                "advanced_fusion_enabled": self.advanced_fusion_enabled,
                "fusion_strategy": self.fusion_strategy_name,
                "auto_select_fusion": self.auto_select_fusion,
                "explanation_enabled": self.explanation_enabled,
                "titleblock_override_enabled": self.titleblock_override_enabled,
                "graph2d_drawing_labels": sorted(self.graph2d_drawing_labels),
                "graph2d_exclude_labels": sorted(self.graph2d_exclude_labels),
                "graph2d_allow_labels": sorted(self.graph2d_allow_labels),
            },
        )

    @property
    def filename_classifier(self):
        """懒加载 FilenameClassifier"""
        if self._filename_classifier is None:
            from src.ml.filename_classifier import FilenameClassifier

            synonyms_path = self._resolve_str(
                "FILENAME_SYNONYMS_PATH",
                self._config.filename.synonyms_path,
            )
            self._filename_classifier = FilenameClassifier(
                synonyms_path=synonyms_path or None
            )
        return self._filename_classifier

    @property
    def graph2d_classifier(self):
        """懒加载 Graph2DClassifier"""
        if self._graph2d_classifier is None:
            try:
                graph2d_ensemble_enabled = (
                    os.getenv("GRAPH2D_ENSEMBLE_ENABLED", "false").strip().lower()
                    == "true"
                )
                if graph2d_ensemble_enabled:
                    from src.ml.vision_2d import get_ensemble_2d_classifier

                    self._graph2d_classifier = get_ensemble_2d_classifier()
                else:
                    from src.ml.vision_2d import get_2d_classifier

                    self._graph2d_classifier = get_2d_classifier()
            except Exception as e:
                logger.warning(f"Graph2D classifier not available: {e}")
                self._graph2d_classifier = None
        return self._graph2d_classifier

    @property
    def titleblock_classifier(self):
        """懒加载 TitleBlockClassifier"""
        if self._titleblock_classifier is None:
            try:
                from src.ml.titleblock_extractor import get_titleblock_classifier

                self._titleblock_classifier = get_titleblock_classifier()
            except Exception as e:
                logger.warning("TitleBlock classifier not available: %s", e)
                self._titleblock_classifier = None
        return self._titleblock_classifier

    @property
    def process_classifier(self):
        """懒加载 ProcessClassifier"""
        if self._process_classifier is None:
            try:
                from src.ml.process_classifier import get_process_classifier

                self._process_classifier = get_process_classifier()
            except Exception as e:
                logger.warning("Process classifier not available: %s", e)
                self._process_classifier = None
        return self._process_classifier

    @property
    def history_sequence_classifier(self):
        """懒加载 HistorySequenceClassifier"""
        if self._history_sequence_classifier is None:
            try:
                from src.ml.history_sequence_classifier import HistorySequenceClassifier

                prototypes_path = self._resolve_str(
                    "HISTORY_SEQUENCE_PROTOTYPES_PATH",
                    self._config.history_sequence.prototypes_path,
                )
                model_path = self._resolve_str(
                    "HISTORY_SEQUENCE_MODEL_PATH",
                    self._config.history_sequence.model_path,
                )
                token_weight = self._resolve_float(
                    "HISTORY_SEQUENCE_PROTOTYPE_TOKEN_WEIGHT",
                    explicit=None,
                    default=self._config.history_sequence.prototype_token_weight,
                )
                bigram_weight = self._resolve_float(
                    "HISTORY_SEQUENCE_PROTOTYPE_BIGRAM_WEIGHT",
                    explicit=None,
                    default=self._config.history_sequence.prototype_bigram_weight,
                )
                self._history_sequence_classifier = HistorySequenceClassifier(
                    prototypes_path=prototypes_path or None,
                    model_path=model_path or None,
                    prototype_token_weight=token_weight,
                    prototype_bigram_weight=bigram_weight,
                )
            except Exception as e:
                logger.warning("History sequence classifier not available: %s", e)
                self._history_sequence_classifier = None
        return self._history_sequence_classifier

    @property
    def fusion_manager(self):
        """懒加载多源融合管理器"""
        if self._fusion_manager is None:
            try:
                from src.ml.hybrid.fusion import FusionStrategy, MultiSourceFusion

                strategy = self._resolve_fusion_strategy(self.fusion_strategy_name)
                self._fusion_manager = MultiSourceFusion(
                    default_strategy=strategy,
                    auto_select=self.auto_select_fusion,
                )
            except Exception as e:
                logger.warning("Advanced fusion manager not available: %s", e)
                self._fusion_manager = None
        return self._fusion_manager

    @property
    def explainer(self):
        """懒加载解释器"""
        if self._explainer is None:
            try:
                from src.ml.hybrid.explainer import HybridExplainer

                self._explainer = HybridExplainer(include_counterfactuals=False)
            except Exception as e:
                logger.warning("Hybrid explainer not available: %s", e)
                self._explainer = None
        return self._explainer

    def _is_filename_enabled(self) -> bool:
        """检查文件名分类是否启用"""
        return self._resolve_bool(
            "FILENAME_CLASSIFIER_ENABLED",
            self._config.filename.enabled,
        )

    def _is_graph2d_enabled(self) -> bool:
        """检查 Graph2D 分类是否启用"""
        return self._resolve_bool("GRAPH2D_ENABLED", self._config.graph2d.enabled)

    def _is_hybrid_enabled(self) -> bool:
        """检查混合分类是否启用"""
        return self._resolve_bool("HYBRID_CLASSIFIER_ENABLED", self._config.enabled)

    def _is_titleblock_enabled(self) -> bool:
        """检查标题栏特征是否启用"""
        return self._resolve_bool("TITLEBLOCK_ENABLED", self._config.titleblock.enabled)

    def _is_process_enabled(self) -> bool:
        """检查工艺特征分类是否启用"""
        return self._resolve_bool(
            "PROCESS_FEATURES_ENABLED", self._config.process.enabled
        )

    def _is_history_enabled(self) -> bool:
        """检查历史命令序列分类是否启用"""
        return self._resolve_bool(
            "HISTORY_SEQUENCE_ENABLED",
            self._config.history_sequence.enabled,
        )

    @staticmethod
    def _has_text_entities(entities: Optional[List[Any]]) -> bool:
        if not entities:
            return False
        for entity in entities:
            try:
                dtype = entity.dxftype()
            except Exception:
                continue
            if dtype in {"TEXT", "MTEXT", "DIMENSION", "ATTRIB"}:
                return True
            if dtype == "INSERT" and getattr(entity, "attribs", None):
                return True
        return False

    def _should_attempt_titleblock(self, entities: Optional[List[Any]]) -> bool:
        if entities is None:
            return False
        if self._is_titleblock_enabled():
            return True
        return self.auto_enable_titleblock and self._has_text_entities(entities)

    def _should_attempt_history(self, history_file_path: Optional[str]) -> bool:
        if not history_file_path:
            return False
        if self._is_history_enabled():
            return True
        return self.auto_enable_history

    @staticmethod
    def _resolve_bool(env_key: str, default: bool) -> bool:
        raw = os.getenv(env_key)
        if raw is None:
            return bool(default)
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _resolve_float(
        env_key: str, explicit: Optional[float], default: float
    ) -> float:
        if explicit is not None:
            base = explicit
        else:
            base = default
        raw = os.getenv(env_key)
        if raw is None:
            return float(base)
        try:
            return float(raw)
        except ValueError:
            logger.warning("Invalid %s=%s, fallback to %s", env_key, raw, base)
            return float(base)

    @staticmethod
    def _resolve_str(env_key: str, default: str) -> str:
        raw = os.getenv(env_key)
        if raw is None:
            return default
        return raw.strip()

    @staticmethod
    def _resolve_fusion_strategy(name: str):
        from src.ml.hybrid.fusion import FusionStrategy

        normalized = str(name or "").strip().lower()
        for strategy in FusionStrategy:
            if strategy.value == normalized:
                return strategy
        return FusionStrategy.WEIGHTED_AVERAGE

    def _apply_advanced_fusion(
        self,
        preds: List[tuple[str, str, str, float, DecisionSource]],
    ) -> Optional[Dict[str, Any]]:
        manager = self.fusion_manager
        if manager is None:
            return None

        try:
            from src.ml.hybrid.fusion import SourcePrediction

            weights = {
                "filename": self.filename_weight,
                "graph2d": self.graph2d_weight,
                "titleblock": self.titleblock_weight,
                "process": self.process_weight,
                "history_sequence": self.history_weight,
            }
            predictions = [
                SourcePrediction(
                    source_name=src_key,
                    label=label_norm,
                    confidence=max(0.0, float(conf)),
                    metadata={
                        "label_raw": label_raw,
                        "decision_source": src.value,
                    },
                )
                for src_key, label_raw, label_norm, conf, src in preds
                if label_norm and float(conf) > 0.0
            ]
            if len(predictions) < 2:
                return None

            strategy = None
            if not self.auto_select_fusion:
                strategy = self._resolve_fusion_strategy(self.fusion_strategy_name)

            fused = manager.fuse(predictions, weights=weights, strategy=strategy)
            payload = fused.to_dict()
            payload["num_sources"] = len(predictions)
            return payload
        except Exception as e:
            logger.warning("Advanced fusion failed, fallback to manual scoring: %s", e)
            return None

    def _attach_explanation(self, result: ClassificationResult) -> None:
        if not self.explanation_enabled:
            return
        explainer = self.explainer
        if explainer is None:
            return
        try:
            explanation = explainer.explain(result, detailed=True)
            result.explanation = explanation.to_dict()
        except Exception as e:
            logger.warning("Hybrid explanation generation failed: %s", e)

    @staticmethod
    def _normalize_label(label: str) -> str:
        text = str(label or "").strip()
        if not text:
            return ""
        if all(ord(ch) < 128 for ch in text):
            return text.lower()
        return text

    @classmethod
    def _parse_label_set(cls, raw: str) -> set[str]:
        if not raw:
            return set()
        tokens = [t.strip() for t in re.split(r"[,\s]+", str(raw)) if t.strip()]
        normalized = {cls._normalize_label(t) for t in tokens}
        return {t for t in normalized if t}

    def _is_graph2d_drawing_type(self, label: Optional[str]) -> bool:
        if not label:
            return False
        return label.strip() in self.graph2d_drawing_labels

    def classify(
        self,
        filename: str,
        file_bytes: Optional[bytes] = None,
        graph2d_result: Optional[Dict[str, Any]] = None,
        history_result: Optional[Dict[str, Any]] = None,
        history_file_path: Optional[str] = None,
    ) -> ClassificationResult:
        """
        执行混合分类

        决策逻辑：
        1. 文件名高置信度 (>= filename_min_conf) → 直接采用
        2. Graph2D 高置信度 (>= graph2d_min_conf) 且文件名低置信度 → 采用 Graph2D
        3. 两者都有预测 → 加权融合
        4. 其他 → 返回可用的预测或 fallback

        Args:
            filename: 文件名
            file_bytes: 文件字节内容（用于 Graph2D）
            graph2d_result: 预计算的 Graph2D 结果（可选）
            history_result: 预计算的历史序列预测结果（可选）
            history_file_path: 历史序列 `.h5` 路径（可选）

        Returns:
            ClassificationResult
        """
        result = ClassificationResult()
        result.decision_path = []

        # 1. 文件名分类
        filename_pred = None
        if self._is_filename_enabled():
            try:
                filename_pred = self.filename_classifier.predict(filename)
                result.filename_prediction = filename_pred
                result.decision_path.append("filename_extracted")
            except Exception as e:
                logger.error(f"Filename classification failed: {e}")
                result.decision_path.append("filename_error")

        # 2. Graph2D 分类
        graph2d_pred = graph2d_result
        if graph2d_pred is None and self._is_graph2d_enabled() and file_bytes:
            try:
                classifier = self.graph2d_classifier
                if classifier:
                    graph2d_pred = classifier.predict_from_bytes(file_bytes, filename)
                    result.decision_path.append("graph2d_predicted")
            except Exception as e:
                logger.error(f"Graph2D classification failed: {e}")
                result.decision_path.append("graph2d_error")

        graph2d_label_raw = graph2d_pred.get("label") if graph2d_pred else None
        graph2d_conf_raw = (
            float(graph2d_pred.get("confidence", 0)) if graph2d_pred else 0.0
        )
        graph2d_margin_raw: Optional[float] = None
        if graph2d_pred:
            try:
                if graph2d_pred.get("margin") is not None:
                    graph2d_margin_raw = float(graph2d_pred.get("margin"))
            except Exception:
                graph2d_margin_raw = None
        graph2d_is_drawing_type = self._is_graph2d_drawing_type(graph2d_label_raw)
        if graph2d_pred:
            graph2d_pred = dict(graph2d_pred)
            graph2d_pred["is_drawing_type"] = graph2d_is_drawing_type
            result.graph2d_prediction = graph2d_pred

        # 3. TitleBlock/Process 共享 DXF 解析（避免重复读文件）
        dxf_entities: Optional[List[Any]] = None
        if (
            self._is_titleblock_enabled()
            or self._is_process_enabled()
            or self.auto_enable_titleblock
        ) and file_bytes:
            try:
                from src.utils.dxf_io import read_dxf_entities_from_bytes

                dxf_entities = read_dxf_entities_from_bytes(file_bytes)
            except Exception as e:
                logger.warning("DXF parse failed for hybrid classifiers: %s", e)
                result.decision_path.append("dxf_parse_error")

        # 3. TitleBlock 分类
        titleblock_pred = None
        if self._should_attempt_titleblock(dxf_entities):
            try:
                if not self._is_titleblock_enabled():
                    result.decision_path.append("titleblock_auto_enabled")
                classifier = self.titleblock_classifier
                if classifier:
                    titleblock_pred = classifier.predict(dxf_entities)
                    result.decision_path.append("titleblock_predicted")
            except Exception as e:
                logger.warning("TitleBlock classification failed: %s", e)
                result.decision_path.append("titleblock_error")

        if titleblock_pred:
            result.titleblock_prediction = titleblock_pred

        # 4. Process 特征分类
        process_pred = None
        process_label = None
        process_conf = 0.0
        if self._is_process_enabled() and dxf_entities is not None:
            try:
                texts = []
                for entity in dxf_entities:
                    dtype = entity.dxftype()
                    if dtype == "TEXT":
                        texts.append(entity.dxf.text)
                    elif dtype == "MTEXT":
                        texts.append(entity.text)
                    elif dtype == "ATTRIB":
                        texts.append(entity.dxf.text)
                for entity in dxf_entities:
                    if entity.dxftype() == "INSERT":
                        for attrib in getattr(entity, "attribs", []):
                            texts.append(attrib.dxf.text)
                combined_text = "\n".join(texts)

                if combined_text.strip():
                    classifier = self.process_classifier
                    if classifier:
                        proc_result = classifier.predict_from_text(combined_text)
                        if (
                            proc_result.suggested_labels
                            and proc_result.confidence >= self.process_min_conf
                        ):
                            process_pred = proc_result.to_dict()
                            process_label = proc_result.suggested_labels[0]
                            process_conf = proc_result.confidence
                            result.decision_path.append("process_predicted")
            except Exception as e:
                logger.warning("Process classification failed: %s", e)
                result.decision_path.append("process_error")

        if process_pred:
            result.process_prediction = process_pred

        # 5. History sequence 分类（HPSketch .h5）
        history_pred = history_result
        if history_pred is None and self._should_attempt_history(history_file_path):
            try:
                if not self._is_history_enabled():
                    result.decision_path.append("history_auto_enabled")
                classifier = self.history_sequence_classifier
                if classifier:
                    history_pred = classifier.predict_from_h5_file(history_file_path)
                    result.decision_path.append("history_predicted")
            except Exception as e:
                logger.warning("History sequence classification failed: %s", e)
                result.decision_path.append("history_error")
        if history_pred:
            result.history_prediction = history_pred

        # 6. 融合决策
        filename_label_raw = filename_pred.get("label") if filename_pred else None
        filename_conf = (
            float(filename_pred.get("confidence", 0)) if filename_pred else 0.0
        )
        filename_label = (
            self._normalize_label(str(filename_label_raw))
            if filename_label_raw
            else None
        )

        graph2d_label_raw_final = graph2d_label_raw
        graph2d_label = graph2d_label_raw
        graph2d_conf = graph2d_conf_raw
        if graph2d_is_drawing_type:
            result.decision_path.append("graph2d_drawing_type_ignored")
            if result.graph2d_prediction is not None:
                result.graph2d_prediction["filtered"] = True
                result.graph2d_prediction["filtered_reason"] = "drawing_type"
            graph2d_label = None
            graph2d_label_raw_final = None
            graph2d_conf = 0.0
        else:
            graph2d_label = (
                self._normalize_label(str(graph2d_label)) if graph2d_label else None
            )

        if graph2d_label:
            if graph2d_label in self.graph2d_exclude_labels:
                result.decision_path.append("graph2d_excluded_label_ignored")
                if result.graph2d_prediction is not None:
                    result.graph2d_prediction["filtered"] = True
                    result.graph2d_prediction["filtered_reason"] = "excluded_label"
                graph2d_label = None
                graph2d_label_raw_final = None
                graph2d_conf = 0.0
            elif self.graph2d_allow_labels and (
                graph2d_label not in self.graph2d_allow_labels
            ):
                result.decision_path.append("graph2d_not_in_allowlist_ignored")
                if result.graph2d_prediction is not None:
                    result.graph2d_prediction["filtered"] = True
                    result.graph2d_prediction["filtered_reason"] = "not_in_allowlist"
                graph2d_label = None
                graph2d_label_raw_final = None
                graph2d_conf = 0.0

        # Graph2D confidence is sensitive to class count (many-class softmax tends to
        # produce lower max probabilities). When class-count metadata is available,
        # apply a conservative dynamic lower bound to avoid filtering everything.
        effective_graph2d_min_conf = self.graph2d_min_conf
        if result.graph2d_prediction is not None:
            try:
                label_map_size = int(
                    result.graph2d_prediction.get("label_map_size") or 0
                )
            except Exception:
                label_map_size = 0
            if label_map_size >= 20:
                uniform = 1.0 / max(1, label_map_size)
                dynamic_min = max(3.0 * uniform, 0.05)
                effective_graph2d_min_conf = min(
                    effective_graph2d_min_conf, dynamic_min
                )
                result.graph2d_prediction["min_confidence_effective"] = (
                    effective_graph2d_min_conf
                )

        if graph2d_label and graph2d_conf < effective_graph2d_min_conf:
            result.decision_path.append("graph2d_below_min_conf_ignored")
            if result.graph2d_prediction is not None:
                result.graph2d_prediction["filtered"] = True
                result.graph2d_prediction["filtered_reason"] = "below_min_conf"
            graph2d_label = None
            graph2d_label_raw_final = None
            graph2d_conf = 0.0

        effective_graph2d_min_margin = self.graph2d_min_margin
        if (
            graph2d_label
            and graph2d_margin_raw is not None
            and graph2d_margin_raw < effective_graph2d_min_margin
        ):
            result.decision_path.append("graph2d_below_min_margin_ignored")
            if result.graph2d_prediction is not None:
                result.graph2d_prediction["filtered"] = True
                result.graph2d_prediction["filtered_reason"] = "below_min_margin"
                result.graph2d_prediction["min_margin_effective"] = (
                    effective_graph2d_min_margin
                )
            graph2d_label = None
            graph2d_label_raw_final = None
            graph2d_conf = 0.0

        titleblock_label_raw = titleblock_pred.get("label") if titleblock_pred else None
        titleblock_label = (
            self._normalize_label(str(titleblock_label_raw))
            if titleblock_label_raw
            else None
        )
        titleblock_conf = (
            float(titleblock_pred.get("confidence", 0.0)) if titleblock_pred else 0.0
        )
        history_label_raw = history_pred.get("label") if history_pred else None
        history_label = (
            self._normalize_label(str(history_label_raw)) if history_label_raw else None
        )
        history_conf = (
            float(history_pred.get("confidence", 0.0)) if history_pred else 0.0
        )
        if history_label and history_conf < self.history_min_conf:
            result.decision_path.append("history_below_min_conf_ignored")
            if result.history_prediction is not None:
                result.history_prediction = dict(result.history_prediction)
                result.history_prediction["filtered"] = True
                result.history_prediction["filtered_reason"] = "below_min_conf"
                result.history_prediction["min_confidence_effective"] = (
                    self.history_min_conf
                )
            history_label = None
            history_conf = 0.0

        # ProcessClassifier often predicts drawing-type labels (e.g. 零件图/装配图).
        # Treat these as auxiliary signals: they can be used when no other label
        # exists, but should not compete with part-name labels from filename/titleblock/graph2d.
        process_label_raw = process_label
        process_is_drawing_type = bool(
            process_label_raw and self._is_graph2d_drawing_type(process_label_raw)
        )
        process_label_normalized = (
            self._normalize_label(str(process_label_raw)) if process_label_raw else None
        )
        process_label_for_fusion = process_label
        process_conf_for_fusion = process_conf

        result.fusion_weights = {
            "filename": self.filename_weight,
            "graph2d": self.graph2d_weight,
            "titleblock": self.titleblock_weight,
            "process": self.process_weight,
            "history_sequence": self.history_weight,
        }

        if titleblock_label and filename_label and titleblock_label != filename_label:
            result.decision_path.append("titleblock_filename_conflict")
            if filename_conf >= self.filename_min_conf:
                result.decision_path.append("titleblock_ignored_filename_high_conf")

        other_labels = {
            self._normalize_label(label)
            for label in (
                filename_label,
                titleblock_label,
                history_label,
                # Only treat process labels as comparable when they're not drawing types.
                process_label_normalized if not process_is_drawing_type else None,
            )
            if label
        }
        if graph2d_label and other_labels and graph2d_label not in other_labels:
            # Guardrail: Graph2D cannot introduce a new label when rules/text found one.
            result.decision_path.append("graph2d_non_matching_ignored")
            if result.graph2d_prediction is not None:
                result.graph2d_prediction["ignored_for_fusion"] = True
                result.graph2d_prediction["ignored_reason"] = "non_matching"
            graph2d_label = None
            graph2d_label_raw_final = None
            graph2d_conf = 0.0

        if process_is_drawing_type and (
            filename_label or titleblock_label or graph2d_label
        ):
            result.decision_path.append("process_drawing_type_ignored_for_fusion")
            process_label_for_fusion = None
            process_label_normalized = None
            process_conf_for_fusion = 0.0

        # 决策逻辑
        if filename_label and filename_conf >= self.filename_min_conf:
            # 文件名高置信度，直接采用
            result.label = str(filename_label_raw or filename_label)
            result.confidence = filename_conf
            result.source = DecisionSource.FILENAME
            result.source_contributions = {
                "filename": float(filename_conf * self.filename_weight)
            }
            result.fusion_metadata = {
                "strategy": "direct_threshold",
                "selected_by": "filename",
                "agreement_score": 1.0,
                "num_sources": 1,
            }
            result.decision_path.append("filename_high_conf_adopted")

        elif (
            self.titleblock_override_enabled
            and titleblock_label
            and titleblock_conf >= self.titleblock_min_conf
            and filename_conf < self.filename_min_conf
        ):
            result.label = str(titleblock_label_raw or titleblock_label)
            result.confidence = titleblock_conf
            result.source = DecisionSource.TITLEBLOCK
            result.source_contributions = {
                "titleblock": float(titleblock_conf * self.titleblock_weight)
            }
            result.fusion_metadata = {
                "strategy": "direct_threshold",
                "selected_by": "titleblock",
                "agreement_score": 1.0,
                "num_sources": 1,
            }
            result.decision_path.append("titleblock_adopted")

        elif (
            history_label
            and history_conf >= self.history_min_conf
            and filename_conf < self.filename_min_conf
        ):
            result.label = str(history_label_raw or history_label)
            result.confidence = history_conf
            result.source = DecisionSource.HISTORY
            result.source_contributions = {
                "history_sequence": float(history_conf * self.history_weight)
            }
            result.fusion_metadata = {
                "strategy": "direct_threshold",
                "selected_by": "history_sequence",
                "agreement_score": 1.0,
                "num_sources": 1,
            }
            result.decision_path.append("history_high_conf_adopted")

        else:
            preds: List[tuple[str, str, str, float, DecisionSource]] = []
            if filename_label:
                preds.append(
                    (
                        "filename",
                        str(filename_label_raw or filename_label),
                        filename_label,
                        filename_conf,
                        DecisionSource.FILENAME,
                    )
                )
            if titleblock_label:
                preds.append(
                    (
                        "titleblock",
                        str(titleblock_label_raw or titleblock_label),
                        titleblock_label,
                        titleblock_conf,
                        DecisionSource.TITLEBLOCK,
                    )
                )
            if process_label_for_fusion:
                preds.append(
                    (
                        "process",
                        str(process_label_raw or process_label_for_fusion),
                        str(process_label_normalized or ""),
                        process_conf_for_fusion,
                        DecisionSource.PROCESS,
                    )
                )
            if history_label:
                preds.append(
                    (
                        "history_sequence",
                        str(history_label_raw or history_label),
                        history_label,
                        history_conf,
                        DecisionSource.HISTORY,
                    )
                )
            if graph2d_label:
                preds.append(
                    (
                        "graph2d",
                        str(graph2d_label_raw_final or graph2d_label),
                        graph2d_label,
                        graph2d_conf,
                        DecisionSource.GRAPH2D,
                    )
                )

            if len(preds) == 1:
                src_key, label_raw, _, conf, src = preds[0]
                result.label = label_raw
                result.confidence = conf
                result.source = src
                result.source_contributions = {
                    src_key: float(conf * result.fusion_weights.get(src_key, 1.0))
                }
                result.fusion_metadata = {
                    "strategy": "single_source",
                    "selected_by": src_key,
                    "agreement_score": 1.0,
                    "num_sources": 1,
                }
                result.decision_path.append(f"{src_key}_only")

            elif preds:
                label_display: Dict[str, str] = {}
                for _, label_raw, label_norm, _, _ in preds:
                    if label_norm and label_norm not in label_display:
                        label_display[label_norm] = label_raw

                advanced_fused = None
                if self.advanced_fusion_enabled:
                    advanced_fused = self._apply_advanced_fusion(preds)

                if advanced_fused and advanced_fused.get("label"):
                    best_label_norm = str(advanced_fused.get("label") or "").strip()
                    result.label = label_display.get(best_label_norm, best_label_norm)
                    result.confidence = min(
                        1.0,
                        max(0.0, float(advanced_fused.get("confidence", 0.0) or 0.0)),
                    )
                    result.source = DecisionSource.FUSION
                    result.source_contributions = {
                        str(key): float(value)
                        for key, value in (
                            advanced_fused.get("source_contributions") or {}
                        ).items()
                    }
                    result.fusion_metadata = {
                        "strategy": str(
                            advanced_fused.get(
                                "fusion_strategy", self.fusion_strategy_name
                            )
                        ),
                        "agreement_score": float(
                            advanced_fused.get("agreement_score", 0.0) or 0.0
                        ),
                        "probabilities": dict(
                            advanced_fused.get("probabilities") or {}
                        ),
                        "num_sources": int(advanced_fused.get("num_sources", len(preds))),
                        "metadata": dict(advanced_fused.get("metadata") or {}),
                    }
                    result.decision_path.append("fusion_scored")
                    result.decision_path.append(
                        f"fusion_engine_{result.fusion_metadata['strategy']}"
                    )
                    if result.fusion_metadata["agreement_score"] >= 0.5:
                        result.decision_path.append("fusion_multi_source_bonus")
                        result.decision_path.append("fusion_high_agreement")

                else:
                    # 多源融合 (filename/graph2d/titleblock/process)
                    label_scores: Dict[str, float] = {}
                    label_support_confs: Dict[str, List[float]] = {}

                    def _add_score(
                        label_norm: Optional[str],
                        label_raw: Optional[str],
                        conf: float,
                        weight: float,
                        source: str,
                    ) -> None:
                        if not label_norm:
                            return
                        score = conf * weight
                        label_scores[label_norm] = label_scores.get(label_norm, 0.0) + score
                        label_support_confs.setdefault(label_norm, []).append(float(conf))
                        result.source_contributions[source] = float(score)
                        if label_raw and label_norm not in label_display:
                            label_display[label_norm] = str(label_raw)

                    _add_score(
                        filename_label,
                        str(filename_label_raw or filename_label),
                        filename_conf,
                        self.filename_weight,
                        "filename",
                    )
                    _add_score(
                        graph2d_label,
                        str(graph2d_label_raw_final or graph2d_label),
                        graph2d_conf,
                        self.graph2d_weight,
                        "graph2d",
                    )
                    _add_score(
                        titleblock_label,
                        str(titleblock_label_raw or titleblock_label),
                        titleblock_conf,
                        self.titleblock_weight,
                        "titleblock",
                    )
                    _add_score(
                        process_label_normalized,
                        str(process_label_raw or process_label_for_fusion),
                        process_conf_for_fusion,
                        self.process_weight,
                        "process",
                    )
                    _add_score(
                        history_label,
                        str(history_label_raw or history_label),
                        history_conf,
                        self.history_weight,
                        "history_sequence",
                    )

                    if label_scores:
                        best_label_norm = max(
                            label_scores.items(), key=lambda item: item[1]
                        )[0]
                        support_confs = label_support_confs.get(best_label_norm, [])
                        base_conf = max(support_confs) if support_confs else 0.0
                        bonus = (
                            min(0.1, 0.05 * (len(support_confs) - 1))
                            if len(support_confs) >= 2
                            else 0.0
                        )
                        fused_conf = min(1.0, float(base_conf) + float(bonus))
                        result.label = label_display.get(best_label_norm, best_label_norm)
                        result.confidence = fused_conf
                        result.source = DecisionSource.FUSION
                        result.fusion_metadata = {
                            "strategy": "manual_weighted",
                            "agreement_score": (
                                len(support_confs) / len(preds) if preds else 0.0
                            ),
                            "num_sources": len(preds),
                        }
                        result.decision_path.append("fusion_scored")
                        if bonus > 0:
                            result.decision_path.append("fusion_multi_source_bonus")

            else:
                # 无预测
                result.source = DecisionSource.FALLBACK
                result.fusion_metadata = {
                    "strategy": "none",
                    "agreement_score": 0.0,
                    "num_sources": 0,
                }
                result.decision_path.append("no_prediction")

        if (
            self.reject_enabled
            and result.label
            and result.confidence < max(0.0, float(self.reject_min_conf))
        ):
            result.rejection = {
                "reason": "below_min_confidence",
                "min_confidence": float(self.reject_min_conf),
                "raw_label": str(result.label),
                "raw_confidence": float(result.confidence),
                "raw_source": result.source.value,
            }
            result.label = None
            result.confidence = 0.0
            result.source = DecisionSource.FALLBACK
            result.decision_path.append("final_below_reject_min_conf")

        self._attach_explanation(result)

        logger.debug(
            "HybridClassifier decision",
            extra={
                "filename": filename,
                "label": result.label,
                "confidence": result.confidence,
                "source": result.source.value,
                "path": result.decision_path,
                "fusion_metadata": result.fusion_metadata,
            },
        )

        return result

    def classify_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[ClassificationResult]:
        """
        批量分类

        Args:
            items: 列表，每项包含 filename 和可选的 file_bytes, graph2d_result

        Returns:
            ClassificationResult 列表
        """
        results = []
        for item in items:
            result = self.classify(
                filename=item.get("filename", ""),
                file_bytes=item.get("file_bytes"),
                graph2d_result=item.get("graph2d_result"),
                history_result=item.get("history_result"),
                history_file_path=item.get("history_file_path"),
            )
            results.append(result)
        return results


# 全局单例
_HYBRID_CLASSIFIER: Optional[HybridClassifier] = None


def get_hybrid_classifier() -> HybridClassifier:
    """获取全局 HybridClassifier 实例"""
    global _HYBRID_CLASSIFIER
    if _HYBRID_CLASSIFIER is None:
        _HYBRID_CLASSIFIER = HybridClassifier()
    return _HYBRID_CLASSIFIER


def reset_hybrid_classifier() -> None:
    """重置全局实例（用于测试）"""
    global _HYBRID_CLASSIFIER
    _HYBRID_CLASSIFIER = None
