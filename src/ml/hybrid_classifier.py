"""
HybridClassifier - 混合分类器

融合多种分类信号：
1. FilenameClassifier (文件名)
2. Graph2DClassifier (几何图神经网络)
3. TitleBlockClassifier (标题栏文本)
4. ProcessClassifier (工艺特征)

Feature Flags:
    HYBRID_CLASSIFIER_ENABLED: 是否启用混合分类 (default: true)
    FILENAME_CLASSIFIER_ENABLED: 是否启用文件名分类 (default: true)
    FILENAME_FUSION_WEIGHT: 文件名分类权重 (default: 0.7)
    GRAPH2D_FUSION_WEIGHT: Graph2D 分类权重 (default: 0.3)
    TITLEBLOCK_ENABLED: 是否启用标题栏特征 (default: false)
    TITLEBLOCK_OVERRIDE_ENABLED: 是否允许标题栏直接覆盖 (default: false)
    PROCESS_FEATURES_ENABLED: 是否启用工艺特征 (default: true)
    PROCESS_FUSION_WEIGHT: 工艺特征权重 (default: 0.15)
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

    # 融合决策详情
    fusion_weights: Dict[str, float] = field(default_factory=dict)
    decision_path: List[str] = field(default_factory=list)

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
            "fusion_weights": self.fusion_weights,
            "decision_path": self.decision_path,
        }


class HybridClassifier:
    """混合分类器"""

    def __init__(
        self,
        filename_weight: Optional[float] = None,
        graph2d_weight: Optional[float] = None,
        titleblock_weight: Optional[float] = None,
        process_weight: Optional[float] = None,
        filename_min_conf: Optional[float] = None,
        graph2d_min_conf: Optional[float] = None,
        titleblock_min_conf: Optional[float] = None,
        process_min_conf: Optional[float] = None,
    ):
        """
        初始化混合分类器

        Args:
            filename_weight: 文件名分类权重
            graph2d_weight: Graph2D 分类权重
            titleblock_weight: 标题栏分类权重
            process_weight: 工艺特征分类权重
            filename_min_conf: 文件名分类最低置信度（高于此值优先采用）
            graph2d_min_conf: Graph2D 分类最低置信度
            titleblock_min_conf: 标题栏分类最低置信度
            process_min_conf: 工艺特征分类最低置信度
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

        logger.info(
            "HybridClassifier initialized",
            extra={
                "filename_weight": self.filename_weight,
                "graph2d_weight": self.graph2d_weight,
                "titleblock_weight": self.titleblock_weight,
                "process_weight": self.process_weight,
                "filename_min_conf": self.filename_min_conf,
                "graph2d_min_conf": self.graph2d_min_conf,
                "titleblock_min_conf": self.titleblock_min_conf,
                "process_min_conf": self.process_min_conf,
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
        if (self._is_titleblock_enabled() or self._is_process_enabled()) and file_bytes:
            try:
                from src.utils.dxf_io import read_dxf_entities_from_bytes

                dxf_entities = read_dxf_entities_from_bytes(file_bytes)
            except Exception as e:
                logger.warning("DXF parse failed for hybrid classifiers: %s", e)
                result.decision_path.append("dxf_parse_error")

        # 3. TitleBlock 分类
        titleblock_pred = None
        if self._is_titleblock_enabled() and dxf_entities is not None:
            try:
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

        # 5. 融合决策
        filename_label = filename_pred.get("label") if filename_pred else None
        filename_conf = (
            float(filename_pred.get("confidence", 0)) if filename_pred else 0.0
        )

        graph2d_label = graph2d_label_raw
        graph2d_conf = graph2d_conf_raw
        if graph2d_is_drawing_type:
            result.decision_path.append("graph2d_drawing_type_ignored")
            if result.graph2d_prediction is not None:
                result.graph2d_prediction["filtered"] = True
                result.graph2d_prediction["filtered_reason"] = "drawing_type"
            graph2d_label = None
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
                graph2d_conf = 0.0
            elif self.graph2d_allow_labels and (
                graph2d_label not in self.graph2d_allow_labels
            ):
                result.decision_path.append("graph2d_not_in_allowlist_ignored")
                if result.graph2d_prediction is not None:
                    result.graph2d_prediction["filtered"] = True
                    result.graph2d_prediction["filtered_reason"] = "not_in_allowlist"
                graph2d_label = None
                graph2d_conf = 0.0

        # Graph2D confidence is sensitive to class count (many-class softmax tends to
        # produce lower max probabilities). When class-count metadata is available,
        # apply a conservative dynamic lower bound to avoid filtering everything.
        effective_graph2d_min_conf = self.graph2d_min_conf
        if result.graph2d_prediction is not None:
            try:
                label_map_size = int(result.graph2d_prediction.get("label_map_size") or 0)
            except Exception:
                label_map_size = 0
            if label_map_size >= 20:
                uniform = 1.0 / max(1, label_map_size)
                dynamic_min = max(3.0 * uniform, 0.05)
                effective_graph2d_min_conf = min(effective_graph2d_min_conf, dynamic_min)
                result.graph2d_prediction["min_confidence_effective"] = effective_graph2d_min_conf

        if graph2d_label and graph2d_conf < effective_graph2d_min_conf:
            result.decision_path.append("graph2d_below_min_conf_ignored")
            if result.graph2d_prediction is not None:
                result.graph2d_prediction["filtered"] = True
                result.graph2d_prediction["filtered_reason"] = "below_min_conf"
            graph2d_label = None
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
                result.graph2d_prediction["min_margin_effective"] = effective_graph2d_min_margin
            graph2d_label = None
            graph2d_conf = 0.0

        titleblock_label = titleblock_pred.get("label") if titleblock_pred else None
        titleblock_conf = (
            float(titleblock_pred.get("confidence", 0.0)) if titleblock_pred else 0.0
        )

        result.fusion_weights = {
            "filename": self.filename_weight,
            "graph2d": self.graph2d_weight,
            "titleblock": self.titleblock_weight,
            "process": self.process_weight,
        }

        if titleblock_label and filename_label and titleblock_label != filename_label:
            result.decision_path.append("titleblock_filename_conflict")
            if filename_conf >= self.filename_min_conf:
                result.decision_path.append("titleblock_ignored_filename_high_conf")

        other_labels = {
            self._normalize_label(label)
            for label in (filename_label, titleblock_label, process_label)
            if label
        }
        if graph2d_label and other_labels and graph2d_label not in other_labels:
            # Guardrail: Graph2D cannot introduce a new label when rules/text found one.
            result.decision_path.append("graph2d_non_matching_ignored")
            if result.graph2d_prediction is not None:
                result.graph2d_prediction["ignored_for_fusion"] = True
                result.graph2d_prediction["ignored_reason"] = "non_matching"
            graph2d_label = None
            graph2d_conf = 0.0

        # 决策逻辑
        if filename_label and filename_conf >= self.filename_min_conf:
            # 文件名高置信度，直接采用
            result.label = filename_label
            result.confidence = filename_conf
            result.source = DecisionSource.FILENAME
            result.decision_path.append("filename_high_conf_adopted")

        elif (
            self.titleblock_override_enabled
            and titleblock_label
            and titleblock_conf >= self.titleblock_min_conf
            and filename_conf < 0.5
        ):
            result.label = titleblock_label
            result.confidence = titleblock_conf
            result.source = DecisionSource.TITLEBLOCK
            result.decision_path.append("titleblock_adopted")

        else:
            preds: List[tuple[str, str, float, DecisionSource]] = []
            if filename_label:
                preds.append(
                    ("filename", filename_label, filename_conf, DecisionSource.FILENAME)
                )
            if titleblock_label:
                preds.append(
                    (
                        "titleblock",
                        titleblock_label,
                        titleblock_conf,
                        DecisionSource.TITLEBLOCK,
                    )
                )
            if process_label:
                preds.append(
                    ("process", process_label, process_conf, DecisionSource.PROCESS)
                )
            if graph2d_label:
                preds.append(
                    ("graph2d", graph2d_label, graph2d_conf, DecisionSource.GRAPH2D)
                )

            if len(preds) == 1:
                src_key, label, conf, src = preds[0]
                result.label = label
                result.confidence = conf
                result.source = src
                result.decision_path.append(f"{src_key}_only")

            elif preds:
                # 多源融合 (filename/graph2d/titleblock/process)
                label_scores: Dict[str, float] = {}
                label_sources: Dict[str, List[str]] = {}

                def _add_score(
                    label: Optional[str], conf: float, weight: float, source: str
                ) -> None:
                    if not label:
                        return
                    label_scores[label] = label_scores.get(label, 0.0) + conf * weight
                    label_sources.setdefault(label, []).append(source)

                _add_score(
                    filename_label, filename_conf, self.filename_weight, "filename"
                )
                _add_score(graph2d_label, graph2d_conf, self.graph2d_weight, "graph2d")
                _add_score(
                    titleblock_label,
                    titleblock_conf,
                    self.titleblock_weight,
                    "titleblock",
                )
                _add_score(process_label, process_conf, self.process_weight, "process")

                if label_scores:
                    best_label = max(label_scores.items(), key=lambda item: item[1])[0]
                    sources = label_sources.get(best_label, [])
                    bonus = 0.1 if len(sources) >= 2 else 0.0
                    fused_conf = min(1.0, label_scores[best_label] + bonus)
                    result.label = best_label
                    result.confidence = fused_conf
                    result.source = DecisionSource.FUSION
                    result.decision_path.append("fusion_scored")
                    if bonus > 0:
                        result.decision_path.append("fusion_multi_source_bonus")

            else:
                # 无预测
                result.source = DecisionSource.FALLBACK
                result.decision_path.append("no_prediction")

        logger.debug(
            "HybridClassifier decision",
            extra={
                "filename": filename,
                "label": result.label,
                "confidence": result.confidence,
                "source": result.source.value,
                "path": result.decision_path,
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
