"""
HybridClassifier - 混合分类器

融合多种分类信号：
1. FilenameClassifier (文件名)
2. Graph2DClassifier (几何图神经网络)
3. TitleBlockClassifier (标题栏文本) [未来]

Feature Flags:
    HYBRID_CLASSIFIER_ENABLED: 是否启用混合分类 (default: true)
    FILENAME_CLASSIFIER_ENABLED: 是否启用文件名分类 (default: true)
    FILENAME_FUSION_WEIGHT: 文件名分类权重 (default: 0.7)
    GRAPH2D_FUSION_WEIGHT: Graph2D 分类权重 (default: 0.3)
    TITLEBLOCK_ENABLED: 是否启用标题栏特征 (default: false)
    TITLEBLOCK_OVERRIDE_ENABLED: 是否允许标题栏直接覆盖 (default: false)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

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
            "fusion_weights": self.fusion_weights,
            "decision_path": self.decision_path,
        }


class HybridClassifier:
    """混合分类器"""

    def __init__(
        self,
        filename_weight: float = 0.7,
        graph2d_weight: float = 0.3,
        titleblock_weight: float = 0.2,
        filename_min_conf: float = 0.8,
        graph2d_min_conf: float = 0.5,
        titleblock_min_conf: float = 0.75,
    ):
        """
        初始化混合分类器

        Args:
            filename_weight: 文件名分类权重
            graph2d_weight: Graph2D 分类权重
            filename_min_conf: 文件名分类最低置信度（高于此值优先采用）
            graph2d_min_conf: Graph2D 分类最低置信度
        """
        self.filename_weight = float(os.getenv("FILENAME_FUSION_WEIGHT", str(filename_weight)))
        self.graph2d_weight = float(os.getenv("GRAPH2D_FUSION_WEIGHT", str(graph2d_weight)))
        self.titleblock_weight = float(
            os.getenv("TITLEBLOCK_FUSION_WEIGHT", str(titleblock_weight))
        )
        self.filename_min_conf = float(os.getenv("FILENAME_MIN_CONF", str(filename_min_conf)))
        self.graph2d_min_conf = float(os.getenv("GRAPH2D_MIN_CONF", str(graph2d_min_conf)))
        self.titleblock_min_conf = float(
            os.getenv("TITLEBLOCK_MIN_CONF", str(titleblock_min_conf))
        )
        self.titleblock_override_enabled = (
            os.getenv("TITLEBLOCK_OVERRIDE_ENABLED", "false").lower() == "true"
        )
        drawing_labels_raw = os.getenv("GRAPH2D_DRAWING_TYPE_LABELS", "").strip()
        if drawing_labels_raw:
            self.graph2d_drawing_labels = {
                label.strip() for label in drawing_labels_raw.split(",") if label.strip()
            }
        else:
            self.graph2d_drawing_labels = set(DEFAULT_GRAPH2D_DRAWING_LABELS)

        # 懒加载分类器
        self._filename_classifier = None
        self._graph2d_classifier = None
        self._titleblock_classifier = None

        logger.info(
            "HybridClassifier initialized",
            extra={
                "filename_weight": self.filename_weight,
                "graph2d_weight": self.graph2d_weight,
                "titleblock_weight": self.titleblock_weight,
                "filename_min_conf": self.filename_min_conf,
                "graph2d_min_conf": self.graph2d_min_conf,
                "titleblock_min_conf": self.titleblock_min_conf,
                "titleblock_override_enabled": self.titleblock_override_enabled,
                "graph2d_drawing_labels": sorted(self.graph2d_drawing_labels),
            },
        )

    @property
    def filename_classifier(self):
        """懒加载 FilenameClassifier"""
        if self._filename_classifier is None:
            from src.ml.filename_classifier import get_filename_classifier
            self._filename_classifier = get_filename_classifier()
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

    def _is_filename_enabled(self) -> bool:
        """检查文件名分类是否启用"""
        return os.getenv("FILENAME_CLASSIFIER_ENABLED", "true").lower() == "true"

    def _is_graph2d_enabled(self) -> bool:
        """检查 Graph2D 分类是否启用"""
        return os.getenv("GRAPH2D_ENABLED", "false").lower() == "true"

    def _is_hybrid_enabled(self) -> bool:
        """检查混合分类是否启用"""
        return os.getenv("HYBRID_CLASSIFIER_ENABLED", "true").lower() == "true"

    def _is_titleblock_enabled(self) -> bool:
        """检查标题栏特征是否启用"""
        return os.getenv("TITLEBLOCK_ENABLED", "false").lower() == "true"

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
        graph2d_conf_raw = float(graph2d_pred.get("confidence", 0)) if graph2d_pred else 0.0
        graph2d_is_drawing_type = self._is_graph2d_drawing_type(graph2d_label_raw)
        if graph2d_pred:
            graph2d_pred = dict(graph2d_pred)
            graph2d_pred["is_drawing_type"] = graph2d_is_drawing_type
            result.graph2d_prediction = graph2d_pred

        # 3. TitleBlock 分类
        titleblock_pred = None
        if self._is_titleblock_enabled() and file_bytes:
            try:
                import tempfile
                import os as _os
                import ezdxf  # type: ignore

                with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name
                try:
                    doc = ezdxf.readfile(tmp_path)
                    msp = doc.modelspace()
                    classifier = self.titleblock_classifier
                    if classifier:
                        titleblock_pred = classifier.predict(list(msp))
                        result.decision_path.append("titleblock_predicted")
                finally:
                    try:
                        _os.unlink(tmp_path)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("TitleBlock classification failed: %s", e)
                result.decision_path.append("titleblock_error")

        if titleblock_pred:
            result.titleblock_prediction = titleblock_pred

        # 3. 融合决策
        filename_label = filename_pred.get("label") if filename_pred else None
        filename_conf = float(filename_pred.get("confidence", 0)) if filename_pred else 0.0

        graph2d_label = graph2d_label_raw
        graph2d_conf = graph2d_conf_raw
        if graph2d_is_drawing_type:
            result.decision_path.append("graph2d_drawing_type_ignored")
            graph2d_label = None
            graph2d_conf = 0.0

        titleblock_label = titleblock_pred.get("label") if titleblock_pred else None
        titleblock_conf = float(titleblock_pred.get("confidence", 0.0)) if titleblock_pred else 0.0

        result.fusion_weights = {
            "filename": self.filename_weight,
            "graph2d": self.graph2d_weight,
            "titleblock": self.titleblock_weight,
        }

        if (
            titleblock_label
            and filename_label
            and titleblock_label != filename_label
        ):
            result.decision_path.append("titleblock_filename_conflict")
            if filename_conf >= self.filename_min_conf:
                result.decision_path.append("titleblock_ignored_filename_high_conf")

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
            and (not graph2d_label or graph2d_conf < self.graph2d_min_conf)
        ):
            result.label = titleblock_label
            result.confidence = titleblock_conf
            result.source = DecisionSource.TITLEBLOCK
            result.decision_path.append("titleblock_adopted")

        elif graph2d_label and graph2d_conf >= self.graph2d_min_conf and filename_conf < 0.5:
            # Graph2D 高置信度，文件名低置信度
            result.label = graph2d_label
            result.confidence = graph2d_conf
            result.source = DecisionSource.GRAPH2D
            result.decision_path.append("graph2d_adopted")

        elif filename_label or graph2d_label or titleblock_label:
            # 多源融合 (filename/graph2d/titleblock)
            label_scores: Dict[str, float] = {}
            label_sources: Dict[str, List[str]] = {}

            def _add_score(label: Optional[str], conf: float, weight: float, source: str) -> None:
                if not label:
                    return
                label_scores[label] = label_scores.get(label, 0.0) + conf * weight
                label_sources.setdefault(label, []).append(source)

            _add_score(filename_label, filename_conf, self.filename_weight, "filename")
            _add_score(graph2d_label, graph2d_conf, self.graph2d_weight, "graph2d")
            _add_score(titleblock_label, titleblock_conf, self.titleblock_weight, "titleblock")

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

        elif filename_label:
            # 只有文件名预测
            result.label = filename_label
            result.confidence = filename_conf
            result.source = DecisionSource.FILENAME
            result.decision_path.append("filename_only")

        elif graph2d_label:
            # 只有 Graph2D 预测
            result.label = graph2d_label
            result.confidence = graph2d_conf
            result.source = DecisionSource.GRAPH2D
            result.decision_path.append("graph2d_only")

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
