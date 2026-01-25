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
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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
        filename_min_conf: float = 0.8,
        graph2d_min_conf: float = 0.5,
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
        self.filename_min_conf = float(os.getenv("FILENAME_MIN_CONF", str(filename_min_conf)))
        self.graph2d_min_conf = float(os.getenv("GRAPH2D_MIN_CONF", str(graph2d_min_conf)))

        # 懒加载分类器
        self._filename_classifier = None
        self._graph2d_classifier = None

        logger.info(
            "HybridClassifier initialized",
            extra={
                "filename_weight": self.filename_weight,
                "graph2d_weight": self.graph2d_weight,
                "filename_min_conf": self.filename_min_conf,
                "graph2d_min_conf": self.graph2d_min_conf,
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

    def _is_filename_enabled(self) -> bool:
        """检查文件名分类是否启用"""
        return os.getenv("FILENAME_CLASSIFIER_ENABLED", "true").lower() == "true"

    def _is_graph2d_enabled(self) -> bool:
        """检查 Graph2D 分类是否启用"""
        return os.getenv("GRAPH2D_ENABLED", "false").lower() == "true"

    def _is_hybrid_enabled(self) -> bool:
        """检查混合分类是否启用"""
        return os.getenv("HYBRID_CLASSIFIER_ENABLED", "true").lower() == "true"

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

        if graph2d_pred:
            result.graph2d_prediction = graph2d_pred

        # 3. 融合决策
        filename_label = filename_pred.get("label") if filename_pred else None
        filename_conf = float(filename_pred.get("confidence", 0)) if filename_pred else 0.0

        graph2d_label = graph2d_pred.get("label") if graph2d_pred else None
        graph2d_conf = float(graph2d_pred.get("confidence", 0)) if graph2d_pred else 0.0

        result.fusion_weights = {
            "filename": self.filename_weight,
            "graph2d": self.graph2d_weight,
        }

        # 决策逻辑
        if filename_label and filename_conf >= self.filename_min_conf:
            # 文件名高置信度，直接采用
            result.label = filename_label
            result.confidence = filename_conf
            result.source = DecisionSource.FILENAME
            result.decision_path.append("filename_high_conf_adopted")

        elif graph2d_label and graph2d_conf >= self.graph2d_min_conf and filename_conf < 0.5:
            # Graph2D 高置信度，文件名低置信度
            result.label = graph2d_label
            result.confidence = graph2d_conf
            result.source = DecisionSource.GRAPH2D
            result.decision_path.append("graph2d_adopted")

        elif filename_label and graph2d_label:
            # 两者都有预测，加权融合
            if filename_label == graph2d_label:
                # 标签一致，增强置信度
                fused_conf = min(1.0, filename_conf * self.filename_weight + graph2d_conf * self.graph2d_weight + 0.1)
                result.label = filename_label
                result.confidence = fused_conf
                result.source = DecisionSource.FUSION
                result.decision_path.append("fusion_agreement")
            else:
                # 标签冲突，选择置信度更高的
                if filename_conf * self.filename_weight >= graph2d_conf * self.graph2d_weight:
                    result.label = filename_label
                    result.confidence = filename_conf
                    result.source = DecisionSource.FILENAME
                    result.decision_path.append("fusion_conflict_filename_wins")
                else:
                    result.label = graph2d_label
                    result.confidence = graph2d_conf
                    result.source = DecisionSource.GRAPH2D
                    result.decision_path.append("fusion_conflict_graph2d_wins")

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
