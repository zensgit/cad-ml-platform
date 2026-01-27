"""
ProcessClassifier - 基于工艺特征的图纸类型推断

根据提取的工艺信息推断图纸类型：
- 热处理/表面处理 → 倾向于零件图
- 焊接信息 → 倾向于装配图/结构件
- 技术要求 → 倾向于零件图

此分类器作为辅助信号，用于增强混合分类器的置信度。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.ocr.base import ProcessRequirements

logger = logging.getLogger(__name__)


# 工艺特征与图纸类型的映射规则
PROCESS_TO_LABEL_RULES = {
    # 热处理通常出现在零件图
    "heat_treatment": {
        "labels": ["零件图", "机械制图"],
        "confidence_boost": 0.15,
    },
    # 表面处理通常出现在零件图
    "surface_treatment": {
        "labels": ["零件图", "机械制图"],
        "confidence_boost": 0.12,
    },
    # 焊接信息可能出现在装配图或结构件
    "welding": {
        "labels": ["装配图", "结构件", "焊接件"],
        "confidence_boost": 0.18,
    },
    # 通用技术要求（公差、圆角等）倾向于零件图
    "general_notes": {
        "labels": ["零件图", "机械制图"],
        "confidence_boost": 0.08,
    },
}


@dataclass
class ProcessClassificationResult:
    """工艺分类结果"""
    suggested_labels: List[str]
    confidence: float
    features_found: Dict[str, int]
    raw: Optional[ProcessRequirements] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suggested_labels": self.suggested_labels,
            "confidence": self.confidence,
            "features_found": self.features_found,
        }


class ProcessClassifier:
    """基于工艺特征的分类器"""

    def __init__(
        self,
        min_confidence: float = 0.3,
        max_confidence: float = 0.7,
    ):
        """
        初始化工艺分类器

        Args:
            min_confidence: 最小置信度（有工艺信息时的基础置信度）
            max_confidence: 最大置信度（多种工艺信息时的上限）
        """
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence

    def predict(self, process_requirements: Optional[ProcessRequirements]) -> ProcessClassificationResult:
        """
        根据工艺信息推断图纸类型

        Args:
            process_requirements: 提取的工艺要求

        Returns:
            ProcessClassificationResult
        """
        if not process_requirements:
            return ProcessClassificationResult(
                suggested_labels=[],
                confidence=0.0,
                features_found={},
                raw=None,
            )

        features_found = {
            "heat_treatment": len(process_requirements.heat_treatments),
            "surface_treatment": len(process_requirements.surface_treatments),
            "welding": len(process_requirements.welding),
            "general_notes": len(process_requirements.general_notes),
        }

        # 计算各标签的分数
        label_scores: Dict[str, float] = {}
        total_boost = 0.0

        for feature_type, count in features_found.items():
            if count > 0:
                rule = PROCESS_TO_LABEL_RULES.get(feature_type)
                if rule:
                    boost = rule["confidence_boost"] * min(count, 3)  # 最多计算3个
                    total_boost += boost
                    for label in rule["labels"]:
                        label_scores[label] = label_scores.get(label, 0.0) + boost

        if not label_scores:
            return ProcessClassificationResult(
                suggested_labels=[],
                confidence=0.0,
                features_found=features_found,
                raw=process_requirements,
            )

        # 按分数排序
        sorted_labels = sorted(label_scores.items(), key=lambda x: -x[1])
        suggested_labels = [label for label, _ in sorted_labels]

        # 计算置信度（基于特征数量和多样性）
        feature_count = sum(1 for v in features_found.values() if v > 0)
        confidence = self.min_confidence + (
            (self.max_confidence - self.min_confidence)
            * min(feature_count / 3.0, 1.0)
        )
        confidence = min(confidence, self.max_confidence)

        logger.debug(
            "ProcessClassifier prediction",
            extra={
                "features_found": features_found,
                "suggested_labels": suggested_labels,
                "confidence": confidence,
            },
        )

        return ProcessClassificationResult(
            suggested_labels=suggested_labels,
            confidence=confidence,
            features_found=features_found,
            raw=process_requirements,
        )

    def predict_from_text(self, text: str) -> ProcessClassificationResult:
        """
        从文本直接推断图纸类型

        Args:
            text: OCR 提取的文本

        Returns:
            ProcessClassificationResult
        """
        from src.core.ocr.parsing.process_parser import parse_process_requirements

        process_requirements = parse_process_requirements(text)
        return self.predict(process_requirements)


# 单例
_process_classifier: Optional[ProcessClassifier] = None


def get_process_classifier() -> ProcessClassifier:
    """获取 ProcessClassifier 单例"""
    global _process_classifier
    if _process_classifier is None:
        _process_classifier = ProcessClassifier()
    return _process_classifier
