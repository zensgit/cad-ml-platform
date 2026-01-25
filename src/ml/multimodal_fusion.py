"""
多模态融合分类器

融合多种分类信号，支持可解释的来源权重输出：
1. 几何分支 (Graph2D GNN)
2. 文本分支 (文件名 + 标题栏)
3. 规则分支 (传统规则匹配)

Feature Flags:
    MULTIMODAL_FUSION_ENABLED: 是否启用多模态融合 (default: true)
    FUSION_GEOMETRY_WEIGHT: 几何分支权重 (default: 0.3)
    FUSION_TEXT_WEIGHT: 文本分支权重 (default: 0.5)
    FUSION_RULE_WEIGHT: 规则分支权重 (default: 0.2)
    FUSION_GATE_TYPE: 门控类型 weighted|attention|learned (default: weighted)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FusionGateType(str, Enum):
    """融合门控类型"""
    WEIGHTED = "weighted"      # 固定权重
    ATTENTION = "attention"    # 注意力机制
    LEARNED = "learned"        # 学习的门控


class ModalitySource(str, Enum):
    """模态来源"""
    GEOMETRY = "geometry"
    FILENAME = "filename"
    TITLEBLOCK = "titleblock"
    RULE = "rule"
    FUSION = "fusion"


@dataclass
class ModalityPrediction:
    """单模态预测结果"""
    source: ModalitySource
    label: Optional[str] = None
    confidence: float = 0.0
    raw_output: Optional[Dict[str, Any]] = None
    available: bool = True


@dataclass
class FusionResult:
    """融合结果"""
    final_label: Optional[str] = None
    final_confidence: float = 0.0
    decision_source: ModalitySource = ModalitySource.FUSION

    # 各模态预测
    modality_predictions: Dict[str, ModalityPrediction] = field(default_factory=dict)

    # 融合权重 (可解释性)
    applied_weights: Dict[str, float] = field(default_factory=dict)

    # 决策路径
    decision_path: List[str] = field(default_factory=list)

    # 冲突信息
    has_conflict: bool = False
    conflict_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "final_label": self.final_label,
            "final_confidence": self.final_confidence,
            "decision_source": self.decision_source.value,
            "modality_predictions": {
                k: {
                    "source": v.source.value,
                    "label": v.label,
                    "confidence": v.confidence,
                    "available": v.available,
                }
                for k, v in self.modality_predictions.items()
            },
            "applied_weights": self.applied_weights,
            "decision_path": self.decision_path,
            "has_conflict": self.has_conflict,
            "conflict_details": self.conflict_details,
        }


class MultimodalFusionClassifier:
    """多模态融合分类器"""

    def __init__(
        self,
        geometry_weight: float = 0.3,
        text_weight: float = 0.5,
        rule_weight: float = 0.2,
        gate_type: str = "weighted",
    ):
        self.geometry_weight = float(os.getenv("FUSION_GEOMETRY_WEIGHT", str(geometry_weight)))
        self.text_weight = float(os.getenv("FUSION_TEXT_WEIGHT", str(text_weight)))
        self.rule_weight = float(os.getenv("FUSION_RULE_WEIGHT", str(rule_weight)))
        self.gate_type = FusionGateType(os.getenv("FUSION_GATE_TYPE", gate_type))

        # 归一化权重
        total = self.geometry_weight + self.text_weight + self.rule_weight
        if total > 0:
            self.geometry_weight /= total
            self.text_weight /= total
            self.rule_weight /= total

        logger.info(
            "MultimodalFusionClassifier initialized",
            extra={
                "geometry_weight": self.geometry_weight,
                "text_weight": self.text_weight,
                "rule_weight": self.rule_weight,
                "gate_type": self.gate_type.value,
            },
        )

    def _aggregate_text_predictions(
        self,
        filename_pred: Optional[ModalityPrediction],
        titleblock_pred: Optional[ModalityPrediction],
    ) -> ModalityPrediction:
        """聚合文本分支预测"""
        preds = [p for p in [filename_pred, titleblock_pred] if p and p.label]

        if not preds:
            return ModalityPrediction(
                source=ModalitySource.FILENAME,
                available=False,
            )

        # 选择置信度最高的
        best = max(preds, key=lambda p: p.confidence)

        # 如果两者一致，增强置信度
        if len(preds) == 2 and preds[0].label == preds[1].label:
            return ModalityPrediction(
                source=ModalitySource.FUSION,
                label=best.label,
                confidence=min(1.0, best.confidence + 0.1),
                available=True,
            )

        return best

    def _weighted_fusion(
        self,
        predictions: Dict[str, ModalityPrediction],
    ) -> Tuple[Optional[str], float, Dict[str, float]]:
        """加权融合"""
        label_scores: Dict[str, float] = {}
        applied_weights: Dict[str, float] = {}

        weight_map = {
            "geometry": self.geometry_weight,
            "text": self.text_weight,
            "rule": self.rule_weight,
        }

        for key, pred in predictions.items():
            if not pred.available or not pred.label:
                continue

            weight = weight_map.get(key, 0.0)
            applied_weights[key] = weight

            score = pred.confidence * weight
            if pred.label in label_scores:
                label_scores[pred.label] += score
            else:
                label_scores[pred.label] = score

        if not label_scores:
            return None, 0.0, applied_weights

        # 选择得分最高的标签
        best_label = max(label_scores, key=label_scores.get)
        best_score = label_scores[best_label]

        # 归一化置信度
        total_score = sum(label_scores.values())
        normalized_conf = best_score / total_score if total_score > 0 else 0.0

        return best_label, normalized_conf, applied_weights

    def _detect_conflict(
        self,
        predictions: Dict[str, ModalityPrediction],
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """检测预测冲突"""
        labels = {}
        for key, pred in predictions.items():
            if pred.available and pred.label:
                labels[key] = (pred.label, pred.confidence)

        if len(set(l for l, _ in labels.values())) <= 1:
            return False, None

        # 存在冲突
        return True, {
            "conflicting_sources": list(labels.keys()),
            "predictions": {k: {"label": l, "confidence": c} for k, (l, c) in labels.items()},
        }

    def classify(
        self,
        geometry_pred: Optional[Dict[str, Any]] = None,
        filename_pred: Optional[Dict[str, Any]] = None,
        titleblock_pred: Optional[Dict[str, Any]] = None,
        rule_pred: Optional[Dict[str, Any]] = None,
    ) -> FusionResult:
        """
        执行多模态融合分类

        Args:
            geometry_pred: 几何分支预测 (Graph2D)
            filename_pred: 文件名分支预测
            titleblock_pred: 标题栏分支预测
            rule_pred: 规则分支预测

        Returns:
            FusionResult
        """
        result = FusionResult()
        result.decision_path = []

        # 转换为统一格式
        modality_preds: Dict[str, ModalityPrediction] = {}

        if geometry_pred:
            modality_preds["geometry"] = ModalityPrediction(
                source=ModalitySource.GEOMETRY,
                label=geometry_pred.get("label"),
                confidence=float(geometry_pred.get("confidence", 0)),
                raw_output=geometry_pred,
                available=geometry_pred.get("status") != "unavailable",
            )
            result.decision_path.append("geometry_received")

        filename_mp = None
        if filename_pred:
            filename_mp = ModalityPrediction(
                source=ModalitySource.FILENAME,
                label=filename_pred.get("label"),
                confidence=float(filename_pred.get("confidence", 0)),
                raw_output=filename_pred,
                available=filename_pred.get("status") == "matched",
            )
            result.decision_path.append("filename_received")

        titleblock_mp = None
        if titleblock_pred:
            titleblock_mp = ModalityPrediction(
                source=ModalitySource.TITLEBLOCK,
                label=titleblock_pred.get("label"),
                confidence=float(titleblock_pred.get("confidence", 0)),
                raw_output=titleblock_pred,
                available=titleblock_pred.get("status") == "matched",
            )
            result.decision_path.append("titleblock_received")

        # 聚合文本分支
        text_pred = self._aggregate_text_predictions(filename_mp, titleblock_mp)
        modality_preds["text"] = text_pred

        if rule_pred:
            modality_preds["rule"] = ModalityPrediction(
                source=ModalitySource.RULE,
                label=rule_pred.get("label") or rule_pred.get("type"),
                confidence=float(rule_pred.get("confidence", 0)),
                raw_output=rule_pred,
                available=True,
            )
            result.decision_path.append("rule_received")

        result.modality_predictions = modality_preds

        # 检测冲突
        has_conflict, conflict_details = self._detect_conflict(modality_preds)
        result.has_conflict = has_conflict
        result.conflict_details = conflict_details

        if has_conflict:
            result.decision_path.append("conflict_detected")

        # 融合决策
        if self.gate_type == FusionGateType.WEIGHTED:
            label, conf, weights = self._weighted_fusion(modality_preds)
            result.final_label = label
            result.final_confidence = conf
            result.applied_weights = weights
            result.decision_path.append("weighted_fusion_applied")

        # 确定决策来源
        if result.final_label:
            # 找到贡献最大的模态
            best_source = None
            best_contrib = 0.0
            for key, pred in modality_preds.items():
                if pred.label == result.final_label:
                    contrib = pred.confidence * result.applied_weights.get(key, 0)
                    if contrib > best_contrib:
                        best_contrib = contrib
                        best_source = pred.source

            result.decision_source = best_source or ModalitySource.FUSION

        return result


# 全局单例
_FUSION_CLASSIFIER: Optional[MultimodalFusionClassifier] = None


def get_multimodal_fusion_classifier() -> MultimodalFusionClassifier:
    """获取全局 MultimodalFusionClassifier 实例"""
    global _FUSION_CLASSIFIER
    if _FUSION_CLASSIFIER is None:
        _FUSION_CLASSIFIER = MultimodalFusionClassifier()
    return _FUSION_CLASSIFIER


def is_multimodal_fusion_enabled() -> bool:
    """检查多模态融合是否启用"""
    return os.getenv("MULTIMODAL_FUSION_ENABLED", "true").lower() == "true"
