"""
Explainability module for HybridClassifier.

Provides explanations for classification decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ExplanationType(str, Enum):
    """Types of explanations."""
    FEATURE_IMPORTANCE = "feature_importance"
    DECISION_PATH = "decision_path"
    COUNTERFACTUAL = "counterfactual"
    RULE_BASED = "rule_based"
    ATTENTION = "attention"


@dataclass
class FeatureContribution:
    """Contribution of a single feature to the decision."""
    feature_name: str
    feature_value: Any
    contribution: float  # Positive = supports prediction, negative = opposes
    source: str
    description: str = ""


@dataclass
class DecisionStep:
    """A step in the decision path."""
    step_number: int
    source: str
    action: str
    result: str
    confidence_change: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Counterfactual:
    """A counterfactual explanation."""
    original_label: str
    counterfactual_label: str
    changes_needed: List[Dict[str, Any]]
    distance: float  # How different from original
    feasibility: float  # How feasible the changes are


@dataclass
class Explanation:
    """Complete explanation for a classification decision."""
    prediction_label: str
    prediction_confidence: float
    explanation_type: ExplanationType
    summary: str

    # Feature-based explanation
    feature_contributions: List[FeatureContribution] = field(default_factory=list)
    top_positive_features: List[str] = field(default_factory=list)
    top_negative_features: List[str] = field(default_factory=list)

    # Decision path
    decision_path: List[DecisionStep] = field(default_factory=list)
    critical_decisions: List[str] = field(default_factory=list)

    # Source contributions
    source_contributions: Dict[str, float] = field(default_factory=dict)

    # Alternative predictions
    alternative_labels: List[Tuple[str, float]] = field(default_factory=list)

    # Counterfactuals
    counterfactuals: List[Counterfactual] = field(default_factory=list)

    # Uncertainty
    uncertainty_score: float = 0.0
    uncertainty_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction": {
                "label": self.prediction_label,
                "confidence": self.prediction_confidence,
            },
            "explanation_type": self.explanation_type.value,
            "summary": self.summary,
            "feature_contributions": [
                {
                    "feature": f.feature_name,
                    "value": f.feature_value,
                    "contribution": f.contribution,
                    "source": f.source,
                    "description": f.description,
                }
                for f in self.feature_contributions
            ],
            "top_positive_features": self.top_positive_features,
            "top_negative_features": self.top_negative_features,
            "decision_path": [
                {
                    "step": s.step_number,
                    "source": s.source,
                    "action": s.action,
                    "result": s.result,
                    "confidence_change": s.confidence_change,
                }
                for s in self.decision_path
            ],
            "source_contributions": self.source_contributions,
            "alternative_labels": [
                {"label": label, "confidence": conf}
                for label, conf in self.alternative_labels
            ],
            "uncertainty": {
                "score": self.uncertainty_score,
                "sources": self.uncertainty_sources,
            },
        }

    def to_natural_language(self) -> str:
        """Generate natural language explanation."""
        lines = []
        lines.append(f"预测结果：{self.prediction_label}（置信度：{self.prediction_confidence:.1%}）")
        lines.append("")
        lines.append(f"解释：{self.summary}")

        if self.top_positive_features:
            lines.append("")
            lines.append("支持该预测的主要因素：")
            for f in self.top_positive_features[:3]:
                lines.append(f"  • {f}")

        if self.top_negative_features:
            lines.append("")
            lines.append("反对该预测的因素：")
            for f in self.top_negative_features[:3]:
                lines.append(f"  • {f}")

        if self.source_contributions:
            lines.append("")
            lines.append("各来源贡献：")
            for source, contrib in sorted(self.source_contributions.items(), key=lambda x: -x[1]):
                lines.append(f"  • {source}: {contrib:.1%}")

        if self.alternative_labels:
            lines.append("")
            lines.append("其他可能的分类：")
            for label, conf in self.alternative_labels[:3]:
                lines.append(f"  • {label}: {conf:.1%}")

        if self.uncertainty_score > 0.3:
            lines.append("")
            lines.append(f"⚠️ 预测不确定性较高 ({self.uncertainty_score:.1%})")
            if self.uncertainty_sources:
                lines.append(f"   原因：{', '.join(self.uncertainty_sources)}")

        return "\n".join(lines)


class HybridExplainer:
    """
    Explainer for HybridClassifier decisions.

    Provides multiple types of explanations for classification decisions.
    """

    def __init__(
        self,
        include_counterfactuals: bool = False,
        max_features: int = 10,
        language: str = "zh",
    ):
        self.include_counterfactuals = include_counterfactuals
        self.max_features = max_features
        self.language = language

        # Feature descriptions
        self._feature_descriptions = {
            "filename": {
                "zh": "文件名",
                "en": "Filename",
            },
            "graph2d": {
                "zh": "图纸几何特征",
                "en": "Drawing geometry",
            },
            "titleblock": {
                "zh": "标题栏信息",
                "en": "Title block",
            },
            "process": {
                "zh": "工艺特征",
                "en": "Process features",
            },
        }

    def explain(
        self,
        result: Any,  # ClassificationResult
        detailed: bool = True,
    ) -> Explanation:
        """
        Generate explanation for a classification result.

        Args:
            result: ClassificationResult from HybridClassifier
            detailed: Whether to include detailed explanations

        Returns:
            Explanation object
        """
        explanation = Explanation(
            prediction_label=result.label or "unknown",
            prediction_confidence=result.confidence,
            explanation_type=ExplanationType.DECISION_PATH,
            summary="",
        )

        # Analyze feature contributions
        self._analyze_features(result, explanation)

        # Build decision path
        self._build_decision_path(result, explanation)

        # Calculate source contributions
        self._calculate_source_contributions(result, explanation)

        # Find alternatives
        self._find_alternatives(result, explanation)

        # Calculate uncertainty
        self._calculate_uncertainty(result, explanation)

        # Generate summary
        explanation.summary = self._generate_summary(result, explanation)

        # Generate counterfactuals if requested
        if self.include_counterfactuals and detailed:
            self._generate_counterfactuals(result, explanation)

        return explanation

    def _analyze_features(self, result: Any, explanation: Explanation) -> None:
        """Analyze feature contributions."""
        contributions = []

        # Filename features
        if result.filename_prediction:
            fp = result.filename_prediction
            label = fp.get("label")
            conf = fp.get("confidence", 0)

            if label == result.label:
                contrib = conf * result.fusion_weights.get("filename", 0.7)
            else:
                contrib = -conf * result.fusion_weights.get("filename", 0.7)

            contributions.append(FeatureContribution(
                feature_name="filename_label",
                feature_value=label,
                contribution=contrib,
                source="filename",
                description=f"文件名分类为 {label}，置信度 {conf:.1%}",
            ))

            # Filename patterns
            patterns = fp.get("matched_patterns", [])
            for pattern in patterns[:3]:
                contributions.append(FeatureContribution(
                    feature_name="filename_pattern",
                    feature_value=pattern,
                    contribution=0.1,
                    source="filename",
                    description=f"匹配模式：{pattern}",
                ))

        # Graph2D features
        if result.graph2d_prediction:
            gp = result.graph2d_prediction
            label = gp.get("label")
            conf = gp.get("confidence", 0)
            is_drawing_type = gp.get("is_drawing_type", False)

            if is_drawing_type:
                contrib = 0  # Ignored
            elif label == result.label:
                contrib = conf * result.fusion_weights.get("graph2d", 0.3)
            else:
                contrib = -conf * result.fusion_weights.get("graph2d", 0.3)

            contributions.append(FeatureContribution(
                feature_name="graph2d_label",
                feature_value=label,
                contribution=contrib,
                source="graph2d",
                description=f"几何分类为 {label}，置信度 {conf:.1%}" + (" (忽略)" if is_drawing_type else ""),
            ))

        # Titleblock features
        if result.titleblock_prediction:
            tp = result.titleblock_prediction
            label = tp.get("label")
            conf = tp.get("confidence", 0)

            if label == result.label:
                contrib = conf * result.fusion_weights.get("titleblock", 0.2)
            else:
                contrib = -conf * result.fusion_weights.get("titleblock", 0.2)

            contributions.append(FeatureContribution(
                feature_name="titleblock_label",
                feature_value=label,
                contribution=contrib,
                source="titleblock",
                description=f"标题栏分类为 {label}，置信度 {conf:.1%}",
            ))

        # Process features
        if result.process_prediction:
            pp = result.process_prediction
            labels = pp.get("suggested_labels", [])
            conf = pp.get("confidence", 0)

            if labels and labels[0] == result.label:
                contrib = conf * result.fusion_weights.get("process", 0.15)
            else:
                contrib = -conf * result.fusion_weights.get("process", 0.15)

            contributions.append(FeatureContribution(
                feature_name="process_label",
                feature_value=labels[0] if labels else None,
                contribution=contrib,
                source="process",
                description=f"工艺分类为 {labels[0] if labels else 'N/A'}，置信度 {conf:.1%}",
            ))

        # Sort by contribution magnitude
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        explanation.feature_contributions = contributions[:self.max_features]

        # Top positive/negative features
        explanation.top_positive_features = [
            f.description for f in contributions if f.contribution > 0
        ][:5]
        explanation.top_negative_features = [
            f.description for f in contributions if f.contribution < 0
        ][:5]

    def _build_decision_path(self, result: Any, explanation: Explanation) -> None:
        """Build decision path from result."""
        path = []
        for i, step in enumerate(result.decision_path):
            action_descriptions = {
                "filename_extracted": "从文件名提取特征",
                "filename_error": "文件名提取失败",
                "graph2d_predicted": "执行几何分类",
                "graph2d_error": "几何分类失败",
                "graph2d_drawing_type_ignored": "忽略绘图类型标签",
                "titleblock_predicted": "从标题栏提取信息",
                "titleblock_error": "标题栏提取失败",
                "process_predicted": "分析工艺特征",
                "process_error": "工艺分析失败",
                "filename_high_conf_adopted": "采用高置信度文件名预测",
                "graph2d_adopted": "采用几何分类预测",
                "titleblock_adopted": "采用标题栏预测",
                "fusion_scored": "执行多源融合",
                "fusion_multi_source_bonus": "多源一致性加分",
                "filename_only": "仅使用文件名预测",
                "graph2d_only": "仅使用几何预测",
                "no_prediction": "无有效预测",
                "titleblock_filename_conflict": "标题栏与文件名冲突",
                "titleblock_ignored_filename_high_conf": "忽略标题栏（文件名置信度高）",
            }

            source = step.split("_")[0] if "_" in step else "system"
            path.append(DecisionStep(
                step_number=i + 1,
                source=source,
                action=step,
                result=action_descriptions.get(step, step),
            ))

        explanation.decision_path = path

        # Identify critical decisions
        critical = []
        if "filename_high_conf_adopted" in result.decision_path:
            critical.append("文件名高置信度直接采用")
        if "fusion_scored" in result.decision_path:
            critical.append("多源融合决策")
        if "fusion_multi_source_bonus" in result.decision_path:
            critical.append("多源一致性加分")
        explanation.critical_decisions = critical

    def _calculate_source_contributions(self, result: Any, explanation: Explanation) -> None:
        """Calculate contribution from each source."""
        contributions = {}

        if result.filename_prediction:
            fp = result.filename_prediction
            if fp.get("label") == result.label:
                contributions["文件名"] = fp.get("confidence", 0) * result.fusion_weights.get("filename", 0.7)

        if result.graph2d_prediction:
            gp = result.graph2d_prediction
            if gp.get("label") == result.label and not gp.get("is_drawing_type", False):
                contributions["几何分析"] = gp.get("confidence", 0) * result.fusion_weights.get("graph2d", 0.3)

        if result.titleblock_prediction:
            tp = result.titleblock_prediction
            if tp.get("label") == result.label:
                contributions["标题栏"] = tp.get("confidence", 0) * result.fusion_weights.get("titleblock", 0.2)

        if result.process_prediction:
            pp = result.process_prediction
            labels = pp.get("suggested_labels", [])
            if labels and labels[0] == result.label:
                contributions["工艺特征"] = pp.get("confidence", 0) * result.fusion_weights.get("process", 0.15)

        explanation.source_contributions = contributions

    def _find_alternatives(self, result: Any, explanation: Explanation) -> None:
        """Find alternative predictions."""
        alternatives: Dict[str, float] = {}

        # Collect all predictions
        if result.filename_prediction:
            label = result.filename_prediction.get("label")
            conf = result.filename_prediction.get("confidence", 0)
            if label and label != result.label:
                alternatives[label] = max(alternatives.get(label, 0), conf * 0.7)

        if result.graph2d_prediction and not result.graph2d_prediction.get("is_drawing_type", False):
            label = result.graph2d_prediction.get("label")
            conf = result.graph2d_prediction.get("confidence", 0)
            if label and label != result.label:
                alternatives[label] = max(alternatives.get(label, 0), conf * 0.3)

        if result.titleblock_prediction:
            label = result.titleblock_prediction.get("label")
            conf = result.titleblock_prediction.get("confidence", 0)
            if label and label != result.label:
                alternatives[label] = max(alternatives.get(label, 0), conf * 0.2)

        if result.process_prediction:
            labels = result.process_prediction.get("suggested_labels", [])
            conf = result.process_prediction.get("confidence", 0)
            for label in labels:
                if label != result.label:
                    alternatives[label] = max(alternatives.get(label, 0), conf * 0.15)

        # Sort by confidence
        explanation.alternative_labels = sorted(
            [(k, v) for k, v in alternatives.items()],
            key=lambda x: -x[1]
        )[:5]

    def _calculate_uncertainty(self, result: Any, explanation: Explanation) -> None:
        """Calculate prediction uncertainty."""
        uncertainty_sources = []
        uncertainty_score = 0.0

        # Low confidence
        if result.confidence < 0.5:
            uncertainty_score += 0.3
            uncertainty_sources.append("置信度较低")

        # Source disagreement
        predictions = []
        if result.filename_prediction:
            predictions.append(result.filename_prediction.get("label"))
        if result.graph2d_prediction and not result.graph2d_prediction.get("is_drawing_type", False):
            predictions.append(result.graph2d_prediction.get("label"))
        if result.titleblock_prediction:
            predictions.append(result.titleblock_prediction.get("label"))

        predictions = [p for p in predictions if p]
        if len(set(predictions)) > 1:
            disagreement = 1 - (max(predictions.count(p) for p in set(predictions)) / len(predictions))
            uncertainty_score += disagreement * 0.4
            uncertainty_sources.append("多源预测不一致")

        # Missing sources
        if not result.graph2d_prediction:
            uncertainty_score += 0.1
            uncertainty_sources.append("缺少几何分析")
        if not result.titleblock_prediction:
            uncertainty_score += 0.05

        # Fallback decision
        if result.source.value == "fallback":
            uncertainty_score += 0.5
            uncertainty_sources.append("无有效预测")

        explanation.uncertainty_score = min(1.0, uncertainty_score)
        explanation.uncertainty_sources = uncertainty_sources

    def _generate_summary(self, result: Any, explanation: Explanation) -> str:
        """Generate explanation summary."""
        if result.source.value == "filename":
            if "filename_high_conf_adopted" in result.decision_path:
                return f"文件名特征明确指向 {result.label}，置信度高于阈值，直接采用"
            else:
                return f"仅有文件名预测可用，分类为 {result.label}"

        elif result.source.value == "graph2d":
            return f"基于图纸几何特征，分类为 {result.label}"

        elif result.source.value == "titleblock":
            return f"基于标题栏信息，分类为 {result.label}"

        elif result.source.value == "fusion":
            sources = [k for k, v in explanation.source_contributions.items() if v > 0]
            if len(sources) >= 2:
                return f"综合 {', '.join(sources)} 多源信息，融合得出 {result.label}"
            else:
                return f"融合多源信息，分类为 {result.label}"

        else:
            return f"预测结果：{result.label}（来源：{result.source.value}）"

    def _generate_counterfactuals(self, result: Any, explanation: Explanation) -> None:
        """Generate counterfactual explanations."""
        # Find what changes would lead to a different prediction
        for alt_label, alt_conf in explanation.alternative_labels[:2]:
            changes = []

            # What would need to change?
            if result.filename_prediction:
                if result.filename_prediction.get("label") != alt_label:
                    changes.append({
                        "feature": "filename",
                        "from": result.filename_prediction.get("label"),
                        "to": alt_label,
                        "description": f"如果文件名指向 {alt_label}",
                    })

            if changes:
                explanation.counterfactuals.append(Counterfactual(
                    original_label=result.label or "unknown",
                    counterfactual_label=alt_label,
                    changes_needed=changes,
                    distance=1.0 - alt_conf,
                    feasibility=alt_conf,
                ))
