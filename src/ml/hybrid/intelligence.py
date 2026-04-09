"""
Intelligence layer for the HybridClassifier.

Adds ensemble uncertainty quantification, disagreement detection,
cross-validation between classifier branches, calibrated confidence
scoring, and context-aware explanation generation.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Branch names recognised by the intelligence layer
# ---------------------------------------------------------------------------
BRANCH_NAMES = ("filename", "graph2d", "titleblock", "process", "history_sequence")

# Human-readable Chinese labels for each branch
BRANCH_DISPLAY_NAMES: Dict[str, str] = {
    "filename": "文件名",
    "graph2d": "几何分析",
    "titleblock": "标题栏",
    "process": "工艺特征",
    "history_sequence": "历史序列",
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EnsembleUncertainty:
    """Quantified uncertainty from the ensemble of classifier branches."""

    vote_entropy: float  # Shannon entropy of label votes (higher = more uncertain)
    agreement_ratio: float  # Fraction of branches agreeing on the top label
    margin: float  # Score gap between the top-2 voted labels
    epistemic_uncertainty: float  # High when branches disagree (model doesn't know)
    aleatoric_uncertainty: float  # High when all branches have low confidence
    severity: str  # "low" / "medium" / "high"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vote_entropy": self.vote_entropy,
            "agreement_ratio": self.agreement_ratio,
            "margin": self.margin,
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "aleatoric_uncertainty": self.aleatoric_uncertainty,
            "severity": self.severity,
        }


@dataclass
class DisagreementReport:
    """Report on inter-branch disagreement."""

    has_disagreement: bool
    disagreeing_branches: List[str]
    majority_label: Optional[str]
    minority_label: Optional[str]
    recommended_action: str  # "accept" / "flag_for_review" / "reject"
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_disagreement": self.has_disagreement,
            "disagreeing_branches": self.disagreeing_branches,
            "majority_label": self.majority_label,
            "minority_label": self.minority_label,
            "recommended_action": self.recommended_action,
            "explanation": self.explanation,
        }


@dataclass
class CrossValidationResult:
    """Result of cross-validating the final prediction against branch signals."""

    is_consistent: bool
    inconsistencies: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_consistent": self.is_consistent,
            "inconsistencies": self.inconsistencies,
            "warnings": self.warnings,
        }


@dataclass
class CalibratedConfidence:
    """A more honest confidence score after ensemble calibration."""

    calibrated_confidence: float
    confidence_interval: Tuple[float, float]  # 90 % interval (lower, upper)
    reliability: str  # "high" / "medium" / "low"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibrated_confidence": self.calibrated_confidence,
            "confidence_interval": list(self.confidence_interval),
            "reliability": self.reliability,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_label(branch_pred: Dict[str, Any]) -> Optional[str]:
    """Extract the predicted label from a branch prediction dict.

    Some branches store labels under ``"label"``, others under
    ``"suggested_labels"`` (process branch).
    """
    label = branch_pred.get("label")
    if label is not None:
        return str(label)
    suggested = branch_pred.get("suggested_labels")
    if isinstance(suggested, list) and suggested:
        return str(suggested[0])
    return None


def _extract_confidence(branch_pred: Dict[str, Any]) -> float:
    """Extract the confidence score from a branch prediction dict."""
    return float(branch_pred.get("confidence", 0.0))


def _shannon_entropy(probs: List[float]) -> float:
    """Compute Shannon entropy for a discrete probability distribution.

    *probs* need not be normalised -- they will be normalised internally.
    Returns 0.0 when the distribution is degenerate.
    """
    total = sum(probs)
    if total <= 0:
        return 0.0
    normed = [p / total for p in probs if p > 0]
    return -sum(p * math.log2(p) for p in normed)


def _max_entropy(n_classes: int) -> float:
    """Maximum possible Shannon entropy for *n_classes* uniform classes."""
    if n_classes <= 1:
        return 0.0
    return math.log2(n_classes)


# ---------------------------------------------------------------------------
# HybridIntelligence
# ---------------------------------------------------------------------------


class HybridIntelligence:
    """Intelligence layer on top of the hybrid classifier.

    Adds: disagreement detection, ensemble uncertainty, adaptive confidence,
    and cross-validation between classifier branches.

    All public methods accept plain ``dict`` branch predictions so that the
    class can be used without importing the actual hybrid classifier.
    """

    def __init__(self) -> None:
        self._branch_accuracy: Dict[str, float] = {}
        self._disagreement_history: List[DisagreementReport] = []

    # ------------------------------------------------------------------
    # Branch accuracy tracking
    # ------------------------------------------------------------------

    def record_branch_accuracy(self, branch_name: str, accuracy: float) -> None:
        """Record observed accuracy for a branch (0-1 scale)."""
        self._branch_accuracy[branch_name] = max(0.0, min(1.0, accuracy))

    # ------------------------------------------------------------------
    # Ensemble uncertainty
    # ------------------------------------------------------------------

    def analyze_ensemble_uncertainty(
        self, branch_predictions: Dict[str, Dict[str, Any]]
    ) -> EnsembleUncertainty:
        """Quantify prediction uncertainty from the ensemble.

        Parameters
        ----------
        branch_predictions:
            Mapping of ``branch_name -> prediction_dict``.  Each prediction
            dict must contain at least ``"label"`` (or ``"suggested_labels"``)
            and ``"confidence"``.

        Returns
        -------
        EnsembleUncertainty
            Detailed uncertainty breakdown.
        """
        labels: List[str] = []
        confidences: List[float] = []

        for _name, pred in branch_predictions.items():
            label = _extract_label(pred)
            if label is not None:
                labels.append(label)
                confidences.append(_extract_confidence(pred))

        n_branches = len(labels)

        # --- edge case: zero or one active branch ---
        if n_branches == 0:
            return EnsembleUncertainty(
                vote_entropy=0.0,
                agreement_ratio=0.0,
                margin=0.0,
                epistemic_uncertainty=1.0,
                aleatoric_uncertainty=1.0,
                severity="high",
            )
        if n_branches == 1:
            conf = confidences[0]
            return EnsembleUncertainty(
                vote_entropy=0.0,
                agreement_ratio=1.0,
                margin=1.0,
                epistemic_uncertainty=0.0,
                aleatoric_uncertainty=max(0.0, 1.0 - conf),
                severity="low" if conf >= 0.7 else ("medium" if conf >= 0.4 else "high"),
            )

        # --- vote counts ---
        vote_counts = Counter(labels)
        sorted_counts = vote_counts.most_common()
        top_count = sorted_counts[0][1]
        second_count = sorted_counts[1][1] if len(sorted_counts) > 1 else 0

        # --- vote entropy (normalised to [0, 1]) ---
        raw_entropy = _shannon_entropy([c for _, c in sorted_counts])
        max_ent = _max_entropy(len(sorted_counts))
        vote_entropy = raw_entropy / max_ent if max_ent > 0 else 0.0

        # --- agreement ratio ---
        agreement_ratio = top_count / n_branches

        # --- margin between top-2 labels ---
        margin = (top_count - second_count) / n_branches

        # --- epistemic uncertainty: disagreement among branches ---
        epistemic_uncertainty = 1.0 - agreement_ratio

        # --- aleatoric uncertainty: average low confidence ---
        mean_conf = sum(confidences) / n_branches
        aleatoric_uncertainty = max(0.0, 1.0 - mean_conf)

        # --- severity ---
        combined = 0.6 * epistemic_uncertainty + 0.4 * aleatoric_uncertainty
        if combined < 0.3:
            severity = "low"
        elif combined < 0.6:
            severity = "medium"
        else:
            severity = "high"

        return EnsembleUncertainty(
            vote_entropy=round(vote_entropy, 4),
            agreement_ratio=round(agreement_ratio, 4),
            margin=round(margin, 4),
            epistemic_uncertainty=round(epistemic_uncertainty, 4),
            aleatoric_uncertainty=round(aleatoric_uncertainty, 4),
            severity=severity,
        )

    # ------------------------------------------------------------------
    # Disagreement detection
    # ------------------------------------------------------------------

    def detect_disagreement(
        self, branch_predictions: Dict[str, Dict[str, Any]]
    ) -> DisagreementReport:
        """Detect when classifier branches significantly disagree.

        Returns a :class:`DisagreementReport` containing the nature of the
        disagreement and a recommended action.
        """
        labels_by_branch: Dict[str, str] = {}
        for name, pred in branch_predictions.items():
            label = _extract_label(pred)
            if label is not None:
                labels_by_branch[name] = label

        n_active = len(labels_by_branch)

        if n_active <= 1:
            report = DisagreementReport(
                has_disagreement=False,
                disagreeing_branches=[],
                majority_label=next(iter(labels_by_branch.values()), None),
                minority_label=None,
                recommended_action="accept",
                explanation="分支数量不足，无法检测分歧。" if n_active == 0 else "仅一个分支有效，无分歧。",
            )
            self._disagreement_history.append(report)
            return report

        vote_counts = Counter(labels_by_branch.values())
        sorted_labels = vote_counts.most_common()
        majority_label = sorted_labels[0][0]
        majority_count = sorted_labels[0][1]

        # Branches that disagree with the majority
        disagreeing_branches = [
            name for name, lbl in labels_by_branch.items() if lbl != majority_label
        ]
        minority_label = sorted_labels[1][0] if len(sorted_labels) > 1 else None

        has_disagreement = len(disagreeing_branches) > 0

        # Recommended action
        agreement_ratio = majority_count / n_active
        if agreement_ratio >= 0.8:
            recommended_action = "accept"
        elif agreement_ratio >= 0.5:
            recommended_action = "flag_for_review"
        else:
            recommended_action = "reject"

        # Human-readable explanation
        if not has_disagreement:
            explanation = f"所有 {n_active} 个分支一致预测为「{majority_label}」。"
        else:
            agreeing_names = [
                BRANCH_DISPLAY_NAMES.get(n, n)
                for n in labels_by_branch
                if labels_by_branch[n] == majority_label
            ]
            dissenting_names = [
                BRANCH_DISPLAY_NAMES.get(n, n) for n in disagreeing_branches
            ]
            explanation = (
                f"多数分支（{', '.join(agreeing_names)}）预测为「{majority_label}」，"
                f"但 {', '.join(dissenting_names)} 预测为「{minority_label}」。"
            )

        report = DisagreementReport(
            has_disagreement=has_disagreement,
            disagreeing_branches=disagreeing_branches,
            majority_label=majority_label,
            minority_label=minority_label,
            recommended_action=recommended_action,
            explanation=explanation,
        )
        self._disagreement_history.append(report)
        return report

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate_prediction(
        self,
        prediction: Dict[str, Any],
        branch_predictions: Dict[str, Dict[str, Any]],
    ) -> CrossValidationResult:
        """Cross-validate the final prediction against individual branch signals.

        Checks:
        1. Is the final label consistent with the filename hint?
        2. Does the geometric analysis (graph2d) support this classification?
        3. Does the titleblock info match?
        4. Is the process recommendation consistent with the part type?
        5. Does the history sequence agree?

        Returns inconsistencies and warnings.
        """
        final_label = prediction.get("label")
        inconsistencies: List[str] = []
        warnings: List[str] = []

        if final_label is None:
            return CrossValidationResult(
                is_consistent=True,
                inconsistencies=[],
                warnings=["最终预测无标签，跳过交叉验证。"],
            )

        # --- filename ---
        fn_pred = branch_predictions.get("filename")
        if fn_pred is not None:
            fn_label = _extract_label(fn_pred)
            fn_conf = _extract_confidence(fn_pred)
            if fn_label is not None and fn_label != final_label:
                if fn_conf >= 0.7:
                    inconsistencies.append(
                        f"文件名分支以 {fn_conf:.0%} 置信度预测为「{fn_label}」，"
                        f"与最终预测「{final_label}」不一致。"
                    )
                else:
                    warnings.append(
                        f"文件名分支预测为「{fn_label}」（置信度 {fn_conf:.0%}），"
                        f"与最终预测「{final_label}」不同。"
                    )

        # --- graph2d ---
        g2d_pred = branch_predictions.get("graph2d")
        if g2d_pred is not None:
            g2d_label = _extract_label(g2d_pred)
            g2d_conf = _extract_confidence(g2d_pred)
            is_drawing_type = g2d_pred.get("is_drawing_type", False)
            if g2d_label is not None and g2d_label != final_label and not is_drawing_type:
                if g2d_conf >= 0.6:
                    inconsistencies.append(
                        f"几何分析分支以 {g2d_conf:.0%} 置信度预测为「{g2d_label}」，"
                        f"与最终预测「{final_label}」不一致。"
                    )
                else:
                    warnings.append(
                        f"几何分析分支预测为「{g2d_label}」（置信度 {g2d_conf:.0%}），"
                        f"与最终预测「{final_label}」不同。"
                    )

        # --- titleblock ---
        tb_pred = branch_predictions.get("titleblock")
        if tb_pred is not None:
            tb_label = _extract_label(tb_pred)
            tb_conf = _extract_confidence(tb_pred)
            if tb_label is not None and tb_label != final_label:
                if tb_conf >= 0.6:
                    inconsistencies.append(
                        f"标题栏分支以 {tb_conf:.0%} 置信度预测为「{tb_label}」，"
                        f"与最终预测「{final_label}」不一致。"
                    )
                else:
                    warnings.append(
                        f"标题栏分支预测为「{tb_label}」（置信度 {tb_conf:.0%}），"
                        f"与最终预测「{final_label}」不同。"
                    )

        # --- process ---
        proc_pred = branch_predictions.get("process")
        if proc_pred is not None:
            proc_label = _extract_label(proc_pred)
            proc_conf = _extract_confidence(proc_pred)
            if proc_label is not None and proc_label != final_label:
                if proc_conf >= 0.5:
                    warnings.append(
                        f"工艺特征分支建议「{proc_label}」（置信度 {proc_conf:.0%}），"
                        f"与最终预测「{final_label}」不同，请确认工艺一致性。"
                    )

        # --- history_sequence ---
        hist_pred = branch_predictions.get("history_sequence")
        if hist_pred is not None:
            hist_label = _extract_label(hist_pred)
            hist_conf = _extract_confidence(hist_pred)
            if hist_label is not None and hist_label != final_label:
                if hist_conf >= 0.7:
                    inconsistencies.append(
                        f"历史序列分支以 {hist_conf:.0%} 置信度预测为「{hist_label}」，"
                        f"与最终预测「{final_label}」不一致。"
                    )
                else:
                    warnings.append(
                        f"历史序列分支预测为「{hist_label}」（置信度 {hist_conf:.0%}），"
                        f"与最终预测「{final_label}」不同。"
                    )

        is_consistent = len(inconsistencies) == 0

        return CrossValidationResult(
            is_consistent=is_consistent,
            inconsistencies=inconsistencies,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Calibrated confidence
    # ------------------------------------------------------------------

    def compute_calibrated_confidence(
        self,
        raw_confidence: float,
        branch_predictions: Dict[str, Dict[str, Any]],
    ) -> CalibratedConfidence:
        """Produce a more honest confidence score.

        Adjusts *raw_confidence* based on:
        - Ensemble agreement (high agreement boosts, low reduces)
        - Historical accuracy of the predicted class (from ``_branch_accuracy``)
        - Branch contribution diversity

        Returns a :class:`CalibratedConfidence` with the adjusted score, a
        90 % confidence interval, and a reliability label.
        """
        uncertainty = self.analyze_ensemble_uncertainty(branch_predictions)

        # --- agreement adjustment ---
        # agreement_ratio in [0, 1]; centre at 0.5 so equal agreement is neutral
        agreement_delta = (uncertainty.agreement_ratio - 0.5) * 0.3
        adjusted = raw_confidence + agreement_delta

        # --- historical accuracy adjustment ---
        if self._branch_accuracy:
            acc_values = list(self._branch_accuracy.values())
            mean_acc = sum(acc_values) / len(acc_values)
            # Shrink towards the mean historical accuracy
            adjusted = 0.8 * adjusted + 0.2 * mean_acc

        # --- diversity penalty ---
        # If very few branches contributed, reduce confidence
        n_active = sum(
            1 for pred in branch_predictions.values()
            if _extract_label(pred) is not None
        )
        if n_active <= 1:
            adjusted *= 0.85
        elif n_active == 2:
            adjusted *= 0.92

        # Clamp
        calibrated = max(0.0, min(1.0, adjusted))

        # --- confidence interval ---
        # Width driven by epistemic + aleatoric uncertainty
        half_width = 0.05 + 0.20 * uncertainty.epistemic_uncertainty + 0.15 * uncertainty.aleatoric_uncertainty
        lower = max(0.0, calibrated - half_width)
        upper = min(1.0, calibrated + half_width)

        # --- reliability ---
        if calibrated >= 0.75 and uncertainty.severity == "low":
            reliability = "high"
        elif calibrated >= 0.45 or uncertainty.severity == "medium":
            reliability = "medium"
        else:
            reliability = "low"

        return CalibratedConfidence(
            calibrated_confidence=round(calibrated, 4),
            confidence_interval=(round(lower, 4), round(upper, 4)),
            reliability=reliability,
        )

    # ------------------------------------------------------------------
    # Smart explanation
    # ------------------------------------------------------------------

    def generate_smart_explanation(
        self,
        prediction: Dict[str, Any],
        branch_predictions: Dict[str, Dict[str, Any]],
        uncertainty: EnsembleUncertainty,
    ) -> str:
        """Generate an intelligent, context-aware explanation.

        Produces a narrative that references specific branch signals and
        their agreement / disagreement rather than just listing numbers.
        """
        final_label = prediction.get("label", "未知")
        final_conf = prediction.get("confidence", 0.0)

        parts: List[str] = []

        # --- Opening with primary evidence ---
        fn_pred = branch_predictions.get("filename")
        if fn_pred is not None:
            fn_label = _extract_label(fn_pred)
            fn_conf = _extract_confidence(fn_pred)
            fn_pattern = fn_pred.get("matched_pattern") or fn_pred.get("part_name")
            if fn_label is not None:
                pattern_hint = f"「{fn_pattern}」" if fn_pattern else "文件名特征"
                parts.append(
                    f"根据{pattern_hint}，文件名分支预测为「{fn_label}」（置信度 {fn_conf:.0%}）。"
                )

        # --- Geometric evidence ---
        g2d_pred = branch_predictions.get("graph2d")
        if g2d_pred is not None:
            g2d_label = _extract_label(g2d_pred)
            g2d_conf = _extract_confidence(g2d_pred)
            if g2d_label is not None and not g2d_pred.get("is_drawing_type", False):
                features_desc = g2d_pred.get("features_summary", "")
                feature_str = f"（{features_desc}）" if features_desc else ""
                verb = "支持" if g2d_label == final_label else "指向"
                parts.append(
                    f"几何分析{feature_str}{verb}「{g2d_label}」（置信度 {g2d_conf:.0%}）。"
                )

        # --- Titleblock evidence ---
        tb_pred = branch_predictions.get("titleblock")
        if tb_pred is not None:
            tb_label = _extract_label(tb_pred)
            tb_conf = _extract_confidence(tb_pred)
            if tb_label is not None:
                material = tb_pred.get("material", "")
                mat_str = f"，材料为「{material}」" if material else ""
                verb = "一致" if tb_label == final_label else "指向"
                parts.append(
                    f"标题栏信息{verb}「{tb_label}」（置信度 {tb_conf:.0%}）{mat_str}。"
                )

        # --- Process evidence ---
        proc_pred = branch_predictions.get("process")
        if proc_pred is not None:
            proc_label = _extract_label(proc_pred)
            proc_conf = _extract_confidence(proc_pred)
            if proc_label is not None:
                verb = "与之一致" if proc_label == final_label else f"建议「{proc_label}」"
                parts.append(f"工艺特征分支{verb}（置信度 {proc_conf:.0%}）。")

        # --- History evidence ---
        hist_pred = branch_predictions.get("history_sequence")
        if hist_pred is not None:
            hist_label = _extract_label(hist_pred)
            hist_conf = _extract_confidence(hist_pred)
            if hist_label is not None:
                verb = "印证" if hist_label == final_label else f"倾向「{hist_label}」"
                parts.append(f"历史序列{verb}（置信度 {hist_conf:.0%}）。")

        # --- Agreement summary ---
        n_active = sum(
            1 for p in branch_predictions.values() if _extract_label(p) is not None
        )
        n_agree = sum(
            1
            for p in branch_predictions.values()
            if _extract_label(p) == final_label
        )
        if n_active > 1:
            parts.append(
                f"综合判定：{n_agree}/{n_active} 个活跃分支支持「{final_label}」，"
                f"置信度{_confidence_word(final_conf)}（{final_conf:.0%}），"
                f"不确定性{uncertainty.severity}。"
            )
        elif n_active == 1:
            parts.append(
                f"仅单一分支有效，最终预测为「{final_label}」（{final_conf:.0%}）。"
            )
        else:
            parts.append("无活跃分支，预测结果不可靠。")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Suggest next action
    # ------------------------------------------------------------------

    def suggest_next_action(
        self,
        prediction: Dict[str, Any],
        uncertainty: EnsembleUncertainty,
    ) -> str:
        """Suggest what the user should do next based on the analysis.

        Returns a Chinese-language recommendation string.
        """
        conf = prediction.get("confidence", 0.0)

        if uncertainty.severity == "high" or conf < 0.3:
            return "置信度较低，请上传更清晰的图纸或提供零件名称。"

        if uncertainty.has_disagreement if hasattr(uncertainty, "has_disagreement") else uncertainty.epistemic_uncertainty > 0.5:
            # Check if there is significant disagreement
            if uncertainty.epistemic_uncertainty > 0.5:
                return "分类器意见不一致，建议人工审核。"

        if uncertainty.severity == "medium" or conf < 0.6:
            return "建议确认零件类型后再进行成本估算。"

        # High confidence
        return "分类结果可直接使用，建议查看推荐工艺。"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _confidence_word(conf: float) -> str:
    """Map a confidence float to a Chinese descriptor."""
    if conf >= 0.85:
        return "很高"
    if conf >= 0.7:
        return "较高"
    if conf >= 0.5:
        return "中等"
    if conf >= 0.3:
        return "较低"
    return "很低"
