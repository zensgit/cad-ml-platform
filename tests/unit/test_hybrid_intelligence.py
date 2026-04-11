"""
Unit tests for HybridIntelligence.

All tests use plain dict-based branch predictions -- no actual classifier
models are required.
"""

from __future__ import annotations

import math

import pytest

from src.ml.hybrid.intelligence import (
    CalibratedConfidence,
    CrossValidationResult,
    DisagreementReport,
    EnsembleUncertainty,
    HybridIntelligence,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def intel() -> HybridIntelligence:
    return HybridIntelligence()


# ---------------------------------------------------------------------------
# Helper factories for branch predictions
# ---------------------------------------------------------------------------


def _branch(label: str, confidence: float, **extra) -> dict:
    """Shorthand to build a branch prediction dict."""
    d: dict = {"label": label, "confidence": confidence}
    d.update(extra)
    return d


def _process_branch(label: str, confidence: float) -> dict:
    """Process branch stores label under ``suggested_labels``."""
    return {"suggested_labels": [label], "confidence": confidence}


# ---------------------------------------------------------------------------
# EnsembleUncertainty tests
# ---------------------------------------------------------------------------


class TestAnalyzeEnsembleUncertainty:
    def test_unanimous_agreement(self, intel: HybridIntelligence):
        """All branches predict the same label -> low uncertainty."""
        preds = {
            "filename": _branch("法兰盘", 0.9),
            "graph2d": _branch("法兰盘", 0.85),
            "titleblock": _branch("法兰盘", 0.8),
            "process": _process_branch("法兰盘", 0.75),
            "history_sequence": _branch("法兰盘", 0.88),
        }
        u = intel.analyze_ensemble_uncertainty(preds)

        assert isinstance(u, EnsembleUncertainty)
        assert u.agreement_ratio == 1.0
        assert u.vote_entropy == 0.0
        assert u.margin == 1.0
        assert u.epistemic_uncertainty == 0.0
        assert u.severity == "low"

    def test_complete_disagreement(self, intel: HybridIntelligence):
        """All branches predict different labels -> high uncertainty."""
        preds = {
            "filename": _branch("法兰盘", 0.5),
            "graph2d": _branch("轴", 0.4),
            "titleblock": _branch("齿轮", 0.35),
            "process": _process_branch("壳体", 0.3),
            "history_sequence": _branch("管件", 0.3),
        }
        u = intel.analyze_ensemble_uncertainty(preds)

        assert u.agreement_ratio == pytest.approx(0.2, abs=0.01)
        assert u.vote_entropy == pytest.approx(1.0, abs=0.01)  # max entropy for 5 classes
        assert u.margin == pytest.approx(0.0, abs=0.01)
        assert u.epistemic_uncertainty == pytest.approx(0.8, abs=0.01)
        assert u.severity == "high"

    def test_majority_three_of_five(self, intel: HybridIntelligence):
        """3 branches agree, 2 disagree -> moderate uncertainty."""
        preds = {
            "filename": _branch("法兰盘", 0.8),
            "graph2d": _branch("法兰盘", 0.75),
            "titleblock": _branch("法兰盘", 0.7),
            "process": _process_branch("轴", 0.6),
            "history_sequence": _branch("齿轮", 0.5),
        }
        u = intel.analyze_ensemble_uncertainty(preds)

        assert u.agreement_ratio == pytest.approx(0.6, abs=0.01)
        assert 0.0 < u.vote_entropy < 1.0
        assert u.epistemic_uncertainty == pytest.approx(0.4, abs=0.01)
        assert u.severity in ("low", "medium")

    def test_single_branch(self, intel: HybridIntelligence):
        """Only one active branch -> no disagreement possible."""
        preds = {"filename": _branch("法兰盘", 0.9)}
        u = intel.analyze_ensemble_uncertainty(preds)

        assert u.agreement_ratio == 1.0
        assert u.vote_entropy == 0.0
        assert u.epistemic_uncertainty == 0.0
        assert u.severity == "low"

    def test_empty_branches(self, intel: HybridIntelligence):
        """No branches -> max uncertainty."""
        u = intel.analyze_ensemble_uncertainty({})

        assert u.agreement_ratio == 0.0
        assert u.epistemic_uncertainty == 1.0
        assert u.aleatoric_uncertainty == 1.0
        assert u.severity == "high"


# ---------------------------------------------------------------------------
# Disagreement detection tests
# ---------------------------------------------------------------------------


class TestDetectDisagreement:
    def test_no_disagreement(self, intel: HybridIntelligence):
        preds = {
            "filename": _branch("法兰盘", 0.9),
            "graph2d": _branch("法兰盘", 0.85),
            "titleblock": _branch("法兰盘", 0.8),
        }
        report = intel.detect_disagreement(preds)

        assert isinstance(report, DisagreementReport)
        assert report.has_disagreement is False
        assert report.recommended_action == "accept"
        assert report.majority_label == "法兰盘"
        assert len(report.disagreeing_branches) == 0

    def test_majority_wins(self, intel: HybridIntelligence):
        """3 agree, 2 disagree -> flag for review."""
        preds = {
            "filename": _branch("法兰盘", 0.8),
            "graph2d": _branch("法兰盘", 0.7),
            "titleblock": _branch("法兰盘", 0.6),
            "process": _process_branch("轴", 0.5),
            "history_sequence": _branch("轴", 0.4),
        }
        report = intel.detect_disagreement(preds)

        assert report.has_disagreement is True
        assert report.majority_label == "法兰盘"
        assert report.minority_label == "轴"
        assert set(report.disagreeing_branches) == {"process", "history_sequence"}
        assert report.recommended_action == "flag_for_review"

    def test_complete_split_rejects(self, intel: HybridIntelligence):
        """No clear majority -> reject."""
        preds = {
            "filename": _branch("法兰盘", 0.5),
            "graph2d": _branch("轴", 0.5),
            "titleblock": _branch("齿轮", 0.5),
            "process": _process_branch("壳体", 0.5),
            "history_sequence": _branch("管件", 0.5),
        }
        report = intel.detect_disagreement(preds)

        assert report.has_disagreement is True
        assert report.recommended_action == "reject"

    def test_strong_majority_accepts(self, intel: HybridIntelligence):
        """4/5 agree -> accept."""
        preds = {
            "filename": _branch("法兰盘", 0.9),
            "graph2d": _branch("法兰盘", 0.85),
            "titleblock": _branch("法兰盘", 0.8),
            "process": _process_branch("法兰盘", 0.7),
            "history_sequence": _branch("轴", 0.4),
        }
        report = intel.detect_disagreement(preds)

        assert report.has_disagreement is True
        assert report.recommended_action == "accept"
        assert report.disagreeing_branches == ["history_sequence"]

    def test_explanation_mentions_branches(self, intel: HybridIntelligence):
        preds = {
            "filename": _branch("法兰盘", 0.9),
            "graph2d": _branch("轴", 0.7),
        }
        report = intel.detect_disagreement(preds)
        assert "几何分析" in report.explanation or "graph2d" in report.explanation


# ---------------------------------------------------------------------------
# Cross-validation tests
# ---------------------------------------------------------------------------


class TestCrossValidatePrediction:
    def test_consistent_prediction(self, intel: HybridIntelligence):
        prediction = {"label": "法兰盘", "confidence": 0.9}
        branches = {
            "filename": _branch("法兰盘", 0.9),
            "graph2d": _branch("法兰盘", 0.85),
        }
        result = intel.cross_validate_prediction(prediction, branches)

        assert isinstance(result, CrossValidationResult)
        assert result.is_consistent is True
        assert len(result.inconsistencies) == 0

    def test_catches_inconsistency(self, intel: HybridIntelligence):
        """Filename says '轴' but graph2d says '法兰盘' -> warning/inconsistency."""
        prediction = {"label": "法兰盘", "confidence": 0.8}
        branches = {
            "filename": _branch("轴", 0.85),
            "graph2d": _branch("法兰盘", 0.8),
        }
        result = intel.cross_validate_prediction(prediction, branches)

        assert result.is_consistent is False
        assert len(result.inconsistencies) >= 1
        assert any("轴" in i for i in result.inconsistencies)

    def test_low_confidence_branch_produces_warning(self, intel: HybridIntelligence):
        """Low-confidence disagreement is a warning, not an inconsistency."""
        prediction = {"label": "法兰盘", "confidence": 0.8}
        branches = {
            "filename": _branch("轴", 0.3),
            "graph2d": _branch("法兰盘", 0.8),
        }
        result = intel.cross_validate_prediction(prediction, branches)

        assert result.is_consistent is True
        assert len(result.warnings) >= 1

    def test_process_branch_inconsistency(self, intel: HybridIntelligence):
        prediction = {"label": "法兰盘", "confidence": 0.8}
        branches = {
            "process": _process_branch("轴", 0.7),
        }
        result = intel.cross_validate_prediction(prediction, branches)
        # Process branch at 0.7 produces a warning
        assert len(result.warnings) >= 1

    def test_no_label_prediction(self, intel: HybridIntelligence):
        prediction = {"label": None, "confidence": 0.0}
        branches = {"filename": _branch("轴", 0.9)}
        result = intel.cross_validate_prediction(prediction, branches)
        assert result.is_consistent is True
        assert len(result.warnings) >= 1


# ---------------------------------------------------------------------------
# Calibrated confidence tests
# ---------------------------------------------------------------------------


class TestComputeCalibratedConfidence:
    def test_high_agreement_boosts(self, intel: HybridIntelligence):
        """Full agreement should not reduce raw confidence."""
        preds = {
            "filename": _branch("法兰盘", 0.9),
            "graph2d": _branch("法兰盘", 0.85),
            "titleblock": _branch("法兰盘", 0.8),
        }
        cal = intel.compute_calibrated_confidence(0.85, preds)

        assert isinstance(cal, CalibratedConfidence)
        assert cal.calibrated_confidence >= 0.85
        assert cal.reliability == "high"
        assert cal.confidence_interval[0] <= cal.calibrated_confidence
        assert cal.confidence_interval[1] >= cal.calibrated_confidence

    def test_disagreement_reduces_confidence(self, intel: HybridIntelligence):
        """Disagreement should lower the calibrated confidence."""
        agreeing = {
            "filename": _branch("法兰盘", 0.9),
            "graph2d": _branch("法兰盘", 0.85),
            "titleblock": _branch("法兰盘", 0.8),
        }
        disagreeing = {
            "filename": _branch("法兰盘", 0.9),
            "graph2d": _branch("轴", 0.85),
            "titleblock": _branch("齿轮", 0.8),
        }

        cal_agree = intel.compute_calibrated_confidence(0.85, agreeing)
        cal_disagree = intel.compute_calibrated_confidence(0.85, disagreeing)

        assert cal_disagree.calibrated_confidence < cal_agree.calibrated_confidence

    def test_single_branch_penalty(self, intel: HybridIntelligence):
        """Single branch -> diversity penalty reduces confidence."""
        preds = {"filename": _branch("法兰盘", 0.9)}
        cal = intel.compute_calibrated_confidence(0.9, preds)

        # Should be lower than 0.9 due to diversity penalty
        assert cal.calibrated_confidence < 0.9

    def test_historical_accuracy_blending(self, intel: HybridIntelligence):
        """Historical accuracy should influence calibrated confidence."""
        intel.record_branch_accuracy("filename", 0.5)
        intel.record_branch_accuracy("graph2d", 0.5)

        preds = {
            "filename": _branch("法兰盘", 0.9),
            "graph2d": _branch("法兰盘", 0.85),
        }
        cal = intel.compute_calibrated_confidence(0.95, preds)

        # Low historical accuracy should pull confidence down
        assert cal.calibrated_confidence < 0.95

    def test_confidence_interval_bounds(self, intel: HybridIntelligence):
        preds = {"filename": _branch("法兰盘", 0.5)}
        cal = intel.compute_calibrated_confidence(0.5, preds)

        lower, upper = cal.confidence_interval
        assert 0.0 <= lower <= cal.calibrated_confidence
        assert cal.calibrated_confidence <= upper <= 1.0


# ---------------------------------------------------------------------------
# Smart explanation tests
# ---------------------------------------------------------------------------


class TestGenerateSmartExplanation:
    def test_mentions_branches(self, intel: HybridIntelligence):
        preds = {
            "filename": _branch("法兰盘", 0.9, part_name="法兰"),
            "graph2d": _branch("法兰盘", 0.85),
        }
        prediction = {"label": "法兰盘", "confidence": 0.88}
        u = intel.analyze_ensemble_uncertainty(preds)
        explanation = intel.generate_smart_explanation(prediction, preds, u)

        assert "法兰盘" in explanation
        assert "文件名" in explanation
        assert "几何分析" in explanation or "几何" in explanation

    def test_mentions_disagreement(self, intel: HybridIntelligence):
        preds = {
            "filename": _branch("法兰盘", 0.9),
            "graph2d": _branch("轴", 0.85),
        }
        prediction = {"label": "法兰盘", "confidence": 0.7}
        u = intel.analyze_ensemble_uncertainty(preds)
        explanation = intel.generate_smart_explanation(prediction, preds, u)

        # Should mention the disagreeing label
        assert "轴" in explanation

    def test_single_branch_explanation(self, intel: HybridIntelligence):
        preds = {"filename": _branch("法兰盘", 0.9)}
        prediction = {"label": "法兰盘", "confidence": 0.9}
        u = intel.analyze_ensemble_uncertainty(preds)
        explanation = intel.generate_smart_explanation(prediction, preds, u)

        assert "单一分支" in explanation or "1/1" in explanation

    def test_titleblock_material_mentioned(self, intel: HybridIntelligence):
        preds = {
            "titleblock": _branch("法兰盘", 0.8, material="碳钢"),
        }
        prediction = {"label": "法兰盘", "confidence": 0.8}
        u = intel.analyze_ensemble_uncertainty(preds)
        explanation = intel.generate_smart_explanation(prediction, preds, u)

        assert "碳钢" in explanation


# ---------------------------------------------------------------------------
# Suggest action tests
# ---------------------------------------------------------------------------


class TestSuggestNextAction:
    def test_high_confidence(self, intel: HybridIntelligence):
        prediction = {"label": "法兰盘", "confidence": 0.92}
        u = EnsembleUncertainty(
            vote_entropy=0.0,
            agreement_ratio=1.0,
            margin=1.0,
            epistemic_uncertainty=0.0,
            aleatoric_uncertainty=0.1,
            severity="low",
        )
        action = intel.suggest_next_action(prediction, u)
        assert "直接使用" in action

    def test_low_confidence(self, intel: HybridIntelligence):
        prediction = {"label": "法兰盘", "confidence": 0.2}
        u = EnsembleUncertainty(
            vote_entropy=1.0,
            agreement_ratio=0.2,
            margin=0.0,
            epistemic_uncertainty=0.8,
            aleatoric_uncertainty=0.8,
            severity="high",
        )
        action = intel.suggest_next_action(prediction, u)
        assert "上传" in action or "图纸" in action or "零件名称" in action

    def test_medium_confidence(self, intel: HybridIntelligence):
        prediction = {"label": "法兰盘", "confidence": 0.55}
        u = EnsembleUncertainty(
            vote_entropy=0.5,
            agreement_ratio=0.6,
            margin=0.2,
            epistemic_uncertainty=0.4,
            aleatoric_uncertainty=0.4,
            severity="medium",
        )
        action = intel.suggest_next_action(prediction, u)
        assert "确认" in action or "成本估算" in action

    def test_disagreement_suggests_review(self, intel: HybridIntelligence):
        prediction = {"label": "法兰盘", "confidence": 0.65}
        u = EnsembleUncertainty(
            vote_entropy=0.8,
            agreement_ratio=0.4,
            margin=0.1,
            epistemic_uncertainty=0.6,
            aleatoric_uncertainty=0.3,
            severity="medium",
        )
        action = intel.suggest_next_action(prediction, u)
        assert "人工审核" in action or "确认" in action


# ---------------------------------------------------------------------------
# Dataclass serialisation sanity checks
# ---------------------------------------------------------------------------


class TestDataclassSerialization:
    def test_ensemble_uncertainty_to_dict(self):
        u = EnsembleUncertainty(0.5, 0.6, 0.2, 0.4, 0.3, "medium")
        d = u.to_dict()
        assert d["vote_entropy"] == 0.5
        assert d["severity"] == "medium"

    def test_disagreement_report_to_dict(self):
        r = DisagreementReport(True, ["graph2d"], "法兰盘", "轴", "flag_for_review", "test")
        d = r.to_dict()
        assert d["has_disagreement"] is True

    def test_cross_validation_result_to_dict(self):
        c = CrossValidationResult(False, ["issue1"], ["warn1"])
        d = c.to_dict()
        assert d["is_consistent"] is False
        assert len(d["inconsistencies"]) == 1

    def test_calibrated_confidence_to_dict(self):
        cc = CalibratedConfidence(0.82, (0.7, 0.9), "high")
        d = cc.to_dict()
        assert d["calibrated_confidence"] == 0.82
        assert d["confidence_interval"] == [0.7, 0.9]
