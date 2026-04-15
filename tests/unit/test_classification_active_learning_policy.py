from __future__ import annotations

import os
from unittest.mock import patch

from src.core.classification.active_learning_policy import (
    flag_classification_for_review,
)


def _payload(**overrides):
    payload = {
        "part_type": "pump_plate",
        "confidence": 0.41,
        "alternatives": [{"label": "plate", "confidence": 0.2}],
        "coarse_part_type": "plate",
        "fine_part_type": "pump_plate",
        "coarse_hybrid_label": "plate",
        "coarse_graph2d_label": "plate",
        "rule_version": "HybridClassifier-v1",
        "model_version": "v1",
        "confidence_source": "hybrid",
        "confidence_breakdown": {"hybrid": 0.41},
        "hybrid_rejection": {"reason": "margin_too_low"},
        "decision_path": ["filename_only"],
        "source_contributions": {"filename": 1.0},
        "history_prediction": {"label": "plate"},
        "fusion_metadata": {"shadow_predictions": {"v16": "plate"}},
        "hybrid_explanation": {"summary": "filename matched"},
        "knowledge_checks": [{"id": "k1"}],
        "violations": [{"id": "v1"}],
        "standards_candidates": ["STD-1"],
        "branch_conflicts": {"hybrid_vs_graph2d": True},
        "needs_review": True,
        "confidence_band": "low",
        "review_priority": "high",
        "review_priority_score": 90,
        "review_reasons": ["branch_conflict", "hybrid_rejection"],
        "review_has_knowledge_conflict": False,
        "review_has_branch_conflict": True,
        "review_has_hybrid_rejection": True,
        "review_is_low_confidence": True,
    }
    payload.update(overrides)
    return payload


def test_flag_classification_for_review_noops_when_disabled():
    assert (
        flag_classification_for_review(
            analysis_id="doc-1",
            cls_payload=_payload(),
            active_learning_enabled=False,
        )
        is None
    )


def test_flag_classification_for_review_noops_without_review_flag():
    assert (
        flag_classification_for_review(
            analysis_id="doc-1",
            cls_payload=_payload(needs_review=False),
            active_learning_enabled=True,
        )
        is None
    )


def test_flag_classification_for_review_queues_branch_conflict_sample():
    captured = {}

    class DummyLearner:
        def flag_for_review(self, **kwargs):
            captured.update(kwargs)
            return {"status": "queued", "doc_id": kwargs["doc_id"]}

    with patch(
        "src.core.active_learning.get_active_learner",
        return_value=DummyLearner(),
    ):
        result = flag_classification_for_review(
            analysis_id="doc-42",
            cls_payload=_payload(),
            active_learning_enabled=True,
        )

    assert result == {"status": "queued", "doc_id": "doc-42"}
    assert captured["doc_id"] == "doc-42"
    assert captured["predicted_type"] == "pump_plate"
    assert captured["sample_type"] == "branch_conflict"
    assert captured["feedback_priority"] == "high"
    assert captured["uncertainty_reason"] == "branch_conflict+hybrid_rejection"
    assert captured["score_breakdown"]["shadow_predictions"] == {"v16": "plate"}
    assert captured["score_breakdown"]["review_has_branch_conflict"] is True
    assert captured["score_breakdown"]["hybrid_rejection"] == {
        "reason": "margin_too_low"
    }


def test_flag_classification_for_review_prefers_knowledge_conflict_sample_type():
    captured = {}

    class DummyLearner:
        def flag_for_review(self, **kwargs):
            captured.update(kwargs)
            return kwargs

    with patch(
        "src.core.active_learning.get_active_learner",
        return_value=DummyLearner(),
    ):
        flag_classification_for_review(
            analysis_id="doc-43",
            cls_payload=_payload(
                review_has_knowledge_conflict=True,
                review_has_branch_conflict=True,
                review_reasons=["knowledge_conflict"],
            ),
            active_learning_enabled=True,
        )

    assert captured["sample_type"] == "knowledge_conflict"
    assert captured["uncertainty_reason"] == "knowledge_conflict"


def test_flag_classification_for_review_uses_hybrid_rejection_sample_type():
    captured = {}

    class DummyLearner:
        def flag_for_review(self, **kwargs):
            captured.update(kwargs)
            return kwargs

    with patch(
        "src.core.active_learning.get_active_learner",
        return_value=DummyLearner(),
    ):
        flag_classification_for_review(
            analysis_id="doc-44",
            cls_payload=_payload(
                review_has_branch_conflict=False,
                review_has_hybrid_rejection=True,
                review_is_low_confidence=False,
                review_reasons=["hybrid_rejection"],
            ),
            active_learning_enabled=True,
        )

    assert captured["sample_type"] == "hybrid_rejection"
    assert captured["uncertainty_reason"] == "hybrid_rejection"


def test_flag_classification_for_review_uses_low_confidence_sample_type():
    captured = {}

    class DummyLearner:
        def flag_for_review(self, **kwargs):
            captured.update(kwargs)
            return kwargs

    with patch(
        "src.core.active_learning.get_active_learner",
        return_value=DummyLearner(),
    ):
        flag_classification_for_review(
            analysis_id="doc-45",
            cls_payload=_payload(
                review_has_branch_conflict=False,
                review_has_hybrid_rejection=False,
                review_is_low_confidence=True,
                review_reasons=[],
            ),
            active_learning_enabled=True,
        )

    assert captured["sample_type"] == "low_confidence"
    assert captured["uncertainty_reason"] == "low_confidence"


def test_flag_classification_for_review_uses_env_gate_when_flag_not_passed():
    captured = {}

    class DummyLearner:
        def flag_for_review(self, **kwargs):
            captured.update(kwargs)
            return kwargs

    with (
        patch.dict(os.environ, {"ACTIVE_LEARNING_ENABLED": "true"}, clear=False),
        patch(
            "src.core.active_learning.get_active_learner",
            return_value=DummyLearner(),
        ),
    ):
        flag_classification_for_review(
            analysis_id="doc-46",
            cls_payload=_payload(),
        )

    assert captured["doc_id"] == "doc-46"


def test_flag_classification_for_review_defaults_priority_to_medium():
    captured = {}

    class DummyLearner:
        def flag_for_review(self, **kwargs):
            captured.update(kwargs)
            return kwargs

    with patch(
        "src.core.active_learning.get_active_learner",
        return_value=DummyLearner(),
    ):
        flag_classification_for_review(
            analysis_id="doc-47",
            cls_payload=_payload(review_priority=""),
            active_learning_enabled=True,
        )

    assert captured["feedback_priority"] == "medium"
