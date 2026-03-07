from __future__ import annotations

import json
import os
import shutil

import pytest

from src.core.active_learning import ActiveLearner, reset_active_learner


@pytest.fixture()
def learner(tmp_path, monkeypatch):
    reset_active_learner()
    data_dir = tmp_path / "active_learning"
    monkeypatch.setenv("ACTIVE_LEARNING_DATA_DIR", str(data_dir))
    monkeypatch.setenv("ACTIVE_LEARNING_STORE", "file")
    instance = ActiveLearner()
    try:
        yield instance
    finally:
        reset_active_learner()
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)


def test_export_training_data_keeps_score_breakdown_and_uncertainty_reason(
    learner: ActiveLearner,
) -> None:
    sample = learner.flag_for_review(
        doc_id="doc-context-1",
        predicted_type="人孔",
        confidence=0.41,
        alternatives=[{"type": "壳体类", "confidence": 0.31}],
        score_breakdown={
            "decision_path": ["fusion_scored", "fusion_engine_weighted_average"],
            "source_contributions": {"filename": 0.61, "titleblock": 0.22},
            "history_prediction": {
                "label": "人孔",
                "confidence": 0.58,
                "shadow_only": True,
                "used_for_fusion": False,
            },
            "shadow_predictions": {
                "history_sequence": {
                    "label": "人孔",
                    "confidence": 0.58,
                    "status": "ok",
                }
            },
            "hybrid_explanation": {"summary": "综合 文件名, 标题栏 多源信息"},
        },
        uncertainty_reason="hybrid_rejected:below_min_confidence+low_confidence",
    )
    learner.submit_feedback(sample.id, "人孔")

    exported = learner.export_training_data(format="jsonl")

    assert exported["status"] == "ok"
    with open(exported["file"], "r", encoding="utf-8") as handle:
        payload = json.loads(handle.readline())

    assert payload["doc_id"] == "doc-context-1"
    assert payload["analysis_id"] == "doc-context-1"
    assert payload["predicted_fine_type"] == "人孔"
    assert payload["predicted_coarse_type"] == "开孔件"
    assert payload["predicted_is_coarse_label"] is False
    assert payload["true_type"] == "人孔"
    assert payload["true_fine_type"] == "人孔"
    assert payload["true_coarse_type"] == "开孔件"
    assert payload["true_is_coarse_label"] is False
    assert payload["correct_label"] == "人孔"
    assert payload["correct_fine_label"] == "人孔"
    assert payload["correct_coarse_label"] == "开孔件"
    assert payload["original_label"] == "人孔"
    assert payload["original_fine_label"] == "人孔"
    assert payload["original_coarse_label"] == "开孔件"
    assert payload["sample_type"] == "hybrid_rejection"
    assert payload["feedback_priority"] == "high"
    assert payload["uncertainty_reason"] == (
        "hybrid_rejected:below_min_confidence+low_confidence"
    )
    assert payload["score_breakdown"]["decision_path"] == [
        "fusion_scored",
        "fusion_engine_weighted_average",
    ]
    assert payload["score_breakdown"]["source_contributions"]["filename"] == 0.61
    assert payload["score_breakdown"]["history_prediction"]["shadow_only"] is True
    assert payload["score_breakdown"]["shadow_predictions"]["history_sequence"] == {
        "label": "人孔",
        "confidence": 0.58,
        "status": "ok",
    }


def test_export_training_data_marks_low_confidence_feedback_priority(
    learner: ActiveLearner,
) -> None:
    sample = learner.flag_for_review(
        doc_id="doc-context-2",
        predicted_type="壳体类",
        confidence=0.52,
        alternatives=[],
        score_breakdown={},
        uncertainty_reason="low_confidence",
    )
    learner.submit_feedback(sample.id, "壳体类")

    exported = learner.export_training_data(format="jsonl")

    assert exported["status"] == "ok"
    with open(exported["file"], "r", encoding="utf-8") as handle:
        payload = json.loads(handle.readline())

    assert payload["sample_type"] == "low_confidence"
    assert payload["feedback_priority"] == "medium"
    assert payload["correct_label"] == "壳体类"
    assert payload["original_label"] == "壳体类"
    assert payload["predicted_fine_type"] == "壳体类"
    assert payload["predicted_coarse_type"] == "壳体类"
    assert payload["predicted_is_coarse_label"] is True
    assert payload["true_fine_type"] == "壳体类"
    assert payload["true_coarse_type"] == "壳体类"
    assert payload["true_is_coarse_label"] is True


def test_export_training_data_uses_review_governance_when_present(
    learner: ActiveLearner,
) -> None:
    sample = learner.flag_for_review(
        doc_id="doc-context-3",
        predicted_type="法兰",
        confidence=0.83,
        alternatives=[],
        score_breakdown={
            "review_priority": "critical",
            "review_has_knowledge_conflict": True,
            "review_has_branch_conflict": False,
            "review_has_hybrid_rejection": False,
            "review_is_low_confidence": False,
        },
        uncertainty_reason="knowledge_conflict",
    )
    learner.submit_feedback(sample.id, "法兰")

    exported = learner.export_training_data(format="jsonl")

    assert exported["status"] == "ok"
    with open(exported["file"], "r", encoding="utf-8") as handle:
        payload = json.loads(handle.readline())

    assert payload["sample_type"] == "knowledge_conflict"
    assert payload["feedback_priority"] == "critical"
    assert payload["correct_label"] == "法兰"
    assert payload["original_label"] == "法兰"
    assert payload["predicted_fine_type"] == "法兰"
    assert payload["predicted_coarse_type"] == "法兰"
    assert payload["predicted_is_coarse_label"] is True
    assert payload["true_fine_type"] == "法兰"
    assert payload["true_coarse_type"] == "法兰"
    assert payload["true_is_coarse_label"] is True
