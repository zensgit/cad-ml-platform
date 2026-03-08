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
    assert payload["uncertainty_reason"] == (
        "hybrid_rejected:below_min_confidence+low_confidence"
    )
    assert payload["score_breakdown"]["decision_path"] == [
        "fusion_scored",
        "fusion_engine_weighted_average",
    ]
    assert payload["score_breakdown"]["source_contributions"]["filename"] == 0.61
    assert payload["evidence_count"] == 4
    assert payload["evidence_sources"] == ["filename", "titleblock"]
    assert payload["evidence_summary"].startswith("综合 文件名, 标题栏 多源信息")
    assert payload["evidence"][0] == {
        "kind": "source_contribution",
        "source": "filename",
        "score": 0.61,
    }
    assert payload["evidence"][1] == {
        "kind": "source_contribution",
        "source": "titleblock",
        "score": 0.22,
    }
