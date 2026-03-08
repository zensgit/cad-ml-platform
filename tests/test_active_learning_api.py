import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.core.active_learning import (
    get_active_learner,
    reset_active_learner,
    SampleStatus,
)
from src.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ACTIVE_LEARNING_DATA_DIR", str(tmp_path / "active_learning"))
    monkeypatch.setenv("ACTIVE_LEARNING_RETRAIN_THRESHOLD", "1")
    reset_active_learner()
    yield TestClient(app)
    reset_active_learner()


def test_active_learning_pending_limit(client):
    learner = get_active_learner()
    sample_one = learner.flag_for_review(
        doc_id="doc-1",
        predicted_type="bolt",
        confidence=0.4,
        alternatives=[],
        score_breakdown={},
        uncertainty_reason="low_confidence",
    )
    learner.flag_for_review(
        doc_id="doc-2",
        predicted_type="screw",
        confidence=0.5,
        alternatives=[],
        score_breakdown={},
        uncertainty_reason="low_confidence",
    )

    resp = client.get("/api/v1/active-learning/pending?limit=1")
    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload) == 1
    assert payload[0]["id"] == sample_one.id
    assert payload[0]["status"] == SampleStatus.PENDING.value
    assert payload[0]["evidence_count"] == 0
    assert payload[0]["evidence_sources"] == []
    assert payload[0]["evidence_summary"] is None
    assert payload[0]["evidence"] == []


def test_active_learning_pending_includes_structured_evidence(client):
    learner = get_active_learner()
    learner.flag_for_review(
        doc_id="doc-evidence-1",
        predicted_type="manhole",
        confidence=0.41,
        alternatives=[{"type": "shell", "confidence": 0.31}],
        score_breakdown={
            "decision_path": ["fusion_scored", "final_below_reject_min_conf"],
            "source_contributions": {"filename": 0.61, "titleblock": 0.22},
            "fusion_metadata": {
                "strategy": "weighted_average",
                "agreement_score": 0.5,
                "num_sources": 2,
            },
            "hybrid_explanation": {"summary": "综合 文件名, 标题栏 多源信息"},
            "hybrid_rejection": {
                "reason": "below_min_confidence",
                "raw_source": "filename",
                "raw_confidence": 0.61,
            },
        },
        uncertainty_reason="hybrid_rejected:below_min_confidence+low_confidence",
    )

    resp = client.get("/api/v1/active-learning/pending?limit=1")
    assert resp.status_code == 200
    payload = resp.json()[0]

    assert payload["evidence_count"] == 6
    assert payload["evidence_sources"] == ["filename", "titleblock"]
    assert payload["evidence_summary"].startswith("综合 文件名, 标题栏 多源信息")
    assert (
        "Rejection: below_min_confidence via filename (0.610)"
        in payload["evidence_summary"]
    )
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
    assert any(item["kind"] == "hybrid_explanation" for item in payload["evidence"])
    assert any(item["kind"] == "decision_path" for item in payload["evidence"])


def test_active_learning_feedback_missing_sample_returns_404(client):
    resp = client.post(
        "/api/v1/active-learning/feedback",
        json={"sample_id": "missing", "true_type": "bolt"},
    )
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["code"] == "DATA_NOT_FOUND"
    assert detail["stage"] == "active_learning_feedback"
    assert detail["context"]["sample_id"] == "missing"


def test_active_learning_feedback_updates_sample(client):
    learner = get_active_learner()
    sample = learner.flag_for_review(
        doc_id="doc-3",
        predicted_type="bolt",
        confidence=0.4,
        alternatives=[],
        score_breakdown={},
        uncertainty_reason="low_confidence",
    )
    resp = client.post(
        "/api/v1/active-learning/feedback",
        json={"sample_id": sample.id, "true_type": "screw", "reviewer_id": "user-1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["is_correction"] is True
    updated = learner.get_sample(sample.id)
    assert updated is not None
    assert updated.status == SampleStatus.LABELED
    assert updated.true_type == "screw"
    assert updated.reviewer_id == "user-1"


def test_active_learning_stats_retrain_ready(client):
    learner = get_active_learner()
    sample = learner.flag_for_review(
        doc_id="doc-4",
        predicted_type="bolt",
        confidence=0.4,
        alternatives=[],
        score_breakdown={},
        uncertainty_reason="low_confidence",
    )
    learner.submit_feedback(sample.id, "bolt")
    resp = client.get("/api/v1/active-learning/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["retrain_ready"] is True
    assert body["labeled_samples"] == 1
    assert body["threshold"] == 1
    assert body["stats"]["total"] == 1


def test_active_learning_export_no_samples(client):
    resp = client.post("/api/v1/active-learning/export", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "error"
    assert "No samples to export" in body["message"]


def test_active_learning_export_labeled(client, tmp_path, monkeypatch):
    monkeypatch.setenv("ACTIVE_LEARNING_DATA_DIR", str(tmp_path / "active_learning"))
    reset_active_learner()
    learner = get_active_learner()
    sample = learner.flag_for_review(
        doc_id="doc-5",
        predicted_type="bolt",
        confidence=0.4,
        alternatives=[],
        score_breakdown={},
        uncertainty_reason="low_confidence",
    )
    learner.submit_feedback(sample.id, "bolt")
    resp = client.post("/api/v1/active-learning/export", json={"format": "jsonl"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["count"] == 1
    export_path = Path(body["file"])
    assert export_path.exists()


def test_active_learning_export_includes_structured_evidence(client):
    learner = get_active_learner()
    sample = learner.flag_for_review(
        doc_id="doc-export-evidence",
        predicted_type="manhole",
        confidence=0.41,
        alternatives=[{"type": "shell", "confidence": 0.31}],
        score_breakdown={
            "decision_path": ["fusion_scored", "final_below_reject_min_conf"],
            "source_contributions": {"filename": 0.61, "titleblock": 0.22},
            "hybrid_explanation": {"summary": "综合 文件名, 标题栏 多源信息"},
        },
        uncertainty_reason="hybrid_rejected:below_min_confidence+low_confidence",
    )
    learner.submit_feedback(sample.id, "manhole")

    resp = client.post("/api/v1/active-learning/export", json={"format": "jsonl"})
    assert resp.status_code == 200
    body = resp.json()
    export_path = Path(body["file"])
    with open(export_path, "r", encoding="utf-8") as handle:
        payload = json.loads(handle.readline())

    assert payload["evidence_count"] == 4
    assert payload["evidence_sources"] == ["filename", "titleblock"]
    assert payload["evidence_summary"].startswith("综合 文件名, 标题栏 多源信息")
