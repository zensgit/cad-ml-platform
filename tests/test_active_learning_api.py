from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.core.active_learning import get_active_learner, reset_active_learner, SampleStatus
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
