import csv
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.core.classification.coarse_labels import normalize_coarse_label
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
    assert payload[0]["predicted_fine_type"] == "bolt"
    assert payload[0]["predicted_coarse_type"] == normalize_coarse_label("bolt")
    assert payload[0]["sample_type"] == "low_confidence"
    assert payload[0]["feedback_priority"] == "medium"


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
        json={
            "sample_id": sample.id,
            "true_type": "人孔",
            "true_fine_type": "人孔",
            "true_coarse_type": "开孔件",
            "reviewer_id": "user-1",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["is_correction"] is True
    updated = learner.get_sample(sample.id)
    assert updated is not None
    assert updated.status == SampleStatus.LABELED
    assert updated.true_type == "人孔"
    assert updated.true_fine_type == "人孔"
    assert updated.true_coarse_type == "开孔件"
    assert updated.true_is_coarse_label is False
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
    assert body["remaining_samples"] == 0
    assert body["retrain_recommendation"] == "threshold_met"
    assert body["stats"]["total"] == 1
    assert body["sample_type_stats"]["low_confidence"] == 1
    assert body["feedback_priority_stats"]["medium"] == 1
    assert body["predicted_fine_stats"]["bolt"] == 1
    assert body["predicted_coarse_stats"][normalize_coarse_label("bolt")] == 1
    assert body["labeled_true_fine_stats"]["bolt"] == 1
    assert body["labeled_true_coarse_stats"][normalize_coarse_label("bolt")] == 1
    assert body["correction_count"] == 0


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


def test_active_learning_review_queue_orders_by_priority_then_confidence(client):
    learner = get_active_learner()
    critical = learner.flag_for_review(
        doc_id="doc-critical",
        predicted_type="法兰",
        confidence=0.82,
        alternatives=[],
        score_breakdown={"review_priority": "critical"},
        uncertainty_reason="knowledge_conflict",
    )
    high = learner.flag_for_review(
        doc_id="doc-high",
        predicted_type="人孔",
        confidence=0.41,
        alternatives=[],
        score_breakdown={},
        uncertainty_reason="hybrid_rejected:below_min_confidence",
    )
    medium = learner.flag_for_review(
        doc_id="doc-medium",
        predicted_type="bolt",
        confidence=0.22,
        alternatives=[],
        score_breakdown={},
        uncertainty_reason="low_confidence",
    )

    resp = client.get("/api/v1/active-learning/review-queue")
    assert resp.status_code == 200

    body = resp.json()
    assert body["total"] == 3
    assert body["returned"] == 3
    assert body["sort_by"] == "priority"
    assert [item["id"] for item in body["items"]] == [critical.id, high.id, medium.id]
    assert body["summary"]["by_feedback_priority"]["critical"] == 1
    assert body["summary"]["by_feedback_priority"]["high"] == 1
    assert body["summary"]["by_feedback_priority"]["medium"] == 1


def test_active_learning_review_queue_supports_filters_and_pagination(client):
    learner = get_active_learner()
    first = learner.flag_for_review(
        doc_id="doc-queue-1",
        predicted_type="法兰",
        confidence=0.61,
        alternatives=[],
        score_breakdown={},
        uncertainty_reason="low_confidence",
    )
    learner.flag_for_review(
        doc_id="doc-queue-2",
        predicted_type="人孔",
        confidence=0.31,
        alternatives=[],
        score_breakdown={},
        uncertainty_reason="hybrid_rejected:below_min_confidence",
    )
    learner.submit_feedback(first.id, "法兰")

    filtered = client.get(
        "/api/v1/active-learning/review-queue",
        params={"status": "pending", "feedback_priority": "high"},
    )
    assert filtered.status_code == 200
    filtered_body = filtered.json()
    assert filtered_body["total"] == 1
    assert filtered_body["items"][0]["doc_id"] == "doc-queue-2"
    assert filtered_body["summary"]["by_sample_type"]["hybrid_rejection"] == 1

    paged = client.get(
        "/api/v1/active-learning/review-queue",
        params={"status": "all", "limit": 1, "offset": 1, "sort_by": "created_at"},
    )
    assert paged.status_code == 200
    paged_body = paged.json()
    assert paged_body["total"] == 2
    assert paged_body["returned"] == 1
    assert paged_body["offset"] == 1
    assert paged_body["has_more"] is False


def test_active_learning_review_queue_summary_includes_decision_source_and_reasons(client):
    learner = get_active_learner()
    learner.flag_for_review(
        doc_id="doc-summary-1",
        predicted_type="法兰",
        confidence=0.55,
        alternatives=[],
        score_breakdown={
            "final_decision_source": "hybrid",
            "review_reasons": ["missing_critical_fields", "low_confidence"],
        },
        uncertainty_reason="low_confidence",
    )

    resp = client.get("/api/v1/active-learning/review-queue")
    assert resp.status_code == 200
    summary = resp.json()["summary"]
    assert summary["by_decision_source"]["hybrid"] == 1
    assert summary["by_uncertainty_reason"]["low_confidence"] == 1
    assert summary["by_review_reason"]["missing_critical_fields"] == 1
    assert summary["by_review_reason"]["low_confidence"] == 1


def test_active_learning_review_queue_export_csv(client):
    learner = get_active_learner()
    learner.flag_for_review(
        doc_id="doc-export-1",
        predicted_type="法兰",
        confidence=0.55,
        alternatives=[],
        score_breakdown={
            "final_decision_source": "hybrid",
            "review_reasons": ["missing_critical_fields"],
        },
        uncertainty_reason="low_confidence",
    )

    resp = client.get("/api/v1/active-learning/review-queue/export")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["count"] == 1
    assert body["format"] == "csv"
    assert body["summary"]["by_decision_source"]["hybrid"] == 1

    export_path = Path(body["file"])
    assert export_path.exists()
    with export_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["doc_id"] == "doc-export-1"
    assert json.loads(rows[0]["review_reasons"]) == ["missing_critical_fields"]


def test_active_learning_review_queue_export_jsonl_empty_returns_error(client):
    resp = client.get(
        "/api/v1/active-learning/review-queue/export",
        params={"format": "jsonl"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "error"
    assert "No review queue samples to export" in body["message"]


def test_active_learning_review_queue_stats_endpoint(client):
    learner = get_active_learner()
    learner.flag_for_review(
        doc_id="doc-stats-1",
        predicted_type="bolt",
        confidence=0.2,
        alternatives=[],
        score_breakdown={"decision_source": "graph2d"},
        uncertainty_reason="low_confidence",
    )
    learner.flag_for_review(
        doc_id="doc-stats-2",
        predicted_type="人孔",
        confidence=0.4,
        alternatives=[],
        score_breakdown={"decision_source": "hybrid", "review_reasons": ["branch_conflict"]},
        uncertainty_reason="branch_conflict",
    )

    resp = client.get("/api/v1/active-learning/review-queue/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert body["by_decision_source"]["graph2d"] == 1
    assert body["by_decision_source"]["hybrid"] == 1
    assert body["by_review_reason"]["branch_conflict"] == 1
