import json

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def _payload():
    return {
        "analysis_id": "analysis-123",
        "corrected_part_type": "人孔",
        "original_part_type": "法兰",
        "original_decision_source": "hybrid",
        "review_outcome": "updated",
        "review_reasons": [" low_confidence ", "", "branch_conflict"],
        "corrected_process": "milling",
        "dfm_feedback": "OK",
        "rating": 4,
    }


def test_submit_feedback_success_writes_log(tmp_path, monkeypatch):
    log_path = tmp_path / "feedback.jsonl"
    monkeypatch.setenv("FEEDBACK_LOG_PATH", str(log_path))

    resp = client.post("/api/v1/feedback/", json=_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["feedback_id"]
    assert "Feedback received" in body["message"]

    stored = log_path.read_text().strip()
    entry = json.loads(stored)
    assert entry["analysis_id"] == "analysis-123"
    assert entry["corrected_part_type"] == "人孔"
    assert entry["corrected_fine_part_type"] == "人孔"
    assert entry["corrected_coarse_part_type"] == "开孔件"
    assert entry["corrected_is_coarse_label"] is False
    assert entry["original_part_type"] == "法兰"
    assert entry["original_fine_part_type"] == "法兰"
    assert entry["original_coarse_part_type"] == "法兰"
    assert entry["original_is_coarse_label"] is True
    assert entry["original_decision_source"] == "hybrid"
    assert entry["review_outcome"] == "updated"
    assert entry["review_reasons"] == ["low_confidence", "branch_conflict"]


def test_submit_feedback_write_failure_returns_500(tmp_path, monkeypatch):
    log_dir = tmp_path / "feedback_dir"
    log_dir.mkdir()
    monkeypatch.setenv("FEEDBACK_LOG_PATH", str(log_dir))

    resp = client.post("/api/v1/feedback/", json=_payload())
    assert resp.status_code == 500
    assert "Failed to save feedback" in resp.json()["detail"]


def test_submit_feedback_accepts_explicit_fine_and_coarse_labels(tmp_path, monkeypatch):
    log_path = tmp_path / "feedback.jsonl"
    monkeypatch.setenv("FEEDBACK_LOG_PATH", str(log_path))

    payload = _payload()
    payload.pop("corrected_part_type")
    payload["corrected_fine_part_type"] = "人孔"
    payload["corrected_coarse_part_type"] = "开孔件"
    payload["original_fine_part_type"] = "捕集口"
    payload["original_coarse_part_type"] = "开孔件"

    resp = client.post("/api/v1/feedback/", json=payload)
    assert resp.status_code == 200

    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert entry["corrected_part_type"] == "人孔"
    assert entry["corrected_fine_part_type"] == "人孔"
    assert entry["corrected_coarse_part_type"] == "开孔件"
    assert entry["original_part_type"] == "捕集口"
    assert entry["original_fine_part_type"] == "捕集口"
    assert entry["original_coarse_part_type"] == "开孔件"


def test_feedback_stats_returns_empty_summary_when_log_missing(tmp_path, monkeypatch):
    log_path = tmp_path / "missing_feedback.jsonl"
    monkeypatch.setenv("FEEDBACK_LOG_PATH", str(log_path))

    resp = client.get("/api/v1/feedback/stats")
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "success"
    assert body["total"] == 0
    assert body["correction_count"] == 0
    assert body["coarse_correction_count"] == 0
    assert body["by_review_reason"] == {}


def test_feedback_stats_aggregates_coarse_and_fine_corrections(tmp_path, monkeypatch):
    log_path = tmp_path / "feedback.jsonl"
    monkeypatch.setenv("FEEDBACK_LOG_PATH", str(log_path))

    updated_payload = _payload()
    client.post("/api/v1/feedback/", json=updated_payload)

    accepted_payload = _payload()
    accepted_payload["analysis_id"] = "analysis-456"
    accepted_payload["corrected_part_type"] = "法兰"
    accepted_payload["original_part_type"] = "法兰"
    accepted_payload["review_outcome"] = "accepted"
    accepted_payload["review_reasons"] = ["confirmed"]
    accepted_payload["rating"] = 5
    client.post("/api/v1/feedback/", json=accepted_payload)

    resp = client.get("/api/v1/feedback/stats")
    assert resp.status_code == 200

    body = resp.json()
    assert body["total"] == 2
    assert body["correction_count"] == 1
    assert body["coarse_correction_count"] == 1
    assert body["average_rating"] == 4.5
    assert body["by_review_outcome"] == {"updated": 1, "accepted": 1}
    assert body["by_review_reason"]["low_confidence"] == 1
    assert body["by_review_reason"]["branch_conflict"] == 1
    assert body["by_review_reason"]["confirmed"] == 1
    assert body["by_corrected_coarse_part_type"]["开孔件"] == 1
    assert body["by_corrected_coarse_part_type"]["法兰"] == 1
    assert body["by_original_coarse_part_type"]["法兰"] == 2
    assert body["by_original_decision_source"]["hybrid"] == 2
