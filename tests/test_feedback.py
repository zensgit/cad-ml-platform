import json

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def _payload():
    return {
        "analysis_id": "analysis-123",
        "corrected_part_type": "bracket",
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


def test_submit_feedback_write_failure_returns_500(tmp_path, monkeypatch):
    log_dir = tmp_path / "feedback_dir"
    log_dir.mkdir()
    monkeypatch.setenv("FEEDBACK_LOG_PATH", str(log_dir))

    resp = client.post("/api/v1/feedback/", json=_payload())
    assert resp.status_code == 500
    assert "Failed to save feedback" in resp.json()["detail"]
