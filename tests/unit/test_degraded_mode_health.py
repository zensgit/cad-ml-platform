import time

from fastapi.testclient import TestClient

from src.main import app


def test_faiss_health_has_next_eta_and_manual_flag():
    client = TestClient(app)
    # Call health; fields should exist even if None
    resp = client.get("/api/v1/health/faiss/health", headers={"X-API-Key": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert "next_recovery_eta" in data
    assert "manual_recovery_in_progress" in data


def test_manual_recover_toggles_flag_and_health_reports():
    client = TestClient(app)
    # Trigger manual recover; status may be success or skipped depending on backend
    r = client.post("/api/v1/faiss/recover", headers={"X-API-Key": "test"})
    assert r.status_code == 200
    assert r.json().get("status") in {"success", "skipped_or_failed"}
    # Health should report manual flag false after handler clears it
    h = client.get("/api/v1/health/faiss/health", headers={"X-API-Key": "test"})
    assert h.status_code == 200
    assert h.json().get("manual_recovery_in_progress") is False
