from fastapi.testclient import TestClient

from src.main import app


def test_drift_baseline_startup_load(monkeypatch):
    # Without Redis this will simply not preload; ensure no crash and state defaults
    client = TestClient(app)
    resp = client.get("/api/v1/analyze/drift", headers={"X-API-Key": "test"})
    assert resp.status_code == 200
    data = resp.json()
    # Baselines may be empty initially; assert keys exist
    assert "material_current" in data
    assert "prediction_current" in data
    assert "status" in data
