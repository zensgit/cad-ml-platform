import json

from fastapi.testclient import TestClient

from src.main import app


def test_drift_baseline_timestamp_persistence(monkeypatch):
    client = TestClient(app)
    # Trigger drift status to potentially establish baseline (threshold may be high; simulate state)
    # Directly patch internal state for test determinism
    from src.api.v1 import analyze as analyze_module

    mats = ["steel"] * 120
    preds = ["part"] * 120
    analyze_module._DRIFT_STATE["materials"] = mats
    analyze_module._DRIFT_STATE["predictions"] = preds
    # First call creates the baseline internally
    resp = client.get("/api/v1/analyze/drift", headers={"X-API-Key": "test"})
    assert resp.status_code == 200
    # Second call returns the baseline that was just created
    resp2 = client.get("/api/v1/analyze/drift", headers={"X-API-Key": "test"})
    assert resp2.status_code == 200
    data = resp2.json()
    # Baseline should now exist
    assert data["material_baseline"] is not None
    # Check baseline status endpoint
    status_resp = client.get("/api/v1/analyze/drift/baseline/status", headers={"X-API-Key": "test"})
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["status"] in ("ok", "stale")
    # Age fields should be present
    assert status_data.get("material_age") is not None
