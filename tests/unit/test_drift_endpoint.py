from fastapi.testclient import TestClient
from src.main import app


def test_drift_endpoint_initial():
    client = TestClient(app)
    r = client.get("/api/v1/analyze/drift", headers={"api-key": "test"})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("baseline_pending", "ok")
    assert "material_current" in data
    # initial scores may be null
    assert "baseline_min_count" in data

