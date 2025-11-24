from fastapi.testclient import TestClient
from src.main import app


def test_drift_reset_endpoint():
    client = TestClient(app)
    # Call drift status first (may establish pending state)
    client.get("/api/v1/analyze/drift", headers={"api-key": "test"})
    r = client.post("/api/v1/analyze/drift/reset", headers={"api-key": "test"})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "reset_material" in data and "reset_predictions" in data

