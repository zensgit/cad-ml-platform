from fastapi.testclient import TestClient

from src.main import app


def test_faiss_health_endpoint_basic(monkeypatch):
    client = TestClient(app)
    # Monkeypatch API key dependency if required
    # Assuming get_api_key validates header 'X-API-Key'
    resp = client.get("/api/v1/faiss/health", headers={"X-API-Key": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert "available" in data
    assert "status" in data
    # If faiss unavailable, status should reflect
    if not data["available"]:
        assert data["status"] == "unavailable"
