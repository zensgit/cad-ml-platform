from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_health_extended_basic():
    r = client.get("/health/extended")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "healthy"
    assert "vector_store" in data
    assert "feature_version_env" in data
    assert "faiss" in data
    faiss = data["faiss"]
    assert "enabled" in faiss
    # if enabled, imported flag should be boolean
    assert isinstance(faiss.get("enabled"), bool)

