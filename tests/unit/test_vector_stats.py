from fastapi.testclient import TestClient
from src.main import app
from src.core.similarity import register_vector
import uuid

client = TestClient(app)


def test_vector_stats_counts():
    # register some vectors
    for m in ["steel", "aluminum", "steel"]:
        vid = str(uuid.uuid4())
        register_vector(vid, [0.1, 0.2, 0.3], meta={"material": m, "complexity": "low", "format": "dxf"})
    resp = client.get("/api/v1/vectors_stats/stats", headers={"X-API-Key": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 3
    assert "steel" in data["by_material"]
    assert data["by_material"]["steel"] >= 2
