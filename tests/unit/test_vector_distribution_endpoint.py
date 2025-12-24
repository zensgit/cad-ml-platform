from fastapi.testclient import TestClient

from src.core.similarity import register_vector
from src.main import app

client = TestClient(app)


def test_vector_distribution_endpoint():
    register_vector(
        "dist_a", [0.1] * 7, meta={"material": "steel", "complexity": "low", "format": "dxf"}
    )
    register_vector(
        "dist_b", [0.2] * 7, meta={"material": "steel", "complexity": "medium", "format": "stl"}
    )
    register_vector(
        "dist_c", [0.3] * 7, meta={"material": "aluminum", "complexity": "high", "format": "step"}
    )
    resp = client.get("/api/v1/vectors_stats/distribution", headers={"x-api-key": "test"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["total"] >= 3
    assert "steel" in data["by_material"] and "aluminum" in data["by_material"]
    assert 0.0 <= data["dominant_ratio"] <= 1.0
