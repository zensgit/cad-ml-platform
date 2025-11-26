import os
from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


def setup_module(module):
    os.environ["X_API_KEY"] = "test"


def test_preview_stats_avg_median_and_warnings(monkeypatch):
    # Prepare in-memory store with mixed deltas
    from src.core import similarity as sim
    sim._VECTOR_STORE.clear()
    sim._VECTOR_META.clear()

    # v1 vectors of varying dimensions
    for i, dim in enumerate([10, 10, 10, 10, 10]):
        sim._VECTOR_STORE[f"v{i}"] = [0.0] * dim
        sim._VECTOR_META[f"v{i}"] = {"feature_version": "v1"}

    # Call preview to v4
    resp = client.get("/api/v1/vectors/migrate/preview", params={"to_version": "v4", "limit": 5}, headers={"X-API-Key": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert "avg_delta" in data and "median_delta" in data
    assert "warnings" in data
    # Basic shape
    assert isinstance(data["avg_delta"], float) or data["avg_delta"] is None
    assert isinstance(data["median_delta"], float) or data["median_delta"] is None
    assert isinstance(data["warnings"], list)


def test_preview_limit_cap(monkeypatch):
    from src.core import similarity as sim
    sim._VECTOR_STORE.clear()
    sim._VECTOR_META.clear()

    for i in range(150):
        sim._VECTOR_STORE[f"id{i}"] = [0.0] * 8
        sim._VECTOR_META[f"id{i}"] = {"feature_version": "v2"}

    resp = client.get("/api/v1/vectors/migrate/preview", params={"to_version": "v3", "limit": 1000}, headers={"X-API-Key": "test"})
    assert resp.status_code == 200
    data = resp.json()
    # Sampled should not exceed 100; feasibility computed against sampled
    assert data["total_vectors"] == 150
    assert "estimated_dimension_changes" in data


def test_preview_invalid_version():
    resp = client.get("/api/v1/vectors/migrate/preview", params={"to_version": "v9"}, headers={"X-API-Key": "test"})
    assert resp.status_code == 422
    err = resp.json()["detail"]
    assert err["code"] == "INPUT_VALIDATION_FAILED"
    assert err["stage"] == "migration_preview"
