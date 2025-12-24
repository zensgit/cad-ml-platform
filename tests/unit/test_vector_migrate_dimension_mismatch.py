"""Tests: vector migration dimension mismatch error handling."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.core import similarity
from src.main import app

client = TestClient(app)


def test_vector_migrate_dimension_mismatch_error() -> None:
    original_vectors = similarity._VECTOR_STORE.copy()
    original_meta = similarity._VECTOR_META.copy()
    similarity._VECTOR_STORE.clear()
    similarity._VECTOR_META.clear()
    try:
        # Seed mismatch: meta says v3 but vector length matches v1 (7)
        similarity._VECTOR_STORE["bad_vec"] = [0.1] * 7
        similarity._VECTOR_META["bad_vec"] = {"feature_version": "v3"}

        response = client.post(
            "/api/v1/vectors/migrate",
            json={"ids": ["bad_vec"], "to_version": "v4"},
            headers={"X-API-Key": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["status"] == "error"
        assert "expected" in data["items"][0]["error"]
    finally:
        similarity._VECTOR_STORE.clear()
        similarity._VECTOR_META.clear()
        similarity._VECTOR_STORE.update(original_vectors)
        similarity._VECTOR_META.update(original_meta)
