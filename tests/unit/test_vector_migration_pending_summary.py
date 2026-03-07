from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_vector_migration_pending_summary_memory():
    vectors = {
        "vec1": [1.0] * 24,
        "vec2": [2.0] * 22,
        "vec3": [3.0] * 12,
        "vec4": [4.0] * 12,
    }
    meta = {
        "vec1": {"feature_version": "v4"},
        "vec2": {"feature_version": "v3"},
        "vec3": {"feature_version": "v2"},
        "vec4": {"feature_version": "v2"},
    }
    with patch("src.core.similarity._VECTOR_STORE", vectors), patch(
        "src.core.similarity._VECTOR_META", meta
    ):
        response = client.get(
            "/api/v1/vectors/migrate/pending/summary",
            headers={"x-api-key": "test"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["total_pending"] == 3
    assert data["pending_ratio"] == 0.75
    assert data["observed_by_from_version"] == {"v3": 1, "v2": 2}


def test_vector_migration_pending_summary_qdrant_complete():
    class DummyPoint:
        def __init__(self, point_id, metadata):
            self.id = point_id
            self.metadata = metadata

    class DummyQdrantStore:
        async def count(self):
            return 4

        async def list_vectors(self, offset=0, limit=50, with_vectors=False):
            items = [
                DummyPoint("vec1", {"feature_version": "v4"}),
                DummyPoint("vec2", {"feature_version": "v3"}),
                DummyPoint("vec3", {"feature_version": "v2"}),
                DummyPoint("vec4", {"feature_version": "v2"}),
            ]
            return items[offset : offset + limit], 4

    with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "qdrant"}), patch(
        "src.api.v1.vectors._get_qdrant_store_or_none",
        return_value=DummyQdrantStore(),
    ):
        response = client.get(
            "/api/v1/vectors/migrate/pending/summary",
            headers={"x-api-key": "test"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["backend"] == "qdrant"
    assert data["total_pending"] == 3
    assert data["pending_ratio"] == 0.75
    assert data["observed_by_from_version"] == {"v3": 1, "v2": 2}


def test_vector_migration_pending_summary_qdrant_partial_hides_exact_total():
    class DummyPoint:
        def __init__(self, point_id, metadata):
            self.id = point_id
            self.metadata = metadata

    class DummyQdrantStore:
        async def count(self):
            return 4

        async def list_vectors(self, offset=0, limit=50, with_vectors=False):
            items = [
                DummyPoint("vec1", {"feature_version": "v4"}),
                DummyPoint("vec2", {"feature_version": "v3"}),
                DummyPoint("vec3", {"feature_version": "v2"}),
                DummyPoint("vec4", {"feature_version": "v2"}),
            ]
            return items[offset : offset + limit], 4

    with patch.dict(
        "os.environ",
        {"VECTOR_STORE_BACKEND": "qdrant", "VECTOR_MIGRATION_SCAN_LIMIT": "2"},
    ), patch(
        "src.api.v1.vectors._get_qdrant_store_or_none",
        return_value=DummyQdrantStore(),
    ):
        response = client.get(
            "/api/v1/vectors/migrate/pending/summary",
            headers={"x-api-key": "test"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["distribution_complete"] is False
    assert data["total_pending"] is None
    assert data["pending_ratio"] is None
    assert data["observed_by_from_version"] == {"v3": 1}
