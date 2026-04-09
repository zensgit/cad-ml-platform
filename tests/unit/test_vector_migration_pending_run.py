from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_vector_migration_pending_run_memory_dry_run_succeeds():
    vectors = {
        "vec1": [1.0] * 24,
        "vec2": [2.0] * 22,
        "vec3": [3.0] * 12,
    }
    meta = {
        "vec1": {"feature_version": "v4"},
        "vec2": {"feature_version": "v3"},
        "vec3": {"feature_version": "v2"},
    }
    with patch("src.core.similarity._VECTOR_STORE", vectors), patch(
        "src.core.similarity._VECTOR_META", meta
    ):
        response = client.post(
            "/api/v1/vectors/migrate/pending/run",
            json={"limit": 10, "dry_run": True},
            headers={"x-api-key": "test"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert data["dry_run_total"] == 2
    assert {item["id"] for item in data["items"]} == {"vec2", "vec3"}


def test_vector_migration_pending_run_qdrant_partial_requires_override():
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
                DummyPoint("vec4", {"feature_version": "v1"}),
            ]
            return items[offset : offset + limit], 4

    with patch.dict(
        "os.environ",
        {"VECTOR_STORE_BACKEND": "qdrant", "VECTOR_MIGRATION_SCAN_LIMIT": "2"},
    ), patch(
        "src.api.v1.vectors._get_qdrant_store_or_none",
        return_value=DummyQdrantStore(),
    ):
        response = client.post(
            "/api/v1/vectors/migrate/pending/run",
            json={"limit": 10, "dry_run": True},
            headers={"x-api-key": "test"},
        )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "CONSTRAINT_VIOLATION"


def test_vector_migration_pending_run_qdrant_partial_allows_override():
    class DummyPoint:
        def __init__(self, point_id, metadata, vector=None):
            self.id = point_id
            self.metadata = metadata
            self.vector = vector or [1.0] * 22

    class DummyQdrantStore:
        async def count(self):
            return 4

        async def list_vectors(self, offset=0, limit=50, with_vectors=False):
            items = [
                DummyPoint("vec1", {"feature_version": "v4"}),
                DummyPoint("vec2", {"feature_version": "v3"}, [2.0] * 22),
                DummyPoint("vec3", {"feature_version": "v2"}, [3.0] * 12),
                DummyPoint("vec4", {"feature_version": "v1"}, [4.0] * 7),
            ]
            return items[offset : offset + limit], 4

        async def get_vector(self, vector_id):
            if vector_id == "vec2":
                return DummyPoint("vec2", {"feature_version": "v3"}, [2.0] * 22)
            return None

        async def register_vector(self, vector_id, vector, metadata=None):
            return None

    with patch.dict(
        "os.environ",
        {"VECTOR_STORE_BACKEND": "qdrant", "VECTOR_MIGRATION_SCAN_LIMIT": "2"},
    ), patch(
        "src.api.v1.vectors._get_qdrant_store_or_none",
        return_value=DummyQdrantStore(),
    ):
        response = client.post(
            "/api/v1/vectors/migrate/pending/run",
            json={"limit": 10, "dry_run": True, "allow_partial_scan": True},
            headers={"x-api-key": "test"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["dry_run_total"] == 1
    assert data["items"][0]["id"] == "vec2"


def test_vector_migration_pending_run_applies_from_version_filter():
    vectors = {
        "vec1": [1.0] * 24,
        "vec2": [2.0] * 22,
        "vec3": [3.0] * 12,
    }
    meta = {
        "vec1": {"feature_version": "v4"},
        "vec2": {"feature_version": "v3"},
        "vec3": {"feature_version": "v2"},
    }
    with patch("src.core.similarity._VECTOR_STORE", vectors), patch(
        "src.core.similarity._VECTOR_META", meta
    ):
        response = client.post(
            "/api/v1/vectors/migrate/pending/run",
            json={"limit": 10, "dry_run": True, "from_version_filter": "v2"},
            headers={"x-api-key": "test"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["id"] == "vec3"
