import os
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.core.similarity import _VECTOR_STORE, register_vector  # type: ignore
from src.main import app

client = TestClient(app)


def test_vector_migrate_dry_run():
    os.environ["FEATURE_VERSION"] = "v1"
    register_vector("migrate_a", [0.1] * 7)
    payload = {"ids": ["migrate_a"], "to_version": "v2", "dry_run": True}
    resp = client.post("/api/v1/vectors/migrate", json=payload, headers={"x-api-key": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"][0]["status"] == "dry_run"
    # original vector unchanged
    assert len(_VECTOR_STORE["migrate_a"]) == 7


def test_vector_migrate_execute():
    os.environ["FEATURE_VERSION"] = "v1"
    register_vector("migrate_b", [0.2] * 7)
    payload = {"ids": ["migrate_b"], "to_version": "v2", "dry_run": False}
    resp = client.post("/api/v1/vectors/migrate", json=payload, headers={"x-api-key": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"][0]["status"] == "migrated"
    # Expect v2 geometric extension (dimension > original)
    assert len(_VECTOR_STORE["migrate_b"]) > 7


def test_vector_migrate_execute_qdrant():
    class DummyVector:
        def __init__(self, vector, metadata):
            self.vector = vector
            self.metadata = metadata

    class DummyQdrantStore:
        def __init__(self):
            self.saved = {}

        async def get_vector(self, vector_id):
            if vector_id != "qdrant-migrate":
                return None
            return DummyVector([0.2] * 7, {"feature_version": "v1", "material": "steel"})

        async def register_vector(self, vector_id, vector, metadata=None):
            self.saved[vector_id] = {"vector": vector, "metadata": metadata or {}}
            return True

    store = DummyQdrantStore()
    with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "qdrant"}), patch(
        "src.api.v1.vectors._get_qdrant_store_or_none",
        return_value=store,
    ):
        resp = client.post(
            "/api/v1/vectors/migrate",
            json={"ids": ["qdrant-migrate"], "to_version": "v2", "dry_run": False},
            headers={"x-api-key": "test"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["items"][0]["status"] == "migrated"
    assert "qdrant-migrate" in store.saved
    assert store.saved["qdrant-migrate"]["metadata"]["feature_version"] == "v2"
    assert len(store.saved["qdrant-migrate"]["vector"]) > 7


def test_vector_migrate_dry_run_qdrant():
    class DummyVector:
        def __init__(self, vector, metadata):
            self.vector = vector
            self.metadata = metadata

    class DummyQdrantStore:
        def __init__(self):
            self.registered = False

        async def get_vector(self, vector_id):
            if vector_id != "qdrant-migrate-dry":
                return None
            return DummyVector([0.2] * 7, {"feature_version": "v1"})

        async def register_vector(self, vector_id, vector, metadata=None):
            self.registered = True
            return True

    store = DummyQdrantStore()
    with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "qdrant"}), patch(
        "src.api.v1.vectors._get_qdrant_store_or_none",
        return_value=store,
    ):
        resp = client.post(
            "/api/v1/vectors/migrate",
            json={"ids": ["qdrant-migrate-dry"], "to_version": "v2", "dry_run": True},
            headers={"x-api-key": "test"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["items"][0]["status"] == "dry_run"
    assert store.registered is False
