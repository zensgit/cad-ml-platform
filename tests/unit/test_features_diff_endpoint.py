from unittest.mock import patch

from fastapi.testclient import TestClient

from src.core.similarity import register_vector  # type: ignore
from src.main import app

client = TestClient(app)


def test_features_diff_basic():
    register_vector("diff_a", [0.1, 0.2, 0.3])
    register_vector("diff_b", [0.1, 0.25, 0.35])
    r = client.get("/api/v1/features/diff?id_a=diff_a&id_b=diff_b", headers={"x-api-key": "test"})
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"
    diffs = data.get("diffs")
    assert isinstance(diffs, list)
    # abs_diff checks (diffs are sorted by abs_diff descending)
    # The response field is "abs_diff", not "delta"
    abs_diffs = [d.get("abs_diff") for d in diffs]
    # Original vectors: [0.1, 0.2, 0.3] vs [0.1, 0.25, 0.35]
    # Differences: 0, 0.05, 0.05 - sorted descending so order may vary
    assert 0.05 in abs_diffs or 0.04999999999999999 in abs_diffs


def test_features_diff_uses_qdrant_when_enabled():
    class DummyQdrantResult:
        def __init__(self, vector_id, vector, metadata=None):
            self.id = vector_id
            self.vector = vector
            self.metadata = metadata or {}

    class DummyQdrantStore:
        async def get_vector(self, vector_id):
            payloads = {
                "qa": DummyQdrantResult("qa", [0.1, 0.2, 0.3], {"feature_version": "v1"}),
                "qb": DummyQdrantResult("qb", [0.1, 0.25, 0.35], {"feature_version": "v1"}),
            }
            return payloads.get(vector_id)

    with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "qdrant"}), patch(
        "src.api.v1.features._get_qdrant_store_or_none",
        return_value=DummyQdrantStore(),
    ):
        r = client.get("/api/v1/features/diff?id_a=qa&id_b=qb", headers={"x-api-key": "test"})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["dimension"] == 3
    assert data["diffs"]


def test_features_diff_qdrant_not_found():
    class DummyQdrantStore:
        async def get_vector(self, vector_id):
            return None

    with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "qdrant"}), patch(
        "src.api.v1.features._get_qdrant_store_or_none",
        return_value=DummyQdrantStore(),
    ):
        r = client.get(
            "/api/v1/features/diff?id_a=missing-a&id_b=missing-b",
            headers={"x-api-key": "test"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "not_found"


def test_features_diff_qdrant_dimension_mismatch():
    class DummyQdrantResult:
        def __init__(self, vector_id, vector):
            self.id = vector_id
            self.vector = vector
            self.metadata = {}

    class DummyQdrantStore:
        async def get_vector(self, vector_id):
            if vector_id == "qa":
                return DummyQdrantResult("qa", [0.1, 0.2])
            if vector_id == "qb":
                return DummyQdrantResult("qb", [0.1, 0.2, 0.3])
            return None

    with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "qdrant"}), patch(
        "src.api.v1.features._get_qdrant_store_or_none",
        return_value=DummyQdrantStore(),
    ):
        r = client.get("/api/v1/features/diff?id_a=qa&id_b=qb", headers={"x-api-key": "test"})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "dimension_mismatch"
