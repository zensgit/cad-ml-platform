from fastapi.testclient import TestClient
from unittest.mock import patch

from src.main import app

client = TestClient(app)


def test_similarity_reference_not_found():
    r = client.post(
        "/api/v1/analyze/similarity",
        json={"reference_id": "nope", "target_id": "also_no"},
        headers={"X-API-Key": "test"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "reference_not_found"
    assert data["error"]["code"] == "DATA_NOT_FOUND"


def test_similarity_dimension_mismatch():
    # Force a mismatch by mutating one stored vector after registration.
    r1 = client.post(
        "/api/v1/analyze",
        files={"file": ("a.dxf", b"1", "application/octet-stream")},
        data={"options": '{"extract_features": true, "classify_parts": false}'},
        headers={"X-API-Key": "test"},
    )
    r2 = client.post(
        "/api/v1/analyze",
        files={"file": ("b.dxf", b"2", "application/octet-stream")},
        data={"options": '{"extract_features": true, "classify_parts": false}'},
        headers={"X-API-Key": "test"},
    )
    assert r1.status_code == 200 and r2.status_code == 200
    id1 = r1.json()["id"]
    id2 = r2.json()["id"]
    # Directly mutate global store to force mismatch
    from src.core import similarity

    similarity._VECTOR_STORE[id2] = similarity._VECTOR_STORE[id2] + [999.0]  # extend vector
    rq = client.post(
        "/api/v1/analyze/similarity",
        json={"reference_id": id1, "target_id": id2},
        headers={"X-API-Key": "test"},
    )
    assert rq.status_code == 200
    data = rq.json()
    assert data["status"] == "dimension_mismatch"
    assert data["error"]["code"] == "VALIDATION_FAILED"


def test_similarity_reference_not_found_qdrant():
    class DummyQdrantStore:
        async def get_vector(self, vector_id):
            return None

    with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "qdrant"}), patch(
        "src.api.v1.analyze._get_qdrant_store_or_none",
        return_value=DummyQdrantStore(),
    ):
        r = client.post(
            "/api/v1/analyze/similarity",
            json={"reference_id": "nope", "target_id": "also_no"},
            headers={"X-API-Key": "test"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "reference_not_found"
    assert data["error"]["code"] == "DATA_NOT_FOUND"


def test_similarity_dimension_mismatch_qdrant():
    class DummyQdrantResult:
        def __init__(self, vector_id, vector):
            self.id = vector_id
            self.vector = vector
            self.metadata = {}

    class DummyQdrantStore:
        async def get_vector(self, vector_id):
            if vector_id == "ref":
                return DummyQdrantResult("ref", [0.1, 0.2])
            if vector_id == "tgt":
                return DummyQdrantResult("tgt", [0.1, 0.2, 0.3])
            return None

    with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "qdrant"}), patch(
        "src.api.v1.analyze._get_qdrant_store_or_none",
        return_value=DummyQdrantStore(),
    ):
        rq = client.post(
            "/api/v1/analyze/similarity",
            json={"reference_id": "ref", "target_id": "tgt"},
            headers={"X-API-Key": "test"},
        )
    assert rq.status_code == 200
    data = rq.json()
    assert data["status"] == "dimension_mismatch"
    assert data["error"]["code"] == "VALIDATION_FAILED"
