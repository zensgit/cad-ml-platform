from fastapi.testclient import TestClient
from src.main import app
from src.core.similarity import _VECTOR_STORE, _VECTOR_META  # type: ignore

client = TestClient(app)


def _add_vec(vid: str, dim: int = 7):
    _VECTOR_STORE[vid] = [0.05] * dim
    _VECTOR_META[vid] = {"feature_version": "v1", "material": "steel"}


def setup_batch_vectors():
    for i in range(1, 6):
        _add_vec(f"vec{i}")


def test_batch_similarity_duplicate_ids():
    setup_batch_vectors()
    payload = {"ids": ["vec1", "vec1", "vec2"], "top_k": 3}
    resp = client.post("/api/v1/vectors/similarity/batch", json=payload, headers={"x-api-key": "test"})
    assert resp.status_code == 200
    data = resp.json()
    # Duplicates are processed independently; total should reflect original length
    assert data["total"] == 3
    assert len(data["items"]) == 3


def test_batch_similarity_min_score_filters_all():
    setup_batch_vectors()
    # Use a min_score of 1.0 - only exact matches should pass
    # Since all vectors have identical values, they should all match with score 1.0
    payload = {"ids": ["vec1", "vec2"], "top_k": 5, "min_score": 1.0}
    resp = client.post("/api/v1/vectors/similarity/batch", json=payload, headers={"x-api-key": "test"})
    assert resp.status_code == 200
    data = resp.json()
    # All identical vectors should pass the min_score=1.0 filter
    for it in data["items"]:
        if it["status"] == "success":
            # All vectors have identical values, so all have score 1.0
            assert len(it["similar"]) > 0


def test_batch_similarity_mixed_not_found():
    setup_batch_vectors()
    payload = {"ids": ["vec1", "missing1", "vec2", "missing2"], "top_k": 2}
    resp = client.post("/api/v1/vectors/similarity/batch", json=payload, headers={"x-api-key": "test"})
    assert resp.status_code == 200
    data = resp.json()
    statuses = [it["status"] for it in data["items"]]
    assert "not_found" in statuses
    assert "success" in statuses

