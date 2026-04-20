from __future__ import annotations

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_vectors_batch_similarity_route_delegates_to_shared_helper(monkeypatch):
    captured = {}

    async def _stub_run_vector_batch_similarity(**kwargs):  # noqa: ANN003, ANN201
        captured.update(kwargs)
        return {
            "total": 1,
            "successful": 1,
            "failed": 0,
            "items": [{"id": "vec1", "status": "success", "similar": [], "error": None}],
            "batch_id": "batch-1",
            "duration_ms": 1.23,
            "fallback": None,
            "degraded": False,
        }

    monkeypatch.setattr(
        "src.api.v1.vectors.run_vector_batch_similarity",
        _stub_run_vector_batch_similarity,
    )

    response = client.post(
        "/api/v1/vectors/similarity/batch",
        json={"ids": ["vec1"], "top_k": 3},
        headers={"X-API-Key": "test"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 1
    assert captured["payload"].ids == ["vec1"]
    assert captured["batch_response_cls"].__name__ == "BatchSimilarityResponse"
