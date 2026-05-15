from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.v1 import vectors as vectors_module
from src.api.v1.vector_similarity_models import (
    BatchSimilarityItem,
    BatchSimilarityResponse,
)
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

    def _sentinel_build_filter_conditions(**kwargs):  # noqa: ANN003, ANN201
        return {"sentinel": kwargs}

    monkeypatch.setattr(
        "src.api.v1.vectors.run_vector_batch_similarity",
        _stub_run_vector_batch_similarity,
    )
    monkeypatch.setattr(
        "src.api.v1.vectors._build_vector_filter_conditions",
        _sentinel_build_filter_conditions,
    )

    response = client.post(
        "/api/v1/vectors/similarity/batch",
        json={"ids": ["vec1"], "top_k": 3},
        headers={"X-API-Key": "test"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 1
    assert body["batch_id"] == "batch-1"
    assert body["items"][0]["status"] == "success"
    assert captured["payload"].ids == ["vec1"]
    assert captured["payload"].top_k == 3
    assert captured["batch_item_cls"] is BatchSimilarityItem
    assert captured["batch_response_cls"] is BatchSimilarityResponse
    assert captured["get_qdrant_store_fn"] is vectors_module._get_qdrant_store_or_none
    assert captured["build_filter_conditions_fn"] is _sentinel_build_filter_conditions
