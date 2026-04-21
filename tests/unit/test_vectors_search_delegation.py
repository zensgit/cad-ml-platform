from __future__ import annotations

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_vectors_search_route_delegates_to_shared_helper(monkeypatch):
    captured = {}

    async def _stub_run_vector_search_pipeline(**kwargs):  # noqa: ANN003, ANN201
        captured.update(kwargs)
        return {"results": [{"id": "vec1", "score": 0.9}], "total": 1}

    monkeypatch.setattr(
        "src.api.v1.vectors.run_vector_search_pipeline",
        _stub_run_vector_search_pipeline,
    )

    response = client.post(
        "/api/v1/vectors/search",
        json={"vector": [0.1] * 7, "k": 3},
        headers={"X-API-Key": "test"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 1
    assert captured["payload"].k == 3
    assert captured["response_cls"].__name__ == "VectorSearchResponse"
