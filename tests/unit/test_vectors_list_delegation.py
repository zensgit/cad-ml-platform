from __future__ import annotations

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_vectors_list_route_delegates_to_shared_helper(monkeypatch):
    captured = {}

    async def _stub_run_vector_list_pipeline(**kwargs):  # noqa: ANN003, ANN201
        captured.update(kwargs)
        return {"total": 1, "vectors": [{"id": "vec1", "dimension": 7}]}

    monkeypatch.setattr(
        "src.api.v1.vectors.run_vector_list_pipeline",
        _stub_run_vector_list_pipeline,
    )

    response = client.get("/api/v1/vectors?source=memory&limit=5", headers={"X-API-Key": "test"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 1
    assert captured["source"] == "memory"
    assert captured["response_cls"].__name__ == "VectorListResponse"

