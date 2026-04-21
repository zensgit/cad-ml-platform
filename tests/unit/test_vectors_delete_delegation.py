from __future__ import annotations

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_vectors_delete_route_delegates_to_shared_helper(monkeypatch):
    captured = {}

    async def _stub_run_vector_delete_pipeline(**kwargs):  # noqa: ANN003, ANN201
        captured.update(kwargs)
        return {"id": "vec1", "status": "deleted", "error": None}

    monkeypatch.setattr(
        "src.api.v1.vectors.run_vector_delete_pipeline",
        _stub_run_vector_delete_pipeline,
    )

    response = client.post(
        "/api/v1/vectors/delete",
        json={"id": "vec1"},
        headers={"X-API-Key": "test"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "deleted"
    assert captured["payload"].id == "vec1"
    assert captured["response_cls"].__name__ == "VectorDeleteResponse"

