from __future__ import annotations

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_vectors_register_route_delegates_to_shared_helper(monkeypatch):
    captured = {}

    async def _stub_run_vector_register_pipeline(**kwargs):  # noqa: ANN003, ANN201
        captured.update(kwargs)
        return {"id": "vec1", "status": "accepted", "dimension": 7, "error": None}

    monkeypatch.setattr(
        "src.api.v1.vectors.run_vector_register_pipeline",
        _stub_run_vector_register_pipeline,
    )

    response = client.post(
        "/api/v1/vectors/register",
        json={"id": "vec1", "vector": [0.1] * 7},
        headers={"X-API-Key": "test"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "accepted"
    assert captured["payload"].id == "vec1"
    assert captured["response_cls"].__name__ == "VectorRegisterResponse"
