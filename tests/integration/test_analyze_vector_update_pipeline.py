from __future__ import annotations

from fastapi.testclient import TestClient

from src.main import app


def test_analyze_vector_update_route_delegates_to_shared_pipeline(monkeypatch) -> None:
    client = TestClient(app)
    captured = {}

    async def _stub_run_vector_update_pipeline(**kwargs):  # noqa: ANN003, ANN201
        captured.update(kwargs)
        return {
            "id": "vec-1",
            "status": "updated",
            "dimension": 5,
            "feature_version": "v2",
        }

    monkeypatch.setattr(
        "src.api.v1.analyze_vector_compat.run_vector_update_pipeline",
        _stub_run_vector_update_pipeline,
    )

    response = client.post(
        "/api/v1/analyze/vectors/update",
        json={"id": "vec-1", "append": [0.1, 0.2]},
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "updated"
    assert payload["dimension"] == 5
    assert captured["payload"].id == "vec-1"
