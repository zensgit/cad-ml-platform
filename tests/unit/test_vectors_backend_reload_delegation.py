from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_vectors_backend_reload_delegates_to_shared_pipeline():
    async def _run(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["backend"] == "memory"
        return {"status": "ok", "backend": "memory"}

    with patch(
        "src.api.v1.vectors.run_vector_backend_reload_pipeline",
        _run,
    ):
        response = client.post(
            "/api/v1/vectors/backend/reload?backend=memory",
            headers={"X-API-Key": "test", "X-Admin-Token": "test"},
        )

    assert response.status_code == 200
    assert response.json()["backend"] == "memory"
