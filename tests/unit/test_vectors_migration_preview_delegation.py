from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    return {"X-API-Key": "test"}


def test_migration_preview_delegates_to_preview_pipeline():
    async def _run_preview(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["to_version"] == "v4"
        assert kwargs["limit"] == 3
        return kwargs["response_cls"](
            total_vectors=1,
            by_version={"v4": 1},
            preview_items=[],
            estimated_dimension_changes={"positive": 0, "negative": 0, "zero": 1},
            migration_feasible=True,
            warnings=[],
            avg_delta=0.0,
            median_delta=0.0,
        )

    with patch("src.api.v1.vectors.run_vector_migration_preview_pipeline", _run_preview):
        response = client.get(
            "/api/v1/vectors/migrate/preview",
            params={"to_version": "v4", "limit": 3},
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    assert response.json()["by_version"] == {"v4": 1}
