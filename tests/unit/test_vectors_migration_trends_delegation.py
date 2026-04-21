from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    return {"X-API-Key": "test"}


def test_migrate_trends_delegates_to_trends_pipeline():
    async def _run_trends(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["window_hours"] == 12
        return kwargs["response_cls"](
            total_migrations=1,
            success_rate=1.0,
            v4_adoption_rate=0.5,
            avg_dimension_delta=2.0,
            window_hours=12,
            version_distribution={"v4": 1, "v3": 1},
            migration_velocity=0.08,
            downgrade_rate=0.0,
            error_rate=0.0,
            time_range={"start": "s", "end": "e"},
            current_total_vectors=2,
            scanned_vectors=2,
            scan_limit=5000,
            distribution_complete=True,
            target_version="v4",
            target_version_vectors=1,
            target_version_ratio=0.5,
            pending_vectors=1,
            migration_ready=False,
        )

    with patch("src.api.v1.vectors.run_vector_migration_trends_pipeline", _run_trends):
        response = client.get(
            "/api/v1/vectors/migrate/trends",
            params={"window_hours": 12},
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    assert response.json()["window_hours"] == 12
