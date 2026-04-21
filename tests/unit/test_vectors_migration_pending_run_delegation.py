from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    return {"X-API-Key": "test"}


def test_migrate_pending_run_delegates_to_pending_run_pipeline():
    async def _run_pending(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["payload"].limit == 5
        assert kwargs["api_key"] == "test"
        return {
            "total": 1,
            "migrated": 0,
            "skipped": 0,
            "items": [{"id": "vec2", "status": "dry_run"}],
            "migration_id": "mid",
            "started_at": None,
            "finished_at": None,
            "dry_run_total": 1,
        }

    with patch("src.api.v1.vectors.run_vector_migration_pending_run_pipeline", _run_pending):
        response = client.post(
            "/api/v1/vectors/migrate/pending/run",
            json={"limit": 5, "dry_run": True},
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    assert response.json()["dry_run_total"] == 1
