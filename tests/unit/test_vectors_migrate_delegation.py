from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_vectors_migrate_delegates_to_shared_pipeline():
    def _sentinel_prepare(*_args):  # noqa: ANN002, ANN202
        return [], [], "sentinel"

    async def _run(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["payload"].to_version == "v2"
        assert kwargs["qdrant_store"] == "qdrant-store"
        assert kwargs["prepare_vector_for_upgrade_fn"] is _sentinel_prepare
        return {
            "total": 1,
            "migrated": 0,
            "skipped": 0,
            "items": [{"id": "vec1", "status": "dry_run"}],
            "migration_id": "mid",
            "started_at": None,
            "finished_at": None,
            "dry_run_total": 1,
        }

    with patch(
        "src.api.v1.vectors._prepare_vector_for_upgrade",
        _sentinel_prepare,
    ), patch(
        "src.api.v1.vectors.run_vector_migrate_pipeline",
        _run,
    ), patch(
        "src.api.v1.vectors._get_qdrant_store_or_none",
        return_value="qdrant-store",
    ):
        response = client.post(
            "/api/v1/vectors/migrate",
            json={"ids": ["vec1"], "to_version": "v2", "dry_run": True},
            headers={"X-API-Key": "test"},
        )

    assert response.status_code == 200
    assert response.json()["dry_run_total"] == 1
