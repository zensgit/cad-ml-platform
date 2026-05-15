from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    return {"X-API-Key": "test"}


def test_migrate_status_uses_facade_scan_limit_resolver() -> None:
    async def _snapshot(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["scan_limit"] == 123
        return {
            "backend": "memory",
            "versions": {"v4": 1},
            "total_vectors": 1,
            "scanned_vectors": 1,
            "scan_limit": 123,
            "distribution_complete": True,
            "readiness": {
                "target_version": "v4",
                "target_version_vectors": 1,
                "target_version_ratio": 1.0,
                "pending_vectors": 0,
                "migration_ready": True,
            },
        }

    def _payload(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["snapshot"]["scan_limit"] == 123
        return {
            "last_migration_id": None,
            "last_started_at": None,
            "last_finished_at": None,
            "last_total": None,
            "last_migrated": None,
            "last_skipped": None,
            "pending_vectors": 0,
            "feature_versions": {"v4": 1},
            "history": [],
            "backend": "memory",
            "current_total_vectors": 1,
            "scanned_vectors": 1,
            "scan_limit": 123,
            "distribution_complete": True,
            "target_version": "v4",
            "target_version_vectors": 1,
            "target_version_ratio": 1.0,
            "migration_ready": True,
        }

    with patch(
        "src.api.v1.vectors._resolve_vector_migration_scan_limit",
        return_value=123,
    ), patch(
        "src.api.v1.vectors.collect_vector_migration_distribution_snapshot",
        _snapshot,
    ), patch(
        "src.api.v1.vectors.build_vector_migration_status_payload",
        _payload,
    ):
        response = client.get("/api/v1/vectors/migrate/status", headers=_auth_headers())

    assert response.status_code == 200
    assert response.json()["scan_limit"] == 123


def test_migrate_pending_uses_facade_target_version_resolver() -> None:
    async def _pending(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["target_version"] == "v3"
        return {
            "target_version": "v3",
            "from_version_filter": None,
            "items": [],
            "listed_count": 0,
            "total_pending": 0,
            "backend": "memory",
            "scanned_vectors": 0,
            "scan_limit": 123,
            "distribution_complete": True,
        }

    with patch(
        "src.api.v1.vectors._resolve_vector_migration_target_version",
        return_value="v3",
    ), patch(
        "src.api.v1.vectors._collect_vector_migration_pending_candidates",
        _pending,
    ):
        response = client.get("/api/v1/vectors/migrate/pending", headers=_auth_headers())

    assert response.status_code == 200
    assert response.json()["target_version"] == "v3"
