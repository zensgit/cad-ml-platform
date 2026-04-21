from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    return {"X-API-Key": "test"}


def test_migrate_status_delegates_to_reporting_pipeline():
    async def _snapshot(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["qdrant_store"] == "store"
        return {
            "backend": "qdrant",
            "versions": {"v4": 2},
            "total_vectors": 2,
            "scanned_vectors": 2,
            "scan_limit": 5000,
            "distribution_complete": True,
            "readiness": {
                "target_version": "v4",
                "target_version_vectors": 2,
                "target_version_ratio": 1.0,
                "pending_vectors": 0,
                "migration_ready": True,
            },
        }

    def _payload(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["snapshot"]["backend"] == "qdrant"
        return {
            "last_migration_id": None,
            "last_started_at": None,
            "last_finished_at": None,
            "last_total": None,
            "last_migrated": None,
            "last_skipped": None,
            "pending_vectors": 0,
            "feature_versions": {"v4": 2},
            "history": [],
            "backend": "qdrant",
            "current_total_vectors": 2,
            "scanned_vectors": 2,
            "scan_limit": 5000,
            "distribution_complete": True,
            "target_version": "v4",
            "target_version_vectors": 2,
            "target_version_ratio": 1.0,
            "migration_ready": True,
        }

    with patch("src.api.v1.vectors._get_qdrant_store_or_none", return_value="store"), patch(
        "src.api.v1.vectors.collect_vector_migration_distribution_snapshot",
        _snapshot,
    ), patch(
        "src.api.v1.vectors.build_vector_migration_status_payload",
        _payload,
    ):
        response = client.get("/api/v1/vectors/migrate/status", headers=_auth_headers())

    assert response.status_code == 200
    assert response.json()["backend"] == "qdrant"


def test_migrate_summary_delegates_to_reporting_pipeline():
    async def _snapshot(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["qdrant_store"] == "store"
        return {
            "backend": "memory",
            "versions": {"v4": 1, "v3": 1},
            "total_vectors": 2,
            "scanned_vectors": 2,
            "scan_limit": 5000,
            "distribution_complete": True,
            "readiness": {
                "target_version": "v4",
                "target_version_vectors": 1,
                "target_version_ratio": 0.5,
                "pending_vectors": 1,
                "migration_ready": False,
            },
        }

    def _payload(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["snapshot"]["versions"] == {"v4": 1, "v3": 1}
        return {
            "counts": {"migrated": 1},
            "total_migrations": 1,
            "history_size": 1,
            "statuses": ["migrated"],
            "backend": "memory",
            "current_version_distribution": {"v4": 1, "v3": 1},
            "current_total_vectors": 2,
            "scanned_vectors": 2,
            "scan_limit": 5000,
            "distribution_complete": True,
            "target_version": "v4",
            "target_version_vectors": 1,
            "target_version_ratio": 0.5,
            "pending_vectors": 1,
            "migration_ready": False,
        }

    with patch.dict(
        "src.api.v1.vectors.__dict__",
        {"_VECTOR_MIGRATION_HISTORY": [{"counts": {"migrated": 1}}]},
    ), patch(
        "src.api.v1.vectors._get_qdrant_store_or_none",
        return_value="store",
    ), patch(
        "src.api.v1.vectors.collect_vector_migration_distribution_snapshot",
        _snapshot,
    ), patch(
        "src.api.v1.vectors.build_vector_migration_summary_payload",
        _payload,
    ):
        response = client.get("/api/v1/vectors/migrate/summary", headers=_auth_headers())

    assert response.status_code == 200
    assert response.json()["counts"] == {"migrated": 1}
