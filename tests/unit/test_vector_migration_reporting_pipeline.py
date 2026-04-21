from __future__ import annotations

from datetime import datetime

import pytest

from src.core.vector_migration_reporting_pipeline import (
    build_vector_migration_status_payload,
    build_vector_migration_summary_payload,
    collect_vector_migration_distribution_snapshot,
)


@pytest.mark.asyncio
async def test_collect_vector_migration_distribution_snapshot_qdrant():
    class DummyStore:
        pass

    async def _collect_qdrant_feature_versions(store, *, scan_limit):  # noqa: ANN001, ANN202
        assert isinstance(store, DummyStore)
        assert scan_limit == 10
        return {"v4": 2, "v3": 1}, 3, 3

    def _build_readiness(versions, *, total_vectors, distribution_complete):  # noqa: ANN001, ANN202
        assert versions == {"v4": 2, "v3": 1}
        assert total_vectors == 3
        assert distribution_complete is True
        return {
            "target_version": "v4",
            "target_version_vectors": 2,
            "target_version_ratio": 0.6667,
            "pending_vectors": 1,
            "migration_ready": False,
        }

    snapshot = await collect_vector_migration_distribution_snapshot(
        qdrant_store=DummyStore(),
        scan_limit=10,
        collect_qdrant_feature_versions_fn=_collect_qdrant_feature_versions,
        build_readiness_fn=_build_readiness,
    )

    assert snapshot == {
        "backend": "qdrant",
        "versions": {"v4": 2, "v3": 1},
        "total_vectors": 3,
        "scanned_vectors": 3,
        "scan_limit": 10,
        "distribution_complete": True,
        "readiness": {
            "target_version": "v4",
            "target_version_vectors": 2,
            "target_version_ratio": 0.6667,
            "pending_vectors": 1,
            "migration_ready": False,
        },
    }


def test_build_vector_migration_status_payload():
    history = [
        {
            "migration_id": "mig-1",
            "started_at": "2026-04-21T10:00:00",
            "finished_at": "2026-04-21T10:05:00",
            "total": 2,
            "migrated": 1,
            "skipped": 1,
        }
    ]
    snapshot = {
        "backend": "memory",
        "versions": {"v4": 3, "v2": 1},
        "total_vectors": 4,
        "scanned_vectors": 4,
        "scan_limit": 5000,
        "distribution_complete": True,
        "readiness": {
            "target_version": "v4",
            "target_version_vectors": 3,
            "target_version_ratio": 0.75,
            "pending_vectors": 1,
            "migration_ready": False,
        },
    }

    payload = build_vector_migration_status_payload(history=history, snapshot=snapshot)

    assert payload["last_migration_id"] == "mig-1"
    assert payload["last_started_at"] == datetime.fromisoformat("2026-04-21T10:00:00")
    assert payload["last_finished_at"] == datetime.fromisoformat("2026-04-21T10:05:00")
    assert payload["feature_versions"] == {"v4": 3, "v2": 1}
    assert payload["pending_vectors"] == 1
    assert payload["current_total_vectors"] == 4


def test_build_vector_migration_summary_payload():
    history = [
        {"counts": {"migrated": 2, "error": 1}},
        {"counts": {"migrated": 1, "dry_run": 3}},
    ]
    snapshot = {
        "backend": "qdrant",
        "versions": {"v4": 2, "v3": 2},
        "total_vectors": 4,
        "scanned_vectors": 4,
        "scan_limit": 5000,
        "distribution_complete": True,
        "readiness": {
            "target_version": "v4",
            "target_version_vectors": 2,
            "target_version_ratio": 0.5,
            "pending_vectors": 2,
            "migration_ready": False,
        },
    }

    payload = build_vector_migration_summary_payload(history=history, snapshot=snapshot)

    assert payload["counts"] == {"dry_run": 3, "error": 1, "migrated": 3}
    assert payload["total_migrations"] == 7
    assert payload["statuses"] == ["dry_run", "error", "migrated"]
    assert payload["backend"] == "qdrant"
    assert payload["current_version_distribution"] == {"v4": 2, "v3": 2}
