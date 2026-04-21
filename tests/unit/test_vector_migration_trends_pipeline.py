from __future__ import annotations

import pytest

from src.api.v1.vectors import VectorMigrationTrendsResponse
from src.core.vector_migration_trends_pipeline import run_vector_migration_trends_pipeline


async def _collect_qdrant_feature_versions(store, *, scan_limit):  # noqa: ANN001, ANN202
    assert store == "store"
    assert scan_limit == 2
    return {"v4": 2, "v3": 1, "v1": 1}, 4, 2


def _build_readiness(versions, *, total_vectors, distribution_complete):  # noqa: ANN001, ANN202
    assert versions == {"v4": 2, "v3": 1, "v1": 1}
    assert total_vectors == 4
    assert distribution_complete is False
    return {
        "target_version": "v4",
        "target_version_vectors": None,
        "target_version_ratio": None,
        "pending_vectors": None,
        "migration_ready": False,
    }


@pytest.mark.asyncio
async def test_run_vector_migration_trends_pipeline_partial_qdrant_snapshot():
    response = await run_vector_migration_trends_pipeline(
        window_hours=24,
        history=[
            {
                "migration_id": "test",
                "started_at": "2026-04-21T10:00:00",
                "total": 10,
                "counts": {"migrated": 6, "downgraded": 2, "error": 1, "not_found": 1},
            }
        ],
        response_cls=VectorMigrationTrendsResponse,
        get_qdrant_store_fn=lambda: "store",
        resolve_scan_limit_fn=lambda: 2,
        collect_qdrant_feature_versions_fn=_collect_qdrant_feature_versions,
        build_readiness_fn=_build_readiness,
    )

    assert response.total_migrations == 10
    assert response.success_rate == 0.8
    assert response.downgrade_rate == 0.2
    assert response.error_rate == 0.2
    assert response.v4_adoption_rate == 0.5
    assert response.current_total_vectors == 4
    assert response.scanned_vectors == 2
    assert response.distribution_complete is False


@pytest.mark.asyncio
async def test_run_vector_migration_trends_pipeline_zero_window_keeps_none_start():
    response = await run_vector_migration_trends_pipeline(
        window_hours=0,
        history=[],
        response_cls=VectorMigrationTrendsResponse,
        get_qdrant_store_fn=lambda: None,
        resolve_scan_limit_fn=lambda: 5000,
        collect_qdrant_feature_versions_fn=None,  # type: ignore[arg-type]
        build_readiness_fn=lambda versions, *, total_vectors, distribution_complete: {  # noqa: ARG005
            "target_version": "v4",
            "target_version_vectors": 0,
            "target_version_ratio": 0.0,
            "pending_vectors": 0,
            "migration_ready": True,
        },
    )

    assert response.window_hours == 0
    assert response.time_range["start"] is None
