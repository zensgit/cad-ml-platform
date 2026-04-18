from __future__ import annotations

from src.api.v1.vector_migration_models import (
    VectorMigrateItem,
    VectorMigrateRequest,
    VectorMigrateResponse,
    VectorMigrationPendingItem,
    VectorMigrationPendingResponse,
    VectorMigrationPendingRunRequest,
    VectorMigrationPendingSummaryResponse,
    VectorMigrationPlanBatch,
    VectorMigrationPlanResponse,
    VectorMigrationPreviewResponse,
    VectorMigrationStatusResponse,
    VectorMigrationSummaryResponse,
    VectorMigrationTrendsResponse,
)


def test_vector_migrate_request_schema():
    payload = VectorMigrateRequest(ids=["vec-1"], to_version="v4", dry_run=True)

    assert payload.ids == ["vec-1"]
    assert payload.to_version == "v4"
    assert payload.dry_run is True


def test_vector_migrate_response_embeds_items():
    response = VectorMigrateResponse(
        total=1,
        migrated=1,
        skipped=0,
        items=[
            VectorMigrateItem(
                id="vec-1",
                status="migrated",
                from_version="v3",
                to_version="v4",
                dimension_before=22,
                dimension_after=24,
            )
        ],
    )

    assert response.items[0].status == "migrated"
    assert response.items[0].dimension_after == 24


def test_vector_migration_status_response_rich_defaults():
    response = VectorMigrationStatusResponse()

    assert response.backend == "memory"
    assert response.distribution_complete is True
    assert response.target_version == "v4"
    assert response.migration_ready is False
    assert response.pending_vectors is None


def test_vector_migration_summary_response_defaults():
    response = VectorMigrationSummaryResponse(
        counts={"migrated": 2, "skipped": 1},
        total_migrations=3,
        history_size=4,
        statuses=["completed", "partial"],
    )

    assert response.backend == "memory"
    assert response.total_migrations == 3
    assert response.statuses == ["completed", "partial"]
    assert response.migration_ready is False


def test_vector_migration_pending_response_embeds_items():
    response = VectorMigrationPendingResponse(
        target_version="v4",
        items=[VectorMigrationPendingItem(id="vec-2", from_version="v3", to_version="v4")],
        listed_count=1,
    )

    assert response.items[0].id == "vec-2"
    assert response.items[0].from_version == "v3"
    assert response.distribution_complete is True


def test_vector_migration_pending_summary_response_defaults():
    response = VectorMigrationPendingSummaryResponse(
        target_version="v4",
        observed_by_from_version={"v3": 2},
    )

    assert response.recommended_from_versions == []
    assert response.backend == "memory"
    assert response.distribution_complete is True


def test_vector_migration_plan_response_embeds_batches():
    batch = VectorMigrationPlanBatch(
        priority=1,
        from_version="v3",
        pending_count=2,
        suggested_run_limit=2,
        request_payload={"limit": 2, "dry_run": True, "from_version_filter": "v3"},
    )
    response = VectorMigrationPlanResponse(
        target_version="v4",
        observed_by_from_version={"v3": 2},
        max_batches=2,
        default_run_limit=2,
        recommended_first_batch=batch,
        batches=[batch],
    )

    assert response.batches[0].from_version == "v3"
    assert response.recommended_first_batch is not None
    assert response.coverage_complete is False


def test_vector_migration_pending_run_request_defaults():
    payload = VectorMigrationPendingRunRequest()

    assert payload.limit == 50
    assert payload.dry_run is False
    assert payload.allow_partial_scan is False


def test_vector_migration_preview_response_contract():
    response = VectorMigrationPreviewResponse(
        total_vectors=3,
        by_version={"v4": 1, "v3": 2},
        preview_items=[VectorMigrateItem(id="vec-1", status="upgrade", from_version="v3")],
        estimated_dimension_changes={"positive": 1, "negative": 0, "zero": 2},
        migration_feasible=True,
    )

    assert response.migration_feasible is True
    assert response.preview_items[0].status == "upgrade"
    assert response.warnings == []


def test_vector_migration_trends_response_defaults():
    response = VectorMigrationTrendsResponse(
        total_migrations=10,
        success_rate=0.9,
        v4_adoption_rate=0.8,
        avg_dimension_delta=1.5,
        window_hours=24,
        version_distribution={"v4": 8, "v3": 2},
        migration_velocity=0.4,
        downgrade_rate=0.1,
        error_rate=0.1,
        time_range={"start": None, "end": None},
    )

    assert response.target_version == "v4"
    assert response.migration_ready is False
    assert response.distribution_complete is True
