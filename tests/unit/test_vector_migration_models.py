from __future__ import annotations

from src.api.v1.vector_migration_models import (
    VectorMigrateItem,
    VectorMigrateRequest,
    VectorMigrateResponse,
    VectorMigrationStatusResponse,
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
