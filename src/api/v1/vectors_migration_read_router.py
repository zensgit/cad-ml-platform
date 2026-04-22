from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_api_key
from src.api.v1.vector_migration_models import (
    VectorMigrationPendingResponse,
    VectorMigrationPendingSummaryResponse,
    VectorMigrationPlanResponse,
    VectorMigrationPreviewResponse,
    VectorMigrationStatusResponse,
    VectorMigrationSummaryResponse,
    VectorMigrationTrendsResponse,
)

router = APIRouter()


@router.get("/migrate/preview", response_model=VectorMigrationPreviewResponse)
async def preview_migration(to_version: str, limit: int = 10, api_key: str = Depends(get_api_key)):
    from src.api.v1 import vectors as vectors_module
    from src.core.feature_extractor import FeatureExtractor

    return await vectors_module.run_vector_migration_preview_pipeline(
        to_version=to_version,
        limit=limit,
        response_cls=VectorMigrationPreviewResponse,
        error_code_cls=vectors_module.ErrorCode,
        build_error_fn=vectors_module.build_error,
        get_qdrant_store_fn=vectors_module._get_qdrant_store_or_none,
        collect_qdrant_preview_samples_fn=vectors_module._collect_qdrant_preview_samples,
        prepare_vector_for_upgrade_fn=vectors_module._prepare_vector_for_upgrade,
        feature_extractor_cls=FeatureExtractor,
    )


@router.get("/migrate/status", response_model=VectorMigrationStatusResponse)
async def migrate_status(api_key: str = Depends(get_api_key)):
    from src.api.v1 import vectors as vectors_module

    history = list(getattr(vectors_module, "_VECTOR_MIGRATION_HISTORY", []))
    snapshot = await vectors_module.collect_vector_migration_distribution_snapshot(
        qdrant_store=vectors_module._get_qdrant_store_or_none(),
        scan_limit=vectors_module._resolve_vector_migration_scan_limit(),
        collect_qdrant_feature_versions_fn=vectors_module._collect_qdrant_feature_versions,
        build_readiness_fn=vectors_module._build_vector_migration_readiness,
    )
    return VectorMigrationStatusResponse(
        **vectors_module.build_vector_migration_status_payload(
            history=history,
            snapshot=snapshot,
        )
    )


@router.get("/migrate/summary", response_model=VectorMigrationSummaryResponse)
async def migrate_summary(api_key: str = Depends(get_api_key)):
    from src.api.v1 import vectors as vectors_module

    history = list(getattr(vectors_module, "_VECTOR_MIGRATION_HISTORY", []))
    snapshot = await vectors_module.collect_vector_migration_distribution_snapshot(
        qdrant_store=vectors_module._get_qdrant_store_or_none(),
        scan_limit=vectors_module._resolve_vector_migration_scan_limit(),
        collect_qdrant_feature_versions_fn=vectors_module._collect_qdrant_feature_versions,
        build_readiness_fn=vectors_module._build_vector_migration_readiness,
    )
    return VectorMigrationSummaryResponse(
        **vectors_module.build_vector_migration_summary_payload(
            history=history,
            snapshot=snapshot,
        )
    )


@router.get("/migrate/pending", response_model=VectorMigrationPendingResponse)
async def migrate_pending(
    limit: int = 50,
    from_version_filter: str | None = Query(default=None),
    api_key: str = Depends(get_api_key),
):
    from src.api.v1 import vectors as vectors_module

    limit = max(min(int(limit or 0), 200), 1)
    target_version = vectors_module._resolve_vector_migration_target_version()
    pending = await vectors_module._collect_vector_migration_pending_candidates(
        limit=limit,
        target_version=target_version,
        from_version_filter=from_version_filter,
    )
    return VectorMigrationPendingResponse(
        target_version=pending["target_version"],
        from_version_filter=pending["from_version_filter"],
        items=pending["items"],
        listed_count=pending["listed_count"],
        total_pending=pending["total_pending"],
        backend=pending["backend"],
        scanned_vectors=pending["scanned_vectors"],
        scan_limit=pending["scan_limit"],
        distribution_complete=pending["distribution_complete"],
    )


@router.get("/migrate/pending/summary", response_model=VectorMigrationPendingSummaryResponse)
async def migrate_pending_summary(
    from_version_filter: str | None = Query(default=None),
    api_key: str = Depends(get_api_key),
):
    from src.api.v1 import vectors as vectors_module

    target_version = vectors_module._resolve_vector_migration_target_version()
    pending = await vectors_module._collect_vector_migration_pending_candidates(
        limit=1,
        target_version=target_version,
        from_version_filter=from_version_filter,
    )
    return VectorMigrationPendingSummaryResponse(
        **vectors_module.build_vector_migration_pending_summary_payload(pending=pending)
    )


@router.get("/migrate/plan", response_model=VectorMigrationPlanResponse)
async def migrate_plan(
    from_version_filter: str | None = Query(default=None),
    max_batches: int = Query(default=3, ge=1, le=10),
    default_run_limit: int = Query(default=50, ge=1, le=200),
    api_key: str = Depends(get_api_key),
):
    from src.api.v1 import vectors as vectors_module

    target_version = vectors_module._resolve_vector_migration_target_version()
    pending = await vectors_module._collect_vector_migration_pending_candidates(
        limit=1,
        target_version=target_version,
        from_version_filter=from_version_filter,
    )
    return VectorMigrationPlanResponse(
        **vectors_module.build_vector_migration_plan_payload(
            pending=pending,
            max_batches=max_batches,
            default_run_limit=default_run_limit,
        )
    )


@router.get("/migrate/trends", response_model=VectorMigrationTrendsResponse)
async def migrate_trends(window_hours: int = 24, api_key: str = Depends(get_api_key)):
    from src.api.v1 import vectors as vectors_module

    return await vectors_module.run_vector_migration_trends_pipeline(
        window_hours=window_hours,
        history=list(getattr(vectors_module, "_VECTOR_MIGRATION_HISTORY", [])),
        response_cls=VectorMigrationTrendsResponse,
        get_qdrant_store_fn=vectors_module._get_qdrant_store_or_none,
        resolve_scan_limit_fn=vectors_module._resolve_vector_migration_scan_limit,
        collect_qdrant_feature_versions_fn=vectors_module._collect_qdrant_feature_versions,
        build_readiness_fn=vectors_module._build_vector_migration_readiness,
    )


__all__ = ["router"]
