"""Vector management endpoints extracted from analyze.py for modularity."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import APIRouter

# Compatibility facade re-export: pre-split vectors.py imported these from
# src.api.dependencies, and tests monkeypatch `src.api.v1.vectors.get_api_key`
# (tests/unit/test_migration_preview_trends.py). Dropping this import in the
# facade refactor (commit 17a28676) broke that surface. get_admin_token is
# restored for symmetry with the pre-split facade; no current test exercises it.
from src.api.dependencies import get_admin_token, get_api_key

from src.api.v1.vector_crud_models import (
    VectorDeleteRequest,
    VectorDeleteResponse,
    VectorRegisterRequest,
    VectorRegisterResponse,
    VectorSearchRequest,
    VectorSearchResponse,
)
from src.api.v1.vectors_admin_router import (
    VectorBackendReloadResponse,
    _vector_reload_admin_token,
    reload_vector_backend,
    router as admin_router,
)
from src.api.v1.vector_list_models import VectorListItem, VectorListResponse
from src.api.v1.vector_similarity_models import (
    BatchSimilarityItem,
    BatchSimilarityRequest,
    BatchSimilarityResponse,
)
from src.api.v1.vectors_crud_router import router as crud_router
from src.api.v1.vectors_list_router import list_vectors, router as list_router
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
from src.api.v1.vectors_migration_read_router import (
    migrate_pending,
    migrate_pending_summary,
    migrate_plan,
    migrate_status,
    migrate_summary,
    migrate_trends,
    preview_migration,
    router as migration_read_router,
)
from src.api.v1.vectors_similarity_router import (
    batch_similarity,
    router as similarity_router,
)
from src.api.v1.vectors_write_router import (
    migrate_pending_run,
    migrate_vectors,
    router as write_router,
    update_vector,
)
from src.core.errors_extended import ErrorCode, build_error
from src.core.qdrant_store_helper import (
    get_qdrant_store_or_none as _get_qdrant_store_or_none,
)
from src.core.vector_delete_pipeline import run_vector_delete_pipeline
from src.core.vector_list_pipeline import run_vector_list_pipeline
from src.core.vector_register_pipeline import run_vector_register_pipeline
from src.core.vector_search_pipeline import run_vector_search_pipeline
from src.core.vector_migration_reporting_pipeline import (
    build_vector_migration_status_payload,
    build_vector_migration_summary_payload,
    collect_vector_migration_distribution_snapshot,
)
from src.core.vector_migration_pending_candidates import (
    collect_vector_migration_pending_candidates,
)
from src.core.vector_migrate_pipeline import run_vector_migrate_pipeline
from src.core.vector_migration_plan_pipeline import (
    build_vector_migration_pending_summary_payload,
    build_vector_migration_plan_payload,
)
from src.core.vector_migration_preview_pipeline import run_vector_migration_preview_pipeline
from src.core.vector_migration_pending_run_pipeline import (
    run_vector_migration_pending_run_pipeline,
)
from src.core.vector_migration_trends_pipeline import run_vector_migration_trends_pipeline
from src.core.vector_batch_similarity import run_vector_batch_similarity
from src.core.vector_backend_reload_pipeline import run_vector_backend_reload_pipeline
from src.core.vector_update_pipeline import run_vector_update_pipeline
from src.core.vector_filtering import (
    build_vector_filter_conditions as _build_vector_filter_conditions,
    build_vector_search_filter_conditions as _build_vector_search_filter_conditions,
    matches_vector_label_filters as _matches_vector_label_filters,
    matches_vector_search_filters as _matches_vector_search_filters,
    vector_item_payload as _vector_item_payload,
)
from src.core.vector_list_sources import (
    resolve_vector_list_source as _resolve_list_source,
)
from src.core.vector_list_memory import list_vectors_memory as _list_vectors_memory_core
from src.core.vector_list_redis import list_vectors_redis as _list_vectors_redis_core
from src.core.vector_migration_config import (
    coerce_optional_int as _coerce_int,
    resolve_vector_migration_scan_limit as _resolve_vector_migration_scan_limit,
    resolve_vector_migration_target_version as _resolve_vector_migration_target_version,
)
from src.core.vector_migration_upgrade import (
    prepare_vector_for_upgrade as _prepare_vector_for_upgrade,
)
from src.core.vector_migration_readiness import (
    build_vector_migration_readiness as _build_vector_migration_readiness_core,
)
from src.core.vector_migration_qdrant_versions import (
    collect_qdrant_feature_versions as _collect_qdrant_feature_versions_core,
)
from src.core.vector_migration_qdrant_preview import (
    collect_qdrant_preview_samples as _collect_qdrant_preview_samples,
)
from src.core.vector_layouts import (
    VECTOR_LAYOUT_BASE,
    VECTOR_LAYOUT_L3,
    VECTOR_LAYOUT_LEGACY,
    layout_has_l3,
)
from src.utils.cache import get_client

router = APIRouter()


if TYPE_CHECKING:
    from src.core.feature_extractor import FeatureExtractor


router.include_router(crud_router)
router.include_router(migration_read_router)
router.include_router(write_router)
router.include_router(list_router)
router.include_router(similarity_router)
router.include_router(admin_router)

def _build_vector_migration_readiness(
    version_distribution: Dict[str, int],
    *,
    total_vectors: int,
    distribution_complete: bool,
) -> Dict[str, Any]:
    return _build_vector_migration_readiness_core(
        version_distribution,
        total_vectors=total_vectors,
        distribution_complete=distribution_complete,
        resolve_target_version_fn=_resolve_vector_migration_target_version,
    )


async def _collect_qdrant_feature_versions(
    qdrant_store,
    *,
    scan_limit: int | None = None,
) -> tuple[Dict[str, int], int, int]:
    return await _collect_qdrant_feature_versions_core(
        qdrant_store,
        scan_limit=scan_limit,
        resolve_scan_limit_fn=_resolve_vector_migration_scan_limit,
    )


def _list_vectors_memory(
    vector_store: Dict[str, list[float]],
    vector_meta: Dict[str, Dict[str, str]],
    offset: int,
    limit: int,
    material_filter: Optional[str],
    complexity_filter: Optional[str],
    fine_part_type_filter: Optional[str],
    coarse_part_type_filter: Optional[str],
    decision_source_filter: Optional[str],
    is_coarse_label_filter: Optional[bool],
) -> VectorListResponse:
    return _list_vectors_memory_core(
        vector_store,
        vector_meta,
        offset,
        limit,
        material_filter,
        complexity_filter,
        fine_part_type_filter,
        coarse_part_type_filter,
        decision_source_filter,
        is_coarse_label_filter,
        item_cls=VectorListItem,
        response_cls=VectorListResponse,
        matches_label_filters_fn=_matches_vector_label_filters,
    )


async def _list_vectors_redis(
    client,
    offset: int,
    limit: int,
    scan_limit: int,
    material_filter: Optional[str],
    complexity_filter: Optional[str],
    fine_part_type_filter: Optional[str],
    coarse_part_type_filter: Optional[str],
    decision_source_filter: Optional[str],
    is_coarse_label_filter: Optional[bool],
) -> VectorListResponse:
    from src.core.similarity import extract_vector_label_contract

    return await _list_vectors_redis_core(
        client,
        offset,
        limit,
        scan_limit,
        material_filter,
        complexity_filter,
        fine_part_type_filter,
        coarse_part_type_filter,
        decision_source_filter,
        is_coarse_label_filter,
        item_cls=VectorListItem,
        response_cls=VectorListResponse,
        matches_label_filters_fn=_matches_vector_label_filters,
        extract_label_contract_fn=extract_vector_label_contract,
        json_loads_fn=json.loads,
    )

__all__ = [
    "router",
    "list_vectors",
    "batch_similarity",
    "update_vector",
    "migrate_vectors",
    "migrate_pending_run",
    "preview_migration",
    "migrate_status",
    "migrate_summary",
    "migrate_pending",
    "migrate_pending_summary",
    "migrate_plan",
    "migrate_trends",
    "VectorMigrateItem",
    "VectorMigrateRequest",
    "VectorMigrateResponse",
    "VectorMigrationStatusResponse",
    "VectorMigrationSummaryResponse",
    "VectorMigrationPendingItem",
    "VectorMigrationPendingResponse",
    "VectorMigrationPendingSummaryResponse",
    "VectorMigrationPlanBatch",
    "VectorMigrationPlanResponse",
    "VectorMigrationPendingRunRequest",
    "VectorMigrationPreviewResponse",
    "VectorMigrationTrendsResponse",
    "VectorListItem",
    "VectorListResponse",
    "BatchSimilarityItem",
    "BatchSimilarityRequest",
    "BatchSimilarityResponse",
    "VectorBackendReloadResponse",
    "_vector_reload_admin_token",
    "reload_vector_backend",
    "get_api_key",
    "get_admin_token",
]


async def _collect_vector_migration_pending_candidates(
    *,
    limit: int,
    target_version: str,
    from_version_filter: Optional[str] = None,
) -> Dict[str, Any]:
    from src.core.similarity import _VECTOR_META, _VECTOR_STORE  # type: ignore

    return await collect_vector_migration_pending_candidates(
        limit=limit,
        target_version=target_version,
        from_version_filter=from_version_filter,
        qdrant_store=_get_qdrant_store_or_none(),
        scan_limit=_resolve_vector_migration_scan_limit(),
        item_cls=VectorMigrationPendingItem,
        vector_meta=_VECTOR_META,
        vector_store=_VECTOR_STORE,
    )
