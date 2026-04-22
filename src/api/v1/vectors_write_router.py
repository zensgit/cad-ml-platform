from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends

from src.api.dependencies import get_api_key
from src.api.v1.analyze_aux_models import VectorUpdateRequest, VectorUpdateResponse
from src.api.v1.vector_migration_models import (
    VectorMigrateRequest,
    VectorMigrateResponse,
    VectorMigrationPendingRunRequest,
)

router = APIRouter()


@router.post("/update", response_model=VectorUpdateResponse)
async def update_vector(payload: VectorUpdateRequest, api_key: str = Depends(get_api_key)):
    from src.api.v1 import vectors as vectors_module

    result = await vectors_module.run_vector_update_pipeline(
        payload=payload,
        qdrant_store=vectors_module._get_qdrant_store_or_none(),
    )
    return VectorUpdateResponse(**result)


@router.post("/migrate", response_model=VectorMigrateResponse)
async def migrate_vectors(
    payload: VectorMigrateRequest,
    api_key: str = Depends(get_api_key),
):
    from src.api.v1 import vectors as vectors_module
    from src.core.feature_extractor import FeatureExtractor
    from src.core.similarity import _VECTOR_META, _VECTOR_STORE  # type: ignore
    from src.utils.analysis_metrics import (
        analysis_error_code_total,
        vector_migrate_dimension_delta,
        vector_migrate_total,
    )

    return await vectors_module.run_vector_migrate_pipeline(
        payload=payload,
        vector_store=_VECTOR_STORE,
        vector_meta=_VECTOR_META,
        qdrant_store=vectors_module._get_qdrant_store_or_none(),
        feature_extractor_cls=FeatureExtractor,
        prepare_vector_for_upgrade_fn=vectors_module._prepare_vector_for_upgrade,
        vector_layout_base=vectors_module.VECTOR_LAYOUT_BASE,
        vector_layout_l3=vectors_module.VECTOR_LAYOUT_L3,
        dimension_delta_metric=vector_migrate_dimension_delta,
        migrate_total_metric=vector_migrate_total,
        analysis_error_code_total_metric=analysis_error_code_total,
        error_code_cls=vectors_module.ErrorCode,
        build_error_fn=vectors_module.build_error,
        item_cls=vectors_module.VectorMigrateItem,
        response_cls=VectorMigrateResponse,
        history=vectors_module.__dict__.setdefault("_VECTOR_MIGRATION_HISTORY", []),
        uuid4_fn=uuid.uuid4,
        utcnow_fn=datetime.utcnow,
    )


@router.post("/migrate/pending/run", response_model=VectorMigrateResponse)
async def migrate_pending_run(
    payload: VectorMigrationPendingRunRequest,
    api_key: str = Depends(get_api_key),
):
    from src.api.v1 import vectors as vectors_module

    return await vectors_module.run_vector_migration_pending_run_pipeline(
        payload=payload,
        api_key=api_key,
        resolve_target_version_fn=vectors_module._resolve_vector_migration_target_version,
        collect_pending_candidates_fn=vectors_module._collect_vector_migration_pending_candidates,
        migrate_vectors_fn=migrate_vectors,
        request_cls=vectors_module.VectorMigrateRequest,
        error_code_cls=vectors_module.ErrorCode,
        build_error_fn=vectors_module.build_error,
    )


__all__ = ["router", "migrate_vectors", "update_vector", "migrate_pending_run"]
