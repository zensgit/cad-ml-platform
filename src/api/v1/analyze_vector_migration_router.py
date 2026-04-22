from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_api_key
from src.api.v1.analyze_aux_models import (
    VectorMigrateRequest,
    VectorMigrateResponse,
    VectorMigrationStatusResponse,
)
from src.core.legacy_vector_migration_pipeline import (
    run_legacy_vector_migrate_pipeline,
    run_legacy_vector_migration_status_pipeline,
)

router = APIRouter()


@router.post("/vectors/migrate", response_model=VectorMigrateResponse)
async def migrate_vectors(
    payload: VectorMigrateRequest, api_key: str = Depends(get_api_key)
):
    result = await run_legacy_vector_migrate_pipeline(payload=payload)
    return VectorMigrateResponse(**result)


@router.get("/vectors/migrate/status", response_model=VectorMigrationStatusResponse)
async def vector_migration_status(api_key: str = Depends(get_api_key)):
    result = run_legacy_vector_migration_status_pipeline()
    return VectorMigrationStatusResponse(**result)


__all__ = ["router"]
