from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_api_key
from src.api.v1.analyze_aux_models import (
    FaissHealthResponse,
    FeatureCacheStatsResponse,
    FeaturesDiffResponse,
    ModelReloadRequest,
    ModelReloadResponse,
    OrphanCleanupResponse,
    VectorDeleteRequest,
    VectorDeleteResponse,
    VectorDistributionResponse,
    VectorListResponse,
    VectorStatsResponse,
)
from src.core.legacy_redirect_pipeline import raise_legacy_redirect

router = APIRouter()


@router.get("/vectors/distribution", response_model=VectorDistributionResponse)
async def vector_distribution_deprecated(api_key: str = Depends(get_api_key)):
    """Deprecated: moved to /api/v1/vectors_stats/distribution"""
    raise_legacy_redirect(
        old_path="/api/v1/analyze/vectors/distribution",
        new_path="/api/v1/vectors_stats/distribution",
        method="GET",
    )


@router.post("/vectors/delete", response_model=VectorDeleteResponse)
async def delete_vector(
    payload: VectorDeleteRequest, api_key: str = Depends(get_api_key)
):
    """Deprecated: moved to /api/v1/vectors/delete"""
    raise_legacy_redirect(
        old_path="/api/v1/analyze/vectors/delete",
        new_path="/api/v1/vectors/delete",
        method="POST",
    )


@router.get("/vectors", response_model=VectorListResponse)
async def list_vectors(api_key: str = Depends(get_api_key)):
    """Deprecated: moved to /api/v1/vectors"""
    raise_legacy_redirect(
        old_path="/api/v1/analyze/vectors",
        new_path="/api/v1/vectors",
        method="GET",
    )


@router.get("/vectors/stats", response_model=VectorStatsResponse)
async def vector_stats(api_key: str = Depends(get_api_key)):
    """Deprecated: moved to /api/v1/vectors_stats/stats"""
    raise_legacy_redirect(
        old_path="/api/v1/analyze/vectors/stats",
        new_path="/api/v1/vectors_stats/stats",
        method="GET",
    )


@router.get("/features/diff", response_model=FeaturesDiffResponse, deprecated=True)
async def features_diff_deprecated(
    id_a: str, id_b: str, api_key: str = Depends(get_api_key)
):
    """Deprecated: moved to /api/v1/features/diff"""
    raise_legacy_redirect(
        old_path="/api/v1/analyze/features/diff",
        new_path="/api/v1/features/diff",
        method="GET",
    )


@router.post("/model/reload", response_model=ModelReloadResponse, deprecated=True)
async def model_reload_deprecated(
    payload: ModelReloadRequest, api_key: str = Depends(get_api_key)
):
    """Deprecated: moved to /api/v1/model/reload"""
    raise_legacy_redirect(
        old_path="/api/v1/analyze/model/reload",
        new_path="/api/v1/model/reload",
        method="POST",
    )


@router.delete(
    "/vectors/orphans", response_model=OrphanCleanupResponse, deprecated=True
)
async def cleanup_orphan_vectors_deprecated(
    threshold: int = Query(0, description="最小孤儿向量数量触发清理"),
    force: bool = Query(False, description="强制执行清理"),
    dry_run: bool = Query(False, description="仅统计不执行删除"),
    verbose: bool = Query(False, description="输出部分孤儿ID样例 (限制10个)"),
    api_key: str = Depends(get_api_key),
):
    """Deprecated: moved to /api/v1/maintenance/orphans"""
    raise_legacy_redirect(
        old_path="/api/v1/analyze/vectors/orphans",
        new_path="/api/v1/maintenance/orphans",
        method="DELETE",
    )


@router.get("/features/cache", response_model=FeatureCacheStatsResponse)
async def feature_cache_stats(api_key: str = Depends(get_api_key)):
    """Backward-compatible redirect stub. Prefer /api/v1/health/features/cache."""
    raise_legacy_redirect(
        old_path="/api/v1/analyze/features/cache",
        new_path="/api/v1/health/features/cache",
        method="GET",
    )


@router.get("/faiss/health", response_model=FaissHealthResponse)
async def faiss_health(api_key: str = Depends(get_api_key)):
    """Deprecated: moved to /api/v1/health/faiss"""
    raise_legacy_redirect(
        old_path="/api/v1/analyze/faiss/health",
        new_path="/api/v1/health/faiss",
        method="GET",
    )


__all__ = ["router"]
