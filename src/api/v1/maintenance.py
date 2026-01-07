"""
Maintenance API endpoints
系统维护相关的API端点 - 包含孤儿向量清理、缓存管理等功能
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.params import Query as QueryParam
from pydantic import BaseModel, Field

from src.api.dependencies import get_api_key
from src.core.errors_extended import ErrorCode, build_error
from src.utils.analysis_metrics import vector_cold_pruned_total, vector_orphan_total
from src.utils.analysis_result_store import (
    cleanup_analysis_results,
    get_analysis_result_store_stats,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class OrphanCleanupResponse(BaseModel):
    """孤儿向量清理响应"""

    orphan_count: int = Field(..., description="孤儿向量数量")
    deleted_count: int = Field(..., description="已删除数量")
    sample_ids: Optional[List[str]] = Field(None, description="孤儿ID样例（verbose模式）")
    status: str = Field(..., description="状态: ok/skipped/dry_run")
    message: str = Field(..., description="操作消息")


class KnowledgeReloadResponse(BaseModel):
    """知识库热加载响应"""

    status: str = Field(..., description="状态: ok/error")
    previous_version: str = Field(..., description="重载前版本")
    current_version: str = Field(..., description="重载后版本")
    changed: bool = Field(..., description="版本是否发生变化")


class KnowledgeStatusResponse(BaseModel):
    """知识库状态响应"""

    version: str = Field(..., description="当前版本")
    total_rules: int = Field(..., description="规则总数")
    by_category: Dict[str, int] = Field(..., description="按类别统计")


class AnalysisResultCleanupResponse(BaseModel):
    """分析结果落盘清理响应"""

    status: str = Field(..., description="状态: ok/dry_run/skipped/disabled")
    total_files: int = Field(..., description="分析结果文件总数")
    eligible_count: int = Field(..., description="符合清理条件的文件数")
    expired_count: int = Field(..., description="超过 TTL 的文件数")
    overflow_count: int = Field(..., description="超出最大保留数量的文件数")
    deleted_count: int = Field(..., description="已删除文件数")
    max_age_seconds: Optional[int] = Field(None, description="生存时间阈值 (秒)")
    max_files: Optional[int] = Field(None, description="最大保留文件数")
    sample_ids: Optional[List[str]] = Field(None, description="样例分析ID (verbose)")
    message: str = Field(..., description="操作消息")


@router.post("/knowledge/reload", response_model=KnowledgeReloadResponse)
async def reload_knowledge(api_key: str = Depends(get_api_key)):
    """手动触发知识库热加载"""
    try:
        from src.core.knowledge.dynamic.manager import get_knowledge_manager

        km = get_knowledge_manager()
        prev = km.get_version()
        km.reload()
        curr = km.get_version()
        return KnowledgeReloadResponse(
            status="ok",
            previous_version=prev,
            current_version=curr,
            changed=prev != curr,
        )
    except Exception as e:
        err = build_error(
            ErrorCode.INTERNAL_ERROR,
            stage="knowledge_reload",
            message="Knowledge reload failed",
            detail=str(e),
        )
        raise HTTPException(status_code=500, detail=err)


@router.delete("/analysis-results", response_model=AnalysisResultCleanupResponse)
async def cleanup_analysis_result_store(
    max_age_seconds: Optional[int] = Query(
        None, description="清理超过该年龄的分析结果 (秒)"
    ),
    max_files: Optional[int] = Query(
        None, description="保留的最大分析结果文件数量"
    ),
    threshold: int = Query(0, description="低于该数量则跳过清理"),
    dry_run: bool = Query(False, description="仅统计不执行删除"),
    verbose: bool = Query(False, description="返回样例分析ID"),
    api_key: str = Depends(get_api_key),
):
    """清理落盘的分析结果文件。"""
    if isinstance(max_age_seconds, QueryParam):
        max_age_seconds = max_age_seconds.default
    if isinstance(max_files, QueryParam):
        max_files = max_files.default
    if isinstance(threshold, QueryParam):
        threshold = threshold.default
    if isinstance(dry_run, QueryParam):
        dry_run = dry_run.default
    if isinstance(verbose, QueryParam):
        verbose = verbose.default

    sample_limit = 10 if verbose else 0
    preview = await cleanup_analysis_results(
        max_age_seconds=max_age_seconds,
        max_files=max_files,
        dry_run=True,
        sample_limit=sample_limit,
    )

    if preview["status"] in {"disabled", "skipped"}:
        return AnalysisResultCleanupResponse(**preview)

    if preview["eligible_count"] < threshold and not dry_run:
        preview["status"] = "skipped"
        preview["message"] = (
            f"Eligible count {preview['eligible_count']} below threshold {threshold}"
        )
        preview["deleted_count"] = 0
        return AnalysisResultCleanupResponse(**preview)

    if dry_run:
        return AnalysisResultCleanupResponse(**preview)

    result = await cleanup_analysis_results(
        max_age_seconds=max_age_seconds,
        max_files=max_files,
        dry_run=False,
        sample_limit=sample_limit,
    )
    return AnalysisResultCleanupResponse(**result)

@router.get("/knowledge/status", response_model=KnowledgeStatusResponse)
async def knowledge_status(api_key: str = Depends(get_api_key)):
    """获取知识库版本与规则统计"""
    try:
        from src.core.knowledge.dynamic.manager import get_knowledge_manager
        from src.core.knowledge.dynamic.models import KnowledgeCategory

        km = get_knowledge_manager()
        by_category: Dict[str, int] = {}
        total = 0
        for cat in KnowledgeCategory:
            count = len(km.get_rules_by_category(cat))
            by_category[cat.value] = count
            total += count

        return KnowledgeStatusResponse(
            version=km.get_version(),
            total_rules=total,
            by_category=by_category,
        )
    except Exception as e:
        err = build_error(
            ErrorCode.INTERNAL_ERROR,
            stage="knowledge_status",
            message="Knowledge status query failed",
            detail=str(e),
        )
        raise HTTPException(status_code=500, detail=err)


@router.delete("/orphans", response_model=OrphanCleanupResponse)
async def cleanup_orphan_vectors(
    threshold: int = Query(0, description="最小孤儿向量数量触发清理"),
    force: bool = Query(False, description="强制执行清理"),
    dry_run: bool = Query(False, description="仅统计不执行删除"),
    verbose: bool = Query(False, description="输出部分孤儿ID样例 (限制10个)"),
    api_key: str = Depends(get_api_key),
):
    """
    清理孤儿向量（存在于向量存储但没有对应缓存结果的向量）

    孤儿向量产生原因：
    - 缓存过期但向量未清理
    - 系统异常导致的不一致
    - 手动删除缓存但保留了向量

    Args:
        threshold: 最小孤儿数量阈值，低于此值不执行清理
        force: 强制清理，忽略阈值
        dry_run: 演习模式，只统计不删除
        verbose: 详细模式，返回部分孤儿ID样例
        api_key: API密钥

    Returns:
        清理结果统计
    """
    from src.core.similarity import _VECTOR_LOCK, _VECTOR_STORE
    from src.utils.cache import get_client

    # Get cache client with error handling
    try:
        client = get_client()
    except Exception as e:
        # Redis connection failed
        err = build_error(
            ErrorCode.SERVICE_UNAVAILABLE,
            stage="orphan_cleanup",
            message="Redis connection failed during orphan cleanup",
            detail=str(e),
            operation="cleanup_orphan_vectors",
            resource_id="vector_store",
            suggestion="Check Redis connectivity and retry",
        )
        vector_orphan_total.inc()  # Track the attempt
        raise HTTPException(status_code=503, detail=err)

    orphan_ids: List[str] = []
    redis_errors = 0

    # Find orphan vectors
    if client is not None:
        with _VECTOR_LOCK:
            keys_snapshot = list(_VECTOR_STORE.keys())

        for vid in keys_snapshot:
            try:
                # Check if corresponding cache entry exists
                raw = await client.get(f"analysis_result:{vid}")  # type: ignore[attr-defined]
                if raw is None:
                    orphan_ids.append(vid)
            except (ConnectionError, TimeoutError) as e:
                # Redis operation failed - track and potentially abort
                redis_errors += 1
                logger.warning(f"Redis error checking {vid}: {e}")

                # If too many Redis errors, abort operation
                if redis_errors > 10:
                    err = build_error(
                        ErrorCode.SERVICE_UNAVAILABLE,
                        stage="orphan_cleanup",
                        message="Redis connection unstable, aborting orphan cleanup",
                        checked_count=len(keys_snapshot),
                        error_count=redis_errors,
                        operation="cleanup_orphan_vectors",
                        resource_id="vector_store",
                        suggestion="Check Redis health before retrying",
                    )
                    vector_orphan_total.inc()
                    raise HTTPException(status_code=503, detail=err)
                continue
            except Exception as e:
                logger.warning(f"Unexpected error checking cache for {vid}: {e}")
                continue
    else:
        logger.warning("No cache client available, assuming all vectors are orphans")
        with _VECTOR_LOCK:
            orphan_ids = list(_VECTOR_STORE.keys())

    orphan_count = len(orphan_ids)

    # Check threshold
    if not force and orphan_count < threshold:
        return OrphanCleanupResponse(
            orphan_count=orphan_count,
            deleted_count=0,
            sample_ids=orphan_ids[:10] if verbose else None,
            status="skipped",
            message=f"Orphan count {orphan_count} below threshold {threshold}",
        )

    # Dry run mode
    if dry_run:
        return OrphanCleanupResponse(
            orphan_count=orphan_count,
            deleted_count=0,
            sample_ids=orphan_ids[:10] if verbose else None,
            status="dry_run",
            message=f"Would delete {orphan_count} orphan vectors",
        )

    # Execute deletion
    deleted_count = 0
    with _VECTOR_LOCK:
        for oid in orphan_ids:
            try:
                if oid in _VECTOR_STORE:
                    del _VECTOR_STORE[oid]
                    deleted_count += 1
                    vector_cold_pruned_total.inc()
            except Exception as e:
                logger.error(f"Error deleting orphan vector {oid}: {e}")
                continue

    # Also clean up metadata if exists
    from src.core.similarity import _VECTOR_META  # type: ignore

    with _VECTOR_LOCK:
        for oid in orphan_ids:
            if oid in _VECTOR_META:
                try:
                    del _VECTOR_META[oid]
                except Exception:
                    pass

    logger.info(f"Cleaned up {deleted_count} orphan vectors out of {orphan_count}")

    # Track orphan metrics
    vector_orphan_total.inc(deleted_count)

    return OrphanCleanupResponse(
        orphan_count=orphan_count,
        deleted_count=deleted_count,
        sample_ids=orphan_ids[:10] if verbose else None,
        status="ok",
        message=f"Successfully deleted {deleted_count} orphan vectors",
    )


@router.post("/cache/clear")
async def clear_cache(
    pattern: str = Query("*", description="缓存键模式，支持通配符"), api_key: str = Depends(get_api_key)
):
    """
    清理缓存

    Args:
        pattern: 键模式，支持通配符 (* 和 ?)
        api_key: API密钥

    Returns:
        清理结果
    """
    from src.utils.cache import get_client

    client = get_client()
    if client is None:
        err = build_error(
            ErrorCode.SERVICE_UNAVAILABLE,
            stage="cache_clear",
            message="Cache client not available",
            reason="redis_not_configured",
            suggestion="Configure Redis connection or use alternative caching",
        )
        raise HTTPException(status_code=503, detail=err)

    try:
        # Get matching keys
        keys = await client.keys(pattern)  # type: ignore[attr-defined]
        if not keys:
            return {"deleted_count": 0, "message": f"No keys matching pattern: {pattern}"}

        # Delete keys
        deleted = await client.delete(*keys)  # type: ignore[attr-defined]

        logger.info(f"Cleared {deleted} cache entries matching pattern: {pattern}")
        return {
            "deleted_count": deleted,
            "message": f"Successfully deleted {deleted} cache entries",
        }

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        err = build_error(
            ErrorCode.INTERNAL_ERROR,
            stage="cache_clear",
            message="Failed to clear cache",
            pattern=pattern,
            detail=str(e),
            suggestion="Check Redis connection and retry",
        )
        raise HTTPException(status_code=500, detail=err)


@router.get("/stats")
async def get_maintenance_stats(api_key: str = Depends(get_api_key)):
    """
    获取系统维护统计信息

    Returns:
        维护相关的统计数据
    """
    from src.core.similarity import _VECTOR_LOCK, _VECTOR_META, _VECTOR_STORE  # type: ignore
    from src.utils.cache import get_client

    try:
        with _VECTOR_LOCK:
            total_vectors = len(_VECTOR_STORE)
            metadata_entries = len(_VECTOR_META)
    except Exception as e:
        err = build_error(
            ErrorCode.INTERNAL_ERROR,
            stage="maintenance_stats",
            message="Failed to read vector store stats",
            detail=str(e),
        )
        raise HTTPException(status_code=500, detail=err)

    stats = {
        "vector_store": {"total_vectors": total_vectors, "metadata_entries": metadata_entries},
        "cache": {"available": False, "size": 0},
        "maintenance": {"orphan_check_available": False, "last_cleanup": None},
        "analysis_result_store": {},
    }

    # Check cache stats
    try:
        client = get_client()
        if client is not None:
            try:
                info = await client.info()  # type: ignore[attr-defined]
                stats["cache"]["available"] = True
                stats["cache"]["size"] = info.get("used_memory", 0)
                stats["maintenance"]["orphan_check_available"] = True
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Redis connection issue while getting stats: {e}")
                # Don't fail the whole request, just mark cache unavailable
                stats["cache"]["available"] = False
            except Exception as e:
                logger.warning(f"Could not get cache stats: {e}")
    except Exception as e:
        logger.warning(f"Could not initialize cache client: {e}")

    try:
        stats["analysis_result_store"] = get_analysis_result_store_stats()
    except Exception as e:
        logger.warning(f"Could not get analysis result store stats: {e}")
        stats["analysis_result_store"] = {"enabled": False, "error": str(e)}

    return stats


class VectorStoreReloadResponse(BaseModel):
    status: str
    backend: Optional[str] = None


@router.post("/vectors/backend/reload", response_model=VectorStoreReloadResponse)
async def reload_vector_backend(api_key: str = Depends(get_api_key)):
    """Force reload of vector store backend selection (clears cached instance)."""
    import os

    from src.core.similarity import reload_vector_store_backend
    from src.utils.analysis_metrics import vector_store_reload_total

    try:
        ok = reload_vector_store_backend()
        backend = os.getenv("VECTOR_STORE_BACKEND", "memory")
        vector_store_reload_total.labels(status="success" if ok else "error").inc()

        if not ok:
            # Reload failed but didn't throw exception
            err = build_error(
                ErrorCode.INTERNAL_ERROR,
                stage="backend_reload",
                message="Vector store backend reload failed",
                backend=backend,
                suggestion="Check backend configuration and logs",
            )
            raise HTTPException(status_code=500, detail=err)

        return VectorStoreReloadResponse(status="ok", backend=backend)
    except HTTPException:
        raise
    except Exception as e:
        vector_store_reload_total.labels(status="error").inc()
        err = build_error(
            ErrorCode.INTERNAL_ERROR,
            stage="backend_reload",
            message="Exception during backend reload",
            detail=str(e),
            suggestion="Check system logs and backend availability",
        )
        raise HTTPException(status_code=500, detail=err)
