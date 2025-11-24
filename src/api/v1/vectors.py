"""Vector management endpoints extracted from analyze.py for modularity."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_api_key
from src.core.errors_extended import build_error, ErrorCode

router = APIRouter()


class VectorDeleteRequest(BaseModel):
    id: str = Field(description="要删除的向量分析ID")


class VectorDeleteResponse(BaseModel):
    id: str
    status: str
    error: Dict[str, Any] | None = None


class VectorListItem(BaseModel):
    id: str
    dimension: int
    material: str | None = None
    complexity: str | None = None
    format: str | None = None


class VectorListResponse(BaseModel):
    total: int
    vectors: list[VectorListItem]


@router.post("/vectors/delete", response_model=VectorDeleteResponse)
async def delete_vector(payload: VectorDeleteRequest, api_key: str = Depends(get_api_key)):
    from src.core.similarity import _VECTOR_STORE, _VECTOR_META, _BACKEND, FaissVectorStore  # type: ignore
    from src.utils.cache import get_client
    if payload.id not in _VECTOR_STORE:
        err = build_error(
            ErrorCode.DATA_NOT_FOUND,
            stage="vector_delete",
            message="Vector not found",
            id=payload.id,
        )
        raise HTTPException(status_code=404, detail=err)
    try:
        if __import__("os").getenv("VECTOR_STORE_BACKEND", "memory") == "faiss":
            try:
                f = FaissVectorStore()
                f.mark_delete(payload.id)  # type: ignore[attr-defined]
            except Exception:
                pass
        del _VECTOR_STORE[payload.id]
        _VECTOR_META.pop(payload.id, None)
        if _BACKEND == "redis":
            client = get_client()
            if client is not None:
                try:
                    client.delete(f"vector:{payload.id}")
                except Exception:
                    pass
        return VectorDeleteResponse(id=payload.id, status="deleted")
    except Exception as e:
        err = build_error(
            ErrorCode.INTERNAL_ERROR,
            stage="vector_delete",
            message="Delete failed",
            id=payload.id,
            detail=str(e),
        )
        raise HTTPException(status_code=500, detail=err)


@router.get("/vectors", response_model=VectorListResponse)
async def list_vectors(api_key: str = Depends(get_api_key)):
    from src.core.similarity import _VECTOR_STORE, _VECTOR_META  # type: ignore
    items: list[VectorListItem] = []
    for vid, vec in _VECTOR_STORE.items():
        meta = _VECTOR_META.get(vid, {})
        items.append(
            VectorListItem(
                id=vid,
                dimension=len(vec),
                material=meta.get("material"),
                complexity=meta.get("complexity"),
                format=meta.get("format"),
            )
        )
    return VectorListResponse(total=len(items), vectors=items)


__all__ = ["router"]
class VectorUpdateRequest(BaseModel):
    id: str = Field(description="要更新的向量分析ID")
    replace: list[float] | None = Field(default=None, description="新的向量 (维度需与原向量一致)")
    append: list[float] | None = Field(default=None, description="追加的向量片段 (若提供 replace 则忽略)")
    material: str | None = Field(default=None, description="更新材料元数据")
    complexity: str | None = Field(default=None, description="更新复杂度元数据")
    format: str | None = Field(default=None, description="更新格式元数据")


class VectorUpdateResponse(BaseModel):
    id: str
    status: str
    dimension: int | None = None
    error: Dict[str, Any] | None = None
    feature_version: str | None = None


@router.post("/vectors/update", response_model=VectorUpdateResponse)
async def update_vector(payload: VectorUpdateRequest, api_key: str = Depends(get_api_key)):
    from src.core.similarity import _VECTOR_STORE, _VECTOR_META  # type: ignore
    from src.utils.analysis_metrics import analysis_error_code_total
    if payload.id not in _VECTOR_STORE:
        err = build_error(ErrorCode.DATA_NOT_FOUND, stage="vector_update", message="Vector not found", id=payload.id)
        analysis_error_code_total.labels(code=ErrorCode.DATA_NOT_FOUND.value).inc()
        return VectorUpdateResponse(id=payload.id, status="not_found", error=err)
    vec = _VECTOR_STORE[payload.id]
    original_dim = len(vec)
    enforce = __import__("os").getenv("ANALYSIS_VECTOR_DIM_CHECK", "0") == "1"
    try:
        if payload.replace is not None:
            if len(payload.replace) != original_dim:
                if enforce:
                    err = build_error(
                        ErrorCode.DIMENSION_MISMATCH,
                        stage="vector_update",
                        message=f"Expected {original_dim}, got {len(payload.replace)}",
                        id=payload.id,
                        expected=original_dim,
                        found=len(payload.replace),
                    )
                    analysis_error_code_total.labels(code=ErrorCode.DIMENSION_MISMATCH.value).inc()
                    from src.utils.analysis_metrics import vector_dimension_rejections_total
                    vector_dimension_rejections_total.labels(reason="dimension_mismatch_replace").inc()
                    raise HTTPException(status_code=409, detail=err)
                return VectorUpdateResponse(
                    id=payload.id,
                    status="dimension_mismatch",
                    dimension=original_dim,
                    error={"code": ErrorCode.DIMENSION_MISMATCH.value, "expected": original_dim, "found": len(payload.replace), "id": payload.id},
                )
            _VECTOR_STORE[payload.id] = [float(x) for x in payload.replace]
        elif payload.append is not None:
            if enforce and original_dim != 0:
                new_dim = original_dim + len(payload.append)
                if new_dim != original_dim:
                    err = build_error(
                        ErrorCode.DIMENSION_MISMATCH,
                        stage="vector_update",
                        message=f"Append changes dimension {original_dim}->{new_dim}",
                        id=payload.id,
                        expected=original_dim,
                        found=new_dim,
                    )
                    analysis_error_code_total.labels(code=ErrorCode.DIMENSION_MISMATCH.value).inc()
                    from src.utils.analysis_metrics import vector_dimension_rejections_total
                    vector_dimension_rejections_total.labels(reason="dimension_mismatch_append").inc()
                    raise HTTPException(status_code=409, detail=err)
            _VECTOR_STORE[payload.id] = vec + [float(float(x)) for x in payload.append]
        meta = _VECTOR_META.get(payload.id, {})
        if payload.material is not None:
            meta["material"] = payload.material
        if payload.complexity is not None:
            meta["complexity"] = payload.complexity
        if payload.format is not None:
            meta["format"] = payload.format
        _VECTOR_META[payload.id] = meta
        return VectorUpdateResponse(
            id=payload.id,
            status="updated",
            dimension=len(_VECTOR_STORE[payload.id]),
            feature_version=_VECTOR_META.get(payload.id, {}).get("feature_version"),
        )
    except HTTPException:
        raise
    except Exception as e:
        err = build_error(ErrorCode.INTERNAL_ERROR, stage="vector_update", message=str(e), id=payload.id)
        analysis_error_code_total.labels(code=ErrorCode.INTERNAL_ERROR.value).inc()
        return VectorUpdateResponse(id=payload.id, status="error", error=err)
class VectorMigrateItem(BaseModel):
    id: str
    status: str
    from_version: str | None = None
    to_version: str | None = None
    dimension_before: int | None = None
    dimension_after: int | None = None
    error: str | None = None


class VectorMigrateRequest(BaseModel):
    ids: list[str] = Field(description="需要迁移的向量ID列表")
    to_version: str = Field(description="目标特征版本")
    dry_run: bool = Field(default=False, description="是否为试运行 (不真正写入)")


class VectorMigrateResponse(BaseModel):
    total: int
    migrated: int
    skipped: int
    items: list[VectorMigrateItem]
    migration_id: str | None = Field(default=None, description="迁移批次ID")
    started_at: datetime | None = None
    finished_at: datetime | None = None
    dry_run_total: int | None = None


class VectorMigrationStatusResponse(BaseModel):
    last_migration_id: str | None = None
    last_started_at: datetime | None = None
    last_finished_at: datetime | None = None
    last_total: int | None = None
    last_migrated: int | None = None
    last_skipped: int | None = None
    pending_vectors: int | None = None
    feature_versions: Dict[str, int] | None = None
    history: list[Dict[str, Any]] | None = None


class VectorMigrationSummaryResponse(BaseModel):
    counts: Dict[str, int]
    total_migrations: int
    history_size: int
    statuses: list[str]


@router.post("/vectors/migrate", response_model=VectorMigrateResponse)
async def migrate_vectors(payload: VectorMigrateRequest, api_key: str = Depends(get_api_key)):
    from src.core.similarity import _VECTOR_STORE, _VECTOR_META  # type: ignore
    from src.core.feature_extractor import FeatureExtractor
    from src.utils.cache import get_cached_result
    import uuid, time, os
    # Validate target version early
    allowed_versions = {"v1", "v2", "v3", "v4"}
    if payload.to_version not in allowed_versions:
        from src.utils.analysis_metrics import analysis_error_code_total
        err = build_error(
            ErrorCode.INPUT_VALIDATION_FAILED,
            stage="vector_migrate",
            message="Unsupported target feature version",
            to_version=payload.to_version,
            allowed=list(sorted(allowed_versions)),
        )
        analysis_error_code_total.labels(code=ErrorCode.INPUT_VALIDATION_FAILED.value).inc()
        raise HTTPException(status_code=422, detail=err)
    migration_id = str(uuid.uuid4())
    started_at = datetime.utcnow()
    target_version = payload.to_version
    items: list[VectorMigrateItem] = []
    migrated = 0
    skipped = 0
    dry_run_total = 0
    # FeatureExtractor expects 'feature_version' parameter; pass target_version explicitly.
    extractor = FeatureExtractor(feature_version=target_version)
    from src.utils.analysis_metrics import vector_migrate_total
    for vid in payload.ids:
        if vid not in _VECTOR_STORE:
            items.append(VectorMigrateItem(id=vid, status="not_found", error="not_found"))
            skipped += 1
            vector_migrate_total.labels(status="not_found").inc()
            continue
        meta = _VECTOR_META.get(vid, {})
        from_version = meta.get("feature_version", "v1")
        if from_version == target_version:
            items.append(VectorMigrateItem(id=vid, status="skipped", from_version=from_version, to_version=target_version))
            skipped += 1
            vector_migrate_total.labels(status="skipped").inc()
            continue
        vec = _VECTOR_STORE[vid]
        dimension_before = len(vec)
        try:
            new_features = extractor.upgrade_vector(vec)
            dimension_after = len(new_features)
            if payload.dry_run:
                items.append(VectorMigrateItem(id=vid, status="dry_run", from_version=from_version, to_version=target_version, dimension_before=dimension_before, dimension_after=dimension_after))
                dry_run_total += 1
                vector_migrate_total.labels(status="dry_run").inc()
            else:
                _VECTOR_STORE[vid] = new_features
                meta["feature_version"] = target_version
                _VECTOR_META[vid] = meta
                # Track downgrade separately if target lower than source
                if (from_version, target_version) in {("v4", "v3"), ("v4", "v2"), ("v4", "v1"), ("v3", "v2"), ("v3", "v1"), ("v2", "v1")}:
                    items.append(VectorMigrateItem(id=vid, status="downgraded", from_version=from_version, to_version=target_version, dimension_before=dimension_before, dimension_after=dimension_after))
                    vector_migrate_total.labels(status="downgraded").inc()
                else:
                    migrated += 1
                    items.append(VectorMigrateItem(id=vid, status="migrated", from_version=from_version, to_version=target_version, dimension_before=dimension_before, dimension_after=dimension_after))
                    vector_migrate_total.labels(status="migrated").inc()
        except Exception as e:
            items.append(VectorMigrateItem(id=vid, status="error", error=str(e)))
            skipped += 1
            vector_migrate_total.labels(status="error").inc()
    finished_at = datetime.utcnow()
    # store simple history (ring buffer)
    history = globals().setdefault("_VECTOR_MIGRATION_HISTORY", [])
    history.append(
        {
            "migration_id": migration_id,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "total": len(payload.ids),
            "migrated": migrated,
            "skipped": skipped,
            "dry_run_total": dry_run_total,
            "counts": {
                "migrated": migrated,
                "skipped": skipped,
                "dry_run": dry_run_total,
                "downgraded": sum(1 for x in items if x.status == "downgraded"),
                "error": sum(1 for x in items if x.status == "error"),
                "not_found": sum(1 for x in items if x.status == "not_found"),
            },
        }
    )
    if len(history) > 10:
        history.pop(0)
    return VectorMigrateResponse(
        total=len(payload.ids),
        migrated=migrated,
        skipped=skipped,
        items=items,
        migration_id=migration_id,
        started_at=started_at,
        finished_at=finished_at,
        dry_run_total=dry_run_total if payload.dry_run else None,
    )


@router.get("/vectors/migrate/status", response_model=VectorMigrationStatusResponse)
async def migrate_status(api_key: str = Depends(get_api_key)):
    from src.core.similarity import _VECTOR_META, _VECTOR_STORE  # type: ignore
    versions: Dict[str, int] = {}
    for vid, meta in _VECTOR_META.items():
        if vid not in _VECTOR_STORE:
            continue
        ver = meta.get("feature_version", "unknown")
        versions[ver] = versions.get(ver, 0) + 1
    history = globals().get("_VECTOR_MIGRATION_HISTORY", [])
    last = history[-1] if history else None
    return VectorMigrationStatusResponse(
        last_migration_id=last.get("migration_id") if last else None,
        last_started_at=datetime.fromisoformat(last.get("started_at")) if last else None,
        last_finished_at=datetime.fromisoformat(last.get("finished_at")) if last else None,
        last_total=last.get("total") if last else None,
        last_migrated=last.get("migrated") if last else None,
        last_skipped=last.get("skipped") if last else None,
        pending_vectors=None,
        feature_versions=versions,
        history=history,
    )


@router.get("/vectors/migrate/summary", response_model=VectorMigrationSummaryResponse)
async def migrate_summary(api_key: str = Depends(get_api_key)):
    history = globals().get("_VECTOR_MIGRATION_HISTORY", [])
    aggregate: Dict[str, int] = {}
    for entry in history:
        counts = entry.get("counts", {})
        for k, v in counts.items():
            aggregate[k] = aggregate.get(k, 0) + int(v)
    total_migrations = sum(aggregate.values())
    statuses = sorted(aggregate.keys())
    return VectorMigrationSummaryResponse(
        counts=aggregate,
        total_migrations=total_migrations,
        history_size=len(history),
        statuses=statuses,
    )


class BatchSimilarityRequest(BaseModel):
    """批量相似度查询请求"""
    ids: list[str] = Field(description="需要查询的向量ID列表")
    top_k: int = Field(default=5, ge=1, le=50, description="每个向量返回的最相似结果数量")
    material: str | None = Field(default=None, description="过滤材料类型")
    complexity: str | None = Field(default=None, description="过滤复杂度")
    format: str | None = Field(default=None, description="过滤CAD格式")
    min_score: float | None = Field(default=None, ge=0.0, le=1.0, description="最小相似度分数阈值")


class BatchSimilarityItem(BaseModel):
    """单个向量的相似度查询结果"""
    id: str
    status: str  # success|not_found|error
    similar: list[Dict[str, Any]] = Field(default_factory=list)
    error: Dict[str, Any] | None = None


class BatchSimilarityResponse(BaseModel):
    """批量相似度查询响应"""
    total: int
    successful: int
    failed: int
    items: list[BatchSimilarityItem]
    batch_id: str | None = None
    duration_ms: float | None = None
    fallback: bool | None = Field(default=None, description="是否使用了内存降级 (Faiss不可用时)")


@router.post("/vectors/similarity/batch", response_model=BatchSimilarityResponse)
async def batch_similarity(payload: BatchSimilarityRequest, api_key: str = Depends(get_api_key)):
    """批量相似度查询 - 支持多个向量ID并行查询相似向量

    功能特性:
    - 支持批量查询多个向量的相似度
    - 可选过滤条件: material, complexity, format
    - 可设置最小相似度分数阈值
    - 自动记录批量查询延迟指标

    Args:
        payload: 批量查询请求
        api_key: API密钥

    Returns:
        批量查询结果，包含每个向量的相似度列表
    """
    import time
    import uuid
    from src.core.similarity import _VECTOR_STORE, _VECTOR_META, get_vector_store
    from src.utils.analysis_metrics import vector_query_batch_latency_seconds, vector_query_backend_total

    batch_id = str(uuid.uuid4())
    # Enforce batch size cap from env or default 200
    import os
    max_batch = int(os.getenv("BATCH_SIMILARITY_MAX_IDS", "200"))
    if len(payload.ids) > max_batch:
        from src.utils.analysis_metrics import analysis_rejections_total, analysis_error_code_total
        err = build_error(
            ErrorCode.INPUT_VALIDATION_FAILED,
            stage="batch_similarity",
            message="Batch size exceeds limit",
            batch_size=len(payload.ids),
            max_batch=max_batch,
        )
        analysis_rejections_total.labels(reason="batch_too_large").inc()
        analysis_error_code_total.labels(code=ErrorCode.INPUT_VALIDATION_FAILED.value).inc()
        raise HTTPException(status_code=422, detail=err)
    start_time = time.time()

    # Determine batch size range for metrics
    batch_size = len(payload.ids)
    if batch_size <= 5:
        size_range = "small"
    elif batch_size <= 20:
        size_range = "medium"
    else:
        size_range = "large"

    items: list[BatchSimilarityItem] = []
    successful = 0
    failed = 0

    # Get vector store using factory (handles backend selection and fallback)
    store = get_vector_store()

    # Detect if fallback occurred (Faiss unavailable -> memory)
    is_fallback = False
    expected_backend = os.getenv("VECTOR_STORE_BACKEND", "memory")
    if expected_backend == "faiss":
        from src.core.similarity import InMemoryVectorStore
        if isinstance(store, InMemoryVectorStore):
            is_fallback = True
            # Record fallback metric
            try:
                vector_query_backend_total.labels(backend="memory_fallback").inc()
            except Exception:
                pass

    # Process each vector ID
    for vid in payload.ids:
        if vid not in _VECTOR_STORE:
            items.append(BatchSimilarityItem(
                id=vid,
                status="not_found",
                error=build_error(ErrorCode.DATA_NOT_FOUND, stage="batch_similarity", message="Vector not found", id=vid)
            ))
            failed += 1
            continue

        try:
            # Get the vector
            vec = _VECTOR_STORE[vid]

            # Query for similar vectors
            results = store.query(vec, top_k=payload.top_k + 1)  # +1 to exclude self

            # Filter results
            similar: list[Dict[str, Any]] = []
            for result_id, score in results:
                # Skip self
                if result_id == vid:
                    continue

                # Apply score threshold
                if payload.min_score is not None and score < payload.min_score:
                    continue

                # Get metadata for filtering
                meta = _VECTOR_META.get(result_id, {})

                # Apply filters
                if payload.material and meta.get("material") != payload.material:
                    continue
                if payload.complexity and meta.get("complexity") != payload.complexity:
                    continue
                if payload.format and meta.get("format") != payload.format:
                    continue

                similar.append({
                    "id": result_id,
                    "score": round(score, 4),
                    "material": meta.get("material"),
                    "complexity": meta.get("complexity"),
                    "format": meta.get("format"),
                    "dimension": len(_VECTOR_STORE.get(result_id, []))
                })

                # Limit to top_k after filtering
                if len(similar) >= payload.top_k:
                    break

            items.append(BatchSimilarityItem(
                id=vid,
                status="success",
                similar=similar
            ))
            successful += 1

        except Exception as e:
            items.append(BatchSimilarityItem(
                id=vid,
                status="error",
                error=build_error(ErrorCode.INTERNAL_ERROR, stage="batch_similarity", message="Query failed", id=vid, detail=str(e))
            ))
            failed += 1

    # Record metrics
    duration = time.time() - start_time
    vector_query_batch_latency_seconds.labels(batch_size_range=size_range).observe(duration)

    # Record empty result metric if every successful item has 0 similar entries
    if successful > 0 and all((not it.similar) for it in items if it.status == "success"):
        try:
            from src.utils.analysis_metrics import analysis_rejections_total
            analysis_rejections_total.labels(reason="batch_empty_results").inc()
        except Exception:
            pass

    return BatchSimilarityResponse(
        total=len(payload.ids),
        successful=successful,
        failed=failed,
        items=items,
        batch_id=batch_id,
        duration_ms=round(duration * 1000, 2),
        fallback=is_fallback if is_fallback else None
    )
