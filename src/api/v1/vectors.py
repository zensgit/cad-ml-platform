"""Vector management endpoints extracted from analyze.py for modularity."""

from __future__ import annotations

from datetime import datetime
import json
import os
from typing import Any, Dict, Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.dependencies import get_api_key, get_admin_token
from src.core.errors_extended import build_error, ErrorCode
from src.utils.cache import get_client

router = APIRouter()


class VectorDeleteRequest(BaseModel):
    id: str = Field(description="要删除的向量分析ID")


class VectorDeleteResponse(BaseModel):
    id: str
    status: str
    error: Optional[Dict[str, Any]] = None


class VectorListItem(BaseModel):
    id: str
    dimension: int
    material: Optional[str] = None
    complexity: Optional[str] = None
    format: Optional[str] = None


class VectorListResponse(BaseModel):
    total: int
    vectors: list[VectorListItem]


class VectorBackendReloadResponse(BaseModel):
    status: str
    backend: Optional[str] = None
    error: Optional[Dict[str, Any]] = None


@router.post("/delete", response_model=VectorDeleteResponse)
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
                    await client.delete(f"vector:{payload.id}")
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


@router.get("/", response_model=VectorListResponse)
async def list_vectors(
    source: str = Query(
        default="auto",
        description="Vector source: auto|memory|redis",
    ),
    offset: int = Query(default=0, ge=0, description="结果偏移用于分页"),
    limit: int = Query(default=50, ge=1, description="返回数量上限"),
    api_key: str = Depends(get_api_key),
):
    from src.core.similarity import _VECTOR_STORE, _VECTOR_META, _BACKEND  # type: ignore

    allowed_sources = {"auto", "memory", "redis"}
    if source not in allowed_sources:
        err = build_error(
            ErrorCode.INPUT_VALIDATION_FAILED,
            stage="vector_list",
            message="Invalid source",
            source=source,
            allowed=list(sorted(allowed_sources)),
        )
        raise HTTPException(status_code=400, detail=err)

    max_limit = int(os.getenv("VECTOR_LIST_LIMIT", "200"))
    limit = min(limit, max_limit)
    scan_limit = int(os.getenv("VECTOR_LIST_SCAN_LIMIT", "5000"))
    resolved = _resolve_list_source(source, _BACKEND)
    if resolved == "redis":
        client = get_client()
        if client is not None:
            return await _list_vectors_redis(client, offset, limit, scan_limit)
    return _list_vectors_memory(_VECTOR_STORE, _VECTOR_META, offset, limit)


def _resolve_list_source(source: str, backend: str) -> str:
    if source == "auto":
        return "redis" if backend == "redis" else "memory"
    return source


def _list_vectors_memory(
    vector_store: Dict[str, list[float]],
    vector_meta: Dict[str, Dict[str, str]],
    offset: int,
    limit: int,
) -> VectorListResponse:
    items: list[VectorListItem] = []
    entries = list(vector_store.items())
    for vid, vec in entries[offset : offset + limit]:
        meta = vector_meta.get(vid, {})
        items.append(
            VectorListItem(
                id=vid,
                dimension=len(vec),
                material=meta.get("material"),
                complexity=meta.get("complexity"),
                format=meta.get("format"),
            )
        )
    return VectorListResponse(total=len(vector_store), vectors=items)


async def _list_vectors_redis(
    client,
    offset: int,
    limit: int,
    scan_limit: int,
) -> VectorListResponse:
    items: list[VectorListItem] = []
    total = 0
    scanned = 0
    cursor = 0
    while True:
        cursor, batch = await client.scan(cursor=cursor, match="vector:*", count=500)  # type: ignore[attr-defined]
        for key in batch:
            scanned += 1
            if scan_limit > 0 and scanned > scan_limit:
                cursor = 0
                break
            data = await client.hgetall(key)  # type: ignore[attr-defined]
            raw_vec = data.get("v") or data.get(b"v")
            if not raw_vec:
                continue
            total += 1
            if total <= offset:
                continue
            raw_meta = data.get("m") or data.get(b"m")
            meta: Dict[str, Any] = {}
            if raw_meta:
                try:
                    meta = json.loads(raw_meta)
                except Exception:
                    meta = {}
            vec_dim = len([p for p in str(raw_vec).split(",") if p])
            key_str = key.decode() if isinstance(key, (bytes, bytearray)) else str(key)
            vid = key_str.split("vector:", 1)[1] if "vector:" in key_str else key_str
            items.append(
                VectorListItem(
                    id=vid,
                    dimension=vec_dim,
                    material=meta.get("material"),
                    complexity=meta.get("complexity"),
                    format=meta.get("format"),
                )
            )
            if len(items) >= limit:
                break
        if len(items) >= limit:
            break
        if cursor == 0:
            break
    return VectorListResponse(total=total, vectors=items)


__all__ = ["router"]
class VectorUpdateRequest(BaseModel):
    id: str = Field(description="要更新的向量分析ID")
    replace: Optional[list[float]] = Field(default=None, description="新的向量 (维度需与原向量一致)")
    append: Optional[list[float]] = Field(default=None, description="追加的向量片段 (若提供 replace 则忽略)")
    material: Optional[str] = Field(default=None, description="更新材料元数据")
    complexity: Optional[str] = Field(default=None, description="更新复杂度元数据")
    format: Optional[str] = Field(default=None, description="更新格式元数据")


class VectorUpdateResponse(BaseModel):
    id: str
    status: str
    dimension: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
    feature_version: Optional[str] = None


@router.post("/update", response_model=VectorUpdateResponse)
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
    from_version: Optional[str] = None
    to_version: Optional[str] = None
    dimension_before: Optional[int] = None
    dimension_after: Optional[int] = None
    error: Optional[str] = None


class VectorMigrateRequest(BaseModel):
    ids: list[str] = Field(description="需要迁移的向量ID列表")
    to_version: str = Field(description="目标特征版本")
    dry_run: bool = Field(default=False, description="是否为试运行 (不真正写入)")


class VectorMigrateResponse(BaseModel):
    total: int
    migrated: int
    skipped: int
    items: List[VectorMigrateItem]
    migration_id: Optional[str] = Field(default=None, description="迁移批次ID")
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    dry_run_total: Optional[int] = None


class VectorMigrationStatusResponse(BaseModel):
    last_migration_id: Optional[str] = None
    last_started_at: Optional[datetime] = None
    last_finished_at: Optional[datetime] = None
    last_total: Optional[int] = None
    last_migrated: Optional[int] = None
    last_skipped: Optional[int] = None
    pending_vectors: Optional[int] = None
    feature_versions: Optional[Dict[str, int]] = None
    history: Optional[List[Dict[str, Any]]] = None


class VectorMigrationSummaryResponse(BaseModel):
    counts: Dict[str, int]
    total_migrations: int
    history_size: int
    statuses: List[str]


class VectorMigrationPreviewResponse(BaseModel):
    """迁移预览响应 - 不执行实际写入"""
    total_vectors: int = Field(description="总向量数量")
    by_version: Dict[str, int] = Field(description="各版本向量数量统计")
    preview_items: List[VectorMigrateItem] = Field(description="预览前N个向量的迁移结果")
    estimated_dimension_changes: Dict[str, int] = Field(description="预计维度变化统计 (positive/negative/zero)")
    migration_feasible: bool = Field(description="迁移是否可行")
    warnings: List[str] = Field(default_factory=list, description="潜在问题警告")
    avg_delta: Optional[float] = Field(default=None, description="采样维度变化平均值")
    median_delta: Optional[float] = Field(default=None, description="采样维度变化中位数")


@router.get("/migrate/preview", response_model=VectorMigrationPreviewResponse)
async def preview_migration(
    to_version: str,
    limit: int = 10,
    api_key: str = Depends(get_api_key)
):
    """
    预览向量特征迁移 - 不执行写入操作

    Args:
        to_version: 目标特征版本 (v1/v2/v3/v4)
        limit: 预览样本数量 (默认10, 最大100)
        api_key: API密钥

    Returns:
        迁移预览信息，包含版本分布、样本预览、维度变化预估等
    """
    from src.core.similarity import _VECTOR_STORE, _VECTOR_META
    from src.core.feature_extractor import FeatureExtractor

    # Validate target version
    allowed_versions = {"v1", "v2", "v3", "v4"}
    if to_version not in allowed_versions:
        err = build_error(
            ErrorCode.INPUT_VALIDATION_FAILED,
            stage="migration_preview",
            message="Unsupported target feature version",
            to_version=to_version,
            allowed=list(allowed_versions)
        )
        from src.utils.analysis_metrics import analysis_error_code_total
        analysis_error_code_total.labels(code=ErrorCode.INPUT_VALIDATION_FAILED.value).inc()
        raise HTTPException(status_code=422, detail=err)

    # Enforce limit cap
    limit = min(limit, 100)

    # Get extractor
    extractor = FeatureExtractor()

    # Collect version distribution
    by_version: Dict[str, int] = {}
    total_vectors = len(_VECTOR_STORE)

    for vid in _VECTOR_STORE.keys():
        meta = _VECTOR_META.get(vid, {})
        current_version = meta.get("feature_version", "v1")
        by_version[current_version] = by_version.get(current_version, 0) + 1

    # Preview sample vectors
    preview_items: list[VectorMigrateItem] = []
    dimension_changes = {"positive": 0, "negative": 0, "zero": 0}
    deltas: list[int] = []
    warnings: list[str] = []
    sampled = 0

    for vid in list(_VECTOR_STORE.keys())[:limit]:
        meta = _VECTOR_META.get(vid, {})
        from_version = meta.get("feature_version", "v1")
        vec = _VECTOR_STORE[vid]
        dimension_before = len(vec)

        if from_version == to_version:
            preview_items.append(VectorMigrateItem(
                id=vid,
                status="skipped",
                from_version=from_version,
                to_version=to_version,
                dimension_before=dimension_before,
                dimension_after=dimension_before
            ))
            dimension_changes["zero"] += 1
            sampled += 1
            continue

        try:
            new_features = extractor.upgrade_vector(vec, current_version=from_version)
            dimension_after = len(new_features)
            dimension_delta = dimension_after - dimension_before
            deltas.append(dimension_delta)

            if dimension_delta > 0:
                dimension_changes["positive"] += 1
            elif dimension_delta < 0:
                dimension_changes["negative"] += 1
            else:
                dimension_changes["zero"] += 1

            # Detect downgrade
            is_downgrade = (from_version, to_version) in {
                ("v4", "v3"), ("v4", "v2"), ("v4", "v1"),
                ("v3", "v2"), ("v3", "v1"),
                ("v2", "v1")
            }

            preview_items.append(VectorMigrateItem(
                id=vid,
                status="downgrade_preview" if is_downgrade else "upgrade_preview",
                from_version=from_version,
                to_version=to_version,
                dimension_before=dimension_before,
                dimension_after=dimension_after
            ))
            sampled += 1

        except Exception as e:
            preview_items.append(VectorMigrateItem(
                id=vid,
                status="error_preview",
                error=str(e)
            ))
            warnings.append(f"Vector {vid} migration would fail: {str(e)}")
            sampled += 1

    # Check migration feasibility
    migration_feasible = True
    total_sampled = max(sampled, 1)
    if dimension_changes["negative"] > total_sampled * 0.5:
        migration_feasible = False
        warnings.append("More than 50% of sampled vectors would lose dimensions")

    # Compute stats
    avg_delta: Optional[float] = None
    median_delta: Optional[float] = None
    if deltas:
        avg_delta = float(sum(deltas) / len(deltas))
        try:
            import statistics
            median_delta = float(statistics.median(deltas))
        except Exception:
            median_delta = float(deltas[len(deltas)//2])
    # Warning heuristics
    if median_delta is not None and median_delta < -5:
        warnings.append("large_negative_skew")
    if avg_delta is not None and abs(avg_delta) > 10:
        warnings.append("growth_spike")

    if len(warnings) > limit * 0.3:
        warnings.append(f"High error rate in preview: {len(warnings)}/{limit}")

    return VectorMigrationPreviewResponse(
        total_vectors=total_vectors,
        by_version=by_version,
        preview_items=preview_items,
        estimated_dimension_changes=dimension_changes,
        migration_feasible=migration_feasible,
        warnings=warnings,
        avg_delta=avg_delta,
        median_delta=median_delta,
    )


@router.post("/migrate", response_model=VectorMigrateResponse)
async def migrate_vectors(payload: VectorMigrateRequest, api_key: str = Depends(get_api_key)):
    from src.core.similarity import _VECTOR_STORE, _VECTOR_META  # type: ignore
    from src.core.feature_extractor import FeatureExtractor
    import uuid
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
    items: List[VectorMigrateItem] = []
    migrated = 0
    skipped = 0
    dry_run_total = 0
    # FeatureExtractor expects 'feature_version' parameter; pass target_version explicitly.
    extractor = FeatureExtractor(feature_version=target_version)
    from src.utils.analysis_metrics import vector_migrate_total, vector_migrate_dimension_delta
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
            new_features = extractor.upgrade_vector(vec, current_version=from_version)
            dimension_after = len(new_features)
            dimension_delta = dimension_after - dimension_before
            # Record dimension delta for observability
            vector_migrate_dimension_delta.observe(dimension_delta)
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


@router.get("/migrate/status", response_model=VectorMigrationStatusResponse)
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


@router.get("/migrate/summary", response_model=VectorMigrationSummaryResponse)
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


class VectorMigrationTrendsResponse(BaseModel):
    """迁移趋势响应"""
    total_migrations: int = Field(description="窗口内总迁移数量")
    success_rate: float = Field(description="迁移成功率 (0.0-1.0)")
    v4_adoption_rate: float = Field(description="v4版本采用率")
    avg_dimension_delta: float = Field(description="平均维度变化")
    window_hours: int = Field(description="统计窗口小时数")
    version_distribution: Dict[str, int] = Field(description="当前版本分布")
    migration_velocity: float = Field(description="每小时迁移数量")
    downgrade_rate: float = Field(description="降级比例")
    error_rate: float = Field(description="错误比例")
    time_range: Dict[str, Optional[str]] = Field(description="统计时间范围")


@router.get("/migrate/trends", response_model=VectorMigrationTrendsResponse)
async def migrate_trends(
    window_hours: int = 24,
    api_key: str = Depends(get_api_key)
):
    """
    获取迁移趋势统计

    Args:
        window_hours: 统计窗口小时数 (默认24小时)
        api_key: API密钥

    Returns:
        迁移趋势统计，包含成功率、v4采用率、维度变化等
    """
    from src.core.similarity import _VECTOR_STORE, _VECTOR_META

    history = globals().get("_VECTOR_MIGRATION_HISTORY", [])

    # Filter history by time window
    from datetime import datetime, timedelta
    cutoff = datetime.utcnow() - timedelta(hours=window_hours)
    filtered_history = []
    for entry in history:
        try:
            entry_time = datetime.fromisoformat(entry.get("started_at", ""))
            if entry_time >= cutoff:
                filtered_history.append(entry)
        except (ValueError, TypeError):
            # Include entries without valid timestamp
            filtered_history.append(entry)

    # Calculate aggregated stats
    total_migrations = 0
    total_migrated = 0
    total_skipped = 0
    total_downgraded = 0
    total_errors = 0

    for entry in filtered_history:
        counts = entry.get("counts", {})
        total_migrations += entry.get("total", 0)
        total_migrated += counts.get("migrated", 0)
        total_skipped += counts.get("skipped", 0)
        total_downgraded += counts.get("downgraded", 0)
        total_errors += counts.get("error", 0) + counts.get("not_found", 0)

    # Calculate rates
    attempted = total_migrated + total_downgraded + total_errors
    success_rate = (total_migrated + total_downgraded) / max(attempted, 1)
    downgrade_rate = total_downgraded / max(attempted, 1)
    error_rate = total_errors / max(attempted, 1)

    # Calculate v4 adoption rate from current vectors
    version_distribution: Dict[str, int] = {}
    total_vectors = 0
    for vid in _VECTOR_STORE.keys():
        meta = _VECTOR_META.get(vid, {})
        version = meta.get("feature_version", "v1")
        version_distribution[version] = version_distribution.get(version, 0) + 1
        total_vectors += 1

    v4_count = version_distribution.get("v4", 0)
    v4_adoption_rate = v4_count / max(total_vectors, 1)

    # Calculate average dimension delta (estimate from history)
    avg_dimension_delta = 0.0
    # For now, estimate based on version changes (v3->v4 adds 2 dimensions)
    if total_migrated > 0:
        # Rough estimate: upgrade to v4 adds 2, downgrade removes dimensions
        avg_dimension_delta = (total_migrated * 2 - total_downgraded * 2) / max(total_migrated + total_downgraded, 1)

    # Migration velocity (per hour)
    migration_velocity = total_migrations / max(window_hours, 1)

    # Time range
    time_range = {
        "start": (datetime.utcnow() - timedelta(hours=window_hours)).isoformat() if window_hours > 0 else None,
        "end": datetime.utcnow().isoformat()
    }

    return VectorMigrationTrendsResponse(
        total_migrations=total_migrations,
        success_rate=round(success_rate, 4),
        v4_adoption_rate=round(v4_adoption_rate, 4),
        avg_dimension_delta=round(avg_dimension_delta, 2),
        window_hours=window_hours,
        version_distribution=version_distribution,
        migration_velocity=round(migration_velocity, 2),
        downgrade_rate=round(downgrade_rate, 4),
        error_rate=round(error_rate, 4),
        time_range=time_range,
    )


class BatchSimilarityRequest(BaseModel):
    """批量相似度查询请求"""
    ids: list[str] = Field(description="需要查询的向量ID列表")
    top_k: int = Field(default=5, ge=1, le=50, description="每个向量返回的最相似结果数量")
    material: Optional[str] = Field(default=None, description="过滤材料类型")
    complexity: Optional[str] = Field(default=None, description="过滤复杂度")
    format: Optional[str] = Field(default=None, description="过滤CAD格式")
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="最小相似度分数阈值")


class BatchSimilarityItem(BaseModel):
    """单个向量的相似度查询结果"""
    id: str
    status: str  # success|not_found|error
    similar: list[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[Dict[str, Any]] = None


class BatchSimilarityResponse(BaseModel):
    """批量相似度查询响应"""
    total: int
    successful: int
    failed: int
    items: list[BatchSimilarityItem]
    batch_id: Optional[str] = None
    duration_ms: Optional[float] = None
    fallback: Optional[bool] = Field(default=None, description="是否使用了内存降级 (Faiss不可用时)")
    degraded: bool = Field(default=False, description="向量存储是否处于降级模式")


@router.post("/similarity/batch", response_model=BatchSimilarityResponse)
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
    from src.core.similarity import _VECTOR_STORE, _VECTOR_META, get_vector_store, get_degraded_mode_info
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

    # Get degraded mode status
    degraded_info = get_degraded_mode_info()

    return BatchSimilarityResponse(
        total=len(payload.ids),
        successful=successful,
        failed=failed,
        items=items,
        batch_id=batch_id,
        duration_ms=round(duration * 1000, 2),
        fallback=is_fallback if is_fallback else None,
        degraded=degraded_info["degraded"]
    )


@router.post("/backend/reload", response_model=VectorBackendReloadResponse)
async def reload_vector_backend(
    backend: Optional[str] = None,
    api_key: str = Depends(get_api_key),
    admin_token: str = Depends(get_admin_token),
):
    """Reload vector store backend (admin token required)."""
    from src.core.similarity import reload_vector_store_backend
    from src.utils.analysis_metrics import vector_store_reload_total
    import os

    allowed = {"memory", "faiss", "redis"}
    if backend is not None:
        backend = backend.strip().lower()
        if backend not in allowed:
            vector_store_reload_total.labels(status="error").inc()
            err = build_error(
                ErrorCode.INPUT_VALIDATION_FAILED,
                stage="backend_reload",
                message="Unsupported vector backend",
                backend=backend,
                supported=sorted(allowed),
            )
            raise HTTPException(status_code=400, detail=err)
        os.environ["VECTOR_STORE_BACKEND"] = backend

    effective_backend = backend or os.getenv("VECTOR_STORE_BACKEND", "memory")
    ok = reload_vector_store_backend()
    vector_store_reload_total.labels(status="success" if ok else "error").inc()
    if not ok:
        err = build_error(
            ErrorCode.INTERNAL_ERROR,
            stage="backend_reload",
            message="Vector store backend reload failed",
            backend=effective_backend,
        )
        raise HTTPException(status_code=500, detail=err)
    return VectorBackendReloadResponse(status="ok", backend=effective_backend)
