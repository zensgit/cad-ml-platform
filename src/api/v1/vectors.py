"""Vector management endpoints extracted from analyze.py for modularity."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, Field

from src.api.dependencies import get_admin_token, get_api_key
from src.api.v1.vector_crud_models import (
    VectorDeleteRequest,
    VectorDeleteResponse,
    VectorRegisterRequest,
    VectorRegisterResponse,
    VectorSearchRequest,
    VectorSearchResponse,
)
from src.api.v1.vector_list_models import VectorListItem, VectorListResponse
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
from src.core.vector_layouts import (
    VECTOR_LAYOUT_BASE,
    VECTOR_LAYOUT_L3,
    VECTOR_LAYOUT_LEGACY,
    layout_has_l3,
)
from src.utils.cache import get_client

router = APIRouter()


async def _vector_reload_admin_token(
    x_admin_token: str = Header(default="", alias="X-Admin-Token"),
) -> str:
    """Admin token dependency that records auth failures for reload metrics."""
    from src.utils.analysis_metrics import vector_store_reload_total

    try:
        return await get_admin_token(x_admin_token)
    except HTTPException:
        try:
            vector_store_reload_total.labels(status="error", reason="auth_failed").inc()
        except Exception:
            pass
        raise

if TYPE_CHECKING:
    from src.core.feature_extractor import FeatureExtractor


class VectorBackendReloadResponse(BaseModel):
    status: str
    backend: Optional[str] = None
    error: Optional[Dict[str, Any]] = None


router.include_router(crud_router)
router.include_router(migration_read_router)
router.include_router(write_router)
router.include_router(list_router)


def _build_vector_filter_conditions(
    *,
    material_filter: Optional[str],
    complexity_filter: Optional[str],
    fine_part_type_filter: Optional[str],
    coarse_part_type_filter: Optional[str],
    decision_source_filter: Optional[str],
    is_coarse_label_filter: Optional[bool],
) -> Dict[str, Any]:
    conditions: Dict[str, Any] = {}
    if material_filter:
        conditions["material"] = material_filter
    if complexity_filter:
        conditions["complexity"] = complexity_filter
    if fine_part_type_filter:
        conditions["fine_part_type"] = fine_part_type_filter
    if coarse_part_type_filter:
        conditions["coarse_part_type"] = coarse_part_type_filter
    if decision_source_filter:
        conditions["decision_source"] = decision_source_filter
    if is_coarse_label_filter is not None:
        conditions["is_coarse_label"] = is_coarse_label_filter
    return conditions


def _build_vector_search_filter_conditions(payload: VectorSearchRequest) -> Dict[str, Any]:
    return _build_vector_filter_conditions(
        material_filter=payload.material_filter,
        complexity_filter=payload.complexity_filter,
        fine_part_type_filter=payload.fine_part_type_filter,
        coarse_part_type_filter=payload.coarse_part_type_filter,
        decision_source_filter=payload.decision_source_filter,
        is_coarse_label_filter=payload.is_coarse_label_filter,
    )


def _vector_item_payload(
    vector_id: str,
    dimension: int,
    meta: Dict[str, Any],
    label_contract: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "id": vector_id,
        "dimension": dimension,
        "material": meta.get("material"),
        "complexity": meta.get("complexity"),
        "format": meta.get("format"),
        "part_type": label_contract.get("part_type"),
        "fine_part_type": label_contract.get("fine_part_type"),
        "coarse_part_type": label_contract.get("coarse_part_type"),
        "decision_source": label_contract.get("decision_source"),
        "is_coarse_label": label_contract.get("is_coarse_label"),
    }
def _resolve_vector_migration_scan_limit(default: int = 5000) -> int:
    try:
        resolved = int(os.getenv("VECTOR_MIGRATION_SCAN_LIMIT", str(default)))
    except (TypeError, ValueError):
        resolved = default
    return max(int(resolved or 0), 1)


def _resolve_vector_migration_target_version(default: str = "v4") -> str:
    allowed_versions = {"v1", "v2", "v3", "v4"}
    raw = str(os.getenv("VECTOR_MIGRATION_TARGET_VERSION", default) or default).strip().lower()
    if raw in allowed_versions:
        return raw
    return default


def _build_vector_migration_readiness(
    version_distribution: Dict[str, int],
    *,
    total_vectors: int,
    distribution_complete: bool,
) -> Dict[str, Any]:
    target_version = _resolve_vector_migration_target_version()
    readiness: Dict[str, Any] = {
        "target_version": target_version,
        "target_version_vectors": None,
        "target_version_ratio": None,
        "pending_vectors": None,
        "migration_ready": False,
    }
    if not distribution_complete:
        return readiness

    target_vectors = int(version_distribution.get(target_version, 0))
    pending_vectors = max(int(total_vectors) - target_vectors, 0)
    readiness["target_version_vectors"] = target_vectors
    readiness["target_version_ratio"] = round(target_vectors / max(int(total_vectors), 1), 4)
    readiness["pending_vectors"] = pending_vectors
    readiness["migration_ready"] = pending_vectors == 0
    return readiness


async def _collect_qdrant_feature_versions(
    qdrant_store,
    *,
    scan_limit: int | None = None,
) -> tuple[Dict[str, int], int, int]:
    resolved_scan_limit = scan_limit
    if resolved_scan_limit is None:
        resolved_scan_limit = _resolve_vector_migration_scan_limit()
    resolved_scan_limit = max(int(resolved_scan_limit or 0), 1)

    total_available = int(await qdrant_store.count())
    versions: Dict[str, int] = {}
    scanned = 0
    offset = 0

    while scanned < min(total_available, resolved_scan_limit):
        batch_limit = min(200, resolved_scan_limit - scanned)
        items, _ = await qdrant_store.list_vectors(
            offset=offset,
            limit=batch_limit,
            with_vectors=False,
        )
        if not items:
            break
        for item in items:
            meta = item.metadata or {}
            version = str(meta.get("feature_version") or "unknown")
            versions[version] = versions.get(version, 0) + 1
        consumed = len(items)
        scanned += consumed
        offset += consumed

    return versions, total_available, scanned


async def _collect_qdrant_preview_samples(
    qdrant_store,
    *,
    limit: int,
) -> tuple[list[tuple[str, list[float], Dict[str, Any]]], int, Dict[str, int]]:
    total_available = int(await qdrant_store.count())
    items, _ = await qdrant_store.list_vectors(
        offset=0,
        limit=max(limit, 1),
        with_vectors=True,
    )
    samples: list[tuple[str, list[float], Dict[str, Any]]] = []
    by_version: Dict[str, int] = {}
    for item in items:
        meta = dict(item.metadata or {})
        version = str(meta.get("feature_version") or "v1")
        by_version[version] = by_version.get(version, 0) + 1
        samples.append((str(item.id), list(item.vector or []), meta))
    if total_available > len(items):
        offset = len(items)
        scan_limit = min(total_available, 5000)
        while offset < scan_limit:
            batch, _ = await qdrant_store.list_vectors(
                offset=offset,
                limit=min(200, scan_limit - offset),
                with_vectors=False,
            )
            if not batch:
                break
            for item in batch:
                meta = dict(item.metadata or {})
                version = str(meta.get("feature_version") or "v1")
                by_version[version] = by_version.get(version, 0) + 1
            offset += len(batch)
    return samples, total_available, by_version


def _matches_vector_label_filters(
    *,
    material_filter: Optional[str],
    complexity_filter: Optional[str],
    fine_part_type_filter: Optional[str],
    coarse_part_type_filter: Optional[str],
    decision_source_filter: Optional[str],
    is_coarse_label_filter: Optional[bool],
    meta: Dict[str, Any],
    label_contract: Dict[str, Any],
) -> bool:
    if material_filter and meta.get("material") != material_filter:
        return False
    if complexity_filter and meta.get("complexity") != complexity_filter:
        return False
    if fine_part_type_filter and label_contract.get("fine_part_type") != fine_part_type_filter:
        return False
    if (
        coarse_part_type_filter
        and label_contract.get("coarse_part_type") != coarse_part_type_filter
    ):
        return False
    if decision_source_filter and label_contract.get("decision_source") != decision_source_filter:
        return False
    if (
        is_coarse_label_filter is not None
        and label_contract.get("is_coarse_label") is not is_coarse_label_filter
    ):
        return False
    return True


def _matches_vector_search_filters(
    payload: VectorSearchRequest,
    meta: Dict[str, Any],
    label_contract: Dict[str, Any],
) -> bool:
    return _matches_vector_label_filters(
        material_filter=payload.material_filter,
        complexity_filter=payload.complexity_filter,
        fine_part_type_filter=payload.fine_part_type_filter,
        coarse_part_type_filter=payload.coarse_part_type_filter,
        decision_source_filter=payload.decision_source_filter,
        is_coarse_label_filter=payload.is_coarse_label_filter,
        meta=meta,
        label_contract=label_contract,
    )


def _resolve_list_source(source: str, backend: str) -> str:
    if source == "auto":
        if backend == "redis":
            return "redis"
        if backend == "qdrant":
            return "qdrant"
        return "memory"
    return source


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
    from src.core.similarity import extract_vector_label_contract

    items: list[VectorListItem] = []
    matched_total = 0
    entries = list(vector_store.items())
    for vid, vec in entries:
        meta = vector_meta.get(vid, {})
        label_contract = extract_vector_label_contract(meta)
        if not _matches_vector_label_filters(
            material_filter=material_filter,
            complexity_filter=complexity_filter,
            fine_part_type_filter=fine_part_type_filter,
            coarse_part_type_filter=coarse_part_type_filter,
            decision_source_filter=decision_source_filter,
            is_coarse_label_filter=is_coarse_label_filter,
            meta=meta,
            label_contract=label_contract,
        ):
            continue
        matched_total += 1
        if matched_total <= offset:
            continue
        if len(items) < limit:
            items.append(
                VectorListItem(
                    id=vid,
                    dimension=len(vec),
                    material=meta.get("material"),
                    complexity=meta.get("complexity"),
                    format=meta.get("format"),
                    part_type=label_contract.get("part_type"),
                    fine_part_type=label_contract.get("fine_part_type"),
                    coarse_part_type=label_contract.get("coarse_part_type"),
                    decision_source=label_contract.get("decision_source"),
                    is_coarse_label=label_contract.get("is_coarse_label"),
                )
            )
    return VectorListResponse(total=matched_total, vectors=items)


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

    items: list[VectorListItem] = []
    matched_total = 0
    scanned = 0
    cursor = 0
    while True:
        cursor, batch = await client.scan(  # type: ignore[attr-defined]
            cursor=cursor,
            match="vector:*",
            count=500,
        )
        for key in batch:
            scanned += 1
            if scan_limit > 0 and scanned > scan_limit:
                cursor = 0
                break
            data = await client.hgetall(key)  # type: ignore[attr-defined]
            raw_vec = data.get("v") or data.get(b"v")
            if not raw_vec:
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
            label_contract = extract_vector_label_contract(meta)
            if not _matches_vector_label_filters(
                material_filter=material_filter,
                complexity_filter=complexity_filter,
                fine_part_type_filter=fine_part_type_filter,
                coarse_part_type_filter=coarse_part_type_filter,
                decision_source_filter=decision_source_filter,
                is_coarse_label_filter=is_coarse_label_filter,
                meta=meta,
                label_contract=label_contract,
            ):
                continue
            matched_total += 1
            if matched_total <= offset:
                continue
            if len(items) < limit:
                items.append(
                    VectorListItem(
                        id=vid,
                        dimension=vec_dim,
                        material=meta.get("material"),
                        complexity=meta.get("complexity"),
                        format=meta.get("format"),
                        part_type=label_contract.get("part_type"),
                        fine_part_type=label_contract.get("fine_part_type"),
                        coarse_part_type=label_contract.get("coarse_part_type"),
                        decision_source=label_contract.get("decision_source"),
                        is_coarse_label=label_contract.get("is_coarse_label"),
                    )
                )
        if cursor == 0:
            break
    return VectorListResponse(total=matched_total, vectors=items)


def _coerce_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _prepare_vector_for_upgrade(
    extractor: "FeatureExtractor",
    vector: list[float],
    meta: Dict[str, Any],
    from_version: str,
) -> tuple[list[float], list[float], str]:
    layout = meta.get("vector_layout") or VECTOR_LAYOUT_BASE
    expected_len = extractor.expected_dim(from_version)
    l3_tail: list[float] = []
    base_vector = vector

    if layout_has_l3(layout):
        l3_dim = _coerce_int(meta.get("l3_3d_dim"))
        if l3_dim is None and len(vector) > expected_len:
            l3_dim = len(vector) - expected_len
        if not l3_dim or len(vector) < expected_len + l3_dim:
            raise ValueError("L3 layout length mismatch")
        base_vector = vector[: len(vector) - l3_dim]
        l3_tail = vector[-l3_dim:]

    if layout == VECTOR_LAYOUT_LEGACY:
        base_vector = extractor.reorder_legacy_vector(base_vector, from_version)

    return base_vector, l3_tail, layout


__all__ = [
    "router",
    "list_vectors",
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
    return await run_vector_batch_similarity(
        payload=payload,
        batch_item_cls=BatchSimilarityItem,
        batch_response_cls=BatchSimilarityResponse,
        error_code_cls=ErrorCode,
        build_error_fn=build_error,
        get_qdrant_store_fn=_get_qdrant_store_or_none,
        build_filter_conditions_fn=_build_vector_filter_conditions,
    )


@router.post("/backend/reload", response_model=VectorBackendReloadResponse)
async def reload_vector_backend(
    backend: Optional[str] = None,
    api_key: str = Depends(get_api_key),
    admin_token: str = Depends(_vector_reload_admin_token),
):
    """Reload vector store backend (admin token required)."""
    from src.core.similarity import reload_vector_store_backend
    from src.utils.analysis_metrics import vector_store_reload_total

    return await run_vector_backend_reload_pipeline(
        backend=backend,
        reload_backend_fn=reload_vector_store_backend,
        reload_metric=vector_store_reload_total,
        error_code_cls=ErrorCode,
        build_error_fn=build_error,
        response_cls=VectorBackendReloadResponse,
    )
