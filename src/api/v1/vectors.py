"""Vector management endpoints extracted from analyze.py for modularity."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Header
from pydantic import BaseModel, Field

from src.api.dependencies import get_admin_token, get_api_key
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
from src.core.vector_migration_plan_pipeline import (
    build_vector_migration_pending_summary_payload,
    build_vector_migration_plan_payload,
)
from src.core.vector_migration_preview_pipeline import run_vector_migration_preview_pipeline
from src.core.vector_migration_trends_pipeline import run_vector_migration_trends_pipeline
from src.core.vector_batch_similarity import run_vector_batch_similarity
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
    part_type: Optional[str] = None
    fine_part_type: Optional[str] = None
    coarse_part_type: Optional[str] = None
    decision_source: Optional[str] = None
    is_coarse_label: Optional[bool] = None


class VectorListResponse(BaseModel):
    total: int
    vectors: list[VectorListItem]


class VectorBackendReloadResponse(BaseModel):
    status: str
    backend: Optional[str] = None
    error: Optional[Dict[str, Any]] = None


class VectorRegisterRequest(BaseModel):
    id: str = Field(description="向量 ID")
    vector: list[float] = Field(description="向量数据")
    meta: Optional[Dict[str, str]] = Field(default=None, description="向量元数据")


class VectorRegisterResponse(BaseModel):
    id: str
    status: str
    dimension: Optional[int] = None
    error: Optional[Dict[str, Any]] = None


class VectorSearchRequest(BaseModel):
    vector: list[float] = Field(description="查询向量")
    k: int = Field(default=10, ge=1, le=100, description="返回数量")
    material_filter: Optional[str] = Field(default=None, description="材料过滤")
    complexity_filter: Optional[str] = Field(default=None, description="复杂度过滤")
    fine_part_type_filter: Optional[str] = Field(default=None, description="细分类过滤")
    coarse_part_type_filter: Optional[str] = Field(default=None, description="粗分类过滤")
    decision_source_filter: Optional[str] = Field(default=None, description="决策来源过滤")
    is_coarse_label_filter: Optional[bool] = Field(
        default=None,
        description="是否仅返回 coarse label 样本",
    )


class VectorSearchResponse(BaseModel):
    results: list[Dict[str, Any]]
    total: int


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


@router.post("/delete", response_model=VectorDeleteResponse)
async def delete_vector(payload: VectorDeleteRequest, api_key: str = Depends(get_api_key)):
    return await run_vector_delete_pipeline(
        payload=payload,
        response_cls=VectorDeleteResponse,
        error_code_cls=ErrorCode,
        build_error_fn=build_error,
        get_qdrant_store_fn=_get_qdrant_store_or_none,
        get_client_fn=get_client,
    )


@router.get("/", response_model=VectorListResponse)
async def list_vectors(
    source: str = Query(
        default="auto",
        description="Vector source: auto|memory|redis",
    ),
    offset: int = Query(default=0, ge=0, description="结果偏移用于分页"),
    limit: int = Query(default=50, ge=1, description="返回数量上限"),
    material_filter: Optional[str] = Query(default=None, description="材料过滤"),
    complexity_filter: Optional[str] = Query(default=None, description="复杂度过滤"),
    fine_part_type_filter: Optional[str] = Query(default=None, description="细分类过滤"),
    coarse_part_type_filter: Optional[str] = Query(default=None, description="粗分类过滤"),
    decision_source_filter: Optional[str] = Query(default=None, description="决策来源过滤"),
    is_coarse_label_filter: Optional[bool] = Query(
        default=None,
        description="是否仅返回 coarse label 样本",
    ),
    api_key: str = Depends(get_api_key),
):
    return await run_vector_list_pipeline(
        source=source,
        offset=offset,
        limit=limit,
        material_filter=material_filter,
        complexity_filter=complexity_filter,
        fine_part_type_filter=fine_part_type_filter,
        coarse_part_type_filter=coarse_part_type_filter,
        decision_source_filter=decision_source_filter,
        is_coarse_label_filter=is_coarse_label_filter,
        response_cls=VectorListResponse,
        item_cls=VectorListItem,
        error_code_cls=ErrorCode,
        build_error_fn=build_error,
        get_qdrant_store_fn=_get_qdrant_store_or_none,
        resolve_list_source_fn=_resolve_list_source,
        build_filter_conditions_fn=_build_vector_filter_conditions,
        list_vectors_redis_fn=_list_vectors_redis,
        list_vectors_memory_fn=_list_vectors_memory,
        get_client_fn=get_client,
    )


@router.post("/register", response_model=VectorRegisterResponse)
async def register_vector_endpoint(
    payload: VectorRegisterRequest,
    api_key: str = Depends(get_api_key),
):
    return await run_vector_register_pipeline(
        payload=payload,
        response_cls=VectorRegisterResponse,
        error_code_cls=ErrorCode,
        build_error_fn=build_error,
        get_qdrant_store_fn=_get_qdrant_store_or_none,
    )


@router.post("/search", response_model=VectorSearchResponse)
async def search_vectors(
    payload: VectorSearchRequest,
    api_key: str = Depends(get_api_key),
):
    return await run_vector_search_pipeline(
        payload=payload,
        response_cls=VectorSearchResponse,
        get_qdrant_store_fn=_get_qdrant_store_or_none,
        build_filter_conditions_fn=_build_vector_search_filter_conditions,
        matches_filters_fn=_matches_vector_search_filters,
        vector_item_payload_fn=_vector_item_payload,
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
    result = await run_vector_update_pipeline(
        payload=payload,
        qdrant_store=_get_qdrant_store_or_none(),
    )
    return VectorUpdateResponse(**result)


@router.get("/migrate/preview", response_model=VectorMigrationPreviewResponse)
async def preview_migration(to_version: str, limit: int = 10, api_key: str = Depends(get_api_key)):
    from src.core.feature_extractor import FeatureExtractor

    return await run_vector_migration_preview_pipeline(
        to_version=to_version,
        limit=limit,
        response_cls=VectorMigrationPreviewResponse,
        error_code_cls=ErrorCode,
        build_error_fn=build_error,
        get_qdrant_store_fn=_get_qdrant_store_or_none,
        collect_qdrant_preview_samples_fn=_collect_qdrant_preview_samples,
        prepare_vector_for_upgrade_fn=_prepare_vector_for_upgrade,
        feature_extractor_cls=FeatureExtractor,
    )


@router.post("/migrate", response_model=VectorMigrateResponse)
async def migrate_vectors(payload: VectorMigrateRequest, api_key: str = Depends(get_api_key)):
    import uuid

    from src.core.feature_extractor import FeatureExtractor
    from src.core.similarity import _VECTOR_META, _VECTOR_STORE  # type: ignore

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
    from src.utils.analysis_metrics import vector_migrate_dimension_delta, vector_migrate_total
    qdrant_store = _get_qdrant_store_or_none()

    for vid in payload.ids:
        if qdrant_store is not None:
            target = await qdrant_store.get_vector(vid)
            if target is None:
                items.append(VectorMigrateItem(id=vid, status="not_found", error="not_found"))
                skipped += 1
                vector_migrate_total.labels(status="not_found").inc()
                continue
            meta = dict(target.metadata or {})
            vec = list(target.vector or [])
        else:
            if vid not in _VECTOR_STORE:
                items.append(VectorMigrateItem(id=vid, status="not_found", error="not_found"))
                skipped += 1
                vector_migrate_total.labels(status="not_found").inc()
                continue
            meta = _VECTOR_META.get(vid, {})
            vec = _VECTOR_STORE[vid]

        from_version = meta.get("feature_version", "v1")
        if from_version == target_version:
            items.append(
                VectorMigrateItem(
                    id=vid, status="skipped", from_version=from_version, to_version=target_version
                )
            )
            skipped += 1
            vector_migrate_total.labels(status="skipped").inc()
            continue
        dimension_before = len(vec)
        try:
            base_vector, l3_tail, _ = _prepare_vector_for_upgrade(
                extractor,
                vec,
                meta,
                from_version,
            )
            new_features = extractor.upgrade_vector(base_vector, current_version=from_version)
            if l3_tail:
                new_features = new_features + l3_tail
            dimension_after = len(new_features)
            dimension_delta = dimension_after - dimension_before
            # Record dimension delta for observability
            vector_migrate_dimension_delta.observe(dimension_delta)
            if payload.dry_run:
                items.append(
                    VectorMigrateItem(
                        id=vid,
                        status="dry_run",
                        from_version=from_version,
                        to_version=target_version,
                        dimension_before=dimension_before,
                        dimension_after=dimension_after,
                    )
                )
                dry_run_total += 1
                vector_migrate_total.labels(status="dry_run").inc()
            else:
                _VECTOR_STORE[vid] = new_features
                meta["feature_version"] = target_version
                expected_2d_dim = extractor.expected_dim(target_version)
                meta["geometric_dim"] = str(expected_2d_dim - 2)
                meta["semantic_dim"] = "2"
                meta["total_dim"] = str(len(new_features))
                meta["vector_layout"] = VECTOR_LAYOUT_L3 if l3_tail else VECTOR_LAYOUT_BASE
                if l3_tail:
                    meta["l3_3d_dim"] = str(len(l3_tail))
                else:
                    meta.pop("l3_3d_dim", None)
                if qdrant_store is not None:
                    await qdrant_store.register_vector(vid, new_features, metadata=meta)
                else:
                    _VECTOR_STORE[vid] = new_features
                    _VECTOR_META[vid] = meta
                # Track downgrade separately if target lower than source
                if (from_version, target_version) in {
                    ("v4", "v3"),
                    ("v4", "v2"),
                    ("v4", "v1"),
                    ("v3", "v2"),
                    ("v3", "v1"),
                    ("v2", "v1"),
                }:
                    items.append(
                        VectorMigrateItem(
                            id=vid,
                            status="downgraded",
                            from_version=from_version,
                            to_version=target_version,
                            dimension_before=dimension_before,
                            dimension_after=dimension_after,
                        )
                    )
                    vector_migrate_total.labels(status="downgraded").inc()
                else:
                    migrated += 1
                    items.append(
                        VectorMigrateItem(
                            id=vid,
                            status="migrated",
                            from_version=from_version,
                            to_version=target_version,
                            dimension_before=dimension_before,
                            dimension_after=dimension_after,
                        )
                    )
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
    history = list(globals().get("_VECTOR_MIGRATION_HISTORY", []))
    snapshot = await collect_vector_migration_distribution_snapshot(
        qdrant_store=_get_qdrant_store_or_none(),
        scan_limit=_resolve_vector_migration_scan_limit(),
        collect_qdrant_feature_versions_fn=_collect_qdrant_feature_versions,
        build_readiness_fn=_build_vector_migration_readiness,
    )
    return VectorMigrationStatusResponse(
        **build_vector_migration_status_payload(history=history, snapshot=snapshot)
    )


@router.get("/migrate/summary", response_model=VectorMigrationSummaryResponse)
async def migrate_summary(api_key: str = Depends(get_api_key)):
    history = list(globals().get("_VECTOR_MIGRATION_HISTORY", []))
    snapshot = await collect_vector_migration_distribution_snapshot(
        qdrant_store=_get_qdrant_store_or_none(),
        scan_limit=_resolve_vector_migration_scan_limit(),
        collect_qdrant_feature_versions_fn=_collect_qdrant_feature_versions,
        build_readiness_fn=_build_vector_migration_readiness,
    )
    return VectorMigrationSummaryResponse(
        **build_vector_migration_summary_payload(history=history, snapshot=snapshot)
    )


async def _collect_vector_migration_pending_candidates(
    *,
    limit: int,
    target_version: str,
    from_version_filter: Optional[str] = None,
) -> Dict[str, Any]:
    qdrant_store = _get_qdrant_store_or_none()
    scan_limit = _resolve_vector_migration_scan_limit()
    normalized_filter = str(from_version_filter or "").strip() or None

    if qdrant_store is not None:
        total_available = int(await qdrant_store.count())
        max_scan = min(total_available, scan_limit)
        scanned = 0
        offset = 0
        items: list[VectorMigrationPendingItem] = []
        pending_ids: list[str] = []
        total_pending = 0
        observed_by_from_version: Dict[str, int] = {}
        while scanned < max_scan:
            batch_limit = min(200, max_scan - scanned)
            points, _ = await qdrant_store.list_vectors(
                offset=offset,
                limit=batch_limit,
                with_vectors=False,
            )
            if not points:
                break
            for point in points:
                meta = point.metadata or {}
                from_version = str(meta.get("feature_version") or "unknown")
                if from_version == target_version:
                    continue
                if normalized_filter and from_version != normalized_filter:
                    continue
                total_pending += 1
                observed_by_from_version[from_version] = (
                    observed_by_from_version.get(from_version, 0) + 1
                )
                pending_ids.append(str(point.id))
                if len(items) < limit:
                    items.append(
                        VectorMigrationPendingItem(
                            id=str(point.id),
                            from_version=from_version,
                            to_version=target_version,
                        )
                    )
            consumed = len(points)
            scanned += consumed
            offset += consumed

        distribution_complete = scanned >= total_available
        return {
            "target_version": target_version,
            "from_version_filter": normalized_filter,
            "items": items,
            "pending_ids": pending_ids[:limit],
            "listed_count": len(items),
            "total_pending": total_pending if distribution_complete else None,
            "observed_by_from_version": observed_by_from_version,
            "backend": "qdrant",
            "scanned_vectors": scanned,
            "scan_limit": scan_limit,
            "distribution_complete": distribution_complete,
        }

    from src.core.similarity import _VECTOR_META, _VECTOR_STORE  # type: ignore

    items: list[VectorMigrationPendingItem] = []
    pending_ids: list[str] = []
    scanned = 0
    total_pending = 0
    observed_by_from_version: Dict[str, int] = {}
    for vid, meta in _VECTOR_META.items():
        if vid not in _VECTOR_STORE:
            continue
        scanned += 1
        from_version = str(meta.get("feature_version") or "unknown")
        if from_version == target_version:
            continue
        if normalized_filter and from_version != normalized_filter:
            continue
        total_pending += 1
        observed_by_from_version[from_version] = observed_by_from_version.get(from_version, 0) + 1
        pending_ids.append(str(vid))
        if len(items) < limit:
            items.append(
                VectorMigrationPendingItem(
                    id=str(vid),
                    from_version=from_version,
                    to_version=target_version,
                )
            )

    return {
        "target_version": target_version,
        "from_version_filter": normalized_filter,
        "items": items,
        "pending_ids": pending_ids[:limit],
        "listed_count": len(items),
        "total_pending": total_pending,
        "observed_by_from_version": observed_by_from_version,
        "backend": "memory",
        "scanned_vectors": scanned,
        "scan_limit": scan_limit,
        "distribution_complete": True,
    }


@router.get("/migrate/pending", response_model=VectorMigrationPendingResponse)
async def migrate_pending(
    limit: int = 50,
    from_version_filter: Optional[str] = Query(default=None),
    api_key: str = Depends(get_api_key),
):
    limit = max(min(int(limit or 0), 200), 1)
    target_version = _resolve_vector_migration_target_version()
    pending = await _collect_vector_migration_pending_candidates(
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
    from_version_filter: Optional[str] = Query(default=None),
    api_key: str = Depends(get_api_key),
):
    target_version = _resolve_vector_migration_target_version()
    pending = await _collect_vector_migration_pending_candidates(
        limit=1,
        target_version=target_version,
        from_version_filter=from_version_filter,
    )
    return VectorMigrationPendingSummaryResponse(
        **build_vector_migration_pending_summary_payload(pending=pending)
    )


@router.get("/migrate/plan", response_model=VectorMigrationPlanResponse)
async def migrate_plan(
    from_version_filter: Optional[str] = Query(default=None),
    max_batches: int = Query(default=3, ge=1, le=10),
    default_run_limit: int = Query(default=50, ge=1, le=200),
    api_key: str = Depends(get_api_key),
):
    target_version = _resolve_vector_migration_target_version()
    pending = await _collect_vector_migration_pending_candidates(
        limit=1,
        target_version=target_version,
        from_version_filter=from_version_filter,
    )
    return VectorMigrationPlanResponse(
        **build_vector_migration_plan_payload(
            pending=pending,
            max_batches=max_batches,
            default_run_limit=default_run_limit,
        )
    )


@router.post("/migrate/pending/run", response_model=VectorMigrateResponse)
async def migrate_pending_run(
    payload: VectorMigrationPendingRunRequest,
    api_key: str = Depends(get_api_key),
):
    target_version = _resolve_vector_migration_target_version()
    pending = await _collect_vector_migration_pending_candidates(
        limit=payload.limit,
        target_version=target_version,
        from_version_filter=payload.from_version_filter,
    )
    if pending["backend"] == "qdrant" and not pending["distribution_complete"]:
        if not payload.allow_partial_scan:
            raise HTTPException(
                status_code=409,
                detail=build_error(
                    ErrorCode.CONSTRAINT_VIOLATION,
                    stage="vector_migrate_pending_run",
                    message=(
                        "Qdrant distribution scan is partial; raise VECTOR_MIGRATION_SCAN_LIMIT "
                        "or set allow_partial_scan=true"
                    ),
                    target_version=target_version,
                    scanned_vectors=pending["scanned_vectors"],
                    scan_limit=pending["scan_limit"],
                ),
            )
    return await migrate_vectors(
        VectorMigrateRequest(
            ids=pending["pending_ids"],
            to_version=target_version,
            dry_run=payload.dry_run,
        ),
        api_key=api_key,
    )


@router.get("/migrate/trends", response_model=VectorMigrationTrendsResponse)
async def migrate_trends(window_hours: int = 24, api_key: str = Depends(get_api_key)):
    return await run_vector_migration_trends_pipeline(
        window_hours=window_hours,
        history=list(globals().get("_VECTOR_MIGRATION_HISTORY", [])),
        response_cls=VectorMigrationTrendsResponse,
        get_qdrant_store_fn=_get_qdrant_store_or_none,
        resolve_scan_limit_fn=_resolve_vector_migration_scan_limit,
        collect_qdrant_feature_versions_fn=_collect_qdrant_feature_versions,
        build_readiness_fn=_build_vector_migration_readiness,
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
    import os

    from src.core.similarity import reload_vector_store_backend
    from src.utils.analysis_metrics import vector_store_reload_total

    allowed = {"memory", "faiss", "redis"}
    if backend is not None:
        backend = backend.strip().lower()
        if backend not in allowed:
            vector_store_reload_total.labels(status="error", reason="invalid_backend").inc()
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
    try:
        ok = reload_vector_store_backend()
    except Exception as e:
        vector_store_reload_total.labels(status="error", reason="init_error").inc()
        err = build_error(
            ErrorCode.INTERNAL_ERROR,
            stage="backend_reload",
            message="Exception during backend reload",
            backend=effective_backend,
            detail=str(e),
        )
        raise HTTPException(status_code=500, detail=err)
    vector_store_reload_total.labels(
        status="success" if ok else "error",
        reason="ok" if ok else "init_error",
    ).inc()
    if not ok:
        err = build_error(
            ErrorCode.INTERNAL_ERROR,
            stage="backend_reload",
            message="Vector store backend reload failed",
            backend=effective_backend,
        )
        raise HTTPException(status_code=500, detail=err)
    return VectorBackendReloadResponse(status="ok", backend=effective_backend)
