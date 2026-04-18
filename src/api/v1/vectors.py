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


def _get_qdrant_store_or_none():
    if os.getenv("VECTOR_STORE_BACKEND", "memory") != "qdrant":
        return None
    try:
        from src.core.vector_stores import get_vector_store as get_managed_vector_store

        return get_managed_vector_store("qdrant")
    except Exception:
        return None


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
    qdrant_store = _get_qdrant_store_or_none()
    if qdrant_store is not None:
        existing = await qdrant_store.get_vector(payload.id)
        if existing is None:
            err = build_error(
                ErrorCode.DATA_NOT_FOUND,
                stage="vector_delete",
                message="Vector not found",
                id=payload.id,
            )
            raise HTTPException(status_code=404, detail=err)
        deleted = await qdrant_store.delete_vector(payload.id)
        if deleted:
            return VectorDeleteResponse(id=payload.id, status="deleted")
        err = build_error(
            ErrorCode.INTERNAL_ERROR,
            stage="vector_delete",
            message="Delete failed",
            id=payload.id,
        )
        raise HTTPException(status_code=500, detail=err)

    from src.core.similarity import (  # type: ignore
        _BACKEND,
        _VECTOR_META,
        _VECTOR_STORE,
        FaissVectorStore,
    )
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
    from src.core.similarity import _BACKEND, _VECTOR_META, _VECTOR_STORE  # type: ignore

    allowed_sources = {"auto", "memory", "redis", "qdrant"}
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
    if resolved == "qdrant":
        qdrant_store = _get_qdrant_store_or_none()
        if qdrant_store is not None:
            from src.core.similarity import extract_vector_label_contract

            results, total = await qdrant_store.list_vectors(
                offset=offset,
                limit=limit,
                filter_conditions=_build_vector_filter_conditions(
                    material_filter=material_filter,
                    complexity_filter=complexity_filter,
                    fine_part_type_filter=fine_part_type_filter,
                    coarse_part_type_filter=coarse_part_type_filter,
                    decision_source_filter=decision_source_filter,
                    is_coarse_label_filter=is_coarse_label_filter,
                ),
                with_vectors=True,
            )
            items = []
            for result in results:
                meta = result.metadata or {}
                label_contract = extract_vector_label_contract(meta)
                items.append(
                    VectorListItem(
                        **_vector_item_payload(
                            result.id,
                            len(result.vector or []),
                            meta,
                            label_contract,
                        )
                    )
                )
            return VectorListResponse(total=total, vectors=items)
    if resolved == "redis":
        client = get_client()
        if client is not None:
            return await _list_vectors_redis(
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
            )
    return _list_vectors_memory(
        _VECTOR_STORE,
        _VECTOR_META,
        offset,
        limit,
        material_filter,
        complexity_filter,
        fine_part_type_filter,
        coarse_part_type_filter,
        decision_source_filter,
        is_coarse_label_filter,
    )


@router.post("/register", response_model=VectorRegisterResponse)
async def register_vector_endpoint(
    payload: VectorRegisterRequest,
    api_key: str = Depends(get_api_key),
):
    qdrant_store = _get_qdrant_store_or_none()
    if qdrant_store is not None:
        meta = dict(payload.meta or {})
        meta.setdefault("total_dim", str(len(payload.vector)))
        await qdrant_store.register_vector(payload.id, payload.vector, metadata=meta)
        return VectorRegisterResponse(
            id=payload.id,
            status="accepted",
            dimension=len(payload.vector),
        )

    from src.core.similarity import FaissVectorStore, last_vector_error, register_vector

    meta = dict(payload.meta or {})
    meta.setdefault("total_dim", str(len(payload.vector)))
    accepted = register_vector(payload.id, payload.vector, meta=meta)
    if accepted:
        if os.getenv("VECTOR_STORE_BACKEND", "memory") == "faiss":
            try:
                fstore = FaissVectorStore()
                fstore.add(payload.id, payload.vector)
            except Exception:
                pass
        return VectorRegisterResponse(
            id=payload.id,
            status="accepted",
            dimension=len(payload.vector),
        )

    err = last_vector_error()
    if err is None:
        err = build_error(
            ErrorCode.DIMENSION_MISMATCH,
            stage="vector_register",
            message="Vector rejected",
            id=payload.id,
        )
    return VectorRegisterResponse(
        id=payload.id,
        status="rejected",
        error=err,
    )


@router.post("/search", response_model=VectorSearchResponse)
async def search_vectors(
    payload: VectorSearchRequest,
    api_key: str = Depends(get_api_key),
):
    from src.core.similarity import (
        _VECTOR_META,
        _VECTOR_STORE,
        extract_vector_label_contract,
        get_vector_store,
    )

    qdrant_store = _get_qdrant_store_or_none()
    if qdrant_store is not None:
        results = await qdrant_store.search_similar(
            payload.vector,
            top_k=payload.k,
            filter_conditions=_build_vector_search_filter_conditions(payload),
            with_vectors=True,
        )
        items = []
        for result in results:
            meta = result.metadata or {}
            label_contract = extract_vector_label_contract(meta)
            items.append(
                {
                    "id": result.id,
                    "score": round(float(result.score), 4),
                    **_vector_item_payload(
                        result.id,
                        len(result.vector or []),
                        meta,
                        label_contract,
                    ),
                }
            )
        return VectorSearchResponse(results=items, total=len(items))

    store = get_vector_store()
    query_k = payload.k
    if any(
        [
            payload.material_filter,
            payload.complexity_filter,
            payload.fine_part_type_filter,
            payload.coarse_part_type_filter,
            payload.decision_source_filter,
            payload.is_coarse_label_filter is not None,
        ]
    ):
        query_k = min(payload.k * 5, payload.k + 100)
    results = store.query(payload.vector, top_k=query_k)
    seen: set[str] = set()
    items: list[Dict[str, Any]] = []
    for vid, score in results:
        if vid in seen:
            continue
        seen.add(vid)
        meta = _VECTOR_META.get(vid) or store.meta(vid) or {}
        label_contract = extract_vector_label_contract(meta)
        if not _matches_vector_search_filters(payload, meta, label_contract):
            continue
        items.append(
            {"id": vid, "score": round(float(score), 4)}
            | _vector_item_payload(
                vid,
                len(_VECTOR_STORE.get(vid, [])),
                meta,
                label_contract,
            )
        )
        if len(items) >= payload.k:
            break

    return VectorSearchResponse(results=items, total=len(items))


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
    """
    预览向量特征迁移 - 不执行写入操作

    Args:
        to_version: 目标特征版本 (v1/v2/v3/v4)
        limit: 预览样本数量 (默认10, 最大100)
        api_key: API密钥

    Returns:
        迁移预览信息，包含版本分布、样本预览、维度变化预估等
    """
    from src.core.feature_extractor import FeatureExtractor
    from src.core.similarity import _VECTOR_META, _VECTOR_STORE

    # Validate target version
    allowed_versions = {"v1", "v2", "v3", "v4"}
    if to_version not in allowed_versions:
        err = build_error(
            ErrorCode.INPUT_VALIDATION_FAILED,
            stage="migration_preview",
            message="Unsupported target feature version",
            to_version=to_version,
            allowed=list(allowed_versions),
        )
        from src.utils.analysis_metrics import analysis_error_code_total

        analysis_error_code_total.labels(code=ErrorCode.INPUT_VALIDATION_FAILED.value).inc()
        raise HTTPException(status_code=422, detail=err)

    # Enforce limit cap
    limit = min(limit, 100)

    # Get extractor for target version
    extractor = FeatureExtractor(feature_version=to_version)

    qdrant_store = _get_qdrant_store_or_none()
    if qdrant_store is not None:
        preview_source, total_vectors, by_version = await _collect_qdrant_preview_samples(
            qdrant_store,
            limit=limit,
        )
    else:
        # Collect version distribution
        by_version = {}
        total_vectors = len(_VECTOR_STORE)
        preview_source = []

        for vid in _VECTOR_STORE.keys():
            meta = _VECTOR_META.get(vid, {})
            current_version = meta.get("feature_version", "v1")
            by_version[current_version] = by_version.get(current_version, 0) + 1

        for vid in list(_VECTOR_STORE.keys())[:limit]:
            preview_source.append((vid, _VECTOR_STORE[vid], _VECTOR_META.get(vid, {})))

    # Preview sample vectors
    preview_items: list[VectorMigrateItem] = []
    dimension_changes = {"positive": 0, "negative": 0, "zero": 0}
    deltas: list[int] = []
    warnings: list[str] = []
    sampled = 0

    for vid, vec, meta in preview_source:
        from_version = meta.get("feature_version", "v1")
        dimension_before = len(vec)

        if from_version == to_version:
            preview_items.append(
                VectorMigrateItem(
                    id=vid,
                    status="skipped",
                    from_version=from_version,
                    to_version=to_version,
                    dimension_before=dimension_before,
                    dimension_after=dimension_before,
                )
            )
            dimension_changes["zero"] += 1
            sampled += 1
            continue

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
            deltas.append(dimension_delta)

            if dimension_delta > 0:
                dimension_changes["positive"] += 1
            elif dimension_delta < 0:
                dimension_changes["negative"] += 1
            else:
                dimension_changes["zero"] += 1

            # Detect downgrade
            is_downgrade = (from_version, to_version) in {
                ("v4", "v3"),
                ("v4", "v2"),
                ("v4", "v1"),
                ("v3", "v2"),
                ("v3", "v1"),
                ("v2", "v1"),
            }

            preview_items.append(
                VectorMigrateItem(
                    id=vid,
                    status="downgrade_preview" if is_downgrade else "upgrade_preview",
                    from_version=from_version,
                    to_version=to_version,
                    dimension_before=dimension_before,
                    dimension_after=dimension_after,
                )
            )
            sampled += 1

        except Exception as e:
            preview_items.append(VectorMigrateItem(id=vid, status="error_preview", error=str(e)))
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
            median_delta = float(deltas[len(deltas) // 2])
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
    qdrant_store = _get_qdrant_store_or_none()
    scan_limit = _resolve_vector_migration_scan_limit()
    if qdrant_store is not None:
        versions, total_available, scanned = await _collect_qdrant_feature_versions(qdrant_store)
        backend = "qdrant"
        distribution_complete = scanned >= total_available
    else:
        from src.core.similarity import _VECTOR_META, _VECTOR_STORE  # type: ignore

        versions = {}
        total_available = 0
        for vid, meta in _VECTOR_META.items():
            if vid not in _VECTOR_STORE:
                continue
            total_available += 1
            ver = meta.get("feature_version", "unknown")
            versions[ver] = versions.get(ver, 0) + 1
        scanned = total_available
        backend = "memory"
        distribution_complete = True
    readiness = _build_vector_migration_readiness(
        versions,
        total_vectors=total_available,
        distribution_complete=distribution_complete,
    )
    history = globals().get("_VECTOR_MIGRATION_HISTORY", [])
    last = history[-1] if history else None
    return VectorMigrationStatusResponse(
        last_migration_id=last.get("migration_id") if last else None,
        last_started_at=datetime.fromisoformat(last.get("started_at")) if last else None,
        last_finished_at=datetime.fromisoformat(last.get("finished_at")) if last else None,
        last_total=last.get("total") if last else None,
        last_migrated=last.get("migrated") if last else None,
        last_skipped=last.get("skipped") if last else None,
        pending_vectors=readiness["pending_vectors"],
        feature_versions=versions,
        history=history,
        backend=backend,
        current_total_vectors=total_available,
        scanned_vectors=scanned,
        scan_limit=scan_limit,
        distribution_complete=distribution_complete,
        target_version=readiness["target_version"],
        target_version_vectors=readiness["target_version_vectors"],
        target_version_ratio=readiness["target_version_ratio"],
        migration_ready=readiness["migration_ready"],
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
    qdrant_store = _get_qdrant_store_or_none()
    scan_limit = _resolve_vector_migration_scan_limit()
    if qdrant_store is not None:
        (
            current_version_distribution,
            current_total_vectors,
            scanned_vectors,
        ) = await _collect_qdrant_feature_versions(qdrant_store)
        backend = "qdrant"
        distribution_complete = scanned_vectors >= current_total_vectors
    else:
        from src.core.similarity import _VECTOR_META, _VECTOR_STORE  # type: ignore

        current_version_distribution = {}
        current_total_vectors = 0
        for vid, meta in _VECTOR_META.items():
            if vid not in _VECTOR_STORE:
                continue
            current_total_vectors += 1
            version = meta.get("feature_version", "unknown")
            current_version_distribution[version] = (
                current_version_distribution.get(version, 0) + 1
            )
        backend = "memory"
        scanned_vectors = current_total_vectors
        distribution_complete = True
    readiness = _build_vector_migration_readiness(
        current_version_distribution,
        total_vectors=current_total_vectors,
        distribution_complete=distribution_complete,
    )
    return VectorMigrationSummaryResponse(
        counts=aggregate,
        total_migrations=total_migrations,
        history_size=len(history),
        statuses=statuses,
        backend=backend,
        current_version_distribution=current_version_distribution,
        current_total_vectors=current_total_vectors,
        scanned_vectors=scanned_vectors,
        scan_limit=scan_limit,
        distribution_complete=distribution_complete,
        target_version=readiness["target_version"],
        target_version_vectors=readiness["target_version_vectors"],
        target_version_ratio=readiness["target_version_ratio"],
        pending_vectors=readiness["pending_vectors"],
        migration_ready=readiness["migration_ready"],
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
    pending_ratio: Optional[float] = None
    recommended_from_versions = [
        key
        for key, _ in sorted(
            pending["observed_by_from_version"].items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )
    ]
    largest_pending_from_version = (
        recommended_from_versions[0] if recommended_from_versions else None
    )
    largest_pending_count = None
    if largest_pending_from_version is not None:
        largest_pending_count = int(
            pending["observed_by_from_version"].get(largest_pending_from_version, 0)
        )
    if pending["distribution_complete"]:
        scanned_vectors = int(pending["scanned_vectors"] or 0)
        pending_ratio = round(int(pending["total_pending"] or 0) / max(scanned_vectors, 1), 4)
    return VectorMigrationPendingSummaryResponse(
        target_version=pending["target_version"],
        from_version_filter=pending["from_version_filter"],
        observed_by_from_version=pending["observed_by_from_version"],
        recommended_from_versions=recommended_from_versions,
        largest_pending_from_version=largest_pending_from_version,
        largest_pending_count=largest_pending_count,
        total_pending=pending["total_pending"],
        pending_ratio=pending_ratio,
        backend=pending["backend"],
        scanned_vectors=pending["scanned_vectors"],
        scan_limit=pending["scan_limit"],
        distribution_complete=pending["distribution_complete"],
    )


def _build_vector_migration_plan_batches(
    *,
    observed_by_from_version: Dict[str, int],
    max_batches: int,
    default_run_limit: int,
    allow_partial_scan_required: bool,
) -> List[VectorMigrationPlanBatch]:
    ordered = sorted(
        observed_by_from_version.items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    )[:max_batches]
    batches: List[VectorMigrationPlanBatch] = []
    for index, (from_version, count) in enumerate(ordered):
        pending_count = int(count)
        suggested_run_limit = min(pending_count, default_run_limit)
        notes: List[str] = []
        if pending_count > suggested_run_limit:
            notes.append("split_batch_required")
        else:
            notes.append("single_batch_ready")
        if allow_partial_scan_required:
            notes.append("partial_scan_override_required")
        batches.append(
            VectorMigrationPlanBatch(
                priority=index + 1,
                from_version=str(from_version),
                pending_count=pending_count,
                suggested_run_limit=suggested_run_limit,
                allow_partial_scan_required=allow_partial_scan_required,
                request_payload={
                    "limit": suggested_run_limit,
                    "dry_run": True,
                    "from_version_filter": str(from_version),
                    "allow_partial_scan": allow_partial_scan_required,
                },
                notes=notes,
            )
        )
    return batches


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
    pending_ratio: Optional[float] = None
    recommended_from_versions = [
        key
        for key, _ in sorted(
            pending["observed_by_from_version"].items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )
    ]
    largest_pending_from_version = (
        recommended_from_versions[0] if recommended_from_versions else None
    )
    largest_pending_count = None
    if largest_pending_from_version is not None:
        largest_pending_count = int(
            pending["observed_by_from_version"].get(largest_pending_from_version, 0)
        )
    if pending["distribution_complete"]:
        scanned_vectors = int(pending["scanned_vectors"] or 0)
        pending_ratio = round(int(pending["total_pending"] or 0) / max(scanned_vectors, 1), 4)
    allow_partial_scan_required = pending["backend"] == "qdrant" and not pending["distribution_complete"]
    batches = _build_vector_migration_plan_batches(
        observed_by_from_version=pending["observed_by_from_version"],
        max_batches=max_batches,
        default_run_limit=default_run_limit,
        allow_partial_scan_required=allow_partial_scan_required,
    )
    blocking_reasons: List[str] = []
    if allow_partial_scan_required:
        blocking_reasons.append("partial_scan_override_required")
    if not batches:
        blocking_reasons.append("no_pending_vectors")
    recommended_first_batch = batches[0] if batches else None
    recommended_first_request_payload = None
    if recommended_first_batch is not None:
        recommended_first_request_payload = dict(recommended_first_batch.request_payload)
    planned_pending_count = sum(int(batch.pending_count) for batch in batches)
    planned_from_versions = {str(batch.from_version) for batch in batches}
    unplanned_from_versions = [
        from_version
        for from_version in recommended_from_versions
        if str(from_version) not in planned_from_versions
    ]
    remaining_pending_count: Optional[int] = None
    planned_pending_ratio: Optional[float] = None
    if pending["distribution_complete"] and pending["total_pending"] is not None:
        remaining_pending_count = max(int(pending["total_pending"]) - planned_pending_count, 0)
        planned_pending_ratio = round(
            planned_pending_count / max(int(pending["total_pending"]), 1),
            4,
        )
    coverage_complete = bool(batches) and not unplanned_from_versions
    suggested_next_max_batches: Optional[int] = None
    if unplanned_from_versions:
        suggested_next_max_batches = len(recommended_from_versions)
    estimated_runs_by_version = {
        str(from_version): max((int(count) + default_run_limit - 1) // default_run_limit, 1)
        for from_version, count in pending["observed_by_from_version"].items()
    }
    return VectorMigrationPlanResponse(
        target_version=pending["target_version"],
        from_version_filter=pending["from_version_filter"],
        observed_by_from_version=pending["observed_by_from_version"],
        recommended_from_versions=recommended_from_versions,
        largest_pending_from_version=largest_pending_from_version,
        largest_pending_count=largest_pending_count,
        total_pending=pending["total_pending"],
        pending_ratio=pending_ratio,
        backend=pending["backend"],
        scanned_vectors=pending["scanned_vectors"],
        scan_limit=pending["scan_limit"],
        distribution_complete=pending["distribution_complete"],
        max_batches=max_batches,
        default_run_limit=default_run_limit,
        estimated_total_runs=sum(estimated_runs_by_version.values()),
        estimated_runs_by_version=estimated_runs_by_version,
        plan_ready=bool(batches) and not blocking_reasons,
        blocking_reasons=blocking_reasons,
        recommended_first_batch=recommended_first_batch,
        recommended_first_request_payload=recommended_first_request_payload,
        planned_pending_count=planned_pending_count,
        remaining_pending_count=remaining_pending_count,
        planned_pending_ratio=planned_pending_ratio,
        coverage_complete=coverage_complete,
        truncated_by_max_batches=bool(unplanned_from_versions),
        unplanned_from_versions=unplanned_from_versions,
        suggested_next_max_batches=suggested_next_max_batches,
        batches=batches,
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
    """
    获取迁移趋势统计

    Args:
        window_hours: 统计窗口小时数 (默认24小时)
        api_key: API密钥

    Returns:
        迁移趋势统计，包含成功率、v4采用率、维度变化等
    """
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
    qdrant_store = _get_qdrant_store_or_none()
    scan_limit = _resolve_vector_migration_scan_limit()
    if qdrant_store is not None:
        version_distribution, total_vectors, scanned_vectors = (
            await _collect_qdrant_feature_versions(qdrant_store)
        )
        distribution_complete = scanned_vectors >= total_vectors
    else:
        from src.core.similarity import _VECTOR_META, _VECTOR_STORE

        version_distribution = {}
        total_vectors = 0
        for vid in _VECTOR_STORE.keys():
            meta = _VECTOR_META.get(vid, {})
            version = meta.get("feature_version", "v1")
            version_distribution[version] = version_distribution.get(version, 0) + 1
            total_vectors += 1
        scanned_vectors = total_vectors
        distribution_complete = True

    v4_count = version_distribution.get("v4", 0)
    v4_adoption_rate = v4_count / max(total_vectors, 1)

    # Calculate average dimension delta (estimate from history)
    avg_dimension_delta = 0.0
    # For now, estimate based on version changes (v3->v4 adds 2 dimensions)
    if total_migrated > 0:
        # Rough estimate: upgrade to v4 adds 2, downgrade removes dimensions
        avg_dimension_delta = (total_migrated * 2 - total_downgraded * 2) / max(
            total_migrated + total_downgraded, 1
        )

    # Migration velocity (per hour)
    migration_velocity = total_migrations / max(window_hours, 1)

    # Time range
    time_range = {
        "start": (datetime.utcnow() - timedelta(hours=window_hours)).isoformat()
        if window_hours > 0
        else None,
        "end": datetime.utcnow().isoformat(),
    }
    readiness = _build_vector_migration_readiness(
        version_distribution,
        total_vectors=total_vectors,
        distribution_complete=distribution_complete,
    )

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
        current_total_vectors=total_vectors,
        scanned_vectors=scanned_vectors,
        scan_limit=scan_limit,
        distribution_complete=distribution_complete,
        target_version=readiness["target_version"],
        target_version_vectors=readiness["target_version_vectors"],
        target_version_ratio=readiness["target_version_ratio"],
        pending_vectors=readiness["pending_vectors"],
        migration_ready=readiness["migration_ready"],
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

    from src.core.similarity import (
        _VECTOR_META,
        _VECTOR_STORE,
        extract_vector_label_contract,
        get_degraded_mode_info,
        get_vector_store,
    )
    from src.utils.analysis_metrics import (
        vector_query_backend_total,
        vector_query_batch_latency_seconds,
    )

    batch_id = str(uuid.uuid4())
    # Enforce batch size cap from env or default 200
    import os

    max_batch = int(os.getenv("BATCH_SIMILARITY_MAX_IDS", "200"))
    if len(payload.ids) > max_batch:
        from src.utils.analysis_metrics import analysis_error_code_total, analysis_rejections_total

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

    qdrant_store = _get_qdrant_store_or_none()
    if qdrant_store is not None:
        from src.core.similarity import extract_vector_label_contract

        filter_conditions = _build_vector_filter_conditions(
            material_filter=payload.material,
            complexity_filter=payload.complexity,
            fine_part_type_filter=None,
            coarse_part_type_filter=None,
            decision_source_filter=None,
            is_coarse_label_filter=None,
        )

        for vid in payload.ids:
            target = await qdrant_store.get_vector(vid)
            if target is None:
                items.append(
                    BatchSimilarityItem(
                        id=vid,
                        status="not_found",
                        error=build_error(
                            ErrorCode.DATA_NOT_FOUND,
                            stage="batch_similarity",
                            message="Vector not found",
                            id=vid,
                        ),
                    )
                )
                failed += 1
                continue

            try:
                query_vector = list(target.vector or [])
                results = await qdrant_store.search_similar(
                    query_vector,
                    top_k=payload.top_k + 1,
                    filter_conditions=filter_conditions or None,
                    score_threshold=payload.min_score,
                    with_vectors=True,
                )

                similar: list[Dict[str, Any]] = []
                for result in results:
                    if result.id == vid:
                        continue
                    meta = result.metadata or {}
                    label_contract = extract_vector_label_contract(meta)
                    dimension = len(result.vector or [])
                    if dimension <= 0:
                        try:
                            dimension = int(meta.get("total_dim") or 0)
                        except (TypeError, ValueError):
                            dimension = 0
                    similar.append(
                        {
                            "id": result.id,
                            "score": round(float(result.score), 4),
                            "material": meta.get("material"),
                            "complexity": meta.get("complexity"),
                            "format": meta.get("format"),
                            "dimension": dimension,
                            "part_type": label_contract.get("part_type"),
                            "fine_part_type": label_contract.get("fine_part_type"),
                            "coarse_part_type": label_contract.get("coarse_part_type"),
                            "decision_source": label_contract.get("decision_source"),
                            "is_coarse_label": label_contract.get("is_coarse_label"),
                        }
                    )
                    if len(similar) >= payload.top_k:
                        break

                items.append(BatchSimilarityItem(id=vid, status="success", similar=similar))
                successful += 1
            except Exception as e:
                items.append(
                    BatchSimilarityItem(
                        id=vid,
                        status="error",
                        error=build_error(
                            ErrorCode.INTERNAL_ERROR,
                            stage="batch_similarity",
                            message="Query failed",
                            id=vid,
                            detail=str(e),
                        ),
                    )
                )
                failed += 1

        duration = time.time() - start_time
        vector_query_batch_latency_seconds.labels(batch_size_range=size_range).observe(duration)

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
            fallback=None,
            degraded=False,
        )

    # Get vector store using factory (handles backend selection and fallback)
    store = get_vector_store()

    # Detect if fallback occurred (Faiss unavailable -> memory)
    is_fallback = bool(getattr(store, "_fallback_from", None))
    requested_backend = getattr(
        store, "_requested_backend", os.getenv("VECTOR_STORE_BACKEND", "memory")
    )
    expected_backend = requested_backend
    if not is_fallback and not getattr(store, "_available", True):
        is_fallback = True
    elif not is_fallback and expected_backend == "faiss":
        from src.core.similarity import FaissVectorStore

        if not isinstance(FaissVectorStore, type):
            is_fallback = True
        elif not isinstance(store, FaissVectorStore):
            is_fallback = True
    if is_fallback:
        # Record fallback metric
        try:
            vector_query_backend_total.labels(backend="memory_fallback").inc()
        except Exception:
            pass

    # Process each vector ID
    for vid in payload.ids:
        if vid not in _VECTOR_STORE:
            items.append(
                BatchSimilarityItem(
                    id=vid,
                    status="not_found",
                    error=build_error(
                        ErrorCode.DATA_NOT_FOUND,
                        stage="batch_similarity",
                        message="Vector not found",
                        id=vid,
                    ),
                )
            )
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

                label_contract = extract_vector_label_contract(meta)
                similar.append(
                    {
                        "id": result_id,
                        "score": round(score, 4),
                        "material": meta.get("material"),
                        "complexity": meta.get("complexity"),
                        "format": meta.get("format"),
                        "dimension": len(_VECTOR_STORE.get(result_id, [])),
                        "part_type": label_contract.get("part_type"),
                        "fine_part_type": label_contract.get("fine_part_type"),
                        "coarse_part_type": label_contract.get("coarse_part_type"),
                        "decision_source": label_contract.get("decision_source"),
                        "is_coarse_label": label_contract.get("is_coarse_label"),
                    }
                )

                # Limit to top_k after filtering
                if len(similar) >= payload.top_k:
                    break

            items.append(BatchSimilarityItem(id=vid, status="success", similar=similar))
            successful += 1

        except Exception as e:
            items.append(
                BatchSimilarityItem(
                    id=vid,
                    status="error",
                    error=build_error(
                        ErrorCode.INTERNAL_ERROR,
                        stage="batch_similarity",
                        message="Query failed",
                        id=vid,
                        detail=str(e),
                    ),
                )
            )
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
    degraded_info = get_degraded_mode_info() or {}
    degraded_flag = bool(degraded_info.get("degraded", False))
    fallback = bool(is_fallback or degraded_flag)
    is_degraded = bool(degraded_flag or is_fallback)

    return BatchSimilarityResponse(
        total=len(payload.ids),
        successful=successful,
        failed=failed,
        items=items,
        batch_id=batch_id,
        duration_ms=round(duration * 1000, 2),
        fallback=fallback if fallback else None,
        degraded=is_degraded,
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
