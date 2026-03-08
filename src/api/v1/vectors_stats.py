"""Vector statistics & distribution endpoints extracted from analyze.py."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.dependencies import get_api_key
from src.utils.cache import get_client

router = APIRouter()


def _get_qdrant_store_or_none():
    if os.getenv("VECTOR_STORE_BACKEND", "memory") != "qdrant":
        return None
    try:
        from src.core.vector_stores import get_vector_store as get_managed_vector_store

        return get_managed_vector_store("qdrant")
    except Exception:
        return None


class VectorStatsResponse(BaseModel):
    backend: str
    total: int
    by_material: Dict[str, int]
    by_complexity: Dict[str, int]
    by_format: Dict[str, int]
    by_coarse_part_type: Optional[Dict[str, int]] = None
    by_decision_source: Optional[Dict[str, int]] = None
    versions: Optional[Dict[str, int]] = None
    backend_health: Optional[Dict[str, Any]] = None


class VectorDistributionResponse(BaseModel):
    total: int
    by_material: Dict[str, int]
    by_complexity: Dict[str, int]
    by_format: Dict[str, int]
    by_coarse_part_type: Optional[Dict[str, int]] = None
    by_decision_source: Optional[Dict[str, int]] = None
    dominant_ratio: float
    dominant_coarse_ratio: float = 0.0
    feature_version: str
    average_dimension: Optional[float] = None
    versions: Optional[Dict[str, int]] = None
    backend_health: Optional[Dict[str, Any]] = None


@router.get("/stats", response_model=VectorStatsResponse)
async def vector_stats(api_key: str = Depends(get_api_key)):
    from src.core.similarity import _BACKEND, _VECTOR_META, _VECTOR_STORE  # type: ignore

    (
        total,
        by_material,
        by_complexity,
        by_format,
        by_coarse_part_type,
        by_decision_source,
        versions,
        _,
    ) = await _summarize_vectors(
        backend=_BACKEND,
        vector_store=_VECTOR_STORE,
        vector_meta=_VECTOR_META,
    )
    backend_health = await _build_backend_health(backend=_BACKEND, observed_count=total)
    return VectorStatsResponse(
        backend=_BACKEND,
        total=total,
        by_material=by_material,
        by_complexity=by_complexity,
        by_format=by_format,
        by_coarse_part_type=by_coarse_part_type,
        by_decision_source=by_decision_source,
        versions=versions,
        backend_health=backend_health,
    )


@router.get("/distribution", response_model=VectorDistributionResponse)
async def vector_distribution(api_key: str = Depends(get_api_key)):
    from src.core.similarity import _BACKEND, _VECTOR_META, _VECTOR_STORE  # type: ignore

    (
        total,
        by_material,
        by_complexity,
        by_format,
        by_coarse_part_type,
        by_decision_source,
        versions,
        avg_dim,
    ) = await _summarize_vectors(
        backend=_BACKEND,
        vector_store=_VECTOR_STORE,
        vector_meta=_VECTOR_META,
    )
    backend_health = await _build_backend_health(backend=_BACKEND, observed_count=total)
    dominant = max(by_material.values()) if by_material else 0
    dominant_ratio = (dominant / total) if total else 0.0
    dominant_coarse = max(by_coarse_part_type.values()) if by_coarse_part_type else 0
    dominant_coarse_ratio = (dominant_coarse / total) if total else 0.0
    feature_version = os.getenv("FEATURE_VERSION", "v1")
    return VectorDistributionResponse(
        total=total,
        by_material=by_material,
        by_complexity=by_complexity,
        by_format=by_format,
        by_coarse_part_type=by_coarse_part_type,
        by_decision_source=by_decision_source,
        dominant_ratio=round(dominant_ratio, 4),
        dominant_coarse_ratio=round(dominant_coarse_ratio, 4),
        feature_version=feature_version,
        average_dimension=round(avg_dim, 3) if total else 0.0,
        versions=versions,
        backend_health=backend_health,
    )


async def _build_backend_health(backend: str, observed_count: int) -> Optional[Dict[str, Any]]:
    if backend != "qdrant":
        return None
    qdrant_store = _get_qdrant_store_or_none()
    if qdrant_store is None:
        return None
    return await _build_qdrant_backend_health(
        qdrant_store=qdrant_store,
        observed_count=observed_count,
    )


async def _build_qdrant_backend_health(
    *,
    qdrant_store,
    observed_count: int,
) -> Dict[str, Any]:
    stats = await qdrant_store.get_stats()
    points_count = int(stats.get("points_count") or stats.get("vectors_count") or 0)
    indexed_vectors_count = int(stats.get("indexed_vectors_count") or 0)
    scan_limit = int(os.getenv("VECTOR_STATS_SCAN_LIMIT", "5000"))
    scan_truncated = bool(scan_limit > 0 and points_count > observed_count)
    unindexed_vectors_count = max(points_count - indexed_vectors_count, 0)
    indexed_ratio = round((indexed_vectors_count / points_count), 4) if points_count else 0.0
    readiness_hints = _build_qdrant_readiness_hints(
        stats=stats,
        scan_truncated=scan_truncated,
        unindexed_vectors_count=unindexed_vectors_count,
    )
    readiness = _classify_qdrant_readiness(
        stats=stats,
        points_count=points_count,
        scan_truncated=scan_truncated,
        unindexed_vectors_count=unindexed_vectors_count,
    )
    config = stats.get("config") or {}
    return {
        "collection_name": stats.get("collection_name"),
        "collection_status": stats.get("status"),
        "points_count": points_count,
        "indexed_vectors_count": indexed_vectors_count,
        "unindexed_vectors_count": unindexed_vectors_count,
        "indexed_ratio": indexed_ratio,
        "observed_vectors_count": observed_count,
        "scan_limit": scan_limit,
        "scan_truncated": scan_truncated,
        "vector_size": config.get("vector_size"),
        "distance": config.get("distance"),
        "readiness": readiness,
        "readiness_hints": readiness_hints,
    }


def _build_qdrant_readiness_hints(
    *,
    stats: Dict[str, Any],
    scan_truncated: bool,
    unindexed_vectors_count: int,
) -> List[str]:
    hints: List[str] = []
    if scan_truncated:
        hints.append("scan_truncated_use_list_or_migration_for_exact_coverage")
    if unindexed_vectors_count > 0:
        hints.append("vector_index_backfill_in_progress")
    status = str(stats.get("status") or "").strip().lower()
    if status and status not in {"green", "ok", "ready"}:
        hints.append(f"collection_status_{status}")
    if int(stats.get("points_count") or stats.get("vectors_count") or 0) <= 0:
        hints.append("collection_empty")
    return hints


def _classify_qdrant_readiness(
    *,
    stats: Dict[str, Any],
    points_count: int,
    scan_truncated: bool,
    unindexed_vectors_count: int,
) -> str:
    status = str(stats.get("status") or "").strip().lower()
    if points_count <= 0:
        return "empty"
    if status in {"red", "error", "failed"}:
        return "degraded"
    if scan_truncated:
        return "partial_scan"
    if unindexed_vectors_count > 0:
        return "indexing"
    return "ready"


async def _summarize_vectors(
    *,
    backend: str,
    vector_store: Dict[str, list[float]],
    vector_meta: Dict[str, Dict[str, str]],
) -> Tuple[
    int,
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    float,
]:
    if backend == "qdrant":
        qdrant_store = _get_qdrant_store_or_none()
        if qdrant_store is not None:
            return await _summarize_vectors_qdrant(qdrant_store)
    if backend == "redis":
        client = get_client()
        if client is not None:
            return await _summarize_vectors_redis(client)
    return _summarize_vectors_memory(vector_store, vector_meta)


def _summarize_vectors_memory(
    vector_store: Dict[str, list[float]],
    vector_meta: Dict[str, Dict[str, str]],
) -> Tuple[
    int,
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    float,
]:
    from src.core.similarity import extract_vector_label_contract

    total = len(vector_store)
    by_material: Dict[str, int] = {}
    by_complexity: Dict[str, int] = {}
    by_format: Dict[str, int] = {}
    by_coarse_part_type: Dict[str, int] = {}
    by_decision_source: Dict[str, int] = {}
    versions: Dict[str, int] = {}
    for meta in vector_meta.values():
        m = meta.get("material", "unknown")
        c = meta.get("complexity", "unknown")
        f = meta.get("format", "unknown")
        ver = meta.get("feature_version", "unknown")
        label_contract = extract_vector_label_contract(meta)
        coarse_label = label_contract.get("coarse_part_type") or "unknown"
        decision_source = label_contract.get("decision_source") or "unknown"
        by_material[m] = by_material.get(m, 0) + 1
        by_complexity[c] = by_complexity.get(c, 0) + 1
        by_format[f] = by_format.get(f, 0) + 1
        by_coarse_part_type[coarse_label] = by_coarse_part_type.get(coarse_label, 0) + 1
        by_decision_source[decision_source] = by_decision_source.get(decision_source, 0) + 1
        versions[ver] = versions.get(ver, 0) + 1
    avg_dim = 0.0
    if total:
        dims = [len(v) for v in vector_store.values()]
        avg_dim = sum(dims) / len(dims)
    return (
        total,
        by_material,
        by_complexity,
        by_format,
        by_coarse_part_type,
        by_decision_source,
        versions,
        avg_dim,
    )


async def _summarize_vectors_redis(
    client,
) -> Tuple[
    int,
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    float,
]:
    from src.core.similarity import extract_vector_label_contract

    total = 0
    dim_sum = 0.0
    by_material: Dict[str, int] = {}
    by_complexity: Dict[str, int] = {}
    by_format: Dict[str, int] = {}
    by_coarse_part_type: Dict[str, int] = {}
    by_decision_source: Dict[str, int] = {}
    versions: Dict[str, int] = {}
    cursor = 0
    scanned = 0
    scan_limit = int(os.getenv("VECTOR_STATS_SCAN_LIMIT", "5000"))
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
            vec_parts = [p for p in str(raw_vec).split(",") if p]
            total += 1
            dim_sum += len(vec_parts)
            raw_meta = data.get("m") or data.get(b"m")
            meta: Dict[str, str] = {}
            if raw_meta:
                try:
                    meta = json.loads(raw_meta)
                except Exception:
                    meta = {}
            m = meta.get("material", "unknown")
            c = meta.get("complexity", "unknown")
            f = meta.get("format", "unknown")
            ver = meta.get("feature_version", "unknown")
            label_contract = extract_vector_label_contract(meta)
            coarse_label = label_contract.get("coarse_part_type") or "unknown"
            decision_source = label_contract.get("decision_source") or "unknown"
            by_material[m] = by_material.get(m, 0) + 1
            by_complexity[c] = by_complexity.get(c, 0) + 1
            by_format[f] = by_format.get(f, 0) + 1
            by_coarse_part_type[coarse_label] = by_coarse_part_type.get(coarse_label, 0) + 1
            by_decision_source[decision_source] = by_decision_source.get(decision_source, 0) + 1
            versions[ver] = versions.get(ver, 0) + 1
        if cursor == 0:
            break
    avg_dim = (dim_sum / total) if total else 0.0
    return (
        total,
        by_material,
        by_complexity,
        by_format,
        by_coarse_part_type,
        by_decision_source,
        versions,
        avg_dim,
    )


async def _summarize_vectors_qdrant(
    qdrant_store,
) -> Tuple[
    int,
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    float,
]:
    from src.core.similarity import extract_vector_label_contract

    scan_limit = int(os.getenv("VECTOR_STATS_SCAN_LIMIT", "5000"))
    results, _ = await qdrant_store.list_vectors(
        offset=0,
        limit=scan_limit,
        with_vectors=True,
    )

    total = len(results)
    dim_sum = 0.0
    by_material: Dict[str, int] = {}
    by_complexity: Dict[str, int] = {}
    by_format: Dict[str, int] = {}
    by_coarse_part_type: Dict[str, int] = {}
    by_decision_source: Dict[str, int] = {}
    versions: Dict[str, int] = {}

    for result in results:
        meta = result.metadata or {}
        m = meta.get("material", "unknown")
        c = meta.get("complexity", "unknown")
        f = meta.get("format", "unknown")
        ver = meta.get("feature_version", "unknown")
        label_contract = extract_vector_label_contract(meta)
        coarse_label = label_contract.get("coarse_part_type") or "unknown"
        decision_source = label_contract.get("decision_source") or "unknown"
        by_material[m] = by_material.get(m, 0) + 1
        by_complexity[c] = by_complexity.get(c, 0) + 1
        by_format[f] = by_format.get(f, 0) + 1
        by_coarse_part_type[coarse_label] = by_coarse_part_type.get(coarse_label, 0) + 1
        by_decision_source[decision_source] = by_decision_source.get(decision_source, 0) + 1
        versions[ver] = versions.get(ver, 0) + 1
        dim_sum += len(result.vector or [])

    avg_dim = (dim_sum / total) if total else 0.0
    return (
        total,
        by_material,
        by_complexity,
        by_format,
        by_coarse_part_type,
        by_decision_source,
        versions,
        avg_dim,
    )


__all__ = ["router"]
