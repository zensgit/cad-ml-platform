"""Vector statistics & distribution endpoints extracted from analyze.py."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.dependencies import get_api_key
from src.utils.cache import get_client

router = APIRouter()


class VectorStatsResponse(BaseModel):
    backend: str
    total: int
    by_material: Dict[str, int]
    by_complexity: Dict[str, int]
    by_format: Dict[str, int]
    versions: Optional[Dict[str, int]] = None


class VectorDistributionResponse(BaseModel):
    total: int
    by_material: Dict[str, int]
    by_complexity: Dict[str, int]
    by_format: Dict[str, int]
    dominant_ratio: float
    feature_version: str
    average_dimension: Optional[float] = None
    versions: Optional[Dict[str, int]] = None


@router.get("/stats", response_model=VectorStatsResponse)
async def vector_stats(api_key: str = Depends(get_api_key)):
    from src.core.similarity import _VECTOR_STORE, _VECTOR_META, _BACKEND  # type: ignore

    total, by_material, by_complexity, by_format, versions, _ = await _summarize_vectors(
        backend=_BACKEND,
        vector_store=_VECTOR_STORE,
        vector_meta=_VECTOR_META,
    )
    return VectorStatsResponse(
        backend=_BACKEND,
        total=total,
        by_material=by_material,
        by_complexity=by_complexity,
        by_format=by_format,
        versions=versions,
    )


@router.get("/distribution", response_model=VectorDistributionResponse)
async def vector_distribution(api_key: str = Depends(get_api_key)):
    from src.core.similarity import _VECTOR_META, _VECTOR_STORE, _BACKEND  # type: ignore

    total, by_material, by_complexity, by_format, versions, avg_dim = await _summarize_vectors(
        backend=_BACKEND,
        vector_store=_VECTOR_STORE,
        vector_meta=_VECTOR_META,
    )
    dominant = max(by_material.values()) if by_material else 0
    dominant_ratio = (dominant / total) if total else 0.0
    feature_version = os.getenv("FEATURE_VERSION", "v1")
    return VectorDistributionResponse(
        total=total,
        by_material=by_material,
        by_complexity=by_complexity,
        by_format=by_format,
        dominant_ratio=round(dominant_ratio, 4),
        feature_version=feature_version,
        average_dimension=round(avg_dim, 3) if total else 0.0,
        versions=versions,
    )


async def _summarize_vectors(
    *,
    backend: str,
    vector_store: Dict[str, list[float]],
    vector_meta: Dict[str, Dict[str, str]],
) -> Tuple[int, Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int], float]:
    if backend == "redis":
        client = get_client()
        if client is not None:
            return await _summarize_vectors_redis(client)
    return _summarize_vectors_memory(vector_store, vector_meta)


def _summarize_vectors_memory(
    vector_store: Dict[str, list[float]],
    vector_meta: Dict[str, Dict[str, str]],
) -> Tuple[int, Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int], float]:
    total = len(vector_store)
    by_material: Dict[str, int] = {}
    by_complexity: Dict[str, int] = {}
    by_format: Dict[str, int] = {}
    versions: Dict[str, int] = {}
    for meta in vector_meta.values():
        m = meta.get("material", "unknown")
        c = meta.get("complexity", "unknown")
        f = meta.get("format", "unknown")
        ver = meta.get("feature_version", "unknown")
        by_material[m] = by_material.get(m, 0) + 1
        by_complexity[c] = by_complexity.get(c, 0) + 1
        by_format[f] = by_format.get(f, 0) + 1
        versions[ver] = versions.get(ver, 0) + 1
    avg_dim = 0.0
    if total:
        dims = [len(v) for v in vector_store.values()]
        avg_dim = sum(dims) / len(dims)
    return total, by_material, by_complexity, by_format, versions, avg_dim


async def _summarize_vectors_redis(
    client,
) -> Tuple[int, Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int], float]:
    total = 0
    dim_sum = 0.0
    by_material: Dict[str, int] = {}
    by_complexity: Dict[str, int] = {}
    by_format: Dict[str, int] = {}
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
            by_material[m] = by_material.get(m, 0) + 1
            by_complexity[c] = by_complexity.get(c, 0) + 1
            by_format[f] = by_format.get(f, 0) + 1
            versions[ver] = versions.get(ver, 0) + 1
        if cursor == 0:
            break
    avg_dim = (dim_sum / total) if total else 0.0
    return total, by_material, by_complexity, by_format, versions, avg_dim


__all__ = ["router"]
