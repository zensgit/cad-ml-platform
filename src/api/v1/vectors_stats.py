"""Vector statistics & distribution endpoints extracted from analyze.py."""

from __future__ import annotations

import os
from typing import Dict, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.dependencies import get_api_key

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


@router.get("/vectors/stats", response_model=VectorStatsResponse)
async def vector_stats(api_key: str = Depends(get_api_key)):
    from src.core.similarity import _VECTOR_STORE, _VECTOR_META, _BACKEND  # type: ignore

    total = len(_VECTOR_STORE)
    by_material: Dict[str, int] = {}
    by_complexity: Dict[str, int] = {}
    by_format: Dict[str, int] = {}
    versions: Dict[str, int] = {}
    for meta in _VECTOR_META.values():
        m = meta.get("material", "unknown")
        c = meta.get("complexity", "unknown")
        f = meta.get("format", "unknown")
        ver = meta.get("feature_version", "unknown")
        by_material[m] = by_material.get(m, 0) + 1
        by_complexity[c] = by_complexity.get(c, 0) + 1
        by_format[f] = by_format.get(f, 0) + 1
        versions[ver] = versions.get(ver, 0) + 1
    return VectorStatsResponse(
        backend=_BACKEND,
        total=total,
        by_material=by_material,
        by_complexity=by_complexity,
        by_format=by_format,
        versions=versions,
    )


@router.get("/vectors/distribution", response_model=VectorDistributionResponse)
async def vector_distribution(api_key: str = Depends(get_api_key)):
    from src.core.similarity import _VECTOR_META, _VECTOR_STORE  # type: ignore

    total = len(_VECTOR_STORE)
    by_material: Dict[str, int] = {}
    by_complexity: Dict[str, int] = {}
    by_format: Dict[str, int] = {}
    versions: Dict[str, int] = {}
    for meta in _VECTOR_META.values():
        m = meta.get("material", "unknown")
        c = meta.get("complexity", "unknown")
        f = meta.get("format", "unknown")
        by_material[m] = by_material.get(m, 0) + 1
        by_complexity[c] = by_complexity.get(c, 0) + 1
        by_format[f] = by_format.get(f, 0) + 1
        ver = meta.get("feature_version", "unknown")
        versions[ver] = versions.get(ver, 0) + 1
    dominant = max(by_material.values()) if by_material else 0
    dominant_ratio = (dominant / total) if total else 0.0
    feature_version = os.getenv("FEATURE_VERSION", "v1")
    avg_dim = 0.0
    if total:
        dims = [len(v) for v in _VECTOR_STORE.values()]
        avg_dim = sum(dims) / len(dims)
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


__all__ = ["router"]
