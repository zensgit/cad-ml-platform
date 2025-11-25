"""Health & cache/faiss status endpoints extracted from analyze.py to reduce file size."""

from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.dependencies import get_api_key

router = APIRouter()


class FeatureCacheStatsResponse(BaseModel):
    status: str
    size: int
    capacity: int
    ttl_seconds: int
    hit_ratio: float | None = None
    hits: int | None = None
    misses: int | None = None
    evictions: int | None = None


class FaissHealthResponse(BaseModel):
    available: bool
    index_size: int | None
    dim: int | None
    age_seconds: int | None
    pending_delete: int | None
    max_pending_delete: int | None
    normalize: bool | None
    status: str
    last_rebuild_status: str | None = None
    last_error: str | None = None
    degraded: bool = False  # True if Faiss fell back to memory
    degraded_reason: str | None = None
    degraded_duration_seconds: float | None = None
    degradation_history_count: int = 0  # Number of degradation events
    degradation_history: list | None = None  # Recent degradation events (last 10)


@router.get("/features/cache", response_model=FeatureCacheStatsResponse)
async def feature_cache_stats(api_key: str = Depends(get_api_key)):
    from src.core.feature_cache import get_feature_cache
    from src.utils.analysis_metrics import (
        feature_cache_size,
    )

    cache = get_feature_cache()
    size = cache.size()
    # update size gauge (best-effort if client absent)
    try:
        feature_cache_size.set(size)
    except Exception:
        pass

    counters = cache.stats()
    hits = counters.get("hits")
    misses = counters.get("misses")
    evictions = counters.get("evictions")
    hit_ratio = None
    if hits is not None and misses is not None:
        total = hits + misses
        if total > 0:
            hit_ratio = hits / total
    return FeatureCacheStatsResponse(
        status="ok",
        size=size,
        capacity=cache.capacity,
        ttl_seconds=cache.ttl_seconds,
        hit_ratio=hit_ratio,
        hits=hits,
        misses=misses,
        evictions=evictions,
    )


@router.get("/faiss/health", response_model=FaissHealthResponse)
async def faiss_health(api_key: str = Depends(get_api_key)):
    from src.core.similarity import (
        _FAISS_AVAILABLE,
        _FAISS_INDEX,
        _FAISS_DIM,
        _FAISS_PENDING_DELETE,
        _FAISS_MAX_PENDING_DELETE,
        FaissVectorStore,
        _FAISS_LAST_EXPORT_TS,
        _FAISS_LAST_IMPORT,
        get_degraded_mode_info,
    )
    import time

    available = bool(_FAISS_AVAILABLE)
    size = None
    if available and _FAISS_INDEX is not None:
        try:
            size = _FAISS_INDEX.ntotal  # type: ignore
        except Exception:
            size = None
    dim = _FAISS_DIM
    age_seconds = None
    now = time.time()
    if _FAISS_LAST_EXPORT_TS:
        age_seconds = int(now - _FAISS_LAST_EXPORT_TS)
    elif _FAISS_LAST_IMPORT:
        age_seconds = int(now - _FAISS_LAST_IMPORT)
    pending = len(_FAISS_PENDING_DELETE) if available else None
    store = FaissVectorStore()
    normalize = store._normalize if available else None  # type: ignore[attr-defined]

    # Get degraded mode information
    degraded_info = get_degraded_mode_info()

    # Determine status: degraded > unavailable > ok
    if degraded_info["degraded"]:
        status = "degraded"
    elif not available:
        status = "unavailable"
    else:
        status = "ok"

    last_rebuild_status = globals().get("_FAISS_LAST_REBUILD_STATUS")
    last_error = globals().get("_FAISS_LAST_ERROR")

    return FaissHealthResponse(
        available=available,
        index_size=size,
        dim=dim,
        age_seconds=age_seconds,
        pending_delete=pending,
        max_pending_delete=_FAISS_MAX_PENDING_DELETE if available else None,
        normalize=normalize,
        status=status,
        last_rebuild_status=str(last_rebuild_status) if last_rebuild_status else None,
        last_error=str(last_error) if last_error else None,
        degraded=degraded_info["degraded"],
        degraded_reason=degraded_info["reason"],
        degraded_duration_seconds=degraded_info["degraded_duration_seconds"],
        degradation_history_count=degraded_info["history_count"],
        degradation_history=degraded_info["history"] if degraded_info["history"] else None,
    )


class ModelHealthResponse(BaseModel):
    status: str
    version: str | None
    hash: str | None
    path: str | None
    loaded: bool
    loaded_at: float | None = None
    uptime_seconds: float | None = None
    last_error: str | None = None
    rollback_level: int = 0
    rollback_reason: str | None = None
    load_seq: int = 0  # Monotonic load sequence for disambiguation


@router.get("/health/model", response_model=ModelHealthResponse)
async def model_health(api_key: str = Depends(get_api_key)):
    from src.ml.classifier import get_model_info
    from src.utils.analysis_metrics import model_health_checks_total
    import time
    info = get_model_info()

    # Determine status based on loaded state and rollback level
    rollback_level = info.get("rollback_level", 0)
    if not info.get("loaded"):
        status = "absent"
    elif rollback_level > 0:
        status = "rollback"
    else:
        status = "ok"

    model_health_checks_total.labels(status=status).inc()

    uptime = None
    if info.get("loaded_at"):
        try:
            uptime = time.time() - float(info.get("loaded_at"))
        except Exception:
            uptime = None

    return ModelHealthResponse(
        status=status,
        version=info.get("version"),
        hash=info.get("hash"),
        path=info.get("path"),
        loaded=bool(info.get("loaded")),
        loaded_at=info.get("loaded_at"),
        uptime_seconds=uptime,
        last_error=info.get("last_error"),
        rollback_level=rollback_level,
        rollback_reason=info.get("rollback_reason"),
        load_seq=info.get("load_seq", 0),
    )


__all__ = ["router"]
