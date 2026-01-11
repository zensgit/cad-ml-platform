"""Health & cache/faiss status endpoints extracted from analyze.py to reduce file size."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.api.dependencies import get_admin_token, get_api_key
from src.api.health_models import HealthResponse
from src.api.health_utils import build_health_payload

router = APIRouter()


class FeatureCacheStatsResponse(BaseModel):
    status: str
    size: int
    capacity: int
    ttl_seconds: int
    hit_ratio: Optional[float] = None
    hits: Optional[int] = None
    misses: Optional[int] = None
    evictions: Optional[int] = None


class CacheTuningRecommendation(BaseModel):
    """缓存调优建议"""

    recommended_capacity: int = Field(description="推荐的缓存容量")
    recommended_ttl_seconds: int = Field(description="推荐的TTL秒数")
    current_capacity: int = Field(description="当前容量")
    current_ttl_seconds: int = Field(description="当前TTL")
    capacity_change_pct: float = Field(description="容量变化百分比")
    ttl_change_pct: float = Field(description="TTL变化百分比")
    reasons: list[str] = Field(description="调优原因列表")
    metrics_summary: dict = Field(description="指标摘要")


class CacheApplyRequest(BaseModel):
    capacity: Optional[int] = Field(None, description="新的缓存容量")
    ttl_seconds: Optional[int] = Field(None, description="新的TTL（秒）")


class CacheApplyResponse(BaseModel):
    status: str
    applied: Optional[dict] = None
    snapshot: Optional[dict] = None
    error: Optional[dict] = None


class CacheRollbackResponse(BaseModel):
    status: str
    restored: Optional[dict] = None
    error: Optional[dict] = None


class FaissHealthResponse(BaseModel):
    available: bool
    index_size: Optional[int]
    dim: Optional[int]
    age_seconds: Optional[int]
    pending_delete: Optional[int]
    max_pending_delete: Optional[int]
    normalize: Optional[bool]
    status: str
    last_rebuild_status: Optional[str] = None
    last_error: Optional[str] = None
    degraded: bool = False  # True if Faiss fell back to memory
    degraded_reason: Optional[str] = None
    degraded_duration_seconds: Optional[float] = None
    degradation_history_count: int = 0  # Number of degradation events
    degradation_history: Optional[List[Dict]] = None  # Recent degradation events (last 10)
    next_recovery_eta: Optional[int] = None
    manual_recovery_in_progress: bool = False


@router.get("/health", response_model=HealthResponse)
async def health_alias() -> Dict[str, Any]:
    """Alias `/api/v1/health` to the root `/health` endpoint."""
    return build_health_payload()


@router.get("/features/cache", response_model=FeatureCacheStatsResponse)
async def feature_cache_stats(api_key: str = Depends(get_api_key)):
    from src.core.feature_cache import get_feature_cache
    from src.utils.analysis_metrics import feature_cache_size

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


@router.get("/features/cache/tuning", response_model=CacheTuningRecommendation)
async def cache_tuning_recommendation(api_key: str = Depends(get_api_key)):
    """
    获取缓存调优建议

    基于当前缓存指标（命中率、驱逐率、使用率等）分析并提供
    容量和TTL的优化建议。

    Returns:
        缓存调优建议，包含推荐的容量和TTL值及调优原因
    """
    from src.core.feature_cache import get_feature_cache

    cache = get_feature_cache()
    size = cache.size()
    capacity = cache.capacity
    ttl_seconds = cache.ttl_seconds

    counters = cache.stats()
    hits = counters.get("hits", 0)
    misses = counters.get("misses", 0)
    evictions = counters.get("evictions", 0)

    # Calculate metrics
    total_requests = hits + misses
    hit_ratio = hits / total_requests if total_requests > 0 else 0.0
    usage_ratio = size / capacity if capacity > 0 else 0.0
    eviction_ratio = evictions / total_requests if total_requests > 0 else 0.0

    # Initialize recommendations with current values
    recommended_capacity = capacity
    recommended_ttl = ttl_seconds
    reasons = []

    # Capacity tuning logic
    if usage_ratio > 0.9 and eviction_ratio > 0.05:
        # High usage + high evictions -> increase capacity
        recommended_capacity = int(capacity * 1.5)
        reasons.append(
            f"High cache usage ({usage_ratio:.1%}) with evictions ({eviction_ratio:.1%}) - increase capacity"
        )
    elif usage_ratio < 0.3 and eviction_ratio < 0.01:
        # Low usage + low evictions -> decrease capacity
        recommended_capacity = max(int(capacity * 0.7), 100)
        reasons.append(
            f"Low cache usage ({usage_ratio:.1%}) with minimal evictions - reduce capacity to save memory"
        )
    elif eviction_ratio > 0.15:
        # Very high evictions -> aggressive increase
        recommended_capacity = int(capacity * 2.0)
        reasons.append(f"Very high eviction rate ({eviction_ratio:.1%}) - double capacity")

    # TTL tuning logic
    if hit_ratio < 0.5 and eviction_ratio < 0.05:
        # Low hit ratio but low evictions -> entries may be stale
        recommended_ttl = max(int(ttl_seconds * 0.7), 60)
        reasons.append(
            f"Low hit ratio ({hit_ratio:.1%}) with low evictions - reduce TTL to refresh entries faster"
        )
    elif hit_ratio > 0.8 and eviction_ratio < 0.02:
        # High hit ratio + low evictions -> can extend TTL
        recommended_ttl = int(ttl_seconds * 1.3)
        reasons.append(
            f"High hit ratio ({hit_ratio:.1%}) with low evictions - extend TTL for efficiency"
        )
    elif eviction_ratio > 0.1 and usage_ratio > 0.8:
        # High evictions + high usage -> reduce TTL to free space faster
        recommended_ttl = max(int(ttl_seconds * 0.8), 60)
        reasons.append(
            f"High evictions ({eviction_ratio:.1%}) with high usage - reduce TTL to free entries"
        )

    # Add default message if no changes
    if recommended_capacity == capacity and recommended_ttl == ttl_seconds:
        reasons.append("Current cache settings are optimal based on observed metrics")

    # Calculate change percentages
    capacity_change_pct = (
        ((recommended_capacity - capacity) / capacity * 100) if capacity > 0 else 0.0
    )
    ttl_change_pct = (
        ((recommended_ttl - ttl_seconds) / ttl_seconds * 100) if ttl_seconds > 0 else 0.0
    )

    try:
        from src.utils.analysis_metrics import (
            feature_cache_tuning_recommended_capacity,
            feature_cache_tuning_recommended_ttl_seconds,
            feature_cache_tuning_requests_total,
        )

        feature_cache_tuning_recommended_capacity.set(recommended_capacity)
        feature_cache_tuning_recommended_ttl_seconds.set(recommended_ttl)
        feature_cache_tuning_requests_total.labels(status="ok").inc()
    except Exception:
        pass

    return CacheTuningRecommendation(
        recommended_capacity=recommended_capacity,
        recommended_ttl_seconds=recommended_ttl,
        current_capacity=capacity,
        current_ttl_seconds=ttl_seconds,
        capacity_change_pct=round(capacity_change_pct, 1),
        ttl_change_pct=round(ttl_change_pct, 1),
        reasons=reasons,
        metrics_summary={
            "hit_ratio": round(hit_ratio, 3),
            "usage_ratio": round(usage_ratio, 3),
            "eviction_ratio": round(eviction_ratio, 3),
            "total_requests": total_requests,
            "current_size": size,
        },
    )


@router.post("/features/cache/apply", response_model=CacheApplyResponse)
async def cache_apply(
    req: CacheApplyRequest,
    api_key: str = Depends(get_api_key),
    admin_token: str = Depends(get_admin_token),
):
    from src.core.feature_cache import apply_cache_settings

    result = apply_cache_settings(req.capacity, req.ttl_seconds)
    return CacheApplyResponse(
        status=result.get("status", "error"),
        applied=result.get("applied"),
        snapshot=result.get("snapshot"),
        error=result.get("error"),
    )


@router.get("/health/features/cache", response_model=FeatureCacheStatsResponse)
async def feature_cache_stats_health_alias(api_key: str = Depends(get_api_key)):
    return await feature_cache_stats(api_key)  # type: ignore


@router.get("/health/features/cache/tuning", response_model=CacheTuningRecommendation)
async def cache_tuning_recommendation_health_alias(api_key: str = Depends(get_api_key)):
    return await cache_tuning_recommendation(api_key)  # type: ignore


@router.post("/health/features/cache/apply", response_model=CacheApplyResponse)
async def cache_apply_health_alias(
    req: CacheApplyRequest,
    api_key: str = Depends(get_api_key),
    admin_token: str = Depends(get_admin_token),
):
    return await cache_apply(req, api_key, admin_token)  # type: ignore


@router.post("/features/cache/rollback", response_model=CacheRollbackResponse)
async def cache_rollback(
    api_key: str = Depends(get_api_key),
    admin_token: str = Depends(get_admin_token),
):
    from src.core.feature_cache import rollback_cache_settings

    result = rollback_cache_settings()
    return CacheRollbackResponse(
        status=result.get("status", "error"),
        restored=result.get("restored"),
        error=result.get("error"),
    )


@router.post("/health/features/cache/rollback", response_model=CacheRollbackResponse)
async def cache_rollback_health_alias(
    api_key: str = Depends(get_api_key),
    admin_token: str = Depends(get_admin_token),
):
    return await cache_rollback(api_key, admin_token)  # type: ignore


@router.post("/features/cache/prewarm")
async def cache_prewarm(
    strategy: str = "auto",
    limit: int = 0,
    api_key: str = Depends(get_api_key),
    admin_token: str = Depends(get_admin_token),
):
    from src.core.feature_cache import prewarm_cache

    result = prewarm_cache(strategy=strategy, limit=limit)
    return result


@router.post("/health/features/cache/prewarm")
async def cache_prewarm_health_alias(
    strategy: str = "auto",
    limit: int = 0,
    api_key: str = Depends(get_api_key),
    admin_token: str = Depends(get_admin_token),
):
    return await cache_prewarm(strategy, limit, api_key, admin_token)  # type: ignore


@router.get("/faiss/health", response_model=FaissHealthResponse)
async def faiss_health(api_key: str = Depends(get_api_key)):
    import time

    from src.core.similarity import (
        _FAISS_AVAILABLE,
        _FAISS_DIM,
        _FAISS_INDEX,
        _FAISS_LAST_EXPORT_TS,
        _FAISS_LAST_IMPORT,
        _FAISS_MANUAL_RECOVERY_IN_PROGRESS,
        _FAISS_MAX_PENDING_DELETE,
        _FAISS_NEXT_RECOVERY_TS,
        _FAISS_PENDING_DELETE,
        FaissVectorStore,
        get_degraded_mode_info,
    )

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

    # Compute next recovery ETA if scheduled
    next_recovery_eta = None
    try:
        if _FAISS_NEXT_RECOVERY_TS:
            next_recovery_eta = int(_FAISS_NEXT_RECOVERY_TS)
            try:
                from src.utils.analysis_metrics import faiss_next_recovery_eta_seconds

                faiss_next_recovery_eta_seconds.set(next_recovery_eta)
            except Exception:
                pass
        else:
            try:
                from src.utils.analysis_metrics import faiss_next_recovery_eta_seconds

                faiss_next_recovery_eta_seconds.set(0)
            except Exception:
                pass
    except Exception:
        next_recovery_eta = None

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
        next_recovery_eta=next_recovery_eta,
        manual_recovery_in_progress=bool(_FAISS_MANUAL_RECOVERY_IN_PROGRESS),
    )


@router.get("/health/faiss/health", response_model=FaissHealthResponse)
async def faiss_health_alias(api_key: str = Depends(get_api_key)):
    return await faiss_health(api_key)  # type: ignore


@router.post("/faiss/recover")
async def faiss_manual_recover(api_key: str = Depends(get_api_key)):
    """Manually trigger a Faiss recovery attempt (respects backoff)."""
    from src.core.similarity import attempt_faiss_recovery

    # Mark manual recovery in progress to coordinate with background loop
    try:
        globals()["_FAISS_MANUAL_RECOVERY_IN_PROGRESS"] = True
    except Exception:
        pass
    ok = attempt_faiss_recovery()
    # Clear the manual flag regardless of outcome
    try:
        globals()["_FAISS_MANUAL_RECOVERY_IN_PROGRESS"] = False
    except Exception:
        pass
    if not ok:
        return {"status": "skipped_or_failed"}
    return {"status": "success"}


class ModelHealthResponse(BaseModel):
    status: str
    version: Optional[str]
    hash: Optional[str]
    path: Optional[str]
    loaded: bool
    loaded_at: Optional[float] = None
    uptime_seconds: Optional[float] = None
    last_error: Optional[str] = None
    rollback_level: int = 0
    rollback_reason: Optional[str] = None
    load_seq: int = 0  # Monotonic load sequence for disambiguation


@router.get("/health/model", response_model=ModelHealthResponse)
async def model_health(api_key: str = Depends(get_api_key)):
    import time

    from src.ml.classifier import get_model_info
    from src.utils.analysis_metrics import model_health_checks_total

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
    try:
        from src.utils.analysis_metrics import model_rollback_level, model_snapshots_available

        snapshots_available = sum(
            1
            for flag in (
                info.get("has_prev"),
                info.get("has_prev2"),
                info.get("has_prev3"),
            )
            if flag
        )
        model_rollback_level.set(rollback_level)
        model_snapshots_available.set(snapshots_available)
    except Exception:
        pass

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
