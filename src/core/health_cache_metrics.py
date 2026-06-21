"""Pure cache metric/tuning helpers for the health API (no FastAPI/metrics).

Extracted from src/api/v1/health.py (behavior-preserving router slimming). The
router keeps the FastAPI handlers + Prometheus side effects and calls these pure
functions; they are re-exported from health.py for compatibility and testing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def compute_hit_ratio(hits: Optional[int], misses: Optional[int]) -> Optional[float]:
    """hits / (hits + misses), or None when unknown or no requests yet."""
    if hits is None or misses is None:
        return None
    total = hits + misses
    return (hits / total) if total > 0 else None


def compute_cache_tuning(
    *,
    size: int,
    capacity: int,
    ttl_seconds: int,
    hits: int,
    misses: int,
    evictions: int,
) -> Dict[str, Any]:
    """Stateless cache capacity/TTL recommendation from observed metrics.

    Returns a dict whose keys match CacheTuningRecommendation's fields, so the
    handler can do `CacheTuningRecommendation(**result)`. Verbatim port of the
    prior inline handler logic — thresholds and messages unchanged.
    """
    total_requests = hits + misses
    hit_ratio = hits / total_requests if total_requests > 0 else 0.0
    usage_ratio = size / capacity if capacity > 0 else 0.0
    eviction_ratio = evictions / total_requests if total_requests > 0 else 0.0

    recommended_capacity = capacity
    recommended_ttl = ttl_seconds
    reasons = []

    # Capacity tuning logic
    if usage_ratio > 0.9 and eviction_ratio > 0.05:
        recommended_capacity = int(capacity * 1.5)
        reasons.append(
            f"High cache usage ({usage_ratio:.1%}) with evictions ({eviction_ratio:.1%}) - increase capacity"
        )
    elif usage_ratio < 0.3 and eviction_ratio < 0.01:
        recommended_capacity = max(int(capacity * 0.7), 100)
        reasons.append(
            f"Low cache usage ({usage_ratio:.1%}) with minimal evictions - reduce capacity to save memory"
        )
    elif eviction_ratio > 0.15:
        recommended_capacity = int(capacity * 2.0)
        reasons.append(
            f"Very high eviction rate ({eviction_ratio:.1%}) - double capacity"
        )

    # TTL tuning logic
    if hit_ratio < 0.5 and eviction_ratio < 0.05:
        recommended_ttl = max(int(ttl_seconds * 0.7), 60)
        reasons.append(
            f"Low hit ratio ({hit_ratio:.1%}) with low evictions - reduce TTL to refresh entries faster"
        )
    elif hit_ratio > 0.8 and eviction_ratio < 0.02:
        recommended_ttl = int(ttl_seconds * 1.3)
        reasons.append(
            f"High hit ratio ({hit_ratio:.1%}) with low evictions - extend TTL for efficiency"
        )
    elif eviction_ratio > 0.1 and usage_ratio > 0.8:
        recommended_ttl = max(int(ttl_seconds * 0.8), 60)
        reasons.append(
            f"High evictions ({eviction_ratio:.1%}) with high usage - reduce TTL to free entries"
        )

    if recommended_capacity == capacity and recommended_ttl == ttl_seconds:
        reasons.append("Current cache settings are optimal based on observed metrics")

    capacity_change_pct = (
        ((recommended_capacity - capacity) / capacity * 100) if capacity > 0 else 0.0
    )
    ttl_change_pct = (
        ((recommended_ttl - ttl_seconds) / ttl_seconds * 100)
        if ttl_seconds > 0
        else 0.0
    )

    return {
        "recommended_capacity": recommended_capacity,
        "recommended_ttl_seconds": recommended_ttl,
        "current_capacity": capacity,
        "current_ttl_seconds": ttl_seconds,
        "capacity_change_pct": round(capacity_change_pct, 1),
        "ttl_change_pct": round(ttl_change_pct, 1),
        "reasons": reasons,
        "metrics_summary": {
            "hit_ratio": round(hit_ratio, 3),
            "usage_ratio": round(usage_ratio, 3),
            "eviction_ratio": round(eviction_ratio, 3),
            "total_requests": total_requests,
            "current_size": size,
        },
    }
