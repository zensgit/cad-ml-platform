from __future__ import annotations

import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union


class FeatureCache:
    """LRU Feature vector cache keyed by content hash + feature version.

    Entry value: (vector: List[float], stored_at: float)
    Expiry: ttl_seconds (0 disables TTL)
    """

    def __init__(self, capacity: int = 256, ttl_seconds: int = 0) -> None:
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._store: "OrderedDict[str, Tuple[List[float], float]]" = OrderedDict()
        # Internal counters to avoid relying on prometheus client internals
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    def _evict_if_needed(self) -> int:
        evicted = 0
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)
            evicted += 1
        return evicted

    def _purge_expired(self) -> int:
        if self.ttl_seconds <= 0:
            return 0
        now = time.time()
        expired_keys = [k for k, (_, ts) in self._store.items() if now - ts > self.ttl_seconds]
        for k in expired_keys:
            self._store.pop(k, None)
        return len(expired_keys)

    def get(self, key: str) -> Optional[List[float]]:
        self._purge_expired()
        if key not in self._store:
            self._misses += 1
            try:
                from src.utils.analysis_metrics import feature_cache_miss_total

                feature_cache_miss_total.inc()
            except Exception:
                pass
            return None
        vec, ts = self._store.pop(key)
        # reinsert as most recently used
        self._store[key] = (vec, ts)
        self._hits += 1
        try:
            from src.utils.analysis_metrics import feature_cache_hits_total

            feature_cache_hits_total.inc()
        except Exception:
            pass
        return vec

    def set(self, key: str, vector: List[float]) -> None:
        self._purge_expired()
        if key in self._store:
            self._store.pop(key)
        self._store[key] = (vector, time.time())
        evicted = self._evict_if_needed()
        if evicted:
            self._evictions += evicted
            try:
                from src.utils.analysis_metrics import feature_cache_evictions_total

                feature_cache_evictions_total.inc(evicted)
            except Exception:
                pass

    def size(self) -> int:
        return len(self._store)

    # Expose internal counters for stats endpoint
    def stats(self) -> Dict[str, int]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
        }

    # Runtime configuration updates
    def update_settings(
        self, *, capacity: Optional[int] = None, ttl_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """Update cache capacity and/or TTL at runtime.

        Returns a dict with previous and new settings plus evicted count if any.
        """
        prev = {"capacity": self.capacity, "ttl_seconds": self.ttl_seconds}
        if capacity is not None and capacity >= 1:
            self.capacity = int(capacity)
        if ttl_seconds is not None and ttl_seconds >= 0:
            self.ttl_seconds = int(ttl_seconds)
        evicted = self._evict_if_needed()
        return {
            "previous": prev,
            "current": {"capacity": self.capacity, "ttl_seconds": self.ttl_seconds},
            "evicted": evicted,
        }


# Global singleton (simple usage pattern for current scope)
_FEATURE_CACHE: Optional[FeatureCache] = None

# Snapshot for apply/rollback window
_CACHE_PREV_SNAPSHOT: Optional[Dict[str, Any]] = None
_CACHE_ROLLBACK_WINDOW_SECONDS = 5 * 60  # 5 minutes


def get_feature_cache() -> FeatureCache:
    global _FEATURE_CACHE
    if _FEATURE_CACHE is None:
        from os import getenv

        cap = int(getenv("FEATURE_CACHE_CAPACITY", "256"))
        ttl = int(getenv("FEATURE_CACHE_TTL_SECONDS", "0"))
        _FEATURE_CACHE = FeatureCache(capacity=cap, ttl_seconds=ttl)
    return _FEATURE_CACHE


def reset_feature_cache_for_tests() -> None:
    """Reset global cache for test isolation (not used in production paths)."""
    global _FEATURE_CACHE, _CACHE_PREV_SNAPSHOT
    _FEATURE_CACHE = None
    _CACHE_PREV_SNAPSHOT = None


def apply_cache_settings(capacity: Optional[int], ttl_seconds: Optional[int]) -> Dict[str, Any]:
    """Apply new cache settings with a rollback snapshot window.

    Reject if a snapshot is still within the rollback window.
    """
    global _CACHE_PREV_SNAPSHOT
    cache = get_feature_cache()
    now = datetime.now(timezone.utc)

    # If there is an active snapshot and still within window, reject
    if _CACHE_PREV_SNAPSHOT is not None:
        until = _CACHE_PREV_SNAPSHOT.get("can_rollback_until")
        if isinstance(until, datetime) and until > now:
            return {
                "status": "window_active",
                "error": {
                    "code": "CACHE_TUNING_ROLLBACK_WINDOW_ACTIVE",
                    "message": "Rollback window active; rollback or wait before applying new settings",
                    "context": {
                        "can_rollback_until": until.isoformat(),
                    },
                },
            }

    # Create snapshot of current settings
    snapshot = {
        "previous_capacity": cache.capacity,
        "previous_ttl": cache.ttl_seconds,
        "applied_at": now,
        "can_rollback_until": now + timedelta(seconds=_CACHE_ROLLBACK_WINDOW_SECONDS),
    }

    result = cache.update_settings(capacity=capacity, ttl_seconds=ttl_seconds)
    _CACHE_PREV_SNAPSHOT = snapshot
    return {
        "status": "applied",
        "applied": {
            "capacity": result["current"]["capacity"],
            "ttl_seconds": result["current"]["ttl_seconds"],
            "evicted": result["evicted"],
        },
        "snapshot": {
            "previous_capacity": snapshot["previous_capacity"],
            "previous_ttl": snapshot["previous_ttl"],
            "applied_at": snapshot["applied_at"].isoformat(),
            "can_rollback_until": snapshot["can_rollback_until"].isoformat(),
        },
    }


def rollback_cache_settings() -> Dict[str, Any]:
    """Rollback to previous cache settings if within the time window."""
    global _CACHE_PREV_SNAPSHOT
    cache = get_feature_cache()
    now = datetime.now(timezone.utc)

    if _CACHE_PREV_SNAPSHOT is None:
        return {
            "status": "no_snapshot",
            "error": {
                "code": "CACHE_TUNING_NO_SNAPSHOT",
                "message": "No previous cache settings snapshot to rollback to",
            },
        }

    until = _CACHE_PREV_SNAPSHOT.get("can_rollback_until")
    if not isinstance(until, datetime) or until <= now:
        # Snapshot expired
        _CACHE_PREV_SNAPSHOT = None
        return {
            "status": "expired",
            "error": {
                "code": "CACHE_TUNING_ROLLBACK_EXPIRED",
                "message": "Rollback window expired",
            },
        }

    prev_cap = int(_CACHE_PREV_SNAPSHOT.get("previous_capacity", cache.capacity))
    prev_ttl = int(_CACHE_PREV_SNAPSHOT.get("previous_ttl", cache.ttl_seconds))
    result = cache.update_settings(capacity=prev_cap, ttl_seconds=prev_ttl)
    # Clear snapshot after rollback
    _CACHE_PREV_SNAPSHOT = None
    return {
        "status": "rolled_back",
        "restored": {
            "capacity": result["current"]["capacity"],
            "ttl_seconds": result["current"]["ttl_seconds"],
        },
    }


def prewarm_cache(strategy: str = "auto", limit: int = 0) -> Dict[str, Any]:
    """Best-effort prewarm.

    Current implementation is limited: as the source of truth for feature vectors
    is external, we only "touch" existing keys to mark them as recently used.
    Returns counts for observability.
    """
    cache = get_feature_cache()
    touched = 0
    if cache._store:
        # Touch up to limit keys (0 means all)
        keys = list(cache._store.keys())
        if limit and limit > 0:
            keys = keys[:limit]
        for k in keys:
            val = cache._store.pop(k)
            cache._store[k] = val
            touched += 1
    try:
        from src.utils.analysis_metrics import feature_cache_prewarm_total

        feature_cache_prewarm_total.labels(result="ok").inc()
    except Exception:
        pass
    return {
        "status": "ok",
        "strategy": strategy,
        "touched": touched,
        "limit": limit,
        "size": cache.size(),
    }


__all__ = [
    "FeatureCache",
    "get_feature_cache",
    "reset_feature_cache_for_tests",
    "apply_cache_settings",
    "rollback_cache_settings",
    "prewarm_cache",
]
