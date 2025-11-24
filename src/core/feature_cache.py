from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Dict, Tuple, List


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

    def get(self, key: str) -> List[float] | None:
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


# Global singleton (simple usage pattern for current scope)
_FEATURE_CACHE: FeatureCache | None = None


def get_feature_cache() -> FeatureCache:
    global _FEATURE_CACHE
    if _FEATURE_CACHE is None:
        from os import getenv
        cap = int(getenv("FEATURE_CACHE_CAPACITY", "256"))
        ttl = int(getenv("FEATURE_CACHE_TTL_SECONDS", "0"))
        _FEATURE_CACHE = FeatureCache(capacity=cap, ttl_seconds=ttl)
    return _FEATURE_CACHE


__all__ = ["FeatureCache", "get_feature_cache"]
