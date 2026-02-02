"""Caching Module.

Provides caching infrastructure:
- Multi-level caching
- Cache-aside pattern
- TTL and invalidation
- Cache statistics
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live only


@dataclass
class CacheEntry(Generic[T]):
    """A cache entry with metadata."""

    key: str
    value: T
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1

    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "hit_rate": self.hit_rate,
            "size": self.size,
            "max_size": self.max_size,
        }


class CacheBackend(ABC):
    """Abstract cache backend."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
    ) -> bool:
        """Set value with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a key."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all entries."""
        pass

    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        pass


class InMemoryCache(CacheBackend):
    """In-memory cache implementation."""

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = None,
        policy: CachePolicy = CachePolicy.LRU,
    ):
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._policy = policy
        self._cache: Dict[str, CacheEntry] = {}
        self._stats = CacheStats(max_size=max_size)
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._stats.misses += 1
                self._stats.expirations += 1
                self._stats.size = len(self._cache)
                return None

            entry.touch()
            self._stats.hits += 1
            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
    ) -> bool:
        """Set value with optional TTL."""
        async with self._lock:
            # Evict if at capacity and key doesn't exist
            if key not in self._cache and len(self._cache) >= self._max_size:
                self._evict()

            expires_at = None
            effective_ttl = ttl if ttl is not None else self._default_ttl
            if effective_ttl is not None:
                expires_at = time.time() + effective_ttl

            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at,
            )

            self._cache[key] = entry
            self._stats.size = len(self._cache)
            return True

    async def delete(self, key: str) -> bool:
        """Delete a key."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size = len(self._cache)
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                del self._cache[key]
                self._stats.expirations += 1
                self._stats.size = len(self._cache)
                return False
            return True

    async def clear(self) -> None:
        """Clear all entries."""
        async with self._lock:
            self._cache.clear()
            self._stats.size = 0

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def _evict(self) -> None:
        """Evict entry based on policy."""
        if not self._cache:
            return

        if self._policy == CachePolicy.LRU:
            # Remove least recently accessed
            key = min(self._cache.keys(), key=lambda k: self._cache[k].last_accessed)
        elif self._policy == CachePolicy.LFU:
            # Remove least frequently used
            key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
        elif self._policy == CachePolicy.FIFO:
            # Remove oldest
            key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        else:
            # Default: remove first
            key = next(iter(self._cache))

        del self._cache[key]
        self._stats.evictions += 1
        self._stats.size = len(self._cache)


class TieredCache(CacheBackend):
    """Multi-level tiered cache."""

    def __init__(self, levels: List[CacheBackend]):
        if not levels:
            raise ValueError("At least one cache level required")
        self._levels = levels
        self._stats = CacheStats()

    async def get(self, key: str) -> Optional[Any]:
        """Get from first available level, populate upper levels."""
        for i, cache in enumerate(self._levels):
            value = await cache.get(key)
            if value is not None:
                # Populate upper levels
                for upper in self._levels[:i]:
                    await upper.set(key, value)
                self._stats.hits += 1
                return value

        self._stats.misses += 1
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
    ) -> bool:
        """Set in all levels."""
        results = await asyncio.gather(*[
            cache.set(key, value, ttl) for cache in self._levels
        ])
        return all(results)

    async def delete(self, key: str) -> bool:
        """Delete from all levels."""
        results = await asyncio.gather(*[
            cache.delete(key) for cache in self._levels
        ])
        return any(results)

    async def exists(self, key: str) -> bool:
        """Check if exists in any level."""
        for cache in self._levels:
            if await cache.exists(key):
                return True
        return False

    async def clear(self) -> None:
        """Clear all levels."""
        await asyncio.gather(*[cache.clear() for cache in self._levels])

    def get_stats(self) -> CacheStats:
        """Get combined stats."""
        return self._stats


class CacheManager:
    """High-level cache manager with patterns."""

    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        key_prefix: str = "",
        default_ttl: Optional[float] = None,
        serializer: Optional[Callable[[Any], bytes]] = None,
        deserializer: Optional[Callable[[bytes], Any]] = None,
    ):
        self._backend = backend or InMemoryCache()
        self._prefix = key_prefix
        self._default_ttl = default_ttl
        self._serializer = serializer or pickle.dumps
        self._deserializer = deserializer or pickle.loads

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        if self._prefix:
            return f"{self._prefix}:{key}"
        return key

    async def get(self, key: str, default: T = None) -> Optional[T]:
        """Get value with default."""
        value = await self._backend.get(self._make_key(key))
        return value if value is not None else default

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
    ) -> bool:
        """Set value."""
        effective_ttl = ttl if ttl is not None else self._default_ttl
        return await self._backend.set(self._make_key(key), value, effective_ttl)

    async def delete(self, key: str) -> bool:
        """Delete value."""
        return await self._backend.delete(self._make_key(key))

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], T],
        ttl: Optional[float] = None,
    ) -> T:
        """Get value or compute and cache it (cache-aside pattern)."""
        value = await self.get(key)
        if value is not None:
            return value

        # Compute value
        result = factory()
        if asyncio.iscoroutine(result):
            result = await result

        # Cache it
        await self.set(key, result, ttl)
        return result

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate keys matching pattern (prefix-based)."""
        # Note: Full pattern matching requires backend support
        # This is a simple prefix-based invalidation
        count = 0
        if hasattr(self._backend, "_cache"):
            cache = self._backend._cache
            full_pattern = self._make_key(pattern)
            keys_to_delete = [
                k for k in list(cache.keys())
                if k.startswith(full_pattern.rstrip("*"))
            ]
            for key in keys_to_delete:
                await self._backend.delete(key)
                count += 1
        return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._backend.get_stats()


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments."""
    key_parts = [str(a) for a in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_str = ":".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def cached(
    ttl: Optional[float] = None,
    key_func: Optional[Callable[..., str]] = None,
    manager: Optional[CacheManager] = None,
):
    """Decorator for caching function results."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        _manager = manager or CacheManager()
        _key_func = key_func or (lambda *a, **kw: f"{func.__name__}:{cache_key(*a, **kw)}")

        async def async_wrapper(*args, **kwargs) -> T:
            key = _key_func(*args, **kwargs)

            # Try cache
            cached_value = await _manager.get(key)
            if cached_value is not None:
                return cached_value

            # Compute
            result = await func(*args, **kwargs)

            # Cache
            await _manager.set(key, result, ttl)
            return result

        def sync_wrapper(*args, **kwargs) -> T:
            return asyncio.run(async_wrapper(*args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class CacheInvalidator:
    """Handles cache invalidation."""

    def __init__(self, manager: CacheManager):
        self._manager = manager
        self._tags: Dict[str, Set[str]] = {}  # tag -> keys

    async def set_with_tags(
        self,
        key: str,
        value: Any,
        tags: List[str],
        ttl: Optional[float] = None,
    ) -> bool:
        """Set value with invalidation tags."""
        result = await self._manager.set(key, value, ttl)

        if result:
            for tag in tags:
                if tag not in self._tags:
                    self._tags[tag] = set()
                self._tags[tag].add(key)

        return result

    async def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all keys with tag."""
        keys = self._tags.pop(tag, set())
        count = 0

        for key in keys:
            if await self._manager.delete(key):
                count += 1

            # Remove from other tags
            for other_tag, other_keys in self._tags.items():
                other_keys.discard(key)

        return count


class WriteThrough:
    """Write-through cache pattern."""

    def __init__(
        self,
        cache: CacheManager,
        backend_get: Callable[[str], Any],
        backend_set: Callable[[str, Any], bool],
    ):
        self._cache = cache
        self._backend_get = backend_get
        self._backend_set = backend_set

    async def get(self, key: str) -> Optional[Any]:
        """Get from cache, fall back to backend."""
        value = await self._cache.get(key)
        if value is not None:
            return value

        # Fetch from backend
        result = self._backend_get(key)
        if asyncio.iscoroutine(result):
            result = await result

        if result is not None:
            await self._cache.set(key, result)

        return result

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Write to backend and cache."""
        # Write to backend first
        result = self._backend_set(key, value)
        if asyncio.iscoroutine(result):
            result = await result

        if result:
            await self._cache.set(key, value, ttl)

        return result


class WriteBehind:
    """Write-behind (write-back) cache pattern."""

    def __init__(
        self,
        cache: CacheManager,
        backend_set: Callable[[str, Any], bool],
        flush_interval: float = 5.0,
    ):
        self._cache = cache
        self._backend_set = backend_set
        self._flush_interval = flush_interval
        self._pending: Dict[str, Any] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Write to cache, queue for backend."""
        await self._cache.set(key, value, ttl)

        async with self._lock:
            self._pending[key] = value

        return True

    async def start(self) -> None:
        """Start background flushing."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        """Stop and flush remaining."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        await self._flush()

    async def _flush_loop(self) -> None:
        """Periodic flush loop."""
        while self._running:
            await asyncio.sleep(self._flush_interval)
            await self._flush()

    async def _flush(self) -> None:
        """Flush pending writes to backend."""
        async with self._lock:
            pending = self._pending.copy()
            self._pending.clear()

        for key, value in pending.items():
            try:
                result = self._backend_set(key, value)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Write-behind flush error for {key}: {e}")
                # Re-queue on failure
                async with self._lock:
                    if key not in self._pending:
                        self._pending[key] = value


__all__ = [
    "CachePolicy",
    "CacheEntry",
    "CacheStats",
    "CacheBackend",
    "InMemoryCache",
    "TieredCache",
    "CacheManager",
    "cache_key",
    "cached",
    "CacheInvalidator",
    "WriteThrough",
    "WriteBehind",
]
