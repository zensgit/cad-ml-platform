"""
Distributed Cache Module - Phase 13.

Provides distributed caching capabilities including cache consistency,
TTL management, cache sharding, and invalidation strategies.
"""

import hashlib
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union

from .base import VisionDescription, VisionProvider

# ============================================================================
# Enums
# ============================================================================


class CacheStrategy(Enum):
    """Cache eviction strategy."""

    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"
    RANDOM = "random"


class ConsistencyLevel(Enum):
    """Cache consistency level."""

    STRONG = "strong"
    EVENTUAL = "eventual"
    READ_YOUR_WRITES = "read_your_writes"
    MONOTONIC_READS = "monotonic_reads"


class InvalidationStrategy(Enum):
    """Cache invalidation strategy."""

    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    WRITE_AROUND = "write_around"
    REFRESH_AHEAD = "refresh_ahead"


class CacheNodeStatus(Enum):
    """Cache node status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SYNCING = "syncing"
    FAILED = "failed"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class CacheEntry:
    """A cache entry."""

    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


@dataclass
class CacheConfig:
    """Cache configuration."""

    cache_id: str
    name: str
    max_size: int = 10000
    default_ttl_seconds: int = 3600
    strategy: CacheStrategy = CacheStrategy.LRU
    consistency: ConsistencyLevel = ConsistencyLevel.EVENTUAL
    invalidation: InvalidationStrategy = InvalidationStrategy.WRITE_THROUGH
    shard_count: int = 1
    replica_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStats:
    """Cache statistics."""

    cache_id: str
    total_entries: int
    hits: int
    misses: int
    evictions: int
    size_bytes: int = 0
    hit_rate: float = 0.0
    avg_latency_ms: float = 0.0


@dataclass
class CacheNode:
    """A cache node in distributed cache."""

    node_id: str
    host: str
    port: int
    status: CacheNodeStatus = CacheNodeStatus.ACTIVE
    shard_ids: List[int] = field(default_factory=list)
    weight: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShardInfo:
    """Information about a cache shard."""

    shard_id: int
    node_id: str
    entry_count: int = 0
    size_bytes: int = 0
    is_primary: bool = True


# ============================================================================
# Cache Shard
# ============================================================================


class CacheShard:
    """A cache shard holding a portion of cached data."""

    def __init__(
        self,
        shard_id: int,
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.LRU,
    ) -> None:
        self._shard_id = shard_id
        self._max_size = max_size
        self._strategy = strategy
        self._entries: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the shard."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.is_expired():
                del self._entries[key]
                self._stats["misses"] += 1
                return None

            entry.touch()
            self._stats["hits"] += 1
            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Set a value in the shard."""
        with self._lock:
            # Evict if necessary
            if len(self._entries) >= self._max_size and key not in self._entries:
                self._evict()

            expires_at = None
            if ttl_seconds:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)

            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at,
            )
            self._entries[key] = entry
            return True

    def delete(self, key: str) -> bool:
        """Delete a key from the shard."""
        with self._lock:
            if key in self._entries:
                del self._entries[key]
                return True
            return False

    def _evict(self) -> None:
        """Evict entries based on strategy."""
        if not self._entries:
            return

        if self._strategy == CacheStrategy.LRU:
            oldest = min(self._entries.values(), key=lambda e: e.last_accessed)
            del self._entries[oldest.key]
        elif self._strategy == CacheStrategy.LFU:
            least_frequent = min(self._entries.values(), key=lambda e: e.access_count)
            del self._entries[least_frequent.key]
        elif self._strategy == CacheStrategy.FIFO:
            oldest = min(self._entries.values(), key=lambda e: e.created_at)
            del self._entries[oldest.key]
        elif self._strategy == CacheStrategy.TTL:
            # Remove expired first, then oldest TTL
            expired = [k for k, v in self._entries.items() if v.is_expired()]
            if expired:
                del self._entries[expired[0]]
            else:
                oldest = min(
                    self._entries.values(),
                    key=lambda e: e.expires_at or datetime.max,
                )
                del self._entries[oldest.key]
        else:  # RANDOM
            import random

            key = random.choice(list(self._entries.keys()))
            del self._entries[key]

        self._stats["evictions"] += 1

    def size(self) -> int:
        """Get number of entries."""
        return len(self._entries)

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._entries.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get shard statistics."""
        return {
            "entries": len(self._entries),
            **self._stats,
        }


# ============================================================================
# Consistent Hashing
# ============================================================================


class ConsistentHashRing:
    """Consistent hash ring for cache sharding."""

    def __init__(self, virtual_nodes: int = 100) -> None:
        self._virtual_nodes = virtual_nodes
        self._ring: Dict[int, str] = {}
        self._sorted_keys: List[int] = []
        self._nodes: Set[str] = set()

    def add_node(self, node_id: str) -> None:
        """Add a node to the ring."""
        self._nodes.add(node_id)
        for i in range(self._virtual_nodes):
            key = self._hash(f"{node_id}:{i}")
            self._ring[key] = node_id
        self._sorted_keys = sorted(self._ring.keys())

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the ring."""
        self._nodes.discard(node_id)
        for i in range(self._virtual_nodes):
            key = self._hash(f"{node_id}:{i}")
            if key in self._ring:
                del self._ring[key]
        self._sorted_keys = sorted(self._ring.keys())

    def get_node(self, key: str) -> Optional[str]:
        """Get the node responsible for a key."""
        if not self._ring:
            return None

        hash_key = self._hash(key)

        # Binary search for the first node >= hash_key
        for ring_key in self._sorted_keys:
            if ring_key >= hash_key:
                return self._ring[ring_key]

        # Wrap around to first node
        return self._ring[self._sorted_keys[0]]

    def _hash(self, key: str) -> int:
        """Hash a key to an integer."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def get_nodes(self) -> Set[str]:
        """Get all nodes."""
        return self._nodes.copy()


# ============================================================================
# Distributed Cache
# ============================================================================


class DistributedCache:
    """Distributed cache implementation."""

    def __init__(self, config: CacheConfig) -> None:
        self._config = config
        self._shards: Dict[int, CacheShard] = {}
        self._hash_ring = ConsistentHashRing()
        self._nodes: Dict[str, CacheNode] = {}
        self._lock = threading.Lock()

        # Initialize shards
        for i in range(config.shard_count):
            self._shards[i] = CacheShard(
                shard_id=i,
                max_size=config.max_size // config.shard_count,
                strategy=config.strategy,
            )
            node_id = f"node_{i}"
            self._hash_ring.add_node(node_id)
            self._nodes[node_id] = CacheNode(
                node_id=node_id,
                host="localhost",
                port=6379 + i,
                shard_ids=[i],
            )

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        shard = self._get_shard(key)
        return shard.get(key)

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Set a value in the cache."""
        shard = self._get_shard(key)
        ttl = ttl_seconds or self._config.default_ttl_seconds
        return shard.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        shard = self._get_shard(key)
        return shard.delete(key)

    def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl_seconds: Optional[int] = None,
    ) -> Any:
        """Get a value or set it using factory if not present."""
        value = self.get(key)
        if value is not None:
            return value

        value = factory()
        self.set(key, value, ttl_seconds)
        return value

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern."""
        count = 0
        for shard in self._shards.values():
            with shard._lock:
                keys_to_delete = [k for k in shard._entries.keys() if pattern in k]
                for key in keys_to_delete:
                    del shard._entries[key]
                    count += 1
        return count

    def clear(self) -> None:
        """Clear all cache entries."""
        for shard in self._shards.values():
            shard.clear()

    def _get_shard(self, key: str) -> CacheShard:
        """Get the shard for a key."""
        node_id = self._hash_ring.get_node(key)
        if node_id and node_id in self._nodes:
            shard_ids = self._nodes[node_id].shard_ids
            if shard_ids:
                return self._shards[shard_ids[0]]
        # Fallback to hash-based shard selection
        shard_id = hash(key) % len(self._shards)
        return self._shards[shard_id]

    def add_node(self, node: CacheNode) -> None:
        """Add a cache node."""
        with self._lock:
            self._nodes[node.node_id] = node
            self._hash_ring.add_node(node.node_id)

    def remove_node(self, node_id: str) -> bool:
        """Remove a cache node."""
        with self._lock:
            if node_id in self._nodes:
                del self._nodes[node_id]
                self._hash_ring.remove_node(node_id)
                return True
            return False

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        total_entries = 0
        total_hits = 0
        total_misses = 0
        total_evictions = 0

        for shard in self._shards.values():
            stats = shard.get_stats()
            total_entries += stats["entries"]
            total_hits += stats["hits"]
            total_misses += stats["misses"]
            total_evictions += stats["evictions"]

        hit_rate = 0.0
        if total_hits + total_misses > 0:
            hit_rate = total_hits / (total_hits + total_misses)

        return CacheStats(
            cache_id=self._config.cache_id,
            total_entries=total_entries,
            hits=total_hits,
            misses=total_misses,
            evictions=total_evictions,
            hit_rate=hit_rate,
        )

    def get_shard_info(self) -> List[ShardInfo]:
        """Get information about all shards."""
        info = []
        for shard_id, shard in self._shards.items():
            node_id = None
            for nid, node in self._nodes.items():
                if shard_id in node.shard_ids:
                    node_id = nid
                    break
            info.append(
                ShardInfo(
                    shard_id=shard_id,
                    node_id=node_id or "unknown",
                    entry_count=shard.size(),
                )
            )
        return info


# ============================================================================
# Cache Manager
# ============================================================================


class CacheManager:
    """Manages multiple distributed caches."""

    def __init__(self) -> None:
        self._caches: Dict[str, DistributedCache] = {}

    def create_cache(self, config: CacheConfig) -> DistributedCache:
        """Create a new cache."""
        cache = DistributedCache(config)
        self._caches[config.cache_id] = cache
        return cache

    def get_cache(self, cache_id: str) -> Optional[DistributedCache]:
        """Get a cache by ID."""
        return self._caches.get(cache_id)

    def delete_cache(self, cache_id: str) -> bool:
        """Delete a cache."""
        if cache_id in self._caches:
            del self._caches[cache_id]
            return True
        return False

    def list_caches(self) -> List[str]:
        """List all cache IDs."""
        return list(self._caches.keys())

    def get_all_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all caches."""
        return {cache_id: cache.get_stats() for cache_id, cache in self._caches.items()}


# ============================================================================
# Distributed Cache Vision Provider
# ============================================================================


class DistributedCacheVisionProvider(VisionProvider):
    """Vision provider with distributed cache."""

    def __init__(
        self,
        provider: VisionProvider,
        cache: DistributedCache,
        ttl_seconds: int = 3600,
    ) -> None:
        self._provider = provider
        self._cache = cache
        self._ttl_seconds = ttl_seconds

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"dcache_{self._provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
        **kwargs: Any,
    ) -> VisionDescription:
        """Analyze image with distributed caching."""
        # Generate cache key
        cache_key = hashlib.md5(image_data).hexdigest()

        # Check cache
        cached = self._cache.get(cache_key)
        if cached:
            return VisionDescription(**cached)

        # Analyze
        result = await self._provider.analyze_image(image_data, include_description)

        # Cache result
        self._cache.set(
            cache_key,
            {
                "summary": result.summary,
                "details": result.details,
                "confidence": result.confidence,
            },
            self._ttl_seconds,
        )

        return result

    def get_cache(self) -> DistributedCache:
        """Get the distributed cache."""
        return self._cache


# ============================================================================
# Factory Functions
# ============================================================================


def create_distributed_cache(
    cache_id: str,
    name: str,
    max_size: int = 10000,
    shard_count: int = 4,
    strategy: CacheStrategy = CacheStrategy.LRU,
) -> DistributedCache:
    """Create a distributed cache."""
    config = CacheConfig(
        cache_id=cache_id,
        name=name,
        max_size=max_size,
        shard_count=shard_count,
        strategy=strategy,
    )
    return DistributedCache(config)


def create_cache_manager() -> CacheManager:
    """Create a cache manager."""
    return CacheManager()


def create_dcache_provider(
    provider: VisionProvider,
    cache: Optional[DistributedCache] = None,
    ttl_seconds: int = 3600,
) -> DistributedCacheVisionProvider:
    """Create a distributed cache vision provider."""
    if cache is None:
        cache = create_distributed_cache("vision_cache", "Vision Cache")
    return DistributedCacheVisionProvider(provider, cache, ttl_seconds)


def create_cache_config(
    cache_id: str,
    name: str,
    max_size: int = 10000,
    strategy: CacheStrategy = CacheStrategy.LRU,
) -> CacheConfig:
    """Create a cache configuration."""
    return CacheConfig(
        cache_id=cache_id,
        name=name,
        max_size=max_size,
        strategy=strategy,
    )


def create_consistent_hash_ring(virtual_nodes: int = 100) -> ConsistentHashRing:
    """Create a consistent hash ring."""
    return ConsistentHashRing(virtual_nodes)
