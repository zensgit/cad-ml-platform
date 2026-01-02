"""Vision response caching.

Provides:
- In-memory LRU cache with TTL
- Content-based cache key generation
- Cache statistics and management
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from .base import VisionDescription

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with value and metadata."""

    value: VisionDescription
    created_at: float
    access_count: int = 0
    last_accessed: float = 0

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if entry has expired."""
        return time.time() - self.created_at > ttl_seconds


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class VisionCache:
    """
    In-memory LRU cache for vision analysis results.

    Features:
    - Content-based cache keys (hash of image data)
    - Configurable TTL (time-to-live)
    - LRU eviction policy
    - Thread-safe operations
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: float = 3600.0,  # 1 hour default
    ):
        """
        Initialize vision cache.

        Args:
            max_size: Maximum number of cached entries
            ttl_seconds: Time-to-live for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats(max_size=max_size)
        self._lock = asyncio.Lock()

    @staticmethod
    def _generate_key(
        image_data: bytes,
        provider: str,
        include_description: bool = True,
    ) -> str:
        """
        Generate cache key from image content and parameters.

        Uses SHA-256 hash of image data combined with provider and options.
        """
        hasher = hashlib.sha256()
        hasher.update(image_data)
        hasher.update(provider.encode())
        hasher.update(str(include_description).encode())
        return hasher.hexdigest()[:32]

    async def get(
        self,
        image_data: bytes,
        provider: str,
        include_description: bool = True,
    ) -> Optional[VisionDescription]:
        """
        Get cached result if available and not expired.

        Args:
            image_data: Raw image bytes
            provider: Provider name for cache isolation
            include_description: Whether description was requested

        Returns:
            Cached VisionDescription or None if not found/expired
        """
        key = self._generate_key(image_data, provider, include_description)

        async with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired(self.ttl_seconds):
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                self._stats.size = len(self._cache)
                logger.debug(f"Cache entry expired: {key[:8]}...")
                return None

            # Update access stats and move to end (LRU)
            entry.access_count += 1
            entry.last_accessed = time.time()
            self._cache.move_to_end(key)

            self._stats.hits += 1
            logger.debug(f"Cache hit: {key[:8]}... (access #{entry.access_count})")
            return entry.value

    async def set(
        self,
        image_data: bytes,
        provider: str,
        result: VisionDescription,
        include_description: bool = True,
    ) -> None:
        """
        Store result in cache.

        Args:
            image_data: Raw image bytes
            provider: Provider name
            result: Analysis result to cache
            include_description: Whether description was requested
        """
        key = self._generate_key(image_data, provider, include_description)

        async with self._lock:
            # Evict oldest entries if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats.evictions += 1
                logger.debug(f"Evicted oldest entry: {oldest_key[:8]}...")

            # Store new entry
            now = time.time()
            self._cache[key] = CacheEntry(
                value=result,
                created_at=now,
                access_count=0,
                last_accessed=now,
            )
            self._stats.size = len(self._cache)
            logger.debug(f"Cached result: {key[:8]}...")

    async def invalidate(
        self,
        image_data: bytes,
        provider: str,
        include_description: bool = True,
    ) -> bool:
        """
        Invalidate specific cache entry.

        Returns:
            True if entry was removed, False if not found
        """
        key = self._generate_key(image_data, provider, include_description)

        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size = len(self._cache)
                logger.debug(f"Invalidated entry: {key[:8]}...")
                return True
            return False

    async def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.size = 0
            logger.info(f"Cache cleared: {count} entries removed")
            return count

    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired(self.ttl_seconds)
            ]

            for key in expired_keys:
                del self._cache[key]
                self._stats.evictions += 1

            self._stats.size = len(self._cache)
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
            return len(expired_keys)

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        self._stats.size = len(self._cache)
        return self._stats

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = CacheStats(
            max_size=self.max_size,
            size=len(self._cache),
        )


class CachedVisionProvider:
    """
    Wrapper that adds caching to any VisionProvider.

    Caches successful responses based on image content hash.
    """

    def __init__(
        self,
        provider,  # VisionProvider
        cache: Optional[VisionCache] = None,
        cache_max_size: int = 100,
        cache_ttl_seconds: float = 3600.0,
    ):
        """
        Initialize cached provider.

        Args:
            provider: The underlying vision provider
            cache: Optional shared cache instance
            cache_max_size: Max cache entries (if creating new cache)
            cache_ttl_seconds: Cache TTL (if creating new cache)
        """
        self._provider = provider
        self._cache = cache or VisionCache(
            max_size=cache_max_size,
            ttl_seconds=cache_ttl_seconds,
        )

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """
        Analyze image with caching.

        Checks cache first, calls provider on miss, caches successful results.
        """
        # Check cache
        cached = await self._cache.get(
            image_data,
            self._provider.provider_name,
            include_description,
        )
        if cached is not None:
            return cached

        # Call provider
        result = await self._provider.analyze_image(image_data, include_description)

        # Cache successful results
        await self._cache.set(
            image_data,
            self._provider.provider_name,
            result,
            include_description,
        )

        return result

    @property
    def provider_name(self) -> str:
        """Return wrapped provider name."""
        return self._provider.provider_name

    @property
    def cache(self) -> VisionCache:
        """Get underlying cache."""
        return self._cache

    @property
    def cache_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._cache.stats


def create_cached_provider(
    provider,
    max_size: int = 100,
    ttl_seconds: float = 3600.0,
) -> CachedVisionProvider:
    """
    Factory to create a cached provider wrapper.

    Args:
        provider: The underlying vision provider
        max_size: Maximum cache entries
        ttl_seconds: Cache entry TTL

    Returns:
        CachedVisionProvider wrapping the original

    Example:
        >>> provider = create_vision_provider("openai")
        >>> cached = create_cached_provider(provider, max_size=200)
        >>> result = await cached.analyze_image(image_bytes)
    """
    return CachedVisionProvider(
        provider=provider,
        cache_max_size=max_size,
        cache_ttl_seconds=ttl_seconds,
    )
