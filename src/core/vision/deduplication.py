"""Request deduplication module for Vision Provider system.

This module provides deduplication capabilities including:
- Content-based request deduplication
- Hash-based duplicate detection
- TTL-based cache expiration
- Deduplication strategies
- Statistics and monitoring
"""

import hashlib
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import VisionDescription, VisionProvider


class DeduplicationStrategy(Enum):
    """Strategy for deduplication."""

    EXACT = "exact"  # Exact content match
    HASH = "hash"  # Hash-based match
    SIMILARITY = "similarity"  # Similarity-based match
    SIZE_AND_HASH = "size_and_hash"  # Combined size and hash


class HashAlgorithm(Enum):
    """Hash algorithm for deduplication."""

    MD5 = "md5"
    SHA1 = "sha1"
    SHA256 = "sha256"
    XXHASH = "xxhash"  # If available


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication."""

    strategy: DeduplicationStrategy = DeduplicationStrategy.HASH
    hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256
    ttl_seconds: float = 300.0  # 5 minutes
    max_cache_size: int = 1000
    enabled: bool = True
    include_params_in_key: bool = True


@dataclass
class CachedResult:
    """Cached result for deduplication."""

    key: str
    result: VisionDescription
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    hit_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)

    def is_expired(self) -> bool:
        """Check if result is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def access(self) -> None:
        """Record an access."""
        self.hit_count += 1
        self.last_accessed = datetime.now()


@dataclass
class DeduplicationStats:
    """Statistics for deduplication."""

    total_requests: int = 0
    deduplicated_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_expirations: int = 0
    current_cache_size: int = 0
    bytes_saved: int = 0

    @property
    def deduplication_rate(self) -> float:
        """Calculate deduplication rate."""
        if self.total_requests == 0:
            return 0.0
        return self.deduplicated_requests / self.total_requests

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total


class KeyGenerator(ABC):
    """Abstract base class for key generators."""

    @abstractmethod
    def generate_key(
        self,
        image_data: bytes,
        include_description: bool = True,
        **kwargs: Any,
    ) -> str:
        """Generate a unique key for the request."""
        pass


class HashKeyGenerator(KeyGenerator):
    """Generate keys using hash algorithms."""

    def __init__(
        self,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
        include_params: bool = True,
    ) -> None:
        """Initialize hash key generator."""
        self._algorithm = algorithm
        self._include_params = include_params

    def generate_key(
        self,
        image_data: bytes,
        include_description: bool = True,
        **kwargs: Any,
    ) -> str:
        """Generate hash-based key."""
        if self._algorithm == HashAlgorithm.MD5:
            hasher = hashlib.md5()
        elif self._algorithm == HashAlgorithm.SHA1:
            hasher = hashlib.sha1()
        else:
            hasher = hashlib.sha256()

        hasher.update(image_data)

        if self._include_params:
            param_str = f"|desc={include_description}"
            for k, v in sorted(kwargs.items()):
                param_str += f"|{k}={v}"
            hasher.update(param_str.encode())

        return hasher.hexdigest()


class SizeHashKeyGenerator(KeyGenerator):
    """Generate keys using size and hash combination."""

    def __init__(self, hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> None:
        """Initialize size-hash key generator."""
        self._hash_gen = HashKeyGenerator(hash_algorithm)

    def generate_key(
        self,
        image_data: bytes,
        include_description: bool = True,
        **kwargs: Any,
    ) -> str:
        """Generate size+hash key."""
        size = len(image_data)
        hash_key = self._hash_gen.generate_key(image_data, include_description, **kwargs)
        return f"{size}_{hash_key}"


class DeduplicationCache:
    """Cache for deduplicated results."""

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 300.0,
    ) -> None:
        """Initialize cache.

        Args:
            max_size: Maximum cache entries
            ttl_seconds: Time-to-live for entries
        """
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cache: Dict[str, CachedResult] = {}
        self._lock = threading.Lock()
        self._stats = DeduplicationStats()

    @property
    def stats(self) -> DeduplicationStats:
        """Get cache statistics."""
        return self._stats

    def get(self, key: str) -> Optional[VisionDescription]:
        """Get cached result.

        Args:
            key: Cache key

        Returns:
            Cached result or None
        """
        with self._lock:
            cached = self._cache.get(key)

            if cached is None:
                self._stats.cache_misses += 1
                return None

            if cached.is_expired():
                del self._cache[key]
                self._stats.cache_expirations += 1
                self._stats.cache_misses += 1
                return None

            cached.access()
            self._stats.cache_hits += 1
            return cached.result

    def put(
        self,
        key: str,
        result: VisionDescription,
        image_size: int = 0,
    ) -> None:
        """Store result in cache.

        Args:
            key: Cache key
            result: Result to cache
            image_size: Size of image in bytes
        """
        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self._max_size:
                self._evict_oldest()

            expires_at = None
            if self._ttl_seconds > 0:
                expires_at = datetime.now() + timedelta(seconds=self._ttl_seconds)

            self._cache[key] = CachedResult(
                key=key,
                result=result,
                expires_at=expires_at,
            )

            self._stats.current_cache_size = len(self._cache)
            if image_size > 0:
                self._stats.bytes_saved += image_size

    def remove(self, key: str) -> bool:
        """Remove entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was removed
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.current_cache_size = len(self._cache)
                return True
            return False

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.current_cache_size = 0
            return count

    def _evict_oldest(self) -> None:
        """Evict oldest entry from cache."""
        if not self._cache:
            return

        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed,
        )
        del self._cache[oldest_key]

    def cleanup_expired(self) -> int:
        """Remove expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [k for k, v in self._cache.items() if v.is_expired()]

            for key in expired_keys:
                del self._cache[key]

            self._stats.cache_expirations += len(expired_keys)
            self._stats.current_cache_size = len(self._cache)
            return len(expired_keys)


class DeduplicationManager:
    """Manages request deduplication."""

    def __init__(
        self,
        config: Optional[DeduplicationConfig] = None,
    ) -> None:
        """Initialize deduplication manager.

        Args:
            config: Deduplication configuration
        """
        self._config = config or DeduplicationConfig()
        self._cache = DeduplicationCache(
            max_size=self._config.max_cache_size,
            ttl_seconds=self._config.ttl_seconds,
        )
        self._key_generator = self._create_key_generator()
        self._pending: Dict[str, List[threading.Event]] = {}
        self._pending_results: Dict[str, VisionDescription] = {}
        self._pending_lock = threading.Lock()

    @property
    def stats(self) -> DeduplicationStats:
        """Get deduplication statistics."""
        return self._cache.stats

    @property
    def config(self) -> DeduplicationConfig:
        """Get configuration."""
        return self._config

    def _create_key_generator(self) -> KeyGenerator:
        """Create appropriate key generator."""
        if self._config.strategy == DeduplicationStrategy.SIZE_AND_HASH:
            return SizeHashKeyGenerator(self._config.hash_algorithm)
        else:
            return HashKeyGenerator(
                self._config.hash_algorithm,
                self._config.include_params_in_key,
            )

    def generate_key(
        self,
        image_data: bytes,
        include_description: bool = True,
        **kwargs: Any,
    ) -> str:
        """Generate deduplication key.

        Args:
            image_data: Image data
            include_description: Whether description is included
            **kwargs: Additional parameters

        Returns:
            Deduplication key
        """
        return self._key_generator.generate_key(image_data, include_description, **kwargs)

    def check_duplicate(
        self,
        image_data: bytes,
        include_description: bool = True,
        **kwargs: Any,
    ) -> Tuple[bool, Optional[VisionDescription], str]:
        """Check if request is a duplicate.

        Args:
            image_data: Image data
            include_description: Whether description is included
            **kwargs: Additional parameters

        Returns:
            Tuple of (is_duplicate, cached_result, key)
        """
        if not self._config.enabled:
            return False, None, ""

        key = self.generate_key(image_data, include_description, **kwargs)
        result = self._cache.get(key)

        self._cache.stats.total_requests += 1

        if result is not None:
            self._cache.stats.deduplicated_requests += 1
            return True, result, key

        return False, None, key

    def store_result(
        self,
        key: str,
        result: VisionDescription,
        image_size: int = 0,
    ) -> None:
        """Store result for future deduplication.

        Args:
            key: Deduplication key
            result: Result to store
            image_size: Size of image in bytes
        """
        if not self._config.enabled:
            return

        self._cache.put(key, result, image_size)

    def invalidate(self, key: str) -> bool:
        """Invalidate a cached result.

        Args:
            key: Cache key

        Returns:
            True if entry was invalidated
        """
        return self._cache.remove(key)

    def clear_cache(self) -> int:
        """Clear all cached results.

        Returns:
            Number of entries cleared
        """
        return self._cache.clear()


class DeduplicatingVisionProvider(VisionProvider):
    """Vision provider with request deduplication."""

    def __init__(
        self,
        provider: VisionProvider,
        manager: Optional[DeduplicationManager] = None,
        config: Optional[DeduplicationConfig] = None,
    ) -> None:
        """Initialize deduplicating provider.

        Args:
            provider: Underlying vision provider
            manager: Optional deduplication manager
            config: Optional configuration
        """
        self._provider = provider

        if manager is not None:
            self._manager = manager
        else:
            self._manager = DeduplicationManager(config)

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"dedup_{self._provider.provider_name}"

    @property
    def manager(self) -> DeduplicationManager:
        """Get deduplication manager."""
        return self._manager

    @property
    def stats(self) -> DeduplicationStats:
        """Get deduplication statistics."""
        return self._manager.stats

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with deduplication.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        # Check for duplicate
        is_dup, cached_result, key = self._manager.check_duplicate(image_data, include_description)

        if is_dup and cached_result is not None:
            return cached_result

        # Process new request
        result = await self._provider.analyze_image(image_data, include_description)

        # Store for future deduplication
        self._manager.store_result(key, result, len(image_data))

        return result


# Global manager instance
_manager: Optional[DeduplicationManager] = None


def get_deduplication_manager() -> DeduplicationManager:
    """Get global deduplication manager."""
    global _manager
    if _manager is None:
        _manager = DeduplicationManager()
    return _manager


def create_deduplicating_provider(
    provider: VisionProvider,
    config: Optional[DeduplicationConfig] = None,
    manager: Optional[DeduplicationManager] = None,
) -> DeduplicatingVisionProvider:
    """Create a deduplicating vision provider.

    Args:
        provider: Underlying vision provider
        config: Optional deduplication configuration
        manager: Optional deduplication manager

    Returns:
        DeduplicatingVisionProvider instance
    """
    return DeduplicatingVisionProvider(
        provider=provider,
        manager=manager,
        config=config,
    )
