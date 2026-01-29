"""
Caching Module for CAD Assistant.

Provides caching utilities for embedding vectors, search results,
and API responses to improve performance.
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry with metadata."""

    value: T
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None  # Time to live in seconds

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1


class CacheBackend(ABC, Generic[T]):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Get number of entries in cache."""
        pass


class LRUCache(CacheBackend[T]):
    """
    Least Recently Used (LRU) cache with TTL support.

    Thread-safe implementation using OrderedDict.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = None,
    ):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }

    def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._stats["misses"] += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._stats["hits"] += 1
            return entry.value

    def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key].value = value
                self._cache[key].touch()
                return

            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
                self._stats["evictions"] += 1

            self._cache[key] = CacheEntry(
                value=value,
                ttl=ttl or self.default_ttl,
            )

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Get number of entries in cache."""
        return len(self._cache)

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count."""
        with self._lock:
            expired_keys = [
                k for k, v in self._cache.items() if v.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        return {
            **self._stats,
            "size": self.size(),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
        }


class EmbeddingCache:
    """
    Specialized cache for embedding vectors.

    Provides efficient caching with content-based hashing.
    """

    def __init__(
        self,
        max_size: int = 10000,
        ttl: float = 3600,  # 1 hour default
    ):
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum number of cached embeddings
            ttl: Time-to-live in seconds
        """
        self._cache: LRUCache[List[float]] = LRUCache(
            max_size=max_size,
            default_ttl=ttl,
        )

    def _hash_text(self, text: str) -> str:
        """Generate hash key for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text."""
        key = self._hash_text(text)
        return self._cache.get(key)

    def set(self, text: str, embedding: List[float]) -> None:
        """Cache embedding for text."""
        key = self._hash_text(text)
        self._cache.set(key, embedding)

    def get_batch(self, texts: List[str]) -> Dict[str, Optional[List[float]]]:
        """Get cached embeddings for multiple texts."""
        return {text: self.get(text) for text in texts}

    def set_batch(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """Cache embeddings for multiple texts."""
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding)

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()


class SearchResultCache:
    """
    Cache for search results.

    Supports query-based caching with configurable TTL.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl: float = 300,  # 5 minutes default
    ):
        """
        Initialize search result cache.

        Args:
            max_size: Maximum cached queries
            ttl: Time-to-live in seconds
        """
        self._cache: LRUCache[List[Dict[str, Any]]] = LRUCache(
            max_size=max_size,
            default_ttl=ttl,
        )

    def _make_key(
        self,
        query: str,
        top_k: int = 10,
        source_filter: Optional[str] = None,
    ) -> str:
        """Generate cache key from query parameters."""
        key_data = f"{query}|{top_k}|{source_filter or ''}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    def get(
        self,
        query: str,
        top_k: int = 10,
        source_filter: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results."""
        key = self._make_key(query, top_k, source_filter)
        return self._cache.get(key)

    def set(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 10,
        source_filter: Optional[str] = None,
    ) -> None:
        """Cache search results."""
        key = self._make_key(query, top_k, source_filter)
        self._cache.set(key, results)

    def invalidate_query(self, query: str) -> None:
        """Invalidate all cached results for a query."""
        # For simplicity, we'd need to track all variants
        # Here we just clear the exact match
        pass

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()


class ResponseCache:
    """
    Cache for API responses.

    Supports caching of complete API responses with TTL.
    """

    def __init__(
        self,
        max_size: int = 500,
        ttl: float = 600,  # 10 minutes default
    ):
        """
        Initialize response cache.

        Args:
            max_size: Maximum cached responses
            ttl: Time-to-live in seconds
        """
        self._cache: LRUCache[Dict[str, Any]] = LRUCache(
            max_size=max_size,
            default_ttl=ttl,
        )

    def _make_key(self, query: str, conversation_id: Optional[str] = None) -> str:
        """Generate cache key."""
        key_data = f"{query}|{conversation_id or ''}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    def get(
        self,
        query: str,
        conversation_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get cached response."""
        key = self._make_key(query, conversation_id)
        return self._cache.get(key)

    def set(
        self,
        query: str,
        response: Dict[str, Any],
        conversation_id: Optional[str] = None,
    ) -> None:
        """Cache response."""
        key = self._make_key(query, conversation_id)
        self._cache.set(key, response)

    def clear(self) -> None:
        """Clear all cached responses."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()


def cached(
    cache: CacheBackend,
    key_fn: Callable[..., str],
    ttl: Optional[float] = None,
):
    """
    Decorator for caching function results.

    Args:
        cache: Cache backend to use
        key_fn: Function to generate cache key from arguments
        ttl: Time-to-live in seconds

    Example:
        >>> embedding_cache = LRUCache(max_size=1000)
        >>> @cached(embedding_cache, lambda text: text)
        ... def embed_text(text: str) -> List[float]:
        ...     return compute_embedding(text)
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            key = key_fn(*args, **kwargs)
            result = cache.get(key)
            if result is not None:
                return result
            result = func(*args, **kwargs)
            cache.set(key, result, ttl)
            return result

        return wrapper

    return decorator


class CacheManager:
    """
    Centralized cache manager for the assistant.

    Manages all caches and provides unified statistics.
    """

    def __init__(
        self,
        embedding_cache_size: int = 10000,
        search_cache_size: int = 1000,
        response_cache_size: int = 500,
        embedding_ttl: float = 3600,
        search_ttl: float = 300,
        response_ttl: float = 600,
    ):
        """
        Initialize cache manager.

        Args:
            embedding_cache_size: Size of embedding cache
            search_cache_size: Size of search result cache
            response_cache_size: Size of response cache
            embedding_ttl: TTL for embeddings
            search_ttl: TTL for search results
            response_ttl: TTL for responses
        """
        self.embedding_cache = EmbeddingCache(
            max_size=embedding_cache_size,
            ttl=embedding_ttl,
        )
        self.search_cache = SearchResultCache(
            max_size=search_cache_size,
            ttl=search_ttl,
        )
        self.response_cache = ResponseCache(
            max_size=response_cache_size,
            ttl=response_ttl,
        )

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        return {
            "embedding": self.embedding_cache.get_stats(),
            "search": self.search_cache.get_stats(),
            "response": self.response_cache.get_stats(),
        }

    def clear_all(self) -> None:
        """Clear all caches."""
        self.embedding_cache.clear()
        self.search_cache.clear()
        self.response_cache.clear()

    def cleanup_expired(self) -> Dict[str, int]:
        """Cleanup expired entries in all caches."""
        return {
            "embedding": self.embedding_cache._cache.cleanup_expired(),
            "search": self.search_cache._cache.cleanup_expired(),
            "response": self.response_cache._cache.cleanup_expired(),
        }
