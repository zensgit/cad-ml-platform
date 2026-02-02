"""DataLoader for N+1 Query Prevention.

Provides batching and caching for efficient data loading.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type


@dataclass
class CacheEntry(Generic[V]):
    """A cached value with expiration."""
    value: V
    created_at: datetime = field(default_factory=datetime.utcnow)
    ttl_seconds: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl_seconds is None:
            return False
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl_seconds


BatchLoadFn = Callable[[List[K]], List[Optional[V]]]


class DataLoader(Generic[K, V]):
    """DataLoader for batching and caching data requests.

    Inspired by Facebook's DataLoader pattern for solving N+1 query problem.

    Example:
        async def batch_load_users(ids: List[str]) -> List[User]:
            return await User.get_many(ids)

        loader = DataLoader(batch_load_users)

        # These calls will be batched into a single query
        user1 = await loader.load("user-1")
        user2 = await loader.load("user-2")
    """

    def __init__(
        self,
        batch_fn: BatchLoadFn,
        max_batch_size: int = 100,
        batch_window_ms: float = 10,
        cache_enabled: bool = True,
        cache_ttl_seconds: Optional[float] = None,
    ):
        """Initialize DataLoader.

        Args:
            batch_fn: Function to batch load values by keys
            max_batch_size: Maximum keys per batch
            batch_window_ms: Time window to collect keys before batching
            cache_enabled: Whether to cache loaded values
            cache_ttl_seconds: Cache TTL (None for unlimited)
        """
        self._batch_fn = batch_fn
        self._max_batch_size = max_batch_size
        self._batch_window_ms = batch_window_ms
        self._cache_enabled = cache_enabled
        self._cache_ttl_seconds = cache_ttl_seconds

        self._cache: Dict[K, CacheEntry[V]] = {}
        self._pending: Dict[K, asyncio.Future] = {}
        self._batch_keys: List[K] = []
        self._batch_lock: Optional[asyncio.Lock] = None
        self._batch_task: Optional[asyncio.Task] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._batch_lock is None:
            self._batch_lock = asyncio.Lock()
        return self._batch_lock

    async def load(self, key: K) -> Optional[V]:
        """Load a single value by key.

        Args:
            key: Key to load

        Returns:
            Loaded value or None
        """
        # Check cache
        if self._cache_enabled and key in self._cache:
            entry = self._cache[key]
            if not entry.is_expired():
                return entry.value
            else:
                del self._cache[key]

        # Check if already pending
        if key in self._pending:
            return await self._pending[key]

        # Add to batch
        async with self._get_lock():
            # Double-check after acquiring lock
            if key in self._pending:
                return await self._pending[key]

            # Create future for this key
            future: asyncio.Future = asyncio.get_event_loop().create_future()
            self._pending[key] = future
            self._batch_keys.append(key)

            # Schedule batch execution
            if self._batch_task is None:
                self._batch_task = asyncio.create_task(self._dispatch_batch())

            # If batch is full, dispatch immediately
            if len(self._batch_keys) >= self._max_batch_size:
                self._batch_task.cancel()
                self._batch_task = asyncio.create_task(self._execute_batch())

        return await future

    async def load_many(self, keys: List[K]) -> List[Optional[V]]:
        """Load multiple values by keys.

        Args:
            keys: Keys to load

        Returns:
            List of loaded values (in same order as keys)
        """
        return await asyncio.gather(*[self.load(key) for key in keys])

    async def _dispatch_batch(self) -> None:
        """Wait for batch window then execute."""
        await asyncio.sleep(self._batch_window_ms / 1000)
        await self._execute_batch()

    async def _execute_batch(self) -> None:
        """Execute the current batch."""
        async with self._get_lock():
            if not self._batch_keys:
                return

            keys = self._batch_keys.copy()
            self._batch_keys.clear()
            self._batch_task = None

        try:
            # Call batch function
            if asyncio.iscoroutinefunction(self._batch_fn):
                values = await self._batch_fn(keys)
            else:
                values = self._batch_fn(keys)

            # Validate result length
            if len(values) != len(keys):
                raise ValueError(
                    f"Batch function returned {len(values)} values for {len(keys)} keys"
                )

            # Resolve futures and cache
            for key, value in zip(keys, values):
                # Cache value
                if self._cache_enabled:
                    self._cache[key] = CacheEntry(
                        value=value,
                        ttl_seconds=self._cache_ttl_seconds,
                    )

                # Resolve future
                if key in self._pending:
                    future = self._pending.pop(key)
                    if not future.done():
                        future.set_result(value)

        except Exception as e:
            logger.error(f"Batch load error: {e}")
            # Reject all pending futures
            for key in keys:
                if key in self._pending:
                    future = self._pending.pop(key)
                    if not future.done():
                        future.set_exception(e)

    def clear(self, key: Optional[K] = None) -> None:
        """Clear cache.

        Args:
            key: Specific key to clear, or None for all
        """
        if key is not None:
            self._cache.pop(key, None)
        else:
            self._cache.clear()

    def prime(self, key: K, value: V) -> None:
        """Prime the cache with a value.

        Args:
            key: Key to cache
            value: Value to cache
        """
        if self._cache_enabled:
            self._cache[key] = CacheEntry(
                value=value,
                ttl_seconds=self._cache_ttl_seconds,
            )

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "pending": len(self._pending),
            "enabled": self._cache_enabled,
            "ttl_seconds": self._cache_ttl_seconds,
        }


class BatchLoader(Generic[K, V]):
    """Simplified batch loader without caching."""

    def __init__(
        self,
        batch_fn: BatchLoadFn,
        max_batch_size: int = 100,
    ):
        self._batch_fn = batch_fn
        self._max_batch_size = max_batch_size

    async def load_many(self, keys: List[K]) -> List[Optional[V]]:
        """Load multiple values, batching automatically."""
        if not keys:
            return []

        results = []
        for i in range(0, len(keys), self._max_batch_size):
            batch = keys[i:i + self._max_batch_size]
            if asyncio.iscoroutinefunction(self._batch_fn):
                batch_results = await self._batch_fn(batch)
            else:
                batch_results = self._batch_fn(batch)
            results.extend(batch_results)

        return results


class CachedDataLoader(DataLoader[K, V]):
    """DataLoader with advanced caching features."""

    def __init__(
        self,
        batch_fn: BatchLoadFn,
        max_batch_size: int = 100,
        batch_window_ms: float = 10,
        cache_ttl_seconds: float = 300,
        max_cache_size: int = 10000,
    ):
        super().__init__(
            batch_fn=batch_fn,
            max_batch_size=max_batch_size,
            batch_window_ms=batch_window_ms,
            cache_enabled=True,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        self._max_cache_size = max_cache_size
        self._access_order: List[K] = []

    def prime(self, key: K, value: V) -> None:
        """Prime cache with LRU eviction."""
        # Evict if at capacity
        while len(self._cache) >= self._max_cache_size:
            if self._access_order:
                oldest_key = self._access_order.pop(0)
                self._cache.pop(oldest_key, None)
            else:
                break

        super().prime(key, value)
        self._access_order.append(key)

    async def load(self, key: K) -> Optional[V]:
        """Load with LRU tracking."""
        result = await super().load(key)

        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        return result


def create_dataloaders(repositories: Dict[str, Any]) -> Dict[str, DataLoader]:
    """Create DataLoaders for common data types.

    Args:
        repositories: Dict of repository instances

    Returns:
        Dict of DataLoaders keyed by type name
    """
    loaders = {}

    if "documents" in repositories:
        async def batch_load_documents(ids: List[str]) -> List[Any]:
            return await repositories["documents"].get_many(ids)

        loaders["documents"] = DataLoader(
            batch_fn=batch_load_documents,
            cache_ttl_seconds=60,
        )

    if "users" in repositories:
        async def batch_load_users(ids: List[str]) -> List[Any]:
            return await repositories["users"].get_many(ids)

        loaders["users"] = DataLoader(
            batch_fn=batch_load_users,
            cache_ttl_seconds=300,
        )

    if "models" in repositories:
        async def batch_load_models(ids: List[str]) -> List[Any]:
            return await repositories["models"].get_many(ids)

        loaders["models"] = DataLoader(
            batch_fn=batch_load_models,
            cache_ttl_seconds=120,
        )

    return loaders


class DataLoaderRegistry:
    """Registry for managing DataLoaders per request."""

    def __init__(self):
        self._factories: Dict[str, Callable[[], DataLoader]] = {}

    def register(self, name: str, factory: Callable[[], DataLoader]) -> None:
        """Register a DataLoader factory.

        Args:
            name: Loader name
            factory: Factory function to create loader
        """
        self._factories[name] = factory

    def create_loaders(self) -> Dict[str, DataLoader]:
        """Create fresh DataLoaders for a request.

        Returns:
            Dict of DataLoaders
        """
        return {name: factory() for name, factory in self._factories.items()}


# Global registry
_dataloader_registry: Optional[DataLoaderRegistry] = None


def get_dataloader_registry() -> DataLoaderRegistry:
    """Get global DataLoader registry."""
    global _dataloader_registry
    if _dataloader_registry is None:
        _dataloader_registry = DataLoaderRegistry()
    return _dataloader_registry
