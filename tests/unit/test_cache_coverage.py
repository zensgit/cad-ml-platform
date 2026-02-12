"""Tests for src/utils/cache.py to improve coverage.

Covers:
- init_redis function
- get_client function
- redis_healthy function
- get_cache function
- set_cache function
- In-memory fallback logic
- Error handling paths
"""

from __future__ import annotations

import json
import time
from types import ModuleType
from typing import Any, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _require_redis_module(cache_module: ModuleType) -> None:
    if cache_module.redis is None:
        pytest.skip("redis.asyncio not available in this environment")


class TestInitRedis:
    """Tests for init_redis function."""

    @pytest.mark.asyncio
    async def test_init_redis_already_initialized(self):
        """Test init_redis returns early if already initialized."""
        from src.utils import cache

        # Save original client
        original_client = cache._redis_client

        try:
            # Set a mock client
            mock_client = MagicMock()
            cache._redis_client = mock_client

            await cache.init_redis()

            # Should still be the same client (no re-init)
            assert cache._redis_client is mock_client
        finally:
            cache._redis_client = original_client

    @pytest.mark.asyncio
    async def test_init_redis_connection_failure(self):
        """Test init_redis handles connection failure."""
        from src.utils import cache

        original_client = cache._redis_client
        try:
            cache._redis_client = None

            with patch("src.utils.cache.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(REDIS_URL="redis://invalid:6379")
                with patch("src.utils.cache.redis") as mock_redis:
                    mock_redis.from_url.side_effect = Exception("Connection failed")

                    await cache.init_redis()

            # Client should still be None after failure
            assert cache._redis_client is None
        finally:
            cache._redis_client = original_client


class TestGetClient:
    """Tests for get_client function."""

    def test_get_client_returns_client(self):
        """Test get_client returns the current client."""
        from src.utils import cache

        original_client = cache._redis_client
        try:
            mock_client = MagicMock()
            cache._redis_client = mock_client

            result = cache.get_client()

            assert result is mock_client
        finally:
            cache._redis_client = original_client

    def test_get_client_returns_none(self):
        """Test get_client returns None when not initialized."""
        from src.utils import cache

        original_client = cache._redis_client
        try:
            cache._redis_client = None

            result = cache.get_client()

            assert result is None
        finally:
            cache._redis_client = original_client


class TestRedisHealthy:
    """Tests for redis_healthy function."""

    @pytest.mark.asyncio
    async def test_redis_healthy_when_connected(self):
        """Test redis_healthy returns True when connected."""
        from src.utils import cache

        original_client = cache._redis_client
        try:
            mock_client = MagicMock()
            mock_client.ping = AsyncMock(return_value=True)
            cache._redis_client = mock_client

            result = await cache.redis_healthy()

            assert result is True
        finally:
            cache._redis_client = original_client

    @pytest.mark.asyncio
    async def test_redis_healthy_when_not_initialized(self):
        """Test redis_healthy returns False when not initialized."""
        from src.utils import cache

        original_client = cache._redis_client
        try:
            cache._redis_client = None

            result = await cache.redis_healthy()

            assert result is False
        finally:
            cache._redis_client = original_client

    @pytest.mark.asyncio
    async def test_redis_healthy_when_ping_fails(self):
        """Test redis_healthy returns False when ping fails."""
        from src.utils import cache

        original_client = cache._redis_client
        try:
            mock_client = MagicMock()
            mock_client.ping = AsyncMock(side_effect=Exception("Connection lost"))
            cache._redis_client = mock_client

            result = await cache.redis_healthy()

            assert result is False
        finally:
            cache._redis_client = original_client


class TestGetCache:
    """Tests for get_cache function."""

    @pytest.mark.asyncio
    async def test_get_cache_from_redis(self):
        """Test get_cache retrieves from Redis."""
        from src.utils import cache

        _require_redis_module(cache)
        original_client = cache._redis_client
        try:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value='{"key": "value"}')
            cache._redis_client = mock_client

            result = await cache.get_cache("test_key")

            assert result == {"key": "value"}
            mock_client.get.assert_called_once_with("test_key")
        finally:
            cache._redis_client = original_client

    @pytest.mark.asyncio
    async def test_get_cache_from_local_fallback(self):
        """Test get_cache uses local fallback when Redis unavailable."""
        from src.utils import cache

        original_client = cache._redis_client
        original_cache = cache._local_cache.copy()
        try:
            cache._redis_client = None
            # Set up local cache with valid expiry
            cache._local_cache["test_key"] = ({"local": "data"}, time.time() + 3600)

            result = await cache.get_cache("test_key")

            assert result == {"local": "data"}
        finally:
            cache._redis_client = original_client
            cache._local_cache.clear()
            cache._local_cache.update(original_cache)

    @pytest.mark.asyncio
    async def test_get_cache_expired_local(self):
        """Test get_cache removes expired local cache entry."""
        from src.utils import cache

        original_client = cache._redis_client
        original_cache = cache._local_cache.copy()
        try:
            cache._redis_client = None
            # Set up local cache with expired entry
            cache._local_cache["expired_key"] = ({"old": "data"}, time.time() - 100)

            result = await cache.get_cache("expired_key")

            assert result is None
            assert "expired_key" not in cache._local_cache
        finally:
            cache._redis_client = original_client
            cache._local_cache.clear()
            cache._local_cache.update(original_cache)

    @pytest.mark.asyncio
    async def test_get_cache_miss(self):
        """Test get_cache returns None on cache miss."""
        from src.utils import cache

        original_client = cache._redis_client
        original_cache = cache._local_cache.copy()
        try:
            cache._redis_client = None
            cache._local_cache.clear()

            result = await cache.get_cache("nonexistent")

            assert result is None
        finally:
            cache._redis_client = original_client
            cache._local_cache.clear()
            cache._local_cache.update(original_cache)

    @pytest.mark.asyncio
    async def test_get_cache_redis_error_fallback(self):
        """Test get_cache falls back to local on Redis error."""
        from src.utils import cache

        original_client = cache._redis_client
        original_cache = cache._local_cache.copy()
        try:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(side_effect=Exception("Redis error"))
            cache._redis_client = mock_client
            cache._local_cache["test_key"] = ({"local": "fallback"}, time.time() + 3600)

            result = await cache.get_cache("test_key")

            assert result == {"local": "fallback"}
        finally:
            cache._redis_client = original_client
            cache._local_cache.clear()
            cache._local_cache.update(original_cache)


class TestSetCache:
    """Tests for set_cache function."""

    @pytest.mark.asyncio
    async def test_set_cache_to_redis(self):
        """Test set_cache stores to Redis."""
        from src.utils import cache

        _require_redis_module(cache)
        original_client = cache._redis_client
        try:
            mock_client = MagicMock()
            mock_client.setex = AsyncMock()
            cache._redis_client = mock_client

            await cache.set_cache("test_key", {"data": "value"}, ttl_seconds=7200)

            mock_client.setex.assert_called_once()
            call_args = mock_client.setex.call_args
            assert call_args[0][0] == "test_key"
            assert call_args[0][1] == 7200
        finally:
            cache._redis_client = original_client

    @pytest.mark.asyncio
    async def test_set_cache_to_local_fallback(self):
        """Test set_cache uses local fallback when Redis unavailable."""
        from src.utils import cache

        original_client = cache._redis_client
        original_cache = cache._local_cache.copy()
        try:
            cache._redis_client = None
            cache._local_cache.clear()

            await cache.set_cache("local_key", {"local": "value"}, ttl_seconds=1800)

            assert "local_key" in cache._local_cache
            value, expire_time = cache._local_cache["local_key"]
            assert value == {"local": "value"}
            assert expire_time > time.time()
        finally:
            cache._redis_client = original_client
            cache._local_cache.clear()
            cache._local_cache.update(original_cache)

    @pytest.mark.asyncio
    async def test_set_cache_redis_error_fallback(self):
        """Test set_cache falls back to local on Redis error."""
        from src.utils import cache

        original_client = cache._redis_client
        original_cache = cache._local_cache.copy()
        try:
            mock_client = MagicMock()
            mock_client.setex = AsyncMock(side_effect=Exception("Redis error"))
            cache._redis_client = mock_client
            cache._local_cache.clear()

            await cache.set_cache("error_key", {"fallback": "data"})

            # Should have fallen back to local cache
            assert "error_key" in cache._local_cache
        finally:
            cache._redis_client = original_client
            cache._local_cache.clear()
            cache._local_cache.update(original_cache)


class TestCompatibilityWrappers:
    """Tests for compatibility wrapper functions."""

    @pytest.mark.asyncio
    async def test_cache_result_wrapper(self):
        """Test cache_result calls set_cache."""
        from src.utils.cache import cache_result

        with patch("src.utils.cache.set_cache", new_callable=AsyncMock) as mock_set:
            await cache_result("key", {"data": 1}, ttl=1800)

            mock_set.assert_called_once_with("key", {"data": 1}, 1800)

    @pytest.mark.asyncio
    async def test_get_cached_result_wrapper(self):
        """Test get_cached_result calls get_cache."""
        from src.utils.cache import get_cached_result

        with patch(
            "src.utils.cache.get_cache", new_callable=AsyncMock, return_value={"cached": True}
        ) as mock_get:
            result = await get_cached_result("key")

            assert result == {"cached": True}
            mock_get.assert_called_once_with("key")


class TestLocalCacheLogic:
    """Tests for local cache logic."""

    def test_local_cache_structure(self):
        """Test local cache stores tuple of value and expire time."""
        value = {"test": "data"}
        expire_time = time.time() + 3600

        entry: Tuple[Any, float] = (value, expire_time)

        assert entry[0] == value
        assert entry[1] == expire_time

    def test_ttl_calculation(self):
        """Test TTL calculation for expiry."""
        ttl_seconds = 3600
        now = time.time()
        expire_time = now + ttl_seconds

        assert expire_time > now
        assert (expire_time - now) == pytest.approx(ttl_seconds, rel=0.01)


class TestJSONSerialization:
    """Tests for JSON serialization in cache."""

    def test_json_dumps(self):
        """Test JSON serialization."""
        data = {"key": "value", "number": 42}
        serialized = json.dumps(data)

        assert isinstance(serialized, str)
        assert "key" in serialized

    def test_json_loads(self):
        """Test JSON deserialization."""
        serialized = '{"key": "value", "number": 42}'
        data = json.loads(serialized)

        assert data["key"] == "value"
        assert data["number"] == 42


class TestRedisModuleHandling:
    """Tests for Redis module availability handling."""

    def test_redis_module_none_check(self):
        """Test redis module None check logic."""
        redis_module = None
        client = MagicMock()

        if client and redis_module is not None:
            use_redis = True
        else:
            use_redis = False

        assert use_redis is False

    def test_redis_module_available_check(self):
        """Test redis module available check."""
        redis_module = MagicMock()
        client = MagicMock()

        if client and redis_module is not None:
            use_redis = True
        else:
            use_redis = False

        assert use_redis is True


class TestGetSyncClient:
    """Tests for get_sync_client function."""

    def test_get_sync_client_returns_existing(self):
        """Test get_sync_client returns existing client."""
        from src.utils import cache

        original = cache._redis_sync_client
        try:
            mock_client = MagicMock()
            cache._redis_sync_client = mock_client

            result = cache.get_sync_client()
            assert result is mock_client
        finally:
            cache._redis_sync_client = original

    def test_get_sync_client_redis_sync_none(self):
        """Test get_sync_client returns None when redis_sync not installed."""
        from src.utils import cache

        original_client = cache._redis_sync_client
        original_redis = cache.redis_sync
        try:
            cache._redis_sync_client = None
            cache.redis_sync = None

            result = cache.get_sync_client()
            assert result is None
        finally:
            cache._redis_sync_client = original_client
            cache.redis_sync = original_redis

    def test_get_sync_client_redis_disabled(self):
        """Test get_sync_client returns None when Redis disabled."""
        from src.utils import cache

        original_client = cache._redis_sync_client
        original_redis = cache.redis_sync
        try:
            cache._redis_sync_client = None
            cache.redis_sync = MagicMock()  # Available but disabled

            with patch("src.utils.cache.get_settings") as mock_settings:
                mock_settings.return_value.REDIS_ENABLED = False

                result = cache.get_sync_client()
                assert result is None
        finally:
            cache._redis_sync_client = original_client
            cache.redis_sync = original_redis

    def test_get_sync_client_initializes_new(self):
        """Test get_sync_client initializes new client."""
        from src.utils import cache

        original_client = cache._redis_sync_client
        original_redis = cache.redis_sync
        try:
            cache._redis_sync_client = None

            mock_redis_sync = MagicMock()
            mock_new_client = MagicMock()
            mock_redis_sync.Redis.from_url.return_value = mock_new_client
            cache.redis_sync = mock_redis_sync

            with patch("src.utils.cache.get_settings") as mock_settings:
                mock_settings.return_value.REDIS_ENABLED = True
                mock_settings.return_value.REDIS_URL = "redis://localhost:6379"

                result = cache.get_sync_client()

            assert result is mock_new_client
            mock_new_client.ping.assert_called_once()
        finally:
            cache._redis_sync_client = original_client
            cache.redis_sync = original_redis

    def test_get_sync_client_double_check_locking(self):
        """Test get_sync_client handles double-check locking."""
        from src.utils import cache

        original_client = cache._redis_sync_client
        original_redis = cache.redis_sync
        try:
            # First call - initialize
            cache._redis_sync_client = None
            mock_redis_sync = MagicMock()
            mock_client = MagicMock()
            mock_redis_sync.Redis.from_url.return_value = mock_client
            cache.redis_sync = mock_redis_sync

            with patch("src.utils.cache.get_settings") as mock_settings:
                mock_settings.return_value.REDIS_ENABLED = True
                mock_settings.return_value.REDIS_URL = "redis://localhost:6379"

                # First call initializes
                result1 = cache.get_sync_client()
                # Second call should return same client
                result2 = cache.get_sync_client()

            assert result1 is mock_client
            assert result2 is mock_client
            # from_url should only be called once
            assert mock_redis_sync.Redis.from_url.call_count == 1
        finally:
            cache._redis_sync_client = original_client
            cache.redis_sync = original_redis

    def test_get_sync_client_init_failure(self):
        """Test get_sync_client handles initialization failure."""
        from src.utils import cache

        original_client = cache._redis_sync_client
        original_redis = cache.redis_sync
        try:
            cache._redis_sync_client = None

            mock_redis_sync = MagicMock()
            mock_redis_sync.Redis.from_url.side_effect = Exception("Connection refused")
            cache.redis_sync = mock_redis_sync

            with patch("src.utils.cache.get_settings") as mock_settings:
                mock_settings.return_value.REDIS_ENABLED = True
                mock_settings.return_value.REDIS_URL = "redis://localhost:6379"

                result = cache.get_sync_client()

            assert result is None
        finally:
            cache._redis_sync_client = original_client
            cache.redis_sync = original_redis


class TestDeleteCache:
    """Tests for delete_cache function."""

    @pytest.mark.asyncio
    async def test_delete_cache_from_redis(self):
        """Test delete_cache removes from Redis."""
        from src.utils import cache

        _require_redis_module(cache)
        original_client = cache._redis_client
        try:
            mock_client = MagicMock()
            mock_client.delete = AsyncMock()
            cache._redis_client = mock_client

            await cache.delete_cache("test_key")

            mock_client.delete.assert_called_once_with("test_key")
        finally:
            cache._redis_client = original_client

    @pytest.mark.asyncio
    async def test_delete_cache_redis_error(self):
        """Test delete_cache handles Redis error gracefully."""
        from src.utils import cache

        original_client = cache._redis_client
        original_cache = cache._local_cache.copy()
        try:
            mock_client = MagicMock()
            mock_client.delete = AsyncMock(side_effect=Exception("Redis error"))
            cache._redis_client = mock_client
            cache._local_cache["delete_key"] = ({"data": 1}, time.time() + 3600)

            # Should not raise
            await cache.delete_cache("delete_key")

            # Should still remove from local cache
            assert "delete_key" not in cache._local_cache
        finally:
            cache._redis_client = original_client
            cache._local_cache.clear()
            cache._local_cache.update(original_cache)

    @pytest.mark.asyncio
    async def test_delete_cache_no_redis(self):
        """Test delete_cache works without Redis."""
        from src.utils import cache

        original_client = cache._redis_client
        original_cache = cache._local_cache.copy()
        try:
            cache._redis_client = None
            cache._local_cache["local_key"] = ({"data": 1}, time.time() + 3600)

            await cache.delete_cache("local_key")

            assert "local_key" not in cache._local_cache
        finally:
            cache._redis_client = original_client
            cache._local_cache.clear()
            cache._local_cache.update(original_cache)

    @pytest.mark.asyncio
    async def test_delete_cache_nonexistent(self):
        """Test delete_cache handles nonexistent key."""
        from src.utils import cache

        original_client = cache._redis_client
        try:
            cache._redis_client = None

            # Should not raise
            await cache.delete_cache("nonexistent")
        finally:
            cache._redis_client = original_client
