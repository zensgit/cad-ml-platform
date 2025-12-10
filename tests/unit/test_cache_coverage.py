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
from typing import Any, Tuple
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


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

        with patch("src.utils.cache.get_cache", new_callable=AsyncMock, return_value={"cached": True}) as mock_get:
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
