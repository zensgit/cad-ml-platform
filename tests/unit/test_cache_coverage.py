"""Tests for cache.py to improve coverage.

Covers:
- init_redis with already initialized client
- get_cache with Redis failure fallback
- set_cache with Redis failure fallback
- Local cache expiration
- redis_healthy edge cases
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_cache_state():
    """Reset cache module state before each test."""
    import src.utils.cache as cache_module
    cache_module._redis_client = None
    cache_module._local_cache.clear()
    yield
    cache_module._redis_client = None
    cache_module._local_cache.clear()


@pytest.mark.asyncio
async def test_init_redis_already_initialized():
    """Test init_redis when client already exists."""
    import src.utils.cache as cache_module

    # Simulate already initialized client
    mock_client = MagicMock()
    cache_module._redis_client = mock_client

    await cache_module.init_redis()

    # Should not create new client
    assert cache_module._redis_client is mock_client


@pytest.mark.asyncio
async def test_init_redis_connection_failure():
    """Test init_redis when Redis connection fails."""
    import src.utils.cache as cache_module

    with patch("src.utils.cache.redis") as mock_redis:
        mock_client = AsyncMock()
        mock_client.ping.side_effect = Exception("Connection refused")
        mock_redis.from_url.return_value = mock_client

        await cache_module.init_redis()

        # Client should be None after failure
        assert cache_module._redis_client is None


@pytest.mark.asyncio
async def test_get_cache_redis_failure_fallback():
    """Test get_cache falls back to local cache on Redis failure."""
    import src.utils.cache as cache_module

    # Set up local cache
    cache_module._local_cache["test_key"] = ({"data": "value"}, time.time() + 3600)

    # Mock Redis client that fails
    mock_client = AsyncMock()
    mock_client.get.side_effect = Exception("Redis error")
    cache_module._redis_client = mock_client

    # Mock redis module as available
    with patch("src.utils.cache.redis", MagicMock()):
        result = await cache_module.get_cache("test_key")

    assert result == {"data": "value"}


@pytest.mark.asyncio
async def test_get_cache_local_expired():
    """Test get_cache removes expired local cache entries."""
    import src.utils.cache as cache_module

    # Set up expired local cache entry
    cache_module._local_cache["expired_key"] = ({"old": "data"}, time.time() - 10)

    result = await cache_module.get_cache("expired_key")

    assert result is None
    assert "expired_key" not in cache_module._local_cache


@pytest.mark.asyncio
async def test_get_cache_local_not_expired():
    """Test get_cache returns valid local cache entries."""
    import src.utils.cache as cache_module

    # Set up valid local cache entry
    cache_module._local_cache["valid_key"] = ({"fresh": "data"}, time.time() + 3600)

    result = await cache_module.get_cache("valid_key")

    assert result == {"fresh": "data"}


@pytest.mark.asyncio
async def test_set_cache_redis_failure_fallback():
    """Test set_cache falls back to local cache on Redis failure."""
    import src.utils.cache as cache_module

    # Mock Redis client that fails
    mock_client = AsyncMock()
    mock_client.setex.side_effect = Exception("Redis error")
    cache_module._redis_client = mock_client

    # Mock redis module as available
    with patch("src.utils.cache.redis", MagicMock()):
        await cache_module.set_cache("fail_key", {"test": "data"}, ttl_seconds=60)

    # Should fall back to local cache
    assert "fail_key" in cache_module._local_cache
    value, expire_time = cache_module._local_cache["fail_key"]
    assert value == {"test": "data"}
    assert expire_time > time.time()


@pytest.mark.asyncio
async def test_set_cache_local_only():
    """Test set_cache with no Redis client."""
    import src.utils.cache as cache_module

    assert cache_module._redis_client is None

    await cache_module.set_cache("local_key", {"local": "value"}, ttl_seconds=120)

    assert "local_key" in cache_module._local_cache
    value, expire_time = cache_module._local_cache["local_key"]
    assert value == {"local": "value"}


@pytest.mark.asyncio
async def test_redis_healthy_no_client():
    """Test redis_healthy returns False when no client."""
    import src.utils.cache as cache_module

    assert cache_module._redis_client is None
    result = await cache_module.redis_healthy()
    assert result is False


@pytest.mark.asyncio
async def test_redis_healthy_ping_fails():
    """Test redis_healthy returns False when ping fails."""
    import src.utils.cache as cache_module

    mock_client = AsyncMock()
    mock_client.ping.side_effect = Exception("Connection lost")
    cache_module._redis_client = mock_client

    result = await cache_module.redis_healthy()
    assert result is False


@pytest.mark.asyncio
async def test_redis_healthy_success():
    """Test redis_healthy returns True when ping succeeds."""
    import src.utils.cache as cache_module

    mock_client = AsyncMock()
    mock_client.ping.return_value = True
    cache_module._redis_client = mock_client

    result = await cache_module.redis_healthy()
    assert result is True


@pytest.mark.asyncio
async def test_get_client_returns_client():
    """Test get_client returns the Redis client."""
    import src.utils.cache as cache_module

    mock_client = MagicMock()
    cache_module._redis_client = mock_client

    result = cache_module.get_client()
    assert result is mock_client


@pytest.mark.asyncio
async def test_get_client_returns_none():
    """Test get_client returns None when no client."""
    import src.utils.cache as cache_module

    assert cache_module._redis_client is None
    result = cache_module.get_client()
    assert result is None


@pytest.mark.asyncio
async def test_cache_result_wrapper():
    """Test cache_result wrapper function."""
    import src.utils.cache as cache_module

    await cache_module.cache_result("wrapper_key", {"wrapped": "data"}, ttl=300)

    # Should be in local cache
    assert "wrapper_key" in cache_module._local_cache


@pytest.mark.asyncio
async def test_get_cached_result_wrapper():
    """Test get_cached_result wrapper function."""
    import src.utils.cache as cache_module

    cache_module._local_cache["result_key"] = ({"result": "data"}, time.time() + 3600)

    result = await cache_module.get_cached_result("result_key")
    assert result == {"result": "data"}


@pytest.mark.asyncio
async def test_get_cache_redis_returns_none():
    """Test get_cache when Redis returns None."""
    import src.utils.cache as cache_module

    mock_client = AsyncMock()
    mock_client.get.return_value = None
    cache_module._redis_client = mock_client

    with patch("src.utils.cache.redis", MagicMock()):
        result = await cache_module.get_cache("nonexistent_key")

    assert result is None


@pytest.mark.asyncio
async def test_set_cache_redis_success():
    """Test set_cache successfully writes to Redis."""
    import src.utils.cache as cache_module

    mock_client = AsyncMock()
    mock_client.setex.return_value = True
    cache_module._redis_client = mock_client

    with patch("src.utils.cache.redis", MagicMock()):
        await cache_module.set_cache("redis_key", {"redis": "data"}, ttl_seconds=60)

    mock_client.setex.assert_called_once()
    # Should NOT be in local cache when Redis succeeds
    assert "redis_key" not in cache_module._local_cache
