"""Redis cache utilities (async wrapper) with in-memory fallback."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

try:
    import redis.asyncio as redis  # type: ignore
except Exception:  # redis not installed
    redis = None  # type: ignore
from src.core.config import get_settings

logger = logging.getLogger(__name__)

_redis_client: Optional[redis.Redis] = None
_init_lock = asyncio.Lock()

# In-memory cache fallback when Redis is unavailable
_local_cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, expire_time)


async def init_redis() -> None:
    global _redis_client
    if _redis_client:
        return
    async with _init_lock:
        if _redis_client:
            return
        settings = get_settings()
        try:
            _redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
            await _redis_client.ping()
            logger.info("Redis connected")
        except Exception as e:  # noqa
            logger.warning(f"Redis init failed: {e}")
            _redis_client = None


def get_client() -> Optional[redis.Redis]:
    return _redis_client


async def redis_healthy() -> bool:
    if not _redis_client:
        return False
    try:
        await _redis_client.ping()
        return True
    except Exception:
        return False


async def get_cache(key: str) -> Optional[dict]:
    # Try Redis first
    if _redis_client and redis is not None:
        try:
            raw = await _redis_client.get(key)
            if raw is not None:
                return json.loads(raw)
        except Exception as e:
            logger.debug(f"Redis get failed {key}: {e}")
    # Fall back to in-memory cache
    if key in _local_cache:
        value, expire_time = _local_cache[key]
        if time.time() < expire_time:
            return value
        else:
            del _local_cache[key]
    return None


async def set_cache(key: str, value: dict, ttl_seconds: int = 3600) -> None:
    # Try Redis first
    if _redis_client and redis is not None:
        try:
            await _redis_client.setex(key, ttl_seconds, json.dumps(value))
            return  # Redis set succeeded
        except Exception as e:
            logger.debug(f"Redis set failed {key}: {e}")
    # Fall back to in-memory cache
    _local_cache[key] = (value, time.time() + ttl_seconds)


# Compatibility wrappers for existing analysis module naming
async def cache_result(key: str, value: dict, ttl: int = 3600) -> None:
    await set_cache(key, value, ttl)


async def get_cached_result(key: str) -> Optional[dict]:
    return await get_cache(key)
