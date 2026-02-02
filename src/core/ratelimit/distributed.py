"""Distributed Rate Limiting.

Provides distributed rate limiting using:
- Redis-based storage
- Lua scripts for atomicity
- Cluster support
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from src.core.ratelimit.algorithms import RateLimitResult, RateLimiter

logger = logging.getLogger(__name__)


class AsyncRedisProtocol(Protocol):
    """Protocol for async Redis client."""

    async def get(self, key: str) -> Optional[bytes]: ...
    async def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool: ...
    async def incr(self, key: str) -> int: ...
    async def expire(self, key: str, seconds: int) -> bool: ...
    async def eval(self, script: str, numkeys: int, *keys_and_args) -> Any: ...
    async def delete(self, *keys: str) -> int: ...


class InMemoryRedis:
    """In-memory Redis-like store for testing."""

    def __init__(self):
        self._data: Dict[str, Tuple[Any, Optional[float]]] = {}  # key -> (value, expire_at)
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _is_expired(self, key: str) -> bool:
        if key not in self._data:
            return True
        _, expire_at = self._data[key]
        if expire_at and time.time() > expire_at:
            del self._data[key]
            return True
        return False

    async def get(self, key: str) -> Optional[bytes]:
        async with self._get_lock():
            if self._is_expired(key):
                return None
            value, _ = self._data[key]
            return str(value).encode() if value is not None else None

    async def set(
        self,
        key: str,
        value: Any,
        ex: Optional[int] = None,
    ) -> bool:
        async with self._get_lock():
            expire_at = time.time() + ex if ex else None
            self._data[key] = (value, expire_at)
            return True

    async def incr(self, key: str) -> int:
        async with self._get_lock():
            if self._is_expired(key):
                self._data[key] = (1, None)
                return 1
            value, expire_at = self._data[key]
            new_value = int(value) + 1
            self._data[key] = (new_value, expire_at)
            return new_value

    async def expire(self, key: str, seconds: int) -> bool:
        async with self._get_lock():
            if key in self._data:
                value, _ = self._data[key]
                self._data[key] = (value, time.time() + seconds)
                return True
            return False

    async def eval(self, script: str, numkeys: int, *keys_and_args) -> Any:
        """Basic Lua script emulation for rate limiting."""
        # This is a simplified implementation
        # Real Redis would execute Lua scripts atomically
        keys = keys_and_args[:numkeys]
        args = keys_and_args[numkeys:]

        # Token bucket script emulation
        if "token" in script.lower():
            return await self._emulate_token_bucket(keys, args)
        # Sliding window script emulation
        elif "sliding" in script.lower():
            return await self._emulate_sliding_window(keys, args)
        # Fixed window script emulation
        else:
            return await self._emulate_fixed_window(keys, args)

    async def _emulate_token_bucket(
        self,
        keys: Tuple[str, ...],
        args: Tuple[Any, ...],
    ) -> List:
        """Emulate token bucket Lua script."""
        key = keys[0] if keys else "default"
        capacity = float(args[0]) if args else 10
        refill_rate = float(args[1]) if len(args) > 1 else 1
        now = float(args[2]) if len(args) > 2 else time.time()
        cost = int(args[3]) if len(args) > 3 else 1

        async with self._get_lock():
            # Get current state
            if not self._is_expired(key):
                data, _ = self._data.get(key, (None, None))
                if data:
                    parts = str(data).split(":")
                    tokens = float(parts[0])
                    last_refill = float(parts[1])
                else:
                    tokens = capacity
                    last_refill = now
            else:
                tokens = capacity
                last_refill = now

            # Refill tokens
            elapsed = now - last_refill
            tokens = min(capacity, tokens + elapsed * refill_rate)

            # Try to consume
            if tokens >= cost:
                tokens -= cost
                self._data[key] = (f"{tokens}:{now}", None)
                return [1, int(tokens)]  # allowed=True, remaining
            else:
                # Calculate retry time
                needed = cost - tokens
                retry_after = needed / refill_rate
                return [0, 0, retry_after]  # allowed=False, remaining, retry_after

    async def _emulate_sliding_window(
        self,
        keys: Tuple[str, ...],
        args: Tuple[Any, ...],
    ) -> List:
        """Emulate sliding window Lua script."""
        key = keys[0] if keys else "default"
        limit = int(args[0]) if args else 10
        window = float(args[1]) if len(args) > 1 else 60
        now = float(args[2]) if len(args) > 2 else time.time()
        cost = int(args[3]) if len(args) > 3 else 1

        async with self._get_lock():
            # Get current and previous window counts
            current_window = int(now // window)
            current_key = f"{key}:{current_window}"
            previous_key = f"{key}:{current_window - 1}"

            current_count = 0
            previous_count = 0

            if not self._is_expired(current_key):
                data, _ = self._data.get(current_key, (0, None))
                current_count = int(data) if data else 0

            if not self._is_expired(previous_key):
                data, _ = self._data.get(previous_key, (0, None))
                previous_count = int(data) if data else 0

            # Calculate weighted count
            window_position = (now % window) / window
            weighted = current_count + previous_count * (1 - window_position)

            if weighted + cost <= limit:
                # Increment current window
                self._data[current_key] = (
                    current_count + cost,
                    now + window * 2,
                )
                remaining = int(limit - weighted - cost)
                return [1, remaining]  # allowed=True, remaining
            else:
                return [0, 0]  # allowed=False, remaining

    async def _emulate_fixed_window(
        self,
        keys: Tuple[str, ...],
        args: Tuple[Any, ...],
    ) -> List:
        """Emulate fixed window Lua script."""
        key = keys[0] if keys else "default"
        limit = int(args[0]) if args else 10
        window = int(args[1]) if len(args) > 1 else 60
        cost = int(args[2]) if len(args) > 2 else 1

        async with self._get_lock():
            now = time.time()
            window_key = int(now // window)
            full_key = f"{key}:{window_key}"

            if self._is_expired(full_key):
                count = 0
            else:
                data, _ = self._data.get(full_key, (0, None))
                count = int(data) if data else 0

            if count + cost <= limit:
                window_end = (window_key + 1) * window
                self._data[full_key] = (count + cost, window_end)
                return [1, limit - count - cost]  # allowed, remaining
            else:
                return [0, 0]  # not allowed, remaining

    async def delete(self, *keys: str) -> int:
        async with self._get_lock():
            count = 0
            for key in keys:
                if key in self._data:
                    del self._data[key]
                    count += 1
            return count


class DistributedTokenBucketLimiter(RateLimiter):
    """Distributed token bucket using Redis."""

    # Lua script for atomic token bucket operation
    LUA_SCRIPT = """
    local key = KEYS[1]
    local capacity = tonumber(ARGV[1])
    local refill_rate = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])
    local cost = tonumber(ARGV[4])
    local ttl = tonumber(ARGV[5])

    local data = redis.call('GET', key)
    local tokens, last_refill

    if data then
        local parts = {}
        for part in string.gmatch(data, "[^:]+") do
            table.insert(parts, part)
        end
        tokens = tonumber(parts[1])
        last_refill = tonumber(parts[2])
    else
        tokens = capacity
        last_refill = now
    end

    -- Refill tokens
    local elapsed = now - last_refill
    tokens = math.min(capacity, tokens + elapsed * refill_rate)

    -- Try to consume
    if tokens >= cost then
        tokens = tokens - cost
        redis.call('SET', key, tokens .. ':' .. now, 'EX', ttl)
        return {1, math.floor(tokens)}
    else
        local needed = cost - tokens
        local retry_after = needed / refill_rate
        return {0, 0, retry_after}
    end
    """

    def __init__(
        self,
        redis_client: AsyncRedisProtocol,
        capacity: int,
        refill_rate: float,
        key_prefix: str = "ratelimit:token:",
        ttl_seconds: int = 3600,
    ):
        self.redis = redis_client
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds

    def _make_key(self, key: str) -> str:
        return f"{self.key_prefix}{key}"

    async def acquire(self, key: str, cost: int = 1) -> RateLimitResult:
        redis_key = self._make_key(key)
        now = time.time()

        try:
            result = await self.redis.eval(
                self.LUA_SCRIPT,
                1,
                redis_key,
                self.capacity,
                self.refill_rate,
                now,
                cost,
                self.ttl_seconds,
            )

            allowed = bool(result[0])
            remaining = int(result[1])

            if allowed:
                return RateLimitResult(
                    allowed=True,
                    remaining=remaining,
                    limit=self.capacity,
                    reset_at=now + 1.0 / self.refill_rate,
                )
            else:
                retry_after = float(result[2]) if len(result) > 2 else 1.0
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    limit=self.capacity,
                    reset_at=now + retry_after,
                    retry_after=retry_after,
                )

        except Exception as e:
            logger.error(f"Redis rate limit error: {e}")
            # Fail open - allow request on error
            return RateLimitResult(
                allowed=True,
                remaining=self.capacity,
                limit=self.capacity,
                reset_at=now + 1.0,
            )

    async def get_status(self, key: str) -> RateLimitResult:
        # Use cost=0 to check without consuming
        return await self.acquire(key, cost=0)

    async def reset(self, key: str) -> None:
        redis_key = self._make_key(key)
        await self.redis.delete(redis_key)


class DistributedSlidingWindowLimiter(RateLimiter):
    """Distributed sliding window counter using Redis."""

    LUA_SCRIPT = """
    local key = KEYS[1]
    local limit = tonumber(ARGV[1])
    local window = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])
    local cost = tonumber(ARGV[4])

    local current_window = math.floor(now / window)
    local current_key = key .. ':' .. current_window
    local previous_key = key .. ':' .. (current_window - 1)

    local current_count = tonumber(redis.call('GET', current_key) or '0')
    local previous_count = tonumber(redis.call('GET', previous_key) or '0')

    local window_position = (now % window) / window
    local weighted = current_count + previous_count * (1 - window_position)

    if weighted + cost <= limit then
        redis.call('INCR', current_key)
        if cost > 1 then
            redis.call('INCRBY', current_key, cost - 1)
        end
        redis.call('EXPIRE', current_key, window * 2)
        return {1, math.floor(limit - weighted - cost)}
    else
        return {0, 0}
    end
    """

    def __init__(
        self,
        redis_client: AsyncRedisProtocol,
        limit: int,
        window_seconds: float,
        key_prefix: str = "ratelimit:sliding:",
    ):
        self.redis = redis_client
        self.limit = limit
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix

    def _make_key(self, key: str) -> str:
        return f"{self.key_prefix}{key}"

    async def acquire(self, key: str, cost: int = 1) -> RateLimitResult:
        redis_key = self._make_key(key)
        now = time.time()

        try:
            result = await self.redis.eval(
                self.LUA_SCRIPT,
                1,
                redis_key,
                self.limit,
                self.window_seconds,
                now,
                cost,
            )

            allowed = bool(result[0])
            remaining = int(result[1])

            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                limit=self.limit,
                reset_at=now + self.window_seconds,
                retry_after=self.window_seconds if not allowed else None,
            )

        except Exception as e:
            logger.error(f"Redis rate limit error: {e}")
            return RateLimitResult(
                allowed=True,
                remaining=self.limit,
                limit=self.limit,
                reset_at=now + self.window_seconds,
            )

    async def get_status(self, key: str) -> RateLimitResult:
        return await self.acquire(key, cost=0)

    async def reset(self, key: str) -> None:
        redis_key = self._make_key(key)
        now = time.time()
        current_window = int(now // self.window_seconds)

        await self.redis.delete(
            f"{redis_key}:{current_window}",
            f"{redis_key}:{current_window - 1}",
        )


@dataclass
class RateLimitConfig:
    """Configuration for distributed rate limiting."""
    algorithm: str = "token_bucket"  # token_bucket, sliding_window, fixed_window
    limit: int = 100
    window_seconds: float = 60.0
    refill_rate: float = 1.67  # For token bucket
    key_prefix: str = "ratelimit:"
    ttl_seconds: int = 3600
    fail_open: bool = True  # Allow on error


def create_distributed_limiter(
    redis_client: AsyncRedisProtocol,
    config: RateLimitConfig,
) -> RateLimiter:
    """Create a distributed rate limiter from config."""
    if config.algorithm == "token_bucket":
        return DistributedTokenBucketLimiter(
            redis_client=redis_client,
            capacity=config.limit,
            refill_rate=config.refill_rate,
            key_prefix=f"{config.key_prefix}token:",
            ttl_seconds=config.ttl_seconds,
        )
    elif config.algorithm == "sliding_window":
        return DistributedSlidingWindowLimiter(
            redis_client=redis_client,
            limit=config.limit,
            window_seconds=config.window_seconds,
            key_prefix=f"{config.key_prefix}sliding:",
        )
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")
