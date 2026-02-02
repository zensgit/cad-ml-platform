"""Distributed Lock Implementations.

Provides lock implementations:
- In-memory lock (testing)
- Redis-based lock (production)
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from src.core.distributed_lock.core import (
    DistributedLock,
    FencingTokenGenerator,
    LockInfo,
    LockResult,
    LockStatus,
)

logger = logging.getLogger(__name__)


class InMemoryLock(DistributedLock):
    """In-memory distributed lock for testing."""

    def __init__(self):
        self._locks: Dict[str, LockInfo] = {}
        self._token_generator = FencingTokenGenerator()
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire(
        self,
        name: str,
        owner: str,
        ttl_seconds: float = 30.0,
        wait_timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LockResult:
        start_time = time.time()

        while True:
            async with self._get_lock():
                # Clean expired locks
                self._cleanup_expired()

                # Check if lock is available
                if name not in self._locks:
                    # Acquire lock
                    now = datetime.utcnow()
                    lock_info = LockInfo(
                        name=name,
                        owner=owner,
                        acquired_at=now,
                        expires_at=now + timedelta(seconds=ttl_seconds),
                        fencing_token=self._token_generator.next(),
                        metadata=metadata or {},
                    )
                    self._locks[name] = lock_info

                    logger.debug(f"Lock '{name}' acquired by '{owner}'")
                    return LockResult(
                        success=True,
                        lock_info=lock_info,
                        status=LockStatus.ACQUIRED,
                    )

                # Lock is held by same owner - extend it
                existing = self._locks[name]
                if existing.owner == owner:
                    existing.expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
                    return LockResult(
                        success=True,
                        lock_info=existing,
                        status=LockStatus.ACQUIRED,
                    )

            # Lock not available
            if wait_timeout is None:
                return LockResult(
                    success=False,
                    error=f"Lock '{name}' is held by '{self._locks[name].owner}'",
                    status=LockStatus.FAILED,
                )

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= wait_timeout:
                return LockResult(
                    success=False,
                    error=f"Timeout waiting for lock '{name}'",
                    status=LockStatus.FAILED,
                )

            # Wait and retry
            await asyncio.sleep(0.1)

    async def release(self, name: str, owner: str) -> bool:
        async with self._get_lock():
            if name not in self._locks:
                return False

            lock_info = self._locks[name]
            if lock_info.owner != owner:
                logger.warning(
                    f"Cannot release lock '{name}': owned by '{lock_info.owner}', "
                    f"not '{owner}'"
                )
                return False

            del self._locks[name]
            logger.debug(f"Lock '{name}' released by '{owner}'")
            return True

    async def extend(
        self,
        name: str,
        owner: str,
        additional_seconds: float,
    ) -> bool:
        async with self._get_lock():
            if name not in self._locks:
                return False

            lock_info = self._locks[name]
            if lock_info.owner != owner:
                return False

            lock_info.expires_at += timedelta(seconds=additional_seconds)
            logger.debug(f"Lock '{name}' extended by {additional_seconds}s")
            return True

    async def get_info(self, name: str) -> Optional[LockInfo]:
        async with self._get_lock():
            self._cleanup_expired()
            return self._locks.get(name)

    async def is_locked(self, name: str) -> bool:
        info = await self.get_info(name)
        return info is not None

    def _cleanup_expired(self) -> None:
        """Remove expired locks."""
        now = datetime.utcnow()
        expired = [
            name for name, info in self._locks.items()
            if info.expires_at < now
        ]
        for name in expired:
            logger.debug(f"Lock '{name}' expired")
            del self._locks[name]


class RedisLock(DistributedLock):
    """Redis-based distributed lock using Redlock algorithm."""

    # Lua script for atomic acquire
    ACQUIRE_SCRIPT = """
    local key = KEYS[1]
    local owner = ARGV[1]
    local ttl_ms = tonumber(ARGV[2])
    local token = tonumber(ARGV[3])
    local metadata = ARGV[4]

    -- Check if lock exists
    local current_owner = redis.call('HGET', key, 'owner')

    if current_owner == false then
        -- Lock is free, acquire it
        redis.call('HSET', key, 'owner', owner)
        redis.call('HSET', key, 'token', token)
        redis.call('HSET', key, 'acquired_at', ARGV[5])
        redis.call('HSET', key, 'metadata', metadata)
        redis.call('PEXPIRE', key, ttl_ms)
        return 1
    elseif current_owner == owner then
        -- Same owner, extend
        redis.call('PEXPIRE', key, ttl_ms)
        return 1
    else
        -- Lock held by someone else
        return 0
    end
    """

    # Lua script for atomic release
    RELEASE_SCRIPT = """
    local key = KEYS[1]
    local owner = ARGV[1]

    local current_owner = redis.call('HGET', key, 'owner')

    if current_owner == owner then
        redis.call('DEL', key)
        return 1
    else
        return 0
    end
    """

    # Lua script for atomic extend
    EXTEND_SCRIPT = """
    local key = KEYS[1]
    local owner = ARGV[1]
    local additional_ms = tonumber(ARGV[2])

    local current_owner = redis.call('HGET', key, 'owner')

    if current_owner == owner then
        local ttl = redis.call('PTTL', key)
        if ttl > 0 then
            redis.call('PEXPIRE', key, ttl + additional_ms)
            return 1
        end
    end
    return 0
    """

    def __init__(
        self,
        redis_client: Any = None,
        key_prefix: str = "lock:",
    ):
        self._redis = redis_client
        self._key_prefix = key_prefix
        self._token_generator = FencingTokenGenerator()
        self._scripts_registered = False

    async def _ensure_scripts(self):
        """Register Lua scripts with Redis."""
        if self._scripts_registered or self._redis is None:
            return

        # Scripts would be registered here for real Redis
        self._scripts_registered = True

    def _make_key(self, name: str) -> str:
        return f"{self._key_prefix}{name}"

    async def acquire(
        self,
        name: str,
        owner: str,
        ttl_seconds: float = 30.0,
        wait_timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LockResult:
        if self._redis is None:
            return LockResult(
                success=False,
                error="Redis client not configured",
                status=LockStatus.FAILED,
            )

        await self._ensure_scripts()
        key = self._make_key(name)
        ttl_ms = int(ttl_seconds * 1000)
        token = self._token_generator.next()
        now = datetime.utcnow()

        import json
        metadata_json = json.dumps(metadata or {})

        start_time = time.time()

        while True:
            try:
                # Execute acquire script
                result = await self._redis.eval(
                    self.ACQUIRE_SCRIPT,
                    1,
                    key,
                    owner,
                    str(ttl_ms),
                    str(token),
                    metadata_json,
                    now.isoformat(),
                )

                if result == 1:
                    lock_info = LockInfo(
                        name=name,
                        owner=owner,
                        acquired_at=now,
                        expires_at=now + timedelta(seconds=ttl_seconds),
                        fencing_token=token,
                        metadata=metadata or {},
                    )

                    logger.debug(f"Lock '{name}' acquired by '{owner}'")
                    return LockResult(
                        success=True,
                        lock_info=lock_info,
                        status=LockStatus.ACQUIRED,
                    )

                # Lock not acquired
                if wait_timeout is None:
                    return LockResult(
                        success=False,
                        error=f"Lock '{name}' is already held",
                        status=LockStatus.FAILED,
                    )

                # Check timeout
                elapsed = time.time() - start_time
                if elapsed >= wait_timeout:
                    return LockResult(
                        success=False,
                        error=f"Timeout waiting for lock '{name}'",
                        status=LockStatus.FAILED,
                    )

                # Wait and retry
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Redis lock acquire error: {e}")
                return LockResult(
                    success=False,
                    error=str(e),
                    status=LockStatus.FAILED,
                )

    async def release(self, name: str, owner: str) -> bool:
        if self._redis is None:
            return False

        key = self._make_key(name)

        try:
            result = await self._redis.eval(
                self.RELEASE_SCRIPT,
                1,
                key,
                owner,
            )
            success = result == 1
            if success:
                logger.debug(f"Lock '{name}' released by '{owner}'")
            return success

        except Exception as e:
            logger.error(f"Redis lock release error: {e}")
            return False

    async def extend(
        self,
        name: str,
        owner: str,
        additional_seconds: float,
    ) -> bool:
        if self._redis is None:
            return False

        key = self._make_key(name)
        additional_ms = int(additional_seconds * 1000)

        try:
            result = await self._redis.eval(
                self.EXTEND_SCRIPT,
                1,
                key,
                owner,
                str(additional_ms),
            )
            success = result == 1
            if success:
                logger.debug(f"Lock '{name}' extended by {additional_seconds}s")
            return success

        except Exception as e:
            logger.error(f"Redis lock extend error: {e}")
            return False

    async def get_info(self, name: str) -> Optional[LockInfo]:
        if self._redis is None:
            return None

        key = self._make_key(name)

        try:
            data = await self._redis.hgetall(key)
            if not data:
                return None

            import json

            return LockInfo(
                name=name,
                owner=data.get(b"owner", b"").decode(),
                acquired_at=datetime.fromisoformat(
                    data.get(b"acquired_at", b"").decode()
                ),
                expires_at=datetime.utcnow() + timedelta(
                    milliseconds=await self._redis.pttl(key)
                ),
                fencing_token=int(data.get(b"token", b"0")),
                metadata=json.loads(data.get(b"metadata", b"{}").decode()),
            )

        except Exception as e:
            logger.error(f"Redis get_info error: {e}")
            return None

    async def is_locked(self, name: str) -> bool:
        if self._redis is None:
            return False

        key = self._make_key(name)

        try:
            return await self._redis.exists(key) > 0
        except Exception:
            return False


class MultiLock:
    """Acquire multiple locks atomically."""

    def __init__(
        self,
        lock: DistributedLock,
        owner: str,
    ):
        self._lock = lock
        self._owner = owner
        self._acquired: list[str] = []

    async def acquire_all(
        self,
        names: list[str],
        ttl_seconds: float = 30.0,
        wait_timeout: Optional[float] = None,
    ) -> bool:
        """Acquire all locks or none."""
        # Sort names to prevent deadlocks
        sorted_names = sorted(names)

        for name in sorted_names:
            result = await self._lock.acquire(
                name=name,
                owner=self._owner,
                ttl_seconds=ttl_seconds,
                wait_timeout=wait_timeout,
            )

            if result.success:
                self._acquired.append(name)
            else:
                # Release acquired locks
                await self.release_all()
                return False

        return True

    async def release_all(self) -> None:
        """Release all acquired locks."""
        for name in reversed(self._acquired):
            await self._lock.release(name, self._owner)
        self._acquired.clear()

    async def __aenter__(self) -> "MultiLock":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release_all()
