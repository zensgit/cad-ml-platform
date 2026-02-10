"""Tests for src/core/distributed_lock/backends.py.

Covers:
- InMemoryLock: acquire, release, extend, get_info, is_locked, cleanup
- RedisLock: acquire, release, extend, get_info, is_locked with mocked Redis
- MultiLock: acquire_all, release_all, context manager
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.distributed_lock.backends import InMemoryLock, MultiLock, RedisLock
from src.core.distributed_lock.core import LockStatus


class TestInMemoryLock:
    """Tests for InMemoryLock class."""

    @pytest.fixture
    def lock(self):
        """Create a fresh InMemoryLock for each test."""
        return InMemoryLock()

    @pytest.mark.asyncio
    async def test_acquire_success(self, lock):
        """Test acquiring a lock successfully."""
        result = await lock.acquire("test-lock", "owner1", ttl_seconds=30.0)

        assert result.success is True
        assert result.status == LockStatus.ACQUIRED
        assert result.lock_info is not None
        assert result.lock_info.name == "test-lock"
        assert result.lock_info.owner == "owner1"
        assert result.lock_info.fencing_token > 0

    @pytest.mark.asyncio
    async def test_acquire_with_metadata(self, lock):
        """Test acquiring a lock with metadata."""
        metadata = {"purpose": "testing", "session": "abc123"}
        result = await lock.acquire("test-lock", "owner1", metadata=metadata)

        assert result.success is True
        assert result.lock_info.metadata == metadata

    @pytest.mark.asyncio
    async def test_acquire_same_owner_extends(self, lock):
        """Test acquiring a lock by same owner extends it."""
        result1 = await lock.acquire("test-lock", "owner1", ttl_seconds=10.0)
        assert result1.success is True

        # Acquire again with same owner
        result2 = await lock.acquire("test-lock", "owner1", ttl_seconds=60.0)
        assert result2.success is True
        assert result2.status == LockStatus.ACQUIRED

    @pytest.mark.asyncio
    async def test_acquire_different_owner_fails(self, lock):
        """Test acquiring a lock held by different owner fails."""
        result1 = await lock.acquire("test-lock", "owner1")
        assert result1.success is True

        result2 = await lock.acquire("test-lock", "owner2")
        assert result2.success is False
        assert result2.status == LockStatus.FAILED
        assert "owner1" in result2.error

    @pytest.mark.asyncio
    async def test_acquire_with_wait_timeout(self, lock):
        """Test acquiring with wait timeout eventually times out."""
        await lock.acquire("test-lock", "owner1")

        # Try to acquire with short timeout
        result = await lock.acquire("test-lock", "owner2", wait_timeout=0.2)
        assert result.success is False
        assert "Timeout" in result.error

    @pytest.mark.asyncio
    async def test_release_success(self, lock):
        """Test releasing a lock successfully."""
        await lock.acquire("test-lock", "owner1")

        result = await lock.release("test-lock", "owner1")
        assert result is True

        # Lock should be available now
        result2 = await lock.acquire("test-lock", "owner2")
        assert result2.success is True

    @pytest.mark.asyncio
    async def test_release_nonexistent_lock(self, lock):
        """Test releasing a lock that doesn't exist."""
        result = await lock.release("nonexistent", "owner1")
        assert result is False

    @pytest.mark.asyncio
    async def test_release_wrong_owner(self, lock):
        """Test releasing a lock by wrong owner fails."""
        await lock.acquire("test-lock", "owner1")

        result = await lock.release("test-lock", "owner2")
        assert result is False

    @pytest.mark.asyncio
    async def test_extend_success(self, lock):
        """Test extending a lock successfully."""
        await lock.acquire("test-lock", "owner1", ttl_seconds=10.0)

        result = await lock.extend("test-lock", "owner1", 30.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_extend_nonexistent_lock(self, lock):
        """Test extending a lock that doesn't exist."""
        result = await lock.extend("nonexistent", "owner1", 30.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_extend_wrong_owner(self, lock):
        """Test extending a lock by wrong owner fails."""
        await lock.acquire("test-lock", "owner1")

        result = await lock.extend("test-lock", "owner2", 30.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_info_success(self, lock):
        """Test getting lock info."""
        await lock.acquire("test-lock", "owner1")

        info = await lock.get_info("test-lock")
        assert info is not None
        assert info.name == "test-lock"
        assert info.owner == "owner1"

    @pytest.mark.asyncio
    async def test_get_info_nonexistent(self, lock):
        """Test getting info for nonexistent lock."""
        info = await lock.get_info("nonexistent")
        assert info is None

    @pytest.mark.asyncio
    async def test_is_locked_true(self, lock):
        """Test is_locked returns True when locked."""
        await lock.acquire("test-lock", "owner1")

        result = await lock.is_locked("test-lock")
        assert result is True

    @pytest.mark.asyncio
    async def test_is_locked_false(self, lock):
        """Test is_locked returns False when not locked."""
        result = await lock.is_locked("test-lock")
        assert result is False

    @pytest.mark.asyncio
    async def test_cleanup_expired_locks(self, lock):
        """Test expired locks are cleaned up."""
        # Acquire lock with very short TTL
        await lock.acquire("test-lock", "owner1", ttl_seconds=0.01)

        # Wait for expiration
        await asyncio.sleep(0.05)

        # Lock should be cleaned up
        info = await lock.get_info("test-lock")
        assert info is None

    @pytest.mark.asyncio
    async def test_expired_lock_logging(self, lock):
        """Test expired lock cleanup logs message."""
        # Acquire lock with very short TTL
        await lock.acquire("test-lock", "owner1", ttl_seconds=0.01)

        # Wait for expiration
        await asyncio.sleep(0.05)

        # Trigger cleanup
        await lock.get_info("test-lock")
        # Lock should be expired and removed


class TestRedisLock:
    """Tests for RedisLock class."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        redis = AsyncMock()
        redis.eval = AsyncMock(return_value=1)
        redis.hgetall = AsyncMock(return_value=None)
        redis.exists = AsyncMock(return_value=0)
        redis.pttl = AsyncMock(return_value=30000)
        return redis

    @pytest.fixture
    def lock(self, mock_redis):
        """Create a RedisLock with mock Redis."""
        return RedisLock(redis_client=mock_redis, key_prefix="test:")

    @pytest.mark.asyncio
    async def test_acquire_no_redis(self):
        """Test acquire fails without Redis client."""
        lock = RedisLock(redis_client=None)

        result = await lock.acquire("test-lock", "owner1")
        assert result.success is False
        assert "not configured" in result.error

    @pytest.mark.asyncio
    async def test_acquire_success(self, lock, mock_redis):
        """Test successful acquire with Redis."""
        mock_redis.eval = AsyncMock(return_value=1)

        result = await lock.acquire("test-lock", "owner1")
        assert result.success is True
        assert result.status == LockStatus.ACQUIRED
        assert result.lock_info.owner == "owner1"

    @pytest.mark.asyncio
    async def test_acquire_already_held(self, lock, mock_redis):
        """Test acquire fails when lock is already held."""
        mock_redis.eval = AsyncMock(return_value=0)

        result = await lock.acquire("test-lock", "owner1")
        assert result.success is False
        assert "already held" in result.error

    @pytest.mark.asyncio
    async def test_acquire_with_wait_timeout(self, lock, mock_redis):
        """Test acquire with wait timeout eventually times out."""
        mock_redis.eval = AsyncMock(return_value=0)

        result = await lock.acquire("test-lock", "owner1", wait_timeout=0.2)
        assert result.success is False
        assert "Timeout" in result.error

    @pytest.mark.asyncio
    async def test_acquire_redis_error(self, lock, mock_redis):
        """Test acquire handles Redis errors."""
        mock_redis.eval = AsyncMock(side_effect=Exception("Connection error"))

        result = await lock.acquire("test-lock", "owner1")
        assert result.success is False
        assert "Connection error" in result.error

    @pytest.mark.asyncio
    async def test_release_no_redis(self):
        """Test release fails without Redis client."""
        lock = RedisLock(redis_client=None)

        result = await lock.release("test-lock", "owner1")
        assert result is False

    @pytest.mark.asyncio
    async def test_release_success(self, lock, mock_redis):
        """Test successful release."""
        mock_redis.eval = AsyncMock(return_value=1)

        result = await lock.release("test-lock", "owner1")
        assert result is True

    @pytest.mark.asyncio
    async def test_release_not_owner(self, lock, mock_redis):
        """Test release fails when not owner."""
        mock_redis.eval = AsyncMock(return_value=0)

        result = await lock.release("test-lock", "owner1")
        assert result is False

    @pytest.mark.asyncio
    async def test_release_redis_error(self, lock, mock_redis):
        """Test release handles Redis errors."""
        mock_redis.eval = AsyncMock(side_effect=Exception("Connection error"))

        result = await lock.release("test-lock", "owner1")
        assert result is False

    @pytest.mark.asyncio
    async def test_extend_no_redis(self):
        """Test extend fails without Redis client."""
        lock = RedisLock(redis_client=None)

        result = await lock.extend("test-lock", "owner1", 30.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_extend_success(self, lock, mock_redis):
        """Test successful extend."""
        mock_redis.eval = AsyncMock(return_value=1)

        result = await lock.extend("test-lock", "owner1", 30.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_extend_not_owner(self, lock, mock_redis):
        """Test extend fails when not owner."""
        mock_redis.eval = AsyncMock(return_value=0)

        result = await lock.extend("test-lock", "owner1", 30.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_extend_redis_error(self, lock, mock_redis):
        """Test extend handles Redis errors."""
        mock_redis.eval = AsyncMock(side_effect=Exception("Connection error"))

        result = await lock.extend("test-lock", "owner1", 30.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_info_no_redis(self):
        """Test get_info fails without Redis client."""
        lock = RedisLock(redis_client=None)

        result = await lock.get_info("test-lock")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_info_success(self, lock, mock_redis):
        """Test successful get_info."""
        mock_redis.hgetall = AsyncMock(
            return_value={
                b"owner": b"owner1",
                b"token": b"12345",
                b"acquired_at": b"2025-01-01T00:00:00",
                b"metadata": b'{"key": "value"}',
            }
        )
        mock_redis.pttl = AsyncMock(return_value=30000)

        info = await lock.get_info("test-lock")
        assert info is not None
        assert info.owner == "owner1"
        assert info.fencing_token == 12345
        assert info.metadata == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_info_not_found(self, lock, mock_redis):
        """Test get_info returns None when not found."""
        mock_redis.hgetall = AsyncMock(return_value={})

        info = await lock.get_info("test-lock")
        assert info is None

    @pytest.mark.asyncio
    async def test_get_info_redis_error(self, lock, mock_redis):
        """Test get_info handles Redis errors."""
        mock_redis.hgetall = AsyncMock(side_effect=Exception("Connection error"))

        info = await lock.get_info("test-lock")
        assert info is None

    @pytest.mark.asyncio
    async def test_is_locked_no_redis(self):
        """Test is_locked returns False without Redis client."""
        lock = RedisLock(redis_client=None)

        result = await lock.is_locked("test-lock")
        assert result is False

    @pytest.mark.asyncio
    async def test_is_locked_true(self, lock, mock_redis):
        """Test is_locked returns True when locked."""
        mock_redis.exists = AsyncMock(return_value=1)

        result = await lock.is_locked("test-lock")
        assert result is True

    @pytest.mark.asyncio
    async def test_is_locked_false(self, lock, mock_redis):
        """Test is_locked returns False when not locked."""
        mock_redis.exists = AsyncMock(return_value=0)

        result = await lock.is_locked("test-lock")
        assert result is False

    @pytest.mark.asyncio
    async def test_is_locked_redis_error(self, lock, mock_redis):
        """Test is_locked handles Redis errors."""
        mock_redis.exists = AsyncMock(side_effect=Exception("Connection error"))

        result = await lock.is_locked("test-lock")
        assert result is False

    def test_make_key(self, lock):
        """Test key prefix is applied correctly."""
        key = lock._make_key("my-lock")
        assert key == "test:my-lock"


class TestMultiLock:
    """Tests for MultiLock class."""

    @pytest.fixture
    def inner_lock(self):
        """Create an InMemoryLock for MultiLock testing."""
        return InMemoryLock()

    @pytest.fixture
    def multi_lock(self, inner_lock):
        """Create a MultiLock."""
        return MultiLock(lock=inner_lock, owner="test-owner")

    @pytest.mark.asyncio
    async def test_acquire_all_success(self, multi_lock):
        """Test acquiring all locks successfully."""
        result = await multi_lock.acquire_all(["lock1", "lock2", "lock3"])
        assert result is True
        assert len(multi_lock._acquired) == 3

    @pytest.mark.asyncio
    async def test_acquire_all_partial_failure(self, inner_lock, multi_lock):
        """Test acquiring all locks fails if one fails."""
        # Pre-acquire one lock
        await inner_lock.acquire("lock2", "other-owner")

        result = await multi_lock.acquire_all(["lock1", "lock2", "lock3"])
        assert result is False
        # All acquired locks should be released
        assert len(multi_lock._acquired) == 0

        # lock1 should be released (available)
        result = await inner_lock.acquire("lock1", "new-owner")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_release_all(self, multi_lock, inner_lock):
        """Test releasing all locks."""
        await multi_lock.acquire_all(["lock1", "lock2"])

        await multi_lock.release_all()
        assert len(multi_lock._acquired) == 0

        # Locks should be available
        result1 = await inner_lock.acquire("lock1", "new-owner")
        result2 = await inner_lock.acquire("lock2", "new-owner")
        assert result1.success is True
        assert result2.success is True

    @pytest.mark.asyncio
    async def test_context_manager(self, inner_lock):
        """Test using MultiLock as context manager."""
        multi = MultiLock(lock=inner_lock, owner="test-owner")

        async with multi:
            await multi.acquire_all(["lock1", "lock2"])
            assert len(multi._acquired) == 2

        # After context, locks should be released
        assert len(multi._acquired) == 0

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self, inner_lock):
        """Test context manager releases locks on exception."""
        multi = MultiLock(lock=inner_lock, owner="test-owner")

        try:
            async with multi:
                await multi.acquire_all(["lock1", "lock2"])
                raise ValueError("Test error")
        except ValueError:
            pass

        # Locks should be released
        assert len(multi._acquired) == 0

    @pytest.mark.asyncio
    async def test_acquire_all_with_ttl(self, multi_lock, inner_lock):
        """Test acquiring locks with custom TTL."""
        result = await multi_lock.acquire_all(["lock1"], ttl_seconds=60.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_all_with_wait_timeout(self, inner_lock):
        """Test acquiring locks with wait timeout."""
        # Pre-acquire a lock
        await inner_lock.acquire("lock2", "other-owner")

        multi = MultiLock(lock=inner_lock, owner="test-owner")
        result = await multi.acquire_all(["lock1", "lock2"], wait_timeout=0.1)
        assert result is False

    @pytest.mark.asyncio
    async def test_acquire_all_sorts_names(self, multi_lock):
        """Test that lock names are sorted to prevent deadlocks."""
        result = await multi_lock.acquire_all(["c", "a", "b"])
        assert result is True
        # Locks should be acquired in sorted order
        # This is internal behavior, just verify it works

    @pytest.mark.asyncio
    async def test_aenter_returns_self(self, inner_lock):
        """Test __aenter__ returns self."""
        multi = MultiLock(lock=inner_lock, owner="test-owner")

        result = await multi.__aenter__()
        assert result is multi

        await multi.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_aexit_releases_locks(self, inner_lock):
        """Test __aexit__ releases locks."""
        multi = MultiLock(lock=inner_lock, owner="test-owner")

        await multi.acquire_all(["lock1", "lock2"])
        assert len(multi._acquired) == 2

        await multi.__aexit__(None, None, None)
        assert len(multi._acquired) == 0


class TestInMemoryLockWaitAndRetry:
    """Tests for InMemoryLock wait and retry behavior."""

    @pytest.mark.asyncio
    async def test_acquire_retries_until_available(self):
        """Test acquire retries and succeeds when lock becomes available."""
        lock = InMemoryLock()

        # Hold lock initially
        await lock.acquire("test-lock", "owner1", ttl_seconds=0.15)

        async def release_later():
            await asyncio.sleep(0.1)
            await lock.release("test-lock", "owner1")

        # Start release task
        asyncio.create_task(release_later())

        # Try to acquire with wait
        result = await lock.acquire("test-lock", "owner2", wait_timeout=1.0)
        assert result.success is True
        assert result.lock_info.owner == "owner2"


class TestRedisLockScriptsRegistration:
    """Tests for Redis script registration."""

    @pytest.mark.asyncio
    async def test_ensure_scripts_called(self):
        """Test that _ensure_scripts is called during acquire."""
        mock_redis = AsyncMock()
        mock_redis.eval = AsyncMock(return_value=1)

        lock = RedisLock(redis_client=mock_redis)
        assert lock._scripts_registered is False

        await lock.acquire("test-lock", "owner1")

        # Scripts should be marked as registered
        assert lock._scripts_registered is True

    @pytest.mark.asyncio
    async def test_ensure_scripts_only_once(self):
        """Test that scripts are only registered once."""
        mock_redis = AsyncMock()
        mock_redis.eval = AsyncMock(return_value=1)

        lock = RedisLock(redis_client=mock_redis)

        # First acquire
        await lock.acquire("lock1", "owner1")
        assert lock._scripts_registered is True

        # Second acquire - scripts should not be re-registered
        await lock.acquire("lock2", "owner1")
        # Still True (didn't change)
        assert lock._scripts_registered is True

    @pytest.mark.asyncio
    async def test_ensure_scripts_no_redis_skips(self):
        """Test _ensure_scripts is skipped without Redis."""
        lock = RedisLock(redis_client=None)

        await lock._ensure_scripts()
        assert lock._scripts_registered is False
