"""Distributed Lock Core.

Provides distributed locking primitives:
- Lock interface
- Fencing tokens
- Lock metadata
"""

from __future__ import annotations

import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Optional


class LockStatus(Enum):
    """Status of a distributed lock."""
    ACQUIRED = "acquired"
    RELEASED = "released"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass
class LockInfo:
    """Information about a lock."""
    name: str
    owner: str
    acquired_at: datetime
    expires_at: datetime
    fencing_token: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at

    @property
    def ttl_seconds(self) -> float:
        remaining = (self.expires_at - datetime.utcnow()).total_seconds()
        return max(0, remaining)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "owner": self.owner,
            "acquired_at": self.acquired_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "fencing_token": self.fencing_token,
            "metadata": self.metadata,
        }


@dataclass
class LockResult:
    """Result of a lock operation."""
    success: bool
    lock_info: Optional[LockInfo] = None
    error: Optional[str] = None
    status: LockStatus = LockStatus.FAILED


class FencingTokenGenerator:
    """Generates monotonically increasing fencing tokens."""

    def __init__(self, initial: int = 0):
        self._counter = initial

    def next(self) -> int:
        self._counter += 1
        return self._counter

    @property
    def current(self) -> int:
        return self._counter


class DistributedLock(ABC):
    """Abstract base class for distributed locks."""

    @abstractmethod
    async def acquire(
        self,
        name: str,
        owner: str,
        ttl_seconds: float = 30.0,
        wait_timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LockResult:
        """Acquire a lock.

        Args:
            name: Name of the lock
            owner: Identifier of the lock owner
            ttl_seconds: Time-to-live in seconds
            wait_timeout: Max time to wait for lock (None = no wait)
            metadata: Optional metadata to store with lock

        Returns:
            LockResult with success status and lock info
        """
        pass

    @abstractmethod
    async def release(self, name: str, owner: str) -> bool:
        """Release a lock.

        Args:
            name: Name of the lock
            owner: Identifier of the lock owner

        Returns:
            True if released, False otherwise
        """
        pass

    @abstractmethod
    async def extend(
        self,
        name: str,
        owner: str,
        additional_seconds: float,
    ) -> bool:
        """Extend lock TTL.

        Args:
            name: Name of the lock
            owner: Identifier of the lock owner
            additional_seconds: Additional time to add

        Returns:
            True if extended, False otherwise
        """
        pass

    @abstractmethod
    async def get_info(self, name: str) -> Optional[LockInfo]:
        """Get lock information.

        Args:
            name: Name of the lock

        Returns:
            LockInfo if lock exists, None otherwise
        """
        pass

    @abstractmethod
    async def is_locked(self, name: str) -> bool:
        """Check if lock is held.

        Args:
            name: Name of the lock

        Returns:
            True if locked, False otherwise
        """
        pass


class LockManager:
    """Manager for distributed locks with helper methods."""

    def __init__(self, lock: DistributedLock, owner: Optional[str] = None):
        self._lock = lock
        self._owner = owner or f"owner_{secrets.token_hex(8)}"

    @property
    def owner(self) -> str:
        return self._owner

    async def acquire(
        self,
        name: str,
        ttl_seconds: float = 30.0,
        wait_timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LockResult:
        """Acquire a lock."""
        return await self._lock.acquire(
            name=name,
            owner=self._owner,
            ttl_seconds=ttl_seconds,
            wait_timeout=wait_timeout,
            metadata=metadata,
        )

    async def release(self, name: str) -> bool:
        """Release a lock."""
        return await self._lock.release(name, self._owner)

    async def extend(self, name: str, additional_seconds: float) -> bool:
        """Extend a lock."""
        return await self._lock.extend(name, self._owner, additional_seconds)

    async def with_lock(
        self,
        name: str,
        func: Callable,
        ttl_seconds: float = 30.0,
        wait_timeout: Optional[float] = None,
    ) -> Any:
        """Execute function while holding lock.

        Args:
            name: Lock name
            func: Function to execute (can be async)
            ttl_seconds: Lock TTL
            wait_timeout: Max wait time

        Returns:
            Function result

        Raises:
            RuntimeError: If lock cannot be acquired
        """
        result = await self.acquire(name, ttl_seconds, wait_timeout)
        if not result.success:
            raise RuntimeError(f"Failed to acquire lock '{name}': {result.error}")

        try:
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return await func()
            return func()
        finally:
            await self.release(name)


class LockContext:
    """Context manager for distributed locks."""

    def __init__(
        self,
        manager: LockManager,
        name: str,
        ttl_seconds: float = 30.0,
        wait_timeout: Optional[float] = None,
        auto_extend: bool = False,
        extend_interval: float = 10.0,
    ):
        self._manager = manager
        self._name = name
        self._ttl_seconds = ttl_seconds
        self._wait_timeout = wait_timeout
        self._auto_extend = auto_extend
        self._extend_interval = extend_interval
        self._lock_result: Optional[LockResult] = None
        self._extend_task: Any = None

    @property
    def fencing_token(self) -> Optional[int]:
        if self._lock_result and self._lock_result.lock_info:
            return self._lock_result.lock_info.fencing_token
        return None

    async def __aenter__(self) -> "LockContext":
        self._lock_result = await self._manager.acquire(
            self._name,
            self._ttl_seconds,
            self._wait_timeout,
        )

        if not self._lock_result.success:
            raise RuntimeError(
                f"Failed to acquire lock '{self._name}': {self._lock_result.error}"
            )

        if self._auto_extend:
            import asyncio
            self._extend_task = asyncio.create_task(self._auto_extend_loop())

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._extend_task:
            self._extend_task.cancel()
            try:
                await self._extend_task
            except Exception:
                pass

        await self._manager.release(self._name)

    async def _auto_extend_loop(self):
        """Automatically extend lock."""
        import asyncio

        while True:
            await asyncio.sleep(self._extend_interval)
            extended = await self._manager.extend(
                self._name,
                self._ttl_seconds / 2,
            )
            if not extended:
                break


def generate_owner_id(prefix: str = "owner") -> str:
    """Generate unique owner identifier."""
    timestamp = int(time.time() * 1000)
    random_part = secrets.token_hex(4)
    return f"{prefix}_{timestamp}_{random_part}"
