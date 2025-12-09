"""
Distributed Locking for Vision Provider.

This module provides distributed locking mechanisms including:
- Lock acquisition and release
- Lock with timeout and automatic expiry
- Reentrant locks
- Read-write locks
- Lock fencing tokens
- Deadlock detection

Phase 10 Feature.
"""

import asyncio
import hashlib
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from .base import VisionDescription, VisionProvider

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Lock Enums
# ============================================================================


class LockType(Enum):
    """Types of locks."""

    EXCLUSIVE = "exclusive"
    SHARED = "shared"
    READ = "read"
    WRITE = "write"
    REENTRANT = "reentrant"


class LockState(Enum):
    """Lock states."""

    UNLOCKED = "unlocked"
    LOCKED = "locked"
    WAITING = "waiting"
    EXPIRED = "expired"
    RELEASED = "released"


class LockAcquisitionResult(Enum):
    """Result of lock acquisition attempt."""

    ACQUIRED = "acquired"
    ALREADY_HELD = "already_held"
    TIMEOUT = "timeout"
    FAILED = "failed"
    DEADLOCK_DETECTED = "deadlock_detected"


# ============================================================================
# Lock Data Classes
# ============================================================================


@dataclass
class LockInfo:
    """Information about a lock."""

    resource_id: str
    lock_type: LockType
    owner_id: str
    acquired_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    fencing_token: int = 0
    reentry_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if lock has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def extend(self, duration: timedelta) -> None:
        """Extend the lock expiration."""
        if self.expires_at:
            self.expires_at = datetime.now() + duration


@dataclass
class LockRequest:
    """Request for acquiring a lock."""

    resource_id: str
    owner_id: str
    lock_type: LockType = LockType.EXCLUSIVE
    timeout_seconds: float = 30.0
    ttl_seconds: Optional[float] = None
    wait: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LockResult:
    """Result of a lock operation."""

    success: bool
    result: LockAcquisitionResult
    lock_info: Optional[LockInfo] = None
    error: Optional[str] = None
    wait_time_ms: float = 0.0


@dataclass
class DeadlockInfo:
    """Information about a detected deadlock."""

    cycle: List[str]  # Owner IDs in the deadlock cycle
    resources: List[str]  # Resources involved
    detected_at: datetime = field(default_factory=datetime.now)


# ============================================================================
# Abstract Lock Interface
# ============================================================================


class DistributedLock(ABC):
    """Abstract interface for distributed locks."""

    @abstractmethod
    async def acquire(
        self,
        resource_id: str,
        owner_id: str,
        timeout_seconds: float = 30.0,
        ttl_seconds: Optional[float] = None,
    ) -> LockResult:
        """Acquire a lock."""
        pass

    @abstractmethod
    async def release(self, resource_id: str, owner_id: str) -> bool:
        """Release a lock."""
        pass

    @abstractmethod
    async def is_locked(self, resource_id: str) -> bool:
        """Check if resource is locked."""
        pass

    @abstractmethod
    async def get_lock_info(self, resource_id: str) -> Optional[LockInfo]:
        """Get information about a lock."""
        pass

    @abstractmethod
    async def extend(
        self,
        resource_id: str,
        owner_id: str,
        extension_seconds: float,
    ) -> bool:
        """Extend a lock's TTL."""
        pass


# ============================================================================
# In-Memory Lock Implementation
# ============================================================================


class InMemoryLock(DistributedLock):
    """
    In-memory distributed lock implementation.

    Suitable for single-process applications and testing.
    """

    def __init__(self) -> None:
        """Initialize in-memory lock store."""
        self._locks: Dict[str, LockInfo] = {}
        self._waiters: Dict[str, List[asyncio.Event]] = {}
        self._fencing_counter = 0
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        resource_id: str,
        owner_id: str,
        timeout_seconds: float = 30.0,
        ttl_seconds: Optional[float] = None,
    ) -> LockResult:
        """Acquire an exclusive lock."""
        start_time = time.time()

        while True:
            async with self._lock:
                # Check for expired lock
                if resource_id in self._locks:
                    existing = self._locks[resource_id]
                    if existing.is_expired:
                        del self._locks[resource_id]
                        logger.debug(f"Expired lock removed: {resource_id}")

                # Try to acquire
                if resource_id not in self._locks:
                    self._fencing_counter += 1
                    expires_at = None
                    if ttl_seconds:
                        expires_at = datetime.now() + timedelta(seconds=ttl_seconds)

                    lock_info = LockInfo(
                        resource_id=resource_id,
                        lock_type=LockType.EXCLUSIVE,
                        owner_id=owner_id,
                        expires_at=expires_at,
                        fencing_token=self._fencing_counter,
                    )
                    self._locks[resource_id] = lock_info

                    logger.debug(
                        f"Lock acquired: {resource_id} by {owner_id} "
                        f"(token: {self._fencing_counter})"
                    )

                    return LockResult(
                        success=True,
                        result=LockAcquisitionResult.ACQUIRED,
                        lock_info=lock_info,
                        wait_time_ms=(time.time() - start_time) * 1000,
                    )

                # Already held by same owner
                existing = self._locks[resource_id]
                if existing.owner_id == owner_id:
                    return LockResult(
                        success=True,
                        result=LockAcquisitionResult.ALREADY_HELD,
                        lock_info=existing,
                    )

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                logger.debug(
                    f"Lock timeout: {resource_id} for {owner_id} after {elapsed:.2f}s"
                )
                return LockResult(
                    success=False,
                    result=LockAcquisitionResult.TIMEOUT,
                    wait_time_ms=elapsed * 1000,
                )

            # Wait for lock release
            event = asyncio.Event()
            async with self._lock:
                if resource_id not in self._waiters:
                    self._waiters[resource_id] = []
                self._waiters[resource_id].append(event)

            try:
                remaining = timeout_seconds - elapsed
                await asyncio.wait_for(event.wait(), timeout=min(remaining, 1.0))
            except asyncio.TimeoutError:
                pass
            finally:
                async with self._lock:
                    if resource_id in self._waiters:
                        try:
                            self._waiters[resource_id].remove(event)
                        except ValueError:
                            pass

    async def release(self, resource_id: str, owner_id: str) -> bool:
        """Release a lock."""
        async with self._lock:
            if resource_id not in self._locks:
                return False

            lock_info = self._locks[resource_id]
            if lock_info.owner_id != owner_id:
                logger.warning(
                    f"Cannot release lock: {resource_id} owned by "
                    f"{lock_info.owner_id}, not {owner_id}"
                )
                return False

            del self._locks[resource_id]
            logger.debug(f"Lock released: {resource_id} by {owner_id}")

            # Notify waiters
            if resource_id in self._waiters:
                for event in self._waiters[resource_id]:
                    event.set()
                del self._waiters[resource_id]

            return True

    async def is_locked(self, resource_id: str) -> bool:
        """Check if resource is locked."""
        async with self._lock:
            if resource_id in self._locks:
                if self._locks[resource_id].is_expired:
                    del self._locks[resource_id]
                    return False
                return True
            return False

    async def get_lock_info(self, resource_id: str) -> Optional[LockInfo]:
        """Get lock information."""
        async with self._lock:
            lock_info = self._locks.get(resource_id)
            if lock_info and lock_info.is_expired:
                del self._locks[resource_id]
                return None
            return lock_info

    async def extend(
        self,
        resource_id: str,
        owner_id: str,
        extension_seconds: float,
    ) -> bool:
        """Extend lock TTL."""
        async with self._lock:
            if resource_id not in self._locks:
                return False

            lock_info = self._locks[resource_id]
            if lock_info.owner_id != owner_id:
                return False

            lock_info.extend(timedelta(seconds=extension_seconds))
            logger.debug(
                f"Lock extended: {resource_id} by {extension_seconds}s"
            )
            return True


# ============================================================================
# Reentrant Lock
# ============================================================================


class ReentrantLock(DistributedLock):
    """
    Reentrant distributed lock.

    Allows the same owner to acquire the lock multiple times.
    """

    def __init__(self, base_lock: Optional[DistributedLock] = None) -> None:
        """Initialize reentrant lock."""
        self._base_lock = base_lock or InMemoryLock()
        self._reentry_counts: Dict[str, Dict[str, int]] = {}
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        resource_id: str,
        owner_id: str,
        timeout_seconds: float = 30.0,
        ttl_seconds: Optional[float] = None,
    ) -> LockResult:
        """Acquire lock with reentry support."""
        async with self._lock:
            # Check for existing reentry
            if resource_id in self._reentry_counts:
                if owner_id in self._reentry_counts[resource_id]:
                    self._reentry_counts[resource_id][owner_id] += 1
                    lock_info = await self._base_lock.get_lock_info(resource_id)
                    if lock_info:
                        lock_info.reentry_count = self._reentry_counts[resource_id][
                            owner_id
                        ]
                    return LockResult(
                        success=True,
                        result=LockAcquisitionResult.ALREADY_HELD,
                        lock_info=lock_info,
                    )

        # Try to acquire base lock
        result = await self._base_lock.acquire(
            resource_id, owner_id, timeout_seconds, ttl_seconds
        )

        if result.success:
            async with self._lock:
                if resource_id not in self._reentry_counts:
                    self._reentry_counts[resource_id] = {}
                self._reentry_counts[resource_id][owner_id] = 1

        return result

    async def release(self, resource_id: str, owner_id: str) -> bool:
        """Release lock with reentry support."""
        async with self._lock:
            if resource_id not in self._reentry_counts:
                return False

            if owner_id not in self._reentry_counts[resource_id]:
                return False

            self._reentry_counts[resource_id][owner_id] -= 1

            if self._reentry_counts[resource_id][owner_id] <= 0:
                del self._reentry_counts[resource_id][owner_id]
                if not self._reentry_counts[resource_id]:
                    del self._reentry_counts[resource_id]

                # Release base lock
                return await self._base_lock.release(resource_id, owner_id)

            # Still have reentry count, don't release base lock
            return True

    async def is_locked(self, resource_id: str) -> bool:
        """Check if locked."""
        return await self._base_lock.is_locked(resource_id)

    async def get_lock_info(self, resource_id: str) -> Optional[LockInfo]:
        """Get lock info with reentry count."""
        lock_info = await self._base_lock.get_lock_info(resource_id)
        if lock_info and resource_id in self._reentry_counts:
            owner = lock_info.owner_id
            if owner in self._reentry_counts[resource_id]:
                lock_info.reentry_count = self._reentry_counts[resource_id][owner]
        return lock_info

    async def extend(
        self,
        resource_id: str,
        owner_id: str,
        extension_seconds: float,
    ) -> bool:
        """Extend lock TTL."""
        return await self._base_lock.extend(resource_id, owner_id, extension_seconds)


# ============================================================================
# Read-Write Lock
# ============================================================================


class ReadWriteLock:
    """
    Read-write lock implementation.

    Allows multiple readers or a single writer.
    """

    def __init__(self) -> None:
        """Initialize read-write lock."""
        self._readers: Dict[str, Set[str]] = {}  # resource_id -> set of reader owners
        self._writers: Dict[str, str] = {}  # resource_id -> writer owner
        self._reader_count: Dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._write_events: Dict[str, asyncio.Event] = {}
        self._read_events: Dict[str, asyncio.Event] = {}

    async def acquire_read(
        self,
        resource_id: str,
        owner_id: str,
        timeout_seconds: float = 30.0,
    ) -> LockResult:
        """Acquire a read lock."""
        start_time = time.time()

        while True:
            async with self._lock:
                # Check if there's a writer
                if resource_id not in self._writers:
                    # No writer, can acquire read lock
                    if resource_id not in self._readers:
                        self._readers[resource_id] = set()
                        self._reader_count[resource_id] = 0

                    self._readers[resource_id].add(owner_id)
                    self._reader_count[resource_id] += 1

                    lock_info = LockInfo(
                        resource_id=resource_id,
                        lock_type=LockType.READ,
                        owner_id=owner_id,
                        metadata={"reader_count": self._reader_count[resource_id]},
                    )

                    return LockResult(
                        success=True,
                        result=LockAcquisitionResult.ACQUIRED,
                        lock_info=lock_info,
                        wait_time_ms=(time.time() - start_time) * 1000,
                    )

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                return LockResult(
                    success=False,
                    result=LockAcquisitionResult.TIMEOUT,
                    wait_time_ms=elapsed * 1000,
                )

            # Wait for writer to release
            event = asyncio.Event()
            async with self._lock:
                self._write_events[resource_id] = event

            try:
                remaining = timeout_seconds - elapsed
                await asyncio.wait_for(event.wait(), timeout=min(remaining, 1.0))
            except asyncio.TimeoutError:
                pass

    async def acquire_write(
        self,
        resource_id: str,
        owner_id: str,
        timeout_seconds: float = 30.0,
    ) -> LockResult:
        """Acquire a write lock."""
        start_time = time.time()

        while True:
            async with self._lock:
                # Check if there are readers or another writer
                reader_count = self._reader_count.get(resource_id, 0)
                has_writer = resource_id in self._writers

                if reader_count == 0 and not has_writer:
                    # Can acquire write lock
                    self._writers[resource_id] = owner_id

                    lock_info = LockInfo(
                        resource_id=resource_id,
                        lock_type=LockType.WRITE,
                        owner_id=owner_id,
                    )

                    return LockResult(
                        success=True,
                        result=LockAcquisitionResult.ACQUIRED,
                        lock_info=lock_info,
                        wait_time_ms=(time.time() - start_time) * 1000,
                    )

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                return LockResult(
                    success=False,
                    result=LockAcquisitionResult.TIMEOUT,
                    wait_time_ms=elapsed * 1000,
                )

            # Wait for readers/writer to release
            event = asyncio.Event()
            async with self._lock:
                self._read_events[resource_id] = event

            try:
                remaining = timeout_seconds - elapsed
                await asyncio.wait_for(event.wait(), timeout=min(remaining, 1.0))
            except asyncio.TimeoutError:
                pass

    async def release_read(self, resource_id: str, owner_id: str) -> bool:
        """Release a read lock."""
        async with self._lock:
            if resource_id not in self._readers:
                return False

            if owner_id not in self._readers[resource_id]:
                return False

            self._readers[resource_id].discard(owner_id)
            self._reader_count[resource_id] -= 1

            if self._reader_count[resource_id] <= 0:
                del self._readers[resource_id]
                del self._reader_count[resource_id]

                # Notify waiting writers
                if resource_id in self._read_events:
                    self._read_events[resource_id].set()
                    del self._read_events[resource_id]

            return True

    async def release_write(self, resource_id: str, owner_id: str) -> bool:
        """Release a write lock."""
        async with self._lock:
            if resource_id not in self._writers:
                return False

            if self._writers[resource_id] != owner_id:
                return False

            del self._writers[resource_id]

            # Notify waiting readers and writers
            if resource_id in self._write_events:
                self._write_events[resource_id].set()
                del self._write_events[resource_id]

            if resource_id in self._read_events:
                self._read_events[resource_id].set()
                del self._read_events[resource_id]

            return True

    async def is_read_locked(self, resource_id: str) -> bool:
        """Check if resource has read locks."""
        async with self._lock:
            return self._reader_count.get(resource_id, 0) > 0

    async def is_write_locked(self, resource_id: str) -> bool:
        """Check if resource has write lock."""
        async with self._lock:
            return resource_id in self._writers

    async def get_reader_count(self, resource_id: str) -> int:
        """Get number of readers."""
        async with self._lock:
            return self._reader_count.get(resource_id, 0)


# ============================================================================
# Lock Manager
# ============================================================================


class LockManager:
    """
    Central manager for distributed locks.

    Provides lock acquisition, monitoring, and deadlock detection.
    """

    def __init__(
        self,
        lock_impl: Optional[DistributedLock] = None,
        enable_deadlock_detection: bool = True,
    ) -> None:
        """Initialize lock manager."""
        self._lock_impl = lock_impl or InMemoryLock()
        self._deadlock_detection = enable_deadlock_detection
        self._wait_graph: Dict[str, Set[str]] = {}  # owner -> set of owners waiting for
        self._held_locks: Dict[str, Set[str]] = {}  # owner -> set of resources held
        self._lock = asyncio.Lock()
        self._deadlock_callbacks: List[Callable[[DeadlockInfo], None]] = []

    async def acquire(self, request: LockRequest) -> LockResult:
        """Acquire a lock with deadlock detection."""
        if self._deadlock_detection:
            # Check for potential deadlock before attempting
            async with self._lock:
                # Record wait edge
                current_owner = await self._get_resource_owner(request.resource_id)
                if current_owner and current_owner != request.owner_id:
                    self._wait_graph.setdefault(request.owner_id, set()).add(
                        current_owner
                    )

                    # Check for cycle
                    cycle = self._detect_cycle(request.owner_id)
                    if cycle:
                        # Remove wait edge
                        self._wait_graph[request.owner_id].discard(current_owner)

                        deadlock = DeadlockInfo(
                            cycle=cycle,
                            resources=[request.resource_id],
                        )
                        self._notify_deadlock(deadlock)

                        return LockResult(
                            success=False,
                            result=LockAcquisitionResult.DEADLOCK_DETECTED,
                            error=f"Deadlock detected: {' -> '.join(cycle)}",
                        )

        # Attempt to acquire lock
        result = await self._lock_impl.acquire(
            resource_id=request.resource_id,
            owner_id=request.owner_id,
            timeout_seconds=request.timeout_seconds,
            ttl_seconds=request.ttl_seconds,
        )

        # Update tracking
        async with self._lock:
            # Remove wait edge
            if request.owner_id in self._wait_graph:
                owner = await self._get_resource_owner(request.resource_id)
                if owner:
                    self._wait_graph[request.owner_id].discard(owner)

            if result.success:
                # Track held lock
                self._held_locks.setdefault(request.owner_id, set()).add(
                    request.resource_id
                )

        return result

    async def release(self, resource_id: str, owner_id: str) -> bool:
        """Release a lock."""
        result = await self._lock_impl.release(resource_id, owner_id)

        if result:
            async with self._lock:
                if owner_id in self._held_locks:
                    self._held_locks[owner_id].discard(resource_id)

        return result

    async def try_acquire(
        self,
        resource_id: str,
        owner_id: str,
        ttl_seconds: Optional[float] = None,
    ) -> LockResult:
        """Try to acquire lock without waiting."""
        request = LockRequest(
            resource_id=resource_id,
            owner_id=owner_id,
            timeout_seconds=0,
            ttl_seconds=ttl_seconds,
            wait=False,
        )
        return await self.acquire(request)

    async def release_all(self, owner_id: str) -> int:
        """Release all locks held by an owner."""
        count = 0
        async with self._lock:
            resources = list(self._held_locks.get(owner_id, set()))

        for resource_id in resources:
            if await self.release(resource_id, owner_id):
                count += 1

        return count

    async def get_held_locks(self, owner_id: str) -> List[LockInfo]:
        """Get all locks held by an owner."""
        async with self._lock:
            resources = list(self._held_locks.get(owner_id, set()))

        locks: List[LockInfo] = []
        for resource_id in resources:
            lock_info = await self._lock_impl.get_lock_info(resource_id)
            if lock_info:
                locks.append(lock_info)

        return locks

    async def _get_resource_owner(self, resource_id: str) -> Optional[str]:
        """Get the owner of a resource lock."""
        lock_info = await self._lock_impl.get_lock_info(resource_id)
        return lock_info.owner_id if lock_info else None

    def _detect_cycle(self, start: str) -> Optional[List[str]]:
        """Detect cycle in wait graph using DFS."""
        visited: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> Optional[List[str]]:
            if node in path:
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]

            if node in visited:
                return None

            visited.add(node)
            path.append(node)

            for neighbor in self._wait_graph.get(node, set()):
                result = dfs(neighbor)
                if result:
                    return result

            path.pop()
            return None

        return dfs(start)

    def on_deadlock(self, callback: Callable[[DeadlockInfo], None]) -> None:
        """Register deadlock callback."""
        self._deadlock_callbacks.append(callback)

    def _notify_deadlock(self, deadlock: DeadlockInfo) -> None:
        """Notify registered callbacks about deadlock."""
        logger.warning(f"Deadlock detected: {deadlock.cycle}")
        for callback in self._deadlock_callbacks:
            try:
                callback(deadlock)
            except Exception as e:
                logger.error(f"Error in deadlock callback: {e}")

    @asynccontextmanager
    async def lock(
        self,
        resource_id: str,
        owner_id: str,
        timeout_seconds: float = 30.0,
        ttl_seconds: Optional[float] = None,
    ) -> AsyncIterator[LockInfo]:
        """Context manager for acquiring and releasing locks."""
        request = LockRequest(
            resource_id=resource_id,
            owner_id=owner_id,
            timeout_seconds=timeout_seconds,
            ttl_seconds=ttl_seconds,
        )

        result = await self.acquire(request)

        if not result.success:
            raise TimeoutError(
                f"Failed to acquire lock: {result.result.value}"
            )

        try:
            yield result.lock_info  # type: ignore
        finally:
            await self.release(resource_id, owner_id)


# ============================================================================
# Fencing Token Manager
# ============================================================================


class FencingTokenManager:
    """
    Manage fencing tokens for lock safety.

    Fencing tokens help prevent issues with delayed messages.
    """

    def __init__(self) -> None:
        """Initialize token manager."""
        self._tokens: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def get_token(self, resource_id: str) -> int:
        """Get current fencing token for a resource."""
        async with self._lock:
            return self._tokens.get(resource_id, 0)

    async def increment_token(self, resource_id: str) -> int:
        """Increment and return new token."""
        async with self._lock:
            current = self._tokens.get(resource_id, 0)
            new_token = current + 1
            self._tokens[resource_id] = new_token
            return new_token

    async def validate_token(self, resource_id: str, token: int) -> bool:
        """Validate that token is current."""
        async with self._lock:
            current = self._tokens.get(resource_id, 0)
            return token >= current


# ============================================================================
# Lock-Protected Vision Provider
# ============================================================================


class LockedVisionProvider(VisionProvider):
    """
    Vision provider with distributed locking.

    Ensures thread-safe access to underlying provider.
    """

    def __init__(
        self,
        provider: VisionProvider,
        lock_manager: LockManager,
        owner_id: Optional[str] = None,
    ) -> None:
        """Initialize locked provider."""
        self._provider = provider
        self._lock_manager = lock_manager
        self._owner_id = owner_id or str(uuid.uuid4())
        self._resource_id = f"vision_provider_{provider.provider_name}"

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"locked_{self._provider.provider_name}"

    async def analyze_image(
        self, image_data: bytes, context: Optional[str] = None
    ) -> VisionDescription:
        """Analyze image with lock protection."""
        async with self._lock_manager.lock(
            resource_id=self._resource_id,
            owner_id=self._owner_id,
            timeout_seconds=60.0,
            ttl_seconds=120.0,
        ):
            return await self._provider.analyze_image(image_data, context)


# ============================================================================
# Semaphore Implementation
# ============================================================================


class DistributedSemaphore:
    """
    Distributed semaphore for limiting concurrent access.

    Allows up to N concurrent holders.
    """

    def __init__(self, max_permits: int = 1) -> None:
        """Initialize semaphore."""
        self._max_permits = max_permits
        self._permits: Dict[str, Set[str]] = {}  # resource_id -> set of owners
        self._lock = asyncio.Lock()
        self._events: Dict[str, asyncio.Event] = {}

    async def acquire(
        self,
        resource_id: str,
        owner_id: str,
        timeout_seconds: float = 30.0,
    ) -> bool:
        """Acquire a permit."""
        start_time = time.time()

        while True:
            async with self._lock:
                if resource_id not in self._permits:
                    self._permits[resource_id] = set()

                current_count = len(self._permits[resource_id])

                if current_count < self._max_permits:
                    self._permits[resource_id].add(owner_id)
                    return True

                if owner_id in self._permits[resource_id]:
                    # Already has permit
                    return True

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                return False

            # Wait for release
            event = asyncio.Event()
            async with self._lock:
                self._events[resource_id] = event

            try:
                remaining = timeout_seconds - elapsed
                await asyncio.wait_for(event.wait(), timeout=min(remaining, 1.0))
            except asyncio.TimeoutError:
                pass

    async def release(self, resource_id: str, owner_id: str) -> bool:
        """Release a permit."""
        async with self._lock:
            if resource_id not in self._permits:
                return False

            if owner_id not in self._permits[resource_id]:
                return False

            self._permits[resource_id].discard(owner_id)

            # Notify waiters
            if resource_id in self._events:
                self._events[resource_id].set()
                del self._events[resource_id]

            return True

    async def get_available_permits(self, resource_id: str) -> int:
        """Get number of available permits."""
        async with self._lock:
            current = len(self._permits.get(resource_id, set()))
            return max(0, self._max_permits - current)

    @asynccontextmanager
    async def permit(
        self, resource_id: str, owner_id: str, timeout_seconds: float = 30.0
    ) -> AsyncIterator[None]:
        """Context manager for permit acquisition."""
        acquired = await self.acquire(resource_id, owner_id, timeout_seconds)
        if not acquired:
            raise TimeoutError(f"Failed to acquire semaphore permit for {resource_id}")

        try:
            yield
        finally:
            await self.release(resource_id, owner_id)


# ============================================================================
# Factory Functions
# ============================================================================


def create_lock_manager(
    enable_deadlock_detection: bool = True,
) -> LockManager:
    """Create a configured lock manager."""
    return LockManager(
        lock_impl=InMemoryLock(),
        enable_deadlock_detection=enable_deadlock_detection,
    )


def create_reentrant_lock() -> ReentrantLock:
    """Create a reentrant lock."""
    return ReentrantLock()


def create_read_write_lock() -> ReadWriteLock:
    """Create a read-write lock."""
    return ReadWriteLock()


def create_semaphore(max_permits: int = 1) -> DistributedSemaphore:
    """Create a distributed semaphore."""
    return DistributedSemaphore(max_permits=max_permits)
