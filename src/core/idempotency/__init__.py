"""Idempotency Module.

Provides idempotency infrastructure:
- Request deduplication
- Idempotency keys management
- Replay protection
- Response caching
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class IdempotencyStatus(Enum):
    """Status of idempotent request."""
    PENDING = "pending"  # Request in progress
    COMPLETED = "completed"  # Request completed successfully
    FAILED = "failed"  # Request failed
    EXPIRED = "expired"  # Request record expired


@dataclass
class IdempotencyRecord:
    """Record of an idempotent request."""

    key: str
    status: IdempotencyStatus = IdempotencyStatus.PENDING
    request_hash: Optional[str] = None
    response: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_completed(self) -> bool:
        return self.status == IdempotencyStatus.COMPLETED

    @property
    def is_pending(self) -> bool:
        return self.status == IdempotencyStatus.PENDING

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "status": self.status.value,
            "request_hash": self.request_hash,
            "response": self.response,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdempotencyRecord":
        return cls(
            key=data["key"],
            status=IdempotencyStatus(data["status"]),
            request_hash=data.get("request_hash"),
            response=data.get("response"),
            error=data.get("error"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            metadata=data.get("metadata", {}),
        )


class IdempotencyStore(ABC):
    """Abstract idempotency store."""

    @abstractmethod
    async def get(self, key: str) -> Optional[IdempotencyRecord]:
        """Get record by key."""
        pass

    @abstractmethod
    async def create(self, record: IdempotencyRecord) -> bool:
        """Create record (returns False if already exists)."""
        pass

    @abstractmethod
    async def update(self, record: IdempotencyRecord) -> bool:
        """Update existing record."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete record."""
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Clean up expired records."""
        pass


class InMemoryIdempotencyStore(IdempotencyStore):
    """In-memory idempotency store."""

    def __init__(self, max_entries: int = 10000):
        self._records: Dict[str, IdempotencyRecord] = {}
        self._max_entries = max_entries
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[IdempotencyRecord]:
        async with self._lock:
            record = self._records.get(key)
            if record and record.is_expired:
                del self._records[key]
                return None
            return record

    async def create(self, record: IdempotencyRecord) -> bool:
        async with self._lock:
            if record.key in self._records:
                existing = self._records[record.key]
                if not existing.is_expired:
                    return False
                # Replace expired record

            # Enforce max entries
            if len(self._records) >= self._max_entries:
                await self._evict_oldest()

            self._records[record.key] = record
            return True

    async def update(self, record: IdempotencyRecord) -> bool:
        async with self._lock:
            if record.key not in self._records:
                return False
            record.updated_at = datetime.utcnow()
            self._records[record.key] = record
            return True

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._records:
                del self._records[key]
                return True
            return False

    async def cleanup_expired(self) -> int:
        async with self._lock:
            now = datetime.utcnow()
            expired_keys = [
                k for k, v in self._records.items()
                if v.expires_at and v.expires_at < now
            ]
            for key in expired_keys:
                del self._records[key]
            return len(expired_keys)

    async def _evict_oldest(self) -> None:
        """Evict oldest records."""
        if not self._records:
            return

        # Sort by created_at and remove oldest 10%
        sorted_keys = sorted(
            self._records.keys(),
            key=lambda k: self._records[k].created_at
        )
        to_remove = max(1, len(sorted_keys) // 10)
        for key in sorted_keys[:to_remove]:
            del self._records[key]


class IdempotencyKeyGenerator:
    """Generates idempotency keys."""

    @staticmethod
    def from_request(
        method: str,
        path: str,
        body: Optional[Any] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Generate key from HTTP request."""
        parts = [method.upper(), path]
        if user_id:
            parts.append(user_id)
        if body:
            body_str = json.dumps(body, sort_keys=True, default=str)
            parts.append(hashlib.md5(body_str.encode()).hexdigest())  # nosec B324 - idempotency fingerprint
        return ":".join(parts)

    @staticmethod
    def from_args(*args, **kwargs) -> str:
        """Generate key from function arguments."""
        parts = [str(a) for a in args]
        parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        combined = ":".join(parts)
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

    @staticmethod
    def generate_uuid() -> str:
        """Generate UUID-based key."""
        return str(uuid.uuid4())


class IdempotencyError(Exception):
    """Base idempotency error."""
    pass


class DuplicateRequestError(IdempotencyError):
    """Raised when duplicate request detected."""

    def __init__(self, key: str, record: IdempotencyRecord):
        self.key = key
        self.record = record
        super().__init__(f"Duplicate request with key: {key}")


class RequestInProgressError(IdempotencyError):
    """Raised when request is already in progress."""

    def __init__(self, key: str):
        self.key = key
        super().__init__(f"Request already in progress: {key}")


class RequestHashMismatchError(IdempotencyError):
    """Raised when request body doesn't match original."""

    def __init__(self, key: str):
        self.key = key
        super().__init__(f"Request body mismatch for key: {key}")


class IdempotencyService:
    """Service for managing idempotent requests."""

    def __init__(
        self,
        store: Optional[IdempotencyStore] = None,
        default_ttl: timedelta = timedelta(hours=24),
        pending_timeout: timedelta = timedelta(minutes=5),
    ):
        self._store = store or InMemoryIdempotencyStore()
        self._default_ttl = default_ttl
        self._pending_timeout = pending_timeout
        self._key_generator = IdempotencyKeyGenerator()

    async def execute(
        self,
        key: str,
        func: Callable[..., T],
        *args,
        request_hash: Optional[str] = None,
        ttl: Optional[timedelta] = None,
        **kwargs,
    ) -> T:
        """Execute function with idempotency guarantee."""
        effective_ttl = ttl or self._default_ttl

        # Check for existing record
        record = await self._store.get(key)

        if record:
            # Check if request is completed
            if record.is_completed:
                # Verify request hash if provided
                if request_hash and record.request_hash and request_hash != record.request_hash:
                    raise RequestHashMismatchError(key)
                logger.debug(f"Returning cached response for key: {key}")
                return record.response

            # Check if request is pending but timed out
            if record.is_pending:
                elapsed = datetime.utcnow() - record.created_at
                if elapsed < self._pending_timeout:
                    raise RequestInProgressError(key)
                # Pending request timed out, allow retry
                logger.warning(f"Pending request timed out, retrying: {key}")

            # Check if request failed
            if record.status == IdempotencyStatus.FAILED:
                # Allow retry of failed requests
                logger.debug(f"Retrying failed request: {key}")

        # Create new record
        new_record = IdempotencyRecord(
            key=key,
            status=IdempotencyStatus.PENDING,
            request_hash=request_hash,
            expires_at=datetime.utcnow() + effective_ttl,
        )

        created = await self._store.create(new_record)
        if not created:
            # Race condition - another request got there first
            record = await self._store.get(key)
            if record and record.is_completed:
                return record.response
            raise RequestInProgressError(key)

        try:
            # Execute the function
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result

            # Update record with success
            new_record.status = IdempotencyStatus.COMPLETED
            new_record.response = result
            await self._store.update(new_record)

            return result

        except Exception as e:
            # Update record with failure
            new_record.status = IdempotencyStatus.FAILED
            new_record.error = str(e)
            await self._store.update(new_record)
            raise

    async def get_status(self, key: str) -> Optional[IdempotencyRecord]:
        """Get status of idempotent request."""
        return await self._store.get(key)

    async def invalidate(self, key: str) -> bool:
        """Invalidate idempotency record."""
        return await self._store.delete(key)

    async def cleanup(self) -> int:
        """Clean up expired records."""
        return await self._store.cleanup_expired()


def idempotent(
    key_func: Optional[Callable[..., str]] = None,
    service: Optional[IdempotencyService] = None,
    ttl: Optional[timedelta] = None,
):
    """Decorator for idempotent functions."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, idempotency_key: Optional[str] = None, **kwargs) -> T:
            _service = service or _default_service
            if not _service:
                # No service configured, just execute
                result = func(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    return await result
                return result

            # Generate key
            if idempotency_key:
                key = idempotency_key
            elif key_func:
                key = key_func(*args, **kwargs)
            else:
                key = IdempotencyKeyGenerator.from_args(*args, **kwargs)

            # Generate request hash
            request_hash = IdempotencyKeyGenerator.from_args(*args, **kwargs)

            return await _service.execute(
                key=key,
                func=func,
                *args,
                request_hash=request_hash,
                ttl=ttl,
                **kwargs,
            )

        @wraps(func)
        def sync_wrapper(*args, idempotency_key: Optional[str] = None, **kwargs) -> T:
            return asyncio.run(async_wrapper(*args, idempotency_key=idempotency_key, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class IdempotencyMiddleware:
    """Middleware for HTTP request idempotency."""

    def __init__(
        self,
        service: IdempotencyService,
        header_name: str = "Idempotency-Key",
        methods: Optional[List[str]] = None,
    ):
        self._service = service
        self._header_name = header_name
        self._methods = methods or ["POST", "PUT", "PATCH"]

    def get_key_from_headers(self, headers: Dict[str, str]) -> Optional[str]:
        """Extract idempotency key from headers."""
        return headers.get(self._header_name) or headers.get(self._header_name.lower())

    def should_apply(self, method: str) -> bool:
        """Check if idempotency should apply to method."""
        return method.upper() in self._methods

    async def process_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: Optional[Any],
        handler: Callable[[], T],
    ) -> tuple[T, bool]:
        """Process request with idempotency.

        Returns (response, was_cached).
        """
        if not self.should_apply(method):
            result = handler()
            if asyncio.iscoroutine(result):
                result = await result
            return result, False

        key = self.get_key_from_headers(headers)
        if not key:
            # No idempotency key, generate one
            key = IdempotencyKeyGenerator.from_request(method, path, body)

        request_hash = None
        if body:
            body_str = json.dumps(body, sort_keys=True, default=str)
            request_hash = hashlib.sha256(body_str.encode()).hexdigest()

        # Check for existing record
        record = await self._service.get_status(key)
        if record and record.is_completed:
            return record.response, True

        # Execute with idempotency
        result = await self._service.execute(
            key=key,
            func=handler,
            request_hash=request_hash,
        )

        return result, False


class ReplayProtection:
    """Protection against replay attacks."""

    def __init__(
        self,
        store: Optional[IdempotencyStore] = None,
        window: timedelta = timedelta(minutes=5),
        max_clock_drift: timedelta = timedelta(seconds=30),
    ):
        self._store = store or InMemoryIdempotencyStore()
        self._window = window
        self._max_clock_drift = max_clock_drift

    async def check_and_record(
        self,
        nonce: str,
        timestamp: datetime,
        additional_data: Optional[str] = None,
    ) -> bool:
        """Check if request is valid and record it.

        Returns True if valid (not a replay), False if replay detected.
        """
        now = datetime.utcnow()

        # Check timestamp within window
        time_diff = abs((now - timestamp).total_seconds())
        max_window = self._window.total_seconds() + self._max_clock_drift.total_seconds()

        if time_diff > max_window:
            logger.warning(f"Request timestamp outside window: {timestamp}")
            return False

        # Check for duplicate nonce
        key = f"replay:{nonce}"
        if additional_data:
            key = f"{key}:{hashlib.sha256(additional_data.encode()).hexdigest()[:16]}"

        record = IdempotencyRecord(
            key=key,
            status=IdempotencyStatus.COMPLETED,
            expires_at=now + self._window + self._max_clock_drift,
            metadata={"timestamp": timestamp.isoformat()},
        )

        created = await self._store.create(record)
        if not created:
            logger.warning(f"Replay attack detected: {nonce}")
            return False

        return True


# Default service
_default_service: Optional[IdempotencyService] = None


def get_default_service() -> Optional[IdempotencyService]:
    """Get default idempotency service."""
    return _default_service


def set_default_service(service: IdempotencyService) -> None:
    """Set default idempotency service."""
    global _default_service
    _default_service = service


__all__ = [
    "IdempotencyStatus",
    "IdempotencyRecord",
    "IdempotencyStore",
    "InMemoryIdempotencyStore",
    "IdempotencyKeyGenerator",
    "IdempotencyError",
    "DuplicateRequestError",
    "RequestInProgressError",
    "RequestHashMismatchError",
    "IdempotencyService",
    "idempotent",
    "IdempotencyMiddleware",
    "ReplayProtection",
    "get_default_service",
    "set_default_service",
]
