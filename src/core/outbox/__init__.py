"""Outbox Pattern Module.

Provides transactional outbox infrastructure:
- Reliable event publishing
- At-least-once delivery guarantee
- Message ordering
- Retry with backoff
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class OutboxStatus(Enum):
    """Outbox message status."""
    PENDING = "pending"
    PROCESSING = "processing"
    PUBLISHED = "published"
    FAILED = "failed"


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class OutboxMessage:
    """A message in the outbox."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    aggregate_id: str = ""
    aggregate_type: str = ""
    event_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    status: OutboxStatus = OutboxStatus.PENDING
    priority: MessagePriority = MessagePriority.NORMAL
    attempts: int = 0
    max_attempts: int = 5
    last_attempt: Optional[datetime] = None
    next_retry: Optional[datetime] = None
    error: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = None
    partition_key: Optional[str] = None
    ordering_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "event_type": self.event_type,
            "payload": self.payload,
            "metadata": self.metadata,
            "status": self.status.value,
            "priority": self.priority.value,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "last_attempt": self.last_attempt.isoformat() if self.last_attempt else None,
            "next_retry": self.next_retry.isoformat() if self.next_retry else None,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "partition_key": self.partition_key,
            "ordering_key": self.ordering_key,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutboxMessage":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            aggregate_id=data.get("aggregate_id", ""),
            aggregate_type=data.get("aggregate_type", ""),
            event_type=data.get("event_type", ""),
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
            status=OutboxStatus(data.get("status", "pending")),
            priority=MessagePriority(data.get("priority", 1)),
            attempts=data.get("attempts", 0),
            max_attempts=data.get("max_attempts", 5),
            last_attempt=datetime.fromisoformat(data["last_attempt"]) if data.get("last_attempt") else None,
            next_retry=datetime.fromisoformat(data["next_retry"]) if data.get("next_retry") else None,
            error=data.get("error"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            published_at=datetime.fromisoformat(data["published_at"]) if data.get("published_at") else None,
            partition_key=data.get("partition_key"),
            ordering_key=data.get("ordering_key"),
        )

    @property
    def is_retriable(self) -> bool:
        """Check if message can be retried."""
        return self.attempts < self.max_attempts

    @property
    def is_ready_for_retry(self) -> bool:
        """Check if message is ready for retry."""
        if self.next_retry is None:
            return True
        return datetime.utcnow() >= self.next_retry


class OutboxStore(ABC):
    """Abstract outbox storage."""

    @abstractmethod
    async def add(self, message: OutboxMessage) -> bool:
        """Add message to outbox."""
        pass

    @abstractmethod
    async def get_pending(
        self,
        limit: int = 100,
        partition_key: Optional[str] = None,
    ) -> List[OutboxMessage]:
        """Get pending messages ready for publishing."""
        pass

    @abstractmethod
    async def mark_processing(self, message_id: str) -> bool:
        """Mark message as being processed."""
        pass

    @abstractmethod
    async def mark_published(self, message_id: str) -> bool:
        """Mark message as published."""
        pass

    @abstractmethod
    async def mark_failed(
        self,
        message_id: str,
        error: str,
        next_retry: Optional[datetime] = None,
    ) -> bool:
        """Mark message as failed."""
        pass

    @abstractmethod
    async def get_by_id(self, message_id: str) -> Optional[OutboxMessage]:
        """Get message by ID."""
        pass

    @abstractmethod
    async def delete(self, message_id: str) -> bool:
        """Delete message."""
        pass

    @abstractmethod
    async def cleanup_published(self, older_than: datetime) -> int:
        """Clean up old published messages."""
        pass


class InMemoryOutboxStore(OutboxStore):
    """In-memory outbox storage."""

    def __init__(self, max_entries: int = 10000):
        self._messages: Dict[str, OutboxMessage] = {}
        self._max_entries = max_entries
        self._lock = asyncio.Lock()

    async def add(self, message: OutboxMessage) -> bool:
        async with self._lock:
            if len(self._messages) >= self._max_entries:
                await self._cleanup_oldest()
            self._messages[message.id] = message
            return True

    async def get_pending(
        self,
        limit: int = 100,
        partition_key: Optional[str] = None,
    ) -> List[OutboxMessage]:
        async with self._lock:
            now = datetime.utcnow()
            pending = [
                m for m in self._messages.values()
                if m.status == OutboxStatus.PENDING
                and m.is_ready_for_retry
                and (partition_key is None or m.partition_key == partition_key)
            ]

            # Sort by priority (desc) and created_at (asc)
            pending.sort(key=lambda m: (-m.priority.value, m.created_at))
            return pending[:limit]

    async def mark_processing(self, message_id: str) -> bool:
        async with self._lock:
            if message_id not in self._messages:
                return False
            message = self._messages[message_id]
            message.status = OutboxStatus.PROCESSING
            message.attempts += 1
            message.last_attempt = datetime.utcnow()
            return True

    async def mark_published(self, message_id: str) -> bool:
        async with self._lock:
            if message_id not in self._messages:
                return False
            message = self._messages[message_id]
            message.status = OutboxStatus.PUBLISHED
            message.published_at = datetime.utcnow()
            return True

    async def mark_failed(
        self,
        message_id: str,
        error: str,
        next_retry: Optional[datetime] = None,
    ) -> bool:
        async with self._lock:
            if message_id not in self._messages:
                return False
            message = self._messages[message_id]
            message.error = error

            if message.is_retriable:
                message.status = OutboxStatus.PENDING
                message.next_retry = next_retry or datetime.utcnow() + timedelta(
                    seconds=min(300, 2 ** message.attempts * 10)  # Exponential backoff, max 5 min
                )
            else:
                message.status = OutboxStatus.FAILED

            return True

    async def get_by_id(self, message_id: str) -> Optional[OutboxMessage]:
        return self._messages.get(message_id)

    async def delete(self, message_id: str) -> bool:
        async with self._lock:
            if message_id in self._messages:
                del self._messages[message_id]
                return True
            return False

    async def cleanup_published(self, older_than: datetime) -> int:
        async with self._lock:
            to_delete = [
                m.id for m in self._messages.values()
                if m.status == OutboxStatus.PUBLISHED
                and m.published_at
                and m.published_at < older_than
            ]
            for msg_id in to_delete:
                del self._messages[msg_id]
            return len(to_delete)

    async def _cleanup_oldest(self) -> None:
        """Remove oldest published messages."""
        published = [
            m for m in self._messages.values()
            if m.status == OutboxStatus.PUBLISHED
        ]
        published.sort(key=lambda m: m.published_at or m.created_at)

        to_remove = max(1, len(published) // 2)
        for msg in published[:to_remove]:
            del self._messages[msg.id]


class MessagePublisher(ABC):
    """Abstract message publisher."""

    @abstractmethod
    async def publish(self, message: OutboxMessage) -> bool:
        """Publish message to external system."""
        pass


class LoggingPublisher(MessagePublisher):
    """Publisher that logs messages (for testing)."""

    def __init__(self):
        self.published: List[OutboxMessage] = []

    async def publish(self, message: OutboxMessage) -> bool:
        logger.info(f"Publishing message: {message.id} - {message.event_type}")
        self.published.append(message)
        return True


class CompositePublisher(MessagePublisher):
    """Publisher that sends to multiple destinations."""

    def __init__(self, publishers: List[MessagePublisher]):
        self._publishers = publishers

    async def publish(self, message: OutboxMessage) -> bool:
        results = await asyncio.gather(*[
            p.publish(message) for p in self._publishers
        ], return_exceptions=True)

        # All must succeed
        for result in results:
            if isinstance(result, Exception):
                raise result
            if not result:
                return False
        return True


class OutboxProcessor:
    """Processes outbox messages."""

    def __init__(
        self,
        store: OutboxStore,
        publisher: MessagePublisher,
        batch_size: int = 100,
        poll_interval: float = 1.0,
        max_concurrency: int = 10,
    ):
        self._store = store
        self._publisher = publisher
        self._batch_size = batch_size
        self._poll_interval = poll_interval
        self._max_concurrency = max_concurrency
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def start(self) -> None:
        """Start the processor."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info("Outbox processor started")

    async def stop(self) -> None:
        """Stop the processor."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Outbox processor stopped")

    async def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                await self._process_batch()
            except Exception as e:
                logger.error(f"Outbox processing error: {e}")

            await asyncio.sleep(self._poll_interval)

    async def _process_batch(self) -> int:
        """Process a batch of messages."""
        messages = await self._store.get_pending(limit=self._batch_size)
        if not messages:
            return 0

        # Process concurrently with semaphore
        tasks = [self._process_message(m) for m in messages]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if r is True)
        logger.debug(f"Processed {success_count}/{len(messages)} messages")
        return success_count

    async def _process_message(self, message: OutboxMessage) -> bool:
        """Process a single message."""
        async with self._semaphore:
            # Mark as processing
            if not await self._store.mark_processing(message.id):
                return False

            try:
                # Publish
                success = await self._publisher.publish(message)

                if success:
                    await self._store.mark_published(message.id)
                    return True
                else:
                    await self._store.mark_failed(message.id, "Publisher returned False")
                    return False

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to publish message {message.id}: {error_msg}")
                await self._store.mark_failed(message.id, error_msg)
                return False

    async def process_single(self, message_id: str) -> bool:
        """Process a single message by ID."""
        message = await self._store.get_by_id(message_id)
        if not message:
            return False
        return await self._process_message(message)


class OutboxService:
    """High-level outbox service."""

    def __init__(
        self,
        store: Optional[OutboxStore] = None,
        publisher: Optional[MessagePublisher] = None,
        auto_start: bool = False,
    ):
        self._store = store or InMemoryOutboxStore()
        self._publisher = publisher or LoggingPublisher()
        self._processor = OutboxProcessor(self._store, self._publisher)

        if auto_start:
            asyncio.create_task(self._processor.start())

    @property
    def store(self) -> OutboxStore:
        return self._store

    @property
    def processor(self) -> OutboxProcessor:
        return self._processor

    async def add_message(
        self,
        event_type: str,
        payload: Dict[str, Any],
        aggregate_id: Optional[str] = None,
        aggregate_type: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        partition_key: Optional[str] = None,
        ordering_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OutboxMessage:
        """Add a message to the outbox."""
        message = OutboxMessage(
            aggregate_id=aggregate_id or "",
            aggregate_type=aggregate_type or "",
            event_type=event_type,
            payload=payload,
            priority=priority,
            partition_key=partition_key,
            ordering_key=ordering_key,
            metadata=metadata or {},
        )

        await self._store.add(message)
        logger.debug(f"Added message to outbox: {message.id}")
        return message

    async def get_message(self, message_id: str) -> Optional[OutboxMessage]:
        """Get message by ID."""
        return await self._store.get_by_id(message_id)

    async def start_processor(self) -> None:
        """Start the outbox processor."""
        await self._processor.start()

    async def stop_processor(self) -> None:
        """Stop the outbox processor."""
        await self._processor.stop()

    async def cleanup(self, retention_hours: int = 24) -> int:
        """Clean up old published messages."""
        older_than = datetime.utcnow() - timedelta(hours=retention_hours)
        return await self._store.cleanup_published(older_than)


def transactional_outbox(
    event_type: str,
    service: Optional[OutboxService] = None,
    aggregate_id_param: Optional[str] = None,
):
    """Decorator to automatically add outbox message after function execution."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        async def async_wrapper(*args, **kwargs) -> T:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result

            _service = service or _default_service
            if _service:
                aggregate_id = None
                if aggregate_id_param and aggregate_id_param in kwargs:
                    aggregate_id = kwargs[aggregate_id_param]

                await _service.add_message(
                    event_type=event_type,
                    payload={"result": result} if result else {},
                    aggregate_id=aggregate_id,
                )

            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return lambda *a, **kw: asyncio.run(async_wrapper(*a, **kw))

    return decorator


# Default service
_default_service: Optional[OutboxService] = None


def get_default_service() -> Optional[OutboxService]:
    """Get default outbox service."""
    return _default_service


def set_default_service(service: OutboxService) -> None:
    """Set default outbox service."""
    global _default_service
    _default_service = service


__all__ = [
    "OutboxStatus",
    "MessagePriority",
    "OutboxMessage",
    "OutboxStore",
    "InMemoryOutboxStore",
    "MessagePublisher",
    "LoggingPublisher",
    "CompositePublisher",
    "OutboxProcessor",
    "OutboxService",
    "transactional_outbox",
    "get_default_service",
    "set_default_service",
]
