"""Dead Letter Queue Module.

Provides failed message handling infrastructure:
- Dead letter queue management
- Retry policies with backoff
- Message inspection and replay
- Alerting and monitoring
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


class FailureReason(Enum):
    """Reasons for message failure."""
    MAX_RETRIES_EXCEEDED = "max_retries_exceeded"
    PROCESSING_ERROR = "processing_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT = "timeout"
    POISON_MESSAGE = "poison_message"  # Message that always fails
    MANUAL_REJECTION = "manual_rejection"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


class DLQAction(Enum):
    """Actions that can be taken on DLQ messages."""
    RETRY = "retry"  # Retry processing
    DISCARD = "discard"  # Permanently discard
    ARCHIVE = "archive"  # Move to archive
    REDIRECT = "redirect"  # Redirect to another queue


@dataclass
class FailedMessage:
    """A failed message in the DLQ."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_id: str = ""
    queue_name: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    failure_reason: FailureReason = FailureReason.UNKNOWN
    error_message: str = ""
    error_type: str = ""
    stack_trace: Optional[str] = None

    attempt_count: int = 0
    max_attempts: int = 3
    first_failure: datetime = field(default_factory=datetime.utcnow)
    last_failure: datetime = field(default_factory=datetime.utcnow)
    retry_count: int = 0
    next_retry: Optional[datetime] = None

    source_system: str = ""
    correlation_id: Optional[str] = None
    partition_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "original_id": self.original_id,
            "queue_name": self.queue_name,
            "payload": self.payload,
            "headers": self.headers,
            "metadata": self.metadata,
            "failure_reason": self.failure_reason.value,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "stack_trace": self.stack_trace,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "first_failure": self.first_failure.isoformat(),
            "last_failure": self.last_failure.isoformat(),
            "retry_count": self.retry_count,
            "next_retry": self.next_retry.isoformat() if self.next_retry else None,
            "source_system": self.source_system,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailedMessage":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            original_id=data.get("original_id", ""),
            queue_name=data.get("queue_name", ""),
            payload=data.get("payload", {}),
            headers=data.get("headers", {}),
            metadata=data.get("metadata", {}),
            failure_reason=FailureReason(data.get("failure_reason", "unknown")),
            error_message=data.get("error_message", ""),
            error_type=data.get("error_type", ""),
            stack_trace=data.get("stack_trace"),
            attempt_count=data.get("attempt_count", 0),
            max_attempts=data.get("max_attempts", 3),
            first_failure=datetime.fromisoformat(data["first_failure"]) if data.get("first_failure") else datetime.utcnow(),
            last_failure=datetime.fromisoformat(data["last_failure"]) if data.get("last_failure") else datetime.utcnow(),
            retry_count=data.get("retry_count", 0),
            next_retry=datetime.fromisoformat(data["next_retry"]) if data.get("next_retry") else None,
            source_system=data.get("source_system", ""),
            correlation_id=data.get("correlation_id"),
        )

    @property
    def is_poison(self) -> bool:
        """Check if message is a poison message (fails repeatedly)."""
        return self.retry_count >= 3 and self.attempt_count >= self.max_attempts

    @property
    def age_seconds(self) -> float:
        """Get message age in seconds."""
        return (datetime.utcnow() - self.first_failure).total_seconds()


@dataclass
class RetryPolicy:
    """Policy for retrying failed messages."""

    max_retries: int = 3
    initial_delay: float = 60.0  # seconds
    max_delay: float = 3600.0  # 1 hour
    multiplier: float = 2.0
    jitter: float = 0.1  # Add random jitter

    def calculate_delay(self, retry_count: int) -> float:
        """Calculate delay for given retry count."""
        import random

        delay = self.initial_delay * (self.multiplier ** retry_count)
        delay = min(delay, self.max_delay)

        # Add jitter
        jitter_amount = delay * self.jitter
        delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0, delay)

    def should_retry(self, retry_count: int) -> bool:
        """Check if message should be retried."""
        return retry_count < self.max_retries


class DLQStore(ABC):
    """Abstract DLQ storage."""

    @abstractmethod
    async def add(self, message: FailedMessage) -> bool:
        """Add message to DLQ."""
        pass

    @abstractmethod
    async def get(self, message_id: str) -> Optional[FailedMessage]:
        """Get message by ID."""
        pass

    @abstractmethod
    async def update(self, message: FailedMessage) -> bool:
        """Update message."""
        pass

    @abstractmethod
    async def remove(self, message_id: str) -> bool:
        """Remove message from DLQ."""
        pass

    @abstractmethod
    async def list_messages(
        self,
        queue_name: Optional[str] = None,
        failure_reason: Optional[FailureReason] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[FailedMessage]:
        """List messages with filters."""
        pass

    @abstractmethod
    async def count(
        self,
        queue_name: Optional[str] = None,
        failure_reason: Optional[FailureReason] = None,
    ) -> int:
        """Count messages."""
        pass

    @abstractmethod
    async def get_ready_for_retry(self, limit: int = 100) -> List[FailedMessage]:
        """Get messages ready for retry."""
        pass


class InMemoryDLQStore(DLQStore):
    """In-memory DLQ storage."""

    def __init__(self, max_entries: int = 10000):
        self._messages: Dict[str, FailedMessage] = {}
        self._max_entries = max_entries
        self._lock = asyncio.Lock()

    async def add(self, message: FailedMessage) -> bool:
        async with self._lock:
            if len(self._messages) >= self._max_entries:
                # Remove oldest
                oldest = min(self._messages.values(), key=lambda m: m.first_failure)
                del self._messages[oldest.id]

            self._messages[message.id] = message
            return True

    async def get(self, message_id: str) -> Optional[FailedMessage]:
        return self._messages.get(message_id)

    async def update(self, message: FailedMessage) -> bool:
        async with self._lock:
            if message.id not in self._messages:
                return False
            self._messages[message.id] = message
            return True

    async def remove(self, message_id: str) -> bool:
        async with self._lock:
            if message_id in self._messages:
                del self._messages[message_id]
                return True
            return False

    async def list_messages(
        self,
        queue_name: Optional[str] = None,
        failure_reason: Optional[FailureReason] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[FailedMessage]:
        messages = list(self._messages.values())

        if queue_name:
            messages = [m for m in messages if m.queue_name == queue_name]

        if failure_reason:
            messages = [m for m in messages if m.failure_reason == failure_reason]

        # Sort by last failure descending
        messages.sort(key=lambda m: m.last_failure, reverse=True)
        return messages[offset:offset + limit]

    async def count(
        self,
        queue_name: Optional[str] = None,
        failure_reason: Optional[FailureReason] = None,
    ) -> int:
        messages = list(self._messages.values())

        if queue_name:
            messages = [m for m in messages if m.queue_name == queue_name]

        if failure_reason:
            messages = [m for m in messages if m.failure_reason == failure_reason]

        return len(messages)

    async def get_ready_for_retry(self, limit: int = 100) -> List[FailedMessage]:
        now = datetime.utcnow()
        ready = [
            m for m in self._messages.values()
            if m.next_retry and m.next_retry <= now
        ]
        ready.sort(key=lambda m: m.next_retry or m.last_failure)
        return ready[:limit]


class AlertHandler(ABC):
    """Abstract alert handler."""

    @abstractmethod
    async def send_alert(
        self,
        message: FailedMessage,
        alert_type: str,
        details: Dict[str, Any],
    ) -> bool:
        """Send alert for failed message."""
        pass


class LoggingAlertHandler(AlertHandler):
    """Alert handler that logs alerts."""

    async def send_alert(
        self,
        message: FailedMessage,
        alert_type: str,
        details: Dict[str, Any],
    ) -> bool:
        logger.warning(
            f"DLQ Alert [{alert_type}]: Message {message.id} - "
            f"{message.error_message} (queue: {message.queue_name})"
        )
        return True


class CallbackAlertHandler(AlertHandler):
    """Alert handler that calls a callback function."""

    def __init__(self, callback: Callable[[FailedMessage, str, Dict[str, Any]], Any]):
        self._callback = callback

    async def send_alert(
        self,
        message: FailedMessage,
        alert_type: str,
        details: Dict[str, Any],
    ) -> bool:
        try:
            result = self._callback(message, alert_type, details)
            if asyncio.iscoroutine(result):
                await result
            return True
        except Exception as e:
            logger.error(f"Alert callback failed: {e}")
            return False


@dataclass
class DLQStats:
    """DLQ statistics."""

    total_messages: int = 0
    by_queue: Dict[str, int] = field(default_factory=dict)
    by_reason: Dict[str, int] = field(default_factory=dict)
    oldest_message_age: Optional[float] = None
    retry_pending: int = 0


class DeadLetterQueue:
    """Dead Letter Queue manager."""

    def __init__(
        self,
        name: str = "default",
        store: Optional[DLQStore] = None,
        retry_policy: Optional[RetryPolicy] = None,
        alert_handler: Optional[AlertHandler] = None,
        alert_threshold: int = 10,  # Alert when queue reaches this size
    ):
        self._name = name
        self._store = store or InMemoryDLQStore()
        self._retry_policy = retry_policy or RetryPolicy()
        self._alert_handler = alert_handler or LoggingAlertHandler()
        self._alert_threshold = alert_threshold
        self._message_processor: Optional[Callable[[FailedMessage], Any]] = None

    @property
    def name(self) -> str:
        return self._name

    def set_message_processor(
        self,
        processor: Callable[[FailedMessage], Any],
    ) -> None:
        """Set the processor for retrying messages."""
        self._message_processor = processor

    async def add_failed_message(
        self,
        payload: Dict[str, Any],
        error: Exception,
        queue_name: str = "",
        original_id: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        attempt_count: int = 1,
    ) -> FailedMessage:
        """Add a failed message to the DLQ."""
        import traceback

        failure_reason = self._determine_failure_reason(error, attempt_count)

        message = FailedMessage(
            original_id=original_id or str(uuid.uuid4()),
            queue_name=queue_name or self._name,
            payload=payload,
            headers=headers or {},
            metadata=metadata or {},
            failure_reason=failure_reason,
            error_message=str(error),
            error_type=type(error).__name__,
            stack_trace=traceback.format_exc(),
            attempt_count=attempt_count,
        )

        # Calculate next retry if policy allows
        if self._retry_policy.should_retry(message.retry_count):
            delay = self._retry_policy.calculate_delay(message.retry_count)
            message.next_retry = datetime.utcnow() + timedelta(seconds=delay)

        await self._store.add(message)
        logger.info(f"Added message to DLQ: {message.id} (reason: {failure_reason.value})")

        # Check alert threshold
        count = await self._store.count(queue_name=queue_name)
        if count >= self._alert_threshold:
            await self._alert_handler.send_alert(
                message,
                "threshold_exceeded",
                {"queue_size": count, "threshold": self._alert_threshold},
            )

        return message

    def _determine_failure_reason(
        self,
        error: Exception,
        attempt_count: int,
    ) -> FailureReason:
        """Determine failure reason from error."""
        error_type = type(error).__name__.lower()

        if "timeout" in error_type or "timeout" in str(error).lower():
            return FailureReason.TIMEOUT

        if "validation" in error_type or "validation" in str(error).lower():
            return FailureReason.VALIDATION_ERROR

        if attempt_count >= 3:
            return FailureReason.MAX_RETRIES_EXCEEDED

        return FailureReason.PROCESSING_ERROR

    async def get_message(self, message_id: str) -> Optional[FailedMessage]:
        """Get message by ID."""
        return await self._store.get(message_id)

    async def list_messages(
        self,
        queue_name: Optional[str] = None,
        failure_reason: Optional[FailureReason] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[FailedMessage]:
        """List messages with filters."""
        return await self._store.list_messages(
            queue_name=queue_name,
            failure_reason=failure_reason,
            limit=limit,
            offset=offset,
        )

    async def retry_message(self, message_id: str) -> bool:
        """Retry a specific message."""
        message = await self._store.get(message_id)
        if not message:
            return False

        if not self._message_processor:
            logger.error("No message processor configured")
            return False

        try:
            result = self._message_processor(message)
            if asyncio.iscoroutine(result):
                await result

            # Remove from DLQ on success
            await self._store.remove(message_id)
            logger.info(f"Successfully retried message: {message_id}")
            return True

        except Exception as e:
            # Update retry count
            message.retry_count += 1
            message.last_failure = datetime.utcnow()
            message.error_message = str(e)

            if self._retry_policy.should_retry(message.retry_count):
                delay = self._retry_policy.calculate_delay(message.retry_count)
                message.next_retry = datetime.utcnow() + timedelta(seconds=delay)
            else:
                message.next_retry = None
                message.failure_reason = FailureReason.MAX_RETRIES_EXCEEDED

            await self._store.update(message)
            logger.warning(f"Retry failed for message {message_id}: {e}")
            return False

    async def discard_message(self, message_id: str) -> bool:
        """Permanently discard a message."""
        result = await self._store.remove(message_id)
        if result:
            logger.info(f"Discarded message: {message_id}")
        return result

    async def bulk_retry(self, message_ids: List[str]) -> Dict[str, bool]:
        """Retry multiple messages."""
        results = {}
        for msg_id in message_ids:
            results[msg_id] = await self.retry_message(msg_id)
        return results

    async def bulk_discard(self, message_ids: List[str]) -> int:
        """Discard multiple messages."""
        count = 0
        for msg_id in message_ids:
            if await self.discard_message(msg_id):
                count += 1
        return count

    async def get_stats(self) -> DLQStats:
        """Get DLQ statistics."""
        messages = await self._store.list_messages(limit=10000)

        by_queue: Dict[str, int] = {}
        by_reason: Dict[str, int] = {}
        oldest_age: Optional[float] = None
        retry_pending = 0

        now = datetime.utcnow()

        for msg in messages:
            by_queue[msg.queue_name] = by_queue.get(msg.queue_name, 0) + 1
            by_reason[msg.failure_reason.value] = by_reason.get(msg.failure_reason.value, 0) + 1

            age = msg.age_seconds
            if oldest_age is None or age > oldest_age:
                oldest_age = age

            if msg.next_retry and msg.next_retry <= now:
                retry_pending += 1

        return DLQStats(
            total_messages=len(messages),
            by_queue=by_queue,
            by_reason=by_reason,
            oldest_message_age=oldest_age,
            retry_pending=retry_pending,
        )

    async def process_retries(self, batch_size: int = 100) -> int:
        """Process messages ready for retry."""
        messages = await self._store.get_ready_for_retry(limit=batch_size)
        success_count = 0

        for message in messages:
            if await self.retry_message(message.id):
                success_count += 1

        return success_count


class DLQRetryWorker:
    """Background worker for processing DLQ retries."""

    def __init__(
        self,
        dlq: DeadLetterQueue,
        interval: float = 60.0,
        batch_size: int = 100,
    ):
        self._dlq = dlq
        self._interval = interval
        self._batch_size = batch_size
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the worker."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info(f"DLQ retry worker started for {self._dlq.name}")

    async def stop(self) -> None:
        """Stop the worker."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"DLQ retry worker stopped for {self._dlq.name}")

    async def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                count = await self._dlq.process_retries(self._batch_size)
                if count > 0:
                    logger.info(f"Processed {count} DLQ retries")
            except Exception as e:
                logger.error(f"DLQ retry worker error: {e}")

            await asyncio.sleep(self._interval)


# Global DLQ registry
_dlq_registry: Dict[str, DeadLetterQueue] = {}


def get_dlq(name: str = "default") -> DeadLetterQueue:
    """Get or create a DLQ by name."""
    if name not in _dlq_registry:
        _dlq_registry[name] = DeadLetterQueue(name=name)
    return _dlq_registry[name]


def register_dlq(dlq: DeadLetterQueue) -> None:
    """Register a DLQ."""
    _dlq_registry[dlq.name] = dlq


__all__ = [
    "FailureReason",
    "DLQAction",
    "FailedMessage",
    "RetryPolicy",
    "DLQStore",
    "InMemoryDLQStore",
    "AlertHandler",
    "LoggingAlertHandler",
    "CallbackAlertHandler",
    "DLQStats",
    "DeadLetterQueue",
    "DLQRetryWorker",
    "get_dlq",
    "register_dlq",
]
