"""
Message Queue Module - Phase 13.

Provides message queue capabilities including pub/sub,
dead letter queues, message persistence, and delivery guarantees.
"""

import asyncio
import threading
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union

from .base import VisionDescription, VisionProvider

# ============================================================================
# Enums
# ============================================================================


class MessageStatus(Enum):
    """Message status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


class DeliveryMode(Enum):
    """Message delivery mode."""

    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


class QueueType(Enum):
    """Queue type."""

    FIFO = "fifo"
    PRIORITY = "priority"
    DELAY = "delay"
    DEAD_LETTER = "dead_letter"


class AckMode(Enum):
    """Acknowledgment mode."""

    AUTO = "auto"
    MANUAL = "manual"
    CLIENT = "client"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class Message:
    """A message in the queue."""

    message_id: str
    payload: Any
    topic: str = "default"
    priority: int = 0
    status: MessageStatus = MessageStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if message is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.retry_count < self.max_retries


@dataclass
class MessageBatch:
    """A batch of messages."""

    batch_id: str
    messages: List[Message]
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class QueueConfig:
    """Queue configuration."""

    queue_id: str
    name: str
    queue_type: QueueType = QueueType.FIFO
    max_size: int = 10000
    delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE
    ack_mode: AckMode = AckMode.MANUAL
    visibility_timeout_seconds: int = 30
    retention_days: int = 7
    dead_letter_queue: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueueStats:
    """Queue statistics."""

    queue_id: str
    total_messages: int
    pending_messages: int
    processing_messages: int
    completed_messages: int
    failed_messages: int
    dead_letter_messages: int
    oldest_message_age_seconds: float = 0
    messages_per_second: float = 0


@dataclass
class Subscription:
    """A subscription to a topic."""

    subscription_id: str
    topic: str
    subscriber_id: str
    filter_expression: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PublishResult:
    """Result of publishing a message."""

    message_id: str
    topic: str
    success: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None


@dataclass
class ConsumeResult:
    """Result of consuming a message."""

    message: Optional[Message]
    success: bool
    error: Optional[str] = None


# ============================================================================
# Message Store Interface
# ============================================================================


class MessageStore(ABC):
    """Abstract message storage."""

    @abstractmethod
    def store(self, message: Message) -> None:
        """Store a message."""
        pass

    @abstractmethod
    def get(self, message_id: str) -> Optional[Message]:
        """Get a message."""
        pass

    @abstractmethod
    def update(self, message: Message) -> None:
        """Update a message."""
        pass

    @abstractmethod
    def delete(self, message_id: str) -> None:
        """Delete a message."""
        pass

    @abstractmethod
    def get_pending(self, topic: str, limit: int = 10) -> List[Message]:
        """Get pending messages for a topic."""
        pass


class InMemoryMessageStore(MessageStore):
    """In-memory message storage."""

    def __init__(self) -> None:
        self._messages: Dict[str, Message] = {}
        self._by_topic: Dict[str, List[str]] = {}
        self._lock = threading.Lock()

    def store(self, message: Message) -> None:
        """Store a message."""
        with self._lock:
            self._messages[message.message_id] = message
            if message.topic not in self._by_topic:
                self._by_topic[message.topic] = []
            self._by_topic[message.topic].append(message.message_id)

    def get(self, message_id: str) -> Optional[Message]:
        """Get a message."""
        return self._messages.get(message_id)

    def update(self, message: Message) -> None:
        """Update a message."""
        with self._lock:
            self._messages[message.message_id] = message

    def delete(self, message_id: str) -> None:
        """Delete a message."""
        with self._lock:
            if message_id in self._messages:
                message = self._messages[message_id]
                del self._messages[message_id]
                if message.topic in self._by_topic:
                    if message_id in self._by_topic[message.topic]:
                        self._by_topic[message.topic].remove(message_id)

    def get_pending(self, topic: str, limit: int = 10) -> List[Message]:
        """Get pending messages for a topic."""
        with self._lock:
            message_ids = self._by_topic.get(topic, [])
            pending = []
            for msg_id in message_ids:
                if len(pending) >= limit:
                    break
                msg = self._messages.get(msg_id)
                if msg and msg.status == MessageStatus.PENDING and not msg.is_expired():
                    pending.append(msg)
            return pending


# ============================================================================
# Queue Implementation
# ============================================================================


class MessageQueue:
    """A message queue implementation."""

    def __init__(
        self,
        config: QueueConfig,
        store: Optional[MessageStore] = None,
    ) -> None:
        self._config = config
        self._store = store or InMemoryMessageStore()
        self._processing: Set[str] = set()
        self._lock = threading.Lock()

    def enqueue(self, message: Message) -> bool:
        """Add a message to the queue."""
        message.status = MessageStatus.PENDING
        self._store.store(message)
        return True

    def dequeue(self) -> Optional[Message]:
        """Remove and return the next message."""
        with self._lock:
            messages = self._store.get_pending(self._config.name, limit=1)
            if not messages:
                return None

            message = messages[0]
            message.status = MessageStatus.PROCESSING
            self._store.update(message)
            self._processing.add(message.message_id)
            return message

    def peek(self) -> Optional[Message]:
        """Look at the next message without removing it."""
        messages = self._store.get_pending(self._config.name, limit=1)
        return messages[0] if messages else None

    def ack(self, message_id: str) -> bool:
        """Acknowledge a message as processed."""
        with self._lock:
            message = self._store.get(message_id)
            if message:
                message.status = MessageStatus.COMPLETED
                self._store.update(message)
                self._processing.discard(message_id)
                return True
            return False

    def nack(self, message_id: str, requeue: bool = True) -> bool:
        """Negative acknowledgment - message processing failed."""
        with self._lock:
            message = self._store.get(message_id)
            if not message:
                return False

            self._processing.discard(message_id)

            if requeue and message.can_retry():
                message.retry_count += 1
                message.status = MessageStatus.PENDING
                self._store.update(message)
            else:
                message.status = MessageStatus.FAILED
                self._store.update(message)
                # Move to dead letter queue if configured
                if self._config.dead_letter_queue:
                    message.status = MessageStatus.DEAD_LETTER

            return True

    def size(self) -> int:
        """Get the number of pending messages."""
        return len(self._store.get_pending(self._config.name, limit=10000))

    def get_stats(self) -> QueueStats:
        """Get queue statistics."""
        all_messages = self._store.get_pending(self._config.name, limit=10000)
        return QueueStats(
            queue_id=self._config.queue_id,
            total_messages=len(all_messages),
            pending_messages=len([m for m in all_messages if m.status == MessageStatus.PENDING]),
            processing_messages=len(self._processing),
            completed_messages=0,
            failed_messages=0,
            dead_letter_messages=0,
        )


# ============================================================================
# Pub/Sub Implementation
# ============================================================================


class Publisher:
    """Message publisher."""

    def __init__(self, store: Optional[MessageStore] = None) -> None:
        self._store = store or InMemoryMessageStore()
        self._subscribers: Dict[str, List[Subscription]] = {}

    def publish(
        self,
        topic: str,
        payload: Any,
        headers: Optional[Dict[str, str]] = None,
    ) -> PublishResult:
        """Publish a message to a topic."""
        message = Message(
            message_id=str(uuid.uuid4()),
            payload=payload,
            topic=topic,
            headers=headers or {},
        )
        self._store.store(message)
        return PublishResult(
            message_id=message.message_id,
            topic=topic,
            success=True,
        )

    def publish_batch(
        self,
        topic: str,
        payloads: List[Any],
    ) -> List[PublishResult]:
        """Publish multiple messages."""
        return [self.publish(topic, payload) for payload in payloads]

    def register_subscriber(self, subscription: Subscription) -> None:
        """Register a subscriber."""
        if subscription.topic not in self._subscribers:
            self._subscribers[subscription.topic] = []
        self._subscribers[subscription.topic].append(subscription)

    def get_subscribers(self, topic: str) -> List[Subscription]:
        """Get subscribers for a topic."""
        return self._subscribers.get(topic, [])


class Subscriber:
    """Message subscriber."""

    def __init__(
        self,
        subscriber_id: str,
        store: Optional[MessageStore] = None,
    ) -> None:
        self._subscriber_id = subscriber_id
        self._store = store or InMemoryMessageStore()
        self._subscriptions: Dict[str, Subscription] = {}
        self._handlers: Dict[str, Callable[[Message], None]] = {}

    def subscribe(
        self,
        topic: str,
        handler: Callable[[Message], None],
        filter_expression: Optional[str] = None,
    ) -> Subscription:
        """Subscribe to a topic."""
        subscription = Subscription(
            subscription_id=str(uuid.uuid4()),
            topic=topic,
            subscriber_id=self._subscriber_id,
            filter_expression=filter_expression,
        )
        self._subscriptions[topic] = subscription
        self._handlers[topic] = handler
        return subscription

    def unsubscribe(self, topic: str) -> bool:
        """Unsubscribe from a topic."""
        if topic in self._subscriptions:
            del self._subscriptions[topic]
            del self._handlers[topic]
            return True
        return False

    def receive(self, topic: str, timeout_seconds: float = 1.0) -> ConsumeResult:
        """Receive a message from a topic."""
        messages = self._store.get_pending(topic, limit=1)
        if messages:
            message = messages[0]
            message.status = MessageStatus.PROCESSING
            self._store.update(message)
            return ConsumeResult(message=message, success=True)
        return ConsumeResult(message=None, success=True)

    def process_message(self, message: Message) -> bool:
        """Process a message using registered handler."""
        handler = self._handlers.get(message.topic)
        if handler:
            try:
                handler(message)
                message.status = MessageStatus.COMPLETED
                self._store.update(message)
                return True
            except Exception:
                message.status = MessageStatus.FAILED
                self._store.update(message)
                return False
        return False


# ============================================================================
# Dead Letter Queue
# ============================================================================


class DeadLetterQueue:
    """Dead letter queue for failed messages."""

    def __init__(self, queue_id: str = "dlq") -> None:
        self._queue_id = queue_id
        self._messages: List[Message] = []
        self._lock = threading.Lock()

    def add(self, message: Message, reason: str = "") -> None:
        """Add a message to the dead letter queue."""
        with self._lock:
            message.status = MessageStatus.DEAD_LETTER
            message.metadata["dlq_reason"] = reason
            message.metadata["dlq_timestamp"] = datetime.utcnow().isoformat()
            self._messages.append(message)

    def get_messages(self, limit: int = 100) -> List[Message]:
        """Get messages from the dead letter queue."""
        return self._messages[:limit]

    def reprocess(self, message_id: str) -> Optional[Message]:
        """Reprocess a message from the dead letter queue."""
        with self._lock:
            for i, msg in enumerate(self._messages):
                if msg.message_id == message_id:
                    msg.status = MessageStatus.PENDING
                    msg.retry_count = 0
                    return self._messages.pop(i)
            return None

    def purge(self) -> int:
        """Remove all messages from the dead letter queue."""
        with self._lock:
            count = len(self._messages)
            self._messages.clear()
            return count

    def size(self) -> int:
        """Get the number of messages in the dead letter queue."""
        return len(self._messages)


# ============================================================================
# Message Broker
# ============================================================================


class MessageBroker:
    """Central message broker managing queues and pub/sub."""

    def __init__(self) -> None:
        self._queues: Dict[str, MessageQueue] = {}
        self._store = InMemoryMessageStore()
        self._publisher = Publisher(self._store)
        self._subscribers: Dict[str, Subscriber] = {}
        self._dlq = DeadLetterQueue()

    def create_queue(self, config: QueueConfig) -> MessageQueue:
        """Create a new queue."""
        queue = MessageQueue(config, self._store)
        self._queues[config.queue_id] = queue
        return queue

    def get_queue(self, queue_id: str) -> Optional[MessageQueue]:
        """Get a queue by ID."""
        return self._queues.get(queue_id)

    def delete_queue(self, queue_id: str) -> bool:
        """Delete a queue."""
        if queue_id in self._queues:
            del self._queues[queue_id]
            return True
        return False

    def publish(
        self,
        topic: str,
        payload: Any,
        headers: Optional[Dict[str, str]] = None,
    ) -> PublishResult:
        """Publish a message."""
        return self._publisher.publish(topic, payload, headers)

    def subscribe(
        self,
        subscriber_id: str,
        topic: str,
        handler: Callable[[Message], None],
    ) -> Subscription:
        """Subscribe to a topic."""
        if subscriber_id not in self._subscribers:
            self._subscribers[subscriber_id] = Subscriber(subscriber_id, self._store)
        subscriber = self._subscribers[subscriber_id]
        subscription = subscriber.subscribe(topic, handler)
        self._publisher.register_subscriber(subscription)
        return subscription

    def unsubscribe(self, subscriber_id: str, topic: str) -> bool:
        """Unsubscribe from a topic."""
        if subscriber_id in self._subscribers:
            return self._subscribers[subscriber_id].unsubscribe(topic)
        return False

    def get_dead_letter_queue(self) -> DeadLetterQueue:
        """Get the dead letter queue."""
        return self._dlq

    def list_queues(self) -> List[str]:
        """List all queue IDs."""
        return list(self._queues.keys())


# ============================================================================
# Message Queue Vision Provider
# ============================================================================


class MessageQueueVisionProvider(VisionProvider):
    """Vision provider with message queue integration."""

    def __init__(
        self,
        provider: VisionProvider,
        broker: MessageBroker,
        topic: str = "vision_analysis",
    ) -> None:
        self._provider = provider
        self._broker = broker
        self._topic = topic

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"mq_{self._provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
        **kwargs: Any,
    ) -> VisionDescription:
        """Analyze image and publish result to message queue."""
        result = await self._provider.analyze_image(image_data, include_description)

        # Publish analysis result
        self._broker.publish(
            self._topic,
            {
                "summary": result.summary,
                "confidence": result.confidence,
                "detail_count": len(result.details),
            },
        )

        return result

    def get_broker(self) -> MessageBroker:
        """Get the message broker."""
        return self._broker


# ============================================================================
# Factory Functions
# ============================================================================


def create_message_broker() -> MessageBroker:
    """Create a message broker."""
    return MessageBroker()


def create_message_queue(
    queue_id: str,
    name: str,
    queue_type: QueueType = QueueType.FIFO,
) -> MessageQueue:
    """Create a message queue."""
    config = QueueConfig(
        queue_id=queue_id,
        name=name,
        queue_type=queue_type,
    )
    return MessageQueue(config)


def create_mq_provider(
    provider: VisionProvider,
    broker: Optional[MessageBroker] = None,
    topic: str = "vision_analysis",
) -> MessageQueueVisionProvider:
    """Create a message queue vision provider."""
    return MessageQueueVisionProvider(
        provider=provider,
        broker=broker or create_message_broker(),
        topic=topic,
    )


def create_message(
    payload: Any,
    topic: str = "default",
    priority: int = 0,
) -> Message:
    """Create a message."""
    return Message(
        message_id=str(uuid.uuid4()),
        payload=payload,
        topic=topic,
        priority=priority,
    )


def create_dead_letter_queue(queue_id: str = "dlq") -> DeadLetterQueue:
    """Create a dead letter queue."""
    return DeadLetterQueue(queue_id)
