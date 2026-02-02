"""Message Bus Module.

Provides messaging infrastructure:
- Pub/sub messaging
- Message routing
- Dead letter queues
- Message persistence
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class MessageStatus(Enum):
    """Message status."""
    PENDING = "pending"
    PROCESSING = "processing"
    DELIVERED = "delivered"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


@dataclass
class Message:
    """A message in the bus."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = ""
    payload: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "topic": self.topic,
            "payload": self.payload,
            "headers": self.headers,
            "priority": self.priority.value,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            topic=data.get("topic", ""),
            payload=data.get("payload"),
            headers=data.get("headers", {}),
            priority=MessagePriority(data.get("priority", 1)),
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
        )

    @property
    def is_expired(self) -> bool:
        """Check if message is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    @property
    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.retry_count < self.max_retries


@dataclass
class Subscription:
    """A topic subscription."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = ""
    pattern: Optional[str] = None  # For pattern-based routing
    handler: Optional[Callable[[Message], Any]] = None
    filter_fn: Optional[Callable[[Message], bool]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def matches(self, topic: str) -> bool:
        """Check if subscription matches topic."""
        if self.pattern:
            # Simple wildcard matching
            if self.pattern.endswith("*"):
                return topic.startswith(self.pattern[:-1])
            elif self.pattern.startswith("*"):
                return topic.endswith(self.pattern[1:])
            elif "*" in self.pattern:
                parts = self.pattern.split("*")
                return topic.startswith(parts[0]) and topic.endswith(parts[-1])
            return self.pattern == topic
        return self.topic == topic


class MessageHandler(ABC):
    """Abstract message handler."""

    @abstractmethod
    async def handle(self, message: Message) -> None:
        """Handle a message."""
        pass


class MessageBus(ABC):
    """Abstract message bus."""

    @abstractmethod
    async def publish(self, message: Message) -> bool:
        """Publish a message."""
        pass

    @abstractmethod
    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Message], Any],
        pattern: Optional[str] = None,
    ) -> str:
        """Subscribe to a topic. Returns subscription ID."""
        pass

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from a topic."""
        pass

    @abstractmethod
    async def request(
        self,
        message: Message,
        timeout: float = 30.0,
    ) -> Optional[Message]:
        """Request-reply pattern."""
        pass


class InMemoryMessageBus(MessageBus):
    """In-memory message bus implementation."""

    def __init__(
        self,
        dead_letter_handler: Optional[Callable[[Message], Any]] = None,
    ):
        self._subscriptions: Dict[str, Subscription] = {}
        self._topic_subscriptions: Dict[str, Set[str]] = {}  # topic -> subscription IDs
        self._pending_replies: Dict[str, asyncio.Future] = {}
        self._dead_letter_queue: List[Message] = []
        self._dead_letter_handler = dead_letter_handler
        self._lock = asyncio.Lock()
        self._message_count = 0

    async def publish(self, message: Message) -> bool:
        """Publish a message to subscribers."""
        if message.is_expired:
            logger.warning(f"Message {message.id} is expired, not publishing")
            return False

        async with self._lock:
            self._message_count += 1

            # Find matching subscriptions
            matching_subs = []

            # Check direct topic subscriptions
            if message.topic in self._topic_subscriptions:
                for sub_id in self._topic_subscriptions[message.topic]:
                    sub = self._subscriptions.get(sub_id)
                    if sub and (not sub.filter_fn or sub.filter_fn(message)):
                        matching_subs.append(sub)

            # Check pattern subscriptions
            for sub in self._subscriptions.values():
                if sub.pattern and sub.matches(message.topic):
                    if not sub.filter_fn or sub.filter_fn(message):
                        if sub not in matching_subs:
                            matching_subs.append(sub)

        if not matching_subs:
            logger.debug(f"No subscribers for topic {message.topic}")
            return True

        # Deliver to all matching subscribers
        for sub in matching_subs:
            if sub.handler:
                try:
                    result = sub.handler(message)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Handler error for {message.id}: {e}")
                    await self._handle_failure(message, e)

        # Handle reply if this is a response
        if message.correlation_id and message.correlation_id in self._pending_replies:
            future = self._pending_replies.pop(message.correlation_id)
            if not future.done():
                future.set_result(message)

        return True

    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Message], Any],
        pattern: Optional[str] = None,
        filter_fn: Optional[Callable[[Message], bool]] = None,
    ) -> str:
        """Subscribe to a topic or pattern."""
        async with self._lock:
            sub = Subscription(
                topic=topic,
                pattern=pattern,
                handler=handler,
                filter_fn=filter_fn,
            )

            self._subscriptions[sub.id] = sub

            if not pattern:
                if topic not in self._topic_subscriptions:
                    self._topic_subscriptions[topic] = set()
                self._topic_subscriptions[topic].add(sub.id)

            logger.info(f"Subscribed {sub.id} to {pattern or topic}")
            return sub.id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from a topic."""
        async with self._lock:
            sub = self._subscriptions.pop(subscription_id, None)
            if not sub:
                return False

            # Remove from topic index
            if sub.topic in self._topic_subscriptions:
                self._topic_subscriptions[sub.topic].discard(subscription_id)

            logger.info(f"Unsubscribed {subscription_id}")
            return True

    async def request(
        self,
        message: Message,
        timeout: float = 30.0,
    ) -> Optional[Message]:
        """Send request and wait for reply."""
        if not message.correlation_id:
            message.correlation_id = message.id

        reply_topic = message.reply_to or f"_reply.{message.id}"
        message.reply_to = reply_topic

        # Create future for reply
        future: asyncio.Future = asyncio.Future()
        self._pending_replies[message.correlation_id] = future

        # Subscribe to reply topic temporarily
        async def reply_handler(reply: Message):
            pass  # Reply is handled via future

        sub_id = await self.subscribe(reply_topic, reply_handler)

        try:
            # Publish request
            await self.publish(message)

            # Wait for reply
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Request {message.id} timed out")
            return None
        finally:
            await self.unsubscribe(sub_id)
            self._pending_replies.pop(message.correlation_id, None)

    async def _handle_failure(self, message: Message, error: Exception) -> None:
        """Handle message delivery failure."""
        message.retry_count += 1

        if message.can_retry:
            logger.info(f"Retrying message {message.id}, attempt {message.retry_count}")
            await asyncio.sleep(min(2 ** message.retry_count, 30))  # Exponential backoff
            await self.publish(message)
        else:
            logger.warning(f"Message {message.id} moved to dead letter queue")
            self._dead_letter_queue.append(message)

            if self._dead_letter_handler:
                try:
                    result = self._dead_letter_handler(message)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Dead letter handler error: {e}")

    def get_dead_letter_queue(self) -> List[Message]:
        """Get messages in dead letter queue."""
        return self._dead_letter_queue.copy()

    def clear_dead_letter_queue(self) -> int:
        """Clear dead letter queue. Returns count of cleared messages."""
        count = len(self._dead_letter_queue)
        self._dead_letter_queue.clear()
        return count

    @property
    def message_count(self) -> int:
        """Get total messages processed."""
        return self._message_count

    @property
    def subscription_count(self) -> int:
        """Get total active subscriptions."""
        return len(self._subscriptions)


class MessageRouter:
    """Routes messages based on content or headers."""

    def __init__(self, bus: MessageBus):
        self._bus = bus
        self._routes: List[tuple[Callable[[Message], bool], str]] = []

    def add_route(
        self,
        condition: Callable[[Message], bool],
        target_topic: str,
    ) -> None:
        """Add a routing rule."""
        self._routes.append((condition, target_topic))

    async def route(self, message: Message) -> bool:
        """Route a message based on rules."""
        for condition, target_topic in self._routes:
            if condition(message):
                message.topic = target_topic
                return await self._bus.publish(message)

        # No matching route, publish to original topic
        return await self._bus.publish(message)


class MessageAggregator:
    """Aggregates messages from multiple topics."""

    def __init__(
        self,
        bus: MessageBus,
        output_topic: str,
        timeout: float = 60.0,
    ):
        self._bus = bus
        self._output_topic = output_topic
        self._timeout = timeout
        self._buffers: Dict[str, List[Message]] = {}  # correlation_id -> messages
        self._expected_counts: Dict[str, int] = {}
        self._timers: Dict[str, asyncio.Task] = {}

    async def add_message(
        self,
        message: Message,
        correlation_id: str,
        expected_count: int,
    ) -> None:
        """Add a message to aggregation buffer."""
        if correlation_id not in self._buffers:
            self._buffers[correlation_id] = []
            self._expected_counts[correlation_id] = expected_count

            # Start timeout timer
            self._timers[correlation_id] = asyncio.create_task(
                self._timeout_handler(correlation_id)
            )

        self._buffers[correlation_id].append(message)

        # Check if complete
        if len(self._buffers[correlation_id]) >= self._expected_counts[correlation_id]:
            await self._complete_aggregation(correlation_id)

    async def _complete_aggregation(self, correlation_id: str) -> None:
        """Complete aggregation and publish result."""
        if correlation_id not in self._buffers:
            return

        messages = self._buffers.pop(correlation_id)
        self._expected_counts.pop(correlation_id, None)

        # Cancel timer
        timer = self._timers.pop(correlation_id, None)
        if timer:
            timer.cancel()

        # Create aggregated message
        aggregated = Message(
            topic=self._output_topic,
            payload=[m.payload for m in messages],
            correlation_id=correlation_id,
            headers={"aggregated_count": str(len(messages))},
        )

        await self._bus.publish(aggregated)

    async def _timeout_handler(self, correlation_id: str) -> None:
        """Handle aggregation timeout."""
        await asyncio.sleep(self._timeout)

        if correlation_id in self._buffers:
            logger.warning(f"Aggregation {correlation_id} timed out")
            await self._complete_aggregation(correlation_id)


class EventEmitter:
    """Event emitter using message bus."""

    def __init__(self, bus: MessageBus, namespace: str = ""):
        self._bus = bus
        self._namespace = namespace

    def _get_topic(self, event: str) -> str:
        """Get full topic name."""
        if self._namespace:
            return f"{self._namespace}.{event}"
        return event

    async def emit(
        self,
        event: str,
        data: Any = None,
        **kwargs,
    ) -> bool:
        """Emit an event."""
        message = Message(
            topic=self._get_topic(event),
            payload=data,
            headers=kwargs.get("headers", {}),
            priority=kwargs.get("priority", MessagePriority.NORMAL),
        )
        return await self._bus.publish(message)

    async def on(
        self,
        event: str,
        handler: Callable[[Message], Any],
    ) -> str:
        """Register event handler. Returns subscription ID."""
        return await self._bus.subscribe(self._get_topic(event), handler)

    async def once(
        self,
        event: str,
        handler: Callable[[Message], Any],
    ) -> str:
        """Register one-time event handler."""
        sub_id_holder = {"id": None}

        async def wrapper(message: Message):
            await self._bus.unsubscribe(sub_id_holder["id"])
            result = handler(message)
            if asyncio.iscoroutine(result):
                await result

        sub_id = await self._bus.subscribe(self._get_topic(event), wrapper)
        sub_id_holder["id"] = sub_id
        return sub_id

    async def off(self, subscription_id: str) -> bool:
        """Remove event handler."""
        return await self._bus.unsubscribe(subscription_id)


__all__ = [
    "MessagePriority",
    "MessageStatus",
    "Message",
    "Subscription",
    "MessageHandler",
    "MessageBus",
    "InMemoryMessageBus",
    "MessageRouter",
    "MessageAggregator",
    "EventEmitter",
]
