"""Message Consumer Implementation.

Provides message consumers for Kafka and RabbitMQ.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class ReceivedMessage:
    """A received message."""

    topic: str
    payload: Any
    key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    message_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    partition: Optional[int] = None
    offset: Optional[int] = None
    raw: Optional[Any] = None  # Original message object

    @classmethod
    def from_bytes(cls, topic: str, data: bytes, **kwargs: Any) -> "ReceivedMessage":
        """Create from serialized bytes."""
        try:
            parsed = json.loads(data.decode("utf-8"))
            return cls(
                topic=topic,
                payload=parsed.get("payload", parsed),
                key=parsed.get("key"),
                headers=parsed.get("headers", {}),
                message_id=parsed.get("message_id"),
                timestamp=datetime.fromisoformat(parsed["timestamp"]) if "timestamp" in parsed else None,
                **kwargs,
            )
        except (json.JSONDecodeError, KeyError):
            return cls(topic=topic, payload=data.decode("utf-8"), **kwargs)


MessageHandler = Callable[[ReceivedMessage], Any]


class MessageConsumer(ABC):
    """Abstract base class for message consumers."""

    @abstractmethod
    async def subscribe(self, topics: List[str]) -> None:
        """Subscribe to topics."""
        pass

    @abstractmethod
    async def consume(self, handler: MessageHandler, max_messages: Optional[int] = None) -> None:
        """Start consuming messages."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the consumer."""
        pass

    @abstractmethod
    async def commit(self) -> None:
        """Commit current offsets."""
        pass


class KafkaConsumer(MessageConsumer):
    """Apache Kafka consumer."""

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        group_id: str = "cad-ml-platform",
        auto_offset_reset: str = "earliest",
        enable_auto_commit: bool = False,
        max_poll_records: int = 500,
    ):
        self.bootstrap_servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.enable_auto_commit = enable_auto_commit
        self.max_poll_records = max_poll_records
        self._consumer: Optional[Any] = None
        self._running = False

    async def _get_consumer(self, topics: List[str]) -> Any:
        """Get or create Kafka consumer."""
        if self._consumer is None:
            try:
                from aiokafka import AIOKafkaConsumer
                self._consumer = AIOKafkaConsumer(
                    *topics,
                    bootstrap_servers=self.bootstrap_servers,
                    group_id=self.group_id,
                    auto_offset_reset=self.auto_offset_reset,
                    enable_auto_commit=self.enable_auto_commit,
                    max_poll_records=self.max_poll_records,
                )
                await self._consumer.start()
                logger.info(f"Kafka consumer connected, subscribed to {topics}")
            except ImportError:
                raise RuntimeError("aiokafka package not installed")
        return self._consumer

    async def subscribe(self, topics: List[str]) -> None:
        await self._get_consumer(topics)

    async def consume(self, handler: MessageHandler, max_messages: Optional[int] = None) -> None:
        if not self._consumer:
            raise RuntimeError("Consumer not subscribed to any topics")

        self._running = True
        message_count = 0

        try:
            async for msg in self._consumer:
                if not self._running:
                    break

                # Parse message
                received = ReceivedMessage.from_bytes(
                    topic=msg.topic,
                    data=msg.value,
                    key=msg.key.decode("utf-8") if msg.key else None,
                    headers={k: v.decode("utf-8") for k, v in (msg.headers or [])},
                    partition=msg.partition,
                    offset=msg.offset,
                    timestamp=datetime.fromtimestamp(msg.timestamp / 1000) if msg.timestamp else None,
                    raw=msg,
                )

                # Handle message
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(received)
                    else:
                        handler(received)

                    message_count += 1

                    if max_messages and message_count >= max_messages:
                        break

                except Exception as e:
                    logger.error(f"Message handler error: {e}")

        except asyncio.CancelledError:
            pass

    async def commit(self) -> None:
        if self._consumer:
            await self._consumer.commit()

    async def close(self) -> None:
        self._running = False
        if self._consumer:
            await self._consumer.stop()
            self._consumer = None
            logger.info("Kafka consumer closed")


class RabbitMQConsumer(MessageConsumer):
    """RabbitMQ consumer."""

    def __init__(
        self,
        url: Optional[str] = None,
        exchange: str = "cad_ml_events",
        queue_prefix: str = "cad_ml",
        prefetch_count: int = 10,
        durable: bool = True,
    ):
        self.url = url or os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost/")
        self.exchange = exchange
        self.queue_prefix = queue_prefix
        self.prefetch_count = prefetch_count
        self.durable = durable
        self._connection: Optional[Any] = None
        self._channel: Optional[Any] = None
        self._queues: Dict[str, Any] = {}
        self._running = False

    async def _setup(self, topics: List[str]) -> None:
        """Setup RabbitMQ connection and queues."""
        try:
            import aio_pika
            self._connection = await aio_pika.connect_robust(self.url)
            self._channel = await self._connection.channel()
            await self._channel.set_qos(prefetch_count=self.prefetch_count)

            # Get exchange
            exchange = await self._channel.get_exchange(self.exchange)

            # Create queues and bind to topics
            for topic in topics:
                queue_name = f"{self.queue_prefix}_{topic.replace('.', '_')}"
                queue = await self._channel.declare_queue(
                    queue_name,
                    durable=self.durable,
                )
                await queue.bind(exchange, routing_key=topic)
                self._queues[topic] = queue

            logger.info(f"RabbitMQ consumer connected, subscribed to {topics}")

        except ImportError:
            raise RuntimeError("aio-pika package not installed")

    async def subscribe(self, topics: List[str]) -> None:
        await self._setup(topics)

    async def consume(self, handler: MessageHandler, max_messages: Optional[int] = None) -> None:
        if not self._queues:
            raise RuntimeError("Consumer not subscribed to any topics")

        self._running = True
        message_count = 0

        async def on_message(message: Any) -> None:
            nonlocal message_count

            async with message.process():
                received = ReceivedMessage.from_bytes(
                    topic=message.routing_key or "unknown",
                    data=message.body,
                    message_id=message.message_id,
                    headers={k: v for k, v in (message.headers or {}).items()},
                    timestamp=message.timestamp,
                    raw=message,
                )

                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(received)
                    else:
                        handler(received)
                    message_count += 1
                except Exception as e:
                    logger.error(f"Message handler error: {e}")

        # Start consuming from all queues
        consumers = []
        for topic, queue in self._queues.items():
            consumer = await queue.consume(on_message)
            consumers.append(consumer)

        # Wait until stopped or max messages reached
        while self._running:
            if max_messages and message_count >= max_messages:
                break
            await asyncio.sleep(0.1)

        # Cancel consumers
        for consumer in consumers:
            await consumer.cancel()

    async def commit(self) -> None:
        # RabbitMQ uses ack per message, handled in on_message
        pass

    async def close(self) -> None:
        self._running = False
        if self._channel:
            await self._channel.close()
            self._channel = None
        if self._connection:
            await self._connection.close()
            self._connection = None
        self._queues.clear()
        logger.info("RabbitMQ consumer closed")


class InMemoryConsumer(MessageConsumer):
    """In-memory consumer for testing."""

    def __init__(self, producer: Any = None):
        self._producer = producer
        self._subscribed_topics: Set[str] = set()
        self._running = False

    async def subscribe(self, topics: List[str]) -> None:
        self._subscribed_topics = set(topics)

    async def consume(self, handler: MessageHandler, max_messages: Optional[int] = None) -> None:
        if not self._producer:
            return

        self._running = True
        message_count = 0

        while self._running:
            for topic in self._subscribed_topics:
                messages = self._producer.get_messages(topic)
                for msg in messages[message_count:]:
                    received = ReceivedMessage(
                        topic=msg.topic,
                        payload=msg.payload,
                        key=msg.key,
                        headers=msg.headers,
                        message_id=msg.message_id,
                        timestamp=msg.timestamp,
                    )

                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(received)
                        else:
                            handler(received)
                        message_count += 1

                        if max_messages and message_count >= max_messages:
                            return
                    except Exception as e:
                        logger.error(f"Handler error: {e}")

            await asyncio.sleep(0.1)

    async def commit(self) -> None:
        pass

    async def close(self) -> None:
        self._running = False


# Handler registry for declarative message handling
_handlers: Dict[str, List[MessageHandler]] = {}


def message_handler(
    topic: str,
    group: str = "default",
) -> Callable[[F], F]:
    """Decorator to register a message handler.

    Args:
        topic: Topic to handle messages from
        group: Consumer group

    Example:
        @message_handler("document.created")
        async def handle_document_created(message: ReceivedMessage):
            ...
    """
    def decorator(func: F) -> F:
        key = f"{group}:{topic}"
        if key not in _handlers:
            _handlers[key] = []
        _handlers[key].append(func)

        @wraps(func)
        async def wrapper(message: ReceivedMessage) -> Any:
            if asyncio.iscoroutinefunction(func):
                return await func(message)
            return func(message)

        return wrapper  # type: ignore

    return decorator


def get_handlers(topic: str, group: str = "default") -> List[MessageHandler]:
    """Get registered handlers for a topic."""
    key = f"{group}:{topic}"
    return _handlers.get(key, [])


class ConsumerWorker:
    """Worker that runs consumers with registered handlers."""

    def __init__(
        self,
        consumer: MessageConsumer,
        group: str = "default",
    ):
        self.consumer = consumer
        self.group = group
        self._task: Optional[asyncio.Task] = None

    async def start(self, topics: List[str]) -> None:
        """Start consuming messages."""
        await self.consumer.subscribe(topics)

        async def dispatch(message: ReceivedMessage) -> None:
            handlers = get_handlers(message.topic, self.group)
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                except Exception as e:
                    logger.error(f"Handler error for {message.topic}: {e}")

        self._task = asyncio.create_task(self.consumer.consume(dispatch))
        logger.info(f"Consumer worker started for topics: {topics}")

    async def stop(self) -> None:
        """Stop consuming messages."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self.consumer.close()
        logger.info("Consumer worker stopped")
