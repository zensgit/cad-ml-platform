"""Message Producer Implementation.

Provides message producers for Kafka and RabbitMQ.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A message to be sent."""

    topic: str
    payload: Any
    key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    partition: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "topic": self.topic,
            "key": self.key,
            "payload": self.payload,
            "headers": self.headers,
            "timestamp": self.timestamp.isoformat(),
        }

    def serialize(self) -> bytes:
        """Serialize message for transmission."""
        return json.dumps(self.to_dict(), default=str).encode("utf-8")


@dataclass
class SendResult:
    """Result of sending a message."""

    success: bool
    message_id: str
    topic: str
    partition: Optional[int] = None
    offset: Optional[int] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MessageProducer(ABC):
    """Abstract base class for message producers."""

    @abstractmethod
    async def send(self, message: Message) -> SendResult:
        """Send a single message."""
        pass

    @abstractmethod
    async def send_batch(self, messages: List[Message]) -> List[SendResult]:
        """Send multiple messages."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the producer."""
        pass

    async def send_event(
        self,
        topic: str,
        event_type: str,
        data: Any,
        key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> SendResult:
        """Convenience method to send an event."""
        message = Message(
            topic=topic,
            payload={"event_type": event_type, "data": data},
            key=key,
            headers=headers or {},
        )
        return await self.send(message)


class KafkaProducer(MessageProducer):
    """Apache Kafka producer."""

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        client_id: str = "cad-ml-platform",
        acks: str = "all",
        retries: int = 3,
        batch_size: int = 16384,
        linger_ms: int = 10,
        compression_type: str = "gzip",
    ):
        self.bootstrap_servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self.client_id = client_id
        self.acks = acks
        self.retries = retries
        self.batch_size = batch_size
        self.linger_ms = linger_ms
        self.compression_type = compression_type
        self._producer: Optional[Any] = None

    async def _get_producer(self) -> Any:
        """Get or create Kafka producer."""
        if self._producer is None:
            try:
                from aiokafka import AIOKafkaProducer
                self._producer = AIOKafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    client_id=self.client_id,
                    acks=self.acks,
                    compression_type=self.compression_type,
                    linger_ms=self.linger_ms,
                    max_batch_size=self.batch_size,
                )
                await self._producer.start()
                logger.info(f"Kafka producer connected to {self.bootstrap_servers}")
            except ImportError:
                raise RuntimeError("aiokafka package not installed")
        return self._producer

    async def send(self, message: Message) -> SendResult:
        try:
            producer = await self._get_producer()

            key = message.key.encode("utf-8") if message.key else None
            value = message.serialize()
            headers = [(k, v.encode("utf-8")) for k, v in message.headers.items()]

            result = await producer.send_and_wait(
                topic=message.topic,
                value=value,
                key=key,
                headers=headers,
                partition=message.partition,
            )

            return SendResult(
                success=True,
                message_id=message.message_id,
                topic=message.topic,
                partition=result.partition,
                offset=result.offset,
            )

        except Exception as e:
            logger.error(f"Kafka send error: {e}")
            return SendResult(
                success=False,
                message_id=message.message_id,
                topic=message.topic,
                error=str(e),
            )

    async def send_batch(self, messages: List[Message]) -> List[SendResult]:
        results = []
        producer = await self._get_producer()

        # Create batch
        batch = producer.create_batch()
        pending_messages = []

        for message in messages:
            key = message.key.encode("utf-8") if message.key else None
            value = message.serialize()
            headers = [(k, v.encode("utf-8")) for k, v in message.headers.items()]

            try:
                metadata = batch.append(
                    key=key,
                    value=value,
                    headers=headers,
                    timestamp=None,
                )
                if metadata is None:
                    # Batch full, send and create new
                    await producer.send_batch(batch, message.topic)
                    batch = producer.create_batch()
                    batch.append(key=key, value=value, headers=headers, timestamp=None)

                pending_messages.append(message)
            except Exception as e:
                results.append(SendResult(
                    success=False,
                    message_id=message.message_id,
                    topic=message.topic,
                    error=str(e),
                ))

        # Send remaining batch
        if pending_messages:
            try:
                await producer.send_batch(batch, pending_messages[0].topic)
                for msg in pending_messages:
                    results.append(SendResult(
                        success=True,
                        message_id=msg.message_id,
                        topic=msg.topic,
                    ))
            except Exception as e:
                for msg in pending_messages:
                    results.append(SendResult(
                        success=False,
                        message_id=msg.message_id,
                        topic=msg.topic,
                        error=str(e),
                    ))

        return results

    async def close(self) -> None:
        if self._producer:
            await self._producer.stop()
            self._producer = None
            logger.info("Kafka producer closed")


class RabbitMQProducer(MessageProducer):
    """RabbitMQ producer."""

    def __init__(
        self,
        url: Optional[str] = None,
        exchange: str = "cad_ml_events",
        exchange_type: str = "topic",
        durable: bool = True,
    ):
        self.url = url or os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost/")
        self.exchange = exchange
        self.exchange_type = exchange_type
        self.durable = durable
        self._connection: Optional[Any] = None
        self._channel: Optional[Any] = None

    async def _get_channel(self) -> Any:
        """Get or create RabbitMQ channel."""
        if self._channel is None:
            try:
                import aio_pika
                self._connection = await aio_pika.connect_robust(self.url)
                self._channel = await self._connection.channel()

                # Declare exchange
                await self._channel.declare_exchange(
                    self.exchange,
                    aio_pika.ExchangeType[self.exchange_type.upper()],
                    durable=self.durable,
                )
                logger.info(f"RabbitMQ producer connected")
            except ImportError:
                raise RuntimeError("aio-pika package not installed")
        return self._channel

    async def send(self, message: Message) -> SendResult:
        try:
            import aio_pika
            channel = await self._get_channel()
            exchange = await channel.get_exchange(self.exchange)

            # Build message
            amqp_message = aio_pika.Message(
                body=message.serialize(),
                message_id=message.message_id,
                timestamp=message.timestamp,
                headers=message.headers,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            )

            # Publish
            routing_key = message.key or message.topic
            await exchange.publish(amqp_message, routing_key=routing_key)

            return SendResult(
                success=True,
                message_id=message.message_id,
                topic=message.topic,
            )

        except Exception as e:
            logger.error(f"RabbitMQ send error: {e}")
            return SendResult(
                success=False,
                message_id=message.message_id,
                topic=message.topic,
                error=str(e),
            )

    async def send_batch(self, messages: List[Message]) -> List[SendResult]:
        results = []
        for message in messages:
            result = await self.send(message)
            results.append(result)
        return results

    async def close(self) -> None:
        if self._channel:
            await self._channel.close()
            self._channel = None
        if self._connection:
            await self._connection.close()
            self._connection = None
        logger.info("RabbitMQ producer closed")


class InMemoryProducer(MessageProducer):
    """In-memory producer for testing."""

    def __init__(self):
        self._messages: Dict[str, List[Message]] = {}
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def send(self, message: Message) -> SendResult:
        async with self._get_lock():
            if message.topic not in self._messages:
                self._messages[message.topic] = []
            self._messages[message.topic].append(message)

        return SendResult(
            success=True,
            message_id=message.message_id,
            topic=message.topic,
            offset=len(self._messages[message.topic]) - 1,
        )

    async def send_batch(self, messages: List[Message]) -> List[SendResult]:
        return [await self.send(msg) for msg in messages]

    async def close(self) -> None:
        self._messages.clear()

    def get_messages(self, topic: str) -> List[Message]:
        """Get all messages for a topic (for testing)."""
        return self._messages.get(topic, [])


# Global producer
_producer: Optional[MessageProducer] = None


def get_producer(producer_type: str = "kafka") -> MessageProducer:
    """Get global message producer."""
    global _producer
    if _producer is None:
        if producer_type == "kafka":
            _producer = KafkaProducer()
        elif producer_type == "rabbitmq":
            _producer = RabbitMQProducer()
        else:
            _producer = InMemoryProducer()
    return _producer


def set_producer(producer: MessageProducer) -> None:
    """Set global message producer."""
    global _producer
    _producer = producer
