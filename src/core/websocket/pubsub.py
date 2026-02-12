"""WebSocket Pub/Sub Backend.

Provides pub/sub backends for distributed WebSocket messaging.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class PubSubMessage:
    """A pub/sub message."""
    channel: str
    data: Any
    message_id: Optional[str] = None


class PubSubBackend(ABC):
    """Abstract base class for pub/sub backends."""

    @abstractmethod
    async def publish(self, channel: str, message: Any) -> bool:
        """Publish a message to a channel.

        Args:
            channel: Channel name
            message: Message data

        Returns:
            True if published
        """
        pass

    @abstractmethod
    async def subscribe(self, channel: str) -> AsyncIterator[PubSubMessage]:
        """Subscribe to a channel.

        Args:
            channel: Channel name

        Yields:
            Messages from the channel
        """
        pass

    @abstractmethod
    async def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from a channel.

        Args:
            channel: Channel name
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the pub/sub connection."""
        pass


class InMemoryPubSub(PubSubBackend):
    """In-memory pub/sub for single-instance deployments."""

    def __init__(self):
        self._channels: Dict[str, List[asyncio.Queue]] = {}
        self._subscriptions: Dict[str, Set[asyncio.Queue]] = {}
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def publish(self, channel: str, message: Any) -> bool:
        """Publish to in-memory subscribers."""
        async with self._get_lock():
            queues = self._subscriptions.get(channel, set())

            # Also publish to pattern subscribers
            for pattern, pattern_queues in self._subscriptions.items():
                if pattern.endswith("*") and channel.startswith(pattern[:-1]):
                    queues = queues | pattern_queues

            if not queues:
                return False

            pub_message = PubSubMessage(channel=channel, data=message)

            for queue in queues:
                try:
                    queue.put_nowait(pub_message)
                except asyncio.QueueFull:
                    logger.warning(f"Queue full for channel {channel}")

            return True

    async def subscribe(self, channel: str) -> AsyncIterator[PubSubMessage]:
        """Subscribe to in-memory channel."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

        async with self._get_lock():
            if channel not in self._subscriptions:
                self._subscriptions[channel] = set()
            self._subscriptions[channel].add(queue)

        try:
            while True:
                message = await queue.get()
                yield message
        finally:
            async with self._get_lock():
                if channel in self._subscriptions:
                    self._subscriptions[channel].discard(queue)
                    if not self._subscriptions[channel]:
                        del self._subscriptions[channel]

    async def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from channel."""
        async with self._get_lock():
            self._subscriptions.pop(channel, None)

    async def close(self) -> None:
        """Clear all subscriptions."""
        async with self._get_lock():
            self._subscriptions.clear()


class RedisPubSub(PubSubBackend):
    """Redis-based pub/sub for distributed deployments."""

    def __init__(
        self,
        url: Optional[str] = None,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
    ):
        self.url = url or os.getenv("REDIS_URL")
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", "6379"))
        self.db = db
        self.password = password or os.getenv("REDIS_PASSWORD")

        self._redis: Optional[Any] = None
        self._pubsub: Optional[Any] = None
        self._subscriptions: Set[str] = set()

    async def _get_redis(self) -> Any:
        """Get or create Redis connection."""
        if self._redis is None:
            try:
                import aioredis

                if self.url:
                    self._redis = await aioredis.from_url(self.url)
                else:
                    self._redis = await aioredis.Redis(
                        host=self.host,
                        port=self.port,
                        db=self.db,
                        password=self.password,
                    )
                logger.info(f"Connected to Redis at {self.host}:{self.port}")

            except ImportError:
                raise RuntimeError("aioredis package not installed")

        return self._redis

    async def _get_pubsub(self) -> Any:
        """Get or create Redis pubsub connection."""
        if self._pubsub is None:
            redis = await self._get_redis()
            self._pubsub = redis.pubsub()
        return self._pubsub

    async def publish(self, channel: str, message: Any) -> bool:
        """Publish to Redis channel."""
        try:
            redis = await self._get_redis()

            # Serialize message
            if isinstance(message, dict):
                message = json.dumps(message, default=str)
            elif not isinstance(message, (str, bytes)):
                message = json.dumps(message, default=str)

            await redis.publish(channel, message)
            return True

        except Exception as e:
            logger.error(f"Redis publish error: {e}")
            return False

    async def subscribe(self, channel: str) -> AsyncIterator[PubSubMessage]:
        """Subscribe to Redis channel."""
        pubsub = await self._get_pubsub()

        # Support pattern subscriptions
        if "*" in channel:
            await pubsub.psubscribe(channel)
        else:
            await pubsub.subscribe(channel)

        self._subscriptions.add(channel)

        try:
            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )

                if message is None:
                    await asyncio.sleep(0.01)
                    continue

                # Parse message
                data = message.get("data")
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        pass

                msg_channel = message.get("channel", channel)
                if isinstance(msg_channel, bytes):
                    msg_channel = msg_channel.decode("utf-8")

                yield PubSubMessage(channel=msg_channel, data=data)

        finally:
            if "*" in channel:
                await pubsub.punsubscribe(channel)
            else:
                await pubsub.unsubscribe(channel)
            self._subscriptions.discard(channel)

    async def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from Redis channel."""
        if self._pubsub:
            if "*" in channel:
                await self._pubsub.punsubscribe(channel)
            else:
                await self._pubsub.unsubscribe(channel)
        self._subscriptions.discard(channel)

    async def close(self) -> None:
        """Close Redis connections."""
        if self._pubsub:
            for channel in list(self._subscriptions):
                await self.unsubscribe(channel)
            await self._pubsub.close()
            self._pubsub = None

        if self._redis:
            await self._redis.close()
            self._redis = None

        logger.info("Redis pub/sub connection closed")


class PubSubRouter:
    """Router for distributing messages across multiple backends."""

    def __init__(self, backends: Optional[List[PubSubBackend]] = None):
        self._backends = backends or []
        self._channel_handlers: Dict[str, List[Callable[[PubSubMessage], Any]]] = {}

    def add_backend(self, backend: PubSubBackend) -> None:
        """Add a pub/sub backend."""
        self._backends.append(backend)

    async def publish(self, channel: str, message: Any) -> int:
        """Publish to all backends.

        Returns:
            Number of successful publishes
        """
        count = 0
        for backend in self._backends:
            try:
                if await backend.publish(channel, message):
                    count += 1
            except Exception as e:
                logger.error(f"Publish error on {backend.__class__.__name__}: {e}")
        return count

    def on(self, channel: str, handler: Callable[[PubSubMessage], Any]) -> None:
        """Register a handler for a channel."""
        if channel not in self._channel_handlers:
            self._channel_handlers[channel] = []
        self._channel_handlers[channel].append(handler)

    async def start_listening(self, channels: List[str]) -> None:
        """Start listening to channels on all backends."""
        tasks = []

        for backend in self._backends:
            for channel in channels:
                task = asyncio.create_task(
                    self._listen_channel(backend, channel)
                )
                tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _listen_channel(self, backend: PubSubBackend, channel: str) -> None:
        """Listen to a single channel."""
        try:
            async for message in backend.subscribe(channel):
                handlers = self._channel_handlers.get(channel, [])
                # Also check pattern handlers
                for pattern, pattern_handlers in self._channel_handlers.items():
                    if pattern.endswith("*") and message.channel.startswith(pattern[:-1]):
                        handlers.extend(pattern_handlers)

                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message)
                        else:
                            handler(message)
                    except Exception as e:
                        logger.error(f"Handler error for {channel}: {e}")

        except Exception as e:
            logger.error(f"Listen error for {channel}: {e}")

    async def close(self) -> None:
        """Close all backends."""
        for backend in self._backends:
            await backend.close()


# Global pub/sub instance
_pubsub: Optional[PubSubBackend] = None


def get_pubsub(backend_type: str = "memory") -> PubSubBackend:
    """Get global pub/sub instance.

    Args:
        backend_type: "memory" or "redis"

    Returns:
        PubSubBackend instance
    """
    global _pubsub

    if _pubsub is None:
        if backend_type == "redis":
            _pubsub = RedisPubSub()
        else:
            _pubsub = InMemoryPubSub()

    return _pubsub


def set_pubsub(backend: PubSubBackend) -> None:
    """Set global pub/sub instance."""
    global _pubsub
    _pubsub = backend
