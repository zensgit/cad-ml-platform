"""Enterprise WebSocket Manager for Real-time Notifications.

Features:
- Connection lifecycle management
- Channel-based subscriptions (per-user, per-tenant, per-topic)
- Broadcast and targeted messaging
- Heartbeat/ping-pong for connection health
- Graceful reconnection handling
- Metrics integration
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """WebSocket message types."""

    # Client -> Server
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PING = "ping"
    MESSAGE = "message"

    # Server -> Client
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    PONG = "pong"
    ERROR = "error"
    ACK = "ack"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"


class NotificationPriority(Enum):
    """Notification priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class WebSocketConnection:
    """Represents a WebSocket connection."""

    connection_id: str
    websocket: WebSocket
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    connected_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    subscriptions: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Notification:
    """Notification message."""

    id: str
    type: MessageType
    channel: str
    payload: Dict[str, Any]
    priority: NotificationPriority = NotificationPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    target_users: Optional[List[str]] = None
    target_tenants: Optional[List[str]] = None


class WebSocketManager:
    """Manages WebSocket connections and message routing."""

    def __init__(
        self,
        heartbeat_interval: float = 30.0,
        heartbeat_timeout: float = 60.0,
        max_connections_per_user: int = 5,
        max_connections_per_tenant: int = 100,
    ):
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.max_connections_per_user = max_connections_per_user
        self.max_connections_per_tenant = max_connections_per_tenant

        # Connection storage
        self._connections: Dict[str, WebSocketConnection] = {}
        self._user_connections: Dict[str, Set[str]] = {}
        self._tenant_connections: Dict[str, Set[str]] = {}
        self._channel_subscriptions: Dict[str, Set[str]] = {}

        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False

        # Message handlers
        self._message_handlers: Dict[str, Callable] = {}

        # Metrics
        self._metrics = {
            "total_connections": 0,
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "total_broadcasts": 0,
        }

    async def start(self) -> None:
        """Start the WebSocket manager."""
        if self._running:
            return

        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("WebSocket manager started")

    async def stop(self) -> None:
        """Stop the WebSocket manager."""
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        for conn_id in list(self._connections.keys()):
            await self.disconnect(conn_id)

        logger.info("WebSocket manager stopped")

    async def connect(
        self,
        websocket: WebSocket,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Accept and register a new WebSocket connection."""
        # Check connection limits
        if user_id and len(self._user_connections.get(user_id, set())) >= self.max_connections_per_user:
            await websocket.close(code=4001, reason="Max connections per user exceeded")
            raise ValueError(f"User {user_id} has too many connections")

        if tenant_id and len(self._tenant_connections.get(tenant_id, set())) >= self.max_connections_per_tenant:
            await websocket.close(code=4002, reason="Max connections per tenant exceeded")
            raise ValueError(f"Tenant {tenant_id} has too many connections")

        # Accept the connection
        await websocket.accept()

        # Create connection record
        connection_id = str(uuid.uuid4())
        connection = WebSocketConnection(
            connection_id=connection_id,
            websocket=websocket,
            user_id=user_id,
            tenant_id=tenant_id,
            metadata=metadata or {},
        )

        # Register connection
        self._connections[connection_id] = connection

        if user_id:
            if user_id not in self._user_connections:
                self._user_connections[user_id] = set()
            self._user_connections[user_id].add(connection_id)

        if tenant_id:
            if tenant_id not in self._tenant_connections:
                self._tenant_connections[tenant_id] = set()
            self._tenant_connections[tenant_id].add(connection_id)

        self._metrics["total_connections"] += 1

        logger.info(f"WebSocket connected: {connection_id} (user={user_id}, tenant={tenant_id})")

        # Send connection acknowledgment
        await self._send_to_connection(
            connection_id,
            {
                "type": MessageType.ACK.value,
                "connection_id": connection_id,
                "timestamp": time.time(),
            },
        )

        return connection_id

    async def disconnect(self, connection_id: str) -> None:
        """Disconnect and clean up a WebSocket connection."""
        connection = self._connections.get(connection_id)
        if not connection:
            return

        # Remove from user connections
        if connection.user_id and connection.user_id in self._user_connections:
            self._user_connections[connection.user_id].discard(connection_id)
            if not self._user_connections[connection.user_id]:
                del self._user_connections[connection.user_id]

        # Remove from tenant connections
        if connection.tenant_id and connection.tenant_id in self._tenant_connections:
            self._tenant_connections[connection.tenant_id].discard(connection_id)
            if not self._tenant_connections[connection.tenant_id]:
                del self._tenant_connections[connection.tenant_id]

        # Remove from channel subscriptions
        for channel in connection.subscriptions:
            if channel in self._channel_subscriptions:
                self._channel_subscriptions[channel].discard(connection_id)
                if not self._channel_subscriptions[channel]:
                    del self._channel_subscriptions[channel]

        # Close WebSocket
        try:
            await connection.websocket.close()
        except Exception:
            pass

        # Remove connection record
        del self._connections[connection_id]

        logger.info(f"WebSocket disconnected: {connection_id}")

    async def subscribe(self, connection_id: str, channel: str) -> bool:
        """Subscribe a connection to a channel."""
        connection = self._connections.get(connection_id)
        if not connection:
            return False

        connection.subscriptions.add(channel)

        if channel not in self._channel_subscriptions:
            self._channel_subscriptions[channel] = set()
        self._channel_subscriptions[channel].add(connection_id)

        # Send subscription confirmation
        await self._send_to_connection(
            connection_id,
            {
                "type": MessageType.SUBSCRIBED.value,
                "channel": channel,
                "timestamp": time.time(),
            },
        )

        logger.debug(f"Connection {connection_id} subscribed to {channel}")
        return True

    async def unsubscribe(self, connection_id: str, channel: str) -> bool:
        """Unsubscribe a connection from a channel."""
        connection = self._connections.get(connection_id)
        if not connection:
            return False

        connection.subscriptions.discard(channel)

        if channel in self._channel_subscriptions:
            self._channel_subscriptions[channel].discard(connection_id)
            if not self._channel_subscriptions[channel]:
                del self._channel_subscriptions[channel]

        # Send unsubscription confirmation
        await self._send_to_connection(
            connection_id,
            {
                "type": MessageType.UNSUBSCRIBED.value,
                "channel": channel,
                "timestamp": time.time(),
            },
        )

        logger.debug(f"Connection {connection_id} unsubscribed from {channel}")
        return True

    async def send_to_user(
        self,
        user_id: str,
        message: Dict[str, Any],
        priority: NotificationPriority = NotificationPriority.NORMAL,
    ) -> int:
        """Send a message to all connections of a user."""
        connection_ids = self._user_connections.get(user_id, set())
        sent_count = 0

        for conn_id in connection_ids:
            if await self._send_to_connection(conn_id, message):
                sent_count += 1

        return sent_count

    async def send_to_tenant(
        self,
        tenant_id: str,
        message: Dict[str, Any],
        priority: NotificationPriority = NotificationPriority.NORMAL,
    ) -> int:
        """Send a message to all connections of a tenant."""
        connection_ids = self._tenant_connections.get(tenant_id, set())
        sent_count = 0

        for conn_id in connection_ids:
            if await self._send_to_connection(conn_id, message):
                sent_count += 1

        return sent_count

    async def send_to_channel(
        self,
        channel: str,
        message: Dict[str, Any],
        priority: NotificationPriority = NotificationPriority.NORMAL,
    ) -> int:
        """Send a message to all subscribers of a channel."""
        connection_ids = self._channel_subscriptions.get(channel, set())
        sent_count = 0

        for conn_id in connection_ids:
            if await self._send_to_connection(conn_id, message):
                sent_count += 1

        return sent_count

    async def broadcast(
        self,
        message: Dict[str, Any],
        exclude_connections: Optional[Set[str]] = None,
    ) -> int:
        """Broadcast a message to all connections."""
        exclude = exclude_connections or set()
        sent_count = 0

        message["type"] = MessageType.BROADCAST.value

        for conn_id in self._connections:
            if conn_id not in exclude:
                if await self._send_to_connection(conn_id, message):
                    sent_count += 1

        self._metrics["total_broadcasts"] += 1
        return sent_count

    async def notify(self, notification: Notification) -> int:
        """Send a notification based on its targeting rules."""
        sent_count = 0
        message = {
            "type": MessageType.NOTIFICATION.value,
            "id": notification.id,
            "channel": notification.channel,
            "payload": notification.payload,
            "priority": notification.priority.value,
            "timestamp": notification.created_at,
        }

        # Target specific users
        if notification.target_users:
            for user_id in notification.target_users:
                sent_count += await self.send_to_user(user_id, message, notification.priority)

        # Target specific tenants
        elif notification.target_tenants:
            for tenant_id in notification.target_tenants:
                sent_count += await self.send_to_tenant(tenant_id, message, notification.priority)

        # Send to channel subscribers
        else:
            sent_count = await self.send_to_channel(notification.channel, message, notification.priority)

        return sent_count

    async def handle_message(self, connection_id: str, data: str) -> None:
        """Handle an incoming WebSocket message."""
        connection = self._connections.get(connection_id)
        if not connection:
            return

        self._metrics["total_messages_received"] += 1

        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            await self._send_error(connection_id, "Invalid JSON")
            return

        msg_type = message.get("type")

        if msg_type == MessageType.PING.value:
            connection.last_heartbeat = time.time()
            await self._send_to_connection(
                connection_id,
                {"type": MessageType.PONG.value, "timestamp": time.time()},
            )

        elif msg_type == MessageType.SUBSCRIBE.value:
            channel = message.get("channel")
            if channel:
                await self.subscribe(connection_id, channel)
            else:
                await self._send_error(connection_id, "Missing channel")

        elif msg_type == MessageType.UNSUBSCRIBE.value:
            channel = message.get("channel")
            if channel:
                await self.unsubscribe(connection_id, channel)
            else:
                await self._send_error(connection_id, "Missing channel")

        elif msg_type == MessageType.MESSAGE.value:
            # Handle custom message via registered handlers
            handler_name = message.get("handler")
            if handler_name and handler_name in self._message_handlers:
                try:
                    await self._message_handlers[handler_name](connection, message)
                except Exception as e:
                    logger.error(f"Message handler error: {e}")
                    await self._send_error(connection_id, str(e))

        else:
            await self._send_error(connection_id, f"Unknown message type: {msg_type}")

    def register_handler(self, name: str, handler: Callable) -> None:
        """Register a custom message handler."""
        self._message_handlers[name] = handler

    async def _send_to_connection(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Send a message to a specific connection."""
        connection = self._connections.get(connection_id)
        if not connection:
            return False

        try:
            await connection.websocket.send_json(message)
            self._metrics["total_messages_sent"] += 1
            return True
        except Exception as e:
            logger.warning(f"Failed to send to {connection_id}: {e}")
            # Connection might be dead, schedule cleanup
            asyncio.create_task(self.disconnect(connection_id))
            return False

    async def _send_error(self, connection_id: str, error: str) -> None:
        """Send an error message to a connection."""
        await self._send_to_connection(
            connection_id,
            {
                "type": MessageType.ERROR.value,
                "error": error,
                "timestamp": time.time(),
            },
        )

    async def _heartbeat_loop(self) -> None:
        """Background task to check connection health."""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                now = time.time()

                # Check for stale connections
                stale_connections = []
                for conn_id, connection in self._connections.items():
                    if now - connection.last_heartbeat > self.heartbeat_timeout:
                        stale_connections.append(conn_id)

                # Disconnect stale connections
                for conn_id in stale_connections:
                    logger.info(f"Disconnecting stale connection: {conn_id}")
                    await self.disconnect(conn_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "active_connections": len(self._connections),
            "users_connected": len(self._user_connections),
            "tenants_connected": len(self._tenant_connections),
            "channels_active": len(self._channel_subscriptions),
            **self._metrics,
        }

    def get_connection(self, connection_id: str) -> Optional[WebSocketConnection]:
        """Get a connection by ID."""
        return self._connections.get(connection_id)

    def get_user_connections(self, user_id: str) -> List[str]:
        """Get all connection IDs for a user."""
        return list(self._user_connections.get(user_id, set()))

    def get_tenant_connections(self, tenant_id: str) -> List[str]:
        """Get all connection IDs for a tenant."""
        return list(self._tenant_connections.get(tenant_id, set()))


# Global manager instance
_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """Get the global WebSocket manager instance."""
    global _manager
    if _manager is None:
        _manager = WebSocketManager()
    return _manager
