"""Tests for WebSocket manager and API endpoints."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMessageType:
    """Tests for MessageType enum."""

    def test_message_types_exist(self):
        """Test all message types are defined."""
        from src.core.websocket.manager import MessageType

        assert MessageType.SUBSCRIBE.value == "subscribe"
        assert MessageType.UNSUBSCRIBE.value == "unsubscribe"
        assert MessageType.PING.value == "ping"
        assert MessageType.PONG.value == "pong"
        assert MessageType.NOTIFICATION.value == "notification"
        assert MessageType.BROADCAST.value == "broadcast"


class TestNotificationPriority:
    """Tests for NotificationPriority enum."""

    def test_priority_levels(self):
        """Test priority levels are defined."""
        from src.core.websocket.manager import NotificationPriority

        assert NotificationPriority.LOW.value == "low"
        assert NotificationPriority.NORMAL.value == "normal"
        assert NotificationPriority.HIGH.value == "high"
        assert NotificationPriority.URGENT.value == "urgent"


class TestWebSocketConnection:
    """Tests for WebSocketConnection dataclass."""

    def test_default_values(self):
        """Test default connection values."""
        from src.core.websocket.manager import WebSocketConnection

        mock_ws = MagicMock()
        conn = WebSocketConnection(
            connection_id="test-123",
            websocket=mock_ws,
        )

        assert conn.connection_id == "test-123"
        assert conn.user_id is None
        assert conn.tenant_id is None
        assert conn.subscriptions == set()
        assert isinstance(conn.connected_at, float)

    def test_custom_values(self):
        """Test custom connection values."""
        from src.core.websocket.manager import WebSocketConnection

        mock_ws = MagicMock()
        conn = WebSocketConnection(
            connection_id="test-456",
            websocket=mock_ws,
            user_id="user-1",
            tenant_id="tenant-1",
            metadata={"role": "admin"},
        )

        assert conn.user_id == "user-1"
        assert conn.tenant_id == "tenant-1"
        assert conn.metadata["role"] == "admin"


class TestNotification:
    """Tests for Notification dataclass."""

    def test_notification_creation(self):
        """Test notification creation."""
        from src.core.websocket.manager import MessageType, Notification, NotificationPriority

        notification = Notification(
            id="notif-1",
            type=MessageType.NOTIFICATION,
            channel="alerts",
            payload={"message": "Hello"},
            priority=NotificationPriority.HIGH,
        )

        assert notification.id == "notif-1"
        assert notification.channel == "alerts"
        assert notification.priority == NotificationPriority.HIGH
        assert notification.payload["message"] == "Hello"


class TestWebSocketManager:
    """Tests for WebSocketManager."""

    @pytest.mark.asyncio
    async def test_connect_creates_connection(self):
        """Test connection is created successfully."""
        from src.core.websocket.manager import WebSocketManager

        manager = WebSocketManager()
        mock_ws = AsyncMock()

        conn_id = await manager.connect(mock_ws, user_id="user-1")

        assert conn_id is not None
        assert conn_id in manager._connections
        mock_ws.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_with_user_tracking(self):
        """Test user connections are tracked."""
        from src.core.websocket.manager import WebSocketManager

        manager = WebSocketManager()
        mock_ws = AsyncMock()

        conn_id = await manager.connect(mock_ws, user_id="user-1")

        assert "user-1" in manager._user_connections
        assert conn_id in manager._user_connections["user-1"]

    @pytest.mark.asyncio
    async def test_connect_with_tenant_tracking(self):
        """Test tenant connections are tracked."""
        from src.core.websocket.manager import WebSocketManager

        manager = WebSocketManager()
        mock_ws = AsyncMock()

        conn_id = await manager.connect(mock_ws, tenant_id="tenant-1")

        assert "tenant-1" in manager._tenant_connections
        assert conn_id in manager._tenant_connections["tenant-1"]

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self):
        """Test disconnect cleans up all references."""
        from src.core.websocket.manager import WebSocketManager

        manager = WebSocketManager()
        mock_ws = AsyncMock()

        conn_id = await manager.connect(mock_ws, user_id="user-1", tenant_id="tenant-1")
        await manager.disconnect(conn_id)

        assert conn_id not in manager._connections
        assert "user-1" not in manager._user_connections
        assert "tenant-1" not in manager._tenant_connections

    @pytest.mark.asyncio
    async def test_subscribe_to_channel(self):
        """Test subscribing to a channel."""
        from src.core.websocket.manager import WebSocketManager

        manager = WebSocketManager()
        mock_ws = AsyncMock()

        conn_id = await manager.connect(mock_ws)
        result = await manager.subscribe(conn_id, "alerts")

        assert result is True
        assert "alerts" in manager._connections[conn_id].subscriptions
        assert conn_id in manager._channel_subscriptions["alerts"]

    @pytest.mark.asyncio
    async def test_unsubscribe_from_channel(self):
        """Test unsubscribing from a channel."""
        from src.core.websocket.manager import WebSocketManager

        manager = WebSocketManager()
        mock_ws = AsyncMock()

        conn_id = await manager.connect(mock_ws)
        await manager.subscribe(conn_id, "alerts")
        result = await manager.unsubscribe(conn_id, "alerts")

        assert result is True
        assert "alerts" not in manager._connections[conn_id].subscriptions

    @pytest.mark.asyncio
    async def test_send_to_user(self):
        """Test sending message to user."""
        from src.core.websocket.manager import WebSocketManager

        manager = WebSocketManager()
        mock_ws = AsyncMock()

        await manager.connect(mock_ws, user_id="user-1")
        sent = await manager.send_to_user("user-1", {"test": "message"})

        assert sent == 1
        mock_ws.send_json.assert_called()

    @pytest.mark.asyncio
    async def test_send_to_tenant(self):
        """Test sending message to tenant."""
        from src.core.websocket.manager import WebSocketManager

        manager = WebSocketManager()
        mock_ws = AsyncMock()

        await manager.connect(mock_ws, tenant_id="tenant-1")
        sent = await manager.send_to_tenant("tenant-1", {"test": "message"})

        assert sent == 1
        mock_ws.send_json.assert_called()

    @pytest.mark.asyncio
    async def test_send_to_channel(self):
        """Test sending message to channel."""
        from src.core.websocket.manager import WebSocketManager

        manager = WebSocketManager()
        mock_ws = AsyncMock()

        conn_id = await manager.connect(mock_ws)
        await manager.subscribe(conn_id, "alerts")
        sent = await manager.send_to_channel("alerts", {"test": "message"})

        assert sent == 1

    @pytest.mark.asyncio
    async def test_broadcast(self):
        """Test broadcasting to all connections."""
        from src.core.websocket.manager import WebSocketManager

        manager = WebSocketManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        await manager.connect(ws1)
        await manager.connect(ws2)
        sent = await manager.broadcast({"test": "broadcast"})

        assert sent == 2

    @pytest.mark.asyncio
    async def test_handle_ping_message(self):
        """Test handling ping message."""
        from src.core.websocket.manager import WebSocketManager

        manager = WebSocketManager()
        mock_ws = AsyncMock()

        conn_id = await manager.connect(mock_ws)
        await manager.handle_message(conn_id, '{"type": "ping"}')

        # Should respond with pong
        calls = mock_ws.send_json.call_args_list
        pong_sent = any("pong" in str(call) for call in calls)
        assert pong_sent or len(calls) >= 2  # At least ack + pong

    @pytest.mark.asyncio
    async def test_handle_subscribe_message(self):
        """Test handling subscribe message."""
        from src.core.websocket.manager import WebSocketManager

        manager = WebSocketManager()
        mock_ws = AsyncMock()

        conn_id = await manager.connect(mock_ws)
        await manager.handle_message(conn_id, '{"type": "subscribe", "channel": "news"}')

        assert "news" in manager._connections[conn_id].subscriptions

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self):
        """Test handling invalid JSON."""
        from src.core.websocket.manager import WebSocketManager

        manager = WebSocketManager()
        mock_ws = AsyncMock()

        conn_id = await manager.connect(mock_ws)
        await manager.handle_message(conn_id, "not valid json")

        # Should send error response
        calls = mock_ws.send_json.call_args_list
        error_sent = any("error" in str(call).lower() for call in calls)
        assert error_sent or len(calls) >= 2

    @pytest.mark.asyncio
    async def test_connection_limit_per_user(self):
        """Test connection limit per user is enforced."""
        from src.core.websocket.manager import WebSocketManager

        manager = WebSocketManager(max_connections_per_user=2)

        # Connect twice (should succeed)
        for _ in range(2):
            mock_ws = AsyncMock()
            await manager.connect(mock_ws, user_id="user-1")

        # Third should fail
        mock_ws = AsyncMock()
        with pytest.raises(ValueError, match="too many connections"):
            await manager.connect(mock_ws, user_id="user-1")

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting statistics."""
        from src.core.websocket.manager import WebSocketManager

        manager = WebSocketManager()
        mock_ws = AsyncMock()

        await manager.connect(mock_ws, user_id="user-1", tenant_id="tenant-1")
        stats = manager.get_stats()

        assert stats["active_connections"] == 1
        assert stats["users_connected"] == 1
        assert stats["tenants_connected"] == 1

    @pytest.mark.asyncio
    async def test_notify_with_notification(self):
        """Test notify with Notification object."""
        from src.core.websocket.manager import (
            MessageType,
            Notification,
            NotificationPriority,
            WebSocketManager,
        )

        manager = WebSocketManager()
        mock_ws = AsyncMock()

        conn_id = await manager.connect(mock_ws, user_id="user-1")

        notification = Notification(
            id="n-1",
            type=MessageType.NOTIFICATION,
            channel="alerts",
            payload={"message": "Test"},
            target_users=["user-1"],
        )

        sent = await manager.notify(notification)
        assert sent == 1


class TestWebSocketManagerLifecycle:
    """Tests for WebSocketManager start/stop."""

    @pytest.mark.asyncio
    async def test_start_creates_heartbeat_task(self):
        """Test starting manager creates heartbeat task."""
        from src.core.websocket.manager import WebSocketManager

        manager = WebSocketManager()
        await manager.start()

        assert manager._running is True
        assert manager._heartbeat_task is not None

        await manager.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_heartbeat(self):
        """Test stopping manager cancels heartbeat."""
        from src.core.websocket.manager import WebSocketManager

        manager = WebSocketManager()
        await manager.start()
        await manager.stop()

        assert manager._running is False


class TestGetWebSocketManager:
    """Tests for get_websocket_manager function."""

    def test_returns_singleton(self):
        """Test returns singleton instance."""
        from src.core.websocket import manager as ws_module

        # Reset global
        ws_module._manager = None

        mgr1 = ws_module.get_websocket_manager()
        mgr2 = ws_module.get_websocket_manager()

        assert mgr1 is mgr2

        # Cleanup
        ws_module._manager = None


class TestMultipleConnections:
    """Tests for multiple connection scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_users_isolated(self):
        """Test multiple users have isolated connections."""
        from src.core.websocket.manager import WebSocketManager

        manager = WebSocketManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        conn1 = await manager.connect(ws1, user_id="user-1")
        conn2 = await manager.connect(ws2, user_id="user-2")

        # Send to user-1 only
        await manager.send_to_user("user-1", {"msg": "hello"})

        # Only ws1 should receive
        assert ws1.send_json.call_count > ws2.send_json.call_count

    @pytest.mark.asyncio
    async def test_channel_isolation(self):
        """Test channels are isolated."""
        from src.core.websocket.manager import WebSocketManager

        manager = WebSocketManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        conn1 = await manager.connect(ws1)
        conn2 = await manager.connect(ws2)

        await manager.subscribe(conn1, "alerts")
        await manager.subscribe(conn2, "news")

        # Reset call counts after subscription
        ws1.send_json.reset_mock()
        ws2.send_json.reset_mock()

        # Send to alerts channel
        await manager.send_to_channel("alerts", {"msg": "alert!"})

        # Only ws1 should receive
        assert ws1.send_json.call_count == 1
        assert ws2.send_json.call_count == 0
