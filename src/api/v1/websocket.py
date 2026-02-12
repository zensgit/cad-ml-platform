"""WebSocket API endpoints for real-time notifications.

Provides endpoints for:
- Connection management
- Channel subscriptions
- Real-time notifications
- Connection statistics
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from src.api.dependencies import get_api_key
from src.core.websocket.manager import (
    MessageType,
    Notification,
    NotificationPriority,
    get_websocket_manager,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["websocket"])


class NotificationRequest(BaseModel):
    """Request to send a notification."""

    channel: str = Field(..., description="Target channel")
    payload: Dict[str, Any] = Field(..., description="Notification payload")
    priority: str = Field("normal", description="Priority level")
    target_users: Optional[list[str]] = Field(None, description="Target user IDs")
    target_tenants: Optional[list[str]] = Field(None, description="Target tenant IDs")


class NotificationResponse(BaseModel):
    """Response from sending a notification."""

    notification_id: str
    sent_count: int
    channel: str


class WebSocketStatsResponse(BaseModel):
    """WebSocket statistics response."""

    active_connections: int
    users_connected: int
    tenants_connected: int
    channels_active: int
    total_connections: int
    total_messages_sent: int
    total_messages_received: int
    total_broadcasts: int


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: Optional[str] = Query(None),
    tenant_id: Optional[str] = Query(None),
) -> None:
    """Main WebSocket endpoint for real-time notifications.

    Query parameters:
    - user_id: Optional user identifier
    - tenant_id: Optional tenant identifier

    Message protocol:
    - Subscribe: {"type": "subscribe", "channel": "channel_name"}
    - Unsubscribe: {"type": "unsubscribe", "channel": "channel_name"}
    - Ping: {"type": "ping"}
    """
    manager = get_websocket_manager()

    try:
        connection_id = await manager.connect(
            websocket=websocket,
            user_id=user_id,
            tenant_id=tenant_id,
        )

        # Main message loop
        while True:
            data = await websocket.receive_text()
            await manager.handle_message(connection_id, data)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: user={user_id}, tenant={tenant_id}")
    except ValueError as e:
        # Connection limit exceeded
        logger.warning(f"WebSocket connection rejected: {e}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Ensure cleanup
        if "connection_id" in locals():
            await manager.disconnect(connection_id)


@router.websocket("/ws/notifications/{channel}")
async def channel_websocket_endpoint(
    websocket: WebSocket,
    channel: str,
    user_id: Optional[str] = Query(None),
    tenant_id: Optional[str] = Query(None),
) -> None:
    """WebSocket endpoint that auto-subscribes to a specific channel.

    Path parameters:
    - channel: Channel to auto-subscribe to
    """
    manager = get_websocket_manager()

    try:
        connection_id = await manager.connect(
            websocket=websocket,
            user_id=user_id,
            tenant_id=tenant_id,
        )

        # Auto-subscribe to the channel
        await manager.subscribe(connection_id, channel)

        # Main message loop
        while True:
            data = await websocket.receive_text()
            await manager.handle_message(connection_id, data)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected from channel {channel}")
    except ValueError as e:
        logger.warning(f"WebSocket connection rejected: {e}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if "connection_id" in locals():
            await manager.disconnect(connection_id)


@router.post("/notifications/send", response_model=NotificationResponse)
async def send_notification(
    request: NotificationRequest,
    api_key: str = Depends(get_api_key),
) -> NotificationResponse:
    """Send a notification to a channel or specific targets.

    Requires API key authentication.
    """
    manager = get_websocket_manager()

    # Map priority string to enum
    priority_map = {
        "low": NotificationPriority.LOW,
        "normal": NotificationPriority.NORMAL,
        "high": NotificationPriority.HIGH,
        "urgent": NotificationPriority.URGENT,
    }
    priority = priority_map.get(request.priority.lower(), NotificationPriority.NORMAL)

    # Create notification
    import uuid

    notification = Notification(
        id=str(uuid.uuid4()),
        type=MessageType.NOTIFICATION,
        channel=request.channel,
        payload=request.payload,
        priority=priority,
        target_users=request.target_users,
        target_tenants=request.target_tenants,
    )

    # Send notification
    sent_count = await manager.notify(notification)

    return NotificationResponse(
        notification_id=notification.id,
        sent_count=sent_count,
        channel=request.channel,
    )


@router.post("/notifications/broadcast")
async def broadcast_notification(
    payload: Dict[str, Any],
    api_key: str = Depends(get_api_key),
    x_admin_token: str = Header(default="", alias="X-Admin-Token"),
) -> Dict[str, Any]:
    """Broadcast a message to all connected clients.

    Requires admin token.
    """
    import os

    expected_token = os.getenv("ADMIN_TOKEN", "test")
    if x_admin_token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid admin token")

    manager = get_websocket_manager()
    sent_count = await manager.broadcast({"payload": payload})

    return {
        "status": "broadcast_sent",
        "sent_count": sent_count,
    }


@router.get("/notifications/stats", response_model=WebSocketStatsResponse)
async def get_websocket_stats(
    api_key: str = Depends(get_api_key),
) -> WebSocketStatsResponse:
    """Get WebSocket connection statistics."""
    manager = get_websocket_manager()
    stats = manager.get_stats()

    return WebSocketStatsResponse(**stats)


@router.get("/notifications/connections")
async def list_connections(
    user_id: Optional[str] = Query(None),
    tenant_id: Optional[str] = Query(None),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """List active WebSocket connections.

    Optionally filter by user_id or tenant_id.
    """
    manager = get_websocket_manager()

    if user_id:
        connection_ids = manager.get_user_connections(user_id)
    elif tenant_id:
        connection_ids = manager.get_tenant_connections(tenant_id)
    else:
        # Return all (admin only in production)
        connection_ids = list(manager._connections.keys())

    connections = []
    for conn_id in connection_ids:
        conn = manager.get_connection(conn_id)
        if conn:
            connections.append({
                "connection_id": conn.connection_id,
                "user_id": conn.user_id,
                "tenant_id": conn.tenant_id,
                "connected_at": conn.connected_at,
                "subscriptions": list(conn.subscriptions),
            })

    return {
        "total": len(connections),
        "connections": connections,
    }
