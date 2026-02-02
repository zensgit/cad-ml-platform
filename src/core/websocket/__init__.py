"""WebSocket module for real-time communication.

Provides:
- Connection management
- Room-based messaging
- Event broadcasting
- Pub/Sub integration
"""

from src.core.websocket.manager import (
    WebSocketManager,
    WebSocketConnection,
    MessageType,
    get_websocket_manager,
)
from src.core.websocket.rooms import (
    Room,
    RoomManager,
    get_room_manager,
)
from src.core.websocket.events import (
    WebSocketEvent,
    EventType,
    EventHandler,
    EventDispatcher,
    get_event_dispatcher,
)
from src.core.websocket.pubsub import (
    PubSubBackend,
    RedisPubSub,
    InMemoryPubSub,
    get_pubsub,
)

__all__ = [
    # Manager
    "WebSocketManager",
    "WebSocketConnection",
    "MessageType",
    "get_websocket_manager",
    # Rooms
    "Room",
    "RoomManager",
    "get_room_manager",
    # Events
    "WebSocketEvent",
    "EventType",
    "EventHandler",
    "EventDispatcher",
    "get_event_dispatcher",
    # PubSub
    "PubSubBackend",
    "RedisPubSub",
    "InMemoryPubSub",
    "get_pubsub",
]
