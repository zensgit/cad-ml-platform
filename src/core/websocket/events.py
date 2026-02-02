"""WebSocket Event System.

Provides event handling for WebSocket messages.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="WebSocketEvent")


class EventType(str, Enum):
    """Standard WebSocket event types."""

    # Connection events
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    RECONNECT = "reconnect"

    # Room events
    JOIN = "join"
    LEAVE = "leave"

    # Message events
    MESSAGE = "message"
    BROADCAST = "broadcast"
    NOTIFICATION = "notification"

    # Document events
    DOCUMENT_CREATED = "document.created"
    DOCUMENT_UPDATED = "document.updated"
    DOCUMENT_DELETED = "document.deleted"
    DOCUMENT_PROCESSED = "document.processed"

    # Job events
    JOB_STARTED = "job.started"
    JOB_PROGRESS = "job.progress"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"

    # Model events
    MODEL_TRAINED = "model.trained"
    MODEL_DEPLOYED = "model.deployed"

    # System events
    SYSTEM_ALERT = "system.alert"
    SYSTEM_MAINTENANCE = "system.maintenance"

    # Control events
    PING = "ping"
    PONG = "pong"
    ACK = "ack"
    ERROR = "error"


@dataclass
class WebSocketEvent:
    """Base WebSocket event."""

    event_type: str
    data: Any = None
    event_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Source info
    connection_id: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.event_type,
            "data": self.data,
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create from dictionary."""
        return cls(
            event_type=data.get("type", "unknown"),
            data=data.get("data"),
            event_id=data.get("event_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


EventHandler = Callable[[WebSocketEvent], Any]


@dataclass
class EventSubscription:
    """Subscription to an event type."""

    event_type: str
    handler: EventHandler
    filters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    once: bool = False

    def matches(self, event: WebSocketEvent) -> bool:
        """Check if event matches subscription filters."""
        if self.filters:
            for key, value in self.filters.items():
                event_value = getattr(event, key, None) or event.metadata.get(key)
                if event_value != value:
                    return False
        return True


class EventDispatcher:
    """Dispatcher for WebSocket events.

    Provides pub/sub pattern for event handling.
    """

    def __init__(self):
        self._handlers: Dict[str, List[EventSubscription]] = {}
        self._wildcard_handlers: List[EventSubscription] = []
        self._middleware: List[Callable[[WebSocketEvent], Optional[WebSocketEvent]]] = []
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def use(self, middleware: Callable[[WebSocketEvent], Optional[WebSocketEvent]]) -> None:
        """Add middleware for event processing.

        Args:
            middleware: Function that transforms or filters events
        """
        self._middleware.append(middleware)

    def on(
        self,
        event_type: str,
        handler: EventHandler,
        filters: Optional[Dict[str, Any]] = None,
        priority: int = 0,
    ) -> EventSubscription:
        """Subscribe to an event type.

        Args:
            event_type: Event type to subscribe to ("*" for all)
            handler: Handler function
            filters: Optional filters
            priority: Handler priority (higher = earlier)

        Returns:
            EventSubscription for unsubscribing
        """
        subscription = EventSubscription(
            event_type=event_type,
            handler=handler,
            filters=filters or {},
            priority=priority,
        )

        if event_type == "*":
            self._wildcard_handlers.append(subscription)
            self._wildcard_handlers.sort(key=lambda s: -s.priority)
        else:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(subscription)
            self._handlers[event_type].sort(key=lambda s: -s.priority)

        return subscription

    def once(
        self,
        event_type: str,
        handler: EventHandler,
    ) -> EventSubscription:
        """Subscribe to an event type for one event only.

        Args:
            event_type: Event type
            handler: Handler function

        Returns:
            EventSubscription
        """
        subscription = EventSubscription(
            event_type=event_type,
            handler=handler,
            once=True,
        )

        if event_type == "*":
            self._wildcard_handlers.append(subscription)
        else:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(subscription)

        return subscription

    def off(
        self,
        event_type: str,
        handler: Optional[EventHandler] = None,
    ) -> None:
        """Unsubscribe from an event type.

        Args:
            event_type: Event type
            handler: Specific handler to remove, or None for all
        """
        if event_type == "*":
            if handler:
                self._wildcard_handlers = [
                    s for s in self._wildcard_handlers if s.handler != handler
                ]
            else:
                self._wildcard_handlers.clear()
        elif event_type in self._handlers:
            if handler:
                self._handlers[event_type] = [
                    s for s in self._handlers[event_type] if s.handler != handler
                ]
            else:
                del self._handlers[event_type]

    async def emit(self, event: WebSocketEvent) -> int:
        """Emit an event to all matching handlers.

        Args:
            event: Event to emit

        Returns:
            Number of handlers called
        """
        # Apply middleware
        processed_event = event
        for middleware in self._middleware:
            try:
                result = middleware(processed_event)
                if result is None:
                    logger.debug(f"Event {event.event_type} filtered by middleware")
                    return 0
                processed_event = result
            except Exception as e:
                logger.error(f"Middleware error: {e}")

        # Get matching handlers
        handlers = list(self._handlers.get(processed_event.event_type, []))
        handlers.extend(self._wildcard_handlers)

        # Track once handlers to remove
        to_remove = []
        count = 0

        for subscription in handlers:
            if not subscription.matches(processed_event):
                continue

            try:
                if asyncio.iscoroutinefunction(subscription.handler):
                    await subscription.handler(processed_event)
                else:
                    subscription.handler(processed_event)
                count += 1

                if subscription.once:
                    to_remove.append(subscription)

            except Exception as e:
                logger.error(f"Event handler error for {processed_event.event_type}: {e}")

        # Remove once handlers
        for sub in to_remove:
            self.off(sub.event_type, sub.handler)

        return count

    async def emit_and_wait(
        self,
        event: WebSocketEvent,
        timeout: float = 30.0,
    ) -> List[Any]:
        """Emit event and wait for all handler results.

        Args:
            event: Event to emit
            timeout: Timeout in seconds

        Returns:
            List of handler results
        """
        handlers = list(self._handlers.get(event.event_type, []))
        handlers.extend(self._wildcard_handlers)

        tasks = []
        for subscription in handlers:
            if not subscription.matches(event):
                continue

            if asyncio.iscoroutinefunction(subscription.handler):
                tasks.append(subscription.handler(event))
            else:
                # Wrap sync handler
                tasks.append(asyncio.get_event_loop().run_in_executor(
                    None, subscription.handler, event
                ))

        if not tasks:
            return []

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )
            return [r for r in results if not isinstance(r, Exception)]
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for handlers of {event.event_type}")
            return []

    def has_handlers(self, event_type: str) -> bool:
        """Check if there are handlers for an event type."""
        return bool(self._handlers.get(event_type) or self._wildcard_handlers)

    def handler_count(self, event_type: Optional[str] = None) -> int:
        """Get count of handlers."""
        if event_type:
            return len(self._handlers.get(event_type, [])) + len(self._wildcard_handlers)
        return sum(len(h) for h in self._handlers.values()) + len(self._wildcard_handlers)


# Global event dispatcher
_event_dispatcher: Optional[EventDispatcher] = None


def get_event_dispatcher() -> EventDispatcher:
    """Get global event dispatcher."""
    global _event_dispatcher
    if _event_dispatcher is None:
        _event_dispatcher = EventDispatcher()
    return _event_dispatcher


# ============================================================================
# Decorator for event handlers
# ============================================================================

def event_handler(
    event_type: str,
    filters: Optional[Dict[str, Any]] = None,
    priority: int = 0,
) -> Callable[[EventHandler], EventHandler]:
    """Decorator to register an event handler.

    Example:
        @event_handler("document.created")
        async def on_document_created(event: WebSocketEvent):
            print(f"Document created: {event.data}")
    """
    def decorator(func: EventHandler) -> EventHandler:
        dispatcher = get_event_dispatcher()
        dispatcher.on(event_type, func, filters=filters, priority=priority)
        return func
    return decorator
