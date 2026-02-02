"""Event Bus Implementation.

Provides an event-driven architecture abstraction.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar

from src.core.messaging.producer import Message, MessageProducer, get_producer

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="Event")


class EventType(str, Enum):
    """Standard event types."""

    # Document events
    DOCUMENT_CREATED = "document.created"
    DOCUMENT_UPDATED = "document.updated"
    DOCUMENT_DELETED = "document.deleted"
    DOCUMENT_PROCESSED = "document.processed"

    # Model events
    MODEL_TRAINED = "model.trained"
    MODEL_DEPLOYED = "model.deployed"
    MODEL_PREDICTION = "model.prediction"

    # User events
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"

    # System events
    SYSTEM_STARTED = "system.started"
    SYSTEM_STOPPED = "system.stopped"
    SYSTEM_ERROR = "system.error"
    SYSTEM_ALERT = "system.alert"

    # Job events
    JOB_STARTED = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    JOB_PROGRESS = "job.progress"


@dataclass
class Event:
    """Base event class."""

    event_type: str
    data: Dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "cad-ml-platform"
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=data["event_type"],
            data=data.get("data", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
            source=data.get("source", "unknown"),
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
            user_id=data.get("user_id"),
            tenant_id=data.get("tenant_id"),
            metadata=data.get("metadata", {}),
        )


EventHandler = Callable[[Event], Any]


class EventBus:
    """Event bus for publishing and subscribing to events."""

    def __init__(
        self,
        producer: Optional[MessageProducer] = None,
        topic_prefix: str = "events",
    ):
        self._producer = producer
        self._topic_prefix = topic_prefix
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._middleware: List[Callable[[Event], Optional[Event]]] = []
        self._dead_letter_handlers: List[Callable[[Event, Exception], None]] = []

    @property
    def producer(self) -> MessageProducer:
        if self._producer is None:
            self._producer = get_producer()
        return self._producer

    def _get_topic(self, event_type: str) -> str:
        """Get topic for event type."""
        return f"{self._topic_prefix}.{event_type}"

    def use(self, middleware: Callable[[Event], Optional[Event]]) -> None:
        """Add middleware to process events before handling.

        Args:
            middleware: Function that transforms or filters events
                       Return None to skip the event
        """
        self._middleware.append(middleware)

    def on_dead_letter(self, handler: Callable[[Event, Exception], None]) -> None:
        """Register handler for failed events."""
        self._dead_letter_handlers.append(handler)

    def subscribe(
        self,
        event_type: str,
        handler: EventHandler,
    ) -> None:
        """Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Function to handle the event
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"Subscribed to event: {event_type}")

    def unsubscribe(
        self,
        event_type: str,
        handler: Optional[EventHandler] = None,
    ) -> None:
        """Unsubscribe from an event type."""
        if event_type in self._handlers:
            if handler:
                self._handlers[event_type] = [
                    h for h in self._handlers[event_type] if h != handler
                ]
            else:
                del self._handlers[event_type]

    async def publish(self, event: Event) -> bool:
        """Publish an event.

        Args:
            event: Event to publish

        Returns:
            True if published successfully
        """
        # Apply middleware
        processed_event = event
        for middleware in self._middleware:
            try:
                result = middleware(processed_event)
                if result is None:
                    logger.debug(f"Event {event.event_id} filtered by middleware")
                    return False
                processed_event = result
            except Exception as e:
                logger.error(f"Middleware error: {e}")

        # Send to message queue
        message = Message(
            topic=self._get_topic(processed_event.event_type),
            payload=processed_event.to_dict(),
            key=processed_event.tenant_id or processed_event.user_id,
            headers={
                "event-type": processed_event.event_type,
                "correlation-id": processed_event.correlation_id or "",
                "source": processed_event.source,
            },
        )

        result = await self.producer.send(message)

        if result.success:
            logger.debug(f"Published event: {processed_event.event_type}")

            # Also dispatch to local handlers
            await self._dispatch_local(processed_event)

        return result.success

    async def _dispatch_local(self, event: Event) -> None:
        """Dispatch event to local handlers."""
        handlers = self._handlers.get(event.event_type, [])
        # Also get wildcard handlers
        handlers.extend(self._handlers.get("*", []))

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
                # Send to dead letter handlers
                for dlh in self._dead_letter_handlers:
                    try:
                        if asyncio.iscoroutinefunction(dlh):
                            await dlh(event, e)
                        else:
                            dlh(event, e)
                    except Exception as dlh_error:
                        logger.error(f"Dead letter handler error: {dlh_error}")

    async def publish_batch(self, events: List[Event]) -> List[bool]:
        """Publish multiple events."""
        results = []
        for event in events:
            result = await self.publish(event)
            results.append(result)
        return results

    def on(self, event_type: str) -> Callable[[EventHandler], EventHandler]:
        """Decorator to subscribe to an event type.

        Example:
            @event_bus.on("document.created")
            async def handle_document(event: Event):
                ...
        """
        def decorator(handler: EventHandler) -> EventHandler:
            self.subscribe(event_type, handler)
            return handler
        return decorator


# Convenience event factory functions
def document_event(
    event_type: EventType,
    document_id: str,
    **kwargs: Any,
) -> Event:
    """Create a document event."""
    return Event(
        event_type=event_type.value,
        data={"document_id": document_id, **kwargs},
    )


def model_event(
    event_type: EventType,
    model_id: str,
    **kwargs: Any,
) -> Event:
    """Create a model event."""
    return Event(
        event_type=event_type.value,
        data={"model_id": model_id, **kwargs},
    )


def user_event(
    event_type: EventType,
    user_id: str,
    **kwargs: Any,
) -> Event:
    """Create a user event."""
    return Event(
        event_type=event_type.value,
        data={"user_id": user_id, **kwargs},
        user_id=user_id,
    )


def job_event(
    event_type: EventType,
    job_id: str,
    progress: Optional[float] = None,
    **kwargs: Any,
) -> Event:
    """Create a job event."""
    data = {"job_id": job_id, **kwargs}
    if progress is not None:
        data["progress"] = progress
    return Event(
        event_type=event_type.value,
        data=data,
    )


# Global event bus
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get global event bus."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def set_event_bus(event_bus: EventBus) -> None:
    """Set global event bus."""
    global _event_bus
    _event_bus = event_bus
