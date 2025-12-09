"""
Event Bus Module for Vision Provider System.

Implements event-driven architecture patterns including event bus,
event routing, event sourcing, and CQRS patterns for vision operations.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

from .base import VisionProvider, VisionDescription

T = TypeVar("T")
E = TypeVar("E", bound="Event")


class EventPriority(Enum):
    """Priority levels for events."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class EventStatus(Enum):
    """Status of an event."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class Event:
    """Base event class."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    version: int = 1

    def __post_init__(self) -> None:
        if not self.event_type:
            self.event_type = self.__class__.__name__


@dataclass
class VisionEvent(Event):
    """Vision-specific event."""

    provider_name: str = ""
    image_hash: str = ""
    confidence: float = 0.0


@dataclass
class ImageAnalyzedEvent(VisionEvent):
    """Event emitted when an image is analyzed."""

    summary: str = ""
    details: List[str] = field(default_factory=list)


@dataclass
class AnalysisFailedEvent(VisionEvent):
    """Event emitted when analysis fails."""

    error_message: str = ""
    error_type: str = ""


@dataclass
class EventEnvelope:
    """Wrapper for events with delivery metadata."""

    event: Event
    status: EventStatus = EventStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    error: Optional[str] = None


class EventHandler(ABC):
    """Abstract base class for event handlers."""

    @abstractmethod
    async def handle(self, event: Event) -> None:
        """Handle an event."""
        pass

    def can_handle(self, event: Event) -> bool:
        """Check if handler can process this event type."""
        return True


class FunctionEventHandler(EventHandler):
    """Event handler that wraps a function."""

    def __init__(
        self,
        func: Callable[[Event], Any],
        event_types: Optional[List[str]] = None,
    ):
        self._func = func
        self._event_types = set(event_types) if event_types else None

    async def handle(self, event: Event) -> None:
        """Handle the event using the wrapped function."""
        if asyncio.iscoroutinefunction(self._func):
            await self._func(event)
        else:
            self._func(event)

    def can_handle(self, event: Event) -> bool:
        """Check if this handler handles the event type."""
        if self._event_types is None:
            return True
        return event.event_type in self._event_types


class EventBus:
    """Central event bus for event-driven architecture."""

    def __init__(self):
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._global_handlers: List[EventHandler] = []
        self._event_history: List[EventEnvelope] = []
        self._max_history: int = 1000
        self._async_mode: bool = True

    def subscribe(
        self,
        event_type: str,
        handler: Union[EventHandler, Callable[[Event], Any]],
    ) -> None:
        """Subscribe a handler to an event type."""
        if callable(handler) and not isinstance(handler, EventHandler):
            handler = FunctionEventHandler(handler, [event_type])

        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def subscribe_all(
        self, handler: Union[EventHandler, Callable[[Event], Any]]
    ) -> None:
        """Subscribe a handler to all events."""
        if callable(handler) and not isinstance(handler, EventHandler):
            handler = FunctionEventHandler(handler)
        self._global_handlers.append(handler)

    def unsubscribe(
        self, event_type: str, handler: EventHandler
    ) -> bool:
        """Unsubscribe a handler from an event type."""
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
                return True
            except ValueError:
                pass
        return False

    async def publish(self, event: Event) -> EventEnvelope:
        """Publish an event to all subscribers."""
        envelope = EventEnvelope(event=event)

        try:
            envelope.status = EventStatus.PROCESSING

            # Get type-specific handlers
            handlers = self._handlers.get(event.event_type, [])

            # Add global handlers
            all_handlers = handlers + self._global_handlers

            # Execute handlers
            for handler in all_handlers:
                if handler.can_handle(event):
                    try:
                        await handler.handle(event)
                    except Exception as e:
                        envelope.error = str(e)

            envelope.status = EventStatus.COMPLETED
            envelope.processed_at = datetime.utcnow()

        except Exception as e:
            envelope.status = EventStatus.FAILED
            envelope.error = str(e)

        self._add_to_history(envelope)
        return envelope

    def publish_sync(self, event: Event) -> EventEnvelope:
        """Publish an event synchronously."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.publish(event))
        finally:
            loop.close()

    def _add_to_history(self, envelope: EventEnvelope) -> None:
        """Add envelope to history with size limit."""
        self._event_history.append(envelope)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history :]

    def get_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[EventEnvelope]:
        """Get event history."""
        history = self._event_history
        if event_type:
            history = [
                e for e in history if e.event.event_type == event_type
            ]
        return history[-limit:]

    def get_handler_count(self, event_type: Optional[str] = None) -> int:
        """Get number of handlers for an event type."""
        if event_type:
            return len(self._handlers.get(event_type, []))
        total = sum(len(h) for h in self._handlers.values())
        return total + len(self._global_handlers)


class EventRouter:
    """Routes events based on rules and filters."""

    def __init__(self):
        self._routes: List[Dict[str, Any]] = []
        self._default_bus: Optional[EventBus] = None

    def set_default_bus(self, bus: EventBus) -> None:
        """Set the default event bus."""
        self._default_bus = bus

    def add_route(
        self,
        name: str,
        predicate: Callable[[Event], bool],
        target: EventBus,
        priority: int = 0,
    ) -> None:
        """Add a routing rule."""
        self._routes.append({
            "name": name,
            "predicate": predicate,
            "target": target,
            "priority": priority,
        })
        self._routes.sort(key=lambda r: r["priority"], reverse=True)

    def remove_route(self, name: str) -> bool:
        """Remove a routing rule."""
        for i, route in enumerate(self._routes):
            if route["name"] == name:
                self._routes.pop(i)
                return True
        return False

    async def route(self, event: Event) -> List[EventEnvelope]:
        """Route an event to matching buses."""
        results: List[EventEnvelope] = []
        routed = False

        for route in self._routes:
            if route["predicate"](event):
                envelope = await route["target"].publish(event)
                results.append(envelope)
                routed = True

        # Use default bus if no routes matched
        if not routed and self._default_bus:
            envelope = await self._default_bus.publish(event)
            results.append(envelope)

        return results

    def get_routes(self) -> List[str]:
        """Get all route names."""
        return [r["name"] for r in self._routes]


class EventStore:
    """Persistent event store for event sourcing."""

    def __init__(self):
        self._events: Dict[str, List[Event]] = {}  # aggregate_id -> events
        self._all_events: List[Event] = []
        self._snapshots: Dict[str, Dict[str, Any]] = {}

    def append(self, aggregate_id: str, event: Event) -> None:
        """Append an event to an aggregate's stream."""
        if aggregate_id not in self._events:
            self._events[aggregate_id] = []

        # Set causation chain
        events = self._events[aggregate_id]
        if events:
            event.causation_id = events[-1].event_id

        event.version = len(events) + 1
        self._events[aggregate_id].append(event)
        self._all_events.append(event)

    def get_events(
        self,
        aggregate_id: str,
        from_version: int = 0,
    ) -> List[Event]:
        """Get events for an aggregate from a specific version."""
        events = self._events.get(aggregate_id, [])
        return [e for e in events if e.version > from_version]

    def get_all_events(
        self,
        from_timestamp: Optional[datetime] = None,
    ) -> List[Event]:
        """Get all events, optionally from a timestamp."""
        if from_timestamp:
            return [
                e for e in self._all_events if e.timestamp >= from_timestamp
            ]
        return self._all_events.copy()

    def save_snapshot(
        self, aggregate_id: str, state: Dict[str, Any], version: int
    ) -> None:
        """Save a snapshot of aggregate state."""
        self._snapshots[aggregate_id] = {
            "state": state,
            "version": version,
            "timestamp": datetime.utcnow(),
        }

    def get_snapshot(
        self, aggregate_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get the latest snapshot for an aggregate."""
        return self._snapshots.get(aggregate_id)

    def get_aggregate_ids(self) -> List[str]:
        """Get all aggregate IDs."""
        return list(self._events.keys())

    def count_events(self, aggregate_id: Optional[str] = None) -> int:
        """Count events."""
        if aggregate_id:
            return len(self._events.get(aggregate_id, []))
        return len(self._all_events)


class EventAggregator:
    """Aggregates events from multiple sources."""

    def __init__(self):
        self._sources: Dict[str, EventBus] = {}
        self._aggregated_bus = EventBus()

    def add_source(self, name: str, bus: EventBus) -> None:
        """Add an event source."""
        self._sources[name] = bus

        # Forward events to aggregated bus
        async def forward(event: Event) -> None:
            event.metadata["source"] = name
            await self._aggregated_bus.publish(event)

        bus.subscribe_all(forward)

    def remove_source(self, name: str) -> bool:
        """Remove an event source."""
        if name in self._sources:
            del self._sources[name]
            return True
        return False

    def subscribe(
        self,
        event_type: str,
        handler: Union[EventHandler, Callable[[Event], Any]],
    ) -> None:
        """Subscribe to aggregated events."""
        self._aggregated_bus.subscribe(event_type, handler)

    def subscribe_all(
        self, handler: Union[EventHandler, Callable[[Event], Any]]
    ) -> None:
        """Subscribe to all aggregated events."""
        self._aggregated_bus.subscribe_all(handler)

    def get_sources(self) -> List[str]:
        """Get all source names."""
        return list(self._sources.keys())


class CommandBus:
    """Command bus for CQRS pattern."""

    def __init__(self):
        self._handlers: Dict[str, Callable[..., Any]] = {}
        self._middleware: List[Callable[..., Any]] = []

    def register(
        self, command_type: str, handler: Callable[..., Any]
    ) -> None:
        """Register a command handler."""
        self._handlers[command_type] = handler

    def add_middleware(self, middleware: Callable[..., Any]) -> None:
        """Add middleware for command processing."""
        self._middleware.append(middleware)

    async def dispatch(
        self, command_type: str, data: Dict[str, Any]
    ) -> Any:
        """Dispatch a command to its handler."""
        handler = self._handlers.get(command_type)
        if not handler:
            raise ValueError(f"No handler for command: {command_type}")

        # Apply middleware
        context = {"command_type": command_type, "data": data}
        for mw in self._middleware:
            if asyncio.iscoroutinefunction(mw):
                await mw(context)
            else:
                mw(context)

        # Execute handler
        if asyncio.iscoroutinefunction(handler):
            return await handler(data)
        return handler(data)

    def has_handler(self, command_type: str) -> bool:
        """Check if a command type has a handler."""
        return command_type in self._handlers


class QueryBus:
    """Query bus for CQRS pattern."""

    def __init__(self):
        self._handlers: Dict[str, Callable[..., Any]] = {}
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl: float = 60.0  # seconds

    def register(
        self, query_type: str, handler: Callable[..., Any]
    ) -> None:
        """Register a query handler."""
        self._handlers[query_type] = handler

    async def query(
        self,
        query_type: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> Any:
        """Execute a query."""
        handler = self._handlers.get(query_type)
        if not handler:
            raise ValueError(f"No handler for query: {query_type}")

        # Check cache
        cache_key = f"{query_type}:{str(params)}"
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            age = (datetime.utcnow() - cached["timestamp"]).total_seconds()
            if age < self._cache_ttl:
                return cached["result"]

        # Execute query
        if asyncio.iscoroutinefunction(handler):
            result = await handler(params or {})
        else:
            result = handler(params or {})

        # Cache result
        if use_cache:
            self._cache[cache_key] = {
                "result": result,
                "timestamp": datetime.utcnow(),
            }

        return result

    def invalidate_cache(self, query_type: Optional[str] = None) -> int:
        """Invalidate cached queries."""
        if query_type:
            keys_to_remove = [
                k for k in self._cache if k.startswith(f"{query_type}:")
            ]
            for key in keys_to_remove:
                del self._cache[key]
            return len(keys_to_remove)
        else:
            count = len(self._cache)
            self._cache.clear()
            return count


class EventDrivenVisionProvider(VisionProvider):
    """Vision provider with event-driven architecture."""

    def __init__(
        self,
        base_provider: VisionProvider,
        event_bus: EventBus,
    ):
        self._base = base_provider
        self._event_bus = event_bus

    @property
    def provider_name(self) -> str:
        return f"event_{self._base.provider_name}"

    async def analyze_image(
        self, image_data: bytes, **kwargs: Any
    ) -> VisionDescription:
        """Analyze image and emit events."""
        correlation_id = str(uuid.uuid4())

        try:
            result = await self._base.analyze_image(image_data, **kwargs)

            # Emit success event
            event = ImageAnalyzedEvent(
                provider_name=self._base.provider_name,
                image_hash=str(hash(image_data)),
                confidence=result.confidence,
                summary=result.summary,
                details=result.details,
                correlation_id=correlation_id,
            )
            await self._event_bus.publish(event)

            return result

        except Exception as e:
            # Emit failure event
            event = AnalysisFailedEvent(
                provider_name=self._base.provider_name,
                image_hash=str(hash(image_data)),
                error_message=str(e),
                error_type=type(e).__name__,
                correlation_id=correlation_id,
            )
            await self._event_bus.publish(event)
            raise

    def get_event_bus(self) -> EventBus:
        """Get the event bus."""
        return self._event_bus


# Factory functions
def create_event_bus() -> EventBus:
    """Create a new event bus."""
    return EventBus()


def create_event_router() -> EventRouter:
    """Create a new event router."""
    return EventRouter()


def create_event_store() -> EventStore:
    """Create a new event store."""
    return EventStore()


def create_event_aggregator() -> EventAggregator:
    """Create a new event aggregator."""
    return EventAggregator()


def create_command_bus() -> CommandBus:
    """Create a new command bus."""
    return CommandBus()


def create_query_bus() -> QueryBus:
    """Create a new query bus."""
    return QueryBus()


def create_event_provider(
    base_provider: VisionProvider,
    event_bus: Optional[EventBus] = None,
) -> EventDrivenVisionProvider:
    """Create an event-driven vision provider."""
    bus = event_bus or create_event_bus()
    return EventDrivenVisionProvider(base_provider, bus)


__all__ = [
    # Enums
    "EventPriority",
    "EventStatus",
    # Data classes
    "Event",
    "VisionEvent",
    "ImageAnalyzedEvent",
    "AnalysisFailedEvent",
    "EventEnvelope",
    # Handler classes
    "EventHandler",
    "FunctionEventHandler",
    # Core classes
    "EventBus",
    "EventRouter",
    "EventStore",
    "EventAggregator",
    "CommandBus",
    "QueryBus",
    "EventDrivenVisionProvider",
    # Factory functions
    "create_event_bus",
    "create_event_router",
    "create_event_store",
    "create_event_aggregator",
    "create_command_bus",
    "create_query_bus",
    "create_event_provider",
]
