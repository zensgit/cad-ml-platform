"""Event Sourcing Module.

Provides event sourcing infrastructure:
- Event store with append-only persistence
- Event streams and versioning
- Projections and read models
- Snapshots for performance
- Event replay and rehydration
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")
E = TypeVar("E", bound="Event")
A = TypeVar("A", bound="Aggregate")


class EventType(Enum):
    """Standard event types."""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    STATE_CHANGED = "state_changed"
    CUSTOM = "custom"


@dataclass
class EventMetadata:
    """Metadata for an event."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "user_id": self.user_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventMetadata":
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
            version=data.get("version", 1),
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
            user_id=data.get("user_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Event:
    """Base event class."""

    aggregate_id: str
    aggregate_type: str
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: EventMetadata = field(default_factory=EventMetadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "event_type": self.event_type,
            "data": self.data,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        return cls(
            aggregate_id=data["aggregate_id"],
            aggregate_type=data["aggregate_type"],
            event_type=data["event_type"],
            data=data.get("data", {}),
            metadata=EventMetadata.from_dict(data.get("metadata", {})),
        )

    @property
    def event_id(self) -> str:
        return self.metadata.event_id

    @property
    def timestamp(self) -> datetime:
        return self.metadata.timestamp

    @property
    def version(self) -> int:
        return self.metadata.version


@dataclass
class EventStream:
    """Stream of events for an aggregate."""

    aggregate_id: str
    aggregate_type: str
    events: List[Event] = field(default_factory=list)
    version: int = 0

    def append(self, event: Event) -> None:
        """Append event to stream."""
        self.version += 1
        event.metadata.version = self.version
        self.events.append(event)

    def get_events_after(self, version: int) -> List[Event]:
        """Get events after specific version."""
        return [e for e in self.events if e.version > version]

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)


@dataclass
class Snapshot:
    """Snapshot of aggregate state."""

    aggregate_id: str
    aggregate_type: str
    version: int
    state: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "version": self.version,
            "state": self.state,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Snapshot":
        return cls(
            aggregate_id=data["aggregate_id"],
            aggregate_type=data["aggregate_type"],
            version=data["version"],
            state=data["state"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
        )


class EventStore(ABC):
    """Abstract event store."""

    @abstractmethod
    async def append(self, event: Event) -> bool:
        """Append event to store."""
        pass

    @abstractmethod
    async def get_events(
        self,
        aggregate_id: str,
        after_version: int = 0,
    ) -> List[Event]:
        """Get events for aggregate."""
        pass

    @abstractmethod
    async def get_all_events(
        self,
        after_position: int = 0,
        limit: int = 100,
    ) -> List[Event]:
        """Get all events from store."""
        pass

    @abstractmethod
    async def get_events_by_type(
        self,
        event_type: str,
        after_position: int = 0,
        limit: int = 100,
    ) -> List[Event]:
        """Get events by type."""
        pass


class InMemoryEventStore(EventStore):
    """In-memory event store implementation."""

    def __init__(self):
        self._events: List[Event] = []
        self._streams: Dict[str, EventStream] = {}
        self._position = 0
        self._lock = asyncio.Lock()

    async def append(self, event: Event) -> bool:
        """Append event to store."""
        async with self._lock:
            # Get or create stream
            stream_key = f"{event.aggregate_type}:{event.aggregate_id}"
            if stream_key not in self._streams:
                self._streams[stream_key] = EventStream(
                    aggregate_id=event.aggregate_id,
                    aggregate_type=event.aggregate_type,
                )

            stream = self._streams[stream_key]

            # Optimistic concurrency check
            if event.metadata.version != 0 and event.metadata.version != stream.version + 1:
                logger.warning(
                    f"Concurrency conflict: expected version {stream.version + 1}, "
                    f"got {event.metadata.version}"
                )
                return False

            # Append to stream and global log
            stream.append(event)
            self._position += 1
            self._events.append(event)

            logger.debug(f"Appended event {event.event_id} to stream {stream_key}")
            return True

    async def get_events(
        self,
        aggregate_id: str,
        after_version: int = 0,
    ) -> List[Event]:
        """Get events for aggregate."""
        events = [
            e for e in self._events
            if e.aggregate_id == aggregate_id and e.version > after_version
        ]
        return sorted(events, key=lambda e: e.version)

    async def get_all_events(
        self,
        after_position: int = 0,
        limit: int = 100,
    ) -> List[Event]:
        """Get all events from store."""
        return self._events[after_position:after_position + limit]

    async def get_events_by_type(
        self,
        event_type: str,
        after_position: int = 0,
        limit: int = 100,
    ) -> List[Event]:
        """Get events by type."""
        matching = [e for e in self._events if e.event_type == event_type]
        return matching[after_position:after_position + limit]


class SnapshotStore(ABC):
    """Abstract snapshot store."""

    @abstractmethod
    async def save(self, snapshot: Snapshot) -> bool:
        """Save snapshot."""
        pass

    @abstractmethod
    async def get(self, aggregate_id: str) -> Optional[Snapshot]:
        """Get latest snapshot for aggregate."""
        pass

    @abstractmethod
    async def delete(self, aggregate_id: str) -> bool:
        """Delete snapshot."""
        pass


class InMemorySnapshotStore(SnapshotStore):
    """In-memory snapshot store."""

    def __init__(self):
        self._snapshots: Dict[str, Snapshot] = {}

    async def save(self, snapshot: Snapshot) -> bool:
        self._snapshots[snapshot.aggregate_id] = snapshot
        return True

    async def get(self, aggregate_id: str) -> Optional[Snapshot]:
        return self._snapshots.get(aggregate_id)

    async def delete(self, aggregate_id: str) -> bool:
        if aggregate_id in self._snapshots:
            del self._snapshots[aggregate_id]
            return True
        return False


class Aggregate(ABC):
    """Base aggregate class."""

    def __init__(self, aggregate_id: str):
        self._id = aggregate_id
        self._version = 0
        self._changes: List[Event] = []

    @property
    def id(self) -> str:
        return self._id

    @property
    def version(self) -> int:
        return self._version

    @property
    @abstractmethod
    def aggregate_type(self) -> str:
        """Return aggregate type name."""
        pass

    def get_uncommitted_changes(self) -> List[Event]:
        """Get uncommitted events."""
        return self._changes.copy()

    def mark_changes_as_committed(self) -> None:
        """Clear uncommitted changes."""
        self._changes.clear()

    def load_from_history(self, events: List[Event]) -> None:
        """Load aggregate state from event history."""
        for event in events:
            self._apply(event)
            self._version = event.version

    def apply_change(self, event: Event) -> None:
        """Apply and record a change."""
        self._apply(event)
        self._version += 1
        event.metadata.version = self._version
        self._changes.append(event)

    @abstractmethod
    def _apply(self, event: Event) -> None:
        """Apply event to aggregate state."""
        pass

    def get_snapshot_state(self) -> Dict[str, Any]:
        """Get state for snapshot."""
        return {}

    def restore_from_snapshot(self, state: Dict[str, Any]) -> None:
        """Restore state from snapshot."""
        pass


class Projection(ABC):
    """Base projection class for building read models."""

    def __init__(self, name: str):
        self._name = name
        self._position = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def position(self) -> int:
        return self._position

    @abstractmethod
    async def apply(self, event: Event) -> None:
        """Apply event to projection."""
        pass

    async def handle_event(self, event: Event, position: int) -> None:
        """Handle event with position tracking."""
        await self.apply(event)
        self._position = position


class InMemoryProjection(Projection):
    """In-memory projection with dict-based read model."""

    def __init__(self, name: str):
        super().__init__(name)
        self._data: Dict[str, Dict[str, Any]] = {}
        self._handlers: Dict[str, Callable[[Event], None]] = {}

    def register_handler(
        self,
        event_type: str,
        handler: Callable[[Event], None],
    ) -> None:
        """Register handler for event type."""
        self._handlers[event_type] = handler

    async def apply(self, event: Event) -> None:
        """Apply event using registered handler."""
        handler = self._handlers.get(event.event_type)
        if handler:
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get read model entry."""
        return self._data.get(key)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Set read model entry."""
        self._data[key] = value

    def delete(self, key: str) -> None:
        """Delete read model entry."""
        self._data.pop(key, None)

    def all(self) -> Dict[str, Dict[str, Any]]:
        """Get all entries."""
        return self._data.copy()


class ProjectionManager:
    """Manages projections and their updates."""

    def __init__(
        self,
        event_store: EventStore,
        poll_interval: float = 1.0,
    ):
        self._event_store = event_store
        self._poll_interval = poll_interval
        self._projections: Dict[str, Projection] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def register(self, projection: Projection) -> None:
        """Register a projection."""
        self._projections[projection.name] = projection
        logger.info(f"Registered projection: {projection.name}")

    def unregister(self, name: str) -> None:
        """Unregister a projection."""
        self._projections.pop(name, None)

    async def start(self) -> None:
        """Start projection updates."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._update_loop())
        logger.info("Projection manager started")

    async def stop(self) -> None:
        """Stop projection updates."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Projection manager stopped")

    async def _update_loop(self) -> None:
        """Continuously update projections."""
        while self._running:
            try:
                await self._update_projections()
            except Exception as e:
                logger.error(f"Projection update error: {e}")

            await asyncio.sleep(self._poll_interval)

    async def _update_projections(self) -> None:
        """Update all projections with new events."""
        for projection in self._projections.values():
            events = await self._event_store.get_all_events(
                after_position=projection.position,
                limit=100,
            )

            for i, event in enumerate(events):
                position = projection.position + i + 1
                await projection.handle_event(event, position)

    async def rebuild(self, projection_name: str) -> None:
        """Rebuild a projection from scratch."""
        projection = self._projections.get(projection_name)
        if not projection:
            return

        # Reset projection state
        if hasattr(projection, "_data"):
            projection._data.clear()
        projection._position = 0

        # Replay all events
        position = 0
        while True:
            events = await self._event_store.get_all_events(
                after_position=position,
                limit=100,
            )
            if not events:
                break

            for event in events:
                position += 1
                await projection.handle_event(event, position)

        logger.info(f"Rebuilt projection {projection_name} to position {position}")


class AggregateRepository(Generic[A]):
    """Repository for loading and saving aggregates."""

    def __init__(
        self,
        aggregate_class: Type[A],
        event_store: EventStore,
        snapshot_store: Optional[SnapshotStore] = None,
        snapshot_frequency: int = 100,  # Create snapshot every N events
    ):
        self._aggregate_class = aggregate_class
        self._event_store = event_store
        self._snapshot_store = snapshot_store
        self._snapshot_frequency = snapshot_frequency

    async def get(self, aggregate_id: str) -> Optional[A]:
        """Load aggregate from event store."""
        aggregate = self._aggregate_class(aggregate_id)

        # Try to load from snapshot first
        start_version = 0
        if self._snapshot_store:
            snapshot = await self._snapshot_store.get(aggregate_id)
            if snapshot:
                aggregate.restore_from_snapshot(snapshot.state)
                aggregate._version = snapshot.version
                start_version = snapshot.version
                logger.debug(f"Loaded snapshot for {aggregate_id} at version {start_version}")

        # Load events after snapshot
        events = await self._event_store.get_events(aggregate_id, start_version)
        if not events and start_version == 0:
            return None

        aggregate.load_from_history(events)
        return aggregate

    async def save(self, aggregate: A) -> bool:
        """Save aggregate to event store."""
        changes = aggregate.get_uncommitted_changes()
        if not changes:
            return True

        # Append events
        for event in changes:
            success = await self._event_store.append(event)
            if not success:
                return False

        aggregate.mark_changes_as_committed()

        # Create snapshot if needed
        if (
            self._snapshot_store
            and aggregate.version % self._snapshot_frequency == 0
            and aggregate.version > 0
        ):
            snapshot = Snapshot(
                aggregate_id=aggregate.id,
                aggregate_type=aggregate.aggregate_type,
                version=aggregate.version,
                state=aggregate.get_snapshot_state(),
            )
            await self._snapshot_store.save(snapshot)
            logger.debug(f"Created snapshot for {aggregate.id} at version {aggregate.version}")

        return True


class EventPublisher:
    """Publishes events to subscribers."""

    def __init__(self):
        self._handlers: Dict[str, List[Callable[[Event], Any]]] = {}

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Event], Any],
    ) -> None:
        """Subscribe to event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def unsubscribe(
        self,
        event_type: str,
        handler: Callable[[Event], Any],
    ) -> None:
        """Unsubscribe from event type."""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h != handler
            ]

    async def publish(self, event: Event) -> None:
        """Publish event to subscribers."""
        handlers = self._handlers.get(event.event_type, [])
        handlers.extend(self._handlers.get("*", []))  # Wildcard subscribers

        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Event handler error for {event.event_type}: {e}")


class EventSourcedService:
    """Base service for event-sourced applications."""

    def __init__(
        self,
        event_store: Optional[EventStore] = None,
        snapshot_store: Optional[SnapshotStore] = None,
        publisher: Optional[EventPublisher] = None,
    ):
        self._event_store = event_store or InMemoryEventStore()
        self._snapshot_store = snapshot_store or InMemorySnapshotStore()
        self._publisher = publisher or EventPublisher()
        self._projection_manager = ProjectionManager(self._event_store)

    @property
    def event_store(self) -> EventStore:
        return self._event_store

    @property
    def snapshot_store(self) -> SnapshotStore:
        return self._snapshot_store

    @property
    def publisher(self) -> EventPublisher:
        return self._publisher

    @property
    def projection_manager(self) -> ProjectionManager:
        return self._projection_manager

    def create_repository(
        self,
        aggregate_class: Type[A],
        snapshot_frequency: int = 100,
    ) -> AggregateRepository[A]:
        """Create repository for aggregate type."""
        return AggregateRepository(
            aggregate_class=aggregate_class,
            event_store=self._event_store,
            snapshot_store=self._snapshot_store,
            snapshot_frequency=snapshot_frequency,
        )

    async def start(self) -> None:
        """Start the service."""
        await self._projection_manager.start()

    async def stop(self) -> None:
        """Stop the service."""
        await self._projection_manager.stop()


__all__ = [
    "EventType",
    "EventMetadata",
    "Event",
    "EventStream",
    "Snapshot",
    "EventStore",
    "InMemoryEventStore",
    "SnapshotStore",
    "InMemorySnapshotStore",
    "Aggregate",
    "Projection",
    "InMemoryProjection",
    "ProjectionManager",
    "AggregateRepository",
    "EventPublisher",
    "EventSourcedService",
]
