"""Event sourcing and CQRS patterns for Vision Provider system.

This module provides event sourcing features including:
- Event store
- Event replay
- Projections
- Snapshots
- CQRS command/query separation
"""

import asyncio
import json
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Set, Type, TypeVar, Union

from .base import VisionDescription, VisionProvider


class EventType(Enum):
    """Event type."""

    IMAGE_ANALYZED = "image_analyzed"
    ANALYSIS_FAILED = "analysis_failed"
    PROVIDER_CHANGED = "provider_changed"
    CONFIGURATION_UPDATED = "configuration_updated"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    RATE_LIMITED = "rate_limited"
    CIRCUIT_OPENED = "circuit_opened"
    CIRCUIT_CLOSED = "circuit_closed"


class AggregateType(Enum):
    """Aggregate type."""

    VISION_REQUEST = "vision_request"
    PROVIDER = "provider"
    CONFIGURATION = "configuration"
    SESSION = "session"


@dataclass
class Event:
    """Base event."""

    event_id: str
    event_type: EventType
    aggregate_id: str
    aggregate_type: AggregateType
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 1
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type.value,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "data": dict(self.data),
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            aggregate_id=data["aggregate_id"],
            aggregate_type=AggregateType(data["aggregate_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            version=data["version"],
            data=data.get("data", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Snapshot:
    """Aggregate snapshot."""

    aggregate_id: str
    aggregate_type: AggregateType
    version: int
    state: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type.value,
            "version": self.version,
            "state": dict(self.state),
            "timestamp": self.timestamp.isoformat(),
        }


class EventStore(ABC):
    """Abstract event store."""

    @abstractmethod
    def append(self, event: Event) -> None:
        """Append event to store."""
        pass

    @abstractmethod
    def get_events(
        self,
        aggregate_id: str,
        from_version: int = 0,
    ) -> List[Event]:
        """Get events for aggregate."""
        pass

    @abstractmethod
    def get_all_events(
        self,
        event_type: Optional[EventType] = None,
        since: Optional[datetime] = None,
    ) -> List[Event]:
        """Get all events."""
        pass


class InMemoryEventStore(EventStore):
    """In-memory event store."""

    def __init__(self) -> None:
        """Initialize store."""
        self._events: List[Event] = []
        self._by_aggregate: Dict[str, List[Event]] = {}
        self._lock = threading.Lock()

    def append(self, event: Event) -> None:
        """Append event to store."""
        with self._lock:
            self._events.append(event)

            if event.aggregate_id not in self._by_aggregate:
                self._by_aggregate[event.aggregate_id] = []
            self._by_aggregate[event.aggregate_id].append(event)

    def get_events(
        self,
        aggregate_id: str,
        from_version: int = 0,
    ) -> List[Event]:
        """Get events for aggregate."""
        with self._lock:
            events = self._by_aggregate.get(aggregate_id, [])
            return [e for e in events if e.version >= from_version]

    def get_all_events(
        self,
        event_type: Optional[EventType] = None,
        since: Optional[datetime] = None,
    ) -> List[Event]:
        """Get all events."""
        with self._lock:
            events = list(self._events)

        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if since:
            events = [e for e in events if e.timestamp >= since]

        return events

    def count(self) -> int:
        """Get event count."""
        with self._lock:
            return len(self._events)


class SnapshotStore:
    """Snapshot store."""

    def __init__(self) -> None:
        """Initialize store."""
        self._snapshots: Dict[str, Snapshot] = {}
        self._lock = threading.Lock()

    def save(self, snapshot: Snapshot) -> None:
        """Save snapshot."""
        with self._lock:
            self._snapshots[snapshot.aggregate_id] = snapshot

    def get(self, aggregate_id: str) -> Optional[Snapshot]:
        """Get snapshot."""
        with self._lock:
            return self._snapshots.get(aggregate_id)

    def delete(self, aggregate_id: str) -> None:
        """Delete snapshot."""
        with self._lock:
            self._snapshots.pop(aggregate_id, None)


T = TypeVar("T")


class Aggregate(ABC, Generic[T]):
    """Base aggregate."""

    def __init__(self, aggregate_id: str) -> None:
        """Initialize aggregate."""
        self._id = aggregate_id
        self._version = 0
        self._pending_events: List[Event] = []

    @property
    def id(self) -> str:
        """Get aggregate ID."""
        return self._id

    @property
    def version(self) -> int:
        """Get version."""
        return self._version

    @abstractmethod
    def apply(self, event: Event) -> None:
        """Apply event to aggregate."""
        pass

    @abstractmethod
    def get_state(self) -> T:
        """Get current state."""
        pass

    def load_from_history(self, events: List[Event]) -> None:
        """Load aggregate from event history."""
        for event in events:
            self.apply(event)
            self._version = event.version

    def raise_event(self, event: Event) -> None:
        """Raise a new event."""
        self._version += 1
        event.version = self._version
        self.apply(event)
        self._pending_events.append(event)

    def get_pending_events(self) -> List[Event]:
        """Get pending events."""
        return list(self._pending_events)

    def clear_pending_events(self) -> None:
        """Clear pending events."""
        self._pending_events.clear()


@dataclass
class VisionRequestState:
    """Vision request aggregate state."""

    request_id: str
    image_hash: str = ""
    provider: str = ""
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class VisionRequestAggregate(Aggregate[VisionRequestState]):
    """Vision request aggregate."""

    def __init__(self, request_id: str) -> None:
        """Initialize aggregate."""
        super().__init__(request_id)
        self._state = VisionRequestState(request_id=request_id)

    def apply(self, event: Event) -> None:
        """Apply event to aggregate."""
        if event.event_type == EventType.IMAGE_ANALYZED:
            self._state.status = "completed"
            self._state.result = event.data.get("result")
            self._state.completed_at = event.timestamp
            self._state.provider = event.data.get("provider", "")

        elif event.event_type == EventType.ANALYSIS_FAILED:
            self._state.status = "failed"
            self._state.error = event.data.get("error")
            self._state.completed_at = event.timestamp

        elif event.event_type == EventType.CACHE_HIT:
            self._state.status = "cache_hit"
            self._state.result = event.data.get("result")

    def get_state(self) -> VisionRequestState:
        """Get current state."""
        return self._state

    def start_analysis(self, image_hash: str, provider: str) -> None:
        """Start analysis."""
        self._state.image_hash = image_hash
        self._state.provider = provider
        self._state.created_at = datetime.now()

    def complete_analysis(self, result: Dict[str, Any], provider: str) -> None:
        """Complete analysis with result."""
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=EventType.IMAGE_ANALYZED,
            aggregate_id=self._id,
            aggregate_type=AggregateType.VISION_REQUEST,
            data={"result": result, "provider": provider},
        )
        self.raise_event(event)

    def fail_analysis(self, error: str) -> None:
        """Fail analysis."""
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=EventType.ANALYSIS_FAILED,
            aggregate_id=self._id,
            aggregate_type=AggregateType.VISION_REQUEST,
            data={"error": error},
        )
        self.raise_event(event)


class Projection(ABC):
    """Base projection."""

    @abstractmethod
    def apply(self, event: Event) -> None:
        """Apply event to projection."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get projection state."""
        pass


class RequestCountProjection(Projection):
    """Request count projection."""

    def __init__(self) -> None:
        """Initialize projection."""
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._by_provider: Dict[str, int] = {}

    def apply(self, event: Event) -> None:
        """Apply event to projection."""
        if event.event_type == EventType.IMAGE_ANALYZED:
            self._total_requests += 1
            self._successful_requests += 1
            provider = event.data.get("provider", "unknown")
            self._by_provider[provider] = self._by_provider.get(provider, 0) + 1

        elif event.event_type == EventType.ANALYSIS_FAILED:
            self._total_requests += 1
            self._failed_requests += 1

    def get_state(self) -> Dict[str, Any]:
        """Get projection state."""
        return {
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "by_provider": dict(self._by_provider),
            "success_rate": (
                self._successful_requests / self._total_requests
                if self._total_requests > 0
                else 0.0
            ),
        }


class LatencyProjection(Projection):
    """Latency projection."""

    def __init__(self) -> None:
        """Initialize projection."""
        self._latencies: List[float] = []
        self._by_provider: Dict[str, List[float]] = {}

    def apply(self, event: Event) -> None:
        """Apply event to projection."""
        if event.event_type == EventType.IMAGE_ANALYZED:
            latency = event.data.get("latency_ms", 0.0)
            self._latencies.append(latency)

            provider = event.data.get("provider", "unknown")
            if provider not in self._by_provider:
                self._by_provider[provider] = []
            self._by_provider[provider].append(latency)

    def get_state(self) -> Dict[str, Any]:
        """Get projection state."""
        avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else 0.0

        by_provider_avg = {}
        for provider, latencies in self._by_provider.items():
            by_provider_avg[provider] = sum(latencies) / len(latencies) if latencies else 0.0

        return {
            "total_requests": len(self._latencies),
            "average_latency_ms": avg_latency,
            "min_latency_ms": min(self._latencies) if self._latencies else 0.0,
            "max_latency_ms": max(self._latencies) if self._latencies else 0.0,
            "by_provider": by_provider_avg,
        }


class ProjectionManager:
    """Manages projections."""

    def __init__(self, event_store: EventStore) -> None:
        """Initialize manager."""
        self._event_store = event_store
        self._projections: Dict[str, Projection] = {}
        self._lock = threading.Lock()

    def register(self, name: str, projection: Projection) -> None:
        """Register projection."""
        with self._lock:
            self._projections[name] = projection

    def unregister(self, name: str) -> None:
        """Unregister projection."""
        with self._lock:
            self._projections.pop(name, None)

    def apply_event(self, event: Event) -> None:
        """Apply event to all projections."""
        with self._lock:
            for projection in self._projections.values():
                projection.apply(event)

    def rebuild(self, name: str) -> None:
        """Rebuild projection from events."""
        with self._lock:
            projection = self._projections.get(name)
            if not projection:
                return

        events = self._event_store.get_all_events()
        for event in events:
            projection.apply(event)

    def rebuild_all(self) -> None:
        """Rebuild all projections."""
        events = self._event_store.get_all_events()

        with self._lock:
            for projection in self._projections.values():
                for event in events:
                    projection.apply(event)

    def get_projection(self, name: str) -> Optional[Projection]:
        """Get projection."""
        with self._lock:
            return self._projections.get(name)

    def get_state(self, name: str) -> Optional[Dict[str, Any]]:
        """Get projection state."""
        projection = self.get_projection(name)
        return projection.get_state() if projection else None


# CQRS Command types


@dataclass
class Command:
    """Base command."""

    command_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyzeImageCommand(Command):
    """Analyze image command."""

    image_data: bytes = b""
    provider: str = ""
    include_description: bool = True


@dataclass
class Query:
    """Base query."""

    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GetRequestStatusQuery(Query):
    """Get request status query."""

    request_id: str = ""


@dataclass
class GetStatisticsQuery(Query):
    """Get statistics query."""

    since: Optional[datetime] = None
    provider: Optional[str] = None


class CommandHandler(ABC):
    """Abstract command handler."""

    @abstractmethod
    async def handle(self, command: Command) -> Any:
        """Handle command."""
        pass


class QueryHandler(ABC):
    """Abstract query handler."""

    @abstractmethod
    def handle(self, query: Query) -> Any:
        """Handle query."""
        pass


class AnalyzeImageCommandHandler(CommandHandler):
    """Handle analyze image commands."""

    def __init__(
        self,
        provider: VisionProvider,
        event_store: EventStore,
    ) -> None:
        """Initialize handler."""
        self._provider = provider
        self._event_store = event_store

    async def handle(self, command: Command) -> Dict[str, Any]:
        """Handle command."""
        if not isinstance(command, AnalyzeImageCommand):
            raise ValueError("Invalid command type")

        request_id = str(uuid.uuid4())
        aggregate = VisionRequestAggregate(request_id)

        import hashlib

        image_hash = hashlib.sha256(command.image_data).hexdigest()[:16]
        aggregate.start_analysis(image_hash, self._provider.provider_name)

        try:
            result = await self._provider.analyze_image(
                command.image_data,
                command.include_description,
            )

            aggregate.complete_analysis(
                result={
                    "summary": result.summary,
                    "details": result.details,
                    "confidence": result.confidence,
                },
                provider=self._provider.provider_name,
            )

        except Exception as e:
            aggregate.fail_analysis(str(e))

        # Persist events
        for event in aggregate.get_pending_events():
            self._event_store.append(event)

        aggregate.clear_pending_events()

        return {
            "request_id": request_id,
            "status": aggregate.get_state().status,
            "result": aggregate.get_state().result,
        }


class GetStatisticsQueryHandler(QueryHandler):
    """Handle statistics queries."""

    def __init__(self, projection_manager: ProjectionManager) -> None:
        """Initialize handler."""
        self._projection_manager = projection_manager

    def handle(self, query: Query) -> Dict[str, Any]:
        """Handle query."""
        if not isinstance(query, GetStatisticsQuery):
            raise ValueError("Invalid query type")

        request_count = self._projection_manager.get_state("request_count")
        latency = self._projection_manager.get_state("latency")

        return {
            "request_count": request_count or {},
            "latency": latency or {},
        }


class EventSourcedVisionProvider(VisionProvider):
    """Vision provider with event sourcing."""

    def __init__(
        self,
        provider: VisionProvider,
        event_store: Optional[EventStore] = None,
    ) -> None:
        """Initialize provider."""
        self._provider = provider
        self._event_store = event_store or InMemoryEventStore()
        self._projection_manager = ProjectionManager(self._event_store)

        # Register default projections
        self._projection_manager.register("request_count", RequestCountProjection())
        self._projection_manager.register("latency", LatencyProjection())

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"event_sourced_{self._provider.provider_name}"

    def get_event_store(self) -> EventStore:
        """Get event store."""
        return self._event_store

    def get_projection_manager(self) -> ProjectionManager:
        """Get projection manager."""
        return self._projection_manager

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with event sourcing."""
        import hashlib

        request_id = str(uuid.uuid4())
        image_hash = hashlib.sha256(image_data).hexdigest()[:16]

        start_time = time.time()

        try:
            result = await self._provider.analyze_image(image_data, include_description)

            latency_ms = (time.time() - start_time) * 1000

            # Create and store event
            event = Event(
                event_id=str(uuid.uuid4()),
                event_type=EventType.IMAGE_ANALYZED,
                aggregate_id=request_id,
                aggregate_type=AggregateType.VISION_REQUEST,
                data={
                    "image_hash": image_hash,
                    "provider": self._provider.provider_name,
                    "latency_ms": latency_ms,
                    "result": {
                        "summary": result.summary,
                        "confidence": result.confidence,
                    },
                },
            )

            self._event_store.append(event)
            self._projection_manager.apply_event(event)

            return result

        except Exception as e:
            # Create failure event
            event = Event(
                event_id=str(uuid.uuid4()),
                event_type=EventType.ANALYSIS_FAILED,
                aggregate_id=request_id,
                aggregate_type=AggregateType.VISION_REQUEST,
                data={
                    "image_hash": image_hash,
                    "error": str(e),
                },
            )

            self._event_store.append(event)
            self._projection_manager.apply_event(event)

            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from projections."""
        return {
            "request_count": self._projection_manager.get_state("request_count"),
            "latency": self._projection_manager.get_state("latency"),
            "event_count": self._event_store.count() if hasattr(self._event_store, "count") else 0,
        }

    def replay_events(self) -> None:
        """Replay all events through projections."""
        self._projection_manager.rebuild_all()


def create_event_sourced_provider(
    provider: VisionProvider,
) -> EventSourcedVisionProvider:
    """Create event-sourced vision provider.

    Args:
        provider: Provider to wrap

    Returns:
        Event-sourced provider
    """
    return EventSourcedVisionProvider(provider)
