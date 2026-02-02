"""Event Store.

Provides event storage and retrieval for event sourcing:
- Event persistence
- Stream management
- Event versioning
- Optimistic concurrency
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Base event class."""
    event_id: str
    event_type: str
    aggregate_id: str
    aggregate_type: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dict."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "data": self.data,
            "metadata": self.metadata,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Deserialize event from dict."""
        return cls(
            event_id=data["event_id"],
            event_type=data["event_type"],
            aggregate_id=data["aggregate_id"],
            aggregate_type=data["aggregate_type"],
            data=data["data"],
            metadata=data.get("metadata", {}),
            version=data.get("version", 1),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class EventStream:
    """A stream of events for an aggregate."""
    aggregate_id: str
    aggregate_type: str
    events: List[Event] = field(default_factory=list)
    version: int = 0

    def append(self, event: Event) -> None:
        """Append an event to the stream."""
        self.events.append(event)
        self.version = event.version


@dataclass
class AppendResult:
    """Result of appending events."""
    success: bool
    new_version: int
    events_appended: int
    error: Optional[str] = None


class ConcurrencyError(Exception):
    """Raised when there's a version conflict."""
    pass


class EventStore(ABC):
    """Abstract base class for event stores."""

    @abstractmethod
    async def append(
        self,
        aggregate_id: str,
        aggregate_type: str,
        events: List[Event],
        expected_version: Optional[int] = None,
    ) -> AppendResult:
        """Append events to an aggregate stream.

        Args:
            aggregate_id: Aggregate identifier.
            aggregate_type: Type of aggregate.
            events: Events to append.
            expected_version: Expected current version for optimistic concurrency.

        Returns:
            AppendResult with operation outcome.

        Raises:
            ConcurrencyError: If expected_version doesn't match.
        """
        pass

    @abstractmethod
    async def get_stream(
        self,
        aggregate_id: str,
        aggregate_type: str,
        from_version: int = 0,
        to_version: Optional[int] = None,
    ) -> EventStream:
        """Get events for an aggregate.

        Args:
            aggregate_id: Aggregate identifier.
            aggregate_type: Type of aggregate.
            from_version: Start version (inclusive).
            to_version: End version (inclusive).

        Returns:
            EventStream with matching events.
        """
        pass

    @abstractmethod
    async def get_all_events(
        self,
        from_position: int = 0,
        batch_size: int = 100,
        event_types: Optional[List[str]] = None,
    ) -> List[Event]:
        """Get all events across aggregates.

        Args:
            from_position: Global position to start from.
            batch_size: Maximum events to return.
            event_types: Filter by event types.

        Returns:
            List of events.
        """
        pass

    @abstractmethod
    async def get_current_version(
        self,
        aggregate_id: str,
        aggregate_type: str,
    ) -> int:
        """Get current version of an aggregate stream."""
        pass


class InMemoryEventStore(EventStore):
    """In-memory event store for testing and development."""

    def __init__(self):
        # {aggregate_type: {aggregate_id: [events]}}
        self._streams: Dict[str, Dict[str, List[Event]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._all_events: List[Event] = []
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def append(
        self,
        aggregate_id: str,
        aggregate_type: str,
        events: List[Event],
        expected_version: Optional[int] = None,
    ) -> AppendResult:
        async with self._get_lock():
            stream = self._streams[aggregate_type][aggregate_id]
            current_version = len(stream)

            # Check optimistic concurrency
            if expected_version is not None and expected_version != current_version:
                raise ConcurrencyError(
                    f"Expected version {expected_version}, but current is {current_version}"
                )

            # Assign versions and append
            for i, event in enumerate(events):
                event.version = current_version + i + 1
                event.aggregate_id = aggregate_id
                event.aggregate_type = aggregate_type
                stream.append(event)
                self._all_events.append(event)

            return AppendResult(
                success=True,
                new_version=current_version + len(events),
                events_appended=len(events),
            )

    async def get_stream(
        self,
        aggregate_id: str,
        aggregate_type: str,
        from_version: int = 0,
        to_version: Optional[int] = None,
    ) -> EventStream:
        async with self._get_lock():
            stream = self._streams[aggregate_type].get(aggregate_id, [])

            filtered = [
                e for e in stream
                if e.version >= from_version
                and (to_version is None or e.version <= to_version)
            ]

            return EventStream(
                aggregate_id=aggregate_id,
                aggregate_type=aggregate_type,
                events=filtered,
                version=len(stream),
            )

    async def get_all_events(
        self,
        from_position: int = 0,
        batch_size: int = 100,
        event_types: Optional[List[str]] = None,
    ) -> List[Event]:
        async with self._get_lock():
            events = self._all_events[from_position:]

            if event_types:
                events = [e for e in events if e.event_type in event_types]

            return events[:batch_size]

    async def get_current_version(
        self,
        aggregate_id: str,
        aggregate_type: str,
    ) -> int:
        async with self._get_lock():
            return len(self._streams[aggregate_type].get(aggregate_id, []))


def create_event(
    event_type: str,
    aggregate_id: str,
    aggregate_type: str,
    data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Event:
    """Helper to create events."""
    return Event(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        aggregate_id=aggregate_id,
        aggregate_type=aggregate_type,
        data=data,
        metadata=metadata or {},
    )


# Global event store
_event_store: Optional[EventStore] = None


def get_event_store() -> EventStore:
    """Get the global event store."""
    global _event_store
    if _event_store is None:
        _event_store = InMemoryEventStore()
    return _event_store


def configure_event_store(store: EventStore) -> None:
    """Configure the global event store."""
    global _event_store
    _event_store = store
