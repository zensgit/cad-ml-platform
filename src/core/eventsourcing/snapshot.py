"""Aggregate Snapshots.

Provides snapshot functionality for event sourcing:
- Snapshot storage
- Snapshot-based aggregate loading
- Snapshot strategies
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar

from src.core.eventsourcing.store import EventStore, get_event_store
from src.core.eventsourcing.aggregate import Aggregate

logger = logging.getLogger(__name__)


T = TypeVar('T', bound=Aggregate)


@dataclass
class Snapshot:
    """A snapshot of aggregate state."""
    aggregate_id: str
    aggregate_type: str
    version: int
    state: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SnapshotStore(ABC):
    """Abstract base class for snapshot storage."""

    @abstractmethod
    async def save(self, snapshot: Snapshot) -> None:
        """Save a snapshot."""
        pass

    @abstractmethod
    async def get(
        self,
        aggregate_id: str,
        aggregate_type: str,
    ) -> Optional[Snapshot]:
        """Get the latest snapshot for an aggregate."""
        pass

    @abstractmethod
    async def delete(
        self,
        aggregate_id: str,
        aggregate_type: str,
    ) -> bool:
        """Delete snapshots for an aggregate."""
        pass


class InMemorySnapshotStore(SnapshotStore):
    """In-memory snapshot store."""

    def __init__(self):
        # {aggregate_type: {aggregate_id: snapshot}}
        self._snapshots: Dict[str, Dict[str, Snapshot]] = {}
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def save(self, snapshot: Snapshot) -> None:
        async with self._get_lock():
            if snapshot.aggregate_type not in self._snapshots:
                self._snapshots[snapshot.aggregate_type] = {}
            self._snapshots[snapshot.aggregate_type][snapshot.aggregate_id] = snapshot

    async def get(
        self,
        aggregate_id: str,
        aggregate_type: str,
    ) -> Optional[Snapshot]:
        async with self._get_lock():
            type_snapshots = self._snapshots.get(aggregate_type, {})
            return type_snapshots.get(aggregate_id)

    async def delete(
        self,
        aggregate_id: str,
        aggregate_type: str,
    ) -> bool:
        async with self._get_lock():
            type_snapshots = self._snapshots.get(aggregate_type, {})
            if aggregate_id in type_snapshots:
                del type_snapshots[aggregate_id]
                return True
            return False


class SnapshotStrategy(ABC):
    """Strategy for determining when to create snapshots."""

    @abstractmethod
    def should_snapshot(
        self,
        aggregate: Aggregate,
        events_since_snapshot: int,
    ) -> bool:
        """Determine if a snapshot should be created."""
        pass


class EventCountStrategy(SnapshotStrategy):
    """Create snapshot after N events."""

    def __init__(self, threshold: int = 100):
        self.threshold = threshold

    def should_snapshot(
        self,
        aggregate: Aggregate,
        events_since_snapshot: int,
    ) -> bool:
        return events_since_snapshot >= self.threshold


class TimeBasedStrategy(SnapshotStrategy):
    """Create snapshot after time period."""

    def __init__(self, interval_seconds: int = 3600):
        self.interval_seconds = interval_seconds
        self._last_snapshot: Dict[str, datetime] = {}

    def should_snapshot(
        self,
        aggregate: Aggregate,
        events_since_snapshot: int,
    ) -> bool:
        last = self._last_snapshot.get(aggregate.id)
        if last is None:
            self._last_snapshot[aggregate.id] = datetime.utcnow()
            return True

        elapsed = (datetime.utcnow() - last).total_seconds()
        if elapsed >= self.interval_seconds:
            self._last_snapshot[aggregate.id] = datetime.utcnow()
            return True

        return False


class SnapshotRepository(Generic[T]):
    """Repository with snapshot support."""

    def __init__(
        self,
        aggregate_class: Type[T],
        event_store: Optional[EventStore] = None,
        snapshot_store: Optional[SnapshotStore] = None,
        snapshot_strategy: Optional[SnapshotStrategy] = None,
        state_serializer: Optional[Callable[[T], Dict[str, Any]]] = None,
        state_deserializer: Optional[Callable[[Dict[str, Any], T], None]] = None,
    ):
        self._aggregate_class = aggregate_class
        self._event_store = event_store or get_event_store()
        self._snapshot_store = snapshot_store or InMemorySnapshotStore()
        self._snapshot_strategy = snapshot_strategy or EventCountStrategy(100)
        self._state_serializer = state_serializer
        self._state_deserializer = state_deserializer

    async def get(self, aggregate_id: str) -> Optional[T]:
        """Load an aggregate, using snapshot if available."""
        aggregate = self._aggregate_class(aggregate_id)

        # Try to load from snapshot
        snapshot = await self._snapshot_store.get(
            aggregate_id=aggregate_id,
            aggregate_type=aggregate.aggregate_type,
        )

        from_version = 0
        if snapshot:
            # Restore state from snapshot
            if self._state_deserializer:
                self._state_deserializer(snapshot.state, aggregate)
            else:
                self._restore_default_state(snapshot.state, aggregate)
            from_version = snapshot.version

        # Load events since snapshot
        stream = await self._event_store.get_stream(
            aggregate_id=aggregate_id,
            aggregate_type=aggregate.aggregate_type,
            from_version=from_version + 1,
        )

        if not stream.events and from_version == 0:
            return None

        aggregate.load_from_history(stream.events)
        return aggregate

    async def save(self, aggregate: T) -> None:
        """Save an aggregate and potentially create a snapshot."""
        events = aggregate.get_uncommitted_events()
        if not events:
            return

        expected_version = aggregate.version - len(events)
        await self._event_store.append(
            aggregate_id=aggregate.id,
            aggregate_type=aggregate.aggregate_type,
            events=events,
            expected_version=expected_version,
        )
        aggregate.mark_events_committed()

        # Check if we should create a snapshot
        snapshot = await self._snapshot_store.get(
            aggregate_id=aggregate.id,
            aggregate_type=aggregate.aggregate_type,
        )

        events_since_snapshot = aggregate.version
        if snapshot:
            events_since_snapshot = aggregate.version - snapshot.version

        if self._snapshot_strategy.should_snapshot(aggregate, events_since_snapshot):
            await self._create_snapshot(aggregate)

    async def _create_snapshot(self, aggregate: T) -> None:
        """Create a snapshot of the aggregate."""
        if self._state_serializer:
            state = self._state_serializer(aggregate)
        else:
            state = self._serialize_default_state(aggregate)

        snapshot = Snapshot(
            aggregate_id=aggregate.id,
            aggregate_type=aggregate.aggregate_type,
            version=aggregate.version,
            state=state,
        )

        await self._snapshot_store.save(snapshot)
        logger.debug(
            f"Created snapshot for {aggregate.aggregate_type}/{aggregate.id} "
            f"at version {aggregate.version}"
        )

    def _serialize_default_state(self, aggregate: T) -> Dict[str, Any]:
        """Default state serialization."""
        state = {}
        for key, value in aggregate.__dict__.items():
            if not key.startswith('_'):
                try:
                    json.dumps(value)  # Check if serializable
                    state[key] = value
                except (TypeError, ValueError):
                    pass
        return state

    def _restore_default_state(self, state: Dict[str, Any], aggregate: T) -> None:
        """Default state restoration."""
        for key, value in state.items():
            if hasattr(aggregate, key):
                setattr(aggregate, key, value)


# Global snapshot store
_snapshot_store: Optional[SnapshotStore] = None


def get_snapshot_store() -> SnapshotStore:
    """Get the global snapshot store."""
    global _snapshot_store
    if _snapshot_store is None:
        _snapshot_store = InMemorySnapshotStore()
    return _snapshot_store
