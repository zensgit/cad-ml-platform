"""Event Sourcing Module.

Provides complete event sourcing capabilities:
- Event store for persisting events
- Aggregate base classes
- Projections for read models
- Snapshots for performance optimization
"""

from src.core.eventsourcing.store import (
    Event,
    EventStream,
    AppendResult,
    ConcurrencyError,
    EventStore,
    InMemoryEventStore,
    create_event,
    get_event_store,
    configure_event_store,
)
from src.core.eventsourcing.aggregate import (
    Aggregate,
    AggregateRepository,
    Order,
)
from src.core.eventsourcing.projection import (
    ProjectionCheckpoint,
    Projection,
    ProjectionManager,
    OrderSummaryProjection,
    CustomerStatsProjection,
)
from src.core.eventsourcing.snapshot import (
    Snapshot,
    SnapshotStore,
    InMemorySnapshotStore,
    SnapshotStrategy,
    EventCountStrategy,
    TimeBasedStrategy,
    SnapshotRepository,
    get_snapshot_store,
)

__all__ = [
    # Store
    "Event",
    "EventStream",
    "AppendResult",
    "ConcurrencyError",
    "EventStore",
    "InMemoryEventStore",
    "create_event",
    "get_event_store",
    "configure_event_store",
    # Aggregate
    "Aggregate",
    "AggregateRepository",
    "Order",
    # Projection
    "ProjectionCheckpoint",
    "Projection",
    "ProjectionManager",
    "OrderSummaryProjection",
    "CustomerStatsProjection",
    # Snapshot
    "Snapshot",
    "SnapshotStore",
    "InMemorySnapshotStore",
    "SnapshotStrategy",
    "EventCountStrategy",
    "TimeBasedStrategy",
    "SnapshotRepository",
    "get_snapshot_store",
]
