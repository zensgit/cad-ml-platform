"""Event Sourced Aggregates.

Provides base classes for event-sourced aggregates:
- Aggregate root with event application
- State reconstruction from events
- Command handling
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar

from src.core.eventsourcing.store import Event, EventStore, create_event, get_event_store

logger = logging.getLogger(__name__)


T = TypeVar('T', bound='Aggregate')


class Aggregate(ABC):
    """Base class for event-sourced aggregates."""

    def __init__(self, aggregate_id: Optional[str] = None):
        self._id = aggregate_id or str(uuid.uuid4())
        self._version = 0
        self._uncommitted_events: List[Event] = []
        self._event_handlers: Dict[str, Callable[[Event], None]] = {}

    @property
    def id(self) -> str:
        return self._id

    @property
    def version(self) -> int:
        return self._version

    @property
    @abstractmethod
    def aggregate_type(self) -> str:
        """Return the aggregate type name."""
        pass

    def _register_handler(self, event_type: str, handler: Callable[[Event], None]) -> None:
        """Register an event handler."""
        self._event_handlers[event_type] = handler

    def apply(self, event: Event) -> None:
        """Apply an event to update state."""
        handler = self._event_handlers.get(event.event_type)
        if handler:
            handler(event)
        self._version = event.version

    def raise_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Raise a new event."""
        event = create_event(
            event_type=event_type,
            aggregate_id=self._id,
            aggregate_type=self.aggregate_type,
            data=data,
            metadata=metadata,
        )
        event.version = self._version + 1
        self._uncommitted_events.append(event)
        self.apply(event)

    def get_uncommitted_events(self) -> List[Event]:
        """Get events that haven't been persisted."""
        return self._uncommitted_events.copy()

    def mark_events_committed(self) -> None:
        """Mark all events as committed."""
        self._uncommitted_events.clear()

    def load_from_history(self, events: List[Event]) -> None:
        """Reconstruct state from historical events."""
        for event in events:
            if event.version <= self._version:
                event.version = self._version + 1
            self.apply(event)


class AggregateRepository(Generic[T]):
    """Repository for loading and saving aggregates."""

    def __init__(
        self,
        aggregate_class: Type[T],
        event_store: Optional[EventStore] = None,
    ):
        self._aggregate_class = aggregate_class
        self._event_store = event_store or get_event_store()

    async def get(self, aggregate_id: str) -> Optional[T]:
        """Load an aggregate by ID.

        Args:
            aggregate_id: Aggregate identifier.

        Returns:
            Aggregate instance or None if not found.
        """
        aggregate = self._aggregate_class(aggregate_id)
        stream = await self._event_store.get_stream(
            aggregate_id=aggregate_id,
            aggregate_type=aggregate.aggregate_type,
        )

        if not stream.events:
            return None

        aggregate.load_from_history(stream.events)
        return aggregate

    async def save(self, aggregate: T) -> None:
        """Save uncommitted events from an aggregate.

        Args:
            aggregate: Aggregate to save.
        """
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

    async def exists(self, aggregate_id: str) -> bool:
        """Check if an aggregate exists."""
        aggregate = self._aggregate_class(aggregate_id)
        version = await self._event_store.get_current_version(
            aggregate_id=aggregate_id,
            aggregate_type=aggregate.aggregate_type,
        )
        return version > 0


# Example: Order Aggregate
class Order(Aggregate):
    """Example order aggregate."""

    def __init__(self, order_id: Optional[str] = None):
        super().__init__(order_id)
        self.customer_id: Optional[str] = None
        self.items: List[Dict[str, Any]] = []
        self.status: str = "draft"
        self.total: float = 0.0

        # Register event handlers
        self._register_handler("OrderCreated", self._on_order_created)
        self._register_handler("ItemAdded", self._on_item_added)
        self._register_handler("ItemRemoved", self._on_item_removed)
        self._register_handler("OrderSubmitted", self._on_order_submitted)
        self._register_handler("OrderCancelled", self._on_order_cancelled)

    @property
    def aggregate_type(self) -> str:
        return "Order"

    def create(self, customer_id: str) -> None:
        """Create a new order."""
        if self.status != "draft":
            raise ValueError("Order already created")

        self.raise_event("OrderCreated", {
            "customer_id": customer_id,
        })

    def add_item(self, product_id: str, quantity: int, price: float) -> None:
        """Add an item to the order."""
        if self.status != "draft":
            raise ValueError("Cannot modify submitted order")

        self.raise_event("ItemAdded", {
            "product_id": product_id,
            "quantity": quantity,
            "price": price,
        })

    def remove_item(self, product_id: str) -> None:
        """Remove an item from the order."""
        if self.status != "draft":
            raise ValueError("Cannot modify submitted order")

        self.raise_event("ItemRemoved", {
            "product_id": product_id,
        })

    def submit(self) -> None:
        """Submit the order."""
        if self.status != "draft":
            raise ValueError("Order already submitted")
        if not self.items:
            raise ValueError("Cannot submit empty order")

        self.raise_event("OrderSubmitted", {
            "total": self.total,
        })

    def cancel(self, reason: str) -> None:
        """Cancel the order."""
        if self.status == "cancelled":
            raise ValueError("Order already cancelled")

        self.raise_event("OrderCancelled", {
            "reason": reason,
        })

    # Event handlers
    def _on_order_created(self, event: Event) -> None:
        self.customer_id = event.data["customer_id"]
        self.status = "draft"

    def _on_item_added(self, event: Event) -> None:
        self.items.append({
            "product_id": event.data["product_id"],
            "quantity": event.data["quantity"],
            "price": event.data["price"],
        })
        self._recalculate_total()

    def _on_item_removed(self, event: Event) -> None:
        product_id = event.data["product_id"]
        self.items = [i for i in self.items if i["product_id"] != product_id]
        self._recalculate_total()

    def _on_order_submitted(self, event: Event) -> None:
        self.status = "submitted"

    def _on_order_cancelled(self, event: Event) -> None:
        self.status = "cancelled"

    def _recalculate_total(self) -> None:
        self.total = sum(
            item["quantity"] * item["price"]
            for item in self.items
        )
