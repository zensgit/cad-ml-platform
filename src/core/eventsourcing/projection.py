"""Event Projections.

Provides event projection capabilities:
- Read model building from events
- Projection handlers
- Checkpoint management
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar

from src.core.eventsourcing.store import Event, EventStore, get_event_store

logger = logging.getLogger(__name__)


@dataclass
class ProjectionCheckpoint:
    """Checkpoint for projection progress."""
    projection_name: str
    position: int
    updated_at: datetime = field(default_factory=datetime.utcnow)


class Projection(ABC):
    """Base class for event projections."""

    def __init__(self, name: str):
        self.name = name
        self._handlers: Dict[str, Callable[[Event], None]] = {}
        self._position = 0

    @property
    def position(self) -> int:
        return self._position

    def handles(self, event_type: str) -> Callable:
        """Decorator to register event handlers."""
        def decorator(func: Callable[[Event], None]) -> Callable[[Event], None]:
            self._handlers[event_type] = func
            return func
        return decorator

    def register_handler(
        self,
        event_type: str,
        handler: Callable[[Event], None],
    ) -> None:
        """Register an event handler."""
        self._handlers[event_type] = handler

    async def handle(self, event: Event) -> bool:
        """Handle an event.

        Returns:
            True if event was handled.
        """
        handler = self._handlers.get(event.event_type)
        if handler:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
            self._position += 1
            return True
        return False

    def get_handled_event_types(self) -> List[str]:
        """Get list of event types this projection handles."""
        return list(self._handlers.keys())


class ProjectionManager:
    """Manages projection execution."""

    def __init__(
        self,
        event_store: Optional[EventStore] = None,
        batch_size: int = 100,
    ):
        self._event_store = event_store or get_event_store()
        self._batch_size = batch_size
        self._projections: Dict[str, Projection] = {}
        self._checkpoints: Dict[str, ProjectionCheckpoint] = {}
        self._running = False

    def register(self, projection: Projection) -> None:
        """Register a projection."""
        self._projections[projection.name] = projection
        self._checkpoints[projection.name] = ProjectionCheckpoint(
            projection_name=projection.name,
            position=0,
        )

    def get_projection(self, name: str) -> Optional[Projection]:
        """Get a projection by name."""
        return self._projections.get(name)

    async def rebuild(self, projection_name: str) -> int:
        """Rebuild a projection from the beginning.

        Args:
            projection_name: Name of projection to rebuild.

        Returns:
            Number of events processed.
        """
        projection = self._projections.get(projection_name)
        if not projection:
            raise ValueError(f"Projection not found: {projection_name}")

        # Reset checkpoint
        self._checkpoints[projection_name] = ProjectionCheckpoint(
            projection_name=projection_name,
            position=0,
        )

        return await self._catch_up_projection(projection)

    async def catch_up(self, projection_name: Optional[str] = None) -> Dict[str, int]:
        """Catch up projections to current events.

        Args:
            projection_name: Specific projection to catch up, or all if None.

        Returns:
            Dict mapping projection names to events processed.
        """
        results = {}

        if projection_name:
            projection = self._projections.get(projection_name)
            if projection:
                results[projection_name] = await self._catch_up_projection(projection)
        else:
            for name, projection in self._projections.items():
                results[name] = await self._catch_up_projection(projection)

        return results

    async def _catch_up_projection(self, projection: Projection) -> int:
        """Catch up a single projection."""
        checkpoint = self._checkpoints.get(projection.name)
        if not checkpoint:
            return 0

        total_processed = 0
        position = checkpoint.position
        event_types = projection.get_handled_event_types()

        while True:
            events = await self._event_store.get_all_events(
                from_position=position,
                batch_size=self._batch_size,
                event_types=event_types if event_types else None,
            )

            if not events:
                break

            for event in events:
                handled = await projection.handle(event)
                if handled:
                    total_processed += 1
                position += 1

            # Update checkpoint
            checkpoint.position = position
            checkpoint.updated_at = datetime.utcnow()

        return total_processed

    async def start_live(self, poll_interval: float = 1.0) -> None:
        """Start processing events in real-time."""
        self._running = True
        logger.info("Starting live projection processing")

        while self._running:
            await self.catch_up()
            await asyncio.sleep(poll_interval)

    def stop_live(self) -> None:
        """Stop live processing."""
        self._running = False


# Example Projections

class OrderSummaryProjection(Projection):
    """Projection that maintains order summaries."""

    def __init__(self):
        super().__init__("order_summary")
        self.orders: Dict[str, Dict[str, Any]] = {}

        self.register_handler("OrderCreated", self._on_order_created)
        self.register_handler("ItemAdded", self._on_item_added)
        self.register_handler("OrderSubmitted", self._on_order_submitted)
        self.register_handler("OrderCancelled", self._on_order_cancelled)

    def _on_order_created(self, event: Event) -> None:
        self.orders[event.aggregate_id] = {
            "order_id": event.aggregate_id,
            "customer_id": event.data["customer_id"],
            "status": "draft",
            "item_count": 0,
            "total": 0.0,
            "created_at": event.timestamp,
        }

    def _on_item_added(self, event: Event) -> None:
        order = self.orders.get(event.aggregate_id)
        if order:
            order["item_count"] += 1
            order["total"] += event.data["quantity"] * event.data["price"]

    def _on_order_submitted(self, event: Event) -> None:
        order = self.orders.get(event.aggregate_id)
        if order:
            order["status"] = "submitted"

    def _on_order_cancelled(self, event: Event) -> None:
        order = self.orders.get(event.aggregate_id)
        if order:
            order["status"] = "cancelled"

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order summary."""
        return self.orders.get(order_id)

    def get_orders_by_customer(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get orders for a customer."""
        return [
            o for o in self.orders.values()
            if o["customer_id"] == customer_id
        ]

    def get_orders_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get orders by status."""
        return [o for o in self.orders.values() if o["status"] == status]


class CustomerStatsProjection(Projection):
    """Projection that maintains customer statistics."""

    def __init__(self):
        super().__init__("customer_stats")
        self.stats: Dict[str, Dict[str, Any]] = {}

        self.register_handler("OrderCreated", self._on_order_created)
        self.register_handler("OrderSubmitted", self._on_order_submitted)
        self.register_handler("OrderCancelled", self._on_order_cancelled)

    def _ensure_customer(self, customer_id: str) -> Dict[str, Any]:
        if customer_id not in self.stats:
            self.stats[customer_id] = {
                "customer_id": customer_id,
                "total_orders": 0,
                "submitted_orders": 0,
                "cancelled_orders": 0,
                "total_spent": 0.0,
            }
        return self.stats[customer_id]

    def _on_order_created(self, event: Event) -> None:
        customer_id = event.data["customer_id"]
        stats = self._ensure_customer(customer_id)
        stats["total_orders"] += 1

    def _on_order_submitted(self, event: Event) -> None:
        # Need to look up customer from order
        pass

    def _on_order_cancelled(self, event: Event) -> None:
        pass

    def get_customer_stats(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a customer."""
        return self.stats.get(customer_id)
