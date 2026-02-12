"""Tests for eventsourcing aggregate module to improve coverage."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.eventsourcing.aggregate import (
    Aggregate,
    AggregateRepository,
    Order,
)
from src.core.eventsourcing.store import (
    Event,
    EventStream,
    InMemoryEventStore,
    create_event,
)


class TestAggregate:
    """Tests for Aggregate base class."""

    def test_aggregate_type_abstract(self):
        """Test aggregate_type is abstract (line 47)."""
        # Verify Aggregate has abstract aggregate_type property
        import inspect
        assert hasattr(Aggregate, "aggregate_type")

        # Cannot instantiate Aggregate directly due to abstract method
        # The test below verifies Order (concrete) works
        order = Order("test-id")
        assert order.aggregate_type == "Order"


class TestAggregateRepository:
    """Tests for AggregateRepository class."""

    @pytest.fixture
    def store(self):
        """Create an InMemoryEventStore."""
        return InMemoryEventStore()

    @pytest.fixture
    def repo(self, store):
        """Create an Order repository."""
        return AggregateRepository(Order, store)

    @pytest.mark.asyncio
    async def test_get_returns_aggregate_with_events(self, store, repo):
        """Test get loads aggregate from events (lines 114-124)."""
        # Add some events to the store
        events = [
            create_event("OrderCreated", "order-1", "Order", {"customer_id": "cust-1"}),
            create_event("ItemAdded", "order-1", "Order", {
                "product_id": "prod-1",
                "quantity": 2,
                "price": 10.0,
            }),
        ]
        await store.append("order-1", "Order", events)

        # Get the aggregate
        order = await repo.get("order-1")

        assert order is not None
        assert order.customer_id == "cust-1"
        assert len(order.items) == 1
        assert order.total == 20.0

    @pytest.mark.asyncio
    async def test_get_returns_none_for_empty_stream(self, store, repo):
        """Test get returns None when no events (lines 120-121)."""
        order = await repo.get("non-existent")
        assert order is None

    @pytest.mark.asyncio
    async def test_save_does_nothing_without_uncommitted_events(self, store, repo):
        """Test save returns early if no uncommitted events (line 134)."""
        order = Order("order-1")
        # Don't raise any events

        await repo.save(order)

        # Verify no events were stored
        stream = await store.get_stream("order-1", "Order")
        assert len(stream.events) == 0

    @pytest.mark.asyncio
    async def test_exists_returns_true_for_existing_aggregate(self, store, repo):
        """Test exists returns True when aggregate exists (lines 147-152)."""
        # Add events for an order
        events = [
            create_event("OrderCreated", "order-1", "Order", {"customer_id": "cust-1"}),
        ]
        await store.append("order-1", "Order", events)

        result = await repo.exists("order-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_exists_returns_false_for_missing_aggregate(self, store, repo):
        """Test exists returns False when aggregate doesn't exist."""
        result = await repo.exists("non-existent")
        assert result is False


class TestOrderAggregate:
    """Tests for Order aggregate implementation."""

    def test_create_order_already_created_error(self):
        """Test create raises error if already created (line 180)."""
        order = Order("order-1")
        order.create("cust-1")
        order.add_item("prod-1", 1, 10.0)
        order.submit()  # Change status to "submitted"

        with pytest.raises(ValueError, match="Order already created"):
            order.create("cust-2")

    def test_add_item_to_submitted_order_error(self):
        """Test add_item raises error for submitted order (line 189)."""
        order = Order("order-1")
        order.create("cust-1")
        order.add_item("prod-1", 1, 10.0)
        order.submit()

        with pytest.raises(ValueError, match="Cannot modify submitted order"):
            order.add_item("prod-2", 1, 5.0)

    def test_remove_item_from_order(self):
        """Test remove_item works correctly (lines 199-202)."""
        order = Order("order-1")
        order.create("cust-1")
        order.add_item("prod-1", 2, 10.0)
        order.add_item("prod-2", 1, 5.0)

        assert len(order.items) == 2
        assert order.total == 25.0

        order.remove_item("prod-1")

        assert len(order.items) == 1
        assert order.items[0]["product_id"] == "prod-2"
        assert order.total == 5.0

    def test_remove_item_from_submitted_order_error(self):
        """Test remove_item raises error for submitted order (lines 199-200)."""
        order = Order("order-1")
        order.create("cust-1")
        order.add_item("prod-1", 1, 10.0)
        order.submit()

        with pytest.raises(ValueError, match="Cannot modify submitted order"):
            order.remove_item("prod-1")

    def test_submit_already_submitted_error(self):
        """Test submit raises error if already submitted (line 209)."""
        order = Order("order-1")
        order.create("cust-1")
        order.add_item("prod-1", 1, 10.0)
        order.submit()

        with pytest.raises(ValueError, match="Order already submitted"):
            order.submit()

    def test_submit_empty_order_error(self):
        """Test submit raises error for empty order (line 211)."""
        order = Order("order-1")
        order.create("cust-1")

        with pytest.raises(ValueError, match="Cannot submit empty order"):
            order.submit()

    def test_cancel_order(self):
        """Test cancel order works (lines 219-222)."""
        order = Order("order-1")
        order.create("cust-1")
        order.add_item("prod-1", 1, 10.0)

        order.cancel("Changed my mind")

        assert order.status == "cancelled"

    def test_cancel_already_cancelled_error(self):
        """Test cancel raises error if already cancelled (lines 219-220)."""
        order = Order("order-1")
        order.create("cust-1")
        order.cancel("Changed my mind")

        with pytest.raises(ValueError, match="Order already cancelled"):
            order.cancel("Again")

    def test_on_item_removed_handler(self):
        """Test _on_item_removed event handler (lines 240-242)."""
        order = Order("order-1")
        order.create("cust-1")
        order.add_item("prod-1", 2, 10.0)
        order.add_item("prod-2", 3, 5.0)

        # Total before removal
        assert order.total == 35.0

        # Remove prod-1
        order.remove_item("prod-1")

        # Total after removal
        assert order.total == 15.0
        assert len(order.items) == 1

    def test_on_order_cancelled_handler(self):
        """Test _on_order_cancelled event handler (line 248)."""
        order = Order("order-1")
        order.create("cust-1")

        assert order.status == "draft"

        order.cancel("Test cancellation")

        assert order.status == "cancelled"


class TestAggregateEventHandling:
    """Tests for aggregate event handling."""

    def test_apply_with_unknown_event_type(self):
        """Test apply ignores unknown event types."""
        order = Order("order-1")

        # Create an event with unknown type
        event = Event(
            event_id="evt-1",
            event_type="UnknownEventType",
            aggregate_id="order-1",
            aggregate_type="Order",
            data={},
            version=1,
        )

        # Should not raise, just update version
        order.apply(event)
        assert order.version == 1

    def test_get_uncommitted_events_returns_copy(self):
        """Test get_uncommitted_events returns a copy."""
        order = Order("order-1")
        order.create("cust-1")

        events1 = order.get_uncommitted_events()
        events2 = order.get_uncommitted_events()

        # Should be equal but different objects
        assert events1 == events2
        assert events1 is not events2

    def test_mark_events_committed_clears_list(self):
        """Test mark_events_committed clears uncommitted events."""
        order = Order("order-1")
        order.create("cust-1")
        order.add_item("prod-1", 1, 10.0)

        assert len(order.get_uncommitted_events()) == 2

        order.mark_events_committed()

        assert len(order.get_uncommitted_events()) == 0

    def test_load_from_history_adjusts_version(self):
        """Test load_from_history adjusts event versions if needed."""
        order = Order("order-1")

        # Create events with version <= current version
        events = [
            Event(
                event_id="evt-1",
                event_type="OrderCreated",
                aggregate_id="order-1",
                aggregate_type="Order",
                data={"customer_id": "cust-1"},
                version=0,  # <= current version (0)
            ),
        ]

        order.load_from_history(events)

        # Version should have been adjusted
        assert order.version == 1


class TestAggregateRepositorySave:
    """Tests for AggregateRepository save functionality."""

    @pytest.fixture
    def store(self):
        return InMemoryEventStore()

    @pytest.fixture
    def repo(self, store):
        return AggregateRepository(Order, store)

    @pytest.mark.asyncio
    async def test_save_with_events(self, store, repo):
        """Test save persists uncommitted events."""
        order = Order("order-1")
        order.create("cust-1")
        order.add_item("prod-1", 2, 10.0)

        await repo.save(order)

        # Verify events were persisted
        stream = await store.get_stream("order-1", "Order")
        assert len(stream.events) == 2

        # Verify uncommitted events were cleared
        assert len(order.get_uncommitted_events()) == 0

    @pytest.mark.asyncio
    async def test_save_uses_default_event_store(self):
        """Test save uses default event store if none provided."""
        import src.core.eventsourcing.store as store_module

        # Set up a mock store
        mock_store = AsyncMock()
        mock_store.append = AsyncMock()
        store_module._event_store = mock_store

        try:
            repo = AggregateRepository(Order)
            order = Order("order-1")
            order.create("cust-1")

            await repo.save(order)

            mock_store.append.assert_called_once()
        finally:
            # Cleanup
            store_module._event_store = None
