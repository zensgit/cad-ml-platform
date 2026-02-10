"""Tests for eventsourcing store module to improve coverage."""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

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


class TestEvent:
    """Tests for Event class."""

    def test_event_to_dict(self):
        """Test Event.to_dict serialization (line 40)."""
        event = Event(
            event_id="evt-1",
            event_type="TestEvent",
            aggregate_id="agg-1",
            aggregate_type="TestAggregate",
            data={"key": "value"},
            metadata={"user": "test"},
            version=1,
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
        )

        result = event.to_dict()

        assert result["event_id"] == "evt-1"
        assert result["event_type"] == "TestEvent"
        assert result["aggregate_id"] == "agg-1"
        assert result["aggregate_type"] == "TestAggregate"
        assert result["data"] == {"key": "value"}
        assert result["metadata"] == {"user": "test"}
        assert result["version"] == 1
        assert result["timestamp"] == "2025-01-01T12:00:00"

    def test_event_from_dict(self):
        """Test Event.from_dict deserialization (line 54)."""
        data = {
            "event_id": "evt-1",
            "event_type": "TestEvent",
            "aggregate_id": "agg-1",
            "aggregate_type": "TestAggregate",
            "data": {"key": "value"},
            "metadata": {"user": "test"},
            "version": 2,
            "timestamp": "2025-01-01T12:00:00",
        }

        event = Event.from_dict(data)

        assert event.event_id == "evt-1"
        assert event.event_type == "TestEvent"
        assert event.aggregate_id == "agg-1"
        assert event.aggregate_type == "TestAggregate"
        assert event.data == {"key": "value"}
        assert event.metadata == {"user": "test"}
        assert event.version == 2
        assert event.timestamp == datetime(2025, 1, 1, 12, 0, 0)

    def test_event_from_dict_without_optional_fields(self):
        """Test Event.from_dict with missing optional fields."""
        data = {
            "event_id": "evt-1",
            "event_type": "TestEvent",
            "aggregate_id": "agg-1",
            "aggregate_type": "TestAggregate",
            "data": {"key": "value"},
            "timestamp": "2025-01-01T12:00:00",
        }

        event = Event.from_dict(data)

        assert event.metadata == {}  # Default
        assert event.version == 1  # Default


class TestEventStream:
    """Tests for EventStream class."""

    def test_stream_append(self):
        """Test EventStream.append method (lines 76-77)."""
        stream = EventStream(
            aggregate_id="agg-1",
            aggregate_type="TestAggregate",
        )

        assert stream.version == 0

        event = Event(
            event_id="evt-1",
            event_type="TestEvent",
            aggregate_id="agg-1",
            aggregate_type="TestAggregate",
            data={},
            version=1,
        )

        stream.append(event)

        assert len(stream.events) == 1
        assert stream.version == 1

    def test_stream_append_updates_version(self):
        """Test EventStream.append updates version correctly."""
        stream = EventStream(
            aggregate_id="agg-1",
            aggregate_type="TestAggregate",
        )

        event1 = Event(
            event_id="evt-1",
            event_type="TestEvent",
            aggregate_id="agg-1",
            aggregate_type="TestAggregate",
            data={},
            version=1,
        )

        event2 = Event(
            event_id="evt-2",
            event_type="TestEvent",
            aggregate_id="agg-1",
            aggregate_type="TestAggregate",
            data={},
            version=2,
        )

        stream.append(event1)
        assert stream.version == 1

        stream.append(event2)
        assert stream.version == 2


class TestAppendResult:
    """Tests for AppendResult class."""

    def test_append_result_version_property(self):
        """Test AppendResult.version backward-compatible property."""
        result = AppendResult(
            success=True,
            new_version=5,
            events_appended=2,
        )

        # version should be an alias for new_version
        assert result.version == 5
        assert result.new_version == 5


class TestEventStoreAbstract:
    """Tests for abstract EventStore methods (lines 124, 145, 164, 173)."""

    def test_abstract_methods_exist(self):
        """Test EventStore abstract methods are defined."""
        # Verify the abstract methods exist
        assert hasattr(EventStore, "append")
        assert hasattr(EventStore, "get_stream")
        assert hasattr(EventStore, "get_all_events")
        assert hasattr(EventStore, "get_current_version")


class TestInMemoryEventStore:
    """Tests for InMemoryEventStore class."""

    @pytest.fixture
    def store(self):
        """Create a fresh event store."""
        return InMemoryEventStore()

    @pytest.mark.asyncio
    async def test_get_all_events_with_type_filter(self, store):
        """Test get_all_events with event type filter (line 256)."""
        events1 = [
            create_event("TypeA", "agg-1", "Aggregate", {"key": "val1"}),
        ]
        events2 = [
            create_event("TypeB", "agg-2", "Aggregate", {"key": "val2"}),
        ]
        events3 = [
            create_event("TypeA", "agg-3", "Aggregate", {"key": "val3"}),
        ]

        await store.append("agg-1", "Aggregate", events1)
        await store.append("agg-2", "Aggregate", events2)
        await store.append("agg-3", "Aggregate", events3)

        # Filter by TypeA only
        filtered = await store.get_all_events(event_types=["TypeA"])

        assert len(filtered) == 2
        assert all(e.event_type == "TypeA" for e in filtered)

    @pytest.mark.asyncio
    async def test_get_current_version_empty_stream(self, store):
        """Test get_current_version for non-existent stream (lines 265-266)."""
        version = await store.get_current_version("non-existent", "Aggregate")
        assert version == 0

    @pytest.mark.asyncio
    async def test_get_current_version_with_events(self, store):
        """Test get_current_version after appending events."""
        events = [
            create_event("TestEvent", "agg-1", "Aggregate", {}),
            create_event("TestEvent", "agg-1", "Aggregate", {}),
        ]
        await store.append("agg-1", "Aggregate", events)

        version = await store.get_current_version("agg-1", "Aggregate")
        assert version == 2


class TestCreateEvent:
    """Tests for create_event helper function."""

    def test_create_event_standard_usage(self):
        """Test create_event with standard arguments."""
        event = create_event(
            event_type="TestEvent",
            aggregate_id="agg-1",
            aggregate_type="Aggregate",
            data={"key": "value"},
            metadata={"user": "test"},
        )

        assert event.event_type == "TestEvent"
        assert event.aggregate_id == "agg-1"
        assert event.aggregate_type == "Aggregate"
        assert event.data == {"key": "value"}
        assert event.metadata == {"user": "test"}
        assert event.event_id is not None

    def test_create_event_legacy_positional_usage(self):
        """Test create_event with legacy positional arguments."""
        # Legacy: (event_type, data, aggregate_id, aggregate_type)
        event = create_event(
            "TestEvent",
            {"key": "value"},  # This is data (dict)
            "agg-1",  # This is aggregate_id (str)
            "Aggregate",  # This is aggregate_type (str)
        )

        assert event.event_type == "TestEvent"
        assert event.data == {"key": "value"}
        assert event.aggregate_id == "agg-1"
        assert event.aggregate_type == "Aggregate"


class TestGlobalEventStore:
    """Tests for global event store functions."""

    def test_get_event_store_creates_default(self):
        """Test get_event_store creates default InMemoryEventStore."""
        import src.core.eventsourcing.store as store_module

        # Reset global store
        store_module._event_store = None

        store = get_event_store()
        assert store is not None
        assert isinstance(store, InMemoryEventStore)

    def test_configure_event_store(self):
        """Test configure_event_store sets global store (line 308)."""
        import src.core.eventsourcing.store as store_module

        custom_store = InMemoryEventStore()

        configure_event_store(custom_store)

        assert store_module._event_store is custom_store

        # Cleanup
        store_module._event_store = None

    def test_get_event_store_returns_configured(self):
        """Test get_event_store returns configured store."""
        import src.core.eventsourcing.store as store_module

        custom_store = InMemoryEventStore()
        store_module._event_store = custom_store

        result = get_event_store()
        assert result is custom_store

        # Cleanup
        store_module._event_store = None


class TestConcurrencyError:
    """Tests for ConcurrencyError exception."""

    def test_concurrency_error_message(self):
        """Test ConcurrencyError can be raised with message."""
        with pytest.raises(ConcurrencyError, match="Version mismatch"):
            raise ConcurrencyError("Version mismatch")

    @pytest.mark.asyncio
    async def test_concurrency_error_on_version_mismatch(self):
        """Test ConcurrencyError is raised on version mismatch."""
        store = InMemoryEventStore()

        events = [create_event("TestEvent", "agg-1", "Aggregate", {})]
        await store.append("agg-1", "Aggregate", events)

        with pytest.raises(ConcurrencyError, match="Expected version 0"):
            await store.append(
                "agg-1",
                "Aggregate",
                events,
                expected_version=0,  # Should be 1
            )


class TestEventStreamFiltering:
    """Tests for EventStream filtering functionality."""

    @pytest.mark.asyncio
    async def test_get_stream_with_version_range(self):
        """Test get_stream with from_version and to_version filters."""
        store = InMemoryEventStore()

        events = [
            create_event("Event1", "agg-1", "Aggregate", {"seq": 1}),
            create_event("Event2", "agg-1", "Aggregate", {"seq": 2}),
            create_event("Event3", "agg-1", "Aggregate", {"seq": 3}),
            create_event("Event4", "agg-1", "Aggregate", {"seq": 4}),
        ]
        await store.append("agg-1", "Aggregate", events)

        # Get events from version 2 to 3
        stream = await store.get_stream("agg-1", "Aggregate", from_version=2, to_version=3)

        # Check we only got versions 2 and 3
        assert len(stream.events) == 2
        assert stream.events[0].version == 2
        assert stream.events[1].version == 3
