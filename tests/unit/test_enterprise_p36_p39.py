"""Unit tests for P36-P39: Event Sourcing, CQRS, Versioning, Rate Limiting."""

import asyncio
import time
import pytest
from datetime import datetime
from typing import Any, Dict, List, Optional


# ============================================================================
# P36: Event Sourcing Tests
# ============================================================================

class TestEventStore:
    """Test event store functionality."""

    @pytest.fixture
    def event_store(self):
        from src.core.eventsourcing import InMemoryEventStore
        return InMemoryEventStore()

    @pytest.fixture
    def sample_events(self):
        from src.core.eventsourcing import create_event
        return [
            create_event("OrderCreated", {"customer_id": "cust-1"}, "order-1", "Order"),
            create_event("ItemAdded", {"product_id": "prod-1", "quantity": 2, "price": 10.0}, "order-1", "Order"),
            create_event("OrderSubmitted", {}, "order-1", "Order"),
        ]

    @pytest.mark.asyncio
    async def test_append_and_get_events(self, event_store, sample_events):
        """Test appending and retrieving events."""
        result = await event_store.append(
            aggregate_id="order-1",
            aggregate_type="Order",
            events=sample_events,
        )
        assert result.success
        assert result.version == 3

        stream = await event_store.get_stream("order-1", "Order")
        assert len(stream.events) == 3
        assert stream.events[0].event_type == "OrderCreated"

    @pytest.mark.asyncio
    async def test_optimistic_concurrency(self, event_store, sample_events):
        """Test optimistic concurrency control."""
        # First append
        await event_store.append(
            aggregate_id="order-1",
            aggregate_type="Order",
            events=sample_events[:1],
            expected_version=0,
        )

        # Second append with wrong version should fail
        from src.core.eventsourcing import ConcurrencyError, create_event
        with pytest.raises(ConcurrencyError):
            await event_store.append(
                aggregate_id="order-1",
                aggregate_type="Order",
                events=[create_event("ItemAdded", {}, "order-1", "Order")],
                expected_version=0,  # Should be 1
            )

    @pytest.mark.asyncio
    async def test_get_all_events(self, event_store, sample_events):
        """Test getting all events across aggregates."""
        # Add events for multiple orders
        await event_store.append("order-1", "Order", sample_events[:2])
        await event_store.append("order-2", "Order", sample_events[2:])

        all_events = await event_store.get_all_events()
        assert len(all_events) >= 2


class TestAggregate:
    """Test aggregate functionality."""

    @pytest.fixture
    def order(self):
        from src.core.eventsourcing import Order
        return Order()

    def test_create_order(self, order):
        """Test creating an order."""
        order.create("customer-1")
        assert order.customer_id == "customer-1"
        assert order.status == "draft"
        assert len(order.get_uncommitted_events()) == 1

    def test_add_item(self, order):
        """Test adding items to order."""
        order.create("customer-1")
        order.add_item("product-1", 2, 10.0)
        assert len(order.items) == 1
        assert order.total == 20.0

    def test_submit_order(self, order):
        """Test submitting an order."""
        order.create("customer-1")
        order.add_item("product-1", 1, 10.0)
        order.submit()
        assert order.status == "submitted"

    def test_load_from_history(self, order):
        """Test loading aggregate from event history."""
        from src.core.eventsourcing import create_event
        events = [
            create_event("OrderCreated", {"customer_id": "cust-1"}, "order-1", "Order"),
            create_event("ItemAdded", {"product_id": "prod-1", "quantity": 2, "price": 10.0}, "order-1", "Order"),
        ]

        order.load_from_history(events)
        assert order.customer_id == "cust-1"
        assert len(order.items) == 1
        assert order.version == 2


class TestProjection:
    """Test projection functionality."""

    @pytest.fixture
    def projection(self):
        from src.core.eventsourcing import OrderSummaryProjection
        return OrderSummaryProjection()

    @pytest.mark.asyncio
    async def test_projection_handles_events(self, projection):
        """Test projection handling events."""
        from src.core.eventsourcing import create_event

        event = create_event(
            "OrderCreated",
            {"customer_id": "cust-1"},
            "order-1",
            "Order",
        )

        handled = await projection.handle(event)
        assert handled
        assert projection.position == 1

        order = projection.get_order("order-1")
        assert order is not None
        assert order["customer_id"] == "cust-1"

    @pytest.mark.asyncio
    async def test_projection_item_added(self, projection):
        """Test projection tracking item additions."""
        from src.core.eventsourcing import create_event

        # Create order first
        await projection.handle(create_event(
            "OrderCreated",
            {"customer_id": "cust-1"},
            "order-1",
            "Order",
        ))

        # Add item
        await projection.handle(create_event(
            "ItemAdded",
            {"product_id": "prod-1", "quantity": 2, "price": 10.0},
            "order-1",
            "Order",
        ))

        order = projection.get_order("order-1")
        assert order["item_count"] == 1
        assert order["total"] == 20.0


class TestSnapshot:
    """Test snapshot functionality."""

    @pytest.fixture
    def snapshot_store(self):
        from src.core.eventsourcing import InMemorySnapshotStore
        return InMemorySnapshotStore()

    @pytest.mark.asyncio
    async def test_save_and_get_snapshot(self, snapshot_store):
        """Test saving and retrieving snapshots."""
        from src.core.eventsourcing import Snapshot

        snapshot = Snapshot(
            aggregate_id="order-1",
            aggregate_type="Order",
            version=10,
            state={"customer_id": "cust-1", "status": "submitted"},
        )

        await snapshot_store.save(snapshot)

        retrieved = await snapshot_store.get("order-1", "Order")
        assert retrieved is not None
        assert retrieved.version == 10
        assert retrieved.state["customer_id"] == "cust-1"

    def test_event_count_strategy(self):
        """Test event count snapshot strategy."""
        from src.core.eventsourcing import EventCountStrategy, Order

        strategy = EventCountStrategy(threshold=5)
        order = Order()

        assert not strategy.should_snapshot(order, 3)
        assert strategy.should_snapshot(order, 5)
        assert strategy.should_snapshot(order, 10)


# ============================================================================
# P37: CQRS Tests
# ============================================================================

class TestCommandBus:
    """Test command bus functionality."""

    @pytest.fixture
    def command_bus(self):
        from src.core.cqrs import CommandBus
        return CommandBus()

    @pytest.mark.asyncio
    async def test_command_handler_registration(self, command_bus):
        """Test registering and dispatching commands."""
        from src.core.cqrs import Command, CommandHandler, CommandResult
        from dataclasses import dataclass

        @dataclass
        class TestCommand(Command):
            value: str = ""

        class TestHandler(CommandHandler[TestCommand]):
            async def handle(self, command: TestCommand) -> CommandResult:
                return CommandResult(
                    success=True,
                    command_id=command.command_id,
                    data={"value": command.value},
                )

        command_bus.register_handler(TestCommand, TestHandler())

        result = await command_bus.dispatch(TestCommand(value="test"))
        assert result.success
        assert result.data["value"] == "test"

    @pytest.mark.asyncio
    async def test_command_validation(self, command_bus):
        """Test command validation."""
        from src.core.cqrs import Command, CommandHandler, CommandResult, CommandValidator
        from dataclasses import dataclass

        @dataclass
        class ValidatedCommand(Command):
            value: str = ""

        class ValueValidator(CommandValidator[ValidatedCommand]):
            def validate(self, command: ValidatedCommand) -> List[str]:
                if not command.value:
                    return ["value is required"]
                return []

        class ValidatedHandler(CommandHandler[ValidatedCommand]):
            async def handle(self, command: ValidatedCommand) -> CommandResult:
                return CommandResult(success=True, command_id=command.command_id)

        command_bus.register_handler(ValidatedCommand, ValidatedHandler())
        command_bus.register_validator(ValidatedCommand, ValueValidator())

        # Should fail validation
        result = await command_bus.dispatch(ValidatedCommand(value=""))
        assert not result.success
        assert "value is required" in result.error

    @pytest.mark.asyncio
    async def test_no_handler_error(self, command_bus):
        """Test error when no handler registered."""
        from src.core.cqrs import Command
        from dataclasses import dataclass

        @dataclass
        class UnhandledCommand(Command):
            pass

        result = await command_bus.dispatch(UnhandledCommand())
        assert not result.success
        assert "No handler" in result.error


class TestQueryBus:
    """Test query bus functionality."""

    @pytest.fixture
    def query_bus(self):
        from src.core.cqrs import QueryBus
        return QueryBus()

    @pytest.mark.asyncio
    async def test_query_handler(self, query_bus):
        """Test registering and dispatching queries."""
        from src.core.cqrs import Query, QueryHandler, QueryResult
        from dataclasses import dataclass

        @dataclass
        class TestQuery(Query):
            filter: str = ""

        class TestHandler(QueryHandler[TestQuery, List[str]]):
            async def handle(self, query: TestQuery) -> QueryResult[List[str]]:
                return QueryResult(
                    success=True,
                    query_id=query.query_id,
                    data=["item1", "item2"],
                )

        query_bus.register_handler(TestQuery, TestHandler())

        result = await query_bus.dispatch(TestQuery(filter="test"))
        assert result.success
        assert len(result.data) == 2

    @pytest.mark.asyncio
    async def test_query_pagination(self, query_bus):
        """Test query with pagination."""
        from src.core.cqrs import Query, QueryHandler, QueryResult
        from dataclasses import dataclass

        @dataclass
        class PaginatedQuery(Query):
            page: int = 1
            page_size: int = 10

        class PaginatedHandler(QueryHandler[PaginatedQuery, List[int]]):
            async def handle(self, query: PaginatedQuery) -> QueryResult[List[int]]:
                all_items = list(range(100))
                start = (query.page - 1) * query.page_size
                end = start + query.page_size
                return QueryResult(
                    success=True,
                    query_id=query.query_id,
                    data=all_items[start:end],
                    total_count=100,
                    page=query.page,
                    page_size=query.page_size,
                )

        query_bus.register_handler(PaginatedQuery, PaginatedHandler())

        result = await query_bus.dispatch(PaginatedQuery(page=2, page_size=10))
        assert result.success
        assert result.total_count == 100
        assert result.data[0] == 10  # First item of page 2


# ============================================================================
# P38: API Versioning Tests
# ============================================================================

class TestSemanticVersion:
    """Test semantic version parsing and comparison."""

    def test_parse_version(self):
        """Test parsing version strings."""
        from src.core.versioning import SemanticVersion

        v = SemanticVersion.parse("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_parse_with_prefix(self):
        """Test parsing version with v prefix."""
        from src.core.versioning import SemanticVersion

        v = SemanticVersion.parse("v2.0.0")
        assert v.major == 2

    def test_parse_with_prerelease(self):
        """Test parsing version with prerelease."""
        from src.core.versioning import SemanticVersion

        v = SemanticVersion.parse("1.0.0-beta.1")
        assert v.prerelease == "beta.1"

    def test_version_comparison(self):
        """Test version comparison."""
        from src.core.versioning import SemanticVersion

        v1 = SemanticVersion.parse("1.0.0")
        v2 = SemanticVersion.parse("2.0.0")
        v3 = SemanticVersion.parse("1.1.0")

        assert v1 < v2
        assert v1 < v3
        assert v2 > v3

    def test_prerelease_comparison(self):
        """Test prerelease version comparison."""
        from src.core.versioning import SemanticVersion

        v1 = SemanticVersion.parse("1.0.0-alpha")
        v2 = SemanticVersion.parse("1.0.0-beta")
        v3 = SemanticVersion.parse("1.0.0")

        assert v1 < v2
        assert v1 < v3
        assert v2 < v3

    def test_version_compatibility(self):
        """Test version compatibility check."""
        from src.core.versioning import SemanticVersion

        v1 = SemanticVersion.parse("1.2.3")
        v2 = SemanticVersion.parse("1.5.0")
        v3 = SemanticVersion.parse("2.0.0")

        assert v1.is_compatible_with(v2)
        assert not v1.is_compatible_with(v3)


class TestVersionRange:
    """Test version range functionality."""

    def test_exact_version(self):
        """Test exact version range."""
        from src.core.versioning import SemanticVersion, VersionRange

        range_ = VersionRange.parse("1.0.0")
        assert range_.contains(SemanticVersion.parse("1.0.0"))
        assert not range_.contains(SemanticVersion.parse("1.0.1"))

    def test_minimum_version(self):
        """Test minimum version range."""
        from src.core.versioning import SemanticVersion, VersionRange

        range_ = VersionRange.parse(">=1.0.0")
        assert range_.contains(SemanticVersion.parse("1.0.0"))
        assert range_.contains(SemanticVersion.parse("2.0.0"))
        assert not range_.contains(SemanticVersion.parse("0.9.0"))

    def test_caret_range(self):
        """Test caret range (^1.0.0)."""
        from src.core.versioning import SemanticVersion, VersionRange

        range_ = VersionRange.parse("^1.0.0")
        assert range_.contains(SemanticVersion.parse("1.0.0"))
        assert range_.contains(SemanticVersion.parse("1.9.9"))
        assert not range_.contains(SemanticVersion.parse("2.0.0"))

    def test_tilde_range(self):
        """Test tilde range (~1.0.0)."""
        from src.core.versioning import SemanticVersion, VersionRange

        range_ = VersionRange.parse("~1.0.0")
        assert range_.contains(SemanticVersion.parse("1.0.0"))
        assert range_.contains(SemanticVersion.parse("1.0.9"))
        assert not range_.contains(SemanticVersion.parse("1.1.0"))


class TestVersionedRouter:
    """Test versioned router functionality."""

    @pytest.fixture
    def router(self):
        from src.core.versioning import (
            VersionedRouter,
            VersioningStrategy,
            SemanticVersion,
        )
        return VersionedRouter(
            strategy=VersioningStrategy.URL_PATH,
            default_version=SemanticVersion.parse("1.0.0"),
        )

    def test_route_registration(self, router):
        """Test registering routes."""
        from src.core.versioning import VersionedRequest, VersionedResponse

        @router.get("/users", version_range=">=1.0.0")
        def get_users(request: VersionedRequest) -> VersionedResponse:
            return VersionedResponse(status_code=200, body={"users": []})

        assert len(router._routes) == 1

    def test_route_matching(self, router):
        """Test route matching with version."""
        from src.core.versioning import VersionedRequest, VersionedResponse

        @router.get("/users", version_range=">=1.0.0")
        def get_users(request: VersionedRequest) -> VersionedResponse:
            return VersionedResponse(status_code=200, body={"users": []})

        request = VersionedRequest(
            method="GET",
            path="/v1/users",
        )

        response = router.handle(request)
        assert response.status_code == 200

    def test_version_not_found(self, router):
        """Test handling of unsupported version."""
        from src.core.versioning import VersionedRequest, VersionedResponse

        @router.get("/users", version_range=">=2.0.0")
        def get_users(request: VersionedRequest) -> VersionedResponse:
            return VersionedResponse(status_code=200, body={"users": []})

        request = VersionedRequest(
            method="GET",
            path="/v1/users",
        )

        response = router.handle(request)
        assert response.status_code == 404


class TestSchemaMigration:
    """Test schema migration functionality."""

    @pytest.fixture
    def migrator(self):
        from src.core.versioning import SchemaMigrator
        return SchemaMigrator()

    def test_register_migration(self, migrator):
        """Test registering migrations."""
        migrator.register_migration(
            schema_name="User",
            from_version="1.0.0",
            to_version="2.0.0",
            description="Add email field",
            transform_request=lambda d: {**d, "email": d.get("email", "")},
        )

        assert "User" in migrator._migrations
        assert len(migrator._migrations["User"]) == 1

    def test_migrate_request(self, migrator):
        """Test request migration."""
        from src.core.versioning import SemanticVersion

        def add_email(data: Dict) -> Dict:
            return {**data, "email": "default@example.com"}

        migrator.register_migration(
            schema_name="User",
            from_version="1.0.0",
            to_version="2.0.0",
            description="Add email field",
            transform_request=add_email,
        )

        result = migrator.migrate_request(
            "User",
            {"name": "John"},
            SemanticVersion.parse("1.0.0"),
            SemanticVersion.parse("2.0.0"),
        )

        assert result["email"] == "default@example.com"


# ============================================================================
# P39: Rate Limiting Tests
# ============================================================================

class TestTokenBucket:
    """Test token bucket rate limiter."""

    @pytest.fixture
    def limiter(self):
        from src.core.ratelimit import TokenBucketLimiter
        return TokenBucketLimiter(
            capacity=10,
            refill_rate=1.0,
            refill_interval=1.0,
        )

    @pytest.mark.asyncio
    async def test_basic_acquire(self, limiter):
        """Test basic token acquisition."""
        result = await limiter.acquire("user-1")
        assert result.allowed
        assert result.remaining == 9

    @pytest.mark.asyncio
    async def test_exhaust_bucket(self, limiter):
        """Test exhausting the bucket."""
        # Consume all tokens
        for i in range(10):
            result = await limiter.acquire("user-1")
            assert result.allowed

        # Next request should be denied
        result = await limiter.acquire("user-1")
        assert not result.allowed
        assert result.retry_after > 0

    @pytest.mark.asyncio
    async def test_different_keys(self, limiter):
        """Test different keys have separate buckets."""
        # Exhaust user-1
        for _ in range(10):
            await limiter.acquire("user-1")

        # user-2 should still have tokens
        result = await limiter.acquire("user-2")
        assert result.allowed


class TestSlidingWindowCounter:
    """Test sliding window counter rate limiter."""

    @pytest.fixture
    def limiter(self):
        from src.core.ratelimit import SlidingWindowCounterLimiter
        return SlidingWindowCounterLimiter(
            limit=10,
            window_seconds=60.0,
        )

    @pytest.mark.asyncio
    async def test_basic_acquire(self, limiter):
        """Test basic sliding window acquisition."""
        result = await limiter.acquire("user-1")
        assert result.allowed
        assert result.remaining == 9

    @pytest.mark.asyncio
    async def test_window_limit(self, limiter):
        """Test hitting window limit."""
        for _ in range(10):
            await limiter.acquire("user-1")

        result = await limiter.acquire("user-1")
        assert not result.allowed


class TestFixedWindowCounter:
    """Test fixed window counter rate limiter."""

    @pytest.fixture
    def limiter(self):
        from src.core.ratelimit import FixedWindowCounterLimiter
        return FixedWindowCounterLimiter(
            limit=5,
            window_seconds=60.0,
        )

    @pytest.mark.asyncio
    async def test_fixed_window(self, limiter):
        """Test fixed window rate limiting."""
        for _ in range(5):
            result = await limiter.acquire("user-1")
            assert result.allowed

        result = await limiter.acquire("user-1")
        assert not result.allowed

    @pytest.mark.asyncio
    async def test_reset(self, limiter):
        """Test resetting rate limit."""
        for _ in range(5):
            await limiter.acquire("user-1")

        await limiter.reset("user-1")

        result = await limiter.acquire("user-1")
        assert result.allowed


class TestDistributedLimiter:
    """Test distributed rate limiting."""

    @pytest.fixture
    def redis(self):
        from src.core.ratelimit import InMemoryRedis
        return InMemoryRedis()

    @pytest.fixture
    def limiter(self, redis):
        from src.core.ratelimit import DistributedTokenBucketLimiter
        return DistributedTokenBucketLimiter(
            redis_client=redis,
            capacity=10,
            refill_rate=1.0,
        )

    @pytest.mark.asyncio
    async def test_distributed_acquire(self, limiter):
        """Test distributed token bucket."""
        result = await limiter.acquire("user-1")
        assert result.allowed

    @pytest.mark.asyncio
    async def test_distributed_exhaust(self, limiter):
        """Test exhausting distributed bucket."""
        for _ in range(10):
            await limiter.acquire("user-1")

        result = await limiter.acquire("user-1")
        assert not result.allowed


class TestMultiTierLimiter:
    """Test multi-tier rate limiting."""

    @pytest.fixture
    def multi_tier(self):
        from src.core.ratelimit import (
            MultiTierRateLimiter,
            TokenBucketLimiter,
            IPKeyExtractor,
            EndpointKeyExtractor,
        )

        limiter = MultiTierRateLimiter()

        # Global tier
        limiter.add_tier(
            name="global",
            limiter=TokenBucketLimiter(capacity=100, refill_rate=10),
            key_extractor=IPKeyExtractor(),
            priority=1,
        )

        # Per-endpoint tier
        limiter.add_tier(
            name="endpoint",
            limiter=TokenBucketLimiter(capacity=10, refill_rate=1),
            key_extractor=EndpointKeyExtractor(),
            priority=2,
        )

        return limiter

    @pytest.mark.asyncio
    async def test_multi_tier_acquire(self, multi_tier):
        """Test multi-tier rate limiting."""
        context = {
            "remote_addr": "192.168.1.1",
            "method": "GET",
            "path": "/api/users",
        }

        allowed, results = await multi_tier.acquire(context)
        assert allowed
        assert "global" in results
        assert "endpoint" in results


class TestKeyExtractors:
    """Test key extraction strategies."""

    def test_ip_extractor(self):
        """Test IP key extraction."""
        from src.core.ratelimit import IPKeyExtractor

        extractor = IPKeyExtractor()
        context = {"remote_addr": "192.168.1.1"}
        assert extractor.extract(context) == "192.168.1.1"

    def test_ip_from_forwarded_header(self):
        """Test IP extraction from X-Forwarded-For."""
        from src.core.ratelimit import IPKeyExtractor

        extractor = IPKeyExtractor()
        context = {
            "headers": {"X-Forwarded-For": "10.0.0.1, 192.168.1.1"},
            "remote_addr": "127.0.0.1",
        }
        assert extractor.extract(context) == "10.0.0.1"

    def test_user_extractor(self):
        """Test user key extraction."""
        from src.core.ratelimit import UserKeyExtractor

        extractor = UserKeyExtractor()
        context = {"user_id": "user-123"}
        assert extractor.extract(context) == "user:user-123"

    def test_api_key_extractor(self):
        """Test API key extraction."""
        from src.core.ratelimit import APIKeyExtractor

        extractor = APIKeyExtractor()
        context = {"headers": {"X-API-Key": "my-secret-key"}}
        key = extractor.extract(context)
        assert key.startswith("api:")


class TestRateLimitHeaders:
    """Test rate limit response headers."""

    @pytest.mark.asyncio
    async def test_headers_generation(self):
        """Test rate limit headers are generated correctly."""
        from src.core.ratelimit import TokenBucketLimiter

        limiter = TokenBucketLimiter(capacity=100, refill_rate=10)
        result = await limiter.acquire("user-1")

        headers = result.headers
        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers
        assert "X-RateLimit-Reset" in headers
        assert headers["X-RateLimit-Limit"] == "100"


# ============================================================================
# Integration Tests
# ============================================================================

class TestEventSourcingCQRSIntegration:
    """Test Event Sourcing + CQRS integration."""

    @pytest.mark.asyncio
    async def test_command_creates_events(self):
        """Test command handler creates events in event store."""
        from src.core.eventsourcing import InMemoryEventStore, AggregateRepository
        from src.core.cqrs import (
            CommandBus,
            CreateOrderCommand,
            CreateOrderHandler,
        )

        event_store = InMemoryEventStore()
        repository = AggregateRepository(event_store)
        handler = CreateOrderHandler(repository)

        command_bus = CommandBus()
        command_bus.register_handler(CreateOrderCommand, handler)

        result = await command_bus.dispatch(
            CreateOrderCommand(customer_id="cust-1")
        )

        assert result.success
        assert result.aggregate_id is not None

    @pytest.mark.asyncio
    async def test_query_reads_projection(self):
        """Test query handler reads from projection."""
        from src.core.eventsourcing import (
            OrderSummaryProjection,
            create_event,
        )
        from src.core.cqrs import (
            QueryBus,
            GetOrderQuery,
            GetOrderHandler,
        )

        # Build projection
        projection = OrderSummaryProjection()
        await projection.handle(create_event(
            "OrderCreated",
            {"customer_id": "cust-1"},
            "order-1",
            "Order",
        ))

        # Query through handler
        handler = GetOrderHandler(projection)
        query_bus = QueryBus()
        query_bus.register_handler(GetOrderQuery, handler)

        result = await query_bus.dispatch(GetOrderQuery(order_id="order-1"))
        assert result.success
        assert result.data["customer_id"] == "cust-1"


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
