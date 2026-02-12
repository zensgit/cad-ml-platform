"""Unit tests for Enterprise Platform P28-P31 modules.

P28: Multi-tenancy Isolation
P29: GraphQL Gateway
P30: WebSocket Real-time Push
P31: Full-text Search (Elasticsearch)
"""

import asyncio
import pytest
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch


# ============================================================================
# P28: Multi-tenancy Tests
# ============================================================================

class TestTenantContext:
    """Test TenantContext and context management."""

    def test_tenant_context_creation(self):
        """Test TenantContext dataclass creation."""
        from src.core.multitenancy.context import TenantContext

        ctx = TenantContext(
            tenant_id="tenant-123",
            tenant_name="Test Tenant",
        )

        assert ctx.tenant_id == "tenant-123"
        assert ctx.tenant_name == "Test Tenant"
        assert ctx.metadata == {}

    def test_tenant_context_with_quotas(self):
        """Test TenantContext with quota tracking."""
        from src.core.multitenancy.context import TenantContext

        ctx = TenantContext(
            tenant_id="tenant-456",
            tenant_name="Enterprise Tenant",
            max_users=1000,
            max_storage_bytes=500 * 1024 * 1024 * 1024,
            metadata={"region": "us-east-1"},
        )

        assert ctx.max_users == 1000
        assert ctx.max_storage_bytes == 500 * 1024 * 1024 * 1024
        assert ctx.metadata["region"] == "us-east-1"

    def test_get_set_current_tenant(self):
        """Test getting and setting current tenant context."""
        from src.core.multitenancy.context import (
            TenantContext,
            get_current_tenant,
            set_current_tenant,
            reset_current_tenant,
        )

        # Initially no tenant
        assert get_current_tenant() is None

        # Set tenant
        ctx = TenantContext(tenant_id="t1", tenant_name="Test")
        token = set_current_tenant(ctx)
        assert get_current_tenant() == ctx

        # Reset tenant
        reset_current_tenant(token)
        assert get_current_tenant() is None

    def test_tenant_context_manager(self):
        """Test tenant_context context manager."""
        from src.core.multitenancy.context import (
            TenantContext,
            tenant_context,
            get_current_tenant,
        )

        ctx = TenantContext(tenant_id="ctx-tenant", tenant_name="Context Test")

        # Before context
        assert get_current_tenant() is None

        # Inside context
        with tenant_context(ctx):
            current = get_current_tenant()
            assert current is not None
            assert current.tenant_id == "ctx-tenant"

        # After context
        assert get_current_tenant() is None

    def test_tenant_context_to_dict(self):
        """Test TenantContext to_dict conversion."""
        from src.core.multitenancy.context import TenantContext

        ctx = TenantContext(
            tenant_id="t1",
            tenant_name="Test",
            max_storage_bytes=1024,
            current_storage_bytes=512,
        )

        data = ctx.to_dict()
        assert data["tenant_id"] == "t1"
        assert data["tenant_name"] == "Test"
        assert data["quotas"]["storage"]["current"] == 512
        assert data["quotas"]["storage"]["max"] == 1024

    def test_tenant_context_quota_exceeded(self):
        """Test quota exceeded check."""
        from src.core.multitenancy.context import TenantContext

        ctx = TenantContext(
            tenant_id="t1",
            tenant_name="Test",
            max_documents=10,
            current_documents=10,
        )

        assert ctx.is_quota_exceeded("documents") is True
        assert ctx.is_quota_exceeded("storage") is False  # None means unlimited


class TestIsolationStrategies:
    """Test tenant isolation strategies."""

    def test_isolation_level_enum(self):
        """Test IsolationLevel enum values."""
        from src.core.multitenancy.isolation import IsolationLevel

        assert IsolationLevel.ROW_LEVEL.value == "row_level"
        assert IsolationLevel.SCHEMA_LEVEL.value == "schema"
        assert IsolationLevel.DATABASE_LEVEL.value == "database"

    def test_row_level_isolation(self):
        """Test row-level isolation strategy."""
        from src.core.multitenancy.isolation import RowLevelIsolation, IsolationConfig, IsolationLevel
        from src.core.multitenancy.context import TenantContext

        config = IsolationConfig(level=IsolationLevel.ROW_LEVEL)
        isolation = RowLevelIsolation(config)

        tenant = TenantContext(tenant_id="tenant-abc", tenant_name="Test")

        # Apply filter to SQL query
        query = "SELECT * FROM documents"
        filtered = isolation.apply_filter(query, tenant)

        assert "tenant_id = 'tenant-abc'" in filtered

    def test_row_level_isolation_inject_tenant(self):
        """Test row-level isolation inject_tenant."""
        from src.core.multitenancy.isolation import RowLevelIsolation, IsolationConfig, IsolationLevel
        from src.core.multitenancy.context import TenantContext

        config = IsolationConfig(level=IsolationLevel.ROW_LEVEL)
        isolation = RowLevelIsolation(config)

        tenant = TenantContext(tenant_id="tenant-xyz", tenant_name="Test")
        data = {"name": "doc1"}

        result = isolation.inject_tenant(data, tenant)
        assert result["tenant_id"] == "tenant-xyz"

    def test_schema_isolation(self):
        """Test schema-level isolation strategy."""
        from src.core.multitenancy.isolation import SchemaIsolation, IsolationConfig, IsolationLevel
        from src.core.multitenancy.context import TenantContext

        config = IsolationConfig(level=IsolationLevel.SCHEMA_LEVEL)
        isolation = SchemaIsolation(config)

        tenant = TenantContext(tenant_id="acme", tenant_name="Acme Corp")

        # Get connection params
        params = isolation.get_connection_params(tenant)
        assert "tenant_acme" in params["schema"]

    def test_database_isolation(self):
        """Test database-level isolation strategy."""
        from src.core.multitenancy.isolation import DatabaseIsolation, IsolationConfig, IsolationLevel
        from src.core.multitenancy.context import TenantContext

        config = IsolationConfig(level=IsolationLevel.DATABASE_LEVEL)
        isolation = DatabaseIsolation(config)

        tenant = TenantContext(tenant_id="corp", tenant_name="Corp Inc")

        params = isolation.get_connection_params(tenant)
        assert "tenant_corp" in params["database"]

    def test_hybrid_isolation(self):
        """Test hybrid isolation strategy."""
        from src.core.multitenancy.isolation import HybridIsolation, IsolationConfig, IsolationLevel
        from src.core.multitenancy.context import TenantContext

        config = IsolationConfig(level=IsolationLevel.ROW_LEVEL)
        isolation = HybridIsolation(config, threshold_documents=100)

        # Small tenant -> row level
        small_tenant = TenantContext(tenant_id="small", tenant_name="Small", current_documents=50)
        assert isolation._get_strategy(small_tenant).__class__.__name__ == "RowLevelIsolation"

        # Large tenant -> schema level
        large_tenant = TenantContext(tenant_id="large", tenant_name="Large", current_documents=5000)
        assert isolation._get_strategy(large_tenant).__class__.__name__ == "SchemaIsolation"


class TestTenantManager:
    """Test TenantManager functionality."""

    def test_tenant_status_enum(self):
        """Test TenantStatus enum values."""
        from src.core.multitenancy.manager import TenantStatus

        assert TenantStatus.PENDING.value == "pending"
        assert TenantStatus.ACTIVE.value == "active"
        assert TenantStatus.SUSPENDED.value == "suspended"

    def test_tenant_tier_enum(self):
        """Test TenantTier enum values."""
        from src.core.multitenancy.manager import TenantTier

        assert TenantTier.FREE.value == "free"
        assert TenantTier.STARTER.value == "starter"
        assert TenantTier.PROFESSIONAL.value == "professional"
        assert TenantTier.ENTERPRISE.value == "enterprise"

    def test_tenant_quotas_for_tier(self):
        """Test quota allocation by tier."""
        from src.core.multitenancy.manager import TenantQuotas, TenantTier

        # Free tier
        free_quotas = TenantQuotas.for_tier(TenantTier.FREE)
        assert free_quotas.max_users == 2
        assert free_quotas.max_documents == 100

        # Enterprise tier
        enterprise_quotas = TenantQuotas.for_tier(TenantTier.ENTERPRISE)
        assert enterprise_quotas.max_users == 500
        assert enterprise_quotas.max_documents == 100000

    def test_tenant_manager_create_tenant(self):
        """Test creating a tenant."""
        from src.core.multitenancy.manager import TenantManager, TenantTier

        manager = TenantManager()

        tenant = asyncio.run(manager.create_tenant(
            name="New Corp",
            slug="new-corp",
            tier=TenantTier.PROFESSIONAL,
        ))

        assert tenant is not None
        assert tenant.name == "New Corp"
        assert tenant.slug == "new-corp"
        assert tenant.tier == TenantTier.PROFESSIONAL

    def test_tenant_manager_get_tenant(self):
        """Test retrieving a tenant."""
        from src.core.multitenancy.manager import TenantManager, TenantTier

        manager = TenantManager()

        # Create tenant
        tenant = asyncio.run(manager.create_tenant(
            name="Get Test",
            slug="get-test",
            tier=TenantTier.STARTER,
        ))

        # Retrieve tenant
        retrieved = manager.get_tenant(tenant.tenant_id)
        assert retrieved is not None
        assert retrieved.tenant_id == tenant.tenant_id
        assert retrieved.name == "Get Test"

    def test_tenant_manager_get_tenant_by_slug(self):
        """Test retrieving a tenant by slug."""
        from src.core.multitenancy.manager import TenantManager, TenantTier

        manager = TenantManager()

        tenant = asyncio.run(manager.create_tenant(
            name="Slug Test",
            slug="slug-test",
        ))

        retrieved = manager.get_tenant_by_slug("slug-test")
        assert retrieved is not None
        assert retrieved.tenant_id == tenant.tenant_id

    def test_tenant_manager_suspend_reactivate(self):
        """Test suspending and reactivating a tenant."""
        from src.core.multitenancy.manager import TenantManager, TenantTier, TenantStatus

        manager = TenantManager()

        # Create and provision tenant
        tenant = asyncio.run(manager.create_tenant(
            name="Suspend Test",
            slug="suspend-test",
            tier=TenantTier.FREE,
        ))
        asyncio.run(manager.provision_tenant(tenant.tenant_id))

        # Suspend
        suspended = asyncio.run(manager.suspend_tenant(
            tenant.tenant_id,
            reason="Non-payment",
        ))
        assert suspended is True

        # Check suspended
        retrieved = manager.get_tenant(tenant.tenant_id)
        assert retrieved.status == TenantStatus.SUSPENDED

        # Reactivate
        reactivated = asyncio.run(manager.reactivate_tenant(tenant.tenant_id))
        assert reactivated is True

        # Check active again
        retrieved = manager.get_tenant(tenant.tenant_id)
        assert retrieved.status == TenantStatus.ACTIVE

    def test_tenant_manager_api_key(self):
        """Test API key generation and validation."""
        from src.core.multitenancy.manager import TenantManager, TenantTier

        manager = TenantManager()

        # Create tenant
        tenant = asyncio.run(manager.create_tenant(
            name="API Key Test",
            slug="api-key-test",
            tier=TenantTier.PROFESSIONAL,
        ))

        # Generate API key
        api_key = asyncio.run(manager.generate_api_key(tenant.tenant_id))
        assert api_key is not None
        assert api_key.startswith("tnt_")

        # Validate API key
        is_valid = manager.validate_api_key(tenant.tenant_id, api_key)
        assert is_valid is True

        # Invalid key should fail
        is_valid = manager.validate_api_key(tenant.tenant_id, "invalid-key")
        assert is_valid is False


class TestTenantMiddleware:
    """Test tenant resolution and middleware."""

    def test_header_tenant_resolver(self):
        """Test header-based tenant resolution."""
        from src.core.multitenancy.middleware import HeaderTenantResolver

        resolver = HeaderTenantResolver(header_name="X-Tenant-ID")

        # Mock request with header
        mock_request = MagicMock()
        mock_request.headers = {"X-Tenant-ID": "header-tenant-123"}

        tenant_id = asyncio.run(resolver.resolve(mock_request))
        assert tenant_id == "header-tenant-123"

        # Request without header
        mock_request.headers = {}
        tenant_id = asyncio.run(resolver.resolve(mock_request))
        assert tenant_id is None

    def test_subdomain_tenant_resolver(self):
        """Test subdomain-based tenant resolution."""
        from src.core.multitenancy.middleware import SubdomainTenantResolver

        resolver = SubdomainTenantResolver(base_domain="myapp.com")

        # Mock request with subdomain
        mock_request = MagicMock()
        mock_request.headers = {"host": "acme.myapp.com"}

        tenant_id = asyncio.run(resolver.resolve(mock_request))
        assert tenant_id == "acme"

        # Excluded subdomain
        mock_request.headers = {"host": "www.myapp.com"}
        tenant_id = asyncio.run(resolver.resolve(mock_request))
        assert tenant_id is None

    def test_path_tenant_resolver(self):
        """Test path-based tenant resolution."""
        from src.core.multitenancy.middleware import PathTenantResolver

        resolver = PathTenantResolver(path_pattern=r"^/tenants/([^/]+)/")

        # Mock request with tenant path
        mock_request = MagicMock()
        mock_request.url = MagicMock()
        mock_request.url.path = "/tenants/acme-corp/api/documents"
        mock_request.path = "/tenants/acme-corp/api/documents"

        tenant_id = asyncio.run(resolver.resolve(mock_request))
        assert tenant_id == "acme-corp"

        # Request without tenant path
        mock_request.url.path = "/api/documents"
        mock_request.path = "/api/documents"
        tenant_id = asyncio.run(resolver.resolve(mock_request))
        assert tenant_id is None

    def test_chained_tenant_resolver(self):
        """Test chained tenant resolution."""
        from src.core.multitenancy.middleware import (
            ChainedTenantResolver,
            HeaderTenantResolver,
            PathTenantResolver,
        )

        resolver = ChainedTenantResolver([
            HeaderTenantResolver(),
            PathTenantResolver(),
        ])

        # First resolver succeeds
        mock_request = MagicMock()
        mock_request.headers = {"X-Tenant-ID": "header-tenant"}
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/docs"
        mock_request.path = "/api/docs"

        tenant_id = asyncio.run(resolver.resolve(mock_request))
        assert tenant_id == "header-tenant"


# ============================================================================
# P29: GraphQL Tests
# ============================================================================

class TestGraphQLTypes:
    """Test GraphQL type definitions."""

    def test_document_type(self):
        """Test DocumentType definition."""
        from src.core.graphql.types import DocumentType, DocumentStatus

        doc = DocumentType(
            id="doc-123",
            name="Test Document",
            file_path="/path/to/doc.dwg",
            file_type="dwg",
            status=DocumentStatus.COMPLETED,
        )

        assert doc.id == "doc-123"
        assert doc.name == "Test Document"
        assert doc.file_type == "dwg"

    def test_page_info(self):
        """Test PageInfo for pagination."""
        from src.core.graphql.types import PageInfo

        page_info = PageInfo(
            has_next_page=True,
            has_previous_page=False,
            start_cursor="cursor_start",
            end_cursor="cursor_end",
        )

        assert page_info.has_next_page is True
        assert page_info.start_cursor == "cursor_start"

    def test_connection_type(self):
        """Test Connection type for cursor pagination."""
        from src.core.graphql.types import Connection, Edge, PageInfo, DocumentType, DocumentStatus

        doc = DocumentType(id="1", name="A", file_path="/a", file_type="dwg", status=DocumentStatus.COMPLETED)
        edges = [Edge(node=doc, cursor="c1")]

        connection = Connection(
            edges=edges,
            page_info=PageInfo(
                has_next_page=False,
                has_previous_page=False,
                start_cursor="c1",
                end_cursor="c1",
            ),
            total_count=1,
        )

        assert len(connection.edges) == 1
        assert connection.total_count == 1

    def test_mutation_response(self):
        """Test MutationResponse type."""
        from src.core.graphql.types import MutationResponse

        # Success response
        response = MutationResponse(
            success=True,
            message="Document created",
            data={"id": "new-doc-123"},
        )
        assert response.success is True
        assert response.data["id"] == "new-doc-123"

        # Error response
        error_response = MutationResponse(
            success=False,
            message="Validation failed",
            errors=["Name is required", "File path invalid"],
        )
        assert error_response.success is False
        assert len(error_response.errors) == 2


class TestGraphQLResolvers:
    """Test GraphQL resolvers."""

    def test_resolver_context(self):
        """Test ResolverContext creation."""
        from src.core.graphql.resolvers import ResolverContext

        ctx = ResolverContext(
            user_id="user-123",
            tenant_id="tenant-456",
            permissions=["read:documents", "write:documents"],
            roles=["admin"],
        )

        assert ctx.user_id == "user-123"
        assert ctx.tenant_id == "tenant-456"
        assert ctx.has_permission("read:documents") is True
        assert ctx.has_permission("delete:documents") is False
        assert ctx.has_role("admin") is True

    def test_resolver_registry(self):
        """Test ResolverRegistry."""
        from src.core.graphql.resolvers import ResolverRegistry, DocumentQueryResolver

        registry = ResolverRegistry()

        resolver = DocumentQueryResolver()
        registry.register_query("document", resolver)

        retrieved = registry.get_resolver("Query", "document")
        assert retrieved is resolver


class TestDataLoader:
    """Test DataLoader for N+1 prevention."""

    def test_dataloader_basic(self):
        """Test basic DataLoader functionality."""
        from src.core.graphql.dataloader import DataLoader

        async def batch_load(keys: List[str]) -> List[Dict]:
            return [{"id": k, "name": f"Item {k}"} for k in keys]

        loader = DataLoader(batch_load)

        async def test():
            result = await loader.load("1")
            return result

        result = asyncio.run(test())
        assert result["id"] == "1"

    def test_dataloader_caching(self):
        """Test DataLoader caching."""
        from src.core.graphql.dataloader import DataLoader

        call_count = 0

        async def batch_load(keys: List[str]) -> List[Dict]:
            nonlocal call_count
            call_count += 1
            return [{"id": k} for k in keys]

        loader = DataLoader(batch_load, cache_enabled=True)

        async def test():
            # First load
            result1 = await loader.load("key1")
            # Second load (should be cached)
            result2 = await loader.load("key1")
            return result1, result2

        result1, result2 = asyncio.run(test())
        assert result1 == result2

    def test_dataloader_registry(self):
        """Test DataLoaderRegistry for per-request loaders."""
        from src.core.graphql.dataloader import DataLoaderRegistry, DataLoader

        registry = DataLoaderRegistry()

        async def user_loader(keys):
            return [{"id": k, "type": "user"} for k in keys]

        # Register loader factory
        registry.register("users", lambda: DataLoader(user_loader))

        # Create loaders for request
        loaders = registry.create_loaders()
        assert "users" in loaders


# ============================================================================
# P30: WebSocket Tests
# ============================================================================

class TestWebSocketRooms:
    """Test WebSocket room management."""

    def test_room_creation(self):
        """Test Room dataclass creation."""
        from src.core.websocket.rooms import Room

        room = Room(
            room_id="room-123",
            name="General Chat",
        )

        assert room.room_id == "room-123"
        assert room.name == "General Chat"
        assert len(room.members) == 0

    def test_room_add_remove_member(self):
        """Test Room add and remove member."""
        from src.core.websocket.rooms import Room

        room = Room(room_id="r1", name="Test", max_members=2)

        # Add members
        assert room.add_member("conn-1") is True
        assert room.add_member("conn-2") is True
        assert room.add_member("conn-3") is False  # Full

        assert room.member_count() == 2
        assert room.has_member("conn-1") is True

        # Remove member
        assert room.remove_member("conn-1") is True
        assert room.has_member("conn-1") is False
        assert room.member_count() == 1

    def test_room_manager_create_room(self):
        """Test creating a room."""
        from src.core.websocket.rooms import RoomManager

        manager = RoomManager()

        room = asyncio.run(manager.create_room(
            room_id="project-alpha",
            name="Project Alpha",
            max_members=50,
        ))

        assert room is not None
        assert room.name == "Project Alpha"
        assert room.max_members == 50

    def test_room_manager_join_leave(self):
        """Test joining and leaving rooms."""
        from src.core.websocket.rooms import RoomManager

        manager = RoomManager()

        # Create room
        room = asyncio.run(manager.create_room(
            room_id="test-room",
            name="Test Room",
        ))

        # Join room
        asyncio.run(manager.join_room(room.room_id, "user-1"))
        asyncio.run(manager.join_room(room.room_id, "user-2"))

        members = manager.get_room_members(room.room_id)
        assert "user-1" in members
        assert "user-2" in members

        # Leave room
        asyncio.run(manager.leave_room(room.room_id, "user-1"))

        members = manager.get_room_members(room.room_id)
        assert "user-1" not in members
        assert "user-2" in members

    def test_room_manager_get_connection_rooms(self):
        """Test getting rooms for a connection."""
        from src.core.websocket.rooms import RoomManager

        manager = RoomManager()

        # Create multiple rooms
        asyncio.run(manager.create_room(room_id="room-1", name="Room 1"))
        asyncio.run(manager.create_room(room_id="room-2", name="Room 2"))
        asyncio.run(manager.create_room(room_id="room-3", name="Room 3"))

        # User joins some rooms
        asyncio.run(manager.join_room("room-1", "user-x"))
        asyncio.run(manager.join_room("room-3", "user-x"))

        # Get user's rooms
        user_rooms = manager.get_connection_rooms("user-x")

        assert len(user_rooms) == 2
        assert "room-1" in user_rooms
        assert "room-3" in user_rooms


class TestWebSocketEvents:
    """Test WebSocket event system."""

    def test_event_type_enum(self):
        """Test EventType enum values."""
        from src.core.websocket.events import EventType

        assert EventType.MESSAGE.value == "message"
        assert EventType.JOIN.value == "join"
        assert EventType.LEAVE.value == "leave"
        assert EventType.ERROR.value == "error"

    def test_websocket_event_creation(self):
        """Test WebSocketEvent creation."""
        from src.core.websocket.events import WebSocketEvent

        event = WebSocketEvent(
            event_type="message",
            data={"content": "Hello!"},
            connection_id="conn-1",
        )

        assert event.event_type == "message"
        assert event.data["content"] == "Hello!"
        assert event.connection_id == "conn-1"

    def test_websocket_event_serialization(self):
        """Test WebSocketEvent to_dict and from_dict."""
        from src.core.websocket.events import WebSocketEvent

        original = WebSocketEvent(
            event_type="message",
            data={"text": "Test message"},
        )

        # Serialize
        data = original.to_dict()
        assert data["type"] == "message"
        assert data["data"]["text"] == "Test message"

        # Deserialize
        restored = WebSocketEvent.from_dict(data)
        assert restored.event_type == "message"
        assert restored.data["text"] == "Test message"

    def test_event_dispatcher_subscribe_emit(self):
        """Test EventDispatcher subscribe and emit."""
        from src.core.websocket.events import EventDispatcher, WebSocketEvent

        dispatcher = EventDispatcher()
        received_events = []

        async def handler(event: WebSocketEvent):
            received_events.append(event)

        # Subscribe
        dispatcher.on("message", handler)

        # Emit
        event = WebSocketEvent(
            event_type="message",
            data={"content": "Test"},
        )

        async def test():
            await dispatcher.emit(event)

        asyncio.run(test())

        assert len(received_events) == 1
        assert received_events[0].data["content"] == "Test"

    def test_event_dispatcher_wildcard(self):
        """Test EventDispatcher wildcard subscription."""
        from src.core.websocket.events import EventDispatcher, WebSocketEvent

        dispatcher = EventDispatcher()
        received_events = []

        async def handler(event: WebSocketEvent):
            received_events.append(event)

        # Subscribe to all events
        dispatcher.on("*", handler)

        async def test():
            await dispatcher.emit(WebSocketEvent(event_type="event1", data={}))
            await dispatcher.emit(WebSocketEvent(event_type="event2", data={}))

        asyncio.run(test())

        assert len(received_events) == 2


class TestWebSocketPubSub:
    """Test WebSocket pub/sub backends."""

    def test_inmemory_pubsub_publish_subscribe(self):
        """Test InMemoryPubSub publish and subscribe."""
        from src.core.websocket.pubsub import InMemoryPubSub

        pubsub = InMemoryPubSub()
        received = []

        async def test():
            # Start subscription in background
            async def subscriber():
                async for msg in pubsub.subscribe("channel1"):
                    received.append(msg)
                    if len(received) >= 1:
                        break

            # Create subscriber task
            sub_task = asyncio.create_task(subscriber())

            # Give subscriber time to start
            await asyncio.sleep(0.05)

            # Publish
            await pubsub.publish("channel1", {"text": "Hello"})

            # Wait for message
            await asyncio.sleep(0.1)
            sub_task.cancel()
            try:
                await sub_task
            except asyncio.CancelledError:
                pass

        asyncio.run(test())

        assert len(received) == 1
        assert received[0].data["text"] == "Hello"

    def test_pubsub_router(self):
        """Test PubSubRouter with multiple backends."""
        from src.core.websocket.pubsub import PubSubRouter, InMemoryPubSub

        router = PubSubRouter()
        backend1 = InMemoryPubSub()
        backend2 = InMemoryPubSub()

        router.add_backend(backend1)
        router.add_backend(backend2)

        async def test():
            count = await router.publish("test-channel", {"data": "test"})
            return count

        # Publishing should try both backends
        count = asyncio.run(test())
        assert count == 0  # No subscribers, but no errors


# ============================================================================
# P31: Elasticsearch Search Tests
# ============================================================================

class TestSearchClient:
    """Test search client implementations."""

    def test_search_hit_dataclass(self):
        """Test SearchHit dataclass."""
        from src.core.search.client import SearchHit

        hit = SearchHit(
            id="doc-1",
            index="documents",
            score=0.95,
            source={"name": "Test Doc", "content": "Hello world"},
            highlight={"content": ["<em>Hello</em> world"]},
        )

        assert hit.id == "doc-1"
        assert hit.score == 0.95
        assert hit.source["name"] == "Test Doc"

    def test_search_result_dataclass(self):
        """Test SearchResult dataclass."""
        from src.core.search.client import SearchResult, SearchHit

        hits = [
            SearchHit(id="1", index="docs", score=0.9, source={"name": "A"}),
            SearchHit(id="2", index="docs", score=0.8, source={"name": "B"}),
        ]

        result = SearchResult(
            hits=hits,
            total=100,
            max_score=0.9,
            took_ms=15,
        )

        assert len(result.hits) == 2
        assert result.total == 100
        assert result.took_ms == 15

    def test_inmemory_client_index_get(self):
        """Test InMemorySearchClient index and get."""
        from src.core.search.client import InMemorySearchClient

        client = InMemorySearchClient()

        # Index document
        success = asyncio.run(client.index(
            index="test-index",
            doc_id="doc-1",
            document={"name": "Test", "content": "Hello world"},
        ))
        assert success is True

        # Get document
        doc = asyncio.run(client.get("test-index", "doc-1"))
        assert doc is not None
        assert doc["name"] == "Test"

        # Get non-existent
        missing = asyncio.run(client.get("test-index", "missing"))
        assert missing is None

    def test_inmemory_client_delete(self):
        """Test InMemorySearchClient delete."""
        from src.core.search.client import InMemorySearchClient

        client = InMemorySearchClient()

        # Index then delete
        asyncio.run(client.index("idx", "d1", {"data": "test"}))
        deleted = asyncio.run(client.delete("idx", "d1"))
        assert deleted is True

        # Verify deleted
        doc = asyncio.run(client.get("idx", "d1"))
        assert doc is None

        # Delete non-existent
        deleted = asyncio.run(client.delete("idx", "missing"))
        assert deleted is False

    def test_inmemory_client_search_match_all(self):
        """Test InMemorySearchClient search with match_all."""
        from src.core.search.client import InMemorySearchClient

        client = InMemorySearchClient()

        # Index documents
        asyncio.run(client.index("idx", "1", {"name": "Doc A"}))
        asyncio.run(client.index("idx", "2", {"name": "Doc B"}))
        asyncio.run(client.index("idx", "3", {"name": "Doc C"}))

        # Search match_all
        result = asyncio.run(client.search("idx", {"match_all": {}}))

        assert result.total == 3
        assert len(result.hits) == 3

    def test_inmemory_client_search_match(self):
        """Test InMemorySearchClient search with match query."""
        from src.core.search.client import InMemorySearchClient

        client = InMemorySearchClient()

        asyncio.run(client.index("idx", "1", {"name": "Python Guide"}))
        asyncio.run(client.index("idx", "2", {"name": "Java Tutorial"}))
        asyncio.run(client.index("idx", "3", {"name": "Python Advanced"}))

        # Search for Python
        result = asyncio.run(client.search("idx", {
            "match": {"name": "Python"}
        }))

        assert result.total == 2

    def test_inmemory_client_search_term(self):
        """Test InMemorySearchClient search with term query."""
        from src.core.search.client import InMemorySearchClient

        client = InMemorySearchClient()

        asyncio.run(client.index("idx", "1", {"status": "active"}))
        asyncio.run(client.index("idx", "2", {"status": "inactive"}))
        asyncio.run(client.index("idx", "3", {"status": "active"}))

        result = asyncio.run(client.search("idx", {
            "term": {"status": "active"}
        }))

        assert result.total == 2

    def test_inmemory_client_bulk_index(self):
        """Test InMemorySearchClient bulk indexing."""
        from src.core.search.client import InMemorySearchClient

        client = InMemorySearchClient()

        documents = [
            ("d1", {"name": "Doc 1"}),
            ("d2", {"name": "Doc 2"}),
            ("d3", {"name": "Doc 3"}),
        ]

        success, errors = asyncio.run(client.bulk_index("idx", documents))

        assert success == 3
        assert errors == 0

        # Verify all indexed
        count = asyncio.run(client.count("idx"))
        assert count == 3


class TestSearchIndex:
    """Test index management."""

    def test_field_type_enum(self):
        """Test FieldType enum values."""
        from src.core.search.index import FieldType

        assert FieldType.TEXT.value == "text"
        assert FieldType.KEYWORD.value == "keyword"
        assert FieldType.LONG.value == "long"
        assert FieldType.DATE.value == "date"
        assert FieldType.DENSE_VECTOR.value == "dense_vector"

    def test_field_mapping_to_dict(self):
        """Test FieldMapping to_dict conversion."""
        from src.core.search.index import FieldMapping, FieldType

        mapping = FieldMapping(
            name="title",
            field_type=FieldType.TEXT,
            analyzer="standard",
            copy_to=["all_text"],
        )

        result = mapping.to_dict()

        assert result["type"] == "text"
        assert result["analyzer"] == "standard"
        assert result["copy_to"] == ["all_text"]

    def test_index_settings_to_dict(self):
        """Test IndexSettings to_dict conversion."""
        from src.core.search.index import IndexSettings

        settings = IndexSettings(
            number_of_shards=3,
            number_of_replicas=2,
            refresh_interval="5s",
        )

        result = settings.to_dict()

        assert result["index"]["number_of_shards"] == 3
        assert result["index"]["number_of_replicas"] == 2
        assert result["index"]["refresh_interval"] == "5s"

    def test_index_mapping_add_field(self):
        """Test IndexMapping add_field."""
        from src.core.search.index import IndexMapping, FieldType

        mapping = IndexMapping()

        mapping.add_field("id", FieldType.KEYWORD)
        mapping.add_field("title", FieldType.TEXT, analyzer="english")
        mapping.add_field("created_at", FieldType.DATE)

        assert len(mapping.fields) == 3
        assert mapping.fields["title"].field_type == FieldType.TEXT
        assert mapping.fields["title"].analyzer == "english"

    def test_index_mapping_to_dict(self):
        """Test IndexMapping to_dict for ES body."""
        from src.core.search.index import IndexMapping, FieldType

        mapping = IndexMapping()
        mapping.add_field("id", FieldType.KEYWORD)
        mapping.add_field("name", FieldType.TEXT)

        result = mapping.to_dict()

        assert "settings" in result
        assert "mappings" in result
        assert "properties" in result["mappings"]
        assert "id" in result["mappings"]["properties"]
        assert "name" in result["mappings"]["properties"]

    def test_predefined_document_mapping(self):
        """Test predefined document index mapping."""
        from src.core.search.index import create_document_index_mapping

        mapping = create_document_index_mapping()

        assert "id" in mapping.fields
        assert "name" in mapping.fields
        assert "file_path" in mapping.fields
        assert "content" in mapping.fields
        assert "created_at" in mapping.fields

    def test_predefined_model_mapping(self):
        """Test predefined model index mapping."""
        from src.core.search.index import create_model_index_mapping

        mapping = create_model_index_mapping()

        assert "id" in mapping.fields
        assert "name" in mapping.fields
        assert "model_type" in mapping.fields
        assert "accuracy" in mapping.fields


class TestSearchQuery:
    """Test query DSL builder."""

    def test_match_all_query(self):
        """Test MatchAllQuery."""
        from src.core.search.query import MatchAllQuery

        query = MatchAllQuery()
        result = query.to_dict()

        assert result == {"match_all": {}}

    def test_match_all_query_with_boost(self):
        """Test MatchAllQuery with boost."""
        from src.core.search.query import MatchAllQuery

        query = MatchAllQuery(boost=1.5)
        result = query.to_dict()

        assert result == {"match_all": {"boost": 1.5}}

    def test_match_query(self):
        """Test MatchQuery."""
        from src.core.search.query import MatchQuery

        query = MatchQuery(field="content", query="search term")
        result = query.to_dict()

        assert "match" in result
        assert result["match"]["content"]["query"] == "search term"

    def test_match_query_with_options(self):
        """Test MatchQuery with options."""
        from src.core.search.query import MatchQuery

        query = MatchQuery(
            field="title",
            query="test",
            operator="and",
            fuzziness="AUTO",
        )
        result = query.to_dict()

        assert result["match"]["title"]["operator"] == "and"
        assert result["match"]["title"]["fuzziness"] == "AUTO"

    def test_term_query(self):
        """Test TermQuery."""
        from src.core.search.query import TermQuery

        query = TermQuery(field="status", value="active")
        result = query.to_dict()

        assert result == {"term": {"status": "active"}}

    def test_terms_query(self):
        """Test TermsQuery."""
        from src.core.search.query import TermsQuery

        query = TermsQuery(field="tags", values=["python", "ml", "ai"])
        result = query.to_dict()

        assert result == {"terms": {"tags": ["python", "ml", "ai"]}}

    def test_range_query(self):
        """Test RangeQuery."""
        from src.core.search.query import RangeQuery

        query = RangeQuery(field="price", gte=10, lte=100)
        result = query.to_dict()

        assert result["range"]["price"]["gte"] == 10
        assert result["range"]["price"]["lte"] == 100

    def test_bool_query(self):
        """Test BoolQuery."""
        from src.core.search.query import BoolQuery, MatchQuery, TermQuery

        bool_query = BoolQuery()
        bool_query.add_must(MatchQuery(field="content", query="python"))
        bool_query.add_filter(TermQuery(field="status", value="published"))
        bool_query.add_must_not(TermQuery(field="archived", value=True))

        result = bool_query.to_dict()

        assert "bool" in result
        assert len(result["bool"]["must"]) == 1
        assert len(result["bool"]["filter"]) == 1
        assert len(result["bool"]["must_not"]) == 1

    def test_query_builder_fluent(self):
        """Test QueryBuilder fluent interface."""
        from src.core.search.query import QueryBuilder, MatchQuery, TermQuery

        query = (QueryBuilder()
            .must(MatchQuery(field="title", query="guide"))
            .filter(TermQuery(field="category", value="tutorial"))
            .should(MatchQuery(field="tags", query="beginner"))
            .minimum_should_match(1)
            .build())

        assert "bool" in query
        assert "must" in query["bool"]
        assert "filter" in query["bool"]
        assert "should" in query["bool"]
        assert query["bool"]["minimum_should_match"] == 1

    def test_query_builder_convenience_methods(self):
        """Test QueryBuilder convenience methods."""
        from src.core.search.query import QueryBuilder

        query = (QueryBuilder()
            .match("content", "search text")
            .term("status", "active")
            .range("created_at", gte="2024-01-01")
            .build())

        assert "bool" in query


class TestSearchAggregations:
    """Test aggregation builders."""

    def test_terms_aggregation(self):
        """Test TermsAggregation."""
        from src.core.search.aggregations import TermsAggregation

        agg = TermsAggregation(name="categories", field="category", size=20)
        result = agg.to_dict()

        assert "categories" in result
        assert result["categories"]["terms"]["field"] == "category"
        assert result["categories"]["terms"]["size"] == 20

    def test_date_histogram_aggregation(self):
        """Test DateHistogramAggregation."""
        from src.core.search.aggregations import DateHistogramAggregation

        agg = DateHistogramAggregation(
            name="monthly",
            field="created_at",
            calendar_interval="month",
            format="yyyy-MM",
        )
        result = agg.to_dict()

        assert "monthly" in result
        assert result["monthly"]["date_histogram"]["calendar_interval"] == "month"
        assert result["monthly"]["date_histogram"]["format"] == "yyyy-MM"

    def test_range_aggregation(self):
        """Test RangeAggregation."""
        from src.core.search.aggregations import RangeAggregation

        agg = RangeAggregation(
            name="price_ranges",
            field="price",
            ranges=[
                {"to": 50},
                {"from": 50, "to": 100},
                {"from": 100},
            ],
        )
        result = agg.to_dict()

        assert "price_ranges" in result
        assert len(result["price_ranges"]["range"]["ranges"]) == 3

    def test_stats_aggregation(self):
        """Test StatsAggregation."""
        from src.core.search.aggregations import StatsAggregation

        agg = StatsAggregation(name="price_stats", field="price")
        result = agg.to_dict()

        assert "price_stats" in result
        assert result["price_stats"]["stats"]["field"] == "price"

    def test_cardinality_aggregation(self):
        """Test CardinalityAggregation."""
        from src.core.search.aggregations import CardinalityAggregation

        agg = CardinalityAggregation(name="unique_users", field="user_id")
        result = agg.to_dict()

        assert "unique_users" in result
        assert result["unique_users"]["cardinality"]["field"] == "user_id"

    def test_nested_aggregations(self):
        """Test nested sub-aggregations."""
        from src.core.search.aggregations import (
            TermsAggregation,
            AvgAggregation,
            MaxAggregation,
        )

        agg = TermsAggregation(
            name="by_category",
            field="category",
            sub_aggregations=[
                AvgAggregation(name="avg_price", field="price"),
                MaxAggregation(name="max_price", field="price"),
            ],
        )
        result = agg.to_dict()

        assert "by_category" in result
        assert "aggs" in result["by_category"]
        assert "avg_price" in result["by_category"]["aggs"]
        assert "max_price" in result["by_category"]["aggs"]

    def test_aggregation_builder(self):
        """Test AggregationBuilder fluent interface."""
        from src.core.search.aggregations import AggregationBuilder

        aggs = (AggregationBuilder()
            .terms("categories", "category", size=10)
            .avg("avg_price", "price")
            .stats("price_stats", "price")
            .cardinality("unique_authors", "author_id")
            .build())

        assert "categories" in aggs
        assert "avg_price" in aggs
        assert "price_stats" in aggs
        assert "unique_authors" in aggs


# ============================================================================
# Integration Tests
# ============================================================================

class TestMultitenancyIntegration:
    """Integration tests for multi-tenancy."""

    def test_tenant_lifecycle_flow(self):
        """Test complete tenant lifecycle."""
        from src.core.multitenancy.manager import TenantManager, TenantTier, TenantStatus
        from src.core.multitenancy.context import tenant_context, get_current_tenant

        manager = TenantManager()

        # Create tenant
        tenant = asyncio.run(manager.create_tenant(
            name="Integration Test Corp",
            slug="integration-test-corp",
            tier=TenantTier.PROFESSIONAL,
        ))

        # Provision tenant
        asyncio.run(manager.provision_tenant(tenant.tenant_id))

        # Use tenant context
        ctx = tenant.to_context()
        with tenant_context(ctx):
            current = get_current_tenant()
            assert current.tenant_id == tenant.tenant_id
            assert current.tenant_name == "Integration Test Corp"

        # Suspend
        asyncio.run(manager.suspend_tenant(tenant.tenant_id, "test"))
        suspended = manager.get_tenant(tenant.tenant_id)
        assert suspended.status == TenantStatus.SUSPENDED

        # Reactivate
        asyncio.run(manager.reactivate_tenant(tenant.tenant_id))
        active = manager.get_tenant(tenant.tenant_id)
        assert active.status == TenantStatus.ACTIVE


class TestSearchIntegration:
    """Integration tests for search functionality."""

    def test_search_workflow(self):
        """Test complete search workflow."""
        from src.core.search.client import InMemorySearchClient
        from src.core.search.query import QueryBuilder, MatchQuery
        from src.core.search.aggregations import AggregationBuilder

        client = InMemorySearchClient()

        # Index documents
        docs = [
            ("1", {"title": "Python Guide", "category": "programming", "views": 100}),
            ("2", {"title": "Python Advanced", "category": "programming", "views": 50}),
            ("3", {"title": "Machine Learning", "category": "ai", "views": 200}),
            ("4", {"title": "Deep Learning", "category": "ai", "views": 150}),
        ]

        asyncio.run(client.bulk_index("articles", docs))

        # Build query
        query = (QueryBuilder()
            .match("title", "Python")
            .build())

        # Search
        result = asyncio.run(client.search(
            index="articles",
            query=query,
        ))

        assert result.total == 2
        titles = [h.source["title"] for h in result.hits]
        assert "Python Guide" in titles
        assert "Python Advanced" in titles
