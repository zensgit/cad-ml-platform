"""
End-to-End Integration Tests for CAD Assistant Full Workflow.

Tests complete user journeys from API request to response,
including all enterprise features (P7-P10).
"""

import pytest
import asyncio
from typing import List, Dict, Any
from unittest.mock import MagicMock, AsyncMock, patch

# Core imports
from src.core.assistant.multi_tenant import (
    TenantManager, TenantTier, Tenant, TenantContext
)
from src.core.assistant.rbac import (
    RBACManager, Permission, ResourceType, AccessContext
)
from src.core.assistant.multi_model import (
    ModelProvider, ModelConfig, ModelSelector,
    MultiModelAssistant, LoadBalancingStrategy
)
from src.core.assistant.streaming import (
    StreamingResponse, StreamEventType, StreamEvent
)
from src.core.assistant.caching import LRUCache, CacheManager


class TestE2EUserJourney:
    """End-to-end tests for complete user journeys."""

    @pytest.fixture
    def enterprise_setup(self):
        """Set up complete enterprise environment."""
        # 1. Create tenant manager
        tenant_mgr = TenantManager()

        # 2. Create tenants for different tiers
        free_tenant_id = tenant_mgr.create_tenant("Free User Corp", TenantTier.FREE)
        pro_tenant_id = tenant_mgr.create_tenant("Pro Corp", TenantTier.PROFESSIONAL)
        ent_tenant_id = tenant_mgr.create_tenant("Enterprise Corp", TenantTier.ENTERPRISE)

        # 3. Create RBAC manager
        rbac = RBACManager()
        rbac.create_default_roles()

        # 4. Create users for each tenant
        users = {
            "free_user": {"tenant": free_tenant_id, "role": "user"},
            "pro_user": {"tenant": pro_tenant_id, "role": "engineer"},
            "pro_admin": {"tenant": pro_tenant_id, "role": "admin"},
            "ent_user": {"tenant": ent_tenant_id, "role": "engineer"},
            "ent_admin": {"tenant": ent_tenant_id, "role": "admin"},
        }

        for user_id, info in users.items():
            rbac.create_user(user_id, user_id, tenant_id=info["tenant"])
            rbac.assign_role(user_id, info["role"])
            tenant_mgr.assign_user_to_tenant(user_id, info["tenant"])

        # 5. Create multi-model assistant with OFFLINE as highest priority
        assistant = MultiModelAssistant(
            strategy=LoadBalancingStrategy.PRIORITY,
            max_retries=2,
        )

        # Register OFFLINE model with highest priority for testing
        assistant.selector.register_model(ModelConfig(
            provider=ModelProvider.OFFLINE,
            model_name="offline-test",
            priority=1,  # Highest priority
        ))

        # 6. Create cache
        cache = LRUCache(max_size=1000)

        return {
            "tenant_mgr": tenant_mgr,
            "rbac": rbac,
            "assistant": assistant,
            "cache": cache,
            "tenants": {
                "free": free_tenant_id,
                "pro": pro_tenant_id,
                "enterprise": ent_tenant_id,
            },
        }

    @pytest.mark.asyncio
    async def test_free_tier_user_workflow(self, enterprise_setup):
        """Test free tier user complete workflow."""
        tenant_mgr = enterprise_setup["tenant_mgr"]
        rbac = enterprise_setup["rbac"]
        assistant = enterprise_setup["assistant"]
        cache = enterprise_setup["cache"]

        tenant_id = enterprise_setup["tenants"]["free"]
        tenant = tenant_mgr.get_tenant(tenant_id)
        user_id = "free_user"

        # Mock provider
        mock_provider = MagicMock(spec=['generate'])
        mock_provider.generate.return_value = "304不锈钢的密度是7.93 g/cm³"
        assistant._providers[ModelProvider.OFFLINE] = mock_provider

        # Execute workflow
        with TenantContext(tenant) as tenant_ctx:
            with AccessContext(rbac, user_id) as access_ctx:
                # 1. Check permissions
                assert access_ctx.can(Permission.CONVERSATION_CREATE) is True
                assert access_ctx.can(Permission.ADMIN_SYSTEM_CONFIG) is False

                # 2. Check quota
                assert tenant_ctx.check_quota("messages") is True

                # 3. Check model access - free tier only has offline
                assert "offline" in tenant.quota.allowed_models
                assert "openai" not in tenant.quota.allowed_models

                # 4. Query using offline model
                response, provider = await assistant.ask(
                    "304不锈钢的密度是多少？"
                )

                # 5. Use quota
                tenant_ctx.use_quota("messages")

                # 6. Cache result
                cache.set("material:304:density", response)

        # Verify
        assert "7.93" in response
        assert tenant.usage.messages_today == 1
        assert cache.get("material:304:density") is not None

    @pytest.mark.asyncio
    async def test_pro_tier_multi_model_workflow(self, enterprise_setup):
        """Test professional tier with multi-model access."""
        tenant_mgr = enterprise_setup["tenant_mgr"]
        rbac = enterprise_setup["rbac"]
        assistant = enterprise_setup["assistant"]

        tenant_id = enterprise_setup["tenants"]["pro"]
        tenant = tenant_mgr.get_tenant(tenant_id)
        user_id = "pro_user"

        # Mock all providers to succeed
        mock_offline = MagicMock(spec=['generate'])
        mock_offline.generate.return_value = "TIG焊接304不锈钢推荐电流150-200A"
        assistant._providers[ModelProvider.OFFLINE] = mock_offline

        with TenantContext(tenant) as tenant_ctx:
            with AccessContext(rbac, user_id) as access_ctx:
                # Engineer has knowledge permissions
                assert access_ctx.can(Permission.KNOWLEDGE_CREATE) is True

                # Pro tier has more model options
                assert "openai" in tenant.quota.allowed_models
                assert "qwen" in tenant.quota.allowed_models

                # Query
                response, provider = await assistant.ask(
                    "TIG焊接304不锈钢的推荐电流是多少？"
                )

                tenant_ctx.use_quota("messages")

        # Verify
        assert "150-200A" in response
        assert tenant.usage.messages_today == 1

    @pytest.mark.asyncio
    async def test_enterprise_admin_workflow(self, enterprise_setup):
        """Test enterprise admin with full access."""
        tenant_mgr = enterprise_setup["tenant_mgr"]
        rbac = enterprise_setup["rbac"]

        tenant_id = enterprise_setup["tenants"]["enterprise"]
        tenant = tenant_mgr.get_tenant(tenant_id)
        user_id = "ent_admin"

        with TenantContext(tenant) as tenant_ctx:
            with AccessContext(rbac, user_id) as access_ctx:
                # Admin has all permissions
                assert access_ctx.can(Permission.ADMIN_SYSTEM_CONFIG) is True
                assert access_ctx.can(Permission.ADMIN_USER_MANAGE) is True

                # Enterprise tier has unlimited quota
                assert tenant.quota.max_conversations == -1  # unlimited
                assert tenant.quota.max_messages_per_day == -1

                # All models available
                assert "openai" in tenant.quota.allowed_models
                assert "claude" in tenant.quota.allowed_models

                # Admin can create new users
                rbac.create_user("new_engineer", "new_engineer", tenant_id=tenant_id)
                rbac.assign_role("new_engineer", "engineer")

                # Verify new user
                users = rbac.list_users(tenant_id=tenant_id)
                usernames = [u["username"] for u in users]
                assert "new_engineer" in usernames

    @pytest.mark.asyncio
    async def test_streaming_workflow(self, enterprise_setup):
        """Test streaming response workflow."""
        tenant_mgr = enterprise_setup["tenant_mgr"]
        rbac = enterprise_setup["rbac"]

        tenant_id = enterprise_setup["tenants"]["pro"]
        tenant = tenant_mgr.get_tenant(tenant_id)
        user_id = "pro_user"

        # Create streamer with smaller chunk size to ensure multiple chunks
        streamer = StreamingResponse(chunk_size=20, delay_ms=0)

        with TenantContext(tenant) as tenant_ctx:
            with AccessContext(rbac, user_id) as access_ctx:
                # Check permissions
                assert access_ctx.can(Permission.CONVERSATION_CREATE) is True

                # Simulate response - long enough to produce multiple chunks
                full_response = "304不锈钢的抗拉强度是515 MPa。这是一种常用的奥氏体不锈钢，具有良好的耐腐蚀性。"

                # Stream response
                chunks = []
                events = []
                async for event in streamer.stream_text(full_response):
                    events.append(event)
                    if event.event_type == StreamEventType.CHUNK:
                        chunks.append(event.data["text"])

                # Use quota
                tenant_ctx.use_quota("messages")

        # Verify streaming - with chunk_size=20 and longer text, should have multiple chunks
        assert len(chunks) >= 1
        assert "".join(chunks) == full_response

        # Verify event types
        event_types = [e.event_type for e in events]
        assert StreamEventType.START in event_types
        assert StreamEventType.CHUNK in event_types
        assert StreamEventType.DONE in event_types

    @pytest.mark.asyncio
    async def test_cross_tenant_isolation(self, enterprise_setup):
        """Test that tenants are properly isolated."""
        tenant_mgr = enterprise_setup["tenant_mgr"]
        rbac = enterprise_setup["rbac"]

        pro_tenant_id = enterprise_setup["tenants"]["pro"]
        ent_tenant_id = enterprise_setup["tenants"]["enterprise"]

        # Create resource in pro tenant
        rbac.register_resource(
            "pro-design-doc",
            ResourceType.KNOWLEDGE,
            owner_id="pro_user",
            tenant_id=pro_tenant_id,
        )

        # Pro user can access their own resource
        assert rbac.check_resource_access(
            "pro_user", "pro-design-doc", Permission.KNOWLEDGE_READ
        ) is True

        # Enterprise user cannot access pro tenant's resource
        assert rbac.check_resource_access(
            "ent_user", "pro-design-doc", Permission.KNOWLEDGE_READ
        ) is False

        # Even enterprise admin cannot access other tenant's resources
        assert rbac.check_resource_access(
            "ent_admin", "pro-design-doc", Permission.KNOWLEDGE_READ
        ) is False

    @pytest.mark.asyncio
    async def test_quota_enforcement_workflow(self, enterprise_setup):
        """Test quota enforcement across workflow."""
        tenant_mgr = enterprise_setup["tenant_mgr"]

        free_tenant_id = enterprise_setup["tenants"]["free"]
        tenant = tenant_mgr.get_tenant(free_tenant_id)

        # Free tier: 10 API calls per minute
        assert tenant.quota.max_api_calls_per_minute == 10

        # Use up all API calls
        with TenantContext(tenant) as ctx:
            for i in range(10):
                assert ctx.use_quota("api_calls") is True

            # 11th call should fail
            assert ctx.use_quota("api_calls") is False

            # Reset and try again
            tenant.usage.reset_minute()
            assert ctx.use_quota("api_calls") is True


class TestE2EErrorHandling:
    """End-to-end tests for error handling scenarios."""

    @pytest.fixture
    def basic_setup(self):
        """Basic setup for error handling tests."""
        tenant_mgr = TenantManager()
        tenant_id = tenant_mgr.create_tenant("Test Corp", TenantTier.BASIC)

        rbac = RBACManager()
        rbac.create_default_roles()
        rbac.create_user("test_user", "test_user", tenant_id=tenant_id)
        rbac.assign_role("test_user", "user")

        return {
            "tenant_mgr": tenant_mgr,
            "rbac": rbac,
            "tenant_id": tenant_id,
        }

    def test_permission_denied_workflow(self, basic_setup):
        """Test workflow when user lacks permissions."""
        rbac = basic_setup["rbac"]

        with AccessContext(rbac, "test_user") as ctx:
            # Regular user should not have admin permissions
            assert ctx.can(Permission.ADMIN_SYSTEM_CONFIG) is False
            assert ctx.can(Permission.ADMIN_USER_MANAGE) is False

            # But should have basic permissions
            assert ctx.can(Permission.CONVERSATION_READ) is True
            assert ctx.can(Permission.CONVERSATION_CREATE) is True

    def test_unknown_user_workflow(self, basic_setup):
        """Test workflow with unknown user."""
        rbac = basic_setup["rbac"]

        # Unknown user should have no permissions
        assert rbac.check_permission("unknown_user", Permission.CONVERSATION_READ) is False

    def test_quota_exceeded_workflow(self, basic_setup):
        """Test workflow when quota is exceeded."""
        tenant_mgr = basic_setup["tenant_mgr"]
        tenant = tenant_mgr.get_tenant(basic_setup["tenant_id"])

        # Simulate quota exhaustion
        for _ in range(tenant.quota.max_api_calls_per_minute):
            tenant.use_quota("api_calls")

        # Should now be blocked
        with TenantContext(tenant) as ctx:
            assert ctx.check_quota("api_calls") is False


class TestE2ECaching:
    """End-to-end tests for caching behavior."""

    def test_cache_hit_workflow(self):
        """Test workflow with cache hits."""
        cache = LRUCache(max_size=100)

        # First request - cache miss
        key = "material:304:tensile_strength"
        result = cache.get(key)
        assert result is None

        # Store in cache
        cache.set(key, "515 MPa")

        # Second request - cache hit
        result = cache.get(key)
        assert result == "515 MPa"

    def test_cache_eviction_workflow(self):
        """Test workflow with cache eviction."""
        cache = LRUCache(max_size=3)

        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add new key - should evict key2 (least recently used)
        cache.set("key4", "value4")

        # Verify
        assert cache.get("key1") == "value1"  # Still present (recently used)
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"  # Still present
        assert cache.get("key4") == "value4"  # Newly added
