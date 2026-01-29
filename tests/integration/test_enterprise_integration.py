"""
Integration tests for Enterprise Features (P8-P9).

Tests cover:
- Multi-tenant + RBAC integration
- Streaming + Multi-model integration
- Full enterprise workflow scenarios
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

# Multi-tenant imports
from src.core.assistant.multi_tenant import (
    TenantStatus,
    TenantTier,
    TenantQuota,
    Tenant,
    TenantManager,
    TenantContext,
)

# RBAC imports
from src.core.assistant.rbac import (
    Permission,
    ResourceType,
    RBACManager,
    AccessContext,
)

# Streaming imports
from src.core.assistant.streaming import (
    StreamEventType,
    StreamingResponse,
    StreamingAssistant,
)

# Multi-model imports
from src.core.assistant.multi_model import (
    ModelProvider,
    ModelConfig,
    LoadBalancingStrategy,
    ModelSelector,
    MultiModelAssistant,
)


class TestMultiTenantRBACIntegration:
    """Integration tests for Multi-tenant + RBAC."""

    @pytest.fixture
    def setup_enterprise(self):
        """Set up multi-tenant environment with RBAC."""
        # Create tenant manager
        tenant_mgr = TenantManager()

        # Create tenants
        acme_id = tenant_mgr.create_tenant("ACME Corp", TenantTier.PROFESSIONAL)
        beta_id = tenant_mgr.create_tenant("Beta Inc", TenantTier.BASIC)

        # Create RBAC manager
        rbac = RBACManager()
        rbac.create_default_roles()

        # Create users
        rbac.create_user("alice", "alice", tenant_id=acme_id)
        rbac.create_user("bob", "bob", tenant_id=acme_id)
        rbac.create_user("charlie", "charlie", tenant_id=beta_id)

        # Assign roles
        rbac.assign_role("alice", "admin")
        rbac.assign_role("bob", "engineer")
        rbac.assign_role("charlie", "user")

        # Assign users to tenants
        tenant_mgr.assign_user_to_tenant("alice", acme_id)
        tenant_mgr.assign_user_to_tenant("bob", acme_id)
        tenant_mgr.assign_user_to_tenant("charlie", beta_id)

        return {
            "tenant_mgr": tenant_mgr,
            "rbac": rbac,
            "acme_id": acme_id,
            "beta_id": beta_id,
        }

    def test_tenant_aware_permission_check(self, setup_enterprise):
        """Test permissions within tenant context."""
        rbac = setup_enterprise["rbac"]
        tenant_mgr = setup_enterprise["tenant_mgr"]
        acme_id = setup_enterprise["acme_id"]

        tenant = tenant_mgr.get_tenant(acme_id)

        # Alice (admin) in ACME tenant context
        with TenantContext(tenant):
            with AccessContext(rbac, "alice") as ctx:
                # Admin should have full access
                assert ctx.can(Permission.ADMIN_SYSTEM_CONFIG) is True
                assert ctx.can(Permission.KNOWLEDGE_CREATE) is True

    def test_cross_tenant_isolation(self, setup_enterprise):
        """Test users cannot access other tenant's resources."""
        rbac = setup_enterprise["rbac"]
        acme_id = setup_enterprise["acme_id"]
        beta_id = setup_enterprise["beta_id"]

        # Register resource owned by Alice in ACME tenant
        rbac.register_resource(
            "acme-doc-1",
            ResourceType.KNOWLEDGE,
            owner_id="alice",
            tenant_id=acme_id,
        )

        # Alice can access
        assert rbac.check_resource_access(
            "alice", "acme-doc-1", Permission.KNOWLEDGE_READ
        ) is True

        # Charlie (different tenant) cannot access
        assert rbac.check_resource_access(
            "charlie", "acme-doc-1", Permission.KNOWLEDGE_READ
        ) is False

    def test_quota_enforcement_with_permissions(self, setup_enterprise):
        """Test quota checking alongside permission checking."""
        rbac = setup_enterprise["rbac"]
        tenant_mgr = setup_enterprise["tenant_mgr"]
        acme_id = setup_enterprise["acme_id"]

        tenant = tenant_mgr.get_tenant(acme_id)

        # Bob (engineer) wants to create knowledge
        with TenantContext(tenant) as ctx:
            # Check permission first
            with AccessContext(rbac, "bob") as access:
                has_permission = access.can(Permission.KNOWLEDGE_CREATE)

            # Then check quota
            has_quota = ctx.check_quota("knowledge")

            # Both must pass
            can_create = has_permission and has_quota
            assert can_create is True

            # Use quota if allowed
            if can_create:
                ctx.use_quota("knowledge")
                assert tenant.usage.knowledge_items == 1

    def test_tier_based_model_access(self, setup_enterprise):
        """Test model access based on tenant tier."""
        tenant_mgr = setup_enterprise["tenant_mgr"]
        acme_id = setup_enterprise["acme_id"]
        beta_id = setup_enterprise["beta_id"]

        acme = tenant_mgr.get_tenant(acme_id)
        beta = tenant_mgr.get_tenant(beta_id)

        # Professional tier (ACME) can use OpenAI
        assert "openai" in acme.quota.allowed_models

        # Basic tier (Beta) cannot use OpenAI
        assert "openai" not in beta.quota.allowed_models
        assert "qwen" in beta.quota.allowed_models

    def test_admin_can_manage_users_in_tenant(self, setup_enterprise):
        """Test admin managing users within their tenant."""
        rbac = setup_enterprise["rbac"]
        tenant_mgr = setup_enterprise["tenant_mgr"]
        acme_id = setup_enterprise["acme_id"]

        tenant = tenant_mgr.get_tenant(acme_id)

        with TenantContext(tenant):
            with AccessContext(rbac, "alice") as ctx:
                # Admin can manage users
                assert ctx.can(Permission.ADMIN_USER_MANAGE) is True

                # Create new user in same tenant
                rbac.create_user("dave", "dave", tenant_id=acme_id)
                rbac.assign_role("dave", "user")

                # Verify new user is in tenant
                users = rbac.list_users(tenant_id=acme_id)
                usernames = [u["username"] for u in users]
                assert "dave" in usernames


class TestStreamingMultiModelIntegration:
    """Integration tests for Streaming + Multi-model."""

    @pytest.fixture
    def multi_model_assistant(self):
        """Create multi-model assistant with mock providers."""
        assistant = MultiModelAssistant(
            strategy=LoadBalancingStrategy.PRIORITY,
            max_retries=3,
        )

        # Register models
        assistant.selector.register_model(ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            priority=1,
        ))
        assistant.selector.register_model(ModelConfig(
            provider=ModelProvider.CLAUDE,
            model_name="claude-3",
            priority=2,
        ))
        assistant.selector.register_model(ModelConfig(
            provider=ModelProvider.OFFLINE,
            model_name="offline",
            priority=3,
        ))

        return assistant

    @pytest.mark.asyncio
    async def test_streaming_with_model_failover(self, multi_model_assistant):
        """Test streaming continues after model failover."""
        # OpenAI fails, Claude succeeds
        mock_openai = MagicMock(spec=['generate'])
        mock_openai.generate.side_effect = Exception("Rate limited")
        multi_model_assistant._providers[ModelProvider.OPENAI] = mock_openai

        mock_claude = MagicMock(spec=['generate'])
        mock_claude.generate.return_value = "Claude's response about steel properties"
        multi_model_assistant._providers[ModelProvider.CLAUDE] = mock_claude

        # Ask question
        response, provider = await multi_model_assistant.ask(
            "What is 304 stainless steel?"
        )

        # Should have failed over to Claude
        assert provider == ModelProvider.CLAUDE
        assert "steel" in response.lower()

        # OpenAI should be marked unhealthy
        health = multi_model_assistant.selector.get_all_health()
        assert health["openai"]["status"] == "unavailable"

    @pytest.mark.asyncio
    async def test_streaming_response_chunking(self):
        """Test streaming breaks response into chunks."""
        long_response = "This is a detailed explanation about steel. " * 10

        streamer = StreamingResponse(chunk_size=50, delay_ms=0)

        chunks = []
        async for event in streamer.stream_text(long_response):
            if event.event_type == StreamEventType.CHUNK:
                chunks.append(event.data["text"])

        # Should have multiple chunks
        assert len(chunks) > 1

        # Reconstructed text should match
        reconstructed = "".join(chunks)
        assert reconstructed == long_response

    @pytest.mark.asyncio
    async def test_model_selection_with_health_monitoring(self, multi_model_assistant):
        """Test model selection considers health status."""
        # Make all providers available
        for provider in [ModelProvider.OPENAI, ModelProvider.CLAUDE, ModelProvider.OFFLINE]:
            mock = MagicMock(spec=['generate'])
            mock.generate.return_value = f"Response from {provider.value}"
            multi_model_assistant._providers[provider] = mock

        # Set latencies for all models
        multi_model_assistant.selector.update_health(
            ModelProvider.OPENAI,
            status=multi_model_assistant.selector._health[ModelProvider.OPENAI].status,
            latency_ms=5000,  # Very slow
        )
        multi_model_assistant.selector.update_health(
            ModelProvider.CLAUDE,
            status=multi_model_assistant.selector._health[ModelProvider.CLAUDE].status,
            latency_ms=100,  # Fast
        )
        multi_model_assistant.selector.update_health(
            ModelProvider.OFFLINE,
            status=multi_model_assistant.selector._health[ModelProvider.OFFLINE].status,
            latency_ms=200,  # Medium
        )

        # With least_latency strategy, should prefer fastest model (Claude)
        multi_model_assistant.selector.strategy = LoadBalancingStrategy.LEAST_LATENCY

        selected = multi_model_assistant.selector.select_model()
        assert selected.provider == ModelProvider.CLAUDE


class TestEnterpriseWorkflow:
    """Full enterprise workflow integration tests."""

    @pytest.mark.asyncio
    async def test_complete_enterprise_workflow(self):
        """Test complete workflow: tenant → user → permission → query → response."""
        # 1. Setup tenant
        tenant_mgr = TenantManager()
        tenant_id = tenant_mgr.create_tenant("Enterprise Corp", TenantTier.ENTERPRISE)
        tenant = tenant_mgr.get_tenant(tenant_id)

        # 2. Setup RBAC
        rbac = RBACManager()
        rbac.create_default_roles()
        rbac.create_user("engineer1", "engineer1", tenant_id=tenant_id)
        rbac.assign_role("engineer1", "engineer")
        tenant_mgr.assign_user_to_tenant("engineer1", tenant_id)

        # 3. Setup multi-model assistant
        assistant = MultiModelAssistant(strategy=LoadBalancingStrategy.PRIORITY)
        assistant.selector.register_model(ModelConfig(
            provider=ModelProvider.OFFLINE,
            model_name="offline",
        ))

        mock_provider = MagicMock(spec=['generate'])
        mock_provider.generate.return_value = "304 stainless steel has tensile strength of 515 MPa."
        assistant._providers[ModelProvider.OFFLINE] = mock_provider

        # 4. Execute workflow in contexts
        with TenantContext(tenant) as tenant_ctx:
            with AccessContext(rbac, "engineer1") as access_ctx:
                # Check permissions
                assert access_ctx.can(Permission.CONVERSATION_CREATE) is True
                assert access_ctx.can(Permission.KNOWLEDGE_READ) is True

                # Check quota
                assert tenant_ctx.check_quota("messages") is True

                # Ask question
                response, provider = await assistant.ask(
                    "What is the tensile strength of 304 stainless steel?"
                )

                # Use quota
                tenant_ctx.use_quota("messages")

                # Verify
                assert "515 MPa" in response
                assert provider == ModelProvider.OFFLINE
                assert tenant.usage.messages_today == 1

    def test_multi_tenant_data_isolation(self):
        """Test data is isolated between tenants."""
        tenant_mgr = TenantManager()
        rbac = RBACManager()
        rbac.create_default_roles()

        # Create two tenants with same-named resources
        tenant_a = tenant_mgr.create_tenant("Tenant A")
        tenant_b = tenant_mgr.create_tenant("Tenant B")

        rbac.create_user("user_a", "user_a", tenant_id=tenant_a)
        rbac.create_user("user_b", "user_b", tenant_id=tenant_b)
        rbac.assign_role("user_a", "engineer")
        rbac.assign_role("user_b", "engineer")

        # Both create knowledge items
        rbac.register_resource("design-spec", ResourceType.KNOWLEDGE, "user_a", tenant_a)
        rbac.register_resource("design-spec-b", ResourceType.KNOWLEDGE, "user_b", tenant_b)

        # Each user can only access their own tenant's resources
        assert rbac.check_resource_access("user_a", "design-spec", Permission.KNOWLEDGE_READ) is True
        assert rbac.check_resource_access("user_a", "design-spec-b", Permission.KNOWLEDGE_READ) is False
        assert rbac.check_resource_access("user_b", "design-spec-b", Permission.KNOWLEDGE_READ) is True
        assert rbac.check_resource_access("user_b", "design-spec", Permission.KNOWLEDGE_READ) is False

    def test_rate_limiting_with_quota(self):
        """Test rate limiting integrated with quota system."""
        tenant_mgr = TenantManager()
        tenant_id = tenant_mgr.create_tenant("Rate Limited Corp", TenantTier.FREE)
        tenant = tenant_mgr.get_tenant(tenant_id)

        # Free tier: 10 API calls per minute
        assert tenant.quota.max_api_calls_per_minute == 10

        # Use up quota
        for i in range(10):
            assert tenant.use_quota("api_calls") is True

        # 11th call should fail
        assert tenant.use_quota("api_calls") is False

        # Reset and try again
        tenant.usage.reset_minute()
        assert tenant.use_quota("api_calls") is True
