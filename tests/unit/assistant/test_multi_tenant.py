"""
Unit tests for multi_tenant.py - Multi-Tenant Support Module.

Tests cover:
- TenantQuota management and tier defaults
- TenantUsage tracking and reset
- Tenant quota checking and consumption
- TenantManager CRUD operations
- TenantContext for request scoping
"""

import pytest
import tempfile
import json
import time
from pathlib import Path

from src.core.assistant.multi_tenant import (
    TenantStatus,
    TenantTier,
    TenantQuota,
    TenantUsage,
    Tenant,
    TenantManager,
    TenantContext,
)


class TestTenantQuota:
    """Tests for TenantQuota class."""

    def test_quota_defaults(self):
        """Test default quota values."""
        quota = TenantQuota()
        assert quota.max_conversations == 100
        assert quota.max_messages_per_day == 1000
        assert quota.max_knowledge_items == 500
        assert quota.max_api_calls_per_minute == 60
        assert quota.max_storage_mb == 100
        assert "offline" in quota.allowed_models

    def test_quota_to_dict(self):
        """Test quota serialization."""
        quota = TenantQuota(max_conversations=50)
        result = quota.to_dict()

        assert result["max_conversations"] == 50
        assert "allowed_models" in result

    def test_quota_for_free_tier(self):
        """Test FREE tier quota limits."""
        quota = TenantQuota.for_tier(TenantTier.FREE)

        assert quota.max_conversations == 10
        assert quota.max_messages_per_day == 100
        assert quota.max_knowledge_items == 50
        assert quota.max_api_calls_per_minute == 10
        assert quota.max_storage_mb == 10
        assert quota.allowed_models == ["offline"]

    def test_quota_for_basic_tier(self):
        """Test BASIC tier quota limits."""
        quota = TenantQuota.for_tier(TenantTier.BASIC)

        assert quota.max_conversations == 100
        assert quota.max_messages_per_day == 1000
        assert "qwen" in quota.allowed_models

    def test_quota_for_professional_tier(self):
        """Test PROFESSIONAL tier quota limits."""
        quota = TenantQuota.for_tier(TenantTier.PROFESSIONAL)

        assert quota.max_conversations == 1000
        assert quota.max_messages_per_day == 10000
        assert "openai" in quota.allowed_models

    def test_quota_for_enterprise_tier(self):
        """Test ENTERPRISE tier has unlimited quotas."""
        quota = TenantQuota.for_tier(TenantTier.ENTERPRISE)

        assert quota.max_conversations == -1  # Unlimited
        assert quota.max_messages_per_day == -1
        assert quota.max_knowledge_items == -1
        assert "claude" in quota.allowed_models


class TestTenantUsage:
    """Tests for TenantUsage class."""

    def test_usage_defaults(self):
        """Test default usage values."""
        usage = TenantUsage()
        assert usage.conversations == 0
        assert usage.messages_today == 0
        assert usage.knowledge_items == 0
        assert usage.api_calls_this_minute == 0
        assert usage.storage_used_mb == 0

    def test_reset_daily(self):
        """Test daily counter reset."""
        usage = TenantUsage(messages_today=100)
        usage.reset_daily()

        assert usage.messages_today == 0
        assert usage.last_reset > 0

    def test_reset_minute(self):
        """Test per-minute counter reset."""
        usage = TenantUsage(api_calls_this_minute=50)
        usage.reset_minute()

        assert usage.api_calls_this_minute == 0


class TestTenant:
    """Tests for Tenant class."""

    @pytest.fixture
    def tenant(self):
        """Create a test tenant."""
        return Tenant(
            id="test-123",
            name="Test Corp",
            tier=TenantTier.BASIC,
            quota=TenantQuota.for_tier(TenantTier.BASIC),
        )

    def test_tenant_creation(self, tenant):
        """Test tenant creation."""
        assert tenant.id == "test-123"
        assert tenant.name == "Test Corp"
        assert tenant.status == TenantStatus.ACTIVE
        assert tenant.tier == TenantTier.BASIC

    def test_tenant_to_dict(self, tenant):
        """Test tenant serialization."""
        result = tenant.to_dict()

        assert result["id"] == "test-123"
        assert result["name"] == "Test Corp"
        assert result["status"] == "active"
        assert result["tier"] == "basic"
        assert "quota" in result
        assert "usage" in result

    def test_check_quota_conversations(self, tenant):
        """Test conversation quota checking."""
        # Should be within quota
        assert tenant.check_quota("conversations") is True

        # Use up quota
        tenant.usage.conversations = tenant.quota.max_conversations
        assert tenant.check_quota("conversations") is False

    def test_check_quota_messages(self, tenant):
        """Test message quota checking."""
        assert tenant.check_quota("messages") is True

        tenant.usage.messages_today = tenant.quota.max_messages_per_day
        assert tenant.check_quota("messages") is False

    def test_check_quota_knowledge(self, tenant):
        """Test knowledge quota checking."""
        assert tenant.check_quota("knowledge") is True

        tenant.usage.knowledge_items = tenant.quota.max_knowledge_items
        assert tenant.check_quota("knowledge") is False

    def test_check_quota_api_calls(self, tenant):
        """Test API call quota checking."""
        assert tenant.check_quota("api_calls") is True

        tenant.usage.api_calls_this_minute = tenant.quota.max_api_calls_per_minute
        assert tenant.check_quota("api_calls") is False

    def test_check_quota_inactive_tenant(self, tenant):
        """Test inactive tenant always fails quota check."""
        tenant.status = TenantStatus.SUSPENDED
        assert tenant.check_quota("conversations") is False

    def test_check_quota_unlimited(self):
        """Test unlimited quota (-1) always passes."""
        tenant = Tenant(
            id="ent-123",
            name="Enterprise",
            tier=TenantTier.ENTERPRISE,
            quota=TenantQuota.for_tier(TenantTier.ENTERPRISE),
        )
        tenant.usage.conversations = 1000000
        assert tenant.check_quota("conversations") is True

    def test_use_quota_success(self, tenant):
        """Test successful quota usage."""
        assert tenant.use_quota("conversations") is True
        assert tenant.usage.conversations == 1

    def test_use_quota_with_amount(self, tenant):
        """Test quota usage with specific amount."""
        assert tenant.use_quota("messages", amount=5) is True
        assert tenant.usage.messages_today == 5

    def test_use_quota_failure(self, tenant):
        """Test quota usage when limit exceeded."""
        tenant.usage.conversations = tenant.quota.max_conversations
        assert tenant.use_quota("conversations") is False

    def test_use_quota_all_resources(self, tenant):
        """Test using quota for all resource types."""
        assert tenant.use_quota("conversations") is True
        assert tenant.use_quota("messages") is True
        assert tenant.use_quota("knowledge") is True
        assert tenant.use_quota("api_calls") is True

        assert tenant.usage.conversations == 1
        assert tenant.usage.messages_today == 1
        assert tenant.usage.knowledge_items == 1
        assert tenant.usage.api_calls_this_minute == 1


class TestTenantManager:
    """Tests for TenantManager class."""

    @pytest.fixture
    def manager(self):
        """Create a tenant manager."""
        return TenantManager()

    def test_create_tenant(self, manager):
        """Test tenant creation."""
        tenant_id = manager.create_tenant("ACME Corp", TenantTier.PROFESSIONAL)

        assert tenant_id is not None
        tenant = manager.get_tenant(tenant_id)
        assert tenant.name == "ACME Corp"
        assert tenant.tier == TenantTier.PROFESSIONAL

    def test_create_tenant_with_settings(self, manager):
        """Test tenant creation with custom settings."""
        tenant_id = manager.create_tenant(
            "Custom Corp",
            TenantTier.BASIC,
            settings={"theme": "dark", "language": "zh-CN"},
        )

        tenant = manager.get_tenant(tenant_id)
        assert tenant.settings["theme"] == "dark"

    def test_get_tenant_not_found(self, manager):
        """Test getting non-existent tenant."""
        assert manager.get_tenant("nonexistent") is None

    def test_get_tenant_for_user(self, manager):
        """Test getting tenant for a user."""
        tenant_id = manager.create_tenant("User Corp")
        manager.assign_user_to_tenant("user-123", tenant_id)

        tenant = manager.get_tenant_for_user("user-123")
        assert tenant is not None
        assert tenant.id == tenant_id

    def test_get_tenant_for_user_not_found(self, manager):
        """Test getting tenant for unassigned user."""
        assert manager.get_tenant_for_user("unknown-user") is None

    def test_assign_user_to_tenant(self, manager):
        """Test user assignment."""
        tenant_id = manager.create_tenant("Corp")
        result = manager.assign_user_to_tenant("user-456", tenant_id)

        assert result is True
        assert manager._user_tenants["user-456"] == tenant_id

    def test_assign_user_to_invalid_tenant(self, manager):
        """Test assigning user to non-existent tenant."""
        result = manager.assign_user_to_tenant("user", "invalid")
        assert result is False

    def test_update_tenant_name(self, manager):
        """Test updating tenant name."""
        tenant_id = manager.create_tenant("Old Name")
        result = manager.update_tenant(tenant_id, name="New Name")

        assert result is True
        assert manager.get_tenant(tenant_id).name == "New Name"

    def test_update_tenant_tier(self, manager):
        """Test updating tenant tier."""
        tenant_id = manager.create_tenant("Corp", TenantTier.FREE)
        manager.update_tenant(tenant_id, tier=TenantTier.PROFESSIONAL)

        tenant = manager.get_tenant(tenant_id)
        assert tenant.tier == TenantTier.PROFESSIONAL
        # Quota should be updated too
        assert tenant.quota.max_conversations == 1000

    def test_update_tenant_status(self, manager):
        """Test updating tenant status."""
        tenant_id = manager.create_tenant("Corp")
        manager.update_tenant(tenant_id, status=TenantStatus.SUSPENDED)

        assert manager.get_tenant(tenant_id).status == TenantStatus.SUSPENDED

    def test_update_tenant_settings(self, manager):
        """Test updating tenant settings."""
        tenant_id = manager.create_tenant("Corp")
        manager.update_tenant(tenant_id, settings={"feature_x": True})

        assert manager.get_tenant(tenant_id).settings["feature_x"] is True

    def test_update_nonexistent_tenant(self, manager):
        """Test updating non-existent tenant."""
        result = manager.update_tenant("invalid", name="Test")
        assert result is False

    def test_delete_tenant(self, manager):
        """Test soft deleting a tenant."""
        tenant_id = manager.create_tenant("Corp")
        result = manager.delete_tenant(tenant_id)

        assert result is True
        tenant = manager.get_tenant(tenant_id)
        assert tenant.status == TenantStatus.DELETED

    def test_delete_nonexistent_tenant(self, manager):
        """Test deleting non-existent tenant."""
        result = manager.delete_tenant("invalid")
        assert result is False

    def test_list_tenants(self, manager):
        """Test listing all tenants."""
        manager.create_tenant("Corp A", TenantTier.FREE)
        manager.create_tenant("Corp B", TenantTier.BASIC)

        tenants = manager.list_tenants()
        assert len(tenants) == 2

    def test_list_tenants_by_status(self, manager):
        """Test listing tenants filtered by status."""
        id1 = manager.create_tenant("Active Corp")
        id2 = manager.create_tenant("Suspended Corp")
        manager.update_tenant(id2, status=TenantStatus.SUSPENDED)

        active = manager.list_tenants(status=TenantStatus.ACTIVE)
        assert len(active) == 1
        assert active[0]["name"] == "Active Corp"

    def test_list_tenants_by_tier(self, manager):
        """Test listing tenants filtered by tier."""
        manager.create_tenant("Free Corp", TenantTier.FREE)
        manager.create_tenant("Pro Corp", TenantTier.PROFESSIONAL)

        free = manager.list_tenants(tier=TenantTier.FREE)
        assert len(free) == 1
        assert free[0]["name"] == "Free Corp"

    def test_reset_usage_counters(self, manager):
        """Test resetting usage counters for all tenants."""
        id1 = manager.create_tenant("Corp A")
        id2 = manager.create_tenant("Corp B")

        # Set some usage
        manager.get_tenant(id1).usage.messages_today = 100
        manager.get_tenant(id2).usage.messages_today = 200

        manager.reset_usage_counters()

        assert manager.get_tenant(id1).usage.messages_today == 0
        assert manager.get_tenant(id2).usage.messages_today == 0

    def test_save_and_load(self):
        """Test persistence."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            storage_path = f.name

        try:
            # Create and save
            manager1 = TenantManager(storage_path=storage_path)
            tenant_id = manager1.create_tenant("Persistent Corp", TenantTier.BASIC)
            manager1.assign_user_to_tenant("user-1", tenant_id)
            result = manager1.save()
            assert result is True

            # Load in new manager
            manager2 = TenantManager(storage_path=storage_path)
            tenant = manager2.get_tenant(tenant_id)

            assert tenant is not None
            assert tenant.name == "Persistent Corp"
            assert tenant.tier == TenantTier.BASIC
            assert manager2._user_tenants["user-1"] == tenant_id
        finally:
            Path(storage_path).unlink(missing_ok=True)

    def test_save_without_path(self, manager):
        """Test save fails without storage path."""
        result = manager.save()
        assert result is False


class TestTenantContext:
    """Tests for TenantContext class."""

    @pytest.fixture
    def tenant(self):
        """Create a test tenant."""
        return Tenant(
            id="ctx-tenant",
            name="Context Corp",
            tier=TenantTier.BASIC,
            quota=TenantQuota.for_tier(TenantTier.BASIC),
        )

    def test_context_entry_exit(self, tenant):
        """Test context manager entry and exit."""
        assert TenantContext.get_current() is None

        with TenantContext(tenant) as ctx:
            assert TenantContext.get_current() == tenant
            assert ctx.tenant == tenant

        assert TenantContext.get_current() is None

    def test_context_nested(self, tenant):
        """Test nested tenant contexts."""
        tenant2 = Tenant(
            id="ctx-tenant-2",
            name="Context Corp 2",
        )

        with TenantContext(tenant):
            assert TenantContext.get_current() == tenant

            with TenantContext(tenant2):
                assert TenantContext.get_current() == tenant2

            # Should restore previous tenant
            assert TenantContext.get_current() == tenant

    def test_context_check_quota(self, tenant):
        """Test quota checking in context."""
        with TenantContext(tenant) as ctx:
            assert ctx.check_quota("conversations") is True
            assert ctx.check_quota("messages") is True

    def test_context_use_quota(self, tenant):
        """Test quota usage in context."""
        with TenantContext(tenant) as ctx:
            assert ctx.use_quota("messages", 5) is True
            assert tenant.usage.messages_today == 5

    def test_get_current_without_context(self):
        """Test get_current returns None outside context."""
        assert TenantContext.get_current() is None


class TestTenantStatus:
    """Tests for TenantStatus enum."""

    def test_all_statuses(self):
        """Test all status values."""
        assert TenantStatus.ACTIVE.value == "active"
        assert TenantStatus.SUSPENDED.value == "suspended"
        assert TenantStatus.PENDING.value == "pending"
        assert TenantStatus.DELETED.value == "deleted"


class TestTenantTier:
    """Tests for TenantTier enum."""

    def test_all_tiers(self):
        """Test all tier values."""
        assert TenantTier.FREE.value == "free"
        assert TenantTier.BASIC.value == "basic"
        assert TenantTier.PROFESSIONAL.value == "professional"
        assert TenantTier.ENTERPRISE.value == "enterprise"
