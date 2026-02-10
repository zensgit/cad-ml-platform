"""Tests for src/core/multitenancy/manager.py to improve coverage.

Covers:
- TenantManager lifecycle operations
- Tenant provisioning and deprovisioning
- API key generation and validation
- Usage tracking
- Quota management
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.core.multitenancy.manager import (
    Tenant,
    TenantManager,
    TenantQuotas,
    TenantStatus,
    TenantTier,
    get_tenant_manager,
)


class TestTenantQuotas:
    """Tests for TenantQuotas class."""

    def test_default_values(self):
        """Test default quota values."""
        quotas = TenantQuotas()

        assert quotas.max_storage_bytes == 1 * 1024 * 1024 * 1024
        assert quotas.max_documents == 1000
        assert quotas.max_users == 5

    def test_for_tier_free(self):
        """Test quotas for free tier."""
        quotas = TenantQuotas.for_tier(TenantTier.FREE)

        assert quotas.max_documents == 100
        assert quotas.max_users == 2
        assert quotas.max_api_calls_per_hour == 100

    def test_for_tier_enterprise(self):
        """Test quotas for enterprise tier."""
        quotas = TenantQuotas.for_tier(TenantTier.ENTERPRISE)

        assert quotas.max_documents == 100000
        assert quotas.max_users == 500
        assert quotas.max_concurrent_jobs == 50


class TestTenant:
    """Tests for Tenant class."""

    def test_is_active(self):
        """Test is_active method."""
        tenant = Tenant(
            tenant_id="t1",
            name="Test",
            slug="test",
            status=TenantStatus.ACTIVE,
        )

        assert tenant.is_active() is True

        tenant.status = TenantStatus.SUSPENDED
        assert tenant.is_active() is False

    def test_can_create_document(self):
        """Test can_create_document method."""
        tenant = Tenant(
            tenant_id="t1",
            name="Test",
            slug="test",
            quotas=TenantQuotas(max_documents=10),
        )

        tenant.current_documents = 5
        assert tenant.can_create_document() is True

        tenant.current_documents = 10
        assert tenant.can_create_document() is False

    def test_can_add_user(self):
        """Test can_add_user method."""
        tenant = Tenant(
            tenant_id="t1",
            name="Test",
            slug="test",
            quotas=TenantQuotas(max_users=5),
        )

        tenant.current_users = 3
        assert tenant.can_add_user() is True

        tenant.current_users = 5
        assert tenant.can_add_user() is False

    def test_has_feature(self):
        """Test has_feature method."""
        tenant = Tenant(
            tenant_id="t1",
            name="Test",
            slug="test",
            features={"feature_a", "feature_b"},
        )

        assert tenant.has_feature("feature_a") is True
        assert tenant.has_feature("feature_c") is False

    def test_to_dict(self):
        """Test to_dict method."""
        tenant = Tenant(
            tenant_id="t1",
            name="Test Tenant",
            slug="test",
            status=TenantStatus.ACTIVE,
            tier=TenantTier.PROFESSIONAL,
            features={"advanced_analytics"},
        )

        data = tenant.to_dict()

        assert data["tenant_id"] == "t1"
        assert data["name"] == "Test Tenant"
        assert data["slug"] == "test"
        assert data["status"] == "active"
        assert data["tier"] == "professional"
        assert "advanced_analytics" in data["features"]

    def test_to_context(self):
        """Test to_context method."""
        tenant = Tenant(
            tenant_id="t1",
            name="Test",
            slug="test",
            schema_name="tenant_test",
            quotas=TenantQuotas(max_documents=100),
        )

        ctx = tenant.to_context()

        assert ctx.tenant_id == "t1"
        assert ctx.tenant_name == "Test"
        assert ctx.schema_name == "tenant_test"
        assert ctx.max_documents == 100


class TestTenantManager:
    """Tests for TenantManager class."""

    @pytest.fixture
    def manager(self):
        """Create a fresh TenantManager."""
        return TenantManager()

    @pytest.mark.asyncio
    async def test_create_tenant(self, manager):
        """Test creating a tenant."""
        tenant = await manager.create_tenant(
            name="Test Company",
            slug="test-company",
            owner_user_id="user123",
            tier=TenantTier.STARTER,
        )

        assert tenant.name == "Test Company"
        assert tenant.slug == "test-company"
        assert tenant.owner_user_id == "user123"
        assert tenant.tier == TenantTier.STARTER
        assert tenant.status == TenantStatus.PENDING

    @pytest.mark.asyncio
    async def test_create_tenant_duplicate_slug_raises(self, manager):
        """Test creating tenant with duplicate slug raises error."""
        await manager.create_tenant(name="First", slug="unique-slug")

        with pytest.raises(ValueError, match="already exists"):
            await manager.create_tenant(name="Second", slug="unique-slug")

    @pytest.mark.asyncio
    async def test_provision_tenant_success(self, manager):
        """Test provisioning a tenant successfully."""
        tenant = await manager.create_tenant(name="Test", slug="test")

        result = await manager.provision_tenant(tenant.tenant_id)

        assert result is True
        assert tenant.status == TenantStatus.ACTIVE
        assert tenant.provisioned_at is not None

    @pytest.mark.asyncio
    async def test_provision_tenant_not_found(self, manager):
        """Test provisioning a nonexistent tenant."""
        result = await manager.provision_tenant("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_provision_tenant_wrong_status(self, manager):
        """Test provisioning a tenant in wrong status."""
        tenant = await manager.create_tenant(name="Test", slug="test")
        await manager.provision_tenant(tenant.tenant_id)  # Now active

        result = await manager.provision_tenant(tenant.tenant_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_provision_tenant_with_sync_hook(self, manager):
        """Test provisioning with sync hook."""
        hook_called = []

        def sync_hook(tenant):
            hook_called.append(tenant.tenant_id)
            return True

        manager.add_provisioning_hook(sync_hook)

        tenant = await manager.create_tenant(name="Test", slug="test")
        await manager.provision_tenant(tenant.tenant_id)

        assert tenant.tenant_id in hook_called

    @pytest.mark.asyncio
    async def test_provision_tenant_with_async_hook(self, manager):
        """Test provisioning with async hook."""
        hook_called = []

        async def async_hook(tenant):
            hook_called.append(tenant.tenant_id)
            return True

        manager.add_provisioning_hook(async_hook)

        tenant = await manager.create_tenant(name="Test", slug="test")
        await manager.provision_tenant(tenant.tenant_id)

        assert tenant.tenant_id in hook_called

    @pytest.mark.asyncio
    async def test_provision_tenant_hook_failure(self, manager):
        """Test provisioning fails when hook fails."""

        def failing_hook(tenant):
            return False

        manager.add_provisioning_hook(failing_hook)

        tenant = await manager.create_tenant(name="Test", slug="test")
        result = await manager.provision_tenant(tenant.tenant_id)

        assert result is False
        assert tenant.status == TenantStatus.PENDING

    @pytest.mark.asyncio
    async def test_provision_tenant_hook_exception(self, manager):
        """Test provisioning handles hook exception."""

        def exception_hook(tenant):
            raise Exception("Hook error")

        manager.add_provisioning_hook(exception_hook)

        tenant = await manager.create_tenant(name="Test", slug="test")
        result = await manager.provision_tenant(tenant.tenant_id)

        assert result is False
        assert tenant.status == TenantStatus.PENDING

    @pytest.mark.asyncio
    async def test_suspend_tenant(self, manager):
        """Test suspending a tenant."""
        tenant = await manager.create_tenant(name="Test", slug="test")
        await manager.provision_tenant(tenant.tenant_id)

        result = await manager.suspend_tenant(tenant.tenant_id, reason="Non-payment")

        assert result is True
        assert tenant.status == TenantStatus.SUSPENDED
        assert tenant.suspended_at is not None
        assert tenant.settings["suspension_reason"] == "Non-payment"

    @pytest.mark.asyncio
    async def test_suspend_tenant_not_found(self, manager):
        """Test suspending nonexistent tenant."""
        result = await manager.suspend_tenant("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_reactivate_tenant(self, manager):
        """Test reactivating a suspended tenant."""
        tenant = await manager.create_tenant(name="Test", slug="test")
        await manager.provision_tenant(tenant.tenant_id)
        await manager.suspend_tenant(tenant.tenant_id, reason="Test")

        result = await manager.reactivate_tenant(tenant.tenant_id)

        assert result is True
        assert tenant.status == TenantStatus.ACTIVE
        assert tenant.suspended_at is None
        assert "suspension_reason" not in tenant.settings

    @pytest.mark.asyncio
    async def test_reactivate_tenant_not_suspended(self, manager):
        """Test reactivating a non-suspended tenant."""
        tenant = await manager.create_tenant(name="Test", slug="test")
        await manager.provision_tenant(tenant.tenant_id)

        result = await manager.reactivate_tenant(tenant.tenant_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_deactivate_tenant(self, manager):
        """Test deactivating a tenant."""
        tenant = await manager.create_tenant(name="Test", slug="test")
        await manager.provision_tenant(tenant.tenant_id)

        result = await manager.deactivate_tenant(tenant.tenant_id)

        assert result is True
        assert tenant.status == TenantStatus.DEACTIVATED

    @pytest.mark.asyncio
    async def test_deactivate_tenant_with_hooks(self, manager):
        """Test deactivating with deprovisioning hooks."""
        hook_called = []

        def deprov_hook(tenant):
            hook_called.append(tenant.tenant_id)
            return True

        manager.add_deprovisioning_hook(deprov_hook)

        tenant = await manager.create_tenant(name="Test", slug="test")
        await manager.deactivate_tenant(tenant.tenant_id)

        assert tenant.tenant_id in hook_called

    @pytest.mark.asyncio
    async def test_deactivate_tenant_async_hook(self, manager):
        """Test deactivating with async deprovisioning hook."""
        hook_called = []

        async def async_deprov_hook(tenant):
            hook_called.append(tenant.tenant_id)
            return True

        manager.add_deprovisioning_hook(async_deprov_hook)

        tenant = await manager.create_tenant(name="Test", slug="test")
        await manager.deactivate_tenant(tenant.tenant_id)

        assert tenant.tenant_id in hook_called

    @pytest.mark.asyncio
    async def test_deactivate_tenant_hook_exception(self, manager):
        """Test deactivating handles hook exception gracefully."""

        def exception_hook(tenant):
            raise Exception("Deprov error")

        manager.add_deprovisioning_hook(exception_hook)

        tenant = await manager.create_tenant(name="Test", slug="test")
        result = await manager.deactivate_tenant(tenant.tenant_id)

        # Should still deactivate despite hook error
        assert result is True
        assert tenant.status == TenantStatus.DEACTIVATED

    @pytest.mark.asyncio
    async def test_deactivate_tenant_not_found(self, manager):
        """Test deactivating nonexistent tenant."""
        result = await manager.deactivate_tenant("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_tenant_soft(self, manager):
        """Test soft deleting a tenant."""
        tenant = await manager.create_tenant(name="Test", slug="test")

        result = await manager.delete_tenant(tenant.tenant_id, hard_delete=False)

        assert result is True
        assert tenant.status == TenantStatus.DELETED
        assert tenant.deleted_at is not None
        assert manager.get_tenant(tenant.tenant_id) is not None

    @pytest.mark.asyncio
    async def test_delete_tenant_hard(self, manager):
        """Test hard deleting a tenant."""
        tenant = await manager.create_tenant(name="Test", slug="test")
        tenant_id = tenant.tenant_id

        result = await manager.delete_tenant(tenant_id, hard_delete=True)

        assert result is True
        assert manager.get_tenant(tenant_id) is None
        assert manager.get_tenant_by_slug("test") is None

    @pytest.mark.asyncio
    async def test_delete_tenant_not_found(self, manager):
        """Test deleting nonexistent tenant."""
        result = await manager.delete_tenant("nonexistent")

        assert result is False

    def test_get_tenant(self, manager):
        """Test getting tenant by ID."""
        # Can't use async fixture result directly, so we'll test sync access
        manager._tenants["t1"] = Tenant(tenant_id="t1", name="Test", slug="test")

        tenant = manager.get_tenant("t1")

        assert tenant is not None
        assert tenant.name == "Test"

    def test_get_tenant_not_found(self, manager):
        """Test getting nonexistent tenant."""
        tenant = manager.get_tenant("nonexistent")

        assert tenant is None

    def test_get_tenant_by_slug(self, manager):
        """Test getting tenant by slug."""
        manager._tenants["t1"] = Tenant(tenant_id="t1", name="Test", slug="test-slug")
        manager._slug_to_id["test-slug"] = "t1"

        tenant = manager.get_tenant_by_slug("test-slug")

        assert tenant is not None
        assert tenant.tenant_id == "t1"

    def test_get_tenant_by_slug_not_found(self, manager):
        """Test getting nonexistent tenant by slug."""
        tenant = manager.get_tenant_by_slug("nonexistent")

        assert tenant is None

    def test_list_tenants(self, manager):
        """Test listing tenants."""
        manager._tenants["t1"] = Tenant(
            tenant_id="t1", name="Test1", slug="test1", status=TenantStatus.ACTIVE
        )
        manager._tenants["t2"] = Tenant(
            tenant_id="t2", name="Test2", slug="test2", status=TenantStatus.SUSPENDED
        )

        all_tenants = manager.list_tenants()
        assert len(all_tenants) == 2

        active_tenants = manager.list_tenants(status=TenantStatus.ACTIVE)
        assert len(active_tenants) == 1
        assert active_tenants[0].tenant_id == "t1"

    def test_list_tenants_by_tier(self, manager):
        """Test listing tenants by tier."""
        manager._tenants["t1"] = Tenant(
            tenant_id="t1", name="Free", slug="free", tier=TenantTier.FREE
        )
        manager._tenants["t2"] = Tenant(
            tenant_id="t2", name="Pro", slug="pro", tier=TenantTier.PROFESSIONAL
        )

        free_tenants = manager.list_tenants(tier=TenantTier.FREE)
        assert len(free_tenants) == 1
        assert free_tenants[0].name == "Free"

    @pytest.mark.asyncio
    async def test_update_tenant(self, manager):
        """Test updating tenant properties."""
        tenant = await manager.create_tenant(name="Original", slug="test")

        updated = await manager.update_tenant(
            tenant.tenant_id,
            name="Updated",
            settings={"key": "value"},
            features={"feature_x"},
        )

        assert updated is not None
        assert updated.name == "Updated"
        assert updated.settings["key"] == "value"
        assert "feature_x" in updated.features

    @pytest.mark.asyncio
    async def test_update_tenant_not_found(self, manager):
        """Test updating nonexistent tenant."""
        updated = await manager.update_tenant("nonexistent", name="Test")

        assert updated is None

    @pytest.mark.asyncio
    async def test_update_tier(self, manager):
        """Test updating tenant tier."""
        tenant = await manager.create_tenant(name="Test", slug="test", tier=TenantTier.FREE)

        updated = await manager.update_tier(tenant.tenant_id, TenantTier.ENTERPRISE)

        assert updated is not None
        assert updated.tier == TenantTier.ENTERPRISE
        assert updated.quotas.max_users == 500  # Enterprise quota

    @pytest.mark.asyncio
    async def test_update_tier_not_found(self, manager):
        """Test updating tier for nonexistent tenant."""
        updated = await manager.update_tier("nonexistent", TenantTier.PROFESSIONAL)

        assert updated is None

    @pytest.mark.asyncio
    async def test_generate_api_key(self, manager):
        """Test generating API key."""
        tenant = await manager.create_tenant(name="Test", slug="test")

        api_key = await manager.generate_api_key(tenant.tenant_id)

        assert api_key is not None
        assert api_key.startswith("tnt_")
        assert tenant.api_key_hash is not None

    @pytest.mark.asyncio
    async def test_generate_api_key_not_found(self, manager):
        """Test generating API key for nonexistent tenant."""
        api_key = await manager.generate_api_key("nonexistent")

        assert api_key is None

    @pytest.mark.asyncio
    async def test_validate_api_key(self, manager):
        """Test validating API key."""
        tenant = await manager.create_tenant(name="Test", slug="test")
        api_key = await manager.generate_api_key(tenant.tenant_id)

        assert manager.validate_api_key(tenant.tenant_id, api_key) is True
        assert manager.validate_api_key(tenant.tenant_id, "wrong_key") is False

    def test_validate_api_key_no_hash(self, manager):
        """Test validating API key when no key set."""
        manager._tenants["t1"] = Tenant(tenant_id="t1", name="Test", slug="test")

        assert manager.validate_api_key("t1", "some_key") is False

    def test_validate_api_key_not_found(self, manager):
        """Test validating API key for nonexistent tenant."""
        assert manager.validate_api_key("nonexistent", "key") is False

    @pytest.mark.asyncio
    async def test_update_usage(self, manager):
        """Test updating tenant usage."""
        tenant = await manager.create_tenant(name="Test", slug="test")

        updated = await manager.update_usage(
            tenant.tenant_id,
            storage_delta=1000,
            documents_delta=5,
            users_delta=2,
        )

        assert updated is not None
        assert updated.current_storage_bytes == 1000
        assert updated.current_documents == 5
        assert updated.current_users == 2

    @pytest.mark.asyncio
    async def test_update_usage_negative_clamps_to_zero(self, manager):
        """Test updating usage with negative delta doesn't go below zero."""
        tenant = await manager.create_tenant(name="Test", slug="test")
        tenant.current_documents = 5

        updated = await manager.update_usage(
            tenant.tenant_id,
            documents_delta=-10,
        )

        assert updated.current_documents == 0

    @pytest.mark.asyncio
    async def test_update_usage_not_found(self, manager):
        """Test updating usage for nonexistent tenant."""
        updated = await manager.update_usage("nonexistent", storage_delta=100)

        assert updated is None


class TestGetTenantManager:
    """Tests for get_tenant_manager function."""

    def test_get_tenant_manager_singleton(self):
        """Test get_tenant_manager returns singleton."""
        import src.core.multitenancy.manager as manager_module

        # Reset singleton
        manager_module._tenant_manager = None

        mgr1 = get_tenant_manager()
        mgr2 = get_tenant_manager()

        assert mgr1 is mgr2

        # Cleanup
        manager_module._tenant_manager = None
