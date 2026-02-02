"""Unit tests for security module - RBAC, Tenant, Vault."""

import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.core.security.rbac import (
    Permission,
    Role,
    RBACPolicy,
    RBACManager,
    ROLE_PERMISSIONS,
    ROLE_HIERARCHY,
    get_rbac_manager,
    require_permission,
    require_role,
)
from src.core.security.tenant import (
    TenantContext,
    TenantIsolation,
    TenantMiddleware,
    get_current_tenant,
    set_tenant_context,
    clear_tenant_context,
    get_tenant_isolation,
    require_tenant,
    require_feature,
)
from src.core.security.vault import (
    VaultConfig,
    SecretLease,
    VaultClient,
    get_vault_client,
    get_secret,
)


# =============================================================================
# RBAC Tests
# =============================================================================

class TestPermission:
    """Test Permission enum."""

    def test_permission_values(self):
        """Test permission string values."""
        assert Permission.DOCUMENT_READ.value == "document:read"
        assert Permission.MODEL_DEPLOY.value == "model:deploy"
        assert Permission.SYSTEM_ADMIN.value == "system:admin"

    def test_permission_categories(self):
        """Test permission categories exist."""
        # Document permissions
        assert Permission.DOCUMENT_READ
        assert Permission.DOCUMENT_WRITE
        assert Permission.DOCUMENT_DELETE
        assert Permission.DOCUMENT_ADMIN

        # Model permissions
        assert Permission.MODEL_PREDICT
        assert Permission.MODEL_TRAIN
        assert Permission.MODEL_DEPLOY
        assert Permission.MODEL_ADMIN


class TestRole:
    """Test Role enum."""

    def test_role_values(self):
        """Test role string values."""
        assert Role.VIEWER.value == "viewer"
        assert Role.EDITOR.value == "editor"
        assert Role.ADMIN.value == "admin"
        assert Role.SUPER_ADMIN.value == "super_admin"

    def test_role_hierarchy_completeness(self):
        """Test all roles have hierarchy entries."""
        for role in Role:
            assert role in ROLE_HIERARCHY

    def test_role_permissions_completeness(self):
        """Test all roles have permission entries."""
        for role in Role:
            assert role in ROLE_PERMISSIONS


class TestRBACPolicy:
    """Test RBACPolicy class."""

    def test_policy_creation(self):
        """Test policy creation with defaults."""
        policy = RBACPolicy(principal_id="user1")
        assert policy.principal_id == "user1"
        assert policy.principal_type == "user"
        assert policy.roles == set()
        assert policy.permissions == set()

    def test_get_effective_permissions_empty(self):
        """Test effective permissions with no roles."""
        policy = RBACPolicy(principal_id="user1")
        assert policy.get_effective_permissions() == set()

    def test_get_effective_permissions_with_role(self):
        """Test effective permissions with viewer role."""
        policy = RBACPolicy(
            principal_id="user1",
            roles={Role.VIEWER},
        )
        permissions = policy.get_effective_permissions()
        assert Permission.DOCUMENT_READ in permissions
        assert Permission.ANALYSIS_READ in permissions

    def test_get_effective_permissions_with_inheritance(self):
        """Test role inheritance in permissions."""
        policy = RBACPolicy(
            principal_id="user1",
            roles={Role.EDITOR},  # Inherits from VIEWER
        )
        permissions = policy.get_effective_permissions()
        # Editor permissions
        assert Permission.DOCUMENT_WRITE in permissions
        # Inherited viewer permissions
        assert Permission.DOCUMENT_READ in permissions

    def test_denied_permissions(self):
        """Test explicit permission denial."""
        policy = RBACPolicy(
            principal_id="user1",
            roles={Role.EDITOR},
            denied_permissions={Permission.DOCUMENT_WRITE},
        )
        permissions = policy.get_effective_permissions()
        assert Permission.DOCUMENT_WRITE not in permissions
        assert Permission.DOCUMENT_READ in permissions

    def test_extra_permissions(self):
        """Test adding extra permissions."""
        policy = RBACPolicy(
            principal_id="user1",
            roles={Role.VIEWER},
            permissions={Permission.MODEL_PREDICT},
        )
        permissions = policy.get_effective_permissions()
        assert Permission.MODEL_PREDICT in permissions
        assert Permission.DOCUMENT_READ in permissions

    def test_has_permission(self):
        """Test has_permission method."""
        policy = RBACPolicy(
            principal_id="user1",
            roles={Role.VIEWER},
        )
        assert policy.has_permission(Permission.DOCUMENT_READ) is True
        assert policy.has_permission(Permission.DOCUMENT_WRITE) is False

    def test_has_any_permission(self):
        """Test has_any_permission method."""
        policy = RBACPolicy(
            principal_id="user1",
            roles={Role.VIEWER},
        )
        assert policy.has_any_permission({Permission.DOCUMENT_READ, Permission.MODEL_DEPLOY}) is True
        assert policy.has_any_permission({Permission.DOCUMENT_WRITE, Permission.MODEL_DEPLOY}) is False

    def test_has_all_permissions(self):
        """Test has_all_permissions method."""
        policy = RBACPolicy(
            principal_id="user1",
            roles={Role.VIEWER},
        )
        assert policy.has_all_permissions({Permission.DOCUMENT_READ, Permission.ANALYSIS_READ}) is True
        assert policy.has_all_permissions({Permission.DOCUMENT_READ, Permission.DOCUMENT_WRITE}) is False

    def test_has_role(self):
        """Test has_role method."""
        policy = RBACPolicy(
            principal_id="user1",
            roles={Role.EDITOR},
        )
        assert policy.has_role(Role.EDITOR) is True
        assert policy.has_role(Role.VIEWER) is True  # Inherited
        assert policy.has_role(Role.ADMIN) is False

    def test_resource_restrictions(self):
        """Test resource-level restrictions."""
        policy = RBACPolicy(
            principal_id="user1",
            roles={Role.VIEWER},
            resource_restrictions={"document": ["doc1", "doc2"]},
        )
        assert policy.can_access_resource("document", "doc1") is True
        assert policy.can_access_resource("document", "doc3") is False
        assert policy.can_access_resource("model", "model1") is True  # No restriction


class TestRBACManager:
    """Test RBACManager class."""

    def test_register_and_get_policy(self):
        """Test policy registration and retrieval."""
        manager = RBACManager()
        policy = RBACPolicy(principal_id="user1", roles={Role.VIEWER})
        manager.register_policy(policy)

        retrieved = manager.get_policy("user1")
        assert retrieved is not None
        assert retrieved.principal_id == "user1"

    def test_tenant_specific_policy(self):
        """Test tenant-specific policies."""
        manager = RBACManager()

        # Global policy
        global_policy = RBACPolicy(principal_id="user1", roles={Role.VIEWER})
        manager.register_policy(global_policy)

        # Tenant-specific policy
        tenant_policy = RBACPolicy(
            principal_id="user1",
            roles={Role.ADMIN},
            tenant_id="tenant1",
        )
        manager.register_policy(tenant_policy)

        # Get global policy
        assert manager.get_policy("user1").roles == {Role.VIEWER}

        # Get tenant-specific policy
        assert manager.get_policy("user1", tenant_id="tenant1").roles == {Role.ADMIN}

    def test_check_permission(self):
        """Test permission checking."""
        manager = RBACManager()
        policy = RBACPolicy(principal_id="user1", roles={Role.VIEWER})
        manager.register_policy(policy)

        assert manager.check_permission("user1", Permission.DOCUMENT_READ) is True
        assert manager.check_permission("user1", Permission.DOCUMENT_WRITE) is False
        assert manager.check_permission("unknown", Permission.DOCUMENT_READ) is False

    def test_check_role(self):
        """Test role checking."""
        manager = RBACManager()
        policy = RBACPolicy(principal_id="user1", roles={Role.EDITOR})
        manager.register_policy(policy)

        assert manager.check_role("user1", Role.EDITOR) is True
        assert manager.check_role("user1", Role.VIEWER) is True  # Inherited
        assert manager.check_role("user1", Role.ADMIN) is False

    def test_grant_permission(self):
        """Test granting permissions."""
        manager = RBACManager()
        manager.grant_permission("user1", Permission.MODEL_PREDICT)

        policy = manager.get_policy("user1")
        assert policy is not None
        assert Permission.MODEL_PREDICT in policy.permissions

    def test_revoke_permission(self):
        """Test revoking permissions."""
        manager = RBACManager()
        policy = RBACPolicy(principal_id="user1", roles={Role.EDITOR})
        manager.register_policy(policy)

        manager.revoke_permission("user1", Permission.DOCUMENT_WRITE)

        assert manager.check_permission("user1", Permission.DOCUMENT_WRITE) is False
        assert Permission.DOCUMENT_WRITE in policy.denied_permissions

    def test_assign_role(self):
        """Test role assignment."""
        manager = RBACManager()
        manager.assign_role("user1", Role.ADMIN)

        policy = manager.get_policy("user1")
        assert Role.ADMIN in policy.roles


# =============================================================================
# Tenant Tests
# =============================================================================

class TestTenantContext:
    """Test TenantContext class."""

    def test_tenant_creation(self):
        """Test tenant context creation."""
        tenant = TenantContext(
            tenant_id="tenant1",
            tenant_name="Test Tenant",
            tier="enterprise",
            features={"feature1", "feature2"},
            quotas={"models": 100, "requests": 10000},
        )
        assert tenant.tenant_id == "tenant1"
        assert tenant.tier == "enterprise"

    def test_has_feature(self):
        """Test feature checking."""
        tenant = TenantContext(
            tenant_id="tenant1",
            features={"ml_training", "advanced_analytics"},
        )
        assert tenant.has_feature("ml_training") is True
        assert tenant.has_feature("unknown_feature") is False

    def test_get_quota(self):
        """Test quota retrieval."""
        tenant = TenantContext(
            tenant_id="tenant1",
            quotas={"models": 100, "requests": 10000},
        )
        assert tenant.get_quota("models") == 100
        assert tenant.get_quota("unknown", default=50) == 50


class TestTenantIsolation:
    """Test TenantIsolation class."""

    def test_register_tenant(self):
        """Test tenant registration."""
        isolation = TenantIsolation()
        tenant = TenantContext(tenant_id="tenant1", tenant_name="Test")
        isolation.register_tenant(tenant)

        assert isolation.get_tenant("tenant1") is not None
        assert isolation.get_tenant("unknown") is None

    def test_list_tenants(self):
        """Test listing tenants."""
        isolation = TenantIsolation()
        isolation.register_tenant(TenantContext(tenant_id="tenant1"))
        isolation.register_tenant(TenantContext(tenant_id="tenant2"))

        tenants = isolation.list_tenants()
        assert len(tenants) == 2

    def test_resource_registration(self):
        """Test resource registration."""
        isolation = TenantIsolation()
        isolation.register_tenant(TenantContext(tenant_id="tenant1"))
        isolation.register_resource("tenant1", "doc1")
        isolation.register_resource("tenant1", "doc2")

        resources = isolation.get_tenant_resources("tenant1")
        assert "doc1" in resources
        assert "doc2" in resources

    def test_resource_access_check(self):
        """Test resource access checking."""
        isolation = TenantIsolation()
        isolation.register_tenant(TenantContext(tenant_id="tenant1"))
        isolation.register_resource("tenant1", "doc1")

        assert isolation.check_resource_access("tenant1", "doc1") is True
        assert isolation.check_resource_access("tenant1", "doc2") is False
        assert isolation.check_resource_access("tenant2", "doc1") is False

    def test_filter_by_tenant(self):
        """Test filtering items by tenant."""
        isolation = TenantIsolation()
        items = [
            {"id": "1", "tenant_id": "tenant1", "name": "Item 1"},
            {"id": "2", "tenant_id": "tenant2", "name": "Item 2"},
            {"id": "3", "tenant_id": "tenant1", "name": "Item 3"},
        ]

        filtered = isolation.filter_by_tenant(items, "tenant1")
        assert len(filtered) == 2
        assert all(item["tenant_id"] == "tenant1" for item in filtered)


class TestTenantContextVar:
    """Test tenant context variable operations."""

    def test_set_and_get_tenant_context(self):
        """Test setting and getting tenant context."""
        tenant = TenantContext(tenant_id="tenant1")

        # Clear any existing context
        clear_tenant_context()
        assert get_current_tenant() is None

        # Set context
        token = set_tenant_context(tenant)
        assert get_current_tenant() is tenant

        # Clear context
        clear_tenant_context()
        assert get_current_tenant() is None


# =============================================================================
# Vault Tests
# =============================================================================

class TestVaultConfig:
    """Test VaultConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = VaultConfig()
        assert config.url == ""
        assert config.kv_version == 2
        assert config.rotation_buffer_seconds == 300

    def test_from_env(self):
        """Test config from environment."""
        with patch.dict("os.environ", {
            "VAULT_ADDR": "https://vault.example.com",
            "VAULT_TOKEN": "test-token",
            "VAULT_KV_VERSION": "1",
        }):
            config = VaultConfig.from_env()
            assert config.url == "https://vault.example.com"
            assert config.token == "test-token"
            assert config.kv_version == 1


class TestSecretLease:
    """Test SecretLease class."""

    def test_lease_creation(self):
        """Test lease creation."""
        lease = SecretLease(
            lease_id="lease-123",
            path="database/creds/myapp",
            secret_data={"username": "user", "password": "pass"},
            lease_duration=3600,
            renewable=True,
        )
        assert lease.lease_id == "lease-123"
        assert lease.renewable is True

    def test_expires_at(self):
        """Test expiration calculation."""
        lease = SecretLease(
            lease_id="lease-123",
            path="test",
            secret_data={},
            lease_duration=3600,
            renewable=True,
        )
        expected = lease.created_at + timedelta(seconds=3600)
        assert lease.expires_at == expected

    def test_is_expired(self):
        """Test expiration check."""
        # Expired lease
        expired_lease = SecretLease(
            lease_id="lease-123",
            path="test",
            secret_data={},
            lease_duration=0,
            renewable=True,
        )
        assert expired_lease.is_expired() is True

        # Valid lease
        valid_lease = SecretLease(
            lease_id="lease-456",
            path="test",
            secret_data={},
            lease_duration=3600,
            renewable=True,
        )
        assert valid_lease.is_expired() is False

        # Check with buffer
        short_lease = SecretLease(
            lease_id="lease-789",
            path="test",
            secret_data={},
            lease_duration=60,  # 1 minute
            renewable=True,
        )
        assert short_lease.is_expired(buffer_seconds=120) is True


class TestVaultClient:
    """Test VaultClient class."""

    def test_client_creation(self):
        """Test client creation."""
        config = VaultConfig(url="https://vault.example.com")
        client = VaultClient(config)
        assert client.config.url == "https://vault.example.com"

    def test_client_without_hvac(self):
        """Test client behavior when hvac is not installed."""
        with patch.dict("src.core.security.vault.__dict__", {"HVAC_AVAILABLE": False}):
            client = VaultClient()
            with pytest.raises(RuntimeError, match="hvac library not installed"):
                client._get_client()

    @patch("src.core.security.vault.HVAC_AVAILABLE", True)
    @patch("src.core.security.vault.hvac")
    def test_get_secret_kv_v2(self, mock_hvac):
        """Test getting secret from KV v2."""
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {"data": {"api_key": "secret-key"}}
        }
        mock_hvac.Client.return_value = mock_client

        config = VaultConfig(
            url="https://vault.example.com",
            token="test-token",
            kv_version=2,
        )
        client = VaultClient(config)

        secret = client.get_secret("myapp/config")
        assert secret == {"api_key": "secret-key"}

    @patch("src.core.security.vault.HVAC_AVAILABLE", True)
    @patch("src.core.security.vault.hvac")
    def test_put_secret_kv_v2(self, mock_hvac):
        """Test storing secret in KV v2."""
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_hvac.Client.return_value = mock_client

        config = VaultConfig(
            url="https://vault.example.com",
            token="test-token",
            kv_version=2,
        )
        client = VaultClient(config)

        client.put_secret("myapp/config", {"api_key": "new-key"})
        mock_client.secrets.kv.v2.create_or_update_secret.assert_called_once()

    @patch("src.core.security.vault.HVAC_AVAILABLE", True)
    @patch("src.core.security.vault.hvac")
    def test_get_database_credentials(self, mock_hvac):
        """Test getting dynamic database credentials."""
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_client.secrets.database.generate_credentials.return_value = {
            "lease_id": "database/creds/myapp/abc123",
            "data": {"username": "v-myapp-123", "password": "generated-pass"},
            "lease_duration": 3600,
            "renewable": True,
        }
        mock_hvac.Client.return_value = mock_client

        config = VaultConfig(url="https://vault.example.com", token="test-token")
        client = VaultClient(config)

        creds = client.get_database_credentials("myapp")
        assert creds["username"] == "v-myapp-123"
        assert "database/creds/myapp/abc123" in client._leases

    @patch("src.core.security.vault.HVAC_AVAILABLE", True)
    @patch("src.core.security.vault.hvac")
    def test_renew_lease(self, mock_hvac):
        """Test lease renewal."""
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_client.sys.renew_lease.return_value = {"lease_duration": 7200}
        mock_hvac.Client.return_value = mock_client

        config = VaultConfig(url="https://vault.example.com", token="test-token")
        client = VaultClient(config)

        # Add a tracked lease
        lease = SecretLease(
            lease_id="test-lease",
            path="test/path",
            secret_data={},
            lease_duration=3600,
            renewable=True,
        )
        client._leases["test-lease"] = lease

        result = client.renew_lease("test-lease")
        assert result is True
        assert client._leases["test-lease"].lease_duration == 7200

    @patch("src.core.security.vault.HVAC_AVAILABLE", True)
    @patch("src.core.security.vault.hvac")
    def test_revoke_lease(self, mock_hvac):
        """Test lease revocation."""
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_hvac.Client.return_value = mock_client

        config = VaultConfig(url="https://vault.example.com", token="test-token")
        client = VaultClient(config)

        # Add a tracked lease
        client._leases["test-lease"] = SecretLease(
            lease_id="test-lease",
            path="test/path",
            secret_data={},
            lease_duration=3600,
            renewable=True,
        )

        result = client.revoke_lease("test-lease")
        assert result is True
        assert "test-lease" not in client._leases

    @patch("src.core.security.vault.HVAC_AVAILABLE", True)
    @patch("src.core.security.vault.hvac")
    def test_health_check(self, mock_hvac):
        """Test health check."""
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_client.sys.is_initialized.return_value = True
        mock_client.sys.is_sealed.return_value = False
        mock_hvac.Client.return_value = mock_client

        config = VaultConfig(url="https://vault.example.com", token="test-token")
        client = VaultClient(config)

        assert client.health_check() is True

    def test_rotation_callback_registration(self):
        """Test rotation callback registration."""
        client = VaultClient(VaultConfig())

        callback = Mock()
        client.register_rotation_callback("database/creds/myapp", callback)

        assert "database/creds/myapp" in client._rotation_callbacks
        assert callback in client._rotation_callbacks["database/creds/myapp"]


# =============================================================================
# Decorator Tests
# =============================================================================

class TestRBACDecorators:
    """Test RBAC decorators."""

    @pytest.fixture
    def setup_manager(self):
        """Setup RBAC manager with test policy."""
        # Reset global manager
        import src.core.security.rbac as rbac_module
        rbac_module._rbac_manager = None

        manager = get_rbac_manager()
        policy = RBACPolicy(
            principal_id="test_user",
            roles={Role.EDITOR},
        )
        manager.register_policy(policy)
        return manager

    def test_require_permission_allowed(self, setup_manager):
        """Test require_permission with allowed permission."""
        @require_permission(Permission.DOCUMENT_READ)
        def read_document(request):
            return "document content"

        # Mock request
        class MockRequest:
            user_id = "test_user"
            tenant_id = None

        result = read_document(MockRequest())
        assert result == "document content"

    def test_require_permission_denied(self, setup_manager):
        """Test require_permission with denied permission."""
        @require_permission(Permission.SYSTEM_ADMIN)
        def admin_action():
            return "admin content"

        class MockRequest:
            user_id = "test_user"
            tenant_id = None

        with pytest.raises(Exception) as exc_info:
            admin_action(MockRequest())
        assert "Permission denied" in str(exc_info.value.detail)


class TestTenantDecorators:
    """Test tenant decorators."""

    def test_require_tenant_with_context(self):
        """Test require_tenant with tenant context."""
        tenant = TenantContext(tenant_id="tenant1")
        set_tenant_context(tenant)

        @require_tenant
        def tenant_action():
            return "tenant content"

        result = tenant_action()
        assert result == "tenant content"

        clear_tenant_context()

    def test_require_tenant_without_context(self):
        """Test require_tenant without tenant context."""
        clear_tenant_context()

        @require_tenant
        def tenant_action():
            return "tenant content"

        with pytest.raises(Exception) as exc_info:
            tenant_action()
        assert "Tenant context required" in str(exc_info.value.detail)

    def test_require_feature_with_feature(self):
        """Test require_feature with feature enabled."""
        tenant = TenantContext(
            tenant_id="tenant1",
            features={"ml_training"},
        )
        set_tenant_context(tenant)

        @require_feature("ml_training")
        def ml_action():
            return "ml content"

        result = ml_action()
        assert result == "ml content"

        clear_tenant_context()

    def test_require_feature_without_feature(self):
        """Test require_feature without feature enabled."""
        tenant = TenantContext(
            tenant_id="tenant1",
            features=set(),
        )
        set_tenant_context(tenant)

        @require_feature("ml_training")
        def ml_action():
            return "ml content"

        with pytest.raises(Exception) as exc_info:
            ml_action()
        assert "Feature not available" in str(exc_info.value.detail)

        clear_tenant_context()


# =============================================================================
# Integration Tests
# =============================================================================

class TestSecurityIntegration:
    """Integration tests for security components."""

    def test_rbac_with_tenant_isolation(self):
        """Test RBAC combined with tenant isolation."""
        # Setup tenant
        isolation = TenantIsolation()
        tenant = TenantContext(
            tenant_id="tenant1",
            features={"advanced_ml"},
        )
        isolation.register_tenant(tenant)

        # Setup RBAC
        manager = RBACManager()
        policy = RBACPolicy(
            principal_id="user1",
            roles={Role.EDITOR},
            tenant_id="tenant1",
        )
        manager.register_policy(policy)

        # Check access
        assert manager.check_permission(
            "user1",
            Permission.DOCUMENT_WRITE,
            tenant_id="tenant1",
        ) is True

        # Different tenant should not have access
        assert manager.get_policy("user1", tenant_id="tenant2") is None

    def test_super_admin_access(self):
        """Test super admin has all permissions."""
        manager = RBACManager()
        policy = RBACPolicy(
            principal_id="super_user",
            roles={Role.SUPER_ADMIN},
        )
        manager.register_policy(policy)

        # Super admin should have all top-level permissions
        assert manager.check_permission("super_user", Permission.SYSTEM_ADMIN) is True
        assert manager.check_permission("super_user", Permission.TENANT_ADMIN) is True
        assert manager.check_permission("super_user", Permission.MODEL_DEPLOY) is True

        # And inherited permissions
        assert manager.check_permission("super_user", Permission.DOCUMENT_READ) is True
        assert manager.check_permission("super_user", Permission.MODEL_PREDICT) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
