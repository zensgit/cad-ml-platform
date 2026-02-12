"""
Unit tests for rbac.py - Role-Based Access Control Module.

Tests cover:
- Role creation and permission management
- User management and role assignment
- Permission checking and inheritance
- Resource-based access control
- Policy enforcement
- AccessContext for request scoping
"""

import pytest
from src.core.assistant.rbac import (
    Permission,
    ResourceType,
    Role,
    User,
    Resource,
    Policy,
    RBACManager,
    AccessContext,
    require_permission,
)


class TestPermission:
    """Tests for Permission enum."""

    def test_conversation_permissions(self):
        """Test conversation permission values."""
        assert Permission.CONVERSATION_CREATE.value == "conversation:create"
        assert Permission.CONVERSATION_READ.value == "conversation:read"
        assert Permission.CONVERSATION_UPDATE.value == "conversation:update"
        assert Permission.CONVERSATION_DELETE.value == "conversation:delete"

    def test_knowledge_permissions(self):
        """Test knowledge permission values."""
        assert Permission.KNOWLEDGE_CREATE.value == "knowledge:create"
        assert Permission.KNOWLEDGE_READ.value == "knowledge:read"
        assert Permission.KNOWLEDGE_IMPORT.value == "knowledge:import"
        assert Permission.KNOWLEDGE_EXPORT.value == "knowledge:export"

    def test_admin_permissions(self):
        """Test admin permission values."""
        assert Permission.ADMIN_USER_MANAGE.value == "admin:user:manage"
        assert Permission.ADMIN_TENANT_MANAGE.value == "admin:tenant:manage"
        assert Permission.ADMIN_SYSTEM_CONFIG.value == "admin:system:config"


class TestResourceType:
    """Tests for ResourceType enum."""

    def test_all_resource_types(self):
        """Test all resource type values."""
        assert ResourceType.CONVERSATION.value == "conversation"
        assert ResourceType.KNOWLEDGE.value == "knowledge"
        assert ResourceType.MODEL.value == "model"
        assert ResourceType.ANALYTICS.value == "analytics"
        assert ResourceType.SYSTEM.value == "system"
        assert ResourceType.USER.value == "user"
        assert ResourceType.TENANT.value == "tenant"


class TestRole:
    """Tests for Role class."""

    def test_role_creation(self):
        """Test basic role creation."""
        role = Role(
            name="test_role",
            description="Test role",
            permissions={Permission.CONVERSATION_READ},
        )
        assert role.name == "test_role"
        assert Permission.CONVERSATION_READ in role.permissions

    def test_has_permission(self):
        """Test permission checking."""
        role = Role(
            name="reader",
            description="Read-only",
            permissions={Permission.CONVERSATION_READ, Permission.KNOWLEDGE_READ},
        )
        assert role.has_permission(Permission.CONVERSATION_READ) is True
        assert role.has_permission(Permission.CONVERSATION_CREATE) is False

    def test_add_permission(self):
        """Test adding permission to role."""
        role = Role(name="test", description="Test")
        role.add_permission(Permission.MODEL_USE)

        assert role.has_permission(Permission.MODEL_USE) is True

    def test_remove_permission(self):
        """Test removing permission from role."""
        role = Role(
            name="test",
            description="Test",
            permissions={Permission.KNOWLEDGE_READ},
        )
        role.remove_permission(Permission.KNOWLEDGE_READ)

        assert role.has_permission(Permission.KNOWLEDGE_READ) is False

    def test_to_dict(self):
        """Test role serialization."""
        role = Role(
            name="admin",
            description="Administrator",
            permissions={Permission.ADMIN_USER_MANAGE},
            inherits_from="manager",
        )
        result = role.to_dict()

        assert result["name"] == "admin"
        assert result["description"] == "Administrator"
        assert "admin:user:manage" in result["permissions"]
        assert result["inherits_from"] == "manager"


class TestUser:
    """Tests for User class."""

    def test_user_creation(self):
        """Test basic user creation."""
        user = User(
            id="user-123",
            username="john",
            email="john@example.com",
        )
        assert user.id == "user-123"
        assert user.username == "john"
        assert user.is_active is True

    def test_user_with_roles(self):
        """Test user with roles."""
        user = User(
            id="user-456",
            username="jane",
            roles={"engineer", "analyst"},
        )
        assert "engineer" in user.roles
        assert "analyst" in user.roles

    def test_user_to_dict(self):
        """Test user serialization excludes sensitive data."""
        user = User(
            id="user-789",
            username="test",
            tenant_id="tenant-1",
        )
        result = user.to_dict()

        assert result["id"] == "user-789"
        assert result["username"] == "test"
        assert result["tenant_id"] == "tenant-1"


class TestResource:
    """Tests for Resource class."""

    def test_resource_creation(self):
        """Test basic resource creation."""
        resource = Resource(
            id="res-123",
            type=ResourceType.CONVERSATION,
            owner_id="user-456",
        )
        assert resource.id == "res-123"
        assert resource.type == ResourceType.CONVERSATION

    def test_resource_to_dict(self):
        """Test resource serialization."""
        resource = Resource(
            id="res-789",
            type=ResourceType.KNOWLEDGE,
            owner_id="user-1",
            tenant_id="tenant-1",
        )
        result = resource.to_dict()

        assert result["id"] == "res-789"
        assert result["type"] == "knowledge"
        assert result["owner_id"] == "user-1"


class TestPolicy:
    """Tests for Policy class."""

    def test_policy_creation(self):
        """Test basic policy creation."""
        policy = Policy(
            name="knowledge_access",
            description="Knowledge access policy",
            resource_type=ResourceType.KNOWLEDGE,
            required_permissions={Permission.KNOWLEDGE_READ},
        )
        assert policy.name == "knowledge_access"
        assert policy.resource_type == ResourceType.KNOWLEDGE

    def test_policy_to_dict(self):
        """Test policy serialization."""
        policy = Policy(
            name="admin_policy",
            description="Admin policy",
            resource_type=ResourceType.SYSTEM,
            required_permissions={Permission.ADMIN_SYSTEM_CONFIG},
            priority=10,
        )
        result = policy.to_dict()

        assert result["name"] == "admin_policy"
        assert result["resource_type"] == "system"
        assert result["priority"] == 10


class TestRBACManager:
    """Tests for RBACManager class."""

    @pytest.fixture
    def rbac(self):
        """Create an RBAC manager with default roles."""
        rbac = RBACManager()
        rbac.create_default_roles()
        return rbac

    def test_create_default_roles(self, rbac):
        """Test default roles are created."""
        assert rbac.get_role("guest") is not None
        assert rbac.get_role("user") is not None
        assert rbac.get_role("engineer") is not None
        assert rbac.get_role("manager") is not None
        assert rbac.get_role("admin") is not None

    def test_create_role(self, rbac):
        """Test custom role creation."""
        role = rbac.create_role(
            "custom",
            "Custom role",
            {Permission.API_ACCESS},
        )
        assert role.name == "custom"
        assert rbac.get_role("custom") is not None

    def test_delete_role(self, rbac):
        """Test role deletion."""
        rbac.create_role("temp", "Temporary", set())
        result = rbac.delete_role("temp")

        assert result is True
        assert rbac.get_role("temp") is None

    def test_get_effective_permissions(self, rbac):
        """Test permission inheritance."""
        # Engineer inherits from user
        perms = rbac.get_effective_permissions("engineer")

        # Should have user permissions
        assert Permission.CONVERSATION_CREATE in perms
        # Should have engineer-specific permissions
        assert Permission.KNOWLEDGE_CREATE in perms

    def test_get_effective_permissions_nested_inheritance(self, rbac):
        """Test nested permission inheritance."""
        # Admin inherits from manager, which inherits from engineer, which inherits from user
        perms = rbac.get_effective_permissions("admin")

        # Should have permissions from all levels
        assert Permission.CONVERSATION_CREATE in perms  # user
        assert Permission.KNOWLEDGE_CREATE in perms  # engineer
        assert Permission.ADMIN_USER_MANAGE in perms  # manager
        assert Permission.ADMIN_SYSTEM_CONFIG in perms  # admin

    def test_create_user(self, rbac):
        """Test user creation."""
        user = rbac.create_user("u1", "testuser", "test@example.com")

        assert user.id == "u1"
        assert user.username == "testuser"
        assert rbac.get_user("u1") is not None

    def test_delete_user(self, rbac):
        """Test user deletion."""
        rbac.create_user("u2", "tempuser")
        result = rbac.delete_user("u2")

        assert result is True
        assert rbac.get_user("u2") is None

    def test_assign_role(self, rbac):
        """Test role assignment."""
        rbac.create_user("u3", "roleuser")
        result = rbac.assign_role("u3", "engineer")

        assert result is True
        user = rbac.get_user("u3")
        assert "engineer" in user.roles

    def test_assign_invalid_role(self, rbac):
        """Test assigning non-existent role."""
        rbac.create_user("u4", "user")
        result = rbac.assign_role("u4", "nonexistent")

        assert result is False

    def test_revoke_role(self, rbac):
        """Test role revocation."""
        rbac.create_user("u5", "user")
        rbac.assign_role("u5", "user")
        result = rbac.revoke_role("u5", "user")

        assert result is True
        user = rbac.get_user("u5")
        assert "user" not in user.roles

    def test_grant_permission(self, rbac):
        """Test granting direct permission."""
        rbac.create_user("u6", "user")
        result = rbac.grant_permission("u6", Permission.ADMIN_AUDIT_LOG)

        assert result is True
        user = rbac.get_user("u6")
        assert Permission.ADMIN_AUDIT_LOG in user.direct_permissions

    def test_deny_permission(self, rbac):
        """Test denying permission."""
        rbac.create_user("u7", "user")
        rbac.assign_role("u7", "admin")
        rbac.deny_permission("u7", Permission.ADMIN_SYSTEM_CONFIG)

        user = rbac.get_user("u7")
        assert Permission.ADMIN_SYSTEM_CONFIG in user.denied_permissions

    def test_get_user_permissions(self, rbac):
        """Test getting all user permissions."""
        rbac.create_user("u8", "user")
        rbac.assign_role("u8", "engineer")
        rbac.grant_permission("u8", Permission.ADMIN_AUDIT_LOG)

        perms = rbac.get_user_permissions("u8")

        # Should have role permissions
        assert Permission.KNOWLEDGE_CREATE in perms
        # Should have direct permission
        assert Permission.ADMIN_AUDIT_LOG in perms

    def test_get_user_permissions_with_denial(self, rbac):
        """Test denied permissions are excluded."""
        rbac.create_user("u9", "user")
        rbac.assign_role("u9", "admin")
        rbac.deny_permission("u9", Permission.ADMIN_SYSTEM_CONFIG)

        perms = rbac.get_user_permissions("u9")

        assert Permission.ADMIN_SYSTEM_CONFIG not in perms

    def test_get_user_permissions_inactive(self, rbac):
        """Test inactive user has no permissions."""
        user = rbac.create_user("u10", "user")
        user.is_active = False

        perms = rbac.get_user_permissions("u10")
        assert len(perms) == 0

    def test_check_permission(self, rbac):
        """Test permission checking."""
        rbac.create_user("u11", "user")
        rbac.assign_role("u11", "user")

        assert rbac.check_permission("u11", Permission.CONVERSATION_CREATE) is True
        assert rbac.check_permission("u11", Permission.ADMIN_SYSTEM_CONFIG) is False

    def test_check_resource_access(self, rbac):
        """Test resource-based access control."""
        rbac.create_user("u12", "owner", tenant_id="t1")
        rbac.assign_role("u12", "user")

        resource = rbac.register_resource(
            "res-1",
            ResourceType.CONVERSATION,
            owner_id="u12",
            tenant_id="t1",
        )

        assert rbac.check_resource_access(
            "u12",
            "res-1",
            Permission.CONVERSATION_READ
        ) is True

    def test_check_resource_access_tenant_isolation(self, rbac):
        """Test tenant isolation in resource access."""
        rbac.create_user("u13", "user1", tenant_id="t1")
        rbac.create_user("u14", "user2", tenant_id="t2")
        rbac.assign_role("u13", "user")
        rbac.assign_role("u14", "user")

        rbac.register_resource(
            "res-2",
            ResourceType.CONVERSATION,
            owner_id="u13",
            tenant_id="t1",
        )

        # Owner can access
        assert rbac.check_resource_access("u13", "res-2", Permission.CONVERSATION_READ) is True
        # Different tenant cannot
        assert rbac.check_resource_access("u14", "res-2", Permission.CONVERSATION_READ) is False

    def test_create_policy(self, rbac):
        """Test policy creation."""
        policy = rbac.create_policy(
            "test_policy",
            "Test policy",
            ResourceType.KNOWLEDGE,
            {Permission.KNOWLEDGE_READ},
        )

        assert policy.name == "test_policy"
        assert "test_policy" in rbac._policies

    def test_list_roles(self, rbac):
        """Test listing roles."""
        roles = rbac.list_roles()

        assert len(roles) >= 5  # Default roles
        names = [r["name"] for r in roles]
        assert "admin" in names
        assert "user" in names

    def test_list_users(self, rbac):
        """Test listing users."""
        rbac.create_user("u15", "user1", tenant_id="t1")
        rbac.create_user("u16", "user2", tenant_id="t1")
        rbac.create_user("u17", "user3", tenant_id="t2")

        all_users = rbac.list_users()
        assert len(all_users) == 3

        t1_users = rbac.list_users(tenant_id="t1")
        assert len(t1_users) == 2


class TestAccessContext:
    """Tests for AccessContext class."""

    @pytest.fixture
    def rbac_with_user(self):
        """Create RBAC with a test user."""
        rbac = RBACManager()
        rbac.create_default_roles()
        rbac.create_user("ctx-user", "contextuser")
        rbac.assign_role("ctx-user", "engineer")
        return rbac

    def test_context_entry_exit(self, rbac_with_user):
        """Test context manager entry and exit."""
        with AccessContext(rbac_with_user, "ctx-user") as ctx:
            assert ctx.user_id == "ctx-user"
            assert ctx._cached_permissions is not None

        assert ctx._cached_permissions is None

    def test_context_can(self, rbac_with_user):
        """Test permission checking in context."""
        with AccessContext(rbac_with_user, "ctx-user") as ctx:
            assert ctx.can(Permission.KNOWLEDGE_CREATE) is True
            assert ctx.can(Permission.ADMIN_SYSTEM_CONFIG) is False

    def test_context_can_access(self, rbac_with_user):
        """Test resource access in context."""
        rbac_with_user.register_resource(
            "ctx-res",
            ResourceType.KNOWLEDGE,
            owner_id="ctx-user",
        )

        with AccessContext(rbac_with_user, "ctx-user") as ctx:
            assert ctx.can_access("ctx-res", Permission.KNOWLEDGE_READ) is True

    def test_context_require_success(self, rbac_with_user):
        """Test require with valid permission."""
        with AccessContext(rbac_with_user, "ctx-user") as ctx:
            ctx.require(Permission.KNOWLEDGE_CREATE)  # Should not raise

    def test_context_require_failure(self, rbac_with_user):
        """Test require with missing permission."""
        with AccessContext(rbac_with_user, "ctx-user") as ctx:
            with pytest.raises(PermissionError):
                ctx.require(Permission.ADMIN_SYSTEM_CONFIG)

    def test_context_get_permissions(self, rbac_with_user):
        """Test getting permissions in context."""
        with AccessContext(rbac_with_user, "ctx-user") as ctx:
            perms = ctx.get_permissions()
            assert Permission.KNOWLEDGE_CREATE in perms


class TestRequirePermissionDecorator:
    """Tests for require_permission decorator."""

    def test_decorator_allows_authorized(self):
        """Test decorator allows authorized access."""
        rbac = RBACManager()
        rbac.create_default_roles()
        rbac.create_user("dec-user", "user")
        rbac.assign_role("dec-user", "user")

        @require_permission(Permission.CONVERSATION_CREATE)
        def create_conversation(user_id: str, data: dict, rbac: RBACManager):
            return "created"

        result = create_conversation("dec-user", {}, rbac=rbac)
        assert result == "created"

    def test_decorator_denies_unauthorized(self):
        """Test decorator denies unauthorized access."""
        rbac = RBACManager()
        rbac.create_default_roles()
        rbac.create_user("dec-user2", "user")
        rbac.assign_role("dec-user2", "guest")  # Guest has limited permissions

        @require_permission(Permission.ADMIN_SYSTEM_CONFIG)
        def admin_action(user_id: str, rbac: RBACManager):
            return "done"

        with pytest.raises(PermissionError):
            admin_action("dec-user2", rbac=rbac)


class TestDefaultRoles:
    """Tests for default role configurations."""

    @pytest.fixture
    def rbac(self):
        """Create RBAC with default roles."""
        rbac = RBACManager()
        rbac.create_default_roles()
        return rbac

    def test_guest_role_permissions(self, rbac):
        """Test guest role has minimal permissions."""
        perms = rbac.get_effective_permissions("guest")
        assert Permission.CONVERSATION_READ in perms
        assert Permission.KNOWLEDGE_READ in perms
        assert Permission.CONVERSATION_CREATE not in perms

    def test_user_role_permissions(self, rbac):
        """Test user role permissions."""
        perms = rbac.get_effective_permissions("user")
        assert Permission.CONVERSATION_CREATE in perms
        assert Permission.MODEL_USE in perms
        assert Permission.API_ACCESS in perms

    def test_engineer_role_permissions(self, rbac):
        """Test engineer role inherits from user."""
        perms = rbac.get_effective_permissions("engineer")
        # User permissions
        assert Permission.CONVERSATION_CREATE in perms
        # Engineer-specific
        assert Permission.KNOWLEDGE_CREATE in perms
        assert Permission.KNOWLEDGE_IMPORT in perms
        assert Permission.ANALYTICS_VIEW in perms

    def test_manager_role_permissions(self, rbac):
        """Test manager role inherits from engineer."""
        perms = rbac.get_effective_permissions("manager")
        # Engineer permissions
        assert Permission.KNOWLEDGE_CREATE in perms
        # Manager-specific
        assert Permission.ADMIN_USER_MANAGE in perms
        assert Permission.ANALYTICS_EXPORT in perms

    def test_admin_role_permissions(self, rbac):
        """Test admin role has full access."""
        perms = rbac.get_effective_permissions("admin")
        # Should have everything
        assert Permission.ADMIN_SYSTEM_CONFIG in perms
        assert Permission.ADMIN_TENANT_MANAGE in perms
        assert Permission.API_RATE_UNLIMITED in perms
        assert Permission.KNOWLEDGE_DELETE in perms


class TestRBACManagerEdgeCases:
    """Tests for RBAC edge cases to improve coverage."""

    @pytest.fixture
    def rbac(self):
        """Create an RBAC manager with default roles."""
        rbac = RBACManager()
        rbac.create_default_roles()
        return rbac

    def test_delete_role_nonexistent(self, rbac):
        """Test deleting a role that doesn't exist returns False."""
        result = rbac.delete_role("nonexistent_role")
        assert result is False

    def test_get_effective_permissions_nonexistent_role(self, rbac):
        """Test get_effective_permissions for nonexistent role returns empty set."""
        perms = rbac.get_effective_permissions("nonexistent_role")
        assert perms == set()

    def test_delete_user_nonexistent(self, rbac):
        """Test deleting a user that doesn't exist returns False."""
        result = rbac.delete_user("nonexistent_user")
        assert result is False

    def test_revoke_role_nonexistent_user(self, rbac):
        """Test revoking role from nonexistent user returns False."""
        result = rbac.revoke_role("nonexistent_user", "user")
        assert result is False

    def test_grant_permission_nonexistent_user(self, rbac):
        """Test granting permission to nonexistent user returns False."""
        result = rbac.grant_permission("nonexistent_user", Permission.API_ACCESS)
        assert result is False

    def test_deny_permission_nonexistent_user(self, rbac):
        """Test denying permission to nonexistent user returns False."""
        result = rbac.deny_permission("nonexistent_user", Permission.API_ACCESS)
        assert result is False

    def test_check_resource_access_missing_user(self, rbac):
        """Test resource access check with nonexistent user returns False."""
        rbac.register_resource("res-test", ResourceType.CONVERSATION, "some-owner")
        result = rbac.check_resource_access("nonexistent_user", "res-test", Permission.CONVERSATION_READ)
        assert result is False

    def test_check_resource_access_missing_resource(self, rbac):
        """Test resource access check with nonexistent resource returns False."""
        rbac.create_user("u-test", "testuser")
        result = rbac.check_resource_access("u-test", "nonexistent_resource", Permission.CONVERSATION_READ)
        assert result is False

    def test_check_resource_access_no_permission(self, rbac):
        """Test resource access check when user lacks permission returns False."""
        rbac.create_user("u-noperm", "noperm", tenant_id="t1")
        # Don't assign any role
        rbac.register_resource("res-perm", ResourceType.CONVERSATION, "other-owner", tenant_id="t1")

        result = rbac.check_resource_access("u-noperm", "res-perm", Permission.CONVERSATION_READ)
        assert result is False

    def test_check_resource_access_with_policy(self, rbac):
        """Test resource access with policy enforcement."""
        rbac.create_user("u-policy", "policyuser", tenant_id="t1")
        rbac.assign_role("u-policy", "user")

        rbac.register_resource("res-policy", ResourceType.KNOWLEDGE, "other-owner", tenant_id="t1")

        # Create a policy that requires KNOWLEDGE_CREATE for KNOWLEDGE resources
        rbac.create_policy(
            "knowledge_edit_policy",
            "Requires create permission",
            ResourceType.KNOWLEDGE,
            {Permission.KNOWLEDGE_CREATE},
            priority=10,
        )

        # User role doesn't have KNOWLEDGE_CREATE
        result = rbac.check_resource_access("u-policy", "res-policy", Permission.KNOWLEDGE_READ)
        assert result is False

    def test_check_resource_access_with_policy_satisfied(self, rbac):
        """Test resource access with policy requirements satisfied."""
        rbac.create_user("u-policy-ok", "policyok", tenant_id="t1")
        rbac.assign_role("u-policy-ok", "engineer")  # Has KNOWLEDGE_CREATE

        rbac.register_resource("res-policy-ok", ResourceType.KNOWLEDGE, "other-owner", tenant_id="t1")

        # Create a policy that requires KNOWLEDGE_CREATE
        rbac.create_policy(
            "knowledge_policy_ok",
            "Requires create permission",
            ResourceType.KNOWLEDGE,
            {Permission.KNOWLEDGE_CREATE},
            priority=5,
        )

        result = rbac.check_resource_access("u-policy-ok", "res-policy-ok", Permission.KNOWLEDGE_READ)
        assert result is True


class TestAccessContextEdgeCases:
    """Tests for AccessContext edge cases."""

    @pytest.fixture
    def rbac_with_user(self):
        """Create RBAC with a test user."""
        rbac = RBACManager()
        rbac.create_default_roles()
        rbac.create_user("edge-user", "edgeuser")
        rbac.assign_role("edge-user", "engineer")
        return rbac

    def test_can_without_cache(self, rbac_with_user):
        """Test can() method without cached permissions."""
        ctx = AccessContext(rbac_with_user, "edge-user")
        # Don't enter context, so cache is not populated
        result = ctx.can(Permission.KNOWLEDGE_CREATE)
        assert result is True

    def test_get_permissions_without_cache(self, rbac_with_user):
        """Test get_permissions() method without cached permissions."""
        ctx = AccessContext(rbac_with_user, "edge-user")
        # Don't enter context, so cache is not populated
        perms = ctx.get_permissions()
        assert Permission.KNOWLEDGE_CREATE in perms
