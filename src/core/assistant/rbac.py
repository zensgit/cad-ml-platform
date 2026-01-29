"""
Role-Based Access Control (RBAC) Module for CAD Assistant.

Provides fine-grained permission management and access control.
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Union


class Permission(Enum):
    """Available permissions in the system."""

    # Conversation permissions
    CONVERSATION_CREATE = "conversation:create"
    CONVERSATION_READ = "conversation:read"
    CONVERSATION_UPDATE = "conversation:update"
    CONVERSATION_DELETE = "conversation:delete"

    # Knowledge permissions
    KNOWLEDGE_CREATE = "knowledge:create"
    KNOWLEDGE_READ = "knowledge:read"
    KNOWLEDGE_UPDATE = "knowledge:update"
    KNOWLEDGE_DELETE = "knowledge:delete"
    KNOWLEDGE_IMPORT = "knowledge:import"
    KNOWLEDGE_EXPORT = "knowledge:export"

    # Model permissions
    MODEL_USE = "model:use"
    MODEL_CONFIGURE = "model:configure"
    MODEL_SWITCH = "model:switch"

    # Analytics permissions
    ANALYTICS_VIEW = "analytics:view"
    ANALYTICS_EXPORT = "analytics:export"
    ANALYTICS_ADMIN = "analytics:admin"

    # Admin permissions
    ADMIN_USER_MANAGE = "admin:user:manage"
    ADMIN_TENANT_MANAGE = "admin:tenant:manage"
    ADMIN_SYSTEM_CONFIG = "admin:system:config"
    ADMIN_AUDIT_LOG = "admin:audit:log"

    # API permissions
    API_ACCESS = "api:access"
    API_RATE_UNLIMITED = "api:rate:unlimited"


class ResourceType(Enum):
    """Types of resources that can be protected."""

    CONVERSATION = "conversation"
    KNOWLEDGE = "knowledge"
    MODEL = "model"
    ANALYTICS = "analytics"
    SYSTEM = "system"
    USER = "user"
    TENANT = "tenant"


@dataclass
class Role:
    """A role with a set of permissions."""

    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    inherits_from: Optional[str] = None  # Parent role name
    created_at: float = field(default_factory=time.time)

    def has_permission(self, permission: Permission) -> bool:
        """Check if role has a specific permission."""
        return permission in self.permissions

    def add_permission(self, permission: Permission) -> None:
        """Add a permission to this role."""
        self.permissions.add(permission)

    def remove_permission(self, permission: Permission) -> None:
        """Remove a permission from this role."""
        self.permissions.discard(permission)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "permissions": [p.value for p in self.permissions],
            "inherits_from": self.inherits_from,
            "created_at": self.created_at,
        }


@dataclass
class User:
    """A user with roles and permissions."""

    id: str
    username: str
    email: Optional[str] = None
    roles: Set[str] = field(default_factory=set)  # Role names
    direct_permissions: Set[Permission] = field(default_factory=set)
    denied_permissions: Set[Permission] = field(default_factory=set)
    tenant_id: Optional[str] = None
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    last_login: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive data)."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "roles": list(self.roles),
            "tenant_id": self.tenant_id,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "last_login": self.last_login,
        }


@dataclass
class Resource:
    """A protected resource."""

    id: str
    type: ResourceType
    owner_id: str
    tenant_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "owner_id": self.owner_id,
            "tenant_id": self.tenant_id,
            "metadata": self.metadata,
        }


@dataclass
class Policy:
    """Access control policy."""

    name: str
    description: str
    resource_type: ResourceType
    required_permissions: Set[Permission]
    conditions: Optional[Dict[str, Any]] = None
    priority: int = 0  # Higher = evaluated first

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "resource_type": self.resource_type.value,
            "required_permissions": [p.value for p in self.required_permissions],
            "conditions": self.conditions,
            "priority": self.priority,
        }


class RBACManager:
    """
    Manages roles, users, and access control.

    Example:
        >>> rbac = RBACManager()
        >>> rbac.create_default_roles()
        >>> rbac.assign_role(user_id, "engineer")
        >>> if rbac.check_permission(user_id, Permission.KNOWLEDGE_READ):
        ...     # Allow access
    """

    def __init__(self):
        """Initialize RBAC manager."""
        self._roles: Dict[str, Role] = {}
        self._users: Dict[str, User] = {}
        self._policies: Dict[str, Policy] = {}
        self._resources: Dict[str, Resource] = {}

    def create_default_roles(self) -> None:
        """Create default system roles."""
        # Guest role - minimal permissions
        self.create_role(
            "guest",
            "Guest user with read-only access",
            {Permission.CONVERSATION_READ, Permission.KNOWLEDGE_READ},
        )

        # User role - standard permissions
        self.create_role(
            "user",
            "Standard user with full conversation access",
            {
                Permission.CONVERSATION_CREATE,
                Permission.CONVERSATION_READ,
                Permission.CONVERSATION_UPDATE,
                Permission.CONVERSATION_DELETE,
                Permission.KNOWLEDGE_READ,
                Permission.MODEL_USE,
                Permission.API_ACCESS,
            },
        )

        # Engineer role - extends user
        self.create_role(
            "engineer",
            "Engineer with knowledge management",
            {
                Permission.KNOWLEDGE_CREATE,
                Permission.KNOWLEDGE_UPDATE,
                Permission.KNOWLEDGE_IMPORT,
                Permission.KNOWLEDGE_EXPORT,
                Permission.ANALYTICS_VIEW,
            },
            inherits_from="user",
        )

        # Manager role - extends engineer
        self.create_role(
            "manager",
            "Manager with team oversight",
            {
                Permission.ANALYTICS_EXPORT,
                Permission.ADMIN_USER_MANAGE,
            },
            inherits_from="engineer",
        )

        # Admin role - full access
        self.create_role(
            "admin",
            "Administrator with full system access",
            {
                Permission.KNOWLEDGE_DELETE,
                Permission.MODEL_CONFIGURE,
                Permission.MODEL_SWITCH,
                Permission.ANALYTICS_ADMIN,
                Permission.ADMIN_TENANT_MANAGE,
                Permission.ADMIN_SYSTEM_CONFIG,
                Permission.ADMIN_AUDIT_LOG,
                Permission.API_RATE_UNLIMITED,
            },
            inherits_from="manager",
        )

    def create_role(
        self,
        name: str,
        description: str,
        permissions: Set[Permission],
        inherits_from: Optional[str] = None,
    ) -> Role:
        """Create a new role."""
        role = Role(
            name=name,
            description=description,
            permissions=permissions,
            inherits_from=inherits_from,
        )
        self._roles[name] = role
        return role

    def get_role(self, name: str) -> Optional[Role]:
        """Get a role by name."""
        return self._roles.get(name)

    def delete_role(self, name: str) -> bool:
        """Delete a role."""
        if name in self._roles:
            del self._roles[name]
            return True
        return False

    def get_effective_permissions(self, role_name: str) -> Set[Permission]:
        """Get all permissions for a role, including inherited."""
        role = self._roles.get(role_name)
        if not role:
            return set()

        permissions = role.permissions.copy()

        # Add inherited permissions
        if role.inherits_from:
            permissions.update(self.get_effective_permissions(role.inherits_from))

        return permissions

    def create_user(
        self,
        user_id: str,
        username: str,
        email: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> User:
        """Create a new user."""
        user = User(
            id=user_id,
            username=username,
            email=email,
            tenant_id=tenant_id,
        )
        self._users[user_id] = user
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        return self._users.get(user_id)

    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        if user_id in self._users:
            del self._users[user_id]
            return True
        return False

    def assign_role(self, user_id: str, role_name: str) -> bool:
        """Assign a role to a user."""
        user = self._users.get(user_id)
        if not user or role_name not in self._roles:
            return False
        user.roles.add(role_name)
        return True

    def revoke_role(self, user_id: str, role_name: str) -> bool:
        """Revoke a role from a user."""
        user = self._users.get(user_id)
        if not user:
            return False
        user.roles.discard(role_name)
        return True

    def grant_permission(self, user_id: str, permission: Permission) -> bool:
        """Grant a direct permission to a user."""
        user = self._users.get(user_id)
        if not user:
            return False
        user.direct_permissions.add(permission)
        return True

    def deny_permission(self, user_id: str, permission: Permission) -> bool:
        """Explicitly deny a permission from a user."""
        user = self._users.get(user_id)
        if not user:
            return False
        user.denied_permissions.add(permission)
        return True

    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all effective permissions for a user."""
        user = self._users.get(user_id)
        if not user or not user.is_active:
            return set()

        permissions = user.direct_permissions.copy()

        # Add permissions from all roles
        for role_name in user.roles:
            permissions.update(self.get_effective_permissions(role_name))

        # Remove denied permissions
        permissions -= user.denied_permissions

        return permissions

    def check_permission(
        self,
        user_id: str,
        permission: Permission,
    ) -> bool:
        """Check if a user has a specific permission."""
        return permission in self.get_user_permissions(user_id)

    def check_resource_access(
        self,
        user_id: str,
        resource_id: str,
        permission: Permission,
    ) -> bool:
        """
        Check if user can access a specific resource.

        Considers ownership, tenant isolation, and policies.
        """
        user = self._users.get(user_id)
        resource = self._resources.get(resource_id)

        if not user or not resource:
            return False

        # Check basic permission
        if not self.check_permission(user_id, permission):
            return False

        # Check tenant isolation
        if resource.tenant_id and user.tenant_id:
            if resource.tenant_id != user.tenant_id:
                return False

        # Owner always has access (for non-admin operations)
        if resource.owner_id == user_id:
            return True

        # Check applicable policies
        for policy in sorted(
            self._policies.values(),
            key=lambda p: p.priority,
            reverse=True,
        ):
            if policy.resource_type == resource.type:
                if not policy.required_permissions <= self.get_user_permissions(
                    user_id
                ):
                    return False

        return True

    def register_resource(
        self,
        resource_id: str,
        resource_type: ResourceType,
        owner_id: str,
        tenant_id: Optional[str] = None,
    ) -> Resource:
        """Register a protected resource."""
        resource = Resource(
            id=resource_id,
            type=resource_type,
            owner_id=owner_id,
            tenant_id=tenant_id,
        )
        self._resources[resource_id] = resource
        return resource

    def create_policy(
        self,
        name: str,
        description: str,
        resource_type: ResourceType,
        required_permissions: Set[Permission],
        conditions: Optional[Dict[str, Any]] = None,
        priority: int = 0,
    ) -> Policy:
        """Create an access control policy."""
        policy = Policy(
            name=name,
            description=description,
            resource_type=resource_type,
            required_permissions=required_permissions,
            conditions=conditions,
            priority=priority,
        )
        self._policies[name] = policy
        return policy

    def list_roles(self) -> List[Dict[str, Any]]:
        """List all roles."""
        return [role.to_dict() for role in self._roles.values()]

    def list_users(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List users, optionally filtered by tenant."""
        users = []
        for user in self._users.values():
            if tenant_id and user.tenant_id != tenant_id:
                continue
            users.append(user.to_dict())
        return users


def require_permission(permission: Permission):
    """
    Decorator to require a permission for a function.

    Example:
        @require_permission(Permission.KNOWLEDGE_CREATE)
        def create_knowledge_item(user_id: str, data: dict):
            ...
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(user_id: str, *args, rbac: RBACManager, **kwargs):
            if not rbac.check_permission(user_id, permission):
                raise PermissionError(
                    f"User {user_id} lacks permission: {permission.value}"
                )
            return func(user_id, *args, rbac=rbac, **kwargs)

        return wrapper

    return decorator


class AccessContext:
    """
    Context for access control during request processing.

    Example:
        >>> with AccessContext(rbac, user_id) as ctx:
        ...     if ctx.can(Permission.KNOWLEDGE_READ):
        ...         return knowledge_base.search(query)
    """

    def __init__(self, rbac: RBACManager, user_id: str):
        """Initialize access context."""
        self.rbac = rbac
        self.user_id = user_id
        self._cached_permissions: Optional[Set[Permission]] = None

    def __enter__(self) -> "AccessContext":
        """Enter context."""
        self._cached_permissions = self.rbac.get_user_permissions(self.user_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context."""
        self._cached_permissions = None

    def can(self, permission: Permission) -> bool:
        """Check if current user has permission."""
        if self._cached_permissions is not None:
            return permission in self._cached_permissions
        return self.rbac.check_permission(self.user_id, permission)

    def can_access(self, resource_id: str, permission: Permission) -> bool:
        """Check if current user can access resource."""
        return self.rbac.check_resource_access(
            self.user_id, resource_id, permission
        )

    def require(self, permission: Permission) -> None:
        """Require permission or raise error."""
        if not self.can(permission):
            raise PermissionError(
                f"Permission denied: {permission.value}"
            )

    def get_permissions(self) -> Set[Permission]:
        """Get all user permissions."""
        if self._cached_permissions is not None:
            return self._cached_permissions.copy()
        return self.rbac.get_user_permissions(self.user_id)
