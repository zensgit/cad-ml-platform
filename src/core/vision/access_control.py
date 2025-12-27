"""Access Control Module for Vision System.

This module provides access control capabilities including:
- Role-Based Access Control (RBAC)
- Permission management
- Authentication integration
- Session management
- Multi-tenancy support
- API key management

Phase 19: Advanced Security & Compliance
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from .base import VisionDescription, VisionProvider

# ========================
# Enums
# ========================


class Permission(str, Enum):
    """System permissions."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE = "execute"
    ANALYZE = "analyze"
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    VIEW_AUDIT = "view_audit"
    MANAGE_POLICIES = "manage_policies"


class RoleType(str, Enum):
    """Types of roles."""

    SYSTEM = "system"
    CUSTOM = "custom"
    TEMPORARY = "temporary"


class AuthMethod(str, Enum):
    """Authentication methods."""

    PASSWORD = "password"
    API_KEY = "api_key"
    OAUTH = "oauth"
    JWT = "jwt"
    SAML = "saml"
    CERTIFICATE = "certificate"


class SessionStatus(str, Enum):
    """Session status."""

    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    LOCKED = "locked"


class AccessDecision(str, Enum):
    """Access control decisions."""

    ALLOW = "allow"
    DENY = "deny"
    ABSTAIN = "abstain"


class ACLResourceType(str, Enum):
    """Types of resources."""

    MODEL = "model"
    DATASET = "dataset"
    PIPELINE = "pipeline"
    EXPERIMENT = "experiment"
    IMAGE = "image"
    API = "api"
    SYSTEM = "system"


# ========================
# Dataclasses
# ========================


@dataclass
class User:
    """A system user."""

    user_id: str
    username: str
    email: str = ""
    roles: List[str] = field(default_factory=list)
    permissions: List[Permission] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    tenant_id: Optional[str] = None


@dataclass
class Role:
    """A system role."""

    role_id: str
    name: str
    role_type: RoleType = RoleType.CUSTOM
    permissions: List[Permission] = field(default_factory=list)
    description: str = ""
    parent_roles: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ACLResource:
    """A protected resource."""

    resource_id: str
    resource_type: ACLResourceType
    name: str
    owner_id: str
    permissions: Dict[str, List[Permission]] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Session:
    """A user session."""

    session_id: str
    user_id: str
    status: SessionStatus = SessionStatus.ACTIVE
    auth_method: AuthMethod = AuthMethod.PASSWORD
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=24))
    last_activity: datetime = field(default_factory=datetime.now)
    ip_address: str = ""
    user_agent: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIKey:
    """An API key for authentication."""

    key_id: str
    key_hash: str
    user_id: str
    name: str
    permissions: List[Permission] = field(default_factory=list)
    rate_limit: int = 1000
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None


@dataclass
class AccessRequest:
    """An access request to evaluate."""

    user_id: str
    resource_id: str
    permission: Permission
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AccessResult:
    """Result of access evaluation."""

    decision: AccessDecision
    request: AccessRequest
    reason: str = ""
    evaluated_at: datetime = field(default_factory=datetime.now)
    policy_id: Optional[str] = None


@dataclass
class Policy:
    """Access control policy."""

    policy_id: str
    name: str
    effect: "PolicyEffect"
    conditions: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class Tenant:
    """A tenant in multi-tenant setup."""

    tenant_id: str
    name: str
    enabled: bool = True
    quota: Dict[str, int] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


# ========================
# Core Classes
# ========================


class UserManager:
    """Manage system users."""

    def __init__(self):
        self._users: Dict[str, User] = {}
        self._username_index: Dict[str, str] = {}
        self._lock = threading.RLock()

    def create_user(self, user: User) -> User:
        """Create a new user."""
        with self._lock:
            if user.username in self._username_index:
                raise ValueError(f"Username {user.username} already exists")
            self._users[user.user_id] = user
            self._username_index[user.username] = user.user_id
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        return self._users.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username."""
        user_id = self._username_index.get(username)
        if user_id:
            return self._users.get(user_id)
        return None

    def update_user(self, user: User) -> User:
        """Update a user."""
        with self._lock:
            if user.user_id in self._users:
                old_user = self._users[user.user_id]
                if old_user.username != user.username:
                    del self._username_index[old_user.username]
                    self._username_index[user.username] = user.user_id
                self._users[user.user_id] = user
        return user

    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        with self._lock:
            user = self._users.get(user_id)
            if user:
                del self._users[user_id]
                del self._username_index[user.username]
                return True
        return False

    def list_users(
        self,
        tenant_id: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> List[User]:
        """List users."""
        users = list(self._users.values())
        if tenant_id:
            users = [u for u in users if u.tenant_id == tenant_id]
        if enabled is not None:
            users = [u for u in users if u.enabled == enabled]
        return users

    def assign_role(self, user_id: str, role_id: str) -> bool:
        """Assign a role to a user."""
        with self._lock:
            user = self._users.get(user_id)
            if user and role_id not in user.roles:
                user.roles.append(role_id)
                return True
        return False

    def revoke_role(self, user_id: str, role_id: str) -> bool:
        """Revoke a role from a user."""
        with self._lock:
            user = self._users.get(user_id)
            if user and role_id in user.roles:
                user.roles.remove(role_id)
                return True
        return False


class RoleManager:
    """Manage roles and permissions."""

    def __init__(self, user_manager: Optional[UserManager] = None):
        self._roles: Dict[str, Role] = {}
        self._user_roles: Dict[str, Set[str]] = defaultdict(set)
        self._user_manager = user_manager
        self._lock = threading.RLock()
        self._initialize_default_roles()

    def _initialize_default_roles(self) -> None:
        """Initialize default system roles."""
        default_roles = [
            Role(
                role_id="admin",
                name="Administrator",
                role_type=RoleType.SYSTEM,
                permissions=list(Permission),
                description="Full system access",
            ),
            Role(
                role_id="user",
                name="User",
                role_type=RoleType.SYSTEM,
                permissions=[Permission.READ, Permission.WRITE, Permission.EXECUTE],
                description="Standard user access",
            ),
            Role(
                role_id="viewer",
                name="Viewer",
                role_type=RoleType.SYSTEM,
                permissions=[Permission.READ],
                description="Read-only access",
            ),
        ]
        for role in default_roles:
            self._roles[role.role_id] = role

    def create_role(
        self,
        role: Role | str,
        permissions: Optional[Set[Permission] | List[Permission]] = None,
        role_type: RoleType = RoleType.CUSTOM,
        description: str = "",
    ) -> Role:
        """Create a new role."""
        if isinstance(role, Role):
            new_role = role
        else:
            new_role = Role(
                role_id=secrets.token_hex(8),
                name=role,
                role_type=role_type,
                permissions=list(permissions or []),
                description=description,
            )
        with self._lock:
            self._roles[new_role.role_id] = new_role
        return new_role

    def get_role(self, role_id: str) -> Optional[Role]:
        """Get a role by ID."""
        return self._roles.get(role_id)

    def update_role(self, role: Role) -> Role:
        """Update a role."""
        with self._lock:
            if role.role_id in self._roles:
                self._roles[role.role_id] = role
        return role

    def delete_role(self, role_id: str) -> bool:
        """Delete a role."""
        with self._lock:
            role = self._roles.get(role_id)
            if role and role.role_type != RoleType.SYSTEM:
                del self._roles[role_id]
                return True
        return False

    def list_roles(self, role_type: Optional[RoleType] = None) -> List[Role]:
        """List roles."""
        roles = list(self._roles.values())
        if role_type:
            roles = [r for r in roles if r.role_type == role_type]
        return roles

    def assign_role(self, user_id: str, role_id: str) -> bool:
        """Assign a role to a user."""
        with self._lock:
            if role_id not in self._roles:
                return False
            self._user_roles[user_id].add(role_id)
            if self._user_manager:
                user = self._user_manager.get_user(user_id)
                if user and role_id not in user.roles:
                    user.roles.append(role_id)
            return True

    def get_user_roles(self, user_id: str) -> List[Role]:
        """Get roles assigned to a user."""
        role_ids: Set[str] = set(self._user_roles.get(user_id, set()))
        if self._user_manager:
            user = self._user_manager.get_user(user_id)
            if user:
                role_ids.update(user.roles)
        return [self._roles[role_id] for role_id in role_ids if role_id in self._roles]

    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if a user has a permission via assigned roles."""
        role_ids = [role.role_id for role in self.get_user_roles(user_id)]
        permissions = self.get_effective_permissions(role_ids)
        return permission in permissions

    def get_effective_permissions(self, role_ids: List[str]) -> Set[Permission]:
        """Get effective permissions for a set of roles."""
        permissions: Set[Permission] = set()

        def collect_permissions(role_id: str, visited: Set[str]) -> None:
            if role_id in visited:
                return
            visited.add(role_id)
            role = self._roles.get(role_id)
            if role:
                permissions.update(role.permissions)
                for parent_id in role.parent_roles:
                    collect_permissions(parent_id, visited)

        visited: Set[str] = set()
        for role_id in role_ids:
            collect_permissions(role_id, visited)
        return permissions


class SessionManager:
    """Manage user sessions."""

    def __init__(self, session_timeout: int = 86400):
        self._sessions: Dict[str, Session] = {}
        self._user_sessions: Dict[str, Set[str]] = defaultdict(set)
        self._session_timeout = session_timeout
        self._lock = threading.RLock()

    def create_session(
        self,
        user_id: str,
        auth_method: AuthMethod = AuthMethod.PASSWORD,
        ip_address: str = "",
        user_agent: str = "",
    ) -> Session:
        """Create a new session."""
        session_id = secrets.token_urlsafe(32)
        session = Session(
            session_id=session_id,
            user_id=user_id,
            auth_method=auth_method,
            expires_at=datetime.now() + timedelta(seconds=self._session_timeout),
            ip_address=ip_address,
            user_agent=user_agent,
        )
        with self._lock:
            self._sessions[session_id] = session
            self._user_sessions[user_id].add(session_id)
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        session = self._sessions.get(session_id)
        if session and session.status == SessionStatus.ACTIVE:
            if datetime.now() > session.expires_at:
                session.status = SessionStatus.EXPIRED
            else:
                session.last_activity = datetime.now()
        return session

    def validate_session(self, session_id: str) -> bool:
        """Validate a session."""
        session = self.get_session(session_id)
        return session is not None and session.status == SessionStatus.ACTIVE

    def revoke_session(self, session_id: str) -> bool:
        """Revoke a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.status = SessionStatus.REVOKED
                return True
        return False

    def revoke_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user."""
        count = 0
        with self._lock:
            session_ids = self._user_sessions.get(user_id, set())
            for session_id in session_ids:
                session = self._sessions.get(session_id)
                if session and session.status == SessionStatus.ACTIVE:
                    session.status = SessionStatus.REVOKED
                    count += 1
        return count

    def list_user_sessions(self, user_id: str) -> List[Session]:
        """List sessions for a user."""
        session_ids = self._user_sessions.get(user_id, set())
        return [self._sessions[sid] for sid in session_ids if sid in self._sessions]


class APIKeyManager:
    """Manage API keys."""

    def __init__(self):
        self._keys: Dict[str, APIKey] = {}
        self._key_hashes: Dict[str, str] = {}
        self._lock = threading.RLock()

    def generate_key(
        self,
        user_id: str,
        name: str,
        permissions: Optional[List[Permission]] = None,
        expires_in_days: Optional[int] = None,
    ) -> Tuple[str, APIKey]:
        """Generate a new API key."""
        key_value = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(key_value.encode()).hexdigest()
        key_id = hashlib.md5(f"{user_id}:{name}:{time.time()}".encode()).hexdigest()[:12]

        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            permissions=permissions or [Permission.READ],
            expires_at=expires_at,
        )

        with self._lock:
            self._keys[key_id] = api_key
            self._key_hashes[key_hash] = key_id
        return key_value, api_key

    def validate_key(self, key_value: str) -> Optional[APIKey]:
        """Validate an API key."""
        key_hash = hashlib.sha256(key_value.encode()).hexdigest()
        key_id = self._key_hashes.get(key_hash)
        if key_id:
            api_key = self._keys.get(key_id)
            if api_key and api_key.enabled:
                if api_key.expires_at and datetime.now() > api_key.expires_at:
                    return None
                api_key.last_used = datetime.now()
                return api_key
        return None

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        with self._lock:
            api_key = self._keys.get(key_id)
            if api_key:
                api_key.enabled = False
                return True
        return False

    def list_user_keys(self, user_id: str) -> List[APIKey]:
        """List API keys for a user."""
        return [k for k in self._keys.values() if k.user_id == user_id]


class AccessController:
    """Main access control component."""

    def __init__(self):
        self._user_manager = UserManager()
        self._role_manager = RoleManager(self._user_manager)
        self._session_manager = SessionManager()
        self._api_key_manager = APIKeyManager()
        self._resources: Dict[str, ACLResource] = {}
        self._lock = threading.RLock()

    @property
    def role_manager(self) -> RoleManager:
        """Expose role manager for convenience."""
        return self._role_manager

    def create_user(
        self,
        username: str,
        email: str = "",
        roles: Optional[List[str]] = None,
        permissions: Optional[Set[Permission]] = None,
    ) -> User:
        """Create and register a user."""
        user = User(
            user_id=secrets.token_hex(8),
            username=username,
            email=email,
            roles=roles or [],
            permissions=list(permissions or []),
        )
        return self._user_manager.create_user(user)

    def evaluate_access(self, request: AccessRequest) -> AccessResult:
        """Evaluate an access request."""
        user = self._user_manager.get_user(request.user_id)

        if user is None or not user.enabled:
            return AccessResult(
                decision=AccessDecision.DENY,
                request=request,
                reason="User not found or disabled",
            )

        role_permissions = self._role_manager.get_effective_permissions(user.roles)
        all_permissions = role_permissions.union(set(user.permissions))

        if request.permission in all_permissions or Permission.ADMIN in all_permissions:
            resource = self._resources.get(request.resource_id)
            if resource:
                if resource.owner_id == request.user_id:
                    return AccessResult(
                        decision=AccessDecision.ALLOW,
                        request=request,
                        reason="Resource owner",
                    )
                for role_id in user.roles:
                    role_perms = resource.permissions.get(role_id, [])
                    if request.permission in role_perms:
                        return AccessResult(
                            decision=AccessDecision.ALLOW,
                            request=request,
                            reason=f"Granted via role {role_id}",
                        )
            return AccessResult(
                decision=AccessDecision.ALLOW,
                request=request,
                reason="Permission granted",
            )

        return AccessResult(
            decision=AccessDecision.DENY,
            request=request,
            reason="Insufficient permissions",
        )

    def check_access(
        self,
        user_id: str,
        resource_id: str,
        permission: Permission,
        context: Optional[Dict[str, Any]] = None,
    ) -> AccessResult:
        """Check access for a user against a resource."""
        request = AccessRequest(
            user_id=user_id,
            resource_id=resource_id,
            permission=permission,
            context=context or {},
        )
        return self.evaluate_access(request)

    def register_resource(
        self,
        resource: ACLResource | ACLResourceType,
        name: str = "",
        owner_id: str = "",
    ) -> ACLResource:
        """Register a protected resource."""
        if isinstance(resource, ACLResource):
            registered = resource
        else:
            resource_name = name or resource.value
            registered = ACLResource(
                resource_id=secrets.token_hex(8),
                resource_type=resource,
                name=resource_name,
                owner_id=owner_id,
            )
        with self._lock:
            self._resources[registered.resource_id] = registered
        return registered

    def get_user_manager(self) -> UserManager:
        """Get the user manager."""
        return self._user_manager

    def get_role_manager(self) -> RoleManager:
        """Get the role manager."""
        return self._role_manager

    def get_session_manager(self) -> SessionManager:
        """Get the session manager."""
        return self._session_manager

    def get_api_key_manager(self) -> APIKeyManager:
        """Get the API key manager."""
        return self._api_key_manager


# ========================
# Policy Engine
# ========================


class PolicyEngine:
    """Simple policy engine for ABAC-like rules."""

    def __init__(self):
        self._policies: Dict[str, Policy] = {}
        self._lock = threading.RLock()

    def add_policy(self, policy: Policy) -> None:
        """Register a policy."""
        with self._lock:
            self._policies[policy.policy_id] = policy

    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Get a policy by ID."""
        return self._policies.get(policy_id)

    def list_policies(self) -> List[Policy]:
        """List all policies."""
        return list(self._policies.values())


# ========================
# Vision Provider
# ========================


class AccessControlVisionProvider(VisionProvider):
    """Vision provider for access control capabilities."""

    def __init__(
        self,
        provider: Optional[VisionProvider] = None,
        controller: Optional[AccessController] = None,
        user_id: str = "",
    ):
        self._provider = provider
        self._controller = controller
        self._user_id = user_id

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return "access_control"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
        resource_id: Optional[str] = None,
        permission: Permission = Permission.ANALYZE,
        **kwargs: Any,
    ) -> VisionDescription:
        """Analyze image for access control context."""
        if self._provider is None:
            return self.get_description()
        if resource_id and self._user_id:
            controller = self.get_controller()
            result = controller.check_access(self._user_id, resource_id, permission)
            if result.decision != AccessDecision.ALLOW:
                raise PermissionError(result.reason or "Access denied")
        return await self._provider.analyze_image(
            image_data,
            include_description=include_description,
            **kwargs,
        )

    def get_description(self) -> VisionDescription:
        """Get provider description."""
        return VisionDescription(
            name="Access Control Vision Provider",
            version="1.0.0",
            description="RBAC, permissions, and authentication",
            capabilities=[
                "user_management",
                "role_management",
                "session_management",
                "api_key_management",
                "access_evaluation",
            ],
        )

    def initialize(self) -> None:
        """Initialize the provider."""
        self._controller = AccessController()

    def shutdown(self) -> None:
        """Shutdown the provider."""
        self._controller = None

    def get_controller(self) -> AccessController:
        """Get the access controller."""
        if self._controller is None:
            self.initialize()
        return self._controller


# ========================
# Factory Functions
# ========================


def create_access_controller() -> AccessController:
    """Create an access controller."""
    return AccessController()


def create_policy_engine() -> PolicyEngine:
    """Create a policy engine."""
    return PolicyEngine()


def create_user(
    user_id: str,
    username: str,
    email: str = "",
    roles: Optional[List[str]] = None,
) -> User:
    """Create a user."""
    return User(
        user_id=user_id,
        username=username,
        email=email,
        roles=roles or [],
    )


def create_role(
    role_id: str,
    name: str,
    permissions: Optional[List[Permission]] = None,
    role_type: RoleType = RoleType.CUSTOM,
) -> Role:
    """Create a role."""
    return Role(
        role_id=role_id,
        name=name,
        permissions=permissions or [],
        role_type=role_type,
    )


def create_acl_resource(
    resource_id: str,
    resource_type: ACLResourceType,
    name: str,
    owner_id: str,
) -> ACLResource:
    """Create a resource."""
    return ACLResource(
        resource_id=resource_id,
        resource_type=resource_type,
        name=name,
        owner_id=owner_id,
    )


def create_access_request(
    user_id: str,
    resource_id: str,
    permission: Permission,
) -> AccessRequest:
    """Create an access request."""
    return AccessRequest(
        user_id=user_id,
        resource_id=resource_id,
        permission=permission,
    )


def create_user_manager() -> UserManager:
    """Create a user manager."""
    return UserManager()


def create_role_manager() -> RoleManager:
    """Create a role manager."""
    return RoleManager()


def create_session_manager(session_timeout: int = 86400) -> SessionManager:
    """Create a session manager."""
    return SessionManager(session_timeout=session_timeout)


def create_api_key_manager() -> APIKeyManager:
    """Create an API key manager."""
    return APIKeyManager()


def create_access_control_provider(
    provider: Optional[VisionProvider] = None,
    controller: Optional[AccessController] = None,
    user_id: str = "",
) -> AccessControlVisionProvider:
    """Create an access control vision provider."""
    return AccessControlVisionProvider(
        provider=provider,
        controller=controller,
        user_id=user_id,
    )


# Aliases for backward compatibility and test compatibility
AccessControlManager = AccessController
Resource = ACLResource
ResourceType = ACLResourceType
PolicyEffect = AccessDecision  # Similar purpose
create_access_manager = create_access_controller
create_acl_provider = create_access_control_provider
