"""Role-Based Access Control (RBAC) Implementation."""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class Permission(str, Enum):
    """API permissions."""

    # Document operations
    DOCUMENT_READ = "document:read"
    DOCUMENT_WRITE = "document:write"
    DOCUMENT_DELETE = "document:delete"
    DOCUMENT_ADMIN = "document:admin"

    # OCR operations
    OCR_EXTRACT = "ocr:extract"
    OCR_BATCH = "ocr:batch"
    OCR_ADMIN = "ocr:admin"

    # Analysis operations
    ANALYSIS_READ = "analysis:read"
    ANALYSIS_CREATE = "analysis:create"
    ANALYSIS_ADMIN = "analysis:admin"

    # Model operations
    MODEL_PREDICT = "model:predict"
    MODEL_TRAIN = "model:train"
    MODEL_DEPLOY = "model:deploy"
    MODEL_ADMIN = "model:admin"

    # Vector store operations
    VECTOR_READ = "vector:read"
    VECTOR_WRITE = "vector:write"
    VECTOR_DELETE = "vector:delete"
    VECTOR_ADMIN = "vector:admin"

    # Admin operations
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_ADMIN = "user:admin"
    TENANT_ADMIN = "tenant:admin"
    SYSTEM_ADMIN = "system:admin"

    # Feature flags
    FEATURE_FLAG_READ = "feature_flag:read"
    FEATURE_FLAG_WRITE = "feature_flag:write"

    # Deployment
    DEPLOYMENT_READ = "deployment:read"
    DEPLOYMENT_EXECUTE = "deployment:execute"
    DEPLOYMENT_ADMIN = "deployment:admin"


class Role(str, Enum):
    """Predefined roles."""

    VIEWER = "viewer"  # Read-only access
    EDITOR = "editor"  # Read + write access
    OPERATOR = "operator"  # Read + write + limited admin
    ADMIN = "admin"  # Full access within tenant
    SUPER_ADMIN = "super_admin"  # Cross-tenant admin


# Permission hierarchy: higher roles inherit lower role permissions
ROLE_HIERARCHY: Dict[Role, List[Role]] = {
    Role.VIEWER: [],
    Role.EDITOR: [Role.VIEWER],
    Role.OPERATOR: [Role.EDITOR],
    Role.ADMIN: [Role.OPERATOR],
    Role.SUPER_ADMIN: [Role.ADMIN],
}

# Default role permissions
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.VIEWER: {
        Permission.DOCUMENT_READ,
        Permission.ANALYSIS_READ,
        Permission.VECTOR_READ,
        Permission.FEATURE_FLAG_READ,
        Permission.DEPLOYMENT_READ,
    },
    Role.EDITOR: {
        Permission.DOCUMENT_WRITE,
        Permission.OCR_EXTRACT,
        Permission.ANALYSIS_CREATE,
        Permission.MODEL_PREDICT,
        Permission.VECTOR_WRITE,
    },
    Role.OPERATOR: {
        Permission.OCR_BATCH,
        Permission.DOCUMENT_DELETE,
        Permission.VECTOR_DELETE,
        Permission.MODEL_TRAIN,
        Permission.FEATURE_FLAG_WRITE,
        Permission.DEPLOYMENT_EXECUTE,
    },
    Role.ADMIN: {
        Permission.DOCUMENT_ADMIN,
        Permission.OCR_ADMIN,
        Permission.ANALYSIS_ADMIN,
        Permission.MODEL_ADMIN,
        Permission.VECTOR_ADMIN,
        Permission.USER_READ,
        Permission.USER_WRITE,
        Permission.DEPLOYMENT_ADMIN,
    },
    Role.SUPER_ADMIN: {
        Permission.USER_ADMIN,
        Permission.TENANT_ADMIN,
        Permission.SYSTEM_ADMIN,
        Permission.MODEL_DEPLOY,
    },
}


@dataclass
class RBACPolicy:
    """A policy defining permissions for a principal."""

    principal_id: str  # User or service account ID
    principal_type: str = "user"  # user, service, api_key
    roles: Set[Role] = field(default_factory=set)
    permissions: Set[Permission] = field(default_factory=set)  # Extra permissions
    denied_permissions: Set[Permission] = field(default_factory=set)  # Explicit denials
    tenant_id: Optional[str] = None
    resource_restrictions: Dict[str, List[str]] = field(default_factory=dict)  # resource_type -> allowed IDs
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_effective_permissions(self) -> Set[Permission]:
        """Get all effective permissions including role inheritance."""
        permissions = set(self.permissions)

        def collect_inherited_permissions(role: Role, visited: Set[Role]) -> None:
            """Recursively collect permissions from inherited roles."""
            if role in visited:
                return
            visited.add(role)

            # Add this role's permissions
            permissions.update(ROLE_PERMISSIONS.get(role, set()))

            # Recursively add inherited role permissions
            for inherited_role in ROLE_HIERARCHY.get(role, []):
                collect_inherited_permissions(inherited_role, visited)

        # Add permissions from all roles
        visited: Set[Role] = set()
        for role in self.roles:
            collect_inherited_permissions(role, visited)

        # Remove denied permissions
        permissions -= self.denied_permissions

        return permissions

    def has_permission(self, permission: Permission) -> bool:
        """Check if policy grants a permission."""
        return permission in self.get_effective_permissions()

    def has_any_permission(self, permissions: Set[Permission]) -> bool:
        """Check if policy grants any of the permissions."""
        return bool(self.get_effective_permissions() & permissions)

    def has_all_permissions(self, permissions: Set[Permission]) -> bool:
        """Check if policy grants all of the permissions."""
        return permissions <= self.get_effective_permissions()

    def has_role(self, role: Role) -> bool:
        """Check if policy has a role (including inherited)."""
        if role in self.roles:
            return True
        for user_role in self.roles:
            if role in ROLE_HIERARCHY.get(user_role, []):
                return True
        return False

    def can_access_resource(self, resource_type: str, resource_id: str) -> bool:
        """Check if policy allows access to a specific resource."""
        if resource_type not in self.resource_restrictions:
            return True  # No restriction means allowed
        return resource_id in self.resource_restrictions[resource_type]


class RBACManager:
    """Manages RBAC policies and authorization checks."""

    def __init__(self):
        self._policies: Dict[str, RBACPolicy] = {}
        self._tenant_policies: Dict[str, Dict[str, RBACPolicy]] = {}

    def register_policy(self, policy: RBACPolicy) -> None:
        """Register a policy for a principal."""
        key = f"{policy.principal_type}:{policy.principal_id}"

        if policy.tenant_id:
            if policy.tenant_id not in self._tenant_policies:
                self._tenant_policies[policy.tenant_id] = {}
            self._tenant_policies[policy.tenant_id][key] = policy
        else:
            self._policies[key] = policy

        logger.info(f"Registered RBAC policy for {key}")

    def get_policy(
        self,
        principal_id: str,
        principal_type: str = "user",
        tenant_id: Optional[str] = None,
    ) -> Optional[RBACPolicy]:
        """Get policy for a principal."""
        key = f"{principal_type}:{principal_id}"

        # Check tenant-specific policy first
        if tenant_id and tenant_id in self._tenant_policies:
            if key in self._tenant_policies[tenant_id]:
                return self._tenant_policies[tenant_id][key]

        # Fall back to global policy
        return self._policies.get(key)

    def check_permission(
        self,
        principal_id: str,
        permission: Permission,
        principal_type: str = "user",
        tenant_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
    ) -> bool:
        """Check if a principal has a permission.

        Args:
            principal_id: Principal identifier
            permission: Permission to check
            principal_type: Type of principal
            tenant_id: Tenant context
            resource_type: Optional resource type for resource-level check
            resource_id: Optional resource ID for resource-level check

        Returns:
            True if permission is granted
        """
        policy = self.get_policy(principal_id, principal_type, tenant_id)

        if policy is None:
            logger.warning(f"No policy found for {principal_type}:{principal_id}")
            return False

        # Check permission
        if not policy.has_permission(permission):
            return False

        # Check resource-level restriction
        if resource_type and resource_id:
            if not policy.can_access_resource(resource_type, resource_id):
                return False

        return True

    def check_role(
        self,
        principal_id: str,
        role: Role,
        principal_type: str = "user",
        tenant_id: Optional[str] = None,
    ) -> bool:
        """Check if a principal has a role."""
        policy = self.get_policy(principal_id, principal_type, tenant_id)

        if policy is None:
            return False

        return policy.has_role(role)

    def grant_permission(
        self,
        principal_id: str,
        permission: Permission,
        principal_type: str = "user",
        tenant_id: Optional[str] = None,
    ) -> None:
        """Grant a permission to a principal."""
        policy = self.get_policy(principal_id, principal_type, tenant_id)

        if policy is None:
            policy = RBACPolicy(
                principal_id=principal_id,
                principal_type=principal_type,
                tenant_id=tenant_id,
            )
            self.register_policy(policy)

        policy.permissions.add(permission)
        logger.info(f"Granted {permission} to {principal_type}:{principal_id}")

    def revoke_permission(
        self,
        principal_id: str,
        permission: Permission,
        principal_type: str = "user",
        tenant_id: Optional[str] = None,
    ) -> None:
        """Revoke a permission from a principal."""
        policy = self.get_policy(principal_id, principal_type, tenant_id)

        if policy:
            policy.permissions.discard(permission)
            policy.denied_permissions.add(permission)
            logger.info(f"Revoked {permission} from {principal_type}:{principal_id}")

    def assign_role(
        self,
        principal_id: str,
        role: Role,
        principal_type: str = "user",
        tenant_id: Optional[str] = None,
    ) -> None:
        """Assign a role to a principal."""
        policy = self.get_policy(principal_id, principal_type, tenant_id)

        if policy is None:
            policy = RBACPolicy(
                principal_id=principal_id,
                principal_type=principal_type,
                tenant_id=tenant_id,
            )
            self.register_policy(policy)

        policy.roles.add(role)
        logger.info(f"Assigned role {role} to {principal_type}:{principal_id}")


# Global RBAC manager
_rbac_manager: Optional[RBACManager] = None


def get_rbac_manager() -> RBACManager:
    """Get global RBAC manager."""
    global _rbac_manager
    if _rbac_manager is None:
        _rbac_manager = RBACManager()
    return _rbac_manager


def require_permission(
    permission: Permission,
    principal_extractor: Optional[Callable[..., tuple]] = None,
) -> Callable[[F], F]:
    """Decorator to require a permission for function execution.

    Args:
        permission: Required permission
        principal_extractor: Function to extract (principal_id, principal_type, tenant_id)

    Returns:
        Decorated function

    Example:
        @require_permission(Permission.DOCUMENT_READ)
        async def get_document(request: Request, doc_id: str):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract principal info
            if principal_extractor:
                principal_id, principal_type, tenant_id = principal_extractor(*args, **kwargs)
            else:
                # Try to extract from request
                request = kwargs.get("request") or (args[0] if args else None)
                principal_id = getattr(request, "user_id", None) or "anonymous"
                principal_type = "user"
                tenant_id = getattr(request, "tenant_id", None)

            # Check permission
            manager = get_rbac_manager()
            if not manager.check_permission(principal_id, permission, principal_type, tenant_id):
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: {permission.value}",
                )

            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if principal_extractor:
                principal_id, principal_type, tenant_id = principal_extractor(*args, **kwargs)
            else:
                request = kwargs.get("request") or (args[0] if args else None)
                principal_id = getattr(request, "user_id", None) or "anonymous"
                principal_type = "user"
                tenant_id = getattr(request, "tenant_id", None)

            manager = get_rbac_manager()
            if not manager.check_permission(principal_id, permission, principal_type, tenant_id):
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: {permission.value}",
                )

            return func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def require_role(
    role: Role,
    principal_extractor: Optional[Callable[..., tuple]] = None,
) -> Callable[[F], F]:
    """Decorator to require a role for function execution.

    Args:
        role: Required role
        principal_extractor: Function to extract principal info

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if principal_extractor:
                principal_id, principal_type, tenant_id = principal_extractor(*args, **kwargs)
            else:
                request = kwargs.get("request") or (args[0] if args else None)
                principal_id = getattr(request, "user_id", None) or "anonymous"
                principal_type = "user"
                tenant_id = getattr(request, "tenant_id", None)

            manager = get_rbac_manager()
            if not manager.check_role(principal_id, role, principal_type, tenant_id):
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=403,
                    detail=f"Role required: {role.value}",
                )

            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if principal_extractor:
                principal_id, principal_type, tenant_id = principal_extractor(*args, **kwargs)
            else:
                request = kwargs.get("request") or (args[0] if args else None)
                principal_id = getattr(request, "user_id", None) or "anonymous"
                principal_type = "user"
                tenant_id = getattr(request, "tenant_id", None)

            manager = get_rbac_manager()
            if not manager.check_role(principal_id, role, principal_type, tenant_id):
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=403,
                    detail=f"Role required: {role.value}",
                )

            return func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator
