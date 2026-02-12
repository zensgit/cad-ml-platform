"""Fine-grained RBAC System for CAD ML Platform.

Provides:
- Role-based access control
- Permission hierarchies
- Resource-level permissions
- Tenant isolation
"""

from src.core.security.rbac import (
    Permission,
    Role,
    RBACPolicy,
    RBACManager,
    get_rbac_manager,
    require_permission,
    require_role,
)
from src.core.security.tenant import (
    TenantContext,
    TenantIsolation,
    get_current_tenant,
    set_tenant_context,
)

__all__ = [
    # RBAC
    "Permission",
    "Role",
    "RBACPolicy",
    "RBACManager",
    "get_rbac_manager",
    "require_permission",
    "require_role",
    # Tenant
    "TenantContext",
    "TenantIsolation",
    "get_current_tenant",
    "set_tenant_context",
]
