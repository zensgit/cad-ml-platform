"""Multi-tenancy module for tenant isolation and management.

Provides:
- Tenant context management
- Data isolation strategies
- Resource quotas
- Cross-tenant operations
"""

from src.core.multitenancy.context import (
    TenantContext,
    get_current_tenant,
    set_current_tenant,
    tenant_context,
)
from src.core.multitenancy.isolation import (
    DataIsolationStrategy,
    IsolationLevel,
    RowLevelIsolation,
    SchemaIsolation,
    DatabaseIsolation,
    get_isolation_strategy,
)
from src.core.multitenancy.manager import (
    Tenant,
    TenantManager,
    TenantStatus,
    get_tenant_manager,
)
from src.core.multitenancy.middleware import (
    TenantMiddleware,
    TenantResolver,
    HeaderTenantResolver,
    SubdomainTenantResolver,
    PathTenantResolver,
)

__all__ = [
    # Context
    "TenantContext",
    "get_current_tenant",
    "set_current_tenant",
    "tenant_context",
    # Isolation
    "DataIsolationStrategy",
    "IsolationLevel",
    "RowLevelIsolation",
    "SchemaIsolation",
    "DatabaseIsolation",
    "get_isolation_strategy",
    # Manager
    "Tenant",
    "TenantManager",
    "TenantStatus",
    "get_tenant_manager",
    # Middleware
    "TenantMiddleware",
    "TenantResolver",
    "HeaderTenantResolver",
    "SubdomainTenantResolver",
    "PathTenantResolver",
]
