"""Tenant Isolation for Multi-tenant Support."""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

# Context variable for tenant
_tenant_context: contextvars.ContextVar[Optional["TenantContext"]] = contextvars.ContextVar(
    "tenant_context", default=None
)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class TenantContext:
    """Context for current tenant."""

    tenant_id: str
    tenant_name: str = ""
    is_active: bool = True
    tier: str = "standard"  # free, standard, enterprise
    features: Set[str] = field(default_factory=set)
    quotas: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None

    def has_feature(self, feature: str) -> bool:
        """Check if tenant has a feature enabled."""
        return feature in self.features

    def get_quota(self, resource: str, default: int = 0) -> int:
        """Get quota for a resource."""
        return self.quotas.get(resource, default)


def get_current_tenant() -> Optional[TenantContext]:
    """Get current tenant context."""
    return _tenant_context.get()


def set_tenant_context(tenant: Optional[TenantContext]) -> contextvars.Token:
    """Set current tenant context.

    Args:
        tenant: Tenant context to set

    Returns:
        Token for resetting context
    """
    return _tenant_context.set(tenant)


def clear_tenant_context() -> None:
    """Clear current tenant context."""
    _tenant_context.set(None)


class TenantIsolation:
    """Manages tenant isolation and resource access."""

    def __init__(self):
        self._tenants: Dict[str, TenantContext] = {}
        self._tenant_resources: Dict[str, Set[str]] = {}  # tenant_id -> resource_ids

    def register_tenant(self, tenant: TenantContext) -> None:
        """Register a tenant."""
        self._tenants[tenant.tenant_id] = tenant
        self._tenant_resources[tenant.tenant_id] = set()
        logger.info(f"Registered tenant: {tenant.tenant_id}")

    def get_tenant(self, tenant_id: str) -> Optional[TenantContext]:
        """Get tenant by ID."""
        return self._tenants.get(tenant_id)

    def list_tenants(self) -> List[TenantContext]:
        """List all tenants."""
        return list(self._tenants.values())

    def register_resource(self, tenant_id: str, resource_id: str) -> None:
        """Register a resource as belonging to a tenant."""
        if tenant_id not in self._tenant_resources:
            self._tenant_resources[tenant_id] = set()
        self._tenant_resources[tenant_id].add(resource_id)

    def check_resource_access(self, tenant_id: str, resource_id: str) -> bool:
        """Check if a tenant can access a resource.

        Args:
            tenant_id: Tenant identifier
            resource_id: Resource identifier

        Returns:
            True if tenant can access the resource
        """
        if tenant_id not in self._tenant_resources:
            return False
        return resource_id in self._tenant_resources[tenant_id]

    def get_tenant_resources(self, tenant_id: str) -> Set[str]:
        """Get all resources belonging to a tenant."""
        return self._tenant_resources.get(tenant_id, set())

    def filter_by_tenant(
        self,
        items: List[Dict[str, Any]],
        tenant_id: str,
        tenant_key: str = "tenant_id",
    ) -> List[Dict[str, Any]]:
        """Filter items by tenant.

        Args:
            items: List of items to filter
            tenant_id: Tenant to filter for
            tenant_key: Key in item dict containing tenant ID

        Returns:
            Filtered list
        """
        return [item for item in items if item.get(tenant_key) == tenant_id]


# Global tenant isolation manager
_tenant_isolation: Optional[TenantIsolation] = None


def get_tenant_isolation() -> TenantIsolation:
    """Get global tenant isolation manager."""
    global _tenant_isolation
    if _tenant_isolation is None:
        _tenant_isolation = TenantIsolation()
    return _tenant_isolation


class TenantMiddleware:
    """FastAPI middleware for tenant context injection."""

    def __init__(self, app: Any, isolation: Optional[TenantIsolation] = None):
        self.app = app
        self.isolation = isolation or get_tenant_isolation()

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            # Extract tenant from headers
            headers = dict(scope.get("headers", []))
            tenant_id = headers.get(b"x-tenant-id", b"").decode()

            if tenant_id:
                tenant = self.isolation.get_tenant(tenant_id)
                if tenant and tenant.is_active:
                    token = set_tenant_context(tenant)
                    try:
                        await self.app(scope, receive, send)
                    finally:
                        _tenant_context.reset(token)
                    return

        await self.app(scope, receive, send)


def require_tenant(func: F) -> F:
    """Decorator to require tenant context.

    Raises 403 if no tenant context is set.
    """
    import functools
    import asyncio

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        tenant = get_current_tenant()
        if tenant is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="Tenant context required")
        return await func(*args, **kwargs)

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        tenant = get_current_tenant()
        if tenant is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="Tenant context required")
        return func(*args, **kwargs)

    if asyncio.iscoroutinefunction(func):
        return async_wrapper  # type: ignore
    return sync_wrapper  # type: ignore


def require_feature(feature: str) -> Callable[[F], F]:
    """Decorator to require a tenant feature.

    Args:
        feature: Required feature name

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        import functools
        import asyncio

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tenant = get_current_tenant()
            if tenant is None or not tenant.has_feature(feature):
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=403,
                    detail=f"Feature not available: {feature}",
                )
            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tenant = get_current_tenant()
            if tenant is None or not tenant.has_feature(feature):
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=403,
                    detail=f"Feature not available: {feature}",
                )
            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator
