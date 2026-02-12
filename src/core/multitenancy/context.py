"""Tenant Context Management.

Provides thread-safe and async-safe tenant context handling.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Context variable for current tenant
_current_tenant: contextvars.ContextVar[Optional["TenantContext"]] = contextvars.ContextVar(
    "current_tenant", default=None
)


@dataclass
class TenantContext:
    """Context information for current tenant."""

    tenant_id: str
    tenant_name: str
    schema_name: Optional[str] = None
    database_name: Optional[str] = None
    settings: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Resource quotas
    max_storage_bytes: Optional[int] = None
    max_documents: Optional[int] = None
    max_users: Optional[int] = None
    max_api_calls_per_hour: Optional[int] = None

    # Current usage
    current_storage_bytes: int = 0
    current_documents: int = 0
    current_users: int = 0

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a tenant-specific setting."""
        return self.settings.get(key, default)

    def is_quota_exceeded(self, resource: str) -> bool:
        """Check if a resource quota is exceeded."""
        quotas = {
            "storage": (self.current_storage_bytes, self.max_storage_bytes),
            "documents": (self.current_documents, self.max_documents),
            "users": (self.current_users, self.max_users),
        }

        if resource not in quotas:
            return False

        current, maximum = quotas[resource]
        if maximum is None:
            return False

        return current >= maximum

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "tenant_name": self.tenant_name,
            "schema_name": self.schema_name,
            "database_name": self.database_name,
            "settings": self.settings,
            "metadata": self.metadata,
            "quotas": {
                "storage": {
                    "current": self.current_storage_bytes,
                    "max": self.max_storage_bytes,
                },
                "documents": {
                    "current": self.current_documents,
                    "max": self.max_documents,
                },
                "users": {
                    "current": self.current_users,
                    "max": self.max_users,
                },
            },
        }


def get_current_tenant() -> Optional[TenantContext]:
    """Get the current tenant context.

    Returns:
        Current TenantContext or None if not set
    """
    return _current_tenant.get()


def set_current_tenant(tenant: Optional[TenantContext]) -> contextvars.Token:
    """Set the current tenant context.

    Args:
        tenant: TenantContext to set

    Returns:
        Token for resetting the context
    """
    return _current_tenant.set(tenant)


def reset_current_tenant(token: contextvars.Token) -> None:
    """Reset tenant context to previous value.

    Args:
        token: Token from set_current_tenant
    """
    _current_tenant.reset(token)


@contextmanager
def tenant_context(tenant: TenantContext):
    """Context manager for tenant scope.

    Args:
        tenant: TenantContext to use

    Example:
        with tenant_context(tenant):
            # All operations here use this tenant
            await process_document(doc)
    """
    token = set_current_tenant(tenant)
    try:
        logger.debug(f"Entered tenant context: {tenant.tenant_id}")
        yield tenant
    finally:
        reset_current_tenant(token)
        logger.debug(f"Exited tenant context: {tenant.tenant_id}")


class TenantContextManager:
    """Advanced tenant context manager with caching."""

    def __init__(self, cache_ttl: int = 300):
        self._cache: Dict[str, TenantContext] = {}
        self._cache_ttl = cache_ttl
        self._cache_timestamps: Dict[str, datetime] = {}

    def get_cached(self, tenant_id: str) -> Optional[TenantContext]:
        """Get cached tenant context."""
        if tenant_id not in self._cache:
            return None

        # Check TTL
        cached_at = self._cache_timestamps.get(tenant_id)
        if cached_at:
            age = (datetime.utcnow() - cached_at).total_seconds()
            if age > self._cache_ttl:
                self._cache.pop(tenant_id, None)
                self._cache_timestamps.pop(tenant_id, None)
                return None

        return self._cache.get(tenant_id)

    def cache(self, tenant: TenantContext) -> None:
        """Cache a tenant context."""
        self._cache[tenant.tenant_id] = tenant
        self._cache_timestamps[tenant.tenant_id] = datetime.utcnow()

    def invalidate(self, tenant_id: str) -> None:
        """Invalidate cached tenant context."""
        self._cache.pop(tenant_id, None)
        self._cache_timestamps.pop(tenant_id, None)

    def clear_cache(self) -> None:
        """Clear all cached contexts."""
        self._cache.clear()
        self._cache_timestamps.clear()


# Global context manager
_context_manager: Optional[TenantContextManager] = None


def get_context_manager() -> TenantContextManager:
    """Get global tenant context manager."""
    global _context_manager
    if _context_manager is None:
        _context_manager = TenantContextManager()
    return _context_manager
