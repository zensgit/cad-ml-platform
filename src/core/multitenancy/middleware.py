"""Tenant Middleware for HTTP Request Handling.

Provides middleware components for resolving and injecting tenant context.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

from src.core.multitenancy.context import TenantContext, set_current_tenant, reset_current_tenant
from src.core.multitenancy.manager import Tenant, TenantManager, TenantStatus, get_tenant_manager

logger = logging.getLogger(__name__)


class TenantResolver(ABC):
    """Abstract base class for tenant resolution strategies."""

    @abstractmethod
    async def resolve(self, request: Any) -> Optional[str]:
        """Resolve tenant identifier from request.

        Args:
            request: HTTP request object

        Returns:
            Tenant identifier (ID or slug) or None
        """
        pass


class HeaderTenantResolver(TenantResolver):
    """Resolve tenant from HTTP header.

    Example:
        X-Tenant-ID: tenant-123
    """

    def __init__(
        self,
        header_name: str = "X-Tenant-ID",
        alt_header_name: Optional[str] = "X-Tenant-Slug",
    ):
        self.header_name = header_name
        self.alt_header_name = alt_header_name

    async def resolve(self, request: Any) -> Optional[str]:
        # Try primary header
        tenant_id = self._get_header(request, self.header_name)
        if tenant_id:
            return tenant_id

        # Try alternate header
        if self.alt_header_name:
            return self._get_header(request, self.alt_header_name)

        return None

    def _get_header(self, request: Any, header_name: str) -> Optional[str]:
        """Get header value from request."""
        # FastAPI/Starlette
        if hasattr(request, "headers"):
            return request.headers.get(header_name)

        # Flask
        if hasattr(request, "environ"):
            header_key = f"HTTP_{header_name.upper().replace('-', '_')}"
            return request.environ.get(header_key)

        # Django
        if hasattr(request, "META"):
            header_key = f"HTTP_{header_name.upper().replace('-', '_')}"
            return request.META.get(header_key)

        return None


class SubdomainTenantResolver(TenantResolver):
    """Resolve tenant from subdomain.

    Example:
        tenant-123.example.com -> tenant-123
    """

    def __init__(
        self,
        base_domain: str = "example.com",
        exclude_subdomains: Optional[List[str]] = None,
    ):
        self.base_domain = base_domain
        self.exclude_subdomains = exclude_subdomains or ["www", "api", "admin"]

    async def resolve(self, request: Any) -> Optional[str]:
        host = self._get_host(request)
        if not host:
            return None

        # Remove port if present
        host = host.split(":")[0]

        # Check if it's a subdomain of our base domain
        if not host.endswith(f".{self.base_domain}"):
            return None

        # Extract subdomain
        subdomain = host[: -len(f".{self.base_domain}")]

        # Skip excluded subdomains
        if subdomain in self.exclude_subdomains:
            return None

        return subdomain

    def _get_host(self, request: Any) -> Optional[str]:
        """Get host from request."""
        if hasattr(request, "headers"):
            return request.headers.get("host")

        if hasattr(request, "environ"):
            return request.environ.get("HTTP_HOST")

        if hasattr(request, "META"):
            return request.META.get("HTTP_HOST")

        return None


class PathTenantResolver(TenantResolver):
    """Resolve tenant from URL path.

    Example:
        /tenants/tenant-123/documents -> tenant-123
    """

    def __init__(
        self,
        path_pattern: str = r"^/tenants/([^/]+)/",
        group_index: int = 1,
    ):
        self.path_pattern = re.compile(path_pattern)
        self.group_index = group_index

    async def resolve(self, request: Any) -> Optional[str]:
        path = self._get_path(request)
        if not path:
            return None

        match = self.path_pattern.match(path)
        if match:
            return match.group(self.group_index)

        return None

    def _get_path(self, request: Any) -> Optional[str]:
        """Get URL path from request."""
        if hasattr(request, "url"):
            return str(request.url.path)

        if hasattr(request, "path"):
            return request.path

        if hasattr(request, "path_info"):
            return request.path_info

        return None


class APIKeyTenantResolver(TenantResolver):
    """Resolve tenant from API key in header or query param.

    Example:
        Authorization: Bearer tnt_xxxxx
        ?api_key=tnt_xxxxx
    """

    def __init__(
        self,
        header_name: str = "Authorization",
        query_param: str = "api_key",
        prefix: str = "Bearer ",
        manager: Optional[TenantManager] = None,
    ):
        self.header_name = header_name
        self.query_param = query_param
        self.prefix = prefix
        self._manager = manager

    @property
    def manager(self) -> TenantManager:
        if self._manager is None:
            self._manager = get_tenant_manager()
        return self._manager

    async def resolve(self, request: Any) -> Optional[str]:
        api_key = self._extract_api_key(request)
        if not api_key:
            return None

        # Find tenant by API key
        for tenant in self.manager.list_tenants():
            if self.manager.validate_api_key(tenant.tenant_id, api_key):
                return tenant.tenant_id

        return None

    def _extract_api_key(self, request: Any) -> Optional[str]:
        """Extract API key from request."""
        # Try header
        auth_header = None
        if hasattr(request, "headers"):
            auth_header = request.headers.get(self.header_name)
        elif hasattr(request, "META"):
            auth_header = request.META.get(f"HTTP_{self.header_name.upper()}")

        if auth_header and auth_header.startswith(self.prefix):
            return auth_header[len(self.prefix) :]

        # Try query param
        if hasattr(request, "query_params"):
            return request.query_params.get(self.query_param)

        return None


class ChainedTenantResolver(TenantResolver):
    """Try multiple resolvers in order.

    Useful for supporting multiple resolution methods.
    """

    def __init__(self, resolvers: List[TenantResolver]):
        self.resolvers = resolvers

    async def resolve(self, request: Any) -> Optional[str]:
        for resolver in self.resolvers:
            try:
                result = await resolver.resolve(request)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Resolver {resolver.__class__.__name__} failed: {e}")

        return None


@dataclass
class TenantMiddlewareConfig:
    """Configuration for tenant middleware."""

    # Whether to require tenant for all requests
    require_tenant: bool = True

    # Paths that don't require tenant
    exclude_paths: List[str] = None

    # Path patterns to exclude (regex)
    exclude_patterns: List[str] = None

    # Whether to allow suspended tenants
    allow_suspended: bool = False

    # Custom error handler
    error_handler: Optional[Callable[[Any, str], Any]] = None

    def __post_init__(self):
        if self.exclude_paths is None:
            self.exclude_paths = ["/health", "/ready", "/metrics", "/docs", "/openapi.json"]
        if self.exclude_patterns is None:
            self.exclude_patterns = []


class TenantMiddleware:
    """Middleware for tenant context injection.

    Works with FastAPI/Starlette, can be adapted for other frameworks.
    """

    def __init__(
        self,
        resolver: TenantResolver,
        manager: Optional[TenantManager] = None,
        config: Optional[TenantMiddlewareConfig] = None,
    ):
        self.resolver = resolver
        self._manager = manager
        self.config = config or TenantMiddlewareConfig()
        self._exclude_patterns = [
            re.compile(p) for p in self.config.exclude_patterns
        ]

    @property
    def manager(self) -> TenantManager:
        if self._manager is None:
            self._manager = get_tenant_manager()
        return self._manager

    def _should_skip(self, path: str) -> bool:
        """Check if path should skip tenant resolution."""
        # Check exact matches
        if path in self.config.exclude_paths:
            return True

        # Check patterns
        for pattern in self._exclude_patterns:
            if pattern.match(path):
                return True

        return False

    async def __call__(self, request: Any, call_next: Callable) -> Any:
        """Process request with tenant context.

        This is the FastAPI/Starlette middleware interface.
        """
        # Get request path
        path = str(getattr(request, "url", request).path) if hasattr(request, "url") else "/"

        # Skip excluded paths
        if self._should_skip(path):
            return await call_next(request)

        # Resolve tenant
        tenant_identifier = await self.resolver.resolve(request)

        if not tenant_identifier:
            if self.config.require_tenant:
                return self._error_response(request, "Tenant not specified")
            return await call_next(request)

        # Load tenant
        tenant = self.manager.get_tenant(tenant_identifier)
        if not tenant:
            tenant = self.manager.get_tenant_by_slug(tenant_identifier)

        if not tenant:
            return self._error_response(request, f"Tenant not found: {tenant_identifier}")

        # Check tenant status
        if tenant.status == TenantStatus.SUSPENDED and not self.config.allow_suspended:
            return self._error_response(request, "Tenant is suspended")

        if tenant.status not in (TenantStatus.ACTIVE, TenantStatus.SUSPENDED):
            return self._error_response(request, f"Tenant is not active: {tenant.status.value}")

        # Set tenant context
        context = tenant.to_context()
        token = set_current_tenant(context)

        try:
            # Add tenant to request state for convenience
            if hasattr(request, "state"):
                request.state.tenant = tenant
                request.state.tenant_context = context

            response = await call_next(request)
            return response

        finally:
            reset_current_tenant(token)

    def _error_response(self, request: Any, message: str) -> Any:
        """Create error response."""
        if self.config.error_handler:
            return self.config.error_handler(request, message)

        # Default: Return JSON error (FastAPI/Starlette style)
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=403,
            content={"detail": message},
        )


def create_tenant_middleware(
    resolution_method: str = "header",
    **kwargs: Any,
) -> TenantMiddleware:
    """Factory function to create tenant middleware.

    Args:
        resolution_method: How to resolve tenant ("header", "subdomain", "path", "api_key")
        **kwargs: Additional configuration

    Returns:
        Configured TenantMiddleware
    """
    resolvers: Dict[str, Type[TenantResolver]] = {
        "header": HeaderTenantResolver,
        "subdomain": SubdomainTenantResolver,
        "path": PathTenantResolver,
        "api_key": APIKeyTenantResolver,
    }

    resolver_class = resolvers.get(resolution_method, HeaderTenantResolver)
    resolver = resolver_class(**{k: v for k, v in kwargs.items() if k in resolver_class.__init__.__code__.co_varnames})

    config_kwargs = {k: v for k, v in kwargs.items() if k in TenantMiddlewareConfig.__dataclass_fields__}
    config = TenantMiddlewareConfig(**config_kwargs) if config_kwargs else None

    return TenantMiddleware(resolver=resolver, config=config)
