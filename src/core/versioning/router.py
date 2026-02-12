"""Version-Aware API Router.

Provides version routing capabilities:
- URL path versioning (/v1/users)
- Header versioning (API-Version: 1.0)
- Query parameter versioning (?version=1.0)
- Content negotiation (Accept: application/vnd.api+json;version=1.0)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from src.core.versioning.version import SemanticVersion, VersionRange, VersionRegistry


class VersioningStrategy(Enum):
    """API versioning strategy."""
    URL_PATH = "url_path"  # /v1/users
    HEADER = "header"  # API-Version: 1.0
    QUERY_PARAM = "query_param"  # ?version=1.0
    CONTENT_TYPE = "content_type"  # Accept: application/vnd.api+json;version=1.0
    CUSTOM = "custom"


@dataclass
class VersionedRequest:
    """HTTP request with version information."""
    method: str
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)
    body: Any = None

    # Resolved version (set by router)
    resolved_version: Optional[SemanticVersion] = None


@dataclass
class VersionedResponse:
    """HTTP response with version headers."""
    status_code: int
    body: Any = None
    headers: Dict[str, str] = field(default_factory=dict)


class VersionExtractor(ABC):
    """Base class for version extraction strategies."""

    @abstractmethod
    def extract(self, request: VersionedRequest) -> Optional[SemanticVersion]:
        """Extract version from request."""
        pass

    @abstractmethod
    def inject(self, path: str, version: SemanticVersion) -> str:
        """Inject version into path/URL."""
        pass


class URLPathVersionExtractor(VersionExtractor):
    """Extract version from URL path (/v1/users)."""

    def __init__(self, prefix: str = "v"):
        self.prefix = prefix
        self._pattern = re.compile(rf'^/{prefix}(\d+(?:\.\d+(?:\.\d+)?)?)/(.*)$')

    def extract(self, request: VersionedRequest) -> Optional[SemanticVersion]:
        match = self._pattern.match(request.path)
        if match:
            version_str = match.group(1)
            # Normalize to full semver
            parts = version_str.split('.')
            if len(parts) == 1:
                version_str = f"{version_str}.0.0"
            elif len(parts) == 2:
                version_str = f"{version_str}.0"
            return SemanticVersion.parse(version_str)
        return None

    def inject(self, path: str, version: SemanticVersion) -> str:
        # Remove existing version prefix if present
        path = self._pattern.sub(r'/\2', path)
        if not path.startswith('/'):
            path = '/' + path
        return f"/{self.prefix}{version.major}/{path.lstrip('/')}"

    def strip_version(self, path: str) -> str:
        """Remove version prefix from path."""
        match = self._pattern.match(path)
        if match:
            return '/' + match.group(2)
        return path


class HeaderVersionExtractor(VersionExtractor):
    """Extract version from HTTP header."""

    def __init__(self, header_name: str = "API-Version"):
        self.header_name = header_name

    def extract(self, request: VersionedRequest) -> Optional[SemanticVersion]:
        version_str = request.headers.get(self.header_name)
        if version_str:
            # Normalize
            parts = version_str.split('.')
            if len(parts) == 1:
                version_str = f"{version_str}.0.0"
            elif len(parts) == 2:
                version_str = f"{version_str}.0"
            return SemanticVersion.parse(version_str)
        return None

    def inject(self, path: str, version: SemanticVersion) -> str:
        # Headers don't modify path
        return path


class QueryParamVersionExtractor(VersionExtractor):
    """Extract version from query parameter."""

    def __init__(self, param_name: str = "version"):
        self.param_name = param_name

    def extract(self, request: VersionedRequest) -> Optional[SemanticVersion]:
        version_str = request.query_params.get(self.param_name)
        if version_str:
            parts = version_str.split('.')
            if len(parts) == 1:
                version_str = f"{version_str}.0.0"
            elif len(parts) == 2:
                version_str = f"{version_str}.0"
            return SemanticVersion.parse(version_str)
        return None

    def inject(self, path: str, version: SemanticVersion) -> str:
        separator = '&' if '?' in path else '?'
        return f"{path}{separator}{self.param_name}={version}"


class ContentTypeVersionExtractor(VersionExtractor):
    """Extract version from Accept header media type."""

    def __init__(self, vendor: str = "vnd.api"):
        self.vendor = vendor
        self._pattern = re.compile(
            rf'application/{vendor}\+json;\s*version=(\d+(?:\.\d+(?:\.\d+)?)?)'
        )

    def extract(self, request: VersionedRequest) -> Optional[SemanticVersion]:
        accept = request.headers.get('Accept', '')
        match = self._pattern.search(accept)
        if match:
            version_str = match.group(1)
            parts = version_str.split('.')
            if len(parts) == 1:
                version_str = f"{version_str}.0.0"
            elif len(parts) == 2:
                version_str = f"{version_str}.0"
            return SemanticVersion.parse(version_str)
        return None

    def inject(self, path: str, version: SemanticVersion) -> str:
        return path


# Handler type
Handler = Callable[[VersionedRequest], VersionedResponse]


@dataclass
class Route:
    """A versioned API route."""
    method: str
    path_pattern: str
    handler: Handler
    version_range: VersionRange
    deprecated: bool = False
    sunset_version: Optional[SemanticVersion] = None

    def matches(self, method: str, path: str, version: SemanticVersion) -> bool:
        """Check if this route matches the request."""
        if method.upper() != self.method.upper():
            return False
        if not self._path_matches(path):
            return False
        if not self.version_range.contains(version):
            return False
        return True

    def _path_matches(self, path: str) -> bool:
        """Check if path matches the pattern."""
        # Convert pattern to regex
        # {param} -> capture group
        pattern = re.sub(r'\{(\w+)\}', r'(?P<\1>[^/]+)', self.path_pattern)
        pattern = f'^{pattern}$'
        return re.match(pattern, path) is not None

    def extract_params(self, path: str) -> Dict[str, str]:
        """Extract path parameters."""
        pattern = re.sub(r'\{(\w+)\}', r'(?P<\1>[^/]+)', self.path_pattern)
        pattern = f'^{pattern}$'
        match = re.match(pattern, path)
        if match:
            return match.groupdict()
        return {}


class VersionedRouter:
    """Version-aware API router."""

    def __init__(
        self,
        strategy: VersioningStrategy = VersioningStrategy.URL_PATH,
        default_version: Optional[SemanticVersion] = None,
        version_registry: Optional[VersionRegistry] = None,
    ):
        self.strategy = strategy
        self.default_version = default_version
        self.registry = version_registry or VersionRegistry()
        self._routes: List[Route] = []
        self._extractor = self._create_extractor(strategy)
        self._middleware: List[Callable] = []

    def _create_extractor(self, strategy: VersioningStrategy) -> VersionExtractor:
        """Create version extractor based on strategy."""
        if strategy == VersioningStrategy.URL_PATH:
            return URLPathVersionExtractor()
        elif strategy == VersioningStrategy.HEADER:
            return HeaderVersionExtractor()
        elif strategy == VersioningStrategy.QUERY_PARAM:
            return QueryParamVersionExtractor()
        elif strategy == VersioningStrategy.CONTENT_TYPE:
            return ContentTypeVersionExtractor()
        else:
            return URLPathVersionExtractor()

    def add_route(
        self,
        method: str,
        path: str,
        handler: Handler,
        version_range: Union[str, VersionRange] = ">=1.0.0",
        deprecated: bool = False,
    ) -> None:
        """Add a versioned route."""
        if isinstance(version_range, str):
            version_range = VersionRange.parse(version_range)

        route = Route(
            method=method.upper(),
            path_pattern=path,
            handler=handler,
            version_range=version_range,
            deprecated=deprecated,
        )
        self._routes.append(route)

    def add_middleware(self, middleware: Callable) -> None:
        """Add middleware to the router."""
        self._middleware.append(middleware)

    def route(
        self,
        method: str,
        path: str,
        version_range: str = ">=1.0.0",
        deprecated: bool = False,
    ) -> Callable[[Handler], Handler]:
        """Decorator to register a route."""
        def decorator(handler: Handler) -> Handler:
            self.add_route(method, path, handler, version_range, deprecated)
            return handler
        return decorator

    def get(self, path: str, **kwargs) -> Callable[[Handler], Handler]:
        """Register GET route."""
        return self.route("GET", path, **kwargs)

    def post(self, path: str, **kwargs) -> Callable[[Handler], Handler]:
        """Register POST route."""
        return self.route("POST", path, **kwargs)

    def put(self, path: str, **kwargs) -> Callable[[Handler], Handler]:
        """Register PUT route."""
        return self.route("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs) -> Callable[[Handler], Handler]:
        """Register DELETE route."""
        return self.route("DELETE", path, **kwargs)

    def handle(self, request: VersionedRequest) -> VersionedResponse:
        """Handle a versioned request."""
        # Extract version
        version = self._extractor.extract(request)
        if not version:
            version = self.default_version

        if not version:
            return VersionedResponse(
                status_code=400,
                body={"error": "API version is required"},
            )

        request.resolved_version = version

        # Strip version from path for URL path strategy
        path = request.path
        if self.strategy == VersioningStrategy.URL_PATH:
            if isinstance(self._extractor, URLPathVersionExtractor):
                path = self._extractor.strip_version(request.path)

        # Find matching route
        matching_routes = [
            r for r in self._routes
            if r.matches(request.method, path, version)
        ]

        if not matching_routes:
            return VersionedResponse(
                status_code=404,
                body={"error": f"No route found for {request.method} {path} v{version}"},
            )

        # Use most specific (highest version) route
        route = max(matching_routes, key=lambda r: r.version_range.min_version or SemanticVersion(0, 0, 0))

        # Build response with version headers
        try:
            # Execute middleware chain
            def execute():
                return route.handler(request)

            result = self._execute_with_middleware(request, execute)

            # Add version headers
            result.headers["X-API-Version"] = str(version)
            if route.deprecated:
                result.headers["X-API-Deprecated"] = "true"
                result.headers["Warning"] = f'299 - "API version {version} is deprecated"'

            return result

        except Exception as e:
            return VersionedResponse(
                status_code=500,
                body={"error": str(e)},
            )

    def _execute_with_middleware(
        self,
        request: VersionedRequest,
        handler: Callable,
    ) -> VersionedResponse:
        """Execute handler with middleware chain."""
        def chain(index: int) -> VersionedResponse:
            if index >= len(self._middleware):
                return handler()

            middleware = self._middleware[index]
            return middleware(request, lambda: chain(index + 1))

        return chain(0)


class VersionedAPIGroup:
    """Group of versioned API endpoints."""

    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self._routes: List[Tuple[str, str, Handler, str, bool]] = []

    def route(
        self,
        method: str,
        path: str,
        version_range: str = ">=1.0.0",
        deprecated: bool = False,
    ) -> Callable[[Handler], Handler]:
        """Decorator to register a route in this group."""
        def decorator(handler: Handler) -> Handler:
            full_path = f"{self.prefix}{path}"
            self._routes.append((method, full_path, handler, version_range, deprecated))
            return handler
        return decorator

    def get(self, path: str, **kwargs) -> Callable[[Handler], Handler]:
        return self.route("GET", path, **kwargs)

    def post(self, path: str, **kwargs) -> Callable[[Handler], Handler]:
        return self.route("POST", path, **kwargs)

    def put(self, path: str, **kwargs) -> Callable[[Handler], Handler]:
        return self.route("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs) -> Callable[[Handler], Handler]:
        return self.route("DELETE", path, **kwargs)

    def register_with(self, router: VersionedRouter) -> None:
        """Register all routes with a router."""
        for method, path, handler, version_range, deprecated in self._routes:
            router.add_route(method, path, handler, version_range, deprecated)
