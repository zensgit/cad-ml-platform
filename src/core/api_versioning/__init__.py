"""API Versioning Module.

Provides API versioning support:
- URL-based versioning
- Header-based versioning
- Media type versioning
- Version deprecation
- Migration support
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class VersioningStrategy(Enum):
    """API versioning strategies."""
    URL = "url"  # /v1/users
    HEADER = "header"  # X-API-Version: 1
    QUERY = "query"  # ?version=1
    MEDIA_TYPE = "media_type"  # Accept: application/vnd.api.v1+json


class VersionStatus(Enum):
    """API version status."""
    CURRENT = "current"
    SUPPORTED = "supported"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"
    RETIRED = "retired"


@dataclass
class Version:
    """API version information."""

    major: int
    minor: int = 0
    patch: int = 0

    def __str__(self) -> str:
        if self.patch > 0:
            return f"{self.major}.{self.minor}.{self.patch}"
        if self.minor > 0:
            return f"{self.major}.{self.minor}"
        return str(self.major)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Version):
            return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
        return False

    def __lt__(self, other: "Version") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __le__(self, other: "Version") -> bool:
        return self == other or self < other

    def __gt__(self, other: "Version") -> bool:
        return not self <= other

    def __ge__(self, other: "Version") -> bool:
        return not self < other

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))

    @classmethod
    def parse(cls, version_str: str) -> "Version":
        """Parse version string."""
        # Handle 'v' prefix
        if version_str.lower().startswith("v"):
            version_str = version_str[1:]

        parts = version_str.split(".")
        major = int(parts[0]) if parts else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0

        return cls(major=major, minor=minor, patch=patch)

    def is_compatible_with(self, other: "Version") -> bool:
        """Check if this version is compatible with another (same major version)."""
        return self.major == other.major


@dataclass
class VersionInfo:
    """Detailed version information."""

    version: Version
    status: VersionStatus = VersionStatus.CURRENT
    release_date: Optional[datetime] = None
    deprecation_date: Optional[datetime] = None
    sunset_date: Optional[datetime] = None
    changelog: str = ""
    breaking_changes: List[str] = field(default_factory=list)
    migration_guide: Optional[str] = None

    @property
    def is_usable(self) -> bool:
        """Check if this version can be used."""
        return self.status in (VersionStatus.CURRENT, VersionStatus.SUPPORTED, VersionStatus.DEPRECATED)

    @property
    def days_until_sunset(self) -> Optional[int]:
        """Get days until sunset."""
        if self.sunset_date is None:
            return None
        delta = self.sunset_date - datetime.utcnow()
        return max(0, delta.days)

    def to_headers(self) -> Dict[str, str]:
        """Convert to response headers."""
        headers = {
            "X-API-Version": str(self.version),
            "X-API-Version-Status": self.status.value,
        }

        if self.status == VersionStatus.DEPRECATED:
            headers["Deprecation"] = "true"
            if self.sunset_date:
                headers["Sunset"] = self.sunset_date.strftime("%a, %d %b %Y %H:%M:%S GMT")

        return headers


class VersionExtractor(ABC):
    """Abstract version extractor."""

    @abstractmethod
    def extract(self, request: Any) -> Optional[str]:
        """Extract version from request."""
        pass


class URLVersionExtractor(VersionExtractor):
    """Extract version from URL path."""

    def __init__(self, pattern: str = r"/v(\d+(?:\.\d+)*)"):
        self._pattern = re.compile(pattern)

    def extract(self, request: Any) -> Optional[str]:
        """Extract version from URL."""
        path = getattr(request, "path", "") or getattr(request, "url", {}).get("path", "")

        match = self._pattern.search(path)
        if match:
            return match.group(1)
        return None


class HeaderVersionExtractor(VersionExtractor):
    """Extract version from HTTP header."""

    def __init__(self, header_name: str = "X-API-Version"):
        self._header_name = header_name

    def extract(self, request: Any) -> Optional[str]:
        """Extract version from header."""
        headers = getattr(request, "headers", {})

        # Handle different header access methods
        if hasattr(headers, "get"):
            version = headers.get(self._header_name)
        elif isinstance(headers, dict):
            version = headers.get(self._header_name)
        else:
            return None

        return version


class QueryVersionExtractor(VersionExtractor):
    """Extract version from query parameter."""

    def __init__(self, param_name: str = "version"):
        self._param_name = param_name

    def extract(self, request: Any) -> Optional[str]:
        """Extract version from query."""
        query = getattr(request, "query_params", {}) or getattr(request, "args", {})

        if hasattr(query, "get"):
            return query.get(self._param_name)
        return None


class MediaTypeVersionExtractor(VersionExtractor):
    """Extract version from Accept header media type."""

    def __init__(self, pattern: str = r"application/vnd\..+\.v(\d+(?:\.\d+)*)\+\w+"):
        self._pattern = re.compile(pattern)

    def extract(self, request: Any) -> Optional[str]:
        """Extract version from Accept header."""
        headers = getattr(request, "headers", {})
        accept = headers.get("Accept", "") if hasattr(headers, "get") else ""

        match = self._pattern.search(accept)
        if match:
            return match.group(1)
        return None


class VersionRegistry:
    """Registry for API versions."""

    def __init__(self, default_version: Optional[Version] = None):
        self._versions: Dict[Version, VersionInfo] = {}
        self._default_version = default_version
        self._handlers: Dict[Version, Dict[str, Callable]] = {}

    def register(
        self,
        version: Union[Version, str],
        status: VersionStatus = VersionStatus.CURRENT,
        **kwargs,
    ) -> "VersionRegistry":
        """Register a new API version."""
        if isinstance(version, str):
            version = Version.parse(version)

        info = VersionInfo(
            version=version,
            status=status,
            **kwargs,
        )

        self._versions[version] = info

        if self._default_version is None or status == VersionStatus.CURRENT:
            self._default_version = version

        logger.info(f"Registered API version {version} ({status.value})")
        return self

    def get_version(self, version: Union[Version, str]) -> Optional[VersionInfo]:
        """Get version info."""
        if isinstance(version, str):
            version = Version.parse(version)
        return self._versions.get(version)

    def get_all_versions(self) -> List[VersionInfo]:
        """Get all registered versions."""
        return sorted(self._versions.values(), key=lambda v: v.version, reverse=True)

    def get_usable_versions(self) -> List[VersionInfo]:
        """Get all usable (non-retired) versions."""
        return [v for v in self.get_all_versions() if v.is_usable]

    def deprecate(
        self,
        version: Union[Version, str],
        sunset_date: Optional[datetime] = None,
        sunset_days: int = 90,
    ) -> None:
        """Mark a version as deprecated."""
        if isinstance(version, str):
            version = Version.parse(version)

        info = self._versions.get(version)
        if info:
            info.status = VersionStatus.DEPRECATED
            info.deprecation_date = datetime.utcnow()
            info.sunset_date = sunset_date or (datetime.utcnow() + timedelta(days=sunset_days))
            logger.warning(f"API version {version} deprecated, sunset: {info.sunset_date}")

    def sunset(self, version: Union[Version, str]) -> None:
        """Mark a version as sunset (no new calls, only existing)."""
        if isinstance(version, str):
            version = Version.parse(version)

        info = self._versions.get(version)
        if info:
            info.status = VersionStatus.SUNSET
            logger.warning(f"API version {version} sunset")

    def retire(self, version: Union[Version, str]) -> None:
        """Mark a version as retired (completely unavailable)."""
        if isinstance(version, str):
            version = Version.parse(version)

        info = self._versions.get(version)
        if info:
            info.status = VersionStatus.RETIRED
            logger.info(f"API version {version} retired")

    def register_handler(
        self,
        version: Union[Version, str],
        route: str,
        handler: Callable,
    ) -> None:
        """Register a route handler for a specific version."""
        if isinstance(version, str):
            version = Version.parse(version)

        if version not in self._handlers:
            self._handlers[version] = {}

        self._handlers[version][route] = handler

    def get_handler(
        self,
        version: Union[Version, str],
        route: str,
    ) -> Optional[Callable]:
        """Get handler for a specific version and route."""
        if isinstance(version, str):
            version = Version.parse(version)

        version_handlers = self._handlers.get(version, {})
        return version_handlers.get(route)

    @property
    def default_version(self) -> Optional[Version]:
        return self._default_version


class VersionNegotiator:
    """Negotiates API version from request."""

    def __init__(
        self,
        registry: VersionRegistry,
        extractors: Optional[List[VersionExtractor]] = None,
        default_extractor: Optional[VersionExtractor] = None,
    ):
        self._registry = registry
        self._extractors = extractors or [
            HeaderVersionExtractor(),
            URLVersionExtractor(),
            QueryVersionExtractor(),
        ]
        self._default_extractor = default_extractor

    def negotiate(self, request: Any) -> Tuple[Optional[VersionInfo], Optional[str]]:
        """
        Negotiate API version from request.

        Returns:
            Tuple of (VersionInfo, error_message)
        """
        # Try each extractor
        version_str = None
        for extractor in self._extractors:
            version_str = extractor.extract(request)
            if version_str:
                break

        # Use default if no version found
        if not version_str:
            if self._registry.default_version:
                return self._registry.get_version(self._registry.default_version), None
            return None, "No API version specified and no default configured"

        # Parse and validate version
        try:
            version = Version.parse(version_str)
        except ValueError:
            return None, f"Invalid version format: {version_str}"

        # Look up version info
        info = self._registry.get_version(version)
        if not info:
            return None, f"Unknown API version: {version}"

        # Check if usable
        if not info.is_usable:
            return None, f"API version {version} is no longer available ({info.status.value})"

        return info, None


class VersionedRouter:
    """Router that handles version-specific routing."""

    def __init__(
        self,
        registry: VersionRegistry,
        negotiator: Optional[VersionNegotiator] = None,
    ):
        self._registry = registry
        self._negotiator = negotiator or VersionNegotiator(registry)
        self._routes: Dict[str, Dict[Version, Callable]] = {}

    def route(
        self,
        path: str,
        versions: Optional[List[Union[Version, str]]] = None,
    ) -> Callable:
        """Decorator to register versioned route."""
        def decorator(handler: Callable) -> Callable:
            target_versions = versions or [self._registry.default_version]

            for v in target_versions:
                if isinstance(v, str):
                    v = Version.parse(v)

                if path not in self._routes:
                    self._routes[path] = {}

                self._routes[path][v] = handler
                self._registry.register_handler(v, path, handler)

            return handler

        return decorator

    def get_handler(
        self,
        path: str,
        version: Union[Version, str],
    ) -> Optional[Callable]:
        """Get handler for path and version."""
        if isinstance(version, str):
            version = Version.parse(version)

        path_handlers = self._routes.get(path, {})

        # Exact match
        if version in path_handlers:
            return path_handlers[version]

        # Find compatible version (same major)
        for v, handler in sorted(path_handlers.items(), reverse=True):
            if v.major == version.major and v <= version:
                return handler

        return None


class VersionMigration:
    """Handles data migration between API versions."""

    def __init__(self):
        self._transformers: Dict[Tuple[Version, Version], Callable] = {}

    def register(
        self,
        from_version: Union[Version, str],
        to_version: Union[Version, str],
        transformer: Callable[[Any], Any],
    ) -> None:
        """Register a data transformer."""
        if isinstance(from_version, str):
            from_version = Version.parse(from_version)
        if isinstance(to_version, str):
            to_version = Version.parse(to_version)

        self._transformers[(from_version, to_version)] = transformer

    def migrate(
        self,
        data: Any,
        from_version: Union[Version, str],
        to_version: Union[Version, str],
    ) -> Any:
        """Migrate data between versions."""
        if isinstance(from_version, str):
            from_version = Version.parse(from_version)
        if isinstance(to_version, str):
            to_version = Version.parse(to_version)

        if from_version == to_version:
            return data

        # Direct migration path
        transformer = self._transformers.get((from_version, to_version))
        if transformer:
            return transformer(data)

        # Build migration chain
        chain = self._build_chain(from_version, to_version)
        if not chain:
            raise ValueError(f"No migration path from {from_version} to {to_version}")

        result = data
        for from_v, to_v in chain:
            transformer = self._transformers[(from_v, to_v)]
            result = transformer(result)

        return result

    def _build_chain(
        self,
        from_version: Version,
        to_version: Version,
    ) -> List[Tuple[Version, Version]]:
        """Build migration chain using BFS."""
        if from_version == to_version:
            return []

        # Get all registered versions as graph nodes
        versions = set()
        for from_v, to_v in self._transformers.keys():
            versions.add(from_v)
            versions.add(to_v)

        # BFS to find shortest path
        queue = [(from_version, [])]
        visited = {from_version}

        while queue:
            current, path = queue.pop(0)

            for (from_v, to_v), _ in self._transformers.items():
                if from_v == current and to_v not in visited:
                    new_path = path + [(from_v, to_v)]

                    if to_v == to_version:
                        return new_path

                    visited.add(to_v)
                    queue.append((to_v, new_path))

        return []


class VersionMiddleware:
    """Middleware for API versioning."""

    def __init__(
        self,
        registry: VersionRegistry,
        negotiator: Optional[VersionNegotiator] = None,
        on_deprecated: Optional[Callable[[Any, VersionInfo], None]] = None,
    ):
        self._registry = registry
        self._negotiator = negotiator or VersionNegotiator(registry)
        self._on_deprecated = on_deprecated

    async def __call__(self, request: Any, next_handler: Callable) -> Any:
        """Process request with version negotiation."""
        info, error = self._negotiator.negotiate(request)

        if error:
            raise VersionError(error)

        # Add version info to request
        request.version_info = info

        # Handle deprecated version
        if info.status == VersionStatus.DEPRECATED:
            if self._on_deprecated:
                self._on_deprecated(request, info)

        # Call next handler
        response = await next_handler(request)

        # Add version headers to response
        if hasattr(response, "headers"):
            for key, value in info.to_headers().items():
                response.headers[key] = value

        return response


class VersionError(Exception):
    """Raised when version negotiation fails."""

    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def create_version_registry(
    versions: List[Dict[str, Any]],
) -> VersionRegistry:
    """Factory function to create configured registry."""
    registry = VersionRegistry()

    for v in versions:
        version = v.get("version")
        if isinstance(version, str):
            version = Version.parse(version)

        status = v.get("status", "current")
        if isinstance(status, str):
            status = VersionStatus(status)

        registry.register(
            version=version,
            status=status,
            release_date=v.get("release_date"),
            deprecation_date=v.get("deprecation_date"),
            sunset_date=v.get("sunset_date"),
            changelog=v.get("changelog", ""),
            breaking_changes=v.get("breaking_changes", []),
            migration_guide=v.get("migration_guide"),
        )

    return registry


__all__ = [
    "VersioningStrategy",
    "VersionStatus",
    "Version",
    "VersionInfo",
    "VersionExtractor",
    "URLVersionExtractor",
    "HeaderVersionExtractor",
    "QueryVersionExtractor",
    "MediaTypeVersionExtractor",
    "VersionRegistry",
    "VersionNegotiator",
    "VersionedRouter",
    "VersionMigration",
    "VersionMiddleware",
    "VersionError",
    "create_version_registry",
]
