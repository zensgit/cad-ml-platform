"""API Versioning Module.

Provides API version management, deprecation handling, and compatibility.
"""

import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

from .base import VisionDescription, VisionProvider

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class VersionStatus(Enum):
    """API version status."""

    CURRENT = "current"
    SUPPORTED = "supported"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"
    RETIRED = "retired"


class CompatibilityLevel(Enum):
    """Compatibility levels between versions."""

    FULL = "full"
    BACKWARD = "backward"
    FORWARD = "forward"
    PARTIAL = "partial"
    NONE = "none"


class ChangeType(Enum):
    """Types of API changes."""

    ADDITION = "addition"
    MODIFICATION = "modification"
    DEPRECATION = "deprecation"
    REMOVAL = "removal"
    BREAKING = "breaking"


@dataclass
class SemanticVersion:
    """Semantic version representation."""

    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None

    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __lt__(self, other: "SemanticVersion") -> bool:
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        return self.patch < other.patch

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return False
        return self.major == other.major and self.minor == other.minor and self.patch == other.patch

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))

    def __le__(self, other: "SemanticVersion") -> bool:
        return self < other or self == other

    def __gt__(self, other: "SemanticVersion") -> bool:
        return not self <= other

    def __ge__(self, other: "SemanticVersion") -> bool:
        return not self < other

    @classmethod
    def parse(cls, version_string: str) -> "SemanticVersion":
        """Parse a version string."""
        pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.]+))?(?:\+([a-zA-Z0-9.]+))?$"
        match = re.match(pattern, version_string)
        if not match:
            raise ValueError(f"Invalid version string: {version_string}")

        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prerelease=match.group(4),
            build=match.group(5),
        )

    def is_compatible_with(self, other: "SemanticVersion") -> bool:
        """Check if this version is backward compatible with another."""
        return self.major == other.major and self >= other


@dataclass
class VersionInfo:
    """Information about an API version."""

    version: SemanticVersion
    status: VersionStatus
    release_date: datetime
    deprecation_date: Optional[datetime] = None
    sunset_date: Optional[datetime] = None
    description: str = ""
    changelog: List[str] = field(default_factory=list)
    breaking_changes: List[str] = field(default_factory=list)


@dataclass
class ApiChange:
    """Record of an API change."""

    change_id: str
    change_type: ChangeType
    endpoint: str
    description: str
    version_introduced: SemanticVersion
    version_deprecated: Optional[SemanticVersion] = None
    version_removed: Optional[SemanticVersion] = None
    migration_guide: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DeprecationWarning:
    """Deprecation warning for API usage."""

    feature: str
    deprecated_in: SemanticVersion
    removal_in: Optional[SemanticVersion]
    message: str
    migration_path: str = ""
    issued_at: datetime = field(default_factory=datetime.now)


class VersionRegistry:
    """Registry for API versions."""

    def __init__(self):
        self._versions: Dict[str, VersionInfo] = {}
        self._current: Optional[SemanticVersion] = None
        self._lock = threading.Lock()

    def register_version(self, info: VersionInfo) -> None:
        """Register a new version."""
        with self._lock:
            key = str(info.version)
            self._versions[key] = info

            if info.status == VersionStatus.CURRENT:
                self._current = info.version

    def get_version(self, version: SemanticVersion) -> Optional[VersionInfo]:
        """Get version info."""
        with self._lock:
            return self._versions.get(str(version))

    def get_current(self) -> Optional[VersionInfo]:
        """Get current version."""
        with self._lock:
            if self._current:
                return self._versions.get(str(self._current))
            return None

    def get_all_versions(self) -> List[VersionInfo]:
        """Get all registered versions."""
        with self._lock:
            return list(self._versions.values())

    def get_supported_versions(self) -> List[VersionInfo]:
        """Get all supported versions."""
        with self._lock:
            return [
                v
                for v in self._versions.values()
                if v.status in (VersionStatus.CURRENT, VersionStatus.SUPPORTED)
            ]

    def update_status(self, version: SemanticVersion, status: VersionStatus) -> bool:
        """Update version status."""
        with self._lock:
            key = str(version)
            if key not in self._versions:
                return False

            self._versions[key].status = status

            if status == VersionStatus.DEPRECATED:
                self._versions[key].deprecation_date = datetime.now()
            elif status == VersionStatus.SUNSET:
                self._versions[key].sunset_date = datetime.now()

            return True


class DeprecationManager:
    """Manages deprecation warnings and sunset schedules."""

    def __init__(
        self,
        warning_period_days: int = 90,
        sunset_period_days: int = 180,
    ):
        self.warning_period_days = warning_period_days
        self.sunset_period_days = sunset_period_days
        self._deprecations: Dict[str, DeprecationWarning] = {}
        self._callbacks: List[Callable[[DeprecationWarning], None]] = []
        self._lock = threading.Lock()

    def deprecate(
        self,
        feature: str,
        deprecated_in: SemanticVersion,
        message: str,
        removal_in: Optional[SemanticVersion] = None,
        migration_path: str = "",
    ) -> DeprecationWarning:
        """Mark a feature as deprecated."""
        warning = DeprecationWarning(
            feature=feature,
            deprecated_in=deprecated_in,
            removal_in=removal_in,
            message=message,
            migration_path=migration_path,
        )

        with self._lock:
            self._deprecations[feature] = warning

            for callback in self._callbacks:
                try:
                    callback(warning)
                except Exception:
                    pass

        return warning

    def get_deprecation(self, feature: str) -> Optional[DeprecationWarning]:
        """Get deprecation warning for a feature."""
        with self._lock:
            return self._deprecations.get(feature)

    def is_deprecated(self, feature: str) -> bool:
        """Check if a feature is deprecated."""
        with self._lock:
            return feature in self._deprecations

    def get_all_deprecations(self) -> List[DeprecationWarning]:
        """Get all deprecation warnings."""
        with self._lock:
            return list(self._deprecations.values())

    def add_callback(self, callback: Callable[[DeprecationWarning], None]) -> None:
        """Add a callback for deprecation events."""
        with self._lock:
            self._callbacks.append(callback)

    def check_sunset(self, current_version: SemanticVersion) -> List[str]:
        """Check for features that should be removed."""
        to_remove = []
        with self._lock:
            for feature, warning in self._deprecations.items():
                if warning.removal_in and current_version >= warning.removal_in:
                    to_remove.append(feature)
        return to_remove


class VersionNegotiator:
    """Negotiates API version between client and server."""

    def __init__(self, registry: VersionRegistry):
        self._registry = registry

    def negotiate(
        self,
        requested_version: Optional[str],
        accept_header: Optional[str] = None,
    ) -> Optional[VersionInfo]:
        """Negotiate the API version to use."""
        # Try explicit version first
        if requested_version:
            try:
                version = SemanticVersion.parse(requested_version)
                info = self._registry.get_version(version)
                if info and info.status in (
                    VersionStatus.CURRENT,
                    VersionStatus.SUPPORTED,
                    VersionStatus.DEPRECATED,
                ):
                    return info
            except ValueError:
                pass

        # Try accept header
        if accept_header:
            versions = self._parse_accept_header(accept_header)
            for version in versions:
                info = self._registry.get_version(version)
                if info and info.status in (
                    VersionStatus.CURRENT,
                    VersionStatus.SUPPORTED,
                ):
                    return info

        # Fall back to current version
        return self._registry.get_current()

    def _parse_accept_header(self, header: str) -> List[SemanticVersion]:
        """Parse version from accept header."""
        versions = []
        # Pattern: application/vnd.api.v1+json, application/vnd.api.v2.1+json
        pattern = r"application/vnd\.[^+]+\.v(\d+(?:\.\d+)*)"
        matches = re.findall(pattern, header)

        for match in matches:
            try:
                parts = match.split(".")
                if len(parts) == 1:
                    versions.append(SemanticVersion(int(parts[0]), 0, 0))
                elif len(parts) == 2:
                    versions.append(SemanticVersion(int(parts[0]), int(parts[1]), 0))
                else:
                    versions.append(SemanticVersion.parse(match))
            except ValueError:
                continue

        return versions

    def get_compatibility(self, v1: SemanticVersion, v2: SemanticVersion) -> CompatibilityLevel:
        """Get compatibility level between two versions."""
        if v1 == v2:
            return CompatibilityLevel.FULL

        if v1.major == v2.major:
            if v1.minor == v2.minor:
                return CompatibilityLevel.FULL
            return CompatibilityLevel.BACKWARD

        if v1.major == v2.major - 1 or v1.major == v2.major + 1:
            return CompatibilityLevel.PARTIAL

        return CompatibilityLevel.NONE


class VersionRouter:
    """Routes requests to version-specific handlers."""

    def __init__(self):
        self._handlers: Dict[str, Dict[str, Callable]] = {}
        self._lock = threading.Lock()

    def register_handler(
        self,
        version: SemanticVersion,
        endpoint: str,
        handler: Callable,
    ) -> None:
        """Register a version-specific handler."""
        with self._lock:
            version_key = str(version)
            if version_key not in self._handlers:
                self._handlers[version_key] = {}
            self._handlers[version_key][endpoint] = handler

    def get_handler(
        self,
        version: SemanticVersion,
        endpoint: str,
    ) -> Optional[Callable]:
        """Get handler for a version and endpoint."""
        with self._lock:
            version_key = str(version)
            if version_key in self._handlers:
                if endpoint in self._handlers[version_key]:
                    return self._handlers[version_key][endpoint]

            # Try to find compatible version
            for v_key, handlers in self._handlers.items():
                try:
                    v = SemanticVersion.parse(v_key)
                    if v.major == version.major and endpoint in handlers:
                        return handlers[endpoint]
                except ValueError:
                    continue

            return None

    def list_endpoints(self, version: SemanticVersion) -> List[str]:
        """List endpoints for a version."""
        with self._lock:
            version_key = str(version)
            if version_key in self._handlers:
                return list(self._handlers[version_key].keys())
            return []


class ApiVersionManager:
    """Main API version manager."""

    def __init__(self):
        self._registry = VersionRegistry()
        self._deprecation_manager = DeprecationManager()
        self._negotiator = VersionNegotiator(self._registry)
        self._router = VersionRouter()
        self._changes: List[ApiChange] = []
        self._lock = threading.Lock()

    def register_version(
        self,
        version: str,
        status: VersionStatus = VersionStatus.SUPPORTED,
        description: str = "",
        changelog: Optional[List[str]] = None,
    ) -> VersionInfo:
        """Register a new API version."""
        semver = SemanticVersion.parse(version)
        info = VersionInfo(
            version=semver,
            status=status,
            release_date=datetime.now(),
            description=description,
            changelog=changelog or [],
        )
        self._registry.register_version(info)
        return info

    def set_current(self, version: str) -> bool:
        """Set the current API version."""
        semver = SemanticVersion.parse(version)
        info = self._registry.get_version(semver)
        if not info:
            return False

        # Demote previous current
        current = self._registry.get_current()
        if current:
            self._registry.update_status(current.version, VersionStatus.SUPPORTED)

        self._registry.update_status(semver, VersionStatus.CURRENT)
        return True

    def deprecate_version(
        self,
        version: str,
        sunset_date: Optional[datetime] = None,
    ) -> bool:
        """Mark a version as deprecated."""
        semver = SemanticVersion.parse(version)
        success = self._registry.update_status(semver, VersionStatus.DEPRECATED)
        if success and sunset_date:
            info = self._registry.get_version(semver)
            if info:
                info.sunset_date = sunset_date
        return success

    def deprecate_feature(
        self,
        feature: str,
        message: str,
        deprecated_in: str,
        removal_in: Optional[str] = None,
        migration_path: str = "",
    ) -> DeprecationWarning:
        """Deprecate a specific feature."""
        dep_version = SemanticVersion.parse(deprecated_in)
        rem_version = SemanticVersion.parse(removal_in) if removal_in else None

        return self._deprecation_manager.deprecate(
            feature=feature,
            deprecated_in=dep_version,
            removal_in=rem_version,
            message=message,
            migration_path=migration_path,
        )

    def negotiate_version(
        self,
        requested: Optional[str] = None,
        accept_header: Optional[str] = None,
    ) -> Optional[VersionInfo]:
        """Negotiate API version."""
        return self._negotiator.negotiate(requested, accept_header)

    def record_change(
        self,
        change_type: ChangeType,
        endpoint: str,
        description: str,
        version: str,
    ) -> ApiChange:
        """Record an API change."""
        semver = SemanticVersion.parse(version)
        change = ApiChange(
            change_id=f"change_{int(time.time() * 1000)}",
            change_type=change_type,
            endpoint=endpoint,
            description=description,
            version_introduced=semver,
        )

        with self._lock:
            self._changes.append(change)

        return change

    def get_changelog(
        self,
        from_version: Optional[str] = None,
        to_version: Optional[str] = None,
    ) -> List[ApiChange]:
        """Get changelog between versions."""
        with self._lock:
            changes = list(self._changes)

        if from_version:
            from_semver = SemanticVersion.parse(from_version)
            changes = [c for c in changes if c.version_introduced >= from_semver]

        if to_version:
            to_semver = SemanticVersion.parse(to_version)
            changes = [c for c in changes if c.version_introduced <= to_semver]

        return sorted(changes, key=lambda c: c.version_introduced)

    def get_supported_versions(self) -> List[str]:
        """Get list of supported versions."""
        return [str(v.version) for v in self._registry.get_supported_versions()]

    def is_version_supported(self, version: str) -> bool:
        """Check if a version is supported."""
        try:
            semver = SemanticVersion.parse(version)
            info = self._registry.get_version(semver)
            return info is not None and info.status in (
                VersionStatus.CURRENT,
                VersionStatus.SUPPORTED,
                VersionStatus.DEPRECATED,
            )
        except ValueError:
            return False


def version_deprecated(
    deprecated_in: str,
    message: str,
    removal_in: Optional[str] = None,
) -> Callable[[F], F]:
    """Decorator to mark a function as deprecated."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import warnings

            warnings.warn(
                f"{func.__name__} is deprecated since {deprecated_in}: {message}",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


class VersionedVisionProvider(VisionProvider):
    """Vision provider with API versioning support."""

    def __init__(
        self,
        wrapped_provider: VisionProvider,
        version_manager: Optional[ApiVersionManager] = None,
        default_version: str = "1.0.0",
    ):
        self._wrapped = wrapped_provider
        self.version_manager = version_manager or ApiVersionManager()
        self._default_version = SemanticVersion.parse(default_version)

        # Register default version
        self.version_manager.register_version(
            default_version,
            status=VersionStatus.CURRENT,
            description="Default API version",
        )

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"versioned_{self._wrapped.provider_name}"

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True, **kwargs: Any
    ) -> VisionDescription:
        """Analyze image with version handling."""
        # Extract version from kwargs
        requested_version = kwargs.pop("api_version", None)

        # Negotiate version
        version_info = self.version_manager.negotiate_version(requested_version)
        if not version_info:
            version_info = self.version_manager._registry.get_current()

        # Check for deprecation warnings
        if version_info and version_info.status == VersionStatus.DEPRECATED:
            import warnings

            warnings.warn(
                f"API version {version_info.version} is deprecated",
                DeprecationWarning,
            )

        return await self._wrapped.analyze_image(image_data, include_description, **kwargs)

    def get_api_version(self) -> str:
        """Get current API version."""
        current = self.version_manager._registry.get_current()
        return str(current.version) if current else str(self._default_version)


# Factory functions
def create_version_manager() -> ApiVersionManager:
    """Create an API version manager."""
    return ApiVersionManager()


def create_versioned_provider(
    provider: VisionProvider,
    version_manager: Optional[ApiVersionManager] = None,
    default_version: str = "1.0.0",
) -> VersionedVisionProvider:
    """Create a versioned vision provider."""
    return VersionedVisionProvider(provider, version_manager, default_version)


def create_semantic_version(major: int, minor: int, patch: int) -> SemanticVersion:
    """Create a semantic version."""
    return SemanticVersion(major, minor, patch)


def parse_version(version_string: str) -> SemanticVersion:
    """Parse a version string."""
    return SemanticVersion.parse(version_string)


def create_version_info(
    version: str,
    status: VersionStatus = VersionStatus.SUPPORTED,
    description: str = "",
) -> VersionInfo:
    """Create version info."""
    return VersionInfo(
        version=SemanticVersion.parse(version),
        status=status,
        release_date=datetime.now(),
        description=description,
    )
