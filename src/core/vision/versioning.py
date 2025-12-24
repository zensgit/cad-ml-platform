"""Provider versioning module for Vision Provider system.

This module provides versioning capabilities including:
- Provider version management
- Version compatibility checking
- Version-based routing
- Rollback support
- Version deprecation handling
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import VisionDescription, VisionProvider


class VersionStatus(Enum):
    """Status of a provider version."""

    DEVELOPMENT = "development"  # In development, not for production
    PREVIEW = "preview"  # Preview/beta release
    STABLE = "stable"  # Stable production release
    DEPRECATED = "deprecated"  # Deprecated, will be removed
    RETIRED = "retired"  # No longer available


class CompatibilityLevel(Enum):
    """Compatibility level between versions."""

    COMPATIBLE = "compatible"  # Fully compatible
    BACKWARD_COMPATIBLE = "backward_compatible"  # Backward compatible only
    BREAKING = "breaking"  # Breaking changes
    UNKNOWN = "unknown"


@dataclass
class SemanticVersion:
    """Semantic version representation."""

    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None

    def __str__(self) -> str:
        """Return string representation."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __lt__(self, other: "SemanticVersion") -> bool:
        """Compare versions."""
        return (self.major, self.minor, self.patch) < (
            other.major,
            other.minor,
            other.patch,
        )

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, SemanticVersion):
            return False
        return self.major == other.major and self.minor == other.minor and self.patch == other.patch

    def __hash__(self) -> int:
        """Return hash."""
        return hash((self.major, self.minor, self.patch))

    @classmethod
    def parse(cls, version_str: str) -> "SemanticVersion":
        """Parse version string.

        Args:
            version_str: Version string (e.g., "1.2.3-beta+build")

        Returns:
            SemanticVersion instance
        """
        build = None
        prerelease = None

        # Extract build metadata
        if "+" in version_str:
            version_str, build = version_str.rsplit("+", 1)

        # Extract prerelease
        if "-" in version_str:
            version_str, prerelease = version_str.split("-", 1)

        parts = version_str.split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0

        return cls(
            major=major,
            minor=minor,
            patch=patch,
            prerelease=prerelease,
            build=build,
        )

    def is_compatible_with(self, other: "SemanticVersion") -> CompatibilityLevel:
        """Check compatibility with another version.

        Args:
            other: Version to compare with

        Returns:
            Compatibility level
        """
        if self == other:
            return CompatibilityLevel.COMPATIBLE

        if self.major != other.major:
            return CompatibilityLevel.BREAKING

        if self.major == 0:
            # Pre-1.0 versions may have breaking changes
            if self.minor != other.minor:
                return CompatibilityLevel.BREAKING

        return CompatibilityLevel.BACKWARD_COMPATIBLE


@dataclass
class ProviderVersion:
    """Version information for a provider."""

    version: SemanticVersion
    provider: VisionProvider
    status: VersionStatus = VersionStatus.STABLE
    release_date: Optional[datetime] = None
    deprecation_date: Optional[datetime] = None
    retirement_date: Optional[datetime] = None
    changelog: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_available(self) -> bool:
        """Check if version is available for use."""
        return self.status not in (VersionStatus.RETIRED,)

    def is_production_ready(self) -> bool:
        """Check if version is production-ready."""
        return self.status == VersionStatus.STABLE


@dataclass
class VersionConstraint:
    """Constraint for version selection."""

    min_version: Optional[SemanticVersion] = None
    max_version: Optional[SemanticVersion] = None
    excluded_versions: List[SemanticVersion] = field(default_factory=list)
    required_status: Optional[VersionStatus] = None
    allow_prerelease: bool = False

    def matches(self, version: ProviderVersion) -> bool:
        """Check if version matches constraint."""
        v = version.version

        if self.min_version and v < self.min_version:
            return False

        if self.max_version and v > self.max_version:
            return False

        if v in self.excluded_versions:
            return False

        if self.required_status and version.status != self.required_status:
            return False

        if not self.allow_prerelease and v.prerelease:
            return False

        return True


@dataclass
class VersionHistory:
    """History of version selections."""

    entries: List[Tuple[datetime, SemanticVersion, str]] = field(default_factory=list)

    def add_entry(self, version: SemanticVersion, reason: str) -> None:
        """Add history entry."""
        self.entries.append((datetime.now(), version, reason))

    def get_recent(self, count: int = 10) -> List[Tuple[datetime, SemanticVersion, str]]:
        """Get recent entries."""
        return self.entries[-count:]


class VersionRegistry:
    """Registry for provider versions."""

    def __init__(self) -> None:
        """Initialize version registry."""
        self._providers: Dict[str, Dict[SemanticVersion, ProviderVersion]] = {}
        self._default_versions: Dict[str, SemanticVersion] = {}
        self._history: Dict[str, VersionHistory] = {}

    def register(
        self,
        provider_name: str,
        version: SemanticVersion,
        provider: VisionProvider,
        status: VersionStatus = VersionStatus.STABLE,
        changelog: Optional[List[str]] = None,
        is_default: bool = False,
    ) -> ProviderVersion:
        """Register a provider version.

        Args:
            provider_name: Name of provider
            version: Version to register
            provider: Provider instance
            status: Version status
            changelog: Version changelog
            is_default: Set as default version

        Returns:
            ProviderVersion instance
        """
        if provider_name not in self._providers:
            self._providers[provider_name] = {}
            self._history[provider_name] = VersionHistory()

        pv = ProviderVersion(
            version=version,
            provider=provider,
            status=status,
            release_date=datetime.now(),
            changelog=changelog or [],
        )

        self._providers[provider_name][version] = pv

        if is_default or provider_name not in self._default_versions:
            self._default_versions[provider_name] = version

        return pv

    def unregister(
        self,
        provider_name: str,
        version: SemanticVersion,
    ) -> bool:
        """Unregister a provider version.

        Args:
            provider_name: Name of provider
            version: Version to unregister

        Returns:
            True if unregistered
        """
        if provider_name not in self._providers:
            return False

        if version not in self._providers[provider_name]:
            return False

        del self._providers[provider_name][version]

        # Update default if needed
        if self._default_versions.get(provider_name) == version:
            versions = list(self._providers[provider_name].keys())
            if versions:
                self._default_versions[provider_name] = max(versions)
            else:
                del self._default_versions[provider_name]

        return True

    def get_version(
        self,
        provider_name: str,
        version: Optional[SemanticVersion] = None,
    ) -> Optional[ProviderVersion]:
        """Get a specific version.

        Args:
            provider_name: Name of provider
            version: Version to get (None for default)

        Returns:
            ProviderVersion or None
        """
        if provider_name not in self._providers:
            return None

        if version is None:
            version = self._default_versions.get(provider_name)
            if version is None:
                return None

        return self._providers[provider_name].get(version)

    def get_all_versions(
        self,
        provider_name: str,
    ) -> List[ProviderVersion]:
        """Get all versions for a provider.

        Args:
            provider_name: Name of provider

        Returns:
            List of versions sorted by version number
        """
        if provider_name not in self._providers:
            return []

        versions = list(self._providers[provider_name].values())
        versions.sort(key=lambda v: v.version, reverse=True)
        return versions

    def find_matching(
        self,
        provider_name: str,
        constraint: VersionConstraint,
    ) -> List[ProviderVersion]:
        """Find versions matching constraint.

        Args:
            provider_name: Name of provider
            constraint: Version constraint

        Returns:
            List of matching versions
        """
        all_versions = self.get_all_versions(provider_name)
        return [v for v in all_versions if constraint.matches(v)]

    def get_latest(
        self,
        provider_name: str,
        status: Optional[VersionStatus] = None,
    ) -> Optional[ProviderVersion]:
        """Get latest version.

        Args:
            provider_name: Name of provider
            status: Filter by status

        Returns:
            Latest version or None
        """
        versions = self.get_all_versions(provider_name)

        if status:
            versions = [v for v in versions if v.status == status]

        if not versions:
            return None

        return versions[0]  # Already sorted descending

    def set_default(
        self,
        provider_name: str,
        version: SemanticVersion,
    ) -> bool:
        """Set default version.

        Args:
            provider_name: Name of provider
            version: Version to set as default

        Returns:
            True if successful
        """
        if provider_name not in self._providers:
            return False

        if version not in self._providers[provider_name]:
            return False

        self._default_versions[provider_name] = version
        return True

    def deprecate(
        self,
        provider_name: str,
        version: SemanticVersion,
        retirement_date: Optional[datetime] = None,
    ) -> bool:
        """Deprecate a version.

        Args:
            provider_name: Name of provider
            version: Version to deprecate
            retirement_date: When version will be retired

        Returns:
            True if successful
        """
        pv = self.get_version(provider_name, version)
        if not pv:
            return False

        pv.status = VersionStatus.DEPRECATED
        pv.deprecation_date = datetime.now()
        pv.retirement_date = retirement_date
        return True

    def retire(
        self,
        provider_name: str,
        version: SemanticVersion,
    ) -> bool:
        """Retire a version.

        Args:
            provider_name: Name of provider
            version: Version to retire

        Returns:
            True if successful
        """
        pv = self.get_version(provider_name, version)
        if not pv:
            return False

        pv.status = VersionStatus.RETIRED
        return True

    def get_providers(self) -> List[str]:
        """Get all registered provider names."""
        return list(self._providers.keys())


class VersionedVisionProvider(VisionProvider):
    """Vision provider with version management."""

    def __init__(
        self,
        provider_name: str,
        registry: VersionRegistry,
        version: Optional[SemanticVersion] = None,
        constraint: Optional[VersionConstraint] = None,
    ) -> None:
        """Initialize versioned provider.

        Args:
            provider_name: Name of provider
            registry: Version registry
            version: Specific version to use
            constraint: Version constraint for selection
        """
        self._provider_name = provider_name
        self._registry = registry
        self._version = version
        self._constraint = constraint

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        v = self._get_current_version()
        if v:
            return f"{self._provider_name}_v{v.version}"
        return f"{self._provider_name}_unversioned"

    @property
    def registry(self) -> VersionRegistry:
        """Get version registry."""
        return self._registry

    @property
    def current_version(self) -> Optional[SemanticVersion]:
        """Get current version."""
        v = self._get_current_version()
        return v.version if v else None

    def _get_current_version(self) -> Optional[ProviderVersion]:
        """Get current provider version."""
        if self._version:
            return self._registry.get_version(self._provider_name, self._version)

        if self._constraint:
            matches = self._registry.find_matching(self._provider_name, self._constraint)
            if matches:
                return matches[0]

        return self._registry.get_version(self._provider_name)

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image using versioned provider.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        pv = self._get_current_version()
        if not pv:
            raise RuntimeError(f"No version available for {self._provider_name}")

        if not pv.is_available():
            raise RuntimeError(f"Version {pv.version} is not available (status: {pv.status})")

        return await pv.provider.analyze_image(image_data, include_description)


# Global registry instance
_registry: Optional[VersionRegistry] = None


def get_version_registry() -> VersionRegistry:
    """Get global version registry."""
    global _registry
    if _registry is None:
        _registry = VersionRegistry()
    return _registry


def create_versioned_provider(
    provider_name: str,
    version: Optional[str] = None,
    constraint: Optional[VersionConstraint] = None,
    registry: Optional[VersionRegistry] = None,
) -> VersionedVisionProvider:
    """Create a versioned vision provider.

    Args:
        provider_name: Name of provider
        version: Specific version string
        constraint: Version constraint
        registry: Optional registry instance

    Returns:
        VersionedVisionProvider instance
    """
    if registry is None:
        registry = get_version_registry()

    sem_version = None
    if version:
        sem_version = SemanticVersion.parse(version)

    return VersionedVisionProvider(
        provider_name=provider_name,
        registry=registry,
        version=sem_version,
        constraint=constraint,
    )
