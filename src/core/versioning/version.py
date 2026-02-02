"""API Version Management.

Provides version parsing and comparison:
- Semantic versioning
- Version ranges
- Compatibility checking
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union


class VersionFormat(Enum):
    """Supported version formats."""
    SEMANTIC = "semantic"  # 1.2.3
    DATE = "date"  # 2024-01-15
    INTEGER = "integer"  # v1, v2


@dataclass(frozen=True)
class SemanticVersion:
    """Semantic version (major.minor.patch)."""
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None

    @classmethod
    def parse(cls, version_str: str) -> "SemanticVersion":
        """Parse a semantic version string.

        Args:
            version_str: Version string like "1.2.3", "v1.2.3", "1.2.3-beta.1+build.123"

        Returns:
            SemanticVersion instance.

        Raises:
            ValueError: If version string is invalid.
        """
        # Remove leading 'v' if present
        if version_str.startswith('v'):
            version_str = version_str[1:]

        # Pattern for semantic versioning
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.-]+))?(?:\+([a-zA-Z0-9.-]+))?$'
        match = re.match(pattern, version_str)

        if not match:
            raise ValueError(f"Invalid semantic version: {version_str}")

        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prerelease=match.group(4),
            build=match.group(5),
        )

    def __str__(self) -> str:
        result = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            result += f"-{self.prerelease}"
        if self.build:
            result += f"+{self.build}"
        return result

    def __lt__(self, other: "SemanticVersion") -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented

        # Compare major, minor, patch
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

        # Prerelease versions have lower precedence
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False
        if self.prerelease and other.prerelease:
            return self._compare_prerelease(self.prerelease, other.prerelease) < 0

        return False

    def __le__(self, other: "SemanticVersion") -> bool:
        return self == other or self < other

    def __gt__(self, other: "SemanticVersion") -> bool:
        return not self <= other

    def __ge__(self, other: "SemanticVersion") -> bool:
        return not self < other

    @staticmethod
    def _compare_prerelease(a: str, b: str) -> int:
        """Compare prerelease strings."""
        parts_a = a.split('.')
        parts_b = b.split('.')

        for pa, pb in zip(parts_a, parts_b):
            # Numeric parts compare as integers
            if pa.isdigit() and pb.isdigit():
                if int(pa) != int(pb):
                    return int(pa) - int(pb)
            # Numeric < non-numeric
            elif pa.isdigit():
                return -1
            elif pb.isdigit():
                return 1
            # Alphanumeric compare lexically
            elif pa != pb:
                return -1 if pa < pb else 1

        return len(parts_a) - len(parts_b)

    def is_compatible_with(self, other: "SemanticVersion") -> bool:
        """Check if this version is backward compatible with another.

        Following semver rules:
        - Same major version = compatible (for major > 0)
        - Major 0 = development, minor changes may break
        """
        if self.major == 0 or other.major == 0:
            # Development versions - minor must match
            return self.major == other.major and self.minor == other.minor
        return self.major == other.major

    def bump_major(self) -> "SemanticVersion":
        """Bump major version."""
        return SemanticVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> "SemanticVersion":
        """Bump minor version."""
        return SemanticVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> "SemanticVersion":
        """Bump patch version."""
        return SemanticVersion(self.major, self.minor, self.patch + 1)


@dataclass
class APIVersion:
    """API version with metadata."""
    version: SemanticVersion
    released_at: Optional[str] = None
    deprecated_at: Optional[str] = None
    sunset_at: Optional[str] = None
    changelog_url: Optional[str] = None

    @property
    def is_deprecated(self) -> bool:
        """Check if version is deprecated."""
        return self.deprecated_at is not None

    @property
    def is_sunset(self) -> bool:
        """Check if version is sunset (no longer available)."""
        return self.sunset_at is not None

    def __str__(self) -> str:
        status = ""
        if self.is_sunset:
            status = " (sunset)"
        elif self.is_deprecated:
            status = " (deprecated)"
        return f"v{self.version}{status}"


class VersionRange:
    """Version range specification."""

    def __init__(
        self,
        min_version: Optional[SemanticVersion] = None,
        max_version: Optional[SemanticVersion] = None,
        include_min: bool = True,
        include_max: bool = True,
    ):
        self.min_version = min_version
        self.max_version = max_version
        self.include_min = include_min
        self.include_max = include_max

    @classmethod
    def parse(cls, range_str: str) -> "VersionRange":
        """Parse a version range string.

        Supported formats:
        - "1.0.0" - Exact version
        - ">=1.0.0" - Minimum version (inclusive)
        - ">1.0.0" - Minimum version (exclusive)
        - "<=2.0.0" - Maximum version (inclusive)
        - "<2.0.0" - Maximum version (exclusive)
        - ">=1.0.0,<2.0.0" - Range
        - "^1.0.0" - Compatible with (same major)
        - "~1.0.0" - Approximately (same minor)

        Args:
            range_str: Version range string.

        Returns:
            VersionRange instance.
        """
        range_str = range_str.strip()

        # Caret range (^1.0.0 = >=1.0.0, <2.0.0)
        if range_str.startswith('^'):
            version = SemanticVersion.parse(range_str[1:])
            return cls(
                min_version=version,
                max_version=version.bump_major(),
                include_min=True,
                include_max=False,
            )

        # Tilde range (~1.0.0 = >=1.0.0, <1.1.0)
        if range_str.startswith('~'):
            version = SemanticVersion.parse(range_str[1:])
            return cls(
                min_version=version,
                max_version=version.bump_minor(),
                include_min=True,
                include_max=False,
            )

        # Range with comma
        if ',' in range_str:
            parts = range_str.split(',')
            result = cls()
            for part in parts:
                part = part.strip()
                partial = cls.parse(part)
                if partial.min_version:
                    result.min_version = partial.min_version
                    result.include_min = partial.include_min
                if partial.max_version:
                    result.max_version = partial.max_version
                    result.include_max = partial.include_max
            return result

        # Operators
        if range_str.startswith('>='):
            return cls(min_version=SemanticVersion.parse(range_str[2:]), include_min=True)
        if range_str.startswith('>'):
            return cls(min_version=SemanticVersion.parse(range_str[1:]), include_min=False)
        if range_str.startswith('<='):
            return cls(max_version=SemanticVersion.parse(range_str[2:]), include_max=True)
        if range_str.startswith('<'):
            return cls(max_version=SemanticVersion.parse(range_str[1:]), include_max=False)

        # Exact version
        version = SemanticVersion.parse(range_str)
        return cls(min_version=version, max_version=version, include_min=True, include_max=True)

    def contains(self, version: SemanticVersion) -> bool:
        """Check if a version is within this range."""
        if self.min_version:
            if self.include_min:
                if version < self.min_version:
                    return False
            else:
                if version <= self.min_version:
                    return False

        if self.max_version:
            if self.include_max:
                if version > self.max_version:
                    return False
            else:
                if version >= self.max_version:
                    return False

        return True

    def __str__(self) -> str:
        parts = []
        if self.min_version:
            op = ">=" if self.include_min else ">"
            parts.append(f"{op}{self.min_version}")
        if self.max_version:
            op = "<=" if self.include_max else "<"
            parts.append(f"{op}{self.max_version}")
        return ", ".join(parts) if parts else "*"


class VersionRegistry:
    """Registry of API versions."""

    def __init__(self):
        self._versions: List[APIVersion] = []
        self._current: Optional[APIVersion] = None

    def register(
        self,
        version: Union[str, SemanticVersion],
        released_at: Optional[str] = None,
        changelog_url: Optional[str] = None,
    ) -> APIVersion:
        """Register a new API version."""
        if isinstance(version, str):
            version = SemanticVersion.parse(version)

        api_version = APIVersion(
            version=version,
            released_at=released_at,
            changelog_url=changelog_url,
        )

        self._versions.append(api_version)
        self._versions.sort(key=lambda v: v.version)

        # Set as current if it's the highest non-deprecated version
        if not api_version.is_deprecated:
            if not self._current or api_version.version > self._current.version:
                self._current = api_version

        return api_version

    def deprecate(
        self,
        version: Union[str, SemanticVersion],
        deprecated_at: str,
        sunset_at: Optional[str] = None,
    ) -> Optional[APIVersion]:
        """Mark a version as deprecated."""
        if isinstance(version, str):
            version = SemanticVersion.parse(version)

        api_version = self.get(version)
        if api_version:
            api_version.deprecated_at = deprecated_at
            api_version.sunset_at = sunset_at
        return api_version

    def get(self, version: Union[str, SemanticVersion]) -> Optional[APIVersion]:
        """Get an API version."""
        if isinstance(version, str):
            version = SemanticVersion.parse(version)

        for v in self._versions:
            if v.version == version:
                return v
        return None

    @property
    def current(self) -> Optional[APIVersion]:
        """Get the current (latest non-deprecated) version."""
        return self._current

    @property
    def latest(self) -> Optional[APIVersion]:
        """Get the latest version (including deprecated)."""
        return self._versions[-1] if self._versions else None

    def get_supported_versions(self) -> List[APIVersion]:
        """Get all non-sunset versions."""
        return [v for v in self._versions if not v.is_sunset]

    def get_deprecated_versions(self) -> List[APIVersion]:
        """Get deprecated but still available versions."""
        return [v for v in self._versions if v.is_deprecated and not v.is_sunset]

    def find_compatible(self, version: SemanticVersion) -> List[APIVersion]:
        """Find all versions compatible with the given version."""
        return [
            v for v in self._versions
            if v.version.is_compatible_with(version) and not v.is_sunset
        ]
