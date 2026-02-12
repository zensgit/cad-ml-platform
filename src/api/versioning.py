"""API Version Management.

Features:
- Multiple API versions (v1, v2, ...)
- Version negotiation via header or path
- Deprecation warnings
- Version-specific feature flags
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import Header, HTTPException, Request, Response
from fastapi.routing import APIRouter

logger = logging.getLogger(__name__)


class VersionStatus(Enum):
    """API version status."""

    CURRENT = "current"  # Active and recommended
    SUPPORTED = "supported"  # Active but not recommended
    DEPRECATED = "deprecated"  # Still works but will be removed
    SUNSET = "sunset"  # No longer available


@dataclass
class APIVersion:
    """Represents an API version."""

    version: str
    status: VersionStatus
    release_date: datetime
    deprecation_date: Optional[datetime] = None
    sunset_date: Optional[datetime] = None
    changelog: str = ""
    features: Set[str] = field(default_factory=set)
    breaking_changes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status.value,
            "release_date": self.release_date.isoformat(),
            "deprecation_date": self.deprecation_date.isoformat() if self.deprecation_date else None,
            "sunset_date": self.sunset_date.isoformat() if self.sunset_date else None,
            "changelog": self.changelog,
            "features": list(self.features),
            "breaking_changes": self.breaking_changes,
        }


class APIVersionManager:
    """Manages API versions and routing."""

    def __init__(self, default_version: str = "v1"):
        self.default_version = default_version
        self._versions: Dict[str, APIVersion] = {}
        self._routers: Dict[str, APIRouter] = {}
        self._deprecation_handlers: List[Callable] = []

    def register_version(
        self,
        version: str,
        router: APIRouter,
        status: VersionStatus = VersionStatus.CURRENT,
        release_date: Optional[datetime] = None,
        deprecation_date: Optional[datetime] = None,
        sunset_date: Optional[datetime] = None,
        changelog: str = "",
        features: Optional[Set[str]] = None,
        breaking_changes: Optional[List[str]] = None,
    ) -> None:
        """Register an API version."""
        api_version = APIVersion(
            version=version,
            status=status,
            release_date=release_date or datetime.now(),
            deprecation_date=deprecation_date,
            sunset_date=sunset_date,
            changelog=changelog,
            features=features or set(),
            breaking_changes=breaking_changes or [],
        )

        self._versions[version] = api_version
        self._routers[version] = router

        logger.info(f"Registered API version: {version} ({status.value})")

    def get_version(self, version: str) -> Optional[APIVersion]:
        """Get version info."""
        return self._versions.get(version)

    def get_router(self, version: str) -> Optional[APIRouter]:
        """Get router for a version."""
        return self._routers.get(version)

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all versions."""
        return [v.to_dict() for v in self._versions.values()]

    def get_current_version(self) -> Optional[str]:
        """Get the current recommended version."""
        for version, info in self._versions.items():
            if info.status == VersionStatus.CURRENT:
                return version
        return self.default_version

    def is_deprecated(self, version: str) -> bool:
        """Check if a version is deprecated."""
        info = self._versions.get(version)
        if not info:
            return False
        return info.status in (VersionStatus.DEPRECATED, VersionStatus.SUNSET)

    def is_sunset(self, version: str) -> bool:
        """Check if a version is sunset."""
        info = self._versions.get(version)
        if not info:
            return True  # Unknown versions are considered sunset
        return info.status == VersionStatus.SUNSET

    def resolve_version(
        self,
        path_version: Optional[str] = None,
        header_version: Optional[str] = None,
    ) -> str:
        """Resolve the API version to use."""
        # Header takes precedence
        if header_version:
            version = header_version.strip().lower()
            if version.startswith("v"):
                return version
            return f"v{version}"

        # Then path version
        if path_version:
            return path_version

        # Default
        return self.default_version

    def add_deprecation_handler(self, handler: Callable[[str, Request], None]) -> None:
        """Add a handler for deprecated version usage."""
        self._deprecation_handlers.append(handler)

    async def check_version(
        self,
        request: Request,
        response: Response,
        version: str,
    ) -> None:
        """Check version and add appropriate headers."""
        info = self._versions.get(version)

        if not info:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown API version: {version}",
            )

        if info.status == VersionStatus.SUNSET:
            raise HTTPException(
                status_code=410,
                detail=f"API version {version} is no longer available. "
                f"Please upgrade to {self.get_current_version()}",
            )

        # Add version headers
        response.headers["X-API-Version"] = version
        response.headers["X-API-Version-Status"] = info.status.value

        if info.status == VersionStatus.DEPRECATED:
            response.headers["X-API-Deprecation-Date"] = (
                info.deprecation_date.isoformat() if info.deprecation_date else ""
            )
            if info.sunset_date:
                response.headers["X-API-Sunset-Date"] = info.sunset_date.isoformat()

            current = self.get_current_version()
            response.headers["X-API-Upgrade-To"] = current or ""

            # Call deprecation handlers
            for handler in self._deprecation_handlers:
                try:
                    handler(version, request)
                except Exception as e:
                    logger.warning(f"Deprecation handler error: {e}")


def version_header_dependency(
    x_api_version: Optional[str] = Header(None, alias="X-API-Version"),
) -> Optional[str]:
    """FastAPI dependency to extract version from header."""
    return x_api_version


# Global version manager
_version_manager: Optional[APIVersionManager] = None


def get_version_manager() -> APIVersionManager:
    """Get the global version manager instance."""
    global _version_manager
    if _version_manager is None:
        _version_manager = APIVersionManager()
    return _version_manager
