"""Tests for API versioning."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from fastapi import Request, Response


class TestVersionStatus:
    """Tests for VersionStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        from src.api.versioning import VersionStatus

        assert VersionStatus.CURRENT.value == "current"
        assert VersionStatus.SUPPORTED.value == "supported"
        assert VersionStatus.DEPRECATED.value == "deprecated"
        assert VersionStatus.SUNSET.value == "sunset"


class TestAPIVersion:
    """Tests for APIVersion dataclass."""

    def test_default_values(self):
        """Test default version values."""
        from src.api.versioning import APIVersion, VersionStatus

        version = APIVersion(
            version="v1",
            status=VersionStatus.CURRENT,
            release_date=datetime(2024, 1, 1),
        )

        assert version.version == "v1"
        assert version.status == VersionStatus.CURRENT
        assert version.features == set()
        assert version.breaking_changes == []

    def test_to_dict(self):
        """Test to_dict conversion."""
        from src.api.versioning import APIVersion, VersionStatus

        version = APIVersion(
            version="v1",
            status=VersionStatus.CURRENT,
            release_date=datetime(2024, 1, 1),
            features={"feature1", "feature2"},
        )

        data = version.to_dict()

        assert data["version"] == "v1"
        assert data["status"] == "current"
        assert "feature1" in data["features"]


class TestAPIVersionManager:
    """Tests for APIVersionManager."""

    def test_register_version(self):
        """Test version registration."""
        from fastapi import APIRouter

        from src.api.versioning import APIVersionManager, VersionStatus

        manager = APIVersionManager()
        router = APIRouter()

        manager.register_version(
            version="v1",
            router=router,
            status=VersionStatus.CURRENT,
        )

        assert "v1" in manager._versions
        assert "v1" in manager._routers

    def test_get_version(self):
        """Test getting version info."""
        from fastapi import APIRouter

        from src.api.versioning import APIVersionManager, VersionStatus

        manager = APIVersionManager()
        router = APIRouter()

        manager.register_version("v1", router, status=VersionStatus.CURRENT)

        version = manager.get_version("v1")
        assert version is not None
        assert version.version == "v1"

    def test_get_version_not_found(self):
        """Test getting non-existent version."""
        from src.api.versioning import APIVersionManager

        manager = APIVersionManager()

        version = manager.get_version("v99")
        assert version is None

    def test_list_versions(self):
        """Test listing versions."""
        from fastapi import APIRouter

        from src.api.versioning import APIVersionManager, VersionStatus

        manager = APIVersionManager()

        manager.register_version("v1", APIRouter(), status=VersionStatus.SUPPORTED)
        manager.register_version("v2", APIRouter(), status=VersionStatus.CURRENT)

        versions = manager.list_versions()

        assert len(versions) == 2
        version_names = [v["version"] for v in versions]
        assert "v1" in version_names
        assert "v2" in version_names

    def test_get_current_version(self):
        """Test getting current version."""
        from fastapi import APIRouter

        from src.api.versioning import APIVersionManager, VersionStatus

        manager = APIVersionManager()

        manager.register_version("v1", APIRouter(), status=VersionStatus.DEPRECATED)
        manager.register_version("v2", APIRouter(), status=VersionStatus.CURRENT)

        current = manager.get_current_version()
        assert current == "v2"

    def test_is_deprecated(self):
        """Test deprecated check."""
        from fastapi import APIRouter

        from src.api.versioning import APIVersionManager, VersionStatus

        manager = APIVersionManager()

        manager.register_version("v1", APIRouter(), status=VersionStatus.DEPRECATED)
        manager.register_version("v2", APIRouter(), status=VersionStatus.CURRENT)

        assert manager.is_deprecated("v1") is True
        assert manager.is_deprecated("v2") is False

    def test_is_sunset(self):
        """Test sunset check."""
        from fastapi import APIRouter

        from src.api.versioning import APIVersionManager, VersionStatus

        manager = APIVersionManager()

        manager.register_version("v0", APIRouter(), status=VersionStatus.SUNSET)
        manager.register_version("v1", APIRouter(), status=VersionStatus.CURRENT)

        assert manager.is_sunset("v0") is True
        assert manager.is_sunset("v1") is False
        assert manager.is_sunset("v99") is True  # Unknown versions

    def test_resolve_version_from_header(self):
        """Test version resolution from header."""
        from src.api.versioning import APIVersionManager

        manager = APIVersionManager(default_version="v1")

        # Header takes precedence
        version = manager.resolve_version(
            path_version="v1",
            header_version="v2",
        )

        assert version == "v2"

    def test_resolve_version_from_path(self):
        """Test version resolution from path."""
        from src.api.versioning import APIVersionManager

        manager = APIVersionManager(default_version="v1")

        version = manager.resolve_version(
            path_version="v2",
            header_version=None,
        )

        assert version == "v2"

    def test_resolve_version_default(self):
        """Test default version resolution."""
        from src.api.versioning import APIVersionManager

        manager = APIVersionManager(default_version="v1")

        version = manager.resolve_version()

        assert version == "v1"

    def test_resolve_version_normalizes_format(self):
        """Test version format normalization."""
        from src.api.versioning import APIVersionManager

        manager = APIVersionManager()

        # Without 'v' prefix
        version = manager.resolve_version(header_version="2")
        assert version == "v2"

        # With 'v' prefix
        version = manager.resolve_version(header_version="v3")
        assert version == "v3"

    @pytest.mark.asyncio
    async def test_check_version_adds_headers(self):
        """Test check_version adds headers."""
        from fastapi import APIRouter

        from src.api.versioning import APIVersionManager, VersionStatus

        manager = APIVersionManager()
        manager.register_version("v1", APIRouter(), status=VersionStatus.CURRENT)

        request = MagicMock(spec=Request)
        response = MagicMock(spec=Response)
        response.headers = {}

        await manager.check_version(request, response, "v1")

        assert response.headers["X-API-Version"] == "v1"
        assert response.headers["X-API-Version-Status"] == "current"

    @pytest.mark.asyncio
    async def test_check_version_deprecated_adds_warning(self):
        """Test deprecated version adds warning headers."""
        from datetime import datetime

        from fastapi import APIRouter

        from src.api.versioning import APIVersionManager, VersionStatus

        manager = APIVersionManager()
        manager.register_version(
            "v1",
            APIRouter(),
            status=VersionStatus.DEPRECATED,
            deprecation_date=datetime(2024, 6, 1),
            sunset_date=datetime(2024, 12, 1),
        )
        manager.register_version("v2", APIRouter(), status=VersionStatus.CURRENT)

        request = MagicMock(spec=Request)
        response = MagicMock(spec=Response)
        response.headers = {}

        await manager.check_version(request, response, "v1")

        assert response.headers["X-API-Version-Status"] == "deprecated"
        assert "X-API-Deprecation-Date" in response.headers
        assert response.headers["X-API-Upgrade-To"] == "v2"

    @pytest.mark.asyncio
    async def test_check_version_sunset_raises(self):
        """Test sunset version raises HTTPException."""
        from fastapi import APIRouter, HTTPException

        from src.api.versioning import APIVersionManager, VersionStatus

        manager = APIVersionManager()
        manager.register_version("v0", APIRouter(), status=VersionStatus.SUNSET)

        request = MagicMock(spec=Request)
        response = MagicMock(spec=Response)
        response.headers = {}

        with pytest.raises(HTTPException) as exc_info:
            await manager.check_version(request, response, "v0")

        assert exc_info.value.status_code == 410

    @pytest.mark.asyncio
    async def test_check_version_unknown_raises(self):
        """Test unknown version raises HTTPException."""
        from fastapi import HTTPException

        from src.api.versioning import APIVersionManager

        manager = APIVersionManager()

        request = MagicMock(spec=Request)
        response = MagicMock(spec=Response)

        with pytest.raises(HTTPException) as exc_info:
            await manager.check_version(request, response, "v99")

        assert exc_info.value.status_code == 400

    def test_deprecation_handler(self):
        """Test deprecation handler is called."""
        from datetime import datetime

        from fastapi import APIRouter

        from src.api.versioning import APIVersionManager, VersionStatus

        manager = APIVersionManager()
        manager.register_version(
            "v1",
            APIRouter(),
            status=VersionStatus.DEPRECATED,
            deprecation_date=datetime(2024, 6, 1),
        )

        called = []

        def handler(version: str, request: Request) -> None:
            called.append(version)

        manager.add_deprecation_handler(handler)

        # Note: handler is called in check_version which is async
        # This test just verifies handler is added
        assert len(manager._deprecation_handlers) == 1


class TestGetVersionManager:
    """Tests for get_version_manager function."""

    def test_returns_singleton(self):
        """Test returns singleton instance."""
        from src.api import versioning as version_module

        # Reset global
        version_module._version_manager = None

        mgr1 = version_module.get_version_manager()
        mgr2 = version_module.get_version_manager()

        assert mgr1 is mgr2

        # Cleanup
        version_module._version_manager = None
