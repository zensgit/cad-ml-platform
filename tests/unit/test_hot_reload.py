"""Tests for model hot reload."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestModelStatus:
    """Tests for ModelStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        from src.core.model.hot_reload import ModelStatus

        assert ModelStatus.LOADING.value == "loading"
        assert ModelStatus.READY.value == "ready"
        assert ModelStatus.FAILED.value == "failed"
        assert ModelStatus.DRAINING.value == "draining"
        assert ModelStatus.RETIRED.value == "retired"


class TestModelVersion:
    """Tests for ModelVersion dataclass."""

    def test_default_values(self):
        """Test default values."""
        from src.core.model.hot_reload import ModelStatus, ModelVersion

        version = ModelVersion(version="v1", path="/path/to/model")

        assert version.version == "v1"
        assert version.status == ModelStatus.LOADING
        assert version.request_count == 0
        assert version.error_count == 0

    def test_to_dict(self):
        """Test to_dict conversion."""
        from src.core.model.hot_reload import ModelStatus, ModelVersion

        version = ModelVersion(
            version="v1",
            path="/path/to/model",
            status=ModelStatus.READY,
        )

        data = version.to_dict()

        assert data["version"] == "v1"
        assert data["status"] == "ready"
        assert data["path"] == "/path/to/model"


class TestReloadConfig:
    """Tests for ReloadConfig."""

    def test_default_values(self):
        """Test default config values."""
        from src.core.model.hot_reload import ReloadConfig

        config = ReloadConfig()

        assert config.health_check_timeout_s == 30.0
        assert config.health_check_retries == 3
        assert config.auto_rollback is True
        assert config.error_threshold == 0.1


class TestModelLoader:
    """Tests for ModelLoader."""

    @pytest.mark.asyncio
    async def test_default_loader_load(self):
        """Test default loader load."""
        from src.core.model.hot_reload import DefaultModelLoader

        loader = DefaultModelLoader()
        model = await loader.load("/path/to/model")

        assert model["loaded"] is True
        assert model["path"] == "/path/to/model"

    @pytest.mark.asyncio
    async def test_default_loader_health_check(self):
        """Test default loader health check."""
        from src.core.model.hot_reload import DefaultModelLoader

        loader = DefaultModelLoader()
        model = {"loaded": True}

        is_healthy = await loader.health_check(model)
        assert is_healthy is True

    def test_compute_checksum_file(self):
        """Test checksum computation for file."""
        from src.core.model.hot_reload import DefaultModelLoader

        loader = DefaultModelLoader()

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            f.flush()

            checksum = loader.compute_checksum(f.name)

            assert checksum != ""
            assert len(checksum) == 32  # MD5 hex length

            # Clean up
            Path(f.name).unlink()

    def test_compute_checksum_missing(self):
        """Test checksum for missing path."""
        from src.core.model.hot_reload import DefaultModelLoader

        loader = DefaultModelLoader()
        checksum = loader.compute_checksum("/nonexistent/path")

        assert checksum == ""


class TestHotReloadManager:
    """Tests for HotReloadManager."""

    @pytest.mark.asyncio
    async def test_load_model_success(self):
        """Test successful model loading."""
        from src.core.model.hot_reload import HotReloadManager, ModelStatus

        manager = HotReloadManager()

        with tempfile.NamedTemporaryFile() as f:
            result = await manager.load_model("v1", f.name)

        assert result is True
        assert manager._active_version == "v1"
        assert manager._versions["v1"].status == ModelStatus.READY

    @pytest.mark.asyncio
    async def test_load_model_failed_health_check(self):
        """Test model loading with failed health check."""
        from src.core.model.hot_reload import (
            HotReloadManager,
            ModelLoader,
            ModelStatus,
        )

        class FailingLoader(ModelLoader):
            async def load(self, path):
                return {"loaded": True}

            async def health_check(self, model):
                return False

        manager = HotReloadManager(loader=FailingLoader())

        with tempfile.NamedTemporaryFile() as f:
            result = await manager.load_model("v1", f.name)

        assert result is False
        assert manager._versions["v1"].status == ModelStatus.FAILED

    @pytest.mark.asyncio
    async def test_load_multiple_versions(self):
        """Test loading multiple versions."""
        from src.core.model.hot_reload import HotReloadManager

        manager = HotReloadManager()

        with tempfile.NamedTemporaryFile() as f:
            await manager.load_model("v1", f.name)
            await manager.load_model("v2", f.name)

        assert manager._active_version == "v2"
        assert manager._previous_version == "v1"

    @pytest.mark.asyncio
    async def test_rollback(self):
        """Test rollback to previous version."""
        from src.core.model.hot_reload import HotReloadManager

        manager = HotReloadManager()

        with tempfile.NamedTemporaryFile() as f:
            await manager.load_model("v1", f.name)
            await manager.load_model("v2", f.name)

            result = await manager.rollback()

        assert result is True
        assert manager._active_version == "v1"

    @pytest.mark.asyncio
    async def test_rollback_no_previous(self):
        """Test rollback when no previous version."""
        from src.core.model.hot_reload import HotReloadManager

        manager = HotReloadManager()

        result = await manager.rollback()

        assert result is False

    @pytest.mark.asyncio
    async def test_unload_version(self):
        """Test unloading a version."""
        from src.core.model.hot_reload import HotReloadManager, ModelStatus

        manager = HotReloadManager()

        with tempfile.NamedTemporaryFile() as f:
            await manager.load_model("v1", f.name)
            await manager.load_model("v2", f.name)

            # Unload v1 (not active)
            result = await manager.unload_version("v1")

        assert result is True
        assert manager._versions["v1"].status == ModelStatus.RETIRED

    @pytest.mark.asyncio
    async def test_unload_active_version_fails(self):
        """Test cannot unload active version."""
        from src.core.model.hot_reload import HotReloadManager

        manager = HotReloadManager()

        with tempfile.NamedTemporaryFile() as f:
            await manager.load_model("v1", f.name)

            result = await manager.unload_version("v1")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_active_model(self):
        """Test getting active model."""
        from src.core.model.hot_reload import HotReloadManager

        manager = HotReloadManager()

        # No active model
        assert manager.get_active_model() is None

        with tempfile.NamedTemporaryFile() as f:
            await manager.load_model("v1", f.name)

        model = manager.get_active_model()
        assert model is not None
        assert model["loaded"] is True

    @pytest.mark.asyncio
    async def test_get_version_info(self):
        """Test getting version info."""
        from src.core.model.hot_reload import HotReloadManager

        manager = HotReloadManager()

        with tempfile.NamedTemporaryFile() as f:
            await manager.load_model("v1", f.name)

        info = manager.get_version_info("v1")

        assert info is not None
        assert info["version"] == "v1"
        assert info["status"] == "ready"

    @pytest.mark.asyncio
    async def test_list_versions(self):
        """Test listing versions."""
        from src.core.model.hot_reload import HotReloadManager

        manager = HotReloadManager()

        with tempfile.NamedTemporaryFile() as f:
            await manager.load_model("v1", f.name)
            await manager.load_model("v2", f.name)

        versions = manager.list_versions()

        assert len(versions) == 2
        version_names = [v["version"] for v in versions]
        assert "v1" in version_names
        assert "v2" in version_names

    @pytest.mark.asyncio
    async def test_record_request(self):
        """Test recording requests."""
        from src.core.model.hot_reload import HotReloadManager

        manager = HotReloadManager()

        with tempfile.NamedTemporaryFile() as f:
            await manager.load_model("v1", f.name)

        manager.record_request(latency_ms=10.0)
        manager.record_request(latency_ms=20.0)
        manager.record_request(latency_ms=30.0, error=True)

        version = manager._versions["v1"]
        assert version.request_count == 3
        assert version.error_count == 1
        assert version.avg_latency_ms > 0

    @pytest.mark.asyncio
    async def test_reload_callback(self):
        """Test reload callback is called."""
        from src.core.model.hot_reload import HotReloadManager

        manager = HotReloadManager()
        callback_called = []

        async def callback(version):
            callback_called.append(version)

        manager.add_reload_callback(callback)

        with tempfile.NamedTemporaryFile() as f:
            await manager.load_model("v1", f.name)

        assert "v1" in callback_called

    @pytest.mark.asyncio
    async def test_get_metrics(self):
        """Test getting metrics."""
        from src.core.model.hot_reload import HotReloadManager

        manager = HotReloadManager()

        with tempfile.NamedTemporaryFile() as f:
            await manager.load_model("v1", f.name)

        metrics = manager.get_metrics()

        assert metrics["reload_count"] == 1
        assert metrics["active_version"] == "v1"
        assert metrics["total_versions"] == 1


class TestGetHotReloadManager:
    """Tests for get_hot_reload_manager function."""

    @pytest.mark.asyncio
    async def test_returns_singleton(self):
        """Test returns singleton instance."""
        from src.core.model import hot_reload as hr_module

        # Reset global
        hr_module._hot_reload_manager = None

        mgr1 = hr_module.get_hot_reload_manager()
        mgr2 = hr_module.get_hot_reload_manager()

        assert mgr1 is mgr2

        # Cleanup
        hr_module._hot_reload_manager = None
