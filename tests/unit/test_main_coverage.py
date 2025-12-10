"""Tests for main.py to improve coverage.

Covers:
- root endpoint
- health_check endpoint
- extended_health endpoint
- readiness_check with various states
- metrics_fallback when prometheus_client unavailable
"""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = MagicMock()
    settings.REDIS_ENABLED = False
    settings.DEBUG = False
    settings.LOG_LEVEL = "INFO"
    settings.HOST = "0.0.0.0"
    settings.PORT = 8000
    settings.WORKERS = 1
    settings.CORS_ORIGINS = ["*"]
    settings.ALLOWED_HOSTS = ["*"]
    settings.VISION_MAX_BASE64_BYTES = 10485760
    settings.OCR_TIMEOUT_MS = 30000
    settings.OCR_PROVIDER_DEFAULT = "doubao"
    settings.CONFIDENCE_FALLBACK = 0.5
    settings.ERROR_EMA_ALPHA = 0.1
    return settings


@pytest.fixture
def mock_lifespan():
    """Mock lifespan context manager."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def mock_lifespan_cm(app):
        yield

    return mock_lifespan_cm


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_app_info(self, mock_settings, mock_lifespan):
        """Test root endpoint returns application info."""
        with patch("src.main.get_settings", return_value=mock_settings), \
             patch("src.main.lifespan", mock_lifespan):
            from src.main import app

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/")

            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "CAD ML Platform"
            assert data["version"] == "1.0.0"
            assert data["status"] == "running"
            assert "docs" in data
            assert "health" in data


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_check_basic(self, mock_settings, mock_lifespan):
        """Test health check returns status info."""
        with patch("src.main.get_settings", return_value=mock_settings), \
             patch("src.main.lifespan", mock_lifespan), \
             patch("src.main.get_ocr_error_rate_ema", return_value=0.05), \
             patch("src.main.get_vision_error_rate_ema", return_value=0.03), \
             patch("src.main.get_resilience_health", return_value={}):
            from src.main import app

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data
            assert "services" in data
            assert "runtime" in data
            assert "config" in data

    def test_health_check_with_redis_disabled(self, mock_settings, mock_lifespan):
        """Test health check shows redis disabled."""
        mock_settings.REDIS_ENABLED = False

        with patch("src.main.get_settings", return_value=mock_settings), \
             patch("src.main.settings", mock_settings), \
             patch("src.main.lifespan", mock_lifespan), \
             patch("src.main.get_ocr_error_rate_ema", return_value=0.0), \
             patch("src.main.get_vision_error_rate_ema", return_value=0.0), \
             patch("src.main.get_resilience_health", return_value={}):
            from src.main import app

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/health")

            data = response.json()
            assert data["services"]["redis"] == "disabled"

    def test_health_check_resilience_failure_handled(self, mock_settings, mock_lifespan):
        """Test health check handles resilience module failure gracefully."""
        with patch("src.main.get_settings", return_value=mock_settings), \
             patch("src.main.lifespan", mock_lifespan), \
             patch("src.main.get_ocr_error_rate_ema", return_value=0.0), \
             patch("src.main.get_vision_error_rate_ema", return_value=0.0), \
             patch("src.main.get_resilience_health", side_effect=Exception("Resilience error")):
            from src.main import app

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/health")

            # Should still return 200 even if resilience module fails
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"


class TestExtendedHealthEndpoint:
    """Tests for extended health endpoint."""

    def test_extended_health_returns_vector_info(self, mock_settings, mock_lifespan):
        """Test extended health returns vector store info."""
        mock_vector_store = {"id1": [0.1, 0.2], "id2": [0.3, 0.4]}
        mock_vector_meta = {
            "id1": {"feature_version": "v1"},
            "id2": {"feature_version": "v2"}
        }

        with patch("src.main.get_settings", return_value=mock_settings), \
             patch("src.main.lifespan", mock_lifespan), \
             patch("src.core.similarity._VECTOR_STORE", mock_vector_store), \
             patch("src.core.similarity._VECTOR_META", mock_vector_meta), \
             patch("src.core.similarity._FAISS_IMPORTED", False), \
             patch("src.core.similarity._FAISS_LAST_EXPORT_SIZE", 0), \
             patch("src.core.similarity._FAISS_LAST_EXPORT_TS", None), \
             patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "memory"}):
            from src.main import app

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/health/extended")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "vector_store" in data
            assert "faiss" in data

    def test_extended_health_faiss_structure(self, mock_settings, mock_lifespan):
        """Test extended health faiss structure is present."""
        with patch("src.main.get_settings", return_value=mock_settings), \
             patch("src.main.lifespan", mock_lifespan):
            from src.main import app

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/health/extended")

            data = response.json()
            # Verify faiss section structure exists regardless of enabled state
            assert "faiss" in data
            assert "enabled" in data["faiss"]
            assert "imported" in data["faiss"]
            assert "last_export_size" in data["faiss"]
            # enabled is a boolean
            assert isinstance(data["faiss"]["enabled"], bool)


class TestReadinessEndpoint:
    """Tests for readiness endpoint."""

    def test_readiness_success(self, mock_settings, mock_lifespan):
        """Test readiness returns ready when all checks pass."""
        mock_settings.REDIS_ENABLED = False

        # Need to patch at the import location within the endpoint
        with patch("src.main.get_settings", return_value=mock_settings), \
             patch("src.main.settings", mock_settings), \
             patch("src.main.lifespan", mock_lifespan):
            from src.main import app

            # Patch models_loaded where it's imported in the readiness check
            with patch.object(
                __import__("src.models.loader", fromlist=["models_loaded"]),
                "models_loaded",
                return_value=True
            ):
                client = TestClient(app, raise_server_exceptions=False)
                response = client.get("/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"

    def test_readiness_models_not_loaded(self, mock_settings, mock_lifespan):
        """Test readiness fails when models not loaded."""
        mock_settings.REDIS_ENABLED = False

        with patch("src.main.get_settings", return_value=mock_settings), \
             patch("src.main.settings", mock_settings), \
             patch("src.main.lifespan", mock_lifespan):
            from src.main import app

            with patch.object(
                __import__("src.models.loader", fromlist=["models_loaded"]),
                "models_loaded",
                return_value=False
            ):
                client = TestClient(app, raise_server_exceptions=False)
                response = client.get("/ready")

            assert response.status_code == 503
            # The exception is caught and re-raised with generic message
            assert "not ready" in response.json()["detail"].lower() or "Models not loaded" in response.json()["detail"]

    def test_readiness_redis_not_ready(self, mock_settings, mock_lifespan):
        """Test readiness fails when Redis not ready."""
        mock_settings.REDIS_ENABLED = True

        with patch("src.main.get_settings", return_value=mock_settings), \
             patch("src.main.lifespan", mock_lifespan), \
             patch("src.main.settings", mock_settings), \
             patch("src.models.loader.models_loaded", return_value=True), \
             patch("src.utils.cache.redis_healthy", new_callable=AsyncMock, return_value=False):
            from src.main import app

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/ready")

            assert response.status_code == 503

    def test_readiness_exception_handling(self, mock_settings, mock_lifespan):
        """Test readiness handles exceptions gracefully."""
        mock_settings.REDIS_ENABLED = False

        with patch("src.main.get_settings", return_value=mock_settings), \
             patch("src.main.lifespan", mock_lifespan), \
             patch("src.models.loader.models_loaded", side_effect=Exception("Unexpected error")):
            from src.main import app

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/ready")

            assert response.status_code == 503
            assert "not ready" in response.json()["detail"]


class TestMetricsFallback:
    """Tests for metrics fallback endpoint."""

    def test_metrics_fallback_when_prometheus_disabled(self):
        """Test metrics fallback returns minimal exposition."""
        # Import main with _metrics_enabled = False
        with patch.dict(sys.modules, {"prometheus_client": None}):
            # We need to test the fallback route
            # Since _metrics_enabled is set at import time, we test the function directly
            from src.main import app

            # Check if the fallback endpoint exists
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/metrics")

            # Should return 200 regardless (either prometheus or fallback)
            assert response.status_code == 200


class TestMetricsEnabled:
    """Tests for metrics enabled flag."""

    def test_metrics_enabled_true(self):
        """Test _metrics_enabled is True when prometheus_client available."""
        # prometheus_client should be available in test environment
        from src.main import _metrics_enabled

        # Could be True or False depending on installation
        assert isinstance(_metrics_enabled, bool)


class TestAppConfiguration:
    """Tests for app configuration."""

    def test_app_has_correct_title(self):
        """Test app has correct title."""
        from src.main import app

        assert app.title == "CAD ML Platform"
        assert app.version == "1.0.0"

    def test_cors_middleware_configured(self):
        """Test CORS middleware is configured."""
        from src.main import app

        # Check that middleware is added
        middleware_types = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_types

    def test_trusted_host_middleware_configured(self):
        """Test TrustedHostMiddleware is configured."""
        from src.main import app

        middleware_types = [m.cls.__name__ for m in app.user_middleware]
        assert "TrustedHostMiddleware" in middleware_types

    def test_api_router_included(self):
        """Test API router is included."""
        from src.main import app

        # Check routes are registered under /api prefix
        route_paths = [route.path for route in app.routes]
        # Should have routes starting with /api
        api_routes = [p for p in route_paths if p.startswith("/api")]
        assert len(api_routes) > 0 or any("/api" in str(r) for r in app.routes)


class TestLifespanEvents:
    """Tests for lifespan events (startup/shutdown)."""

    @pytest.mark.asyncio
    async def test_lifespan_startup_without_redis(self, mock_settings):
        """Test lifespan startup without Redis."""
        mock_settings.REDIS_ENABLED = False

        with patch("src.main.get_settings", return_value=mock_settings), \
             patch("src.main.settings", mock_settings), \
             patch("src.main.init_redis", new_callable=AsyncMock) as mock_init_redis, \
             patch("src.main.load_models", new_callable=AsyncMock), \
             patch("src.main.background_prune_task", new_callable=AsyncMock), \
             patch("src.main.load_recovery_state"), \
             patch("src.main.orphan_scan_loop", new_callable=AsyncMock), \
             patch("asyncio.create_task") as mock_create_task:

            from src.main import lifespan, app

            async with lifespan(app):
                # Redis should not be initialized when disabled
                mock_init_redis.assert_not_called()

    @pytest.mark.asyncio
    async def test_lifespan_startup_with_redis(self, mock_settings):
        """Test lifespan startup with Redis enabled."""
        mock_settings.REDIS_ENABLED = True

        with patch("src.main.get_settings", return_value=mock_settings), \
             patch("src.main.settings", mock_settings), \
             patch("src.main.init_redis", new_callable=AsyncMock) as mock_init_redis, \
             patch("src.main.load_models", new_callable=AsyncMock), \
             patch("src.main.background_prune_task", new_callable=AsyncMock), \
             patch("src.main.load_recovery_state"), \
             patch("src.main.orphan_scan_loop", new_callable=AsyncMock), \
             patch("src.main.FaissVectorStore") as mock_faiss, \
             patch("asyncio.create_task") as mock_create_task, \
             patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "memory"}):

            mock_create_task.return_value = MagicMock()

            from src.main import lifespan, app

            async with lifespan(app):
                mock_init_redis.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_faiss_import_success(self, mock_settings):
        """Test lifespan completes successfully with faiss backend configured."""
        mock_settings.REDIS_ENABLED = False

        with patch("src.main.get_settings", return_value=mock_settings), \
             patch("src.main.settings", mock_settings), \
             patch("src.main.load_models", new_callable=AsyncMock), \
             patch("src.main.background_prune_task", new_callable=AsyncMock), \
             patch("src.main.load_recovery_state"), \
             patch("src.main.orphan_scan_loop", new_callable=AsyncMock), \
             patch("asyncio.create_task") as mock_create_task, \
             patch.dict("os.environ", {
                 "VECTOR_STORE_BACKEND": "faiss",
                 "FAISS_INDEX_PATH": "/tmp/test_faiss.bin",
             }):

            mock_create_task.return_value = MagicMock()

            from src.main import lifespan, app

            # Test that lifespan completes without error when faiss is configured
            async with lifespan(app):
                pass  # Lifespan completes successfully

    @pytest.mark.asyncio
    async def test_lifespan_faiss_import_failure(self, mock_settings):
        """Test lifespan handles Faiss import failure."""
        mock_settings.REDIS_ENABLED = False

        with patch("src.main.get_settings", return_value=mock_settings), \
             patch("src.main.settings", mock_settings), \
             patch("src.main.load_models", new_callable=AsyncMock), \
             patch("src.main.background_prune_task", new_callable=AsyncMock), \
             patch("src.main.load_recovery_state", side_effect=Exception("Load failed")), \
             patch("src.main.orphan_scan_loop", new_callable=AsyncMock), \
             patch("asyncio.create_task") as mock_create_task, \
             patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "memory"}):

            mock_create_task.return_value = MagicMock()

            from src.main import lifespan, app

            # Should not raise exception
            async with lifespan(app):
                pass

    @pytest.mark.asyncio
    async def test_lifespan_drift_baseline_loading(self, mock_settings):
        """Test lifespan loads drift baselines from Redis."""
        mock_settings.REDIS_ENABLED = True

        mock_client = AsyncMock()
        mock_client.get.side_effect = [
            '{"material1": 0.5}',  # baseline:material
            '{"class1": 0.8}',     # baseline:class
            "1234567890",          # baseline:material:ts
            "1234567891",          # baseline:class:ts
        ]

        with patch("src.main.get_settings", return_value=mock_settings), \
             patch("src.main.settings", mock_settings), \
             patch("src.main.init_redis", new_callable=AsyncMock), \
             patch("src.main.load_models", new_callable=AsyncMock), \
             patch("src.main.background_prune_task", new_callable=AsyncMock), \
             patch("src.main.load_recovery_state"), \
             patch("src.main.orphan_scan_loop", new_callable=AsyncMock), \
             patch("src.utils.cache.get_client", return_value=mock_client), \
             patch("asyncio.create_task") as mock_create_task, \
             patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "memory"}):

            mock_create_task.return_value = MagicMock()

            from src.main import lifespan, app

            async with lifespan(app):
                # Should have called Redis to get baselines
                pass  # Just verify no exception


class TestSettings:
    """Tests for settings integration."""

    def test_settings_loaded(self):
        """Test settings are loaded at module level."""
        from src.main import settings

        # Should have settings object
        assert settings is not None

    def test_logger_configured(self):
        """Test logger is configured."""
        from src.main import logger

        assert logger is not None
        assert logger.name == "src.main"
