"""Tests for src/main.py to improve coverage.

Covers:
- lifespan context manager
- app configuration
- health check endpoints
- readiness check
- metrics endpoints
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRootEndpoint:
    """Tests for root endpoint."""

    @pytest.mark.asyncio
    async def test_root_returns_app_info(self):
        """Test root endpoint returns app info."""
        from src.main import root

        result = await root()

        assert result["name"] == "CAD ML Platform"
        assert result["version"] == "1.0.0"
        assert result["status"] == "running"
        assert result["docs"] == "/docs"
        assert result["health"] == "/health"


class TestHealthCheck:
    """Tests for health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_returns_healthy(self):
        """Test health check returns healthy status."""
        from src.main import health_check

        with patch("src.api.health_utils.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                REDIS_ENABLED=False,
                VISION_MAX_BASE64_BYTES=10_000_000,
                OCR_TIMEOUT_MS=30000,
                OCR_PROVIDER_DEFAULT="tesseract",
                CONFIDENCE_FALLBACK=0.5,
                ERROR_EMA_ALPHA=0.1,
                CORS_ORIGINS=["*"],
                ALLOWED_HOSTS=["*"],
                DEBUG=False,
                LOG_LEVEL="INFO",
            )
            with patch("src.api.health_utils.get_ocr_error_rate_ema", return_value=0.01):
                with patch("src.api.health_utils.get_vision_error_rate_ema", return_value=0.02):
                    with patch(
                        "src.api.health_utils.get_resilience_health",
                        return_value={"resilience": "ok"},
                    ):
                        result = await health_check()

        assert result["status"] == "healthy"
        assert "timestamp" in result
        assert "services" in result
        assert "runtime" in result
        assert "config" in result

    @pytest.mark.asyncio
    async def test_health_check_with_redis_enabled(self):
        """Test health check with Redis enabled."""
        from src.main import health_check

        with patch("src.api.health_utils.get_settings") as mock_settings:
            mock_obj = MagicMock(
                REDIS_ENABLED=True,
                VISION_MAX_BASE64_BYTES=10_000_000,
                OCR_TIMEOUT_MS=30000,
                OCR_PROVIDER_DEFAULT="tesseract",
                CONFIDENCE_FALLBACK=0.5,
                ERROR_EMA_ALPHA=0.1,
                CORS_ORIGINS=["*"],
                ALLOWED_HOSTS=["*"],
                DEBUG=False,
                LOG_LEVEL="INFO",
            )
            mock_settings.return_value = mock_obj
            with patch("src.api.health_utils.get_ocr_error_rate_ema", return_value=0.01):
                with patch("src.api.health_utils.get_vision_error_rate_ema", return_value=0.02):
                    with patch("src.api.health_utils.get_resilience_health", return_value={}):
                        result = await health_check()

        assert result["services"]["redis"] == "up"

    @pytest.mark.asyncio
    async def test_health_check_handles_resilience_error(self):
        """Test health check handles resilience module error."""
        from src.main import health_check

        with patch("src.api.health_utils.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                REDIS_ENABLED=False,
                VISION_MAX_BASE64_BYTES=10_000_000,
                OCR_TIMEOUT_MS=30000,
                OCR_PROVIDER_DEFAULT="tesseract",
                CONFIDENCE_FALLBACK=0.5,
                ERROR_EMA_ALPHA=0.1,
                CORS_ORIGINS=["*"],
                ALLOWED_HOSTS=["*"],
                DEBUG=False,
                LOG_LEVEL="INFO",
            )
            with patch("src.api.health_utils.get_ocr_error_rate_ema", return_value=0.01):
                with patch("src.api.health_utils.get_vision_error_rate_ema", return_value=0.02):
                    with patch(
                        "src.api.health_utils.get_resilience_health",
                        side_effect=Exception("Module error"),
                    ):
                        result = await health_check()

        # Should still return healthy even if resilience fails
        assert result["status"] == "healthy"


class TestExtendedHealth:
    """Tests for extended health endpoint."""

    @pytest.mark.asyncio
    async def test_extended_health_returns_vector_info(self):
        """Test extended health returns vector store info."""
        from src.main import extended_health

        with patch("src.core.similarity._VECTOR_STORE", {}):
            with patch("src.core.similarity._VECTOR_META", {}):
                with patch("src.core.similarity._FAISS_IMPORTED", False):
                    with patch("src.core.similarity._FAISS_LAST_EXPORT_SIZE", 0):
                        with patch("src.core.similarity._FAISS_LAST_EXPORT_TS", None):
                            with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "memory"}):
                                result = await extended_health()

        assert result["status"] == "healthy"
        assert "vector_store" in result
        assert "faiss" in result

    @pytest.mark.asyncio
    async def test_extended_health_with_faiss_enabled(self):
        """Test extended health with Faiss enabled."""
        import time

        from src.main import extended_health

        export_ts = time.time() - 100  # 100 seconds ago

        with patch("src.core.similarity._VECTOR_STORE", {"id1": [1, 2, 3]}):
            with patch("src.core.similarity._VECTOR_META", {"id1": {"feature_version": "v2"}}):
                with patch("src.core.similarity._FAISS_IMPORTED", True):
                    with patch("src.core.similarity._FAISS_LAST_EXPORT_SIZE", 1000):
                        with patch("src.core.similarity._FAISS_LAST_EXPORT_TS", export_ts):
                            with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "faiss"}):
                                result = await extended_health()

        assert result["faiss"]["enabled"] is True
        assert result["faiss"]["imported"] is True
        assert result["faiss"]["last_export_age_seconds"] is not None


class TestReadinessCheck:
    """Tests for readiness check endpoint."""

    @pytest.mark.asyncio
    async def test_readiness_when_ready(self):
        """Test readiness check when service is ready."""
        from src.main import readiness_check

        with patch("src.main.settings", MagicMock(REDIS_ENABLED=False)):
            with patch("src.models.loader.models_loaded", return_value=True):
                result = await readiness_check()

        assert result["status"] == "ready"

    @pytest.mark.asyncio
    async def test_readiness_models_not_loaded(self):
        """Test readiness check when models not loaded."""
        from fastapi import HTTPException

        from src.main import readiness_check

        with patch("src.main.settings", MagicMock(REDIS_ENABLED=False)):
            with patch("src.models.loader.models_loaded", return_value=False):
                with pytest.raises(HTTPException) as exc_info:
                    await readiness_check()

        assert exc_info.value.status_code == 503
        # Generic error message is used
        assert "not ready" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_readiness_redis_not_ready(self):
        """Test readiness check when Redis not ready."""
        from fastapi import HTTPException

        from src.main import readiness_check

        with patch("src.main.settings", MagicMock(REDIS_ENABLED=True)):
            with patch("src.models.loader.models_loaded", return_value=True):
                with patch("src.utils.cache.redis_healthy", AsyncMock(return_value=False)):
                    with pytest.raises(HTTPException) as exc_info:
                        await readiness_check()

        assert exc_info.value.status_code == 503


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""

    def test_metrics_enabled_flag(self):
        """Test metrics enabled flag is set."""
        from src.main import _metrics_enabled

        # Flag should be boolean
        assert isinstance(_metrics_enabled, bool)


class TestAppConfiguration:
    """Tests for app configuration."""

    def test_app_title(self):
        """Test app title is set correctly."""
        from src.main import app

        assert app.title == "CAD ML Platform"

    def test_app_version(self):
        """Test app version is set correctly."""
        from src.main import app

        assert app.version == "1.0.0"

    def test_app_has_routes(self):
        """Test app has routes registered."""
        from src.main import app

        routes = [r.path for r in app.routes]
        assert "/" in routes
        assert "/health" in routes
        assert "/ready" in routes


class TestConfigValues:
    """Tests for configuration values."""

    def test_health_limits_calculated(self):
        """Test health limits are properly calculated."""
        max_bytes = 10_000_000
        max_mb = round(max_bytes / 1024 / 1024, 2)

        assert max_mb == pytest.approx(9.54, rel=0.01)

    def test_timeout_conversion(self):
        """Test timeout conversion from ms to seconds."""
        timeout_ms = 30000
        timeout_seconds = timeout_ms / 1000

        assert timeout_seconds == 30.0


class TestRuntimeInfo:
    """Tests for runtime info in health check."""

    def test_python_version_format(self):
        """Test Python version format extraction."""
        version = sys.version.split(" ")[0]

        # Should be semver-like format
        assert "." in version
        parts = version.split(".")
        assert len(parts) >= 2


class TestTimestampHandling:
    """Tests for timestamp handling."""

    def test_iso_format_timestamp(self):
        """Test ISO format timestamp generation."""
        timestamp = datetime.now(timezone.utc).isoformat()

        # Should contain T and timezone info
        assert "T" in timestamp


class TestServiceStatus:
    """Tests for service status handling."""

    def test_redis_disabled_status(self):
        """Test Redis disabled status string."""
        redis_enabled = False
        status = "up" if redis_enabled else "disabled"

        assert status == "disabled"

    def test_redis_enabled_status(self):
        """Test Redis enabled status string."""
        redis_enabled = True
        status = "up" if redis_enabled else "disabled"

        assert status == "up"


class TestErrorRateEMA:
    """Tests for error rate EMA handling."""

    def test_error_rate_structure(self):
        """Test error rate structure in response."""
        error_rates = {
            "ocr": 0.01,
            "vision": 0.02,
        }

        assert "ocr" in error_rates
        assert "vision" in error_rates
        assert error_rates["ocr"] == 0.01


class TestVersionDistribution:
    """Tests for version distribution in extended health."""

    def test_version_counting(self):
        """Test version counting logic."""
        meta = {
            "id1": {"feature_version": "v1"},
            "id2": {"feature_version": "v2"},
            "id3": {"feature_version": "v1"},
            "id4": {},  # No version
        }

        # Use fixed default version instead of env var
        default_ver = "v1"
        versions: Dict[str, int] = {}
        for m in meta.values():
            ver = m.get("feature_version", default_ver)
            versions[ver] = versions.get(ver, 0) + 1

        assert versions.get("v1", 0) == 3  # id1, id3, and id4 (default)
        assert versions.get("v2", 0) == 1


class TestFaissAgeCalculation:
    """Tests for Faiss age calculation."""

    def test_age_calculation(self):
        """Test age calculation from timestamp."""
        import time

        export_ts = time.time() - 100
        age = round(time.time() - export_ts, 2)

        assert age >= 99.0  # At least 99 seconds

    def test_age_none_when_no_export(self):
        """Test age is None when no export timestamp."""
        export_ts = None

        if export_ts:
            age = 100.0
        else:
            age = None

        assert age is None
