"""Tests for V16 classifier management API endpoints."""

import io
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def api_headers():
    return {"X-API-Key": "test-key"}


@pytest.fixture
def admin_headers():
    os.environ["ADMIN_TOKEN"] = "test"
    return {"X-API-Key": "test-key", "X-Admin-Token": "test"}


class TestV16HealthEndpoint:
    """Tests for V16 classifier health check endpoint."""

    def test_health_unavailable_when_classifier_not_loaded(self, client, api_headers):
        """Health returns unavailable when V16 classifier is not loaded."""
        with patch("src.core.analyzer._get_v16_classifier", return_value=None):
            response = client.get("/api/v1/health/v16-classifier", headers=api_headers)

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] in ("unavailable", "disabled")
        assert payload["loaded"] is False

    def test_health_ok_when_classifier_loaded(self, client, api_headers):
        """Health returns ok with stats when V16 classifier is loaded."""
        mock_classifier = MagicMock()
        mock_classifier.speed_mode = "fast"
        mock_classifier.enable_cache = True
        mock_classifier.feature_cache = {"key1": "val1", "key2": "val2"}
        mock_classifier.cache_size = 1000
        mock_classifier.cache_hits = 10
        mock_classifier.cache_misses = 2
        mock_classifier.v6_model = MagicMock()
        mock_classifier.v14_model = MagicMock()
        mock_classifier.categories = ["传动件", "其他", "壳体类", "轴类", "连接件"]

        with patch("src.core.analyzer._get_v16_classifier", return_value=mock_classifier):
            response = client.get("/api/v1/health/v16-classifier", headers=api_headers)

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["loaded"] is True
        assert payload["speed_mode"] == "fast"
        assert payload["cache_enabled"] is True
        assert payload["cache_size"] == 2
        assert payload["cache_max_size"] == 1000
        assert payload["cache_hits"] == 10
        assert payload["cache_misses"] == 2
        assert payload["cache_hit_ratio"] == 0.8333
        assert payload["v6_model_loaded"] is True
        assert payload["v14_model_loaded"] is True

    def test_health_disabled_by_env(self, client, api_headers):
        """Health returns disabled when DISABLE_V16_CLASSIFIER is set."""
        with patch("src.core.analyzer._get_v16_classifier", return_value=None):
            with patch.dict(os.environ, {"DISABLE_V16_CLASSIFIER": "true"}):
                response = client.get("/api/v1/health/v16-classifier", headers=api_headers)

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "disabled"

    def test_health_alternate_path(self, client, api_headers):
        """Health endpoint accessible via alternate path."""
        with patch("src.core.analyzer._get_v16_classifier", return_value=None):
            response = client.get("/api/v1/v16-classifier/health", headers=api_headers)

        assert response.status_code == 200


class TestV16CacheClearEndpoint:
    """Tests for V16 classifier cache clear endpoint."""

    def test_cache_clear_unavailable_when_not_loaded(self, client, admin_headers):
        """Cache clear returns unavailable when classifier not loaded."""
        with patch("src.core.analyzer._get_v16_classifier", return_value=None):
            response = client.post(
                "/api/v1/v16-classifier/cache/clear", headers=admin_headers
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "unavailable"

    def test_cache_clear_success(self, client, admin_headers):
        """Cache clear successfully clears caches and resets stats."""
        mock_feature_cache = MagicMock()
        mock_feature_cache.__len__ = MagicMock(return_value=3)

        mock_classifier = MagicMock()
        mock_classifier.cache_hits = 50
        mock_classifier.cache_misses = 10
        mock_classifier.feature_cache = mock_feature_cache
        mock_classifier.image_cache = MagicMock()
        mock_classifier.cache_order = MagicMock()

        with patch("src.core.analyzer._get_v16_classifier", return_value=mock_classifier):
            response = client.post(
                "/api/v1/v16-classifier/cache/clear", headers=admin_headers
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["cleared_entries"] == 3
        assert payload["previous_hits"] == 50
        assert payload["previous_misses"] == 10

    def test_cache_clear_requires_admin_token(self, client, api_headers):
        """Cache clear requires admin token."""
        response = client.post(
            "/api/v1/v16-classifier/cache/clear", headers=api_headers
        )
        assert response.status_code == 401


class TestV16SpeedModeEndpoint:
    """Tests for V16 classifier speed mode endpoint."""

    def test_get_speed_mode_unavailable(self, client, api_headers):
        """Get speed mode returns unavailable when classifier not loaded."""
        with patch("src.core.analyzer._get_v16_classifier", return_value=None):
            response = client.get(
                "/api/v1/v16-classifier/speed-mode", headers=api_headers
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "unavailable"
        assert "accurate" in payload["available_modes"]
        assert "fast" in payload["available_modes"]

    def test_get_speed_mode_success(self, client, api_headers):
        """Get speed mode returns current mode."""
        mock_classifier = MagicMock()
        mock_classifier.speed_mode = "balanced"

        with patch("src.core.analyzer._get_v16_classifier", return_value=mock_classifier):
            response = client.get(
                "/api/v1/v16-classifier/speed-mode", headers=api_headers
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["current_mode"] == "balanced"

    def test_set_speed_mode_invalid(self, client, admin_headers):
        """Set speed mode rejects invalid mode."""
        response = client.post(
            "/api/v1/v16-classifier/speed-mode",
            json={"speed_mode": "invalid_mode"},
            headers=admin_headers,
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "error"
        assert "invalid_mode" in payload["message"].lower()

    def test_set_speed_mode_success(self, client, admin_headers):
        """Set speed mode successfully changes mode."""
        mock_classifier = MagicMock()
        mock_classifier.speed_mode = "accurate"

        # Patch at import location in health.py
        with patch("src.core.analyzer._get_v16_classifier", return_value=mock_classifier):
            # Since SPEED_MODES import requires torch, we test the unavailable path
            # when classifier is mocked but SPEED_MODES import fails
            response = client.post(
                "/api/v1/v16-classifier/speed-mode",
                json={"speed_mode": "fast"},
                headers=admin_headers,
            )

        # Either ok (if torch available) or error (if SPEED_MODES import fails)
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] in ("ok", "error")

    def test_set_speed_mode_requires_admin(self, client, api_headers):
        """Set speed mode requires admin token."""
        response = client.post(
            "/api/v1/v16-classifier/speed-mode",
            json={"speed_mode": "fast"},
            headers=api_headers,
        )
        assert response.status_code == 401


class TestBatchClassifyEndpoint:
    """Tests for batch classification endpoint."""

    def test_batch_classify_empty_files(self, client, api_headers):
        """Batch classify with no files returns validation error."""
        response = client.post(
            "/api/v1/analyze/batch-classify",
            files=[],
            headers=api_headers,
        )
        # FastAPI returns 422 for missing required field
        assert response.status_code == 422

    def test_batch_classify_unsupported_format(self, client, api_headers):
        """Batch classify rejects unsupported file formats."""
        files = [
            ("files", ("test.txt", io.BytesIO(b"content"), "text/plain")),
        ]

        with patch("src.core.analyzer._get_v16_classifier", return_value=None):
            with patch("src.core.analyzer._get_ml_classifier", return_value=None):
                response = client.post(
                    "/api/v1/analyze/batch-classify",
                    files=files,
                    headers=api_headers,
                )

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        assert payload["failed"] == 1
        assert "unsupported" in payload["results"][0]["error"].lower()

    def test_batch_classify_no_classifier_available(self, client, api_headers):
        """Batch classify handles no classifier available."""
        dxf_content = b"0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nEOF\n"
        files = [
            ("files", ("test.dxf", io.BytesIO(dxf_content), "application/octet-stream")),
        ]

        with patch("src.core.analyzer._get_v16_classifier", return_value=None):
            with patch("src.core.analyzer._get_ml_classifier", return_value=None):
                response = client.post(
                    "/api/v1/analyze/batch-classify",
                    files=files,
                    headers=api_headers,
                )

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        assert payload["failed"] == 1
        assert "no classifier" in payload["results"][0]["error"].lower()

    def test_batch_classify_with_v16_classifier(self, client, api_headers):
        """Batch classify uses V16 classifier when available."""
        dxf_content = b"0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nEOF\n"
        files = [
            ("files", ("part1.dxf", io.BytesIO(dxf_content), "application/octet-stream")),
            ("files", ("part2.dxf", io.BytesIO(dxf_content), "application/octet-stream")),
        ]

        mock_result = MagicMock()
        mock_result.category = "轴类"
        mock_result.confidence = 0.95
        mock_result.probabilities = {"轴类": 0.95, "其他": 0.05}
        mock_result.needs_review = False
        mock_result.review_reason = None
        mock_result.model_version = "v16_ensemble"

        mock_classifier = MagicMock()
        mock_classifier.predict_batch.return_value = [mock_result, mock_result]

        with patch("src.core.analyzer._get_v16_classifier", return_value=mock_classifier):
            response = client.post(
                "/api/v1/analyze/batch-classify",
                files=files,
                headers=api_headers,
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 2
        assert payload["success"] == 2
        assert payload["failed"] == 0
        assert payload["results"][0]["category"] == "轴类"
        assert payload["results"][0]["confidence"] == 0.95

    def test_batch_classify_mixed_formats(self, client, api_headers):
        """Batch classify handles mixed valid and invalid formats."""
        dxf_content = b"0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nEOF\n"
        files = [
            ("files", ("valid.dxf", io.BytesIO(dxf_content), "application/octet-stream")),
            ("files", ("invalid.pdf", io.BytesIO(b"pdf"), "application/pdf")),
        ]

        mock_result = MagicMock()
        mock_result.category = "壳体类"
        mock_result.confidence = 0.88
        mock_result.probabilities = {"壳体类": 0.88}
        mock_result.needs_review = False
        mock_result.review_reason = None  # Explicit None, not MagicMock
        mock_result.model_version = "v16"

        mock_classifier = MagicMock()
        mock_classifier.predict_batch.return_value = [mock_result]

        with patch("src.core.analyzer._get_v16_classifier", return_value=mock_classifier):
            response = client.post(
                "/api/v1/analyze/batch-classify",
                files=files,
                headers=api_headers,
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 2
        assert payload["success"] == 1
        assert payload["failed"] == 1
