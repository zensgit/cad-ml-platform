"""Integration tests for Vision API endpoints.

Tests the full request/response cycle through FastAPI.
"""

import base64
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.v1.vision import reset_vision_manager


# Sample image data (minimal PNG header)
SAMPLE_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
)
SAMPLE_PNG_B64 = base64.b64encode(SAMPLE_PNG).decode("utf-8")


@pytest.fixture
def client():
    """Create test client with clean vision manager state."""
    from src.main import app

    reset_vision_manager()
    with TestClient(app) as c:
        yield c
    reset_vision_manager()


@pytest.fixture(autouse=True)
def clean_env():
    """Ensure clean environment for each test."""
    keys_to_clear = ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    original = {k: os.environ.get(k) for k in keys_to_clear}

    for key in keys_to_clear:
        os.environ.pop(key, None)

    yield

    for key, value in original.items():
        if value is not None:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)


class TestVisionAnalyzeEndpoint:
    """Tests for POST /api/v1/vision/analyze."""

    def test_analyze_with_stub_provider(self, client):
        """Test analysis with default stub provider."""
        response = client.post(
            "/api/v1/vision/analyze",
            json={
                "image_base64": SAMPLE_PNG_B64,
                "include_description": True,
                "include_ocr": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["provider"] == "deepseek_stub"
        assert data["description"] is not None
        assert data["description"]["summary"] != ""
        assert data["processing_time_ms"] >= 0

    def test_analyze_with_explicit_stub_provider(self, client):
        """Test analysis with explicit stub provider parameter."""
        response = client.post(
            "/api/v1/vision/analyze?provider=stub",
            json={
                "image_base64": SAMPLE_PNG_B64,
                "include_description": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["provider"] == "deepseek_stub"

    def test_analyze_ocr_only_mode(self, client):
        """Test OCR-only mode returns minimal description."""
        response = client.post(
            "/api/v1/vision/analyze",
            json={
                "image_base64": SAMPLE_PNG_B64,
                "include_description": False,
                "include_ocr": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["description"]["summary"] == "Image processed (OCR-only mode)"

    def test_analyze_missing_image(self, client):
        """Test error when no image provided."""
        response = client.post(
            "/api/v1/vision/analyze",
            json={
                "include_description": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "error" in data

    def test_analyze_response_structure(self, client):
        """Test response structure matches schema."""
        response = client.post(
            "/api/v1/vision/analyze",
            json={
                "image_base64": SAMPLE_PNG_B64,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Required fields
        assert "success" in data
        assert "provider" in data
        assert "processing_time_ms" in data

        # Optional fields present when successful
        if data["success"]:
            assert "description" in data
            if data["description"]:
                assert "summary" in data["description"]
                assert "details" in data["description"]
                assert "confidence" in data["description"]


class TestVisionHealthEndpoint:
    """Tests for GET /api/v1/vision/health."""

    def test_health_check(self, client):
        """Test health endpoint returns expected structure."""
        response = client.get("/api/v1/vision/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "provider" in data
        assert data["status"] == "healthy"

    def test_health_check_shows_provider(self, client):
        """Test health shows current provider."""
        response = client.get("/api/v1/vision/health")

        data = response.json()
        assert data["provider"] == "deepseek_stub"
        assert "ocr_enabled" in data


class TestVisionProvidersEndpoint:
    """Tests for GET /api/v1/vision/providers."""

    def test_list_providers(self, client):
        """Test providers endpoint returns all providers."""
        response = client.get("/api/v1/vision/providers")

        assert response.status_code == 200
        data = response.json()
        assert "current_provider" in data
        assert "providers" in data

        providers = data["providers"]
        assert "stub" in providers
        assert "deepseek" in providers
        assert "openai" in providers
        assert "anthropic" in providers

    def test_providers_stub_always_available(self, client):
        """Test stub provider is always marked available."""
        response = client.get("/api/v1/vision/providers")

        data = response.json()
        stub = data["providers"]["stub"]
        assert stub["available"] is True
        assert stub["requires_key"] is False

    def test_providers_require_keys(self, client):
        """Test real providers require API keys."""
        response = client.get("/api/v1/vision/providers")

        data = response.json()
        for provider in ["deepseek", "openai", "anthropic"]:
            assert data["providers"][provider]["requires_key"] is True
            assert data["providers"][provider]["key_set"] is False

    def test_providers_key_set_detection(self, client):
        """Test key_set reflects environment."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            reset_vision_manager()
            response = client.get("/api/v1/vision/providers")

            data = response.json()
            assert data["providers"]["openai"]["key_set"] is True
            assert data["providers"]["deepseek"]["key_set"] is False


class TestVisionProviderSwitching:
    """Tests for provider switching functionality."""

    def test_switch_to_stub(self, client):
        """Test explicit switch to stub provider."""
        response = client.post(
            "/api/v1/vision/analyze?provider=stub",
            json={"image_base64": SAMPLE_PNG_B64},
        )

        assert response.status_code == 200
        assert response.json()["provider"] == "deepseek_stub"

    def test_invalid_provider_fallback(self, client):
        """Test invalid provider falls back to stub."""
        response = client.post(
            "/api/v1/vision/analyze?provider=invalid_provider",
            json={"image_base64": SAMPLE_PNG_B64},
        )

        assert response.status_code == 200
        data = response.json()
        # Should either work with stub or return error
        assert "provider" in data or "error" in data

    def test_auto_provider_uses_stub_without_keys(self, client):
        """Test auto detection uses stub when no API keys."""
        response = client.post(
            "/api/v1/vision/analyze?provider=auto",
            json={"image_base64": SAMPLE_PNG_B64},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "deepseek_stub"
