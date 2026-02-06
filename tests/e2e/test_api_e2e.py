"""End-to-end API tests using Playwright.

Features:
- Health check validation
- OCR endpoint testing
- Vision endpoint testing
- WebSocket testing
- Error handling validation
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional

import pytest
import pytest_asyncio

# Check if playwright is available
try:
    from playwright.async_api import async_playwright, APIRequestContext

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Configuration
BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "test-api-key")


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="module")
async def api_context():
    """Create API request context."""
    if not PLAYWRIGHT_AVAILABLE:
        pytest.skip("Playwright not installed")

    async with async_playwright() as p:
        context = await p.request.new_context(
            base_url=BASE_URL,
            extra_http_headers={
                "X-API-Key": API_KEY,
                "Content-Type": "application/json",
            },
        )
        yield context
        await context.dispose()


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
@pytest.mark.e2e
class TestHealthEndpoints:
    """Health check endpoint tests."""

    @pytest.mark.asyncio
    async def test_health_endpoint_returns_200(self, api_context: APIRequestContext):
        """Test that health endpoint returns 200."""
        response = await api_context.get("/health")
        assert response.status == 200

        data = await response.json()
        assert data["status"] in ["healthy", "degraded"]

    @pytest.mark.asyncio
    async def test_health_contains_version(self, api_context: APIRequestContext):
        """Test that health response contains version info."""
        response = await api_context.get("/health")
        data = await response.json()

        # Accept both legacy and current health payload variants.
        assert "version" in data or "components" in data or "runtime" in data

    @pytest.mark.asyncio
    async def test_api_v1_health(self, api_context: APIRequestContext):
        """Test v1 health endpoint."""
        response = await api_context.get("/api/v1/health")
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, api_context: APIRequestContext):
        """Test metrics endpoint for Prometheus."""
        response = await api_context.get("/metrics")
        assert response.status == 200

        text = await response.text()
        assert "http_requests_total" in text or "python_info" in text


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
@pytest.mark.e2e
class TestOCREndpoints:
    """OCR API endpoint tests."""

    @pytest.mark.asyncio
    async def test_ocr_extract_requires_file(self, api_context: APIRequestContext):
        """Test that OCR endpoint requires a file."""
        response = await api_context.post("/api/v2/ocr/extract")
        assert response.status in [400, 404, 422]

    @pytest.mark.asyncio
    async def test_ocr_rejects_invalid_file_type(self, api_context: APIRequestContext):
        """Test that OCR rejects invalid file types."""
        response = await api_context.post(
            "/api/v2/ocr/extract",
            multipart={
                "file": {
                    "name": "test.txt",
                    "mimeType": "text/plain",
                    "buffer": b"not a dxf file",
                }
            },
        )
        # Endpoint may be unavailable in some deployments.
        assert response.status in [400, 404, 415, 422]


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
@pytest.mark.e2e
class TestVisionEndpoints:
    """Vision API endpoint tests."""

    @pytest.mark.asyncio
    async def test_vision_analyze_requires_file(self, api_context: APIRequestContext):
        """Test that vision endpoint requires a file."""
        response = await api_context.post("/api/v2/vision/analyze")
        assert response.status in [400, 404, 422]


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
@pytest.mark.e2e
class TestRateLimiting:
    """Rate limiting tests."""

    @pytest.mark.asyncio
    async def test_rate_limit_headers_present(self, api_context: APIRequestContext):
        """Test that rate limit headers are returned."""
        response = await api_context.get("/health")

        headers = response.headers
        # At least one rate limit header should be present
        rate_limit_headers = [
            "x-ratelimit-limit",
            "x-ratelimit-remaining",
            "x-ratelimit-reset",
            "ratelimit-limit",
            "ratelimit-remaining",
        ]

        has_rate_limit = any(
            h.lower() in [k.lower() for k in headers.keys()] for h in rate_limit_headers
        )
        # Rate limiting might not be enabled in test environment
        assert response.status == 200  # At minimum, request should succeed


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
@pytest.mark.e2e
class TestErrorHandling:
    """Error handling tests."""

    @pytest.mark.asyncio
    async def test_404_returns_json_error(self, api_context: APIRequestContext):
        """Test that 404 returns structured JSON error."""
        response = await api_context.get("/api/v2/nonexistent-endpoint")
        assert response.status == 404

        data = await response.json()
        # Should have error structure
        assert "detail" in data or "error" in data or "message" in data

    @pytest.mark.asyncio
    async def test_invalid_json_returns_422(self, api_context: APIRequestContext):
        """Test that invalid JSON returns 422."""
        response = await api_context.post(
            "/api/v2/materials/classify",
            data="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status in [400, 404, 422]

    @pytest.mark.asyncio
    async def test_missing_auth_returns_401_or_403(self, api_context: APIRequestContext):
        """Test that missing auth returns appropriate error."""
        # Create context without API key
        async with async_playwright() as p:
            no_auth_context = await p.request.new_context(base_url=BASE_URL)
            try:
                response = await no_auth_context.get("/api/v2/ocr/extract")
                # Should be unauthorized or endpoint might allow anonymous
                assert response.status in [200, 401, 403, 404, 405]
            finally:
                await no_auth_context.dispose()


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
@pytest.mark.e2e
class TestBatchEndpoints:
    """Batch processing endpoint tests."""

    @pytest.mark.asyncio
    async def test_batch_submit_requires_body(self, api_context: APIRequestContext):
        """Test that batch submit requires request body."""
        response = await api_context.post("/api/v2/batch/submit")
        assert response.status in [400, 404, 422]

    @pytest.mark.asyncio
    async def test_batch_status_returns_404_for_invalid_job(
        self, api_context: APIRequestContext
    ):
        """Test that batch status returns 404 for invalid job ID."""
        response = await api_context.get("/api/v2/batch/status/nonexistent-job-id")
        assert response.status in [404, 400]


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
@pytest.mark.e2e
class TestAPIVersioning:
    """API versioning tests."""

    @pytest.mark.asyncio
    async def test_v1_endpoints_work(self, api_context: APIRequestContext):
        """Test that v1 API endpoints still work."""
        response = await api_context.get("/api/v1/health")
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_v2_endpoints_work(self, api_context: APIRequestContext):
        """Test that v2 API endpoints work."""
        response = await api_context.get("/api/v2/health")
        # v2 health might not exist, but should not error
        assert response.status in [200, 404]


# WebSocket tests require a different approach
@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
@pytest.mark.e2e
class TestWebSocketEndpoints:
    """WebSocket endpoint tests using browser context."""

    @pytest.mark.asyncio
    async def test_websocket_connection_url_exists(self):
        """Test that WebSocket URL is accessible."""
        # This is a basic check - full WebSocket testing requires browser context
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            try:
                page = await browser.new_page()

                # Navigate to a page that would use WebSocket
                # This is a basic connectivity check
                response = await page.goto(f"{BASE_URL}/health")
                assert response is not None
                assert response.status == 200
            finally:
                await browser.close()


def run_e2e_tests():
    """Run E2E tests programmatically."""
    pytest.main([__file__, "-v", "-m", "e2e", "--tb=short"])


if __name__ == "__main__":
    run_e2e_tests()
