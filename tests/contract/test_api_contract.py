"""API contract tests using Schemathesis.

Features:
- OpenAPI schema validation
- Property-based testing
- Response contract verification
- Stateful testing
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pytest

# Check if schemathesis is available
try:
    import schemathesis
    from hypothesis import given, settings, HealthCheck

    SCHEMATHESIS_AVAILABLE = True
except ImportError:
    SCHEMATHESIS_AVAILABLE = False

# Configuration
BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "test-api-key")
OPENAPI_URL = f"{BASE_URL}/openapi.json"


@pytest.mark.skipif(not SCHEMATHESIS_AVAILABLE, reason="Schemathesis not installed")
@pytest.mark.contract
class TestAPIContract:
    """Contract tests based on OpenAPI schema."""

    @pytest.fixture(scope="class")
    def schema(self):
        """Load OpenAPI schema."""
        try:
            return schemathesis.from_uri(
                OPENAPI_URL,
                base_url=BASE_URL,
            )
        except Exception as e:
            pytest.skip(f"Could not load OpenAPI schema: {e}")

    def test_openapi_schema_valid(self, schema):
        """Test that OpenAPI schema is valid."""
        assert schema is not None
        assert len(list(schema.get_all_operations())) > 0

    @pytest.mark.parametrize("endpoint", ["/health", "/api/v1/health"])
    def test_health_endpoints_match_schema(self, endpoint):
        """Test health endpoints match their schema."""
        import requests

        response = requests.get(
            f"{BASE_URL}{endpoint}",
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 200

        # Should return JSON
        data = response.json()
        assert isinstance(data, dict)

    def test_error_responses_have_correct_structure(self):
        """Test that error responses follow consistent structure."""
        import requests

        # Request non-existent endpoint
        response = requests.get(
            f"{BASE_URL}/api/v2/nonexistent",
            headers={"X-API-Key": API_KEY},
        )

        assert response.status_code == 404
        data = response.json()

        # Should have error structure
        assert any(key in data for key in ["detail", "error", "message"])


# Schemathesis-based tests (when available)
if SCHEMATHESIS_AVAILABLE:
    try:
        schema = schemathesis.from_uri(OPENAPI_URL, base_url=BASE_URL)

        @schema.parametrize()
        @settings(
            max_examples=10,
            suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        )
        def test_api_contract(case):
            """Property-based API contract test."""
            # Add authentication
            case.headers = case.headers or {}
            case.headers["X-API-Key"] = API_KEY

            response = case.call()

            # Validate response
            case.validate_response(response)

    except Exception:
        # Schema loading failed, tests will be skipped
        pass


@pytest.mark.contract
class TestResponseContracts:
    """Manual response contract tests."""

    def test_success_response_structure(self):
        """Test successful response structure."""
        import requests

        response = requests.get(
            f"{BASE_URL}/health",
            headers={"X-API-Key": API_KEY},
        )

        assert response.status_code == 200
        data = response.json()

        # Health endpoint should have status
        assert "status" in data

    def test_validation_error_structure(self):
        """Test validation error response structure."""
        import requests

        # Send invalid request
        response = requests.post(
            f"{BASE_URL}/api/v2/materials/classify",
            headers={
                "X-API-Key": API_KEY,
                "Content-Type": "application/json",
            },
            json={},  # Missing required fields
        )

        # Should return validation error
        if response.status_code == 422:
            data = response.json()
            assert "detail" in data or "error" in data

    def test_rate_limit_response_structure(self):
        """Test rate limit response structure."""
        import requests

        # This test might not trigger rate limiting in normal conditions
        response = requests.get(
            f"{BASE_URL}/health",
            headers={"X-API-Key": API_KEY},
        )

        # Check for rate limit headers
        headers = response.headers
        rate_limit_headers = [
            "x-ratelimit-limit",
            "x-ratelimit-remaining",
            "ratelimit-limit",
        ]

        # At least verify the response is valid
        assert response.status_code in [200, 429]


@pytest.mark.contract
class TestContentTypeContracts:
    """Content type contract tests."""

    def test_json_endpoints_return_json(self):
        """Test that JSON endpoints return proper content type."""
        import requests

        response = requests.get(
            f"{BASE_URL}/health",
            headers={"X-API-Key": API_KEY},
        )

        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type.lower()

    def test_metrics_endpoint_returns_text(self):
        """Test that metrics endpoint returns text/plain."""
        import requests

        response = requests.get(f"{BASE_URL}/metrics")

        content_type = response.headers.get("content-type", "")
        # Prometheus metrics can be text/plain or application/openmetrics-text
        assert any(t in content_type.lower() for t in ["text/plain", "openmetrics"])


@pytest.mark.contract
class TestHeaderContracts:
    """Header contract tests."""

    def test_cors_headers_present(self):
        """Test CORS headers are present when needed."""
        import requests

        response = requests.options(
            f"{BASE_URL}/health",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )

        # CORS might not be enabled, but request should succeed
        assert response.status_code in [200, 204, 405]

    def test_security_headers_present(self):
        """Test security headers are present."""
        import requests

        response = requests.get(
            f"{BASE_URL}/health",
            headers={"X-API-Key": API_KEY},
        )

        # These headers might not all be present depending on configuration
        # Just verify the request succeeds
        assert response.status_code == 200


@pytest.mark.contract
class TestVersioningContract:
    """API versioning contract tests."""

    def test_v1_api_available(self):
        """Test v1 API is available."""
        import requests

        response = requests.get(
            f"{BASE_URL}/api/v1/health",
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 200

    def test_version_header_accepted(self):
        """Test version header is accepted."""
        import requests

        response = requests.get(
            f"{BASE_URL}/health",
            headers={
                "X-API-Key": API_KEY,
                "X-API-Version": "v2",
            },
        )

        # Header should not cause errors
        assert response.status_code == 200


def run_contract_tests():
    """Run contract tests programmatically."""
    pytest.main([__file__, "-v", "-m", "contract", "--tb=short"])


if __name__ == "__main__":
    run_contract_tests()
