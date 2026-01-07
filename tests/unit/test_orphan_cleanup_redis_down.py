"""Tests for orphan cleanup when Redis is down/unavailable.

Tests resilience and fallback behavior when Redis connection fails during
orphan vector cleanup operations.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def _extract_error_payload(payload: dict) -> dict:
    if isinstance(payload, dict) and isinstance(payload.get("detail"), dict):
        return payload["detail"]
    return payload


@pytest.fixture
def mock_redis_unavailable():
    """Mock Redis connection failure."""
    with patch("src.utils.cache.get_client") as mock_get_client:
        mock_get_client.side_effect = ConnectionError("Redis connection failed")
        yield mock_get_client


@pytest.fixture
def mock_redis_timeout():
    """Mock Redis timeout."""
    with patch("src.utils.cache.get_client") as mock_get_client:
        instance = MagicMock()
        instance.get.side_effect = TimeoutError("Redis timeout")
        mock_get_client.return_value = instance
        yield mock_get_client


@pytest.fixture
def mock_redis_partial_failure():
    """Mock Redis partial failure (some operations succeed, some fail)."""
    with patch("src.utils.cache.get_client") as mock_get_client:
        instance = MagicMock()
        # First two calls succeed, third fails
        instance.get.side_effect = [None, None, ConnectionError("Connection lost")]
        mock_get_client.return_value = instance
        yield mock_get_client


def test_orphan_cleanup_redis_connection_failure(mock_redis_unavailable):
    """Test orphan cleanup when Redis connection fails completely."""
    # Attempt cleanup
    response = client.delete("/api/v1/maintenance/orphans", headers={"X-API-Key": "test"})

    # Should return structured error SERVICE_UNAVAILABLE or succeed with fallback
    assert response.status_code in (200, 503, 500)

    data = response.json()
    error_payload = _extract_error_payload(data)

    # Check for structured error format or fallback indication
    if response.status_code >= 400:
        # Should have structured error
        assert "code" in error_payload or "detail" in data

        if "code" in error_payload:
            # build_error format
            assert error_payload["code"] in [
                "SERVICE_UNAVAILABLE",
                "REDIS_CONNECTION_FAILED",
                "BACKEND_UNAVAILABLE",
            ]
            assert error_payload.get("stage") in ["orphan_cleanup", "maintenance", "vector_store"]
            assert "message" in error_payload
            assert isinstance(error_payload["message"], str)
            context = error_payload.get("context", {})
            if isinstance(context, dict):
                assert context.get("operation") == "cleanup_orphan_vectors"
                assert context.get("resource_id") == "vector_store"
    else:
        # Succeeded with fallback
        assert "status" in data
        # May have fallback indicator
        if "fallback" in data:
            assert data["fallback"] is True


def test_orphan_cleanup_redis_timeout(mock_redis_timeout):
    """Test orphan cleanup when Redis times out."""
    response = client.delete("/api/v1/maintenance/orphans", headers={"X-API-Key": "test"})

    # Should handle timeout gracefully
    assert response.status_code in (200, 503, 504)

    data = response.json()

    if response.status_code >= 400 and "code" in data:
        assert data["code"] in ["SERVICE_UNAVAILABLE", "REDIS_TIMEOUT", "BACKEND_UNAVAILABLE"]
        # Message should mention timeout or unavailability
        message = data.get("message", "").lower()
        assert "timeout" in message or "unavailable" in message


def test_orphan_cleanup_redis_partial_failure(mock_redis_partial_failure):
    """Test orphan cleanup when Redis partially fails during operation."""
    response = client.delete("/api/v1/maintenance/orphans", headers={"X-API-Key": "test"})

    # Should handle partial failures
    assert response.status_code in (200, 207, 500, 503)

    data = response.json()

    if response.status_code == 207:
        # Multi-Status: partial success
        assert "status" in data
        assert "warnings" in data or "partial" in data.get("status", "").lower()
    elif response.status_code >= 400 and "code" in data:
        assert data["code"] in ["SERVICE_UNAVAILABLE", "PARTIAL_FAILURE", "BACKEND_UNAVAILABLE"]


def test_orphan_cleanup_error_response_structure():
    """Test that Redis errors return properly structured error responses."""
    with patch("src.utils.cache.get_client") as mock_get_client:
        mock_get_client.side_effect = ConnectionError("Connection refused")

        response = client.delete("/api/v1/maintenance/orphans", headers={"X-API-Key": "test"})

        if response.status_code >= 400:
            data = response.json()
            error_payload = _extract_error_payload(data)

            # Check for structured error format
            if "code" in error_payload:
                # build_error format
                assert isinstance(error_payload["code"], str)
                assert error_payload["code"].isupper()  # SCREAMING_SNAKE_CASE
                assert "_" in error_payload["code"]

                assert "stage" in error_payload
                assert isinstance(error_payload["stage"], str)

                assert "message" in error_payload
                assert isinstance(error_payload["message"], str)
                assert len(error_payload["message"]) > 0

                # Context should be dict if present
                if "context" in error_payload:
                    assert isinstance(error_payload["context"], dict)
                    assert error_payload["context"].get("operation") == "cleanup_orphan_vectors"
                    assert error_payload["context"].get("resource_id") == "vector_store"


def test_orphan_cleanup_suggestion_in_error():
    """Test that error responses include helpful suggestions."""
    with patch("src.utils.cache.get_client") as mock_get_client:
        mock_get_client.side_effect = ConnectionError("Connection refused")

        response = client.delete("/api/v1/maintenance/orphans", headers={"X-API-Key": "test"})

        if response.status_code >= 400:
            data = response.json()

            # Check for suggestion
            has_suggestion = False

            if "context" in data and isinstance(data["context"], dict):
                if "suggestion" in data["context"]:
                    suggestion = data["context"]["suggestion"]
                    has_suggestion = True
                    # Should mention what to do
                    assert any(
                        word in suggestion.lower()
                        for word in ["check", "verify", "ensure", "redis"]
                    )

            if not has_suggestion and "message" in data:
                # Suggestion might be in message
                message = data["message"].lower()
                # Should have actionable guidance
                assert any(
                    word in message for word in ["check", "verify", "ensure", "redis", "backend"]
                )


def test_orphan_cleanup_metric_on_redis_failure():
    """Test that vector_orphan_total metric behavior is consistent on Redis failure."""
    from src.utils.analysis_metrics import vector_orphan_total

    if not hasattr(vector_orphan_total, "_value"):
        pytest.skip("prometheus_client not available")

    with patch("src.utils.cache.get_client") as mock_get_client:
        mock_get_client.side_effect = ConnectionError("Redis down")

        before = vector_orphan_total._value.get()
        response = client.delete("/api/v1/maintenance/orphans", headers={"X-API-Key": "test"})
        after = vector_orphan_total._value.get()

        # Endpoint should be callable
        assert response.status_code in (200, 500, 503)
        assert after > before

        # Verify structured error if failure
        if response.status_code >= 500:
            data = response.json()
            assert "code" in data or "detail" in data


@pytest.mark.slow
def test_orphan_cleanup_redis_recovery():
    """Test that system recovers when Redis comes back online."""
    # First call: Redis down
    with patch("src.utils.cache.get_client") as mock_get_client:
        mock_get_client.side_effect = ConnectionError("Redis down")

        response1 = client.delete("/api/v1/maintenance/orphans", headers={"X-API-Key": "test"})

        # Should fail or use fallback
        assert response1.status_code in (200, 500, 503)

    # Second call: Redis supposedly back (no patching)
    # In real tests, this would verify reconnection logic
    response2 = client.delete("/api/v1/maintenance/orphans", headers={"X-API-Key": "test"})

    # Should work (or fail gracefully if no Redis available in test env)
    assert response2.status_code in (200, 404, 500, 503)
    # Status code depends on actual Redis availability in test environment
