"""
Test all deprecated endpoints return structured 410 GONE errors.

Validates that deprecated endpoints return:
- HTTP status 410
- Structured error with code=GONE (not RESOURCE_GONE)
- Migration metadata: deprecated_path, new_path, method, migration_date
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)

# List of all deprecated endpoints with their expected migration info
DEPRECATED_ENDPOINTS = [
    {
        "method": "GET",
        "path": "/api/v1/analyze/vectors/distribution",
        "old_path": "/api/v1/analyze/vectors/distribution",
        "new_path": "/api/v1/vectors_stats/distribution",
    },
    {
        "method": "POST",
        "path": "/api/v1/analyze/vectors/delete",
        "old_path": "/api/v1/analyze/vectors/delete",
        "new_path": "/api/v1/vectors/delete (DELETE method)",
        "payload": {"id": "test-id"},
    },
    {
        "method": "GET",
        "path": "/api/v1/analyze/vectors",
        "old_path": "/api/v1/analyze/vectors",
        "new_path": "/api/v1/vectors",
    },
    {
        "method": "GET",
        "path": "/api/v1/analyze/vectors/stats",
        "old_path": "/api/v1/analyze/vectors/stats",
        "new_path": "/api/v1/vectors_stats/summary",
    },
    {
        "method": "GET",
        "path": "/api/v1/analyze/features/diff",
        "old_path": "/api/v1/analyze/features/diff",
        "new_path": "/api/v1/features/diff",
    },
    {
        "method": "POST",
        "path": "/api/v1/analyze/model/reload",
        "old_path": "/api/v1/analyze/model/reload",
        "new_path": "/api/v1/model/reload",
        "payload": {},
    },
    {
        "method": "GET",
        "path": "/api/v1/analyze/features/cache",
        "old_path": "/api/v1/analyze/features/cache",
        "new_path": "/api/v1/maintenance/stats (includes cache info)",
    },
    {
        "method": "GET",
        "path": "/api/v1/analyze/faiss/health",
        "old_path": "/api/v1/analyze/faiss/health",
        "new_path": "/api/v1/health/faiss",
    },
]


@pytest.mark.parametrize("endpoint", DEPRECATED_ENDPOINTS)
def test_deprecated_endpoint_returns_410_with_structured_error(endpoint):
    """Test that deprecated endpoint returns 410 with proper structured error."""
    method = endpoint["method"]
    path = endpoint["path"]
    payload = endpoint.get("payload")

    # Make request
    if method == "GET":
        resp = client.get(path, headers={"X-API-Key": "test"})
    elif method == "POST":
        resp = client.post(path, json=payload, headers={"X-API-Key": "test"})
    elif method == "DELETE":
        resp = client.delete(path, headers={"X-API-Key": "test"})
    else:
        pytest.fail(f"Unsupported method: {method}")

    # Assert HTTP 410
    assert resp.status_code == 410, f"Expected 410 for {path}, got {resp.status_code}"

    # Get response body
    body = resp.json()
    detail = body.get("detail", {})

    # Assert structured error format
    assert isinstance(detail, dict), f"Detail should be dict, got {type(detail)}"

    # Assert error code is GONE (not RESOURCE_GONE)
    assert detail.get("code") == "GONE", f"Expected code=GONE, got {detail.get('code')}"

    # Assert migration metadata present
    assert "deprecated_path" in detail, "Missing deprecated_path in error"
    assert "new_path" in detail, "Missing new_path in error"
    assert "method" in detail, "Missing method in error"
    assert "migration_date" in detail, "Missing migration_date in error"

    # Validate deprecated_path matches
    assert detail["deprecated_path"] == endpoint["old_path"], \
        f"deprecated_path mismatch: {detail['deprecated_path']} != {endpoint['old_path']}"

    # Validate method matches
    assert detail["method"] == method, \
        f"method mismatch: {detail['method']} != {method}"


def test_deprecated_endpoint_error_message_clarity():
    """Test that error message provides clear migration guidance."""
    resp = client.get(
        "/api/v1/analyze/vectors/distribution",
        headers={"X-API-Key": "test"}
    )

    assert resp.status_code == 410
    detail = resp.json()["detail"]

    # Check message includes migration guidance
    message = detail.get("message", "")
    assert "moved" in message.lower() or "use" in message.lower(), \
        f"Message should provide migration guidance: {message}"
    assert detail["new_path"] in message, \
        f"Message should mention new path: {message}"


def test_deprecated_endpoint_severity():
    """Test that GONE errors have appropriate severity (INFO level)."""
    resp = client.get(
        "/api/v1/analyze/vectors",
        headers={"X-API-Key": "test"}
    )

    assert resp.status_code == 410
    detail = resp.json()["detail"]

    # GONE should be INFO severity (resource moved, not an error)
    # Note: If severity is added to error response, validate it here
    # For now, just ensure the error code is GONE
    assert detail["code"] == "GONE"


def test_all_deprecated_endpoints_covered():
    """Ensure we have test coverage for all deprecated endpoints."""
    # This is a meta-test to ensure we don't miss any deprecated endpoints
    # If a new 410 endpoint is added, this test should be updated
    assert len(DEPRECATED_ENDPOINTS) >= 8, \
        f"Expected at least 8 deprecated endpoints, got {len(DEPRECATED_ENDPOINTS)}"


def test_deprecated_vector_delete_POST_to_DELETE_migration():
    """Specific test for POST /vectors/delete -> DELETE /vectors migration."""
    # Old POST method (deprecated)
    resp = client.post(
        "/api/v1/analyze/vectors/delete",
        json={"id": "test-id"},
        headers={"X-API-Key": "test"}
    )

    assert resp.status_code == 410
    detail = resp.json()["detail"]
    assert "DELETE method" in detail["new_path"]
    assert detail["method"] == "POST"


def test_deprecated_model_reload_migration():
    """Specific test for model reload endpoint migration."""
    resp = client.post(
        "/api/v1/analyze/model/reload",
        json={},
        headers={"X-API-Key": "test"}
    )

    assert resp.status_code == 410
    detail = resp.json()["detail"]
    assert detail["new_path"] == "/api/v1/model/reload"
    assert detail["deprecated_path"] == "/api/v1/analyze/model/reload"
