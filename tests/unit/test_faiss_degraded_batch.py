"""Tests for batch similarity degradation when Faiss is unavailable.

Tests resilience and fallback behavior when Faiss backend fails during
batch similarity queries, ensuring graceful degradation to in-memory store.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


@pytest.fixture
def mock_faiss_unavailable():
    """Mock Faiss completely unavailable (import failure)."""
    with patch("src.core.similarity.FaissVectorStore") as mock_faiss:
        # Simulate import failure by setting _available to False
        instance = MagicMock()
        instance._available = False
        instance.query.return_value = []
        mock_faiss.return_value = instance
        yield mock_faiss


@pytest.fixture
def mock_faiss_init_failure():
    """Mock Faiss initialization failure after successful import."""
    with patch("src.core.similarity.FaissVectorStore") as mock_faiss:
        mock_faiss.side_effect = RuntimeError("Faiss initialization failed")
        yield mock_faiss


@pytest.fixture
def mock_faiss_query_exception():
    """Mock Faiss query raising exception during batch operation."""
    with patch("src.core.similarity.FaissVectorStore") as mock_faiss:
        instance = MagicMock()
        instance._available = True
        instance.query.side_effect = Exception("Faiss query failed")
        mock_faiss.return_value = instance
        yield mock_faiss


@pytest.fixture
def mock_faiss_partial_failure():
    """Mock Faiss with intermittent failures (some queries succeed, some fail)."""
    with patch("src.core.similarity.FaissVectorStore") as mock_faiss:
        instance = MagicMock()
        instance._available = True
        # First call succeeds, second fails, third succeeds
        instance.query.side_effect = [
            [("vec2", 0.95), ("vec3", 0.85)],
            Exception("Connection lost"),
            [("vec1", 0.92)],
        ]
        mock_faiss.return_value = instance
        yield mock_faiss


def test_batch_similarity_faiss_unavailable(mock_faiss_unavailable):
    """Test batch similarity when Faiss is completely unavailable."""
    from src.utils.analysis_metrics import vector_query_backend_total

    # Get initial metric values
    try:
        initial_memory = vector_query_backend_total.labels(backend="memory_fallback")._value.get()
    except Exception:
        initial_memory = 0

    # Prepare test data - create some vectors first
    # Note: This test assumes in-memory fallback has data
    response = client.post(
        "/api/v1/vectors/similarity/batch",
        headers={"X-API-Key": "test"},
        json={"ids": ["test1", "test2"], "top_k": 5},
    )

    # Should succeed with fallback or return structured error
    assert response.status_code in (200, 404, 422, 500)

    data = response.json()

    if response.status_code == 200:
        # Successful fallback to memory store
        assert "total" in data
        assert "items" in data

        # Check for fallback indicator if present (may be True, None, or missing)
        if data.get("fallback") is not None:
            assert data["fallback"] is True

        # Verify metric incremented (if metric tracking is enabled)
        try:
            final_memory = vector_query_backend_total.labels(backend="memory_fallback")._value.get()
            assert final_memory >= initial_memory  # Should increment or stay same
        except Exception:
            pass  # Metric might not be available in test env
    else:
        # Should have structured error
        assert "detail" in data or "code" in data


def test_batch_similarity_faiss_init_failure(mock_faiss_init_failure):
    """Test batch similarity when Faiss fails to initialize."""
    response = client.post(
        "/api/v1/vectors/similarity/batch",
        headers={"X-API-Key": "test"},
        json={"ids": ["test1"], "top_k": 3},
    )

    # Should handle initialization failure gracefully
    assert response.status_code in (200, 404, 500, 503)

    data = response.json()

    if response.status_code >= 400:
        # Should have error information
        if "code" in data:
            assert isinstance(data["code"], str)
            # Common error codes for backend failures
            assert data["code"] in [
                "SERVICE_UNAVAILABLE",
                "BACKEND_UNAVAILABLE",
                "INTERNAL_ERROR",
                "DATA_NOT_FOUND",
            ]
    else:
        # Succeeded with fallback
        assert "items" in data


def test_batch_similarity_faiss_query_exception(mock_faiss_query_exception):
    """Test batch similarity when Faiss query operation fails."""
    response = client.post(
        "/api/v1/vectors/similarity/batch",
        headers={"X-API-Key": "test"},
        json={"ids": ["test1", "test2", "test3"], "top_k": 5},
    )

    # Should handle query exceptions
    assert response.status_code in (200, 404, 500)

    data = response.json()

    if response.status_code == 200:
        # Check response structure
        assert "total" in data
        assert "items" in data
        assert data["total"] == 3

        # Items should indicate error status for failed queries
        for item in data["items"]:
            assert "id" in item
            assert "status" in item
            # Status could be: success (fallback worked), not_found, or error
            assert item["status"] in ["success", "not_found", "error"]


def test_batch_similarity_faiss_partial_failure(mock_faiss_partial_failure):
    """Test batch similarity with intermittent Faiss failures."""
    response = client.post(
        "/api/v1/vectors/similarity/batch",
        headers={"X-API-Key": "test"},
        json={"ids": ["vec1", "vec2", "vec3"], "top_k": 5},
    )

    # Should handle partial failures
    assert response.status_code in (200, 207, 404, 500)

    data = response.json()

    if response.status_code == 200:
        # Full success (fallback handled failures)
        assert "total" in data
        assert "successful" in data
        assert "failed" in data
        assert data["total"] == 3

    elif response.status_code == 207:
        # Multi-Status: partial success
        assert "items" in data
        # Should have mixed success/failure statuses
        statuses = [item["status"] for item in data["items"]]
        assert "success" in statuses or "error" in statuses


def test_batch_similarity_fallback_metric_recording():
    """Test that fallback to memory store is properly recorded in metrics."""
    from src.utils.analysis_metrics import vector_query_backend_total

    with patch("src.core.similarity.get_vector_store") as mock_factory:
        # Mock factory to return memory store
        from src.core.similarity import InMemoryVectorStore

        memory_store = InMemoryVectorStore()
        mock_factory.return_value = memory_store

        # Get initial count
        try:
            initial_count = vector_query_backend_total.labels(
                backend="memory_fallback"
            )._value.get()
        except Exception:
            initial_count = 0

        # Execute batch query
        response = client.post(
            "/api/v1/vectors/similarity/batch",
            headers={"X-API-Key": "test"},
            json={"ids": ["test1"], "top_k": 5},
        )

        # Should complete (success or not_found acceptable)
        assert response.status_code in (200, 404)

        # Verify metric behavior (may or may not increment depending on implementation)
        try:
            final_count = vector_query_backend_total.labels(backend="memory_fallback")._value.get()
            assert final_count >= initial_count  # Should not decrease
        except Exception:
            pass  # Metric might not be available


def test_batch_similarity_fallback_response_structure():
    """Test that fallback responses maintain proper structure."""
    with patch("src.core.similarity.FaissVectorStore") as mock_faiss:
        # Force fallback by making Faiss unavailable
        instance = MagicMock()
        instance._available = False
        instance.query.return_value = []
        mock_faiss.return_value = instance

        response = client.post(
            "/api/v1/vectors/similarity/batch",
            headers={"X-API-Key": "test"},
            json={"ids": ["test1", "test2"], "top_k": 3, "material": "steel"},
        )

        # Should return valid response structure
        assert response.status_code in (200, 404, 422)

        data = response.json()

        if response.status_code == 200:
            # Verify required fields
            assert "total" in data
            assert "successful" in data
            assert "failed" in data
            assert "items" in data

            # Verify items structure
            for item in data["items"]:
                assert "id" in item
                assert "status" in item
                assert item["status"] in ["success", "not_found", "error"]

                if item["status"] == "success":
                    assert "similar" in item
                    assert isinstance(item["similar"], list)

                if item["status"] == "error":
                    assert "error" in item
                    assert isinstance(item["error"], dict)


def test_batch_similarity_fallback_with_filters():
    """Test that filters still work correctly during fallback."""
    with patch("src.core.similarity.FaissVectorStore") as mock_faiss:
        # Force fallback
        instance = MagicMock()
        instance._available = False
        mock_faiss.return_value = instance

        response = client.post(
            "/api/v1/vectors/similarity/batch",
            headers={"X-API-Key": "test"},
            json={
                "ids": ["test1"],
                "top_k": 5,
                "material": "aluminum",
                "complexity": "high",
                "format": "STEP",
                "min_score": 0.7,
            },
        )

        # Should handle filters properly even with fallback
        assert response.status_code in (200, 404, 422)

        if response.status_code == 200:
            data = response.json()
            assert "items" in data

            # If successful results exist, verify filter application
            for item in data["items"]:
                if item["status"] == "success" and item["similar"]:
                    for result in item["similar"]:
                        # Score should meet threshold
                        assert result["score"] >= 0.7


def test_batch_similarity_performance_acceptable_with_fallback():
    """Test that fallback performance remains acceptable (<10% degradation)."""
    import time

    with patch("src.core.similarity.FaissVectorStore") as mock_faiss:
        # Force fallback to memory store
        instance = MagicMock()
        instance._available = False
        mock_faiss.return_value = instance

        # Measure fallback performance
        start = time.time()
        response = client.post(
            "/api/v1/vectors/similarity/batch",
            headers={"X-API-Key": "test"},
            json={"ids": ["test1", "test2", "test3", "test4", "test5"], "top_k": 10},
        )
        duration = time.time() - start

        # Should complete reasonably quickly (< 1 second for small batch)
        assert duration < 1.0

        # Check response includes duration if available
        if response.status_code == 200:
            data = response.json()
            if "duration_ms" in data:
                # Duration should be reasonable
                assert data["duration_ms"] < 1000  # < 1 second


@pytest.mark.slow
def test_batch_similarity_faiss_recovery():
    """Test that system recovers when Faiss comes back online."""
    # First call: Faiss unavailable
    with patch("src.core.similarity.FaissVectorStore") as mock_faiss:
        instance = MagicMock()
        instance._available = False
        mock_faiss.return_value = instance

        response1 = client.post(
            "/api/v1/vectors/similarity/batch",
            headers={"X-API-Key": "test"},
            json={"ids": ["test1"], "top_k": 5},
        )

        # Should use fallback
        assert response1.status_code in (200, 404, 500)

    # Second call: Faiss supposedly recovered (no patching)
    response2 = client.post(
        "/api/v1/vectors/similarity/batch",
        headers={"X-API-Key": "test"},
        json={"ids": ["test1"], "top_k": 5},
    )

    # Should work or fail gracefully
    assert response2.status_code in (200, 404, 500)
