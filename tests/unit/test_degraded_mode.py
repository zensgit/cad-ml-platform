"""Tests for vector store degraded mode functionality.

Tests that degraded mode flag is set when Faiss falls back to memory,
and that the health endpoint correctly reports degraded status.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_store():
    """Reset vector store state before each test."""
    from src.core.similarity import reset_default_store
    reset_default_store()
    yield
    reset_default_store()


def test_degraded_mode_set_when_faiss_unavailable():
    """Test that degraded mode is set when Faiss library is unavailable."""
    from src.core.similarity import get_vector_store, get_degraded_mode_info
    import os

    # Set backend to faiss
    original_backend = os.getenv("VECTOR_STORE_BACKEND")
    os.environ["VECTOR_STORE_BACKEND"] = "faiss"

    try:
        # Mock FaissVectorStore to report unavailable
        with patch("src.core.similarity.FaissVectorStore") as MockFaiss:
            mock_instance = MagicMock()
            mock_instance._available = False
            MockFaiss.return_value = mock_instance

            # Get store - should trigger degradation
            store = get_vector_store("faiss")

            # Check degraded mode is set
            degraded_info = get_degraded_mode_info()
            assert degraded_info["degraded"] is True
            assert degraded_info["reason"] == "Faiss library unavailable"
            assert degraded_info["degraded_at"] is not None
            assert degraded_info["degraded_duration_seconds"] is not None
            assert degraded_info["degraded_duration_seconds"] >= 0
    finally:
        # Restore original backend
        if original_backend:
            os.environ["VECTOR_STORE_BACKEND"] = original_backend
        else:
            os.environ.pop("VECTOR_STORE_BACKEND", None)


def test_degraded_mode_set_on_faiss_init_exception():
    """Test that degraded mode is set when Faiss initialization fails."""
    from src.core.similarity import get_vector_store, get_degraded_mode_info
    import os

    os.environ["VECTOR_STORE_BACKEND"] = "faiss"

    try:
        # Mock FaissVectorStore to raise exception
        with patch("src.core.similarity.FaissVectorStore") as MockFaiss:
            MockFaiss.side_effect = RuntimeError("Faiss init failed")

            # Get store - should trigger degradation
            store = get_vector_store("faiss")

            # Check degraded mode is set
            degraded_info = get_degraded_mode_info()
            assert degraded_info["degraded"] is True
            assert "Faiss initialization failed" in degraded_info["reason"]
            assert "RuntimeError" in degraded_info["reason"] or "init failed" in degraded_info["reason"]
            assert degraded_info["degraded_at"] is not None
    finally:
        os.environ.pop("VECTOR_STORE_BACKEND", None)


def test_degraded_mode_not_set_for_memory_backend():
    """Test that degraded mode is not set when using memory backend."""
    from src.core.similarity import get_vector_store, get_degraded_mode_info

    # Get memory store (no degradation expected)
    store = get_vector_store("memory")

    degraded_info = get_degraded_mode_info()
    assert degraded_info["degraded"] is False
    assert degraded_info["reason"] is None
    assert degraded_info["degraded_at"] is None
    assert degraded_info["degraded_duration_seconds"] is None


def test_degraded_mode_reset_on_store_reset():
    """Test that degraded mode flags are cleared on store reset."""
    from src.core.similarity import get_vector_store, get_degraded_mode_info, reset_default_store
    import os

    os.environ["VECTOR_STORE_BACKEND"] = "faiss"

    try:
        # Trigger degradation
        with patch("src.core.similarity.FaissVectorStore") as MockFaiss:
            mock_instance = MagicMock()
            mock_instance._available = False
            MockFaiss.return_value = mock_instance

            get_vector_store("faiss")

            # Verify degraded
            assert get_degraded_mode_info()["degraded"] is True

            # Reset store
            reset_default_store()

            # Verify degraded mode cleared
            degraded_info = get_degraded_mode_info()
            assert degraded_info["degraded"] is False
            assert degraded_info["reason"] is None
            assert degraded_info["degraded_at"] is None
    finally:
        os.environ.pop("VECTOR_STORE_BACKEND", None)


def test_faiss_health_endpoint_shows_degraded_status():
    """Test that /faiss/health endpoint reports degraded status."""
    from src.core import similarity

    # Mock degraded state
    similarity._VECTOR_DEGRADED = True
    similarity._VECTOR_DEGRADED_REASON = "Test degradation"
    import time
    similarity._VECTOR_DEGRADED_AT = time.time()

    try:
        response = client.get(
            "/api/v1/health/faiss/health",
            headers={"X-API-Key": "test"}
        )

        assert response.status_code == 200
        data = response.json()

        # Status should be "degraded" not "ok"
        assert data["status"] == "degraded"
        assert data["degraded"] is True
        assert data["degraded_reason"] == "Test degradation"
        assert data["degraded_duration_seconds"] is not None
        assert data["degraded_duration_seconds"] >= 0
    finally:
        # Reset degraded state
        similarity._VECTOR_DEGRADED = False
        similarity._VECTOR_DEGRADED_REASON = None
        similarity._VECTOR_DEGRADED_AT = None


def test_faiss_health_endpoint_normal_status():
    """Test that /faiss/health endpoint reports ok status when not degraded."""
    from src.core import similarity

    # Ensure not degraded
    similarity._VECTOR_DEGRADED = False
    similarity._VECTOR_DEGRADED_REASON = None
    similarity._VECTOR_DEGRADED_AT = None

    # Mock Faiss as available
    similarity._FAISS_AVAILABLE = True

    try:
        response = client.get(
            "/api/v1/health/faiss/health",
            headers={"X-API-Key": "test"}
        )

        assert response.status_code == 200
        data = response.json()

        # Status should be "ok" not "degraded"
        assert data["status"] == "ok"
        assert data["degraded"] is False
        assert data["degraded_reason"] is None
        assert data["degraded_duration_seconds"] is None
    finally:
        # Reset state
        similarity._FAISS_AVAILABLE = None


def test_degraded_mode_duration_calculation():
    """Test that degraded duration is calculated correctly."""
    from src.core.similarity import get_degraded_mode_info
    from src.core import similarity
    import time

    # Set degraded 5 seconds ago
    similarity._VECTOR_DEGRADED = True
    similarity._VECTOR_DEGRADED_REASON = "Test"
    similarity._VECTOR_DEGRADED_AT = time.time() - 5.0

    try:
        degraded_info = get_degraded_mode_info()

        assert degraded_info["degraded"] is True
        assert degraded_info["degraded_duration_seconds"] is not None
        # Should be ~5 seconds (allow small variance)
        assert 4.8 <= degraded_info["degraded_duration_seconds"] <= 5.5
    finally:
        # Reset
        similarity._VECTOR_DEGRADED = False
        similarity._VECTOR_DEGRADED_REASON = None
        similarity._VECTOR_DEGRADED_AT = None
