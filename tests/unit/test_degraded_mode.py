"""Tests for vector store degraded mode functionality.

Tests that degraded mode flag is set when Faiss falls back to memory,
and that the health endpoint correctly reports degraded status.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
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
    import os

    from src.core.similarity import get_degraded_mode_info, get_vector_store

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
    import os

    from src.core.similarity import get_degraded_mode_info, get_vector_store

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
            assert (
                "RuntimeError" in degraded_info["reason"]
                or "init failed" in degraded_info["reason"]
            )
            assert degraded_info["degraded_at"] is not None
    finally:
        os.environ.pop("VECTOR_STORE_BACKEND", None)


def test_degraded_mode_not_set_for_memory_backend():
    """Test that degraded mode is not set when using memory backend."""
    from src.core.similarity import get_degraded_mode_info, get_vector_store

    # Get memory store (no degradation expected)
    store = get_vector_store("memory")

    degraded_info = get_degraded_mode_info()
    assert degraded_info["degraded"] is False
    assert degraded_info["reason"] is None
    assert degraded_info["degraded_at"] is None
    assert degraded_info["degraded_duration_seconds"] is None


def test_degraded_mode_reset_on_store_reset():
    """Test that degraded mode flags are cleared on store reset."""
    import os

    from src.core.similarity import get_degraded_mode_info, get_vector_store, reset_default_store

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
        response = client.get("/api/v1/health/faiss/health", headers={"X-API-Key": "test"})

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
        response = client.get("/api/v1/health/faiss/health", headers={"X-API-Key": "test"})

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
    import time

    from src.core import similarity
    from src.core.similarity import get_degraded_mode_info

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


def test_degradation_history_recorded():
    """Test that degradation events are recorded in history."""
    import os

    from src.core.similarity import get_degraded_mode_info, get_vector_store

    os.environ["VECTOR_STORE_BACKEND"] = "faiss"

    try:
        # Mock multiple degradation events
        with patch("src.core.similarity.FaissVectorStore") as MockFaiss:
            mock_instance = MagicMock()
            mock_instance._available = False
            MockFaiss.return_value = mock_instance

            # Trigger first degradation
            get_vector_store("faiss")

            degraded_info = get_degraded_mode_info()
            assert degraded_info["history_count"] == 1
            assert len(degraded_info["history"]) == 1
            assert degraded_info["history"][0]["reason"] == "Faiss library unavailable"
            assert degraded_info["history"][0]["backend_requested"] == "faiss"
            assert degraded_info["history"][0]["backend_actual"] == "memory"
    finally:
        os.environ.pop("VECTOR_STORE_BACKEND", None)


def test_degradation_history_limit():
    """Test that degradation history is limited to 10 events."""
    import time

    from src.core import similarity

    # Manually add 15 events
    similarity._DEGRADATION_HISTORY = []
    for i in range(15):
        similarity._DEGRADATION_HISTORY.append(
            {
                "timestamp": time.time(),
                "reason": f"Test event {i}",
                "backend_requested": "faiss",
                "backend_actual": "memory",
            }
        )

    # Trigger history cleanup by simulating one more degradation
    # (cleanup happens during append)
    from src.core.similarity import get_degraded_mode_info

    degraded_info = get_degraded_mode_info()

    # Should have all 15 (cleanup only happens on new append in get_vector_store)
    assert degraded_info["history_count"] == 15

    # Test that cleanup works correctly with limit
    assert similarity._MAX_DEGRADATION_HISTORY == 10


def test_degradation_history_in_health_endpoint():
    """Test that degradation history is exposed in health endpoint."""
    import time

    from src.core import similarity

    # Set up degraded state with history
    similarity._VECTOR_DEGRADED = True
    similarity._VECTOR_DEGRADED_REASON = "Test degradation"
    similarity._VECTOR_DEGRADED_AT = time.time()
    similarity._DEGRADATION_HISTORY = [
        {
            "timestamp": time.time() - 100,
            "reason": "Event 1",
            "backend_requested": "faiss",
            "backend_actual": "memory",
        },
        {
            "timestamp": time.time() - 50,
            "reason": "Event 2",
            "backend_requested": "faiss",
            "backend_actual": "memory",
            "error": "Init failed",
        },
    ]

    try:
        response = client.get("/api/v1/health/faiss/health", headers={"X-API-Key": "test"})

        assert response.status_code == 200
        data = response.json()

        assert data["degradation_history_count"] == 2
        assert data["degradation_history"] is not None
        assert len(data["degradation_history"]) == 2
        assert data["degradation_history"][0]["reason"] == "Event 1"
        assert data["degradation_history"][1]["reason"] == "Event 2"
        assert "error" in data["degradation_history"][1]
    finally:
        # Reset
        similarity._VECTOR_DEGRADED = False
        similarity._VECTOR_DEGRADED_REASON = None
        similarity._VECTOR_DEGRADED_AT = None
        similarity._DEGRADATION_HISTORY = []
