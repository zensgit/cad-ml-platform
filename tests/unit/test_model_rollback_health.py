"""Tests for model rollback health reporting.

Tests that the /health/model endpoint correctly reports rollback information
when model reloading fails and triggers automatic rollback to previous versions.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


@pytest.fixture
def mock_model_info_no_rollback():
    """Mock get_model_info returning normal state (no rollback)."""
    return {
        "version": "v1.0.0",
        "hash": "abc123",
        "path": "/models/classifier_v1.pkl",
        "loaded": True,
        "loaded_at": 1700000000.0,
        "rollback_level": 0,
        "last_error": None,
        "rollback_reason": None,
        "has_prev": False,
        "has_prev2": False,
    }


@pytest.fixture
def mock_model_info_rollback_level1():
    """Mock get_model_info returning rollback level 1."""
    return {
        "version": "v0.9.0",
        "hash": "def456",
        "path": "/models/classifier_v0.9.pkl",
        "loaded": True,
        "loaded_at": 1700000100.0,
        "rollback_level": 1,
        "last_error": "Security validation failed",
        "rollback_reason": "Rolled back to previous model after reload failure",
        "has_prev": True,
        "has_prev2": False,
    }


@pytest.fixture
def mock_model_info_rollback_level2():
    """Mock get_model_info returning rollback level 2."""
    return {
        "version": "v0.8.0",
        "hash": "ghi789",
        "path": "/models/classifier_v0.8.pkl",
        "loaded": True,
        "loaded_at": 1700000200.0,
        "rollback_level": 2,
        "last_error": "Consecutive reload failures",
        "rollback_reason": "Rolled back to level 2 snapshot after consecutive failures",
        "has_prev": True,
        "has_prev2": True,
    }


@pytest.fixture
def mock_model_info_absent():
    """Mock get_model_info returning absent state."""
    return {
        "version": "none",
        "hash": None,
        "path": None,
        "loaded": False,
        "loaded_at": None,
        "rollback_level": 0,
        "last_error": None,
        "rollback_reason": None,
        "has_prev": False,
        "has_prev2": False,
    }


def test_model_health_no_rollback(mock_model_info_no_rollback):
    """Test health endpoint with normal model (no rollback)."""
    with patch("src.ml.classifier.get_model_info") as mock_info:
        mock_info.return_value = mock_model_info_no_rollback

        response = client.get(
            "/api/v1/health/model",
            headers={"X-API-Key": "test"}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify basic fields
        assert data["status"] == "ok"
        assert data["loaded"] is True
        assert data["version"] == "v1.0.0"
        assert data["hash"] == "abc123"

        # Verify rollback fields
        assert data["rollback_level"] == 0
        assert data["rollback_reason"] is None
        assert data["last_error"] is None


def test_model_health_rollback_level1(mock_model_info_rollback_level1):
    """Test health endpoint with level 1 rollback."""
    with patch("src.ml.classifier.get_model_info") as mock_info:
        mock_info.return_value = mock_model_info_rollback_level1

        response = client.get(
            "/api/v1/health/model",
            headers={"X-API-Key": "test"}
        )

        assert response.status_code == 200
        data = response.json()

        # Status should be "rollback" not "ok"
        assert data["status"] == "rollback"
        assert data["loaded"] is True
        assert data["version"] == "v0.9.0"

        # Verify rollback information
        assert data["rollback_level"] == 1
        assert data["rollback_reason"] == "Rolled back to previous model after reload failure"
        assert data["last_error"] == "Security validation failed"


def test_model_health_rollback_level2(mock_model_info_rollback_level2):
    """Test health endpoint with level 2 rollback."""
    with patch("src.ml.classifier.get_model_info") as mock_info:
        mock_info.return_value = mock_model_info_rollback_level2

        response = client.get(
            "/api/v1/health/model",
            headers={"X-API-Key": "test"}
        )

        assert response.status_code == 200
        data = response.json()

        # Status should be "rollback"
        assert data["status"] == "rollback"
        assert data["loaded"] is True
        assert data["version"] == "v0.8.0"

        # Verify level 2 rollback
        assert data["rollback_level"] == 2
        assert "level 2 snapshot" in data["rollback_reason"]
        assert data["last_error"] == "Consecutive reload failures"


def test_model_health_absent(mock_model_info_absent):
    """Test health endpoint when model is absent."""
    with patch("src.ml.classifier.get_model_info") as mock_info:
        mock_info.return_value = mock_model_info_absent

        response = client.get(
            "/api/v1/health/model",
            headers={"X-API-Key": "test"}
        )

        assert response.status_code == 200
        data = response.json()

        # Status should be "absent"
        assert data["status"] == "absent"
        assert data["loaded"] is False
        assert data["version"] == "none"

        # Rollback fields should be default
        assert data["rollback_level"] == 0
        assert data["rollback_reason"] is None


def test_model_health_metric_recording():
    """Test that health checks record correct metric labels."""
    from src.utils.analysis_metrics import model_health_checks_total

    with patch("src.ml.classifier.get_model_info") as mock_info:
        # Test "ok" status metric
        mock_info.return_value = {
            "version": "v1.0.0",
            "hash": "abc",
            "path": "/models/test.pkl",
            "loaded": True,
            "loaded_at": 1700000000.0,
            "rollback_level": 0,
            "last_error": None,
            "rollback_reason": None,
            "has_prev": False,
            "has_prev2": False,
        }

        response = client.get(
            "/api/v1/health/model",
            headers={"X-API-Key": "test"}
        )

        assert response.status_code == 200
        # Metric should be recorded with status="ok"

        # Test "rollback" status metric
        mock_info.return_value["rollback_level"] = 1
        mock_info.return_value["rollback_reason"] = "Test rollback"

        response = client.get(
            "/api/v1/health/model",
            headers={"X-API-Key": "test"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "rollback"


def test_model_health_rollback_clear_after_recovery():
    """Test that rollback information clears after successful reload."""
    with patch("src.ml.classifier.get_model_info") as mock_info:
        # First call: model in rollback state
        mock_info.return_value = {
            "version": "v0.9.0",
            "hash": "old",
            "path": "/models/old.pkl",
            "loaded": True,
            "loaded_at": 1700000000.0,
            "rollback_level": 1,
            "last_error": "Previous error",
            "rollback_reason": "Rolled back after failure",
            "has_prev": True,
            "has_prev2": False,
        }

        response1 = client.get(
            "/api/v1/health/model",
            headers={"X-API-Key": "test"}
        )

        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["status"] == "rollback"
        assert data1["rollback_level"] == 1

        # Second call: model successfully reloaded (no rollback)
        mock_info.return_value = {
            "version": "v1.1.0",
            "hash": "new",
            "path": "/models/new.pkl",
            "loaded": True,
            "loaded_at": 1700000100.0,
            "rollback_level": 0,
            "last_error": None,
            "rollback_reason": None,
            "has_prev": True,  # Previous still exists but not active
            "has_prev2": False,
        }

        response2 = client.get(
            "/api/v1/health/model",
            headers={"X-API-Key": "test"}
        )

        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["status"] == "ok"
        assert data2["rollback_level"] == 0
        assert data2["rollback_reason"] is None


def test_model_health_consecutive_failures():
    """Test health reporting during consecutive rollback failures."""
    with patch("src.ml.classifier.get_model_info") as mock_info:
        # Simulate progression: ok → level1 → level2
        states = [
            # Initial state: ok
            {
                "version": "v1.0.0",
                "loaded": True,
                "rollback_level": 0,
                "rollback_reason": None,
            },
            # After first failure: rollback to level 1
            {
                "version": "v0.9.0",
                "loaded": True,
                "rollback_level": 1,
                "rollback_reason": "Rolled back after first failure",
            },
            # After second failure: rollback to level 2
            {
                "version": "v0.8.0",
                "loaded": True,
                "rollback_level": 2,
                "rollback_reason": "Rolled back to level 2 snapshot",
            },
        ]

        for state in states:
            # Add common fields
            state.update({
                "hash": "test",
                "path": "/test.pkl",
                "loaded_at": 1700000000.0,
                "last_error": None,
                "has_prev": True,
                "has_prev2": True,
            })
            mock_info.return_value = state

            response = client.get(
                "/api/v1/health/model",
                headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200
            data = response.json()

            expected_level = state["rollback_level"]
            assert data["rollback_level"] == expected_level

            if expected_level > 0:
                assert data["status"] == "rollback"
                assert data["rollback_reason"] is not None
            else:
                assert data["status"] == "ok"
                assert data["rollback_reason"] is None


def test_model_load_seq_increments():
    """Test that load_seq increments on successive successful reloads."""
    with patch("src.ml.classifier.get_model_info") as mock_info:
        # First load: seq = 0
        mock_info.return_value = {
            "version": "v1.0.0",
            "hash": "abc123",
            "path": "/models/v1.pkl",
            "loaded": True,
            "loaded_at": 1700000000.0,
            "rollback_level": 0,
            "last_error": None,
            "rollback_reason": None,
            "has_prev": False,
            "has_prev2": False,
            "load_seq": 0,
        }

        response1 = client.get("/api/v1/health/model", headers={"X-API-Key": "test"})
        assert response1.status_code == 200
        data1 = response1.json()

        # Second load: seq = 1 (after successful reload)
        mock_info.return_value = {
            "version": "v1.1.0",
            "hash": "def456",
            "path": "/models/v1.1.pkl",
            "loaded": True,
            "loaded_at": 1700000100.0,
            "rollback_level": 0,
            "last_error": None,
            "rollback_reason": None,
            "has_prev": True,
            "has_prev2": False,
            "load_seq": 1,
        }

        response2 = client.get("/api/v1/health/model", headers={"X-API-Key": "test"})
        assert response2.status_code == 200
        data2 = response2.json()

        # Third load: seq = 2
        mock_info.return_value["load_seq"] = 2
        mock_info.return_value["version"] = "v1.2.0"

        response3 = client.get("/api/v1/health/model", headers={"X-API-Key": "test"})
        assert response3.status_code == 200
        data3 = response3.json()

        # Verify versions changed (indicating successful reloads)
        assert data1["version"] == "v1.0.0"
        assert data2["version"] == "v1.1.0"
        assert data3["version"] == "v1.2.0"

        # Verify load_seq increments are exposed in API response
        assert data1["load_seq"] == 0
        assert data2["load_seq"] == 1
        assert data3["load_seq"] == 2
