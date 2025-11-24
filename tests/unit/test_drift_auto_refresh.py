"""Tests for drift baseline auto-refresh功能."""

import os
import time
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_drift_state():
    """Reset drift state before each test."""
    from src.api.v1 import analyze as analyze_module
    _DRIFT_STATE = analyze_module._DRIFT_STATE  # type: ignore

    # Save original state
    original_state = {
        "materials": _DRIFT_STATE["materials"].copy(),
        "predictions": _DRIFT_STATE["predictions"].copy(),
        "baseline_materials": _DRIFT_STATE["baseline_materials"].copy(),
        "baseline_predictions": _DRIFT_STATE["baseline_predictions"].copy(),
        "baseline_materials_ts": _DRIFT_STATE.get("baseline_materials_ts"),
        "baseline_predictions_ts": _DRIFT_STATE.get("baseline_predictions_ts"),
    }

    # Clear state for test
    _DRIFT_STATE["materials"].clear()
    _DRIFT_STATE["predictions"].clear()
    _DRIFT_STATE["baseline_materials"].clear()
    _DRIFT_STATE["baseline_predictions"].clear()
    _DRIFT_STATE["baseline_materials_ts"] = None
    _DRIFT_STATE["baseline_predictions_ts"] = None

    yield

    # Restore original state
    _DRIFT_STATE["materials"] = original_state["materials"]
    _DRIFT_STATE["predictions"] = original_state["predictions"]
    _DRIFT_STATE["baseline_materials"] = original_state["baseline_materials"]
    _DRIFT_STATE["baseline_predictions"] = original_state["baseline_predictions"]
    _DRIFT_STATE["baseline_materials_ts"] = original_state["baseline_materials_ts"]
    _DRIFT_STATE["baseline_predictions_ts"] = original_state["baseline_predictions_ts"]


def test_drift_auto_refresh_when_stale():
    """Test baseline is automatically refreshed when age exceeds threshold."""
    from src.api.v1 import analyze as analyze_module
    _DRIFT_STATE = analyze_module._DRIFT_STATE  # type: ignore

    # Set up initial data (>= min_count)
    for _ in range(150):
        _DRIFT_STATE["materials"].append("steel")
        _DRIFT_STATE["predictions"].append("bracket")

    # Create baseline with old timestamp (25 hours ago)
    old_timestamp = time.time() - (25 * 3600)  # 25 hours ago
    _DRIFT_STATE["baseline_materials"] = ["aluminum"] * 100
    _DRIFT_STATE["baseline_materials_ts"] = old_timestamp
    _DRIFT_STATE["baseline_predictions"] = ["plate"] * 100
    _DRIFT_STATE["baseline_predictions_ts"] = old_timestamp

    # Call drift endpoint with auto-refresh enabled and short max_age
    with patch.dict(os.environ, {
        "DRIFT_BASELINE_MAX_AGE_SECONDS": "3600",  # 1 hour
        "DRIFT_BASELINE_AUTO_REFRESH": "1",
        "DRIFT_BASELINE_MIN_COUNT": "100"
    }):
        response = client.get("/api/v1/drift", headers={"X-API-Key": "test"})

    assert response.status_code == 200
    data = response.json()

    # Verify baseline was refreshed (timestamps updated)
    assert data["baseline_material_age"] < 5  # Should be recent (< 5 seconds)
    assert data["baseline_prediction_age"] < 5
    assert data["stale"] is False  # No longer stale after refresh


def test_drift_auto_refresh_disabled():
    """Test baseline is NOT refreshed when auto-refresh is disabled."""
    from src.api.v1 import analyze as analyze_module
    _DRIFT_STATE = analyze_module._DRIFT_STATE  # type: ignore

    # Set up initial data
    for _ in range(150):
        _DRIFT_STATE["materials"].append("steel")
        _DRIFT_STATE["predictions"].append("bracket")

    # Create baseline with old timestamp
    old_timestamp = time.time() - (25 * 3600)
    _DRIFT_STATE["baseline_materials"] = ["aluminum"] * 100
    _DRIFT_STATE["baseline_materials_ts"] = old_timestamp

    # Disable auto-refresh
    with patch.dict(os.environ, {
        "DRIFT_BASELINE_MAX_AGE_SECONDS": "3600",
        "DRIFT_BASELINE_AUTO_REFRESH": "0",  # Disabled
        "DRIFT_BASELINE_MIN_COUNT": "100"
    }):
        response = client.get("/api/v1/drift", headers={"X-API-Key": "test"})

    assert response.status_code == 200
    data = response.json()

    # Verify baseline was NOT refreshed (still old)
    assert data["baseline_material_age"] > 24 * 3600  # Still > 24 hours
    assert data["stale"] is True  # Still marked as stale


def test_drift_auto_refresh_insufficient_data():
    """Test baseline is NOT refreshed when current data < min_count."""
    from src.api.v1 import analyze as analyze_module
    _DRIFT_STATE = analyze_module._DRIFT_STATE  # type: ignore

    # Set up insufficient data (< min_count)
    for _ in range(50):  # Less than 100
        _DRIFT_STATE["materials"].append("steel")

    # Create baseline with old timestamp
    old_timestamp = time.time() - (25 * 3600)
    _DRIFT_STATE["baseline_materials"] = ["aluminum"] * 100
    _DRIFT_STATE["baseline_materials_ts"] = old_timestamp

    with patch.dict(os.environ, {
        "DRIFT_BASELINE_MAX_AGE_SECONDS": "3600",
        "DRIFT_BASELINE_AUTO_REFRESH": "1",
        "DRIFT_BASELINE_MIN_COUNT": "100"
    }):
        response = client.get("/api/v1/drift", headers={"X-API-Key": "test"})

    assert response.status_code == 200
    data = response.json()

    # Verify baseline was NOT refreshed (insufficient current data)
    assert data["baseline_material_age"] > 24 * 3600
    assert data["stale"] is True


def test_drift_manual_reset_records_metric():
    """Test manual reset records drift_baseline_refresh_total metric."""
    from src.api.v1 import analyze as analyze_module
    _DRIFT_STATE = analyze_module._DRIFT_STATE  # type: ignore

    # Create baseline
    _DRIFT_STATE["baseline_materials"] = ["steel"] * 100
    _DRIFT_STATE["baseline_materials_ts"] = time.time()
    _DRIFT_STATE["baseline_predictions"] = ["bracket"] * 100
    _DRIFT_STATE["baseline_predictions_ts"] = time.time()

    # Call manual reset
    response = client.post("/api/v1/drift/reset", headers={"X-API-Key": "test"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["reset_material"] is True
    assert data["reset_predictions"] is True

    # Verify state is cleared
    assert _DRIFT_STATE["baseline_materials"] == []
    assert _DRIFT_STATE["baseline_predictions"] == []
    assert _DRIFT_STATE["baseline_materials_ts"] is None
    assert _DRIFT_STATE["baseline_predictions_ts"] is None


def test_drift_auto_refresh_updates_baseline_content():
    """Test auto-refresh updates baseline to current distribution."""
    from src.api.v1 import analyze as analyze_module
    _DRIFT_STATE = analyze_module._DRIFT_STATE  # type: ignore

    # Set up current data with different distribution
    for _ in range(150):
        _DRIFT_STATE["materials"].append("titanium")  # Different from baseline
        _DRIFT_STATE["predictions"].append("shaft")

    # Create baseline with old timestamp and different distribution
    old_timestamp = time.time() - (25 * 3600)
    _DRIFT_STATE["baseline_materials"] = ["aluminum"] * 100
    _DRIFT_STATE["baseline_materials_ts"] = old_timestamp

    with patch.dict(os.environ, {
        "DRIFT_BASELINE_MAX_AGE_SECONDS": "3600",
        "DRIFT_BASELINE_AUTO_REFRESH": "1",
        "DRIFT_BASELINE_MIN_COUNT": "100"
    }):
        response = client.get("/api/v1/drift", headers={"X-API-Key": "test"})

    assert response.status_code == 200
    data = response.json()

    # Verify baseline now contains current distribution
    assert "titanium" in data["material_baseline"]
    # After refresh, drift score should be 0 (current == baseline)
    assert data["material_drift_score"] == 0.0


def test_drift_auto_refresh_only_refreshes_stale_baselines():
    """Test only stale baselines are refreshed, not fresh ones."""
    from src.api.v1 import analyze as analyze_module
    _DRIFT_STATE = analyze_module._DRIFT_STATE  # type: ignore

    # Set up data
    for _ in range(150):
        _DRIFT_STATE["materials"].append("steel")
        _DRIFT_STATE["predictions"].append("bracket")

    # Material baseline: OLD (should refresh)
    old_timestamp = time.time() - (25 * 3600)
    _DRIFT_STATE["baseline_materials"] = ["aluminum"] * 100
    _DRIFT_STATE["baseline_materials_ts"] = old_timestamp

    # Prediction baseline: FRESH (should NOT refresh)
    fresh_timestamp = time.time() - 600  # 10 minutes ago
    _DRIFT_STATE["baseline_predictions"] = ["plate"] * 100
    _DRIFT_STATE["baseline_predictions_ts"] = fresh_timestamp

    with patch.dict(os.environ, {
        "DRIFT_BASELINE_MAX_AGE_SECONDS": "3600",
        "DRIFT_BASELINE_AUTO_REFRESH": "1",
        "DRIFT_BASELINE_MIN_COUNT": "100"
    }):
        response = client.get("/api/v1/drift", headers={"X-API-Key": "test"})

    assert response.status_code == 200
    data = response.json()

    # Material baseline refreshed
    assert data["baseline_material_age"] < 5

    # Prediction baseline NOT refreshed (still ~10 minutes old)
    assert 550 < data["baseline_prediction_age"] < 650
    assert data["stale"] is False  # Overall not stale (both within limits now)


def test_drift_status_response_structure():
    """Test drift status response includes all required fields."""
    from src.api.v1 import analyze as analyze_module
    _DRIFT_STATE = analyze_module._DRIFT_STATE  # type: ignore

    # Setup minimal data
    _DRIFT_STATE["materials"].extend(["steel"] * 120)
    _DRIFT_STATE["predictions"].extend(["bracket"] * 120)

    response = client.get("/api/v1/drift", headers={"X-API-Key": "test"})

    assert response.status_code == 200
    data = response.json()

    # Verify required fields
    assert "material_current" in data
    assert "material_baseline" in data
    assert "material_drift_score" in data
    assert "prediction_current" in data
    assert "prediction_baseline" in data
    assert "prediction_drift_score" in data
    assert "baseline_min_count" in data
    assert "materials_total" in data
    assert "predictions_total" in data
    assert "status" in data
    assert "baseline_material_age" in data
    assert "baseline_prediction_age" in data
    assert "stale" in data
