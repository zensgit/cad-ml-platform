from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def _headers():
    return {"X-API-Key": "test"}


def test_drift_baseline_export_empty():
    response = client.post("/api/v1/analyze/drift/baseline/export", headers=_headers())
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"empty", "ok"}
    snapshot = data["snapshot"]
    assert snapshot["material_baseline"] == []
    assert snapshot["prediction_baseline"] == []
    assert "exported_at" in snapshot


def test_drift_baseline_import_roundtrip():
    payload = {
        "material_baseline": ["steel", "aluminum"],
        "prediction_baseline": ["bracket"],
        "material_baseline_ts": 1700000000,
        "prediction_baseline_ts": 1700001234,
    }
    import_resp = client.post(
        "/api/v1/analyze/drift/baseline/import",
        headers=_headers(),
        json=payload,
    )
    assert import_resp.status_code == 200
    import_data = import_resp.json()
    assert import_data["status"] == "ok"
    assert import_data["imported_materials"] is True
    assert import_data["imported_predictions"] is True

    export_resp = client.post("/api/v1/analyze/drift/baseline/export", headers=_headers())
    assert export_resp.status_code == 200
    export_data = export_resp.json()["snapshot"]
    assert export_data["material_baseline"] == payload["material_baseline"]
    assert export_data["prediction_baseline"] == payload["prediction_baseline"]
    assert export_data["material_baseline_ts"] == payload["material_baseline_ts"]
    assert export_data["prediction_baseline_ts"] == payload["prediction_baseline_ts"]


def test_drift_baseline_import_no_op_when_empty():
    """Test import returns no_op when both baselines are None."""
    payload = {}  # No material_baseline and no prediction_baseline
    import_resp = client.post(
        "/api/v1/analyze/drift/baseline/import",
        headers=_headers(),
        json=payload,
    )
    assert import_resp.status_code == 200
    import_data = import_resp.json()
    assert import_data["status"] == "no_op"
    assert import_data["imported_materials"] is False
    assert import_data["imported_predictions"] is False


def test_drift_baseline_import_clear_material_baseline():
    """Test import with empty material list clears the baseline."""
    # First, set some baseline
    setup_payload = {
        "material_baseline": ["steel"],
        "material_baseline_ts": 1700000000,
    }
    client.post(
        "/api/v1/analyze/drift/baseline/import",
        headers=_headers(),
        json=setup_payload,
    )

    # Now clear it with empty list
    clear_payload = {
        "material_baseline": [],  # Empty list should clear the baseline
    }
    import_resp = client.post(
        "/api/v1/analyze/drift/baseline/import",
        headers=_headers(),
        json=clear_payload,
    )
    assert import_resp.status_code == 200
    import_data = import_resp.json()
    assert import_data["status"] == "ok"
    assert import_data["imported_materials"] is True
    assert import_data["material_baseline_ts"] is None  # Should be cleared


def test_drift_baseline_import_clear_prediction_baseline():
    """Test import with empty prediction list clears the baseline."""
    # First, set some baseline
    setup_payload = {
        "prediction_baseline": ["bolt", "washer"],
        "prediction_baseline_ts": 1700000000,
    }
    client.post(
        "/api/v1/analyze/drift/baseline/import",
        headers=_headers(),
        json=setup_payload,
    )

    # Now clear it with empty list
    clear_payload = {
        "prediction_baseline": [],  # Empty list should clear the baseline
    }
    import_resp = client.post(
        "/api/v1/analyze/drift/baseline/import",
        headers=_headers(),
        json=clear_payload,
    )
    assert import_resp.status_code == 200
    import_data = import_resp.json()
    assert import_data["status"] == "ok"
    assert import_data["imported_predictions"] is True
    assert import_data["prediction_baseline_ts"] is None  # Should be cleared


def test_drift_baseline_import_with_redis_success():
    """Test import persists to Redis when available."""
    mock_client = AsyncMock()
    mock_client.set = AsyncMock()

    with patch("src.utils.cache.get_client", return_value=mock_client):
        payload = {
            "material_baseline": ["steel", "aluminum"],
            "prediction_baseline": ["bracket"],
            "material_baseline_ts": 1700000000,
            "prediction_baseline_ts": 1700001234,
        }
        import_resp = client.post(
            "/api/v1/analyze/drift/baseline/import",
            headers=_headers(),
            json=payload,
        )
        assert import_resp.status_code == 200

    # Verify Redis was called
    assert mock_client.set.call_count >= 2  # At least material and prediction


def test_drift_baseline_import_with_redis_clear():
    """Test import deletes from Redis when clearing baselines."""
    mock_client = AsyncMock()
    mock_client.delete = AsyncMock()

    with patch("src.utils.cache.get_client", return_value=mock_client):
        # Clear both baselines
        clear_payload = {
            "material_baseline": [],
            "prediction_baseline": [],
        }
        import_resp = client.post(
            "/api/v1/analyze/drift/baseline/import",
            headers=_headers(),
            json=clear_payload,
        )
        assert import_resp.status_code == 200

    # Verify Redis delete was called
    assert mock_client.delete.call_count >= 2


def test_drift_baseline_import_redis_error_handled():
    """Test import handles Redis errors gracefully."""
    mock_client = AsyncMock()
    mock_client.set = AsyncMock(side_effect=Exception("Redis error"))

    with patch("src.utils.cache.get_client", return_value=mock_client):
        payload = {
            "material_baseline": ["steel"],
        }
        import_resp = client.post(
            "/api/v1/analyze/drift/baseline/import",
            headers=_headers(),
            json=payload,
        )
        # Should still return success even if Redis fails
        assert import_resp.status_code == 200
        import_data = import_resp.json()
        assert import_data["status"] == "ok"


def test_drift_baseline_import_only_materials():
    """Test import with only material baseline."""
    payload = {
        "material_baseline": ["copper", "bronze"],
    }
    import_resp = client.post(
        "/api/v1/analyze/drift/baseline/import",
        headers=_headers(),
        json=payload,
    )
    assert import_resp.status_code == 200
    import_data = import_resp.json()
    assert import_data["status"] == "ok"
    assert import_data["imported_materials"] is True
    assert import_data["imported_predictions"] is False
    assert import_data["material_baseline_ts"] is not None


def test_drift_baseline_import_only_predictions():
    """Test import with only prediction baseline."""
    payload = {
        "prediction_baseline": ["nut", "bolt"],
    }
    import_resp = client.post(
        "/api/v1/analyze/drift/baseline/import",
        headers=_headers(),
        json=payload,
    )
    assert import_resp.status_code == 200
    import_data = import_resp.json()
    assert import_data["status"] == "ok"
    assert import_data["imported_materials"] is False
    assert import_data["imported_predictions"] is True
    assert import_data["prediction_baseline_ts"] is not None
