from __future__ import annotations

from fastapi.testclient import TestClient

from src.main import app


def test_drift_route_delegates_to_shared_status_pipeline(monkeypatch):
    captured: dict[str, object] = {}

    def fake_pipeline(drift_state, **kwargs):
        captured["drift_state"] = drift_state
        captured["kwargs"] = kwargs
        return {
            "material_current": {},
            "material_baseline": None,
            "material_drift_score": None,
            "prediction_current": {},
            "prediction_baseline": None,
            "prediction_drift_score": None,
            "prediction_current_coarse": {"unknown": 1},
            "prediction_baseline_coarse": None,
            "prediction_coarse_drift_score": None,
            "baseline_min_count": 100,
            "materials_total": 0,
            "predictions_total": 0,
            "status": "baseline_pending",
            "baseline_material_age": None,
            "baseline_prediction_age": None,
            "baseline_material_created_at": None,
            "baseline_prediction_created_at": None,
            "stale": None,
        }

    monkeypatch.setattr("src.api.v1.drift.run_drift_status_pipeline", fake_pipeline)

    client = TestClient(app)
    response = client.get("/api/v1/analyze/drift", headers={"api-key": "test"})

    assert response.status_code == 200
    assert response.json()["prediction_current_coarse"] == {"unknown": 1}
    assert captured["kwargs"]["include_prediction_coarse"] is True
    assert callable(captured["kwargs"]["coarse_label_normalizer"])


def test_drift_reset_route_delegates_to_shared_reset_pipeline(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_pipeline(drift_state, **kwargs):
        captured["drift_state"] = drift_state
        captured["kwargs"] = kwargs
        return {
            "status": "ok",
            "reset_material": False,
            "reset_predictions": True,
        }

    monkeypatch.setattr("src.api.v1.drift.run_drift_reset_pipeline", fake_pipeline)

    client = TestClient(app)
    response = client.post("/api/v1/analyze/drift/reset", headers={"api-key": "test"})

    assert response.status_code == 200
    assert response.json()["reset_predictions"] is True
    assert captured["kwargs"]["record_manual_refresh_metrics"] is True


def test_drift_baseline_status_route_delegates_to_shared_pipeline(monkeypatch):
    captured: dict[str, object] = {}

    def fake_pipeline(drift_state, **kwargs):
        captured["drift_state"] = drift_state
        captured["kwargs"] = kwargs
        return {
            "status": "ok",
            "material_age": 1,
            "prediction_age": 2,
            "material_created_at": None,
            "prediction_created_at": None,
            "stale": False,
            "max_age_seconds": 86400,
        }

    monkeypatch.setattr(
        "src.api.v1.drift.run_drift_baseline_status_pipeline", fake_pipeline
    )

    client = TestClient(app)
    response = client.get(
        "/api/v1/analyze/drift/baseline/status", headers={"api-key": "test"}
    )

    assert response.status_code == 200
    assert response.json()["material_age"] == 1
    assert captured["kwargs"] == {}
