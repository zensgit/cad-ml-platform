"""Tests: drift baseline startup refresh trigger metric."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from src.api.health_utils import metrics_enabled
from src.main import app

client = TestClient(app)


def test_drift_startup_trigger_metric_present() -> None:
    if not metrics_enabled():
        pytest.skip("metrics client disabled in this environment")

    from src.api.v1 import analyze as analyze_module

    drift_state = analyze_module._DRIFT_STATE  # type: ignore
    drift_state["baseline_materials"] = ["steel"] * 10
    drift_state["baseline_predictions"] = ["bracket"] * 10
    drift_state["baseline_materials_ts"] = time.time() - 10
    drift_state["baseline_predictions_ts"] = time.time() - 10
    drift_state.pop("baseline_materials_startup_mark", None)
    drift_state.pop("baseline_predictions_startup_mark", None)

    status_response = client.get("/api/v1/analyze/drift/baseline/status")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["material_age"] is not None
    assert status_data["prediction_age"] is not None

    from src.utils.analysis_metrics import drift_baseline_refresh_total

    def counter_value(labels):  # type: ignore[no-untyped-def]
        for sample in drift_baseline_refresh_total.collect()[0].samples:
            if sample.labels == labels:
                return sample.value
        return 0.0

    before_material = counter_value({"type": "material", "trigger": "startup"})
    before_prediction = counter_value({"type": "prediction", "trigger": "startup"})

    drift_response = client.get("/api/v1/analyze/drift", headers={"X-API-Key": "test"})
    assert drift_response.status_code == 200

    after_material = counter_value({"type": "material", "trigger": "startup"})
    after_prediction = counter_value({"type": "prediction", "trigger": "startup"})
    assert after_material > before_material
    assert after_prediction > before_prediction
