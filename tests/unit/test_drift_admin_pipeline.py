from __future__ import annotations

from datetime import timezone

import pytest

from src.core.drift_admin_pipeline import (
    run_drift_baseline_status_pipeline,
    run_drift_reset_pipeline,
    run_drift_status_pipeline,
)


def _state() -> dict:
    return {
        "materials": [],
        "predictions": [],
        "baseline_materials": [],
        "baseline_predictions": [],
        "baseline_materials_ts": None,
        "baseline_predictions_ts": None,
        "baseline_materials_startup_mark": None,
        "baseline_predictions_startup_mark": None,
    }


def test_run_drift_status_pipeline_creates_baselines_when_threshold_met():
    drift_state = _state()
    drift_state["materials"] = ["steel", "steel"]
    drift_state["predictions"] = ["simple", "simple"]

    payload = run_drift_status_pipeline(
        drift_state,
        min_count=2,
        max_age_seconds=3600,
        auto_refresh_enabled=False,
        now_ts=1_700_000_000,
    )

    assert payload["status"] == "ok"
    assert drift_state["baseline_materials"] == ["steel", "steel"]
    assert drift_state["baseline_predictions"] == ["simple", "simple"]
    assert payload["material_baseline"] is None
    assert payload["prediction_baseline"] is None


def test_run_drift_status_pipeline_supports_coarse_counts_and_stale_refresh():
    drift_state = _state()
    drift_state["materials"] = ["steel", "steel"]
    drift_state["predictions"] = ["complex_a", "complex_b"]
    drift_state["baseline_materials"] = ["aluminum", "aluminum"]
    drift_state["baseline_predictions"] = ["legacy_a", "legacy_b"]
    drift_state["baseline_materials_ts"] = 100
    drift_state["baseline_predictions_ts"] = 100

    payload = run_drift_status_pipeline(
        drift_state,
        include_prediction_coarse=True,
        coarse_label_normalizer=lambda value: value.split("_")[0],
        min_count=2,
        max_age_seconds=10,
        auto_refresh_enabled=True,
        now_ts=200,
    )

    assert drift_state["baseline_materials"] == ["steel", "steel"]
    assert drift_state["baseline_predictions"] == ["complex_a", "complex_b"]
    assert payload["prediction_current_coarse"] == {"complex": 2}
    assert payload["prediction_baseline_coarse"] == {"complex": 2}
    assert payload["stale"] is False
    assert payload["baseline_material_created_at"].tzinfo == timezone.utc


@pytest.mark.asyncio
async def test_run_drift_reset_pipeline_clears_baselines_and_cache():
    class FakeCacheClient:
        def __init__(self) -> None:
            self.deleted: list[str] = []

        async def delete(self, key: str) -> None:
            self.deleted.append(key)

    client = FakeCacheClient()
    drift_state = _state()
    drift_state["baseline_materials"] = ["steel"]
    drift_state["baseline_predictions"] = ["simple"]
    drift_state["baseline_materials_ts"] = 123
    drift_state["baseline_predictions_ts"] = 456

    payload = await run_drift_reset_pipeline(
        drift_state,
        clear_persisted_cache=True,
        cache_client_factory=lambda: client,
    )

    assert payload == {
        "status": "ok",
        "reset_material": True,
        "reset_predictions": True,
    }
    assert drift_state["baseline_materials"] == []
    assert drift_state["baseline_predictions"] == []
    assert client.deleted == ["baseline:material", "baseline:class"]


def test_run_drift_baseline_status_pipeline_handles_no_baseline_and_stale():
    empty_payload = run_drift_baseline_status_pipeline(
        _state(),
        max_age_seconds=10,
        now_ts=200,
    )
    assert empty_payload["status"] == "no_baseline"
    assert empty_payload["stale"] is None

    drift_state = _state()
    drift_state["baseline_materials_ts"] = 100
    drift_state["baseline_predictions_ts"] = 195
    payload = run_drift_baseline_status_pipeline(
        drift_state,
        max_age_seconds=10,
        now_ts=200,
    )

    assert payload["status"] == "stale"
    assert payload["material_age"] == 100
    assert payload["prediction_age"] == 5
    assert payload["stale"] is True
