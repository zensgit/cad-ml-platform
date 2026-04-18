from __future__ import annotations

from typing import Any

import pytest

from src.core.analysis_drift_pipeline import run_analysis_drift_pipeline


class _DummyCacheClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def set(self, key: str, value: str) -> None:
        self.calls.append((key, value))


@pytest.mark.asyncio
async def test_run_analysis_drift_pipeline_creates_baselines_and_persists_cache() -> None:
    drift_state: dict[str, Any] = {
        "materials": [],
        "predictions": [],
        "baseline_materials": [],
        "baseline_predictions": [],
    }
    cache_client = _DummyCacheClient()
    observed_material: list[float] = []
    observed_prediction: list[float] = []

    await run_analysis_drift_pipeline(
        drift_state=drift_state,
        material="steel",
        classification_payload={"type": "bracket"},
        material_drift_observer=observed_material.append,
        prediction_drift_observer=observed_prediction.append,
        compute_drift_fn=lambda current, baseline: 1.0 if current == baseline else 0.0,
        cache_client_factory=lambda: cache_client,
        baseline_min_count=1,
    )

    assert drift_state["materials"] == ["steel"]
    assert drift_state["predictions"] == ["bracket"]
    assert drift_state["baseline_materials"] == ["steel"]
    assert drift_state["baseline_predictions"] == ["bracket"]
    assert observed_material == [1.0]
    assert observed_prediction == [1.0]
    assert ("baseline:material", '[\"steel\"]') in cache_client.calls
    assert ("baseline:class", '[\"bracket\"]') in cache_client.calls
    assert any(key == "baseline:material:ts" for key, _ in cache_client.calls)
    assert any(key == "baseline:class:ts" for key, _ in cache_client.calls)


@pytest.mark.asyncio
async def test_run_analysis_drift_pipeline_uses_ml_predicted_type_fallback() -> None:
    drift_state: dict[str, Any] = {
        "materials": ["steel"],
        "predictions": [],
        "baseline_materials": ["steel"],
        "baseline_predictions": [],
    }
    observed_prediction: list[float] = []

    await run_analysis_drift_pipeline(
        drift_state=drift_state,
        material=None,
        classification_payload={"ml_predicted_type": "plate"},
        material_drift_observer=lambda score: None,
        prediction_drift_observer=observed_prediction.append,
        compute_drift_fn=lambda current, baseline: float(len(current) + len(baseline)),
        baseline_min_count=1,
    )

    assert drift_state["materials"] == ["steel", "unknown"]
    assert drift_state["predictions"] == ["plate"]
    assert drift_state["baseline_predictions"] == ["plate"]
    assert observed_prediction == [2.0]
