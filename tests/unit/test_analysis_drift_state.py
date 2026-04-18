from __future__ import annotations

from src.api.v1 import analyze
from src.core.analysis_drift_state import ANALYSIS_DRIFT_STATE, build_analysis_drift_state


def test_build_analysis_drift_state_returns_expected_shape() -> None:
    state = build_analysis_drift_state()

    assert state == {
        "materials": [],
        "predictions": [],
        "baseline_materials": [],
        "baseline_predictions": [],
        "baseline_materials_ts": None,
        "baseline_predictions_ts": None,
    }


def test_build_analysis_drift_state_returns_fresh_lists() -> None:
    first = build_analysis_drift_state()
    second = build_analysis_drift_state()

    first["materials"].append("steel")

    assert second["materials"] == []
    assert first is not second


def test_analyze_reexports_shared_drift_state() -> None:
    assert analyze._DRIFT_STATE is ANALYSIS_DRIFT_STATE
