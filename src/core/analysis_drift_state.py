from __future__ import annotations

from typing import Any, Dict


def build_analysis_drift_state() -> Dict[str, Any]:
    """Create the shared in-memory drift state shape used by analysis routes."""
    return {
        "materials": [],
        "predictions": [],
        "baseline_materials": [],
        "baseline_predictions": [],
        "baseline_materials_ts": None,
        "baseline_predictions_ts": None,
    }


ANALYSIS_DRIFT_STATE: Dict[str, Any] = build_analysis_drift_state()


__all__ = ["ANALYSIS_DRIFT_STATE", "build_analysis_drift_state"]
