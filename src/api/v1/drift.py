from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.dependencies import get_api_key

router = APIRouter()

# Drift state will be imported from analyze module to reuse existing structure
from src.api.v1 import analyze as analyze_module  # type: ignore


class DriftStatusResponse(BaseModel):
    material_current: Dict[str, int]
    material_baseline: Optional[Dict[str, int]] = None
    material_drift_score: Optional[float] = None
    prediction_current: Dict[str, int]
    prediction_baseline: Optional[Dict[str, int]] = None
    prediction_drift_score: Optional[float] = None
    baseline_min_count: int
    materials_total: int
    predictions_total: int
    status: str
    baseline_material_age: Optional[int] = None
    baseline_prediction_age: Optional[int] = None
    baseline_material_created_at: Optional[datetime] = None
    baseline_prediction_created_at: Optional[datetime] = None
    stale: Optional[bool] = None


class DriftResetResponse(BaseModel):
    status: str
    reset_material: bool
    reset_predictions: bool


class DriftBaselineStatusResponse(BaseModel):
    status: str
    material_age: Optional[int] = None
    prediction_age: Optional[int] = None
    material_created_at: Optional[datetime] = None
    prediction_created_at: Optional[datetime] = None
    stale: Optional[bool] = None
    max_age_seconds: int


@router.get("/drift", response_model=DriftStatusResponse)
async def drift_status(api_key: str = Depends(get_api_key)):
    from collections import Counter
    import os, time
    from src.utils.drift import compute_drift
    from src.utils.analysis_metrics import drift_baseline_refresh_total, drift_baseline_created_total
    _DRIFT_STATE: Dict[str, Any] = analyze_module._DRIFT_STATE  # type: ignore
    min_count = int(os.getenv("DRIFT_BASELINE_MIN_COUNT", "100"))
    max_age = int(os.getenv("DRIFT_BASELINE_MAX_AGE_SECONDS", "86400"))
    auto_refresh_enabled = os.getenv("DRIFT_BASELINE_AUTO_REFRESH", "1") == "1"

    mats = _DRIFT_STATE["materials"]
    preds = _DRIFT_STATE["predictions"]
    material_current_counts = dict(Counter(mats))
    prediction_current_counts = dict(Counter(preds))
    material_baseline_counts = dict(Counter(_DRIFT_STATE["baseline_materials"])) if _DRIFT_STATE["baseline_materials"] else None
    prediction_baseline_counts = dict(Counter(_DRIFT_STATE["baseline_predictions"])) if _DRIFT_STATE["baseline_predictions"] else None

    # Check material baseline age and soft refresh if stale
    mat_score = None
    if material_baseline_counts:
        material_age = int(time.time() - _DRIFT_STATE.get("baseline_materials_ts", 0))
        if auto_refresh_enabled and material_age > max_age and len(mats) >= min_count:
            # Soft refresh: replace with current distribution
            _DRIFT_STATE["baseline_materials"] = list(mats)
            _DRIFT_STATE["baseline_materials_ts"] = time.time()
            drift_baseline_refresh_total.labels(type="material", trigger="stale").inc()
            material_baseline_counts = dict(Counter(mats))  # Update for response
        mat_score = compute_drift(mats, _DRIFT_STATE["baseline_materials"])  # type: ignore
    else:
        if len(mats) >= min_count:
            _DRIFT_STATE["baseline_materials"] = list(mats)
            _DRIFT_STATE["baseline_materials_ts"] = time.time()
            drift_baseline_created_total.labels(type="material").inc()

    # Check prediction baseline age and soft refresh if stale
    pred_score = None
    if prediction_baseline_counts:
        prediction_age = int(time.time() - _DRIFT_STATE.get("baseline_predictions_ts", 0))
        if auto_refresh_enabled and prediction_age > max_age and len(preds) >= min_count:
            # Soft refresh: replace with current distribution
            _DRIFT_STATE["baseline_predictions"] = list(preds)
            _DRIFT_STATE["baseline_predictions_ts"] = time.time()
            drift_baseline_refresh_total.labels(type="prediction", trigger="stale").inc()
            prediction_baseline_counts = dict(Counter(preds))  # Update for response
        pred_score = compute_drift(preds, _DRIFT_STATE["baseline_predictions"])  # type: ignore
    else:
        if len(preds) >= min_count:
            _DRIFT_STATE["baseline_predictions"] = list(preds)
            _DRIFT_STATE["baseline_predictions_ts"] = time.time()
            drift_baseline_created_total.labels(type="prediction").inc()
    status = "baseline_pending" if (len(mats) < min_count or len(preds) < min_count) else "ok"
    baseline_material_age = None
    baseline_prediction_age = None
    baseline_material_created_at = None
    baseline_prediction_created_at = None
    if _DRIFT_STATE.get("baseline_materials_ts"):
        baseline_material_age = int(time.time() - _DRIFT_STATE["baseline_materials_ts"])
        baseline_material_created_at = datetime.fromtimestamp(_DRIFT_STATE["baseline_materials_ts"], tz=timezone.utc)
        if _DRIFT_STATE.get("baseline_materials_startup_mark") is None:
            drift_baseline_refresh_total.labels(type="material", trigger="startup").inc()
            _DRIFT_STATE["baseline_materials_startup_mark"] = True
    if _DRIFT_STATE.get("baseline_predictions_ts"):
        baseline_prediction_age = int(time.time() - _DRIFT_STATE["baseline_predictions_ts"])
        baseline_prediction_created_at = datetime.fromtimestamp(_DRIFT_STATE["baseline_predictions_ts"], tz=timezone.utc)
        if _DRIFT_STATE.get("baseline_predictions_startup_mark") is None:
            drift_baseline_refresh_total.labels(type="prediction", trigger="startup").inc()
            _DRIFT_STATE["baseline_predictions_startup_mark"] = True
    stale_flag = None
    max_age = int(os.getenv("DRIFT_BASELINE_MAX_AGE_SECONDS", "86400"))
    if baseline_material_age and baseline_material_age > max_age:
        stale_flag = True
    if baseline_prediction_age and baseline_prediction_age > max_age:
        stale_flag = True if stale_flag is None else True
    if stale_flag is None and (baseline_material_age or baseline_prediction_age):
        stale_flag = False
    return DriftStatusResponse(
        material_current=material_current_counts,
        material_baseline=material_baseline_counts,
        material_drift_score=mat_score,
        prediction_current=prediction_current_counts,
        prediction_baseline=prediction_baseline_counts,
        prediction_drift_score=pred_score,
        baseline_min_count=min_count,
        materials_total=len(mats),
        predictions_total=len(preds),
        status=status,
        baseline_material_age=baseline_material_age,
        baseline_prediction_age=baseline_prediction_age,
        baseline_material_created_at=baseline_material_created_at,
        baseline_prediction_created_at=baseline_prediction_created_at,
        stale=stale_flag,
    )


@router.post("/drift/reset", response_model=DriftResetResponse)
async def drift_reset(api_key: str = Depends(get_api_key)):
    from src.utils.analysis_metrics import drift_baseline_refresh_total
    _DRIFT_STATE: Dict[str, Any] = analyze_module._DRIFT_STATE  # type: ignore
    reset_material = bool(_DRIFT_STATE["baseline_materials"])
    reset_predictions = bool(_DRIFT_STATE["baseline_predictions"])

    # Record manual refresh metrics
    if reset_material:
        drift_baseline_refresh_total.labels(type="material", trigger="manual").inc()
    if reset_predictions:
        drift_baseline_refresh_total.labels(type="prediction", trigger="manual").inc()

    _DRIFT_STATE["baseline_materials"] = []
    _DRIFT_STATE["baseline_predictions"] = []
    _DRIFT_STATE["baseline_materials_ts"] = None
    _DRIFT_STATE["baseline_predictions_ts"] = None
    return DriftResetResponse(status="ok", reset_material=reset_material, reset_predictions=reset_predictions)


@router.get("/drift/baseline/status", response_model=DriftBaselineStatusResponse)
async def drift_baseline_status(api_key: str = Depends(get_api_key)):
    import os, time
    _DRIFT_STATE: Dict[str, Any] = analyze_module._DRIFT_STATE  # type: ignore
    max_age = int(os.getenv("DRIFT_BASELINE_MAX_AGE_SECONDS", "86400"))
    material_age = None
    prediction_age = None
    material_created_at = None
    prediction_created_at = None
    if _DRIFT_STATE.get("baseline_materials_ts"):
        material_age = int(time.time() - _DRIFT_STATE["baseline_materials_ts"])
        material_created_at = datetime.fromtimestamp(_DRIFT_STATE["baseline_materials_ts"], tz=timezone.utc)
    if _DRIFT_STATE.get("baseline_predictions_ts"):
        prediction_age = int(time.time() - _DRIFT_STATE["baseline_predictions_ts"])
        prediction_created_at = datetime.fromtimestamp(_DRIFT_STATE["baseline_predictions_ts"], tz=timezone.utc)
    stale_flag = None
    if material_age and material_age > max_age:
        stale_flag = True
    if prediction_age and prediction_age > max_age:
        stale_flag = True if stale_flag is None else True
    if stale_flag is None and (material_age or prediction_age):
        stale_flag = False
    status = "stale" if stale_flag else "ok"
    if material_age is None and prediction_age is None:
        status = "no_baseline"
    return DriftBaselineStatusResponse(
        status=status,
        material_age=material_age,
        prediction_age=prediction_age,
        material_created_at=material_created_at,
        prediction_created_at=prediction_created_at,
        stale=stale_flag,
        max_age_seconds=max_age,
    )


__all__ = ["router"]
