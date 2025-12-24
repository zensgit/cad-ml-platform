from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

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


class DriftBaselineSnapshot(BaseModel):
    material_baseline: List[str] = Field(default_factory=list)
    prediction_baseline: List[str] = Field(default_factory=list)
    material_baseline_ts: Optional[int] = None
    prediction_baseline_ts: Optional[int] = None
    exported_at: datetime


class DriftBaselineExportResponse(BaseModel):
    status: str
    snapshot: DriftBaselineSnapshot


class DriftBaselineImportRequest(BaseModel):
    material_baseline: Optional[List[str]] = None
    prediction_baseline: Optional[List[str]] = None
    material_baseline_ts: Optional[int] = None
    prediction_baseline_ts: Optional[int] = None


class DriftBaselineImportResponse(BaseModel):
    status: str
    imported_materials: bool
    imported_predictions: bool
    material_baseline_ts: Optional[int] = None
    prediction_baseline_ts: Optional[int] = None


@router.get("/drift", response_model=DriftStatusResponse)
async def drift_status(api_key: str = Depends(get_api_key)):
    import os
    import time
    from collections import Counter

    from src.utils.analysis_metrics import (
        drift_baseline_created_total,
        drift_baseline_refresh_total,
    )
    from src.utils.drift import compute_drift

    _DRIFT_STATE: Dict[str, Any] = analyze_module._DRIFT_STATE  # type: ignore
    min_count = int(os.getenv("DRIFT_BASELINE_MIN_COUNT", "100"))
    max_age = int(os.getenv("DRIFT_BASELINE_MAX_AGE_SECONDS", "86400"))
    auto_refresh_enabled = os.getenv("DRIFT_BASELINE_AUTO_REFRESH", "1") == "1"

    mats = _DRIFT_STATE["materials"]
    preds = _DRIFT_STATE["predictions"]
    material_current_counts = dict(Counter(mats))
    prediction_current_counts = dict(Counter(preds))
    material_baseline_counts = (
        dict(Counter(_DRIFT_STATE["baseline_materials"]))
        if _DRIFT_STATE["baseline_materials"]
        else None
    )
    prediction_baseline_counts = (
        dict(Counter(_DRIFT_STATE["baseline_predictions"]))
        if _DRIFT_STATE["baseline_predictions"]
        else None
    )

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
        baseline_material_created_at = datetime.fromtimestamp(
            _DRIFT_STATE["baseline_materials_ts"], tz=timezone.utc
        )
        if _DRIFT_STATE.get("baseline_materials_startup_mark") is None:
            drift_baseline_refresh_total.labels(type="material", trigger="startup").inc()
            _DRIFT_STATE["baseline_materials_startup_mark"] = True
    if _DRIFT_STATE.get("baseline_predictions_ts"):
        baseline_prediction_age = int(time.time() - _DRIFT_STATE["baseline_predictions_ts"])
        baseline_prediction_created_at = datetime.fromtimestamp(
            _DRIFT_STATE["baseline_predictions_ts"], tz=timezone.utc
        )
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
    return DriftResetResponse(
        status="ok", reset_material=reset_material, reset_predictions=reset_predictions
    )


@router.get("/drift/baseline/status", response_model=DriftBaselineStatusResponse)
async def drift_baseline_status(api_key: str = Depends(get_api_key)):
    import os
    import time

    _DRIFT_STATE: Dict[str, Any] = analyze_module._DRIFT_STATE  # type: ignore
    max_age = int(os.getenv("DRIFT_BASELINE_MAX_AGE_SECONDS", "86400"))
    material_age = None
    prediction_age = None
    material_created_at = None
    prediction_created_at = None
    if _DRIFT_STATE.get("baseline_materials_ts"):
        material_age = int(time.time() - _DRIFT_STATE["baseline_materials_ts"])
        material_created_at = datetime.fromtimestamp(
            _DRIFT_STATE["baseline_materials_ts"], tz=timezone.utc
        )
    if _DRIFT_STATE.get("baseline_predictions_ts"):
        prediction_age = int(time.time() - _DRIFT_STATE["baseline_predictions_ts"])
        prediction_created_at = datetime.fromtimestamp(
            _DRIFT_STATE["baseline_predictions_ts"], tz=timezone.utc
        )
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


@router.post("/drift/baseline/export", response_model=DriftBaselineExportResponse)
async def drift_baseline_export(api_key: str = Depends(get_api_key)):
    _DRIFT_STATE: Dict[str, Any] = analyze_module._DRIFT_STATE  # type: ignore
    material_baseline = list(_DRIFT_STATE.get("baseline_materials") or [])
    prediction_baseline = list(_DRIFT_STATE.get("baseline_predictions") or [])
    material_ts = _DRIFT_STATE.get("baseline_materials_ts")
    prediction_ts = _DRIFT_STATE.get("baseline_predictions_ts")
    status = "empty" if not material_baseline and not prediction_baseline else "ok"
    snapshot = DriftBaselineSnapshot(
        material_baseline=material_baseline,
        prediction_baseline=prediction_baseline,
        material_baseline_ts=int(material_ts) if material_ts else None,
        prediction_baseline_ts=int(prediction_ts) if prediction_ts else None,
        exported_at=datetime.now(tz=timezone.utc),
    )
    return DriftBaselineExportResponse(status=status, snapshot=snapshot)


@router.post("/drift/baseline/import", response_model=DriftBaselineImportResponse)
async def drift_baseline_import(
    payload: DriftBaselineImportRequest,
    api_key: str = Depends(get_api_key),
):
    import json
    import time

    from src.utils.analysis_metrics import drift_baseline_refresh_total
    from src.utils.cache import get_client

    _DRIFT_STATE: Dict[str, Any] = analyze_module._DRIFT_STATE  # type: ignore
    imported_materials = payload.material_baseline is not None
    imported_predictions = payload.prediction_baseline is not None
    if not imported_materials and not imported_predictions:
        return DriftBaselineImportResponse(
            status="no_op",
            imported_materials=False,
            imported_predictions=False,
            material_baseline_ts=_DRIFT_STATE.get("baseline_materials_ts"),
            prediction_baseline_ts=_DRIFT_STATE.get("baseline_predictions_ts"),
        )

    now_ts = int(time.time())
    material_ts = _DRIFT_STATE.get("baseline_materials_ts")
    prediction_ts = _DRIFT_STATE.get("baseline_predictions_ts")

    if imported_materials:
        material_baseline = list(payload.material_baseline or [])
        _DRIFT_STATE["baseline_materials"] = material_baseline
        if material_baseline:
            material_ts = payload.material_baseline_ts or now_ts
        else:
            material_ts = None
        _DRIFT_STATE["baseline_materials_ts"] = material_ts
        drift_baseline_refresh_total.labels(type="material", trigger="manual").inc()

    if imported_predictions:
        prediction_baseline = list(payload.prediction_baseline or [])
        _DRIFT_STATE["baseline_predictions"] = prediction_baseline
        if prediction_baseline:
            prediction_ts = payload.prediction_baseline_ts or now_ts
        else:
            prediction_ts = None
        _DRIFT_STATE["baseline_predictions_ts"] = prediction_ts
        drift_baseline_refresh_total.labels(type="prediction", trigger="manual").inc()

    # Persist to Redis if available (best effort)
    client = get_client()
    if client is not None:
        try:
            if imported_materials:
                if _DRIFT_STATE["baseline_materials"]:
                    await client.set(
                        "baseline:material", json.dumps(_DRIFT_STATE["baseline_materials"])
                    )
                    await client.set("baseline:material:ts", str(material_ts or now_ts))
                else:
                    await client.delete("baseline:material")
                    await client.delete("baseline:material:ts")
            if imported_predictions:
                if _DRIFT_STATE["baseline_predictions"]:
                    await client.set(
                        "baseline:class", json.dumps(_DRIFT_STATE["baseline_predictions"])
                    )
                    await client.set("baseline:class:ts", str(prediction_ts or now_ts))
                else:
                    await client.delete("baseline:class")
                    await client.delete("baseline:class:ts")
        except Exception:
            pass

    return DriftBaselineImportResponse(
        status="ok",
        imported_materials=imported_materials,
        imported_predictions=imported_predictions,
        material_baseline_ts=material_ts,
        prediction_baseline_ts=prediction_ts,
    )


__all__ = ["router"]
