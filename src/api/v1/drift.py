from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.api.dependencies import get_api_key
from src.core.classification.coarse_labels import normalize_coarse_label
from src.core.drift_admin_pipeline import (
    run_drift_baseline_status_pipeline,
    run_drift_reset_pipeline,
    run_drift_status_pipeline,
)

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
    prediction_current_coarse: Dict[str, int] = Field(default_factory=dict)
    prediction_baseline_coarse: Optional[Dict[str, int]] = None
    prediction_coarse_drift_score: Optional[float] = None
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


@router.get(
    "/drift",
    response_model=DriftStatusResponse,
    operation_id="drift_status_v1",
)
async def drift_status(api_key: str = Depends(get_api_key)):
    _DRIFT_STATE: Dict[str, Any] = analyze_module._DRIFT_STATE  # type: ignore
    return DriftStatusResponse(
        **run_drift_status_pipeline(
            _DRIFT_STATE,
            include_prediction_coarse=True,
            coarse_label_normalizer=normalize_coarse_label,
        )
    )


@router.post(
    "/drift/reset",
    response_model=DriftResetResponse,
    operation_id="drift_reset_v1",
)
async def drift_reset(api_key: str = Depends(get_api_key)):
    _DRIFT_STATE: Dict[str, Any] = analyze_module._DRIFT_STATE  # type: ignore
    return DriftResetResponse(
        **await run_drift_reset_pipeline(
            _DRIFT_STATE,
            record_manual_refresh_metrics=True,
        )
    )


@router.get(
    "/drift/baseline/status",
    response_model=DriftBaselineStatusResponse,
    operation_id="drift_baseline_status_v1",
)
async def drift_baseline_status(api_key: str = Depends(get_api_key)):
    _DRIFT_STATE: Dict[str, Any] = analyze_module._DRIFT_STATE  # type: ignore
    return DriftBaselineStatusResponse(
        **run_drift_baseline_status_pipeline(_DRIFT_STATE)
    )


@router.post(
    "/drift/baseline/export",
    response_model=DriftBaselineExportResponse,
    operation_id="drift_baseline_export_v1",
)
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


@router.post(
    "/drift/baseline/import",
    response_model=DriftBaselineImportResponse,
    operation_id="drift_baseline_import_v1",
)
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
