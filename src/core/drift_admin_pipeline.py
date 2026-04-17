from __future__ import annotations

import os
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional


def _safe_inc(counter, **labels: str) -> None:
    try:
        counter.labels(**labels).inc()
    except Exception:
        pass


def _safe_set(gauge, value: int) -> None:
    try:
        gauge.set(value)
    except Exception:
        pass


def run_drift_status_pipeline(
    drift_state: Dict[str, Any],
    *,
    include_prediction_coarse: bool = False,
    coarse_label_normalizer: Optional[Callable[[str], Optional[str]]] = None,
    min_count: Optional[int] = None,
    max_age_seconds: Optional[int] = None,
    auto_refresh_enabled: Optional[bool] = None,
    now_ts: Optional[float] = None,
) -> Dict[str, Any]:
    from src.utils.analysis_metrics import (
        baseline_material_age_seconds,
        baseline_prediction_age_seconds,
        drift_baseline_created_total,
        drift_baseline_refresh_total,
    )
    from src.utils.drift import compute_drift

    min_count = (
        min_count
        if min_count is not None
        else int(os.getenv("DRIFT_BASELINE_MIN_COUNT", "100"))
    )
    max_age_seconds = (
        max_age_seconds
        if max_age_seconds is not None
        else int(os.getenv("DRIFT_BASELINE_MAX_AGE_SECONDS", "86400"))
    )
    auto_refresh_enabled = (
        auto_refresh_enabled
        if auto_refresh_enabled is not None
        else os.getenv("DRIFT_BASELINE_AUTO_REFRESH", "1") == "1"
    )
    now_ts = now_ts if now_ts is not None else time.time()

    mats = list(drift_state.get("materials") or [])
    preds = list(drift_state.get("predictions") or [])
    material_current_counts = dict(Counter(mats))
    prediction_current_counts = dict(Counter(preds))

    material_baseline = list(drift_state.get("baseline_materials") or [])
    prediction_baseline = list(drift_state.get("baseline_predictions") or [])
    material_baseline_counts = dict(Counter(material_baseline)) if material_baseline else None
    prediction_baseline_counts = (
        dict(Counter(prediction_baseline)) if prediction_baseline else None
    )

    coarse_preds: Optional[list[str]] = None
    prediction_current_coarse_counts: Dict[str, int] = {}
    prediction_baseline_coarse_counts: Optional[Dict[str, int]] = None
    if include_prediction_coarse:
        normalize = coarse_label_normalizer or (lambda value: value)
        coarse_preds = [normalize(pred) or "unknown" for pred in preds]
        prediction_current_coarse_counts = dict(Counter(coarse_preds))
        if prediction_baseline:
            prediction_baseline_coarse_counts = dict(
                Counter(normalize(pred) or "unknown" for pred in prediction_baseline)
            )

    mat_score = None
    if material_baseline_counts:
        material_age = int(now_ts - (drift_state.get("baseline_materials_ts") or 0))
        if auto_refresh_enabled and material_age > max_age_seconds and len(mats) >= min_count:
            drift_state["baseline_materials"] = list(mats)
            drift_state["baseline_materials_ts"] = now_ts
            material_baseline = list(mats)
            material_baseline_counts = dict(Counter(material_baseline))
            _safe_inc(drift_baseline_refresh_total, type="material", trigger="stale")
        mat_score = compute_drift(mats, drift_state["baseline_materials"])
    elif len(mats) >= min_count:
        drift_state["baseline_materials"] = list(mats)
        drift_state["baseline_materials_ts"] = now_ts
        _safe_inc(drift_baseline_created_total, type="material")

    pred_score = None
    pred_coarse_score = None
    if prediction_baseline_counts:
        prediction_age = int(now_ts - (drift_state.get("baseline_predictions_ts") or 0))
        if auto_refresh_enabled and prediction_age > max_age_seconds and len(preds) >= min_count:
            drift_state["baseline_predictions"] = list(preds)
            drift_state["baseline_predictions_ts"] = now_ts
            prediction_baseline = list(preds)
            prediction_baseline_counts = dict(Counter(prediction_baseline))
            _safe_inc(drift_baseline_refresh_total, type="prediction", trigger="stale")
            if include_prediction_coarse:
                prediction_baseline_coarse_counts = dict(
                    Counter(coarse_preds or [])
                )
        pred_score = compute_drift(preds, drift_state["baseline_predictions"])
        if include_prediction_coarse and coarse_preds is not None:
            normalize = coarse_label_normalizer or (lambda value: value)
            pred_coarse_score = compute_drift(
                coarse_preds,
                [
                    normalize(pred) or "unknown"
                    for pred in drift_state["baseline_predictions"]
                ],
            )
    elif len(preds) >= min_count:
        drift_state["baseline_predictions"] = list(preds)
        drift_state["baseline_predictions_ts"] = now_ts
        _safe_inc(drift_baseline_created_total, type="prediction")

    status = "baseline_pending" if (len(mats) < min_count or len(preds) < min_count) else "ok"

    baseline_material_age = None
    baseline_prediction_age = None
    baseline_material_created_at = None
    baseline_prediction_created_at = None
    stale_flag = None

    if drift_state.get("baseline_materials_ts"):
        baseline_material_age = int(now_ts - drift_state["baseline_materials_ts"])
        baseline_material_created_at = datetime.fromtimestamp(
            drift_state["baseline_materials_ts"], tz=timezone.utc
        )
        _safe_set(baseline_material_age_seconds, baseline_material_age)
        if drift_state.get("baseline_materials_startup_mark") is None:
            _safe_inc(drift_baseline_refresh_total, type="material", trigger="startup")
            drift_state["baseline_materials_startup_mark"] = True

    if drift_state.get("baseline_predictions_ts"):
        baseline_prediction_age = int(now_ts - drift_state["baseline_predictions_ts"])
        baseline_prediction_created_at = datetime.fromtimestamp(
            drift_state["baseline_predictions_ts"], tz=timezone.utc
        )
        _safe_set(baseline_prediction_age_seconds, baseline_prediction_age)
        if drift_state.get("baseline_predictions_startup_mark") is None:
            _safe_inc(drift_baseline_refresh_total, type="prediction", trigger="startup")
            drift_state["baseline_predictions_startup_mark"] = True

    if baseline_material_age is not None and baseline_material_age > max_age_seconds:
        stale_flag = True
    if baseline_prediction_age is not None and baseline_prediction_age > max_age_seconds:
        stale_flag = True
    if stale_flag is None and (
        baseline_material_age is not None or baseline_prediction_age is not None
    ):
        stale_flag = False

    payload: Dict[str, Any] = {
        "material_current": material_current_counts,
        "material_baseline": material_baseline_counts,
        "material_drift_score": mat_score,
        "prediction_current": prediction_current_counts,
        "prediction_baseline": prediction_baseline_counts,
        "prediction_drift_score": pred_score,
        "baseline_min_count": min_count,
        "materials_total": len(mats),
        "predictions_total": len(preds),
        "status": status,
        "baseline_material_age": baseline_material_age,
        "baseline_prediction_age": baseline_prediction_age,
        "baseline_material_created_at": baseline_material_created_at,
        "baseline_prediction_created_at": baseline_prediction_created_at,
        "stale": stale_flag,
    }
    if include_prediction_coarse:
        payload.update(
            {
                "prediction_current_coarse": prediction_current_coarse_counts,
                "prediction_baseline_coarse": prediction_baseline_coarse_counts,
                "prediction_coarse_drift_score": pred_coarse_score,
            }
        )
    return payload


async def run_drift_reset_pipeline(
    drift_state: Dict[str, Any],
    *,
    record_manual_refresh_metrics: bool = False,
    clear_persisted_cache: bool = False,
    cache_client_factory: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    reset_material = bool(drift_state.get("baseline_materials"))
    reset_predictions = bool(drift_state.get("baseline_predictions"))

    if record_manual_refresh_metrics:
        from src.utils.analysis_metrics import drift_baseline_refresh_total

        if reset_material:
            _safe_inc(drift_baseline_refresh_total, type="material", trigger="manual")
        if reset_predictions:
            _safe_inc(drift_baseline_refresh_total, type="prediction", trigger="manual")

    drift_state["baseline_materials"] = []
    drift_state["baseline_predictions"] = []
    drift_state["baseline_materials_ts"] = None
    drift_state["baseline_predictions_ts"] = None

    if clear_persisted_cache and cache_client_factory is not None:
        try:
            client = cache_client_factory()
            if client is not None:
                await client.delete("baseline:material")
                await client.delete("baseline:class")
        except Exception:
            pass

    return {
        "status": "ok",
        "reset_material": reset_material,
        "reset_predictions": reset_predictions,
    }


def run_drift_baseline_status_pipeline(
    drift_state: Dict[str, Any],
    *,
    max_age_seconds: Optional[int] = None,
    now_ts: Optional[float] = None,
) -> Dict[str, Any]:
    max_age_seconds = (
        max_age_seconds
        if max_age_seconds is not None
        else int(os.getenv("DRIFT_BASELINE_MAX_AGE_SECONDS", "86400"))
    )
    now_ts = now_ts if now_ts is not None else time.time()

    material_age = None
    prediction_age = None
    material_created_at = None
    prediction_created_at = None

    if drift_state.get("baseline_materials_ts"):
        material_age = int(now_ts - drift_state["baseline_materials_ts"])
        material_created_at = datetime.fromtimestamp(
            drift_state["baseline_materials_ts"], tz=timezone.utc
        )
    if drift_state.get("baseline_predictions_ts"):
        prediction_age = int(now_ts - drift_state["baseline_predictions_ts"])
        prediction_created_at = datetime.fromtimestamp(
            drift_state["baseline_predictions_ts"], tz=timezone.utc
        )

    stale_flag = None
    if material_age and material_age > max_age_seconds:
        stale_flag = True
    if prediction_age and prediction_age > max_age_seconds:
        stale_flag = True if stale_flag is None else stale_flag or True
    if stale_flag is None and (material_age or prediction_age):
        stale_flag = False

    status = "stale" if stale_flag else "ok"
    if material_age is None and prediction_age is None:
        status = "no_baseline"

    return {
        "status": status,
        "material_age": material_age,
        "prediction_age": prediction_age,
        "material_created_at": material_created_at,
        "prediction_created_at": prediction_created_at,
        "stale": stale_flag,
        "max_age_seconds": max_age_seconds,
    }


__all__ = [
    "run_drift_baseline_status_pipeline",
    "run_drift_reset_pipeline",
    "run_drift_status_pipeline",
]
