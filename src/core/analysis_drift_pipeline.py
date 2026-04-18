"""Shared drift recording helper for analyze flows."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional


CacheClientFactory = Callable[[], Any]
DriftObserver = Callable[[float], Any]
ComputeDriftFn = Callable[[list[str], list[str]], float]


async def run_analysis_drift_pipeline(
    *,
    drift_state: Dict[str, Any],
    material: Optional[str],
    classification_payload: Optional[Mapping[str, Any]],
    material_drift_observer: DriftObserver,
    prediction_drift_observer: DriftObserver,
    compute_drift_fn: Optional[ComputeDriftFn] = None,
    cache_client_factory: Optional[CacheClientFactory] = None,
    baseline_min_count: Optional[int] = None,
) -> None:
    """Record one analysis result into the in-memory drift state."""
    if compute_drift_fn is None:
        from src.utils.drift import compute_drift as compute_drift_fn

    state = drift_state
    state.setdefault("materials", [])
    state.setdefault("predictions", [])
    state.setdefault("baseline_materials", [])
    state.setdefault("baseline_predictions", [])

    material_used = material or "unknown"
    state["materials"].append(material_used)

    cls_payload = classification_payload or {}
    predicted_label = cls_payload.get("type") or cls_payload.get("ml_predicted_type")
    if predicted_label:
        state["predictions"].append(str(predicted_label))

    min_count = (
        baseline_min_count
        if baseline_min_count is not None
        else int(os.getenv("DRIFT_BASELINE_MIN_COUNT", "100"))
    )

    if len(state["baseline_materials"]) == 0 and len(state["materials"]) >= min_count:
        state["baseline_materials"] = list(state["materials"])
        if cache_client_factory is not None:
            try:
                client = cache_client_factory()
                if client is not None:
                    now_ts = str(int(time.time()))
                    await client.set(
                        "baseline:material",
                        json.dumps(state["baseline_materials"]),
                    )
                    await client.set("baseline:material:ts", now_ts)
            except Exception:
                pass

    if len(state["baseline_predictions"]) == 0 and len(state["predictions"]) >= min_count:
        state["baseline_predictions"] = list(state["predictions"])
        if cache_client_factory is not None:
            try:
                client = cache_client_factory()
                if client is not None:
                    now_ts = str(int(time.time()))
                    await client.set(
                        "baseline:class",
                        json.dumps(state["baseline_predictions"]),
                    )
                    await client.set("baseline:class:ts", now_ts)
            except Exception:
                pass

    if state["baseline_materials"]:
        material_drift_observer(
            compute_drift_fn(state["materials"], state["baseline_materials"])
        )

    if state["baseline_predictions"]:
        prediction_drift_observer(
            compute_drift_fn(state["predictions"], state["baseline_predictions"])
        )


__all__ = ["run_analysis_drift_pipeline"]
