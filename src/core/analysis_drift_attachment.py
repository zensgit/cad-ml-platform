from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Optional


DriftPipeline = Callable[..., Awaitable[None]]
MetricObserver = Callable[[float], Any]


async def attach_analysis_drift(
    *,
    drift_state: Dict[str, Any],
    material: Optional[str],
    classification_payload: Dict[str, Any],
    drift_pipeline: DriftPipeline,
    material_drift_observer: MetricObserver,
    prediction_drift_observer: MetricObserver,
) -> None:
    try:
        from src.utils.cache import get_client

        await drift_pipeline(
            drift_state=drift_state,
            material=material,
            classification_payload=classification_payload,
            material_drift_observer=material_drift_observer,
            prediction_drift_observer=prediction_drift_observer,
            cache_client_factory=get_client,
        )
    except Exception:
        return None
