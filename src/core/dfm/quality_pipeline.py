"""Quality / DFM orchestration helpers for analyze flows."""

from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

QualityFallbackFn = Callable[[Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]
PayloadGetter = Callable[[], Mapping[str, Any]]
MetricObserver = Callable[[float], Any]
DFMAnalyzerFactory = Callable[[], Any]
GeometryEngineFactory = Callable[[], Any]


def _default_get_geometry_engine() -> Any:
    from src.core.geometry.engine import get_geometry_engine

    return get_geometry_engine()


def _default_get_dfm_analyzer() -> Any:
    from src.core.dfm.analyzer import get_dfm_analyzer

    return get_dfm_analyzer()


def _resolve_classification_payload(
    *,
    classification_payload: Optional[Mapping[str, Any]],
    classification_payload_getter: Optional[PayloadGetter],
) -> Dict[str, Any]:
    if classification_payload_getter is not None:
        return dict(classification_payload_getter() or {})
    return dict(classification_payload or {})


async def run_quality_pipeline(
    *,
    doc: Any,
    features: Mapping[str, Any],
    features_3d: Optional[Mapping[str, Any]],
    check_quality: QualityFallbackFn,
    classification_payload: Optional[Mapping[str, Any]] = None,
    classification_payload_getter: Optional[PayloadGetter] = None,
    logger_instance: Optional[logging.Logger] = None,
    dfm_analyzer_factory: Optional[DFMAnalyzerFactory] = None,
    geometry_engine_factory: Optional[GeometryEngineFactory] = None,
    dfm_latency_observer: Optional[MetricObserver] = None,
) -> Dict[str, Any]:
    """Run DFM-first quality analysis while preserving legacy fallback behavior."""
    active_logger = logger_instance or logger
    quality_features = dict(features)

    if features_3d:
        try:
            dfm_start = time.time()
            if "thin_walls_detected" not in features_3d:
                (geometry_engine_factory or _default_get_geometry_engine)()

            dfm = (dfm_analyzer_factory or _default_get_dfm_analyzer)()
            cls_payload = _resolve_classification_payload(
                classification_payload=classification_payload,
                classification_payload_getter=classification_payload_getter,
            )
            part_type = str(cls_payload.get("part_type") or "unknown")
            dfm_result = dfm.analyze(dict(features_3d), part_type)
            elapsed = time.time() - dfm_start
            if dfm_latency_observer is not None:
                dfm_latency_observer(elapsed)
            return {
                "mode": "L4_DFM",
                "score": dfm_result["dfm_score"],
                "issues": dfm_result["issues"],
                "manufacturability": dfm_result["manufacturability"],
            }
        except Exception as exc:  # noqa: BLE001
            active_logger.error("DFM check failed: %s", exc)
            return await check_quality(doc, quality_features)

    quality = await check_quality(doc, quality_features)
    return {
        "score": quality["score"],
        "issues": quality.get("issues", []),
        "suggestions": quality.get("suggestions", []),
    }


__all__ = ["run_quality_pipeline"]
