"""Process recommendation and cost estimation helpers for analyze flows."""

from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

ProcessFallbackFn = Callable[[Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]
PayloadGetter = Callable[[], Mapping[str, Any]]
MetricObserver = Callable[[float], Any]
RuleVersionObserver = Callable[[str], Any]
ProcessRecommenderFactory = Callable[[], Any]
CostEstimatorFactory = Callable[[], Any]


def _default_get_process_recommender() -> Any:
    from src.core.process.ai_recommender import get_process_recommender

    return get_process_recommender()


def _default_get_cost_estimator() -> Any:
    from src.core.cost.estimator import get_cost_estimator

    return get_cost_estimator()


def _resolve_classification_payload(
    *,
    classification_payload: Optional[Mapping[str, Any]],
    classification_payload_getter: Optional[PayloadGetter],
) -> Dict[str, Any]:
    if classification_payload_getter is not None:
        return dict(classification_payload_getter() or {})
    return dict(classification_payload or {})


def _build_primary_process_payload(proc_result: Mapping[str, Any]) -> Dict[str, Any]:
    primary = proc_result.get("primary_recommendation")
    if isinstance(primary, Mapping) and primary:
        return dict(primary)
    if "process" in proc_result:
        return {
            "process": proc_result.get("process"),
            "method": proc_result.get("method", "standard"),
        }
    return {}


async def run_process_pipeline(
    *,
    doc: Any,
    features: Mapping[str, Any],
    features_3d: Optional[Mapping[str, Any]],
    recommend_process: ProcessFallbackFn,
    material: Optional[str] = None,
    estimate_cost: bool = False,
    classification_payload: Optional[Mapping[str, Any]] = None,
    classification_payload_getter: Optional[PayloadGetter] = None,
    logger_instance: Optional[logging.Logger] = None,
    process_recommender_factory: Optional[ProcessRecommenderFactory] = None,
    cost_estimator_factory: Optional[CostEstimatorFactory] = None,
    process_rule_version_observer: Optional[RuleVersionObserver] = None,
    cost_latency_observer: Optional[MetricObserver] = None,
) -> Dict[str, Any]:
    """Run process recommendation with optional cost estimation."""
    active_logger = logger_instance or logger
    process_payload: Any = {}
    proc_result: Dict[str, Any] = {}
    cost_payload: Optional[Dict[str, Any]] = None
    process_features = dict(features)
    effective_material = material or "steel"

    if features_3d:
        try:
            recommender = (process_recommender_factory or _default_get_process_recommender)()
            cls_payload = _resolve_classification_payload(
                classification_payload=classification_payload,
                classification_payload_getter=classification_payload_getter,
            )
            part_type = str(cls_payload.get("part_type") or "unknown")
            process_payload = recommender.recommend(
                dict(features_3d),
                part_type,
                effective_material,
            )
            proc_result = process_payload if isinstance(process_payload, dict) else {}
        except Exception as exc:  # noqa: BLE001
            active_logger.error("AI Process failed: %s", exc)
            process_payload = await recommend_process(doc, process_features)
            proc_result = process_payload if isinstance(process_payload, dict) else {}
    else:
        process_payload = await recommend_process(doc, process_features)
        if isinstance(process_payload, dict):
            proc_result = process_payload
            rule_version = process_payload.get("rule_version")
            if rule_version and process_rule_version_observer is not None:
                process_rule_version_observer(str(rule_version))

    if estimate_cost and features_3d:
        try:
            estimator = (cost_estimator_factory or _default_get_cost_estimator)()
            primary_proc = _build_primary_process_payload(proc_result)
            cost_start = time.time()
            cost_payload = estimator.estimate(
                dict(features_3d),
                primary_proc,
                material=effective_material,
            )
            if cost_latency_observer is not None:
                cost_latency_observer(time.time() - cost_start)
        except Exception as exc:  # noqa: BLE001
            active_logger.error("Cost estimation failed: %s", exc)

    return {
        "process": process_payload,
        "cost_estimation": cost_payload,
    }


__all__ = ["run_process_pipeline"]
