from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Optional, Protocol


class SupportsAnalysisParallelOptions(Protocol):
    classify_parts: bool
    quality_check: bool
    process_recommendation: bool
    estimate_cost: bool


StageDurationObserver = Callable[[str, float], Any]
MetricObserver = Callable[[float], Any]


async def run_analysis_parallel_pipeline(
    *,
    analysis_id: str,
    analysis_options: SupportsAnalysisParallelOptions,
    doc: Any,
    features: Dict[str, Any],
    features_3d: Dict[str, Any],
    file_name: Optional[str],
    file_format: str,
    content: bytes,
    material: Optional[str],
    results: Dict[str, Any],
    classify_pipeline: Callable[..., Awaitable[Dict[str, Any]]],
    classify_part: Callable[..., Any],
    quality_pipeline: Callable[..., Awaitable[Dict[str, Any]]],
    check_quality: Callable[..., Any],
    process_pipeline: Callable[..., Awaitable[Dict[str, Any]]],
    recommend_process: Callable[..., Any],
    logger_instance: logging.Logger,
    classification_latency_observer: MetricObserver,
    dfm_latency_observer: MetricObserver,
    process_latency_observer: MetricObserver,
    process_rule_version_observer: Callable[[str], Any],
    cost_latency_observer: MetricObserver,
    stage_duration_observer: StageDurationObserver,
    parallel_enabled_setter: Callable[[int], Any],
    parallel_savings_observer: MetricObserver,
) -> Dict[str, float]:
    parallel_tasks = []

    if analysis_options.classify_parts:

        async def _run_classify() -> tuple[str, float]:
            t0 = time.time()
            cls_payload = await classify_pipeline(
                analysis_id=analysis_id,
                doc=doc,
                features=features,
                features_3d=features_3d,
                file_name=file_name,
                file_format=file_format,
                content=content,
                analysis_options=analysis_options,
                classify_part=classify_part,
                logger_instance=logger_instance,
            )
            results["classification"] = cls_payload
            dur = time.time() - t0
            classification_latency_observer(dur)
            return ("classify", dur)

        parallel_tasks.append(_run_classify())

    if analysis_options.quality_check:

        async def _run_quality() -> tuple[str, float]:
            t0 = time.time()
            results["quality"] = await quality_pipeline(
                doc=doc,
                features=features,
                features_3d=features_3d,
                check_quality=check_quality,
                classification_payload_getter=lambda: results.get("classification", {}),
                logger_instance=logger_instance,
                dfm_latency_observer=dfm_latency_observer,
            )
            return ("quality", time.time() - t0)

        parallel_tasks.append(_run_quality())

    if analysis_options.process_recommendation:

        async def _run_process() -> tuple[str, float]:
            t0 = time.time()
            process_context = await process_pipeline(
                doc=doc,
                features=features,
                features_3d=features_3d,
                recommend_process=recommend_process,
                material=material,
                estimate_cost=analysis_options.estimate_cost,
                classification_payload_getter=lambda: results.get("classification", {}),
                logger_instance=logger_instance,
                process_rule_version_observer=process_rule_version_observer,
                cost_latency_observer=cost_latency_observer,
            )
            results["process"] = process_context["process"]
            if process_context["cost_estimation"] is not None:
                results["cost_estimation"] = process_context["cost_estimation"]
            dur = time.time() - t0
            process_latency_observer(dur)
            return ("process", dur)

        parallel_tasks.append(_run_process())

    if not parallel_tasks:
        parallel_enabled_setter(0)
        return {}

    parallel_enabled_setter(1)
    wall_start = time.time()
    stage_results = await asyncio.gather(*parallel_tasks)
    wall_total = time.time() - wall_start

    stage_times: Dict[str, float] = {}
    serial_sum = 0.0
    for stage_name, indiv_dur in stage_results:
        stage_times[stage_name] = indiv_dur
        serial_sum += indiv_dur
        stage_duration_observer(stage_name, indiv_dur)

    savings = serial_sum - wall_total
    if savings < 0:
        savings = 0.0
    parallel_savings_observer(savings)
    return stage_times
