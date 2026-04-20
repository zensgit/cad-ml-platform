from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional


AsyncCallable = Callable[..., Awaitable[Any]]
SyncCallable = Callable[..., Any]


async def run_analysis_live_pipeline(
    *,
    file_name: str,
    content: bytes,
    options_raw: str,
    material: Optional[str],
    project_id: Optional[str],
    analysis_id: str,
    start_time: datetime,
    options_model_cls: type[Any],
    result_model_cls: type[Any],
    analyzer_factory: SyncCallable,
    run_preflight_fn: AsyncCallable,
    run_document_pipeline_fn: AsyncCallable,
    run_feature_pipeline_fn: AsyncCallable,
    run_parallel_pipeline_fn: AsyncCallable,
    attach_manufacturing_summary_fn: SyncCallable,
    build_manufacturing_summary_fn: SyncCallable,
    attach_drift_fn: AsyncCallable,
    drift_state: Dict[str, Any],
    run_drift_pipeline_fn: AsyncCallable,
    attach_vector_context_fn: AsyncCallable,
    run_vector_pipeline_fn: AsyncCallable,
    get_qdrant_store_fn: SyncCallable,
    compute_qdrant_similarity_fn: AsyncCallable,
    attach_ocr_payload_fn: AsyncCallable,
    run_ocr_pipeline_fn: AsyncCallable,
    build_response_fn: AsyncCallable,
    finalize_analysis_success_fn: AsyncCallable,
    classification_pipeline_fn: AsyncCallable,
    quality_pipeline_fn: AsyncCallable,
    process_pipeline_fn: AsyncCallable,
    logger_instance: Any,
) -> Any:
    from src.utils.analysis_metrics import (
        analysis_feature_vector_dimension,
        analysis_parallel_enabled,
        analysis_stage_duration_seconds,
        classification_latency_seconds,
        classification_prediction_drift_score,
        cost_estimation_latency_seconds,
        dfm_analysis_latency_seconds,
        material_distribution_drift_score,
        process_recommend_latency_seconds,
        process_rule_version_total,
        vector_store_material_total,
    )
    from src.utils.analysis_metrics import (
        analysis_parallel_savings_seconds,
    )

    started = time.time()
    stage_times: Dict[str, float] = {}

    preflight = await run_preflight_fn(
        file_name=file_name,
        options_raw=options_raw,
        content=content,
        analysis_id=analysis_id,
        timestamp=start_time,
        options_model_cls=options_model_cls,
    )
    analysis_options = preflight["analysis_options"]
    analysis_cache_key = preflight["analysis_cache_key"]
    cached_response = preflight["cached_response"]
    if cached_response is not None:
        logger_instance.info("Cache hit for %s", file_name)
        return result_model_cls(**cached_response)

    document_context = await run_document_pipeline_fn(
        file_name=file_name,
        content=content,
        started_at=started,
        material=material,
        project_id=project_id,
    )
    file_format = document_context["file_format"]
    doc = document_context["doc"]
    unified_data = document_context["unified_data"]
    stage_times["parse"] = document_context["parse_stage_duration"]

    analyzer = analyzer_factory()
    results: Dict[str, Any] = {}
    features: Dict[str, Any] = {
        "geometric": [],
        "semantic": [],
    }
    features_3d: Dict[str, Any] = {}

    feature_context = await run_feature_pipeline_fn(
        extract_features=analysis_options.extract_features,
        file_format=file_format,
        file_name=file_name,
        content=content,
        doc=doc,
        started_at=started,
        stage_times=stage_times,
        logger_instance=logger_instance,
    )
    features = feature_context["features"]
    features_3d = feature_context["features_3d"]
    results.update(feature_context["results_patch"])
    if feature_context["features_3d_stage_duration"] is not None:
        stage_times["features_3d"] = feature_context["features_3d_stage_duration"]
    if feature_context["features_stage_duration"] is not None:
        stage_times["features"] = feature_context["features_stage_duration"]
        analysis_stage_duration_seconds.labels(stage="features").observe(
            stage_times["features"]
        )

    stage_times.update(
        await run_parallel_pipeline_fn(
            analysis_id=analysis_id,
            analysis_options=analysis_options,
            doc=doc,
            features=features,
            features_3d=features_3d,
            file_name=file_name,
            file_format=file_format,
            content=content,
            material=material,
            results=results,
            classify_pipeline=classification_pipeline_fn,
            classify_part=analyzer.classify_part,
            quality_pipeline=quality_pipeline_fn,
            check_quality=analyzer.check_quality,
            process_pipeline=process_pipeline_fn,
            recommend_process=analyzer.recommend_process,
            logger_instance=logger_instance,
            classification_latency_observer=classification_latency_seconds.observe,
            dfm_latency_observer=dfm_analysis_latency_seconds.observe,
            process_latency_observer=process_recommend_latency_seconds.observe,
            process_rule_version_observer=lambda version: process_rule_version_total.labels(
                version=str(version)
            ).inc(),
            cost_latency_observer=cost_estimation_latency_seconds.observe,
            stage_duration_observer=lambda stage_name, indiv_dur: analysis_stage_duration_seconds.labels(
                stage=stage_name
            ).observe(indiv_dur),
            parallel_enabled_setter=analysis_parallel_enabled.set,
            parallel_savings_observer=analysis_parallel_savings_seconds.observe,
        )
    )

    attach_manufacturing_summary_fn(
        results=results,
        summary_builder=build_manufacturing_summary_fn,
        logger_instance=logger_instance,
    )

    await attach_drift_fn(
        drift_state=drift_state,
        material=material,
        classification_payload=results.get("classification", {}),
        drift_pipeline=run_drift_pipeline_fn,
        material_drift_observer=material_distribution_drift_score.observe,
        prediction_drift_observer=classification_prediction_drift_score.observe,
    )

    vector_context = await attach_vector_context_fn(
        analysis_id=analysis_id,
        doc=doc,
        features=features,
        features_3d=features_3d,
        material=material,
        classification_meta=results.get("classification", {}),
        calculate_similarity=analysis_options.calculate_similarity,
        reference_id=analysis_options.reference_id,
        results=results,
        stage_times=stage_times,
        started_at=started,
        vector_pipeline=run_vector_pipeline_fn,
        get_qdrant_store=get_qdrant_store_fn,
        compute_qdrant_similarity=compute_qdrant_similarity_fn,
        vector_material_observer=lambda m_used: vector_store_material_total.labels(
            material=m_used
        ).inc(),
        feature_dimension_observer=analysis_feature_vector_dimension.observe,
        similarity_stage_observer=lambda duration: analysis_stage_duration_seconds.labels(
            stage="similarity"
        ).observe(duration),
    )

    await attach_ocr_payload_fn(
        enable_ocr=analysis_options.enable_ocr,
        ocr_provider_strategy=analysis_options.ocr_provider,
        unified_data=unified_data,
        results=results,
        ocr_pipeline=run_ocr_pipeline_fn,
    )

    return await build_response_fn(
        result_model_cls=result_model_cls,
        finalize_analysis_success_fn=finalize_analysis_success_fn,
        analysis_id=analysis_id,
        start_time=start_time,
        file_name=file_name,
        file_format=file_format,
        results=results,
        doc=doc,
        stage_times=stage_times,
        analysis_cache_key=analysis_cache_key,
        vector_context=vector_context,
        material=material,
        unified_data=unified_data,
        logger_instance=logger_instance,
    )


__all__ = ["run_analysis_live_pipeline"]
