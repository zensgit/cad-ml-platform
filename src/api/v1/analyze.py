"""
CAD文件分析API端点
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from src.api.dependencies import get_api_key
from src.api.v1.analyze_legacy_redirects import router as legacy_redirect_router
from src.api.v1.analyze_live_models import (
    AnalysisOptions,
    AnalysisResult,
    BatchClassifyResponse,
    BatchClassifyResultItem,
    SimilarityQuery,
    SimilarityResult,
    SimilarityTopKItem,
    SimilarityTopKQuery,
    SimilarityTopKResponse,
)
from src.api.v1.analyze_vector_compat import router as vector_compat_router
from src.api.v1.process import process_rules_audit
from src.api.v1.analyze_shadow_compat import (
    _build_graph2d_soft_override_suggestion,
    _enrich_graph2d_prediction,
    _graph2d_is_drawing_type,
    _resolve_history_sequence_file_path,
)
from src.core.analysis_batch_pipeline import run_batch_analysis
from src.core.analysis_drift_attachment import attach_analysis_drift
from src.core.analysis_drift_pipeline import run_analysis_drift_pipeline
from src.core.analysis_drift_state import ANALYSIS_DRIFT_STATE as _DRIFT_STATE
from src.core.analysis_error_handling import (
    handle_analysis_http_exception,
    handle_analysis_options_json_error,
    handle_analysis_unexpected_exception,
)
from src.core.analysis_manufacturing_summary import (
    attach_manufacturing_decision_summary,
)
from src.core.analysis_parallel_pipeline import run_analysis_parallel_pipeline
from src.core.analysis_preflight import run_analysis_request_preflight
from src.core.analysis_result_envelope import finalize_analysis_success
from src.core.analysis_vector_attachment import attach_analysis_vector_context
from src.core.analyzer import CADAnalyzer
from src.core.classification import (
    extract_label_decision_contract,
    run_batch_classify_pipeline,
    run_classification_pipeline,
)
from src.core.document_pipeline import run_document_pipeline
from src.core.dfm.quality_pipeline import run_quality_pipeline
from src.core.feature_pipeline import run_feature_pipeline
from src.core.ocr.analysis_ocr_pipeline import run_analysis_ocr_pipeline
from src.core.process import (
    build_manufacturing_decision_summary,
    run_process_pipeline,
)
from src.core.qdrant_store_helper import (
    get_qdrant_store_or_none as _get_qdrant_store_or_none,
)
from src.core.qdrant_similarity_helper import compute_qdrant_cosine_similarity
from src.core.vector_query_pipeline import (
    run_similarity_query_pipeline,
    run_similarity_topk_pipeline,
)
from src.core.vector_pipeline import run_vector_pipeline
from src.models.cad_document import CadDocument
from src.utils.analysis_metrics import (
    analysis_error_code_total,
    analysis_errors_total,
    analysis_feature_vector_dimension,
    analysis_material_usage_total,
    analysis_parallel_enabled,
    analysis_parse_latency_budget_ratio,
    analysis_rejections_total,
    analysis_requests_total,
    analysis_stage_duration_seconds,
    classification_latency_seconds,
    classification_prediction_drift_score,
    cost_estimation_latency_seconds,
    dfm_analysis_latency_seconds,
    material_distribution_drift_score,
    process_recommend_latency_seconds,
    process_rule_version_total,
    vector_query_latency_seconds,
    vector_store_material_total,
)
from src.utils.analysis_result_store import load_analysis_result
from src.utils.cache import get_cached_result, set_cache

logger = logging.getLogger(__name__)

router = APIRouter()
router.include_router(legacy_redirect_router)
router.include_router(vector_compat_router)


@router.post("/", response_model=AnalysisResult)
async def analyze_cad_file(
    file: UploadFile = File(..., description="CAD文件"),
    options: str = Form(default='{"extract_features": true, "classify_parts": true}'),
    material: Optional[str] = Form(default=None, description="材料标注"),
    project_id: Optional[str] = Form(default=None, description="项目ID"),
    api_key: str = Depends(get_api_key),
):
    """
    分析CAD文件

    支持格式:
    - DXF (AutoCAD)
    - DWG (通过转换)
    - STEP
    - IGES
    - STL
    """
    start_time = datetime.now(timezone.utc)
    analysis_id = str(uuid.uuid4())

    stage_times: Dict[str, float] = {}
    import time

    started = time.time()
    try:
        content = await file.read()
        preflight = await run_analysis_request_preflight(
            file_name=file.filename,
            options_raw=options,
            content=content,
            analysis_id=analysis_id,
            timestamp=start_time,
            options_model_cls=AnalysisOptions,
        )
        analysis_options = preflight["analysis_options"]
        analysis_cache_key = preflight["analysis_cache_key"]
        cached_response = preflight["cached_response"]
        if cached_response is not None:
            logger.info("Cache hit for %s", file.filename)
            return AnalysisResult(**cached_response)

        document_context = await run_document_pipeline(
            file_name=file.filename,
            content=content,
            started_at=started,
            material=material,
            project_id=project_id,
        )
        file_format = document_context["file_format"]
        doc = document_context["doc"]
        unified_data = document_context["unified_data"]
        stage_times["parse"] = document_context["parse_stage_duration"]

        # 创建分析器
        analyzer = CADAnalyzer()
        results: Dict[str, Any] = {}
        features: Dict[str, Any] = {
            "geometric": [],
            "semantic": [],
        }  # ensure defined even if skipped
        features_3d: Dict[str, Any] = {}

        feature_context = await run_feature_pipeline(
            extract_features=analysis_options.extract_features,
            file_format=file_format,
            file_name=file.filename,
            content=content,
            doc=doc,
            started_at=started,
            stage_times=stage_times,
            logger_instance=logger,
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

        from src.utils.analysis_metrics import analysis_parallel_savings_seconds

        stage_times.update(
            await run_analysis_parallel_pipeline(
                analysis_id=analysis_id,
                analysis_options=analysis_options,
                doc=doc,
                features=features,
                features_3d=features_3d,
                file_name=file.filename,
                file_format=file_format,
                content=content,
                material=material,
                results=results,
                classify_pipeline=run_classification_pipeline,
                classify_part=analyzer.classify_part,
                quality_pipeline=run_quality_pipeline,
                check_quality=analyzer.check_quality,
                process_pipeline=run_process_pipeline,
                recommend_process=analyzer.recommend_process,
                logger_instance=logger,
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

        attach_manufacturing_decision_summary(
            results=results,
            summary_builder=build_manufacturing_decision_summary,
            logger_instance=logger,
        )

        await attach_analysis_drift(
            drift_state=_DRIFT_STATE,
            material=material,
            classification_payload=results.get("classification", {}),
            drift_pipeline=run_analysis_drift_pipeline,
            material_drift_observer=material_distribution_drift_score.observe,
            prediction_drift_observer=classification_prediction_drift_score.observe,
        )

        vector_context = await attach_analysis_vector_context(
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
            vector_pipeline=run_vector_pipeline,
            get_qdrant_store=_get_qdrant_store_or_none,
            compute_qdrant_similarity=compute_qdrant_cosine_similarity,
            vector_material_observer=lambda m_used: vector_store_material_total.labels(
                material=m_used
            ).inc(),
            feature_dimension_observer=analysis_feature_vector_dimension.observe,
            similarity_stage_observer=lambda duration: analysis_stage_duration_seconds.labels(
                stage="similarity"
            ).observe(duration),
        )

        ocr_payload = await run_analysis_ocr_pipeline(
            enable_ocr=analysis_options.enable_ocr,
            ocr_provider_strategy=analysis_options.ocr_provider,
            unified_data=unified_data,
        )
        if ocr_payload is not None:
            results["ocr"] = ocr_payload

        response_payload = await finalize_analysis_success(
            analysis_id=analysis_id,
            start_time=start_time,
            file_name=file.filename,
            file_format=file_format,
            results=results,
            doc=doc,
            stage_times=stage_times,
            analysis_cache_key=analysis_cache_key,
            vector_context=vector_context,
            material=material,
            unified_data=unified_data,
            logger_instance=logger,
        )
        return AnalysisResult(**response_payload)

    except json.JSONDecodeError:
        handle_analysis_options_json_error()
    except HTTPException as he:
        handle_analysis_http_exception(he)
    except Exception as e:
        handle_analysis_unexpected_exception(
            file_name=file.filename,
            exc=e,
            logger_instance=logger,
        )


@router.post("/batch")
async def batch_analyze(
    files: list[UploadFile] = File(..., description="多个CAD文件"),
    options: str = Form(default='{"extract_features": true}'),
    api_key: str = Depends(get_api_key),
):
    """批量分析CAD文件"""
    return await run_batch_analysis(
        files=files,
        options=options,
        api_key=api_key,
        analyze_file_fn=analyze_cad_file,
    )


@router.post("/similarity", response_model=SimilarityResult)
async def similarity_query(
    payload: SimilarityQuery, api_key: str = Depends(get_api_key)
):
    """在已存在的向量之间计算相似度。"""
    result = await run_similarity_query_pipeline(
        payload,
        get_qdrant_store=_get_qdrant_store_or_none,
        error_recorder=lambda code: analysis_error_code_total.labels(code=code).inc(),
    )
    return SimilarityResult(**result)


@router.post("/similarity/topk", response_model=SimilarityTopKResponse)
async def similarity_topk(
    payload: SimilarityTopKQuery, api_key: str = Depends(get_api_key)
):
    """基于已存储向量的 Top-K 相似检索。"""
    result = await run_similarity_topk_pipeline(
        payload,
        get_qdrant_store=_get_qdrant_store_or_none,
        error_recorder=lambda code: analysis_error_code_total.labels(code=code).inc(),
        latency_observer=lambda backend, duration: vector_query_latency_seconds.labels(
            backend=backend
        ).observe(duration),
    )
    return SimilarityTopKResponse(**result)


@router.post("/batch-classify", response_model=BatchClassifyResponse)
async def batch_classify(
    files: List[UploadFile] = File(..., description="CAD文件列表(DXF/DWG)"),
    max_workers: Optional[int] = Form(default=None, description="并行工作线程数"),
    api_key: str = Depends(get_api_key),
):
    """
    批量分类CAD文件

    使用V16超级集成分类器并行处理多个文件，支持DXF和DWG格式。
    相比逐个调用，批量处理可获得约3倍性能提升。
    """
    result = await run_batch_classify_pipeline(
        files=files,
        max_workers=max_workers,
        logger=logger,
    )
    return BatchClassifyResponse(**result)


# IMPORTANT: This catch-all route MUST be at the end of the file
# to prevent it from matching specific paths like /drift, /vectors, etc.
@router.get("/{analysis_id}")
async def get_analysis_result(analysis_id: str, api_key: str = Depends(get_api_key)):
    """获取分析结果"""
    # 从缓存或落盘存储获取历史分析结果
    cache_key = f"analysis_result:{analysis_id}"
    result = await get_cached_result(cache_key)

    if not result:
        result = await load_analysis_result(analysis_id)
        if result:
            await set_cache(cache_key, result, ttl_seconds=3600)

    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return result
