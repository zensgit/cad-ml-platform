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
from src.api.v1.analyze_batch_router import build_batch_router
from src.api.v1.analyze_faiss_admin_router import router as faiss_admin_router
from src.api.v1.analyze_legacy_redirects import router as legacy_redirect_router
from src.api.v1.analyze_live_models import (
    AnalysisOptions,
    AnalysisResult,
)
from src.api.v1.analyze_vector_update_router import (
    router as vector_update_router,
)
from src.api.v1.analyze_vector_migration_router import (
    router as vector_migration_router,
)
from src.api.v1.analyze_result_router import build_result_router
from src.api.v1.analyze_similarity_router import router as similarity_router
from src.api.v1.process import process_rules_audit
from src.api.v1.analyze_shadow_compat import (
    _build_graph2d_soft_override_suggestion,
    _enrich_graph2d_prediction,
    _graph2d_is_drawing_type,
    _resolve_history_sequence_file_path,
)
from src.core.analysis_drift_attachment import attach_analysis_drift
from src.core.analysis_drift_pipeline import run_analysis_drift_pipeline
from src.core.analysis_drift_state import ANALYSIS_DRIFT_STATE as _DRIFT_STATE
from src.core.analysis_error_handling import (
    handle_analysis_http_exception,
    handle_analysis_options_json_error,
    handle_analysis_unexpected_exception,
)
from src.core.analysis_live_pipeline import run_analysis_live_pipeline
from src.core.analysis_manufacturing_summary import (
    attach_manufacturing_decision_summary,
)
from src.core.analysis_ocr_attachment import attach_analysis_ocr_payload
from src.core.analysis_parallel_pipeline import run_analysis_parallel_pipeline
from src.core.analysis_preflight import run_analysis_request_preflight
from src.core.analysis_response_builder import build_analysis_response
from src.core.analysis_result_envelope import finalize_analysis_success
from src.core.analysis_vector_attachment import attach_analysis_vector_context
from src.core.analyzer import CADAnalyzer
from src.core.classification import (
    extract_label_decision_contract,
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
from src.core.vector_pipeline import run_vector_pipeline
from src.models.cad_document import CadDocument
from src.utils.analysis_result_store import load_analysis_result
from src.utils.cache import get_cached_result, set_cache

logger = logging.getLogger(__name__)

router = APIRouter()
router.include_router(legacy_redirect_router)
router.include_router(similarity_router)
router.include_router(faiss_admin_router)
router.include_router(vector_migration_router)
router.include_router(vector_update_router)


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
    try:
        content = await file.read()
        return await run_analysis_live_pipeline(
            file_name=file.filename,
            content=content,
            options_raw=options,
            material=material,
            project_id=project_id,
            analysis_id=analysis_id,
            start_time=start_time,
            options_model_cls=AnalysisOptions,
            result_model_cls=AnalysisResult,
            analyzer_factory=CADAnalyzer,
            run_preflight_fn=run_analysis_request_preflight,
            run_document_pipeline_fn=run_document_pipeline,
            run_feature_pipeline_fn=run_feature_pipeline,
            run_parallel_pipeline_fn=run_analysis_parallel_pipeline,
            attach_manufacturing_summary_fn=attach_manufacturing_decision_summary,
            build_manufacturing_summary_fn=build_manufacturing_decision_summary,
            attach_drift_fn=attach_analysis_drift,
            drift_state=_DRIFT_STATE,
            run_drift_pipeline_fn=run_analysis_drift_pipeline,
            attach_vector_context_fn=attach_analysis_vector_context,
            run_vector_pipeline_fn=run_vector_pipeline,
            get_qdrant_store_fn=_get_qdrant_store_or_none,
            compute_qdrant_similarity_fn=compute_qdrant_cosine_similarity,
            attach_ocr_payload_fn=attach_analysis_ocr_payload,
            run_ocr_pipeline_fn=run_analysis_ocr_pipeline,
            build_response_fn=build_analysis_response,
            finalize_analysis_success_fn=finalize_analysis_success,
            classification_pipeline_fn=run_classification_pipeline,
            quality_pipeline_fn=run_quality_pipeline,
            process_pipeline_fn=run_process_pipeline,
            logger_instance=logger,
        )

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

router.include_router(
    build_batch_router(analyze_file_fn=analyze_cad_file, logger_instance=logger)
)
router.include_router(
    build_result_router(
        get_cached_result_fn=get_cached_result,
        load_analysis_result_fn=load_analysis_result,
        set_cache_fn=set_cache,
    )
)
