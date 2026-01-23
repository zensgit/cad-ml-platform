"""
CAD文件分析API端点
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from src.adapters.factory import AdapterFactory
from src.api.dependencies import get_api_key
from src.core.analyzer import CADAnalyzer
from src.core.errors_extended import (
    ErrorCode,
    build_error,
    create_extended_error,
    create_migration_error,
)
from src.core.feature_extractor import FeatureExtractor
from src.core.ocr.manager import OcrManager
from src.core.ocr.providers.deepseek_hf import DeepSeekHfProvider
from src.core.ocr.providers.paddle import PaddleOcrProvider
from src.core.similarity import FaissVectorStore, compute_similarity, has_vector, register_vector
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
    vector_store_material_total,
)
from src.utils.analysis_result_store import load_analysis_result, store_analysis_result
from src.utils.cache import cache_result, get_cached_result, set_cache

logger = logging.getLogger(__name__)

router = APIRouter()

# Local helper for env float parsing to avoid runtime 500s on bad values.
def _safe_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%s; using default %.2f", name, raw, default)
        return float(default)

# Drift state (in-memory); keys: materials, predictions, baseline_materials, baseline_predictions
_DRIFT_STATE: Dict[str, Any] = {
    "materials": [],
    "predictions": [],
    "baseline_materials": [],
    "baseline_predictions": [],
    "baseline_materials_ts": None,
    "baseline_predictions_ts": None,
}


class AnalysisOptions(BaseModel):
    """分析选项"""

    extract_features: bool = Field(default=True, description="是否提取特征")
    classify_parts: bool = Field(default=True, description="是否分类零件")
    calculate_similarity: bool = Field(default=False, description="是否计算相似度")
    reference_id: Optional[str] = Field(default=None, description="参考文件ID")
    quality_check: bool = Field(default=True, description="是否质量检查")
    process_recommendation: bool = Field(default=False, description="是否推荐工艺")
    estimate_cost: bool = Field(default=False, description="是否估算成本 (L4)")
    enable_ocr: bool = Field(default=False, description="是否启用OCR解析 (默认关闭保障向后兼容)")
    ocr_provider: str = Field(default="auto", description="OCR provider策略 auto|paddle|deepseek_hf")


class AnalysisResult(BaseModel):
    """分析结果"""

    id: str = Field(description="分析ID")
    timestamp: datetime = Field(description="分析时间")
    file_name: str = Field(description="文件名")
    file_format: str = Field(description="文件格式")
    results: Dict[str, Any] = Field(description="分析结果")
    processing_time: float = Field(description="处理时间(秒)")
    cache_hit: bool = Field(default=False, description="是否缓存命中")
    cad_document: Optional[Dict[str, Any]] = Field(
        default=None,
        description="统一的CAD文档结构 (序列化) 包含实体/图层/边界框/复杂度等, 便于下游直接使用。",
    )
    feature_version: str = Field(default="v1", description="特征版本 (用于兼容后续维度或语义扩展)")


class SimilarityQuery(BaseModel):
    reference_id: str = Field(description="参考分析ID")
    target_id: str = Field(description="目标分析ID")


class SimilarityResult(BaseModel):
    reference_id: str
    target_id: str
    score: float
    method: str
    dimension: int
    status: Optional[str] = None
    error: Optional[Dict[str, Any]] = None


class SimilarityTopKQuery(BaseModel):
    target_id: str = Field(description="用于检索相似向量的分析ID")
    k: int = Field(default=5, description="返回的最大数量 (包含自身)")
    exclude_self: bool = Field(default=False, description="是否排除自身向量")
    offset: int = Field(default=0, description="结果偏移用于分页")
    material_filter: Optional[str] = Field(default=None, description="按材料过滤")
    complexity_filter: Optional[str] = Field(default=None, description="按复杂度过滤")


class SimilarityTopKItem(BaseModel):
    id: str
    score: float
    material: Optional[str] = None
    complexity: Optional[str] = None
    format: Optional[str] = None


class SimilarityTopKResponse(BaseModel):
    target_id: str
    k: int
    results: list[SimilarityTopKItem]
    status: Optional[str] = None
    error: Optional[Dict[str, Any]] = None


class VectorDeleteRequest(BaseModel):  # deprecated moved to vectors.py
    id: str = Field(description="要删除的向量分析ID")


class VectorDeleteResponse(BaseModel):  # deprecated moved to vectors.py
    id: str
    status: str


class VectorListItem(BaseModel):  # deprecated moved to vectors.py
    id: str
    dimension: int
    material: Optional[str] = None
    complexity: Optional[str] = None
    format: Optional[str] = None


class VectorListResponse(BaseModel):  # deprecated moved to vectors.py
    total: int
    vectors: list[VectorListItem]


class VectorUpdateRequest(BaseModel):
    id: str = Field(description="要更新的向量分析ID")
    replace: Optional[list[float]] = Field(default=None, description="新的向量 (维度需与原向量一致)")
    append: Optional[list[float]] = Field(default=None, description="追加的向量片段 (若提供 replace 则忽略)")
    material: Optional[str] = Field(default=None, description="更新材料元数据")
    complexity: Optional[str] = Field(default=None, description="更新复杂度元数据")
    format: Optional[str] = Field(default=None, description="更新格式元数据")


class VectorUpdateResponse(BaseModel):
    id: str
    status: str
    dimension: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
    feature_version: Optional[str] = None


class VectorStatsResponse(BaseModel):  # deprecated moved to vectors_stats.py
    backend: str
    total: int
    by_material: Dict[str, int]
    by_complexity: Dict[str, int]
    by_format: Dict[str, int]
    versions: Optional[Dict[str, int]] = None


class VectorDistributionResponse(BaseModel):  # deprecated moved to vectors_stats.py
    total: int
    by_material: Dict[str, int]
    by_complexity: Dict[str, int]
    by_format: Dict[str, int]
    dominant_ratio: float
    feature_version: str
    average_dimension: Optional[float] = None
    versions: Optional[Dict[str, int]] = None


class VectorMigrateItem(BaseModel):
    id: str
    status: str
    from_version: Optional[str] = None
    to_version: Optional[str] = None
    dimension_before: Optional[int] = None
    dimension_after: Optional[int] = None
    error: Optional[str] = None


class VectorMigrateRequest(BaseModel):
    ids: list[str] = Field(description="需要迁移的向量ID列表")
    to_version: str = Field(description="目标特征版本")
    dry_run: bool = Field(default=False, description="是否为试运行 (不真正写入)")


class VectorMigrateResponse(BaseModel):
    total: int
    migrated: int
    skipped: int
    items: list[VectorMigrateItem]
    migration_id: Optional[str] = Field(default=None, description="迁移批次ID")
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    dry_run_total: Optional[int] = None


class VectorMigrationStatusResponse(BaseModel):
    last_migration_id: Optional[str] = None
    last_started_at: Optional[datetime] = None
    last_finished_at: Optional[datetime] = None
    last_total: Optional[int] = None
    last_migrated: Optional[int] = None
    last_skipped: Optional[int] = None
    pending_vectors: Optional[int] = None
    feature_versions: Optional[Dict[str, int]] = None
    history: Optional[list[Dict[str, Any]]] = None


class ProcessRulesAuditResponse(BaseModel):
    version: str
    source: str
    hash: Optional[str] = None
    materials: list[str]
    complexities: Dict[str, list[str]]
    raw: Dict[str, Any]


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
        # 解析选项
        analysis_options = AnalysisOptions(**json.loads(options))

        # 计算内容哈希用于更精确缓存键 (避免同名不同内容误命中)
        import hashlib

        content_peek = await file.read()  # read for hash then reset below
        file.file.seek(0)
        content_hash = hashlib.sha256(content_peek).hexdigest()[:16]
        analysis_cache_key = f"analysis:{file.filename}:{content_hash}:{options}"
        cached = await get_cached_result(analysis_cache_key)
        if cached:
            logger.info(f"Cache hit for {file.filename}")
            from src.utils.analysis_metrics import analysis_cache_hits_total

            analysis_cache_hits_total.inc()
            # sliding window update (best-effort)
            try:
                from collections import deque
                from time import time as _t_now

                from src.utils.analysis_metrics import feature_cache_hits_last_hour

                _CACHE_HIT_EVENTS = globals().setdefault("_CACHE_HIT_EVENTS", deque())
                now = _t_now()
                _CACHE_HIT_EVENTS.append(now)
                # prune older than 3600s
                while _CACHE_HIT_EVENTS and now - _CACHE_HIT_EVENTS[0] > 3600:
                    _CACHE_HIT_EVENTS.popleft()
                feature_cache_hits_last_hour.set(len(_CACHE_HIT_EVENTS))
            except Exception:
                pass
            feature_version = __import__("os").getenv("FEATURE_VERSION", "v1")
            return AnalysisResult(
                id=analysis_id,
                timestamp=start_time,
                file_name=file.filename,
                file_format=file.filename.split(".")[-1].upper(),
                results=cached,
                processing_time=0.1,
                cache_hit=True,
                cad_document=None,
                feature_version=feature_version,
            )
        else:
            from src.utils.analysis_metrics import analysis_cache_miss_total

            analysis_cache_miss_total.inc()
            try:
                from collections import deque
                from time import time as _t_now

                from src.utils.analysis_metrics import feature_cache_miss_last_hour

                _CACHE_MISS_EVENTS = globals().setdefault("_CACHE_MISS_EVENTS", deque())
                now = _t_now()
                _CACHE_MISS_EVENTS.append(now)
                while _CACHE_MISS_EVENTS and now - _CACHE_MISS_EVENTS[0] > 3600:
                    _CACHE_MISS_EVENTS.popleft()
                feature_cache_miss_last_hour.set(len(_CACHE_MISS_EVENTS))
            except Exception:
                pass

        # 读取文件内容
        content = await file.read()
        # MIME sniff (best-effort); reject if clearly unsupported
        from src.security.input_validator import (
            deep_format_validate,
            is_supported_mime,
            sniff_mime,
            verify_signature,
        )

        mime, reliable = sniff_mime(content[:4096])  # peek first 4KB
        if reliable and not is_supported_mime(mime):
            analysis_rejections_total.labels(reason="mime_mismatch").inc()
            # ErrorCode and build_error imported at module level
            err = build_error(
                ErrorCode.INPUT_FORMAT_INVALID,
                stage="input",
                message=f"Unsupported MIME type: {mime}",
                mime=mime,
            )
            raise HTTPException(status_code=415, detail=err)
        # Safety: file size limit (10MB default)
        max_mb = float(__import__("os").getenv("ANALYSIS_MAX_FILE_MB", "10"))
        size_mb = len(content) / (1024 * 1024)
        if size_mb > max_mb:
            analysis_requests_total.labels(status="error").inc()
            analysis_errors_total.labels(stage="input", code="file_too_large").inc()
            # ErrorCode and build_error imported at module level
            err = build_error(
                ErrorCode.INPUT_SIZE_EXCEEDED,
                stage="input",
                message="File too large",
                size_mb=round(size_mb, 3),
                max_mb=max_mb,
            )
            raise HTTPException(status_code=413, detail=err)
        if not content:
            # ErrorCode and build_error imported at module level
            err = build_error(
                ErrorCode.INPUT_ERROR,
                stage="input",
                message="Empty file",
            )
            raise HTTPException(status_code=400, detail=err)

        # 获取文件格式
        file_format = file.filename.split(".")[-1].lower()
        if file_format not in ["dxf", "dwg", "json", "step", "stp", "iges", "igs", "stl"]:
            analysis_requests_total.labels(status="error").inc()
            analysis_errors_total.labels(stage="input", code="unsupported_format").inc()
            # ErrorCode and build_error imported at module level
            err = build_error(
                ErrorCode.UNSUPPORTED_FORMAT,
                stage="input",
                message=f"Unsupported file format: {file_format}",
                format=file_format,
            )
            raise HTTPException(status_code=400, detail=err)

        # 使用适配器转换格式
        adapter = AdapterFactory.get_adapter(file_format)
        # Convert to unified dict (legacy) using adapter; new adapters can also return CadDocument via parse
        # Adapter may return legacy dict (convert()) or we can call parse if available.
        # Parse with timeout protection
        parse_timeout = float(os.getenv("PARSE_TIMEOUT_SECONDS", "10"))
        import time as _t

        _parse_start = _t.time()
        try:
            import asyncio

            doc: CadDocument
            if hasattr(adapter, "parse"):
                # adapter.parse may be async; wrap with wait_for
                doc = await asyncio.wait_for(
                    adapter.parse(content, file_name=file.filename),  # type: ignore[attr-defined]
                    timeout=parse_timeout,
                )
            else:
                # legacy convert path (may be async as well)
                _legacy = await asyncio.wait_for(
                    adapter.convert(content, file_name=file.filename),
                    timeout=parse_timeout,
                )
                doc = CadDocument(file_name=file.filename, format=file_format)
                doc.metadata.update({"legacy": True})
            unified_data = doc.to_unified_dict()
        except asyncio.TimeoutError:
            from src.utils.analysis_metrics import parse_timeout_total

            parse_timeout_total.inc()
            analysis_errors_total.labels(stage="parse", code="timeout").inc()
            # ErrorCode and build_error imported at module level
            err = build_error(
                ErrorCode.TIMEOUT,
                stage="parse",
                message="Parse stage timeout",
                timeout_seconds=parse_timeout,
                file=file.filename,
            )
            raise HTTPException(status_code=504, detail=err)
        except Exception:
            doc = CadDocument(file_name=file.filename, format=file_format)
            unified_data = doc.to_unified_dict()
        try:
            from src.utils.analysis_metrics import parse_stage_latency_seconds

            parse_stage_latency_seconds.labels(format=file_format).observe(_t.time() - _parse_start)
        except Exception:
            pass
        # Signature validation (heuristic)
        valid_sig, expectation = verify_signature(content[:256], file_format)
        if not valid_sig:
            from src.utils.analysis_metrics import signature_validation_fail_total

            signature_validation_fail_total.labels(format=file_format).inc()
            analysis_rejections_total.labels(reason="signature_mismatch").inc()
            # ErrorCode and build_error imported at module level
            from src.security.input_validator import signature_hex_prefix

            err = build_error(
                ErrorCode.INPUT_FORMAT_INVALID,
                stage="input",
                message="Signature validation failed",
                format=file_format,
                signature_prefix=signature_hex_prefix(content[:32]),
                expected_signature=expectation,
            )
            raise HTTPException(status_code=415, detail=err)

        # Deep format validation (strict mode optional) + matrix validation
        strict_mode = os.getenv("FORMAT_STRICT_MODE", "0") == "1"
        from src.utils.analysis_metrics import format_validation_fail_total, strict_mode_enabled

        if strict_mode:
            strict_mode_enabled.set(1)
            ok_deep, reason_deep = deep_format_validate(content[:2048], file_format)
            if not ok_deep:
                format_validation_fail_total.labels(format=file_format, reason=reason_deep).inc()
                analysis_rejections_total.labels(reason="deep_format_invalid").inc()
                # ErrorCode imported at module level
                # ErrorCode and build_error imported at module level
                err = build_error(
                    ErrorCode.INPUT_FORMAT_INVALID,
                    stage="input",
                    message="Deep format validation failed",
                    format=file_format,
                    reason=reason_deep,
                )
                raise HTTPException(status_code=415, detail=err)
            # matrix validation
            from src.security.input_validator import matrix_validate

            ok_matrix, reason_matrix = matrix_validate(content[:4096], file_format, project_id)
            if not ok_matrix:
                format_validation_fail_total.labels(format=file_format, reason=reason_matrix).inc()
                analysis_rejections_total.labels(reason="matrix_format_invalid").inc()
                # ErrorCode imported at module level
                # ErrorCode and build_error imported at module level
                err = build_error(
                    ErrorCode.INPUT_FORMAT_INVALID,
                    stage="input",
                    message="Matrix format validation failed",
                    format=file_format,
                    reason=reason_matrix,
                    project_id=project_id,
                )
                raise HTTPException(status_code=415, detail=err)
        else:
            strict_mode_enabled.set(0)
        # attach metadata
        if material:
            doc.metadata["material"] = material
            analysis_material_usage_total.labels(material=material).inc()
        if project_id:
            doc.metadata["project_id"] = project_id
        stage_times["parse"] = time.time() - started
        analysis_stage_duration_seconds.labels(stage="parse").observe(stage_times["parse"])
        # Budget ratio metric (parse latency / target)
        target_ms = float(__import__("os").getenv("ANALYSIS_PARSE_P95_TARGET_MS", "250"))
        if target_ms > 0:
            ratio = (stage_times["parse"] * 1000.0) / target_ms
            analysis_parse_latency_budget_ratio.observe(ratio)

        # Complexity limits (configurable via env): reject overly large entity counts to protect service
        max_entities = int(__import__("os").getenv("ANALYSIS_MAX_ENTITIES", "50000"))
        if doc.entity_count() > max_entities:
            analysis_rejections_total.labels(reason="entity_count_exceeded").inc()
            # ErrorCode and build_error imported at module level
            err = build_error(
                ErrorCode.VALIDATION_FAILED,
                stage="input",
                message="Entity count exceeds limit",
                entity_count=doc.entity_count(),
                max_entities=max_entities,
            )
            raise HTTPException(status_code=422, detail=err)

        # 创建分析器
        analyzer = CADAnalyzer()
        results: Dict[str, Any] = {}
        features: Dict[str, Any] = {
            "geometric": [],
            "semantic": [],
        }  # ensure defined even if skipped
        features_3d: Dict[str, Any] = {}

        # L3: 3D Feature Extraction (run before 2D feature extraction)
        if analysis_options.extract_features and file_format in ["step", "stp", "iges", "igs"]:
            try:
                # Lazy import to avoid startup overhead if not used
                from src.core.geometry.cache import get_feature_cache
                from src.core.geometry.engine import get_geometry_engine
                from src.ml.vision_3d import get_3d_encoder

                _geo_start = time.time()

                # Check Cache
                f_cache = get_feature_cache()
                # Use feature version from env or default
                f_ver = "l4_v1"
                cache_key = f_cache.generate_key(content, f_ver)
                cached_3d = f_cache.get(cache_key)

                if cached_3d:
                    features_3d = cached_3d
                    logger.info(f"3D Feature Cache HIT for {file.filename}")
                else:
                    geo_engine = get_geometry_engine()
                    # Parse 3D content
                    shape = geo_engine.load_step(content, file_name=file.filename)
                    if shape:
                        features_3d = geo_engine.extract_brep_features(shape)
                        # L4: Extract DFM features (Wall thickness, etc.)
                        dfm_feats = geo_engine.extract_dfm_features(shape)
                        features_3d.update(dfm_feats)

                        # Deep Learning Embedding
                        encoder = get_3d_encoder()
                        embedding_3d = encoder.encode(features_3d)
                        features_3d["embedding_vector"] = embedding_3d

                        # Save to Cache
                        f_cache.set(cache_key, features_3d)

                # Add to result payload for debugging/inspection
                if "embedding_vector" in features_3d:
                    results["features_3d"] = {
                        k: v for k, v in features_3d.items() if k != "embedding_vector"
                    }
                    results["features_3d"]["embedding_dim"] = len(features_3d["embedding_vector"])

                stage_times["features_3d"] = time.time() - _geo_start
            except Exception as e:
                logger.error(f"L3 Analysis failed: {e}")

        # 执行分析 (带特征缓存)
        if analysis_options.extract_features:
            import hashlib as _hl

            from src.core.feature_cache import get_feature_cache

            feature_version = __import__("os").getenv("FEATURE_VERSION", "v1")
            # Use full content bytes for cache key
            content_hash_full = _hl.sha256(content).hexdigest()
            cache_key = f"{content_hash_full}:{feature_version}:layout_v2"
            from src.utils.analysis_metrics import (
                feature_cache_hits_total,
                feature_cache_miss_total,
                feature_cache_size,
            )

            feature_cache = get_feature_cache()
            # Measure lookup latency
            import time as _t

            _lk_start = _t.time()
            cached_vector = feature_cache.get(cache_key)
            try:
                from src.utils.analysis_metrics import feature_cache_lookup_seconds

                feature_cache_lookup_seconds.observe(_t.time() - _lk_start)
            except Exception:
                pass
            extractor = FeatureExtractor()
            combined_vec: list[float] | None = None
            if cached_vector is not None:
                feature_cache_hits_total.inc()
                # rehydrate cached vector into geometric/semantic split
                features = extractor.rehydrate(cached_vector, version=feature_version)
                combined_vec = cached_vector
            else:
                feature_cache_miss_total.inc()
                features = await extractor.extract(doc, brep_features=features_3d)
                try:
                    combined_vec = extractor.flatten(features)
                    feature_cache.set(cache_key, combined_vec)
                    feature_cache_size.set(feature_cache.size())
                except Exception:
                    pass
            if combined_vec is None:
                try:
                    combined_vec = extractor.flatten(features)
                except Exception:
                    combined_vec = []
            feature_slots = extractor.slots(feature_version)
            results["features"] = {
                "geometric": [float(x) for x in features["geometric"]],
                "semantic": [float(x) for x in features["semantic"]],
                "combined": [float(x) for x in combined_vec],
                "dimension": len(features["geometric"]) + len(features["semantic"]),
                "feature_version": feature_version,
                "feature_slots": feature_slots,
                "cache_hit": cached_vector is not None,
            }
            stage_times["features"] = time.time() - started - sum(stage_times.values())
            analysis_stage_duration_seconds.labels(stage="features").observe(
                stage_times["features"]
            )

        # Parallelize classification / quality / process recommendation if multiple enabled
        import asyncio

        parallel_tasks = []
        if analysis_options.classify_parts:

            async def _run_classify():
                t0 = time.time()

                def _build_text_signals(doc: CadDocument) -> str:
                    parts = []
                    stem, _ = os.path.splitext(doc.file_name or "")
                    if stem:
                        parts.append(stem)
                    text = doc.metadata.get("text")
                    if text:
                        parts.append(str(text))
                    text_content = doc.metadata.get("text_content")
                    if isinstance(text_content, list):
                        parts.extend([str(t) for t in text_content if str(t).strip()])
                    meta = doc.metadata.get("meta")
                    if isinstance(meta, dict):
                        for key in (
                            "drawing_number",
                            "drawing_no",
                            "drawingNo",
                            "drawingNumber",
                            "number",
                        ):
                            val = meta.get(key)
                            if val:
                                parts.append(str(val))
                    return " ".join(parts).strip()

                # Check if we can use L3 Fusion
                ent_counts = {}
                for e in doc.entities:
                    ent_counts[e.kind] = ent_counts.get(e.kind, 0) + 1
                text_signals = _build_text_signals(doc)
                try:
                    from src.core.knowledge.fusion_analyzer import (
                        build_doc_metadata,
                        build_l2_features,
                    )

                    doc_metadata = build_doc_metadata(doc)
                    l2_features = build_l2_features(doc)
                except Exception:
                    doc_metadata = {}
                    l2_features = {}
                l3_features = (
                    {k: v for k, v in features_3d.items() if k != "embedding_vector"}
                    if features_3d
                    else {}
                )
                if features_3d:
                    try:
                        from src.core.knowledge.fusion import get_fusion_classifier

                        fusion = get_fusion_classifier()

                        # Perform fused classification
                        fused_result = fusion.classify(
                            text_signals=text_signals,
                            features_2d={
                                "geometric_features": l2_features,
                                "entity_counts": ent_counts,
                            },
                            features_3d=features_3d,
                        )

                        classification = {
                            "type": fused_result["type"],
                            "confidence": fused_result["confidence"],
                            "sub_type": None,
                            "characteristics": [],
                            "rule_version": "L3-Fusion-v1",
                            "alternatives": fused_result.get("alternatives", []),
                            "confidence_breakdown": fused_result.get("fusion_breakdown"),
                        }
                    except Exception as e:
                        logger.error(f"Fusion failed, falling back to L1: {e}")
                        classification = await analyzer.classify_part(doc, features)
                else:
                    classification = await analyzer.classify_part(doc, features)
                    if text_signals or ent_counts:
                        try:
                            from src.core.knowledge.fusion import get_fusion_classifier

                            fusion = get_fusion_classifier()
                            fused_result = fusion.classify(
                                text_signals=text_signals,
                                features_2d={
                                    "geometric_features": l2_features,
                                    "entity_counts": ent_counts,
                                },
                                features_3d={},
                            )
                            if (
                                fused_result.get("type") not in {None, "unknown"}
                                and float(fused_result.get("confidence") or 0.0) > 0.0
                            ):
                                classification = {
                                    "type": fused_result["type"],
                                    "confidence": fused_result["confidence"],
                                    "sub_type": None,
                                    "characteristics": [],
                                    "rule_version": "L2-Fusion-v1",
                                    "alternatives": fused_result.get("alternatives", []),
                                    "confidence_breakdown": fused_result.get("fusion_breakdown"),
                                }
                        except Exception as e:
                            logger.error(f"Fusion failed, falling back to L1: {e}")

                cls_payload = {
                    "part_type": classification["type"],
                    "confidence": classification["confidence"],
                    "sub_type": classification.get("sub_type"),
                    "characteristics": classification.get("characteristics", []),
                    "rule_version": classification.get("rule_version"),
                    "alternatives": classification.get("alternatives", []),
                    "confidence_breakdown": classification.get("confidence_breakdown"),
                }
                rule_version = str(cls_payload.get("rule_version") or "")
                cls_payload["confidence_source"] = (
                    "fusion"
                    if rule_version.startswith("L3-Fusion") or rule_version.startswith("L2-Fusion")
                    else "rules"
                )
                # Attempt ML classification overlay
                ml_result: Dict[str, Any] | None = None
                try:
                    from src.ml.classifier import predict

                    vec_for_model = FeatureExtractor().flatten(features)
                    ml_result = predict(vec_for_model)
                    if ml_result.get("predicted_type"):
                        cls_payload["ml_predicted_type"] = ml_result["predicted_type"]
                        cls_payload["model_version"] = ml_result.get("model_version")
                    else:
                        cls_payload["model_version"] = ml_result.get("status")
                except Exception:
                    ml_result = None
                    cls_payload["model_version"] = "ml_error"
                # Optional 2D graph classifier (shadow by default)
                graph2d_result: Dict[str, Any] | None = None
                graph2d_enabled = os.getenv("GRAPH2D_ENABLED", "false").lower() == "true"
                if graph2d_enabled and file_format == "dxf":
                    try:
                        from src.ml.vision_2d import get_2d_classifier

                        graph2d_result = get_2d_classifier().predict_from_bytes(
                            content, file.filename
                        )
                        if graph2d_result.get("status") != "model_unavailable":
                            cls_payload["graph2d_prediction"] = graph2d_result
                    except Exception:
                        graph2d_result = None
                # Optional FusionAnalyzer (shadow by default)
                fusion_enabled = os.getenv("FUSION_ANALYZER_ENABLED", "false").lower() == "true"
                fusion_override = (
                    os.getenv("FUSION_ANALYZER_OVERRIDE", "false").lower() == "true"
                )
                fusion_override_min_conf = _safe_float_env(
                    "FUSION_ANALYZER_OVERRIDE_MIN_CONF", 0.5
                )
                if fusion_enabled:
                    try:
                        from src.core.knowledge.fusion_analyzer import get_fusion_analyzer

                        l4_prediction = None
                        graph2d_fusion = (
                            os.getenv("GRAPH2D_FUSION_ENABLED", "false").lower() == "true"
                        )
                        if (
                            graph2d_fusion
                            and graph2d_result
                            and graph2d_result.get("label")
                        ):
                            l4_prediction = {
                                "label": graph2d_result["label"],
                                "confidence": float(graph2d_result.get("confidence", 0.0)),
                                "source": "graph2d",
                            }
                        elif ml_result and ml_result.get("predicted_type"):
                            l4_prediction = {
                                "label": ml_result["predicted_type"],
                                "confidence": float(ml_result.get("confidence", 0.0)),
                                "source": "ml",
                            }

                        fusion_decision = get_fusion_analyzer().analyze(
                            doc_metadata=doc_metadata,
                            l2_features=l2_features,
                            l3_features=l3_features,
                            l4_prediction=l4_prediction,
                        )
                        cls_payload["fusion_decision"] = fusion_decision.model_dump()
                        cls_payload["fusion_inputs"] = {
                            "l1": doc_metadata,
                            "l2": l2_features,
                            "l3": l3_features,
                            "l4": l4_prediction,
                        }
                        if fusion_override and fusion_decision.confidence >= fusion_override_min_conf:
                            from src.core.knowledge.fusion_contracts import DecisionSource

                            is_default_rule = (
                                fusion_decision.source == DecisionSource.RULE_BASED
                                and fusion_decision.rule_hits == ["RULE_DEFAULT"]
                            )
                            if is_default_rule:
                                cls_payload["fusion_override_skipped"] = {
                                    "min_confidence": fusion_override_min_conf,
                                    "decision_confidence": fusion_decision.confidence,
                                    "reason": "default_rule_only",
                                }
                            else:
                                cls_payload["part_type"] = fusion_decision.primary_label
                                cls_payload["confidence"] = fusion_decision.confidence
                                cls_payload["rule_version"] = (
                                    f"FusionAnalyzer-{fusion_decision.schema_version}"
                                )
                                cls_payload["confidence_source"] = "fusion"
                        elif fusion_override:
                            cls_payload["fusion_override_skipped"] = {
                                "min_confidence": fusion_override_min_conf,
                                "decision_confidence": fusion_decision.confidence,
                            }
                    except Exception as e:
                        logger.error(f"FusionAnalyzer failed: {e}")
                results["classification"] = cls_payload
                # Active learning: flag low-confidence samples for review
                try:
                    enabled = (
                        __import__("os").getenv("ACTIVE_LEARNING_ENABLED", "false").lower()
                        == "true"
                    )
                    threshold = float(
                        __import__("os").getenv("ACTIVE_LEARNING_CONFIDENCE_THRESHOLD", "0.6")
                    )
                    if enabled and float(cls_payload.get("confidence", 1.0)) < threshold:
                        from src.core.active_learning import get_active_learner

                        learner = get_active_learner()
                        learner.flag_for_review(
                            doc_id=analysis_id,
                            predicted_type=str(cls_payload.get("part_type", "unknown")),
                            confidence=float(cls_payload.get("confidence", 0.0)),
                            alternatives=cls_payload.get("alternatives", []),
                            score_breakdown={
                                "rule_version": cls_payload.get("rule_version"),
                                "model_version": cls_payload.get("model_version"),
                                "confidence_source": cls_payload.get("confidence_source"),
                                "confidence_breakdown": cls_payload.get("confidence_breakdown"),
                            },
                            uncertainty_reason="low_confidence",
                        )
                except Exception as e:
                    logger.warning(f"Active learning flag failed: {e}")
                dur = time.time() - t0
                classification_latency_seconds.observe(dur)
                return ("classify", dur)

            parallel_tasks.append(_run_classify())

        if analysis_options.quality_check:

            async def _run_quality():
                t0 = time.time()
                # L4 DFM Check
                if "features_3d" in locals() and features_3d:
                    try:
                        dfm_start = time.time()
                        # Extract extra DFM features if not already done
                        if "thin_walls_detected" not in features_3d:
                            from src.core.geometry.engine import get_geometry_engine

                            geo = get_geometry_engine()
                            # Re-load shape from cache or content not ideal here,
                            # but for prototype we assume features_3d already has what we need
                            # OR we enhanced extract_brep_features to include DFM.
                            # Let's assume we call extract_dfm_features here if shape is available:
                            # shape = geo.load_step(...)  # Expensive, in prod pass shape around
                            pass

                        from src.core.dfm.analyzer import get_dfm_analyzer

                        dfm = get_dfm_analyzer()
                        # Use classified type or unknown
                        ptype = results.get("classification", {}).get("part_type", "unknown")
                        dfm_result = dfm.analyze(features_3d, ptype)
                        dfm_analysis_latency_seconds.observe(time.time() - dfm_start)

                        results["quality"] = {
                            "mode": "L4_DFM",
                            "score": dfm_result["dfm_score"],
                            "issues": dfm_result["issues"],
                            "manufacturability": dfm_result["manufacturability"],
                        }
                    except Exception as e:
                        logger.error(f"DFM check failed: {e}")
                        # Fallback
                        quality = await analyzer.check_quality(doc, features)
                        results["quality"] = quality
                else:
                    quality = await analyzer.check_quality(doc, features)
                    results["quality"] = {
                        "score": quality["score"],
                        "issues": quality.get("issues", []),
                        "suggestions": quality.get("suggestions", []),
                    }
                return ("quality", time.time() - t0)

            parallel_tasks.append(_run_quality())

        if analysis_options.process_recommendation:

            async def _run_process():
                t0 = time.time()
                # L4 AI Process Recommendation
                proc_result = None
                if "features_3d" in locals() and features_3d:
                    try:
                        from src.core.process.ai_recommender import get_process_recommender

                        recommender = get_process_recommender()
                        ptype = results.get("classification", {}).get("part_type", "unknown")
                        mat = material or "steel"  # Default

                        proc_result = recommender.recommend(features_3d, ptype, mat)
                        results["process"] = proc_result
                    except Exception as e:
                        logger.error(f"AI Process failed: {e}")
                        process = await analyzer.recommend_process(doc, features)
                        results["process"] = process
                        proc_result = process if isinstance(process, dict) else {}
                else:
                    process = await analyzer.recommend_process(doc, features)
                    if isinstance(process, dict) and process.get("rule_version"):
                        process_rule_version_total.labels(
                            version=str(process.get("rule_version"))
                        ).inc()
                    results["process"] = process
                    proc_result = process if isinstance(process, dict) else {}

                # L4 Cost Estimation (Chained after Process)
                if analysis_options.estimate_cost and "features_3d" in locals() and features_3d:
                    try:
                        from src.core.cost.estimator import get_cost_estimator

                        estimator = get_cost_estimator()
                        # Use primary recommendation if available
                        primary_proc = proc_result.get("primary_recommendation", {})
                        if not primary_proc and "process" in proc_result:
                            # Fallback to legacy rule structure
                            primary_proc = {
                                "process": proc_result.get("process"),
                                "method": "standard",
                            }
                        cost_start = time.time()
                        cost_est = estimator.estimate(
                            features_3d, primary_proc, material=material or "steel"
                        )
                        results["cost_estimation"] = cost_est
                        cost_estimation_latency_seconds.observe(time.time() - cost_start)
                    except Exception as e:
                        logger.error(f"Cost estimation failed: {e}")

                dur = time.time() - t0
                process_recommend_latency_seconds.observe(dur)
                return ("process", dur)

            parallel_tasks.append(_run_process())

        if parallel_tasks:
            analysis_parallel_enabled.set(1)
            # gather returns (stage_name, duration)
            import time as _t_parallel

            _wall_start = _t_parallel.time()
            stage_results = await asyncio.gather(*parallel_tasks)
            _wall_total = _t_parallel.time() - _wall_start
            serial_sum = 0.0
            for stage_name, indiv_dur in stage_results:
                stage_times[stage_name] = indiv_dur
                serial_sum += indiv_dur
                analysis_stage_duration_seconds.labels(stage=stage_name).observe(indiv_dur)
            # Savings = sum of individual durations - wall time (non-negative)
            from src.utils.analysis_metrics import analysis_parallel_savings_seconds

            savings = serial_sum - _wall_total
            if savings < 0:
                savings = 0.0
            analysis_parallel_savings_seconds.observe(savings)
        else:
            analysis_parallel_enabled.set(0)

        # Manufacturing decision summary (quality + process + cost)
        try:
            quality = results.get("quality", {}) if isinstance(results, dict) else {}
            process = results.get("process", {}) if isinstance(results, dict) else {}
            cost = results.get("cost_estimation", {}) if isinstance(results, dict) else {}

            primary_proc = {}
            if isinstance(process, dict):
                primary_proc = process.get("primary_recommendation", {})
                if not primary_proc and "process" in process:
                    primary_proc = {
                        "process": process.get("process"),
                        "method": process.get("method"),
                    }

            total_cost = None
            if isinstance(cost, dict):
                total_cost = cost.get("total_unit_cost")

            cost_range = None
            if isinstance(total_cost, (int, float)):
                cost_range = {
                    "low": round(total_cost * 0.9, 2),
                    "high": round(total_cost * 1.1, 2),
                }

            if quality or process or cost:
                results["manufacturing_decision"] = {
                    "feasibility": quality.get("manufacturability")
                    if isinstance(quality, dict)
                    else None,
                    "risks": quality.get("issues", []) if isinstance(quality, dict) else [],
                    "process": primary_proc or None,
                    "cost_estimate": cost if isinstance(cost, dict) else None,
                    "cost_range": cost_range,
                    "currency": cost.get("currency") if isinstance(cost, dict) else None,
                }
        except Exception as e:
            logger.warning(f"Manufacturing decision summary failed: {e}")

        # Drift metrics (executed once per analysis after classification if present)
        try:
            from src.utils.drift import compute_drift

            # Material drift: compare current material tag vs baseline (simple ring buffer)
            _DRIFT_STATE = __import__("src.api.v1.analyze", fromlist=["_DRIFT_STATE"])
        except Exception:
            _DRIFT_STATE = None  # type: ignore
        try:
            if _DRIFT_STATE is not None:
                st = getattr(
                    _DRIFT_STATE,
                    "_DRIFT_STATE",
                    {
                        "materials": [],
                        "predictions": [],
                        "baseline_materials": [],
                        "baseline_predictions": [],
                    },
                )
                m_used = material or "unknown"
                st["materials"].append(m_used)
                pred_label = results.get("classification", {}).get("type") or results.get(
                    "classification", {}
                ).get("ml_predicted_type")
                if pred_label:
                    st["predictions"].append(str(pred_label))
                # establish baselines once minimum count reached
                min_count = int(__import__("os").getenv("DRIFT_BASELINE_MIN_COUNT", "100"))
                if len(st["baseline_materials"]) == 0 and len(st["materials"]) >= min_count:
                    st["baseline_materials"] = list(st["materials"])
                    # persist baseline to redis if available
                    try:
                        client = __import__("src.utils.cache", fromlist=["get_client"]).get_client()
                        if client is not None:
                            await client.set("baseline:material", json.dumps(st["baseline_materials"]))  # type: ignore[attr-defined]
                            await client.set("baseline:material:ts", str(int(__import__("time").time())))  # type: ignore[attr-defined]
                    except Exception:
                        pass
                if len(st["baseline_predictions"]) == 0 and len(st["predictions"]) >= min_count:
                    st["baseline_predictions"] = list(st["predictions"])
                    try:
                        client = __import__("src.utils.cache", fromlist=["get_client"]).get_client()
                        if client is not None:
                            await client.set("baseline:class", json.dumps(st["baseline_predictions"]))  # type: ignore[attr-defined]
                            await client.set("baseline:class:ts", str(int(__import__("time").time())))  # type: ignore[attr-defined]
                    except Exception:
                        pass
                if st["baseline_materials"]:
                    mat_score = compute_drift(st["materials"], st["baseline_materials"])
                    material_distribution_drift_score.observe(mat_score)
                if st["baseline_predictions"]:
                    cls_score = compute_drift(st["predictions"], st["baseline_predictions"])
                    classification_prediction_drift_score.observe(cls_score)
                setattr(_DRIFT_STATE, "_DRIFT_STATE", st)
        except Exception:
            pass

        # Register vector for later similarity queries (use geometric+semantic concatenation + L3 embedding)
        try:
            from src.core.vector_layouts import VECTOR_LAYOUT_BASE, VECTOR_LAYOUT_L3

            feature_version = __import__("os").getenv("FEATURE_VERSION", "v1")
            feature_vector: list[float] = FeatureExtractor().flatten(features)
            vector_layout = VECTOR_LAYOUT_BASE
            l3_dim: int | None = None

            # L3 Integration: Append 3D embedding if available
            if "features_3d" in locals() and "embedding_vector" in features_3d:
                l3_dim = len(features_3d["embedding_vector"])
                feature_vector.extend([float(x) for x in features_3d["embedding_vector"]])
                vector_layout = VECTOR_LAYOUT_L3

            m_used = material or "unknown"
            meta = {
                "material": m_used,
                "complexity": doc.complexity_bucket(),
                "format": doc.format,
                "feature_version": feature_version,
                "vector_layout": vector_layout,
                "geometric_dim": str(len(features.get("geometric", []))),
                "semantic_dim": str(len(features.get("semantic", []))),
                "total_dim": str(len(feature_vector)),
            }
            if l3_dim is not None:
                meta["l3_3d_dim"] = str(l3_dim)

            accepted = register_vector(
                analysis_id,
                feature_vector,
                meta=meta,
            )
            if accepted:
                vector_store_material_total.labels(material=m_used).inc()
                # Optional FAISS backend add if enabled
                if os.getenv("VECTOR_STORE_BACKEND", "memory") == "faiss":
                    try:
                        fstore = FaissVectorStore()
                        fstore.add(analysis_id, feature_vector)
                    except Exception:
                        pass
                analysis_feature_vector_dimension.observe(len(feature_vector))
                # enrich meta with dimension breakdown for future migrations
                try:
                    _VECTOR_META = __import__("src.core.similarity", fromlist=["_VECTOR_META"])._VECTOR_META  # type: ignore
                    _VECTOR_META[analysis_id].update(meta)
                except Exception:
                    pass
        except Exception:
            pass

        if analysis_options.calculate_similarity and analysis_options.reference_id:
            sim = compute_similarity(analysis_options.reference_id, feature_vector)
            results["similarity"] = sim
        elif analysis_options.reference_id and not has_vector(analysis_options.reference_id):
            results["similarity"] = {
                "reference_id": analysis_options.reference_id,
                "status": "reference_not_found",
            }
        if "similarity" in results:
            stage_times["similarity"] = time.time() - started - sum(stage_times.values())
            analysis_stage_duration_seconds.labels(stage="similarity").observe(
                stage_times["similarity"]
            )

        # 可选 OCR 集成 (向后兼容: 默认不启用)
        if analysis_options.enable_ocr:
            ocr_manager = OcrManager(confidence_fallback=0.85)
            ocr_manager.register_provider("paddle", PaddleOcrProvider())
            ocr_manager.register_provider("deepseek_hf", DeepSeekHfProvider())
            # 简单处理: 如果是图像/含预览可抽取, 此处示例假设 unified_data 带有 preview_image_bytes
            img_bytes = unified_data.get("preview_image_bytes")
            if img_bytes:
                ocr_result = await ocr_manager.extract(
                    img_bytes, strategy=analysis_options.ocr_provider
                )
                results["ocr"] = {
                    "provider": ocr_result.provider,
                    "confidence": ocr_result.calibrated_confidence or ocr_result.confidence,
                    "fallback_level": ocr_result.fallback_level,
                    "dimensions": [d.model_dump() for d in ocr_result.dimensions],
                    "symbols": [s.model_dump() for s in ocr_result.symbols],
                    "completeness": ocr_result.completeness,
                }
            else:
                results["ocr"] = {"status": "no_preview_image"}

        # 添加统计信息
        results["statistics"] = {
            "entity_count": doc.entity_count(),
            "layer_count": len(doc.layers),
            "bounding_box": doc.bounding_box.model_dump(),
            "complexity": doc.complexity_bucket(),
            "stages": stage_times,
        }

        # 缓存结果 (使用 analysis_cache_key 而非被覆盖的 cache_key)
        await cache_result(analysis_cache_key, results, ttl=3600)
        # Persist full result by analysis id for retrieval endpoint
        await set_cache(f"analysis_result:{analysis_id}", results, ttl_seconds=3600)
        await store_analysis_result(analysis_id, results)

        # 计算处理时间
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        logger.info(
            "analysis.completed",
            extra={
                "file": file.filename,
                "analysis_id": analysis_id,
                "processing_time_s": round(processing_time, 4),
                "stages": stage_times,
                "feature_vector_dim": len(feature_vector) if "feature_vector" in locals() else 0,
                "material": material,
                "complexity": unified_data.get("complexity"),
            },
        )

        analysis_requests_total.labels(status="success").inc()
        feature_version = __import__("os").getenv("FEATURE_VERSION", "v1")
        return AnalysisResult(
            id=analysis_id,
            timestamp=start_time,
            file_name=file.filename,
            file_format=file_format.upper(),
            results=results,
            processing_time=processing_time,
            cache_hit=False,
            cad_document={
                "file_name": doc.file_name,
                "format": doc.format,
                "entity_count": doc.entity_count(),
                "entities": [
                    e.model_dump() for e in doc.entities[:200]
                ],  # limit entities for payload size
                "layers": doc.layers,
                "bounding_box": doc.bounding_box.model_dump(),
                "complexity": doc.complexity_bucket(),
                "metadata": doc.metadata,
                "raw_stats": doc.raw_stats,
            },
            feature_version=feature_version,
        )

    except json.JSONDecodeError:
        analysis_requests_total.labels(status="error").inc()
        analysis_errors_total.labels(stage="options", code="json_decode").inc()
        analysis_error_code_total.labels(code=ErrorCode.JSON_PARSE_ERROR.value).inc()
        # ErrorCode and build_error imported at module level
        err = build_error(
            ErrorCode.JSON_PARSE_ERROR, stage="options", message="Invalid options JSON format"
        )
        raise HTTPException(status_code=400, detail=err)
    except HTTPException as he:
        # If detail is already structured keep it, otherwise wrap
        analysis_requests_total.labels(status="error").inc()
        code = ErrorCode.INTERNAL_ERROR
        if he.status_code == 400:
            code = ErrorCode.INPUT_ERROR
        elif he.status_code == 404:
            code = ErrorCode.DATA_NOT_FOUND
        elif he.status_code == 413:
            code = ErrorCode.INPUT_SIZE_EXCEEDED
        elif he.status_code == 422:
            code = ErrorCode.BUSINESS_RULE_VIOLATION
        analysis_errors_total.labels(stage="general", code=str(he.status_code)).inc()
        if isinstance(he.detail, dict):  # already structured
            raise
        analysis_error_code_total.labels(code=code.value).inc()
        # build_error imported at module level
        err = build_error(code, stage="analysis", message=str(he.detail))
        raise HTTPException(status_code=he.status_code, detail=err)
    except Exception as e:
        analysis_requests_total.labels(status="error").inc()
        analysis_errors_total.labels(stage="general", code=ErrorCode.INTERNAL_ERROR.value).inc()
        analysis_error_code_total.labels(code=ErrorCode.INTERNAL_ERROR.value).inc()
        logger.error(f"Analysis failed for {file.filename}: {str(e)}")
        # ErrorCode and build_error imported at module level
        err = build_error(
            ErrorCode.INTERNAL_ERROR, stage="analysis", message=f"Analysis failed: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=err)


@router.post("/batch")
async def batch_analyze(
    files: list[UploadFile] = File(..., description="多个CAD文件"),
    options: str = Form(default='{"extract_features": true}'),
    api_key: str = Depends(get_api_key),
):
    """批量分析CAD文件"""
    results = []

    for file in files:
        try:
            result = await analyze_cad_file(file, options, api_key)
            results.append(result)
        except Exception as e:
            results.append({"file_name": file.filename, "error": str(e)})

    return {
        "total": len(files),
        "successful": len([r for r in results if "error" not in r]),
        "failed": len([r for r in results if "error" in r]),
        "results": results,
    }


@router.post("/similarity", response_model=SimilarityResult)
async def similarity_query(payload: SimilarityQuery, api_key: str = Depends(get_api_key)):
    """在已存在的向量之间计算相似度。"""
    from src.core.similarity import _VECTOR_STORE  # type: ignore

    if payload.reference_id not in _VECTOR_STORE:
        # ErrorCode and build_error imported at module level
        err = build_error(
            ErrorCode.DATA_NOT_FOUND,
            stage="similarity",
            message="Reference vector not found",
            id=payload.reference_id,
        )
        analysis_error_code_total.labels(code=ErrorCode.DATA_NOT_FOUND.value).inc()
        return SimilarityResult(
            reference_id=payload.reference_id,
            target_id=payload.target_id,
            score=0.0,
            method="cosine",
            dimension=0,
            status="reference_not_found",
            error=err,
        )
    if payload.target_id not in _VECTOR_STORE:
        # ErrorCode and build_error imported at module level
        err = build_error(
            ErrorCode.DATA_NOT_FOUND,
            stage="similarity",
            message="Target vector not found",
            id=payload.target_id,
        )
        analysis_error_code_total.labels(code=ErrorCode.DATA_NOT_FOUND.value).inc()
        return SimilarityResult(
            reference_id=payload.reference_id,
            target_id=payload.target_id,
            score=0.0,
            method="cosine",
            dimension=0,
            status="target_not_found",
            error=err,
        )
    ref = _VECTOR_STORE[payload.reference_id]
    tgt = _VECTOR_STORE[payload.target_id]
    if len(ref) != len(tgt):
        # ErrorCode and build_error imported at module level
        err = build_error(
            ErrorCode.VALIDATION_FAILED,
            stage="similarity",
            message="Vector dimension mismatch",
            expected=len(ref),
            found=len(tgt),
        )
        analysis_error_code_total.labels(code=ErrorCode.VALIDATION_FAILED.value).inc()
        return SimilarityResult(
            reference_id=payload.reference_id,
            target_id=payload.target_id,
            score=0.0,
            method="cosine",
            dimension=min(len(ref), len(tgt)),
            status="dimension_mismatch",
            error=err,
        )
    from src.core.similarity import _cosine  # type: ignore

    score = _cosine(ref, tgt)
    return SimilarityResult(
        reference_id=payload.reference_id,
        target_id=payload.target_id,
        score=round(score, 4),
        method="cosine",
        dimension=len(ref),
    )


@router.post("/similarity/topk", response_model=SimilarityTopKResponse)
async def similarity_topk(payload: SimilarityTopKQuery, api_key: str = Depends(get_api_key)):
    """基于已存储向量的 Top-K 相似检索。"""
    from src.core.similarity import InMemoryVectorStore  # type: ignore

    store = InMemoryVectorStore()
    if not store.exists(payload.target_id):
        # create_extended_error and ErrorCode imported at module level
        ext = create_extended_error(
            ErrorCode.DATA_NOT_FOUND, "Target vector not found", stage="similarity"
        )
        analysis_error_code_total.labels(code=ErrorCode.DATA_NOT_FOUND.value).inc()
        return SimilarityTopKResponse(
            target_id=payload.target_id,
            k=payload.k,
            results=[],
            status="target_not_found",
            error=ext.to_dict(),
        )
    base_vec = store.get(payload.target_id)
    assert base_vec is not None  # for type checker
    # Choose backend dynamically
    backend = os.getenv("VECTOR_STORE_BACKEND", "memory")
    import time as _time

    t0 = _time.time()
    if backend == "faiss":
        fstore = FaissVectorStore()
        raw = fstore.query(base_vec, top_k=max(1, payload.k + payload.offset))
        if not raw:  # fallback to memory if faiss unavailable
            store = InMemoryVectorStore()
            raw = store.query(base_vec, top_k=max(1, payload.k + payload.offset))
            backend = "memory_fallback"
    else:
        raw = store.query(base_vec, top_k=max(1, payload.k + payload.offset))
    from src.utils.analysis_metrics import vector_query_latency_seconds

    vector_query_latency_seconds.labels(backend=backend).observe(_time.time() - t0)
    items: list[SimilarityTopKItem] = []
    sliced = raw[payload.offset : payload.offset + payload.k]
    from src.core.similarity import InMemoryVectorStore  # type: ignore

    meta_store = InMemoryVectorStore()
    for vid, score in sliced:
        if payload.exclude_self and vid == payload.target_id:
            continue
        meta = meta_store.meta(vid) or {}
        if payload.material_filter and meta.get("material") != payload.material_filter:
            continue
        if payload.complexity_filter and meta.get("complexity") != payload.complexity_filter:
            continue
        items.append(
            SimilarityTopKItem(
                id=vid,
                score=round(score, 4),
                material=meta.get("material"),
                complexity=meta.get("complexity"),
                format=meta.get("format"),
            )
        )
    return SimilarityTopKResponse(
        target_id=payload.target_id,
        k=payload.k,
        results=items,
    )


@router.get("/vectors/distribution", response_model=VectorDistributionResponse)
async def vector_distribution_deprecated(api_key: str = Depends(get_api_key)):
    """Deprecated: moved to /api/v1/vectors_stats/distribution"""
    raise HTTPException(
        status_code=410,
        detail=create_migration_error(
            old_path="/api/v1/analyze/vectors/distribution",
            new_path="/api/v1/vectors_stats/distribution",
            method="GET",
        ),
    )


@router.post("/vectors/delete", response_model=VectorDeleteResponse)
async def delete_vector(payload: VectorDeleteRequest, api_key: str = Depends(get_api_key)):
    """Deprecated: moved to /api/v1/vectors/delete"""
    raise HTTPException(
        status_code=410,
        detail=create_migration_error(
            old_path="/api/v1/analyze/vectors/delete",
            new_path="/api/v1/vectors/delete",
            method="POST",
        ),
    )


@router.get("/vectors", response_model=VectorListResponse)
async def list_vectors(api_key: str = Depends(get_api_key)):
    """Deprecated: moved to /api/v1/vectors"""
    raise HTTPException(
        status_code=410,
        detail=create_migration_error(
            old_path="/api/v1/analyze/vectors", new_path="/api/v1/vectors", method="GET"
        ),
    )


@router.get("/vectors/stats", response_model=VectorStatsResponse)
async def vector_stats(api_key: str = Depends(get_api_key)):
    """Deprecated: moved to /api/v1/vectors_stats/stats"""
    raise HTTPException(
        status_code=410,
        detail=create_migration_error(
            old_path="/api/v1/analyze/vectors/stats",
            new_path="/api/v1/vectors_stats/stats",
            method="GET",
        ),
    )


@router.get("/process/rules/audit", response_model=ProcessRulesAuditResponse)
async def process_rules_audit(raw: bool = True, api_key: str = Depends(get_api_key)):
    import hashlib
    import os

    from src.core.process_rules import load_rules

    path = os.getenv("PROCESS_RULES_FILE", "config/process_rules.yaml")
    rules = load_rules(force_reload=True)
    version = rules.get("__meta__", {}).get("version", "v1")
    materials = sorted([m for m in rules.keys() if not m.startswith("__")])
    complexities: Dict[str, list[str]] = {}
    for m in materials:
        cm = rules.get(m, {})
        if isinstance(cm, dict):
            complexities[m] = sorted([c for c in cm.keys() if isinstance(cm.get(c), list)])
    file_hash: Optional[str] = None
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        file_hash = None
    return ProcessRulesAuditResponse(
        version=version,
        source=path if os.path.exists(path) else "embedded-defaults",
        hash=file_hash,
        materials=materials,
        complexities=complexities,
        raw=rules if raw else {},
    )


@router.post("/vectors/faiss/rebuild")
async def faiss_rebuild(api_key: str = Depends(get_api_key)):
    """手动触发 Faiss 索引重建 (延迟删除生效)."""
    if os.getenv("VECTOR_STORE_BACKEND", "memory") != "faiss":
        return {"rebuilt": False, "reason": "backend_not_faiss"}
    from src.core.similarity import FaissVectorStore  # type: ignore

    store = FaissVectorStore()
    ok = store.rebuild()  # type: ignore[attr-defined]
    return {"rebuilt": ok, "message": "Index rebuilt successfully" if ok else "Rebuild failed"}


@router.post("/vectors/update", response_model=VectorUpdateResponse)
async def update_vector(payload: VectorUpdateRequest, api_key: str = Depends(get_api_key)):
    from src.core.similarity import _VECTOR_META, _VECTOR_STORE  # type: ignore

    # create_extended_error and ErrorCode imported at module level
    from src.utils.analysis_metrics import analysis_error_code_total

    if payload.id not in _VECTOR_STORE:
        ext = create_extended_error(
            ErrorCode.DATA_NOT_FOUND, "Vector not found", stage="vector_update", id=payload.id
        )
        analysis_error_code_total.labels(code=ErrorCode.DATA_NOT_FOUND.value).inc()
        return VectorUpdateResponse(id=payload.id, status="not_found", error=ext.to_dict())
    vec = _VECTOR_STORE[payload.id]
    original_dim = len(vec)
    # Optional dimension enforcement via env flag
    enforce = __import__("os").getenv("ANALYSIS_VECTOR_DIM_CHECK", "0") == "1"
    try:
        if payload.replace is not None:
            if len(payload.replace) != original_dim:
                if enforce:
                    # build_error imported at module level
                    err = build_error(
                        ErrorCode.DIMENSION_MISMATCH,
                        stage="vector_update",
                        message=f"Expected {original_dim}, got {len(payload.replace)}",
                        id=payload.id,
                        expected=original_dim,
                        found=len(payload.replace),
                    )
                    analysis_error_code_total.labels(code=ErrorCode.DIMENSION_MISMATCH.value).inc()
                    from src.utils.analysis_metrics import vector_dimension_rejections_total

                    vector_dimension_rejections_total.labels(
                        reason="dimension_mismatch_replace"
                    ).inc()
                    raise HTTPException(status_code=409, detail=err)
                return VectorUpdateResponse(
                    id=payload.id,
                    status="dimension_mismatch",
                    dimension=original_dim,
                    error={
                        "code": ErrorCode.DIMENSION_MISMATCH.value,
                        "expected": original_dim,
                        "found": len(payload.replace),
                        "id": payload.id,
                    },
                )
            _VECTOR_STORE[payload.id] = [float(x) for x in payload.replace]
        elif payload.append is not None:
            if enforce and original_dim != 0:
                new_dim = original_dim + len(payload.append)
                if new_dim != original_dim:
                    # build_error imported at module level
                    err = build_error(
                        ErrorCode.DIMENSION_MISMATCH,
                        stage="vector_update",
                        message=f"Append changes dimension {original_dim}->{new_dim}",
                        id=payload.id,
                        expected=original_dim,
                        found=new_dim,
                    )
                    analysis_error_code_total.labels(code=ErrorCode.DIMENSION_MISMATCH.value).inc()
                    from src.utils.analysis_metrics import vector_dimension_rejections_total

                    vector_dimension_rejections_total.labels(
                        reason="dimension_mismatch_append"
                    ).inc()
                    raise HTTPException(status_code=409, detail=err)
            _VECTOR_STORE[payload.id] = vec + [float(float(x)) for x in payload.append]
        # update meta
        meta = _VECTOR_META.get(payload.id, {})
        if payload.material is not None:
            meta["material"] = payload.material
        if payload.complexity is not None:
            meta["complexity"] = payload.complexity
        if payload.format is not None:
            meta["format"] = payload.format
        _VECTOR_META[payload.id] = meta
        return VectorUpdateResponse(
            id=payload.id,
            status="updated",
            dimension=len(_VECTOR_STORE[payload.id]),
            feature_version=_VECTOR_META.get(payload.id, {}).get("feature_version"),
        )
    except Exception as e:
        ext = create_extended_error(ErrorCode.INTERNAL_ERROR, str(e), stage="vector_update")
        analysis_error_code_total.labels(code=ErrorCode.INTERNAL_ERROR.value).inc()
        return VectorUpdateResponse(id=payload.id, status="error", error=ext.to_dict())


@router.post("/vectors/migrate", response_model=VectorMigrateResponse)
async def migrate_vectors(payload: VectorMigrateRequest, api_key: str = Depends(get_api_key)):
    """在线迁移指定向量到目标特征版本 (重算特征并替换)."""
    from src.core.feature_extractor import FeatureExtractor
    from src.core.similarity import _VECTOR_META, _VECTOR_STORE  # type: ignore
    from src.models.cad_document import CadDocument
    from src.utils.cache import get_cached_result

    items: list[VectorMigrateItem] = []
    migrated = 0
    skipped = 0
    default_version = os.getenv("FEATURE_VERSION", "v1")
    started_at = datetime.now(timezone.utc)
    batch_id = str(uuid.uuid4())
    dry_run_total = 0
    for vid in payload.ids:
        meta = _VECTOR_META.get(vid, {})
        if vid not in _VECTOR_STORE:
            items.append(
                VectorMigrateItem(
                    id=vid,
                    status="not_found",
                    to_version=payload.to_version,
                    error="vector_missing",
                )
            )
            skipped += 1
            continue
        from_version = meta.get("feature_version", default_version)
        original_dim = len(_VECTOR_STORE[vid])
        if from_version == payload.to_version:
            items.append(
                VectorMigrateItem(
                    id=vid,
                    status="skipped",
                    from_version=from_version,
                    to_version=payload.to_version,
                    dimension_before=original_dim,
                    dimension_after=original_dim,
                )
            )
            skipped += 1
            continue
        cached = await get_cached_result(f"analysis_result:{vid}")
        if not cached:
            items.append(
                VectorMigrateItem(
                    id=vid,
                    status="skipped",
                    from_version=from_version,
                    to_version=payload.to_version,
                    error="cached_result_missing",
                )
            )
            skipped += 1
            continue
        stats = cached.get("statistics", {})
        bbox = stats.get("bounding_box", {})
        doc = CadDocument(
            file_name=cached.get("file_name", vid), format=cached.get("file_format", "unknown")
        )
        doc.bounding_box.min_x = bbox.get("min_x", 0.0)
        doc.bounding_box.min_y = bbox.get("min_y", 0.0)
        doc.bounding_box.min_z = bbox.get("min_z", 0.0)
        doc.bounding_box.max_x = bbox.get("max_x", 0.0)
        doc.bounding_box.max_y = bbox.get("max_y", 0.0)
        doc.bounding_box.max_z = bbox.get("max_z", 0.0)
        # Use explicit version parameter instead of modifying os.environ (concurrent-safe)
        extractor = FeatureExtractor(feature_version=payload.to_version)
        try:
            new_features = await extractor.extract(doc)
            new_vector = extractor.flatten(new_features)
            if payload.dry_run:
                items.append(
                    VectorMigrateItem(
                        id=vid,
                        status="dry_run",
                        from_version=from_version,
                        to_version=payload.to_version,
                        dimension_before=original_dim,
                        dimension_after=len(new_vector),
                    )
                )
                skipped += 1
                dry_run_total += 1
            else:
                from src.core.vector_layouts import VECTOR_LAYOUT_BASE

                _VECTOR_STORE[vid] = [float(x) for x in new_vector]
                meta.update(
                    {
                        "feature_version": payload.to_version,
                        "geometric_dim": str(len(new_features.get("geometric", []))),
                        "semantic_dim": str(len(new_features.get("semantic", []))),
                        "total_dim": str(len(new_vector)),
                        "vector_layout": VECTOR_LAYOUT_BASE,
                    }
                )
                meta.pop("l3_3d_dim", None)
                items.append(
                    VectorMigrateItem(
                        id=vid,
                        status="migrated",
                        from_version=from_version,
                        to_version=payload.to_version,
                        dimension_before=original_dim,
                        dimension_after=len(new_vector),
                    )
                )
                migrated += 1
        except Exception as e:
            items.append(
                VectorMigrateItem(
                    id=vid,
                    status="error",
                    from_version=from_version,
                    to_version=payload.to_version,
                    error=str(e),
                )
            )
            skipped += 1
    finished_at = datetime.now(timezone.utc)
    try:
        import src.core.similarity as _sim  # type: ignore

        if not hasattr(_sim, "_MIGRATION_STATUS"):
            _sim._MIGRATION_STATUS = {}
        hist = _sim._MIGRATION_STATUS.get("history", [])
        entry = {
            "migration_id": batch_id,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "total": len(payload.ids),
            "migrated": migrated,
            "skipped": skipped,
            "dry_run_total": dry_run_total,
        }
        hist.append(entry)
        if len(hist) > 10:
            hist = hist[-10:]
        _sim._MIGRATION_STATUS.update(
            {
                "last_migration_id": batch_id,
                "last_started_at": started_at.isoformat(),
                "last_finished_at": finished_at.isoformat(),
                "last_total": len(payload.ids),
                "last_migrated": migrated,
                "last_skipped": skipped,
                "last_dry_run_total": dry_run_total,
                "history": hist,
            }
        )
    except Exception:
        pass
    return VectorMigrateResponse(
        total=len(payload.ids),
        migrated=migrated,
        skipped=skipped,
        items=items,
        migration_id=batch_id,
        started_at=started_at,
        finished_at=finished_at,
        dry_run_total=dry_run_total,
    )


@router.get("/vectors/migrate/status", response_model=VectorMigrationStatusResponse)
async def vector_migration_status(api_key: str = Depends(get_api_key)):
    import src.core.similarity as _sim  # type: ignore
    from src.core.similarity import _VECTOR_META, _VECTOR_STORE  # type: ignore

    versions: Dict[str, int] = {}
    for meta in _VECTOR_META.values():
        ver = meta.get("feature_version", "unknown")
        versions[ver] = versions.get(ver, 0) + 1
    st = getattr(_sim, "_MIGRATION_STATUS", {})

    def _dt(val: Optional[str]):
        if not val:
            return None
        try:
            return datetime.fromisoformat(val)
        except Exception:
            return None

    return VectorMigrationStatusResponse(
        last_migration_id=st.get("last_migration_id"),
        last_started_at=_dt(st.get("last_started_at")),
        last_finished_at=_dt(st.get("last_finished_at")),
        last_total=st.get("last_total"),
        last_migrated=st.get("last_migrated"),
        last_skipped=st.get("last_skipped"),
        pending_vectors=len(_VECTOR_STORE),
        feature_versions=versions,
        history=st.get("history"),
    )


class FeaturesDiffResponse(BaseModel):
    id_a: str
    id_b: str
    dimension: Optional[int] = None
    diffs: list[Dict[str, Any]]
    status: str
    error: Optional[Dict[str, Any]] = None


@router.get("/features/diff", response_model=FeaturesDiffResponse, deprecated=True)
async def features_diff_deprecated(id_a: str, id_b: str, api_key: str = Depends(get_api_key)):
    """Deprecated: moved to /api/v1/features/diff"""
    raise HTTPException(
        status_code=410,
        detail=create_migration_error(
            old_path="/api/v1/analyze/features/diff", new_path="/api/v1/features/diff", method="GET"
        ),
    )


class ModelReloadRequest(BaseModel):
    path: str = Field(description="模型文件路径")
    expected_version: Optional[str] = Field(default=None, description="期望模型版本")
    force: bool = Field(default=False, description="强制重载忽略版本校验")


class ModelReloadResponse(BaseModel):
    status: str
    model_version: Optional[str] = None
    hash: Optional[str] = None
    error: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(protected_namespaces=())


@router.post("/model/reload", response_model=ModelReloadResponse, deprecated=True)
async def model_reload_deprecated(payload: ModelReloadRequest, api_key: str = Depends(get_api_key)):
    """Deprecated: moved to /api/v1/model/reload"""
    raise HTTPException(
        status_code=410,
        detail=create_migration_error(
            old_path="/api/v1/analyze/model/reload", new_path="/api/v1/model/reload", method="POST"
        ),
    )


class OrphanCleanupResponse(BaseModel):
    status: str
    cleaned: int
    total_orphans_detected: Optional[int] = None
    error: Optional[Dict[str, Any]] = None


@router.delete("/vectors/orphans", response_model=OrphanCleanupResponse, deprecated=True)
async def cleanup_orphan_vectors_deprecated(
    threshold: int = Query(0, description="最小孤儿向量数量触发清理"),
    force: bool = Query(False, description="强制执行清理"),
    dry_run: bool = Query(False, description="仅统计不执行删除"),
    verbose: bool = Query(False, description="输出部分孤儿ID样例 (限制10个)"),
    api_key: str = Depends(get_api_key),
):
    """Deprecated: moved to /api/v1/maintenance/orphans"""
    raise HTTPException(
        status_code=410,
        detail=create_migration_error(
            old_path="/api/v1/analyze/vectors/orphans",
            new_path="/api/v1/maintenance/orphans",
            method="DELETE",
        ),
    )


class DriftStatusResponse(BaseModel):
    material_current: Dict[str, int]
    material_baseline: Optional[Dict[str, int]] = None
    material_drift_score: Optional[float] = None
    prediction_current: Dict[str, int]
    prediction_baseline: Optional[Dict[str, int]] = None
    prediction_drift_score: Optional[float] = None
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


@router.get("/drift", response_model=DriftStatusResponse)
async def drift_status(api_key: str = Depends(get_api_key)):
    import os
    import time
    from collections import Counter

    min_count = int(os.getenv("DRIFT_BASELINE_MIN_COUNT", "100"))
    max_age = int(os.getenv("DRIFT_BASELINE_MAX_AGE_SECONDS", "86400"))
    auto_refresh_enabled = os.getenv("DRIFT_BASELINE_AUTO_REFRESH", "1") == "1"
    mats = _DRIFT_STATE["materials"]
    preds = _DRIFT_STATE["predictions"]
    material_current_counts = dict(Counter(mats))
    prediction_current_counts = dict(Counter(preds))
    material_baseline_counts = (
        dict(Counter(_DRIFT_STATE["baseline_materials"]))
        if _DRIFT_STATE["baseline_materials"]
        else None
    )
    prediction_baseline_counts = (
        dict(Counter(_DRIFT_STATE["baseline_predictions"]))
        if _DRIFT_STATE["baseline_predictions"]
        else None
    )
    from src.utils.drift import compute_drift

    mat_score = None
    if material_baseline_counts:
        material_age = int(time.time() - _DRIFT_STATE.get("baseline_materials_ts", 0))
        if auto_refresh_enabled and material_age > max_age and len(mats) >= min_count:
            _DRIFT_STATE["baseline_materials"] = list(mats)
            _DRIFT_STATE["baseline_materials_ts"] = time.time()
            try:
                from src.utils.analysis_metrics import drift_baseline_refresh_total

                drift_baseline_refresh_total.labels(type="material", trigger="stale").inc()
            except Exception:
                pass
            material_baseline_counts = dict(Counter(mats))
        mat_score = compute_drift(mats, _DRIFT_STATE["baseline_materials"])  # type: ignore
    else:
        # establish baseline when threshold met
        if len(mats) >= min_count:
            _DRIFT_STATE["baseline_materials"] = list(mats)
            _DRIFT_STATE["baseline_materials_ts"] = time.time()
            try:
                from src.utils.analysis_metrics import drift_baseline_created_total

                drift_baseline_created_total.labels(type="material").inc()
            except Exception:
                pass
    pred_score = None
    if prediction_baseline_counts:
        prediction_age = int(time.time() - _DRIFT_STATE.get("baseline_predictions_ts", 0))
        if auto_refresh_enabled and prediction_age > max_age and len(preds) >= min_count:
            _DRIFT_STATE["baseline_predictions"] = list(preds)
            _DRIFT_STATE["baseline_predictions_ts"] = time.time()
            try:
                from src.utils.analysis_metrics import drift_baseline_refresh_total

                drift_baseline_refresh_total.labels(type="prediction", trigger="stale").inc()
            except Exception:
                pass
            prediction_baseline_counts = dict(Counter(preds))
        pred_score = compute_drift(preds, _DRIFT_STATE["baseline_predictions"])  # type: ignore
    else:
        if len(preds) >= min_count:
            _DRIFT_STATE["baseline_predictions"] = list(preds)
            _DRIFT_STATE["baseline_predictions_ts"] = time.time()
            try:
                from src.utils.analysis_metrics import drift_baseline_created_total

                drift_baseline_created_total.labels(type="prediction").inc()
            except Exception:
                pass
    status = "baseline_pending" if (len(mats) < min_count or len(preds) < min_count) else "ok"
    baseline_material_age = None
    baseline_prediction_age = None
    # Use first timestamp index to approximate age (list length as proxy)
    if _DRIFT_STATE["baseline_materials_ts"]:
        baseline_material_age = int(
            time.time() - _DRIFT_STATE["baseline_materials_ts"]
        )  # seconds since baseline snapshot
        try:
            from src.utils.analysis_metrics import baseline_material_age_seconds

            baseline_material_age_seconds.set(baseline_material_age)
        except Exception:
            pass
    if _DRIFT_STATE["baseline_predictions_ts"]:
        baseline_prediction_age = int(
            time.time() - _DRIFT_STATE["baseline_predictions_ts"]
        )  # seconds since baseline snapshot
        try:
            from src.utils.analysis_metrics import baseline_prediction_age_seconds

            baseline_prediction_age_seconds.set(baseline_prediction_age)
        except Exception:
            pass
    stale_flag = None
    try:
        if baseline_material_age is not None and baseline_material_age > max_age:
            stale_flag = True
        if baseline_prediction_age is not None and baseline_prediction_age > max_age:
            stale_flag = True
        if stale_flag is None and (
            baseline_material_age is not None or baseline_prediction_age is not None
        ):
            stale_flag = False
    except Exception:
        pass
    baseline_material_created_at = None
    baseline_prediction_created_at = None
    if _DRIFT_STATE.get("baseline_materials_ts"):
        baseline_material_created_at = datetime.fromtimestamp(
            _DRIFT_STATE["baseline_materials_ts"], tz=timezone.utc
        )
        if _DRIFT_STATE.get("baseline_materials_startup_mark") is None:
            try:
                from src.utils.analysis_metrics import drift_baseline_refresh_total

                drift_baseline_refresh_total.labels(type="material", trigger="startup").inc()
            except Exception:
                pass
            _DRIFT_STATE["baseline_materials_startup_mark"] = True
    if _DRIFT_STATE.get("baseline_predictions_ts"):
        baseline_prediction_created_at = datetime.fromtimestamp(
            _DRIFT_STATE["baseline_predictions_ts"], tz=timezone.utc
        )
        if _DRIFT_STATE.get("baseline_predictions_startup_mark") is None:
            try:
                from src.utils.analysis_metrics import drift_baseline_refresh_total

                drift_baseline_refresh_total.labels(type="prediction", trigger="startup").inc()
            except Exception:
                pass
            _DRIFT_STATE["baseline_predictions_startup_mark"] = True
    return DriftStatusResponse(
        material_current=material_current_counts,
        material_baseline=material_baseline_counts,
        material_drift_score=mat_score,
        prediction_current=prediction_current_counts,
        prediction_baseline=prediction_baseline_counts,
        prediction_drift_score=pred_score,
        baseline_min_count=min_count,
        materials_total=len(mats),
        predictions_total=len(preds),
        status=status,
        baseline_material_age=baseline_material_age,
        baseline_prediction_age=baseline_prediction_age,
        baseline_material_created_at=baseline_material_created_at,
        baseline_prediction_created_at=baseline_prediction_created_at,
        stale=stale_flag,
    )


@router.post("/drift/reset", response_model=DriftResetResponse)
async def drift_reset(api_key: str = Depends(get_api_key)):
    # Reset baseline lists only; keep current observations for immediate recompute when threshold met again
    reset_material = bool(_DRIFT_STATE["baseline_materials"])
    reset_predictions = bool(_DRIFT_STATE["baseline_predictions"])
    _DRIFT_STATE["baseline_materials"] = []
    _DRIFT_STATE["baseline_predictions"] = []
    _DRIFT_STATE["baseline_materials_ts"] = None
    _DRIFT_STATE["baseline_predictions_ts"] = None
    # Remove persisted baselines if redis present
    try:
        client = __import__("src.utils.cache", fromlist=["get_client"]).get_client()
        if client is not None:
            await client.delete("baseline:material")  # type: ignore[attr-defined]
            await client.delete("baseline:class")  # type: ignore[attr-defined]
    except Exception:
        pass
    return DriftResetResponse(
        status="ok", reset_material=reset_material, reset_predictions=reset_predictions
    )


class DriftBaselineStatusResponse(BaseModel):
    status: str
    material_age: Optional[int] = None
    prediction_age: Optional[int] = None
    material_created_at: Optional[datetime] = None
    prediction_created_at: Optional[datetime] = None
    stale: Optional[bool] = None
    max_age_seconds: int


class FeatureCacheStatsResponse(BaseModel):
    """(Deprecated location) Moved to health.py"""

    status: str
    size: int
    capacity: int
    ttl_seconds: int
    hit_ratio: Optional[float] = None
    hits: Optional[int] = None
    misses: Optional[int] = None
    evictions: Optional[int] = None


class FaissHealthResponse(BaseModel):
    """(Deprecated location) Moved to health.py"""

    available: bool
    index_size: Optional[int]
    dim: Optional[int]
    age_seconds: Optional[int]
    pending_delete: Optional[int]
    max_pending_delete: Optional[int]
    normalize: Optional[bool]
    status: str


@router.get("/features/cache", response_model=FeatureCacheStatsResponse)
async def feature_cache_stats(api_key: str = Depends(get_api_key)):
    """Backward-compatible redirect stub. Prefer /api/v1/health/features/cache."""
    raise HTTPException(
        status_code=410,
        detail=create_migration_error(
            old_path="/api/v1/analyze/features/cache",
            new_path="/api/v1/health/features/cache",
            method="GET",
        ),
    )


@router.get("/drift/baseline/status", response_model=DriftBaselineStatusResponse)
async def drift_baseline_status(api_key: str = Depends(get_api_key)):
    import os
    import time

    max_age = int(os.getenv("DRIFT_BASELINE_MAX_AGE_SECONDS", "86400"))
    material_age = None
    prediction_age = None
    material_created_at = None
    prediction_created_at = None
    if _DRIFT_STATE.get("baseline_materials_ts"):
        material_age = int(time.time() - _DRIFT_STATE["baseline_materials_ts"])
        material_created_at = datetime.fromtimestamp(
            _DRIFT_STATE["baseline_materials_ts"], tz=timezone.utc
        )
    if _DRIFT_STATE.get("baseline_predictions_ts"):
        prediction_age = int(time.time() - _DRIFT_STATE["baseline_predictions_ts"])
        prediction_created_at = datetime.fromtimestamp(
            _DRIFT_STATE["baseline_predictions_ts"], tz=timezone.utc
        )
    stale_flag = None
    if material_age and material_age > max_age:
        stale_flag = True
    if prediction_age and prediction_age > max_age:
        stale_flag = True if stale_flag is None else stale_flag or True
    if stale_flag is None and (material_age or prediction_age):
        stale_flag = False
    status = "stale" if stale_flag else "ok"
    if material_age is None and prediction_age is None:
        status = "no_baseline"
    return DriftBaselineStatusResponse(
        status=status,
        material_age=material_age,
        prediction_age=prediction_age,
        material_created_at=material_created_at,
        prediction_created_at=prediction_created_at,
        stale=stale_flag,
        max_age_seconds=max_age,
    )


@router.get("/faiss/health", response_model=FaissHealthResponse)
async def faiss_health(api_key: str = Depends(get_api_key)):
    """Deprecated: moved to /api/v1/health/faiss"""
    raise HTTPException(
        status_code=410,
        detail=create_migration_error(
            old_path="/api/v1/analyze/faiss/health", new_path="/api/v1/health/faiss", method="GET"
        ),
    )


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
