"""Shared success-result envelope helpers for analyze flows."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

from src.utils.analysis_metrics import analysis_requests_total
from src.utils.analysis_result_store import store_analysis_result
from src.utils.cache import cache_result, set_cache

PersistCacheFn = Callable[[str, Dict[str, Any]], Awaitable[Any]]
PersistStoreFn = Callable[[str, Dict[str, Any]], Awaitable[Any]]
PersistResultFn = Callable[[str, Dict[str, Any], int], Awaitable[Any]]
TimeFactory = Callable[[], datetime]


def build_analysis_statistics(
    *,
    doc: Any,
    stage_times: Mapping[str, float],
) -> Dict[str, Any]:
    return {
        "entity_count": doc.entity_count(),
        "layer_count": len(doc.layers),
        "bounding_box": doc.bounding_box.model_dump(),
        "complexity": doc.complexity_bucket(),
        "stages": dict(stage_times),
    }


def build_analysis_cad_document_payload(doc: Any) -> Dict[str, Any]:
    return {
        "file_name": doc.file_name,
        "format": doc.format,
        "entity_count": doc.entity_count(),
        "entities": [entity.model_dump() for entity in doc.entities[:200]],
        "layers": doc.layers,
        "bounding_box": doc.bounding_box.model_dump(),
        "complexity": doc.complexity_bucket(),
        "metadata": doc.metadata,
        "raw_stats": doc.raw_stats,
    }


async def finalize_analysis_success(
    *,
    analysis_id: str,
    start_time: datetime,
    file_name: str,
    file_format: str,
    results: Dict[str, Any],
    doc: Any,
    stage_times: Mapping[str, float],
    analysis_cache_key: str,
    vector_context: Optional[Mapping[str, Any]] = None,
    material: Optional[str] = None,
    unified_data: Optional[Mapping[str, Any]] = None,
    logger_instance: Optional[Any] = None,
    cache_result_fn: Optional[PersistResultFn] = None,
    set_cache_fn: Optional[PersistCacheFn] = None,
    store_analysis_result_fn: Optional[PersistStoreFn] = None,
    time_factory: Optional[TimeFactory] = None,
) -> Dict[str, Any]:
    results["statistics"] = build_analysis_statistics(doc=doc, stage_times=stage_times)

    await (cache_result_fn or cache_result)(analysis_cache_key, results, ttl=3600)
    await (set_cache_fn or set_cache)(f"analysis_result:{analysis_id}", results, ttl_seconds=3600)
    await (store_analysis_result_fn or store_analysis_result)(analysis_id, results)

    now = (time_factory or (lambda: datetime.now(timezone.utc)))()
    processing_time = (now - start_time).total_seconds()

    if logger_instance is not None:
        logger_instance.info(
            "analysis.completed",
            extra={
                "file": file_name,
                "analysis_id": analysis_id,
                "processing_time_s": round(processing_time, 4),
                "stages": dict(stage_times),
                "feature_vector_dim": (vector_context or {}).get("feature_vector_dim", 0),
                "material": material,
                "complexity": (unified_data or {}).get("complexity"),
            },
        )

    analysis_requests_total.labels(status="success").inc()
    feature_version = os.getenv("FEATURE_VERSION", "v1")
    return {
        "id": analysis_id,
        "timestamp": start_time,
        "file_name": file_name,
        "file_format": file_format.upper(),
        "results": results,
        "processing_time": processing_time,
        "cache_hit": False,
        "cad_document": build_analysis_cad_document_payload(doc),
        "feature_version": feature_version,
    }


__all__ = [
    "build_analysis_cad_document_payload",
    "build_analysis_statistics",
    "finalize_analysis_success",
]
