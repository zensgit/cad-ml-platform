from __future__ import annotations

import os
import time
import uuid
from typing import Any, Callable

from fastapi import HTTPException


def _build_qdrant_similar_item(
    *,
    result: Any,
    extract_vector_label_contract_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    meta = result.metadata or {}
    label_contract = extract_vector_label_contract_fn(meta)
    dimension = len(result.vector or [])
    if dimension <= 0:
        try:
            dimension = int(meta.get("total_dim") or 0)
        except (TypeError, ValueError):
            dimension = 0
    return {
        "id": result.id,
        "score": round(float(result.score), 4),
        "material": meta.get("material"),
        "complexity": meta.get("complexity"),
        "format": meta.get("format"),
        "dimension": dimension,
        "part_type": label_contract.get("part_type"),
        "fine_part_type": label_contract.get("fine_part_type"),
        "coarse_part_type": label_contract.get("coarse_part_type"),
        "decision_source": label_contract.get("decision_source"),
        "is_coarse_label": label_contract.get("is_coarse_label"),
    }


def _build_memory_similar_item(
    *,
    result_id: str,
    score: float,
    meta: dict[str, Any],
    vector_dimension: int,
    extract_vector_label_contract_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    label_contract = extract_vector_label_contract_fn(meta)
    return {
        "id": result_id,
        "score": round(score, 4),
        "material": meta.get("material"),
        "complexity": meta.get("complexity"),
        "format": meta.get("format"),
        "dimension": vector_dimension,
        "part_type": label_contract.get("part_type"),
        "fine_part_type": label_contract.get("fine_part_type"),
        "coarse_part_type": label_contract.get("coarse_part_type"),
        "decision_source": label_contract.get("decision_source"),
        "is_coarse_label": label_contract.get("is_coarse_label"),
    }


async def run_vector_batch_similarity(
    *,
    payload: Any,
    batch_item_cls: type[Any],
    batch_response_cls: type[Any],
    error_code_cls: Any,
    build_error_fn: Callable[..., dict[str, Any]],
    get_qdrant_store_fn: Callable[[], Any],
    build_filter_conditions_fn: Callable[..., dict[str, Any]],
) -> Any:
    from src.core.similarity import (
        _VECTOR_META,
        _VECTOR_STORE,
        extract_vector_label_contract,
        get_degraded_mode_info,
        get_vector_store,
    )
    from src.utils.analysis_metrics import (
        vector_query_backend_total,
        vector_query_batch_latency_seconds,
    )

    batch_id = str(uuid.uuid4())
    max_batch = int(os.getenv("BATCH_SIMILARITY_MAX_IDS", "200"))
    if len(payload.ids) > max_batch:
        from src.utils.analysis_metrics import analysis_error_code_total, analysis_rejections_total

        err = build_error_fn(
            error_code_cls.INPUT_VALIDATION_FAILED,
            stage="batch_similarity",
            message="Batch size exceeds limit",
            batch_size=len(payload.ids),
            max_batch=max_batch,
        )
        analysis_rejections_total.labels(reason="batch_too_large").inc()
        analysis_error_code_total.labels(code=error_code_cls.INPUT_VALIDATION_FAILED.value).inc()
        raise HTTPException(status_code=422, detail=err)

    start_time = time.time()
    batch_size = len(payload.ids)
    if batch_size <= 5:
        size_range = "small"
    elif batch_size <= 20:
        size_range = "medium"
    else:
        size_range = "large"

    items: list[Any] = []
    successful = 0
    failed = 0

    qdrant_store = get_qdrant_store_fn()
    if qdrant_store is not None:
        filter_conditions = build_filter_conditions_fn(
            material_filter=payload.material,
            complexity_filter=payload.complexity,
            fine_part_type_filter=None,
            coarse_part_type_filter=None,
            decision_source_filter=None,
            is_coarse_label_filter=None,
        )

        for vid in payload.ids:
            target = await qdrant_store.get_vector(vid)
            if target is None:
                items.append(
                    batch_item_cls(
                        id=vid,
                        status="not_found",
                        error=build_error_fn(
                            error_code_cls.DATA_NOT_FOUND,
                            stage="batch_similarity",
                            message="Vector not found",
                            id=vid,
                        ),
                    )
                )
                failed += 1
                continue

            try:
                query_vector = list(target.vector or [])
                results = await qdrant_store.search_similar(
                    query_vector,
                    top_k=payload.top_k + 1,
                    filter_conditions=filter_conditions or None,
                    score_threshold=payload.min_score,
                    with_vectors=True,
                )

                similar: list[dict[str, Any]] = []
                for result in results:
                    if result.id == vid:
                        continue
                    similar.append(
                        _build_qdrant_similar_item(
                            result=result,
                            extract_vector_label_contract_fn=extract_vector_label_contract,
                        )
                    )
                    if len(similar) >= payload.top_k:
                        break

                items.append(batch_item_cls(id=vid, status="success", similar=similar))
                successful += 1
            except Exception as exc:
                items.append(
                    batch_item_cls(
                        id=vid,
                        status="error",
                        error=build_error_fn(
                            error_code_cls.INTERNAL_ERROR,
                            stage="batch_similarity",
                            message="Query failed",
                            id=vid,
                            detail=str(exc),
                        ),
                    )
                )
                failed += 1

        duration = time.time() - start_time
        vector_query_batch_latency_seconds.labels(batch_size_range=size_range).observe(duration)

        if successful > 0 and all((not it.similar) for it in items if it.status == "success"):
            try:
                from src.utils.analysis_metrics import analysis_rejections_total

                analysis_rejections_total.labels(reason="batch_empty_results").inc()
            except Exception:
                pass

        return batch_response_cls(
            total=len(payload.ids),
            successful=successful,
            failed=failed,
            items=items,
            batch_id=batch_id,
            duration_ms=round(duration * 1000, 2),
            fallback=None,
            degraded=False,
        )

    store = get_vector_store()
    is_fallback = bool(getattr(store, "_fallback_from", None))
    requested_backend = getattr(store, "_requested_backend", os.getenv("VECTOR_STORE_BACKEND", "memory"))
    expected_backend = requested_backend
    if not is_fallback and not getattr(store, "_available", True):
        is_fallback = True
    elif not is_fallback and expected_backend == "faiss":
        from src.core.similarity import FaissVectorStore

        if not isinstance(FaissVectorStore, type):
            is_fallback = True
        elif not isinstance(store, FaissVectorStore):
            is_fallback = True
    if is_fallback:
        try:
            vector_query_backend_total.labels(backend="memory_fallback").inc()
        except Exception:
            pass

    for vid in payload.ids:
        if vid not in _VECTOR_STORE:
            items.append(
                batch_item_cls(
                    id=vid,
                    status="not_found",
                    error=build_error_fn(
                        error_code_cls.DATA_NOT_FOUND,
                        stage="batch_similarity",
                        message="Vector not found",
                        id=vid,
                    ),
                )
            )
            failed += 1
            continue

        try:
            vec = _VECTOR_STORE[vid]
            results = store.query(vec, top_k=payload.top_k + 1)

            similar: list[dict[str, Any]] = []
            for result_id, score in results:
                if result_id == vid:
                    continue
                if payload.min_score is not None and score < payload.min_score:
                    continue

                meta = _VECTOR_META.get(result_id, {})
                if payload.material and meta.get("material") != payload.material:
                    continue
                if payload.complexity and meta.get("complexity") != payload.complexity:
                    continue
                if payload.format and meta.get("format") != payload.format:
                    continue

                similar.append(
                    _build_memory_similar_item(
                        result_id=result_id,
                        score=score,
                        meta=meta,
                        vector_dimension=len(_VECTOR_STORE.get(result_id, [])),
                        extract_vector_label_contract_fn=extract_vector_label_contract,
                    )
                )
                if len(similar) >= payload.top_k:
                    break

            items.append(batch_item_cls(id=vid, status="success", similar=similar))
            successful += 1
        except Exception as exc:
            items.append(
                batch_item_cls(
                    id=vid,
                    status="error",
                    error=build_error_fn(
                        error_code_cls.INTERNAL_ERROR,
                        stage="batch_similarity",
                        message="Query failed",
                        id=vid,
                        detail=str(exc),
                    ),
                )
            )
            failed += 1

    duration = time.time() - start_time
    vector_query_batch_latency_seconds.labels(batch_size_range=size_range).observe(duration)

    if successful > 0 and all((not it.similar) for it in items if it.status == "success"):
        try:
            from src.utils.analysis_metrics import analysis_rejections_total

            analysis_rejections_total.labels(reason="batch_empty_results").inc()
        except Exception:
            pass

    degraded_info = get_degraded_mode_info() or {}
    degraded_flag = bool(degraded_info.get("degraded", False))
    fallback = bool(is_fallback or degraded_flag)
    is_degraded = bool(degraded_flag or is_fallback)

    return batch_response_cls(
        total=len(payload.ids),
        successful=successful,
        failed=failed,
        items=items,
        batch_id=batch_id,
        duration_ms=round(duration * 1000, 2),
        fallback=fallback if fallback else None,
        degraded=is_degraded,
    )


__all__ = ["run_vector_batch_similarity"]
