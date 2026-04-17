"""Shared similarity query helpers for analyze vector endpoints."""

from __future__ import annotations

import os
import time as _time
from typing import Any, Awaitable, Callable, Dict, Optional, Protocol

from src.core.errors_extended import ErrorCode, build_error, create_extended_error
from src.core.similarity import FaissVectorStore

ErrorRecorder = Callable[[str], Any]
LatencyObserver = Callable[[str, float], Any]
QdrantStoreGetter = Callable[[], Any]


class SimilarityQueryPayload(Protocol):
    reference_id: str
    target_id: str


class SimilarityTopKPayload(Protocol):
    target_id: str
    k: int
    exclude_self: bool
    offset: int
    material_filter: Optional[str]
    complexity_filter: Optional[str]
    fine_part_type_filter: Optional[str]
    coarse_part_type_filter: Optional[str]
    decision_source_filter: Optional[str]
    is_coarse_label_filter: Optional[bool]


def _record_error(
    error_code: ErrorCode,
    error_recorder: Optional[ErrorRecorder],
) -> None:
    if error_recorder is None:
        return
    error_recorder(error_code.value)


def _build_similarity_result_payload(
    *,
    reference_id: str,
    target_id: str,
    score: float = 0.0,
    method: str = "cosine",
    dimension: int = 0,
    status: Optional[str] = None,
    error: Optional[Dict[str, Any]] = None,
    reference_contract: Optional[Dict[str, Any]] = None,
    target_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    reference_contract = reference_contract or {}
    target_contract = target_contract or {}
    return {
        "reference_id": reference_id,
        "target_id": target_id,
        "score": score,
        "method": method,
        "dimension": dimension,
        "reference_part_type": reference_contract.get("part_type"),
        "reference_fine_part_type": reference_contract.get("fine_part_type"),
        "reference_coarse_part_type": reference_contract.get("coarse_part_type"),
        "reference_decision_source": reference_contract.get("decision_source"),
        "reference_is_coarse_label": reference_contract.get("is_coarse_label"),
        "target_part_type": target_contract.get("part_type"),
        "target_fine_part_type": target_contract.get("fine_part_type"),
        "target_coarse_part_type": target_contract.get("coarse_part_type"),
        "target_decision_source": target_contract.get("decision_source"),
        "target_is_coarse_label": target_contract.get("is_coarse_label"),
        "status": status,
        "error": error,
    }


def _build_similarity_topk_error_payload(
    *,
    target_id: str,
    k: int,
    status: str,
    error: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "target_id": target_id,
        "k": k,
        "results": [],
        "status": status,
        "error": error,
    }


def _build_vector_filter_conditions(payload: SimilarityTopKPayload) -> Dict[str, Any]:
    filter_conditions: Dict[str, Any] = {}
    if payload.material_filter:
        filter_conditions["material"] = payload.material_filter
    if payload.complexity_filter:
        filter_conditions["complexity"] = payload.complexity_filter
    if payload.fine_part_type_filter:
        filter_conditions["fine_part_type"] = payload.fine_part_type_filter
    if payload.coarse_part_type_filter:
        filter_conditions["coarse_part_type"] = payload.coarse_part_type_filter
    if payload.decision_source_filter:
        filter_conditions["decision_source"] = payload.decision_source_filter
    if payload.is_coarse_label_filter is not None:
        filter_conditions["is_coarse_label"] = payload.is_coarse_label_filter
    return filter_conditions


def matches_similarity_topk_filters(
    payload: SimilarityTopKPayload,
    meta: Dict[str, Any],
    label_contract: Dict[str, Any],
) -> bool:
    if payload.material_filter and meta.get("material") != payload.material_filter:
        return False
    if payload.complexity_filter and meta.get("complexity") != payload.complexity_filter:
        return False
    if payload.fine_part_type_filter and (
        label_contract.get("fine_part_type") != payload.fine_part_type_filter
    ):
        return False
    if payload.coarse_part_type_filter and (
        label_contract.get("coarse_part_type") != payload.coarse_part_type_filter
    ):
        return False
    if payload.decision_source_filter and (
        label_contract.get("decision_source") != payload.decision_source_filter
    ):
        return False
    if payload.is_coarse_label_filter is not None and (
        label_contract.get("is_coarse_label") is not payload.is_coarse_label_filter
    ):
        return False
    return True


async def run_similarity_query_pipeline(
    payload: SimilarityQueryPayload,
    *,
    get_qdrant_store: Optional[QdrantStoreGetter] = None,
    error_recorder: Optional[ErrorRecorder] = None,
) -> Dict[str, Any]:
    from src.core.similarity import _cosine, extract_vector_label_contract

    qdrant_store = get_qdrant_store() if get_qdrant_store is not None else None
    if qdrant_store is not None:
        ref_result = await qdrant_store.get_vector(payload.reference_id)
        if ref_result is None:
            _record_error(ErrorCode.DATA_NOT_FOUND, error_recorder)
            return _build_similarity_result_payload(
                reference_id=payload.reference_id,
                target_id=payload.target_id,
                status="reference_not_found",
                error=build_error(
                    ErrorCode.DATA_NOT_FOUND,
                    stage="similarity",
                    message="Reference vector not found",
                    id=payload.reference_id,
                ),
            )
        tgt_result = await qdrant_store.get_vector(payload.target_id)
        if tgt_result is None:
            _record_error(ErrorCode.DATA_NOT_FOUND, error_recorder)
            return _build_similarity_result_payload(
                reference_id=payload.reference_id,
                target_id=payload.target_id,
                status="target_not_found",
                error=build_error(
                    ErrorCode.DATA_NOT_FOUND,
                    stage="similarity",
                    message="Target vector not found",
                    id=payload.target_id,
                ),
            )
        ref = list(ref_result.vector or [])
        tgt = list(tgt_result.vector or [])
        reference_contract = extract_vector_label_contract(ref_result.metadata)
        target_contract = extract_vector_label_contract(tgt_result.metadata)
    else:
        from src.core.similarity import _VECTOR_META, _VECTOR_STORE  # type: ignore

        if payload.reference_id not in _VECTOR_STORE:
            _record_error(ErrorCode.DATA_NOT_FOUND, error_recorder)
            return _build_similarity_result_payload(
                reference_id=payload.reference_id,
                target_id=payload.target_id,
                status="reference_not_found",
                error=build_error(
                    ErrorCode.DATA_NOT_FOUND,
                    stage="similarity",
                    message="Reference vector not found",
                    id=payload.reference_id,
                ),
            )
        if payload.target_id not in _VECTOR_STORE:
            _record_error(ErrorCode.DATA_NOT_FOUND, error_recorder)
            return _build_similarity_result_payload(
                reference_id=payload.reference_id,
                target_id=payload.target_id,
                status="target_not_found",
                error=build_error(
                    ErrorCode.DATA_NOT_FOUND,
                    stage="similarity",
                    message="Target vector not found",
                    id=payload.target_id,
                ),
            )
        ref = _VECTOR_STORE[payload.reference_id]
        tgt = _VECTOR_STORE[payload.target_id]
        reference_contract = extract_vector_label_contract(_VECTOR_META.get(payload.reference_id))
        target_contract = extract_vector_label_contract(_VECTOR_META.get(payload.target_id))

    if len(ref) != len(tgt):
        _record_error(ErrorCode.VALIDATION_FAILED, error_recorder)
        return _build_similarity_result_payload(
            reference_id=payload.reference_id,
            target_id=payload.target_id,
            method="cosine",
            dimension=min(len(ref), len(tgt)),
            status="dimension_mismatch",
            error=build_error(
                ErrorCode.VALIDATION_FAILED,
                stage="similarity",
                message="Vector dimension mismatch",
                expected=len(ref),
                found=len(tgt),
            ),
        )

    score = _cosine(ref, tgt)
    return _build_similarity_result_payload(
        reference_id=payload.reference_id,
        target_id=payload.target_id,
        score=round(score, 4),
        method="cosine",
        dimension=len(ref),
        reference_contract=reference_contract,
        target_contract=target_contract,
    )


async def run_similarity_topk_pipeline(
    payload: SimilarityTopKPayload,
    *,
    get_qdrant_store: Optional[QdrantStoreGetter] = None,
    error_recorder: Optional[ErrorRecorder] = None,
    latency_observer: Optional[LatencyObserver] = None,
) -> Dict[str, Any]:
    from src.core.similarity import InMemoryVectorStore, extract_vector_label_contract

    qdrant_store = get_qdrant_store() if get_qdrant_store is not None else None
    if qdrant_store is not None:
        try:
            target = await qdrant_store.get_vector(payload.target_id)
            if target is None:
                _record_error(ErrorCode.DATA_NOT_FOUND, error_recorder)
                return _build_similarity_topk_error_payload(
                    target_id=payload.target_id,
                    k=payload.k,
                    status="target_not_found",
                    error=create_extended_error(
                        ErrorCode.DATA_NOT_FOUND,
                        "Target vector not found",
                        stage="similarity",
                    ).to_dict(),
                )
            query_vector = target.vector or []
            query_k = max(payload.k + payload.offset + 1, min(payload.k * 5, payload.k + 100))
            raw = await qdrant_store.search_similar(
                query_vector,
                top_k=query_k,
                filter_conditions=_build_vector_filter_conditions(payload) or None,
            )
            items: list[Dict[str, Any]] = []
            matched = 0
            for result in raw:
                if payload.exclude_self and result.id == payload.target_id:
                    continue
                if matched < payload.offset:
                    matched += 1
                    continue
                meta = result.metadata or {}
                label_contract = extract_vector_label_contract(meta)
                items.append(
                    {
                        "id": result.id,
                        "score": round(float(result.score), 4),
                        "material": meta.get("material"),
                        "complexity": meta.get("complexity"),
                        "format": meta.get("format"),
                        "part_type": label_contract.get("part_type"),
                        "fine_part_type": label_contract.get("fine_part_type"),
                        "coarse_part_type": label_contract.get("coarse_part_type"),
                        "decision_source": label_contract.get("decision_source"),
                        "is_coarse_label": label_contract.get("is_coarse_label"),
                    }
                )
                if len(items) >= payload.k:
                    break
            return {
                "target_id": payload.target_id,
                "k": payload.k,
                "results": items,
            }
        except Exception:
            pass

    backend = os.getenv("VECTOR_STORE_BACKEND", "memory")
    store = InMemoryVectorStore()
    if not store.exists(payload.target_id):
        _record_error(ErrorCode.DATA_NOT_FOUND, error_recorder)
        return _build_similarity_topk_error_payload(
            target_id=payload.target_id,
            k=payload.k,
            status="target_not_found",
            error=create_extended_error(
                ErrorCode.DATA_NOT_FOUND,
                "Target vector not found",
                stage="similarity",
            ).to_dict(),
        )

    base_vec = store.get(payload.target_id)
    assert base_vec is not None
    t0 = _time.time()
    if backend == "faiss":
        fstore = FaissVectorStore()
        raw = fstore.query(base_vec, top_k=max(1, payload.k + payload.offset))
        if not raw:
            store = InMemoryVectorStore()
            raw = store.query(base_vec, top_k=max(1, payload.k + payload.offset))
            backend = "memory_fallback"
    else:
        raw = store.query(base_vec, top_k=max(1, payload.k + payload.offset))
    if latency_observer is not None:
        latency_observer(backend, _time.time() - t0)

    items: list[Dict[str, Any]] = []
    sliced = raw[payload.offset : payload.offset + payload.k]
    meta_store = InMemoryVectorStore()
    for vector_id, score in sliced:
        if payload.exclude_self and vector_id == payload.target_id:
            continue
        meta = meta_store.meta(vector_id) or {}
        label_contract = extract_vector_label_contract(meta)
        if not matches_similarity_topk_filters(payload, meta, label_contract):
            continue
        items.append(
            {
                "id": vector_id,
                "score": round(score, 4),
                "material": meta.get("material"),
                "complexity": meta.get("complexity"),
                "format": meta.get("format"),
                "part_type": label_contract.get("part_type"),
                "fine_part_type": label_contract.get("fine_part_type"),
                "coarse_part_type": label_contract.get("coarse_part_type"),
                "decision_source": label_contract.get("decision_source"),
                "is_coarse_label": label_contract.get("is_coarse_label"),
            }
        )
    return {
        "target_id": payload.target_id,
        "k": payload.k,
        "results": items,
    }


__all__ = [
    "matches_similarity_topk_filters",
    "run_similarity_query_pipeline",
    "run_similarity_topk_pipeline",
]
