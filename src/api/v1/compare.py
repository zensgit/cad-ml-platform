"""Compatibility endpoint for feature vector comparison."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_api_key
from src.core.errors_extended import ErrorCode, build_error
from src.core.qdrant_store_helper import (
    get_qdrant_store_or_none as _get_qdrant_store_or_none,
)
from src.core.similarity import (
    InMemoryVectorStore,
    _VECTOR_META,
    _cosine,
    extract_vector_label_contract,
)
from src.utils.analysis_metrics import compare_requests_total

router = APIRouter()
class CompareRequest(BaseModel):
    query_features: list[float] = Field(description="Query feature vector")
    candidate_hash: str = Field(description="Candidate vector id")


class CompareResponse(BaseModel):
    similarity: float
    score: float
    feature_distance: float
    category_match: bool = False
    ocr_match: float = 0.0
    method: str
    dimension: int
    reference_id: str
    reference_part_type: str | None = None
    reference_fine_part_type: str | None = None
    reference_coarse_part_type: str | None = None
    reference_decision_source: str | None = None
    reference_is_coarse_label: bool | None = None


@router.post("", response_model=CompareResponse)
async def compare_features(
    payload: CompareRequest,
    api_key: str = Depends(get_api_key),
) -> CompareResponse:
    """Compare query features against a stored candidate vector."""
    if not payload.query_features:
        compare_requests_total.labels(status="invalid").inc()
        err = build_error(
            ErrorCode.INPUT_VALIDATION_FAILED,
            stage="compare",
            message="query_features is empty",
        )
        raise HTTPException(status_code=400, detail=err)
    candidate_id = payload.candidate_hash.strip()
    if not candidate_id:
        compare_requests_total.labels(status="invalid").inc()
        err = build_error(
            ErrorCode.INPUT_VALIDATION_FAILED,
            stage="compare",
            message="candidate_hash is required",
        )
        raise HTTPException(status_code=400, detail=err)

    qdrant_store = _get_qdrant_store_or_none()
    reference = None
    meta = {}
    if qdrant_store is not None:
        result = await qdrant_store.get_vector(candidate_id)
        if result is not None:
            reference = list(result.vector or [])
            meta = dict(result.metadata or {})
    else:
        store = InMemoryVectorStore()
        reference = store.get(candidate_id)
        if reference is not None:
            meta = _VECTOR_META.get(candidate_id, {})
    if reference is None:
        compare_requests_total.labels(status="not_found").inc()
        err = build_error(
            ErrorCode.DATA_NOT_FOUND,
            stage="compare",
            message="Reference vector not found",
            id=candidate_id,
        )
        raise HTTPException(status_code=404, detail=err)
    if len(reference) != len(payload.query_features):
        compare_requests_total.labels(status="dimension_mismatch").inc()
        err = build_error(
            ErrorCode.DIMENSION_MISMATCH,
            stage="compare",
            message="Vector dimension mismatch",
            expected=len(reference),
            found=len(payload.query_features),
        )
        raise HTTPException(status_code=400, detail=err)

    score = _cosine(reference, payload.query_features)
    similarity = round(score, 4)
    compare_requests_total.labels(status="success").inc()
    label_contract = extract_vector_label_contract(meta)
    return CompareResponse(
        similarity=similarity,
        score=similarity,
        feature_distance=round(1.0 - score, 4),
        category_match=False,
        ocr_match=0.0,
        method="cosine",
        dimension=len(reference),
        reference_id=candidate_id,
        reference_part_type=label_contract.get("part_type"),
        reference_fine_part_type=label_contract.get("fine_part_type"),
        reference_coarse_part_type=label_contract.get("coarse_part_type"),
        reference_decision_source=label_contract.get("decision_source"),
        reference_is_coarse_label=label_contract.get("is_coarse_label"),
    )
