"""Compatibility endpoint for feature vector comparison."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_api_key
from src.core.errors_extended import ErrorCode, build_error
from src.core.similarity import InMemoryVectorStore, _cosine

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


@router.post("/", response_model=CompareResponse)
async def compare_features(
    payload: CompareRequest,
    api_key: str = Depends(get_api_key),
):
    """Compare query features against a stored candidate vector."""
    if not payload.query_features:
        err = build_error(
            ErrorCode.INPUT_VALIDATION_FAILED,
            stage="compare",
            message="query_features is empty",
        )
        raise HTTPException(status_code=400, detail=err)
    candidate_id = payload.candidate_hash.strip()
    if not candidate_id:
        err = build_error(
            ErrorCode.INPUT_VALIDATION_FAILED,
            stage="compare",
            message="candidate_hash is required",
        )
        raise HTTPException(status_code=400, detail=err)

    store = InMemoryVectorStore()
    reference = store.get(candidate_id)
    if reference is None:
        err = build_error(
            ErrorCode.DATA_NOT_FOUND,
            stage="compare",
            message="Reference vector not found",
            id=candidate_id,
        )
        raise HTTPException(status_code=404, detail=err)
    if len(reference) != len(payload.query_features):
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
    return CompareResponse(
        similarity=similarity,
        score=similarity,
        feature_distance=round(1.0 - score, 4),
        category_match=False,
        ocr_match=0.0,
        method="cosine",
        dimension=len(reference),
        reference_id=candidate_id,
    )
