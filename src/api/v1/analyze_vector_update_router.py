from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_api_key
from src.api.v1.analyze_aux_models import (
    VectorUpdateRequest,
    VectorUpdateResponse,
)
from src.core.vector_update_pipeline import run_vector_update_pipeline

router = APIRouter()


@router.post("/vectors/update", response_model=VectorUpdateResponse)
async def update_vector(
    payload: VectorUpdateRequest, api_key: str = Depends(get_api_key)
):
    result = await run_vector_update_pipeline(payload=payload)
    return VectorUpdateResponse(**result)


__all__ = ["router"]
