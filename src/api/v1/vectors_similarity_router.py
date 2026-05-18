from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_api_key
from src.api.v1.vector_similarity_models import (
    BatchSimilarityItem,
    BatchSimilarityRequest,
    BatchSimilarityResponse,
)

router = APIRouter()


@router.post("/similarity/batch", response_model=BatchSimilarityResponse)
async def batch_similarity(payload: BatchSimilarityRequest, api_key: str = Depends(get_api_key)):
    """批量相似度查询 - 支持多个向量ID并行查询相似向量"""
    from src.api.v1 import vectors as vectors_module

    return await vectors_module.run_vector_batch_similarity(
        payload=payload,
        batch_item_cls=BatchSimilarityItem,
        batch_response_cls=BatchSimilarityResponse,
        error_code_cls=vectors_module.ErrorCode,
        build_error_fn=vectors_module.build_error,
        get_qdrant_store_fn=vectors_module._get_qdrant_store_or_none,
        build_filter_conditions_fn=vectors_module._build_vector_filter_conditions,
    )


__all__ = ["batch_similarity", "router"]

