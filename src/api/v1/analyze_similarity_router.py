from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_api_key
from src.api.v1.analyze_live_models import (
    SimilarityQuery,
    SimilarityResult,
    SimilarityTopKQuery,
    SimilarityTopKResponse,
)
from src.core.qdrant_store_helper import (
    get_qdrant_store_or_none as _get_qdrant_store_or_none,
)
from src.core.vector_query_pipeline import (
    run_similarity_query_pipeline,
    run_similarity_topk_pipeline,
)
from src.utils.analysis_metrics import (
    analysis_error_code_total,
    vector_query_latency_seconds,
)

router = APIRouter()


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


__all__ = [
    "router",
    "_get_qdrant_store_or_none",
    "run_similarity_query_pipeline",
    "run_similarity_topk_pipeline",
]
