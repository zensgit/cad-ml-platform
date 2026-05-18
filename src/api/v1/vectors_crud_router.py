from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_api_key
from src.api.v1.vector_crud_models import (
    VectorDeleteRequest,
    VectorDeleteResponse,
    VectorRegisterRequest,
    VectorRegisterResponse,
    VectorSearchRequest,
    VectorSearchResponse,
)

router = APIRouter()


@router.post("/delete", response_model=VectorDeleteResponse)
async def delete_vector(payload: VectorDeleteRequest, api_key: str = Depends(get_api_key)):
    from src.api.v1 import vectors as vectors_module

    return await vectors_module.run_vector_delete_pipeline(
        payload=payload,
        response_cls=VectorDeleteResponse,
        error_code_cls=vectors_module.ErrorCode,
        build_error_fn=vectors_module.build_error,
        get_qdrant_store_fn=vectors_module._get_qdrant_store_or_none,
        get_client_fn=vectors_module.get_client,
    )


@router.post("/register", response_model=VectorRegisterResponse)
async def register_vector_endpoint(
    payload: VectorRegisterRequest,
    api_key: str = Depends(get_api_key),
):
    from src.api.v1 import vectors as vectors_module

    return await vectors_module.run_vector_register_pipeline(
        payload=payload,
        response_cls=VectorRegisterResponse,
        error_code_cls=vectors_module.ErrorCode,
        build_error_fn=vectors_module.build_error,
        get_qdrant_store_fn=vectors_module._get_qdrant_store_or_none,
    )


@router.post("/search", response_model=VectorSearchResponse)
async def search_vectors(
    payload: VectorSearchRequest,
    api_key: str = Depends(get_api_key),
):
    from src.api.v1 import vectors as vectors_module

    return await vectors_module.run_vector_search_pipeline(
        payload=payload,
        response_cls=VectorSearchResponse,
        get_qdrant_store_fn=vectors_module._get_qdrant_store_or_none,
        build_filter_conditions_fn=vectors_module._build_vector_search_filter_conditions,
        matches_filters_fn=vectors_module._matches_vector_search_filters,
        vector_item_payload_fn=vectors_module._vector_item_payload,
    )


__all__ = ["router"]
