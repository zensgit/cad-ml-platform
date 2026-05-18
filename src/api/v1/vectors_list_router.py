from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_api_key
from src.api.v1.vector_list_models import VectorListItem, VectorListResponse

router = APIRouter()


@router.get("/", response_model=VectorListResponse)
async def list_vectors(
    source: str = Query(
        default="auto",
        description="Vector source: auto|memory|redis",
    ),
    offset: int = Query(default=0, ge=0, description="结果偏移用于分页"),
    limit: int = Query(default=50, ge=1, description="返回数量上限"),
    material_filter: Optional[str] = Query(default=None, description="材料过滤"),
    complexity_filter: Optional[str] = Query(default=None, description="复杂度过滤"),
    fine_part_type_filter: Optional[str] = Query(default=None, description="细分类过滤"),
    coarse_part_type_filter: Optional[str] = Query(default=None, description="粗分类过滤"),
    decision_source_filter: Optional[str] = Query(default=None, description="决策来源过滤"),
    is_coarse_label_filter: Optional[bool] = Query(
        default=None,
        description="是否仅返回 coarse label 样本",
    ),
    api_key: str = Depends(get_api_key),
):
    from src.api.v1 import vectors as vectors_module

    return await vectors_module.run_vector_list_pipeline(
        source=source,
        offset=offset,
        limit=limit,
        material_filter=material_filter,
        complexity_filter=complexity_filter,
        fine_part_type_filter=fine_part_type_filter,
        coarse_part_type_filter=coarse_part_type_filter,
        decision_source_filter=decision_source_filter,
        is_coarse_label_filter=is_coarse_label_filter,
        response_cls=VectorListResponse,
        item_cls=VectorListItem,
        error_code_cls=vectors_module.ErrorCode,
        build_error_fn=vectors_module.build_error,
        get_qdrant_store_fn=vectors_module._get_qdrant_store_or_none,
        resolve_list_source_fn=vectors_module._resolve_list_source,
        build_filter_conditions_fn=vectors_module._build_vector_filter_conditions,
        list_vectors_redis_fn=vectors_module._list_vectors_redis,
        list_vectors_memory_fn=vectors_module._list_vectors_memory,
        get_client_fn=vectors_module.get_client,
    )


__all__ = ["list_vectors", "router"]

