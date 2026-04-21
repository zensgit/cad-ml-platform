from __future__ import annotations

import os

from fastapi import APIRouter, Depends

from src.api.dependencies import get_api_key
from src.core.legacy_admin_pipeline import run_faiss_rebuild_pipeline
from src.core.similarity import FaissVectorStore

router = APIRouter()


@router.post("/vectors/faiss/rebuild")
async def faiss_rebuild(api_key: str = Depends(get_api_key)):
    """手动触发 Faiss 索引重建 (延迟删除生效)."""
    return run_faiss_rebuild_pipeline(
        vector_store_backend=os.getenv("VECTOR_STORE_BACKEND", "memory"),
        store_factory=FaissVectorStore,
    )


__all__ = ["router"]
