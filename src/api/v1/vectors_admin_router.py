"""Admin endpoints for vector backend operations."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel

from src.api.dependencies import get_admin_token, get_api_key

router = APIRouter()


async def _vector_reload_admin_token(
    x_admin_token: str = Header(default="", alias="X-Admin-Token"),
) -> str:
    """Admin token dependency that records auth failures for reload metrics."""
    from src.utils.analysis_metrics import vector_store_reload_total

    try:
        return await get_admin_token(x_admin_token)
    except HTTPException:
        try:
            vector_store_reload_total.labels(status="error", reason="auth_failed").inc()
        except Exception:
            pass
        raise


class VectorBackendReloadResponse(BaseModel):
    status: str
    backend: Optional[str] = None
    error: Optional[Dict[str, Any]] = None


@router.post("/backend/reload", response_model=VectorBackendReloadResponse)
async def reload_vector_backend(
    backend: Optional[str] = None,
    api_key: str = Depends(get_api_key),
    admin_token: str = Depends(_vector_reload_admin_token),
):
    """Reload vector store backend (admin token required)."""
    from src.api.v1 import vectors as vectors_module
    from src.core.similarity import reload_vector_store_backend
    from src.utils.analysis_metrics import vector_store_reload_total

    return await vectors_module.run_vector_backend_reload_pipeline(
        backend=backend,
        reload_backend_fn=reload_vector_store_backend,
        reload_metric=vector_store_reload_total,
        error_code_cls=vectors_module.ErrorCode,
        build_error_fn=vectors_module.build_error,
        response_cls=vectors_module.VectorBackendReloadResponse,
    )


__all__ = [
    "VectorBackendReloadResponse",
    "_vector_reload_admin_token",
    "reload_vector_backend",
    "router",
]
