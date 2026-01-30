"""Rendering endpoints for CAD previews."""
from __future__ import annotations

import logging
import os

import httpx
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response

from src.core.dedupcad_2d_worker import _render_cad_to_png

router = APIRouter()
logger = logging.getLogger(__name__)


def _resolve_bearer(token: str) -> str:
    token = token.strip()
    if not token:
        return ""
    if token.lower().startswith("bearer "):
        return token
    return f"Bearer {token}"


async def _render_via_fallback(
    *,
    request: Request,
    file_name: str,
    file_bytes: bytes,
    content_type: str,
) -> Optional[bytes]:
    base_url = os.getenv("CAD_RENDER_FALLBACK_URL", "").strip()
    if not base_url:
        return None
    headers: dict[str, str] = {}
    token = os.getenv("CAD_RENDER_FALLBACK_TOKEN", "").strip()
    if token:
        headers["Authorization"] = _resolve_bearer(token)
    else:
        auth_header = request.headers.get("Authorization", "").strip()
        if auth_header:
            headers["Authorization"] = auth_header
    url = f"{base_url.rstrip('/')}/api/v1/render/cad"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                url,
                files={"file": (file_name, file_bytes, content_type)},
                headers=headers,
            )
            resp.raise_for_status()
            return resp.content
    except Exception as exc:
        logger.warning("CAD render fallback failed: %s", exc)
        return None


@router.post("/cad")
async def render_cad_preview(request: Request, file: UploadFile = File(...)) -> Response:
    """Render a CAD file (DWG/DXF) to PNG.

    Returns raw PNG bytes for downstream preview services.
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    file_name = file.filename or "drawing.dwg"
    content_type = file.content_type or "application/octet-stream"

    try:
        _, png_bytes, _ = _render_cad_to_png(
            file_name=file_name,
            file_bytes=file_bytes,
            content_type=content_type,
        )
        return Response(content=png_bytes, media_type="image/png")
    except Exception as exc:
        fallback_bytes = await _render_via_fallback(
            request=request,
            file_name=file_name,
            file_bytes=file_bytes,
            content_type=content_type,
        )
        if fallback_bytes:
            logger.info("CAD render fallback used")
            return Response(content=fallback_bytes, media_type="image/png")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
