"""Rendering endpoints for CAD previews."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from src.core.dedupcad_2d_worker import _render_cad_to_png

router = APIRouter()


@router.post("/cad")
async def render_cad_preview(file: UploadFile = File(...)) -> Response:
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
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return Response(content=png_bytes, media_type="image/png")
