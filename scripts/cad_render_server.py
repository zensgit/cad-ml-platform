"""Standalone CAD render service for dev usage."""
from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response

from src.core.dedupcad_2d_worker import _render_cad_to_png

from prometheus_client import Counter, Histogram, make_asgi_app
import os

app = FastAPI(title="CAD Render Service", version="0.1.0")
app.mount("/metrics", make_asgi_app())

auth_token = os.getenv("CAD_RENDER_AUTH_TOKEN", "").strip()

render_requests_total = Counter(
    "cad_render_requests_total",
    "Total CAD render requests",
    ["status"],
)
render_duration_seconds = Histogram(
    "cad_render_duration_seconds",
    "CAD render duration in seconds",
    buckets=(0.5, 1, 2, 5, 10, 20, 30),
)
render_input_bytes = Histogram(
    "cad_render_input_bytes",
    "CAD render input size in bytes",
    buckets=(10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000),
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/render/cad")
async def render_cad(request: Request, file: UploadFile = File(...)) -> Response:
    """Render a CAD file (DWG/DXF) to PNG."""
    if auth_token:
        auth_header = request.headers.get("Authorization", "")
        expected = f"Bearer {auth_token}"
        if auth_header != expected:
            render_requests_total.labels(status="unauthorized").inc()
            raise HTTPException(status_code=401, detail="Unauthorized")
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    render_input_bytes.observe(len(file_bytes))
    try:
        with render_duration_seconds.time():
            _, png_bytes, _ = _render_cad_to_png(
                file_name=file.filename or "drawing.dwg",
                file_bytes=file_bytes,
                content_type=file.content_type or "application/octet-stream",
            )
        render_requests_total.labels(status="ok").inc()
    except Exception as exc:
        render_requests_total.labels(status="error").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return Response(content=png_bytes, media_type="image/png")
