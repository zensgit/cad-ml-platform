import sys
from types import ModuleType
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def render_client(monkeypatch):
    stub_worker = ModuleType("src.core.dedupcad_2d_worker")
    stub_worker._render_cad_to_png = lambda **_: ("stub", b"stub", None)
    monkeypatch.setitem(sys.modules, "src.core.dedupcad_2d_worker", stub_worker)

    from importlib import reload
    from src.api.v1 import render

    reload(render)
    app = FastAPI()
    app.include_router(render.router, prefix="/api/v1/render")
    return TestClient(app)


def test_render_cad_preview_empty_file_returns_400(render_client):
    files = {"file": ("empty.dwg", b"", "application/acad")}
    resp = render_client.post("/api/v1/render/cad", files=files)
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Empty file"


def test_render_cad_preview_success_returns_png(render_client):
    png_bytes = b"\x89PNG\r\n"
    with patch(
        "src.api.v1.render._render_cad_to_png",
        return_value=("mock", png_bytes, None),
    ) as mock_render:
        files = {"file": ("sample.dwg", b"dwgdata", "application/acad")}
        resp = render_client.post("/api/v1/render/cad", files=files)
    assert resp.status_code == 200
    assert resp.content == png_bytes
    assert resp.headers["content-type"] == "image/png"
    mock_render.assert_called_once()


def test_render_cad_preview_render_failure_returns_422(render_client):
    with patch(
        "src.api.v1.render._render_cad_to_png",
        side_effect=RuntimeError("bad cad"),
    ):
        files = {"file": ("bad.dwg", b"dwgdata", "application/acad")}
        resp = render_client.post("/api/v1/render/cad", files=files)
    assert resp.status_code == 422
    assert resp.json()["detail"] == "bad cad"


def test_render_cad_preview_fallback_returns_png(render_client):
    with patch(
        "src.api.v1.render._render_cad_to_png",
        side_effect=RuntimeError("bad cad"),
    ), patch(
        "src.api.v1.render._render_via_fallback",
        new=AsyncMock(return_value=b"fallback-png"),
    ) as mock_fallback:
        files = {"file": ("bad.dwg", b"dwgdata", "application/acad")}
        resp = render_client.post("/api/v1/render/cad", files=files)
    assert resp.status_code == 200
    assert resp.content == b"fallback-png"
    assert resp.headers["content-type"] == "image/png"
    mock_fallback.assert_awaited_once()


def test_resolve_bearer_helpers():
    from src.api.v1.render import _resolve_bearer

    assert _resolve_bearer("") == ""
    assert _resolve_bearer("token") == "Bearer token"
    assert _resolve_bearer("Bearer token") == "Bearer token"
    assert _resolve_bearer("bearer token") == "bearer token"
