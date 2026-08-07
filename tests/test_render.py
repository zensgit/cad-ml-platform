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
    monkeypatch.setenv("DWG_CONVERTER", "auto")

    from importlib import reload
    from src.api.v1 import render

    reload(render)
    app = FastAPI()
    app.include_router(render.router, prefix="/api/v1/render")
    return TestClient(app, headers={"X-API-Key": "test"})


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


def test_disabled_dwg_never_reaches_local_or_fallback_renderer(
    render_client, monkeypatch
):
    monkeypatch.setenv("DWG_CONVERTER", "disabled")
    with patch(
        "src.api.v1.render._render_cad_to_png",
        side_effect=AssertionError("disabled DWG must not render locally"),
    ) as local_render, patch(
        "src.api.v1.render._render_via_fallback",
        new=AsyncMock(side_effect=AssertionError("disabled DWG must not use fallback")),
    ) as fallback:
        files = {"file": ("sample.dwg", b"dwgdata", "application/acad")}
        resp = render_client.post("/api/v1/render/cad", files=files)

    assert resp.status_code == 422
    assert resp.json()["detail"] == "DWG conversion disabled by DWG_CONVERTER"
    local_render.assert_not_called()
    fallback.assert_not_awaited()


@pytest.mark.parametrize(
    ("file_name", "content_type", "payload"),
    [
        ("sample.bin", "application/dwg", b"opaque"),
        ("sample.bin", "application/octet-stream", b"AC1032payload"),
    ],
)
def test_disabled_disguised_dwg_never_reaches_fallback(
    render_client, monkeypatch, file_name, content_type, payload
):
    monkeypatch.setenv("DWG_CONVERTER", "disabled")
    with patch(
        "src.api.v1.render._render_via_fallback",
        new=AsyncMock(side_effect=AssertionError("disabled DWG must not use fallback")),
    ) as fallback:
        resp = render_client.post(
            "/api/v1/render/cad",
            files={"file": (file_name, payload, content_type)},
        )

    assert resp.status_code == 422
    assert resp.json()["detail"] == "DWG conversion disabled by DWG_CONVERTER"
    fallback.assert_not_awaited()


def test_disabled_mode_keeps_dxf_fallback_available(render_client, monkeypatch):
    monkeypatch.setenv("DWG_CONVERTER", "disabled")
    with patch(
        "src.api.v1.render._render_cad_to_png",
        side_effect=RuntimeError("bad dxf"),
    ), patch(
        "src.api.v1.render._render_via_fallback",
        new=AsyncMock(return_value=b"fallback-png"),
    ) as fallback:
        files = {"file": ("sample.dxf", b"dxfdata", "application/dxf")}
        resp = render_client.post("/api/v1/render/cad", files=files)

    assert resp.status_code == 200
    fallback.assert_awaited_once()


def test_resolve_bearer_helpers():
    from src.api.v1.render import _resolve_bearer

    assert _resolve_bearer("") == ""
    assert _resolve_bearer("token") == "Bearer token"
    assert _resolve_bearer("Bearer token") == "Bearer token"
    assert _resolve_bearer("bearer token") == "bearer token"
