from __future__ import annotations

import io
import json
import os

from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_analyze_dxf_uses_shared_http_exception_handler(monkeypatch):
    captured = {}

    async def _stub_run_document_pipeline(**kwargs):  # noqa: ANN003, ANN201
        raise HTTPException(status_code=404, detail="missing-analysis-input")

    def _stub_handle_analysis_http_exception(exc):  # noqa: ANN001, ANN202
        captured["exc"] = exc
        raise HTTPException(status_code=404, detail={"code": "HANDLED_HTTP"})

    monkeypatch.setattr(
        "src.api.v1.analyze.run_document_pipeline",
        _stub_run_document_pipeline,
    )
    monkeypatch.setattr(
        "src.api.v1.analyze.handle_analysis_http_exception",
        _stub_handle_analysis_http_exception,
    )

    dxf_payload = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"
    options = {
        "extract_features": False,
        "classify_parts": False,
        "quality_check": False,
        "process_recommendation": False,
    }
    resp = client.post(
        "/api/v1/analyze/",
        files={"file": ("ErrProbe.dxf", io.BytesIO(dxf_payload), "application/dxf")},
        data={"options": json.dumps(options)},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )

    assert resp.status_code == 404
    assert resp.json()["detail"] == {"code": "HANDLED_HTTP"}
    assert isinstance(captured["exc"], HTTPException)
    assert captured["exc"].detail == "missing-analysis-input"
