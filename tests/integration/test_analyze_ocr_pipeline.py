from __future__ import annotations

import io
import json
import os

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_analyze_dxf_uses_shared_ocr_pipeline(monkeypatch):
    captured: dict[str, object] = {}

    async def _stub_run_analysis_ocr_pipeline(**kwargs):  # noqa: ANN003, ANN201
        captured.update(kwargs)
        return {"status": "no_preview_image"}

    monkeypatch.setattr(
        "src.api.v1.analyze.run_analysis_ocr_pipeline",
        _stub_run_analysis_ocr_pipeline,
    )

    dxf_payload = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"
    options = {
        "extract_features": False,
        "classify_parts": False,
        "calculate_similarity": False,
        "quality_check": False,
        "process_recommendation": False,
        "enable_ocr": True,
        "ocr_provider": "deepseek_hf",
    }
    resp = client.post(
        "/api/v1/analyze/",
        files={
            "file": ("OcrProbe.dxf", io.BytesIO(dxf_payload), "application/dxf"),
        },
        data={"options": json.dumps(options)},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["results"]["ocr"] == {"status": "no_preview_image"}
    assert captured["enable_ocr"] is True
    assert captured["ocr_provider_strategy"] == "deepseek_hf"
    assert isinstance(captured["unified_data"], dict)
