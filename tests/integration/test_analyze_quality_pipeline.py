from __future__ import annotations

import io
import json
import os

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_analyze_dxf_quality_check_uses_shared_quality_pipeline(monkeypatch):
    captured: dict[str, object] = {}

    async def _stub_run_quality_pipeline(**kwargs):  # noqa: ANN003, ANN201
        captured.update(kwargs)
        return {
            "mode": "L4_DFM",
            "score": 86.0,
            "issues": [],
            "manufacturability": "high",
        }

    monkeypatch.setattr("src.api.v1.analyze.run_quality_pipeline", _stub_run_quality_pipeline)

    dxf_payload = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"
    options = {
        "extract_features": True,
        "classify_parts": False,
        "quality_check": True,
        "process_recommendation": False,
    }
    resp = client.post(
        "/api/v1/analyze/",
        files={
            "file": ("QualityProbe.dxf", io.BytesIO(dxf_payload), "application/dxf"),
        },
        data={"options": json.dumps(options)},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("results", {}).get("quality") == {
        "mode": "L4_DFM",
        "score": 86.0,
        "issues": [],
        "manufacturability": "high",
    }
    assert callable(captured["classification_payload_getter"])
    assert captured["classification_payload_getter"]() == {}
    assert callable(captured["dfm_latency_observer"])
