from __future__ import annotations

import io
import json
import os

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_analyze_route_delegates_to_live_pipeline(monkeypatch):
    captured = {}

    async def _stub_run_analysis_live_pipeline(**kwargs):  # noqa: ANN003, ANN201
        captured["kwargs"] = kwargs
        return {
            "id": kwargs["analysis_id"],
            "file_name": kwargs["file_name"],
            "file_format": "dxf",
            "timestamp": "2026-04-20T00:00:00Z",
            "results": {
                "classification": {"part_type": "support"},
                "statistics": {"entity_count": 1},
            },
            "processing_time": 0.01,
            "cache_hit": False,
            "cad_document": None,
            "feature_version": "v-test",
        }

    monkeypatch.setattr(
        "src.api.v1.analyze.run_analysis_live_pipeline",
        _stub_run_analysis_live_pipeline,
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
        files={"file": ("Delegated.dxf", io.BytesIO(dxf_payload), "application/dxf")},
        data={"options": json.dumps(options), "material": "steel", "project_id": "proj-1"},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["feature_version"] == "v-test"
    assert body["file_name"] == "Delegated.dxf"
    assert captured["kwargs"]["file_name"] == "Delegated.dxf"
    assert captured["kwargs"]["material"] == "steel"
    assert captured["kwargs"]["project_id"] == "proj-1"
