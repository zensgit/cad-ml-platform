from __future__ import annotations

import io
import json
import os

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_analyze_dxf_uses_shared_result_envelope(monkeypatch):
    captured: dict[str, object] = {}

    async def _stub_finalize_analysis_success(**kwargs):  # noqa: ANN003, ANN201
        captured.update(kwargs)
        return {
            "id": kwargs["analysis_id"],
            "timestamp": kwargs["start_time"],
            "file_name": kwargs["file_name"],
            "file_format": kwargs["file_format"].upper(),
            "results": {
                "statistics": {
                    "entity_count": 1,
                    "layer_count": 0,
                    "bounding_box": {
                        "min_x": 0.0,
                        "min_y": 0.0,
                        "min_z": 0.0,
                        "max_x": 0.0,
                        "max_y": 0.0,
                        "max_z": 0.0,
                    },
                    "complexity": "low",
                    "stages": kwargs["stage_times"],
                }
            },
            "processing_time": 0.25,
            "cache_hit": False,
            "cad_document": {
                "file_name": kwargs["file_name"],
                "format": kwargs["file_format"],
                "entity_count": 1,
                "entities": [],
                "layers": {},
                "bounding_box": {
                    "min_x": 0.0,
                    "min_y": 0.0,
                    "min_z": 0.0,
                    "max_x": 0.0,
                    "max_y": 0.0,
                    "max_z": 0.0,
                },
                "complexity": "low",
                "metadata": {},
                "raw_stats": {},
            },
            "feature_version": "v-test",
        }

    monkeypatch.setattr(
        "src.api.v1.analyze.finalize_analysis_success",
        _stub_finalize_analysis_success,
    )

    dxf_payload = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"
    options = {
        "extract_features": False,
        "classify_parts": False,
        "calculate_similarity": False,
        "quality_check": False,
        "process_recommendation": False,
    }
    resp = client.post(
        "/api/v1/analyze/",
        files={
            "file": ("EnvelopeProbe.dxf", io.BytesIO(dxf_payload), "application/dxf"),
        },
        data={"options": json.dumps(options)},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["processing_time"] == 0.25
    assert data["feature_version"] == "v-test"
    assert data["results"]["statistics"]["entity_count"] == 1
    assert captured["analysis_cache_key"].startswith("analysis:EnvelopeProbe.dxf:")
    assert captured["file_name"] == "EnvelopeProbe.dxf"
    assert captured["file_format"] == "dxf"
