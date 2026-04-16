from __future__ import annotations

import io
import json
import os

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_analyze_dxf_process_check_uses_shared_process_pipeline(monkeypatch):
    captured: dict[str, object] = {}

    async def _stub_run_process_pipeline(**kwargs):  # noqa: ANN003, ANN201
        captured.update(kwargs)
        return {
            "process": {
                "primary_recommendation": {
                    "process": "turning",
                    "method": "cnc_lathe",
                }
            },
            "cost_estimation": {
                "total_unit_cost": 16.8,
                "currency": "USD",
            },
        }

    monkeypatch.setattr("src.api.v1.analyze.run_process_pipeline", _stub_run_process_pipeline)

    dxf_payload = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"
    options = {
        "extract_features": True,
        "classify_parts": False,
        "quality_check": False,
        "process_recommendation": True,
        "estimate_cost": True,
    }
    resp = client.post(
        "/api/v1/analyze/",
        files={
            "file": ("ProcessProbe.dxf", io.BytesIO(dxf_payload), "application/dxf"),
        },
        data={"options": json.dumps(options)},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("results", {}).get("process") == {
        "primary_recommendation": {
            "process": "turning",
            "method": "cnc_lathe",
        }
    }
    assert data.get("results", {}).get("cost_estimation") == {
        "total_unit_cost": 16.8,
        "currency": "USD",
    }
    assert callable(captured["classification_payload_getter"])
    assert captured["classification_payload_getter"]() == {}
    assert callable(captured["process_rule_version_observer"])
    assert callable(captured["cost_latency_observer"])
