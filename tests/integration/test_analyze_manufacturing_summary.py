from __future__ import annotations

import io
import json
import os

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_analyze_dxf_uses_shared_manufacturing_summary_builder(monkeypatch):
    captured: dict[str, object] = {}
    quality_payload = {
        "mode": "L4_DFM",
        "score": 92.0,
        "issues": ["draft_angle"],
        "manufacturability": "high",
    }
    process_payload = {
        "primary_recommendation": {
            "process": "turning",
            "method": "cnc_lathe",
        }
    }
    cost_payload = {
        "total_unit_cost": 19.6,
        "currency": "USD",
    }
    summary_payload = {
        "feasibility": "high",
        "risks": ["draft_angle"],
        "process": {
            "process": "turning",
            "method": "cnc_lathe",
        },
        "cost_estimate": cost_payload,
        "cost_range": {"low": 17.64, "high": 21.56},
        "currency": "USD",
    }

    async def _stub_run_quality_pipeline(**kwargs):  # noqa: ANN003, ANN201
        return quality_payload

    async def _stub_run_process_pipeline(**kwargs):  # noqa: ANN003, ANN201
        return {
            "process": process_payload,
            "cost_estimation": cost_payload,
        }

    def _stub_build_manufacturing_decision_summary(**kwargs):  # noqa: ANN003, ANN201
        captured.update(kwargs)
        return summary_payload

    monkeypatch.setattr("src.api.v1.analyze.run_quality_pipeline", _stub_run_quality_pipeline)
    monkeypatch.setattr("src.api.v1.analyze.run_process_pipeline", _stub_run_process_pipeline)
    monkeypatch.setattr(
        "src.api.v1.analyze.build_manufacturing_decision_summary",
        _stub_build_manufacturing_decision_summary,
    )

    dxf_payload = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"
    options = {
        "extract_features": True,
        "classify_parts": False,
        "quality_check": True,
        "process_recommendation": True,
        "estimate_cost": True,
    }
    resp = client.post(
        "/api/v1/analyze/",
        files={
            "file": ("ManufacturingSummaryProbe.dxf", io.BytesIO(dxf_payload), "application/dxf"),
        },
        data={"options": json.dumps(options)},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("results", {}).get("manufacturing_decision") == summary_payload
    assert captured == {
        "quality_payload": quality_payload,
        "process_payload": process_payload,
        "cost_payload": cost_payload,
    }


def test_analyze_dxf_manufacturing_summary_uses_real_builder_wiring(monkeypatch):
    async def _stub_run_quality_pipeline(**kwargs):  # noqa: ANN003, ANN201
        return {
            "manufacturability": "medium",
            "issues": ["thin_wall"],
        }

    async def _stub_run_process_pipeline(**kwargs):  # noqa: ANN003, ANN201
        return {
            "process": {
                "primary_recommendation": {},
                "process": "cnc_milling",
                "method": "standard",
            },
            "cost_estimation": {
                "total_unit_cost": 18.0,
                "currency": "USD",
            },
        }

    monkeypatch.setattr("src.api.v1.analyze.run_quality_pipeline", _stub_run_quality_pipeline)
    monkeypatch.setattr("src.api.v1.analyze.run_process_pipeline", _stub_run_process_pipeline)

    dxf_payload = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"
    options = {
        "extract_features": True,
        "classify_parts": False,
        "quality_check": True,
        "process_recommendation": True,
        "estimate_cost": True,
    }
    resp = client.post(
        "/api/v1/analyze/",
        files={
            "file": ("ManufacturingDecisionRealBuilder.dxf", io.BytesIO(dxf_payload), "application/dxf"),
        },
        data={"options": json.dumps(options)},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json().get("results", {}).get("manufacturing_decision") == {
        "feasibility": "medium",
        "risks": ["thin_wall"],
        "process": {
            "process": "cnc_milling",
            "method": "standard",
        },
        "cost_estimate": {
            "total_unit_cost": 18.0,
            "currency": "USD",
        },
        "cost_range": {
            "low": 16.2,
            "high": 19.8,
        },
        "currency": "USD",
    }
