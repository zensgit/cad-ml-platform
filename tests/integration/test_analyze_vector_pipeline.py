from __future__ import annotations

import io
import json
import os

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_analyze_dxf_uses_shared_vector_pipeline(monkeypatch):
    captured: dict[str, object] = {}

    async def _stub_run_vector_pipeline(**kwargs):  # noqa: ANN003, ANN201
        captured.update(kwargs)
        return {
            "registered": True,
            "similarity": {
                "reference_id": "ref-123",
                "score": 0.88,
            },
            "vector_metadata": {
                "material": "steel",
                "part_type": "法兰",
            },
        }

    monkeypatch.setattr("src.api.v1.analyze.run_vector_pipeline", _stub_run_vector_pipeline)

    dxf_payload = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"
    options = {
        "extract_features": True,
        "classify_parts": False,
        "calculate_similarity": True,
        "reference_id": "ref-123",
        "quality_check": False,
        "process_recommendation": False,
    }
    resp = client.post(
        "/api/v1/analyze/",
        files={
            "file": ("VectorProbe.dxf", io.BytesIO(dxf_payload), "application/dxf"),
        },
        data={"options": json.dumps(options)},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("results", {}).get("similarity") == {
        "reference_id": "ref-123",
        "score": 0.88,
    }
    assert captured["calculate_similarity"] is True
    assert captured["reference_id"] == "ref-123"
    assert captured["classification_meta"] == {}
    assert callable(captured["get_qdrant_store"])
    assert callable(captured["compute_qdrant_similarity"])
    assert callable(captured["vector_material_observer"])
    assert callable(captured["feature_dimension_observer"])
