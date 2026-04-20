from __future__ import annotations

import io
import json
import os

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_analyze_batch_uses_shared_batch_pipeline(monkeypatch):
    captured = {}

    async def _stub_run_batch_analysis(**kwargs):  # noqa: ANN003, ANN201
        captured.update(kwargs)
        return {
            "total": 2,
            "successful": 1,
            "failed": 1,
            "results": [
                {"file_name": "a.dxf", "ok": True},
                {"file_name": "b.dxf", "error": "boom"},
            ],
        }

    monkeypatch.setattr(
        "src.api.v1.analyze_batch_router.run_batch_analysis",
        _stub_run_batch_analysis,
    )

    response = client.post(
        "/api/v1/analyze/batch",
        files=[
            ("files", ("a.dxf", io.BytesIO(b"0\nEOF\n"), "application/dxf")),
            ("files", ("b.dxf", io.BytesIO(b"0\nEOF\n"), "application/dxf")),
        ],
        data={"options": json.dumps({"extract_features": False})},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 2
    assert payload["successful"] == 1
    assert payload["failed"] == 1
    assert captured["options"] == '{"extract_features": false}'
    assert captured["api_key"] == os.getenv("API_KEY", "test")
    assert captured["analyze_file_fn"].__name__ == "analyze_cad_file"
    assert len(captured["files"]) == 2
