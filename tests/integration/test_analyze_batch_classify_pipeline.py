from __future__ import annotations

import io

from fastapi.testclient import TestClient

from src.main import app


def test_batch_classify_route_delegates_to_shared_pipeline(monkeypatch) -> None:
    client = TestClient(app)
    captured = {}

    async def _stub_run_batch_classify_pipeline(**kwargs):  # noqa: ANN003, ANN201
        captured.update(kwargs)
        return {
            "total": 1,
            "success": 1,
            "failed": 0,
            "processing_time": 0.123,
            "results": [
                {
                    "file_name": "part.dxf",
                    "category": "轴类",
                    "fine_category": "轴类",
                    "coarse_category": "轴类",
                    "is_coarse_label": True,
                    "fine_part_type": "轴类",
                    "coarse_part_type": "轴类",
                    "decision_source": "v16",
                    "contract_version": "classification_decision.v1",
                    "decision_contract": {
                        "fine_part_type": "轴类",
                        "coarse_part_type": "轴类",
                        "confidence": 0.95,
                        "decision_source": "v16",
                        "branch_conflicts": {},
                        "evidence": [],
                        "review_reasons": [],
                        "fallback_flags": [],
                        "contract_version": "classification_decision.v1",
                    },
                    "confidence": 0.95,
                }
            ],
        }

    monkeypatch.setattr(
        "src.api.v1.analyze_batch_router.run_batch_classify_pipeline",
        _stub_run_batch_classify_pipeline,
    )

    response = client.post(
        "/api/v1/analyze/batch-classify",
        files=[
            ("files", ("part.dxf", io.BytesIO(b"0\nEOF\n"), "application/octet-stream"))
        ],
        data={"max_workers": "3"},
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] == 1
    assert payload["results"][0]["coarse_category"] == "轴类"
    assert payload["results"][0]["contract_version"] == "classification_decision.v1"
    assert captured["max_workers"] == 3
    assert callable(captured["logger"].warning)
    assert len(captured["files"]) == 1
