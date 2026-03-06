from __future__ import annotations

import io
import json
import os

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_analyze_dxf_active_learning_keeps_hybrid_context(monkeypatch):
    captured: dict[str, object] = {}

    class _StubLearner:
        def flag_for_review(self, **kwargs):  # noqa: ANN003, ANN201
            captured.update(kwargs)
            return {"status": "ok"}

    class _StubHybridClassifier:
        class _Result:
            def __init__(self, payload):  # noqa: ANN001
                self._payload = payload

            def to_dict(self):  # noqa: D401
                return dict(self._payload)

        def classify(  # noqa: ANN201
            self,
            filename,
            file_bytes=None,  # noqa: ANN001
            graph2d_result=None,  # noqa: ANN001
            history_result=None,  # noqa: ANN001
            history_file_path=None,  # noqa: ANN001
        ):
            return self._Result(
                {
                    "label": None,
                    "confidence": 0.0,
                    "source": "fallback",
                    "filename_prediction": {"label": "人孔", "confidence": 0.62},
                    "source_contributions": {"filename": 0.434},
                    "fusion_metadata": {
                        "strategy": "weighted_average",
                        "agreement_score": 0.5,
                        "num_sources": 1,
                    },
                    "decision_path": [
                        "filename_extracted",
                        "final_below_reject_min_conf",
                    ],
                    "explanation": {
                        "summary": "文件名支持 人孔，但最终低于拒识阈值",
                    },
                    "rejection": {
                        "reason": "below_min_confidence",
                        "min_confidence": 0.8,
                        "raw_label": "人孔",
                        "raw_confidence": 0.62,
                        "raw_source": "filename",
                    },
                }
            )

    monkeypatch.setenv("PROVIDER_REGISTRY_CACHE_ENABLED", "false")
    monkeypatch.setenv("GRAPH2D_ENABLED", "false")
    monkeypatch.setenv("HYBRID_CLASSIFIER_ENABLED", "true")
    monkeypatch.setenv("ACTIVE_LEARNING_ENABLED", "true")
    monkeypatch.setenv("ACTIVE_LEARNING_CONFIDENCE_THRESHOLD", "0.01")
    monkeypatch.setattr(
        "src.core.providers.classifier.HybridClassifierProviderAdapter._build_default_classifier",
        lambda self: _StubHybridClassifier(),
    )
    monkeypatch.setattr(
        "src.core.active_learning.get_active_learner", lambda: _StubLearner()
    )

    dxf_payload = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"
    options = {"extract_features": True, "classify_parts": True}
    resp = client.post(
        "/api/v1/analyze/",
        files={
            "file": ("sample.dxf", io.BytesIO(dxf_payload), "application/dxf"),
        },
        data={"options": json.dumps(options)},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )

    assert resp.status_code == 200, resp.text
    score_breakdown = captured.get("score_breakdown") or {}
    assert score_breakdown["decision_path"] == [
        "filename_extracted",
        "final_below_reject_min_conf",
    ]
    assert score_breakdown["source_contributions"]["filename"] == 0.434
    assert score_breakdown["fusion_metadata"]["strategy"] == "weighted_average"
    assert score_breakdown["hybrid_explanation"]["summary"].startswith("文件名支持")
