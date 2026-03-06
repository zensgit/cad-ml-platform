from __future__ import annotations

import io
import json
import os

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_analyze_dxf_exposes_hybrid_explanation_and_fusion_metadata(monkeypatch):
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
                    "label": "人孔",
                    "confidence": 0.91,
                    "source": "fusion",
                    "filename_prediction": {"label": "人孔", "confidence": 0.86},
                    "titleblock_prediction": {"label": "人孔", "confidence": 0.81},
                    "history_prediction": {
                        "label": "人孔",
                        "confidence": 0.78,
                        "status": "ok",
                        "file_path": history_file_path,
                    },
                    "process_prediction": None,
                    "fusion_weights": {
                        "filename": 0.7,
                        "graph2d": 0.3,
                        "titleblock": 0.2,
                        "process": 0.15,
                        "history_sequence": 0.2,
                    },
                    "source_contributions": {
                        "filename": 0.602,
                        "titleblock": 0.162,
                        "history_sequence": 0.156,
                    },
                    "fusion_metadata": {
                        "strategy": "weighted_average",
                        "agreement_score": 1.0,
                        "num_sources": 3,
                    },
                    "decision_path": [
                        "filename_extracted",
                        "titleblock_auto_enabled",
                        "titleblock_predicted",
                        "history_auto_enabled",
                        "history_predicted",
                        "fusion_scored",
                        "fusion_engine_weighted_average",
                    ],
                    "explanation": {
                        "summary": "综合 文件名, 标题栏, 历史序列 多源信息，融合得出 人孔",
                        "source_contributions": {
                            "文件名": 0.602,
                            "标题栏": 0.162,
                            "历史序列": 0.156,
                        },
                    },
                }
            )

    monkeypatch.setenv("PROVIDER_REGISTRY_CACHE_ENABLED", "false")
    monkeypatch.setenv("GRAPH2D_ENABLED", "false")
    monkeypatch.setenv("HYBRID_CLASSIFIER_ENABLED", "true")
    monkeypatch.setattr(
        "src.core.providers.classifier.HybridClassifierProviderAdapter._build_default_classifier",
        lambda self: _StubHybridClassifier(),
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
    data = resp.json()
    classification = data.get("results", {}).get("classification", {})

    assert classification.get("fine_part_type") == "人孔"
    assert classification.get("history_prediction", {}).get("label") == "人孔"
    assert classification.get("source_contributions", {}).get("filename") == 0.602
    assert classification.get("fusion_metadata", {}).get("strategy") == (
        "weighted_average"
    )
    assert "fusion_engine_weighted_average" in (
        classification.get("decision_path") or []
    )
    assert (classification.get("hybrid_explanation") or {}).get("summary")
