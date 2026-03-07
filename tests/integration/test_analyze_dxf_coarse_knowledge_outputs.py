from __future__ import annotations

import io
import json
import os

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_analyze_dxf_exposes_coarse_labels_and_knowledge_outputs(monkeypatch):
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
                        "used_for_fusion": False,
                    },
                    "fusion_metadata": {
                        "strategy": "weighted_average",
                        "agreement_score": 0.5,
                        "num_sources": 3,
                    },
                    "decision_path": [
                        "filename_extracted",
                        "history_predicted",
                        "fusion_engine_weighted_average",
                    ],
                    "source_contributions": {
                        "filename": 0.602,
                        "history_sequence": 0.156,
                    },
                    "explanation": {
                        "summary": "综合 文件名, 历史序列 多源信息，融合得出 人孔",
                    },
                }
            )

    class _StubGraph2D:
        def predict_from_bytes(self, data, file_name):  # noqa: ANN001
            return {"label": "传动件", "confidence": 0.82, "status": "ok"}

    monkeypatch.setenv("PROVIDER_REGISTRY_CACHE_ENABLED", "false")
    monkeypatch.setenv("GRAPH2D_ENABLED", "true")
    monkeypatch.setenv("HYBRID_CLASSIFIER_ENABLED", "true")
    monkeypatch.setenv("ACTIVE_LEARNING_ENABLED", "false")
    monkeypatch.setattr(
        "src.core.providers.classifier.HybridClassifierProviderAdapter._build_default_classifier",
        lambda self: _StubHybridClassifier(),
    )
    monkeypatch.setattr(
        "src.core.providers.classifier.Graph2DClassifierProviderAdapter._build_default_classifier",
        lambda self: _StubGraph2D(),
    )

    dxf_payload = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"
    options = {"extract_features": True, "classify_parts": True}
    resp = client.post(
        "/api/v1/analyze/",
        files={
            "file": (
                "M10x1.25_ISO2768-mK_GBT1804-M_IT7_位置度0.2_A_C_B_"
                "材料304_Ra3.2_N8_人孔.dxf",
                io.BytesIO(dxf_payload),
                "application/dxf",
            ),
        },
        data={"options": json.dumps(options)},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )

    assert resp.status_code == 200, resp.text
    classification = resp.json().get("results", {}).get("classification", {})

    assert classification.get("coarse_hybrid_label") == "开孔件"
    assert classification.get("coarse_filename_label") == "开孔件"
    assert classification.get("coarse_graph2d_label") == "传动件"
    assert classification.get("has_branch_conflict") is True
    assert classification.get("needs_review") is True
    assert classification.get("confidence_band") == "high"
    assert classification.get("review_priority") == "critical"
    assert "knowledge_conflict" in (classification.get("review_reasons") or [])
    assert "branch_conflict" in (classification.get("review_reasons") or [])

    knowledge_checks = classification.get("knowledge_checks") or []
    categories = {item.get("category") for item in knowledge_checks}
    assert "thread_standard" in categories
    assert "general_tolerance" in categories
    assert "it_grade" in categories
    assert "gdt" in categories
    assert "material" in categories
    assert "surface_finish" in categories

    standards = classification.get("standards_candidates") or []
    assert any(item.get("designation") == "GB/T 1804-M" for item in standards)
    assert any(
        item.get("type") == "material" and item.get("grade") == "S30408"
        for item in standards
    )
    assert any(
        item.get("type") == "surface_finish" and item.get("grade") == "N8"
        for item in standards
    )
    assert classification.get("violations")
