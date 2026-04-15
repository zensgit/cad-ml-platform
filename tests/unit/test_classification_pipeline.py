from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.core.classification.classification_pipeline import run_classification_pipeline


@pytest.mark.asyncio
async def test_run_classification_pipeline_happy_path(monkeypatch):
    calls: list[str] = []
    captured: dict[str, object] = {}

    async def _baseline(**kwargs):
        calls.append("baseline")
        return {
            "payload": {"part_type": "simple_plate", "confidence": 0.2, "rule_version": "v1"},
            "text_signals": "sig",
            "entity_counts": {"LINE": 2},
            "doc_metadata": {"fmt": "dxf"},
            "l2_features": {"aspect_ratio": 2.0},
            "l3_features": {"faces": 4},
        }

    async def _shadow(payload, **kwargs):
        calls.append("shadow")
        assert payload["part_type"] == "simple_plate"
        return {
            "payload": {**payload, "fine_part_type": "人孔"},
            "ml_result": {"predicted_type": "法兰", "confidence": 0.61},
            "graph2d_fusable": {"label": "人孔", "confidence": 0.91},
            "hybrid_result": {"label": "人孔", "confidence": 0.95},
        }

    def _fusion(payload, **kwargs):
        calls.append("fusion")
        return {"payload": {**payload, "fusion_decision": {"schema_version": "v1.2"}}}

    def _hybrid(payload, **kwargs):
        calls.append("hybrid")
        return {"payload": {**payload, "part_type": "人孔", "confidence_source": "hybrid"}}

    def _finalize(payload, **kwargs):
        calls.append("finalize")
        captured["finalize_kwargs"] = kwargs
        return {**payload, "needs_review": True, "coarse_part_type": "壳体类"}

    def _flag(**kwargs):
        calls.append("flag")
        captured["flag_kwargs"] = kwargs
        return {"status": "queued"}

    monkeypatch.setenv("ANALYSIS_REVIEW_LOW_CONFIDENCE_THRESHOLD", "0.55")
    monkeypatch.setenv("ANALYSIS_REVIEW_HIGH_CONFIDENCE_THRESHOLD", "0.91")

    with (
        patch(
            "src.core.classification.classification_pipeline.build_baseline_classification_context",
            _baseline,
        ),
        patch(
            "src.core.classification.classification_pipeline.build_shadow_classification_context",
            _shadow,
        ),
        patch(
            "src.core.classification.classification_pipeline.build_fusion_classification_context",
            _fusion,
        ),
        patch(
            "src.core.classification.classification_pipeline.build_hybrid_override_context",
            _hybrid,
        ),
        patch(
            "src.core.classification.classification_pipeline.finalize_classification_payload",
            _finalize,
        ),
        patch(
            "src.core.classification.classification_pipeline.flag_classification_for_review",
            _flag,
        ),
    ):
        result = await run_classification_pipeline(
            analysis_id="analysis-1",
            doc=SimpleNamespace(metadata={"text_content": ["A1"]}),
            features={"geometric": [], "semantic": []},
            features_3d={"faces": 4},
            file_name="sample.dxf",
            file_format="dxf",
            content=b"dxf",
            analysis_options=SimpleNamespace(),
            classify_part=lambda *_args, **_kwargs: None,
            logger_instance=logging.getLogger("test"),
        )

    assert calls == ["baseline", "shadow", "fusion", "hybrid", "finalize", "flag"]
    assert result["part_type"] == "人孔"
    assert captured["finalize_kwargs"]["low_confidence_threshold"] == 0.55
    assert captured["finalize_kwargs"]["high_confidence_threshold"] == 0.91
    assert captured["flag_kwargs"] == {
        "analysis_id": "analysis-1",
        "cls_payload": result,
    }


@pytest.mark.asyncio
async def test_run_classification_pipeline_survives_fusion_failure(monkeypatch, caplog):
    calls: list[str] = []

    async def _baseline(**kwargs):
        calls.append("baseline")
        return {
            "payload": {"part_type": "simple_plate", "confidence": 0.2, "rule_version": "v1"},
            "text_signals": "sig",
            "entity_counts": {},
            "doc_metadata": {},
            "l2_features": {},
            "l3_features": {},
        }

    async def _shadow(payload, **kwargs):
        calls.append("shadow")
        return {
            "payload": payload,
            "ml_result": None,
            "graph2d_fusable": None,
            "hybrid_result": {"label": "人孔", "confidence": 0.95},
        }

    def _fusion(payload, **kwargs):
        calls.append("fusion")
        raise RuntimeError("fusion unavailable")

    def _hybrid(payload, **kwargs):
        calls.append("hybrid")
        return {"payload": {**payload, "part_type": "人孔"}}

    def _finalize(payload, **kwargs):
        calls.append("finalize")
        return {**payload, "needs_review": False}

    with (
        patch(
            "src.core.classification.classification_pipeline.build_baseline_classification_context",
            _baseline,
        ),
        patch(
            "src.core.classification.classification_pipeline.build_shadow_classification_context",
            _shadow,
        ),
        patch(
            "src.core.classification.classification_pipeline.build_fusion_classification_context",
            _fusion,
        ),
        patch(
            "src.core.classification.classification_pipeline.build_hybrid_override_context",
            _hybrid,
        ),
        patch(
            "src.core.classification.classification_pipeline.finalize_classification_payload",
            _finalize,
        ),
        patch(
            "src.core.classification.classification_pipeline.flag_classification_for_review",
        ) as flag_mock,
        caplog.at_level(logging.ERROR),
    ):
        result = await run_classification_pipeline(
            analysis_id="analysis-2",
            doc=SimpleNamespace(metadata={}),
            features={"geometric": [], "semantic": []},
            features_3d=None,
            file_name="sample.dxf",
            file_format="dxf",
            content=b"dxf",
            analysis_options=SimpleNamespace(),
            classify_part=lambda *_args, **_kwargs: None,
        )

    assert calls == ["baseline", "shadow", "fusion", "hybrid", "finalize"]
    assert result["part_type"] == "人孔"
    assert "FusionAnalyzer failed: fusion unavailable" in caplog.text
    flag_mock.assert_called_once()


@pytest.mark.asyncio
async def test_run_classification_pipeline_survives_active_learning_failure(
    monkeypatch, caplog
):
    async def _baseline(**kwargs):
        return {
            "payload": {"part_type": "simple_plate", "confidence": 0.2, "rule_version": "v1"},
            "text_signals": "sig",
            "entity_counts": {},
            "doc_metadata": {},
            "l2_features": {},
            "l3_features": {},
        }

    async def _shadow(payload, **kwargs):
        return {
            "payload": payload,
            "ml_result": None,
            "graph2d_fusable": None,
            "hybrid_result": None,
        }

    def _finalize(payload, **kwargs):
        return {**payload, "needs_review": True}

    def _flag(**kwargs):
        raise RuntimeError("queue unavailable")

    with (
        patch(
            "src.core.classification.classification_pipeline.build_baseline_classification_context",
            _baseline,
        ),
        patch(
            "src.core.classification.classification_pipeline.build_shadow_classification_context",
            _shadow,
        ),
        patch(
            "src.core.classification.classification_pipeline.build_fusion_classification_context",
            lambda payload, **kwargs: {"payload": payload},
        ),
        patch(
            "src.core.classification.classification_pipeline.build_hybrid_override_context",
            lambda payload, **kwargs: {"payload": payload},
        ),
        patch(
            "src.core.classification.classification_pipeline.finalize_classification_payload",
            _finalize,
        ),
        patch(
            "src.core.classification.classification_pipeline.flag_classification_for_review",
            _flag,
        ),
        caplog.at_level(logging.WARNING),
    ):
        result = await run_classification_pipeline(
            analysis_id="analysis-3",
            doc=SimpleNamespace(metadata={}),
            features={"geometric": [], "semantic": []},
            features_3d=None,
            file_name="sample.dxf",
            file_format="dxf",
            content=b"dxf",
            analysis_options=SimpleNamespace(),
            classify_part=lambda *_args, **_kwargs: None,
            logger_instance=logging.getLogger("test"),
        )

    assert result["needs_review"] is True
    assert "Active learning flag failed: queue unavailable" in caplog.text
