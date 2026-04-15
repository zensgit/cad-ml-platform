from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.classification import shadow_pipeline


class _MetricRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def labels(self, **labels):  # noqa: ANN003, ANN201
        self.calls.append(("labels", labels))
        return self

    def inc(self) -> None:
        self.calls.append(("inc", None))

    def observe(self, value) -> None:  # noqa: ANN001
        self.calls.append(("observe", value))


class _AsyncProvider:
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        self.calls: list[tuple[object, dict[str, object]]] = []

    async def process(self, request, **kwargs):  # noqa: ANN001, ANN201
        self.calls.append((request, kwargs))
        return dict(self.result)


@pytest.mark.asyncio
async def test_build_shadow_classification_context_populates_graph2d_hybrid_and_soft_override(
    monkeypatch,
) -> None:
    graph2d_provider = _AsyncProvider(
        {
            "label": "人孔",
            "confidence": 0.91,
            "margin": 0.3,
            "status": "ok",
        }
    )
    hybrid_provider = _AsyncProvider(
        {
            "label": "法兰",
            "confidence": 0.95,
            "source": "filename_exact",
            "filename_prediction": {
                "label": "法兰",
                "confidence": 0.95,
                "source": "filename",
            },
            "titleblock_prediction": None,
            "history_prediction": {
                "label": "法兰",
                "confidence": 0.72,
                "source": "history_sequence",
            },
            "process_prediction": {"label": "laser_cut", "confidence": 0.8},
            "decision_path": ["filename_exact"],
            "source_contributions": {"filename": 0.95},
            "fusion_metadata": {"mode": "unit"},
            "explanation": "unit test",
        }
    )

    monkeypatch.setattr(
        shadow_pipeline,
        "_predict_ml_from_features",
        lambda _features: {
            "predicted_type": "stub_ml",
            "confidence": 0.44,
            "model_version": "ml_v1",
        },
    )
    monkeypatch.setattr(
        shadow_pipeline,
        "_resolve_history_sequence_file_path",
        lambda **_kwargs: ("/tmp/Bolt_M6x20.h5", "sidecar_exact"),
    )
    monkeypatch.setattr(
        shadow_pipeline,
        "_make_classifier_request",
        lambda **kwargs: dict(kwargs),
    )
    monkeypatch.setattr(
        shadow_pipeline,
        "_get_classifier_provider",
        lambda provider_name: {
            "graph2d": graph2d_provider,
            "hybrid": hybrid_provider,
        }[provider_name],
    )

    monkeypatch.setenv("GRAPH2D_ENABLED", "true")
    monkeypatch.setenv("GRAPH2D_MIN_CONF", "0.4")
    monkeypatch.setenv("GRAPH2D_MIN_MARGIN", "0.1")
    monkeypatch.setenv("GRAPH2D_ALLOW_LABELS", "人孔")
    monkeypatch.setenv("GRAPH2D_EXCLUDE_LABELS", "other")
    monkeypatch.setenv("GRAPH2D_SOFT_OVERRIDE_MIN_CONF", "0.5")
    monkeypatch.setenv("HYBRID_CLASSIFIER_ENABLED", "true")
    monkeypatch.setenv("PART_CLASSIFIER_PROVIDER_ENABLED", "false")

    context = await shadow_pipeline.build_shadow_classification_context(
        {
            "part_type": "unknown",
            "confidence": 0.2,
            "rule_version": "v1",
            "confidence_source": "rules",
        },
        features={"geometric": [0.1], "semantic": [0.2]},
        file_name="Bolt_M6x20.dxf",
        file_format="dxf",
        content=b"0\nSECTION\n2\nEOF\n",
        analysis_options=SimpleNamespace(history_file_path=None),
    )

    payload = context["payload"]
    assert payload["ml_predicted_type"] == "stub_ml"
    assert payload["model_version"] == "ml_v1"
    assert payload["graph2d_prediction"]["label"] == "人孔"
    assert payload["soft_override_suggestion"]["eligible"] is True
    assert payload["history_sequence_input"] == {
        "resolved": True,
        "source": "sidecar_exact",
        "file_name": "Bolt_M6x20.h5",
    }
    assert payload["hybrid_decision"]["label"] == "法兰"
    assert payload["hybrid_rejected"] is False
    assert payload["fine_part_type"] == "法兰"
    assert payload["fine_confidence"] == 0.95
    assert context["graph2d_fusable"]["label"] == "人孔"
    assert context["hybrid_result"]["label"] == "法兰"

    graph2d_request, _ = graph2d_provider.calls[0]
    assert graph2d_request["filename"] == "Bolt_M6x20.dxf"
    hybrid_request, hybrid_kwargs = hybrid_provider.calls[0]
    assert hybrid_request["history_file_path"] == "/tmp/Bolt_M6x20.h5"
    assert hybrid_kwargs["graph2d_result"]["label"] == "人孔"


@pytest.mark.asyncio
async def test_build_shadow_classification_context_records_hybrid_rejection_metrics(
    monkeypatch,
) -> None:
    hybrid_provider = _AsyncProvider(
        {
            "label": None,
            "confidence": 0.0,
            "source": "fallback",
            "filename_prediction": {"label": "人孔", "confidence": 0.65},
            "titleblock_prediction": None,
            "rejection": {
                "reason": "below_min_confidence",
                "min_confidence": 0.8,
                "raw_label": "人孔",
                "raw_confidence": 0.65,
                "raw_source": "filename",
            },
        }
    )
    metric = _MetricRecorder()

    monkeypatch.setattr(shadow_pipeline, "analysis_hybrid_rejections_total", metric)
    monkeypatch.setattr(
        shadow_pipeline,
        "_predict_ml_from_features",
        lambda _features: {"status": "ml_skipped"},
    )
    monkeypatch.setattr(
        shadow_pipeline,
        "_resolve_history_sequence_file_path",
        lambda **_kwargs: (None, None),
    )
    monkeypatch.setattr(
        shadow_pipeline,
        "_make_classifier_request",
        lambda **kwargs: dict(kwargs),
    )
    monkeypatch.setattr(
        shadow_pipeline,
        "_get_classifier_provider",
        lambda provider_name: hybrid_provider,
    )

    monkeypatch.setenv("GRAPH2D_ENABLED", "false")
    monkeypatch.setenv("HYBRID_CLASSIFIER_ENABLED", "true")
    monkeypatch.setenv("PART_CLASSIFIER_PROVIDER_ENABLED", "false")

    context = await shadow_pipeline.build_shadow_classification_context(
        {
            "part_type": "unknown",
            "confidence": 0.2,
            "rule_version": "v1",
            "confidence_source": "rules",
        },
        features={"geometric": [], "semantic": []},
        file_name="Bolt_M6x20.dxf",
        file_format="dxf",
        content=b"0\nSECTION\n2\nEOF\n",
        analysis_options=SimpleNamespace(history_file_path=None),
    )

    payload = context["payload"]
    assert payload["hybrid_rejected"] is True
    assert payload["hybrid_rejection"]["reason"] == "below_min_confidence"
    assert ("labels", {"reason": "below_min_confidence", "raw_source": "filename"}) in metric.calls
    assert ("inc", None) in metric.calls


@pytest.mark.asyncio
async def test_build_shadow_classification_context_normalizes_part_classifier_shadow(
    monkeypatch,
    tmp_path: Path,
) -> None:
    part_provider = _AsyncProvider(
        {
            "status": "ok",
            "label": "stub_part",
            "confidence": 0.99,
            "model_version": "stub_v1",
            "needs_review": True,
            "review_reason": "edge_case",
            "top2_category": "stub_part_2",
            "top2_confidence": 0.2,
        }
    )
    req_metric = _MetricRecorder()
    sec_metric = _MetricRecorder()
    skipped_metric = _MetricRecorder()

    monkeypatch.setattr(
        shadow_pipeline,
        "_predict_ml_from_features",
        lambda _features: {"status": "ml_skipped"},
    )
    monkeypatch.setattr(
        shadow_pipeline,
        "analysis_part_classifier_requests_total",
        req_metric,
    )
    monkeypatch.setattr(
        shadow_pipeline,
        "analysis_part_classifier_seconds",
        sec_metric,
    )
    monkeypatch.setattr(
        shadow_pipeline,
        "analysis_part_classifier_skipped_total",
        skipped_metric,
    )
    monkeypatch.setattr(
        shadow_pipeline,
        "_make_classifier_request",
        lambda **kwargs: dict(kwargs),
    )
    monkeypatch.setattr(
        shadow_pipeline,
        "_get_classifier_provider",
        lambda provider_name: part_provider,
    )

    monkeypatch.setenv("GRAPH2D_ENABLED", "false")
    monkeypatch.setenv("HYBRID_CLASSIFIER_ENABLED", "false")
    monkeypatch.setenv("PART_CLASSIFIER_PROVIDER_ENABLED", "true")
    monkeypatch.setenv("PART_CLASSIFIER_PROVIDER_NAME", "part_stub")

    context = await shadow_pipeline.build_shadow_classification_context(
        {
            "part_type": "bolt",
            "confidence": 0.9,
            "rule_version": "L2-Fusion-v1",
            "confidence_source": "fusion",
        },
        features={"geometric": [], "semantic": []},
        file_name="Bolt_M6x20.dxf",
        file_format="dxf",
        content=b"0\nSECTION\n2\nEOF\n",
        analysis_options=SimpleNamespace(history_file_path=str(tmp_path / "unused.h5")),
    )

    payload = context["payload"]
    prediction = payload["part_classifier_prediction"]
    assert prediction["status"] == "ok"
    assert prediction["provider"] == "part_stub"
    assert payload["part_family"] == "stub_part"
    assert payload["part_family_source"] == "provider:part_stub"
    assert payload["part_family_model_version"] == "stub_v1"
    assert payload["part_family_needs_review"] is True
    assert payload["part_family_review_reason"] == "edge_case"
    assert payload["part_family_top2"] == {
        "label": "stub_part_2",
        "confidence": 0.2,
    }
    assert ("labels", {"status": "success", "provider": "part_stub"}) in req_metric.calls
    assert any(call[0] == "observe" for call in sec_metric.calls)
    assert skipped_metric.calls == []


@pytest.mark.asyncio
async def test_build_shadow_classification_context_marks_file_too_large_for_part_classifier(
    monkeypatch,
) -> None:
    skipped_metric = _MetricRecorder()

    monkeypatch.setattr(
        shadow_pipeline,
        "_predict_ml_from_features",
        lambda _features: {"status": "ml_skipped"},
    )
    monkeypatch.setattr(
        shadow_pipeline,
        "analysis_part_classifier_skipped_total",
        skipped_metric,
    )
    monkeypatch.setenv("GRAPH2D_ENABLED", "false")
    monkeypatch.setenv("HYBRID_CLASSIFIER_ENABLED", "false")
    monkeypatch.setenv("PART_CLASSIFIER_PROVIDER_ENABLED", "true")
    monkeypatch.setenv("PART_CLASSIFIER_PROVIDER_MAX_MB", "0.000001")

    context = await shadow_pipeline.build_shadow_classification_context(
        {
            "part_type": "bolt",
            "confidence": 0.9,
            "rule_version": "L2-Fusion-v1",
            "confidence_source": "fusion",
        },
        features={"geometric": [], "semantic": []},
        file_name="Bolt_M6x20.dxf",
        file_format="dxf",
        content=b"x" * 2048,
        analysis_options=SimpleNamespace(history_file_path=None),
    )

    payload = context["payload"]
    assert payload["part_classifier_prediction"]["status"] == "file_too_large"
    assert ("labels", {"reason": "file_too_large"}) in skipped_metric.calls
    assert ("inc", None) in skipped_metric.calls


@pytest.mark.asyncio
async def test_build_shadow_classification_context_disabled_path_returns_full_key_set(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        shadow_pipeline,
        "_predict_ml_from_features",
        lambda _features: {"status": "ml_skipped"},
    )
    monkeypatch.setenv("GRAPH2D_ENABLED", "false")
    monkeypatch.setenv("HYBRID_CLASSIFIER_ENABLED", "false")
    monkeypatch.setenv("PART_CLASSIFIER_PROVIDER_ENABLED", "false")

    context = await shadow_pipeline.build_shadow_classification_context(
        {
            "part_type": "bolt",
            "confidence": 0.9,
            "rule_version": "L2-Fusion-v1",
            "confidence_source": "fusion",
        },
        features={"geometric": [], "semantic": []},
        file_name="Bolt_M6x20.dxf",
        file_format="dxf",
        content=b"0\nSECTION\n2\nEOF\n",
        analysis_options=SimpleNamespace(history_file_path=None),
    )

    assert context["payload"]["model_version"] == "ml_skipped"
    assert context["ml_result"] == {"status": "ml_skipped"}
    assert context["graph2d_result"] is None
    assert context["graph2d_fusable"] is None
    assert context["hybrid_result"] is None
