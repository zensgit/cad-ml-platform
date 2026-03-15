from __future__ import annotations

from src.api.v1.analyze import (
    _build_graph2d_soft_override_suggestion,
    _enrich_graph2d_prediction,
)


def _set_graph2d_gate_env(monkeypatch) -> None:
    monkeypatch.setenv("GRAPH2D_MIN_CONF", "0.4")
    monkeypatch.setenv("GRAPH2D_MIN_MARGIN", "0.1")
    monkeypatch.setenv("GRAPH2D_ALLOW_LABELS", "人孔,出料凸缘,装配图")
    monkeypatch.setenv("GRAPH2D_EXCLUDE_LABELS", "other,禁止件")
    monkeypatch.setenv("GRAPH2D_DRAWING_TYPE_LABELS", "装配图,原理图")
    monkeypatch.setenv("GRAPH2D_COARSE_LABELS", "传动件,连接件")


def test_enrich_graph2d_prediction_unavailable_not_fusable(monkeypatch) -> None:
    _set_graph2d_gate_env(monkeypatch)

    result, fusable = _enrich_graph2d_prediction(
        {"status": "model_unavailable", "confidence": 0.0},
        graph2d_ensemble_enabled=False,
    )

    assert result["status"] == "model_unavailable"
    assert result["min_confidence"] == 0.4
    assert result["min_margin"] == 0.1
    assert result["ensemble_enabled"] is False
    assert fusable is None


def test_enrich_graph2d_prediction_threshold_and_margin_gate(monkeypatch) -> None:
    _set_graph2d_gate_env(monkeypatch)

    result, fusable = _enrich_graph2d_prediction(
        {"label": "人孔", "confidence": 0.39, "margin": 0.05},
        graph2d_ensemble_enabled=True,
    )

    assert result["passed_threshold"] is False
    assert result["passed_margin"] is False
    assert result["allowed"] is True
    assert result["excluded"] is False
    assert fusable is None


def test_enrich_graph2d_prediction_drawing_type_not_fusable(monkeypatch) -> None:
    _set_graph2d_gate_env(monkeypatch)

    result, fusable = _enrich_graph2d_prediction(
        {"label": "装配图", "confidence": 0.95, "margin": 0.8},
        graph2d_ensemble_enabled=True,
    )

    assert result["passed_threshold"] is True
    assert result["passed_margin"] is True
    assert result["allowed"] is True
    assert result["is_drawing_type"] is True
    assert fusable is None


def test_enrich_graph2d_prediction_valid_label_is_fusable(monkeypatch) -> None:
    _set_graph2d_gate_env(monkeypatch)

    result, fusable = _enrich_graph2d_prediction(
        {"label": "出料凸缘", "confidence": 0.91, "margin": 0.3},
        graph2d_ensemble_enabled=True,
    )

    assert result["passed_threshold"] is True
    assert result["passed_margin"] is True
    assert result["allowed"] is True
    assert result["excluded"] is False
    assert result["is_drawing_type"] is False
    assert fusable is not None
    assert fusable["label"] == "出料凸缘"


def test_build_graph2d_soft_override_suggestion_reason_priority(monkeypatch) -> None:
    monkeypatch.setenv("GRAPH2D_SOFT_OVERRIDE_MIN_CONF", "0.8")

    graph2d_result = {
        "label": "出料凸缘",
        "confidence": 0.75,
        "allowed": True,
        "excluded": False,
        "passed_margin": True,
        "min_margin": 0.1,
        "is_drawing_type": False,
        "is_coarse_label": False,
        "min_confidence": 0.4,
    }
    cls_payload = {"confidence_source": "rules", "rule_version": "v1"}
    suggestion = _build_graph2d_soft_override_suggestion(
        graph2d_result=graph2d_result,
        cls_payload=cls_payload,
    )
    assert suggestion is not None
    assert suggestion["eligible"] is False
    assert suggestion["reason"] == "below_threshold"
    assert suggestion["threshold"] == 0.8

    suggestion2 = _build_graph2d_soft_override_suggestion(
        graph2d_result=graph2d_result,
        cls_payload={"confidence_source": "model", "rule_version": "v1"},
    )
    assert suggestion2 is not None
    assert suggestion2["eligible"] is False
    assert suggestion2["reason"] == "confidence_source_not_rules"
