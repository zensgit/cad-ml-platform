from __future__ import annotations

import pytest

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


def test_build_graph2d_soft_override_suggestion_none_input_returns_none() -> None:
    suggestion = _build_graph2d_soft_override_suggestion(
        graph2d_result=None,
        cls_payload={"confidence_source": "rules", "rule_version": "v1"},
    )
    assert suggestion is None


def test_build_graph2d_soft_override_suggestion_model_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("GRAPH2D_SOFT_OVERRIDE_MIN_CONF", "0.6")
    suggestion = _build_graph2d_soft_override_suggestion(
        graph2d_result={"status": "model_unavailable", "confidence": 0.0},
        cls_payload={"confidence_source": "rules", "rule_version": "v1"},
    )
    assert suggestion is not None
    assert suggestion["eligible"] is False
    assert suggestion["reason"] == "graph2d_unavailable"
    assert suggestion["threshold"] == 0.6


@pytest.mark.parametrize(
    "mutate,expected_reason",
    [
        (lambda payload, g: payload.update({"confidence_source": "model"}), "confidence_source_not_rules"),
        (lambda payload, g: payload.update({"rule_version": "v2"}), "rule_version_not_v1"),
        (lambda payload, g: g.update({"excluded": True}), "graph2d_excluded"),
        (lambda payload, g: g.update({"allowed": False}), "graph2d_not_allowed"),
        (lambda payload, g: g.update({"is_drawing_type": True}), "graph2d_drawing_type"),
        (lambda payload, g: g.update({"is_coarse_label": True}), "graph2d_coarse_label"),
        (lambda payload, g: g.update({"passed_margin": False}), "below_margin"),
        (lambda payload, g: g.update({"confidence": 0.2}), "below_threshold"),
    ],
)
def test_build_graph2d_soft_override_suggestion_reason_matrix(
    monkeypatch, mutate, expected_reason
) -> None:
    monkeypatch.delenv("GRAPH2D_SOFT_OVERRIDE_MIN_CONF", raising=False)
    cls_payload = {"confidence_source": "rules", "rule_version": "v1"}
    graph2d_result = {
        "label": "出料凸缘",
        "confidence": 0.9,
        "allowed": True,
        "excluded": False,
        "passed_margin": True,
        "min_margin": 0.1,
        "is_drawing_type": False,
        "is_coarse_label": False,
        "min_confidence": 0.4,
    }
    mutate(cls_payload, graph2d_result)
    suggestion = _build_graph2d_soft_override_suggestion(
        graph2d_result=graph2d_result,
        cls_payload=cls_payload,
    )
    assert suggestion is not None
    assert suggestion["eligible"] is False
    assert suggestion["reason"] == expected_reason


def test_build_graph2d_soft_override_suggestion_eligible_case(monkeypatch) -> None:
    monkeypatch.setenv("GRAPH2D_SOFT_OVERRIDE_MIN_CONF", "0.5")
    suggestion = _build_graph2d_soft_override_suggestion(
        graph2d_result={
            "label": "出料凸缘",
            "confidence": 0.95,
            "allowed": True,
            "excluded": False,
            "passed_margin": True,
            "min_margin": 0.1,
            "is_drawing_type": False,
            "is_coarse_label": False,
            "min_confidence": 0.4,
        },
        cls_payload={"confidence_source": "rules", "rule_version": "v1"},
    )
    assert suggestion is not None
    assert suggestion["eligible"] is True
    assert suggestion["reason"] == "eligible"
