from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core.classification.fusion_pipeline import (
    build_fusion_classification_context,
)
from src.core.knowledge.fusion_contracts import DecisionSource


class _StubFusionDecision:
    def __init__(
        self,
        *,
        primary_label: str = "bolt",
        confidence: float = 0.58,
        schema_version: str = "v1.2",
        source: object = DecisionSource.AI_MODEL,
        rule_hits: list[str] | None = None,
    ) -> None:
        self.primary_label = primary_label
        self.confidence = confidence
        self.schema_version = schema_version
        self.source = source
        self.rule_hits = list(rule_hits or [])

    def model_dump(self) -> dict[str, object]:
        return {
            "primary_label": self.primary_label,
            "confidence": self.confidence,
            "schema_version": self.schema_version,
            "source": getattr(self.source, "value", self.source),
            "rule_hits": list(self.rule_hits),
        }


def test_build_fusion_classification_context_prefers_graph2d_l4(monkeypatch):
    captured: dict[str, object] = {}

    class _StubFusionAnalyzer:
        def analyze(self, **kwargs):  # noqa: ANN003, ANN201
            captured["kwargs"] = kwargs
            return _StubFusionDecision()

    monkeypatch.setenv("FUSION_ANALYZER_ENABLED", "true")
    monkeypatch.setenv("FUSION_ANALYZER_OVERRIDE", "false")
    monkeypatch.setenv("GRAPH2D_FUSION_ENABLED", "true")

    result = build_fusion_classification_context(
        {"part_type": "simple_plate", "confidence": 0.2, "rule_version": "v1"},
        doc_metadata={"source": "unit"},
        l2_features={"aspect_ratio": 2.1},
        l3_features={"faces": 12},
        ml_result={"predicted_type": "ml_part", "confidence": 0.72},
        graph2d_fusable={"label": "人孔", "confidence": 0.91},
        fusion_analyzer_factory=lambda: _StubFusionAnalyzer(),
    )

    assert result["enabled"] is True
    assert result["l4_prediction"] == {
        "label": "人孔",
        "confidence": 0.91,
        "source": "graph2d",
    }
    payload = result["payload"]
    assert payload["part_type"] == "simple_plate"
    assert payload["fusion_decision"]["schema_version"] == "v1.2"
    assert payload["fusion_inputs"] == {
        "l1": {"source": "unit"},
        "l2": {"aspect_ratio": 2.1},
        "l3": {"faces": 12},
        "l4": {
            "label": "人孔",
            "confidence": 0.91,
            "source": "graph2d",
        },
    }
    assert (captured["kwargs"])["l4_prediction"] == payload["fusion_inputs"]["l4"]


def test_build_fusion_classification_context_falls_back_to_ml_l4(monkeypatch):
    captured: dict[str, object] = {}

    class _StubFusionAnalyzer:
        def analyze(self, **kwargs):  # noqa: ANN003, ANN201
            captured["kwargs"] = kwargs
            return _StubFusionDecision(primary_label="ml_part", confidence=0.61)

    monkeypatch.setenv("FUSION_ANALYZER_ENABLED", "true")
    monkeypatch.setenv("FUSION_ANALYZER_OVERRIDE", "false")
    monkeypatch.setenv("GRAPH2D_FUSION_ENABLED", "true")

    result = build_fusion_classification_context(
        {"part_type": "simple_plate", "confidence": 0.2, "rule_version": "v1"},
        doc_metadata={"source": "unit"},
        l2_features={"aspect_ratio": 1.2},
        l3_features={},
        ml_result={"predicted_type": "法兰", "confidence": 0.77},
        graph2d_fusable=None,
        fusion_analyzer_factory=lambda: _StubFusionAnalyzer(),
    )

    assert result["l4_prediction"] == {
        "label": "法兰",
        "confidence": 0.77,
        "source": "ml",
    }
    assert (captured["kwargs"])["l4_prediction"] == result["l4_prediction"]


def test_build_fusion_classification_context_skips_graph2d_when_fusion_disabled(
    monkeypatch,
):
    captured: dict[str, object] = {}

    class _StubFusionAnalyzer:
        def analyze(self, **kwargs):  # noqa: ANN003, ANN201
            captured["kwargs"] = kwargs
            return _StubFusionDecision(primary_label="ml_part", confidence=0.61)

    monkeypatch.setenv("FUSION_ANALYZER_ENABLED", "true")
    monkeypatch.setenv("FUSION_ANALYZER_OVERRIDE", "false")
    monkeypatch.setenv("GRAPH2D_FUSION_ENABLED", "false")

    result = build_fusion_classification_context(
        {"part_type": "simple_plate", "confidence": 0.2, "rule_version": "v1"},
        doc_metadata={"source": "unit"},
        l2_features={"aspect_ratio": 1.2},
        l3_features={},
        ml_result={"predicted_type": "法兰", "confidence": 0.77},
        graph2d_fusable={"label": "人孔", "confidence": 0.91},
        fusion_analyzer_factory=lambda: _StubFusionAnalyzer(),
    )

    assert result["l4_prediction"] == {
        "label": "法兰",
        "confidence": 0.77,
        "source": "ml",
    }
    assert (captured["kwargs"])["l4_prediction"] == result["l4_prediction"]


def test_build_fusion_classification_context_applies_override(monkeypatch):
    monkeypatch.setenv("FUSION_ANALYZER_ENABLED", "true")
    monkeypatch.setenv("FUSION_ANALYZER_OVERRIDE", "true")
    monkeypatch.setenv("FUSION_ANALYZER_OVERRIDE_MIN_CONF", "0.5")
    monkeypatch.setenv("GRAPH2D_FUSION_ENABLED", "false")

    result = build_fusion_classification_context(
        {
            "part_type": "simple_plate",
            "confidence": 0.2,
            "rule_version": "v1",
            "confidence_source": "rules",
        },
        doc_metadata={"source": "unit"},
        l2_features={"aspect_ratio": 1.4},
        l3_features={},
        ml_result={"predicted_type": "法兰", "confidence": 0.77},
        graph2d_fusable=None,
        fusion_analyzer_factory=lambda: SimpleNamespace(
            analyze=lambda **_: _StubFusionDecision(
                primary_label="bolt",
                confidence=0.88,
                schema_version="v1.2",
                source=DecisionSource.AI_MODEL,
            )
        ),
    )

    payload = result["payload"]
    assert payload["part_type"] == "bolt"
    assert payload["confidence"] == 0.88
    assert payload["rule_version"] == "FusionAnalyzer-v1.2"
    assert payload["confidence_source"] == "fusion"


def test_build_fusion_classification_context_preserves_default_rule_skip(monkeypatch):
    monkeypatch.setenv("FUSION_ANALYZER_ENABLED", "true")
    monkeypatch.setenv("FUSION_ANALYZER_OVERRIDE", "true")
    monkeypatch.setenv("FUSION_ANALYZER_OVERRIDE_MIN_CONF", "0.5")

    result = build_fusion_classification_context(
        {
            "part_type": "simple_plate",
            "confidence": 0.2,
            "rule_version": "v1",
            "confidence_source": "rules",
        },
        doc_metadata={"source": "unit"},
        l2_features={},
        l3_features={},
        ml_result=None,
        graph2d_fusable=None,
        fusion_analyzer_factory=lambda: SimpleNamespace(
            analyze=lambda **_: _StubFusionDecision(
                primary_label="unknown",
                confidence=0.91,
                source=DecisionSource.RULE_BASED,
                rule_hits=["RULE_DEFAULT"],
            )
        ),
    )

    payload = result["payload"]
    assert payload["part_type"] == "simple_plate"
    assert payload["fusion_override_skipped"] == {
        "min_confidence": 0.5,
        "decision_confidence": 0.91,
        "reason": "default_rule_only",
    }


def test_build_fusion_classification_context_disabled_path(monkeypatch):
    monkeypatch.setenv("FUSION_ANALYZER_ENABLED", "false")
    monkeypatch.setenv("GRAPH2D_FUSION_ENABLED", "true")

    result = build_fusion_classification_context(
        {"part_type": "simple_plate", "confidence": 0.2, "rule_version": "v1"},
        doc_metadata={"source": "unit"},
        l2_features={},
        l3_features={},
        ml_result={"predicted_type": "法兰", "confidence": 0.77},
        graph2d_fusable={"label": "人孔", "confidence": 0.91},
        fusion_analyzer_factory=lambda: pytest.fail("fusion analyzer should not run"),
    )

    assert result == {
        "payload": {
            "part_type": "simple_plate",
            "confidence": 0.2,
            "rule_version": "v1",
        },
        "enabled": False,
        "graph2d_fusion_enabled": True,
        "l4_prediction": None,
    }


def test_build_fusion_classification_context_propagates_analyzer_errors(monkeypatch):
    monkeypatch.setenv("FUSION_ANALYZER_ENABLED", "true")

    with pytest.raises(RuntimeError, match="fusion unavailable"):
        build_fusion_classification_context(
            {"part_type": "simple_plate", "confidence": 0.2, "rule_version": "v1"},
            doc_metadata={"source": "unit"},
            l2_features={},
            l3_features={},
            ml_result=None,
            graph2d_fusable=None,
            fusion_analyzer_factory=lambda: SimpleNamespace(
                analyze=lambda **_: (_ for _ in ()).throw(
                    RuntimeError("fusion unavailable")
                )
            ),
        )
