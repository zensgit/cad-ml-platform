from types import SimpleNamespace

from src.core.classification.override_policy import (
    apply_fusion_override,
    apply_hybrid_override,
)
from src.core.knowledge.fusion_contracts import DecisionSource


def test_apply_fusion_override_applies_when_threshold_passes():
    decision = SimpleNamespace(
        primary_label="bolt",
        confidence=0.88,
        schema_version="v1",
        source=DecisionSource.AI_MODEL,
        rule_hits=[],
    )

    result = apply_fusion_override(
        {"part_type": "unknown", "confidence": 0.2, "rule_version": "v1"},
        fusion_decision=decision,
        override_enabled=True,
        min_confidence=0.5,
    )

    assert result["part_type"] == "bolt"
    assert result["confidence"] == 0.88
    assert result["rule_version"] == "FusionAnalyzer-v1"
    assert result["confidence_source"] == "fusion"


def test_apply_fusion_override_skips_default_rule_only():
    decision = SimpleNamespace(
        primary_label="unknown",
        confidence=0.91,
        schema_version="v1",
        source=DecisionSource.RULE_BASED,
        rule_hits=["RULE_DEFAULT"],
    )

    result = apply_fusion_override(
        {"part_type": "simple_plate", "confidence": 0.3, "rule_version": "v1"},
        fusion_decision=decision,
        override_enabled=True,
        min_confidence=0.5,
    )

    assert result["part_type"] == "simple_plate"
    assert result["fusion_override_skipped"] == {
        "min_confidence": 0.5,
        "decision_confidence": 0.91,
        "reason": "default_rule_only",
    }


def test_apply_hybrid_override_auto_placeholder():
    result = apply_hybrid_override(
        {
            "part_type": "unknown",
            "confidence": 0.2,
            "rule_version": "v1",
            "confidence_source": "rules",
        },
        hybrid_result={"label": "人孔", "confidence": 0.95},
        override_enabled=False,
        auto_override_enabled=True,
        min_confidence=0.8,
        base_max_confidence=0.7,
        is_drawing_type=lambda _: False,
    )

    assert result["part_type"] == "人孔"
    assert result["confidence_source"] == "hybrid"
    assert result["rule_version"] == "HybridClassifier-v1"
    assert result["hybrid_override_applied"]["mode"] == "auto"


def test_apply_hybrid_override_auto_low_confidence():
    result = apply_hybrid_override(
        {
            "part_type": "传动件",
            "confidence": 0.2,
            "rule_version": "v2",
            "confidence_source": "rules",
        },
        hybrid_result={"label": "人孔", "confidence": 0.9},
        override_enabled=False,
        auto_override_enabled=True,
        min_confidence=0.8,
        base_max_confidence=0.7,
        is_drawing_type=lambda _: False,
    )

    assert result["part_type"] == "人孔"
    assert result["hybrid_override_applied"]["mode"] == "auto_low_conf"


def test_apply_hybrid_override_env_skip_records_reason():
    result = apply_hybrid_override(
        {
            "part_type": "simple_plate",
            "confidence": 0.2,
            "rule_version": "v1",
            "confidence_source": "rules",
        },
        hybrid_result={"label": "人孔", "confidence": 0.6},
        override_enabled=True,
        auto_override_enabled=True,
        min_confidence=0.8,
        base_max_confidence=0.7,
        is_drawing_type=lambda _: False,
    )

    assert result["part_type"] == "simple_plate"
    assert result["hybrid_override_skipped"] == {
        "min_confidence": 0.8,
        "decision_confidence": 0.6,
        "label": "人孔",
    }
