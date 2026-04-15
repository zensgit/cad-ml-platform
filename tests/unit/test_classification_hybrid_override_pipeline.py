from __future__ import annotations

from src.core.classification.hybrid_override_pipeline import (
    build_hybrid_override_context,
)


def test_build_hybrid_override_context_applies_env_override(monkeypatch):
    monkeypatch.setenv("HYBRID_CLASSIFIER_OVERRIDE", "true")
    monkeypatch.setenv("HYBRID_CLASSIFIER_AUTO_OVERRIDE", "false")
    monkeypatch.setenv("HYBRID_OVERRIDE_MIN_CONF", "0.5")
    monkeypatch.setenv("HYBRID_OVERRIDE_BASE_MAX_CONF", "0.4")

    result = build_hybrid_override_context(
        {
            "part_type": "传动件",
            "confidence": 0.95,
            "rule_version": "v2",
            "confidence_source": "rules",
        },
        hybrid_result={"label": "人孔", "confidence": 0.91},
        is_drawing_type=lambda _: False,
    )

    assert result["override_enabled"] is True
    assert result["auto_override_enabled"] is False
    assert result["min_confidence"] == 0.5
    assert result["base_max_confidence"] == 0.4
    assert result["payload"]["part_type"] == "人孔"
    assert result["payload"]["confidence_source"] == "hybrid"
    assert result["payload"]["hybrid_override_applied"]["mode"] == "env"


def test_build_hybrid_override_context_auto_override_uses_defaults(monkeypatch):
    monkeypatch.delenv("HYBRID_CLASSIFIER_OVERRIDE", raising=False)
    monkeypatch.delenv("HYBRID_CLASSIFIER_AUTO_OVERRIDE", raising=False)
    monkeypatch.delenv("HYBRID_OVERRIDE_MIN_CONF", raising=False)
    monkeypatch.delenv("HYBRID_OVERRIDE_BASE_MAX_CONF", raising=False)

    result = build_hybrid_override_context(
        {
            "part_type": "unknown",
            "confidence": 0.2,
            "rule_version": "v1",
            "confidence_source": "rules",
        },
        hybrid_result={"label": "人孔", "confidence": 0.95},
        is_drawing_type=lambda _: False,
    )

    assert result["override_enabled"] is False
    assert result["auto_override_enabled"] is True
    assert result["min_confidence"] == 0.8
    assert result["base_max_confidence"] == 0.7
    assert result["payload"]["part_type"] == "人孔"
    assert result["payload"]["hybrid_override_applied"]["mode"] == "auto"


def test_build_hybrid_override_context_records_env_skip(monkeypatch):
    monkeypatch.setenv("HYBRID_CLASSIFIER_OVERRIDE", "true")
    monkeypatch.setenv("HYBRID_OVERRIDE_MIN_CONF", "0.8")

    result = build_hybrid_override_context(
        {
            "part_type": "simple_plate",
            "confidence": 0.2,
            "rule_version": "v1",
            "confidence_source": "rules",
        },
        hybrid_result={"label": "人孔", "confidence": 0.6},
        is_drawing_type=lambda _: False,
    )

    assert result["payload"]["part_type"] == "simple_plate"
    assert result["payload"]["hybrid_override_skipped"] == {
        "min_confidence": 0.8,
        "decision_confidence": 0.6,
        "label": "人孔",
    }


def test_build_hybrid_override_context_auto_low_conf_mode(monkeypatch):
    monkeypatch.delenv("HYBRID_CLASSIFIER_OVERRIDE", raising=False)
    monkeypatch.delenv("HYBRID_CLASSIFIER_AUTO_OVERRIDE", raising=False)

    result = build_hybrid_override_context(
        {
            "part_type": "传动件",
            "confidence": 0.2,
            "rule_version": "v2",
            "confidence_source": "rules",
        },
        hybrid_result={"label": "人孔", "confidence": 0.9},
        is_drawing_type=lambda _: False,
    )

    assert result["payload"]["part_type"] == "人孔"
    assert result["payload"]["hybrid_override_applied"]["mode"] == "auto_low_conf"


def test_build_hybrid_override_context_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("HYBRID_CLASSIFIER_OVERRIDE", "true")
    monkeypatch.setenv("HYBRID_OVERRIDE_MIN_CONF", "oops")
    monkeypatch.setenv("HYBRID_OVERRIDE_BASE_MAX_CONF", "nope")

    result = build_hybrid_override_context(
        {
            "part_type": "simple_plate",
            "confidence": 0.2,
            "rule_version": "v1",
            "confidence_source": "rules",
        },
        hybrid_result={"label": "人孔", "confidence": 0.6},
        is_drawing_type=lambda _: False,
    )

    assert result["min_confidence"] == 0.8
    assert result["base_max_confidence"] == 0.7
    assert result["payload"]["hybrid_override_skipped"] == {
        "min_confidence": 0.8,
        "decision_confidence": 0.6,
        "label": "人孔",
    }


def test_build_hybrid_override_context_without_hybrid_result_returns_payload(monkeypatch):
    monkeypatch.setenv("HYBRID_CLASSIFIER_OVERRIDE", "false")

    result = build_hybrid_override_context(
        {
            "part_type": "simple_plate",
            "confidence": 0.2,
            "rule_version": "v1",
            "confidence_source": "rules",
        },
        hybrid_result=None,
        is_drawing_type=lambda _: False,
    )

    assert result["payload"] == {
        "part_type": "simple_plate",
        "confidence": 0.2,
        "rule_version": "v1",
        "confidence_source": "rules",
    }


def test_build_hybrid_override_context_accepts_none_payload(monkeypatch):
    monkeypatch.setenv("HYBRID_CLASSIFIER_OVERRIDE", "false")

    result = build_hybrid_override_context(
        None,
        hybrid_result=None,
        is_drawing_type=lambda _: False,
    )

    assert result["payload"] == {}
