"""F4 — a degraded history prediction must EXIT model fusion.

Owner ruling (L3 #532 round-2 review, design-lock §403–410): the history
sequence family keeps status="ok" for legacy compat, but when the pinned model
is unavailable and only the rule-based prototype answered, the payload is
stamped degraded=True / model_available=False. Such a prediction MUST NOT be
selected as a DecisionSource.HISTORY model vote nor fused as a model signal; it
may only be recorded as an explicit rule/fallback-labeled auxiliary result.

These tests are DISCRIMINATING: a non-degraded, otherwise-identical high-conf
history prediction IS selected as DecisionSource.HISTORY, while the degraded
twin is NOT.
"""

from __future__ import annotations

from src.ml.hybrid_config import reset_config
from src.ml.hybrid_classifier import DecisionSource, HybridClassifier


def _history_only_env(monkeypatch) -> None:
    monkeypatch.setenv("FILENAME_CLASSIFIER_ENABLED", "false")
    monkeypatch.setenv("GRAPH2D_ENABLED", "false")
    monkeypatch.setenv("TITLEBLOCK_ENABLED", "false")
    monkeypatch.setenv("PROCESS_FEATURES_ENABLED", "false")
    monkeypatch.setenv("HISTORY_SEQUENCE_ENABLED", "true")


def test_non_degraded_history_is_selected(monkeypatch) -> None:
    """Positive control: a healthy high-conf history prediction wins."""
    _history_only_env(monkeypatch)
    reset_config()
    classifier = HybridClassifier(history_min_conf=0.4, history_weight=0.5)
    result = classifier.classify(
        filename="sample.dxf",
        history_result={
            "status": "ok",
            "label": "轴类",
            "confidence": 0.99,
            "source": "history_sequence_prototype",
            "model_available": True,
            "degraded": False,
        },
    )

    assert result.label == "轴类"
    assert result.source == DecisionSource.HISTORY
    assert "history_high_conf_adopted" in result.decision_path
    assert "history_degraded_excluded_from_fusion" not in result.decision_path
    reset_config()


def test_degraded_history_is_not_selected_as_history_source(monkeypatch) -> None:
    """Degraded twin of the positive control must NOT win as HISTORY.

    Same label and IDENTICAL confidence (0.99) as the healthy case above — the
    ONLY difference is degraded=True / model_available=False. If the degrade marker
    were ignored, this would be adopted exactly like the positive control.
    """
    _history_only_env(monkeypatch)
    reset_config()
    classifier = HybridClassifier(history_min_conf=0.4, history_weight=0.5)
    result = classifier.classify(
        filename="sample.dxf",
        history_result={
            "status": "ok",  # legacy-compat status is deliberately still "ok"
            "label": "轴类",
            "confidence": 0.99,  # identical to the positive control above
            "source": "history_sequence_prototype",
            "model_available": False,
            "degraded": True,
            "degraded_reason": "pinned_model_unavailable",
        },
    )

    # It must NOT be selected as a HISTORY model vote.
    assert result.source != DecisionSource.HISTORY
    assert result.source == DecisionSource.FALLBACK
    assert result.label != "轴类"
    assert "history_high_conf_adopted" not in result.decision_path

    # It IS retained as an explicit rule/fallback-labeled auxiliary record.
    assert "history_degraded_excluded_from_fusion" in result.decision_path
    assert result.history_prediction is not None
    assert result.history_prediction.get("used_for_fusion") is False
    assert (
        result.history_prediction.get("fusion_excluded_reason")
        == "degraded_model_unavailable"
    )
    assert result.history_prediction.get("auxiliary_role") == "rule_fallback"
    assert result.history_prediction.get("auxiliary_label") == "轴类"
    reset_config()


def test_degraded_via_model_available_false_only(monkeypatch) -> None:
    """model_available=False alone (no explicit degraded flag) also excludes."""
    _history_only_env(monkeypatch)
    reset_config()
    classifier = HybridClassifier(history_min_conf=0.4, history_weight=0.5)
    result = classifier.classify(
        filename="sample.dxf",
        history_result={
            "status": "ok",
            "label": "轴类",
            "confidence": 0.95,
            "source": "history_sequence_prototype",
            "model_available": False,
        },
    )

    assert result.source != DecisionSource.HISTORY
    assert "history_degraded_excluded_from_fusion" in result.decision_path
    reset_config()
