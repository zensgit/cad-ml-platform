"""P2 — the explainer must NOT treat a fusion-excluded history as a MODEL signal.

Fusion already excludes a degraded history: ``hybrid_classifier`` stamps
``used_for_fusion=False`` (with ``degraded=True`` / ``model_available=False``,
design-lock §403–410 F4). But ``HybridExplainer`` previously still computed the
history contribution, candidate label, source contribution, and source-disagreement
as if history were a participating model signal — over-crediting a signal the
classifier already dropped.

These tests are DISCRIMINATING: a healthy (``used_for_fusion`` not False) history
prediction DOES produce a model contribution / candidate label / disagreement,
while its degraded twin — IDENTICAL label and confidence, only the degrade markers
differ — does NOT. If the explainer ignored the markers, the degraded assertions
below would fail exactly like the healthy positive controls.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.ml.hybrid.explainer import HybridExplainer
from src.ml.hybrid_classifier import ClassificationResult, DecisionSource


def _make_result(
    *,
    label: str,
    history_prediction: Optional[Dict[str, Any]],
) -> ClassificationResult:
    """A minimal fusion result: a healthy filename model vote + a history payload.

    ``source_contributions`` is deliberately left empty ({} default) so the
    explainer's manual source-contribution branch (the code under test) runs
    instead of short-circuiting on a precomputed map.
    """
    return ClassificationResult(
        label=label,
        confidence=0.9,  # >= 0.5 so "置信度较低" never clutters uncertainty_sources
        source=DecisionSource.FUSION,
        filename_prediction={"label": label, "confidence": 0.9},
        history_prediction=history_prediction,
        fusion_weights={"filename": 0.7, "history_sequence": 0.2},
    )


# A degraded history payload as stamped by hybrid_classifier's F4 path.
_DEGRADED = {
    "label": "轴类",
    "confidence": 0.99,
    "status": "ok",
    "used_for_fusion": False,
    "degraded": True,
    "model_available": False,
    "auxiliary_role": "rule_fallback",
}
# Identical label/confidence, but a healthy signal that DID participate in fusion.
_HEALTHY = {
    "label": "轴类",
    "confidence": 0.99,
    "status": "ok",
    "used_for_fusion": True,
    "model_available": True,
    "degraded": False,
}


def _history_model_contributions(explanation) -> list:
    return [
        f for f in explanation.feature_contributions if f.source == "history_sequence"
    ]


def test_degraded_history_produces_no_model_contribution() -> None:
    """Degraded history must not yield a source=="history_sequence" contribution."""
    explainer = HybridExplainer()
    exp = explainer.explain(_make_result(label="阀体", history_prediction=_DEGRADED))

    # No MODEL contribution for the excluded history.
    assert _history_model_contributions(exp) == []
    # If surfaced at all, it is an explicit rule/fallback auxiliary entry (zero weight),
    # never a model source and never a positive/negative model factor.
    aux = [f for f in exp.feature_contributions if f.source == "rule_fallback"]
    for f in aux:
        assert f.contribution == 0.0
        assert f.description not in exp.top_positive_features
        assert f.description not in exp.top_negative_features


def test_healthy_history_does_produce_a_model_contribution() -> None:
    """Positive control: an identical but healthy history IS a model signal."""
    explainer = HybridExplainer()
    exp = explainer.explain(_make_result(label="阀体", history_prediction=_HEALTHY))

    model_contribs = _history_model_contributions(exp)
    assert len(model_contribs) == 1
    # Label mismatch ("轴类" != "阀体") → opposing model signal, non-zero weight.
    assert model_contribs[0].contribution != 0.0


def test_degraded_history_is_not_a_candidate_label() -> None:
    """Degraded history's label must not appear among alternative model labels."""
    explainer = HybridExplainer()
    deg = explainer.explain(_make_result(label="阀体", history_prediction=_DEGRADED))
    healthy = explainer.explain(_make_result(label="阀体", history_prediction=_HEALTHY))

    deg_labels = [label for label, _ in deg.alternative_labels]
    healthy_labels = [label for label, _ in healthy.alternative_labels]
    assert "轴类" not in deg_labels          # excluded
    assert "轴类" in healthy_labels           # positive control


def test_degraded_history_is_not_a_disagreeing_model_source() -> None:
    """Degraded history must not trigger the multi-source-disagreement flag."""
    explainer = HybridExplainer()
    deg = explainer.explain(_make_result(label="阀体", history_prediction=_DEGRADED))
    healthy = explainer.explain(_make_result(label="阀体", history_prediction=_HEALTHY))

    # Only filename ("阀体") remains as a model source → no disagreement.
    assert "多源预测不一致" not in deg.uncertainty_sources
    # Healthy history ("轴类") disagrees with filename ("阀体") → disagreement flagged.
    assert "多源预测不一致" in healthy.uncertainty_sources


def test_degraded_history_is_not_counted_in_source_contributions() -> None:
    """When history agrees with the result label, a degraded one still contributes 0.

    Here history label == result label ("阀体"); a healthy history would add a
    positive "历史序列" source contribution, but the degraded twin must not.
    """
    explainer = HybridExplainer()

    deg_hist = dict(_DEGRADED, label="阀体")
    healthy_hist = dict(_HEALTHY, label="阀体")

    deg = explainer.explain(_make_result(label="阀体", history_prediction=deg_hist))
    healthy = explainer.explain(
        _make_result(label="阀体", history_prediction=healthy_hist)
    )

    assert "历史序列" not in deg.source_contributions          # excluded
    assert healthy.source_contributions.get("历史序列", 0.0) > 0.0  # positive control
