"""Unit-level coverage for decision_service evidence helpers.

The integration end (`tests/test_*_classify*.py`) exercises these helpers
indirectly via full decision flows. This module asserts their behaviour on
isolated inputs so future refactors don't silently change evidence shape.
"""

from __future__ import annotations

from src.core.classification.decision_service import (
    _brep_evidence,
    _prediction_evidence,
    _top_brep_hint,
)


# --- _prediction_evidence -------------------------------------------------


def test_prediction_evidence_returns_none_for_non_mapping() -> None:
    assert _prediction_evidence(source="graph2d", prediction=None) is None
    assert _prediction_evidence(source="graph2d", prediction="") is None
    assert _prediction_evidence(source="graph2d", prediction=[1, 2]) is None
    assert _prediction_evidence(source="graph2d", prediction=42) is None


def test_prediction_evidence_returns_none_when_label_confidence_status_all_missing() -> None:
    # Mapping exists but carries no actionable signal — must yield None to
    # avoid polluting the evidence list with empty rows.
    assert _prediction_evidence(source="graph2d", prediction={}) is None
    assert _prediction_evidence(source="graph2d", prediction={"reason": "x"}) is None


def test_prediction_evidence_extracts_label_confidence_status_and_kind() -> None:
    row = _prediction_evidence(
        source="graph2d",
        prediction={"label": "BRACKET", "confidence": 0.91, "status": "ok"},
    )
    assert row is not None
    assert row["source"] == "graph2d"
    assert row["kind"] == "prediction"
    assert row["label"] == "BRACKET"
    assert row["confidence"] == 0.91
    assert row["status"] == "ok"
    assert "details" not in row  # no detail keys provided


def test_prediction_evidence_prefers_label_key_over_alternates() -> None:
    # `label` wins over `predicted_type` / `part_type` / `primary_label`.
    row = _prediction_evidence(
        source="filename",
        prediction={
            "label": "BRACKET",
            "predicted_type": "SHAFT",
            "part_type": "PLATE",
            "primary_label": "WASHER",
            "confidence": 0.7,
        },
    )
    assert row is not None
    assert row["label"] == "BRACKET"


def test_prediction_evidence_falls_back_through_alternate_label_keys() -> None:
    # When `label` is missing, fall back in defined order.
    row = _prediction_evidence(
        source="ocr",
        prediction={"predicted_type": "SHAFT", "confidence": 0.6},
    )
    assert row is not None
    assert row["label"] == "SHAFT"


def test_prediction_evidence_attaches_contribution_when_provided() -> None:
    row = _prediction_evidence(
        source="hybrid",
        prediction={"label": "BRACKET", "confidence": 0.9},
        contribution=0.35,
    )
    assert row is not None
    assert row["contribution"] == 0.35


def test_prediction_evidence_omits_contribution_when_none() -> None:
    row = _prediction_evidence(
        source="hybrid",
        prediction={"label": "BRACKET", "confidence": 0.9},
        contribution=None,
    )
    assert row is not None
    assert "contribution" not in row


def test_prediction_evidence_compacts_detail_fields() -> None:
    row = _prediction_evidence(
        source="hybrid",
        prediction={
            "label": "BRACKET",
            "confidence": 0.91,
            "rule_version": "v2.3",
            "model_version": "graph2d-v4",
            "margin": 0.12,
            "passed_threshold": True,
            "reason": "",  # empty string should be dropped
            "allowed": None,  # None should be dropped
        },
    )
    assert row is not None
    details = row.get("details") or {}
    assert details.get("rule_version") == "v2.3"
    assert details.get("model_version") == "graph2d-v4"
    assert details.get("margin") == 0.12
    assert details.get("passed_threshold") is True
    assert "reason" not in details  # empty filtered
    assert "allowed" not in details  # None filtered


def test_prediction_evidence_kind_override() -> None:
    row = _prediction_evidence(
        source="baseline",
        prediction={"label": "X", "confidence": 0.5},
        kind="decision",
    )
    assert row is not None
    assert row["kind"] == "decision"


# --- _top_brep_hint -------------------------------------------------------


def test_top_brep_hint_empty_when_features_missing_hints() -> None:
    assert _top_brep_hint({}) == (None, None)
    assert _top_brep_hint({"feature_hints": None}) == (None, None)
    assert _top_brep_hint({"feature_hints": {}}) == (None, None)
    assert _top_brep_hint({"feature_hints": "not-a-dict"}) == (None, None)


def test_top_brep_hint_returns_argmax_label_and_score() -> None:
    label, score = _top_brep_hint(
        {"feature_hints": {"BRACKET": 0.3, "PLATE": 0.9, "WASHER": 0.6}}
    )
    assert label == "PLATE"
    assert score == 0.9


def test_top_brep_hint_skips_non_numeric_scores() -> None:
    # Non-coercible values are skipped, so the next-best numeric label wins.
    label, score = _top_brep_hint(
        {"feature_hints": {"PLATE": "not-a-number", "BRACKET": 0.4}}
    )
    assert label == "BRACKET"
    assert score == 0.4


# --- _brep_evidence ------------------------------------------------------


def test_brep_evidence_returns_none_for_empty_or_non_mapping() -> None:
    assert _brep_evidence(None) is None
    assert _brep_evidence({}) is None


def test_brep_evidence_promotes_top_hint_and_emits_valid_status() -> None:
    row = _brep_evidence(
        {
            "valid_3d": True,
            "feature_hints": {"PLATE": 0.9, "BRACKET": 0.3},
            "faces": 6,
            "edges": 12,
            "solids": 1,
            "surface_types": ["plane", "cylinder"],
        }
    )
    assert row is not None
    assert row["source"] == "brep"
    assert row["kind"] == "geometric_hint"
    assert row["label"] == "PLATE"
    assert row["confidence"] == 0.9
    # `valid` only emitted when valid_3d is True.
    assert row["status"] == "valid"
    details = row.get("details") or {}
    assert details.get("faces") == 6
    assert details.get("edges") == 12
    assert details.get("surface_types") == ["plane", "cylinder"]


def test_brep_evidence_status_none_when_valid_3d_not_true() -> None:
    row = _brep_evidence({"valid_3d": False, "feature_hints": {"PLATE": 0.9}})
    assert row is not None
    assert row["status"] is None


def test_brep_evidence_returns_none_when_no_signal() -> None:
    # Mapping exists but has neither a usable label / score nor any detail
    # fields that would justify emitting a row.
    assert _brep_evidence({"unused_key": "x"}) is None
