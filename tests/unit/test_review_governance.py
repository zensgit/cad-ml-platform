from __future__ import annotations

from src.core.classification import build_review_governance, derive_confidence_band


def test_derive_confidence_band_supports_rejected_and_thresholds() -> None:
    assert derive_confidence_band(0.91) == "high"
    assert derive_confidence_band(0.72) == "medium"
    assert derive_confidence_band(0.31) == "low"
    assert derive_confidence_band(0.0, rejected=True) == "rejected"
    assert derive_confidence_band(None) == "unknown"


def test_build_review_governance_prioritizes_knowledge_conflict() -> None:
    payload = build_review_governance(
        confidence=0.55,
        hybrid_rejection={"reason": "below_min_confidence"},
        branch_conflicts={"hybrid_vs_graph2d": True},
        violations=[{"category": "knowledge_conflict", "severity": "warn"}],
        low_confidence_threshold=0.6,
    )

    assert payload["needs_review"] is True
    assert payload["confidence_band"] == "rejected"
    assert payload["review_priority"] == "critical"
    assert payload["review_reasons"] == [
        "hybrid_rejected:below_min_confidence",
        "knowledge_conflict",
        "branch_conflict",
        "low_confidence",
    ]
    assert payload["review_priority_score"] > 4.0


def test_build_review_governance_no_review_for_high_confidence_clean_case() -> None:
    payload = build_review_governance(
        confidence=0.92,
        hybrid_rejection=None,
        branch_conflicts={},
        violations=[],
        low_confidence_threshold=0.6,
        high_confidence_threshold=0.85,
    )

    assert payload["needs_review"] is False
    assert payload["confidence_band"] == "high"
    assert payload["review_priority"] == "none"
    assert payload["review_reasons"] == []
    assert payload["review_priority_score"] == 0.0
