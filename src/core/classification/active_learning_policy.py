"""Helpers for dispatching finalized classification results into active learning."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def _is_active_learning_enabled() -> bool:
    return os.getenv("ACTIVE_LEARNING_ENABLED", "false").lower() == "true"


def _normalize_hybrid_rejection(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    hybrid_rejection = payload.get("hybrid_rejection")
    return hybrid_rejection if isinstance(hybrid_rejection, dict) else None


def _derive_uncertainty_reason(payload: Dict[str, Any]) -> str:
    review_reasons = [
        str(reason).strip()
        for reason in (payload.get("review_reasons") or [])
        if str(reason).strip()
    ]
    return "+".join(review_reasons) or "low_confidence"


def _derive_sample_type(payload: Dict[str, Any]) -> str:
    if payload.get("review_has_knowledge_conflict"):
        return "knowledge_conflict"
    if payload.get("review_has_branch_conflict"):
        return "branch_conflict"
    if payload.get("review_has_hybrid_rejection"):
        return "hybrid_rejection"
    if payload.get("review_is_low_confidence"):
        return "low_confidence"
    return "review"


def _build_score_breakdown(payload: Dict[str, Any]) -> Dict[str, Any]:
    fusion_metadata = payload.get("fusion_metadata")
    shadow_predictions = None
    if isinstance(fusion_metadata, dict):
        shadow_predictions = fusion_metadata.get("shadow_predictions")

    return {
        "coarse_part_type": payload.get("coarse_part_type"),
        "fine_part_type": payload.get("fine_part_type"),
        "coarse_hybrid_label": payload.get("coarse_hybrid_label"),
        "coarse_graph2d_label": payload.get("coarse_graph2d_label"),
        "rule_version": payload.get("rule_version"),
        "model_version": payload.get("model_version"),
        "confidence_source": payload.get("confidence_source"),
        "confidence_breakdown": payload.get("confidence_breakdown"),
        "hybrid_rejection": _normalize_hybrid_rejection(payload),
        "decision_path": payload.get("decision_path"),
        "source_contributions": payload.get("source_contributions"),
        "history_prediction": payload.get("history_prediction"),
        "fusion_metadata": fusion_metadata,
        "shadow_predictions": shadow_predictions,
        "hybrid_explanation": payload.get("hybrid_explanation"),
        "knowledge_checks": payload.get("knowledge_checks"),
        "violations": payload.get("violations"),
        "standards_candidates": payload.get("standards_candidates"),
        "branch_conflicts": payload.get("branch_conflicts"),
        "needs_review": payload.get("needs_review"),
        "confidence_band": payload.get("confidence_band"),
        "review_priority": payload.get("review_priority"),
        "review_priority_score": payload.get("review_priority_score"),
        "review_reasons": payload.get("review_reasons"),
        "review_has_knowledge_conflict": payload.get(
            "review_has_knowledge_conflict"
        ),
        "review_has_branch_conflict": payload.get("review_has_branch_conflict"),
        "review_has_hybrid_rejection": payload.get("review_has_hybrid_rejection"),
        "review_is_low_confidence": payload.get("review_is_low_confidence"),
    }


def flag_classification_for_review(
    *,
    analysis_id: str,
    cls_payload: Dict[str, Any],
    active_learning_enabled: Optional[bool] = None,
) -> Optional[Any]:
    """Queue a finalized classification payload for human review when enabled."""
    enabled = (
        _is_active_learning_enabled()
        if active_learning_enabled is None
        else bool(active_learning_enabled)
    )
    if not enabled or not bool(cls_payload.get("needs_review")):
        return None

    review_priority = str(cls_payload.get("review_priority") or "medium").strip()
    if not review_priority:
        review_priority = "medium"

    from src.core.active_learning import get_active_learner

    learner = get_active_learner()
    return learner.flag_for_review(
        doc_id=analysis_id,
        predicted_type=str(cls_payload.get("part_type", "unknown")),
        confidence=float(cls_payload.get("confidence", 0.0)),
        alternatives=cls_payload.get("alternatives", []),
        score_breakdown=_build_score_breakdown(cls_payload),
        uncertainty_reason=_derive_uncertainty_reason(cls_payload),
        sample_type=_derive_sample_type(cls_payload),
        feedback_priority=review_priority,
    )


__all__ = ["flag_classification_for_review"]
