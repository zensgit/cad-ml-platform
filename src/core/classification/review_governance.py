"""Helpers for consistent review governance across API, eval, and review-pack."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def _has_items(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, list):
        return bool(value)
    return False


def derive_confidence_band(
    confidence: Any,
    *,
    low_confidence_threshold: float = 0.6,
    high_confidence_threshold: float = 0.85,
    rejected: bool = False,
) -> str:
    """Bucket a confidence score into a stable review band."""
    if rejected:
        return "rejected"
    try:
        score = float(confidence)
    except (TypeError, ValueError):
        return "unknown"
    if score >= float(high_confidence_threshold):
        return "high"
    if score >= float(low_confidence_threshold):
        return "medium"
    if score >= 0.0:
        return "low"
    return "unknown"


def _priority_score(
    *,
    has_knowledge_conflict: bool,
    has_rejection: bool,
    has_branch_conflict: bool,
    low_confidence: bool,
    confidence: float,
    low_confidence_threshold: float,
) -> float:
    score = 0.0
    if has_knowledge_conflict:
        score += 4.0
    if has_rejection:
        score += 3.0
    if has_branch_conflict:
        score += 2.0
    if low_confidence:
        gap = max(0.0, float(low_confidence_threshold) - float(confidence))
        score += 1.0 + min(1.0, gap)
    return float(score)


def build_review_governance(
    *,
    confidence: Any,
    hybrid_rejection: Optional[Dict[str, Any]] = None,
    branch_conflicts: Optional[Dict[str, Any]] = None,
    violations: Optional[Iterable[Dict[str, Any]]] = None,
    low_confidence_threshold: float = 0.6,
    high_confidence_threshold: float = 0.85,
) -> Dict[str, Any]:
    """Build stable review governance signals from classification outputs."""
    try:
        score = float(confidence)
    except (TypeError, ValueError):
        score = 0.0

    has_rejection = _has_items(hybrid_rejection)
    has_branch_conflict = _has_items(branch_conflicts)
    violation_rows = list(violations or [])
    has_knowledge_conflict = bool(violation_rows)
    is_low_confidence = score < float(low_confidence_threshold)

    reasons: List[str] = []
    if has_rejection:
        reason = str((hybrid_rejection or {}).get("reason") or "unknown").strip() or "unknown"
        reasons.append(f"hybrid_rejected:{reason}")
    if has_knowledge_conflict:
        reasons.append("knowledge_conflict")
    if has_branch_conflict:
        reasons.append("branch_conflict")
    if is_low_confidence:
        reasons.append("low_confidence")

    needs_review = bool(reasons)
    confidence_band = derive_confidence_band(
        score,
        low_confidence_threshold=low_confidence_threshold,
        high_confidence_threshold=high_confidence_threshold,
        rejected=has_rejection,
    )
    review_priority = "none"
    if has_knowledge_conflict:
        review_priority = "critical"
    elif has_rejection or has_branch_conflict:
        review_priority = "high"
    elif is_low_confidence:
        review_priority = "medium"

    return {
        "needs_review": needs_review,
        "confidence_band": confidence_band,
        "review_priority": review_priority,
        "review_priority_score": _priority_score(
            has_knowledge_conflict=has_knowledge_conflict,
            has_rejection=has_rejection,
            has_branch_conflict=has_branch_conflict,
            low_confidence=is_low_confidence,
            confidence=score,
            low_confidence_threshold=low_confidence_threshold,
        ),
        "review_reasons": reasons,
        "review_reason_text": ";".join(reasons),
        "review_has_hybrid_rejection": has_rejection,
        "review_has_branch_conflict": has_branch_conflict,
        "review_has_knowledge_conflict": has_knowledge_conflict,
        "review_is_low_confidence": is_low_confidence,
        "review_low_confidence_threshold": float(low_confidence_threshold),
        "review_high_confidence_threshold": float(high_confidence_threshold),
    }
