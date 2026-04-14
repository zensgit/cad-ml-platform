"""Helpers for finalizing analysis classification payloads."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from src.core.classification.coarse_labels import (
    labels_conflict,
    normalize_coarse_label,
)
from src.core.classification.decision_contract import (
    build_classification_decision_contract,
)
from src.core.classification.review_governance import build_review_governance
from src.core.knowledge.analysis_summary import build_knowledge_summary


def _coarse_label_from_prediction(prediction: Any) -> Optional[str]:
    if not isinstance(prediction, dict):
        return None
    return normalize_coarse_label(prediction.get("label"))


def _list_text_items(text_items: Any) -> Optional[list[Any]]:
    return text_items if isinstance(text_items, list) else None


def finalize_classification_payload(
    payload: Optional[Dict[str, Any]],
    *,
    text_signals: Optional[Iterable[Dict[str, Any]]] = None,
    text_items: Any = None,
    geometric_features: Optional[Dict[str, Any]] = None,
    entity_counts: Optional[Dict[str, Any]] = None,
    low_confidence_threshold: float = 0.6,
    high_confidence_threshold: float = 0.85,
) -> Dict[str, Any]:
    """Finalize stable classification fields without changing decision order."""
    cls_payload = dict(payload or {})

    for key, value in build_classification_decision_contract(cls_payload).items():
        if value is not None:
            cls_payload[key] = value

    cls_payload["coarse_fine_part_type"] = normalize_coarse_label(
        cls_payload.get("fine_part_type")
    )
    cls_payload["coarse_hybrid_label"] = _coarse_label_from_prediction(
        cls_payload.get("hybrid_decision")
    )
    cls_payload["coarse_graph2d_label"] = _coarse_label_from_prediction(
        cls_payload.get("graph2d_prediction")
    )
    cls_payload["coarse_filename_label"] = _coarse_label_from_prediction(
        cls_payload.get("filename_prediction")
    )
    cls_payload["coarse_titleblock_label"] = _coarse_label_from_prediction(
        cls_payload.get("titleblock_prediction")
    )
    cls_payload["coarse_history_label"] = _coarse_label_from_prediction(
        cls_payload.get("history_prediction")
    )
    cls_payload["coarse_process_label"] = _coarse_label_from_prediction(
        cls_payload.get("process_prediction")
    )
    cls_payload["coarse_part_family"] = normalize_coarse_label(
        cls_payload.get("part_family")
    )

    branch_conflicts = {
        "hybrid_vs_graph2d": labels_conflict(
            cls_payload.get("coarse_hybrid_label"),
            cls_payload.get("coarse_graph2d_label"),
        ),
        "filename_vs_graph2d": labels_conflict(
            cls_payload.get("coarse_filename_label"),
            cls_payload.get("coarse_graph2d_label"),
        ),
        "titleblock_vs_graph2d": labels_conflict(
            cls_payload.get("coarse_titleblock_label"),
            cls_payload.get("coarse_graph2d_label"),
        ),
        "history_vs_final": labels_conflict(
            cls_payload.get("coarse_history_label"),
            cls_payload.get("coarse_part_type"),
        ),
    }
    cls_payload["branch_conflicts"] = {
        key: value for key, value in branch_conflicts.items() if value
    }
    cls_payload["has_branch_conflict"] = bool(cls_payload["branch_conflicts"])

    knowledge_payload = build_knowledge_summary(
        text_signals=text_signals,
        text_items=_list_text_items(text_items),
        geometric_features=geometric_features,
        entity_counts=entity_counts,
        fine_part_type=cls_payload.get("fine_part_type"),
        coarse_part_type=cls_payload.get("coarse_part_type"),
    )
    cls_payload["knowledge_checks"] = knowledge_payload.get("knowledge_checks", [])
    cls_payload["violations"] = knowledge_payload.get("violations", [])
    cls_payload["standards_candidates"] = knowledge_payload.get(
        "standards_candidates", []
    )
    cls_payload["knowledge_hints"] = knowledge_payload.get("knowledge_hints", [])

    cls_payload.update(
        build_review_governance(
            confidence=cls_payload.get("confidence", 0.0),
            hybrid_rejection=cls_payload.get("hybrid_rejection"),
            branch_conflicts=cls_payload.get("branch_conflicts"),
            violations=cls_payload.get("violations"),
            low_confidence_threshold=low_confidence_threshold,
            high_confidence_threshold=high_confidence_threshold,
        )
    )

    for key, value in build_classification_decision_contract(cls_payload).items():
        if value is not None:
            cls_payload[key] = value
    return cls_payload


__all__ = ["finalize_classification_payload"]
