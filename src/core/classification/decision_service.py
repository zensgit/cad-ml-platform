"""Centralized final classification decision service."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from src.core.classification.finalization import finalize_classification_payload

DECISION_CONTRACT_VERSION = "classification_decision.v1"

FinalizeFn = Callable[..., Dict[str, Any]]


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compact_details(payload: Mapping[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    return {
        key: payload[key]
        for key in keys
        if key in payload and payload.get(key) not in (None, "", [], {})
    }


def _first_present(payload: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in payload and payload.get(key) is not None:
            return payload.get(key)
    return None


def _prediction_evidence(
    *,
    source: str,
    prediction: Any,
    contribution: Optional[float] = None,
    kind: str = "prediction",
) -> Optional[Dict[str, Any]]:
    if not isinstance(prediction, Mapping):
        return None
    label = _clean_text(
        _first_present(
            prediction,
            ("label", "predicted_type", "part_type", "primary_label"),
        )
    )
    confidence = _safe_float(_first_present(prediction, ("confidence", "score", "probability")))
    status = _clean_text(prediction.get("status"))
    if not any((label, confidence is not None, status)):
        return None

    row: Dict[str, Any] = {
        "source": source,
        "kind": kind,
        "label": label,
        "confidence": confidence,
        "status": status,
    }
    if contribution is not None:
        row["contribution"] = contribution
    details = _compact_details(
        prediction,
        (
            "reason",
            "source",
            "provider",
            "rule_version",
            "model_version",
            "margin",
            "min_confidence",
            "min_margin",
            "passed_threshold",
            "passed_margin",
            "allowed",
            "excluded",
            "is_drawing_type",
            "is_coarse_label",
            "rejection",
        ),
    )
    if details:
        row["details"] = details
    return row


def _top_brep_hint(features_3d: Mapping[str, Any]) -> tuple[Optional[str], Optional[float]]:
    hints = features_3d.get("feature_hints")
    if not isinstance(hints, Mapping) or not hints:
        return None, None
    best_label = None
    best_score = None
    for label, score_raw in hints.items():
        score = _safe_float(score_raw)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_label = str(label)
            best_score = score
    return best_label, best_score


def _brep_evidence(features_3d: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(features_3d, Mapping) or not features_3d:
        return None
    top_label, top_score = _top_brep_hint(features_3d)
    details = _compact_details(
        features_3d,
        (
            "valid_3d",
            "faces",
            "edges",
            "vertices",
            "solids",
            "shells",
            "surface_types",
            "thin_walls_detected",
            "stock_removal_ratio",
            "embedding_dim",
        ),
    )
    if not any((top_label, top_score is not None, details)):
        return None
    row: Dict[str, Any] = {
        "source": "brep",
        "kind": "geometric_hint",
        "label": top_label,
        "confidence": top_score,
        "status": "valid" if features_3d.get("valid_3d") is True else None,
    }
    if details:
        row["details"] = details
    return row


def _structured_evidence(
    payload: Mapping[str, Any],
    *,
    features_3d: Optional[Mapping[str, Any]],
    vector_neighbors: Optional[Iterable[Mapping[str, Any]]],
    active_learning_history: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    contributions = payload.get("source_contributions")
    contribution_map = contributions if isinstance(contributions, Mapping) else {}

    baseline = {
        "label": payload.get("part_type"),
        "confidence": payload.get("confidence"),
        "source": payload.get("confidence_source"),
        "rule_version": payload.get("rule_version"),
        "model_version": payload.get("model_version"),
    }
    baseline_row = _prediction_evidence(
        source="baseline",
        prediction=baseline,
        contribution=_safe_float(contribution_map.get("baseline")),
        kind="decision",
    )
    if baseline_row:
        evidence.append(baseline_row)

    for source, key in (
        ("filename", "filename_prediction"),
        ("titleblock", "titleblock_prediction"),
        ("ocr", "ocr_prediction"),
        ("graph2d", "graph2d_prediction"),
        ("history_sequence", "history_prediction"),
        ("process", "process_prediction"),
        ("hybrid", "hybrid_decision"),
        ("part_classifier", "part_classifier_prediction"),
        ("fusion", "fusion_decision"),
    ):
        row = _prediction_evidence(
            source=source,
            prediction=payload.get(key),
            contribution=_safe_float(contribution_map.get(source)),
        )
        if row:
            evidence.append(row)

    brep_row = _brep_evidence(features_3d)
    if brep_row:
        evidence.append(brep_row)

    knowledge_checks = payload.get("knowledge_checks")
    if isinstance(knowledge_checks, list) and knowledge_checks:
        violations_raw = payload.get("violations")
        standards_candidates_raw = payload.get("standards_candidates")
        knowledge_hints_raw = payload.get("knowledge_hints")
        violations = violations_raw if isinstance(violations_raw, list) else []
        standards_candidates = (
            standards_candidates_raw
            if isinstance(standards_candidates_raw, list)
            else []
        )
        knowledge_hints = knowledge_hints_raw if isinstance(knowledge_hints_raw, list) else []

        def _metadata_tokens(rows: Iterable[Any], key: str) -> List[str]:
            tokens: List[str] = []
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                token = _clean_text(row.get(key))
                if token:
                    tokens.append(token)
            return list(dict.fromkeys(tokens))

        all_knowledge_rows = [
            *knowledge_checks,
            *violations,
            *standards_candidates,
            *knowledge_hints,
        ]
        evidence.append(
            {
                "source": "knowledge",
                "kind": "checks",
                "label": None,
                "confidence": None,
                "status": "violated" if violations else "checked",
                "details": {
                    "checks_count": len(knowledge_checks),
                    "violations_count": len(violations),
                    "standards_candidates_count": len(standards_candidates),
                    "knowledge_hints_count": len(knowledge_hints),
                    "check_categories": _metadata_tokens(knowledge_checks, "category"),
                    "standards_candidate_types": _metadata_tokens(
                        standards_candidates,
                        "type",
                    ),
                    "rule_sources": _metadata_tokens(all_knowledge_rows, "rule_source"),
                    "rule_versions": _metadata_tokens(
                        all_knowledge_rows,
                        "rule_version",
                    ),
                },
            }
        )

    neighbors = list(vector_neighbors or [])
    if neighbors:
        evidence.append(
            {
                "source": "vector_neighbors",
                "kind": "neighbors",
                "label": None,
                "confidence": None,
                "status": "available",
                "details": {"neighbor_count": len(neighbors), "top_neighbor": neighbors[0]},
            }
        )

    if isinstance(active_learning_history, Mapping) and active_learning_history:
        evidence.append(
            {
                "source": "active_learning_history",
                "kind": "history",
                "label": None,
                "confidence": None,
                "status": "available",
                "details": dict(active_learning_history),
            }
        )

    return evidence


def _fallback_flags(
    payload: Mapping[str, Any],
    *,
    features_3d: Optional[Mapping[str, Any]],
) -> List[str]:
    flags: List[str] = []
    if str(payload.get("confidence_source") or "").strip() == "rules":
        flags.append("rules_baseline")
    if str(payload.get("model_version") or "").strip().lower() in {
        "ml_error",
        "fallback",
        "unavailable",
        "model_unavailable",
    }:
        flags.append("ml_unavailable")
    if payload.get("hybrid_error"):
        flags.append("hybrid_error")
    if isinstance(payload.get("hybrid_rejection"), Mapping):
        flags.append("hybrid_rejected")

    graph2d = payload.get("graph2d_prediction")
    if isinstance(graph2d, Mapping):
        status = str(graph2d.get("status") or "").strip().lower()
        if status in {"model_unavailable", "unavailable", "error", "timeout"}:
            flags.append(f"graph2d_{status}")

    part_classifier = payload.get("part_classifier_prediction")
    if isinstance(part_classifier, Mapping):
        status = str(part_classifier.get("status") or "").strip().lower()
        if status in {"model_unavailable", "unavailable", "error", "timeout"}:
            flags.append(f"part_classifier_{status}")

    if isinstance(features_3d, Mapping) and features_3d:
        if features_3d.get("valid_3d") is False:
            flags.append("brep_invalid")

    return list(dict.fromkeys(flags))


def _decision_contract(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "fine_part_type": payload.get("fine_part_type"),
        "coarse_part_type": payload.get("coarse_part_type"),
        "confidence": payload.get("confidence"),
        "decision_source": payload.get("decision_source"),
        "branch_conflicts": payload.get("branch_conflicts") or {},
        "evidence": payload.get("evidence") or [],
        "review_reasons": payload.get("review_reasons") or [],
        "fallback_flags": payload.get("fallback_flags") or [],
        "contract_version": DECISION_CONTRACT_VERSION,
    }


class DecisionService:
    """Build the canonical final decision payload for CAD classification flows."""

    def __init__(self, *, finalize_fn: FinalizeFn = finalize_classification_payload):
        self._finalize_fn = finalize_fn

    def decide(
        self,
        payload: Optional[Dict[str, Any]],
        *,
        text_signals: Optional[Iterable[Dict[str, Any]]] = None,
        text_items: Any = None,
        geometric_features: Optional[Dict[str, Any]] = None,
        entity_counts: Optional[Dict[str, Any]] = None,
        features_3d: Optional[Mapping[str, Any]] = None,
        vector_neighbors: Optional[Iterable[Mapping[str, Any]]] = None,
        active_learning_history: Optional[Mapping[str, Any]] = None,
        low_confidence_threshold: float = 0.6,
        high_confidence_threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """Finalize a classification payload and attach the stable decision contract."""
        final_payload = self._finalize_fn(
            payload,
            text_signals=text_signals,
            text_items=text_items,
            geometric_features=geometric_features,
            entity_counts=entity_counts,
            low_confidence_threshold=low_confidence_threshold,
            high_confidence_threshold=high_confidence_threshold,
        )
        final_payload["contract_version"] = DECISION_CONTRACT_VERSION
        final_payload["decision_contract_version"] = DECISION_CONTRACT_VERSION
        final_payload["fallback_flags"] = _fallback_flags(
            final_payload,
            features_3d=features_3d,
        )
        final_payload["evidence"] = _structured_evidence(
            final_payload,
            features_3d=features_3d,
            vector_neighbors=vector_neighbors,
            active_learning_history=active_learning_history,
        )
        final_payload["decision_contract"] = _decision_contract(final_payload)
        return final_payload


__all__ = [
    "DECISION_CONTRACT_VERSION",
    "DecisionService",
]
