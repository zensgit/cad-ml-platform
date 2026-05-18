"""Forward-looking CAD ML scorecard helpers."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional

FORWARD_STATUSES = {
    "release_ready",
    "benchmark_ready_with_gap",
    "shadow_only",
    "blocked",
}
MANUFACTURING_EVIDENCE_SOURCES = (
    "dfm",
    "manufacturing_process",
    "manufacturing_cost",
    "manufacturing_decision",
)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _nested_float(payload: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return _to_float(current, default=default)


def _first_float(payload: Dict[str, Any], paths: Iterable[tuple[str, ...]]) -> float:
    for path in paths:
        value = _nested_float(payload, *path, default=-1.0)
        if value >= 0.0:
            return value
    return 0.0


def _status_rank(status: str) -> int:
    return {
        "release_ready": 3,
        "benchmark_ready_with_gap": 2,
        "shadow_only": 1,
        "blocked": 0,
    }.get(status, 0)


def _metric_status(
    *,
    sample_size: int,
    primary_score: float,
    secondary_score: Optional[float] = None,
    low_conf_rate: Optional[float] = None,
    release_threshold: float = 0.85,
    gap_threshold: float = 0.65,
    min_release_samples: int = 30,
    min_gap_samples: int = 10,
) -> str:
    if sample_size <= 0:
        return "blocked"
    if sample_size < min_gap_samples:
        return "shadow_only"
    secondary_ok = secondary_score is None or secondary_score >= max(
        0.5, gap_threshold - 0.1
    )
    low_conf_ok = low_conf_rate is None or low_conf_rate <= 0.25
    if (
        sample_size >= min_release_samples
        and primary_score >= release_threshold
        and secondary_ok
        and low_conf_ok
    ):
        return "release_ready"
    if primary_score >= gap_threshold:
        return "benchmark_ready_with_gap"
    return "shadow_only"


def _hybrid_component(summary: Dict[str, Any]) -> Dict[str, Any]:
    sample_size = _to_int(summary.get("sample_size") or summary.get("total"))
    coarse_accuracy = _first_float(
        summary,
        (
            ("coarse_accuracy", "hybrid_label", "accuracy"),
            ("coarse_scores", "hybrid_label", "accuracy"),
            ("coarse_accuracy_overall",),
        ),
    )
    macro_f1 = _first_float(
        summary,
        (
            ("coarse_macro_f1", "hybrid_label"),
            ("coarse_macro_f1_overall",),
            ("macro_f1_overall",),
        ),
    )
    exact_accuracy = _first_float(
        summary,
        (
            ("exact_accuracy", "hybrid_label", "accuracy"),
            ("exact_scores", "hybrid_label", "accuracy"),
            ("accuracy_overall",),
        ),
    )
    low_conf_rate = _first_float(
        summary,
        (
            ("confidence", "hybrid_label", "low_conf_rate"),
            ("confidence_stats", "hybrid_label", "low_conf_rate"),
            ("low_conf_rate",),
        ),
    )
    status = _metric_status(
        sample_size=sample_size,
        primary_score=coarse_accuracy,
        secondary_score=macro_f1,
        low_conf_rate=low_conf_rate if low_conf_rate > 0 else None,
        release_threshold=0.85,
        gap_threshold=0.7,
    )
    return {
        "status": status,
        "sample_size": sample_size,
        "coarse_accuracy": round(coarse_accuracy, 6),
        "coarse_macro_f1": round(macro_f1, 6),
        "exact_accuracy": round(exact_accuracy, 6),
        "low_conf_rate": round(low_conf_rate, 6),
    }


def _graph2d_component(summary: Dict[str, Any]) -> Dict[str, Any]:
    sample_size = _to_int(summary.get("sample_size") or summary.get("total"))
    blind_accuracy = _to_float(
        summary.get("blind_accuracy")
        or summary.get("accuracy")
        or summary.get("diagnose_accuracy")
    )
    low_conf_rate = _to_float(summary.get("low_conf_rate"))
    status = _metric_status(
        sample_size=sample_size,
        primary_score=blind_accuracy,
        low_conf_rate=low_conf_rate if low_conf_rate > 0 else None,
        release_threshold=0.8,
        gap_threshold=0.55,
        min_release_samples=30,
        min_gap_samples=10,
    )
    return {
        "status": status,
        "sample_size": sample_size,
        "blind_accuracy": round(blind_accuracy, 6),
        "low_conf_rate": round(low_conf_rate, 6),
    }


def _history_component(summary: Dict[str, Any]) -> Dict[str, Any]:
    sample_size = _to_int(summary.get("sample_size") or summary.get("total"))
    coarse_accuracy = _to_float(summary.get("coarse_accuracy_overall"))
    coarse_macro_f1 = _to_float(summary.get("coarse_macro_f1_overall"))
    exact_accuracy = _to_float(summary.get("accuracy_overall"))
    status = _metric_status(
        sample_size=sample_size,
        primary_score=coarse_accuracy,
        secondary_score=coarse_macro_f1,
        release_threshold=0.8,
        gap_threshold=0.6,
        min_release_samples=30,
        min_gap_samples=10,
    )
    return {
        "status": status,
        "sample_size": sample_size,
        "coarse_accuracy": round(coarse_accuracy, 6),
        "coarse_macro_f1": round(coarse_macro_f1, 6),
        "exact_accuracy": round(exact_accuracy, 6),
        "low_conf_rate": round(_to_float(summary.get("low_conf_rate")), 6),
    }


def _brep_component(summary: Dict[str, Any]) -> Dict[str, Any]:
    sample_size = _to_int(summary.get("sample_size") or summary.get("total"))
    parse_success = _to_int(
        summary.get("parse_success_count") or summary.get("ok_count")
    )
    valid_3d = _to_int(summary.get("valid_3d_count"))
    graph_valid = _to_int(
        summary.get("graph_valid_count") or summary.get("valid_graph_count")
    )
    if graph_valid <= 0:
        schema_counts = summary.get("graph_schema_version_counts") or {}
        graph_valid = _to_int(schema_counts.get("v2"))
    parse_ratio = parse_success / sample_size if sample_size else 0.0
    graph_ratio = graph_valid / sample_size if sample_size else 0.0
    if sample_size <= 0:
        status = "blocked"
    elif sample_size < 20:
        status = "shadow_only"
    elif parse_ratio >= 0.95 and graph_ratio >= 0.9:
        status = "release_ready"
    elif parse_ratio >= 0.7 or graph_ratio >= 0.5 or valid_3d > 0:
        status = "benchmark_ready_with_gap"
    else:
        status = "shadow_only"
    return {
        "status": status,
        "sample_size": sample_size,
        "parse_success_count": parse_success,
        "valid_3d_count": valid_3d,
        "graph_valid_count": graph_valid,
        "parse_success_ratio": round(parse_ratio, 6),
        "graph_valid_ratio": round(graph_ratio, 6),
        "failure_reasons": dict(summary.get("failure_reasons") or {}),
    }


def _qdrant_component(summary: Dict[str, Any]) -> Dict[str, Any]:
    backend_health = summary.get("backend_health") if isinstance(summary, dict) else {}
    health = backend_health if isinstance(backend_health, dict) else summary
    readiness = str(
        (health or {}).get("readiness") or (health or {}).get("status") or ""
    ).lower()
    indexed_ratio = _to_float((health or {}).get("indexed_ratio"), default=-1.0)
    unindexed = _to_int((health or {}).get("unindexed_vectors_count"))
    scan_truncated = bool((health or {}).get("scan_truncated"))
    if not health:
        status = "shadow_only"
    elif (
        readiness in {"ready", "green", "ok"}
        and indexed_ratio >= 0.99
        and unindexed == 0
    ):
        status = "release_ready"
    elif readiness in {"ready", "green", "ok", "partial", "indexing"}:
        status = "benchmark_ready_with_gap"
    elif scan_truncated or indexed_ratio >= 0.5:
        status = "benchmark_ready_with_gap"
    else:
        status = "blocked"
    return {
        "status": status,
        "readiness": readiness or "missing",
        "indexed_ratio": round(indexed_ratio, 6) if indexed_ratio >= 0.0 else None,
        "unindexed_vectors_count": unindexed,
        "scan_truncated": scan_truncated,
        "readiness_hints": list((health or {}).get("readiness_hints") or []),
    }


def _review_queue_component(summary: Dict[str, Any]) -> Dict[str, Any]:
    total = _to_int(summary.get("total"))
    by_priority = summary.get("by_feedback_priority") or {}
    critical = _to_int(by_priority.get("critical"))
    high = _to_int(by_priority.get("high"))
    evidence_ratio = _to_float(summary.get("records_with_evidence_ratio"), default=1.0)
    automation_ready_ratio = _to_float(summary.get("automation_ready_ratio"))
    if critical > 0:
        status = "blocked"
    elif high > 0 or evidence_ratio < 0.7:
        status = "benchmark_ready_with_gap"
    else:
        status = "release_ready"
    return {
        "status": status,
        "total": total,
        "critical_count": critical,
        "high_count": high,
        "automation_ready_ratio": round(automation_ready_ratio, 6),
        "records_with_evidence_ratio": round(evidence_ratio, 6),
    }


def _knowledge_component(summary: Dict[str, Any]) -> Dict[str, Any]:
    component = summary.get("knowledge_readiness") or summary
    grounding = (
        summary.get("knowledge_grounding") or component.get("knowledge_grounding") or {}
    )
    raw_status = str(component.get("status") or "").lower()
    ready = _to_int(component.get("ready_component_count"))
    partial = _to_int(component.get("partial_component_count"))
    missing = _to_int(component.get("missing_component_count"))
    total_reference_items = _to_int(component.get("total_reference_items"))

    grounding_sample_size = _to_int(
        grounding.get("sample_size")
        or grounding.get("total")
        or grounding.get("row_count")
    )

    def _grounding_rate(rate_key: str, count_key: str) -> float:
        explicit_rate = _to_float(grounding.get(rate_key), default=-1.0)
        if explicit_rate >= 0.0:
            return explicit_rate
        count = _to_int(grounding.get(count_key))
        return count / grounding_sample_size if grounding_sample_size else 0.0

    evidence_rate = _grounding_rate(
        "knowledge_evidence_coverage_rate",
        "records_with_knowledge_evidence",
    )
    rule_source_rate = _grounding_rate(
        "rule_source_coverage_rate",
        "records_with_rule_sources",
    )
    rule_version_rate = _grounding_rate(
        "rule_version_coverage_rate",
        "records_with_rule_versions",
    )
    rule_sources = list(grounding.get("rule_sources") or [])
    rule_versions = list(grounding.get("rule_versions") or [])
    grounding_gaps: List[str] = []
    if grounding_sample_size <= 0:
        grounding_gaps.append("knowledge_grounding_sample_missing")
    if evidence_rate < 0.8:
        grounding_gaps.append("knowledge_evidence_coverage_below_release")
    if rule_source_rate < 0.95:
        grounding_gaps.append("rule_source_coverage_below_release")
    if rule_version_rate < 0.95:
        grounding_gaps.append("rule_version_coverage_below_release")
    if not rule_sources:
        grounding_gaps.append("rule_sources_missing")
    if not rule_versions:
        grounding_gaps.append("rule_versions_missing")

    readiness_release_ready = "ready" in raw_status and missing == 0
    readiness_present = ready > 0 or partial > 0 or total_reference_items > 0
    grounding_release_ready = not grounding_gaps
    grounding_present = grounding_sample_size > 0 or bool(rule_sources or rule_versions)

    if readiness_release_ready and grounding_release_ready:
        status = "release_ready"
    elif readiness_release_ready or readiness_present or grounding_present:
        status = "benchmark_ready_with_gap"
    elif raw_status:
        status = "shadow_only"
    else:
        status = "blocked"
    return {
        "status": status,
        "source_status": raw_status or "missing",
        "ready_component_count": ready,
        "partial_component_count": partial,
        "missing_component_count": missing,
        "total_reference_items": total_reference_items,
        "focus_areas": list(component.get("focus_areas") or []),
        "grounding_sample_size": grounding_sample_size,
        "knowledge_evidence_coverage_rate": round(evidence_rate, 6),
        "rule_source_coverage_rate": round(rule_source_rate, 6),
        "rule_version_coverage_rate": round(rule_version_rate, 6),
        "rule_sources": rule_sources,
        "rule_versions": rule_versions,
        "grounding_gaps": grounding_gaps,
    }


def _manufacturing_evidence_component(summary: Dict[str, Any]) -> Dict[str, Any]:
    component = summary.get("manufacturing_evidence") or summary
    sample_size = _to_int(
        component.get("sample_size")
        or component.get("total")
        or component.get("row_count")
    )
    records_with_evidence = _to_int(
        component.get("records_with_manufacturing_evidence")
        or component.get("records_with_evidence")
        or component.get("evidence_count")
    )
    evidence_rate = _to_float(
        component.get("manufacturing_evidence_coverage_rate")
        or component.get("evidence_coverage_rate"),
        default=-1.0,
    )
    if evidence_rate < 0.0:
        evidence_rate = records_with_evidence / sample_size if sample_size else 0.0

    source_counts = dict(
        component.get("source_counts") or component.get("evidence_source_counts") or {}
    )
    explicit_source_rates = component.get("source_coverage_rates") or {}
    source_rates: Dict[str, float] = {}
    for source in MANUFACTURING_EVIDENCE_SOURCES:
        explicit_rate = _to_float(explicit_source_rates.get(source), default=-1.0)
        if explicit_rate >= 0.0:
            source_rates[source] = explicit_rate
            continue
        source_count = _to_int(source_counts.get(source))
        source_rates[source] = source_count / sample_size if sample_size else 0.0

    sources = list(component.get("sources") or source_counts.keys())
    source_set = {str(source) for source in sources}
    reviewed_sample_count = _to_int(
        component.get("reviewed_sample_count")
        or component.get("manufacturing_evidence_reviewed_sample_count")
    )
    correctness_available = _to_bool(
        component.get("source_correctness_available")
    ) or reviewed_sample_count > 0
    source_precision = _to_float(component.get("source_precision"), default=0.0)
    source_recall = _to_float(component.get("source_recall"), default=0.0)
    source_f1 = _to_float(component.get("source_f1"), default=0.0)
    source_exact_match_rate = _to_float(
        component.get("source_exact_match_rate"),
        default=0.0,
    )
    source_correctness = dict(component.get("source_correctness") or {})
    payload_quality_reviewed_sample_count = _to_int(
        component.get("payload_quality_reviewed_sample_count")
    )
    payload_quality_available = _to_bool(
        component.get("payload_quality_available")
    ) or payload_quality_reviewed_sample_count > 0
    payload_quality_accuracy = _to_float(
        component.get("payload_quality_accuracy"),
        default=0.0,
    )
    payload_detail_quality_reviewed_sample_count = _to_int(
        component.get("payload_detail_quality_reviewed_sample_count")
    )
    payload_detail_quality_available = _to_bool(
        component.get("payload_detail_quality_available")
    ) or payload_detail_quality_reviewed_sample_count > 0
    payload_detail_quality_accuracy = _to_float(
        component.get("payload_detail_quality_accuracy"),
        default=0.0,
    )
    payload_quality = dict(component.get("payload_quality") or {})
    evidence_gaps: List[str] = []
    if sample_size <= 0:
        evidence_gaps.append("manufacturing_evidence_sample_missing")
    if evidence_rate < 0.9:
        evidence_gaps.append("manufacturing_evidence_coverage_below_release")
    for source, rate in source_rates.items():
        if source not in source_set and rate <= 0.0:
            evidence_gaps.append(f"{source}_missing")
        elif rate < 0.8:
            evidence_gaps.append(f"{source}_coverage_below_release")
    if reviewed_sample_count < 30:
        evidence_gaps.append("manufacturing_evidence_correctness_review_sample_below_release")
    if not correctness_available:
        evidence_gaps.append("manufacturing_evidence_correctness_review_missing")
    elif source_precision < 0.9:
        evidence_gaps.append("manufacturing_evidence_source_precision_below_release")
    if correctness_available and source_recall < 0.9:
        evidence_gaps.append("manufacturing_evidence_source_recall_below_release")
    if payload_quality_reviewed_sample_count < 30:
        evidence_gaps.append("manufacturing_evidence_payload_quality_sample_below_release")
    if not payload_quality_available:
        evidence_gaps.append("manufacturing_evidence_payload_quality_missing")
    elif payload_quality_accuracy < 0.9:
        evidence_gaps.append("manufacturing_evidence_payload_quality_below_release")
    if payload_detail_quality_reviewed_sample_count < 30:
        evidence_gaps.append(
            "manufacturing_evidence_payload_detail_quality_sample_below_release"
        )
    if not payload_detail_quality_available:
        evidence_gaps.append("manufacturing_evidence_payload_detail_quality_missing")
    elif payload_detail_quality_accuracy < 0.9:
        evidence_gaps.append(
            "manufacturing_evidence_payload_detail_quality_below_release"
        )

    release_ready = sample_size >= 30 and not evidence_gaps
    gap_ready = sample_size >= 10 and evidence_rate >= 0.5 and len(source_set) >= 2
    if release_ready:
        status = "release_ready"
    elif gap_ready:
        status = "benchmark_ready_with_gap"
    elif sample_size > 0:
        status = "shadow_only"
    else:
        status = "blocked"

    return {
        "status": status,
        "sample_size": sample_size,
        "records_with_manufacturing_evidence": records_with_evidence,
        "manufacturing_evidence_coverage_rate": round(evidence_rate, 6),
        "required_sources": list(MANUFACTURING_EVIDENCE_SOURCES),
        "sources": sources,
        "source_coverage_rates": {
            source: round(rate, 6) for source, rate in source_rates.items()
        },
        "source_correctness_available": correctness_available,
        "reviewed_sample_count": reviewed_sample_count,
        "source_exact_match_rate": round(source_exact_match_rate, 6),
        "source_precision": round(source_precision, 6),
        "source_recall": round(source_recall, 6),
        "source_f1": round(source_f1, 6),
        "source_correctness": source_correctness,
        "payload_quality_available": payload_quality_available,
        "payload_quality_reviewed_sample_count": payload_quality_reviewed_sample_count,
        "payload_quality_accuracy": round(payload_quality_accuracy, 6),
        "payload_detail_quality_available": payload_detail_quality_available,
        "payload_detail_quality_reviewed_sample_count": (
            payload_detail_quality_reviewed_sample_count
        ),
        "payload_detail_quality_accuracy": round(payload_detail_quality_accuracy, 6),
        "payload_quality": payload_quality,
        "evidence_gaps": evidence_gaps,
    }


def _manufacturing_review_manifest_validation(summary: Dict[str, Any]) -> Dict[str, Any]:
    if not summary:
        return {}
    blocking_reasons = [
        str(item)
        for item in (summary.get("blocking_reasons") or [])
        if str(item).strip()
    ]
    return {
        "status": str(summary.get("status") or "unknown"),
        "row_count": _to_int(summary.get("row_count")),
        "min_reviewed_samples": _to_int(summary.get("min_reviewed_samples")),
        "approved_review_statuses": list(summary.get("approved_review_statuses") or []),
        "approved_review_sample_count": _to_int(
            summary.get("approved_review_sample_count")
        ),
        "unapproved_review_sample_count": _to_int(
            summary.get("unapproved_review_sample_count")
        ),
        "require_reviewer_metadata": _to_bool(
            summary.get("require_reviewer_metadata")
        ),
        "reviewer_metadata_missing_sample_count": _to_int(
            summary.get("reviewer_metadata_missing_sample_count")
        ),
        "source_reviewed_sample_count": _to_int(
            summary.get("source_reviewed_sample_count")
        ),
        "payload_reviewed_sample_count": _to_int(
            summary.get("payload_reviewed_sample_count")
        ),
        "payload_detail_reviewed_sample_count": _to_int(
            summary.get("payload_detail_reviewed_sample_count")
        ),
        "payload_expected_field_total": _to_int(
            summary.get("payload_expected_field_total")
        ),
        "payload_detail_expected_field_total": _to_int(
            summary.get("payload_detail_expected_field_total")
        ),
        "blocking_reasons": blocking_reasons,
    }


def _attach_manufacturing_review_manifest_validation(
    component: Dict[str, Any],
    validation_summary: Dict[str, Any],
) -> None:
    validation = _manufacturing_review_manifest_validation(validation_summary)
    if not validation:
        return
    component["review_manifest_validation"] = validation
    if validation.get("status") == "release_label_ready":
        return
    evidence_gaps = component.setdefault("evidence_gaps", [])
    if "manufacturing_review_manifest_validation_blocked" not in evidence_gaps:
        evidence_gaps.append("manufacturing_review_manifest_validation_blocked")
    if component.get("status") == "release_ready":
        component["status"] = "benchmark_ready_with_gap"


def _model_readiness_component(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    items = snapshot.get("items") or {}
    if not snapshot:
        return {
            "status": "blocked",
            "registry_status": "missing",
            "fallback_models": [],
            "blocking_reasons": ["model_readiness_snapshot_missing"],
        }
    fallback_models = [
        name
        for name, row in items.items()
        if isinstance(row, dict) and row.get("status") == "fallback"
    ]
    blocking_reasons = list(snapshot.get("blocking_reasons") or [])
    registry_status = str(snapshot.get("status") or "unknown")
    if blocking_reasons or snapshot.get("ok") is False:
        status = "blocked"
    elif fallback_models or snapshot.get("degraded"):
        status = "benchmark_ready_with_gap"
    else:
        status = "release_ready"
    return {
        "status": status,
        "registry_status": registry_status,
        "fallback_models": fallback_models,
        "blocking_reasons": blocking_reasons,
        "degraded_reasons": list(snapshot.get("degraded_reasons") or []),
        "model_statuses": {
            name: row.get("status")
            for name, row in items.items()
            if isinstance(row, dict)
        },
    }


def _overall_status(components: Dict[str, Dict[str, Any]]) -> str:
    model = components["model_readiness"]
    hybrid = components["hybrid_dxf"]
    if model.get("status") == "blocked":
        return "blocked"
    if hybrid.get("status") == "blocked":
        return "blocked"
    if all(row.get("status") == "release_ready" for row in components.values()):
        return "release_ready"
    if _status_rank(str(hybrid.get("status"))) >= 2:
        return "benchmark_ready_with_gap"
    return "shadow_only"


def _recommendations(components: Dict[str, Dict[str, Any]]) -> List[str]:
    items: List[str] = []
    if components["model_readiness"].get("fallback_models"):
        items.append(
            "Replace fallback-only model branches with checkpoint-backed evidence."
        )
    if components["hybrid_dxf"].get("status") in {"blocked", "shadow_only"}:
        items.append(
            "Add or refresh Hybrid DXF benchmark metrics before making release claims."
        )
    if components["graph2d"].get("status") in {"blocked", "shadow_only"}:
        items.append(
            "Keep Graph2D as shadow evidence until blind accuracy and confidence improve."
        )
    if components["history_sequence"].get("status") in {"blocked", "shadow_only"}:
        items.append("Expand history-sequence evaluation beyond smoke evidence.")
    if components["brep"].get("status") in {"blocked", "shadow_only"}:
        items.append(
            "Build a strict STEP/IGES B-Rep golden set with parse and graph metrics."
        )
    if components["qdrant"].get("status") in {"blocked", "shadow_only"}:
        items.append("Provide Qdrant/vector migration readiness evidence.")
    if components["review_queue"].get("status") == "blocked":
        items.append("Drain critical active-learning review backlog before release.")
    if components["knowledge"].get("status") in {"blocked", "shadow_only"}:
        items.append(
            "Backfill knowledge coverage before claiming manufacturing intelligence."
        )
    elif components["knowledge"].get("grounding_gaps"):
        items.append(
            "Backfill knowledge grounding rule-source and rule-version coverage before release."
        )
    if components["manufacturing_evidence"].get("status") in {"blocked", "shadow_only"}:
        items.append(
            "Add manufacturing evidence fixtures before claiming manufacturing intelligence."
        )
    elif components["manufacturing_evidence"].get("evidence_gaps"):
        review_validation = (
            components["manufacturing_evidence"].get("review_manifest_validation") or {}
        )
        if review_validation.get("status") == "blocked":
            items.append(
                "Populate the manufacturing review manifest before release claims."
            )
        else:
            items.append(
                "Raise manufacturing evidence coverage for DFM, process, cost, and summary sources."
            )
    if not items:
        items.append("Scorecard evidence supports release-readiness review.")
    return items


def build_forward_scorecard(
    *,
    title: str = "CAD ML Forward Scorecard",
    model_readiness: Dict[str, Any],
    hybrid_summary: Optional[Dict[str, Any]] = None,
    graph2d_summary: Optional[Dict[str, Any]] = None,
    history_summary: Optional[Dict[str, Any]] = None,
    brep_summary: Optional[Dict[str, Any]] = None,
    qdrant_summary: Optional[Dict[str, Any]] = None,
    review_queue_summary: Optional[Dict[str, Any]] = None,
    knowledge_summary: Optional[Dict[str, Any]] = None,
    manufacturing_summary: Optional[Dict[str, Any]] = None,
    manufacturing_review_manifest_validation: Optional[Dict[str, Any]] = None,
    artifact_paths: Optional[Dict[str, str]] = None,
    generated_at: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a forward scorecard with the canonical four release statuses."""
    components = {
        "model_readiness": _model_readiness_component(model_readiness),
        "hybrid_dxf": _hybrid_component(hybrid_summary or {}),
        "graph2d": _graph2d_component(graph2d_summary or {}),
        "history_sequence": _history_component(history_summary or {}),
        "brep": _brep_component(brep_summary or {}),
        "qdrant": _qdrant_component(qdrant_summary or {}),
        "review_queue": _review_queue_component(review_queue_summary or {}),
        "knowledge": _knowledge_component(knowledge_summary or {}),
        "manufacturing_evidence": _manufacturing_evidence_component(
            manufacturing_summary or {}
        ),
    }
    _attach_manufacturing_review_manifest_validation(
        components["manufacturing_evidence"],
        manufacturing_review_manifest_validation or {},
    )
    overall_status = _overall_status(components)
    return {
        "title": title,
        "generated_at": int(generated_at if generated_at is not None else time.time()),
        "overall_status": overall_status,
        "allowed_statuses": sorted(FORWARD_STATUSES),
        "release_claim_rule": "Release claims must cite this scorecard artifact.",
        "components": components,
        "recommendations": _recommendations(components),
        "artifacts": dict(artifact_paths or {}),
    }


def render_forward_scorecard_markdown(payload: Dict[str, Any], title: str) -> str:
    """Render the forward scorecard as Markdown."""
    lines = [
        f"# {title}",
        "",
        "## Overview",
        "",
        f"- `overall_status`: `{payload.get('overall_status', 'unknown')}`",
        f"- `generated_at`: `{payload.get('generated_at', '')}`",
        f"- `release_claim_rule`: {payload.get('release_claim_rule', '')}",
        "",
        "## Components",
        "",
        "| Component | Status | Key Evidence |",
        "| --- | --- | --- |",
    ]
    components = payload.get("components") or {}
    for name, row in components.items():
        if not isinstance(row, dict):
            continue
        evidence = _component_evidence(name, row)
        lines.append(f"| `{name}` | `{row.get('status', 'unknown')}` | {evidence} |")
    lines.extend(["", "## Recommendations", ""])
    for item in payload.get("recommendations") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Artifacts", ""])
    artifacts = payload.get("artifacts") or {}
    if artifacts:
        for name, path in artifacts.items():
            lines.append(f"- `{name}`: `{path}`")
    else:
        lines.append("- No external benchmark artifact paths were provided.")
    lines.append("")
    return "\n".join(lines)


def _component_evidence(name: str, row: Dict[str, Any]) -> str:
    if name == "model_readiness":
        fallbacks = ",".join(row.get("fallback_models") or [])
        return f"registry={row.get('registry_status')}; fallbacks={fallbacks or 'none'}"
    if name == "hybrid_dxf":
        return (
            f"n={row.get('sample_size')}; coarse={row.get('coarse_accuracy')}; "
            f"f1={row.get('coarse_macro_f1')}"
        )
    if name == "graph2d":
        return (
            f"n={row.get('sample_size')}; blind={row.get('blind_accuracy')}; "
            f"low_conf={row.get('low_conf_rate')}"
        )
    if name == "history_sequence":
        return (
            f"n={row.get('sample_size')}; coarse={row.get('coarse_accuracy')}; "
            f"f1={row.get('coarse_macro_f1')}"
        )
    if name == "brep":
        return (
            f"n={row.get('sample_size')}; parse={row.get('parse_success_ratio')}; "
            f"graph={row.get('graph_valid_ratio')}"
        )
    if name == "qdrant":
        return (
            f"readiness={row.get('readiness')}; indexed_ratio={row.get('indexed_ratio')}; "
            f"unindexed={row.get('unindexed_vectors_count')}"
        )
    if name == "review_queue":
        return (
            f"total={row.get('total')}; critical={row.get('critical_count')}; "
            f"high={row.get('high_count')}"
        )
    if name == "knowledge":
        return (
            f"source={row.get('source_status')}; ready={row.get('ready_component_count')}; "
            f"refs={row.get('total_reference_items')}; "
            f"rule_src={row.get('rule_source_coverage_rate')}; "
            f"rule_ver={row.get('rule_version_coverage_rate')}"
        )
    if name == "manufacturing_evidence":
        return (
            f"n={row.get('sample_size')}; "
            f"coverage={row.get('manufacturing_evidence_coverage_rate')}; "
            f"reviewed={row.get('reviewed_sample_count')}; "
            f"precision={row.get('source_precision')}; "
            f"recall={row.get('source_recall')}; "
            f"payload_acc={row.get('payload_quality_accuracy')}; "
            f"detail_acc={row.get('payload_detail_quality_accuracy')}; "
            f"review_manifest="
            f"{(row.get('review_manifest_validation') or {}).get('status', 'not_provided')}; "
            f"sources={','.join(row.get('sources') or []) or 'none'}"
        )
    return "none"


__all__ = [
    "FORWARD_STATUSES",
    "build_forward_scorecard",
    "render_forward_scorecard_markdown",
]
