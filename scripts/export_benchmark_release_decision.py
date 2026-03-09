#!/usr/bin/env python3
"""Export a benchmark release decision summary."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CRITICAL_STATUSES = {
    "blocked",
    "critical_backlog",
    "missing",
}

REVIEW_STATUSES = {
    "attention_required",
    "gap_detected",
    "knowledge_application_partial",
    "managed_backlog",
    "partial_coverage",
    "review_heavy",
    "weak_coverage",
    "feedback_collected",
    "passive_feedback_only",
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _load_json(path_text: str) -> Dict[str, Any]:
    path = Path(path_text).expanduser()
    if not path.exists():
        raise SystemExit(f"JSON input not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected object JSON in {path}")
    return payload


def _maybe_load_json(path_text: str) -> Dict[str, Any]:
    if not str(path_text or "").strip():
        return {}
    return _load_json(path_text)


def _write_output(path_text: str, content: str) -> None:
    output_path = Path(path_text).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def _compact(items: Iterable[Any], *, limit: int = 5) -> List[str]:
    out: List[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _artifact_row(name: str, path_text: str) -> Dict[str, Any]:
    path_value = _text(path_text)
    return {
        "name": name,
        "path": path_value,
        "present": bool(path_value),
    }


def _component_statuses(
    benchmark_scorecard: Dict[str, Any],
    benchmark_operational_summary: Dict[str, Any],
    benchmark_artifact_bundle: Dict[str, Any],
    benchmark_companion_summary: Dict[str, Any],
    benchmark_knowledge_readiness: Dict[str, Any],
    benchmark_knowledge_drift: Dict[str, Any],
    benchmark_engineering_signals: Dict[str, Any],
    benchmark_realdata_signals: Dict[str, Any],
    benchmark_operator_adoption: Dict[str, Any],
    benchmark_realdata_scorecard: Dict[str, Any] | None = None,
    benchmark_knowledge_application: Dict[str, Any] | None = None,
    benchmark_knowledge_realdata_correlation: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_matrix: Dict[str, Any] | None = None,
    benchmark_knowledge_outcome_correlation: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    benchmark_realdata_scorecard = benchmark_realdata_scorecard or {}
    benchmark_knowledge_application = benchmark_knowledge_application or {}
    benchmark_knowledge_realdata_correlation = (
        benchmark_knowledge_realdata_correlation or {}
    )
    benchmark_knowledge_domain_matrix = benchmark_knowledge_domain_matrix or {}
    benchmark_knowledge_outcome_correlation = (
        benchmark_knowledge_outcome_correlation or {}
    )
    scorecard_components = benchmark_scorecard.get("components") or {}
    operational_components = benchmark_operational_summary.get("component_statuses") or {}
    bundle_components = benchmark_artifact_bundle.get("component_statuses") or {}
    companion_components = benchmark_companion_summary.get("component_statuses") or {}
    knowledge_component = (
        benchmark_knowledge_readiness.get("knowledge_readiness")
        or benchmark_knowledge_readiness
        or {}
    )
    engineering_component = (
        benchmark_engineering_signals.get("engineering_signals")
        or benchmark_engineering_signals
        or {}
    )
    realdata_component = (
        benchmark_realdata_signals.get("realdata_signals")
        or benchmark_realdata_signals
        or {}
    )
    realdata_scorecard_component = (
        benchmark_realdata_scorecard.get("realdata_scorecard")
        or benchmark_realdata_scorecard
        or {}
    )
    knowledge_application_component = (
        benchmark_knowledge_application.get("knowledge_application")
        or benchmark_knowledge_application
        or {}
    )
    knowledge_realdata_correlation_component = (
        benchmark_knowledge_realdata_correlation.get("knowledge_realdata_correlation")
        or benchmark_knowledge_realdata_correlation
        or {}
    )
    knowledge_domain_matrix_component = (
        benchmark_knowledge_domain_matrix.get("knowledge_domain_matrix")
        or benchmark_knowledge_domain_matrix
        or {}
    )
    knowledge_outcome_correlation_component = (
        benchmark_knowledge_outcome_correlation.get("knowledge_outcome_correlation")
        or benchmark_knowledge_outcome_correlation
        or {}
    )
    drift_component = (
        benchmark_knowledge_drift.get("knowledge_drift")
        or benchmark_knowledge_drift
        or {}
    )

    def pick(name: str) -> str:
        if isinstance(companion_components, dict) and companion_components.get(name):
            return str(companion_components.get(name))
        if isinstance(bundle_components, dict) and bundle_components.get(name):
            return str(bundle_components.get(name))
        if isinstance(operational_components, dict) and operational_components.get(name):
            return str(operational_components.get(name))
        scorecard_row = scorecard_components.get(name) or {}
        if isinstance(scorecard_row, dict) and scorecard_row.get("status"):
            return str(scorecard_row.get("status"))
        return "unknown"

    return {
        "hybrid": pick("hybrid"),
        "history_sequence": pick("history_sequence"),
        "brep": pick("brep"),
        "migration_governance": pick("migration_governance"),
        "feedback_flywheel": pick("feedback_flywheel"),
        "assistant_explainability": pick("assistant_explainability"),
        "review_queue": pick("review_queue"),
        "ocr_review": pick("ocr_review"),
        "qdrant_backend": pick("qdrant_backend"),
        "knowledge_readiness": str(
            companion_components.get("knowledge_readiness")
            or bundle_components.get("knowledge_readiness")
            or knowledge_component.get("status")
            or (scorecard_components.get("knowledge_readiness") or {}).get("status")
            or "unknown"
        ),
        "knowledge_drift": str(
            companion_components.get("knowledge_drift")
            or bundle_components.get("knowledge_drift")
            or drift_component.get("status")
            or (scorecard_components.get("knowledge_drift") or {}).get("status")
            or "unknown"
        ),
        "engineering_signals": str(
            companion_components.get("engineering_signals")
            or bundle_components.get("engineering_signals")
            or engineering_component.get("status")
            or (scorecard_components.get("engineering_signals") or {}).get("status")
            or "unknown"
        ),
        "realdata_signals": str(
            companion_components.get("realdata_signals")
            or bundle_components.get("realdata_signals")
            or realdata_component.get("status")
            or (scorecard_components.get("realdata_signals") or {}).get("status")
            or "unknown"
        ),
        "realdata_scorecard": str(
            companion_components.get("realdata_scorecard")
            or bundle_components.get("realdata_scorecard")
            or realdata_scorecard_component.get("status")
            or (scorecard_components.get("realdata_scorecard") or {}).get("status")
            or "unknown"
        ),
        "operator_adoption": str(
            companion_components.get("operator_adoption")
            or bundle_components.get("operator_adoption")
            or benchmark_operator_adoption.get("adoption_readiness")
            or "unknown"
        ),
        "knowledge_application": str(
            companion_components.get("knowledge_application")
            or bundle_components.get("knowledge_application")
            or knowledge_application_component.get("status")
            or (scorecard_components.get("knowledge_application") or {}).get("status")
            or "unknown"
        ),
        "knowledge_realdata_correlation": str(
            companion_components.get("knowledge_realdata_correlation")
            or bundle_components.get("knowledge_realdata_correlation")
            or knowledge_realdata_correlation_component.get("status")
            or (
                scorecard_components.get("knowledge_realdata_correlation") or {}
            ).get("status")
            or "unknown"
        ),
        "knowledge_domain_matrix": str(
            companion_components.get("knowledge_domain_matrix")
            or bundle_components.get("knowledge_domain_matrix")
            or knowledge_domain_matrix_component.get("status")
            or (scorecard_components.get("knowledge_domain_matrix") or {}).get("status")
            or "unknown"
        ),
        "knowledge_outcome_correlation": str(
            companion_components.get("knowledge_outcome_correlation")
            or bundle_components.get("knowledge_outcome_correlation")
            or knowledge_outcome_correlation_component.get("status")
            or (
                scorecard_components.get("knowledge_outcome_correlation") or {}
            ).get("status")
            or "unknown"
        ),
    }


def _pick_signal_source(
    benchmark_companion_summary: Dict[str, Any],
    benchmark_artifact_bundle: Dict[str, Any],
    benchmark_operational_summary: Dict[str, Any],
    benchmark_scorecard: Dict[str, Any],
) -> str:
    if benchmark_companion_summary:
        return "benchmark_companion_summary"
    if benchmark_artifact_bundle:
        return "benchmark_artifact_bundle"
    if benchmark_operational_summary:
        return "benchmark_operational_summary"
    if benchmark_scorecard:
        return "benchmark_scorecard"
    return "none"


def _decision(
    component_statuses: Dict[str, str], blockers: List[str], review_signals: List[str]
) -> Tuple[str, bool]:
    decision_statuses = {
        name: status
        for name, status in component_statuses.items()
        if name not in {"operator_adoption", "knowledge_drift"}
    }
    if blockers:
        return "blocked", False
    if any(status in CRITICAL_STATUSES for status in decision_statuses.values()):
        return "blocked", False
    if review_signals or any(status in REVIEW_STATUSES for status in decision_statuses.values()):
        return "review_required", False
    return "ready", True


def _engineering_review_signals(
    benchmark_engineering_signals: Dict[str, Any],
    component_statuses: Dict[str, str],
) -> List[str]:
    status = str(component_statuses.get("engineering_signals") or "").strip()
    if status in {"engineering_semantics_ready", "unknown", ""}:
        return []
    return _compact(benchmark_engineering_signals.get("recommendations") or [], limit=6)


def _realdata_review_signals(
    benchmark_realdata_signals: Dict[str, Any],
    component_statuses: Dict[str, str],
) -> List[str]:
    status = str(component_statuses.get("realdata_signals") or "").strip()
    if status in {"realdata_foundation_ready", "unknown", ""}:
        return []
    return _compact(benchmark_realdata_signals.get("recommendations") or [], limit=6)


def _realdata_scorecard_review_signals(
    benchmark_realdata_scorecard: Dict[str, Any],
    component_statuses: Dict[str, str],
) -> List[str]:
    status = str(component_statuses.get("realdata_scorecard") or "").strip()
    if status in {"realdata_scorecard_ready", "unknown", ""}:
        return []
    return _compact(benchmark_realdata_scorecard.get("recommendations") or [], limit=6)


def _knowledge_review_signals(
    benchmark_knowledge_readiness: Dict[str, Any],
    component_statuses: Dict[str, str],
) -> List[str]:
    status = str(component_statuses.get("knowledge_readiness") or "").strip()
    if status in {"knowledge_foundation_ready", "unknown", ""}:
        return []
    return _compact(benchmark_knowledge_readiness.get("recommendations") or [], limit=6)


def _knowledge_application_review_signals(
    benchmark_knowledge_application: Dict[str, Any],
    component_statuses: Dict[str, str],
) -> List[str]:
    status = str(component_statuses.get("knowledge_application") or "").strip()
    if status in {"knowledge_application_ready", "unknown", ""}:
        return []
    return _compact(benchmark_knowledge_application.get("recommendations") or [], limit=6)


def _knowledge_realdata_correlation_review_signals(
    benchmark_knowledge_realdata_correlation: Dict[str, Any],
    component_statuses: Dict[str, str],
) -> List[str]:
    status = str(component_statuses.get("knowledge_realdata_correlation") or "").strip()
    if status in {"knowledge_realdata_ready", "unknown", ""}:
        return []
    return _compact(
        benchmark_knowledge_realdata_correlation.get("recommendations") or [],
        limit=6,
    )


def _knowledge_domain_matrix_review_signals(
    benchmark_knowledge_domain_matrix: Dict[str, Any],
    component_statuses: Dict[str, str],
) -> List[str]:
    status = str(component_statuses.get("knowledge_domain_matrix") or "").strip()
    if status in {"knowledge_domain_matrix_ready", "unknown", ""}:
        return []
    return _compact(
        benchmark_knowledge_domain_matrix.get("recommendations") or [],
        limit=6,
    )


def _knowledge_drift_summary(status: str, counts: Dict[str, int]) -> str:
    if status == "baseline_missing":
        return "Knowledge drift baseline is missing."
    if status == "regressed":
        return (
            "Knowledge drift regressed in "
            f"{counts['regressions']} component(s) against the previous baseline."
        )
    if status == "mixed":
        return (
            "Knowledge drift is mixed with "
            f"{counts['regressions']} regression(s) and "
            f"{counts['improvements']} improvement(s)."
        )
    if status == "improved":
        return (
            "Knowledge drift improved in "
            f"{counts['improvements']} component(s) against the previous baseline."
        )
    if counts["new_focus_areas"]:
        return (
            "Knowledge drift is stable but introduced "
            f"{counts['new_focus_areas']} new focus area(s)."
        )
    return "Knowledge drift is stable against the previous baseline."


def _knowledge_drift_payload(
    benchmark_knowledge_drift: Dict[str, Any],
) -> Dict[str, Any]:
    component = (
        benchmark_knowledge_drift.get("knowledge_drift")
        or benchmark_knowledge_drift
        or {}
    )
    regressions = _compact(component.get("regressions") or [], limit=10)
    improvements = _compact(component.get("improvements") or [], limit=10)
    resolved_focus_areas = _compact(
        component.get("resolved_focus_areas") or [],
        limit=10,
    )
    new_focus_areas = _compact(component.get("new_focus_areas") or [], limit=10)
    domain_regressions = _compact(component.get("domain_regressions") or [], limit=10)
    domain_improvements = _compact(component.get("domain_improvements") or [], limit=10)
    resolved_priority_domains = _compact(
        component.get("resolved_priority_domains") or [],
        limit=10,
    )
    new_priority_domains = _compact(component.get("new_priority_domains") or [], limit=10)
    counts = {
        "regressions": len(regressions),
        "improvements": len(improvements),
        "new_focus_areas": len(new_focus_areas),
        "resolved_focus_areas": len(resolved_focus_areas),
        "domain_regressions": len(domain_regressions),
        "domain_improvements": len(domain_improvements),
        "resolved_priority_domains": len(resolved_priority_domains),
        "new_priority_domains": len(new_priority_domains),
    }
    status = _text(component.get("status")) or (
        "provided" if benchmark_knowledge_drift else "unknown"
    )
    summary = _text(benchmark_knowledge_drift.get("summary")) or _knowledge_drift_summary(
        status,
        counts,
    )
    return {
        "status": status,
        "summary": summary,
        "regressions": regressions,
        "improvements": improvements,
        "new_focus_areas": new_focus_areas,
        "resolved_focus_areas": resolved_focus_areas,
        "domain_regressions": domain_regressions,
        "domain_improvements": domain_improvements,
        "resolved_priority_domains": resolved_priority_domains,
        "new_priority_domains": new_priority_domains,
        "counts": counts,
        "recommendations": _compact(
            benchmark_knowledge_drift.get("recommendations") or [],
            limit=6,
        ),
        "has_drift": bool(benchmark_knowledge_drift),
    }


def _knowledge_drift_review_signals(knowledge_drift: Dict[str, Any]) -> List[str]:
    status = _text(knowledge_drift.get("status"))
    if status in {"", "unknown", "stable"} and not knowledge_drift.get(
        "new_focus_areas"
    ):
        return []
    if status == "improved" and not knowledge_drift.get("new_focus_areas"):
        return []
    recommendations = _compact(knowledge_drift.get("recommendations") or [], limit=6)
    if recommendations:
        return recommendations
    summary = _text(knowledge_drift.get("summary"))
    return [summary] if summary else []
def _operator_adoption_knowledge_drift(
    benchmark_operator_adoption: Dict[str, Any],
) -> Dict[str, Any]:
    drift = benchmark_operator_adoption.get("knowledge_drift") or {}
    return {
        "status": str(
            benchmark_operator_adoption.get("knowledge_drift_status")
            or drift.get("status")
            or "unknown"
        ),
        "summary": str(
            benchmark_operator_adoption.get("knowledge_drift_summary")
            or drift.get("summary")
            or "none"
        ),
        "recommendations": _compact(
            drift.get("recommendations")
            or benchmark_operator_adoption.get("recommended_actions")
            or [],
            limit=6,
        ),
    }
def build_release_decision(
    *,
    title: str,
    benchmark_scorecard: Dict[str, Any],
    benchmark_operational_summary: Dict[str, Any],
    benchmark_artifact_bundle: Dict[str, Any],
    benchmark_companion_summary: Dict[str, Any],
    benchmark_knowledge_readiness: Dict[str, Any],
    benchmark_knowledge_drift: Dict[str, Any],
    benchmark_engineering_signals: Dict[str, Any],
    benchmark_realdata_signals: Dict[str, Any],
    benchmark_operator_adoption: Dict[str, Any],
    artifact_paths: Dict[str, str],
    benchmark_realdata_scorecard: Dict[str, Any] | None = None,
    benchmark_knowledge_application: Dict[str, Any] | None = None,
    benchmark_knowledge_realdata_correlation: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_matrix: Dict[str, Any] | None = None,
    benchmark_knowledge_outcome_correlation: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    benchmark_realdata_scorecard = benchmark_realdata_scorecard or {}
    benchmark_knowledge_application = benchmark_knowledge_application or {}
    benchmark_knowledge_realdata_correlation = (
        benchmark_knowledge_realdata_correlation or {}
    )
    benchmark_knowledge_domain_matrix = benchmark_knowledge_domain_matrix or {}
    benchmark_knowledge_outcome_correlation = (
        benchmark_knowledge_outcome_correlation or {}
    )
    component_statuses = _component_statuses(
        benchmark_scorecard,
        benchmark_operational_summary,
        benchmark_artifact_bundle,
        benchmark_companion_summary,
        benchmark_knowledge_readiness,
        benchmark_knowledge_drift,
        benchmark_engineering_signals,
        benchmark_realdata_signals,
        benchmark_operator_adoption,
        benchmark_realdata_scorecard,
        benchmark_knowledge_application,
        benchmark_knowledge_realdata_correlation,
        benchmark_knowledge_domain_matrix,
        benchmark_knowledge_outcome_correlation,
    )
    knowledge_drift = _knowledge_drift_payload(benchmark_knowledge_drift)
    realdata_component = (
        benchmark_realdata_signals.get("realdata_signals")
        or benchmark_realdata_signals
        or {}
    )
    realdata_scorecard_component = (
        benchmark_realdata_scorecard.get("realdata_scorecard")
        or benchmark_realdata_scorecard
        or {}
    )
    knowledge_application_component = (
        benchmark_knowledge_application.get("knowledge_application")
        or benchmark_knowledge_application
        or {}
    )
    knowledge_realdata_correlation_component = (
        benchmark_knowledge_realdata_correlation.get("knowledge_realdata_correlation")
        or benchmark_knowledge_realdata_correlation
        or {}
    )
    knowledge_domain_matrix_component = (
        benchmark_knowledge_domain_matrix.get("knowledge_domain_matrix")
        or benchmark_knowledge_domain_matrix
        or {}
    )
    knowledge_outcome_correlation_component = (
        benchmark_knowledge_outcome_correlation.get("knowledge_outcome_correlation")
        or benchmark_knowledge_outcome_correlation
        or {}
    )
    knowledge_focus_areas = list(
        (
            benchmark_knowledge_readiness.get("knowledge_readiness")
            or benchmark_knowledge_readiness
            or {}
        ).get("focus_areas_detail")
        or []
    )
    operator_adoption_knowledge_drift = _operator_adoption_knowledge_drift(
        benchmark_operator_adoption
    )
    knowledge_root = (
        benchmark_knowledge_readiness.get("knowledge_readiness")
        or benchmark_knowledge_readiness
        or {}
    )
    knowledge_domains = knowledge_root.get("domains") or {}
    knowledge_domain_focus_areas = list(
        knowledge_root.get("domain_focus_areas") or []
    )
    knowledge_priority_domains = list(knowledge_root.get("priority_domains") or [])
    knowledge_application_focus_areas = list(
        knowledge_application_component.get("focus_areas_detail") or []
    )
    knowledge_application_domains = knowledge_application_component.get("domains") or {}
    knowledge_application_priority_domains = list(
        knowledge_application_component.get("priority_domains") or []
    )
    knowledge_realdata_correlation_domains = (
        knowledge_realdata_correlation_component.get("domains") or {}
    )
    knowledge_realdata_correlation_priority_domains = list(
        knowledge_realdata_correlation_component.get("priority_domains") or []
    )
    knowledge_domain_matrix_domains = knowledge_domain_matrix_component.get("domains") or {}
    knowledge_domain_matrix_priority_domains = list(
        knowledge_domain_matrix_component.get("priority_domains") or []
    )
    knowledge_outcome_correlation_domains = (
        knowledge_outcome_correlation_component.get("domains") or {}
    )
    knowledge_outcome_correlation_priority_domains = list(
        knowledge_outcome_correlation_component.get("priority_domains") or []
    )
    blockers = _compact(
        benchmark_companion_summary.get("blockers")
        or benchmark_artifact_bundle.get("blockers")
        or benchmark_operational_summary.get("blockers")
        or [],
        limit=6,
    )
    if not blockers:
        blockers = _compact(
            benchmark_operator_adoption.get("blocking_signals") or [],
            limit=6,
        )
    review_signals = _compact(
        benchmark_companion_summary.get("recommended_actions")
        or benchmark_artifact_bundle.get("recommendations")
        or benchmark_operational_summary.get("recommendations")
        or benchmark_scorecard.get("recommendations")
        or [],
        limit=6,
    )
    review_signals.extend(
        item
        for item in _engineering_review_signals(
            benchmark_engineering_signals,
            component_statuses,
        )
        if item not in review_signals
    )
    review_signals.extend(
        item
        for item in _realdata_review_signals(
            benchmark_realdata_signals,
            component_statuses,
        )
        if item not in review_signals
    )
    review_signals.extend(
        item
        for item in _realdata_scorecard_review_signals(
            benchmark_realdata_scorecard,
            component_statuses,
        )
        if item not in review_signals
    )
    review_signals.extend(
        item
        for item in _knowledge_review_signals(
            benchmark_knowledge_readiness,
            component_statuses,
        )
        if item not in review_signals
    )
    review_signals.extend(
        item
        for item in _knowledge_application_review_signals(
            benchmark_knowledge_application,
            component_statuses,
        )
        if item not in review_signals
    )
    review_signals.extend(
        item
        for item in _knowledge_realdata_correlation_review_signals(
            benchmark_knowledge_realdata_correlation,
            component_statuses,
        )
        if item not in review_signals
    )
    review_signals.extend(
        item
        for item in _knowledge_domain_matrix_review_signals(
            benchmark_knowledge_domain_matrix,
            component_statuses,
        )
        if item not in review_signals
    )
    knowledge_outcome_correlation_status = (
        str(knowledge_outcome_correlation_component.get("status") or "unknown").strip()
        or "unknown"
    )
    if knowledge_outcome_correlation_status not in {
        "",
        "unknown",
        "knowledge_outcome_correlation_ready",
    }:
        for item in _compact(
            benchmark_knowledge_outcome_correlation.get("recommendations") or [],
            limit=6,
        ):
            if item not in review_signals:
                review_signals.append(item)
    review_signals.extend(
        item
        for item in _knowledge_drift_review_signals(knowledge_drift)
        if item not in review_signals
    )
    operator_adoption_status = str(
        benchmark_operator_adoption.get("adoption_readiness") or ""
    ).strip()
    if not review_signals and operator_adoption_status not in {
        "",
        "operator_ready",
        "ready",
        "unknown",
    }:
        review_signals.extend(
            item
            for item in _compact(
                benchmark_operator_adoption.get("recommended_actions")
                or benchmark_operator_adoption.get("review_signals")
                or [],
                limit=6,
            )
            if item not in review_signals
        )
    if operator_adoption_knowledge_drift.get("status") == "regressed":
        for item in operator_adoption_knowledge_drift.get("recommendations") or []:
            if item not in review_signals:
                review_signals.append(item)
    release_status, automation_ready = _decision(
        component_statuses,
        blockers,
        review_signals,
    )
    primary_signal_source = _pick_signal_source(
        benchmark_companion_summary,
        benchmark_artifact_bundle,
        benchmark_operational_summary,
        benchmark_scorecard,
    )
    return {
        "title": title,
        "generated_at": int(time.time()),
        "release_status": release_status,
        "automation_ready": automation_ready,
        "primary_signal_source": primary_signal_source,
        "component_statuses": component_statuses,
        "knowledge_focus_areas": knowledge_focus_areas,
        "knowledge_drift_status": knowledge_drift["status"],
        "knowledge_drift_summary": knowledge_drift["summary"],
        "knowledge_drift": knowledge_drift,
        "knowledge_drift_domain_regressions": list(
            knowledge_drift.get("domain_regressions") or []
        ),
        "knowledge_drift_domain_improvements": list(
            knowledge_drift.get("domain_improvements") or []
        ),
        "knowledge_drift_resolved_priority_domains": list(
            knowledge_drift.get("resolved_priority_domains") or []
        ),
        "knowledge_drift_new_priority_domains": list(
            knowledge_drift.get("new_priority_domains") or []
        ),
        "realdata_status": str(realdata_component.get("status") or "unknown"),
        "realdata_signals": realdata_component,
        "realdata_recommendations": _compact(
            benchmark_realdata_signals.get("recommendations") or [],
            limit=6,
        ),
        "realdata_scorecard_status": (
            realdata_scorecard_component.get("status") or "unknown"
        ),
        "realdata_scorecard": realdata_scorecard_component,
        "realdata_scorecard_recommendations": _compact(
            benchmark_realdata_scorecard.get("recommendations") or [],
            limit=6,
        ),
        "operator_adoption_knowledge_drift": operator_adoption_knowledge_drift,
        "knowledge_domains": knowledge_domains,
        "knowledge_domain_focus_areas": knowledge_domain_focus_areas,
        "knowledge_priority_domains": knowledge_priority_domains,
        "knowledge_application_status": knowledge_application_component.get("status")
        or "unknown",
        "knowledge_application": knowledge_application_component,
        "knowledge_application_focus_areas": knowledge_application_focus_areas,
        "knowledge_application_domains": knowledge_application_domains,
        "knowledge_application_priority_domains": knowledge_application_priority_domains,
        "knowledge_application_recommendations": _compact(
            benchmark_knowledge_application.get("recommendations") or [],
            limit=6,
        ),
        "knowledge_realdata_correlation_status": (
            knowledge_realdata_correlation_component.get("status") or "unknown"
        ),
        "knowledge_realdata_correlation": knowledge_realdata_correlation_component,
        "knowledge_realdata_correlation_domains": knowledge_realdata_correlation_domains,
        "knowledge_realdata_correlation_priority_domains": (
            knowledge_realdata_correlation_priority_domains
        ),
        "knowledge_realdata_correlation_recommendations": _compact(
            benchmark_knowledge_realdata_correlation.get("recommendations") or [],
            limit=6,
        ),
        "knowledge_domain_matrix_status": (
            knowledge_domain_matrix_component.get("status") or "unknown"
        ),
        "knowledge_domain_matrix": knowledge_domain_matrix_component,
        "knowledge_domain_matrix_domains": knowledge_domain_matrix_domains,
        "knowledge_domain_matrix_priority_domains": (
            knowledge_domain_matrix_priority_domains
        ),
        "knowledge_domain_matrix_recommendations": _compact(
            benchmark_knowledge_domain_matrix.get("recommendations") or [],
            limit=6,
        ),
        "knowledge_outcome_correlation_status": knowledge_outcome_correlation_status,
        "knowledge_outcome_correlation": knowledge_outcome_correlation_component,
        "knowledge_outcome_correlation_domains": knowledge_outcome_correlation_domains,
        "knowledge_outcome_correlation_priority_domains": (
            knowledge_outcome_correlation_priority_domains
        ),
        "knowledge_outcome_correlation_recommendations": _compact(
            benchmark_knowledge_outcome_correlation.get("recommendations") or [],
            limit=6,
        ),
        "blocking_signals": blockers,
        "review_signals": review_signals,
        "artifacts": {
            "benchmark_scorecard": _artifact_row(
                "benchmark_scorecard", artifact_paths.get("benchmark_scorecard", "")
            ),
            "benchmark_operational_summary": _artifact_row(
                "benchmark_operational_summary",
                artifact_paths.get("benchmark_operational_summary", ""),
            ),
            "benchmark_artifact_bundle": _artifact_row(
                "benchmark_artifact_bundle",
                artifact_paths.get("benchmark_artifact_bundle", ""),
            ),
            "benchmark_companion_summary": _artifact_row(
                "benchmark_companion_summary",
                artifact_paths.get("benchmark_companion_summary", ""),
            ),
            "benchmark_knowledge_readiness": _artifact_row(
                "benchmark_knowledge_readiness",
                artifact_paths.get("benchmark_knowledge_readiness", ""),
            ),
            "benchmark_knowledge_drift": _artifact_row(
                "benchmark_knowledge_drift",
                artifact_paths.get("benchmark_knowledge_drift", ""),
            ),
            "benchmark_engineering_signals": _artifact_row(
                "benchmark_engineering_signals",
                artifact_paths.get("benchmark_engineering_signals", ""),
            ),
            "benchmark_realdata_signals": _artifact_row(
                "benchmark_realdata_signals",
                artifact_paths.get("benchmark_realdata_signals", ""),
            ),
            "benchmark_realdata_scorecard": _artifact_row(
                "benchmark_realdata_scorecard",
                artifact_paths.get("benchmark_realdata_scorecard", ""),
            ),
            "benchmark_operator_adoption": _artifact_row(
                "benchmark_operator_adoption",
                artifact_paths.get("benchmark_operator_adoption", ""),
            ),
            "benchmark_knowledge_application": _artifact_row(
                "benchmark_knowledge_application",
                artifact_paths.get("benchmark_knowledge_application", ""),
            ),
            "benchmark_knowledge_realdata_correlation": _artifact_row(
                "benchmark_knowledge_realdata_correlation",
                artifact_paths.get(
                    "benchmark_knowledge_realdata_correlation",
                    "",
                ),
            ),
            "benchmark_knowledge_domain_matrix": _artifact_row(
                "benchmark_knowledge_domain_matrix",
                artifact_paths.get("benchmark_knowledge_domain_matrix", ""),
            ),
            "benchmark_knowledge_outcome_correlation": _artifact_row(
                "benchmark_knowledge_outcome_correlation",
                artifact_paths.get("benchmark_knowledge_outcome_correlation", ""),
            ),
        },
    }


def render_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        f"# {payload.get('title') or 'Benchmark Release Decision'}",
        "",
        f"- `release_status`: `{payload.get('release_status')}`",
        f"- `automation_ready`: `{payload.get('automation_ready')}`",
        f"- `primary_signal_source`: `{payload.get('primary_signal_source')}`",
        f"- `realdata_scorecard_status`: "
        f"`{payload.get('realdata_scorecard_status') or 'unknown'}`",
        "",
        "## Component Statuses",
        "",
    ]
    for name, status in (payload.get("component_statuses") or {}).items():
        lines.append(f"- `{name}`: `{status}`")
    drift = payload.get("operator_adoption_knowledge_drift") or {}
    lines.append(
        f"- `operator_adoption_knowledge_drift`: `{drift.get('status') or 'unknown'}`"
    )
    lines.append(
        f"- `operator_adoption_knowledge_drift_summary`: "
        f"{drift.get('summary') or 'none'}"
    )
    lines.append(
        f"- `knowledge_application_status`: "
        f"`{payload.get('knowledge_application_status') or 'unknown'}`"
    )
    lines.append(
        f"- `knowledge_realdata_correlation_status`: "
        f"`{payload.get('knowledge_realdata_correlation_status') or 'unknown'}`"
    )
    lines.append(
        f"- `knowledge_domain_matrix_status`: "
        f"`{payload.get('knowledge_domain_matrix_status') or 'unknown'}`"
    )
    lines.append(
        f"- `knowledge_outcome_correlation_status`: "
        f"`{payload.get('knowledge_outcome_correlation_status') or 'unknown'}`"
    )
    lines.append(f"- `realdata_status`: `{payload.get('realdata_status')}`")
    lines.extend(["", "## Blocking Signals", ""])
    blockers = payload.get("blocking_signals") or []
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- none")
    lines.extend(["", "## Review Signals", ""])
    review_signals = payload.get("review_signals") or []
    if review_signals:
        lines.extend(f"- {item}" for item in review_signals)
    else:
        lines.append("- none")
    lines.extend(["", "## Knowledge Focus Areas", ""])
    focus_areas = payload.get("knowledge_focus_areas") or []
    if focus_areas:
        for row in focus_areas:
            lines.append(
                "- "
                f"`{row.get('component')}` "
                f"status=`{row.get('status')}` "
                f"priority=`{row.get('priority')}` "
                f"action=`{row.get('action')}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Knowledge Drift", ""])
    knowledge_drift = payload.get("knowledge_drift") or {}
    lines.append(f"- `status`: `{payload.get('knowledge_drift_status')}`")
    lines.append(
        "- `summary`: "
        + (_text(payload.get("knowledge_drift_summary")) or "none")
    )
    counts = knowledge_drift.get("counts") or {}
    lines.append(
        "- `counts`: "
        f"regressions={counts.get('regressions', 0)} "
        f"improvements={counts.get('improvements', 0)} "
        f"new_focus_areas={counts.get('new_focus_areas', 0)} "
        f"resolved_focus_areas={counts.get('resolved_focus_areas', 0)}"
    )
    for label in (
        "regressions",
        "improvements",
        "new_focus_areas",
        "resolved_focus_areas",
    ):
        values = knowledge_drift.get(label) or []
        lines.append(
            f"- `{label}`: `{', '.join(str(item) for item in values) or 'none'}`"
        )
    for label in (
        "knowledge_drift_domain_regressions",
        "knowledge_drift_domain_improvements",
        "knowledge_drift_resolved_priority_domains",
        "knowledge_drift_new_priority_domains",
    ):
        values = payload.get(label) or []
        lines.append(
            f"- `{label}`: `{', '.join(str(item) for item in values) or 'none'}`"
        )
    lines.extend(["", "## Knowledge Domains", ""])
    knowledge_domains = payload.get("knowledge_domains") or {}
    if knowledge_domains:
        for name, row in knowledge_domains.items():
            lines.append(
                "- "
                f"`{name}` "
                f"status=`{row.get('status')}` "
                f"focus_components=`{', '.join(row.get('focus_components') or []) or 'none'}` "
                f"missing_metrics=`{', '.join(row.get('missing_metrics') or []) or 'none'}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Knowledge Domain Focus Areas", ""])
    domain_focus_areas = payload.get("knowledge_domain_focus_areas") or []
    if domain_focus_areas:
        for row in domain_focus_areas:
            lines.append(
                "- "
                f"`{row.get('domain')}` "
                f"status=`{row.get('status')}` "
                f"priority=`{row.get('priority')}` "
                f"action=`{row.get('action')}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Knowledge Application", ""])
    knowledge_application_domains = payload.get("knowledge_application_domains") or {}
    lines.append(
        f"- `status`: `{payload.get('knowledge_application_status') or 'unknown'}`"
    )
    if knowledge_application_domains:
        for name, row in knowledge_application_domains.items():
            lines.append(
                "- "
                f"`{name}` "
                f"status=`{row.get('status')}` "
                f"readiness=`{row.get('readiness_status')}` "
                f"evidence=`{row.get('evidence_status')}` "
                f"signal_count=`{row.get('signal_count')}`"
            )
    else:
        lines.append("- none")
    knowledge_application_recommendations = (
        payload.get("knowledge_application_recommendations") or []
    )
    if knowledge_application_recommendations:
        for item in knowledge_application_recommendations:
            lines.append(f"- recommendation: {item}")
    lines.extend(["", "## Knowledge Real-Data Correlation", ""])
    knowledge_realdata_domains = payload.get("knowledge_realdata_correlation_domains") or {}
    lines.append(
        "- `status`: "
        f"`{payload.get('knowledge_realdata_correlation_status') or 'unknown'}`"
    )
    if knowledge_realdata_domains:
        for name, row in knowledge_realdata_domains.items():
            lines.append(
                "- "
                f"`{name}` "
                f"status=`{row.get('status')}` "
                f"readiness=`{row.get('readiness_status')}` "
                f"application=`{row.get('application_status')}` "
                f"realdata=`{row.get('realdata_status')}`"
            )
    else:
        lines.append("- none")
    knowledge_realdata_recommendations = (
        payload.get("knowledge_realdata_correlation_recommendations") or []
    )
    if knowledge_realdata_recommendations:
        for item in knowledge_realdata_recommendations:
            lines.append(f"- recommendation: {item}")
    lines.extend(["", "## Knowledge Domain Matrix", ""])
    lines.append(
        f"- `status`: `{payload.get('knowledge_domain_matrix_status') or 'unknown'}`"
    )
    knowledge_domain_matrix_domains = payload.get("knowledge_domain_matrix_domains") or {}
    if knowledge_domain_matrix_domains:
        for name, row in knowledge_domain_matrix_domains.items():
            lines.append(
                "- "
                f"`{name}` "
                f"status=`{row.get('status')}` "
                f"readiness=`{row.get('readiness_status')}` "
                f"application=`{row.get('application_status')}` "
                f"realdata=`{row.get('realdata_status')}`"
            )
    else:
        lines.append("- none")
    knowledge_domain_matrix_recommendations = (
        payload.get("knowledge_domain_matrix_recommendations") or []
    )
    if knowledge_domain_matrix_recommendations:
        for item in knowledge_domain_matrix_recommendations:
            lines.append(f"- recommendation: {item}")
    lines.extend(["", "## Knowledge Outcome Correlation", ""])
    lines.append(
        "- `status`: "
        f"`{payload.get('knowledge_outcome_correlation_status') or 'unknown'}`"
    )
    knowledge_outcome_domains = payload.get("knowledge_outcome_correlation_domains") or {}
    if knowledge_outcome_domains:
        for name, row in knowledge_outcome_domains.items():
            lines.append(
                "- "
                f"`{name}` "
                f"status=`{row.get('status')}` "
                f"matrix=`{row.get('matrix_status')}` "
                f"best_surface=`{row.get('best_surface')}` "
                f"best_surface_score=`{row.get('best_surface_score')}`"
            )
    else:
        lines.append("- none")
    knowledge_outcome_recommendations = (
        payload.get("knowledge_outcome_correlation_recommendations") or []
    )
    if knowledge_outcome_recommendations:
        for item in knowledge_outcome_recommendations:
            lines.append(f"- recommendation: {item}")
    lines.extend(["", "## Real-Data Signals", ""])
    realdata = payload.get("realdata_signals") or {}
    lines.append(f"- `status`: `{payload.get('realdata_status')}`")
    component_rows = realdata.get("components") or {}
    if component_rows:
        for name, row in component_rows.items():
            lines.append(
                "- "
                f"`{name}` "
                f"status=`{row.get('status')}` "
                f"sample_size=`{row.get('sample_size', 'n/a')}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Real-Data Recommendations", ""])
    realdata_recommendations = payload.get("realdata_recommendations") or []
    if realdata_recommendations:
        lines.extend(f"- {item}" for item in realdata_recommendations)
    else:
        lines.append("- none")
    lines.extend(["", "## Real-Data Scorecard", ""])
    realdata_scorecard = payload.get("realdata_scorecard") or {}
    lines.append(
        f"- `status`: `{payload.get('realdata_scorecard_status') or 'unknown'}`"
    )
    if realdata_scorecard.get("best_surface"):
        lines.append(f"- `best_surface`: `{realdata_scorecard.get('best_surface')}`")
    scorecard_rows = realdata_scorecard.get("components") or {}
    if scorecard_rows:
        for name, row in scorecard_rows.items():
            lines.append(
                "- "
                f"`{name}` "
                f"status=`{row.get('status')}` "
                f"coarse_accuracy=`{row.get('coarse_accuracy', 'n/a')}` "
                f"sample_size=`{row.get('sample_size', 'n/a')}`"
            )
    else:
        lines.append("- none")
    realdata_scorecard_recommendations = (
        payload.get("realdata_scorecard_recommendations") or []
    )
    if realdata_scorecard_recommendations:
        lines.extend(f"- {item}" for item in realdata_scorecard_recommendations)
    else:
        lines.append("- none")
    lines.extend(["", "## Artifacts", ""])
    for name, row in (payload.get("artifacts") or {}).items():
        lines.append(
            f"- `{name}`: present=`{row.get('present')}` path=`{row.get('path')}`"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a benchmark release decision summary."
    )
    parser.add_argument("--title", default="Benchmark Release Decision")
    parser.add_argument("--benchmark-scorecard", default="")
    parser.add_argument("--benchmark-operational-summary", default="")
    parser.add_argument("--benchmark-artifact-bundle", default="")
    parser.add_argument("--benchmark-companion-summary", default="")
    parser.add_argument("--benchmark-knowledge-readiness", default="")
    parser.add_argument("--benchmark-knowledge-drift", default="")
    parser.add_argument("--benchmark-engineering-signals", default="")
    parser.add_argument("--benchmark-realdata-signals", default="")
    parser.add_argument("--benchmark-realdata-scorecard", default="")
    parser.add_argument("--benchmark-operator-adoption", default="")
    parser.add_argument("--benchmark-knowledge-application", default="")
    parser.add_argument("--benchmark-knowledge-realdata-correlation", default="")
    parser.add_argument("--benchmark-knowledge-domain-matrix", default="")
    parser.add_argument("--benchmark-knowledge-outcome-correlation", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    artifact_paths = {
        "benchmark_scorecard": args.benchmark_scorecard,
        "benchmark_operational_summary": args.benchmark_operational_summary,
        "benchmark_artifact_bundle": args.benchmark_artifact_bundle,
        "benchmark_companion_summary": args.benchmark_companion_summary,
        "benchmark_knowledge_readiness": args.benchmark_knowledge_readiness,
        "benchmark_knowledge_drift": args.benchmark_knowledge_drift,
        "benchmark_engineering_signals": args.benchmark_engineering_signals,
        "benchmark_realdata_signals": args.benchmark_realdata_signals,
        "benchmark_realdata_scorecard": args.benchmark_realdata_scorecard,
        "benchmark_operator_adoption": args.benchmark_operator_adoption,
        "benchmark_knowledge_application": args.benchmark_knowledge_application,
        "benchmark_knowledge_realdata_correlation": (
            args.benchmark_knowledge_realdata_correlation
        ),
        "benchmark_knowledge_domain_matrix": args.benchmark_knowledge_domain_matrix,
        "benchmark_knowledge_outcome_correlation": (
            args.benchmark_knowledge_outcome_correlation
        ),
    }
    payload = build_release_decision(
        title=args.title,
        benchmark_scorecard=_maybe_load_json(args.benchmark_scorecard),
        benchmark_operational_summary=_maybe_load_json(
            args.benchmark_operational_summary
        ),
        benchmark_artifact_bundle=_maybe_load_json(args.benchmark_artifact_bundle),
        benchmark_companion_summary=_maybe_load_json(
            args.benchmark_companion_summary
        ),
        benchmark_knowledge_readiness=_maybe_load_json(
            args.benchmark_knowledge_readiness
        ),
        benchmark_knowledge_drift=_maybe_load_json(args.benchmark_knowledge_drift),
        benchmark_engineering_signals=_maybe_load_json(
            args.benchmark_engineering_signals
        ),
        benchmark_realdata_signals=_maybe_load_json(
            args.benchmark_realdata_signals
        ),
        benchmark_operator_adoption=_maybe_load_json(
            args.benchmark_operator_adoption
        ),
        benchmark_realdata_scorecard=_maybe_load_json(
            args.benchmark_realdata_scorecard
        ),
        benchmark_knowledge_application=_maybe_load_json(
            args.benchmark_knowledge_application
        ),
        benchmark_knowledge_realdata_correlation=_maybe_load_json(
            args.benchmark_knowledge_realdata_correlation
        ),
        benchmark_knowledge_domain_matrix=_maybe_load_json(
            args.benchmark_knowledge_domain_matrix
        ),
        benchmark_knowledge_outcome_correlation=_maybe_load_json(
            args.benchmark_knowledge_outcome_correlation
        ),
        artifact_paths=artifact_paths,
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        _write_output(args.output_json, rendered + "\n")
    if args.output_md:
        _write_output(args.output_md, render_markdown(payload))
    print(rendered)


if __name__ == "__main__":
    main()
