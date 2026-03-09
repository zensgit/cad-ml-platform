#!/usr/bin/env python3
"""Export a compact benchmark artifact bundle manifest and markdown summary."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.append(str(Path(__file__).resolve().parents[1]))


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


def _text(value: Any) -> str:
    return str(value or "").strip()


def _scorecard_operator_adoption(scorecard: Dict[str, Any]) -> Dict[str, Any]:
    component = (scorecard.get("components") or {}).get("operator_adoption") or {}
    return {
        "status": str(component.get("status") or "unknown"),
        "operator_mode": str(component.get("operator_mode") or "unknown"),
        "knowledge_outcome_drift_status": str(
            component.get("knowledge_outcome_drift_status") or "unknown"
        ),
        "knowledge_outcome_drift_summary": str(
            component.get("knowledge_outcome_drift_summary") or "none"
        ),
    }


def _operational_operator_adoption(
    operational_summary: Dict[str, Any],
) -> Dict[str, Any]:
    components = operational_summary.get("component_statuses") or {}
    return {
        "status": str(components.get("operator_adoption") or "unknown"),
        "knowledge_outcome_drift_status": str(
            operational_summary.get("operator_adoption_knowledge_outcome_drift_status")
            or "unknown"
        ),
        "knowledge_outcome_drift_summary": str(
            operational_summary.get("operator_adoption_knowledge_outcome_drift_summary")
            or "none"
        ),
    }


def _artifact_row(
    *,
    name: str,
    path_text: str,
    payload: Dict[str, Any],
    fallback_status: str = "missing",
) -> Dict[str, Any]:
    path_value = str(path_text or "").strip()
    return {
        "name": name,
        "path": path_value,
        "present": bool(payload) or bool(path_value),
        "status": str(payload.get("overall_status") or payload.get("status") or fallback_status),
    }


def _component_statuses(
    scorecard: Dict[str, Any],
    operational_summary: Dict[str, Any],
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
    benchmark_knowledge_outcome_drift: Dict[str, Any] | None = None,
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
    benchmark_knowledge_outcome_drift = benchmark_knowledge_outcome_drift or {}
    components = scorecard.get("components") or {}
    companion_components = benchmark_companion_summary.get("component_statuses") or {}
    knowledge_component = (
        benchmark_knowledge_readiness.get("knowledge_readiness")
        or benchmark_knowledge_readiness
        or {}
    )
    knowledge_drift_component = _knowledge_drift_component(benchmark_knowledge_drift)
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
    scorecard_operator_adoption = _scorecard_operator_adoption(scorecard)
    operational_operator_adoption = _operational_operator_adoption(
        operational_summary
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
    knowledge_outcome_drift_component = _knowledge_outcome_drift_component(
        benchmark_knowledge_outcome_drift
    )
    component_rows = {
        "hybrid": str(
            companion_components.get("hybrid")
            or (components.get("hybrid") or {}).get("status")
            or "unknown"
        ),
        "history_sequence": str(
            (components.get("history_sequence") or {}).get("status") or "unknown"
        ),
        "brep": str((components.get("brep") or {}).get("status") or "unknown"),
        "migration_governance": str(
            (components.get("migration_governance") or {}).get("status") or "unknown"
        ),
    }
    operational_components = operational_summary.get("component_statuses") or {}
    component_rows["feedback_flywheel"] = str(
        operational_components.get("feedback_flywheel")
        or (components.get("feedback_flywheel") or {}).get("status")
        or "unknown"
    )
    component_rows["assistant_explainability"] = str(
        companion_components.get("assistant_explainability")
        or operational_components.get("assistant_explainability")
        or (components.get("assistant_explainability") or {}).get("status")
        or "unknown"
    )
    component_rows["review_queue"] = str(
        companion_components.get("review_queue")
        or operational_components.get("review_queue")
        or (components.get("review_queue") or {}).get("status")
        or "unknown"
    )
    component_rows["ocr_review"] = str(
        companion_components.get("ocr_review")
        or operational_components.get("ocr_review")
        or (components.get("ocr_review") or {}).get("status")
        or "unknown"
    )
    component_rows["realdata_scorecard"] = str(
        companion_components.get("realdata_scorecard")
        or operational_components.get("realdata_scorecard")
        or (components.get("realdata_scorecard") or {}).get("status")
        or realdata_scorecard_component.get("status")
        or "unknown"
    )
    component_rows["qdrant_backend"] = str(
        companion_components.get("qdrant_backend")
        or (components.get("qdrant_backend") or {}).get("status")
        or "unknown"
    )
    component_rows["knowledge_readiness"] = str(
        companion_components.get("knowledge_readiness")
        or knowledge_component.get("status")
        or (components.get("knowledge_readiness") or {}).get("status")
        or "unknown"
    )
    component_rows["knowledge_drift"] = str(
        companion_components.get("knowledge_drift")
        or knowledge_drift_component.get("status")
        or "unknown"
    )
    component_rows["engineering_signals"] = str(
        companion_components.get("engineering_signals")
        or engineering_component.get("status")
        or (components.get("engineering_signals") or {}).get("status")
        or "unknown"
    )
    component_rows["realdata_signals"] = str(
        companion_components.get("realdata_signals")
        or realdata_component.get("status")
        or (components.get("realdata_signals") or {}).get("status")
        or "unknown"
    )
    component_rows["operator_adoption"] = str(
        benchmark_operator_adoption.get("adoption_readiness") or "unknown"
    )
    if component_rows["operator_adoption"] == "unknown":
        component_rows["operator_adoption"] = str(
            operational_operator_adoption.get("status")
            or scorecard_operator_adoption.get("status")
            or "unknown"
        )
    component_rows["knowledge_application"] = str(
        companion_components.get("knowledge_application")
        or operational_components.get("knowledge_application")
        or knowledge_application_component.get("status")
        or (components.get("knowledge_application") or {}).get("status")
        or "unknown"
    )
    component_rows["knowledge_realdata_correlation"] = str(
        companion_components.get("knowledge_realdata_correlation")
        or operational_components.get("knowledge_realdata_correlation")
        or knowledge_realdata_correlation_component.get("status")
        or (components.get("knowledge_realdata_correlation") or {}).get("status")
        or "unknown"
    )
    component_rows["knowledge_domain_matrix"] = str(
        companion_components.get("knowledge_domain_matrix")
        or operational_components.get("knowledge_domain_matrix")
        or knowledge_domain_matrix_component.get("status")
        or (components.get("knowledge_domain_matrix") or {}).get("status")
        or "unknown"
    )
    component_rows["knowledge_outcome_correlation"] = str(
        companion_components.get("knowledge_outcome_correlation")
        or operational_components.get("knowledge_outcome_correlation")
        or knowledge_outcome_correlation_component.get("status")
        or (components.get("knowledge_outcome_correlation") or {}).get("status")
        or "unknown"
    )
    component_rows["knowledge_outcome_drift"] = str(
        companion_components.get("knowledge_outcome_drift")
        or operational_components.get("knowledge_outcome_drift")
        or knowledge_outcome_drift_component.get("status")
        or (components.get("knowledge_outcome_drift") or {}).get("status")
        or "unknown"
    )
    return component_rows


def _compact_list(items: Iterable[Any]) -> List[str]:
    return [str(item).strip() for item in items if str(item).strip()]


def _knowledge_drift_component(payload: Dict[str, Any]) -> Dict[str, Any]:
    return payload.get("knowledge_drift") or payload or {}


def _knowledge_outcome_drift_component(payload: Dict[str, Any]) -> Dict[str, Any]:
    return payload.get("knowledge_outcome_drift") or payload or {}


def _knowledge_drift_summary(component: Dict[str, Any]) -> str:
    if not component:
        return ""
    parts = [f"status={component.get('status') or 'unknown'}"]
    current_status = str(component.get("current_status") or "").strip()
    previous_status = str(component.get("previous_status") or "").strip()
    if current_status:
        parts.append(f"current={current_status}")
    if previous_status:
        parts.append(f"previous={previous_status}")
    if component.get("reference_item_delta") is not None:
        parts.append(f"reference_item_delta={component.get('reference_item_delta')}")
    regressions = _compact_list(component.get("regressions") or [])
    improvements = _compact_list(component.get("improvements") or [])
    new_focus_areas = _compact_list(component.get("new_focus_areas") or [])
    if regressions:
        parts.append("regressions=" + ", ".join(regressions))
    if improvements:
        parts.append("improvements=" + ", ".join(improvements))
    if new_focus_areas:
        parts.append("new_focus_areas=" + ", ".join(new_focus_areas))
    return "; ".join(parts)


def _knowledge_outcome_drift_summary(component: Dict[str, Any]) -> str:
    if not component:
        return ""
    parts = [f"status={component.get('status') or 'unknown'}"]
    current_status = str(component.get("current_status") or "").strip()
    previous_status = str(component.get("previous_status") or "").strip()
    if current_status:
        parts.append(f"current={current_status}")
    if previous_status:
        parts.append(f"previous={previous_status}")
    regressions = _compact_list(component.get("regressions") or [])
    improvements = _compact_list(component.get("improvements") or [])
    new_focus_areas = _compact_list(component.get("new_focus_areas") or [])
    if regressions:
        parts.append("regressions=" + ", ".join(regressions))
    if improvements:
        parts.append("improvements=" + ", ".join(improvements))
    if new_focus_areas:
        parts.append("new_focus_areas=" + ", ".join(new_focus_areas))
    return "; ".join(parts)


def _pick_summary_items(
    benchmark_release_decision: Dict[str, Any],
    benchmark_companion_summary: Dict[str, Any],
    benchmark_operational_summary: Dict[str, Any],
    benchmark_scorecard: Dict[str, Any],
    benchmark_knowledge_drift: Dict[str, Any],
    benchmark_engineering_signals: Dict[str, Any],
    benchmark_realdata_signals: Dict[str, Any],
    benchmark_operator_adoption: Dict[str, Any],
    benchmark_realdata_scorecard: Dict[str, Any] | None = None,
    benchmark_knowledge_application: Dict[str, Any] | None = None,
    benchmark_knowledge_realdata_correlation: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_matrix: Dict[str, Any] | None = None,
    benchmark_knowledge_outcome_correlation: Dict[str, Any] | None = None,
    benchmark_knowledge_outcome_drift: Dict[str, Any] | None = None,
) -> tuple[str, List[str], List[str]]:
    benchmark_realdata_scorecard = benchmark_realdata_scorecard or {}
    benchmark_knowledge_application = benchmark_knowledge_application or {}
    benchmark_knowledge_realdata_correlation = (
        benchmark_knowledge_realdata_correlation or {}
    )
    benchmark_knowledge_domain_matrix = benchmark_knowledge_domain_matrix or {}
    benchmark_knowledge_outcome_correlation = (
        benchmark_knowledge_outcome_correlation or {}
    )
    benchmark_knowledge_outcome_drift = benchmark_knowledge_outcome_drift or {}
    overall_status = (
        str(benchmark_release_decision.get("release_status") or "").strip()
        or str(benchmark_companion_summary.get("overall_status") or "").strip()
        or str(benchmark_operational_summary.get("overall_status") or "").strip()
        or str(benchmark_scorecard.get("overall_status") or "").strip()
        or "unknown"
    )
    blockers = _compact_list(
        benchmark_release_decision.get("blocking_signals")
        or benchmark_companion_summary.get("blockers")
        or benchmark_operational_summary.get("blockers")
        or benchmark_operator_adoption.get("blocking_signals")
        or []
    )
    recommendations = _compact_list(
        benchmark_release_decision.get("review_signals")
        or benchmark_companion_summary.get("recommended_actions")
        or benchmark_operational_summary.get("recommendations")
        or benchmark_scorecard.get("recommendations")
        or benchmark_knowledge_drift.get("recommendations")
        or benchmark_engineering_signals.get("recommendations")
        or benchmark_realdata_signals.get("recommendations")
        or benchmark_realdata_scorecard.get("recommendations")
        or benchmark_operator_adoption.get("recommended_actions")
        or benchmark_knowledge_application.get("recommendations")
        or benchmark_knowledge_realdata_correlation.get("recommendations")
        or benchmark_knowledge_domain_matrix.get("recommendations")
        or benchmark_knowledge_outcome_correlation.get("recommendations")
        or benchmark_knowledge_outcome_drift.get("recommendations")
        or []
    )
    return overall_status, blockers, recommendations


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
        "recommendations": _compact_list(
            drift.get("recommendations")
            or benchmark_operator_adoption.get("recommended_actions")
            or []
        ),
    }


def build_bundle(
    *,
    title: str,
    benchmark_scorecard: Dict[str, Any],
    benchmark_operational_summary: Dict[str, Any],
    benchmark_companion_summary: Dict[str, Any],
    benchmark_release_decision: Dict[str, Any],
    benchmark_knowledge_readiness: Dict[str, Any],
    benchmark_knowledge_drift: Dict[str, Any],
    benchmark_engineering_signals: Dict[str, Any],
    benchmark_realdata_signals: Dict[str, Any],
    benchmark_realdata_scorecard: Dict[str, Any] | None = None,
    benchmark_operator_adoption: Dict[str, Any],
    benchmark_knowledge_application: Dict[str, Any] | None = None,
    benchmark_knowledge_realdata_correlation: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_matrix: Dict[str, Any] | None = None,
    benchmark_knowledge_outcome_correlation: Dict[str, Any] | None = None,
    benchmark_knowledge_outcome_drift: Dict[str, Any] | None = None,
    feedback_flywheel: Dict[str, Any],
    assistant_evidence: Dict[str, Any],
    review_queue: Dict[str, Any],
    ocr_review: Dict[str, Any],
    artifact_paths: Dict[str, str],
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
    benchmark_knowledge_outcome_drift = benchmark_knowledge_outcome_drift or {}
    overall_status, blockers, recommendations = _pick_summary_items(
        benchmark_release_decision,
        benchmark_companion_summary,
        benchmark_operational_summary,
        benchmark_scorecard,
        benchmark_knowledge_drift,
        benchmark_engineering_signals,
        benchmark_realdata_signals,
        benchmark_realdata_scorecard,
        benchmark_operator_adoption,
        benchmark_knowledge_application,
        benchmark_knowledge_realdata_correlation,
        benchmark_knowledge_domain_matrix,
        benchmark_knowledge_outcome_correlation,
        benchmark_knowledge_outcome_drift,
    )
    knowledge_component = (
        benchmark_knowledge_readiness.get("knowledge_readiness")
        or benchmark_knowledge_readiness
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
    knowledge_outcome_drift_component = _knowledge_outcome_drift_component(
        benchmark_knowledge_outcome_drift
    )
    realdata_scorecard_recommendations = (
        benchmark_realdata_scorecard.get("recommendations") or []
    )
    knowledge_drift_component = _knowledge_drift_component(benchmark_knowledge_drift)
    knowledge_focus_areas = list(knowledge_component.get("focus_areas_detail") or [])
    knowledge_drift_recommendations = _compact_list(
        benchmark_knowledge_drift.get("recommendations") or []
    )
    operator_adoption_knowledge_drift = _operator_adoption_knowledge_drift(
        benchmark_operator_adoption
    )
    scorecard_operator_adoption = _scorecard_operator_adoption(benchmark_scorecard)
    operational_operator_adoption = _operational_operator_adoption(
        benchmark_operational_summary
    )
    knowledge_domains = knowledge_component.get("domains") or {}
    knowledge_domain_focus_areas = list(
        knowledge_component.get("domain_focus_areas") or []
    )
    knowledge_priority_domains = list(knowledge_component.get("priority_domains") or [])
    knowledge_application_focus_areas = list(
        knowledge_application_component.get("focus_areas_detail") or []
    )
    knowledge_application_domains = knowledge_application_component.get("domains") or {}
    knowledge_application_priority_domains = list(
        knowledge_application_component.get("priority_domains") or []
    )
    knowledge_application_recommendations = _compact_list(
        benchmark_knowledge_application.get("recommendations") or []
    )
    knowledge_realdata_correlation_recommendations = _compact_list(
        benchmark_knowledge_realdata_correlation.get("recommendations") or []
    )
    knowledge_domain_matrix_recommendations = _compact_list(
        benchmark_knowledge_domain_matrix.get("recommendations") or []
    )
    knowledge_outcome_correlation_recommendations = _compact_list(
        benchmark_knowledge_outcome_correlation.get("recommendations") or []
    )
    knowledge_outcome_drift_recommendations = _compact_list(
        benchmark_knowledge_outcome_drift.get("recommendations") or []
    )
    artifact_rows = {
        "benchmark_scorecard": _artifact_row(
            name="benchmark_scorecard",
            path_text=artifact_paths.get("benchmark_scorecard", ""),
            payload=benchmark_scorecard,
        ),
        "benchmark_operational_summary": _artifact_row(
            name="benchmark_operational_summary",
            path_text=artifact_paths.get("benchmark_operational_summary", ""),
            payload=benchmark_operational_summary,
        ),
        "benchmark_companion_summary": _artifact_row(
            name="benchmark_companion_summary",
            path_text=artifact_paths.get("benchmark_companion_summary", ""),
            payload=benchmark_companion_summary,
        ),
        "benchmark_release_decision": _artifact_row(
            name="benchmark_release_decision",
            path_text=artifact_paths.get("benchmark_release_decision", ""),
            payload=benchmark_release_decision,
        ),
        "benchmark_knowledge_readiness": _artifact_row(
            name="benchmark_knowledge_readiness",
            path_text=artifact_paths.get("benchmark_knowledge_readiness", ""),
            payload=benchmark_knowledge_readiness,
        ),
        "benchmark_knowledge_drift": _artifact_row(
            name="benchmark_knowledge_drift",
            path_text=artifact_paths.get("benchmark_knowledge_drift", ""),
            payload=benchmark_knowledge_drift,
        ),
        "benchmark_engineering_signals": _artifact_row(
            name="benchmark_engineering_signals",
            path_text=artifact_paths.get("benchmark_engineering_signals", ""),
            payload=benchmark_engineering_signals,
        ),
        "benchmark_realdata_signals": _artifact_row(
            name="benchmark_realdata_signals",
            path_text=artifact_paths.get("benchmark_realdata_signals", ""),
            payload=benchmark_realdata_signals,
        ),
        "benchmark_realdata_scorecard": _artifact_row(
            name="benchmark_realdata_scorecard",
            path_text=artifact_paths.get("benchmark_realdata_scorecard", ""),
            payload=benchmark_realdata_scorecard,
        ),
        "benchmark_operator_adoption": _artifact_row(
            name="benchmark_operator_adoption",
            path_text=artifact_paths.get("benchmark_operator_adoption", ""),
            payload=benchmark_operator_adoption,
        ),
        "benchmark_knowledge_application": _artifact_row(
            name="benchmark_knowledge_application",
            path_text=artifact_paths.get("benchmark_knowledge_application", ""),
            payload=benchmark_knowledge_application,
        ),
        "benchmark_knowledge_realdata_correlation": _artifact_row(
            name="benchmark_knowledge_realdata_correlation",
            path_text=artifact_paths.get(
                "benchmark_knowledge_realdata_correlation",
                "",
            ),
            payload=benchmark_knowledge_realdata_correlation,
        ),
        "benchmark_knowledge_domain_matrix": _artifact_row(
            name="benchmark_knowledge_domain_matrix",
            path_text=artifact_paths.get("benchmark_knowledge_domain_matrix", ""),
            payload=benchmark_knowledge_domain_matrix,
        ),
        "benchmark_knowledge_outcome_correlation": _artifact_row(
            name="benchmark_knowledge_outcome_correlation",
            path_text=artifact_paths.get(
                "benchmark_knowledge_outcome_correlation", ""
            ),
            payload=benchmark_knowledge_outcome_correlation,
        ),
        "benchmark_knowledge_outcome_drift": _artifact_row(
            name="benchmark_knowledge_outcome_drift",
            path_text=artifact_paths.get("benchmark_knowledge_outcome_drift", ""),
            payload=benchmark_knowledge_outcome_drift,
        ),
        "feedback_flywheel": _artifact_row(
            name="feedback_flywheel",
            path_text=artifact_paths.get("feedback_flywheel", ""),
            payload=feedback_flywheel,
        ),
        "assistant_evidence": _artifact_row(
            name="assistant_evidence",
            path_text=artifact_paths.get("assistant_evidence", ""),
            payload=assistant_evidence,
        ),
        "review_queue": _artifact_row(
            name="review_queue",
            path_text=artifact_paths.get("review_queue", ""),
            payload=review_queue,
        ),
        "ocr_review": _artifact_row(
            name="ocr_review",
            path_text=artifact_paths.get("ocr_review", ""),
            payload=ocr_review,
        ),
    }
    available_count = len([row for row in artifact_rows.values() if row["present"]])
    return {
        "title": title,
        "generated_at": int(time.time()),
        "overall_status": overall_status,
        "available_artifact_count": available_count,
        "knowledge_focus_area_count": len(knowledge_focus_areas),
        "knowledge_focus_areas": knowledge_focus_areas,
        "knowledge_drift": knowledge_drift_component,
        "knowledge_drift_summary": _knowledge_drift_summary(knowledge_drift_component),
        "knowledge_drift_recommendations": knowledge_drift_recommendations,
        "knowledge_drift_component_changes": list(
            knowledge_drift_component.get("component_changes") or []
        ),
        "knowledge_drift_domain_regressions": list(
            knowledge_drift_component.get("domain_regressions") or []
        ),
        "knowledge_drift_domain_improvements": list(
            knowledge_drift_component.get("domain_improvements") or []
        ),
        "knowledge_drift_resolved_priority_domains": list(
            knowledge_drift_component.get("resolved_priority_domains") or []
        ),
        "knowledge_drift_new_priority_domains": list(
            knowledge_drift_component.get("new_priority_domains") or []
        ),
        "realdata_status": str(realdata_component.get("status") or "unknown"),
        "realdata_signals": realdata_component,
        "realdata_recommendations": _compact_list(
            benchmark_realdata_signals.get("recommendations") or []
        ),
        "realdata_scorecard_status": (
            realdata_scorecard_component.get("status") or "unknown"
        ),
        "realdata_scorecard": realdata_scorecard_component,
        "realdata_scorecard_recommendations": _compact_list(
            realdata_scorecard_recommendations
        ),
        "knowledge_application_status": knowledge_application_component.get("status")
        or "unknown",
        "knowledge_application": knowledge_application_component,
        "knowledge_application_focus_areas": knowledge_application_focus_areas,
        "knowledge_application_domains": knowledge_application_domains,
        "knowledge_application_priority_domains": knowledge_application_priority_domains,
        "knowledge_application_recommendations": knowledge_application_recommendations,
        "knowledge_realdata_correlation_status": (
            knowledge_realdata_correlation_component.get("status") or "unknown"
        ),
        "knowledge_realdata_correlation": knowledge_realdata_correlation_component,
        "knowledge_realdata_correlation_domains": (
            knowledge_realdata_correlation_component.get("domains") or {}
        ),
        "knowledge_realdata_correlation_priority_domains": list(
            knowledge_realdata_correlation_component.get("priority_domains") or []
        ),
        "knowledge_realdata_correlation_recommendations": (
            knowledge_realdata_correlation_recommendations
        ),
        "knowledge_domain_matrix_status": (
            knowledge_domain_matrix_component.get("status") or "unknown"
        ),
        "knowledge_domain_matrix": knowledge_domain_matrix_component,
        "knowledge_domain_matrix_domains": (
            knowledge_domain_matrix_component.get("domains") or {}
        ),
        "knowledge_domain_matrix_priority_domains": list(
            knowledge_domain_matrix_component.get("priority_domains") or []
        ),
        "knowledge_domain_matrix_recommendations": (
            knowledge_domain_matrix_recommendations
        ),
        "knowledge_outcome_correlation_status": (
            knowledge_outcome_correlation_component.get("status") or "unknown"
        ),
        "knowledge_outcome_correlation": knowledge_outcome_correlation_component,
        "knowledge_outcome_correlation_domains": (
            knowledge_outcome_correlation_component.get("domains") or {}
        ),
        "knowledge_outcome_correlation_priority_domains": list(
            knowledge_outcome_correlation_component.get("priority_domains") or []
        ),
        "knowledge_outcome_correlation_recommendations": (
            knowledge_outcome_correlation_recommendations
        ),
        "knowledge_outcome_drift_status": (
            knowledge_outcome_drift_component.get("status") or "unknown"
        ),
        "knowledge_outcome_drift": knowledge_outcome_drift_component,
        "knowledge_outcome_drift_summary": _knowledge_outcome_drift_summary(
            knowledge_outcome_drift_component
        ),
        "knowledge_outcome_drift_domain_regressions": list(
            knowledge_outcome_drift_component.get("domain_regressions") or []
        ),
        "knowledge_outcome_drift_domain_improvements": list(
            knowledge_outcome_drift_component.get("domain_improvements") or []
        ),
        "knowledge_outcome_drift_resolved_priority_domains": list(
            knowledge_outcome_drift_component.get("resolved_priority_domains") or []
        ),
        "knowledge_outcome_drift_new_priority_domains": list(
            knowledge_outcome_drift_component.get("new_priority_domains") or []
        ),
        "knowledge_outcome_drift_recommendations": (
            knowledge_outcome_drift_recommendations
        ),
        "operator_adoption_knowledge_drift": operator_adoption_knowledge_drift,
        "scorecard_operator_adoption": scorecard_operator_adoption,
        "operational_operator_adoption": operational_operator_adoption,
        "knowledge_domains": knowledge_domains,
        "knowledge_domain_focus_areas": knowledge_domain_focus_areas,
        "knowledge_priority_domains": knowledge_priority_domains,
        "component_statuses": _component_statuses(
            benchmark_scorecard,
            benchmark_operational_summary,
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
            benchmark_knowledge_outcome_drift,
        ),
        "blockers": blockers,
        "recommendations": recommendations,
        "artifacts": artifact_rows,
    }


def render_markdown(payload: Dict[str, Any]) -> str:
    components = payload.get("component_statuses") or {}
    artifacts = payload.get("artifacts") or {}
    lines = [
        f"# {payload.get('title') or 'Benchmark Artifact Bundle'}",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- overall_status: `{payload.get('overall_status')}`",
        f"- available_artifact_count: `{payload.get('available_artifact_count')}`",
        f"- knowledge_focus_area_count: `{payload.get('knowledge_focus_area_count')}`",
        f"- knowledge_drift_summary: `{payload.get('knowledge_drift_summary') or 'none'}`",
        f"- knowledge_application_status: "
        f"`{payload.get('knowledge_application_status') or 'unknown'}`",
        f"- knowledge_realdata_correlation_status: "
        f"`{payload.get('knowledge_realdata_correlation_status') or 'unknown'}`",
        f"- knowledge_domain_matrix_status: "
        f"`{payload.get('knowledge_domain_matrix_status') or 'unknown'}`",
        f"- knowledge_outcome_correlation_status: "
        f"`{payload.get('knowledge_outcome_correlation_status') or 'unknown'}`",
        f"- knowledge_outcome_drift_status: "
        f"`{payload.get('knowledge_outcome_drift_status') or 'unknown'}`",
        f"- realdata_status: `{payload.get('realdata_status') or 'unknown'}`",
        f"- realdata_scorecard_status: "
        f"`{payload.get('realdata_scorecard_status') or 'unknown'}`",
        "",
        "## Component Statuses",
        "",
    ]
    for name, status in components.items():
        lines.append(f"- `{name}`: `{status}`")
    lines.extend(["", "## Artifacts", ""])
    for name, row in artifacts.items():
        lines.append(
            f"- `{name}`: present=`{row.get('present')}` status=`{row.get('status')}` "
            f"path=`{row.get('path')}`"
        )
    blockers = payload.get("blockers") or []
    recommendations = payload.get("recommendations") or []
    lines.extend(["", "## Blockers", ""])
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- none")
    lines.extend(["", "## Recommendations", ""])
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- none")
    lines.extend(["", "## Knowledge Application", ""])
    lines.append(
        f"- `status`: `{payload.get('knowledge_application_status') or 'unknown'}`"
    )
    knowledge_application_domains = payload.get("knowledge_application_domains") or {}
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
    lines.append(
        "- `status`: "
        f"`{payload.get('knowledge_realdata_correlation_status') or 'unknown'}`"
    )
    knowledge_realdata_domains = payload.get("knowledge_realdata_correlation_domains") or {}
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
        f"- `status`: `{payload.get('knowledge_outcome_correlation_status') or 'unknown'}`"
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
    lines.extend(["", "## Knowledge Outcome Drift", ""])
    lines.append(
        f"- `status`: `{payload.get('knowledge_outcome_drift_status') or 'unknown'}`"
    )
    lines.append(
        "- summary: "
        + (_text(payload.get("knowledge_outcome_drift_summary")) or "none")
    )
    lines.append(
        "- domain_regressions: "
        + (
            ", ".join(
                str(item)
                for item in (payload.get("knowledge_outcome_drift_domain_regressions") or [])
            )
            or "none"
        )
    )
    lines.append(
        "- domain_improvements: "
        + (
            ", ".join(
                str(item)
                for item in (payload.get("knowledge_outcome_drift_domain_improvements") or [])
            )
            or "none"
        )
    )
    lines.append(
        "- resolved_priority_domains: "
        + (
            ", ".join(
                str(item)
                for item in (
                    payload.get("knowledge_outcome_drift_resolved_priority_domains") or []
                )
            )
            or "none"
        )
    )
    lines.append(
        "- new_priority_domains: "
        + (
            ", ".join(
                str(item)
                for item in (payload.get("knowledge_outcome_drift_new_priority_domains") or [])
            )
            or "none"
        )
    )
    knowledge_outcome_drift_recommendations = (
        payload.get("knowledge_outcome_drift_recommendations") or []
    )
    if knowledge_outcome_drift_recommendations:
        for item in knowledge_outcome_drift_recommendations:
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
        for item in realdata_recommendations:
            lines.append(f"- recommendation: {item}")
    else:
        lines.append("- recommendation: none")
    lines.extend(["", "## Real-Data Scorecard", ""])
    lines.append(
        f"- `status`: `{payload.get('realdata_scorecard_status') or 'unknown'}`"
    )
    realdata_scorecard = payload.get("realdata_scorecard") or {}
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
        for item in realdata_scorecard_recommendations:
            lines.append(f"- recommendation: {item}")
    lines.extend(["", "## Knowledge Drift", ""])
    knowledge_drift_changes = payload.get("knowledge_drift_component_changes") or []
    knowledge_drift_domain_regressions = (
        payload.get("knowledge_drift_domain_regressions") or []
    )
    knowledge_drift_domain_improvements = (
        payload.get("knowledge_drift_domain_improvements") or []
    )
    knowledge_drift_resolved_priority_domains = (
        payload.get("knowledge_drift_resolved_priority_domains") or []
    )
    knowledge_drift_new_priority_domains = (
        payload.get("knowledge_drift_new_priority_domains") or []
    )
    knowledge_drift_recommendations = payload.get("knowledge_drift_recommendations") or []
    if payload.get("knowledge_drift_summary"):
        lines.append(f"- summary: {payload.get('knowledge_drift_summary')}")
    else:
        lines.append("- summary: none")
    if knowledge_drift_changes:
        for row in knowledge_drift_changes:
            lines.append(
                "- "
                f"`{row.get('component')}` "
                f"`{row.get('previous_status')}` -> `{row.get('current_status')}` "
                f"trend=`{row.get('trend')}` "
                f"delta=`{row.get('reference_item_delta')}`"
            )
    else:
        lines.append("- component_changes: none")
    lines.append(
        "- domain_regressions: "
        + (", ".join(str(item) for item in knowledge_drift_domain_regressions) or "none")
    )
    lines.append(
        "- domain_improvements: "
        + (", ".join(str(item) for item in knowledge_drift_domain_improvements) or "none")
    )
    lines.append(
        "- resolved_priority_domains: "
        + (
            ", ".join(str(item) for item in knowledge_drift_resolved_priority_domains)
            or "none"
        )
    )
    lines.append(
        "- new_priority_domains: "
        + (", ".join(str(item) for item in knowledge_drift_new_priority_domains) or "none")
    )
    if knowledge_drift_recommendations:
        lines.extend(f"- recommendation: {item}" for item in knowledge_drift_recommendations)
    else:
        lines.append("- recommendation: none")
    lines.extend(["", "## Knowledge Focus Areas", ""])
    focus_areas = payload.get("knowledge_focus_areas") or []
    if focus_areas:
        for row in focus_areas:
            lines.append(
                "- "
                f"`{row.get('component')}` "
                f"status=`{row.get('status')}` "
                f"priority=`{row.get('priority')}` "
                f"missing_metrics=`{', '.join(row.get('missing_metrics') or []) or 'none'}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Operator Adoption Knowledge Drift", ""])
    drift = payload.get("operator_adoption_knowledge_drift") or {}
    lines.append(f"- `status`: `{drift.get('status') or 'unknown'}`")
    lines.append(f"- `summary`: {drift.get('summary') or 'none'}")
    recommendations = drift.get("recommendations") or []
    if recommendations:
        for item in recommendations:
            lines.append(f"- recommendation: {item}")
    else:
        lines.append("- recommendation: none")
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
    lines.extend(["", "## Scorecard Operator Adoption", ""])
    scorecard_operator = payload.get("scorecard_operator_adoption") or {}
    lines.append(f"- `status`: `{scorecard_operator.get('status') or 'unknown'}`")
    lines.append(
        f"- `operator_mode`: `{scorecard_operator.get('operator_mode') or 'unknown'}`"
    )
    lines.append(
        "- `knowledge_outcome_drift_status`: "
        f"`{scorecard_operator.get('knowledge_outcome_drift_status') or 'unknown'}`"
    )
    lines.append(
        "- `knowledge_outcome_drift_summary`: "
        + (scorecard_operator.get("knowledge_outcome_drift_summary") or "none")
    )
    lines.extend(["", "## Operational Operator Adoption", ""])
    operational_operator = payload.get("operational_operator_adoption") or {}
    lines.append(f"- `status`: `{operational_operator.get('status') or 'unknown'}`")
    lines.append(
        "- `knowledge_outcome_drift_status`: "
        f"`{operational_operator.get('knowledge_outcome_drift_status') or 'unknown'}`"
    )
    lines.append(
        "- `knowledge_outcome_drift_summary`: "
        + (operational_operator.get("knowledge_outcome_drift_summary") or "none")
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a compact benchmark artifact bundle."
    )
    parser.add_argument("--title", default="Benchmark Artifact Bundle")
    parser.add_argument("--benchmark-scorecard", default="")
    parser.add_argument("--benchmark-operational-summary", default="")
    parser.add_argument("--benchmark-companion-summary", default="")
    parser.add_argument("--benchmark-release-decision", default="")
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
    parser.add_argument("--benchmark-knowledge-outcome-drift", default="")
    parser.add_argument("--feedback-flywheel", default="")
    parser.add_argument("--assistant-evidence", default="")
    parser.add_argument("--review-queue", default="")
    parser.add_argument("--ocr-review", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    artifact_paths = {
        "benchmark_scorecard": args.benchmark_scorecard,
        "benchmark_operational_summary": args.benchmark_operational_summary,
        "benchmark_companion_summary": args.benchmark_companion_summary,
        "benchmark_release_decision": args.benchmark_release_decision,
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
        "benchmark_knowledge_outcome_drift": args.benchmark_knowledge_outcome_drift,
        "feedback_flywheel": args.feedback_flywheel,
        "assistant_evidence": args.assistant_evidence,
        "review_queue": args.review_queue,
        "ocr_review": args.ocr_review,
    }
    payload = build_bundle(
        title=args.title,
        benchmark_scorecard=_maybe_load_json(args.benchmark_scorecard),
        benchmark_operational_summary=_maybe_load_json(args.benchmark_operational_summary),
        benchmark_companion_summary=_maybe_load_json(args.benchmark_companion_summary),
        benchmark_release_decision=_maybe_load_json(args.benchmark_release_decision),
        benchmark_knowledge_readiness=_maybe_load_json(
            args.benchmark_knowledge_readiness
        ),
        benchmark_knowledge_drift=_maybe_load_json(args.benchmark_knowledge_drift),
        benchmark_engineering_signals=_maybe_load_json(
            args.benchmark_engineering_signals
        ),
        benchmark_realdata_signals=_maybe_load_json(args.benchmark_realdata_signals),
        benchmark_realdata_scorecard=_maybe_load_json(
            args.benchmark_realdata_scorecard
        ),
        benchmark_operator_adoption=_maybe_load_json(args.benchmark_operator_adoption),
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
        benchmark_knowledge_outcome_drift=_maybe_load_json(
            args.benchmark_knowledge_outcome_drift
        ),
        feedback_flywheel=_maybe_load_json(args.feedback_flywheel),
        assistant_evidence=_maybe_load_json(args.assistant_evidence),
        review_queue=_maybe_load_json(args.review_queue),
        ocr_review=_maybe_load_json(args.ocr_review),
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
