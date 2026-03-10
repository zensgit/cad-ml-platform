#!/usr/bin/env python3
"""Export a benchmark companion summary for operators and reviewers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


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


def _compact(items: Iterable[Any], *, limit: int = 5) -> List[str]:
    out: List[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


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
    regressions = _compact(component.get("regressions") or [])
    improvements = _compact(component.get("improvements") or [])
    new_focus_areas = _compact(component.get("new_focus_areas") or [])
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
    regressions = _compact(component.get("regressions") or [])
    improvements = _compact(component.get("improvements") or [])
    new_focus_areas = _compact(component.get("new_focus_areas") or [])
    if regressions:
        parts.append("regressions=" + ", ".join(regressions))
    if improvements:
        parts.append("improvements=" + ", ".join(improvements))
    if new_focus_areas:
        parts.append("new_focus_areas=" + ", ".join(new_focus_areas))
    return "; ".join(parts)


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


def _operator_adoption_release_surface_alignment(
    benchmark_operator_adoption: Dict[str, Any],
) -> Dict[str, Any]:
    alignment = benchmark_operator_adoption.get("release_surface_alignment") or {}
    return {
        "status": str(
            benchmark_operator_adoption.get("release_surface_alignment_status")
            or alignment.get("status")
            or "unknown"
        ),
        "summary": str(
            benchmark_operator_adoption.get("release_surface_alignment_summary")
            or alignment.get("summary")
            or "none"
        ),
        "mismatches": list(alignment.get("mismatches") or []),
        "release_decision": dict(alignment.get("release_decision") or {}),
        "release_runbook": dict(alignment.get("release_runbook") or {}),
    }


def _component_statuses(
    scorecard: Dict[str, Any],
    operational_summary: Dict[str, Any],
    artifact_bundle: Dict[str, Any],
    knowledge_readiness_summary: Dict[str, Any],
    knowledge_drift_summary: Dict[str, Any],
    engineering_signals_summary: Dict[str, Any],
    realdata_signals_summary: Dict[str, Any],
    operator_adoption_summary: Dict[str, Any],
    realdata_scorecard_summary: Dict[str, Any] | None = None,
    knowledge_application_summary: Dict[str, Any] | None = None,
    knowledge_realdata_correlation_summary: Dict[str, Any] | None = None,
    knowledge_domain_matrix_summary: Dict[str, Any] | None = None,
    knowledge_outcome_correlation_summary: Dict[str, Any] | None = None,
    knowledge_outcome_drift_summary: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    realdata_scorecard_summary = realdata_scorecard_summary or {}
    knowledge_application_summary = knowledge_application_summary or {}
    knowledge_realdata_correlation_summary = knowledge_realdata_correlation_summary or {}
    knowledge_domain_matrix_summary = knowledge_domain_matrix_summary or {}
    knowledge_outcome_correlation_summary = (
        knowledge_outcome_correlation_summary or {}
    )
    knowledge_outcome_drift_summary = knowledge_outcome_drift_summary or {}
    scorecard_components = scorecard.get("components") or {}
    operational_components = operational_summary.get("component_statuses") or {}
    bundle_components = artifact_bundle.get("component_statuses") or {}
    knowledge_component = (
        knowledge_readiness_summary.get("knowledge_readiness")
        or knowledge_readiness_summary
        or {}
    )
    knowledge_drift_component = _knowledge_drift_component(knowledge_drift_summary)
    engineering_component = (
        engineering_signals_summary.get("engineering_signals")
        or engineering_signals_summary
        or {}
    )
    realdata_component = (
        realdata_signals_summary.get("realdata_signals")
        or realdata_signals_summary
        or {}
    )
    realdata_scorecard_component = (
        realdata_scorecard_summary.get("realdata_scorecard")
        or realdata_scorecard_summary
        or {}
    )
    knowledge_application_component = (
        knowledge_application_summary.get("knowledge_application")
        or knowledge_application_summary
        or {}
    )
    knowledge_realdata_correlation_component = (
        knowledge_realdata_correlation_summary.get("knowledge_realdata_correlation")
        or knowledge_realdata_correlation_summary
        or {}
    )
    knowledge_domain_matrix_component = (
        knowledge_domain_matrix_summary.get("knowledge_domain_matrix")
        or knowledge_domain_matrix_summary
        or {}
    )
    knowledge_outcome_correlation_component = (
        knowledge_outcome_correlation_summary.get("knowledge_outcome_correlation")
        or knowledge_outcome_correlation_summary
        or {}
    )
    knowledge_outcome_drift_component = _knowledge_outcome_drift_component(
        knowledge_outcome_drift_summary
    )

    def pick(name: str) -> str:
        if isinstance(bundle_components, dict) and bundle_components.get(name):
            return str(bundle_components.get(name))
        if isinstance(operational_components, dict) and operational_components.get(name):
            return str(operational_components.get(name))
        if isinstance(scorecard_components, dict):
            value = scorecard_components.get(name) or {}
            if isinstance(value, dict) and value.get("status"):
                return str(value.get("status"))
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
            bundle_components.get("knowledge_readiness")
            or operational_components.get("knowledge_readiness")
            or knowledge_component.get("status")
            or (scorecard_components.get("knowledge_readiness") or {}).get("status")
            or "unknown"
        ),
        "knowledge_drift": str(
            bundle_components.get("knowledge_drift")
            or knowledge_drift_component.get("status")
            or "unknown"
        ),
        "engineering_signals": str(
            engineering_component.get("status")
            or (scorecard_components.get("engineering_signals") or {}).get("status")
            or "unknown"
        ),
        "realdata_signals": str(realdata_component.get("status") or "unknown"),
        "realdata_scorecard": str(
            bundle_components.get("realdata_scorecard")
            or operational_components.get("realdata_scorecard")
            or realdata_scorecard_component.get("status")
            or (scorecard_components.get("realdata_scorecard") or {}).get("status")
            or "unknown"
        ),
        "operator_adoption": str(
            operator_adoption_summary.get("adoption_readiness")
            or operational_summary.get("component_statuses", {}).get("operator_adoption")
            or ((scorecard.get("components") or {}).get("operator_adoption") or {}).get(
                "status"
            )
            or "unknown"
        ),
        "knowledge_application": str(
            bundle_components.get("knowledge_application")
            or operational_components.get("knowledge_application")
            or knowledge_application_component.get("status")
            or (scorecard_components.get("knowledge_application") or {}).get("status")
            or "unknown"
        ),
        "knowledge_realdata_correlation": str(
            bundle_components.get("knowledge_realdata_correlation")
            or operational_components.get("knowledge_realdata_correlation")
            or knowledge_realdata_correlation_component.get("status")
            or (
                scorecard_components.get("knowledge_realdata_correlation") or {}
            ).get("status")
            or "unknown"
        ),
        "knowledge_domain_matrix": str(
            bundle_components.get("knowledge_domain_matrix")
            or operational_components.get("knowledge_domain_matrix")
            or knowledge_domain_matrix_component.get("status")
            or (scorecard_components.get("knowledge_domain_matrix") or {}).get("status")
            or "unknown"
        ),
        "knowledge_outcome_correlation": str(
            bundle_components.get("knowledge_outcome_correlation")
            or operational_components.get("knowledge_outcome_correlation")
            or knowledge_outcome_correlation_component.get("status")
            or (
                scorecard_components.get("knowledge_outcome_correlation") or {}
            ).get("status")
            or "unknown"
        ),
        "knowledge_outcome_drift": str(
            bundle_components.get("knowledge_outcome_drift")
            or operational_components.get("knowledge_outcome_drift")
            or knowledge_outcome_drift_component.get("status")
            or (
                scorecard_components.get("knowledge_outcome_drift") or {}
            ).get("status")
            or "unknown"
        ),
    }


def _primary_gap(
    component_statuses: Dict[str, str], blockers: List[str], recommendations: List[str]
) -> str:
    if blockers:
        return blockers[0]
    for name, status in component_statuses.items():
        if status in {
            "missing",
            "attention_required",
            "gap_detected",
            "evidence_gap",
            "critical_backlog",
            "managed_backlog",
            "review_heavy",
            "partial_coverage",
            "passive_feedback_only",
            "feedback_collected",
            "baseline_missing",
            "regressed",
            "mixed",
        }:
            return f"{name}:{status}"
    if recommendations:
        return recommendations[0]
    return "none"


def _artifact_rows(
    scorecard_path: str,
    operational_path: str,
    bundle_path: str,
    knowledge_path: str,
    knowledge_drift_path: str,
    engineering_path: str,
    realdata_path: str,
    realdata_scorecard_path: str,
    operator_adoption_path: str,
    knowledge_application_path: str,
    knowledge_realdata_correlation_path: str,
    knowledge_domain_matrix_path: str,
    knowledge_outcome_correlation_path: str,
    knowledge_outcome_drift_path: str,
) -> Dict[str, Dict[str, Any]]:
    def row(name: str, path_text: str) -> Dict[str, Any]:
        path_value = str(path_text or "").strip()
        return {
            "name": name,
            "path": path_value,
            "present": bool(path_value),
        }

    return {
        "benchmark_scorecard": row("benchmark_scorecard", scorecard_path),
        "benchmark_operational_summary": row(
            "benchmark_operational_summary", operational_path
        ),
        "benchmark_artifact_bundle": row("benchmark_artifact_bundle", bundle_path),
        "benchmark_knowledge_readiness": row(
            "benchmark_knowledge_readiness", knowledge_path
        ),
        "benchmark_knowledge_drift": row(
            "benchmark_knowledge_drift", knowledge_drift_path
        ),
        "benchmark_engineering_signals": row(
            "benchmark_engineering_signals", engineering_path
        ),
        "benchmark_realdata_signals": row(
            "benchmark_realdata_signals", realdata_path
        ),
        "benchmark_realdata_scorecard": row(
            "benchmark_realdata_scorecard", realdata_scorecard_path
        ),
        "benchmark_operator_adoption": row(
            "benchmark_operator_adoption", operator_adoption_path
        ),
        "benchmark_knowledge_application": row(
            "benchmark_knowledge_application", knowledge_application_path
        ),
        "benchmark_knowledge_realdata_correlation": row(
            "benchmark_knowledge_realdata_correlation",
            knowledge_realdata_correlation_path,
        ),
        "benchmark_knowledge_domain_matrix": row(
            "benchmark_knowledge_domain_matrix",
            knowledge_domain_matrix_path,
        ),
        "benchmark_knowledge_outcome_correlation": row(
            "benchmark_knowledge_outcome_correlation",
            knowledge_outcome_correlation_path,
        ),
        "benchmark_knowledge_outcome_drift": row(
            "benchmark_knowledge_outcome_drift",
            knowledge_outcome_drift_path,
        ),
    }


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
            limit=5,
        ),
    }


def _operator_adoption_knowledge_outcome_drift(
    benchmark_operator_adoption: Dict[str, Any],
) -> Dict[str, Any]:
    drift = benchmark_operator_adoption.get("knowledge_outcome_drift") or {}
    return {
        "status": str(
            benchmark_operator_adoption.get("knowledge_outcome_drift_status")
            or drift.get("status")
            or "unknown"
        ),
        "summary": str(
            benchmark_operator_adoption.get("knowledge_outcome_drift_summary")
            or drift.get("summary")
            or "none"
        ),
        "recommendations": _compact(
            drift.get("recommendations")
            or benchmark_operator_adoption.get("recommended_actions")
            or [],
            limit=5,
        ),
    }


def build_companion_summary(
    *,
    title: str,
    benchmark_scorecard: Dict[str, Any],
    benchmark_operational_summary: Dict[str, Any],
    benchmark_artifact_bundle: Dict[str, Any],
    benchmark_knowledge_readiness: Dict[str, Any],
    benchmark_knowledge_drift: Dict[str, Any],
    benchmark_engineering_signals: Dict[str, Any],
    benchmark_realdata_signals: Dict[str, Any],
    benchmark_realdata_scorecard: Dict[str, Any] | None = None,
    benchmark_operator_adoption: Dict[str, Any],
    artifact_paths: Dict[str, str],
    benchmark_knowledge_application: Dict[str, Any] | None = None,
    benchmark_knowledge_realdata_correlation: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_matrix: Dict[str, Any] | None = None,
    benchmark_knowledge_outcome_correlation: Dict[str, Any] | None = None,
    benchmark_knowledge_outcome_drift: Dict[str, Any] | None = None,
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
    overall_status = (
        str(benchmark_artifact_bundle.get("overall_status") or "").strip()
        or str(benchmark_operational_summary.get("overall_status") or "").strip()
        or str(benchmark_scorecard.get("overall_status") or "").strip()
        or "unknown"
    )
    bundle_blockers = benchmark_artifact_bundle.get("blockers") or []
    operational_blockers = benchmark_operational_summary.get("blockers") or []
    blockers = _compact(bundle_blockers or operational_blockers, limit=5)
    bundle_recommendations = benchmark_artifact_bundle.get("recommendations") or []
    operational_recommendations = benchmark_operational_summary.get("recommendations") or []
    scorecard_recommendations = benchmark_scorecard.get("recommendations") or []
    knowledge_drift_component = (
        benchmark_artifact_bundle.get("knowledge_drift")
        or _knowledge_drift_component(benchmark_knowledge_drift)
    )
    knowledge_drift_recommendations = (
        benchmark_artifact_bundle.get("knowledge_drift_recommendations")
        or benchmark_knowledge_drift.get("recommendations")
        or []
    )
    engineering_recommendations = benchmark_engineering_signals.get("recommendations") or []
    realdata_root = (
        benchmark_realdata_signals.get("realdata_signals")
        or benchmark_realdata_signals
        or {}
    )
    realdata_scorecard_root = (
        benchmark_realdata_scorecard.get("realdata_scorecard")
        or benchmark_realdata_scorecard
        or {}
    )
    knowledge_application_root = (
        benchmark_knowledge_application.get("knowledge_application")
        or benchmark_knowledge_application
        or {}
    )
    knowledge_realdata_correlation_root = (
        benchmark_knowledge_realdata_correlation.get("knowledge_realdata_correlation")
        or benchmark_knowledge_realdata_correlation
        or {}
    )
    knowledge_domain_matrix_root = (
        benchmark_knowledge_domain_matrix.get("knowledge_domain_matrix")
        or benchmark_knowledge_domain_matrix
        or {}
    )
    knowledge_outcome_correlation_root = (
        benchmark_knowledge_outcome_correlation.get("knowledge_outcome_correlation")
        or benchmark_knowledge_outcome_correlation
        or {}
    )
    knowledge_outcome_drift_root = _knowledge_outcome_drift_component(
        benchmark_knowledge_outcome_drift
    )
    realdata_recommendations = benchmark_realdata_signals.get("recommendations") or []
    knowledge_application_recommendations = (
        benchmark_knowledge_application.get("recommendations") or []
    )
    knowledge_realdata_correlation_recommendations = (
        benchmark_knowledge_realdata_correlation.get("recommendations") or []
    )
    knowledge_domain_matrix_recommendations = (
        benchmark_knowledge_domain_matrix.get("recommendations") or []
    )
    knowledge_outcome_correlation_recommendations = (
        benchmark_knowledge_outcome_correlation.get("recommendations") or []
    )
    knowledge_outcome_drift_recommendations = (
        benchmark_knowledge_outcome_drift.get("recommendations") or []
    )
    realdata_scorecard_recommendations = (
        benchmark_realdata_scorecard.get("recommendations") or []
    )
    operator_adoption_recommendations = (
        benchmark_operator_adoption.get("recommended_actions") or []
    )
    recommendations = _compact(
        bundle_recommendations
        or operational_recommendations
        or knowledge_drift_recommendations
        or knowledge_application_recommendations
        or knowledge_realdata_correlation_recommendations
        or knowledge_domain_matrix_recommendations
        or knowledge_outcome_correlation_recommendations
        or knowledge_outcome_drift_recommendations
        or realdata_scorecard_recommendations
        or operator_adoption_recommendations
        or engineering_recommendations
        or realdata_recommendations
        or scorecard_recommendations,
        limit=5,
    )
    component_statuses = _component_statuses(
        benchmark_scorecard,
        benchmark_operational_summary,
        benchmark_artifact_bundle,
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
    operator_adoption_knowledge_outcome_drift = (
        _operator_adoption_knowledge_outcome_drift(benchmark_operator_adoption)
    )
    scorecard_operator_adoption = _scorecard_operator_adoption(benchmark_scorecard)
    operational_operator_adoption = _operational_operator_adoption(
        benchmark_operational_summary
    )
    operator_adoption_release_surface_alignment = (
        _operator_adoption_release_surface_alignment(benchmark_operator_adoption)
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
        knowledge_application_root.get("focus_areas_detail") or []
    )
    knowledge_application_domains = knowledge_application_root.get("domains") or {}
    knowledge_application_priority_domains = list(
        knowledge_application_root.get("priority_domains") or []
    )
    knowledge_realdata_correlation_domains = (
        knowledge_realdata_correlation_root.get("domains") or {}
    )
    knowledge_realdata_correlation_priority_domains = list(
        knowledge_realdata_correlation_root.get("priority_domains") or []
    )
    knowledge_domain_matrix_domains = knowledge_domain_matrix_root.get("domains") or {}
    knowledge_domain_matrix_priority_domains = list(
        knowledge_domain_matrix_root.get("priority_domains") or []
    )
    knowledge_outcome_correlation_domains = (
        knowledge_outcome_correlation_root.get("domains") or {}
    )
    knowledge_outcome_correlation_priority_domains = list(
        knowledge_outcome_correlation_root.get("priority_domains") or []
    )
    operator_adoption_status = component_statuses.get("operator_adoption")
    primary_gap = _primary_gap(component_statuses, blockers, recommendations)
    review_surface = (
        "ready"
        if component_statuses.get("review_queue") not in {"critical_backlog", "managed_backlog"}
        and component_statuses.get("assistant_explainability")
        not in {"missing", "partial_coverage", "weak_coverage"}
        and component_statuses.get("ocr_review") not in {"missing", "review_heavy"}
        and component_statuses.get("knowledge_readiness")
        not in {"knowledge_foundation_missing", "knowledge_foundation_partial"}
        and component_statuses.get("engineering_signals")
        not in {"unknown", "partial_engineering_semantics", "engineering_signal_gap"}
        and component_statuses.get("knowledge_application")
        not in {"unknown", "knowledge_application_partial", "knowledge_application_missing"}
        and component_statuses.get("knowledge_realdata_correlation")
        not in {
            "unknown",
            "knowledge_realdata_partial",
            "knowledge_realdata_missing",
            "blocked",
        }
        and component_statuses.get("knowledge_domain_matrix")
        not in {
            "unknown",
            "knowledge_domain_matrix_partial",
            "knowledge_domain_matrix_missing",
            "blocked",
        }
        and component_statuses.get("knowledge_outcome_correlation")
        not in {
            "unknown",
            "knowledge_outcome_correlation_partial",
            "knowledge_outcome_correlation_missing",
            "blocked",
        }
        and component_statuses.get("realdata_scorecard")
        not in {
            "realdata_scorecard_partial",
            "realdata_scorecard_missing",
            "environment_blocked",
            "weak",
            "smoke_only",
        }
        and operator_adoption_status not in {"unknown", "guided_manual", "blocked"}
        else "attention_required"
    )
    artifacts = _artifact_rows(
        artifact_paths.get("benchmark_scorecard", ""),
        artifact_paths.get("benchmark_operational_summary", ""),
        artifact_paths.get("benchmark_artifact_bundle", ""),
        artifact_paths.get("benchmark_knowledge_readiness", ""),
        artifact_paths.get("benchmark_knowledge_drift", ""),
        artifact_paths.get("benchmark_engineering_signals", ""),
        artifact_paths.get("benchmark_realdata_signals", ""),
        artifact_paths.get("benchmark_realdata_scorecard", ""),
        artifact_paths.get("benchmark_operator_adoption", ""),
        artifact_paths.get("benchmark_knowledge_application", ""),
        artifact_paths.get("benchmark_knowledge_realdata_correlation", ""),
        artifact_paths.get("benchmark_knowledge_domain_matrix", ""),
        artifact_paths.get("benchmark_knowledge_outcome_correlation", ""),
        artifact_paths.get("benchmark_knowledge_outcome_drift", ""),
    )
    return {
        "title": title,
        "generated_at": int(time.time()),
        "overall_status": overall_status,
        "review_surface": review_surface,
        "primary_gap": primary_gap,
        "component_statuses": component_statuses,
        "knowledge_focus_areas": knowledge_focus_areas,
        "knowledge_drift": knowledge_drift_component,
        "knowledge_drift_summary": (
            benchmark_artifact_bundle.get("knowledge_drift_summary")
            or _knowledge_drift_summary(knowledge_drift_component)
        ),
        "knowledge_drift_recommendations": _compact(
            knowledge_drift_recommendations, limit=5
        ),
        "knowledge_drift_component_changes": list(
            benchmark_artifact_bundle.get("knowledge_drift_component_changes")
            or knowledge_drift_component.get("component_changes")
            or []
        ),
        "knowledge_drift_domain_regressions": list(
            benchmark_artifact_bundle.get("knowledge_drift_domain_regressions")
            or knowledge_drift_component.get("domain_regressions")
            or []
        ),
        "knowledge_drift_domain_improvements": list(
            benchmark_artifact_bundle.get("knowledge_drift_domain_improvements")
            or knowledge_drift_component.get("domain_improvements")
            or []
        ),
        "knowledge_drift_resolved_priority_domains": list(
            benchmark_artifact_bundle.get("knowledge_drift_resolved_priority_domains")
            or knowledge_drift_component.get("resolved_priority_domains")
            or []
        ),
        "knowledge_drift_new_priority_domains": list(
            benchmark_artifact_bundle.get("knowledge_drift_new_priority_domains")
            or knowledge_drift_component.get("new_priority_domains")
            or []
        ),
        "operator_adoption_knowledge_drift": operator_adoption_knowledge_drift,
        "knowledge_domains": knowledge_domains,
        "knowledge_domain_focus_areas": knowledge_domain_focus_areas,
        "knowledge_priority_domains": knowledge_priority_domains,
        "knowledge_application": knowledge_application_root,
        "knowledge_application_status": knowledge_application_root.get("status")
        or "unknown",
        "knowledge_application_focus_areas": knowledge_application_focus_areas,
        "knowledge_application_domains": knowledge_application_domains,
        "knowledge_application_priority_domains": knowledge_application_priority_domains,
        "knowledge_application_recommendations": _compact(
            knowledge_application_recommendations, limit=5
        ),
        "knowledge_realdata_correlation": knowledge_realdata_correlation_root,
        "knowledge_realdata_correlation_status": (
            knowledge_realdata_correlation_root.get("status") or "unknown"
        ),
        "knowledge_realdata_correlation_domains": knowledge_realdata_correlation_domains,
        "knowledge_realdata_correlation_priority_domains": (
            knowledge_realdata_correlation_priority_domains
        ),
        "knowledge_realdata_correlation_recommendations": _compact(
            knowledge_realdata_correlation_recommendations,
            limit=5,
        ),
        "knowledge_domain_matrix": knowledge_domain_matrix_root,
        "knowledge_domain_matrix_status": (
            knowledge_domain_matrix_root.get("status") or "unknown"
        ),
        "knowledge_domain_matrix_domains": knowledge_domain_matrix_domains,
        "knowledge_domain_matrix_priority_domains": (
            knowledge_domain_matrix_priority_domains
        ),
        "knowledge_domain_matrix_recommendations": _compact(
            knowledge_domain_matrix_recommendations,
            limit=5,
        ),
        "knowledge_outcome_correlation": knowledge_outcome_correlation_root,
        "knowledge_outcome_correlation_status": (
            knowledge_outcome_correlation_root.get("status") or "unknown"
        ),
        "knowledge_outcome_correlation_domains": knowledge_outcome_correlation_domains,
        "knowledge_outcome_correlation_priority_domains": (
            knowledge_outcome_correlation_priority_domains
        ),
        "knowledge_outcome_correlation_recommendations": _compact(
            knowledge_outcome_correlation_recommendations,
            limit=5,
        ),
        "knowledge_outcome_drift": knowledge_outcome_drift_root,
        "knowledge_outcome_drift_status": (
            knowledge_outcome_drift_root.get("status") or "unknown"
        ),
        "knowledge_outcome_drift_summary": _knowledge_outcome_drift_summary(
            knowledge_outcome_drift_root
        ),
        "knowledge_outcome_drift_domain_regressions": list(
            knowledge_outcome_drift_root.get("domain_regressions") or []
        ),
        "knowledge_outcome_drift_domain_improvements": list(
            knowledge_outcome_drift_root.get("domain_improvements") or []
        ),
        "knowledge_outcome_drift_resolved_priority_domains": list(
            knowledge_outcome_drift_root.get("resolved_priority_domains") or []
        ),
        "knowledge_outcome_drift_new_priority_domains": list(
            knowledge_outcome_drift_root.get("new_priority_domains") or []
        ),
        "knowledge_outcome_drift_recommendations": _compact(
            knowledge_outcome_drift_recommendations,
            limit=5,
        ),
        "realdata_signals": realdata_root,
        "realdata_status": realdata_root.get("status") or "unknown",
        "realdata_recommendations": _compact(realdata_recommendations, limit=5),
        "realdata_scorecard": realdata_scorecard_root,
        "realdata_scorecard_status": (
            realdata_scorecard_root.get("status") or "unknown"
        ),
        "realdata_scorecard_recommendations": _compact(
            realdata_scorecard_recommendations, limit=5
        ),
        "operator_adoption_knowledge_outcome_drift": (
            operator_adoption_knowledge_outcome_drift
        ),
        "operator_adoption_release_surface_alignment": (
            operator_adoption_release_surface_alignment
        ),
        "scorecard_operator_adoption": scorecard_operator_adoption,
        "operational_operator_adoption": operational_operator_adoption,
        "recommended_actions": recommendations,
        "blockers": blockers,
        "artifacts": artifacts,
    }


def render_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        f"# {payload.get('title') or 'Benchmark Companion Summary'}",
        "",
        f"- `overall_status`: `{payload.get('overall_status')}`",
        f"- `review_surface`: `{payload.get('review_surface')}`",
        f"- `primary_gap`: `{payload.get('primary_gap')}`",
        f"- `knowledge_drift_summary`: `{payload.get('knowledge_drift_summary') or 'none'}`",
        f"- `knowledge_application_status`: "
        f"`{payload.get('knowledge_application_status') or 'unknown'}`",
        f"- `knowledge_realdata_correlation_status`: "
        f"`{payload.get('knowledge_realdata_correlation_status') or 'unknown'}`",
        f"- `knowledge_domain_matrix_status`: "
        f"`{payload.get('knowledge_domain_matrix_status') or 'unknown'}`",
        f"- `knowledge_outcome_correlation_status`: "
        f"`{payload.get('knowledge_outcome_correlation_status') or 'unknown'}`",
        f"- `knowledge_outcome_drift_status`: "
        f"`{payload.get('knowledge_outcome_drift_status') or 'unknown'}`",
        f"- `realdata_status`: `{payload.get('realdata_status') or 'unknown'}`",
        f"- `realdata_scorecard_status`: "
        f"`{payload.get('realdata_scorecard_status') or 'unknown'}`",
        "",
        "## Component Statuses",
        "",
    ]
    for name, status in (payload.get("component_statuses") or {}).items():
        lines.append(f"- `{name}`: `{status}`")
    lines.extend(["", "## Blockers", ""])
    blockers = payload.get("blockers") or []
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- none")
    lines.extend(["", "## Knowledge Drift", ""])
    if payload.get("knowledge_drift_summary"):
        lines.append(f"- summary: {payload.get('knowledge_drift_summary')}")
    else:
        lines.append("- summary: none")
    drift_changes = payload.get("knowledge_drift_component_changes") or []
    drift_domain_regressions = payload.get("knowledge_drift_domain_regressions") or []
    drift_domain_improvements = payload.get("knowledge_drift_domain_improvements") or []
    drift_resolved_priority_domains = (
        payload.get("knowledge_drift_resolved_priority_domains") or []
    )
    drift_new_priority_domains = payload.get("knowledge_drift_new_priority_domains") or []
    if drift_changes:
        for row in drift_changes:
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
        + (", ".join(str(item) for item in drift_domain_regressions) or "none")
    )
    lines.append(
        "- domain_improvements: "
        + (", ".join(str(item) for item in drift_domain_improvements) or "none")
    )
    lines.append(
        "- resolved_priority_domains: "
        + (", ".join(str(item) for item in drift_resolved_priority_domains) or "none")
    )
    lines.append(
        "- new_priority_domains: "
        + (", ".join(str(item) for item in drift_new_priority_domains) or "none")
    )
    drift_recommendations = payload.get("knowledge_drift_recommendations") or []
    if drift_recommendations:
        lines.extend(f"- recommendation: {item}" for item in drift_recommendations)
    else:
        lines.append("- recommendation: none")
    lines.extend(["", "## Real-Data Signals", ""])
    realdata = payload.get("realdata_signals") or {}
    lines.append(f"- `status`: `{payload.get('realdata_status') or 'unknown'}`")
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
    realdata_recommendations = payload.get("realdata_recommendations") or []
    if realdata_recommendations:
        for item in realdata_recommendations:
            lines.append(f"- recommendation: {item}")
    else:
        lines.append("- recommendation: none")
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
        for item in realdata_scorecard_recommendations:
            lines.append(f"- recommendation: {item}")
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
                f"action=`{row.get('action')}`"
            )
    else:
        lines.append("- none")
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
        "- `status`: "
        f"`{payload.get('knowledge_domain_matrix_status') or 'unknown'}`"
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
    lines.extend(["", "## Knowledge Outcome Drift", ""])
    lines.append(
        f"- `status`: `{payload.get('knowledge_outcome_drift_status') or 'unknown'}`"
    )
    lines.append(
        "- `summary`: "
        + (_text(payload.get("knowledge_outcome_drift_summary")) or "none")
    )
    for label in (
        "knowledge_outcome_drift_domain_regressions",
        "knowledge_outcome_drift_domain_improvements",
        "knowledge_outcome_drift_resolved_priority_domains",
        "knowledge_outcome_drift_new_priority_domains",
    ):
        values = payload.get(label) or []
        lines.append(
            f"- `{label}`: `{', '.join(str(item) for item in values) or 'none'}`"
        )
    knowledge_outcome_drift_recommendations = (
        payload.get("knowledge_outcome_drift_recommendations") or []
    )
    if knowledge_outcome_drift_recommendations:
        for item in knowledge_outcome_drift_recommendations:
            lines.append(f"- recommendation: {item}")
    else:
        lines.append("- recommendation: none")
    lines.extend(["", "## Operator Adoption Knowledge Drift", ""])
    drift = payload.get("operator_adoption_knowledge_drift") or {}
    lines.append(f"- `status`: `{drift.get('status') or 'unknown'}`")
    lines.append(f"- `summary`: {drift.get('summary') or 'none'}")
    drift_recommendations = drift.get("recommendations") or []
    if drift_recommendations:
        for item in drift_recommendations:
            lines.append(f"- recommendation: {item}")
    else:
        lines.append("- recommendation: none")
    lines.extend(["", "## Operator Adoption Knowledge Outcome Drift", ""])
    outcome_drift = payload.get("operator_adoption_knowledge_outcome_drift") or {}
    lines.append(f"- `status`: `{outcome_drift.get('status') or 'unknown'}`")
    lines.append(f"- `summary`: {outcome_drift.get('summary') or 'none'}")
    outcome_drift_recommendations = outcome_drift.get("recommendations") or []
    if outcome_drift_recommendations:
        for item in outcome_drift_recommendations:
            lines.append(f"- recommendation: {item}")
    else:
        lines.append("- recommendation: none")
    lines.extend(["", "## Operator Adoption Release Surface Alignment", ""])
    alignment = payload.get("operator_adoption_release_surface_alignment") or {}
    lines.append(f"- `status`: `{alignment.get('status') or 'unknown'}`")
    lines.append(f"- `summary`: {alignment.get('summary') or 'none'}")
    mismatches = alignment.get("mismatches") or []
    lines.append(
        "- `mismatches`: "
        + (", ".join(str(item) for item in mismatches) if mismatches else "none")
    )
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
    lines.extend(["", "## Recommended Actions", ""])
    actions = payload.get("recommended_actions") or []
    if actions:
        lines.extend(f"- {item}" for item in actions)
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
        description="Export a benchmark companion summary."
    )
    parser.add_argument("--title", default="Benchmark Companion Summary")
    parser.add_argument("--benchmark-scorecard", default="")
    parser.add_argument("--benchmark-operational-summary", default="")
    parser.add_argument("--benchmark-artifact-bundle", default="")
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
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    artifact_paths = {
        "benchmark_scorecard": args.benchmark_scorecard,
        "benchmark_operational_summary": args.benchmark_operational_summary,
        "benchmark_artifact_bundle": args.benchmark_artifact_bundle,
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
    }
    payload = build_companion_summary(
        title=args.title,
        benchmark_scorecard=_maybe_load_json(args.benchmark_scorecard),
        benchmark_operational_summary=_maybe_load_json(
            args.benchmark_operational_summary
        ),
        benchmark_artifact_bundle=_maybe_load_json(args.benchmark_artifact_bundle),
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
