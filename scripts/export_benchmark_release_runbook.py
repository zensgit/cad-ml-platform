#!/usr/bin/env python3
"""Export an operator-facing benchmark release runbook."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


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


def _compact(items: Iterable[Any], *, limit: int = 6) -> List[str]:
    out: List[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _artifact_row(name: str, path_text: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    path_value = _text(path_text)
    return {
        "name": name,
        "path": path_value,
        "present": bool(path_value) or bool(payload),
    }


def _knowledge_domain_capability_drift_component(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    return payload.get("knowledge_domain_capability_drift") or payload or {}


def _knowledge_domain_control_plane_component(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    return payload.get("knowledge_domain_control_plane") or payload or {}


def _knowledge_domain_control_plane_drift_component(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    return payload.get("knowledge_domain_control_plane_drift") or payload or {}


def _knowledge_domain_release_gate_component(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    return payload.get("knowledge_domain_release_gate") or payload or {}


def _knowledge_domain_release_readiness_matrix_component(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    return payload.get("knowledge_domain_release_readiness_matrix") or payload or {}


def _knowledge_domain_release_surface_alignment(
    benchmark_knowledge_domain_release_surface_alignment: Dict[str, Any],
) -> Dict[str, Any]:
    alignment = (
        benchmark_knowledge_domain_release_surface_alignment.get(
            "knowledge_domain_release_surface_alignment"
        )
        or benchmark_knowledge_domain_release_surface_alignment
        or {}
    )
    return {
        "status": str(alignment.get("status") or "unknown"),
        "summary": str(alignment.get("summary") or "none"),
        "mismatches": list(alignment.get("mismatches") or []),
        "domain_mismatches": list(alignment.get("domain_mismatches") or []),
        "release_blocker_mismatches": list(
            alignment.get("release_blocker_mismatches") or []
        ),
        "release_decision": dict(alignment.get("release_decision") or {}),
        "release_runbook": dict(alignment.get("release_runbook") or {}),
    }


def _artifacts(
    *,
    benchmark_release_decision: Dict[str, Any],
    benchmark_companion_summary: Dict[str, Any],
    benchmark_artifact_bundle: Dict[str, Any],
    benchmark_knowledge_readiness: Dict[str, Any],
    benchmark_knowledge_drift: Dict[str, Any],
    benchmark_engineering_signals: Dict[str, Any],
    benchmark_realdata_signals: Dict[str, Any],
    benchmark_operator_adoption: Dict[str, Any],
    benchmark_realdata_scorecard: Dict[str, Any] | None = None,
    benchmark_knowledge_application: Dict[str, Any] | None = None,
    benchmark_knowledge_realdata_correlation: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_matrix: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_capability_matrix: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_validation_matrix: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_capability_drift: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_action_plan: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_control_plane: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_control_plane_drift: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_release_gate: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_release_readiness_matrix: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_release_surface_alignment: Dict[str, Any] | None = None,
    benchmark_knowledge_reference_inventory: Dict[str, Any] | None = None,
    benchmark_knowledge_source_action_plan: Dict[str, Any] | None = None,
    benchmark_knowledge_source_coverage: Dict[str, Any] | None = None,
    benchmark_knowledge_source_drift: Dict[str, Any] | None = None,
    benchmark_knowledge_outcome_correlation: Dict[str, Any] | None = None,
    benchmark_knowledge_outcome_drift: Dict[str, Any] | None = None,
    benchmark_competitive_surpass_index: Dict[str, Any] | None = None,
    benchmark_competitive_surpass_trend: Dict[str, Any] | None = None,
    benchmark_competitive_surpass_action_plan: Dict[str, Any] | None = None,
    artifact_paths: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    benchmark_realdata_scorecard = benchmark_realdata_scorecard or {}
    benchmark_knowledge_application = benchmark_knowledge_application or {}
    benchmark_knowledge_realdata_correlation = (
        benchmark_knowledge_realdata_correlation or {}
    )
    benchmark_knowledge_domain_matrix = benchmark_knowledge_domain_matrix or {}
    benchmark_knowledge_domain_capability_matrix = (
        benchmark_knowledge_domain_capability_matrix or {}
    )
    benchmark_knowledge_domain_validation_matrix = (
        benchmark_knowledge_domain_validation_matrix or {}
    )
    benchmark_knowledge_domain_capability_drift = (
        benchmark_knowledge_domain_capability_drift or {}
    )
    benchmark_knowledge_domain_action_plan = (
        benchmark_knowledge_domain_action_plan or {}
    )
    benchmark_knowledge_domain_control_plane = (
        benchmark_knowledge_domain_control_plane or {}
    )
    benchmark_knowledge_domain_control_plane_drift = (
        benchmark_knowledge_domain_control_plane_drift or {}
    )
    benchmark_knowledge_domain_release_gate = (
        benchmark_knowledge_domain_release_gate or {}
    )
    benchmark_knowledge_domain_release_readiness_matrix = (
        benchmark_knowledge_domain_release_readiness_matrix or {}
    )
    benchmark_knowledge_domain_release_surface_alignment = (
        benchmark_knowledge_domain_release_surface_alignment or {}
    )
    benchmark_knowledge_reference_inventory = (
        benchmark_knowledge_reference_inventory or {}
    )
    benchmark_knowledge_source_action_plan = (
        benchmark_knowledge_source_action_plan or {}
    )
    benchmark_knowledge_source_coverage = (
        benchmark_knowledge_source_coverage or {}
    )
    benchmark_knowledge_source_drift = benchmark_knowledge_source_drift or {}
    benchmark_knowledge_outcome_correlation = (
        benchmark_knowledge_outcome_correlation or {}
    )
    benchmark_knowledge_outcome_drift = benchmark_knowledge_outcome_drift or {}
    benchmark_competitive_surpass_index = benchmark_competitive_surpass_index or {}
    benchmark_competitive_surpass_trend = benchmark_competitive_surpass_trend or {}
    benchmark_competitive_surpass_action_plan = (
        benchmark_competitive_surpass_action_plan or {}
    )
    decision_artifacts = benchmark_release_decision.get("artifacts") or {}

    def pick_path(name: str) -> str:
        direct = _text(artifact_paths.get(name, ""))
        if direct:
            return direct
        nested = decision_artifacts.get(name) or {}
        if isinstance(nested, dict):
            return _text(nested.get("path"))
        return ""

    return {
        "benchmark_release_decision": _artifact_row(
            "benchmark_release_decision",
            artifact_paths.get("benchmark_release_decision", ""),
            benchmark_release_decision,
        ),
        "benchmark_companion_summary": _artifact_row(
            "benchmark_companion_summary",
            pick_path("benchmark_companion_summary"),
            benchmark_companion_summary,
        ),
        "benchmark_artifact_bundle": _artifact_row(
            "benchmark_artifact_bundle",
            pick_path("benchmark_artifact_bundle"),
            benchmark_artifact_bundle,
        ),
        "benchmark_knowledge_readiness": _artifact_row(
            "benchmark_knowledge_readiness",
            artifact_paths.get("benchmark_knowledge_readiness", ""),
            benchmark_knowledge_readiness,
        ),
        "benchmark_knowledge_drift": _artifact_row(
            "benchmark_knowledge_drift",
            artifact_paths.get("benchmark_knowledge_drift", ""),
            benchmark_knowledge_drift,
        ),
        "benchmark_engineering_signals": _artifact_row(
            "benchmark_engineering_signals",
            pick_path("benchmark_engineering_signals")
            or artifact_paths.get("benchmark_engineering_signals", ""),
            benchmark_engineering_signals,
        ),
        "benchmark_realdata_signals": _artifact_row(
            "benchmark_realdata_signals",
            pick_path("benchmark_realdata_signals")
            or artifact_paths.get("benchmark_realdata_signals", ""),
            benchmark_realdata_signals,
        ),
        "benchmark_realdata_scorecard": _artifact_row(
            "benchmark_realdata_scorecard",
            artifact_paths.get("benchmark_realdata_scorecard", ""),
            benchmark_realdata_scorecard,
        ),
        "benchmark_operator_adoption": _artifact_row(
            "benchmark_operator_adoption",
            artifact_paths.get("benchmark_operator_adoption", ""),
            benchmark_operator_adoption,
        ),
        "benchmark_knowledge_application": _artifact_row(
            "benchmark_knowledge_application",
            artifact_paths.get("benchmark_knowledge_application", ""),
            benchmark_knowledge_application,
        ),
        "benchmark_knowledge_realdata_correlation": _artifact_row(
            "benchmark_knowledge_realdata_correlation",
            artifact_paths.get("benchmark_knowledge_realdata_correlation", ""),
            benchmark_knowledge_realdata_correlation,
        ),
        "benchmark_knowledge_domain_matrix": _artifact_row(
            "benchmark_knowledge_domain_matrix",
            artifact_paths.get("benchmark_knowledge_domain_matrix", ""),
            benchmark_knowledge_domain_matrix,
        ),
        "benchmark_knowledge_domain_capability_matrix": _artifact_row(
            "benchmark_knowledge_domain_capability_matrix",
            artifact_paths.get("benchmark_knowledge_domain_capability_matrix", ""),
            benchmark_knowledge_domain_capability_matrix,
        ),
        "benchmark_knowledge_domain_capability_drift": _artifact_row(
            "benchmark_knowledge_domain_capability_drift",
            artifact_paths.get("benchmark_knowledge_domain_capability_drift", ""),
            benchmark_knowledge_domain_capability_drift,
        ),
        "benchmark_knowledge_domain_action_plan": _artifact_row(
            "benchmark_knowledge_domain_action_plan",
            artifact_paths.get("benchmark_knowledge_domain_action_plan", ""),
            benchmark_knowledge_domain_action_plan,
        ),
        "benchmark_knowledge_domain_control_plane": _artifact_row(
            "benchmark_knowledge_domain_control_plane",
            artifact_paths.get("benchmark_knowledge_domain_control_plane", ""),
            benchmark_knowledge_domain_control_plane,
        ),
        "benchmark_knowledge_domain_control_plane_drift": _artifact_row(
            "benchmark_knowledge_domain_control_plane_drift",
            artifact_paths.get(
                "benchmark_knowledge_domain_control_plane_drift",
                "",
            ),
            benchmark_knowledge_domain_control_plane_drift,
        ),
        "benchmark_knowledge_domain_release_gate": _artifact_row(
            "benchmark_knowledge_domain_release_gate",
            artifact_paths.get("benchmark_knowledge_domain_release_gate", ""),
            benchmark_knowledge_domain_release_gate,
        ),
        "benchmark_knowledge_domain_release_readiness_matrix": _artifact_row(
            "benchmark_knowledge_domain_release_readiness_matrix",
            artifact_paths.get(
                "benchmark_knowledge_domain_release_readiness_matrix",
                "",
            ),
            benchmark_knowledge_domain_release_readiness_matrix,
        ),
        "benchmark_knowledge_domain_release_surface_alignment": _artifact_row(
            "benchmark_knowledge_domain_release_surface_alignment",
            artifact_paths.get(
                "benchmark_knowledge_domain_release_surface_alignment",
                "",
            ),
            benchmark_knowledge_domain_release_surface_alignment,
        ),
        "benchmark_knowledge_reference_inventory": _artifact_row(
            "benchmark_knowledge_reference_inventory",
            artifact_paths.get("benchmark_knowledge_reference_inventory", ""),
            benchmark_knowledge_reference_inventory,
        ),
        "benchmark_knowledge_domain_validation_matrix": _artifact_row(
            "benchmark_knowledge_domain_validation_matrix",
            artifact_paths.get("benchmark_knowledge_domain_validation_matrix", ""),
            benchmark_knowledge_domain_validation_matrix,
        ),
        "benchmark_knowledge_source_action_plan": _artifact_row(
            "benchmark_knowledge_source_action_plan",
            artifact_paths.get("benchmark_knowledge_source_action_plan", ""),
            benchmark_knowledge_source_action_plan,
        ),
        "benchmark_knowledge_source_coverage": _artifact_row(
            "benchmark_knowledge_source_coverage",
            artifact_paths.get("benchmark_knowledge_source_coverage", ""),
            benchmark_knowledge_source_coverage,
        ),
        "benchmark_knowledge_source_drift": _artifact_row(
            "benchmark_knowledge_source_drift",
            artifact_paths.get("benchmark_knowledge_source_drift", ""),
            benchmark_knowledge_source_drift,
        ),
        "benchmark_knowledge_outcome_correlation": _artifact_row(
            "benchmark_knowledge_outcome_correlation",
            artifact_paths.get("benchmark_knowledge_outcome_correlation", ""),
            benchmark_knowledge_outcome_correlation,
        ),
        "benchmark_knowledge_outcome_drift": _artifact_row(
            "benchmark_knowledge_outcome_drift",
            artifact_paths.get("benchmark_knowledge_outcome_drift", ""),
            benchmark_knowledge_outcome_drift,
        ),
        "benchmark_competitive_surpass_index": _artifact_row(
            "benchmark_competitive_surpass_index",
            artifact_paths.get("benchmark_competitive_surpass_index", ""),
            benchmark_competitive_surpass_index,
        ),
        "benchmark_competitive_surpass_trend": _artifact_row(
            "benchmark_competitive_surpass_trend",
            artifact_paths.get("benchmark_competitive_surpass_trend", ""),
            benchmark_competitive_surpass_trend,
        ),
        "benchmark_competitive_surpass_action_plan": _artifact_row(
            "benchmark_competitive_surpass_action_plan",
            artifact_paths.get("benchmark_competitive_surpass_action_plan", ""),
            benchmark_competitive_surpass_action_plan,
        ),
        "benchmark_scorecard": _artifact_row(
            "benchmark_scorecard",
            pick_path("benchmark_scorecard"),
            decision_artifacts.get("benchmark_scorecard") or {},
        ),
        "benchmark_operational_summary": _artifact_row(
            "benchmark_operational_summary",
            pick_path("benchmark_operational_summary"),
            decision_artifacts.get("benchmark_operational_summary") or {},
        ),
    }


def _release_status(
    benchmark_release_decision: Dict[str, Any],
    benchmark_companion_summary: Dict[str, Any],
    benchmark_artifact_bundle: Dict[str, Any],
) -> str:
    return (
        str(benchmark_release_decision.get("release_status") or "").strip()
        or str(benchmark_companion_summary.get("overall_status") or "").strip()
        or str(benchmark_artifact_bundle.get("overall_status") or "").strip()
        or "unknown"
    )


def _primary_signal_source(
    benchmark_release_decision: Dict[str, Any],
    benchmark_companion_summary: Dict[str, Any],
    benchmark_artifact_bundle: Dict[str, Any],
) -> str:
    return (
        str(benchmark_release_decision.get("primary_signal_source") or "").strip()
        or ("benchmark_companion_summary" if benchmark_companion_summary else "")
        or ("benchmark_artifact_bundle" if benchmark_artifact_bundle else "")
        or "none"
    )


def _operator_adoption_payload(
    benchmark_operator_adoption: Dict[str, Any]
) -> Dict[str, Any]:
    status = (
        _text(benchmark_operator_adoption.get("status"))
        or _text(benchmark_operator_adoption.get("overall_status"))
        or _text(benchmark_operator_adoption.get("adoption_status"))
        or ("provided" if benchmark_operator_adoption else "unknown")
    )
    summary = (
        _text(benchmark_operator_adoption.get("summary"))
        or _text(benchmark_operator_adoption.get("headline"))
        or _text(benchmark_operator_adoption.get("status_summary"))
    )
    signals = _compact(
        benchmark_operator_adoption.get("signals")
        or benchmark_operator_adoption.get("adoption_signals")
        or benchmark_operator_adoption.get("risks")
        or [],
    )
    actions = _compact(
        benchmark_operator_adoption.get("actions")
        or benchmark_operator_adoption.get("recommended_actions")
        or benchmark_operator_adoption.get("guidance")
        or [],
    )
    knowledge_drift = benchmark_operator_adoption.get("knowledge_drift") or {}
    knowledge_outcome_drift = (
        benchmark_operator_adoption.get("knowledge_outcome_drift") or {}
    )
    return {
        "status": status,
        "summary": summary,
        "signals": signals,
        "actions": actions,
        "knowledge_drift_status": _text(
            benchmark_operator_adoption.get("knowledge_drift_status")
        )
        or _text(knowledge_drift.get("status"))
        or "unknown",
        "knowledge_drift_summary": _text(
            benchmark_operator_adoption.get("knowledge_drift_summary")
        )
        or _text(knowledge_drift.get("summary"))
        or "none",
        "knowledge_drift_recommendations": _compact(
            knowledge_drift.get("recommendations")
            or benchmark_operator_adoption.get("recommended_actions")
            or []
        ),
        "knowledge_outcome_drift_status": _text(
            benchmark_operator_adoption.get("knowledge_outcome_drift_status")
        )
        or _text(knowledge_outcome_drift.get("status"))
        or "unknown",
        "knowledge_outcome_drift_summary": _text(
            benchmark_operator_adoption.get("knowledge_outcome_drift_summary")
        )
        or _text(knowledge_outcome_drift.get("summary"))
        or "none",
        "knowledge_outcome_drift_recommendations": _compact(
            knowledge_outcome_drift.get("recommendations")
            or benchmark_operator_adoption.get("recommended_actions")
            or []
        ),
        "has_guidance": bool(signals or actions),
    }


def _scorecard_operator_adoption(
    benchmark_scorecard: Dict[str, Any],
) -> Dict[str, Any]:
    component = (benchmark_scorecard.get("components") or {}).get("operator_adoption") or {}
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
    benchmark_operational_summary: Dict[str, Any],
) -> Dict[str, Any]:
    components = benchmark_operational_summary.get("component_statuses") or {}
    return {
        "status": str(components.get("operator_adoption") or "unknown"),
        "knowledge_outcome_drift_status": str(
            benchmark_operational_summary.get(
                "operator_adoption_knowledge_outcome_drift_status"
            )
            or "unknown"
        ),
        "knowledge_outcome_drift_summary": str(
            benchmark_operational_summary.get(
                "operator_adoption_knowledge_outcome_drift_summary"
            )
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


def _step(
    *,
    order: int,
    key: str,
    status: str,
    title: str,
    reason: str,
    action: str,
) -> Dict[str, Any]:
    return {
        "order": order,
        "key": key,
        "status": status,
        "title": title,
        "reason": reason,
        "action": action,
    }


def build_release_runbook(
    *,
    title: str,
    benchmark_release_decision: Dict[str, Any],
    benchmark_scorecard: Dict[str, Any] | None = None,
    benchmark_operational_summary: Dict[str, Any] | None = None,
    benchmark_companion_summary: Dict[str, Any],
    benchmark_artifact_bundle: Dict[str, Any],
    benchmark_knowledge_readiness: Dict[str, Any],
    benchmark_knowledge_drift: Dict[str, Any],
    benchmark_engineering_signals: Dict[str, Any],
    benchmark_realdata_signals: Dict[str, Any],
    benchmark_realdata_scorecard: Dict[str, Any] | None = None,
    benchmark_operator_adoption: Dict[str, Any] | None = None,
    benchmark_knowledge_application: Dict[str, Any] | None = None,
    benchmark_knowledge_realdata_correlation: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_matrix: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_capability_matrix: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_validation_matrix: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_capability_drift: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_action_plan: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_control_plane: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_control_plane_drift: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_release_gate: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_release_readiness_matrix: Dict[str, Any] | None = None,
    benchmark_knowledge_domain_release_surface_alignment: Dict[str, Any] | None = None,
    benchmark_knowledge_reference_inventory: Dict[str, Any] | None = None,
    benchmark_knowledge_source_action_plan: Dict[str, Any] | None = None,
    benchmark_knowledge_source_coverage: Dict[str, Any] | None = None,
    benchmark_knowledge_source_drift: Dict[str, Any] | None = None,
    benchmark_knowledge_outcome_correlation: Dict[str, Any] | None = None,
    benchmark_knowledge_outcome_drift: Dict[str, Any] | None = None,
    benchmark_competitive_surpass_index: Dict[str, Any] | None = None,
    benchmark_competitive_surpass_trend: Dict[str, Any] | None = None,
    benchmark_competitive_surpass_action_plan: Dict[str, Any] | None = None,
    artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
    benchmark_scorecard = benchmark_scorecard or {}
    benchmark_operational_summary = benchmark_operational_summary or {}
    benchmark_realdata_scorecard = benchmark_realdata_scorecard or {}
    benchmark_operator_adoption = benchmark_operator_adoption or {}
    benchmark_knowledge_application = benchmark_knowledge_application or {}
    benchmark_knowledge_realdata_correlation = (
        benchmark_knowledge_realdata_correlation or {}
    )
    benchmark_knowledge_domain_matrix = benchmark_knowledge_domain_matrix or {}
    benchmark_knowledge_domain_capability_matrix = (
        benchmark_knowledge_domain_capability_matrix or {}
    )
    benchmark_knowledge_domain_validation_matrix = (
        benchmark_knowledge_domain_validation_matrix or {}
    )
    benchmark_knowledge_domain_capability_drift = (
        benchmark_knowledge_domain_capability_drift or {}
    )
    benchmark_knowledge_domain_action_plan = (
        benchmark_knowledge_domain_action_plan or {}
    )
    benchmark_knowledge_domain_control_plane = (
        benchmark_knowledge_domain_control_plane or {}
    )
    benchmark_knowledge_domain_control_plane_drift = (
        benchmark_knowledge_domain_control_plane_drift or {}
    )
    benchmark_knowledge_domain_release_gate = (
        benchmark_knowledge_domain_release_gate or {}
    )
    benchmark_knowledge_domain_release_readiness_matrix = (
        benchmark_knowledge_domain_release_readiness_matrix or {}
    )
    benchmark_knowledge_domain_release_surface_alignment = (
        benchmark_knowledge_domain_release_surface_alignment or {}
    )
    benchmark_knowledge_reference_inventory = (
        benchmark_knowledge_reference_inventory or {}
    )
    benchmark_knowledge_source_action_plan = (
        benchmark_knowledge_source_action_plan or {}
    )
    benchmark_knowledge_source_coverage = (
        benchmark_knowledge_source_coverage or {}
    )
    benchmark_knowledge_source_drift = benchmark_knowledge_source_drift or {}
    benchmark_knowledge_outcome_correlation = (
        benchmark_knowledge_outcome_correlation or {}
    )
    benchmark_knowledge_outcome_drift = benchmark_knowledge_outcome_drift or {}
    benchmark_competitive_surpass_index = benchmark_competitive_surpass_index or {}
    benchmark_competitive_surpass_trend = benchmark_competitive_surpass_trend or {}
    benchmark_competitive_surpass_action_plan = (
        benchmark_competitive_surpass_action_plan or {}
    )
    knowledge_component = (
        benchmark_knowledge_readiness.get("knowledge_readiness")
        or benchmark_knowledge_readiness
        or {}
    )
    knowledge_status = (
        str(knowledge_component.get("status") or "unknown").strip() or "unknown"
    )
    knowledge_focus_areas = list(knowledge_component.get("focus_areas_detail") or [])
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
    knowledge_domain_capability_matrix_component = (
        benchmark_knowledge_domain_capability_matrix.get(
            "knowledge_domain_capability_matrix"
        )
        or benchmark_knowledge_domain_capability_matrix
        or {}
    )
    knowledge_domain_capability_drift_component = (
        _knowledge_domain_capability_drift_component(
            benchmark_knowledge_domain_capability_drift
        )
    )
    knowledge_domain_action_plan_component = (
        benchmark_knowledge_domain_action_plan.get("knowledge_domain_action_plan")
        or benchmark_knowledge_domain_action_plan
        or {}
    )
    knowledge_domain_control_plane_component = (
        _knowledge_domain_control_plane_component(
            benchmark_knowledge_domain_control_plane
        )
    )
    knowledge_domain_control_plane_drift_component = (
        _knowledge_domain_control_plane_drift_component(
            benchmark_knowledge_domain_control_plane_drift
        )
    )
    knowledge_domain_release_gate_component = _knowledge_domain_release_gate_component(
        benchmark_knowledge_domain_release_gate
    )
    knowledge_domain_release_readiness_matrix_component = (
        _knowledge_domain_release_readiness_matrix_component(
            benchmark_knowledge_domain_release_readiness_matrix
        )
    )
    knowledge_domain_release_surface_alignment_component = (
        _knowledge_domain_release_surface_alignment(
            benchmark_knowledge_domain_release_surface_alignment
        )
    )
    knowledge_reference_inventory_component = (
        benchmark_knowledge_reference_inventory.get("knowledge_reference_inventory")
        or benchmark_knowledge_reference_inventory
        or {}
    )
    knowledge_domain_validation_matrix_component = (
        benchmark_knowledge_domain_validation_matrix.get(
            "knowledge_domain_validation_matrix"
        )
        or benchmark_knowledge_domain_validation_matrix
        or {}
    )
    knowledge_source_action_plan_component = (
        benchmark_knowledge_source_action_plan.get("knowledge_source_action_plan")
        or benchmark_knowledge_source_action_plan
        or {}
    )
    knowledge_source_coverage_component = (
        benchmark_knowledge_source_coverage.get("knowledge_source_coverage")
        or benchmark_knowledge_source_coverage
        or {}
    )
    knowledge_source_drift_component = (
        benchmark_knowledge_source_drift.get("knowledge_source_drift")
        or benchmark_knowledge_source_drift
        or {}
    )
    knowledge_outcome_correlation_component = (
        benchmark_knowledge_outcome_correlation.get("knowledge_outcome_correlation")
        or benchmark_knowledge_outcome_correlation
        or {}
    )
    knowledge_outcome_drift_component = (
        benchmark_knowledge_outcome_drift.get("knowledge_outcome_drift")
        or benchmark_knowledge_outcome_drift
        or {}
    )
    competitive_surpass_component = (
        benchmark_competitive_surpass_index.get("competitive_surpass_index")
        or benchmark_competitive_surpass_index
        or {}
    )
    competitive_surpass_trend_component = (
        benchmark_competitive_surpass_trend.get("competitive_surpass_trend")
        or benchmark_competitive_surpass_trend
        or {}
    )
    competitive_surpass_action_plan_component = (
        benchmark_competitive_surpass_action_plan.get(
            "competitive_surpass_action_plan"
        )
        or benchmark_competitive_surpass_action_plan
        or {}
    )
    realdata_status = (
        str(realdata_component.get("status") or "unknown").strip() or "unknown"
    )
    realdata_scorecard_status = (
        str(realdata_scorecard_component.get("status") or "unknown").strip()
        or "unknown"
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
    knowledge_domain_capability_matrix_domains = (
        knowledge_domain_capability_matrix_component.get("domains") or {}
    )
    knowledge_domain_capability_matrix_priority_domains = list(
        knowledge_domain_capability_matrix_component.get("priority_domains") or []
    )
    knowledge_domain_action_plan_actions = list(
        knowledge_domain_action_plan_component.get("actions") or []
    )
    knowledge_domain_action_plan_priority_domains = list(
        knowledge_domain_action_plan_component.get("priority_domains") or []
    )
    knowledge_domain_control_plane_domains = (
        knowledge_domain_control_plane_component.get("domains") or {}
    )
    knowledge_domain_control_plane_focus_areas = list(
        knowledge_domain_control_plane_component.get("focus_areas_detail") or []
    )
    knowledge_source_action_plan_priority_domains = list(
        knowledge_source_action_plan_component.get("priority_domains") or []
    )
    knowledge_source_coverage_domains = (
        knowledge_source_coverage_component.get("domains") or {}
    )
    knowledge_source_coverage_expansion_candidates = list(
        knowledge_source_coverage_component.get("expansion_candidates") or []
    )
    knowledge_outcome_correlation_domains = (
        knowledge_outcome_correlation_component.get("domains") or {}
    )
    knowledge_outcome_correlation_priority_domains = list(
        knowledge_outcome_correlation_component.get("priority_domains") or []
    )
    release_status = _release_status(
        benchmark_release_decision,
        benchmark_companion_summary,
        benchmark_artifact_bundle,
    )
    automation_ready = bool(benchmark_release_decision.get("automation_ready"))
    blockers = _compact(
        benchmark_release_decision.get("blocking_signals")
        or benchmark_companion_summary.get("blockers")
        or benchmark_artifact_bundle.get("blockers")
        or [],
    )
    knowledge_domain_release_gate_status = (
        str(knowledge_domain_release_gate_component.get("status") or "unknown").strip()
        or "unknown"
    )
    if knowledge_domain_release_gate_status in {
        "knowledge_domain_release_gate_blocked",
        "knowledge_domain_release_gate_unavailable",
    } or not bool(knowledge_domain_release_gate_component.get("gate_open", False)):
        for item in _compact(
            benchmark_knowledge_domain_release_gate.get("blocking_reasons")
            or benchmark_knowledge_domain_release_gate.get("recommendations")
            or [],
        ):
            if item not in blockers:
                blockers.append(item)
    knowledge_domain_release_readiness_matrix_status = (
        str(
            knowledge_domain_release_readiness_matrix_component.get("status")
            or "unknown"
        ).strip()
        or "unknown"
    )
    if knowledge_domain_release_readiness_matrix_status in {
        "knowledge_domain_release_readiness_unavailable",
        "knowledge_domain_release_readiness_blocked",
    }:
        for item in _compact(
            benchmark_knowledge_domain_release_readiness_matrix.get("recommendations")
            or [],
            limit=6,
        ):
            if item not in blockers:
                blockers.append(item)
    review_signals = _compact(
        benchmark_release_decision.get("review_signals")
        or benchmark_companion_summary.get("recommended_actions")
        or benchmark_artifact_bundle.get("recommendations")
        or [],
    )
    engineering_component = (
        benchmark_engineering_signals.get("engineering_signals")
        or benchmark_engineering_signals
        or {}
    )
    engineering_status = str(engineering_component.get("status") or "unknown").strip() or "unknown"
    if str(engineering_component.get("status") or "").strip() not in {
        "",
        "unknown",
        "engineering_semantics_ready",
    }:
        for item in _compact(benchmark_engineering_signals.get("recommendations") or []):
            if item not in review_signals:
                review_signals.append(item)
    if knowledge_status not in {"", "unknown", "knowledge_foundation_ready"}:
        for item in _compact(benchmark_knowledge_readiness.get("recommendations") or []):
            if item not in review_signals:
                review_signals.append(item)
    if realdata_status not in {"", "unknown", "realdata_foundation_ready"}:
        for item in _compact(benchmark_realdata_signals.get("recommendations") or []):
            if item not in review_signals:
                review_signals.append(item)
    if realdata_scorecard_status not in {
        "",
        "unknown",
        "realdata_scorecard_ready",
    }:
        for item in _compact(benchmark_realdata_scorecard.get("recommendations") or []):
            if item not in review_signals:
                review_signals.append(item)
    knowledge_application_status = (
        str(knowledge_application_component.get("status") or "unknown").strip()
        or "unknown"
    )
    if knowledge_application_status not in {
        "",
        "unknown",
        "knowledge_application_ready",
    }:
        for item in _compact(benchmark_knowledge_application.get("recommendations") or []):
            if item not in review_signals:
                review_signals.append(item)
    knowledge_realdata_correlation_status = (
        str(knowledge_realdata_correlation_component.get("status") or "unknown").strip()
        or "unknown"
    )
    if knowledge_realdata_correlation_status not in {
        "",
        "unknown",
        "knowledge_realdata_ready",
    }:
        for item in _compact(
            benchmark_knowledge_realdata_correlation.get("recommendations") or []
        ):
            if item not in review_signals:
                review_signals.append(item)
    knowledge_domain_matrix_status = (
        str(knowledge_domain_matrix_component.get("status") or "unknown").strip()
        or "unknown"
    )
    if knowledge_domain_matrix_status not in {
        "",
        "unknown",
        "knowledge_domain_matrix_ready",
    }:
        for item in _compact(benchmark_knowledge_domain_matrix.get("recommendations") or []):
            if item not in review_signals:
                review_signals.append(item)
    knowledge_domain_capability_matrix_status = (
        str(knowledge_domain_capability_matrix_component.get("status") or "unknown").strip()
        or "unknown"
    )
    if knowledge_domain_capability_matrix_status not in {
        "",
        "unknown",
        "knowledge_domain_capability_ready",
    }:
        for item in _compact(
            benchmark_knowledge_domain_capability_matrix.get("recommendations") or []
        ):
            if item not in review_signals:
                review_signals.append(item)
    knowledge_domain_capability_drift_status = (
        str(knowledge_domain_capability_drift_component.get("status") or "unknown").strip()
        or "unknown"
    )
    if knowledge_domain_capability_drift_status in {"regressed", "mixed"}:
        for item in _compact(
            benchmark_knowledge_domain_capability_drift.get("recommendations") or []
        ):
            if item not in review_signals:
                review_signals.append(item)
    knowledge_domain_action_plan_status = (
        str(knowledge_domain_action_plan_component.get("status") or "unknown").strip()
        or "unknown"
    )
    if knowledge_domain_action_plan_status not in {
        "",
        "unknown",
        "knowledge_domain_action_plan_ready",
    }:
        for item in _compact(
            benchmark_knowledge_domain_action_plan.get("recommendations") or []
        ):
            if item not in review_signals:
                review_signals.append(item)
    knowledge_domain_control_plane_status = (
        str(knowledge_domain_control_plane_component.get("status") or "unknown").strip()
        or "unknown"
    )
    if knowledge_domain_control_plane_status not in {
        "",
        "unknown",
        "knowledge_domain_control_plane_ready",
    }:
        for item in _compact(
            benchmark_knowledge_domain_control_plane.get("recommendations") or []
        ):
            if item not in review_signals:
                review_signals.append(item)
    knowledge_domain_control_plane_drift_status = (
        str(
            knowledge_domain_control_plane_drift_component.get("status") or "unknown"
        ).strip()
        or "unknown"
    )
    if knowledge_domain_control_plane_drift_status in {"regressed", "mixed"}:
        for item in _compact(
            benchmark_knowledge_domain_control_plane_drift.get("recommendations")
            or []
        ):
            if item not in review_signals:
                review_signals.append(item)
    if knowledge_domain_release_gate_status not in {
        "",
        "unknown",
        "knowledge_domain_release_gate_ready",
    }:
        for item in _compact(
            benchmark_knowledge_domain_release_gate.get("warning_reasons")
            or benchmark_knowledge_domain_release_gate.get("recommendations")
            or [],
        ):
            if item not in review_signals:
                review_signals.append(item)
    if knowledge_domain_release_readiness_matrix_status not in {
        "",
        "unknown",
        "knowledge_domain_release_readiness_ready",
    }:
        for item in _compact(
            benchmark_knowledge_domain_release_readiness_matrix.get("recommendations")
            or [],
            limit=6,
        ):
            if item not in review_signals:
                review_signals.append(item)
    if (
        knowledge_domain_release_surface_alignment_component.get("status")
        not in {"", "unknown", "aligned"}
    ):
        for item in _compact(
            benchmark_knowledge_domain_release_surface_alignment.get(
                "recommendations"
            )
            or []
        ):
            if item not in review_signals:
                review_signals.append(item)
    knowledge_source_action_plan_status = (
        str(knowledge_source_action_plan_component.get("status") or "unknown").strip()
        or "unknown"
    )
    if knowledge_source_action_plan_status not in {
        "",
        "unknown",
        "knowledge_source_action_plan_ready",
    }:
        for item in _compact(
            benchmark_knowledge_source_action_plan.get("recommendations") or []
        ):
            if item not in review_signals:
                review_signals.append(item)
    knowledge_source_coverage_status = (
        str(knowledge_source_coverage_component.get("status") or "unknown").strip()
        or "unknown"
    )
    knowledge_reference_inventory_status = (
        str(knowledge_reference_inventory_component.get("status") or "unknown").strip()
        or "unknown"
    )
    if knowledge_source_coverage_status not in {
        "",
        "unknown",
        "knowledge_source_coverage_ready",
    }:
        for item in _compact(
            benchmark_knowledge_source_coverage.get("recommendations") or []
        ):
            if item not in review_signals:
                review_signals.append(item)
    if knowledge_reference_inventory_status not in {
        "",
        "unknown",
        "knowledge_reference_inventory_ready",
    }:
        for item in _compact(
            benchmark_knowledge_reference_inventory.get("recommendations") or []
        ):
            if item not in review_signals:
                review_signals.append(item)
    knowledge_domain_validation_matrix_status = (
        str(knowledge_domain_validation_matrix_component.get("status") or "unknown").strip()
        or "unknown"
    )
    if knowledge_domain_validation_matrix_status not in {
        "",
        "unknown",
        "knowledge_domain_validation_ready",
    }:
        for item in _compact(
            benchmark_knowledge_domain_validation_matrix.get("recommendations") or []
        ):
            if item not in review_signals:
                review_signals.append(item)
    knowledge_source_drift_status = (
        str(knowledge_source_drift_component.get("status") or "unknown").strip()
        or "unknown"
    )
    if knowledge_source_drift_status not in {"", "unknown", "stable", "improved"}:
        for item in _compact(
            benchmark_knowledge_source_drift.get("recommendations") or []
        ):
            if item not in review_signals:
                review_signals.append(item)
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
            benchmark_knowledge_outcome_correlation.get("recommendations") or []
        ):
            if item not in review_signals:
                review_signals.append(item)
    knowledge_outcome_drift_status = (
        str(knowledge_outcome_drift_component.get("status") or "unknown").strip()
        or "unknown"
    )
    if knowledge_outcome_drift_status not in {"", "unknown", "stable", "improved"}:
        for item in _compact(
            benchmark_knowledge_outcome_drift.get("recommendations") or []
        ):
            if item not in review_signals:
                review_signals.append(item)
    competitive_surpass_status = (
        str(competitive_surpass_component.get("status") or "unknown").strip()
        or "unknown"
    )
    if competitive_surpass_status not in {
        "",
        "unknown",
        "competitive_surpass_ready",
    }:
        for item in _compact(
            benchmark_competitive_surpass_index.get("recommendations") or []
        ):
            if item not in review_signals:
                review_signals.append(item)
    competitive_surpass_trend_status = (
        str(competitive_surpass_trend_component.get("status") or "unknown").strip()
        or "unknown"
    )
    if competitive_surpass_trend_status not in {"", "unknown", "stable", "improved"}:
        for item in _compact(
            benchmark_competitive_surpass_trend.get("recommendations") or []
        ):
            if item not in review_signals:
                review_signals.append(item)
    competitive_surpass_action_plan_status = (
        str(competitive_surpass_action_plan_component.get("status") or "unknown").strip()
        or "unknown"
    )
    if competitive_surpass_action_plan_status not in {
        "",
        "unknown",
        "competitive_surpass_action_plan_ready",
    }:
        for item in _compact(
            benchmark_competitive_surpass_action_plan.get("recommendations") or []
        ):
            if item not in review_signals:
                review_signals.append(item)
    for item in _knowledge_drift_review_signals(knowledge_drift):
        if item not in review_signals:
            review_signals.append(item)
    artifacts = _artifacts(
        benchmark_release_decision=benchmark_release_decision,
        benchmark_companion_summary=benchmark_companion_summary,
        benchmark_artifact_bundle=benchmark_artifact_bundle,
        benchmark_knowledge_readiness=benchmark_knowledge_readiness,
        benchmark_knowledge_drift=benchmark_knowledge_drift,
        benchmark_engineering_signals=benchmark_engineering_signals,
        benchmark_realdata_signals=benchmark_realdata_signals,
        benchmark_operator_adoption=benchmark_operator_adoption,
        benchmark_realdata_scorecard=benchmark_realdata_scorecard,
        benchmark_knowledge_application=benchmark_knowledge_application,
        benchmark_knowledge_realdata_correlation=benchmark_knowledge_realdata_correlation,
        benchmark_knowledge_domain_matrix=benchmark_knowledge_domain_matrix,
        benchmark_knowledge_domain_capability_matrix=(
            benchmark_knowledge_domain_capability_matrix
        ),
        benchmark_knowledge_domain_action_plan=benchmark_knowledge_domain_action_plan,
        benchmark_knowledge_domain_control_plane=(
            benchmark_knowledge_domain_control_plane
        ),
        benchmark_knowledge_domain_control_plane_drift=(
            benchmark_knowledge_domain_control_plane_drift
        ),
        benchmark_knowledge_domain_release_gate=(
            benchmark_knowledge_domain_release_gate
        ),
        benchmark_knowledge_domain_release_readiness_matrix=(
            benchmark_knowledge_domain_release_readiness_matrix
        ),
        benchmark_knowledge_domain_release_surface_alignment=(
            benchmark_knowledge_domain_release_surface_alignment
        ),
        benchmark_knowledge_reference_inventory=(
            benchmark_knowledge_reference_inventory
        ),
        benchmark_knowledge_domain_capability_drift=(
            benchmark_knowledge_domain_capability_drift
        ),
        benchmark_knowledge_source_action_plan=benchmark_knowledge_source_action_plan,
        benchmark_knowledge_source_coverage=benchmark_knowledge_source_coverage,
        benchmark_knowledge_source_drift=benchmark_knowledge_source_drift,
        benchmark_knowledge_outcome_correlation=benchmark_knowledge_outcome_correlation,
        benchmark_knowledge_outcome_drift=benchmark_knowledge_outcome_drift,
        benchmark_competitive_surpass_index=benchmark_competitive_surpass_index,
        benchmark_competitive_surpass_trend=benchmark_competitive_surpass_trend,
        benchmark_competitive_surpass_action_plan=(
            benchmark_competitive_surpass_action_plan
        ),
        artifact_paths=artifact_paths,
    )
    operator_adoption = _operator_adoption_payload(benchmark_operator_adoption)
    scorecard_operator_adoption = _scorecard_operator_adoption(benchmark_scorecard)
    operational_operator_adoption = _operational_operator_adoption(
        benchmark_operational_summary
    )
    operator_adoption_release_surface_alignment = (
        _operator_adoption_release_surface_alignment(benchmark_operator_adoption)
    )
    missing_artifacts = [
        name
        for name, row in artifacts.items()
        if name
        not in {
            "benchmark_release_decision",
            "benchmark_operator_adoption",
            "benchmark_knowledge_readiness",
            "benchmark_knowledge_drift",
            "benchmark_knowledge_domain_control_plane",
            "benchmark_realdata_scorecard",
        }
        and not row["present"]
    ]

    operator_steps: List[Dict[str, Any]] = []
    operator_steps.append(
        _step(
            order=1,
            key="collect_artifacts",
            status="required" if missing_artifacts else "ready",
            title="Collect benchmark artifacts",
            reason=(
                "Missing artifacts: " + ", ".join(missing_artifacts)
                if missing_artifacts
                else "All required benchmark artifacts are present."
            ),
            action=(
                "Regenerate or attach the missing benchmark scorecard / operational "
                "summary / companion / bundle artifacts before freezing a release."
                if missing_artifacts
                else "No artifact backfill required."
            ),
        )
    )
    operator_steps.append(
        _step(
            order=2,
            key="resolve_blockers",
            status="required" if blockers else "ready",
            title="Resolve blocking signals",
            reason=(
                "; ".join(blockers)
                if blockers
                else "No release blockers were reported by the benchmark decision."
            ),
            action=(
                "Address the blocking components, then rerun evaluation-report.yml "
                "with benchmark exporters enabled."
                if blockers
                else "Blocker resolution is not required."
            ),
        )
    )
    operator_steps.append(
        _step(
            order=3,
            key="review_signals",
            status="required" if release_status == "review_required" or review_signals else "ready",
            title="Review non-blocking benchmark gaps",
            reason=(
                "; ".join(review_signals)
                if review_signals
                else "No additional review-only signals were emitted."
            ),
            action=(
                "Route the highlighted items through review queue / OCR guidance, "
                "then rerun the benchmark before promoting the next baseline."
                if release_status == "review_required" or review_signals
                else "No manual review escalation is required."
            ),
        )
    )
    operator_steps.append(
        _step(
            order=4,
            key="operator_adoption_guidance",
            status="guidance" if operator_adoption["has_guidance"] else "ready",
            title="Review operator adoption guidance",
            reason=(
                " | ".join(
                    part
                    for part in [
                        operator_adoption.get("summary") or "",
                        (
                            "Signals: "
                            + "; ".join(operator_adoption.get("signals") or [])
                        )
                        if operator_adoption.get("signals")
                        else "",
                    ]
                    if part
                )
                or "No operator adoption guidance was supplied."
            ),
            action=(
                "Use operator adoption actions as low-priority guidance after "
                "required blockers, artifact gaps, and review signals are cleared: "
                + "; ".join(operator_adoption.get("actions") or [])
                if operator_adoption["has_guidance"]
                else "No operator adoption follow-up is required."
            ),
        )
    )
    operator_steps.append(
        _step(
            order=5,
            key="rerun_benchmark",
            status=(
                "required"
                if blockers or review_signals or missing_artifacts
                else "ready"
            ),
            title="Rerun benchmark workflow",
            reason=(
                "Release evidence changed or remains incomplete."
                if blockers or review_signals or missing_artifacts
                else "Current evidence is already consistent."
            ),
            action=(
                "Trigger evaluation-report.yml and verify scorecard, companion "
                "summary, artifact bundle, and release decision artifacts."
            ),
        )
    )
    operator_steps.append(
        _step(
            order=6,
            key="freeze_release_baseline",
            status=(
                "ready"
                if (
                    automation_ready
                    and not blockers
                    and not review_signals
                    and not missing_artifacts
                )
                else "blocked"
            ),
            title="Freeze release benchmark baseline",
            reason=(
                "Automation-ready release decision with no outstanding blockers."
                if (
                    automation_ready
                    and not blockers
                    and not review_signals
                    and not missing_artifacts
                )
                else "Outstanding gaps still prevent freezing the next baseline."
            ),
            action=(
                "Promote this run as the benchmark baseline and attach the generated "
                "artifact bundle to the release record."
                if (
                    automation_ready
                    and not blockers
                    and not review_signals
                    and not missing_artifacts
                )
                else "Do not freeze the release baseline until earlier steps are green."
            ),
        )
    )

    next_action = next(
        (
            step["key"]
            for step in operator_steps
            if step["status"] in {"required", "blocked"}
        ),
        "freeze_release_baseline" if automation_ready else "rerun_benchmark",
    )
    ready_to_freeze = (
        automation_ready
        and release_status == "ready"
        and not blockers
        and not review_signals
        and not missing_artifacts
    )
    return {
        "title": title,
        "generated_at": int(time.time()),
        "release_status": release_status,
        "automation_ready": automation_ready,
        "ready_to_freeze_baseline": ready_to_freeze,
        "engineering_status": engineering_status,
        "knowledge_status": knowledge_status,
        "realdata_status": realdata_status,
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
        "realdata_signals": realdata_component,
        "realdata_recommendations": _compact(
            benchmark_realdata_signals.get("recommendations") or [],
            limit=6,
        ),
        "realdata_scorecard_status": realdata_scorecard_status,
        "realdata_scorecard": realdata_scorecard_component,
        "realdata_scorecard_recommendations": _compact(
            benchmark_realdata_scorecard.get("recommendations") or [],
            limit=6,
        ),
        "knowledge_domains": knowledge_domains,
        "knowledge_domain_focus_areas": knowledge_domain_focus_areas,
        "knowledge_priority_domains": knowledge_priority_domains,
        "knowledge_application_status": knowledge_application_status,
        "knowledge_application": knowledge_application_component,
        "knowledge_application_focus_areas": knowledge_application_focus_areas,
        "knowledge_application_domains": knowledge_application_domains,
        "knowledge_application_priority_domains": knowledge_application_priority_domains,
        "knowledge_application_recommendations": _compact(
            benchmark_knowledge_application.get("recommendations") or []
        ),
        "knowledge_realdata_correlation_status": knowledge_realdata_correlation_status,
        "knowledge_realdata_correlation": knowledge_realdata_correlation_component,
        "knowledge_realdata_correlation_domains": knowledge_realdata_correlation_domains,
        "knowledge_realdata_correlation_priority_domains": (
            knowledge_realdata_correlation_priority_domains
        ),
        "knowledge_realdata_correlation_recommendations": _compact(
            benchmark_knowledge_realdata_correlation.get("recommendations") or []
        ),
        "knowledge_domain_matrix_status": knowledge_domain_matrix_status,
        "knowledge_domain_matrix": knowledge_domain_matrix_component,
        "knowledge_domain_matrix_domains": knowledge_domain_matrix_domains,
        "knowledge_domain_matrix_priority_domains": (
            knowledge_domain_matrix_priority_domains
        ),
        "knowledge_domain_matrix_recommendations": _compact(
            benchmark_knowledge_domain_matrix.get("recommendations") or []
        ),
        "knowledge_domain_capability_matrix_status": (
            knowledge_domain_capability_matrix_status
        ),
        "knowledge_domain_capability_matrix": (
            knowledge_domain_capability_matrix_component
        ),
        "knowledge_domain_capability_matrix_domains": (
            knowledge_domain_capability_matrix_domains
        ),
        "knowledge_domain_capability_matrix_priority_domains": (
            knowledge_domain_capability_matrix_priority_domains
        ),
        "knowledge_domain_capability_matrix_focus_areas_detail": list(
            knowledge_domain_capability_matrix_component.get("focus_areas_detail")
            or []
        ),
        "knowledge_domain_capability_matrix_recommendations": _compact(
            benchmark_knowledge_domain_capability_matrix.get("recommendations") or []
        ),
        "knowledge_domain_capability_drift_status": (
            knowledge_domain_capability_drift_status
        ),
        "knowledge_domain_capability_drift": (
            knowledge_domain_capability_drift_component
        ),
        "knowledge_domain_capability_drift_domain_regressions": list(
            knowledge_domain_capability_drift_component.get("domain_regressions") or []
        ),
        "knowledge_domain_capability_drift_domain_improvements": list(
            knowledge_domain_capability_drift_component.get("domain_improvements") or []
        ),
        "knowledge_domain_capability_drift_recommendations": _compact(
            benchmark_knowledge_domain_capability_drift.get("recommendations") or []
        ),
        "knowledge_domain_action_plan_status": knowledge_domain_action_plan_status,
        "knowledge_domain_action_plan": knowledge_domain_action_plan_component,
        "knowledge_domain_action_plan_actions": knowledge_domain_action_plan_actions,
        "knowledge_domain_action_plan_priority_domains": (
            knowledge_domain_action_plan_priority_domains
        ),
        "knowledge_domain_action_plan_recommendations": _compact(
            benchmark_knowledge_domain_action_plan.get("recommendations") or []
        ),
        "knowledge_domain_control_plane_status": (
            knowledge_domain_control_plane_status
        ),
        "knowledge_domain_control_plane": knowledge_domain_control_plane_component,
        "knowledge_domain_control_plane_domains": knowledge_domain_control_plane_domains,
        "knowledge_domain_control_plane_focus_areas": (
            knowledge_domain_control_plane_focus_areas
        ),
        "knowledge_domain_control_plane_release_blockers": list(
            knowledge_domain_control_plane_component.get("release_blockers") or []
        ),
        "knowledge_domain_control_plane_recommendations": _compact(
            benchmark_knowledge_domain_control_plane.get("recommendations") or []
        ),
        "knowledge_domain_control_plane_drift_status": (
            knowledge_domain_control_plane_drift_status
        ),
        "knowledge_domain_control_plane_drift": (
            knowledge_domain_control_plane_drift_component
        ),
        "knowledge_domain_control_plane_drift_domain_regressions": list(
            knowledge_domain_control_plane_drift_component.get("domain_regressions")
            or []
        ),
        "knowledge_domain_control_plane_drift_domain_improvements": list(
            knowledge_domain_control_plane_drift_component.get("domain_improvements")
            or []
        ),
        "knowledge_domain_control_plane_drift_resolved_release_blockers": list(
            knowledge_domain_control_plane_drift_component.get(
                "resolved_release_blockers"
            )
            or []
        ),
        "knowledge_domain_control_plane_drift_new_release_blockers": list(
            knowledge_domain_control_plane_drift_component.get(
                "new_release_blockers"
            )
            or []
        ),
        "knowledge_domain_control_plane_drift_recommendations": _compact(
            benchmark_knowledge_domain_control_plane_drift.get("recommendations")
            or []
        ),
        "knowledge_domain_release_gate_status": knowledge_domain_release_gate_status,
        "knowledge_domain_release_gate": knowledge_domain_release_gate_component,
        "knowledge_domain_release_gate_gate_open": bool(
            knowledge_domain_release_gate_component.get("gate_open")
        ),
        "knowledge_domain_release_gate_releasable_domains": list(
            knowledge_domain_release_gate_component.get("releasable_domains") or []
        ),
        "knowledge_domain_release_gate_blocked_domains": list(
            knowledge_domain_release_gate_component.get("blocked_domains") or []
        ),
        "knowledge_domain_release_gate_priority_domains": list(
            knowledge_domain_release_gate_component.get("priority_domains") or []
        ),
        "knowledge_domain_release_gate_blocking_reasons": _compact(
            benchmark_knowledge_domain_release_gate.get("blocking_reasons") or []
        ),
        "knowledge_domain_release_gate_warning_reasons": _compact(
            benchmark_knowledge_domain_release_gate.get("warning_reasons") or []
        ),
        "knowledge_domain_release_gate_recommendations": _compact(
            benchmark_knowledge_domain_release_gate.get("recommendations") or []
        ),
        "knowledge_domain_release_readiness_matrix_status": (
            knowledge_domain_release_readiness_matrix_status
        ),
        "knowledge_domain_release_readiness_matrix": (
            knowledge_domain_release_readiness_matrix_component
        ),
        "knowledge_domain_release_readiness_matrix_summary": (
            knowledge_domain_release_readiness_matrix_component.get("summary")
            or "none"
        ),
        "knowledge_domain_release_readiness_matrix_priority_domains": list(
            knowledge_domain_release_readiness_matrix_component.get(
                "priority_domains"
            )
            or []
        ),
        "knowledge_domain_release_readiness_matrix_releasable_domains": list(
            knowledge_domain_release_readiness_matrix_component.get(
                "releasable_domains"
            )
            or []
        ),
        "knowledge_domain_release_readiness_matrix_blocked_domains": list(
            knowledge_domain_release_readiness_matrix_component.get("blocked_domains")
            or []
        ),
        "knowledge_domain_release_readiness_matrix_focus_areas_detail": list(
            knowledge_domain_release_readiness_matrix_component.get(
                "focus_areas_detail"
            )
            or []
        ),
        "knowledge_domain_release_readiness_matrix_recommendations": _compact(
            benchmark_knowledge_domain_release_readiness_matrix.get("recommendations")
            or []
        ),
        "knowledge_domain_release_surface_alignment_status": (
            knowledge_domain_release_surface_alignment_component.get("status")
            or "unknown"
        ),
        "knowledge_domain_release_surface_alignment": (
            knowledge_domain_release_surface_alignment_component
        ),
        "knowledge_domain_release_surface_alignment_recommendations": _compact(
            benchmark_knowledge_domain_release_surface_alignment.get(
                "recommendations"
            )
            or []
        ),
        "knowledge_reference_inventory_status": (
            knowledge_reference_inventory_component.get("status") or "unknown"
        ),
        "knowledge_reference_inventory": knowledge_reference_inventory_component,
        "knowledge_reference_inventory_summary": (
            knowledge_reference_inventory_component.get("summary") or "none"
        ),
        "knowledge_reference_inventory_priority_domains": list(
            knowledge_reference_inventory_component.get("priority_domains") or []
        ),
        "knowledge_reference_inventory_total_reference_items": (
            knowledge_reference_inventory_component.get("total_reference_items") or 0
        ),
        "knowledge_reference_inventory_focus_tables_detail": list(
            knowledge_reference_inventory_component.get("focus_tables_detail") or []
        ),
        "knowledge_reference_inventory_recommendations": _compact(
            benchmark_knowledge_reference_inventory.get("recommendations") or []
        ),
        "knowledge_domain_validation_matrix_status": (
            knowledge_domain_validation_matrix_component.get("status") or "unknown"
        ),
        "knowledge_domain_validation_matrix": (
            knowledge_domain_validation_matrix_component
        ),
        "knowledge_domain_validation_matrix_summary": (
            knowledge_domain_validation_matrix_component.get("summary") or "none"
        ),
        "knowledge_domain_validation_matrix_domains": (
            knowledge_domain_validation_matrix_component.get("domains") or {}
        ),
        "knowledge_domain_validation_matrix_priority_domains": list(
            knowledge_domain_validation_matrix_component.get("priority_domains") or []
        ),
        "knowledge_domain_validation_matrix_total_test_count": (
            knowledge_domain_validation_matrix_component.get("total_test_count") or 0
        ),
        "knowledge_domain_validation_matrix_recommendations": _compact(
            benchmark_knowledge_domain_validation_matrix.get("recommendations") or []
        ),
        "knowledge_source_action_plan_status": knowledge_source_action_plan_status,
        "knowledge_source_action_plan": knowledge_source_action_plan_component,
        "knowledge_source_action_plan_total_action_count": (
            knowledge_source_action_plan_component.get("total_action_count") or 0
        ),
        "knowledge_source_action_plan_high_priority_action_count": (
            knowledge_source_action_plan_component.get("high_priority_action_count")
            or 0
        ),
        "knowledge_source_action_plan_medium_priority_action_count": (
            knowledge_source_action_plan_component.get("medium_priority_action_count")
            or 0
        ),
        "knowledge_source_action_plan_priority_domains": (
            knowledge_source_action_plan_priority_domains
        ),
        "knowledge_source_action_plan_recommended_first_actions": list(
            knowledge_source_action_plan_component.get("recommended_first_actions")
            or []
        ),
        "knowledge_source_action_plan_source_group_action_counts": dict(
            knowledge_source_action_plan_component.get("source_group_action_counts")
            or {}
        ),
        "knowledge_source_action_plan_expansion_action_count": (
            knowledge_source_action_plan_component.get("expansion_action_count") or 0
        ),
        "knowledge_source_action_plan_recommendations": _compact(
            benchmark_knowledge_source_action_plan.get("recommendations") or []
        ),
        "knowledge_source_coverage_status": knowledge_source_coverage_status,
        "knowledge_source_coverage": knowledge_source_coverage_component,
        "knowledge_source_coverage_domains": knowledge_source_coverage_domains,
        "knowledge_source_coverage_expansion_candidates": (
            knowledge_source_coverage_expansion_candidates
        ),
        "knowledge_source_coverage_recommendations": _compact(
            benchmark_knowledge_source_coverage.get("recommendations") or []
        ),
        "knowledge_source_drift_status": knowledge_source_drift_status,
        "knowledge_source_drift": knowledge_source_drift_component,
        "knowledge_source_drift_summary": _text(
            benchmark_knowledge_source_drift.get("summary")
        ) or "none",
        "knowledge_source_drift_source_group_regressions": list(
            knowledge_source_drift_component.get("source_group_regressions") or []
        ),
        "knowledge_source_drift_source_group_improvements": list(
            knowledge_source_drift_component.get("source_group_improvements") or []
        ),
        "knowledge_source_drift_resolved_priority_domains": list(
            knowledge_source_drift_component.get("resolved_priority_domains") or []
        ),
        "knowledge_source_drift_new_priority_domains": list(
            knowledge_source_drift_component.get("new_priority_domains") or []
        ),
        "knowledge_source_drift_recommendations": _compact(
            benchmark_knowledge_source_drift.get("recommendations") or []
        ),
        "knowledge_outcome_correlation_status": knowledge_outcome_correlation_status,
        "knowledge_outcome_correlation": knowledge_outcome_correlation_component,
        "knowledge_outcome_correlation_domains": knowledge_outcome_correlation_domains,
        "knowledge_outcome_correlation_priority_domains": (
            knowledge_outcome_correlation_priority_domains
        ),
        "knowledge_outcome_correlation_recommendations": _compact(
            benchmark_knowledge_outcome_correlation.get("recommendations") or []
        ),
        "knowledge_outcome_drift_status": knowledge_outcome_drift_status,
        "knowledge_outcome_drift": knowledge_outcome_drift_component,
        "knowledge_outcome_drift_summary": _text(
            benchmark_knowledge_outcome_drift.get("summary")
        ) or "none",
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
        "knowledge_outcome_drift_recommendations": _compact(
            benchmark_knowledge_outcome_drift.get("recommendations") or []
        ),
        "competitive_surpass_index_status": competitive_surpass_status,
        "competitive_surpass_index": competitive_surpass_component,
        "competitive_surpass_primary_gaps": list(
            competitive_surpass_component.get("primary_gaps") or []
        ),
        "competitive_surpass_recommendations": _compact(
            benchmark_competitive_surpass_index.get("recommendations") or []
        ),
        "competitive_surpass_trend_status": competitive_surpass_trend_status,
        "competitive_surpass_trend": competitive_surpass_trend_component,
        "competitive_surpass_trend_summary": _text(
            benchmark_competitive_surpass_trend.get("summary")
        ) or "none",
        "competitive_surpass_trend_score_delta": (
            competitive_surpass_trend_component.get("score_delta") or 0
        ),
        "competitive_surpass_trend_pillar_improvements": list(
            competitive_surpass_trend_component.get("pillar_improvements") or []
        ),
        "competitive_surpass_trend_pillar_regressions": list(
            competitive_surpass_trend_component.get("pillar_regressions") or []
        ),
        "competitive_surpass_trend_resolved_primary_gaps": list(
            competitive_surpass_trend_component.get("resolved_primary_gaps") or []
        ),
        "competitive_surpass_trend_new_primary_gaps": list(
            competitive_surpass_trend_component.get("new_primary_gaps") or []
        ),
        "competitive_surpass_trend_recommendations": _compact(
            benchmark_competitive_surpass_trend.get("recommendations") or []
        ),
        "competitive_surpass_action_plan_status": (
            competitive_surpass_action_plan_status
        ),
        "competitive_surpass_action_plan": competitive_surpass_action_plan_component,
        "competitive_surpass_action_plan_total_action_count": (
            competitive_surpass_action_plan_component.get("total_action_count") or 0
        ),
        "competitive_surpass_action_plan_high_priority_action_count": (
            competitive_surpass_action_plan_component.get("high_priority_action_count")
            or 0
        ),
        "competitive_surpass_action_plan_medium_priority_action_count": (
            competitive_surpass_action_plan_component.get(
                "medium_priority_action_count"
            )
            or 0
        ),
        "competitive_surpass_action_plan_priority_pillars": list(
            competitive_surpass_action_plan_component.get("priority_pillars") or []
        ),
        "competitive_surpass_action_plan_recommended_first_actions": list(
            competitive_surpass_action_plan_component.get(
                "recommended_first_actions"
            )
            or []
        ),
        "competitive_surpass_action_plan_recommendations": _compact(
            benchmark_competitive_surpass_action_plan.get("recommendations") or []
        ),
        "primary_signal_source": _primary_signal_source(
            benchmark_release_decision,
            benchmark_companion_summary,
            benchmark_artifact_bundle,
        ),
        "missing_artifacts": missing_artifacts,
        "blocking_signals": blockers,
        "review_signals": review_signals,
        "operator_adoption": operator_adoption,
        "scorecard_operator_adoption": scorecard_operator_adoption,
        "operational_operator_adoption": operational_operator_adoption,
        "operator_adoption_release_surface_alignment": (
            operator_adoption_release_surface_alignment
        ),
        "next_action": next_action,
        "operator_steps": operator_steps,
        "artifacts": artifacts,
    }


def render_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        f"# {payload.get('title') or 'Benchmark Release Runbook'}",
        "",
        f"- `release_status`: `{payload.get('release_status')}`",
        f"- `automation_ready`: `{payload.get('automation_ready')}`",
        f"- `ready_to_freeze_baseline`: `{payload.get('ready_to_freeze_baseline')}`",
        f"- `engineering_status`: `{payload.get('engineering_status')}`",
        f"- `knowledge_status`: `{payload.get('knowledge_status')}`",
        f"- `knowledge_application_status`: "
        f"`{payload.get('knowledge_application_status')}`",
        f"- `knowledge_realdata_correlation_status`: "
        f"`{payload.get('knowledge_realdata_correlation_status')}`",
        f"- `knowledge_domain_matrix_status`: "
        f"`{payload.get('knowledge_domain_matrix_status')}`",
        f"- `knowledge_domain_capability_matrix_status`: "
        f"`{payload.get('knowledge_domain_capability_matrix_status') or 'unknown'}`",
        f"- `knowledge_domain_capability_drift_status`: "
        f"`{payload.get('knowledge_domain_capability_drift_status') or 'unknown'}`",
        f"- `knowledge_domain_control_plane_status`: "
        f"`{payload.get('knowledge_domain_control_plane_status') or 'unknown'}`",
        f"- `knowledge_domain_control_plane_drift_status`: "
        f"`{payload.get('knowledge_domain_control_plane_drift_status') or 'unknown'}`",
        f"- `knowledge_domain_release_gate_status`: "
        f"`{payload.get('knowledge_domain_release_gate_status') or 'unknown'}`",
        f"- `knowledge_reference_inventory_status`: "
        f"`{payload.get('knowledge_reference_inventory_status') or 'unknown'}`",
        f"- `knowledge_domain_release_surface_alignment_status`: "
        f"`{payload.get('knowledge_domain_release_surface_alignment_status') or 'unknown'}`",
        f"- `knowledge_domain_action_plan_status`: "
        f"`{payload.get('knowledge_domain_action_plan_status') or 'unknown'}`",
        f"- `knowledge_source_coverage_status`: "
        f"`{payload.get('knowledge_source_coverage_status') or 'unknown'}`",
        f"- `knowledge_source_drift_status`: "
        f"`{payload.get('knowledge_source_drift_status') or 'unknown'}`",
        f"- `knowledge_outcome_correlation_status`: "
        f"`{payload.get('knowledge_outcome_correlation_status')}`",
        f"- `knowledge_outcome_drift_status`: "
        f"`{payload.get('knowledge_outcome_drift_status') or 'unknown'}`",
        f"- `realdata_status`: `{payload.get('realdata_status')}`",
        f"- `realdata_scorecard_status`: "
        f"`{payload.get('realdata_scorecard_status') or 'unknown'}`",
        f"- `competitive_surpass_index_status`: "
        f"`{payload.get('competitive_surpass_index_status') or 'unknown'}`",
        f"- `competitive_surpass_trend_status`: "
        f"`{payload.get('competitive_surpass_trend_status') or 'unknown'}`",
        f"- `primary_signal_source`: `{payload.get('primary_signal_source')}`",
        f"- `next_action`: `{payload.get('next_action')}`",
        "",
        "## Missing Artifacts",
        "",
    ]
    missing_artifacts = payload.get("missing_artifacts") or []
    if missing_artifacts:
        lines.extend(f"- `{name}`" for name in missing_artifacts)
    else:
        lines.append("- none")
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
    lines.extend(["", "## Knowledge Domain Capability Matrix", ""])
    lines.append(
        f"- `status`: `{payload.get('knowledge_domain_capability_matrix_status') or 'unknown'}`"
    )
    capability_domains = payload.get("knowledge_domain_capability_matrix_domains") or {}
    if capability_domains:
        for name, row in capability_domains.items():
            lines.append(
                "- "
                f"`{name}` "
                f"status=`{row.get('status')}` "
                f"foundation=`{row.get('foundation_status')}` "
                f"application=`{row.get('application_status')}` "
                f"provider=`{row.get('provider_status')}` "
                f"surface=`{row.get('surface_status')}` "
                f"reference_items=`{row.get('reference_item_count')}`"
            )
    else:
        lines.append("- none")
    capability_recommendations = (
        payload.get("knowledge_domain_capability_matrix_recommendations") or []
    )
    if capability_recommendations:
        for item in capability_recommendations:
            lines.append(f"- recommendation: {item}")
    lines.extend(["", "## Knowledge Domain Capability Drift", ""])
    capability_drift_regressions = ", ".join(
        payload.get("knowledge_domain_capability_drift_domain_regressions") or []
    ) or "none"
    capability_drift_improvements = ", ".join(
        payload.get("knowledge_domain_capability_drift_domain_improvements") or []
    ) or "none"
    lines.append(
        f"- `status`: `{payload.get('knowledge_domain_capability_drift_status') or 'unknown'}`"
    )
    lines.append("- `domain_regressions`: " f"`{capability_drift_regressions}`")
    lines.append("- `domain_improvements`: " f"`{capability_drift_improvements}`")
    capability_drift_recommendations = (
        payload.get("knowledge_domain_capability_drift_recommendations") or []
    )
    if capability_drift_recommendations:
        for item in capability_drift_recommendations:
            lines.append(f"- recommendation: {item}")
    lines.extend(["", "## Knowledge Domain Control Plane", ""])
    lines.append(
        f"- `status`: `{payload.get('knowledge_domain_control_plane_status') or 'unknown'}`"
    )
    control_plane_release_blockers = ", ".join(
        payload.get("knowledge_domain_control_plane_release_blockers") or []
    ) or "none"
    lines.append("- `release_blockers`: " f"`{control_plane_release_blockers}`")
    control_plane_domains = (
        payload.get("knowledge_domain_control_plane_domains") or {}
    )
    if control_plane_domains:
        for name, row in control_plane_domains.items():
            lines.append(
                "- "
                f"`{name}` "
                f"status=`{row.get('status')}` "
                f"priority=`{row.get('priority')}` "
                f"release_blocker=`{row.get('release_blocker')}`"
            )
    else:
        lines.append("- none")
    control_plane_recommendations = (
        payload.get("knowledge_domain_control_plane_recommendations") or []
    )
    if control_plane_recommendations:
        for item in control_plane_recommendations:
            lines.append(f"- recommendation: {item}")
    lines.extend(["", "## Knowledge Domain Control Plane Drift", ""])
    control_plane_drift_regressions = ", ".join(
        payload.get("knowledge_domain_control_plane_drift_domain_regressions") or []
    ) or "none"
    control_plane_drift_improvements = ", ".join(
        payload.get("knowledge_domain_control_plane_drift_domain_improvements") or []
    ) or "none"
    control_plane_drift_new_blockers = ", ".join(
        payload.get("knowledge_domain_control_plane_drift_new_release_blockers") or []
    ) or "none"
    control_plane_drift_resolved_blockers = ", ".join(
        payload.get(
            "knowledge_domain_control_plane_drift_resolved_release_blockers"
        )
        or []
    ) or "none"
    lines.append(
        f"- `status`: `{payload.get('knowledge_domain_control_plane_drift_status') or 'unknown'}`"
    )
    lines.append(
        "- `domain_regressions`: " f"`{control_plane_drift_regressions}`"
    )
    lines.append(
        "- `domain_improvements`: " f"`{control_plane_drift_improvements}`"
    )
    lines.append(
        "- `new_release_blockers`: " f"`{control_plane_drift_new_blockers}`"
    )
    lines.append(
        "- `resolved_release_blockers`: "
        f"`{control_plane_drift_resolved_blockers}`"
    )
    control_plane_drift_recommendations = (
        payload.get("knowledge_domain_control_plane_drift_recommendations") or []
    )
    if control_plane_drift_recommendations:
        for item in control_plane_drift_recommendations:
            lines.append(f"- recommendation: {item}")
    lines.extend(["", "## Knowledge Domain Release Gate", ""])
    lines.append(
        f"- `status`: `{payload.get('knowledge_domain_release_gate_status') or 'unknown'}`"
    )
    lines.append(
        "- `gate_open`: "
        + str(bool(payload.get("knowledge_domain_release_gate_gate_open"))).lower()
    )
    lines.append(
        "- `releasable_domains`: "
        + (
            ", ".join(payload.get("knowledge_domain_release_gate_releasable_domains") or [])
            or "none"
        )
    )
    lines.append(
        "- `blocked_domains`: "
        + (
            ", ".join(payload.get("knowledge_domain_release_gate_blocked_domains") or [])
            or "none"
        )
    )
    lines.append(
        "- `priority_domains`: "
        + (
            ", ".join(payload.get("knowledge_domain_release_gate_priority_domains") or [])
            or "none"
        )
    )
    lines.append(
        "- `blocking_reasons`: "
        + (
            ", ".join(payload.get("knowledge_domain_release_gate_blocking_reasons") or [])
            or "none"
        )
    )
    gate_warnings = payload.get("knowledge_domain_release_gate_warning_reasons") or []
    if gate_warnings:
        for item in gate_warnings:
            lines.append(f"- warning: {item}")
    gate_recommendations = (
        payload.get("knowledge_domain_release_gate_recommendations") or []
    )
    if gate_recommendations:
        for item in gate_recommendations:
            lines.append(f"- recommendation: {item}")
    lines.extend(["", "## Knowledge Domain Release Readiness Matrix", ""])
    lines.append(
        "- `status`: "
        f"`{payload.get('knowledge_domain_release_readiness_matrix_status') or 'unknown'}`"
    )
    lines.append(
        "- `summary`: "
        f"`{payload.get('knowledge_domain_release_readiness_matrix_summary') or 'none'}`"
    )
    lines.append(
        "- `priority_domains`: "
        + (
            ", ".join(
                payload.get(
                    "knowledge_domain_release_readiness_matrix_priority_domains"
                )
                or []
            )
            or "none"
        )
    )
    lines.append(
        "- `releasable_domains`: "
        + (
            ", ".join(
                payload.get(
                    "knowledge_domain_release_readiness_matrix_releasable_domains"
                )
                or []
            )
            or "none"
        )
    )
    lines.append(
        "- `blocked_domains`: "
        + (
            ", ".join(
                payload.get(
                    "knowledge_domain_release_readiness_matrix_blocked_domains"
                )
                or []
            )
            or "none"
        )
    )
    readiness_domains = (
        payload.get("knowledge_domain_release_readiness_matrix", {}).get("domains")
        or {}
    )
    if readiness_domains:
        for name, row in readiness_domains.items():
            lines.append(
                "- "
                f"`{name}` "
                f"status=`{row.get('status')}` "
                f"validation=`{row.get('validation_status')}` "
                f"inventory=`{row.get('inventory_status')}` "
                f"gate=`{row.get('release_gate_status')}` "
                f"alignment_warning=`{bool(row.get('alignment_warning'))}`"
            )
    else:
        lines.append("- none")
    readiness_recommendations = (
        payload.get("knowledge_domain_release_readiness_matrix_recommendations")
        or []
    )
    if readiness_recommendations:
        for item in readiness_recommendations:
            lines.append(f"- recommendation: {item}")
    lines.extend(["", "## Knowledge Domain Release Surface Alignment", ""])
    knowledge_alignment = (
        payload.get("knowledge_domain_release_surface_alignment") or {}
    )
    lines.append(
        f"- `status`: `{knowledge_alignment.get('status') or 'unknown'}`"
    )
    lines.append(
        "- `summary`: " + (_text(knowledge_alignment.get("summary")) or "none")
    )
    lines.append(
        "- `mismatches`: "
        + (
            ", ".join(str(item) for item in knowledge_alignment.get("mismatches") or [])
            or "none"
        )
    )
    knowledge_alignment_recommendations = (
        payload.get("knowledge_domain_release_surface_alignment_recommendations")
        or []
    )
    if knowledge_alignment_recommendations:
        for item in knowledge_alignment_recommendations:
            lines.append(f"- recommendation: {item}")
    lines.extend(["", "## Knowledge Reference Inventory", ""])
    lines.append(
        f"- `status`: `{payload.get('knowledge_reference_inventory_status') or 'unknown'}`"
    )
    lines.append(
        "- `summary`: "
        f"`{payload.get('knowledge_reference_inventory_summary') or 'none'}`"
    )
    priority_domains_text = ", ".join(
        payload.get("knowledge_reference_inventory_priority_domains") or []
    )
    lines.append(f"- `priority_domains`: `{priority_domains_text or 'none'}`")
    lines.append(
        "- `total_reference_items`: "
        f"`{payload.get('knowledge_reference_inventory_total_reference_items') or 0}`"
    )
    knowledge_reference_inventory = payload.get("knowledge_reference_inventory") or {}
    knowledge_reference_domains = knowledge_reference_inventory.get("domains") or {}
    if knowledge_reference_domains:
        for name, row in knowledge_reference_domains.items():
            lines.append(
                "- "
                f"`{name}` "
                f"status=`{row.get('status')}` "
                f"reference_items=`{row.get('total_reference_items', 0)}` "
                f"tables=`{row.get('populated_table_count', 0)}/"
                f"{row.get('total_table_count', 0)}` "
                f"missing_tables=`{', '.join(row.get('missing_tables') or []) or 'none'}`"
            )
    else:
        lines.append("- none")
    focus_tables = payload.get("knowledge_reference_inventory_focus_tables_detail") or []
    if focus_tables:
        for row in focus_tables[:5]:
            lines.append(
                "- focus: "
                f"`{row.get('domain')}` "
                f"missing_tables=`{', '.join(row.get('missing_tables') or []) or 'none'}` "
                f"action=`{row.get('action') or 'Backfill missing tables.'}`"
            )
    knowledge_reference_inventory_recommendations = (
        payload.get("knowledge_reference_inventory_recommendations") or []
    )
    if knowledge_reference_inventory_recommendations:
        for item in knowledge_reference_inventory_recommendations:
            lines.append(f"- recommendation: {item}")
    lines.extend(["", "## Knowledge Domain Validation Matrix", ""])
    lines.append(
        f"- `status`: `{payload.get('knowledge_domain_validation_matrix_status') or 'unknown'}`"
    )
    lines.append(
        "- `summary`: "
        f"`{payload.get('knowledge_domain_validation_matrix_summary') or 'none'}`"
    )
    priority_validation_domains = ", ".join(
        payload.get("knowledge_domain_validation_matrix_priority_domains") or []
    )
    lines.append(
        f"- `priority_domains`: `{priority_validation_domains or 'none'}`"
    )
    lines.append(
        "- `total_test_count`: "
        f"`{payload.get('knowledge_domain_validation_matrix_total_test_count') or 0}`"
    )
    knowledge_validation_domains = (
        payload.get("knowledge_domain_validation_matrix_domains") or {}
    )
    if knowledge_validation_domains:
        for name, row in knowledge_validation_domains.items():
            lines.append(
                "- "
                f"`{name}` "
                f"status=`{row.get('status')}` "
                f"provider=`{row.get('provider_status')}` "
                f"api=`{row.get('public_surface_count')}` "
                f"tests=`{row.get('total_test_count')}` "
                f"missing_layers=`{', '.join(row.get('missing_layers') or []) or 'none'}`"
            )
    else:
        lines.append("- none")
    validation_recommendations = (
        payload.get("knowledge_domain_validation_matrix_recommendations") or []
    )
    if validation_recommendations:
        for item in validation_recommendations:
            lines.append(f"- recommendation: {item}")
    lines.extend(["", "## Knowledge Domain Action Plan", ""])
    lines.append(
        f"- `status`: `{payload.get('knowledge_domain_action_plan_status') or 'unknown'}`"
    )
    action_plan_actions = payload.get("knowledge_domain_action_plan_actions") or []
    if action_plan_actions:
        for item in action_plan_actions:
            lines.append(
                "- "
                f"`{item.get('id')}` "
                f"domain=`{item.get('domain')}` "
                f"stage=`{item.get('stage')}` "
                f"priority=`{item.get('priority')}` "
                f"status=`{item.get('status')}`"
            )
    else:
        lines.append("- none")
    action_plan_recommendations = (
        payload.get("knowledge_domain_action_plan_recommendations") or []
    )
    if action_plan_recommendations:
        for item in action_plan_recommendations:
            lines.append(f"- recommendation: {item}")
    lines.extend(["", "## Knowledge Source Action Plan", ""])
    lines.append(
        f"- `status`: `{payload.get('knowledge_source_action_plan_status') or 'unknown'}`"
    )
    lines.append(
        "- `counts`: "
        f"total=`{payload.get('knowledge_source_action_plan_total_action_count') or 0}` "
        f"high=`{payload.get('knowledge_source_action_plan_high_priority_action_count') or 0}` "
        f"medium=`{payload.get('knowledge_source_action_plan_medium_priority_action_count') or 0}` "
        f"expansion=`{payload.get('knowledge_source_action_plan_expansion_action_count') or 0}`"
    )
    priority_domains_text = ", ".join(
        payload.get("knowledge_source_action_plan_priority_domains") or []
    )
    lines.append(
        f"- `priority_domains`: `{priority_domains_text or 'none'}`"
    )
    source_action_items = (
        payload.get("knowledge_source_action_plan_recommended_first_actions") or []
    )
    if source_action_items:
        for item in source_action_items:
            lines.append(
                "- action: "
                f"`{item.get('id')}` priority=`{item.get('priority')}` "
                f"stage=`{item.get('stage')}`"
            )
    else:
        lines.append("- none")
    source_action_recommendations = (
        payload.get("knowledge_source_action_plan_recommendations") or []
    )
    if source_action_recommendations:
        for item in source_action_recommendations:
            lines.append(f"- recommendation: {item}")
    lines.extend(["", "## Knowledge Source Coverage", ""])
    lines.append(
        f"- `status`: `{payload.get('knowledge_source_coverage_status') or 'unknown'}`"
    )
    knowledge_source_domains = payload.get("knowledge_source_coverage_domains") or {}
    if knowledge_source_domains:
        for name, row in knowledge_source_domains.items():
            lines.append(
                "- "
                f"`{name}` "
                f"status=`{row.get('status')}` "
                f"source_groups=`{', '.join(row.get('source_groups') or []) or 'none'}` "
                f"source_items=`{row.get('source_item_count')}`"
            )
    else:
        lines.append("- none")
    knowledge_source_expansion = (
        payload.get("knowledge_source_coverage_expansion_candidates") or []
    )
    if knowledge_source_expansion:
        for row in knowledge_source_expansion[:4]:
            lines.append(
                "- expansion: "
                f"`{row.get('name')}` status=`{row.get('status')}` "
                f"source_items=`{row.get('source_item_count')}`"
            )
    knowledge_source_recommendations = (
        payload.get("knowledge_source_coverage_recommendations") or []
    )
    if knowledge_source_recommendations:
        for item in knowledge_source_recommendations:
            lines.append(f"- recommendation: {item}")
    lines.extend(["", "## Knowledge Source Drift", ""])
    knowledge_source_drift = payload.get("knowledge_source_drift") or {}
    lines.append(
        f"- `status`: `{payload.get('knowledge_source_drift_status') or 'unknown'}`"
    )
    lines.append(
        "- `summary`: "
        + (_text(payload.get("knowledge_source_drift_summary")) or "none")
    )
    counts = knowledge_source_drift.get("counts") or {}
    if counts:
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
        values = knowledge_source_drift.get(label) or []
        if values:
            lines.append(
                f"- `{label}`: `{', '.join(str(item) for item in values) or 'none'}`"
            )
    for label in (
        "knowledge_source_drift_source_group_regressions",
        "knowledge_source_drift_source_group_improvements",
        "knowledge_source_drift_resolved_priority_domains",
        "knowledge_source_drift_new_priority_domains",
    ):
        values = payload.get(label) or []
        lines.append(
            f"- `{label}`: `{', '.join(str(item) for item in values) or 'none'}`"
        )
    knowledge_source_drift_recommendations = (
        payload.get("knowledge_source_drift_recommendations") or []
    )
    if knowledge_source_drift_recommendations:
        lines.extend(
            f"- recommendation: {item}" for item in knowledge_source_drift_recommendations
        )
    lines.extend(["", "## Knowledge Outcome Correlation", ""])
    knowledge_outcome_domains = payload.get("knowledge_outcome_correlation_domains") or {}
    lines.append(
        "- `status`: "
        f"`{payload.get('knowledge_outcome_correlation_status') or 'unknown'}`"
    )
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
        lines.extend(f"- recommendation: {item}" for item in knowledge_outcome_recommendations)
    lines.extend(["", "## Knowledge Outcome Drift", ""])
    knowledge_outcome_drift = payload.get("knowledge_outcome_drift") or {}
    lines.append(
        f"- `status`: `{payload.get('knowledge_outcome_drift_status') or 'unknown'}`"
    )
    lines.append(
        "- `summary`: "
        + (_text(payload.get("knowledge_outcome_drift_summary")) or "none")
    )
    counts = knowledge_outcome_drift.get("counts") or {}
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
        values = knowledge_outcome_drift.get(label) or []
        lines.append(
            f"- `{label}`: `{', '.join(str(item) for item in values) or 'none'}`"
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
        lines.extend(
            f"- recommendation: {item}"
            for item in knowledge_outcome_drift_recommendations
        )
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
    lines.extend(["", "## Competitive Surpass Index", ""])
    competitive_surpass = payload.get("competitive_surpass_index") or {}
    lines.append(
        f"- `status`: `{payload.get('competitive_surpass_index_status') or 'unknown'}`"
    )
    lines.append(f"- `score`: `{competitive_surpass.get('score', 0)}`")
    gaps = payload.get("competitive_surpass_primary_gaps") or []
    lines.append(
        "- `primary_gaps`: "
        + (", ".join(str(item) for item in gaps) if gaps else "none")
    )
    competitive_surpass_recommendations = (
        payload.get("competitive_surpass_recommendations") or []
    )
    if competitive_surpass_recommendations:
        lines.extend(f"- {item}" for item in competitive_surpass_recommendations)
    else:
        lines.append("- none")
    lines.extend(["", "## Competitive Surpass Trend", ""])
    lines.append(
        f"- `status`: `{payload.get('competitive_surpass_trend_status') or 'unknown'}`"
    )
    lines.append(
        f"- `score_delta`: `{payload.get('competitive_surpass_trend_score_delta') or 0}`"
    )
    lines.append(
        "- `summary`: "
        + (_text(payload.get("competitive_surpass_trend_summary")) or "none")
    )
    for label in (
        "competitive_surpass_trend_pillar_improvements",
        "competitive_surpass_trend_pillar_regressions",
        "competitive_surpass_trend_resolved_primary_gaps",
        "competitive_surpass_trend_new_primary_gaps",
    ):
        values = payload.get(label) or []
        lines.append(
            f"- `{label}`: `{', '.join(str(item) for item in values) or 'none'}`"
        )
    trend_recommendations = payload.get("competitive_surpass_trend_recommendations") or []
    if trend_recommendations:
        lines.extend(f"- recommendation: {item}" for item in trend_recommendations)
    else:
        lines.append("- recommendation: none")
    lines.extend(["", "## Competitive Surpass Action Plan", ""])
    lines.append(
        "- `status`: "
        f"`{payload.get('competitive_surpass_action_plan_status') or 'unknown'}`"
    )
    lines.append(
        "- `total_action_count`: "
        f"`{payload.get('competitive_surpass_action_plan_total_action_count') or 0}`"
    )
    lines.append(
        "- `high_priority_action_count`: "
        f"`{payload.get('competitive_surpass_action_plan_high_priority_action_count') or 0}`"
    )
    lines.append(
        "- `medium_priority_action_count`: "
        f"`{payload.get('competitive_surpass_action_plan_medium_priority_action_count') or 0}`"
    )
    lines.append(
        "- `priority_pillars`: "
        + (
            ", ".join(
                str(item)
                for item in (
                    payload.get("competitive_surpass_action_plan_priority_pillars") or []
                )
            )
            or "none"
        )
    )
    first_actions = (
        payload.get("competitive_surpass_action_plan_recommended_first_actions") or []
    )
    if first_actions:
        for item in first_actions:
            lines.append(
                "- first_action: "
                f"{item.get('pillar') or 'unknown'} -> {item.get('action') or 'none'}"
            )
    else:
        lines.append("- first_action: none")
    action_plan_recommendations = (
        payload.get("competitive_surpass_action_plan_recommendations") or []
    )
    if action_plan_recommendations:
        lines.extend(f"- recommendation: {item}" for item in action_plan_recommendations)
    else:
        lines.append("- recommendation: none")
    lines.extend(["", "## Operator Adoption", ""])
    operator_adoption = payload.get("operator_adoption") or {}
    lines.append(f"- `status`: `{operator_adoption.get('status')}`")
    lines.append(f"- `has_guidance`: `{operator_adoption.get('has_guidance')}`")
    lines.append(
        f"- `knowledge_drift_status`: "
        f"`{operator_adoption.get('knowledge_drift_status')}`"
    )
    lines.append(
        f"- `knowledge_outcome_drift_status`: "
        f"`{operator_adoption.get('knowledge_outcome_drift_status')}`"
    )
    lines.append(
        "- `summary`: "
        + (_text(operator_adoption.get("summary")) or "none")
    )
    lines.append(
        "- `knowledge_drift_summary`: "
        + (_text(operator_adoption.get("knowledge_drift_summary")) or "none")
    )
    lines.append(
        "- `knowledge_outcome_drift_summary`: "
        + (_text(operator_adoption.get("knowledge_outcome_drift_summary")) or "none")
    )
    operator_signals = operator_adoption.get("signals") or []
    if operator_signals:
        lines.extend(f"- signal: {item}" for item in operator_signals)
    else:
        lines.append("- signal: none")
    operator_actions = operator_adoption.get("actions") or []
    if operator_actions:
        lines.extend(f"- action: {item}" for item in operator_actions)
    else:
        lines.append("- action: none")
    drift_recommendations = operator_adoption.get("knowledge_drift_recommendations") or []
    if drift_recommendations:
        lines.extend(f"- drift_recommendation: {item}" for item in drift_recommendations)
    else:
        lines.append("- drift_recommendation: none")
    outcome_drift_recommendations = (
        operator_adoption.get("knowledge_outcome_drift_recommendations") or []
    )
    if outcome_drift_recommendations:
        lines.extend(
            f"- outcome_drift_recommendation: {item}"
            for item in outcome_drift_recommendations
        )
    else:
        lines.append("- outcome_drift_recommendation: none")
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
        + (_text(scorecard_operator.get("knowledge_outcome_drift_summary")) or "none")
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
        + (_text(operational_operator.get("knowledge_outcome_drift_summary")) or "none")
    )
    lines.extend(["", "## Operator Adoption Release Surface Alignment", ""])
    alignment = payload.get("operator_adoption_release_surface_alignment") or {}
    lines.append(f"- `status`: `{alignment.get('status') or 'unknown'}`")
    lines.append(
        "- `summary`: "
        + (_text(alignment.get("summary")) or "none")
    )
    mismatches = alignment.get("mismatches") or []
    lines.append(
        "- `mismatches`: "
        + (", ".join(str(item) for item in mismatches) if mismatches else "none")
    )
    lines.extend(["", "## Operator Steps", ""])
    for step in payload.get("operator_steps") or []:
        lines.append(
            f"{step.get('order')}. `{step.get('key')}` `{step.get('status')}` "
            f"{step.get('title')}"
        )
        lines.append(f"   - reason: {step.get('reason')}")
        lines.append(f"   - action: {step.get('action')}")
    lines.extend(["", "## Artifacts", ""])
    for name, row in (payload.get("artifacts") or {}).items():
        lines.append(
            f"- `{name}`: present=`{row.get('present')}` path=`{row.get('path')}`"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a benchmark release operator runbook."
    )
    parser.add_argument("--title", default="Benchmark Release Runbook")
    parser.add_argument("--benchmark-release-decision", default="")
    parser.add_argument("--benchmark-scorecard", default="")
    parser.add_argument("--benchmark-operational-summary", default="")
    parser.add_argument("--benchmark-companion-summary", default="")
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
    parser.add_argument("--benchmark-knowledge-domain-capability-matrix", default="")
    parser.add_argument("--benchmark-knowledge-domain-validation-matrix", default="")
    parser.add_argument("--benchmark-knowledge-domain-capability-drift", default="")
    parser.add_argument("--benchmark-knowledge-domain-action-plan", default="")
    parser.add_argument("--benchmark-knowledge-domain-control-plane", default="")
    parser.add_argument("--benchmark-knowledge-domain-control-plane-drift", default="")
    parser.add_argument("--benchmark-knowledge-domain-release-gate", default="")
    parser.add_argument(
        "--benchmark-knowledge-domain-release-readiness-matrix", default=""
    )
    parser.add_argument(
        "--benchmark-knowledge-domain-release-surface-alignment", default=""
    )
    parser.add_argument("--benchmark-knowledge-reference-inventory", default="")
    parser.add_argument("--benchmark-knowledge-source-action-plan", default="")
    parser.add_argument("--benchmark-knowledge-source-coverage", default="")
    parser.add_argument("--benchmark-knowledge-source-drift", default="")
    parser.add_argument("--benchmark-knowledge-outcome-correlation", default="")
    parser.add_argument("--benchmark-knowledge-outcome-drift", default="")
    parser.add_argument("--benchmark-competitive-surpass-index", default="")
    parser.add_argument("--benchmark-competitive-surpass-trend", default="")
    parser.add_argument("--benchmark-competitive-surpass-action-plan", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    artifact_paths = {
        "benchmark_release_decision": args.benchmark_release_decision,
        "benchmark_scorecard": args.benchmark_scorecard,
        "benchmark_operational_summary": args.benchmark_operational_summary,
        "benchmark_companion_summary": args.benchmark_companion_summary,
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
        "benchmark_knowledge_domain_capability_matrix": (
            args.benchmark_knowledge_domain_capability_matrix
        ),
        "benchmark_knowledge_domain_validation_matrix": (
            args.benchmark_knowledge_domain_validation_matrix
        ),
        "benchmark_knowledge_domain_capability_drift": (
            args.benchmark_knowledge_domain_capability_drift
        ),
        "benchmark_knowledge_domain_action_plan": (
            args.benchmark_knowledge_domain_action_plan
        ),
        "benchmark_knowledge_domain_control_plane": (
            args.benchmark_knowledge_domain_control_plane
        ),
        "benchmark_knowledge_domain_control_plane_drift": (
            args.benchmark_knowledge_domain_control_plane_drift
        ),
        "benchmark_knowledge_domain_release_gate": (
            args.benchmark_knowledge_domain_release_gate
        ),
        "benchmark_knowledge_domain_release_readiness_matrix": (
            args.benchmark_knowledge_domain_release_readiness_matrix
        ),
        "benchmark_knowledge_domain_release_surface_alignment": (
            args.benchmark_knowledge_domain_release_surface_alignment
        ),
        "benchmark_knowledge_reference_inventory": (
            args.benchmark_knowledge_reference_inventory
        ),
        "benchmark_knowledge_source_action_plan": (
            args.benchmark_knowledge_source_action_plan
        ),
        "benchmark_knowledge_source_coverage": (
            args.benchmark_knowledge_source_coverage
        ),
        "benchmark_knowledge_source_drift": args.benchmark_knowledge_source_drift,
        "benchmark_knowledge_outcome_correlation": (
            args.benchmark_knowledge_outcome_correlation
        ),
        "benchmark_knowledge_outcome_drift": args.benchmark_knowledge_outcome_drift,
        "benchmark_competitive_surpass_index": (
            args.benchmark_competitive_surpass_index
        ),
        "benchmark_competitive_surpass_trend": (
            args.benchmark_competitive_surpass_trend
        ),
        "benchmark_competitive_surpass_action_plan": (
            args.benchmark_competitive_surpass_action_plan
        ),
    }
    payload = build_release_runbook(
        title=args.title,
        benchmark_release_decision=_maybe_load_json(args.benchmark_release_decision),
        benchmark_scorecard=_maybe_load_json(args.benchmark_scorecard),
        benchmark_operational_summary=_maybe_load_json(
            args.benchmark_operational_summary
        ),
        benchmark_companion_summary=_maybe_load_json(args.benchmark_companion_summary),
        benchmark_artifact_bundle=_maybe_load_json(args.benchmark_artifact_bundle),
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
        benchmark_realdata_scorecard=_maybe_load_json(
            args.benchmark_realdata_scorecard
        ),
        benchmark_operator_adoption=_maybe_load_json(
            args.benchmark_operator_adoption
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
        benchmark_knowledge_domain_capability_matrix=_maybe_load_json(
            args.benchmark_knowledge_domain_capability_matrix
        ),
        benchmark_knowledge_domain_validation_matrix=_maybe_load_json(
            args.benchmark_knowledge_domain_validation_matrix
        ),
        benchmark_knowledge_domain_capability_drift=_maybe_load_json(
            args.benchmark_knowledge_domain_capability_drift
        ),
        benchmark_knowledge_domain_action_plan=_maybe_load_json(
            args.benchmark_knowledge_domain_action_plan
        ),
        benchmark_knowledge_domain_control_plane=_maybe_load_json(
            args.benchmark_knowledge_domain_control_plane
        ),
        benchmark_knowledge_domain_control_plane_drift=_maybe_load_json(
            args.benchmark_knowledge_domain_control_plane_drift
        ),
        benchmark_knowledge_domain_release_gate=_maybe_load_json(
            args.benchmark_knowledge_domain_release_gate
        ),
        benchmark_knowledge_domain_release_readiness_matrix=_maybe_load_json(
            args.benchmark_knowledge_domain_release_readiness_matrix
        ),
        benchmark_knowledge_domain_release_surface_alignment=_maybe_load_json(
            args.benchmark_knowledge_domain_release_surface_alignment
        ),
        benchmark_knowledge_reference_inventory=_maybe_load_json(
            args.benchmark_knowledge_reference_inventory
        ),
        benchmark_knowledge_source_action_plan=_maybe_load_json(
            args.benchmark_knowledge_source_action_plan
        ),
        benchmark_knowledge_source_coverage=_maybe_load_json(
            args.benchmark_knowledge_source_coverage
        ),
        benchmark_knowledge_source_drift=_maybe_load_json(
            args.benchmark_knowledge_source_drift
        ),
        benchmark_knowledge_outcome_correlation=_maybe_load_json(
            args.benchmark_knowledge_outcome_correlation
        ),
        benchmark_knowledge_outcome_drift=_maybe_load_json(
            args.benchmark_knowledge_outcome_drift
        ),
        benchmark_competitive_surpass_index=_maybe_load_json(
            args.benchmark_competitive_surpass_index
        ),
        benchmark_competitive_surpass_trend=_maybe_load_json(
            args.benchmark_competitive_surpass_trend
        ),
        benchmark_competitive_surpass_action_plan=_maybe_load_json(
            args.benchmark_competitive_surpass_action_plan
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
