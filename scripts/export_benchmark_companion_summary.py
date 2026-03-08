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


def _component_statuses(
    scorecard: Dict[str, Any],
    operational_summary: Dict[str, Any],
    artifact_bundle: Dict[str, Any],
    knowledge_readiness_summary: Dict[str, Any],
    knowledge_drift_summary: Dict[str, Any],
    engineering_signals_summary: Dict[str, Any],
    operator_adoption_summary: Dict[str, Any],
) -> Dict[str, str]:
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
        "operator_adoption": str(
            operator_adoption_summary.get("adoption_readiness") or "unknown"
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
    operator_adoption_path: str,
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
        "benchmark_operator_adoption": row(
            "benchmark_operator_adoption", operator_adoption_path
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


def build_companion_summary(
    *,
    title: str,
    benchmark_scorecard: Dict[str, Any],
    benchmark_operational_summary: Dict[str, Any],
    benchmark_artifact_bundle: Dict[str, Any],
    benchmark_knowledge_readiness: Dict[str, Any],
    benchmark_knowledge_drift: Dict[str, Any],
    benchmark_engineering_signals: Dict[str, Any],
    benchmark_operator_adoption: Dict[str, Any],
    artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
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
    operator_adoption_recommendations = (
        benchmark_operator_adoption.get("recommended_actions") or []
    )
    recommendations = _compact(
        bundle_recommendations
        or operational_recommendations
        or knowledge_drift_recommendations
        or operator_adoption_recommendations
        or engineering_recommendations
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
        benchmark_operator_adoption,
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
        artifact_paths.get("benchmark_operator_adoption", ""),
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
        "operator_adoption_knowledge_drift": operator_adoption_knowledge_drift,
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
    drift_recommendations = payload.get("knowledge_drift_recommendations") or []
    if drift_recommendations:
        lines.extend(f"- recommendation: {item}" for item in drift_recommendations)
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
    lines.extend(["", "## Operator Adoption Knowledge Drift", ""])
    drift = payload.get("operator_adoption_knowledge_drift") or {}
    lines.append(f"- `status`: `{drift.get('status') or 'unknown'}`")
    lines.append(f"- `summary`: {drift.get('summary') or 'none'}")
    drift_recommendations = drift.get("recommendations") or []
    if drift_recommendations:
        for item in drift_recommendations:
            lines.append(f"- recommendation: {item}")
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
    parser.add_argument("--benchmark-operator-adoption", default="")
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
        "benchmark_operator_adoption": args.benchmark_operator_adoption,
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
        benchmark_operator_adoption=_maybe_load_json(args.benchmark_operator_adoption),
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
