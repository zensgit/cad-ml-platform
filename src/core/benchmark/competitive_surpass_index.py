"""Benchmark helpers for a unified competitive-surpass index."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _compact(items: Iterable[Any], *, limit: int = 6) -> List[str]:
    out: List[str] = []
    for item in items:
        text = _text(item)
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _normalize_tier(status: str) -> str:
    value = _text(status).lower()
    if value in {
        "engineering_semantics_ready",
        "knowledge_foundation_ready",
        "knowledge_application_ready",
        "knowledge_realdata_ready",
        "knowledge_domain_matrix_ready",
        "knowledge_domain_action_plan_ready",
        "knowledge_source_coverage_ready",
        "knowledge_source_action_plan_ready",
        "stable",
        "improved",
        "knowledge_outcome_correlation_ready",
        "competitive_surpass_knowledge_ready",
        "competitive_surpass_realdata_ready",
        "realdata_foundation_ready",
        "realdata_scorecard_ready",
        "operator_ready",
        "aligned",
        "ready",
    }:
        return "ready"
    if value in {
        "partial_engineering_semantics",
        "knowledge_foundation_partial",
        "knowledge_application_partial",
        "knowledge_realdata_partial",
        "knowledge_domain_matrix_partial",
        "knowledge_domain_action_plan_partial",
        "knowledge_source_coverage_partial",
        "knowledge_source_action_plan_partial",
        "knowledge_outcome_correlation_partial",
        "competitive_surpass_knowledge_partial",
        "competitive_surpass_realdata_partial",
        "mixed",
        "realdata_foundation_partial",
        "realdata_scorecard_partial",
        "guided_manual",
        "assisted_review",
        "review_required",
        "diverged",
        "partial",
        "weak",
    }:
        return "partial"
    return "blocked"


def _tier_weight(tier: str) -> float:
    return {"ready": 1.0, "partial": 0.5}.get(tier, 0.0)


def _pillar_row(name: str, status: str, summary: str, details: Dict[str, Any]) -> Dict[str, Any]:
    tier = _normalize_tier(status)
    return {
        "name": name,
        "status": _text(status) or "unknown",
        "tier": tier,
        "summary": _text(summary) or "none",
        "details": details,
    }


def _engineering_pillar(
    benchmark_engineering_signals: Dict[str, Any]
) -> Dict[str, Any]:
    component = (
        benchmark_engineering_signals.get("engineering_signals")
        or benchmark_engineering_signals
    )
    status = _text(component.get("status")) or "missing"
    recommendations = _compact(benchmark_engineering_signals.get("recommendations") or [])
    summary = (
        f"coverage={component.get('coverage_ratio', 0.0)}; "
        f"standards_rows={component.get('rows_with_standards_candidates', 0)}; "
        f"violations_rows={component.get('rows_with_violations', 0)}"
    )
    if recommendations:
        summary = f"{summary}; next={recommendations[0]}"
    return _pillar_row(
        "engineering",
        status,
        summary,
        {
            "coverage_ratio": component.get("coverage_ratio", 0.0),
            "recommendations": recommendations,
        },
    )


def _knowledge_pillar(
    benchmark_knowledge_readiness: Dict[str, Any],
    benchmark_knowledge_application: Dict[str, Any],
    benchmark_knowledge_realdata_correlation: Dict[str, Any],
    benchmark_knowledge_domain_matrix: Dict[str, Any],
    benchmark_knowledge_domain_action_plan: Dict[str, Any] | None,
    benchmark_knowledge_source_coverage: Dict[str, Any] | None,
    benchmark_knowledge_source_action_plan: Dict[str, Any] | None,
    benchmark_knowledge_source_drift: Dict[str, Any] | None,
    benchmark_knowledge_outcome_correlation: Dict[str, Any],
    benchmark_knowledge_outcome_drift: Dict[str, Any],
) -> Dict[str, Any]:
    readiness = (
        benchmark_knowledge_readiness.get("knowledge_readiness")
        or benchmark_knowledge_readiness
    )
    application = (
        benchmark_knowledge_application.get("knowledge_application")
        or benchmark_knowledge_application
    )
    realdata = (
        benchmark_knowledge_realdata_correlation.get("knowledge_realdata_correlation")
        or benchmark_knowledge_realdata_correlation
    )
    domain_matrix = (
        benchmark_knowledge_domain_matrix.get("knowledge_domain_matrix")
        or benchmark_knowledge_domain_matrix
    )
    action_plan_root = benchmark_knowledge_domain_action_plan or {}
    action_plan = (
        action_plan_root.get("knowledge_domain_action_plan")
        or action_plan_root
        or {}
    )
    source_coverage_root = benchmark_knowledge_source_coverage or {}
    source_coverage = (
        source_coverage_root.get("knowledge_source_coverage")
        or source_coverage_root
        or {}
    )
    source_action_plan_root = benchmark_knowledge_source_action_plan or {}
    source_action_plan = (
        source_action_plan_root.get("knowledge_source_action_plan")
        or source_action_plan_root
        or {}
    )
    source_drift_root = benchmark_knowledge_source_drift or {}
    source_drift = (
        source_drift_root.get("knowledge_source_drift")
        or source_drift_root
        or {}
    )
    outcome_corr = (
        benchmark_knowledge_outcome_correlation.get("knowledge_outcome_correlation")
        or benchmark_knowledge_outcome_correlation
    )
    outcome_drift = (
        benchmark_knowledge_outcome_drift.get("knowledge_outcome_drift")
        or benchmark_knowledge_outcome_drift
    )
    rows = {
        "readiness": _text(readiness.get("status")) or "missing",
        "application": _text(application.get("status")) or "missing",
        "realdata": _text(realdata.get("status")) or "missing",
        "domain_matrix": _text(domain_matrix.get("status")) or "missing",
        "action_plan": _text(action_plan.get("status")) or "missing",
        "source_drift": _text(source_drift.get("status")) or "unknown",
        "outcome_correlation": _text(outcome_corr.get("status")) or "missing",
        "outcome_drift": _text(outcome_drift.get("status")) or "unknown",
    }
    source_coverage_status = _text(source_coverage.get("status"))
    if source_coverage_status:
        rows["source_coverage"] = source_coverage_status
    source_action_plan_status = _text(source_action_plan.get("status"))
    if source_action_plan_status:
        rows["source_action_plan"] = source_action_plan_status
    tiers = {name: _normalize_tier(status) for name, status in rows.items()}
    if all(tier == "ready" for tier in tiers.values()):
        status = "competitive_surpass_knowledge_ready"
    elif any(tier == "blocked" for tier in tiers.values()):
        status = "competitive_surpass_knowledge_partial"
    elif any(tier == "partial" for tier in tiers.values()):
        status = "competitive_surpass_knowledge_partial"
    else:
        status = "competitive_surpass_knowledge_missing"
    focus_areas = _compact(
        [row.get("component") for row in readiness.get("focus_areas_detail") or []]
        + [row.get("domain") for row in application.get("focus_areas_detail") or []]
        + [row.get("domain") for row in realdata.get("focus_areas") or []]
        + list(domain_matrix.get("priority_domains") or [])
        + list(action_plan.get("priority_domains") or [])
        + list(source_coverage.get("priority_domains") or [])
        + list(source_action_plan.get("priority_domains") or [])
        + list(source_drift.get("new_priority_domains") or [])
        + list(source_drift.get("resolved_priority_domains") or [])
        + [row.get("name") for row in source_coverage.get("expansion_candidates") or []]
        + [item.get("source_group") for item in source_action_plan.get("actions") or []]
        + list(outcome_corr.get("priority_domains") or [])
        + list(outcome_drift.get("new_priority_domains") or []),
        limit=6,
    )
    summary = (
        f"readiness={rows['readiness']}; application={rows['application']}; "
        f"realdata={rows['realdata']}; domain_matrix={rows['domain_matrix']}; "
        f"action_plan={rows['action_plan']}; source_drift={rows['source_drift']}; "
        f"outcome={rows['outcome_correlation']}; drift={rows['outcome_drift']}"
    )
    if source_coverage_status:
        summary = f"{summary}; source_coverage={source_coverage_status}"
    if source_action_plan_status:
        summary = f"{summary}; source_action_plan={source_action_plan_status}"
    if focus_areas:
        summary = f"{summary}; focus={', '.join(focus_areas)}"
    return _pillar_row(
        "knowledge",
        status,
        summary,
        {
            "component_statuses": rows,
            "focus_areas": focus_areas,
            "expansion_candidates": _compact(
                [row.get("name") for row in source_coverage.get("expansion_candidates") or []],
                limit=6,
            ),
            "source_action_items": _compact(
                [item.get("id") for item in source_action_plan.get("actions") or []],
                limit=6,
            ),
            "source_group_regressions": _compact(
                source_drift.get("source_group_regressions") or [],
                limit=6,
            ),
            "source_group_improvements": _compact(
                source_drift.get("source_group_improvements") or [],
                limit=6,
            ),
        },
    )


def _realdata_pillar(
    benchmark_realdata_signals: Dict[str, Any],
    benchmark_realdata_scorecard: Dict[str, Any],
) -> Dict[str, Any]:
    signals = (
        benchmark_realdata_signals.get("realdata_signals")
        or benchmark_realdata_signals
    )
    scorecard = (
        benchmark_realdata_scorecard.get("realdata_scorecard")
        or benchmark_realdata_scorecard
    )
    signal_status = _text(signals.get("status")) or "missing"
    scorecard_status = _text(scorecard.get("status")) or "missing"
    signal_tier = _normalize_tier(signal_status)
    scorecard_tier = _normalize_tier(scorecard_status)
    if signal_tier == "ready" and scorecard_tier == "ready":
        status = "competitive_surpass_realdata_ready"
    elif signal_tier == "blocked" and scorecard_tier == "blocked":
        status = "competitive_surpass_realdata_missing"
    else:
        status = "competitive_surpass_realdata_partial"
    summary = (
        f"signals={signal_status}; scorecard={scorecard_status}; "
        f"ready_components={signals.get('ready_component_count', 0)}"
    )
    best_surface = _text(scorecard.get("best_surface"))
    if best_surface:
        summary = f"{summary}; best_surface={best_surface}"
    return _pillar_row(
        "realdata",
        status,
        summary,
        {
            "signals_status": signal_status,
            "scorecard_status": scorecard_status,
            "best_surface": best_surface,
        },
    )


def _operator_adoption_pillar(benchmark_operator_adoption: Dict[str, Any]) -> Dict[str, Any]:
    readiness = (
        _text(benchmark_operator_adoption.get("adoption_readiness"))
        or _text(benchmark_operator_adoption.get("status"))
        or "missing"
    )
    mode = _text(benchmark_operator_adoption.get("operator_mode")) or "unknown"
    drift_status = _text(benchmark_operator_adoption.get("knowledge_outcome_drift_status"))
    summary = f"readiness={readiness}; mode={mode}"
    if drift_status:
        summary = f"{summary}; outcome_drift={drift_status}"
    return _pillar_row(
        "operator_adoption",
        readiness,
        summary,
        {
            "mode": mode,
            "knowledge_outcome_drift_status": drift_status or "unknown",
            "recommended_actions": _compact(
                benchmark_operator_adoption.get("recommended_actions")
                or benchmark_operator_adoption.get("actions")
                or [],
                limit=4,
            ),
        },
    )


def _release_alignment_pillar(benchmark_operator_adoption: Dict[str, Any]) -> Dict[str, Any]:
    alignment = benchmark_operator_adoption.get("release_surface_alignment") or {}
    status = (
        _text(benchmark_operator_adoption.get("release_surface_alignment_status"))
        or _text(alignment.get("status"))
        or "unavailable"
    )
    summary = (
        _text(benchmark_operator_adoption.get("release_surface_alignment_summary"))
        or _text(alignment.get("summary"))
        or "none"
    )
    mismatches = list(alignment.get("mismatches") or [])
    if mismatches:
        summary = f"{summary}; mismatches={'; '.join(_compact(mismatches, limit=4))}"
    return _pillar_row(
        "release_alignment",
        status,
        summary,
        {"mismatches": mismatches},
    )


def _overall_status(pillars: Dict[str, Dict[str, Any]]) -> Tuple[str, int]:
    rows = list(pillars.values())
    weights = [_tier_weight(row["tier"]) for row in rows]
    score = int(round((sum(weights) / float(len(rows) or 1)) * 100))
    blocked = [row["name"] for row in rows if row["tier"] == "blocked"]
    partial = [row["name"] for row in rows if row["tier"] == "partial"]
    if not blocked and not partial:
        return "competitive_surpass_ready", score
    if len(blocked) >= 2 or ("realdata" in blocked and "release_alignment" in blocked):
        return "competitive_surpass_blocked", score
    return "competitive_surpass_attention_required", score


def build_competitive_surpass_index(
    *,
    benchmark_engineering_signals: Dict[str, Any],
    benchmark_knowledge_readiness: Dict[str, Any],
    benchmark_knowledge_application: Dict[str, Any],
    benchmark_knowledge_realdata_correlation: Dict[str, Any],
    benchmark_knowledge_domain_matrix: Dict[str, Any],
    benchmark_knowledge_domain_action_plan: Dict[str, Any] | None = None,
    benchmark_knowledge_source_coverage: Dict[str, Any] | None = None,
    benchmark_knowledge_source_action_plan: Dict[str, Any] | None = None,
    benchmark_knowledge_source_drift: Dict[str, Any] | None = None,
    benchmark_knowledge_outcome_correlation: Dict[str, Any],
    benchmark_knowledge_outcome_drift: Dict[str, Any],
    benchmark_realdata_signals: Dict[str, Any],
    benchmark_realdata_scorecard: Dict[str, Any],
    benchmark_operator_adoption: Dict[str, Any],
) -> Dict[str, Any]:
    pillars = {
        "engineering": _engineering_pillar(benchmark_engineering_signals),
        "knowledge": _knowledge_pillar(
            benchmark_knowledge_readiness,
            benchmark_knowledge_application,
            benchmark_knowledge_realdata_correlation,
            benchmark_knowledge_domain_matrix,
            benchmark_knowledge_domain_action_plan,
            benchmark_knowledge_source_coverage,
            benchmark_knowledge_source_action_plan,
            benchmark_knowledge_source_drift,
            benchmark_knowledge_outcome_correlation,
            benchmark_knowledge_outcome_drift,
        ),
        "realdata": _realdata_pillar(
            benchmark_realdata_signals, benchmark_realdata_scorecard
        ),
        "operator_adoption": _operator_adoption_pillar(benchmark_operator_adoption),
        "release_alignment": _release_alignment_pillar(benchmark_operator_adoption),
    }
    status, score = _overall_status(pillars)
    blocked = [row["name"] for row in pillars.values() if row["tier"] == "blocked"]
    partial = [row["name"] for row in pillars.values() if row["tier"] == "partial"]
    primary_gaps = blocked or partial
    return {
        "status": status,
        "score": score,
        "pillars": pillars,
        "ready_pillars": [row["name"] for row in pillars.values() if row["tier"] == "ready"],
        "partial_pillars": partial,
        "blocked_pillars": blocked,
        "primary_gaps": primary_gaps,
    }


def competitive_surpass_index_recommendations(component: Dict[str, Any]) -> List[str]:
    gaps = list(component.get("primary_gaps") or [])
    recommendations: List[str] = []
    if "engineering" in gaps:
        recommendations.append(
            "Raise engineering signal coverage until standards, violations, and hints are stable."
        )
    if "knowledge" in gaps:
        recommendations.append(
            "Close tolerance/standards/GD&T knowledge gaps before claiming "
            "benchmark surpass readiness."
        )
    if "realdata" in gaps:
        recommendations.append(
            "Expand DXF/STEP/history real-data validation so benchmark claims "
            "are grounded in production evidence."
        )
    if "operator_adoption" in gaps:
        recommendations.append(
            "Reduce operator friction and align review guidance before "
            "promoting the benchmark to release mode."
        )
    if "release_alignment" in gaps:
        recommendations.append(
            "Keep release decision, runbook, and operator adoption surfaces "
            "aligned so release guidance stays trustworthy."
        )
    if not recommendations:
        recommendations.append(
            "Competitive-surpass benchmark pillars are aligned across "
            "engineering, knowledge, real-data, operator adoption, and "
            "release surfaces."
        )
    return recommendations


def render_competitive_surpass_markdown(payload: Dict[str, Any], title: str) -> str:
    component = payload.get("competitive_surpass_index") or {}
    pillars = component.get("pillars") or {}
    recommendations = payload.get("recommendations") or []
    lines = [
        f"# {title}",
        "",
        "## Overview",
        "",
        f"- `status`: `{component.get('status', 'competitive_surpass_blocked')}`",
        f"- `score`: `{component.get('score', 0)}`",
        f"- `ready_pillars`: `{component.get('ready_pillars', [])}`",
        f"- `partial_pillars`: `{component.get('partial_pillars', [])}`",
        f"- `blocked_pillars`: `{component.get('blocked_pillars', [])}`",
        f"- `primary_gaps`: `{component.get('primary_gaps', [])}`",
        "",
        "## Pillars",
        "",
    ]
    for name in (
        "engineering",
        "knowledge",
        "realdata",
        "operator_adoption",
        "release_alignment",
    ):
        row = pillars.get(name) or {}
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- `status`: `{row.get('status', 'unknown')}`")
        lines.append(f"- `tier`: `{row.get('tier', 'blocked')}`")
        lines.append(f"- `summary`: `{row.get('summary', 'none')}`")
        lines.append("")
    lines.extend(["## Recommendations", ""])
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- No recommendations.")
    lines.append("")
    return "\n".join(lines)
