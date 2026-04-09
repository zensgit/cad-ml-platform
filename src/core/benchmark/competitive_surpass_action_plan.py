"""Turn competitive-surpass signals into an executable action plan."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _text(value: Any) -> str:
    return str(value or "").strip()


def _compact(items: Iterable[Any], *, limit: int = 12) -> List[str]:
    out: List[str] = []
    for item in items:
        text = _text(item)
        if not text or text in out:
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _component(payload: Dict[str, Any], key: str) -> Dict[str, Any]:
    return payload.get(key) or payload or {}


def _tier(value: Any) -> str:
    status = _text(value).lower()
    if status in {
        "engineering_semantics_ready",
        "competitive_surpass_knowledge_ready",
        "knowledge_domain_action_plan_ready",
        "knowledge_foundation_ready",
        "realdata_foundation_ready",
        "realdata_scorecard_ready",
        "operator_ready",
        "aligned",
        "ready",
    }:
        return "ready"
    if status in {
        "partial_engineering_semantics",
        "competitive_surpass_knowledge_partial",
        "knowledge_domain_action_plan_partial",
        "knowledge_foundation_partial",
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


def _priority(*, tier: str, regressed: bool) -> str:
    if regressed or tier == "blocked":
        return "high"
    if tier == "partial":
        return "medium"
    return "low"


def _item_status(*, tier: str, regressed: bool) -> str:
    if regressed or tier == "blocked":
        return "blocked"
    if tier == "partial":
        return "required"
    return "ready"


def _first_recommendation(recommendations: Iterable[Any], fallback: str) -> str:
    items = _compact(recommendations, limit=1)
    return items[0] if items else fallback


def _pillar_action(
    *,
    pillar: str,
    status: str,
    summary: str,
    recommendations: Iterable[Any],
    regressed: bool = False,
    gaps: Iterable[Any] = (),
) -> Dict[str, Any] | None:
    tier = _tier(status)
    if tier == "ready" and not regressed:
        return None
    priority = _priority(tier=tier, regressed=regressed)
    return {
        "id": f"{pillar}:action",
        "pillar": pillar,
        "priority": priority,
        "status": _item_status(tier=tier, regressed=regressed),
        "trend_regressed": regressed,
        "summary": summary or "none",
        "gaps": _compact(gaps, limit=6),
        "action": _first_recommendation(
            recommendations,
            f"Stabilize {pillar} until competitive-surpass signals are ready.",
        ),
    }


def _trend_action(
    trend_component: Dict[str, Any],
    recommendations: Iterable[Any],
) -> Dict[str, Any] | None:
    status = _text(trend_component.get("status")) or "unknown"
    if status in {"stable", "improved", "unknown", ""}:
        return None
    priority = "high" if status == "regressed" else "medium"
    return {
        "id": "competitive_surpass:trend",
        "pillar": "competitive_surpass",
        "priority": priority,
        "status": "blocked" if priority == "high" else "required",
        "trend_regressed": status == "regressed",
        "summary": (
            f"status={status}; score_delta={trend_component.get('score_delta') or 0}; "
            "regressed="
            f"{', '.join(_compact(trend_component.get('pillar_regressions') or [])) or 'none'}; "
            "new_gaps="
            f"{', '.join(_compact(trend_component.get('new_primary_gaps') or [])) or 'none'}"
        ),
        "gaps": _compact(trend_component.get("new_primary_gaps") or [], limit=6),
        "action": _first_recommendation(
            recommendations,
            "Persist and review the current competitive-surpass baseline before promotion.",
        ),
    }


def build_competitive_surpass_action_plan(
    *,
    benchmark_competitive_surpass_index: Dict[str, Any],
    benchmark_competitive_surpass_trend: Dict[str, Any],
    benchmark_engineering_signals: Dict[str, Any],
    benchmark_knowledge_domain_action_plan: Dict[str, Any],
    benchmark_realdata_signals: Dict[str, Any],
    benchmark_realdata_scorecard: Dict[str, Any],
    benchmark_operator_adoption: Dict[str, Any],
) -> Dict[str, Any]:
    """Build an executable action plan from competitive-surpass benchmark inputs."""
    index_component = _component(
        benchmark_competitive_surpass_index, "competitive_surpass_index"
    )
    trend_component = _component(
        benchmark_competitive_surpass_trend, "competitive_surpass_trend"
    )
    engineering_component = _component(
        benchmark_engineering_signals, "engineering_signals"
    )
    knowledge_component = _component(
        benchmark_knowledge_domain_action_plan, "knowledge_domain_action_plan"
    )
    realdata_component = _component(
        benchmark_realdata_scorecard, "realdata_scorecard"
    )
    if not realdata_component:
        realdata_component = _component(
            benchmark_realdata_signals, "realdata_signals"
        )
    operator_component = benchmark_operator_adoption or {}
    knowledge_priority_domain_summary = (
        ", ".join(_compact(knowledge_component.get("priority_domains") or [])) or "none"
    )
    operator_release_alignment_status = (
        operator_component.get("release_surface_alignment_status") or "unknown"
    )

    pillar_regressions = set(_compact(trend_component.get("pillar_regressions") or []))
    primary_gaps = _compact(index_component.get("primary_gaps") or [], limit=12)

    actions: List[Dict[str, Any]] = []
    actions.extend(
        filter(
            None,
            [
                _pillar_action(
                    pillar="engineering",
                    status=_text(engineering_component.get("status")) or "unknown",
                    summary=(
                        f"coverage={engineering_component.get('coverage_ratio', 0.0)}; "
                        f"rows_with_candidates="
                        f"{engineering_component.get('rows_with_standards_candidates', 0)}"
                    ),
                    recommendations=benchmark_engineering_signals.get("recommendations")
                    or [],
                    regressed="engineering" in pillar_regressions,
                    gaps=[gap for gap in primary_gaps if "engineering" in gap],
                ),
                _pillar_action(
                    pillar="knowledge",
                    status=_text(knowledge_component.get("status")) or "unknown",
                    summary=(
                        f"actions={knowledge_component.get('total_action_count', 0)}; "
                        f"priority_domains={knowledge_priority_domain_summary}"
                    ),
                    recommendations=benchmark_knowledge_domain_action_plan.get(
                        "recommendations"
                    )
                    or [],
                    regressed="knowledge" in pillar_regressions,
                    gaps=[gap for gap in primary_gaps if "knowledge" in gap],
                ),
                _pillar_action(
                    pillar="realdata",
                    status=_text(realdata_component.get("status")) or "unknown",
                    summary=(
                        f"ready={realdata_component.get('ready_component_count', 0)}; "
                        f"partial={realdata_component.get('partial_component_count', 0)}; "
                        f"blocked={realdata_component.get('environment_blocked_count', 0)}"
                    ),
                    recommendations=(benchmark_realdata_scorecard.get("recommendations"))
                    or (benchmark_realdata_signals.get("recommendations"))
                    or [],
                    regressed="realdata" in pillar_regressions,
                    gaps=[
                        gap
                        for gap in primary_gaps
                        if "realdata" in gap or "step" in gap
                    ],
                ),
                _pillar_action(
                    pillar="operator_adoption",
                    status=_text(operator_component.get("adoption_readiness"))
                    or "unknown",
                    summary=(
                        f"mode={operator_component.get('operator_mode') or 'unknown'}; "
                        f"release={operator_release_alignment_status}"
                    ),
                    recommendations=operator_component.get("recommended_actions") or [],
                    regressed="operator_adoption" in pillar_regressions,
                    gaps=[gap for gap in primary_gaps if "operator" in gap],
                ),
                _pillar_action(
                    pillar="release_alignment",
                    status=_text(
                        operator_component.get("release_surface_alignment_status")
                    )
                    or "unknown",
                    summary=_text(
                        operator_component.get("release_surface_alignment_summary")
                    )
                    or "none",
                    recommendations=operator_component.get("recommended_actions") or [],
                    regressed="release_alignment" in pillar_regressions,
                    gaps=[gap for gap in primary_gaps if "release" in gap],
                ),
                _trend_action(
                    trend_component,
                    benchmark_competitive_surpass_trend.get("recommendations") or [],
                ),
            ],
        )
    )

    actions.sort(
        key=lambda item: (
            {"high": 0, "medium": 1, "low": 2}.get(item.get("priority") or "", 3),
            item.get("pillar") or "",
            item.get("id") or "",
        )
    )

    pillar_action_counts: Dict[str, int] = {}
    for item in actions:
        pillar = _text(item.get("pillar")) or "unknown"
        pillar_action_counts[pillar] = pillar_action_counts.get(pillar, 0) + 1

    ready_pillars = [
        pillar
        for pillar in (
            "engineering",
            "knowledge",
            "realdata",
            "operator_adoption",
            "release_alignment",
        )
        if pillar_action_counts.get(pillar, 0) == 0
    ]
    priority_pillars = _compact(
        [item.get("pillar") for item in actions if item.get("priority") == "high"]
        + [item.get("pillar") for item in actions if item.get("priority") == "medium"],
        limit=8,
    )
    high_priority_actions = [item for item in actions if item.get("priority") == "high"]
    medium_priority_actions = [
        item for item in actions if item.get("priority") == "medium"
    ]

    if not actions:
        status = "competitive_surpass_action_plan_ready"
    elif high_priority_actions:
        status = "competitive_surpass_action_plan_blocked"
    else:
        status = "competitive_surpass_action_plan_partial"

    return {
        "status": status,
        "total_action_count": len(actions),
        "high_priority_action_count": len(high_priority_actions),
        "medium_priority_action_count": len(medium_priority_actions),
        "pillar_action_counts": pillar_action_counts,
        "priority_pillars": priority_pillars,
        "ready_pillars": ready_pillars,
        "recommended_first_actions": actions[:3],
        "actions": actions,
    }


def competitive_surpass_action_plan_recommendations(
    component: Dict[str, Any],
) -> List[str]:
    status = _text(component.get("status")) or "unknown"
    if status == "competitive_surpass_action_plan_ready":
        return [
            "Competitive surpass action plan is clear: no blocked or required actions remain."
        ]
    recommendations: List[str] = []
    for item in component.get("actions") or []:
        if item.get("priority") not in {"high", "medium"}:
            continue
        recommendations.append(
            f"{item.get('pillar')}: {item.get('action')} ({item.get('summary')})"
        )
    return _compact(recommendations, limit=12)


def render_competitive_surpass_action_plan_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    component = payload.get("competitive_surpass_action_plan") or {}
    lines = [
        f"# {title}",
        "",
        "## Overview",
        "",
        f"- `status`: `{component.get('status') or 'unknown'}`",
        f"- `total_action_count`: `{component.get('total_action_count') or 0}`",
        f"- `high_priority_action_count`: `{component.get('high_priority_action_count') or 0}`",
        f"- `medium_priority_action_count`: `{component.get('medium_priority_action_count') or 0}`",
        "- `priority_pillars`: `"
        + (", ".join(_compact(component.get("priority_pillars") or [])) or "none")
        + "`",
        "",
        "## Actions",
        "",
    ]
    actions = component.get("actions") or []
    if actions:
        for item in actions:
            lines.append(
                f"- `{item.get('id')}` pillar=`{item.get('pillar')}` "
                f"priority=`{item.get('priority')}` status=`{item.get('status')}`"
            )
            lines.append(f"  summary: {item.get('summary') or 'none'}")
            lines.append(f"  action: {item.get('action') or 'none'}")
    else:
        lines.append("- none")
    recommendations = payload.get("recommendations") or []
    lines.extend(["", "## Recommendations", ""])
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"
