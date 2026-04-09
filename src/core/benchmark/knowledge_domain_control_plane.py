"""Unified benchmark control-plane for standards, tolerance, and GD&T."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


DOMAIN_SPECS: Dict[str, Dict[str, str]] = {
    "tolerance": {"label": "Tolerance & Fits"},
    "standards": {"label": "Standards & Design Tables"},
    "gdt": {"label": "GD&T & Datums"},
}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _compact(items: Iterable[Any], *, limit: int = 10) -> List[str]:
    out: List[str] = []
    for item in items:
        text = _text(item)
        if not text or text in out:
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _status_tier(status: Any) -> str:
    text = _text(status).lower()
    if text in {"ready", "healthy", "aligned"} or text.endswith("_ready"):
        return "ready"
    if not text or text in {"missing", "blocked"} or text.endswith("_missing"):
        return "blocked"
    return "partial"


def _drift_status(
    domain: str,
    drift_component: Dict[str, Any],
) -> str:
    regressions = {
        _text(item)
        for item in list(drift_component.get("domain_regressions") or [])
        + list(drift_component.get("new_priority_domains") or [])
    }
    improvements = {
        _text(item)
        for item in list(drift_component.get("domain_improvements") or [])
        + list(drift_component.get("resolved_priority_domains") or [])
    }
    if domain in regressions:
        return "regressed"
    if domain in improvements:
        return "improved"
    return "stable"


def _next_action(domain: str, action_plan_component: Dict[str, Any]) -> Dict[str, Any]:
    for item in action_plan_component.get("actions") or []:
        if _text(item.get("domain")) == domain:
            return dict(item)
    return {}


def _domain_status(
    *,
    capability_status: str,
    realdata_status: str,
    outcome_status: str,
    drift_status: str,
    high_priority_action_count: int,
    medium_priority_action_count: int,
) -> str:
    tiers = [
        _status_tier(capability_status),
        _status_tier(realdata_status),
        _status_tier(outcome_status),
    ]
    if (
        all(tier == "ready" for tier in tiers)
        and drift_status != "regressed"
        and high_priority_action_count == 0
        and medium_priority_action_count == 0
    ):
        return "ready"
    if (
        all(tier == "blocked" for tier in tiers)
        and drift_status == "stable"
        and high_priority_action_count == 0
        and medium_priority_action_count == 0
    ):
        return "missing"
    if (
        "blocked" in tiers
        or drift_status == "regressed"
        or high_priority_action_count > 0
    ):
        return "blocked"
    return "partial"


def build_knowledge_domain_control_plane(
    *,
    benchmark_knowledge_domain_capability_matrix: Dict[str, Any],
    benchmark_knowledge_domain_capability_drift: Dict[str, Any],
    benchmark_knowledge_realdata_correlation: Dict[str, Any],
    benchmark_knowledge_outcome_correlation: Dict[str, Any],
    benchmark_knowledge_domain_action_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a release-facing control-plane over standards/tolerance/GD&T."""
    capability_component = (
        benchmark_knowledge_domain_capability_matrix.get(
            "knowledge_domain_capability_matrix"
        )
        or benchmark_knowledge_domain_capability_matrix
        or {}
    )
    drift_component = (
        benchmark_knowledge_domain_capability_drift.get(
            "knowledge_domain_capability_drift"
        )
        or benchmark_knowledge_domain_capability_drift
        or {}
    )
    realdata_component = (
        benchmark_knowledge_realdata_correlation.get("knowledge_realdata_correlation")
        or benchmark_knowledge_realdata_correlation
        or {}
    )
    outcome_component = (
        benchmark_knowledge_outcome_correlation.get("knowledge_outcome_correlation")
        or benchmark_knowledge_outcome_correlation
        or {}
    )
    action_plan_component = (
        benchmark_knowledge_domain_action_plan.get("knowledge_domain_action_plan")
        or benchmark_knowledge_domain_action_plan
        or {}
    )

    capability_domains = capability_component.get("domains") or {}
    realdata_domains = realdata_component.get("domains") or {}
    outcome_domains = outcome_component.get("domains") or {}

    domains: Dict[str, Dict[str, Any]] = {}
    focus_areas_detail: List[Dict[str, Any]] = []
    release_blockers: List[str] = []
    ready_count = 0
    partial_count = 0
    blocked_count = 0
    missing_count = 0
    total_action_count = 0
    high_priority_action_count = 0

    for domain, spec in DOMAIN_SPECS.items():
        capability_row = capability_domains.get(domain) or {}
        realdata_row = realdata_domains.get(domain) or {}
        outcome_row = outcome_domains.get(domain) or {}
        action_item = _next_action(domain, action_plan_component)
        action_priority = _text(action_item.get("priority")).lower()
        high_count = 1 if action_priority == "high" else 0
        medium_count = 1 if action_priority == "medium" else 0
        total_action_count += 1 if action_item else 0
        high_priority_action_count += high_count
        drift_status = _drift_status(domain, drift_component)
        capability_status = _text(capability_row.get("status")) or "missing"
        realdata_status = _text(realdata_row.get("status")) or "missing"
        outcome_status = _text(outcome_row.get("status")) or "missing"
        status = _domain_status(
            capability_status=capability_status,
            realdata_status=realdata_status,
            outcome_status=outcome_status,
            drift_status=drift_status,
            high_priority_action_count=high_count,
            medium_priority_action_count=medium_count,
        )
        if status in {"blocked", "missing"}:
            priority = "high"
        elif status == "partial":
            priority = "medium"
        else:
            priority = "low"
        primary_gaps = _compact(
            list(capability_row.get("primary_gaps") or [])
            + (
                [f"realdata:{realdata_status}"]
                if _status_tier(realdata_status) != "ready"
                else []
            )
            + (
                [f"outcome:{outcome_status}"]
                if _status_tier(outcome_status) != "ready"
                else []
            )
            + (
                [f"capability_drift:{drift_status}"]
                if drift_status != "stable"
                else []
            )
            + (
                [f"action_plan:{action_item.get('stage')}"]
                if action_item
                else []
            ),
            limit=10,
        )
        release_blocker = status in {"blocked", "missing"}
        row = {
            "domain": domain,
            "label": _text(capability_row.get("label")) or spec["label"],
            "status": status,
            "priority": priority,
            "release_blocker": release_blocker,
            "capability_status": capability_status,
            "foundation_status": _text(capability_row.get("foundation_status"))
            or "missing",
            "application_status": _text(capability_row.get("application_status"))
            or "missing",
            "matrix_status": _text(capability_row.get("matrix_status")) or "missing",
            "provider_status": _text(capability_row.get("provider_status"))
            or "missing",
            "surface_status": _text(capability_row.get("surface_status"))
            or "missing",
            "realdata_status": realdata_status,
            "outcome_status": outcome_status,
            "drift_status": drift_status,
            "action_item": action_item,
            "action_stage": _text(action_item.get("stage")) or "none",
            "action_priority": action_priority or "none",
            "high_priority_action_count": high_count,
            "medium_priority_action_count": medium_count,
            "next_action": _text(action_item.get("action"))
            or _text(capability_row.get("action"))
            or _text(realdata_row.get("action"))
            or _text(outcome_row.get("action")),
            "primary_gaps": primary_gaps,
            "focus_components": list(capability_row.get("focus_components") or []),
            "missing_metrics": list(capability_row.get("missing_metrics") or []),
            "realdata_components": dict(realdata_row.get("realdata_components") or {}),
            "missing_realdata_components": list(
                realdata_row.get("missing_realdata_components") or []
            ),
            "best_surface": _text(outcome_row.get("best_surface")) or "none",
            "best_surface_score": outcome_row.get("best_surface_score", 0.0),
            "weak_surfaces": list(outcome_row.get("weak_surfaces") or []),
            "missing_surfaces": list(outcome_row.get("missing_surfaces") or []),
        }
        domains[domain] = row
        if release_blocker:
            release_blockers.append(domain)
            focus_areas_detail.append(row)
        elif status == "partial":
            focus_areas_detail.append(row)

        if status == "ready":
            ready_count += 1
        elif status == "partial":
            partial_count += 1
        elif status == "missing":
            missing_count += 1
        else:
            blocked_count += 1

    if ready_count == len(DOMAIN_SPECS):
        overall_status = "knowledge_domain_control_plane_ready"
    elif blocked_count > 0 or missing_count > 0:
        overall_status = "knowledge_domain_control_plane_blocked"
    elif partial_count > 0:
        overall_status = "knowledge_domain_control_plane_partial"
    else:
        overall_status = "knowledge_domain_control_plane_missing"

    return {
        "status": overall_status,
        "ready_domain_count": ready_count,
        "partial_domain_count": partial_count,
        "blocked_domain_count": blocked_count,
        "missing_domain_count": missing_count,
        "total_domain_count": len(DOMAIN_SPECS),
        "release_blockers": release_blockers,
        "priority_domains": [row["domain"] for row in focus_areas_detail],
        "focus_areas": [row["domain"] for row in focus_areas_detail],
        "focus_areas_detail": focus_areas_detail,
        "domains": domains,
        "total_action_count": total_action_count,
        "high_priority_action_count": high_priority_action_count,
    }


def knowledge_domain_control_plane_recommendations(
    component: Dict[str, Any],
) -> List[str]:
    status = _text(component.get("status")).lower()
    if status == "knowledge_domain_control_plane_ready":
        return [
            "Standards, tolerance, and GD&T domains are aligned across capability, "
            "real-data, outcome, and action-plan benchmark surfaces."
        ]

    items: List[str] = []
    for row in component.get("focus_areas_detail") or []:
        domain = _text(row.get("domain")) or "unknown"
        next_action = _text(row.get("next_action")) or "Review domain blockers."
        primary_gaps = ", ".join(row.get("primary_gaps") or [])
        if primary_gaps:
            items.append(f"{domain}: {next_action} ({primary_gaps})")
        else:
            items.append(f"{domain}: {next_action}")
    if status == "knowledge_domain_control_plane_blocked":
        items.append(
            "Treat blocked domain rows as release blockers until control-plane status is "
            "at least partial across standards, tolerance, and GD&T."
        )
    return _compact(items, limit=12)


def render_knowledge_domain_control_plane_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    component = payload.get("knowledge_domain_control_plane") or {}
    lines = [
        f"# {title}",
        "",
        "## Overview",
        "",
        f"- `status`: `{component.get('status', 'knowledge_domain_control_plane_missing')}`",
        f"- `ready_domain_count`: `{component.get('ready_domain_count', 0)}`",
        f"- `partial_domain_count`: `{component.get('partial_domain_count', 0)}`",
        f"- `blocked_domain_count`: `{component.get('blocked_domain_count', 0)}`",
        f"- `missing_domain_count`: `{component.get('missing_domain_count', 0)}`",
        f"- `release_blockers`: "
        f"`{', '.join(component.get('release_blockers') or []) or 'none'}`",
        "",
        "## Domains",
        "",
    ]
    for row in (component.get("domains") or {}).values():
        lines.extend(
            [
                f"### {row.get('label') or row.get('domain')}",
                "",
                f"- `status`: `{row.get('status', 'unknown')}`",
                f"- `capability_status`: `{row.get('capability_status', 'unknown')}`",
                f"- `realdata_status`: `{row.get('realdata_status', 'unknown')}`",
                f"- `outcome_status`: `{row.get('outcome_status', 'unknown')}`",
                f"- `drift_status`: `{row.get('drift_status', 'stable')}`",
                f"- `release_blocker`: `{row.get('release_blocker', False)}`",
                f"- `primary_gaps`: `{', '.join(row.get('primary_gaps') or []) or 'none'}`",
                f"- `next_action`: `{row.get('next_action') or 'none'}`",
                "",
            ]
        )
    recommendations = payload.get("recommendations") or []
    if recommendations:
        lines.extend(["## Recommendations", ""])
        for item in recommendations:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
