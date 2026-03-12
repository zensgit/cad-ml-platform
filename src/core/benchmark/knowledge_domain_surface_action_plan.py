"""Turn knowledge-domain surface gaps into an executable action plan."""

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


def _normalize_status(value: Any) -> str:
    status = _text(value).lower()
    return status or "missing"


def _priority(status: str) -> str:
    if status in {"missing", "blocked"}:
        return "high"
    if status == "partial":
        return "medium"
    return "low"


def _item_status(status: str) -> str:
    if status in {"missing", "blocked"}:
        return "blocked"
    if status == "partial":
        return "required"
    return "ready"


def _join(items: Iterable[Any]) -> str:
    values = _compact(items, limit=8)
    return ", ".join(values) if values else "none"


def build_knowledge_domain_surface_action_plan(
    knowledge_domain_surface_matrix_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Build an action plan from the public knowledge surface matrix."""
    component = (
        knowledge_domain_surface_matrix_summary.get("knowledge_domain_surface_matrix")
        or knowledge_domain_surface_matrix_summary
        or {}
    )
    domains = component.get("domains") or {}
    actions: List[Dict[str, Any]] = []
    ready_domains: List[str] = []
    partial_domains: List[str] = []
    blocked_domains: List[str] = []
    domain_action_counts: Dict[str, int] = {}
    total_subcapability_count = 0

    for domain, row in domains.items():
        label = _text(row.get("label")) or domain
        domain_rows = row.get("subcapabilities") or {}
        total_subcapability_count += len(domain_rows)
        domain_actions: List[Dict[str, Any]] = []
        for key, subcapability in domain_rows.items():
            status = _normalize_status(subcapability.get("status"))
            if status == "ready":
                continue
            missing_routes = list(subcapability.get("missing_routes") or [])
            action = {
                "id": f"{domain}:{key}",
                "domain": domain,
                "label": label,
                "subcapability": key,
                "subcapability_label": subcapability.get("label") or key,
                "priority": _priority(status),
                "status": _item_status(status),
                "reason": (
                    f"Surface status is {status}; "
                    f"missing_routes={_join(missing_routes)}; "
                    f"present_routes={subcapability.get('present_route_count', 0)}/"
                    f"{subcapability.get('expected_route_count', 0)}; "
                    f"reference_items={subcapability.get('reference_item_count', 0)}"
                ),
                "action": subcapability.get("action")
                or (
                    f"Expose {subcapability.get('label') or key} through stable public "
                    "knowledge routes and backfill reference depth."
                ),
                "missing_routes": missing_routes,
                "present_route_count": int(subcapability.get("present_route_count") or 0),
                "expected_route_count": int(subcapability.get("expected_route_count") or 0),
                "reference_item_count": int(subcapability.get("reference_item_count") or 0),
            }
            domain_actions.append(action)

        domain_action_counts[domain] = len(domain_actions)
        if not domain_actions:
            ready_domains.append(domain)
        elif _normalize_status(row.get("status")) in {"missing", "blocked"}:
            blocked_domains.append(domain)
        else:
            partial_domains.append(domain)
        actions.extend(domain_actions)

    actions.sort(
        key=lambda item: (
            {"high": 0, "medium": 1, "low": 2}.get(item.get("priority") or "", 3),
            item.get("domain") or "",
            item.get("subcapability") or "",
        )
    )
    high_priority_actions = [
        item["id"] for item in actions if item.get("priority") == "high"
    ]
    medium_priority_actions = [
        item["id"] for item in actions if item.get("priority") == "medium"
    ]
    if not actions:
        status = "knowledge_domain_surface_action_plan_ready"
    elif high_priority_actions:
        status = "knowledge_domain_surface_action_plan_blocked"
    else:
        status = "knowledge_domain_surface_action_plan_partial"
    return {
        "status": status,
        "total_subcapability_count": total_subcapability_count,
        "domain_action_counts": domain_action_counts,
        "ready_domains": ready_domains,
        "partial_domains": partial_domains,
        "blocked_domains": blocked_domains,
        "total_action_count": len(actions),
        "high_priority_action_count": len(high_priority_actions),
        "medium_priority_action_count": len(medium_priority_actions),
        "priority_domains": blocked_domains or partial_domains,
        "recommended_first_actions": actions[:3],
        "actions": actions,
    }


def knowledge_domain_surface_action_plan_recommendations(
    component: Dict[str, Any],
) -> List[str]:
    status = _normalize_status(component.get("status"))
    if status == "knowledge_domain_surface_action_plan_ready":
        return [
            "Knowledge-domain surface action plan is clear: no open public surface "
            "actions remain."
        ]
    recommendations: List[str] = []
    for item in component.get("actions") or []:
        if item.get("priority") not in {"high", "medium"}:
            continue
        recommendations.append(
            f"{item.get('subcapability_label')}: {item.get('action')} "
            f"({item.get('reason')})"
        )
    if status == "knowledge_domain_surface_action_plan_blocked":
        recommendations.append(
            "Treat high-priority surface action items as blockers before promoting "
            "knowledge domains through release surfaces."
        )
    return _compact(recommendations, limit=12)


def render_knowledge_domain_surface_action_plan_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    component = payload.get("knowledge_domain_surface_action_plan") or {}
    lines = [
        f"# {title}",
        "",
        "## Overview",
        "",
        "- "
        f"`status`: `{component.get('status', 'knowledge_domain_surface_action_plan_blocked')}`",
        f"- `total_action_count`: `{component.get('total_action_count', 0)}`",
        f"- `high_priority_action_count`: `{component.get('high_priority_action_count', 0)}`",
        f"- `medium_priority_action_count`: `{component.get('medium_priority_action_count', 0)}`",
        f"- `priority_domains`: `{', '.join(component.get('priority_domains') or []) or 'none'}`",
        "",
        "## Actions",
        "",
    ]
    actions = component.get("actions") or []
    if actions:
        for item in actions:
            lines.append(
                f"- `{item.get('id')}` "
                f"domain=`{item.get('domain')}` "
                f"subcapability=`{item.get('subcapability')}` "
                f"priority=`{item.get('priority')}` "
                f"status=`{item.get('status')}`"
            )
            lines.append(f"  reason: {item.get('reason') or 'none'}")
    else:
        lines.append("- none")
    lines.extend(["", "## Recommendations", ""])
    recommendations = payload.get("recommendations") or []
    if recommendations:
        for item in recommendations:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"
