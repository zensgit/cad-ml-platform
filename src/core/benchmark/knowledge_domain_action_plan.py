"""Turn knowledge-domain benchmark gaps into an executable action plan."""

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
    if not status:
        return "missing"
    return status


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


def _foundation_item(domain: str, label: str, row: Dict[str, Any]) -> Dict[str, Any] | None:
    readiness_status = _normalize_status(row.get("readiness_status"))
    if readiness_status == "ready":
        return None
    missing_metrics = list(row.get("missing_metrics") or [])
    focus_components = list(row.get("focus_components") or [])
    reason = (
        f"Foundation status is {readiness_status}; "
        f"missing_metrics={_join(missing_metrics)}; "
        f"focus_components={_join(focus_components)}"
    )
    return {
        "id": f"{domain}:foundation",
        "domain": domain,
        "label": label,
        "stage": "foundation",
        "status": _item_status(readiness_status),
        "priority": _priority(readiness_status),
        "reason": reason,
        "action": (
            f"Backfill {label} foundation metrics and coverage before promoting "
            "this domain as benchmark-ready."
        ),
        "missing_metrics": missing_metrics,
        "focus_components": focus_components,
    }


def _application_item(
    domain: str,
    label: str,
    row: Dict[str, Any],
) -> Dict[str, Any] | None:
    application_status = _normalize_status(row.get("application_status"))
    if application_status == "ready":
        return None
    signal_count = int(row.get("application_signal_count") or 0)
    reason = (
        f"Application status is {application_status}; "
        f"signal_count={signal_count}"
    )
    return {
        "id": f"{domain}:application",
        "domain": domain,
        "label": label,
        "stage": "application",
        "status": _item_status(application_status),
        "priority": _priority(application_status),
        "reason": reason,
        "action": (
            f"Promote {label} application evidence into benchmark surfaces and "
            "raise usable signal count."
        ),
        "signal_count": signal_count,
    }


def _realdata_item(domain: str, label: str, row: Dict[str, Any]) -> Dict[str, Any] | None:
    realdata_status = _normalize_status(row.get("realdata_status"))
    if realdata_status == "ready":
        return None
    blocked_components = list(row.get("blocked_realdata_components") or [])
    missing_components = list(row.get("missing_realdata_components") or [])
    partial_components = list(row.get("partial_realdata_components") or [])
    components = blocked_components or missing_components or partial_components
    reason = (
        f"Real-data status is {realdata_status}; "
        f"components={_join(components)}"
    )
    return {
        "id": f"{domain}:realdata",
        "domain": domain,
        "label": label,
        "stage": "realdata",
        "status": _item_status(realdata_status),
        "priority": _priority(realdata_status),
        "reason": reason,
        "action": (
            f"Expand {label} real-data validation across DXF, history, and STEP/B-Rep "
            "surfaces until benchmark evidence is stable."
        ),
        "components": components,
        "blocked_components": blocked_components,
        "missing_components": missing_components,
        "partial_components": partial_components,
    }


def build_knowledge_domain_action_plan(
    knowledge_domain_matrix_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Build an execution plan from benchmark knowledge-domain matrix gaps."""
    component = (
        knowledge_domain_matrix_summary.get("knowledge_domain_matrix")
        or knowledge_domain_matrix_summary
        or {}
    )
    domains = component.get("domains") or {}
    actions: List[Dict[str, Any]] = []
    ready_domains: List[str] = []
    blocked_domains: List[str] = []
    partial_domains: List[str] = []
    domain_action_counts: Dict[str, int] = {}

    for domain, row in domains.items():
        label = _text(row.get("label")) or domain
        items = [
            _foundation_item(domain, label, row),
            _application_item(domain, label, row),
            _realdata_item(domain, label, row),
        ]
        present = [item for item in items if item]
        if not present:
            ready_domains.append(domain)
            domain_action_counts[domain] = 0
            continue
        if _normalize_status(row.get("status")) in {"blocked", "missing"}:
            blocked_domains.append(domain)
        else:
            partial_domains.append(domain)
        actions.extend(present)
        domain_action_counts[domain] = len(present)

    actions.sort(
        key=lambda item: (
            {"high": 0, "medium": 1, "low": 2}.get(item.get("priority") or "", 3),
            item.get("domain") or "",
            item.get("stage") or "",
        )
    )
    high_priority_actions = [
        item["id"] for item in actions if item.get("priority") == "high"
    ]
    medium_priority_actions = [
        item["id"] for item in actions if item.get("priority") == "medium"
    ]
    if not actions:
        status = "knowledge_domain_action_plan_ready"
    elif high_priority_actions:
        status = "knowledge_domain_action_plan_blocked"
    else:
        status = "knowledge_domain_action_plan_partial"
    return {
        "status": status,
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


def knowledge_domain_action_plan_recommendations(component: Dict[str, Any]) -> List[str]:
    status = _normalize_status(component.get("status"))
    if status == "knowledge_domain_action_plan_ready":
        return [
            "Knowledge-domain action plan is clear: no open foundation, application, "
            "or real-data actions remain."
        ]
    recommendations: List[str] = []
    for item in component.get("actions") or []:
        if item.get("priority") not in {"high", "medium"}:
            continue
        recommendations.append(
            f"{item.get('stage')}: {item.get('action')} ({item.get('reason')})"
        )
    if status == "knowledge_domain_action_plan_blocked":
        recommendations.append(
            "Treat high-priority knowledge-domain actions as release blockers until "
            "their missing metrics and real-data gaps are resolved."
        )
    return _compact(recommendations, limit=12)


def render_knowledge_domain_action_plan_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    component = payload.get("knowledge_domain_action_plan") or {}
    lines = [
        f"# {title}",
        "",
        "## Overview",
        "",
        f"- `status`: `{component.get('status', 'knowledge_domain_action_plan_blocked')}`",
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
                f"stage=`{item.get('stage')}` "
                f"priority=`{item.get('priority')}` "
                f"status=`{item.get('status')}`"
            )
            lines.append(f"  reason: {item.get('reason') or 'none'}")
            lines.append(f"  action: {item.get('action') or 'none'}")
    else:
        lines.append("- none")
    lines.extend(["", "## Recommendations", ""])
    recommendations = payload.get("recommendations") or []
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)
