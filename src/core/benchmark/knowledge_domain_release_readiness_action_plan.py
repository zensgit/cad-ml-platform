"""Action-plan helpers for knowledge-domain release readiness."""

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


def _priority(*, blocked: bool = False, warning: bool = False) -> str:
    if blocked:
        return "high"
    if warning:
        return "medium"
    return "low"


def _item_status(priority: str) -> str:
    if priority == "high":
        return "blocked"
    if priority == "medium":
        return "required"
    return "ready"


def build_knowledge_domain_release_readiness_action_plan(
    *,
    benchmark_knowledge_domain_release_readiness_matrix: Dict[str, Any],
    benchmark_knowledge_domain_release_readiness_drift: Dict[str, Any],
    benchmark_knowledge_domain_release_gate: Dict[str, Any],
) -> Dict[str, Any]:
    """Build an execution plan from release-readiness matrix, drift, and gate."""
    matrix_component = _component(
        benchmark_knowledge_domain_release_readiness_matrix,
        "knowledge_domain_release_readiness_matrix",
    )
    drift_component = _component(
        benchmark_knowledge_domain_release_readiness_drift,
        "knowledge_domain_release_readiness_drift",
    )
    gate_component = _component(
        benchmark_knowledge_domain_release_gate,
        "knowledge_domain_release_gate",
    )

    matrix_domains = matrix_component.get("domains") or {}
    drift_changes = {
        _text(item.get("domain")): item
        for item in (drift_component.get("domain_changes") or [])
        if _text(item.get("domain"))
    }

    domain_names = sorted(
        set(matrix_domains)
        | set(drift_changes)
        | set(gate_component.get("blocked_domains") or [])
        | set(gate_component.get("partial_domains") or [])
        | set(gate_component.get("priority_domains") or [])
        | set(gate_component.get("releasable_domains") or [])
    )

    actions: List[Dict[str, Any]] = []
    ready_domains: List[str] = []
    partial_domains: List[str] = []
    blocked_domains: List[str] = []

    gate_blocked_domains = set(
        _compact(gate_component.get("blocked_domains") or [], limit=50)
    )
    gate_partial_domains = set(
        _compact(gate_component.get("partial_domains") or [], limit=50)
    )
    gate_priority_domains = set(
        _compact(gate_component.get("priority_domains") or [], limit=50)
    )

    for domain in domain_names:
        matrix_row = matrix_domains.get(domain) or {}
        drift_row = drift_changes.get(domain) or {}
        matrix_status = _text(matrix_row.get("status")) or "unknown"
        drift_trend = _text(drift_row.get("trend")) or "stable"
        alignment_warning = bool(matrix_row.get("alignment_warning"))

        domain_actions: List[Dict[str, Any]] = []

        if matrix_status in {"blocked", "partial", "unknown"}:
            priority = _priority(
                blocked=matrix_status == "blocked",
                warning=matrix_status in {"partial", "unknown"},
            )
            domain_actions.append(
                {
                    "id": f"{domain}:readiness",
                    "domain": domain,
                    "label": matrix_row.get("label") or domain,
                    "stage": "readiness",
                    "priority": priority,
                    "status": _item_status(priority),
                    "reason": (
                        f"matrix={matrix_status}; "
                        f"gate={_text(matrix_row.get('release_gate_status')) or 'unknown'}; "
                        f"validation={_text(matrix_row.get('validation_status')) or 'unknown'}; "
                        f"inventory={_text(matrix_row.get('inventory_status')) or 'unknown'}"
                    ),
                    "action": _text(matrix_row.get("action"))
                    or f"Unblock {domain} release-readiness matrix signals.",
                    "blocking_reasons": list(matrix_row.get("blocking_reasons") or []),
                    "warning_reasons": list(matrix_row.get("warning_reasons") or []),
                }
            )

        if drift_trend in {"regressed", "mixed"}:
            priority = _priority(blocked=True)
            domain_actions.append(
                {
                    "id": f"{domain}:drift",
                    "domain": domain,
                    "label": (
                        drift_row.get("label")
                        or matrix_row.get("label")
                        or domain
                    ),
                    "stage": "drift",
                    "priority": priority,
                    "status": _item_status(priority),
                    "reason": (
                        f"trend={drift_trend}; "
                        f"previous={_text(drift_row.get('previous_status')) or 'unknown'}; "
                        f"current={_text(drift_row.get('current_status')) or 'unknown'}"
                    ),
                    "action": (
                        f"Resolve {domain} release-readiness regressions before promotion."
                    ),
                    "previous_status": _text(drift_row.get("previous_status")) or "unknown",
                    "current_status": _text(drift_row.get("current_status")) or "unknown",
                    "previous_priority": _text(drift_row.get("previous_priority")) or "unknown",
                    "current_priority": _text(drift_row.get("current_priority")) or "unknown",
                }
            )

        if (
            domain in gate_blocked_domains
            or domain in gate_partial_domains
            or domain in gate_priority_domains
            or alignment_warning
        ):
            blocked = domain in gate_blocked_domains
            warning = (
                domain in gate_partial_domains
                or domain in gate_priority_domains
                or alignment_warning
            )
            priority = _priority(blocked=blocked, warning=warning and not blocked)
            reasons: List[str] = []
            if blocked:
                reasons.append("release_gate:blocked")
            if domain in gate_partial_domains:
                reasons.append("release_gate:partial")
            if domain in gate_priority_domains:
                reasons.append("release_gate:priority")
            if alignment_warning:
                reasons.append("release_surface_alignment:mismatch")
            domain_actions.append(
                {
                    "id": f"{domain}:gate",
                    "domain": domain,
                    "label": matrix_row.get("label") or drift_row.get("label") or domain,
                    "stage": "release_gate",
                    "priority": priority,
                    "status": _item_status(priority),
                    "reason": ", ".join(reasons) or "release_gate:review",
                    "action": f"Clear {domain} release-gate blockers and warnings.",
                    "gate_reasons": reasons,
                }
            )

        actions.extend(domain_actions)
        if any(item["priority"] == "high" for item in domain_actions):
            blocked_domains.append(domain)
        elif domain_actions:
            partial_domains.append(domain)
        else:
            ready_domains.append(domain)

    actions.sort(
        key=lambda item: (
            {"high": 0, "medium": 1, "low": 2}.get(item.get("priority") or "", 3),
            item.get("domain") or "",
            item.get("stage") or "",
        )
    )

    high_priority_action_count = sum(1 for item in actions if item["priority"] == "high")
    medium_priority_action_count = sum(
        1 for item in actions if item["priority"] == "medium"
    )

    if not actions:
        status = "knowledge_domain_release_readiness_action_plan_ready"
    elif high_priority_action_count:
        status = "knowledge_domain_release_readiness_action_plan_blocked"
    else:
        status = "knowledge_domain_release_readiness_action_plan_partial"

    priority_domains = blocked_domains or partial_domains
    return {
        "status": status,
        "total_action_count": len(actions),
        "high_priority_action_count": high_priority_action_count,
        "medium_priority_action_count": medium_priority_action_count,
        "ready_domains": ready_domains,
        "partial_domains": partial_domains,
        "blocked_domains": blocked_domains,
        "priority_domains": priority_domains,
        "gate_open": bool(gate_component.get("gate_open")),
        "recommended_first_actions": actions[:3],
        "actions": actions,
    }


def knowledge_domain_release_readiness_action_plan_recommendations(
    component: Dict[str, Any],
) -> List[str]:
    status = _text(component.get("status")) or "unknown"
    if status == "knowledge_domain_release_readiness_action_plan_ready":
        return [
            "Knowledge-domain release-readiness action plan is clear: "
            "no open matrix, drift, or gate actions remain."
        ]
    recommendations: List[str] = []
    for item in component.get("actions") or []:
        if item.get("priority") not in {"high", "medium"}:
            continue
        recommendations.append(
            f"{item.get('stage')}: {item.get('action')} ({item.get('reason') or 'none'})"
        )
    return _compact(recommendations, limit=12)


def render_knowledge_domain_release_readiness_action_plan_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    component = (
        payload.get("knowledge_domain_release_readiness_action_plan") or payload or {}
    )
    lines = [
        f"# {title}",
        "",
        "## Overview",
        "",
        f"- `status`: `{component.get('status') or 'unknown'}`",
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
                f"- `{item.get('id')}` domain=`{item.get('domain')}` stage=`{item.get('stage')}` "
                f"priority=`{item.get('priority')}` status=`{item.get('status')}`"
            )
            lines.append(f"  reason: {item.get('reason') or 'none'}")
            lines.append(f"  action: {item.get('action') or 'none'}")
    else:
        lines.append("- none")
    recommendations = payload.get("recommendations") or []
    if recommendations:
        lines.extend(["", "## Recommendations", ""])
        for item in recommendations:
            lines.append(f"- {item}")
    return "\n".join(lines).strip() + "\n"
