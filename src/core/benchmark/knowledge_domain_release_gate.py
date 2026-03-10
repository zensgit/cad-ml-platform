"""Release-gate view over knowledge-domain benchmark control-plane signals."""

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


def _release_gate_domain_statuses(
    control_plane_component: Dict[str, Any],
    capability_component: Dict[str, Any],
) -> Dict[str, str]:
    domain_names = sorted(
        set(control_plane_component.get("domains") or {})
        | set(capability_component.get("domains") or {})
    )
    out: Dict[str, str] = {}
    for name in domain_names:
        control_row = (control_plane_component.get("domains") or {}).get(name) or {}
        capability_row = (capability_component.get("domains") or {}).get(name) or {}
        out[name] = (
            _text(control_row.get("status"))
            or _text(capability_row.get("status"))
            or "unknown"
        )
    return out


def build_knowledge_domain_release_gate(
    *,
    benchmark_knowledge_domain_capability_matrix: Dict[str, Any],
    benchmark_knowledge_domain_capability_drift: Dict[str, Any],
    benchmark_knowledge_domain_action_plan: Dict[str, Any],
    benchmark_knowledge_domain_control_plane: Dict[str, Any],
    benchmark_knowledge_domain_control_plane_drift: Dict[str, Any],
    benchmark_knowledge_domain_release_surface_alignment: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a release-gate decision over knowledge benchmark surfaces."""
    capability_component = _component(
        benchmark_knowledge_domain_capability_matrix,
        "knowledge_domain_capability_matrix",
    )
    capability_drift_component = _component(
        benchmark_knowledge_domain_capability_drift,
        "knowledge_domain_capability_drift",
    )
    action_plan_component = _component(
        benchmark_knowledge_domain_action_plan,
        "knowledge_domain_action_plan",
    )
    control_plane_component = _component(
        benchmark_knowledge_domain_control_plane,
        "knowledge_domain_control_plane",
    )
    control_plane_drift_component = _component(
        benchmark_knowledge_domain_control_plane_drift,
        "knowledge_domain_control_plane_drift",
    )
    alignment_component = _component(
        benchmark_knowledge_domain_release_surface_alignment,
        "knowledge_domain_release_surface_alignment",
    )

    domain_statuses = _release_gate_domain_statuses(
        control_plane_component,
        capability_component,
    )
    blocked_domains = _compact(
        list(control_plane_component.get("release_blockers") or [])
        + list(capability_component.get("priority_domains") or [])
        + list(capability_drift_component.get("new_priority_domains") or [])
    )
    releasable_domains = sorted(
        name
        for name, status in domain_statuses.items()
        if status == "ready" and name not in blocked_domains
    )
    partial_domains = sorted(
        name for name, status in domain_statuses.items() if status == "partial"
    )
    recommended_first_action = (
        (action_plan_component.get("recommended_first_actions") or [{}])[0] or {}
    )

    capability_status = _text(capability_component.get("status")) or "unknown"
    capability_drift_status = _text(capability_drift_component.get("status")) or "unknown"
    control_plane_status = _text(control_plane_component.get("status")) or "unknown"
    control_plane_drift_status = (
        _text(control_plane_drift_component.get("status")) or "unknown"
    )
    alignment_status = _text(alignment_component.get("status")) or "unknown"
    action_plan_status = _text(action_plan_component.get("status")) or "unknown"

    blocking_reasons = _compact(
        (
            [f"control_plane:{control_plane_status}"]
            if control_plane_status not in {"ready", "knowledge_domain_control_plane_ready"}
            else []
        )
        + (
            [f"control_plane_drift:{control_plane_drift_status}"]
            if control_plane_drift_status in {"regressed", "mixed"}
            else []
        )
        + (
            [f"capability:{capability_status}"]
            if capability_status
            not in {"ready", "knowledge_domain_capability_ready"}
            else []
        )
        + (
            [f"capability_drift:{capability_drift_status}"]
            if capability_drift_status in {"regressed", "mixed"}
            else []
        )
        + (
            [f"release_surface_alignment:{alignment_status}"]
            if alignment_status not in {"aligned"}
            else []
        )
        + (
            [f"action_plan:{action_plan_status}"]
            if action_plan_status == "knowledge_domain_action_plan_blocked"
            else []
        )
        + ([f"blocked_domains:{','.join(blocked_domains)}"] if blocked_domains else [])
    )

    warning_reasons = _compact(
        (
            [f"control_plane_drift:{control_plane_drift_status}"]
            if control_plane_drift_status == "improved"
            else []
        )
        + (
            [f"capability_drift:{capability_drift_status}"]
            if capability_drift_status == "improved"
            else []
        )
        + (
            [f"action_plan:{action_plan_status}"]
            if action_plan_status == "knowledge_domain_action_plan_partial"
            else []
        )
        + ([f"partial_domains:{','.join(partial_domains)}"] if partial_domains else [])
    )

    known = any(
        value != "unknown"
        for value in (
            capability_status,
            capability_drift_status,
            control_plane_status,
            control_plane_drift_status,
            alignment_status,
            action_plan_status,
        )
    ) or bool(domain_statuses)

    if not known:
        status = "knowledge_domain_release_gate_unavailable"
        gate_open = False
        summary = "knowledge-domain release gate unavailable"
    elif blocking_reasons:
        status = "knowledge_domain_release_gate_blocked"
        gate_open = False
        summary = "; ".join(blocking_reasons[:4])
    elif warning_reasons:
        status = "knowledge_domain_release_gate_partial"
        gate_open = False
        summary = "; ".join(warning_reasons[:4])
    else:
        status = "knowledge_domain_release_gate_ready"
        gate_open = True
        summary = "knowledge-domain release gate open"

    return {
        "status": status,
        "summary": summary,
        "gate_open": gate_open,
        "blocking_reasons": blocking_reasons,
        "warning_reasons": warning_reasons,
        "releasable_domains": releasable_domains,
        "blocked_domains": blocked_domains,
        "partial_domains": partial_domains,
        "priority_domains": _compact(action_plan_component.get("priority_domains") or []),
        "domain_statuses": domain_statuses,
        "capability_status": capability_status,
        "capability_drift_status": capability_drift_status,
        "control_plane_status": control_plane_status,
        "control_plane_drift_status": control_plane_drift_status,
        "release_surface_alignment_status": alignment_status,
        "action_plan_status": action_plan_status,
        "recommended_first_action": recommended_first_action,
        "recommended_first_action_id": _text(recommended_first_action.get("id")) or "none",
        "high_priority_action_count": int(
            action_plan_component.get("high_priority_action_count") or 0
        ),
        "medium_priority_action_count": int(
            action_plan_component.get("medium_priority_action_count") or 0
        ),
    }


def knowledge_domain_release_gate_recommendations(
    component: Dict[str, Any],
) -> List[str]:
    status = _text(component.get("status")) or "knowledge_domain_release_gate_unavailable"
    if status == "knowledge_domain_release_gate_ready":
        return [
            "Knowledge-domain release gate is open; keep capability, control-plane, "
            "and release-surface alignment baselines synchronized."
        ]
    recommendations: List[str] = []
    if status == "knowledge_domain_release_gate_blocked":
        recommendations.append(
            "Do not promote standards/tolerance/GD&T release readiness until "
            "knowledge-domain release gate blockers are cleared."
        )
    elif status == "knowledge_domain_release_gate_partial":
        recommendations.append(
            "Clear remaining knowledge-domain release gate warnings before "
            "advertising stable release readiness."
        )
    else:
        recommendations.append(
            "Emit capability, control-plane, and release-surface artifacts before "
            "evaluating the knowledge-domain release gate."
        )
    if component.get("recommended_first_action"):
        action = component["recommended_first_action"]
        recommendations.append(
            "Recommended first action: "
            f"{_text(action.get('domain')) or 'unknown'} / "
            f"{_text(action.get('stage')) or 'none'} / "
            f"{_text(action.get('action')) or 'none'}"
        )
    blocking = _compact(component.get("blocking_reasons") or [], limit=4)
    if blocking:
        recommendations.append("Blocking reasons: " + ", ".join(blocking))
    warnings = _compact(component.get("warning_reasons") or [], limit=4)
    if warnings and status != "knowledge_domain_release_gate_blocked":
        recommendations.append("Warnings: " + ", ".join(warnings))
    return recommendations


def render_knowledge_domain_release_gate_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    component = payload.get("knowledge_domain_release_gate") or {}
    lines = [
        f"# {title}",
        "",
        "## Overview",
        "",
        f"- `status`: `{component.get('status', 'knowledge_domain_release_gate_unavailable')}`",
        f"- `summary`: {component.get('summary') or 'none'}",
        f"- `gate_open`: `{bool(component.get('gate_open'))}`",
        "- `releasable_domains`: `"
        + (", ".join(component.get("releasable_domains") or []) or "none")
        + "`",
        "- `blocked_domains`: `"
        + (", ".join(component.get("blocked_domains") or []) or "none")
        + "`",
        "- `partial_domains`: `"
        + (", ".join(component.get("partial_domains") or []) or "none")
        + "`",
        "- `blocking_reasons`: `"
        + (", ".join(component.get("blocking_reasons") or []) or "none")
        + "`",
        "- `warning_reasons`: `"
        + (", ".join(component.get("warning_reasons") or []) or "none")
        + "`",
        "",
        "## Recommendations",
        "",
    ]
    recommendations = payload.get("recommendations") or []
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"
