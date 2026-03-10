"""Release-surface alignment for the knowledge-domain control plane."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


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


def _domain_statuses(payload: Dict[str, Any]) -> Dict[str, str]:
    rows = payload.get("knowledge_domain_control_plane_domains") or {}
    if not isinstance(rows, dict):
        return {}
    out: Dict[str, str] = {}
    for name, row in rows.items():
        if not isinstance(row, dict):
            continue
        out[str(name)] = _text(row.get("status")) or "unknown"
    return out


def _release_surface(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "control_plane_status": (
            _text(payload.get("knowledge_domain_control_plane_status")) or "unknown"
        ),
        "control_plane_drift_status": (
            _text(payload.get("knowledge_domain_control_plane_drift_status"))
            or "unknown"
        ),
        "release_blockers": sorted(
            _compact(payload.get("knowledge_domain_control_plane_release_blockers") or [])
        ),
        "domain_statuses": _domain_statuses(payload),
    }


def build_knowledge_domain_release_surface_alignment(
    *,
    benchmark_release_decision: Dict[str, Any],
    benchmark_release_runbook: Dict[str, Any],
) -> Dict[str, Any]:
    release_decision = _release_surface(benchmark_release_decision)
    release_runbook = _release_surface(benchmark_release_runbook)

    domain_names = sorted(
        set(release_decision.get("domain_statuses") or {})
        | set(release_runbook.get("domain_statuses") or {})
    )
    mismatches: List[str] = []
    domain_mismatches: List[str] = []
    blocker_mismatches: List[str] = []

    for key, label in (
        ("control_plane_status", "control_plane_status"),
        ("control_plane_drift_status", "control_plane_drift_status"),
    ):
        left = release_decision.get(key) or "unknown"
        right = release_runbook.get(key) or "unknown"
        if left != right:
            mismatches.append(f"{label}:{left}->{right}")

    release_decision_blockers = release_decision.get("release_blockers") or []
    release_runbook_blockers = release_runbook.get("release_blockers") or []
    if release_decision_blockers != release_runbook_blockers:
        blocker_mismatches.append(
            "release_blockers:"
            + ",".join(release_decision_blockers or ["none"])
            + "->"
            + ",".join(release_runbook_blockers or ["none"])
        )
        mismatches.extend(blocker_mismatches)

    for domain in domain_names:
        left = (release_decision.get("domain_statuses") or {}).get(domain) or "unknown"
        right = (release_runbook.get("domain_statuses") or {}).get(domain) or "unknown"
        if left != right:
            domain_mismatches.append(f"{domain}:{left}->{right}")
    mismatches.extend(domain_mismatches)

    known = any(
        item != "unknown"
        for item in (
            release_decision.get("control_plane_status"),
            release_decision.get("control_plane_drift_status"),
            release_runbook.get("control_plane_status"),
            release_runbook.get("control_plane_drift_status"),
        )
    ) or bool(domain_names) or bool(release_decision_blockers) or bool(release_runbook_blockers)

    if not known:
        status = "unavailable"
        summary = "knowledge-domain release surface alignment unavailable"
    elif mismatches:
        status = "diverged"
        summary = "; ".join(_compact(mismatches, limit=5))
    else:
        status = "aligned"
        summary = (
            "release_decision and release_runbook agree on knowledge-domain control-plane "
            "status, drift, blockers, and per-domain states"
        )

    return {
        "status": status,
        "summary": summary,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "domain_mismatches": domain_mismatches,
        "release_blocker_mismatches": blocker_mismatches,
        "release_decision": release_decision,
        "release_runbook": release_runbook,
    }


def knowledge_domain_release_surface_alignment_recommendations(
    component: Dict[str, Any],
) -> List[str]:
    status = _text(component.get("status")) or "unavailable"
    if status == "aligned":
        return [
            "Keep knowledge-domain control-plane fields aligned between release decision "
            "and release runbook."
        ]
    if status == "diverged":
        recommendations = [
            "Reconcile knowledge-domain release-surface mismatches before claiming "
            "standards/tolerance/GD&T release readiness."
        ]
        domain_mismatches = _compact(component.get("domain_mismatches") or [], limit=4)
        if domain_mismatches:
            recommendations.append("Domain mismatches: " + ", ".join(domain_mismatches))
        blocker_mismatches = _compact(
            component.get("release_blocker_mismatches") or [], limit=2
        )
        if blocker_mismatches:
            recommendations.append(
                "Release-blocker mismatches: " + ", ".join(blocker_mismatches)
            )
        return recommendations
    return [
        "Emit knowledge-domain control-plane status and drift into both release decision "
        "and release runbook before benchmarking release alignment."
    ]


def render_knowledge_domain_release_surface_alignment_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    component = payload.get("knowledge_domain_release_surface_alignment") or {}
    lines = [
        f"# {title}",
        "",
        "## Overview",
        "",
        f"- `status`: `{component.get('status', 'unavailable')}`",
        f"- `summary`: {component.get('summary') or 'none'}",
        f"- `mismatch_count`: `{component.get('mismatch_count', 0)}`",
        "- `domain_mismatches`: `"
        + (", ".join(component.get("domain_mismatches") or []) or "none")
        + "`",
        "- `release_blocker_mismatches`: `"
        + (", ".join(component.get("release_blocker_mismatches") or []) or "none")
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
