"""Drift helpers for benchmark knowledge domain control plane."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


STATUS_RANK = {
    "ready": 3,
    "partial": 2,
    "blocked": 1,
    "missing": 0,
    "knowledge_domain_control_plane_ready": 3,
    "knowledge_domain_control_plane_partial": 2,
    "knowledge_domain_control_plane_blocked": 1,
    "knowledge_domain_control_plane_missing": 0,
}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _status(value: Any) -> str:
    return _text(value) or "missing"


def _rank(value: Any) -> int:
    return STATUS_RANK.get(_status(value), 0)


def _compact(items: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for item in items:
        text = _text(item)
        if text and text not in out:
            out.append(text)
    return out


def _component(summary: Dict[str, Any]) -> Dict[str, Any]:
    return summary.get("knowledge_domain_control_plane") or summary or {}


def _domains(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = _component(summary).get("domains") or {}
    return rows if isinstance(rows, dict) else {}


def _release_blockers(summary: Dict[str, Any]) -> List[str]:
    return _compact(_component(summary).get("release_blockers") or [])


def build_knowledge_domain_control_plane_drift_status(
    current_summary: Dict[str, Any],
    previous_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare current and previous knowledge domain control-plane summaries."""
    current_component = _component(current_summary)
    previous_component = _component(previous_summary)

    if not previous_component:
        return {
            "status": "baseline_missing",
            "current_status": _status(current_component.get("status")),
            "previous_status": "baseline_missing",
            "ready_domain_delta": int(current_component.get("ready_domain_count") or 0),
            "partial_domain_delta": int(
                current_component.get("partial_domain_count") or 0
            ),
            "blocked_domain_delta": int(
                current_component.get("blocked_domain_count") or 0
            ),
            "missing_domain_delta": int(
                current_component.get("missing_domain_count") or 0
            ),
            "total_action_delta": int(current_component.get("total_action_count") or 0),
            "high_priority_action_delta": int(
                current_component.get("high_priority_action_count") or 0
            ),
            "regressions": [],
            "improvements": [],
            "resolved_release_blockers": [],
            "new_release_blockers": _release_blockers(current_summary),
            "domain_regressions": [],
            "domain_improvements": [],
            "domain_changes": [],
        }

    current_domains = _domains(current_summary)
    previous_domains = _domains(previous_summary)
    domain_names = sorted(set(current_domains) | set(previous_domains))
    domain_changes: List[Dict[str, Any]] = []
    domain_regressions: List[str] = []
    domain_improvements: List[str] = []

    for name in domain_names:
        current_row = current_domains.get(name) or {}
        previous_row = previous_domains.get(name) or {}
        current_status = _status(current_row.get("status"))
        previous_status = _status(previous_row.get("status"))
        current_rank = _rank(current_status)
        previous_rank = _rank(previous_status)
        if current_rank > previous_rank:
            trend = "improved"
            domain_improvements.append(name)
        elif current_rank < previous_rank:
            trend = "regressed"
            domain_regressions.append(name)
        else:
            trend = "stable"
        domain_changes.append(
            {
                "domain": name,
                "label": _text(current_row.get("label"))
                or _text(previous_row.get("label"))
                or name,
                "previous_status": previous_status,
                "current_status": current_status,
                "trend": trend,
                "previous_priority": _text(previous_row.get("priority")) or "unknown",
                "current_priority": _text(current_row.get("priority")) or "unknown",
                "previous_release_blocker": bool(previous_row.get("release_blocker")),
                "current_release_blocker": bool(current_row.get("release_blocker")),
                "previous_primary_gaps": _compact(
                    previous_row.get("primary_gaps") or []
                ),
                "current_primary_gaps": _compact(current_row.get("primary_gaps") or []),
                "previous_next_action": _text(previous_row.get("next_action")) or "none",
                "current_next_action": _text(current_row.get("next_action")) or "none",
            }
        )

    regressions = _compact(
        [row["domain"] for row in domain_changes if row.get("trend") == "regressed"]
    )
    improvements = _compact(
        [row["domain"] for row in domain_changes if row.get("trend") == "improved"]
    )
    if regressions and improvements:
        status = "mixed"
    elif regressions:
        status = "regressed"
    elif improvements:
        status = "improved"
    else:
        status = "stable"

    previous_blockers = set(_release_blockers(previous_summary))
    current_blockers = set(_release_blockers(current_summary))

    return {
        "status": status,
        "current_status": _status(current_component.get("status")),
        "previous_status": _status(previous_component.get("status")),
        "ready_domain_delta": int(current_component.get("ready_domain_count") or 0)
        - int(previous_component.get("ready_domain_count") or 0),
        "partial_domain_delta": int(current_component.get("partial_domain_count") or 0)
        - int(previous_component.get("partial_domain_count") or 0),
        "blocked_domain_delta": int(current_component.get("blocked_domain_count") or 0)
        - int(previous_component.get("blocked_domain_count") or 0),
        "missing_domain_delta": int(current_component.get("missing_domain_count") or 0)
        - int(previous_component.get("missing_domain_count") or 0),
        "total_action_delta": int(current_component.get("total_action_count") or 0)
        - int(previous_component.get("total_action_count") or 0),
        "high_priority_action_delta": int(
            current_component.get("high_priority_action_count") or 0
        )
        - int(previous_component.get("high_priority_action_count") or 0),
        "regressions": regressions,
        "improvements": improvements,
        "resolved_release_blockers": sorted(previous_blockers - current_blockers),
        "new_release_blockers": sorted(current_blockers - previous_blockers),
        "domain_regressions": domain_regressions,
        "domain_improvements": domain_improvements,
        "domain_changes": domain_changes,
    }


def knowledge_domain_control_plane_drift_recommendations(
    summary: Dict[str, Any],
) -> List[str]:
    status = _status(summary.get("status"))
    if status == "baseline_missing":
        items = [
            "Persist the current knowledge domain control-plane as the next "
            "benchmark baseline."
        ]
        blockers = _compact(summary.get("new_release_blockers") or [])
        if blockers:
            items.append("Initial release blockers: " + ", ".join(blockers))
        return items
    if status == "regressed":
        items = [
            "Restore regressed knowledge-domain control-plane rows before claiming "
            "release alignment."
        ]
        regressed = _compact(summary.get("domain_regressions") or [])
        if regressed:
            items.append("Regressed domains: " + ", ".join(regressed))
        blockers = _compact(summary.get("new_release_blockers") or [])
        if blockers:
            items.append("New release blockers: " + ", ".join(blockers))
        return items
    if status == "improved":
        items = [
            "Carry the improved knowledge-domain control-plane forward as the new "
            "release benchmark baseline."
        ]
        improved = _compact(summary.get("domain_improvements") or [])
        if improved:
            items.append("Improved domains: " + ", ".join(improved))
        resolved = _compact(summary.get("resolved_release_blockers") or [])
        if resolved:
            items.append("Resolved release blockers: " + ", ".join(resolved))
        return items
    if status == "mixed":
        return [
            "Review mixed knowledge-domain control-plane movement before release.",
            "Regressions: " + ", ".join(_compact(summary.get("domain_regressions") or []))
            if _compact(summary.get("domain_regressions") or [])
            else "Regressions: none",
            "Improvements: "
            + ", ".join(_compact(summary.get("domain_improvements") or []))
            if _compact(summary.get("domain_improvements") or [])
            else "Improvements: none",
        ]
    return [
        "Knowledge-domain control-plane is stable; keep monitoring release blockers "
        "and action count deltas."
    ]


def render_knowledge_domain_control_plane_drift_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    component = payload.get("knowledge_domain_control_plane_drift") or {}
    lines = [
        f"# {title}",
        "",
        "## Overview",
        "",
        f"- `status`: `{component.get('status', 'baseline_missing')}`",
        f"- `current_status`: `{component.get('current_status', 'unknown')}`",
        f"- `previous_status`: `{component.get('previous_status', 'unknown')}`",
        f"- `ready_domain_delta`: `{component.get('ready_domain_delta', 0)}`",
        f"- `partial_domain_delta`: `{component.get('partial_domain_delta', 0)}`",
        f"- `blocked_domain_delta`: `{component.get('blocked_domain_delta', 0)}`",
        f"- `missing_domain_delta`: `{component.get('missing_domain_delta', 0)}`",
        f"- `total_action_delta`: `{component.get('total_action_delta', 0)}`",
        f"- `high_priority_action_delta`: "
        f"`{component.get('high_priority_action_delta', 0)}`",
        f"- `new_release_blockers`: "
        f"`{', '.join(component.get('new_release_blockers') or []) or 'none'}`",
        f"- `resolved_release_blockers`: "
        f"`{', '.join(component.get('resolved_release_blockers') or []) or 'none'}`",
        "",
        "## Domain Changes",
        "",
    ]
    for row in component.get("domain_changes") or []:
        lines.extend(
            [
                f"### {row.get('label') or row.get('domain')}",
                "",
                f"- `trend`: `{row.get('trend', 'stable')}`",
                f"- `status`: `{row.get('previous_status', 'unknown')}` -> "
                f"`{row.get('current_status', 'unknown')}`",
                f"- `release_blocker`: `{row.get('previous_release_blocker', False)}` "
                f"-> `{row.get('current_release_blocker', False)}`",
                f"- `priority`: `{row.get('previous_priority', 'unknown')}` -> "
                f"`{row.get('current_priority', 'unknown')}`",
                f"- `primary_gaps`: "
                f"`{', '.join(row.get('previous_primary_gaps') or []) or 'none'}` -> "
                f"`{', '.join(row.get('current_primary_gaps') or []) or 'none'}`",
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
