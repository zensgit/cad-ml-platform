"""Drift helpers for benchmark knowledge domain capability matrix."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


STATUS_RANK = {
    "ready": 3,
    "partial": 2,
    "blocked": 1,
    "missing": 0,
    "knowledge_domain_capability_ready": 3,
    "knowledge_domain_capability_partial": 2,
    "knowledge_domain_capability_missing": 0,
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
    return summary.get("knowledge_domain_capability_drift") or summary or {}


def _matrix(summary: Dict[str, Any]) -> Dict[str, Any]:
    return summary.get("knowledge_domain_capability_matrix") or summary or {}


def _domains(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = _matrix(summary).get("domains") or {}
    return rows if isinstance(rows, dict) else {}


def _focus_areas(summary: Dict[str, Any]) -> List[str]:
    root = _matrix(summary)
    detail = root.get("focus_areas_detail") or []
    names = []
    for row in detail:
        if not isinstance(row, dict):
            continue
        name = _text(row.get("domain"))
        if name and name not in names:
            names.append(name)
    return names or _compact(root.get("focus_areas") or [])


def _priority_domains(summary: Dict[str, Any]) -> List[str]:
    return _compact(_matrix(summary).get("priority_domains") or [])


def _provider_gap_domains(summary: Dict[str, Any]) -> List[str]:
    return _compact(_matrix(summary).get("provider_gap_domains") or [])


def _surface_gap_domains(summary: Dict[str, Any]) -> List[str]:
    return _compact(_matrix(summary).get("surface_gap_domains") or [])


def build_knowledge_domain_capability_drift_status(
    current_summary: Dict[str, Any],
    previous_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare current and previous knowledge domain capability summaries."""
    current_component = _matrix(current_summary)
    previous_component = _matrix(previous_summary)

    if not previous_component:
        return {
            "status": "baseline_missing",
            "current_status": _status(current_component.get("status")),
            "previous_status": "baseline_missing",
            "ready_domain_delta": int(current_component.get("ready_domain_count") or 0),
            "blocked_domain_delta": int(
                current_component.get("blocked_domain_count") or 0
            ),
            "provider_gap_delta": len(_provider_gap_domains(current_summary)),
            "surface_gap_delta": len(_surface_gap_domains(current_summary)),
            "regressions": [],
            "improvements": [],
            "resolved_focus_areas": [],
            "new_focus_areas": _focus_areas(current_summary),
            "resolved_priority_domains": [],
            "new_priority_domains": _priority_domains(current_summary),
            "resolved_provider_gap_domains": [],
            "new_provider_gap_domains": _provider_gap_domains(current_summary),
            "resolved_surface_gap_domains": [],
            "new_surface_gap_domains": _surface_gap_domains(current_summary),
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
                "previous_provider_status": _status(previous_row.get("provider_status")),
                "current_provider_status": _status(current_row.get("provider_status")),
                "previous_surface_status": _status(previous_row.get("surface_status")),
                "current_surface_status": _status(current_row.get("surface_status")),
                "previous_priority": _text(previous_row.get("priority")) or "unknown",
                "current_priority": _text(current_row.get("priority")) or "unknown",
                "previous_primary_gaps": _compact(
                    previous_row.get("primary_gaps") or []
                ),
                "current_primary_gaps": _compact(current_row.get("primary_gaps") or []),
            }
        )

    previous_focus = set(_focus_areas(previous_summary))
    current_focus = set(_focus_areas(current_summary))
    previous_priority = set(_priority_domains(previous_summary))
    current_priority = set(_priority_domains(current_summary))
    previous_provider_gaps = set(_provider_gap_domains(previous_summary))
    current_provider_gaps = set(_provider_gap_domains(current_summary))
    previous_surface_gaps = set(_surface_gap_domains(previous_summary))
    current_surface_gaps = set(_surface_gap_domains(current_summary))

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

    return {
        "status": status,
        "current_status": _status(current_component.get("status")),
        "previous_status": _status(previous_component.get("status")),
        "ready_domain_delta": int(current_component.get("ready_domain_count") or 0)
        - int(previous_component.get("ready_domain_count") or 0),
        "blocked_domain_delta": int(current_component.get("blocked_domain_count") or 0)
        - int(previous_component.get("blocked_domain_count") or 0),
        "provider_gap_delta": len(current_provider_gaps) - len(previous_provider_gaps),
        "surface_gap_delta": len(current_surface_gaps) - len(previous_surface_gaps),
        "regressions": regressions,
        "improvements": improvements,
        "resolved_focus_areas": sorted(previous_focus - current_focus),
        "new_focus_areas": sorted(current_focus - previous_focus),
        "resolved_priority_domains": sorted(previous_priority - current_priority),
        "new_priority_domains": sorted(current_priority - previous_priority),
        "resolved_provider_gap_domains": sorted(
            previous_provider_gaps - current_provider_gaps
        ),
        "new_provider_gap_domains": sorted(current_provider_gaps - previous_provider_gaps),
        "resolved_surface_gap_domains": sorted(
            previous_surface_gaps - current_surface_gaps
        ),
        "new_surface_gap_domains": sorted(current_surface_gaps - previous_surface_gaps),
        "domain_regressions": domain_regressions,
        "domain_improvements": domain_improvements,
        "domain_changes": domain_changes,
    }


def knowledge_domain_capability_drift_recommendations(
    summary: Dict[str, Any],
) -> List[str]:
    status = _status(summary.get("status"))
    if status == "baseline_missing":
        items = [
            "Persist the current knowledge domain capability matrix as the next benchmark baseline."
        ]
        new_priority = _compact(summary.get("new_priority_domains") or [])
        if new_priority:
            items.append("Initial priority domains: " + ", ".join(new_priority))
        return items
    if status == "regressed":
        items = [
            (
                "Restore regressed knowledge domain capabilities before claiming "
                "benchmark capability stability."
            )
        ]
        regressed = _compact(summary.get("domain_regressions") or [])
        if regressed:
            items.append("Regressed domains: " + ", ".join(regressed))
        new_provider = _compact(summary.get("new_provider_gap_domains") or [])
        if new_provider:
            items.append("New provider gaps: " + ", ".join(new_provider))
        new_surface = _compact(summary.get("new_surface_gap_domains") or [])
        if new_surface:
            items.append("New surface gaps: " + ", ".join(new_surface))
        return items
    if status == "improved":
        items = [
            "Promote the improved knowledge capability coverage after CI surfaces refresh."
        ]
        improved = _compact(summary.get("domain_improvements") or [])
        if improved:
            items.append("Improved domains: " + ", ".join(improved))
        resolved = _compact(summary.get("resolved_provider_gap_domains") or [])
        if resolved:
            items.append("Resolved provider gaps: " + ", ".join(resolved))
        return items
    if status == "mixed":
        return [
            "Keep the previous knowledge capability baseline until regressions are cleared.",
            "Regressed domains: "
            + (", ".join(_compact(summary.get("domain_regressions") or [])) or "none"),
            "Improved domains: "
            + (", ".join(_compact(summary.get("domain_improvements") or [])) or "none"),
        ]
    if summary.get("new_priority_domains"):
        return [
            "Watch new priority domains: "
            + ", ".join(_compact(summary.get("new_priority_domains") or []))
        ]
    return [
        "Knowledge domain capability matrix is stable against the previous benchmark baseline."
    ]


def render_knowledge_domain_capability_drift_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    component = payload.get("knowledge_domain_capability_drift") or {}
    lines = [
        f"# {title}",
        "",
        "## Status",
        "",
        f"- `status`: `{component.get('status', 'baseline_missing')}`",
        f"- `current_status`: `{component.get('current_status', 'unknown')}`",
        f"- `previous_status`: `{component.get('previous_status', 'baseline_missing')}`",
        f"- `ready_domain_delta`: `{component.get('ready_domain_delta', 0)}`",
        f"- `blocked_domain_delta`: `{component.get('blocked_domain_delta', 0)}`",
        f"- `provider_gap_delta`: `{component.get('provider_gap_delta', 0)}`",
        f"- `surface_gap_delta`: `{component.get('surface_gap_delta', 0)}`",
        "",
        "## Changes",
        "",
        f"- `regressions`: `{', '.join(component.get('regressions') or []) or 'none'}`",
        f"- `improvements`: `{', '.join(component.get('improvements') or []) or 'none'}`",
        f"- `resolved_focus_areas`: "
        f"`{', '.join(component.get('resolved_focus_areas') or []) or 'none'}`",
        f"- `new_focus_areas`: "
        f"`{', '.join(component.get('new_focus_areas') or []) or 'none'}`",
        f"- `resolved_provider_gap_domains`: "
        f"`{', '.join(component.get('resolved_provider_gap_domains') or []) or 'none'}`",
        f"- `new_provider_gap_domains`: "
        f"`{', '.join(component.get('new_provider_gap_domains') or []) or 'none'}`",
        f"- `resolved_surface_gap_domains`: "
        f"`{', '.join(component.get('resolved_surface_gap_domains') or []) or 'none'}`",
        f"- `new_surface_gap_domains`: "
        f"`{', '.join(component.get('new_surface_gap_domains') or []) or 'none'}`",
        "",
        "## Domain Changes",
        "",
    ]
    for row in component.get("domain_changes") or []:
        lines.append(
            "- "
            f"`{row.get('domain')}` "
            f"`{row.get('previous_status')}` -> `{row.get('current_status')}` "
            f"trend=`{row.get('trend')}` "
            f"provider=`{row.get('previous_provider_status')}` -> "
            f"`{row.get('current_provider_status')}` "
            f"surface=`{row.get('previous_surface_status')}` -> "
            f"`{row.get('current_surface_status')}`"
        )
    if not component.get("domain_changes"):
        lines.append("- none")
    lines.extend(["", "## Recommendations", ""])
    recommendations = payload.get("recommendations") or []
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)
