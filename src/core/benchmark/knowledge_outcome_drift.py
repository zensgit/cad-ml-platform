"""Drift helpers for benchmark knowledge outcome correlation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


STATUS_RANK = {
    "ready": 3,
    "partial": 2,
    "blocked": 1,
    "missing": 0,
    "knowledge_outcome_correlation_ready": 3,
    "knowledge_outcome_correlation_partial": 2,
    "knowledge_outcome_correlation_missing": 0,
}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _status(value: Any) -> str:
    return _text(value) or "missing"


def _rank(value: Any) -> int:
    return STATUS_RANK.get(_status(value), 0)


def _compact(items: Iterable[Any]) -> List[str]:
    return [_text(item) for item in items if _text(item)]


def _component(summary: Dict[str, Any]) -> Dict[str, Any]:
    return summary.get("knowledge_outcome_correlation") or summary or {}


def _domains(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    domains = _component(summary).get("domains") or {}
    return domains if isinstance(domains, dict) else {}


def _focus_areas(summary: Dict[str, Any]) -> List[str]:
    return _compact(_component(summary).get("focus_areas") or [])


def _priority_domains(summary: Dict[str, Any]) -> List[str]:
    return _compact(_component(summary).get("priority_domains") or [])


def build_knowledge_outcome_drift_status(
    current_summary: Dict[str, Any],
    previous_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare current and previous knowledge outcome correlation summaries."""
    current_component = _component(current_summary)
    previous_component = _component(previous_summary)

    if not previous_component:
        return {
            "status": "baseline_missing",
            "current_status": _status(current_component.get("status")),
            "previous_status": "baseline_missing",
            "ready_domain_delta": int(current_component.get("ready_domain_count") or 0),
            "blocked_domain_delta": int(
                current_component.get("blocked_domain_count") or 0
            ),
            "regressions": [],
            "improvements": [],
            "resolved_focus_areas": [],
            "new_focus_areas": _focus_areas(current_summary),
            "resolved_priority_domains": [],
            "new_priority_domains": _priority_domains(current_summary),
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
                "previous_best_surface": _text(previous_row.get("best_surface")) or "none",
                "current_best_surface": _text(current_row.get("best_surface")) or "none",
                "previous_best_surface_score": round(
                    _to_float(previous_row.get("best_surface_score")), 6
                ),
                "current_best_surface_score": round(
                    _to_float(current_row.get("best_surface_score")), 6
                ),
                "best_surface_score_delta": round(
                    _to_float(current_row.get("best_surface_score"))
                    - _to_float(previous_row.get("best_surface_score")),
                    6,
                ),
            }
        )

    previous_focus = set(_focus_areas(previous_summary))
    current_focus = set(_focus_areas(current_summary))
    previous_priority = set(_priority_domains(previous_summary))
    current_priority = set(_priority_domains(current_summary))

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
        "regressions": regressions,
        "improvements": improvements,
        "resolved_focus_areas": sorted(previous_focus - current_focus),
        "new_focus_areas": sorted(current_focus - previous_focus),
        "resolved_priority_domains": sorted(previous_priority - current_priority),
        "new_priority_domains": sorted(current_priority - previous_priority),
        "domain_regressions": domain_regressions,
        "domain_improvements": domain_improvements,
        "domain_changes": domain_changes,
    }


def knowledge_outcome_drift_recommendations(summary: Dict[str, Any]) -> List[str]:
    status = _status(summary.get("status"))
    if status == "baseline_missing":
        items = [
            "Persist the current knowledge outcome correlation as the next benchmark baseline."
        ]
        new_priority = _compact(summary.get("new_priority_domains") or [])
        if new_priority:
            items.append("Initial priority domains: " + ", ".join(new_priority))
        return items
    if status == "regressed":
        items = [
            "Resolve knowledge outcome regressions before claiming benchmark outcome stability."
        ]
        domain_regressions = _compact(summary.get("domain_regressions") or [])
        if domain_regressions:
            items.append("Regressed domains: " + ", ".join(domain_regressions))
        return items
    if status == "improved":
        items = [
            "Promote the improved knowledge outcome correlation after CI surfaces refresh."
        ]
        resolved = _compact(summary.get("resolved_priority_domains") or [])
        improved = _compact(summary.get("domain_improvements") or [])
        if resolved:
            items.append("Resolved priority domains: " + ", ".join(resolved))
        if improved:
            items.append("Improved domains: " + ", ".join(improved))
        return items
    if status == "mixed":
        return [
            "Keep the previous knowledge outcome baseline until regressions are cleared.",
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
        "Knowledge outcome correlation is stable against the previous benchmark baseline."
    ]


def render_knowledge_outcome_drift_markdown(
    payload: Dict[str, Any], title: str
) -> str:
    component = payload.get("knowledge_outcome_drift") or {}
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
        "",
        "## Changes",
        "",
        f"- `regressions`: `{', '.join(component.get('regressions') or []) or 'none'}`",
        f"- `improvements`: `{', '.join(component.get('improvements') or []) or 'none'}`",
        f"- `domain_regressions`: "
        f"`{', '.join(component.get('domain_regressions') or []) or 'none'}`",
        f"- `domain_improvements`: "
        f"`{', '.join(component.get('domain_improvements') or []) or 'none'}`",
        f"- `resolved_focus_areas`: "
        f"`{', '.join(component.get('resolved_focus_areas') or []) or 'none'}`",
        f"- `new_focus_areas`: `{', '.join(component.get('new_focus_areas') or []) or 'none'}`",
        f"- `resolved_priority_domains`: "
        f"`{', '.join(component.get('resolved_priority_domains') or []) or 'none'}`",
        f"- `new_priority_domains`: "
        f"`{', '.join(component.get('new_priority_domains') or []) or 'none'}`",
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
            f"surface=`{row.get('previous_best_surface')}` -> "
            f"`{row.get('current_best_surface')}` "
            f"score_delta=`{row.get('best_surface_score_delta')}`"
        )
    lines.extend(["", "## Recommendations", ""])
    recommendations = payload.get("recommendations") or []
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)
