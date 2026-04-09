"""Drift helpers for benchmark knowledge source coverage."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


STATUS_RANK = {
    "ready": 3,
    "partial": 2,
    "missing": 0,
    "knowledge_source_coverage_ready": 3,
    "knowledge_source_coverage_partial": 2,
    "knowledge_source_coverage_missing": 0,
}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _status(value: Any) -> str:
    return _text(value) or "missing"


def _rank(value: Any) -> int:
    return STATUS_RANK.get(_status(value), 0)


def _compact(items: Iterable[Any]) -> List[str]:
    return [_text(item) for item in items if _text(item)]


def _component(summary: Dict[str, Any]) -> Dict[str, Any]:
    return summary.get("knowledge_source_coverage") or summary or {}


def _source_groups(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    groups = _component(summary).get("source_groups") or {}
    return groups if isinstance(groups, dict) else {}


def _focus_areas(summary: Dict[str, Any]) -> List[str]:
    return _compact(_component(summary).get("focus_areas") or [])


def _priority_domains(summary: Dict[str, Any]) -> List[str]:
    return _compact(_component(summary).get("priority_domains") or [])


def _expansion_names(summary: Dict[str, Any]) -> List[str]:
    rows = _component(summary).get("expansion_candidates") or []
    names: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = _text(row.get("name"))
        if name and name not in names:
            names.append(name)
    return names


def build_knowledge_source_drift_status(
    current_summary: Dict[str, Any],
    previous_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare current and previous knowledge source coverage summaries."""
    current_component = _component(current_summary)
    previous_component = _component(previous_summary)

    if not previous_component:
        return {
            "status": "baseline_missing",
            "current_status": _status(current_component.get("status")),
            "previous_status": "baseline_missing",
            "ready_source_group_delta": int(
                current_component.get("ready_source_group_count") or 0
            ),
            "missing_source_group_delta": int(
                current_component.get("missing_source_group_count") or 0
            ),
            "regressions": [],
            "improvements": [],
            "resolved_focus_areas": [],
            "new_focus_areas": _focus_areas(current_summary),
            "resolved_priority_domains": [],
            "new_priority_domains": _priority_domains(current_summary),
            "resolved_expansion_candidates": [],
            "new_expansion_candidates": _expansion_names(current_summary),
            "source_group_regressions": [],
            "source_group_improvements": [],
            "source_group_changes": [],
        }

    current_groups = _source_groups(current_summary)
    previous_groups = _source_groups(previous_summary)
    group_names = sorted(set(current_groups) | set(previous_groups))
    source_group_changes: List[Dict[str, Any]] = []
    source_group_regressions: List[str] = []
    source_group_improvements: List[str] = []

    for name in group_names:
        current_row = current_groups.get(name) or {}
        previous_row = previous_groups.get(name) or {}
        current_status = _status(current_row.get("status"))
        previous_status = _status(previous_row.get("status"))
        current_rank = _rank(current_status)
        previous_rank = _rank(previous_status)
        if current_rank > previous_rank:
            trend = "improved"
            source_group_improvements.append(name)
        elif current_rank < previous_rank:
            trend = "regressed"
            source_group_regressions.append(name)
        else:
            trend = "stable"
        source_group_changes.append(
            {
                "name": name,
                "label": _text(current_row.get("label"))
                or _text(previous_row.get("label"))
                or name,
                "domain": _text(current_row.get("domain"))
                or _text(previous_row.get("domain"))
                or name,
                "previous_status": previous_status,
                "current_status": current_status,
                "trend": trend,
                "previous_source_item_count": int(
                    previous_row.get("source_item_count") or 0
                ),
                "current_source_item_count": int(current_row.get("source_item_count") or 0),
                "source_item_delta": int(current_row.get("source_item_count") or 0)
                - int(previous_row.get("source_item_count") or 0),
                "previous_missing_source_tables": _compact(
                    previous_row.get("missing_source_tables") or []
                ),
                "current_missing_source_tables": _compact(
                    current_row.get("missing_source_tables") or []
                ),
            }
        )

    previous_focus = set(_focus_areas(previous_summary))
    current_focus = set(_focus_areas(current_summary))
    previous_priority = set(_priority_domains(previous_summary))
    current_priority = set(_priority_domains(current_summary))
    previous_expansion = set(_expansion_names(previous_summary))
    current_expansion = set(_expansion_names(current_summary))

    regressions = _compact(
        [row["name"] for row in source_group_changes if row.get("trend") == "regressed"]
    )
    improvements = _compact(
        [row["name"] for row in source_group_changes if row.get("trend") == "improved"]
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
        "ready_source_group_delta": int(
            current_component.get("ready_source_group_count") or 0
        )
        - int(previous_component.get("ready_source_group_count") or 0),
        "missing_source_group_delta": int(
            current_component.get("missing_source_group_count") or 0
        )
        - int(previous_component.get("missing_source_group_count") or 0),
        "regressions": regressions,
        "improvements": improvements,
        "resolved_focus_areas": sorted(previous_focus - current_focus),
        "new_focus_areas": sorted(current_focus - previous_focus),
        "resolved_priority_domains": sorted(previous_priority - current_priority),
        "new_priority_domains": sorted(current_priority - previous_priority),
        "resolved_expansion_candidates": sorted(previous_expansion - current_expansion),
        "new_expansion_candidates": sorted(current_expansion - previous_expansion),
        "source_group_regressions": source_group_regressions,
        "source_group_improvements": source_group_improvements,
        "source_group_changes": source_group_changes,
    }


def knowledge_source_drift_recommendations(summary: Dict[str, Any]) -> List[str]:
    status = _status(summary.get("status"))
    if status == "baseline_missing":
        items = [
            "Persist the current knowledge source coverage as the next benchmark baseline."
        ]
        new_priority = _compact(summary.get("new_priority_domains") or [])
        if new_priority:
            items.append("Initial priority domains: " + ", ".join(new_priority))
        return items
    if status == "regressed":
        items = [
            "Restore regressed knowledge source groups before claiming benchmark source stability."
        ]
        regressed = _compact(summary.get("source_group_regressions") or [])
        if regressed:
            items.append("Regressed source groups: " + ", ".join(regressed))
        return items
    if status == "improved":
        items = [
            "Promote the improved knowledge source coverage after CI surfaces refresh."
        ]
        resolved = _compact(summary.get("resolved_priority_domains") or [])
        improved = _compact(summary.get("source_group_improvements") or [])
        if resolved:
            items.append("Resolved priority domains: " + ", ".join(resolved))
        if improved:
            items.append("Improved source groups: " + ", ".join(improved))
        return items
    if status == "mixed":
        return [
            "Keep the previous knowledge source coverage baseline until regressions are cleared.",
            "Regressed source groups: "
            + (", ".join(_compact(summary.get("source_group_regressions") or [])) or "none"),
            "Improved source groups: "
            + (", ".join(_compact(summary.get("source_group_improvements") or [])) or "none"),
        ]
    if summary.get("new_priority_domains"):
        return [
            "Watch new priority domains: "
            + ", ".join(_compact(summary.get("new_priority_domains") or []))
        ]
    return [
        "Knowledge source coverage is stable against the previous benchmark baseline."
    ]


def render_knowledge_source_drift_markdown(
    payload: Dict[str, Any], title: str
) -> str:
    component = payload.get("knowledge_source_drift") or {}
    lines = [
        f"# {title}",
        "",
        "## Status",
        "",
        f"- `status`: `{component.get('status', 'baseline_missing')}`",
        f"- `current_status`: `{component.get('current_status', 'unknown')}`",
        f"- `previous_status`: `{component.get('previous_status', 'baseline_missing')}`",
        f"- `ready_source_group_delta`: `{component.get('ready_source_group_delta', 0)}`",
        f"- `missing_source_group_delta`: `{component.get('missing_source_group_delta', 0)}`",
        "",
        "## Changes",
        "",
        f"- `regressions`: `{', '.join(component.get('regressions') or []) or 'none'}`",
        f"- `improvements`: `{', '.join(component.get('improvements') or []) or 'none'}`",
        f"- `resolved_focus_areas`: `{', '.join(component.get('resolved_focus_areas') or []) or 'none'}`",
        f"- `new_focus_areas`: `{', '.join(component.get('new_focus_areas') or []) or 'none'}`",
        f"- `resolved_priority_domains`: `{', '.join(component.get('resolved_priority_domains') or []) or 'none'}`",
        f"- `new_priority_domains`: `{', '.join(component.get('new_priority_domains') or []) or 'none'}`",
        f"- `resolved_expansion_candidates`: `{', '.join(component.get('resolved_expansion_candidates') or []) or 'none'}`",
        f"- `new_expansion_candidates`: `{', '.join(component.get('new_expansion_candidates') or []) or 'none'}`",
        "",
        "## Source Group Changes",
        "",
    ]
    changes = component.get("source_group_changes") or []
    if changes:
        for row in changes:
            lines.append(
                "- "
                + (
                    f"{row.get('name')}: {row.get('previous_status')} -> "
                    f"{row.get('current_status')} ({row.get('trend')}); "
                    f"source_item_delta={row.get('source_item_delta', 0)}"
                )
            )
    else:
        lines.append("- none")
    recommendations = payload.get("recommendations") or []
    lines.extend(["", "## Recommendations", ""])
    if recommendations:
        for item in recommendations:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"
