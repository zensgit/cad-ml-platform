"""Reusable benchmark helpers for knowledge readiness drift signals."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


STATUS_RANK = {
    "ready": 3,
    "partial": 2,
    "missing": 1,
    "knowledge_foundation_ready": 3,
    "knowledge_foundation_partial": 2,
    "knowledge_foundation_missing": 1,
}


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _component_status(value: Any) -> str:
    return str(value or "missing").strip() or "missing"


def _status_rank(value: Any) -> int:
    return STATUS_RANK.get(_component_status(value), 0)


def _component_map(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    component = summary.get("knowledge_readiness") or summary
    components = component.get("components") or {}
    return components if isinstance(components, dict) else {}


def _focus_area_names(summary: Dict[str, Any]) -> List[str]:
    component = summary.get("knowledge_readiness") or summary
    focus = component.get("focus_areas") or []
    return [str(item).strip() for item in focus if str(item).strip()]


def _compact(items: Iterable[str]) -> List[str]:
    return [str(item).strip() for item in items if str(item).strip()]


def build_knowledge_drift_status(
    current_summary: Dict[str, Any],
    previous_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare current and previous benchmark knowledge-readiness summaries."""
    current_component = current_summary.get("knowledge_readiness") or current_summary or {}
    previous_component = previous_summary.get("knowledge_readiness") or previous_summary or {}

    if not previous_component:
        return {
            "status": "baseline_missing",
            "current_status": _component_status(current_component.get("status")),
            "previous_status": "baseline_missing",
            "reference_item_delta": _to_int(current_component.get("total_reference_items")),
            "regressions": [],
            "improvements": [],
            "resolved_focus_areas": [],
            "new_focus_areas": _focus_area_names(current_summary),
            "component_changes": [],
        }

    current_components = _component_map(current_summary)
    previous_components = _component_map(previous_summary)
    component_names = sorted(set(current_components) | set(previous_components))
    component_changes: List[Dict[str, Any]] = []
    regressions: List[str] = []
    improvements: List[str] = []

    for name in component_names:
        current_row = current_components.get(name) or {}
        previous_row = previous_components.get(name) or {}
        current_status = _component_status(current_row.get("status"))
        previous_status = _component_status(previous_row.get("status"))
        current_rank = _status_rank(current_status)
        previous_rank = _status_rank(previous_status)
        if current_rank > previous_rank:
            trend = "improved"
            improvements.append(name)
        elif current_rank < previous_rank:
            trend = "regressed"
            regressions.append(name)
        else:
            trend = "stable"
        component_changes.append(
            {
                "component": name,
                "previous_status": previous_status,
                "current_status": current_status,
                "trend": trend,
                "previous_reference_items": _to_int(previous_row.get("total_reference_items")),
                "current_reference_items": _to_int(current_row.get("total_reference_items")),
                "reference_item_delta": (
                    _to_int(current_row.get("total_reference_items"))
                    - _to_int(previous_row.get("total_reference_items"))
                ),
            }
        )

    previous_focus = set(_focus_area_names(previous_summary))
    current_focus = set(_focus_area_names(current_summary))
    resolved_focus_areas = sorted(previous_focus - current_focus)
    new_focus_areas = sorted(current_focus - previous_focus)

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
        "current_status": _component_status(current_component.get("status")),
        "previous_status": _component_status(previous_component.get("status")),
        "current_total_reference_items": _to_int(
            current_component.get("total_reference_items")
        ),
        "previous_total_reference_items": _to_int(
            previous_component.get("total_reference_items")
        ),
        "reference_item_delta": (
            _to_int(current_component.get("total_reference_items"))
            - _to_int(previous_component.get("total_reference_items"))
        ),
        "regressions": regressions,
        "improvements": improvements,
        "resolved_focus_areas": resolved_focus_areas,
        "new_focus_areas": new_focus_areas,
        "component_changes": component_changes,
    }


def knowledge_drift_recommendations(summary: Dict[str, Any]) -> List[str]:
    status = _component_status(summary.get("status"))
    regressions = _compact(summary.get("regressions") or [])
    improvements = _compact(summary.get("improvements") or [])
    resolved_focus_areas = _compact(summary.get("resolved_focus_areas") or [])
    new_focus_areas = _compact(summary.get("new_focus_areas") or [])

    if status == "baseline_missing":
        return [
            "Persist the current benchmark knowledge-readiness summary as the next drift baseline."
        ]
    if status == "regressed":
        return [
            "Resolve knowledge regressions before claiming the benchmark surpass "
            "baseline remains stable.",
            "Regressed components: " + ", ".join(regressions),
        ]
    if status == "improved":
        items = [
            "Promote the improved knowledge baseline after CI and review "
            "surfaces are refreshed."
        ]
        if resolved_focus_areas:
            items.append("Resolved focus areas: " + ", ".join(resolved_focus_areas))
        return items
    if status == "mixed":
        items = [
            "Keep the previous baseline until knowledge regressions are cleared.",
            "Regressed components: " + ", ".join(regressions or ["none"]),
            "Improved components: " + ", ".join(improvements or ["none"]),
        ]
        return items
    if new_focus_areas:
        return ["Watch new focus areas: " + ", ".join(new_focus_areas)]
    return ["Knowledge readiness is stable against the previous benchmark baseline."]


def render_knowledge_drift_markdown(payload: Dict[str, Any], title: str) -> str:
    component = payload.get("knowledge_drift") or {}
    lines = [
        f"# {title}",
        "",
        "## Status",
        "",
        f"- `status`: `{component.get('status', 'baseline_missing')}`",
        f"- `current_status`: `{component.get('current_status', 'unknown')}`",
        f"- `previous_status`: `{component.get('previous_status', 'baseline_missing')}`",
        f"- `reference_item_delta`: `{component.get('reference_item_delta', 0)}`",
        "",
        "## Changes",
        "",
        f"- `regressions`: `{', '.join(component.get('regressions') or []) or 'none'}`",
        f"- `improvements`: `{', '.join(component.get('improvements') or []) or 'none'}`",
        f"- `resolved_focus_areas`: "
        f"`{', '.join(component.get('resolved_focus_areas') or []) or 'none'}`",
        f"- `new_focus_areas`: `{', '.join(component.get('new_focus_areas') or []) or 'none'}`",
        "",
        "## Component Changes",
        "",
    ]
    for row in component.get("component_changes") or []:
        lines.append(
            "- "
            f"`{row.get('component')}` "
            f"`{row.get('previous_status')}` -> `{row.get('current_status')}` "
            f"trend=`{row.get('trend')}` "
            f"delta=`{row.get('reference_item_delta')}`"
        )
    lines.extend(["", "## Recommendations", ""])
    recommendations = payload.get("recommendations") or []
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)
