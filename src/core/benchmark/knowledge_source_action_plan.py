"""Turn knowledge source coverage gaps into an executable action plan."""

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


def _priority(status: str) -> str:
    if status == "missing":
        return "high"
    if status == "partial":
        return "medium"
    return "low"


def _item_status(status: str) -> str:
    if status == "missing":
        return "blocked"
    if status == "partial":
        return "required"
    return "ready"


def _join(items: Iterable[Any]) -> str:
    values = _compact(items, limit=8)
    return ", ".join(values) if values else "none"


def _source_group_item(name: str, row: Dict[str, Any]) -> Dict[str, Any] | None:
    status = _text(row.get("status")).lower()
    if status == "ready":
        return None
    missing_tables = list(row.get("missing_source_tables") or [])
    return {
        "id": f"{name}:coverage",
        "source_group": name,
        "domain": _text(row.get("domain")) or name,
        "label": _text(row.get("label")) or name,
        "stage": "source_group",
        "status": _item_status(status),
        "priority": _priority(status),
        "reason": (
            f"Source group is {status}; "
            f"missing_tables={_join(missing_tables)}; "
            f"source_items={row.get('source_item_count', 0)}; "
            f"reference_standards={row.get('reference_standard_count', 0)}"
        ),
        "action": (
            f"Raise {_text(row.get('label')) or name} to ready by restoring missing "
            "tables and ensuring source items / reference standards remain non-zero."
        ),
        "missing_tables": missing_tables,
    }


def _domain_item(name: str, row: Dict[str, Any]) -> Dict[str, Any] | None:
    status = _text(row.get("status")).lower()
    if status == "ready":
        return None
    focus_groups = list(row.get("focus_source_groups") or [])
    return {
        "id": f"{name}:domain",
        "source_group": "",
        "domain": name,
        "label": _text(row.get("label")) or name,
        "stage": "domain",
        "status": _item_status(status),
        "priority": _priority(status),
        "reason": (
            f"Domain is {status}; "
            f"focus_source_groups={_join(focus_groups)}"
        ),
        "action": (
            f"Close {_text(row.get('label')) or name} domain source gaps before "
            "calling this knowledge area benchmark-ready."
        ),
        "focus_source_groups": focus_groups,
    }


def _expansion_item(row: Dict[str, Any]) -> Dict[str, Any] | None:
    status = _text(row.get("status")).lower()
    if status != "ready":
        return None
    name = _text(row.get("name")) or "unknown"
    return {
        "id": f"{name}:expansion",
        "source_group": name,
        "domain": _text(row.get("domain")) or "manufacturing",
        "label": _text(row.get("label")) or name,
        "stage": "expansion",
        "status": "recommended",
        "priority": "medium",
        "reason": (
            f"Expansion candidate is ready; "
            f"source_tables={row.get('source_table_count', 0)}; "
            f"source_items={row.get('source_item_count', 0)}"
        ),
        "action": (
            f"Promote {_text(row.get('label')) or name} into benchmark bundle / "
            "companion / release surfaces as the next-wave knowledge domain."
        ),
    }


def build_knowledge_source_action_plan(
    knowledge_source_coverage_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Build an action plan from knowledge source coverage gaps."""
    component = (
        knowledge_source_coverage_summary.get("knowledge_source_coverage")
        or knowledge_source_coverage_summary
        or {}
    )
    source_groups = component.get("source_groups") or {}
    domains = component.get("domains") or {}
    expansion_candidates = component.get("expansion_candidates") or []

    actions: List[Dict[str, Any]] = []
    source_group_action_counts: Dict[str, int] = {}

    for name, row in source_groups.items():
        item = _source_group_item(name, row)
        if item:
            actions.append(item)
            source_group_action_counts[name] = source_group_action_counts.get(name, 0) + 1

    for name, row in domains.items():
        item = _domain_item(name, row)
        if item:
            actions.append(item)

    for row in expansion_candidates:
        item = _expansion_item(row)
        if item:
            actions.append(item)
            name = _text(row.get("name")) or "unknown"
            source_group_action_counts[name] = source_group_action_counts.get(name, 0) + 1

    actions.sort(
        key=lambda item: (
            {"high": 0, "medium": 1, "low": 2}.get(item.get("priority") or "", 3),
            item.get("domain") or "",
            item.get("stage") or "",
            item.get("source_group") or "",
        )
    )
    high_priority_actions = [
        item["id"] for item in actions if item.get("priority") == "high"
    ]
    medium_priority_actions = [
        item["id"] for item in actions if item.get("priority") == "medium"
    ]
    if not actions:
        status = "knowledge_source_action_plan_ready"
    elif high_priority_actions:
        status = "knowledge_source_action_plan_blocked"
    else:
        status = "knowledge_source_action_plan_partial"

    priority_domains = _compact(
        [item.get("domain") for item in actions if item.get("priority") in {"high", "medium"}],
        limit=8,
    )
    expansion_actions = [
        item["id"] for item in actions if item.get("stage") == "expansion"
    ]
    return {
        "status": status,
        "total_action_count": len(actions),
        "high_priority_action_count": len(high_priority_actions),
        "medium_priority_action_count": len(medium_priority_actions),
        "priority_domains": priority_domains,
        "recommended_first_actions": actions[:3],
        "actions": actions,
        "source_group_action_counts": source_group_action_counts,
        "expansion_action_count": len(expansion_actions),
    }


def knowledge_source_action_plan_recommendations(component: Dict[str, Any]) -> List[str]:
    status = _text(component.get("status")).lower()
    if status == "knowledge_source_action_plan_ready":
        return [
            "Knowledge source action plan is clear: no remaining source-group, domain, "
            "or expansion actions are blocking benchmark rollout."
        ]
    recommendations: List[str] = []
    for item in component.get("actions") or []:
        if item.get("priority") not in {"high", "medium"}:
            continue
        recommendations.append(
            f"{item.get('stage')}: {_text(item.get('action')) or 'Investigate next source action.'}"
        )
    return _compact(recommendations, limit=10)


def render_knowledge_source_action_plan_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    component = payload.get("knowledge_source_action_plan") or {}
    lines = [
        f"# {title}",
        "",
        "## Overview",
        "",
        f"- `status`: `{component.get('status', 'knowledge_source_action_plan_blocked')}`",
        f"- `total_action_count`: `{component.get('total_action_count', 0)}`",
        f"- `high_priority_action_count`: `{component.get('high_priority_action_count', 0)}`",
        f"- `medium_priority_action_count`: `{component.get('medium_priority_action_count', 0)}`",
        f"- `priority_domains`: `{', '.join(component.get('priority_domains') or []) or 'none'}`",
        f"- `expansion_action_count`: `{component.get('expansion_action_count', 0)}`",
        "",
        "## Actions",
        "",
    ]
    actions = component.get("actions") or []
    if actions:
        for item in actions:
            lines.append(
                "- "
                f"`{item.get('id')}` priority=`{item.get('priority')}` "
                f"stage=`{item.get('stage')}` action=`{item.get('action')}`"
            )
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
