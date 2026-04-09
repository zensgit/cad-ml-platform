"""Aggregate benchmark knowledge domain signals into a single matrix view."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


DOMAIN_ORDER = ("tolerance", "standards", "gdt")

DOMAIN_LABELS: Dict[str, str] = {
    "tolerance": "Tolerance & Fits",
    "standards": "Standards & Design Tables",
    "gdt": "GD&T & Datums",
}


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


def _matrix_status(
    readiness_status: str,
    application_status: str,
    realdata_status: str,
) -> str:
    if all(
        status == "ready"
        for status in (readiness_status, application_status, realdata_status)
    ):
        return "ready"
    if all(
        status == "missing"
        for status in (readiness_status, application_status, realdata_status)
    ):
        return "missing"
    if (
        readiness_status == "missing"
        or application_status == "missing"
        or realdata_status in {"missing", "blocked"}
    ):
        return "blocked"
    return "partial"


def _priority(status: str) -> str:
    if status in {"blocked", "missing"}:
        return "high"
    if status == "partial":
        return "medium"
    return "low"


def _merge_focus_components(
    readiness_row: Dict[str, Any],
    application_row: Dict[str, Any],
    realdata_row: Dict[str, Any],
) -> List[str]:
    items: List[str] = []
    for name in readiness_row.get("focus_components") or []:
        text = _text(name)
        if text and text not in items:
            items.append(text)
    if _normalize_status(application_row.get("status")) != "ready":
        domain_name = _text(application_row.get("domain"))
        if domain_name and domain_name not in items:
            items.append(domain_name)
    for group_name in (
        "missing_realdata_components",
        "blocked_realdata_components",
        "partial_realdata_components",
    ):
        for name in realdata_row.get(group_name) or []:
            text = _text(name)
            if text and text not in items:
                items.append(text)
    return items


def _merge_missing_metrics(
    readiness_row: Dict[str, Any],
    application_row: Dict[str, Any],
    realdata_row: Dict[str, Any],
) -> List[str]:
    items: List[str] = []
    for name in readiness_row.get("missing_metrics") or []:
        text = _text(name)
        if text and text not in items:
            items.append(text)
    if int(application_row.get("signal_count") or 0) <= 0:
        items.append("application_signal_count")
    for group_name in ("missing_realdata_components", "blocked_realdata_components"):
        for name in realdata_row.get(group_name) or []:
            text = _text(name)
            if not text:
                continue
            metric = f"realdata:{text}"
            if metric not in items:
                items.append(metric)
    return items


def _pick_action(
    readiness_row: Dict[str, Any],
    application_row: Dict[str, Any],
    realdata_row: Dict[str, Any],
) -> str:
    for value in (
        realdata_row.get("action"),
        application_row.get("action"),
        readiness_row.get("action"),
    ):
        text = _text(value)
        if text:
            return text
    return (
        "Raise benchmark knowledge-domain coverage across foundation, application, "
        "and real-data evidence."
    )


def build_knowledge_domain_matrix_status(
    knowledge_readiness_summary: Dict[str, Any],
    knowledge_application_summary: Dict[str, Any],
    knowledge_realdata_correlation_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine benchmark knowledge views into a single per-domain matrix."""
    readiness_root = (
        knowledge_readiness_summary.get("knowledge_readiness")
        or knowledge_readiness_summary
        or {}
    )
    application_root = (
        knowledge_application_summary.get("knowledge_application")
        or knowledge_application_summary
        or {}
    )
    realdata_root = (
        knowledge_realdata_correlation_summary.get("knowledge_realdata_correlation")
        or knowledge_realdata_correlation_summary
        or {}
    )
    readiness_domains = readiness_root.get("domains") or {}
    application_domains = application_root.get("domains") or {}
    realdata_domains = realdata_root.get("domains") or {}

    domains: Dict[str, Dict[str, Any]] = {}
    focus_areas_detail: List[Dict[str, Any]] = []
    ready_count = 0
    partial_count = 0
    blocked_count = 0

    for name in DOMAIN_ORDER:
        readiness_row = readiness_domains.get(name) or {}
        application_row = application_domains.get(name) or {}
        realdata_row = realdata_domains.get(name) or {}
        readiness_status = _normalize_status(readiness_row.get("status"))
        application_status = _normalize_status(application_row.get("status"))
        realdata_status = _normalize_status(realdata_row.get("status"))
        status = _matrix_status(
            readiness_status,
            application_status,
            realdata_status,
        )
        row = {
            "domain": name,
            "label": _text(readiness_row.get("label"))
            or _text(application_row.get("label"))
            or _text(realdata_row.get("label"))
            or DOMAIN_LABELS.get(name, name),
            "status": status,
            "priority": _priority(status),
            "readiness_status": readiness_status,
            "application_status": application_status,
            "realdata_status": realdata_status,
            "focus_components": _merge_focus_components(
                readiness_row,
                application_row,
                realdata_row,
            ),
            "missing_metrics": _merge_missing_metrics(
                readiness_row,
                application_row,
                realdata_row,
            ),
            "application_signal_count": int(application_row.get("signal_count") or 0),
            "ready_realdata_components": list(
                realdata_row.get("ready_realdata_components") or []
            ),
            "partial_realdata_components": list(
                realdata_row.get("partial_realdata_components") or []
            ),
            "blocked_realdata_components": list(
                realdata_row.get("blocked_realdata_components") or []
            ),
            "missing_realdata_components": list(
                realdata_row.get("missing_realdata_components") or []
            ),
            "action": _pick_action(readiness_row, application_row, realdata_row),
        }
        domains[name] = row
        if status == "ready":
            ready_count += 1
        elif status == "partial":
            partial_count += 1
            focus_areas_detail.append(row)
        else:
            blocked_count += 1
            focus_areas_detail.append(row)

    if ready_count == len(DOMAIN_ORDER):
        overall_status = "knowledge_domain_matrix_ready"
    elif ready_count > 0 or partial_count > 0:
        overall_status = "knowledge_domain_matrix_partial"
    else:
        overall_status = "knowledge_domain_matrix_missing"

    return {
        "status": overall_status,
        "ready_domain_count": ready_count,
        "partial_domain_count": partial_count,
        "blocked_domain_count": blocked_count,
        "total_domain_count": len(DOMAIN_ORDER),
        "focus_areas": [row["domain"] for row in focus_areas_detail],
        "focus_areas_detail": focus_areas_detail,
        "priority_domains": [
            row["domain"]
            for row in focus_areas_detail
            if _text(row.get("priority")) == "high"
        ],
        "domains": domains,
        "matrix": domains,
    }


def knowledge_domain_matrix_recommendations(summary: Dict[str, Any]) -> List[str]:
    status = _normalize_status(summary.get("status"))
    if status == "knowledge_domain_matrix_missing":
        return [
            "Build shared tolerance/standards/GD&T benchmark evidence before "
            "claiming a complete knowledge domain matrix."
        ]

    items: List[str] = []
    for row in summary.get("focus_areas_detail") or []:
        domain = _text(row.get("domain")) or "unknown"
        readiness_status = _normalize_status(row.get("readiness_status"))
        application_status = _normalize_status(row.get("application_status"))
        realdata_status = _normalize_status(row.get("realdata_status"))
        action = _text(row.get("action"))
        if readiness_status == "missing":
            items.append(f"Backfill {domain} foundation: {action}")
        if application_status == "missing":
            items.append(f"Promote {domain} application evidence: {action}")
        if realdata_status in {"missing", "blocked"}:
            detail = ", ".join(row.get("blocked_realdata_components") or []) or ", ".join(
                row.get("missing_realdata_components") or []
            )
            items.append(
                f"Expand {domain} real-data coverage: {detail or 'real-data evidence missing'}"
            )
        elif realdata_status == "partial":
            detail = ", ".join(row.get("partial_realdata_components") or [])
            items.append(
                f"Raise {domain} real-data depth: {detail or 'partial real-data coverage'}"
            )
    if status == "knowledge_domain_matrix_partial":
        items.append(
            "Use companion, artifact bundle, release decision, and release runbook "
            "surfaces to track the same domain gaps until all three knowledge "
            "layers align."
        )
    return _compact(items, limit=12)


def render_knowledge_domain_matrix_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    component = payload.get("knowledge_domain_matrix") or {}
    lines = [
        f"# {title}",
        "",
        "## Overview",
        "",
        f"- `status`: `{component.get('status', 'knowledge_domain_matrix_missing')}`",
        f"- `ready_domain_count`: `{component.get('ready_domain_count', 0)}`",
        f"- `partial_domain_count`: `{component.get('partial_domain_count', 0)}`",
        f"- `blocked_domain_count`: `{component.get('blocked_domain_count', 0)}`",
        f"- `priority_domains`: `{', '.join(component.get('priority_domains') or []) or 'none'}`",
        "",
        "## Domains",
        "",
    ]
    for name, row in (component.get("domains") or {}).items():
        lines.append(f"### {name}")
        lines.append("")
        for key in (
            "status",
            "priority",
            "readiness_status",
            "application_status",
            "realdata_status",
            "application_signal_count",
        ):
            lines.append(f"- `{key}`: `{row.get(key)}`")
        lines.append(
            "- `focus_components`: "
            f"`{', '.join(row.get('focus_components') or []) or 'none'}`"
        )
        lines.append(
            "- `missing_metrics`: "
            f"`{', '.join(row.get('missing_metrics') or []) or 'none'}`"
        )
        lines.append(
            "- `action`: "
            f"{row.get('action') or 'none'}"
        )
        lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    recommendations = payload.get("recommendations") or []
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)
