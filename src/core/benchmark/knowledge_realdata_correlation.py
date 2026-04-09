"""Correlate benchmark knowledge domains with real-data validation coverage."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


DOMAIN_REALDATA_COMPONENTS: Dict[str, Dict[str, Any]] = {
    "tolerance": {
        "label": "Tolerance & Fits",
        "components": ["hybrid_dxf", "history_h5", "step_dir"],
        "action": (
            "Connect tolerance/fits benchmark coverage to DXF hybrid, history, and STEP "
            "directory real-data validation."
        ),
    },
    "standards": {
        "label": "Standards & Design Tables",
        "components": ["hybrid_dxf", "history_h5"],
        "action": (
            "Verify standard-part and design-table evidence against DXF hybrid and "
            "history-sequence real-data runs."
        ),
    },
    "gdt": {
        "label": "GD&T & Datums",
        "components": ["hybrid_dxf", "step_smoke", "step_dir"],
        "action": (
            "Tie GD&T benchmark evidence to DXF hybrid plus STEP/B-Rep smoke and "
            "directory validations."
        ),
    },
}

READY_STATUSES = {"ready"}
PARTIAL_STATUSES = {"partial", "weak"}
BLOCKED_STATUSES = {"environment_blocked", "load_failed"}
MISSING_STATUSES = {"missing", ""}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _compact(items: Iterable[Any], *, limit: int = 6) -> List[str]:
    out: List[str] = []
    for item in items:
        text = _text(item)
        if not text:
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _realdata_domain_status(component_statuses: Dict[str, Any], names: List[str]) -> Dict[str, Any]:
    rows = {name: _text(component_statuses.get(name)) or "missing" for name in names}
    ready = [name for name, status in rows.items() if status in READY_STATUSES]
    partial = [name for name, status in rows.items() if status in PARTIAL_STATUSES]
    blocked = [name for name, status in rows.items() if status in BLOCKED_STATUSES]
    missing = [name for name, status in rows.items() if status in MISSING_STATUSES]
    if len(ready) == len(names):
        status = "ready"
    elif blocked and not (ready or partial):
        status = "blocked"
    elif ready or partial:
        status = "partial"
    else:
        status = "missing"
    return {
        "status": status,
        "components": rows,
        "ready_components": ready,
        "partial_components": partial,
        "blocked_components": blocked,
        "missing_components": missing,
    }


def _domain_status(
    readiness_status: str,
    application_status: str,
    realdata_status: str,
) -> str:
    if readiness_status == "ready" and application_status == "ready" and realdata_status == "ready":
        return "ready"
    if (
        readiness_status == "missing"
        and application_status == "missing"
        and realdata_status == "missing"
    ):
        return "missing"
    if realdata_status == "blocked":
        return "blocked"
    if "missing" in {readiness_status, application_status, realdata_status}:
        return "blocked"
    return "partial"


def _priority(status: str) -> str:
    if status in {"blocked", "missing"}:
        return "high"
    if status == "partial":
        return "medium"
    return "low"


def build_knowledge_realdata_correlation_status(
    knowledge_readiness_summary: Dict[str, Any],
    knowledge_application_summary: Dict[str, Any],
    realdata_signals_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a domain-level benchmark summary that ties knowledge to real-data evidence."""
    readiness_component = (
        knowledge_readiness_summary.get("knowledge_readiness")
        or knowledge_readiness_summary
        or {}
    )
    application_component = (
        knowledge_application_summary.get("knowledge_application")
        or knowledge_application_summary
        or {}
    )
    realdata_component = (
        realdata_signals_summary.get("realdata_signals")
        or realdata_signals_summary
        or {}
    )

    readiness_domains = readiness_component.get("domains") or {}
    application_domains = application_component.get("domains") or {}
    realdata_components = realdata_component.get("component_statuses") or {}

    domains: Dict[str, Dict[str, Any]] = {}
    focus_areas_detail: List[Dict[str, Any]] = []
    ready_count = 0
    partial_count = 0
    blocked_count = 0

    for name, spec in DOMAIN_REALDATA_COMPONENTS.items():
        readiness_row = readiness_domains.get(name) or {}
        application_row = application_domains.get(name) or {}
        realdata_row = _realdata_domain_status(realdata_components, list(spec["components"]))
        readiness_status = _text(readiness_row.get("status")) or "missing"
        application_status = _text(application_row.get("status")) or "missing"
        status = _domain_status(
            readiness_status,
            application_status,
            _text(realdata_row.get("status")) or "missing",
        )
        row = {
            "domain": name,
            "label": _text(spec.get("label")) or name,
            "status": status,
            "priority": _priority(status),
            "readiness_status": readiness_status,
            "application_status": application_status,
            "realdata_status": _text(realdata_row.get("status")) or "missing",
            "realdata_components": realdata_row.get("components") or {},
            "ready_realdata_components": list(realdata_row.get("ready_components") or []),
            "partial_realdata_components": list(realdata_row.get("partial_components") or []),
            "blocked_realdata_components": list(realdata_row.get("blocked_components") or []),
            "missing_realdata_components": list(realdata_row.get("missing_components") or []),
            "knowledge_focus_components": list(readiness_row.get("focus_components") or []),
            "knowledge_missing_metrics": list(readiness_row.get("missing_metrics") or []),
            "application_signal_count": int(application_row.get("signal_count") or 0),
            "action": _text(application_row.get("action")) or _text(spec.get("action")),
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

    if ready_count == len(DOMAIN_REALDATA_COMPONENTS):
        overall_status = "knowledge_realdata_ready"
    elif ready_count > 0 or partial_count > 0:
        overall_status = "knowledge_realdata_partial"
    else:
        overall_status = "knowledge_realdata_missing"

    return {
        "status": overall_status,
        "ready_domain_count": ready_count,
        "partial_domain_count": partial_count,
        "blocked_domain_count": blocked_count,
        "total_domain_count": len(DOMAIN_REALDATA_COMPONENTS),
        "priority_domains": [
            row["domain"]
            for row in focus_areas_detail
            if _text(row.get("priority")) == "high"
        ],
        "focus_areas": [row["domain"] for row in focus_areas_detail],
        "focus_areas_detail": focus_areas_detail,
        "domains": domains,
    }


def knowledge_realdata_correlation_recommendations(summary: Dict[str, Any]) -> List[str]:
    status = _text(summary.get("status")).lower()
    if status == "knowledge_realdata_missing":
        return [
            "Establish shared knowledge + real-data evidence before treating "
            "standards/tolerance/GD&T benchmark status as release-relevant."
        ]

    items: List[str] = []
    for row in summary.get("focus_areas_detail") or []:
        domain = _text(row.get("domain")) or "unknown"
        readiness_status = _text(row.get("readiness_status")) or "missing"
        application_status = _text(row.get("application_status")) or "missing"
        realdata_status = _text(row.get("realdata_status")) or "missing"
        action = _text(row.get("action"))
        if readiness_status == "missing":
            items.append(f"Backfill {domain} knowledge foundation: {action}")
        if application_status == "missing":
            items.append(f"Promote {domain} application evidence: {action}")
        if realdata_status in {"blocked", "missing"}:
            missing_realdata = ", ".join(row.get("missing_realdata_components") or [])
            blocked_realdata = ", ".join(row.get("blocked_realdata_components") or [])
            detail = blocked_realdata or missing_realdata or "real-data coverage"
            items.append(f"Expand {domain} real-data coverage: {detail}")
        elif realdata_status == "partial":
            partial_realdata = ", ".join(row.get("partial_realdata_components") or [])
            detail = partial_realdata or "partial real-data coverage"
            items.append(f"Raise {domain} real-data depth: {detail}")
    if status == "knowledge_realdata_partial":
        items.append(
            "Use companion, release decision, and release runbook surfaces to track "
            "whether knowledge-domain gaps align with real-data validation gaps."
        )
    return _compact(items, limit=12)


def render_knowledge_realdata_correlation_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    component = payload.get("knowledge_realdata_correlation") or {}
    lines = [
        f"# {title}",
        "",
        "## Overview",
        "",
        f"- `status`: `{component.get('status', 'knowledge_realdata_missing')}`",
        f"- `ready_domain_count`: `{component.get('ready_domain_count', 0)}`",
        f"- `partial_domain_count`: `{component.get('partial_domain_count', 0)}`",
        f"- `blocked_domain_count`: `{component.get('blocked_domain_count', 0)}`",
        f"- `priority_domains`: "
        f"`{', '.join(component.get('priority_domains') or []) or 'none'}`",
        "",
        "## Domains",
        "",
    ]
    domains = component.get("domains") or {}
    for name, row in domains.items():
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
            "- `realdata_components`: "
            f"`{row.get('realdata_components') or {}}`"
        )
        lines.append(
            "- `knowledge_missing_metrics`: "
            f"`{', '.join(row.get('knowledge_missing_metrics') or []) or 'none'}`"
        )
        lines.append(
            "- `action`: "
            f"`{row.get('action') or 'n/a'}`"
        )
        lines.append("")
    lines.extend(["## Recommendations", ""])
    recommendations = payload.get("recommendations") or []
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- none")
    return "\n".join(lines).rstrip() + "\n"
