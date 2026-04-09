"""Benchmark helpers for standards / tolerance / GD&T reference inventory."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .knowledge_source_coverage import collect_builtin_knowledge_source_snapshot


DOMAIN_SPECS: Dict[str, Dict[str, str]] = {
    "tolerance": {"label": "Tolerance & Fits"},
    "standards": {"label": "Standards & Design Tables"},
    "gdt": {"label": "GD&T & Datums"},
}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _compact(items: Iterable[Any], *, limit: int = 8) -> List[str]:
    out: List[str] = []
    for item in items:
        text = _text(item)
        if not text or text in out:
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _domain_status(*, total_tables: int, populated_tables: int, total_items: int) -> str:
    if total_tables == 0 or total_items <= 0 or populated_tables == 0:
        return "blocked"
    if populated_tables == total_tables:
        return "ready"
    return "partial"


def build_knowledge_reference_inventory_status(
    snapshot: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a standards / tolerance / GD&T reference inventory summary."""
    snapshot = snapshot or collect_builtin_knowledge_source_snapshot()
    domains: Dict[str, Dict[str, Any]] = {}
    priority_domains: List[str] = []
    focus_tables_detail: List[Dict[str, Any]] = []
    total_reference_items = 0
    total_table_count = 0
    populated_table_count = 0
    ready_domain_count = 0
    partial_domain_count = 0
    blocked_domain_count = 0

    for domain, spec in DOMAIN_SPECS.items():
        source_tables = dict(
            (snapshot.get(domain) or {}).get("source_tables") or {}
        )
        total_tables = len(source_tables)
        populated = {
            name: int(value or 0)
            for name, value in source_tables.items()
            if int(value or 0) > 0
        }
        missing_tables = [
            name for name, value in source_tables.items() if int(value or 0) <= 0
        ]
        domain_total_items = sum(int(value or 0) for value in source_tables.values())
        status = _domain_status(
            total_tables=total_tables,
            populated_tables=len(populated),
            total_items=domain_total_items,
        )
        if status != "ready":
            priority_domains.append(domain)
        if status == "ready":
            ready_domain_count += 1
        elif status == "partial":
            partial_domain_count += 1
        else:
            blocked_domain_count += 1
        total_reference_items += domain_total_items
        total_table_count += total_tables
        populated_table_count += len(populated)
        if missing_tables:
            focus_tables_detail.append(
                {
                    "domain": domain,
                    "label": spec["label"],
                    "status": status,
                    "missing_tables": missing_tables,
                    "action": (
                        "Backfill "
                        f"{domain} reference tables: {', '.join(missing_tables[:3])}"
                    ),
                }
            )
        domains[domain] = {
            "label": spec["label"],
            "status": status,
            "total_reference_items": domain_total_items,
            "total_table_count": total_tables,
            "populated_table_count": len(populated),
            "missing_table_count": len(missing_tables),
            "top_tables": _compact(
                [
                    f"{name}:{value}"
                    for name, value in sorted(
                        populated.items(), key=lambda item: (-item[1], item[0])
                    )
                ],
                limit=5,
            ),
            "missing_tables": missing_tables,
            "source_tables": source_tables,
        }

    if blocked_domain_count:
        status = "knowledge_reference_inventory_blocked"
    elif partial_domain_count:
        status = "knowledge_reference_inventory_partial"
    else:
        status = "knowledge_reference_inventory_ready"

    return {
        "status": status,
        "ready_domain_count": ready_domain_count,
        "partial_domain_count": partial_domain_count,
        "blocked_domain_count": blocked_domain_count,
        "total_domain_count": len(DOMAIN_SPECS),
        "total_reference_items": total_reference_items,
        "total_table_count": total_table_count,
        "populated_table_count": populated_table_count,
        "priority_domains": priority_domains,
        "focus_tables_detail": focus_tables_detail,
        "domains": domains,
    }


def knowledge_reference_inventory_recommendations(
    component: Dict[str, Any],
) -> List[str]:
    recommendations: List[str] = []
    for row in component.get("focus_tables_detail") or []:
        action = _text(row.get("action"))
        if action and action not in recommendations:
            recommendations.append(action)
    if (
        not recommendations
        and component.get("status") == "knowledge_reference_inventory_ready"
    ):
        recommendations.append(
            "Promote standards/tolerance/GD&T reference inventory into "
            "benchmark control-plane reviews."
        )
    return recommendations


def render_knowledge_reference_inventory_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    component = payload.get("knowledge_reference_inventory") or payload or {}
    recommendations = payload.get("recommendations") or []
    lines = [
        f"# {title}",
        "",
        f"- Status: `{component.get('status') or 'unknown'}`",
        f"- Ready domains: `{component.get('ready_domain_count', 0)}`",
        f"- Partial domains: `{component.get('partial_domain_count', 0)}`",
        f"- Blocked domains: `{component.get('blocked_domain_count', 0)}`",
        f"- Total reference items: `{component.get('total_reference_items', 0)}`",
        "- Populated tables: "
        f"`{component.get('populated_table_count', 0)}/{component.get('total_table_count', 0)}`",
        "",
        "## Domains",
        "",
        "| Domain | Status | Reference Items | Populated Tables | Missing Tables | Top Tables |",
        "|--------|--------|-----------------|------------------|----------------|------------|",
    ]
    for name, row in (component.get("domains") or {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("label") or name,
                    row.get("status") or "unknown",
                    str(row.get("total_reference_items", 0)),
                    f"{row.get('populated_table_count', 0)}/{row.get('total_table_count', 0)}",
                    ", ".join(row.get("missing_tables") or []) or "n/a",
                    ", ".join(row.get("top_tables") or []) or "n/a",
                ]
            )
            + " |"
        )
    focus = component.get("focus_tables_detail") or []
    if focus:
        lines.extend(["", "## Focus Areas", ""])
        for row in focus:
            lines.append(
                f"- `{row.get('domain')}`: {row.get('action') or 'Backfill missing tables.'}"
            )
    if recommendations:
        lines.extend(["", "## Recommendations", ""])
        for item in recommendations:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"
