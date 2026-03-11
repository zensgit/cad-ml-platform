"""Release-readiness matrix over knowledge-domain benchmark signals."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


DOMAIN_LABELS: Dict[str, str] = {
    "tolerance": "Tolerance & Fits",
    "standards": "Standards & Design Tables",
    "gdt": "GD&T & Datums",
}


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


def _component(payload: Dict[str, Any], key: str) -> Dict[str, Any]:
    return payload.get(key) or payload or {}


def _domain_gate_status(name: str, gate: Dict[str, Any]) -> str:
    if name in (gate.get("blocked_domains") or []):
        return "blocked"
    if name in (gate.get("releasable_domains") or []):
        return "ready"
    if name in (gate.get("partial_domains") or []):
        return "partial"
    if name in (gate.get("priority_domains") or []):
        return "partial"
    return "unknown"


def _alignment_warning(name: str, alignment: Dict[str, Any]) -> bool:
    checks = list(alignment.get("domain_mismatches") or []) + list(
        alignment.get("mismatches") or []
    )
    return any(name in _text(item) for item in checks)


def build_knowledge_domain_release_readiness_matrix(
    *,
    benchmark_knowledge_domain_validation_matrix: Dict[str, Any],
    benchmark_knowledge_domain_release_gate: Dict[str, Any],
    benchmark_knowledge_reference_inventory: Dict[str, Any],
    benchmark_knowledge_domain_release_surface_alignment: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build release-readiness status for standards/tolerance/GD&T domains."""
    validation = _component(
        benchmark_knowledge_domain_validation_matrix,
        "knowledge_domain_validation_matrix",
    )
    gate = _component(
        benchmark_knowledge_domain_release_gate,
        "knowledge_domain_release_gate",
    )
    inventory = _component(
        benchmark_knowledge_reference_inventory,
        "knowledge_reference_inventory",
    )
    alignment = _component(
        benchmark_knowledge_domain_release_surface_alignment or {},
        "knowledge_domain_release_surface_alignment",
    )

    domain_names = sorted(
        set(DOMAIN_LABELS)
        | set((validation.get("domains") or {}).keys())
        | set((inventory.get("domains") or {}).keys())
        | set(gate.get("releasable_domains") or [])
        | set(gate.get("blocked_domains") or [])
        | set(gate.get("partial_domains") or [])
        | set(gate.get("priority_domains") or [])
    )

    domains: Dict[str, Dict[str, Any]] = {}
    focus_areas_detail: List[Dict[str, Any]] = []
    ready_domain_count = 0
    partial_domain_count = 0
    blocked_domain_count = 0
    ready_domains: List[str] = []
    partial_domains: List[str] = []
    blocked_domains: List[str] = []
    releasable_domains: List[str] = []

    for name in domain_names:
        validation_row = (validation.get("domains") or {}).get(name) or {}
        inventory_row = (inventory.get("domains") or {}).get(name) or {}
        validation_status = _text(validation_row.get("status")) or "unknown"
        inventory_status = _text(inventory_row.get("status")) or "unknown"
        gate_status = _domain_gate_status(name, gate)
        alignment_warning = _alignment_warning(name, alignment)

        blocking_reasons: List[str] = []
        warning_reasons: List[str] = []

        if validation_status in {"blocked", "missing"}:
            blocking_reasons.append(f"validation:{validation_status}")
        elif validation_status in {"partial"}:
            warning_reasons.append(f"validation:{validation_status}")

        if inventory_status in {"blocked", "missing"}:
            blocking_reasons.append(f"inventory:{inventory_status}")
        elif inventory_status in {"partial"}:
            warning_reasons.append(f"inventory:{inventory_status}")

        if gate_status == "blocked":
            blocking_reasons.append("release_gate:blocked")
        elif gate_status == "partial":
            warning_reasons.append("release_gate:partial")
        elif gate_status == "unknown":
            warning_reasons.append("release_gate:unknown")

        if alignment_warning:
            warning_reasons.append("release_surface_alignment:mismatch")

        if blocking_reasons:
            status = "blocked"
            priority = "high"
            blocked_domain_count += 1
            blocked_domains.append(name)
        elif (
            validation_status == "ready"
            and inventory_status == "ready"
            and gate_status == "ready"
        ):
            status = "ready"
            priority = "low"
            ready_domain_count += 1
            ready_domains.append(name)
            releasable_domains.append(name)
        else:
            status = "partial"
            priority = "medium"
            partial_domain_count += 1
            partial_domains.append(name)

        action = (
            f"Promote {name} release readiness."
            if status == "ready"
            else (
                f"Unblock {name} release readiness: "
                + ", ".join(_compact(blocking_reasons or warning_reasons, limit=4))
            )
        )
        row = {
            "domain": name,
            "label": validation_row.get("label")
            or inventory_row.get("label")
            or DOMAIN_LABELS.get(name, name),
            "status": status,
            "priority": priority,
            "validation_status": validation_status,
            "inventory_status": inventory_status,
            "release_gate_status": gate_status,
            "alignment_warning": alignment_warning,
            "blocking_reasons": blocking_reasons,
            "warning_reasons": warning_reasons,
            "action": action,
        }
        domains[name] = row
        if status != "ready":
            focus_areas_detail.append(row)

    if not domains:
        status = "knowledge_domain_release_readiness_unavailable"
    elif blocked_domain_count:
        status = "knowledge_domain_release_readiness_blocked"
    elif partial_domain_count:
        status = "knowledge_domain_release_readiness_partial"
    else:
        status = "knowledge_domain_release_readiness_ready"

    summary = (
        f"ready={ready_domain_count}; partial={partial_domain_count}; "
        f"blocked={blocked_domain_count}"
    )

    return {
        "status": status,
        "summary": summary,
        "ready_domain_count": ready_domain_count,
        "partial_domain_count": partial_domain_count,
        "blocked_domain_count": blocked_domain_count,
        "total_domain_count": len(domains),
        "ready_domains": ready_domains,
        "partial_domains": partial_domains,
        "blocked_domains": blocked_domains,
        "releasable_domains": releasable_domains,
        "priority_domains": [row["domain"] for row in focus_areas_detail],
        "focus_areas_detail": focus_areas_detail,
        "domains": domains,
        "gate_open": bool(gate.get("gate_open")),
    }


def knowledge_domain_release_readiness_matrix_recommendations(
    component: Dict[str, Any],
) -> List[str]:
    if _text(component.get("status")) == "knowledge_domain_release_readiness_ready":
        return [
            "Standards, tolerance, and GD&T release-readiness signals are aligned for promotion."
        ]

    items: List[str] = []
    for row in component.get("focus_areas_detail") or []:
        items.append(_text(row.get("action")) or f"Unblock {row.get('domain')}.")
    return _compact(items, limit=10)


def render_knowledge_domain_release_readiness_matrix_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    component = payload.get("knowledge_domain_release_readiness_matrix") or payload or {}
    lines = [
        f"# {title}",
        "",
        f"- `status`: `{component.get('status') or 'unknown'}`",
        f"- `summary`: `{component.get('summary') or 'none'}`",
        f"- `ready_domain_count`: `{component.get('ready_domain_count', 0)}`",
        f"- `partial_domain_count`: `{component.get('partial_domain_count', 0)}`",
        f"- `blocked_domain_count`: `{component.get('blocked_domain_count', 0)}`",
        "",
        "## Domains",
        "",
    ]
    for row in (component.get("domains") or {}).values():
        lines.extend(
            [
                f"### {row.get('label') or row.get('domain')}",
                "",
                f"- `status`: `{row.get('status') or 'unknown'}`",
                f"- `validation_status`: `{row.get('validation_status') or 'unknown'}`",
                f"- `inventory_status`: `{row.get('inventory_status') or 'unknown'}`",
                f"- `release_gate_status`: `{row.get('release_gate_status') or 'unknown'}`",
                f"- `alignment_warning`: `{bool(row.get('alignment_warning'))}`",
                f"- `blocking_reasons`: `{', '.join(row.get('blocking_reasons') or []) or 'none'}`",
                f"- `warning_reasons`: `{', '.join(row.get('warning_reasons') or []) or 'none'}`",
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
