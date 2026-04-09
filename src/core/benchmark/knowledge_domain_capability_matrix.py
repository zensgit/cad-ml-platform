"""Benchmark control-plane for standards/tolerance/GD&T domain capabilities."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from src.core.benchmark.knowledge_readiness import (
    DOMAIN_COMPONENTS,
    collect_builtin_knowledge_snapshot,
)


DOMAIN_SPECS: Dict[str, Dict[str, Any]] = {
    "tolerance": {
        "label": "Tolerance & Fits",
        "components": ["tolerance"],
        "provider_names": ["tolerance"],
        "public_surfaces": [
            "/api/v1/tolerance/it",
            "/api/v1/tolerance/fit",
            "/api/v1/tolerance/limit-deviations",
        ],
        "assistant_surfaces": ["tolerance"],
        "action": (
            "Keep ISO 286 / GB-T 1800 tolerance and fit surfaces aligned across provider, "
            "API, assistant retrieval, and benchmark release artifacts."
        ),
    },
    "standards": {
        "label": "Standards & Design Tables",
        "components": ["standards", "design_standards"],
        "provider_names": ["standards", "design_standards"],
        "public_surfaces": [
            "/api/v1/standards/thread",
            "/api/v1/standards/bearing",
            "/api/v1/standards/oring",
            "/api/v1/design-standards/general-tolerances/linear",
            "/api/v1/design-standards/general-tolerances/angular",
            "/api/v1/design-standards/design-features/preferred-diameters",
        ],
        "assistant_surfaces": ["standards", "design_standards"],
        "action": (
            "Expose standard-part and design-standard knowledge as a stable benchmark "
            "domain, not only as raw lookup tables."
        ),
    },
    "gdt": {
        "label": "GD&T & Datums",
        "components": ["gdt"],
        "provider_names": ["gdt"],
        "public_surfaces": [],
        "assistant_surfaces": ["gdt"],
        "action": (
            "Promote GD&T from assistant-only knowledge into a benchmark-visible domain "
            "with provider and release-surface coverage."
        ),
    },
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


def _status(value: Any, default: str = "missing") -> str:
    text = _text(value).lower()
    return text or default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _reference_counts(snapshot: Dict[str, Any], names: Iterable[str]) -> Tuple[int, Dict[str, int]]:
    total = 0
    detail: Dict[str, int] = {}
    for name in names:
        rows = snapshot.get(name) or {}
        subtotal = sum(_to_int(value) for value in rows.values())
        detail[str(name)] = subtotal
        total += subtotal
    return total, detail


def _provider_status(names: Iterable[str]) -> Tuple[str, List[str], List[str]]:
    from src.core.providers.knowledge import bootstrap_core_knowledge_providers
    from src.core.providers.registry import ProviderRegistry

    bootstrap_core_knowledge_providers()
    required = [_text(name) for name in names if _text(name)]
    available = [
        name for name in required if ProviderRegistry.exists("knowledge", name)
    ]
    missing = [name for name in required if name not in available]
    if available and not missing:
        return "ready", available, missing
    if available:
        return "partial", available, missing
    return "missing", available, missing


def _surface_status(public_surfaces: List[str], assistant_surfaces: List[str]) -> str:
    if public_surfaces and assistant_surfaces:
        return "ready"
    if public_surfaces or assistant_surfaces:
        return "partial"
    return "missing"


def _domain_status(
    *,
    foundation_status: str,
    application_status: str,
    matrix_status: str,
    provider_status: str,
    surface_status: str,
) -> str:
    if all(
        status == "ready"
        for status in (
            foundation_status,
            application_status,
            matrix_status,
            provider_status,
            surface_status,
        )
    ):
        return "ready"
    if (
        foundation_status == "missing"
        and application_status == "missing"
        and matrix_status in {"missing", "blocked"}
        and provider_status == "missing"
        and surface_status == "missing"
    ):
        return "missing"
    if provider_status == "missing" or surface_status == "missing" or matrix_status == "blocked":
        return "blocked"
    return "partial"


def _priority(status: str) -> str:
    if status in {"missing", "blocked"}:
        return "high"
    if status == "partial":
        return "medium"
    return "low"


def _primary_gaps(
    *,
    foundation_status: str,
    application_status: str,
    matrix_status: str,
    provider_status: str,
    surface_status: str,
    missing_providers: List[str],
    public_surfaces: List[str],
) -> List[str]:
    gaps: List[str] = []
    if foundation_status != "ready":
        gaps.append("foundation_gap")
    if application_status != "ready":
        gaps.append("application_gap")
    if matrix_status in {"missing", "blocked", "partial"}:
        gaps.append(f"matrix_{matrix_status}")
    if provider_status != "ready":
        gaps.append(
            f"provider_missing:{','.join(missing_providers) or 'unknown'}"
        )
    if surface_status == "missing":
        gaps.append("surface_missing")
    elif surface_status == "partial" and not public_surfaces:
        gaps.append("public_surface_missing")
    return gaps


def build_knowledge_domain_capability_matrix(
    *,
    knowledge_readiness_summary: Dict[str, Any],
    knowledge_application_summary: Dict[str, Any],
    knowledge_domain_matrix_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a benchmark control-plane over tolerance/standards/GD&T capabilities."""
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
    matrix_root = (
        knowledge_domain_matrix_summary.get("knowledge_domain_matrix")
        or knowledge_domain_matrix_summary
        or {}
    )
    snapshot = collect_builtin_knowledge_snapshot()
    readiness_domains = readiness_root.get("domains") or {}
    application_domains = application_root.get("domains") or {}
    matrix_domains = matrix_root.get("domains") or {}

    domains: Dict[str, Dict[str, Any]] = {}
    focus_areas_detail: List[Dict[str, Any]] = []
    ready_count = 0
    partial_count = 0
    blocked_count = 0

    for name, spec in DOMAIN_SPECS.items():
        readiness_row = readiness_domains.get(name) or {}
        application_row = application_domains.get(name) or {}
        matrix_row = matrix_domains.get(name) or {}
        foundation_status = _status(readiness_row.get("status"))
        application_status = _status(application_row.get("status"))
        matrix_status = _status(matrix_row.get("status"), default="blocked")
        provider_status, available_providers, missing_providers = _provider_status(
            spec.get("provider_names") or []
        )
        public_surfaces = list(spec.get("public_surfaces") or [])
        assistant_surfaces = list(spec.get("assistant_surfaces") or [])
        surface_status = _surface_status(public_surfaces, assistant_surfaces)
        reference_item_count, reference_item_counts = _reference_counts(
            snapshot,
            spec.get("components") or [],
        )
        status = _domain_status(
            foundation_status=foundation_status,
            application_status=application_status,
            matrix_status=matrix_status,
            provider_status=provider_status,
            surface_status=surface_status,
        )
        row = {
            "domain": name,
            "label": _text(spec.get("label")) or name,
            "status": status,
            "priority": _priority(status),
            "foundation_status": foundation_status,
            "application_status": application_status,
            "matrix_status": matrix_status,
            "provider_status": provider_status,
            "surface_status": surface_status,
            "provider_names": list(spec.get("provider_names") or []),
            "available_providers": available_providers,
            "missing_providers": missing_providers,
            "public_surface_count": len(public_surfaces),
            "assistant_surface_count": len(assistant_surfaces),
            "public_surfaces": public_surfaces,
            "assistant_surfaces": assistant_surfaces,
            "reference_item_count": reference_item_count,
            "reference_item_counts": reference_item_counts,
            "primary_gaps": _primary_gaps(
                foundation_status=foundation_status,
                application_status=application_status,
                matrix_status=matrix_status,
                provider_status=provider_status,
                surface_status=surface_status,
                missing_providers=missing_providers,
                public_surfaces=public_surfaces,
            ),
            "action": _text(matrix_row.get("action"))
            or _text(readiness_row.get("action"))
            or _text(application_row.get("action"))
            or _text(spec.get("action")),
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

    if ready_count == len(DOMAIN_SPECS):
        overall_status = "knowledge_domain_capability_ready"
    elif ready_count > 0 or partial_count > 0:
        overall_status = "knowledge_domain_capability_partial"
    else:
        overall_status = "knowledge_domain_capability_missing"

    return {
        "status": overall_status,
        "ready_domain_count": ready_count,
        "partial_domain_count": partial_count,
        "blocked_domain_count": blocked_count,
        "total_domain_count": len(DOMAIN_SPECS),
        "priority_domains": [
            row["domain"] for row in focus_areas_detail if row.get("priority") == "high"
        ],
        "provider_gap_domains": [
            row["domain"]
            for row in focus_areas_detail
            if row.get("provider_status") != "ready"
        ],
        "surface_gap_domains": [
            row["domain"]
            for row in focus_areas_detail
            if row.get("surface_status") != "ready"
        ],
        "focus_areas": [row["domain"] for row in focus_areas_detail],
        "focus_areas_detail": focus_areas_detail,
        "domains": domains,
        "matrix": domains,
    }


def knowledge_domain_capability_matrix_recommendations(
    component: Dict[str, Any]
) -> List[str]:
    status = _status(component.get("status"))
    if status == "knowledge_domain_capability_ready":
        return [
            "Tolerance, standards/design-tables, and GD&T are benchmark-visible across "
            "provider, surface, and downstream release artifacts."
        ]

    items: List[str] = []
    for row in component.get("focus_areas_detail") or []:
        label = _text(row.get("label")) or _text(row.get("domain")) or "unknown"
        if row.get("provider_status") != "ready":
            missing = ", ".join(row.get("missing_providers") or []) or "provider"
            items.append(f"Backfill {label} provider coverage: {missing}")
        if row.get("surface_status") == "missing":
            items.append(f"Expose benchmark-facing surfaces for {label}")
        elif row.get("surface_status") == "partial" and not row.get("public_surfaces"):
            items.append(f"Add a public benchmark/API surface for {label}")
        if row.get("foundation_status") != "ready":
            items.append(f"Backfill {label} foundation metrics")
        if row.get("application_status") != "ready":
            items.append(f"Raise {label} application evidence in benchmark outputs")
    return _compact(items, limit=12)


def render_knowledge_domain_capability_matrix_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    component = payload.get("knowledge_domain_capability_matrix") or {}
    lines = [
        f"# {title}",
        "",
        "## Overview",
        "",
        f"- `status`: `{component.get('status', 'knowledge_domain_capability_missing')}`",
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
    if domains:
        for name, row in domains.items():
            lines.append(
                "- "
                f"`{name}` "
                f"status=`{row.get('status')}` "
                f"foundation=`{row.get('foundation_status')}` "
                f"application=`{row.get('application_status')}` "
                f"provider=`{row.get('provider_status')}` "
                f"surface=`{row.get('surface_status')}` "
                f"reference_items=`{row.get('reference_item_count')}`"
            )
            lines.append(
                "  "
                f"primary_gaps={', '.join(row.get('primary_gaps') or []) or 'none'} "
                f"action={row.get('action') or 'none'}"
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
