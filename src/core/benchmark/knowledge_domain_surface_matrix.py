"""Benchmark control-plane for public knowledge sub-capability surfaces."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.core.benchmark.knowledge_readiness import collect_builtin_knowledge_snapshot


ROUTE_PATTERN = re.compile(
    r'@router\.(?:get|post|put|patch|delete)\(\s*"([^"]+)"',
    re.MULTILINE,
)


DOMAIN_SURFACE_SPECS: Dict[str, Dict[str, Any]] = {
    "tolerance": {
        "label": "Tolerance & Fits",
        "reference_names": ["tolerance"],
        "subcapabilities": [
            {
                "key": "it_tolerance",
                "label": "IT tolerances",
                "routes": ["/api/v1/tolerance/it"],
                "route_file": "src/api/v1/tolerance.py",
            },
            {
                "key": "fit_deviations",
                "label": "Fit deviations",
                "routes": ["/api/v1/tolerance/fit"],
                "route_file": "src/api/v1/tolerance.py",
            },
            {
                "key": "limit_deviations",
                "label": "Limit deviations",
                "routes": ["/api/v1/tolerance/limit-deviations"],
                "route_file": "src/api/v1/tolerance.py",
            },
        ],
        "action": (
            "Keep ISO 286 / GB-T 1800 tolerance endpoints benchmark-visible and "
            "backed by reference data."
        ),
    },
    "standards": {
        "label": "Standards & Design Standards",
        "reference_names": ["standards", "design_standards"],
        "subcapabilities": [
            {
                "key": "thread_lookup",
                "label": "Thread lookup",
                "routes": ["/api/v1/standards/thread"],
                "route_file": "src/api/v1/standards.py",
            },
            {
                "key": "bearing_lookup",
                "label": "Bearing lookup",
                "routes": [
                    "/api/v1/standards/bearing",
                    "/api/v1/standards/bearing/by-bore",
                ],
                "route_file": "src/api/v1/standards.py",
            },
            {
                "key": "oring_lookup",
                "label": "O-ring lookup",
                "routes": [
                    "/api/v1/standards/oring",
                    "/api/v1/standards/oring/by-id",
                ],
                "route_file": "src/api/v1/standards.py",
            },
            {
                "key": "surface_finish",
                "label": "Surface finish",
                "routes": [
                    "/api/v1/design-standards/surface-finish/grade",
                    "/api/v1/design-standards/surface-finish/grades",
                    "/api/v1/design-standards/surface-finish/application",
                    "/api/v1/design-standards/surface-finish/suggest",
                ],
                "route_file": "src/api/v1/design_standards.py",
            },
            {
                "key": "general_tolerances",
                "label": "General tolerances",
                "routes": [
                    "/api/v1/design-standards/general-tolerances/linear",
                    "/api/v1/design-standards/general-tolerances/angular",
                    "/api/v1/design-standards/general-tolerances/table",
                ],
                "route_file": "src/api/v1/design_standards.py",
            },
            {
                "key": "preferred_diameters",
                "label": "Preferred diameters",
                "routes": [
                    "/api/v1/design-standards/preferred-diameter",
                    "/api/v1/design-standards/design-features/preferred-diameters",
                ],
                "route_file": "src/api/v1/design_standards.py",
            },
            {
                "key": "design_features",
                "label": "Design features",
                "routes": [
                    "/api/v1/design-standards/design-features/chamfer",
                    "/api/v1/design-standards/design-features/fillet",
                ],
                "route_file": "src/api/v1/design_standards.py",
            },
        ],
        "action": (
            "Expose standards and design-standard lookups as stable public benchmark "
            "surfaces instead of isolated route groups."
        ),
    },
    "gdt": {
        "label": "GD&T & Datums",
        "reference_names": ["gdt"],
        "subcapabilities": [
            {
                "key": "gdt_public_api",
                "label": "GD&T public API",
                "routes": ["/api/v1/gdt"],
                "route_file": "src/api/v1/gdt.py",
            }
        ],
        "action": (
            "Promote GD&T from reference-only knowledge into public benchmark-facing "
            "API surfaces."
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
        subtotal = sum(_to_int(item) for item in rows.values())
        detail[str(name)] = subtotal
        total += subtotal
    return total, detail


def _collect_routes(route_file: str) -> List[str]:
    path = Path(route_file)
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    return [match.group(1).strip() for match in ROUTE_PATTERN.finditer(content)]


def _route_matches(expected_route: str, present_route: str) -> bool:
    expected = _text(expected_route)
    present = _text(present_route)
    if not expected or not present:
        return False
    return expected == present or expected.endswith(present)


def _subcapability_status(
    *,
    present_route_count: int,
    expected_route_count: int,
    reference_item_count: int,
) -> str:
    if present_route_count == expected_route_count and reference_item_count > 0:
        return "ready"
    if present_route_count > 0 or reference_item_count > 0:
        return "partial"
    return "blocked"


def _domain_status(rows: List[Dict[str, Any]]) -> str:
    statuses = {str(row.get("status") or "unknown") for row in rows}
    if statuses == {"ready"}:
        return "ready"
    if statuses <= {"blocked"}:
        return "blocked"
    return "partial"


def _priority(status: str) -> str:
    if status == "blocked":
        return "high"
    if status == "partial":
        return "medium"
    return "low"


def build_knowledge_domain_surface_matrix() -> Dict[str, Any]:
    """Build a fine-grained public knowledge surface matrix."""
    snapshot = collect_builtin_knowledge_snapshot()
    domains: Dict[str, Dict[str, Any]] = {}
    focus_areas_detail: List[Dict[str, Any]] = []
    public_surface_gap_domains: List[str] = []
    reference_gap_domains: List[str] = []
    ready_count = 0
    partial_count = 0
    blocked_count = 0

    for domain_name, spec in DOMAIN_SURFACE_SPECS.items():
        reference_item_count, reference_item_counts = _reference_counts(
            snapshot,
            spec.get("reference_names") or [],
        )
        rows: Dict[str, Dict[str, Any]] = {}
        for subcapability in spec.get("subcapabilities") or []:
            expected_routes = list(subcapability.get("routes") or [])
            present_routes = _collect_routes(str(subcapability.get("route_file") or ""))
            matching_routes = [
                route
                for route in expected_routes
                if any(_route_matches(route, present) for present in present_routes)
            ]
            missing_routes = [route for route in expected_routes if route not in matching_routes]
            status = _subcapability_status(
                present_route_count=len(matching_routes),
                expected_route_count=len(expected_routes),
                reference_item_count=reference_item_count,
            )
            row = {
                "status": status,
                "label": subcapability.get("label") or subcapability.get("key") or "",
                "expected_route_count": len(expected_routes),
                "present_route_count": len(matching_routes),
                "missing_route_count": len(missing_routes),
                "expected_routes": expected_routes,
                "present_routes": matching_routes,
                "missing_routes": missing_routes,
                "route_file": subcapability.get("route_file") or "",
                "reference_item_count": reference_item_count,
                "reference_item_counts": dict(reference_item_counts),
                "action": (
                    f"Expose {subcapability.get('label') or subcapability.get('key')} "
                    "through stable public knowledge routes."
                    if missing_routes
                    else (
                        "Backfill reference depth for "
                        f"{subcapability.get('label') or subcapability.get('key')}."
                        if reference_item_count <= 0
                        else "Keep this public knowledge surface aligned."
                    )
                ),
            }
            rows[str(subcapability.get("key") or "")] = row
            if status != "ready":
                focus_areas_detail.append(
                    {
                        "domain": domain_name,
                        "component": str(subcapability.get("key") or ""),
                        "status": status,
                        "priority": _priority(status),
                        "missing_metrics": _compact(
                            [
                                "public_routes" if missing_routes else "",
                                "reference_depth" if reference_item_count <= 0 else "",
                            ]
                        ),
                        "action": row["action"],
                    }
                )

        domain_rows = list(rows.values())
        status = _domain_status(domain_rows)
        if status == "ready":
            ready_count += 1
        elif status == "partial":
            partial_count += 1
        else:
            blocked_count += 1
        if any(row["missing_route_count"] > 0 for row in domain_rows):
            public_surface_gap_domains.append(domain_name)
        if reference_item_count <= 0:
            reference_gap_domains.append(domain_name)
        domains[domain_name] = {
            "status": status,
            "label": spec.get("label") or domain_name,
            "priority": _priority(status),
            "reference_item_count": reference_item_count,
            "reference_item_counts": reference_item_counts,
            "subcapability_count": len(domain_rows),
            "ready_subcapability_count": sum(
                1 for row in domain_rows if row.get("status") == "ready"
            ),
            "partial_subcapability_count": sum(
                1 for row in domain_rows if row.get("status") == "partial"
            ),
            "blocked_subcapability_count": sum(
                1 for row in domain_rows if row.get("status") == "blocked"
            ),
            "subcapabilities": rows,
            "action": spec.get("action") or "",
        }

    overall_status = (
        "knowledge_domain_surface_matrix_ready"
        if blocked_count == 0 and partial_count == 0
        else (
            "knowledge_domain_surface_matrix_blocked"
            if blocked_count and not ready_count and not partial_count
            else "knowledge_domain_surface_matrix_partial"
        )
    )
    priority_domains = [
        name
        for name, row in domains.items()
        if row.get("status") in {"blocked", "partial"}
    ]
    recommendations = knowledge_domain_surface_matrix_recommendations(
        {
            "status": overall_status,
            "domains": domains,
            "priority_domains": priority_domains,
            "focus_areas_detail": focus_areas_detail,
            "public_surface_gap_domains": public_surface_gap_domains,
            "reference_gap_domains": reference_gap_domains,
        }
    )
    return {
        "status": overall_status,
        "ready_domain_count": ready_count,
        "partial_domain_count": partial_count,
        "blocked_domain_count": blocked_count,
        "total_domain_count": len(domains),
        "domains": domains,
        "priority_domains": priority_domains,
        "focus_areas_detail": focus_areas_detail,
        "public_surface_gap_domains": public_surface_gap_domains,
        "reference_gap_domains": reference_gap_domains,
        "recommendations": recommendations,
    }


def knowledge_domain_surface_matrix_recommendations(
    component: Dict[str, Any]
) -> List[str]:
    """Derive concise recommendations from the surface matrix."""
    component = component or {}
    recommendations = _compact(component.get("recommendations") or [], limit=12)
    if recommendations:
        return recommendations

    derived: List[str] = []
    domains = component.get("domains") or {}
    for name, row in domains.items():
        if row.get("status") == "ready":
            continue
        gaps: List[str] = []
        if name in (component.get("public_surface_gap_domains") or []):
            gaps.append("public API routes")
        if name in (component.get("reference_gap_domains") or []):
            gaps.append("reference depth")
        target = row.get("label") or name
        if gaps:
            derived.append(f"Backfill {target} {', '.join(gaps)} before benchmark promotion.")
        else:
            derived.append(
                f"Stabilize {target} sub-capability surfaces before benchmark promotion."
            )
    return _compact(derived, limit=12)


def render_knowledge_domain_surface_matrix_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    """Render a markdown summary for the surface matrix."""
    component = payload.get("knowledge_domain_surface_matrix") or payload or {}
    lines = [f"# {title}", ""]
    lines.append(f"- `status`: `{component.get('status') or 'unknown'}`")
    lines.append("")
    lines.append("## Domains")
    domains = component.get("domains") or {}
    if domains:
        for name, row in domains.items():
            lines.append(
                "- "
                f"`{name}` "
                f"status=`{row.get('status')}` "
                f"reference_items=`{row.get('reference_item_count')}` "
                f"ready_subcapabilities=`{row.get('ready_subcapability_count')}`"
            )
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Recommendations")
    recommendations = payload.get("recommendations") or component.get("recommendations") or []
    if recommendations:
        for item in recommendations:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"
