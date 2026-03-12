"""Benchmark control-plane for knowledge subdomain public surfaces."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.core.benchmark.knowledge_readiness import collect_builtin_knowledge_snapshot


ROUTE_PATTERN = re.compile(
    r'@router\.(?:get|post|put|patch|delete)\(\s*"([^"]+)"',
    re.MULTILINE,
)


SUBDOMAIN_SURFACE_SPECS: List[Dict[str, Any]] = [
    {
        "key": "tolerance.it_grades",
        "domain": "tolerance",
        "label": "Tolerance IT grades",
        "reference_names": ["tolerance"],
        "routes": ["/api/v1/tolerance/it"],
        "route_file": "src/api/v1/tolerance.py",
        "module_file": "src/core/knowledge/tolerance/it_grades.py",
        "action": "Keep IT-grade tolerance lookup benchmark-visible and reference-backed.",
    },
    {
        "key": "tolerance.fit_deviations",
        "domain": "tolerance",
        "label": "Tolerance fit deviations",
        "reference_names": ["tolerance"],
        "routes": ["/api/v1/tolerance/fit"],
        "route_file": "src/api/v1/tolerance.py",
        "module_file": "src/core/knowledge/tolerance/fits.py",
        "action": "Keep fit deviation lookup benchmark-visible and reference-backed.",
    },
    {
        "key": "tolerance.limit_deviations",
        "domain": "tolerance",
        "label": "Tolerance limit deviations",
        "reference_names": ["tolerance"],
        "routes": ["/api/v1/tolerance/limit-deviations"],
        "route_file": "src/api/v1/tolerance.py",
        "module_file": "src/core/knowledge/tolerance/fits.py",
        "action": "Keep limit deviation lookup aligned with ISO 286 tables.",
    },
    {
        "key": "standards.threads",
        "domain": "standards",
        "label": "Standards threads",
        "reference_names": ["standards"],
        "routes": ["/api/v1/standards/thread"],
        "route_file": "src/api/v1/standards.py",
        "module_file": "src/core/knowledge/standards/threads.py",
        "action": "Expose thread standards as a stable benchmark-facing lookup surface.",
    },
    {
        "key": "standards.bearings",
        "domain": "standards",
        "label": "Standards bearings",
        "reference_names": ["standards"],
        "routes": ["/api/v1/standards/bearing", "/api/v1/standards/bearing/by-bore"],
        "route_file": "src/api/v1/standards.py",
        "module_file": "src/core/knowledge/standards/bearings.py",
        "action": "Keep bearing standards lookup complete across size and bore surfaces.",
    },
    {
        "key": "standards.seals",
        "domain": "standards",
        "label": "Standards seals",
        "reference_names": ["standards"],
        "routes": ["/api/v1/standards/oring", "/api/v1/standards/oring/by-id"],
        "route_file": "src/api/v1/standards.py",
        "module_file": "src/core/knowledge/standards/seals.py",
        "action": "Keep seal standards lookup complete across section and ID surfaces.",
    },
    {
        "key": "design_standards.surface_finish",
        "domain": "design_standards",
        "label": "Design standards surface finish",
        "reference_names": ["design_standards"],
        "routes": [
            "/api/v1/design-standards/surface-finish/grade",
            "/api/v1/design-standards/surface-finish/grades",
            "/api/v1/design-standards/surface-finish/application",
            "/api/v1/design-standards/surface-finish/suggest",
        ],
        "route_file": "src/api/v1/design_standards.py",
        "module_file": "src/core/knowledge/design_standards/surface_finish.py",
        "action": "Expose surface-finish knowledge as a stable benchmark control surface.",
    },
    {
        "key": "design_standards.general_tolerances",
        "domain": "design_standards",
        "label": "Design standards general tolerances",
        "reference_names": ["design_standards"],
        "routes": [
            "/api/v1/design-standards/general-tolerances/linear",
            "/api/v1/design-standards/general-tolerances/angular",
            "/api/v1/design-standards/general-tolerances/table",
        ],
        "route_file": "src/api/v1/design_standards.py",
        "module_file": "src/core/knowledge/design_standards/general_tolerances.py",
        "action": "Expose general tolerance knowledge as a public benchmark surface.",
    },
    {
        "key": "design_standards.design_features",
        "domain": "design_standards",
        "label": "Design standards design features",
        "reference_names": ["design_standards"],
        "routes": [
            "/api/v1/design-standards/design-features/chamfer",
            "/api/v1/design-standards/design-features/fillet",
        ],
        "route_file": "src/api/v1/design_standards.py",
        "module_file": "src/core/knowledge/design_standards/design_features.py",
        "action": "Keep chamfer and fillet recommendations benchmark-visible.",
    },
    {
        "key": "design_standards.preferred_diameters",
        "domain": "design_standards",
        "label": "Design standards preferred diameters",
        "reference_names": ["design_standards"],
        "routes": [
            "/api/v1/design-standards/preferred-diameter",
            "/api/v1/design-standards/design-features/preferred-diameters",
        ],
        "route_file": "src/api/v1/design_standards.py",
        "module_file": "src/core/knowledge/design_standards/design_features.py",
        "action": "Keep preferred diameter guidance benchmark-visible.",
    },
    {
        "key": "gdt.symbols",
        "domain": "gdt",
        "label": "GD&T symbols",
        "reference_names": ["gdt"],
        "routes": ["/api/v1/gdt/symbols"],
        "route_file": "src/api/v1/gdt.py",
        "module_file": "src/core/knowledge/gdt/symbols.py",
        "action": "Promote GD&T symbol knowledge into a public benchmark-facing API.",
    },
    {
        "key": "gdt.datums",
        "domain": "gdt",
        "label": "GD&T datums",
        "reference_names": ["gdt"],
        "routes": ["/api/v1/gdt/datums"],
        "route_file": "src/api/v1/gdt.py",
        "module_file": "src/core/knowledge/gdt/datums.py",
        "action": "Promote datum guidance into a public benchmark-facing API.",
    },
    {
        "key": "gdt.tolerances",
        "domain": "gdt",
        "label": "GD&T tolerances",
        "reference_names": ["gdt"],
        "routes": ["/api/v1/gdt/tolerances"],
        "route_file": "src/api/v1/gdt.py",
        "module_file": "src/core/knowledge/gdt/tolerances.py",
        "action": "Promote GD&T tolerance knowledge into a public benchmark-facing API.",
    },
    {
        "key": "gdt.application",
        "domain": "gdt",
        "label": "GD&T application",
        "reference_names": ["gdt"],
        "routes": ["/api/v1/gdt/application"],
        "route_file": "src/api/v1/gdt.py",
        "module_file": "src/core/knowledge/gdt/application.py",
        "action": "Promote GD&T application guidance into a public benchmark-facing API.",
    },
]


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


def _priority(status: str) -> str:
    if status == "blocked":
        return "high"
    if status == "partial":
        return "medium"
    return "low"


def _row_status(
    *,
    present_route_count: int,
    expected_route_count: int,
    reference_item_count: int,
) -> str:
    if (
        expected_route_count > 0
        and present_route_count == expected_route_count
        and reference_item_count > 0
    ):
        return "ready"
    if present_route_count > 0 or reference_item_count > 0:
        return "partial"
    return "blocked"


def build_knowledge_subdomain_surface_matrix() -> Dict[str, Any]:
    """Build a fine-grained subdomain matrix for standards/tolerance/GD&T surfaces."""
    snapshot = collect_builtin_knowledge_snapshot()
    subdomains: Dict[str, Dict[str, Any]] = {}
    domains: Dict[str, Dict[str, Any]] = {}
    focus_areas_detail: List[Dict[str, Any]] = []
    public_api_gap_subdomains: List[str] = []
    reference_gap_subdomains: List[str] = []
    ready_count = 0
    partial_count = 0
    blocked_count = 0

    for spec in SUBDOMAIN_SURFACE_SPECS:
        key = str(spec.get("key") or "")
        domain = str(spec.get("domain") or "unknown")
        label = str(spec.get("label") or key)
        expected_routes = list(spec.get("routes") or [])
        present_routes = _collect_routes(str(spec.get("route_file") or ""))
        matching_routes = [
            route
            for route in expected_routes
            if any(_route_matches(route, present) for present in present_routes)
        ]
        missing_routes = [route for route in expected_routes if route not in matching_routes]
        reference_item_count, reference_item_counts = _reference_counts(
            snapshot,
            spec.get("reference_names") or [],
        )
        status = _row_status(
            present_route_count=len(matching_routes),
            expected_route_count=len(expected_routes),
            reference_item_count=reference_item_count,
        )
        if status == "ready":
            ready_count += 1
        elif status == "partial":
            partial_count += 1
        else:
            blocked_count += 1
        module_file = str(spec.get("module_file") or "")
        row = {
            "status": status,
            "priority": _priority(status),
            "domain": domain,
            "label": label,
            "expected_route_count": len(expected_routes),
            "present_route_count": len(matching_routes),
            "missing_route_count": len(missing_routes),
            "expected_routes": expected_routes,
            "present_routes": matching_routes,
            "missing_routes": missing_routes,
            "route_file": str(spec.get("route_file") or ""),
            "module_file": module_file,
            "module_present": Path(module_file).exists() if module_file else False,
            "reference_item_count": reference_item_count,
            "reference_item_counts": dict(reference_item_counts),
            "public_api_gap": bool(missing_routes),
            "reference_gap": reference_item_count <= 0,
            "action": str(spec.get("action") or ""),
        }
        subdomains[key] = row
        domains.setdefault(
            domain,
            {
                "subdomain_count": 0,
                "ready_subdomain_count": 0,
                "partial_subdomain_count": 0,
                "blocked_subdomain_count": 0,
            },
        )
        domains[domain]["subdomain_count"] += 1
        domains[domain][f"{status}_subdomain_count"] += 1
        if missing_routes:
            public_api_gap_subdomains.append(key)
        if reference_item_count <= 0:
            reference_gap_subdomains.append(key)
        if status != "ready":
            focus_areas_detail.append(
                {
                    "domain": domain,
                    "component": key,
                    "status": status,
                    "priority": _priority(status),
                    "missing_metrics": _compact(
                        [
                            "public_api_routes" if missing_routes else "",
                            "reference_depth" if reference_item_count <= 0 else "",
                        ]
                    ),
                    "action": row["action"],
                }
            )

    overall_status = (
        "knowledge_subdomain_surface_matrix_ready"
        if blocked_count == 0 and partial_count == 0
        else (
            "knowledge_subdomain_surface_matrix_blocked"
            if blocked_count and not ready_count and not partial_count
            else "knowledge_subdomain_surface_matrix_partial"
        )
    )
    priority_subdomains = [
        key
        for key, row in subdomains.items()
        if row.get("status") in {"blocked", "partial"}
    ]
    recommendations = knowledge_subdomain_surface_matrix_recommendations(
        {
            "status": overall_status,
            "subdomains": subdomains,
            "public_api_gap_subdomains": public_api_gap_subdomains,
            "reference_gap_subdomains": reference_gap_subdomains,
        }
    )
    return {
        "status": overall_status,
        "total_subdomain_count": len(subdomains),
        "ready_subdomain_count": ready_count,
        "partial_subdomain_count": partial_count,
        "blocked_subdomain_count": blocked_count,
        "subdomains": subdomains,
        "domains": domains,
        "priority_subdomains": priority_subdomains,
        "public_api_gap_subdomains": public_api_gap_subdomains,
        "reference_gap_subdomains": reference_gap_subdomains,
        "focus_areas_detail": focus_areas_detail,
        "recommendations": recommendations,
    }


def knowledge_subdomain_surface_matrix_recommendations(component: Dict[str, Any]) -> List[str]:
    """Derive concise recommendations from the subdomain matrix."""
    component = component or {}
    derived: List[str] = []
    for key, row in (component.get("subdomains") or {}).items():
        status = str(row.get("status") or "")
        if status == "ready":
            continue
        label = row.get("label") or key
        gaps: List[str] = []
        if key in (component.get("public_api_gap_subdomains") or []):
            gaps.append("public API routes")
        if key in (component.get("reference_gap_subdomains") or []):
            gaps.append("reference depth")
        if gaps:
            derived.append(f"Backfill {label} {', '.join(gaps)} before benchmark promotion.")
        else:
            derived.append(f"Stabilize {label} before benchmark promotion.")
    return _compact(derived, limit=12)


def render_knowledge_subdomain_surface_matrix_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    """Render a markdown summary for the subdomain matrix."""
    component = payload.get("knowledge_subdomain_surface_matrix") or payload or {}
    lines = [f"# {title}", ""]
    lines.append(f"- `status`: `{component.get('status') or 'unknown'}`")
    lines.append("")
    lines.append("## Priority Subdomains")
    priority_subdomains = component.get("priority_subdomains") or []
    if priority_subdomains:
        for item in priority_subdomains:
            lines.append(f"- `{item}`")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Recommendations")
    recommendations = component.get("recommendations") or []
    if recommendations:
        for item in recommendations:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    return "\n".join(lines).strip() + "\n"
