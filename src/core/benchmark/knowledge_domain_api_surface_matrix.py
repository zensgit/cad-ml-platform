"""Benchmark control-plane for public API coverage of knowledge domains."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.core.benchmark.knowledge_readiness import collect_builtin_knowledge_snapshot


DOMAIN_SPECS: Dict[str, Dict[str, Any]] = {
    "tolerance": {
        "label": "Tolerance & Fits",
        "components": ["tolerance"],
        "files": [
            {
                "path": "src/api/v1/tolerance.py",
                "prefix": "/api/v1/tolerance",
            }
        ],
        "action": (
            "Keep tolerance/fits endpoints stable and benchmark-visible so ISO 286 and "
            "GB/T 1800 coverage is directly observable."
        ),
    },
    "standards": {
        "label": "Standards & Design Tables",
        "components": ["standards", "design_standards"],
        "files": [
            {
                "path": "src/api/v1/standards.py",
                "prefix": "/api/v1/standards",
            },
            {
                "path": "src/api/v1/design_standards.py",
                "prefix": "/api/v1/design-standards",
            },
        ],
        "action": (
            "Expose standards and design-standards knowledge as first-class public API "
            "surfaces, not only internal lookup tables."
        ),
    },
    "gdt": {
        "label": "GD&T & Datums",
        "components": ["gdt"],
        "files": [
            {
                "path": "src/api/v1/gdt.py",
                "prefix": "/api/v1/gdt",
            }
        ],
        "action": (
            "Add a dedicated GD&T API surface so symbols, datums, and tolerance guidance "
            "become externally benchmarkable."
        ),
    },
}

ROUTE_RE = re.compile(
    r'@router\.(get|post|put|delete|patch)\(\s*["\']([^"\']+)["\']',
    re.MULTILINE,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _join_api_path(prefix: str, suffix: str) -> str:
    prefix_text = "/" + _text(prefix).strip("/")
    suffix_text = "/" + _text(suffix).lstrip("/")
    if suffix_text == "/":
        return prefix_text
    return prefix_text + suffix_text


def _scan_route_file(path_text: str, prefix: str) -> Tuple[bool, List[Dict[str, str]]]:
    source_path = _repo_root() / path_text
    if not source_path.exists():
        return False, []
    source = source_path.read_text(encoding="utf-8")
    routes: List[Dict[str, str]] = []
    for method, suffix in ROUTE_RE.findall(source):
        routes.append(
            {
                "method": method.upper(),
                "path": _join_api_path(prefix, suffix),
            }
        )
    return True, routes


def _reference_counts(components: Iterable[str]) -> Tuple[int, Dict[str, int]]:
    snapshot = collect_builtin_knowledge_snapshot()
    detail: Dict[str, int] = {}
    total = 0
    for component in components:
        counts = snapshot.get(component) or {}
        subtotal = sum(_to_int(value) for value in counts.values())
        detail[str(component)] = subtotal
        total += subtotal
    return total, detail


def _domain_status(
    *,
    api_route_count: int,
    reference_item_count: int,
    capability_status: str,
) -> str:
    if api_route_count > 0 and reference_item_count > 0:
        if capability_status in {"blocked", "missing"}:
            return "partial"
        return "ready"
    if api_route_count == 0 and reference_item_count > 0:
        return "blocked"
    if api_route_count > 0 or reference_item_count > 0 or capability_status in {
        "ready",
        "partial",
    }:
        return "partial"
    return "missing"


def _overall_status(domain_statuses: Iterable[str]) -> str:
    statuses = [str(item or "missing") for item in domain_statuses]
    if statuses and all(item == "ready" for item in statuses):
        return "knowledge_domain_api_surface_matrix_ready"
    if statuses and all(item in {"missing", "blocked"} for item in statuses):
        return "knowledge_domain_api_surface_matrix_blocked"
    return "knowledge_domain_api_surface_matrix_partial"


def build_knowledge_domain_api_surface_matrix(
    *,
    knowledge_domain_capability_matrix_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build benchmark visibility over public API surfaces for core knowledge domains."""
    capability_root = (
        (knowledge_domain_capability_matrix_summary or {}).get(
            "knowledge_domain_capability_matrix"
        )
        or knowledge_domain_capability_matrix_summary
        or {}
    )
    capability_domains = capability_root.get("domains") or {}

    domains: Dict[str, Dict[str, Any]] = {}
    focus_areas_detail: List[Dict[str, Any]] = []
    public_api_gap_domains: List[str] = []
    reference_gap_domains: List[str] = []
    ready_count = 0
    partial_count = 0
    blocked_count = 0
    total_api_route_count = 0

    for name, spec in DOMAIN_SPECS.items():
        file_specs = list(spec.get("files") or [])
        route_files: List[Dict[str, Any]] = []
        api_routes: List[Dict[str, str]] = []
        missing_files: List[str] = []
        for file_spec in file_specs:
            exists, file_routes = _scan_route_file(
                str(file_spec.get("path") or ""),
                str(file_spec.get("prefix") or ""),
            )
            route_files.append(
                {
                    "path": str(file_spec.get("path") or ""),
                    "prefix": str(file_spec.get("prefix") or ""),
                    "exists": exists,
                    "route_count": len(file_routes),
                }
            )
            if exists:
                api_routes.extend(file_routes)
            else:
                missing_files.append(str(file_spec.get("path") or ""))

        reference_item_count, reference_item_counts = _reference_counts(
            spec.get("components") or []
        )
        capability_row = capability_domains.get(name) or {}
        capability_status = _text(capability_row.get("status")).lower() or "unknown"
        api_route_count = len(api_routes)
        api_surface_status = "ready" if api_route_count > 0 else "missing"
        reference_status = "ready" if reference_item_count > 0 else "missing"
        domain_status = _domain_status(
            api_route_count=api_route_count,
            reference_item_count=reference_item_count,
            capability_status=capability_status,
        )
        if domain_status == "ready":
            ready_count += 1
        elif domain_status == "blocked":
            blocked_count += 1
        else:
            partial_count += 1
        total_api_route_count += api_route_count
        if api_surface_status != "ready":
            public_api_gap_domains.append(name)
        if reference_status != "ready":
            reference_gap_domains.append(name)

        domain_row = {
            "domain": name,
            "label": str(spec.get("label") or name),
            "status": domain_status,
            "priority": (
                "high"
                if domain_status in {"blocked", "missing"}
                else "medium" if domain_status == "partial" else "low"
            ),
            "capability_status": capability_status,
            "api_surface_status": api_surface_status,
            "reference_status": reference_status,
            "reference_item_count": reference_item_count,
            "reference_item_counts": reference_item_counts,
            "api_route_count": api_route_count,
            "api_routes": api_routes,
            "api_route_examples": [row["path"] for row in api_routes[:6]],
            "route_files": route_files,
            "missing_route_files": missing_files,
            "action": str(spec.get("action") or ""),
        }
        domains[name] = domain_row

        if domain_status != "ready":
            focus_areas_detail.append(
                {
                    "domain": name,
                    "label": domain_row["label"],
                    "status": domain_status,
                    "priority": domain_row["priority"],
                    "api_route_count": api_route_count,
                    "reference_item_count": reference_item_count,
                    "capability_status": capability_status,
                    "missing_route_files": missing_files,
                    "action": domain_row["action"],
                }
            )

    priority_domains = [
        row["domain"]
        for row in sorted(
            focus_areas_detail,
            key=lambda item: (
                {"high": 0, "medium": 1, "low": 2}.get(
                    str(item.get("priority") or "medium"),
                    1,
                ),
                str(item.get("domain") or ""),
            ),
        )
    ]

    return {
        "status": _overall_status(domains[name]["status"] for name in DOMAIN_SPECS),
        "ready_domain_count": ready_count,
        "partial_domain_count": partial_count,
        "blocked_domain_count": blocked_count,
        "total_domain_count": len(DOMAIN_SPECS),
        "total_api_route_count": total_api_route_count,
        "domains": domains,
        "priority_domains": priority_domains,
        "public_api_gap_domains": public_api_gap_domains,
        "reference_gap_domains": reference_gap_domains,
        "focus_areas_detail": focus_areas_detail,
    }


def knowledge_domain_api_surface_matrix_recommendations(
    component: Dict[str, Any],
) -> List[str]:
    items: List[str] = []
    for domain in component.get("focus_areas_detail") or []:
        label = _text(domain.get("label") or domain.get("domain") or "domain")
        status = _text(domain.get("status") or "unknown")
        if _to_int(domain.get("api_route_count")) <= 0:
            items.append(
                f"Expose a stable public API surface for {label} before treating it as "
                f"release-ready benchmark knowledge ({status})."
            )
        else:
            items.append(
                f"Keep {label} API routes aligned with benchmark knowledge artifacts and "
                f"close the remaining readiness gaps ({status})."
            )
    return _compact(items, limit=8)


def render_knowledge_domain_api_surface_matrix_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    component = (
        payload.get("knowledge_domain_api_surface_matrix")
        or payload
        or {}
    )
    lines = [f"# {title}", ""]
    lines.append(f"- `status`: `{component.get('status') or 'unknown'}`")
    lines.append(
        "- `counts`: "
        f"ready=`{component.get('ready_domain_count') or 0}` "
        f"partial=`{component.get('partial_domain_count') or 0}` "
        f"blocked=`{component.get('blocked_domain_count') or 0}` "
        f"routes=`{component.get('total_api_route_count') or 0}`"
    )
    lines.append(
        "- `priority_domains`: "
        + (", ".join(component.get("priority_domains") or []) or "none")
    )
    lines.extend(["", "## Domains", ""])
    domains = component.get("domains") or {}
    if domains:
        for name, row in domains.items():
            lines.append(
                "- "
                f"`{name}` "
                f"status=`{row.get('status')}` "
                f"api=`{row.get('api_surface_status')}` "
                f"capability=`{row.get('capability_status')}` "
                f"reference_items=`{row.get('reference_item_count')}` "
                f"routes=`{row.get('api_route_count')}`"
            )
    else:
        lines.append("- none")
    recommendations = payload.get("recommendations") or []
    if recommendations:
        lines.extend(["", "## Recommendations", ""])
        for item in recommendations:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"
