"""Benchmark helpers for standards/tolerance/GD&T validation layers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]

DOMAIN_SPECS: Dict[str, Dict[str, Any]] = {
    "tolerance": {
        "label": "Tolerance & Fits",
        "provider_names": ["tolerance"],
        "public_surfaces": [
            "/api/v1/tolerance/it",
            "/api/v1/tolerance/fit",
            "/api/v1/tolerance/limit-deviations",
        ],
        "unit_tests": [
            "tests/unit/knowledge/test_tolerance.py",
            "tests/unit/test_tolerance_api_normalization.py",
            "tests/unit/test_tolerance_limit_deviations.py",
            "tests/unit/test_tolerance_fundamental_deviation.py",
            "tests/test_tolerance_fits.py",
        ],
        "integration_tests": [
            "tests/integration/test_tolerance_api.py",
            "tests/integration/test_tolerance_api_errors.py",
        ],
        "assistant_tests": [],
    },
    "standards": {
        "label": "Standards & Design Tables",
        "provider_names": ["standards", "design_standards"],
        "public_surfaces": [
            "/api/v1/standards/thread",
            "/api/v1/standards/bearing",
            "/api/v1/standards/oring",
            "/api/v1/design-standards/general-tolerances/linear",
            "/api/v1/design-standards/general-tolerances/angular",
            "/api/v1/design-standards/design-features/preferred-diameters",
        ],
        "unit_tests": [
            "tests/unit/knowledge/test_standards.py",
            "tests/unit/knowledge/test_design_standards.py",
        ],
        "integration_tests": [
            "tests/integration/test_standards_api.py",
            "tests/integration/test_design_standards_api.py",
        ],
        "assistant_tests": [],
    },
    "gdt": {
        "label": "GD&T & Datums",
        "provider_names": ["gdt"],
        "public_surfaces": [],
        "unit_tests": [
            "tests/unit/knowledge/test_gdt.py",
            "tests/test_gdt_application.py",
        ],
        "integration_tests": [],
        "assistant_tests": [
            "tests/unit/assistant/test_gdt_retrieval.py",
        ],
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


def _existing_paths(paths: Iterable[str]) -> List[str]:
    existing: List[str] = []
    for rel in paths:
        path = REPO_ROOT / rel
        if path.exists():
            existing.append(rel)
    return existing


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


def _domain_status(
    *,
    provider_status: str,
    public_surface_count: int,
    unit_test_count: int,
    integration_test_count: int,
) -> str:
    if (
        provider_status == "missing"
        and public_surface_count == 0
        and unit_test_count == 0
        and integration_test_count == 0
    ):
        return "missing"
    if (
        provider_status == "ready"
        and public_surface_count > 0
        and unit_test_count > 0
        and integration_test_count > 0
    ):
        return "ready"
    if provider_status == "missing" or public_surface_count == 0 or integration_test_count == 0:
        return "blocked"
    return "partial"


def _priority(status: str) -> str:
    if status in {"blocked", "missing"}:
        return "high"
    if status == "partial":
        return "medium"
    return "low"


def _summary(
    *,
    ready_domain_count: int,
    partial_domain_count: int,
    blocked_domain_count: int,
    total_test_count: int,
) -> str:
    return (
        f"ready={ready_domain_count}; partial={partial_domain_count}; "
        f"blocked={blocked_domain_count}; total_tests={total_test_count}"
    )


def build_knowledge_domain_validation_matrix() -> Dict[str, Any]:
    """Build a benchmark validation matrix for standards/tolerance/GD&T."""
    domains: Dict[str, Dict[str, Any]] = {}
    focus_areas_detail: List[Dict[str, Any]] = []
    ready_domain_count = 0
    partial_domain_count = 0
    blocked_domain_count = 0
    total_test_count = 0
    api_covered_domain_count = 0
    provider_ready_domain_count = 0

    for domain, spec in DOMAIN_SPECS.items():
        provider_status, available_providers, missing_providers = _provider_status(
            spec.get("provider_names") or []
        )
        public_surfaces = list(spec.get("public_surfaces") or [])
        unit_tests = _existing_paths(spec.get("unit_tests") or [])
        integration_tests = _existing_paths(spec.get("integration_tests") or [])
        assistant_tests = _existing_paths(spec.get("assistant_tests") or [])

        public_surface_count = len(public_surfaces)
        unit_test_count = len(unit_tests)
        integration_test_count = len(integration_tests)
        assistant_test_count = len(assistant_tests)
        domain_total_test_count = (
            unit_test_count + integration_test_count + assistant_test_count
        )
        total_test_count += domain_total_test_count
        if public_surface_count > 0:
            api_covered_domain_count += 1
        if provider_status == "ready":
            provider_ready_domain_count += 1

        status = _domain_status(
            provider_status=provider_status,
            public_surface_count=public_surface_count,
            unit_test_count=unit_test_count,
            integration_test_count=integration_test_count,
        )
        priority = _priority(status)

        missing_layers: List[str] = []
        if provider_status != "ready":
            missing_layers.append("provider")
        if public_surface_count == 0:
            missing_layers.append("api")
        if unit_test_count == 0:
            missing_layers.append("unit_tests")
        if integration_test_count == 0:
            missing_layers.append("integration_tests")

        row = {
            "domain": domain,
            "label": spec["label"],
            "status": status,
            "priority": priority,
            "provider_status": provider_status,
            "available_providers": available_providers,
            "missing_providers": missing_providers,
            "public_surface_count": public_surface_count,
            "public_surfaces": public_surfaces,
            "unit_test_count": unit_test_count,
            "integration_test_count": integration_test_count,
            "assistant_test_count": assistant_test_count,
            "total_test_count": domain_total_test_count,
            "unit_tests": unit_tests,
            "integration_tests": integration_tests,
            "assistant_tests": assistant_tests,
            "missing_layers": missing_layers,
            "action": (
                f"Close {domain} validation gaps: "
                f"{', '.join(missing_layers) or 'maintain current coverage'}"
            ),
        }
        domains[domain] = row

        if status == "ready":
            ready_domain_count += 1
        elif status == "partial":
            partial_domain_count += 1
            focus_areas_detail.append(row)
        else:
            blocked_domain_count += 1
            focus_areas_detail.append(row)

    if blocked_domain_count:
        status = "knowledge_domain_validation_blocked"
    elif partial_domain_count:
        status = "knowledge_domain_validation_partial"
    else:
        status = "knowledge_domain_validation_ready"

    return {
        "status": status,
        "summary": _summary(
            ready_domain_count=ready_domain_count,
            partial_domain_count=partial_domain_count,
            blocked_domain_count=blocked_domain_count,
            total_test_count=total_test_count,
        ),
        "ready_domain_count": ready_domain_count,
        "partial_domain_count": partial_domain_count,
        "blocked_domain_count": blocked_domain_count,
        "total_domain_count": len(DOMAIN_SPECS),
        "total_test_count": total_test_count,
        "api_covered_domain_count": api_covered_domain_count,
        "provider_ready_domain_count": provider_ready_domain_count,
        "priority_domains": [row["domain"] for row in focus_areas_detail],
        "focus_areas_detail": focus_areas_detail,
        "domains": domains,
    }


def knowledge_domain_validation_matrix_recommendations(
    component: Dict[str, Any],
) -> List[str]:
    if _text(component.get("status")) == "knowledge_domain_validation_ready":
        return [
            "Standards, tolerance, and GD&T validation layers are aligned across "
            "providers, APIs, and tests."
        ]

    items: List[str] = []
    for row in component.get("focus_areas_detail") or []:
        domain = _text(row.get("domain")) or "unknown"
        missing_layers = ", ".join(row.get("missing_layers") or []) or "none"
        items.append(f"{domain}: close validation gaps in {missing_layers}")
    return _compact(items, limit=10)


def render_knowledge_domain_validation_matrix_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    component = payload.get("knowledge_domain_validation_matrix") or payload or {}
    lines = [
        f"# {title}",
        "",
        f"- `status`: `{component.get('status') or 'unknown'}`",
        f"- `summary`: `{component.get('summary') or 'none'}`",
        f"- `ready_domain_count`: `{component.get('ready_domain_count', 0)}`",
        f"- `partial_domain_count`: `{component.get('partial_domain_count', 0)}`",
        f"- `blocked_domain_count`: `{component.get('blocked_domain_count', 0)}`",
        f"- `total_test_count`: `{component.get('total_test_count', 0)}`",
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
                f"- `provider_status`: `{row.get('provider_status') or 'unknown'}`",
                f"- `public_surface_count`: `{row.get('public_surface_count', 0)}`",
                f"- `unit_test_count`: `{row.get('unit_test_count', 0)}`",
                f"- `integration_test_count`: `{row.get('integration_test_count', 0)}`",
                f"- `assistant_test_count`: `{row.get('assistant_test_count', 0)}`",
                f"- `missing_layers`: `{', '.join(row.get('missing_layers') or []) or 'none'}`",
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
