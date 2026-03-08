"""Reusable benchmark helpers for knowledge readiness signals."""

from __future__ import annotations

from typing import Any, Dict, List


COMPONENT_REQUIREMENTS: Dict[str, List[str]] = {
    "tolerance": ["it_grade_count", "common_fit_count"],
    "standards": ["thread_count", "bearing_count", "oring_count"],
    "design_standards": [
        "surface_finish_grade_count",
        "linear_tolerance_range_count",
        "preferred_diameter_count",
        "standard_chamfer_count",
        "standard_fillet_count",
    ],
    "gdt": [
        "symbol_count",
        "application_count",
        "datum_feature_type_count",
        "tolerance_recommendation_count",
    ],
}

FOCUS_ACTIONS: Dict[str, str] = {
    "tolerance": "Backfill ISO 286 / GB-T 1800 grades and common-fit lookup coverage.",
    "standards": "Expand standard-part dictionaries for threads, bearings, and O-rings.",
    "design_standards": (
        "Complete general tolerance, surface finish, chamfer, fillet, and preferred-size "
        "reference tables."
    ),
    "gdt": (
        "Promote GD&T symbols, datum-feature mappings, and tolerance recommendations into "
        "the benchmark baseline."
    ),
}


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def collect_builtin_knowledge_snapshot() -> Dict[str, Any]:
    """Collect a deterministic snapshot of built-in knowledge coverage."""
    from src.core.knowledge.design_standards import (
        LINEAR_TOLERANCE_TABLE,
        PREFERRED_DIAMETERS,
        STANDARD_CHAMFERS,
        STANDARD_FILLETS,
        SURFACE_FINISH_TABLE,
    )
    from src.core.knowledge.gdt import (
        COMMON_APPLICATIONS,
        DATUM_FEATURE_TYPES,
        GDT_SYMBOLS,
        TOLERANCE_RECOMMENDATIONS,
    )
    from src.core.knowledge.standards import (
        BEARING_DATABASE,
        METRIC_THREADS,
        ORING_DATABASE,
    )
    from src.core.knowledge.tolerance import COMMON_FITS, TOLERANCE_GRADES

    return {
        "tolerance": {
            "it_grade_count": len(TOLERANCE_GRADES),
            "common_fit_count": len(COMMON_FITS),
        },
        "standards": {
            "thread_count": len(METRIC_THREADS),
            "bearing_count": len(BEARING_DATABASE),
            "oring_count": len(ORING_DATABASE),
        },
        "design_standards": {
            "surface_finish_grade_count": len(SURFACE_FINISH_TABLE),
            "linear_tolerance_range_count": len(LINEAR_TOLERANCE_TABLE),
            "preferred_diameter_count": len(PREFERRED_DIAMETERS),
            "standard_chamfer_count": len(STANDARD_CHAMFERS),
            "standard_fillet_count": len(STANDARD_FILLETS),
        },
        "gdt": {
            "symbol_count": len(GDT_SYMBOLS),
            "application_count": len(COMMON_APPLICATIONS),
            "datum_feature_type_count": len(DATUM_FEATURE_TYPES),
            "tolerance_recommendation_count": len(TOLERANCE_RECOMMENDATIONS),
        },
    }


def _component_from_counts(
    counts: Dict[str, Any],
    *,
    required_keys: List[str],
) -> Dict[str, Any]:
    normalized = {key: _to_int(counts.get(key)) for key in required_keys}
    total_reference_items = sum(normalized.values())
    available_keys = sum(1 for key in required_keys if normalized[key] > 0)
    missing_metrics = [key for key in required_keys if normalized[key] <= 0]
    if available_keys == len(required_keys):
        status = "ready"
    elif available_keys > 0:
        status = "partial"
    else:
        status = "missing"
    return {
        "status": status,
        "total_reference_items": total_reference_items,
        "ready_metric_count": available_keys,
        "missing_metric_count": len(missing_metrics),
        "total_metric_count": len(required_keys),
        "missing_metrics": missing_metrics,
        **normalized,
    }


def build_knowledge_focus_areas(components: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return actionable focus areas for non-ready knowledge components."""
    items: List[Dict[str, Any]] = []
    for name, component in components.items():
        status = str(component.get("status") or "").strip()
        if status == "ready":
            continue
        items.append(
            {
                "component": name,
                "status": status or "missing",
                "priority": "high" if status == "missing" else "medium",
                "ready_metric_count": _to_int(component.get("ready_metric_count")),
                "missing_metric_count": _to_int(component.get("missing_metric_count")),
                "total_metric_count": _to_int(component.get("total_metric_count")),
                "total_reference_items": _to_int(component.get("total_reference_items")),
                "missing_metrics": list(component.get("missing_metrics") or []),
                "action": FOCUS_ACTIONS.get(
                    name,
                    "Expand built-in knowledge coverage for this benchmark component.",
                ),
            }
        )
    return items


def build_knowledge_readiness_status(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Build a normalized benchmark knowledge-readiness summary."""
    snapshot = snapshot or {}
    tolerance = _component_from_counts(
        snapshot.get("tolerance") or {},
        required_keys=COMPONENT_REQUIREMENTS["tolerance"],
    )
    standards = _component_from_counts(
        snapshot.get("standards") or {},
        required_keys=COMPONENT_REQUIREMENTS["standards"],
    )
    design_standards = _component_from_counts(
        snapshot.get("design_standards") or {},
        required_keys=COMPONENT_REQUIREMENTS["design_standards"],
    )
    gdt = _component_from_counts(
        snapshot.get("gdt") or {},
        required_keys=COMPONENT_REQUIREMENTS["gdt"],
    )
    components = {
        "tolerance": tolerance,
        "standards": standards,
        "design_standards": design_standards,
        "gdt": gdt,
    }
    statuses = [component["status"] for component in components.values()]
    ready_count = statuses.count("ready")
    partial_count = statuses.count("partial")
    missing_count = statuses.count("missing")
    total_reference_items = sum(
        int(component.get("total_reference_items") or 0)
        for component in components.values()
    )
    focus_areas = [
        name for name, component in components.items() if component.get("status") != "ready"
    ]
    focus_area_statuses = [
        f"{name}:{component.get('status', 'unknown')}"
        for name, component in components.items()
        if component.get("status") != "ready"
    ]
    focus_areas_detail = build_knowledge_focus_areas(components)

    if ready_count == len(components):
        status = "knowledge_foundation_ready"
    elif ready_count > 0 or partial_count > 0:
        status = "knowledge_foundation_partial"
    else:
        status = "knowledge_foundation_missing"

    return {
        "status": status,
        "ready_component_count": ready_count,
        "partial_component_count": partial_count,
        "missing_component_count": missing_count,
        "total_reference_items": total_reference_items,
        "focus_areas": focus_areas,
        "focus_area_statuses": focus_area_statuses,
        "focus_areas_detail": focus_areas_detail,
        "components": components,
    }


def knowledge_readiness_recommendations(summary: Dict[str, Any]) -> List[str]:
    status = str(summary.get("status") or "").strip().lower()
    components = summary.get("components") or {}
    if status == "knowledge_foundation_missing":
        return [
            "Restore built-in tolerance, standards, design-standards, and GD&T "
            "coverage before claiming benchmark knowledge readiness."
        ]

    items: List[str] = []
    tolerance = components.get("tolerance") or {}
    standards = components.get("standards") or {}
    design_standards = components.get("design_standards") or {}
    gdt = components.get("gdt") or {}

    if tolerance.get("status") != "ready":
        items.append(
            "Stabilize ISO 286 / GB-T 1800 tolerance-grade and common-fit coverage."
        )
    if standards.get("status") != "ready":
        items.append(
            "Expand standard-part coverage for threads, bearings, and O-rings."
        )
    if design_standards.get("status") != "ready":
        items.append(
            "Complete general-tolerance, surface-finish, chamfer, and fillet knowledge."
        )
    if gdt.get("status") != "ready":
        items.append(
            "Promote GD&T symbols, applications, datum features, and tolerance "
            "recommendations into the benchmark baseline."
        )
    focus_areas_detail = summary.get("focus_areas_detail") or []
    if focus_areas_detail:
        focus_actions = [
            f"{row.get('component')}: {row.get('action')}"
            for row in focus_areas_detail[:3]
            if row.get("component") and row.get("action")
        ]
        if focus_actions:
            items.append("Prioritize knowledge gaps: " + " | ".join(focus_actions))
    if not items and status == "knowledge_foundation_ready":
        return []
    if status == "knowledge_foundation_partial":
        items.append(
            "Lift knowledge readiness into companion, release decision, and runbook "
            "views before calling the benchmark fully surpass-ready."
        )
    return items


def render_knowledge_readiness_markdown(payload: Dict[str, Any], title: str) -> str:
    component = payload.get("knowledge_readiness") or {}
    components = component.get("components") or {}
    recommendations = payload.get("recommendations") or []
    lines = [
        f"# {title}",
        "",
        "## Status",
        "",
        f"- `status`: `{component.get('status', 'knowledge_foundation_missing')}`",
        f"- `ready_component_count`: `{component.get('ready_component_count', 0)}`",
        f"- `partial_component_count`: `{component.get('partial_component_count', 0)}`",
        f"- `missing_component_count`: `{component.get('missing_component_count', 0)}`",
        f"- `total_reference_items`: `{component.get('total_reference_items', 0)}`",
        f"- `focus_areas`: `{', '.join(component.get('focus_areas') or []) or 'none'}`",
        "",
        "## Components",
        "",
    ]
    for name, row in components.items():
        lines.append(
            f"- `{name}`: status=`{row.get('status')}` "
            f"total_reference_items=`{row.get('total_reference_items')}`"
        )
    lines.extend(["", "## Focus Areas", ""])
    focus_areas = component.get("focus_areas_detail") or []
    if focus_areas:
        for row in focus_areas:
            lines.append(
                "- "
                f"`{row.get('component')}` "
                f"status=`{row.get('status')}` "
                f"priority=`{row.get('priority')}` "
                f"missing_metrics=`{', '.join(row.get('missing_metrics') or []) or 'none'}` "
                f"action=`{row.get('action')}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Recommendations", ""])
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- Knowledge readiness is healthy.")
    lines.append("")
    return "\n".join(lines)
