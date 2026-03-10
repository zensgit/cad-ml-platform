"""Benchmark helpers for built-in knowledge source coverage."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


CORE_SOURCE_GROUPS: Dict[str, Dict[str, Any]] = {
    "tolerance": {
        "label": "Tolerance & Fits",
        "domain": "tolerance",
        "reference_standard_count": 3,
    },
    "standards": {
        "label": "Standard Parts",
        "domain": "standards",
        "reference_standard_count": 4,
    },
    "design_standards": {
        "label": "Design Standards",
        "domain": "standards",
        "reference_standard_count": 4,
    },
    "gdt": {
        "label": "GD&T & Datums",
        "domain": "gdt",
        "reference_standard_count": 4,
    },
}

EXPANSION_SOURCE_GROUPS: Dict[str, Dict[str, Any]] = {
    "machining": {
        "label": "Machining Knowledge",
        "domain": "manufacturing",
        "reference_standard_count": 0,
        "allow_reference_free": True,
    },
    "welding": {
        "label": "Welding Knowledge",
        "domain": "manufacturing",
        "reference_standard_count": 0,
        "allow_reference_free": True,
    },
    "surface_treatment": {
        "label": "Surface Treatment Knowledge",
        "domain": "manufacturing",
        "reference_standard_count": 0,
        "allow_reference_free": True,
    },
    "heat_treatment": {
        "label": "Heat Treatment Knowledge",
        "domain": "manufacturing",
        "reference_standard_count": 0,
        "allow_reference_free": True,
    },
}

DOMAIN_ORDER = ("tolerance", "standards", "gdt")
DOMAIN_LABELS = {
    "tolerance": "Tolerance & Fits",
    "standards": "Standards & Design Tables",
    "gdt": "GD&T & Datums",
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


def collect_builtin_knowledge_source_snapshot() -> Dict[str, Any]:
    """Collect a deterministic snapshot of built-in knowledge source tables."""
    from src.core.knowledge.design_standards import (
        LINEAR_TOLERANCE_TABLE,
        PREFERRED_DIAMETERS,
        STANDARD_CHAMFERS,
        STANDARD_FILLETS,
        SURFACE_FINISH_TABLE,
    )
    from src.core.knowledge.design_standards.general_tolerances import (
        ANGULAR_TOLERANCE_TABLE,
        CHAMFER_FILLET_TOLERANCE,
    )
    from src.core.knowledge.design_standards.surface_finish import (
        SURFACE_FINISH_APPLICATIONS,
    )
    from src.core.knowledge.gdt import (
        COMMON_APPLICATIONS,
        DATUM_FEATURE_TYPES,
        GDT_SYMBOLS,
        TOLERANCE_RECOMMENDATIONS,
    )
    from src.core.knowledge.gdt.tolerances import TOLERANCE_ZONE_SHAPES
    from src.core.knowledge.heat_treatment.annealing import (
        ANNEALING_DATABASE,
        STRESS_RELIEF_GUIDELINES,
    )
    from src.core.knowledge.heat_treatment.hardening import (
        HARDENING_DATABASE,
        TEMPERING_CURVES,
    )
    from src.core.knowledge.heat_treatment.processes import (
        HEAT_TREATMENT_DATABASE,
        QUENCH_MEDIA_DATA,
    )
    from src.core.knowledge.machining.cutting import (
        CUTTING_SPEED_TABLE,
        DEPTH_TABLE,
        FEED_TABLE,
    )
    from src.core.knowledge.machining.materials import MACHINABILITY_DATABASE
    from src.core.knowledge.machining.tooling import (
        GEOMETRY_RECOMMENDATIONS,
        TOOL_DATABASE,
        TOOL_MATERIAL_MATRIX,
    )
    from src.core.knowledge.standards import (
        BEARING_DATABASE,
        METRIC_THREADS,
        ORING_DATABASE,
    )
    from src.core.knowledge.standards.bearings import (
        SERIES_60_DATA,
        SERIES_62_DATA,
        SERIES_63_DATA,
    )
    from src.core.knowledge.standards.seals import (
        METRIC_ORING_DATA,
        ORING_TOLERANCES,
        STANDARD_CROSS_SECTIONS,
    )
    from src.core.knowledge.standards.threads import (
        METRIC_COARSE_DATA,
        METRIC_FINE_DATA,
    )
    from src.core.knowledge.surface_treatment.anodizing import (
        ANODIZE_COLORS,
        ANODIZING_DATABASE,
    )
    from src.core.knowledge.surface_treatment.coating import (
        COATING_DATABASE,
        CORROSIVITY_RECOMMENDATIONS,
    )
    from src.core.knowledge.surface_treatment.electroplating import (
        PLATING_DATABASE,
        THICKNESS_SPECIFICATIONS,
    )
    from src.core.knowledge.tolerance import COMMON_FITS, TOLERANCE_GRADES
    from src.core.knowledge.tolerance.fits import HOLE_SYMBOLS, SHAFT_SYMBOLS
    from src.core.knowledge.tolerance.it_grades import (
        GRADE_APPLICATIONS,
        SIZE_RANGES,
    )
    from src.core.knowledge.tolerance.selection import FIT_SELECTION_RULES
    from src.core.knowledge.welding.joints import (
        FILLET_WELD_SIZES,
        JOINT_DESIGN_DATABASE,
    )
    from src.core.knowledge.welding.materials import PREHEAT_CHART, WELDABILITY_DATABASE
    from src.core.knowledge.welding.parameters import (
        FILLER_MATERIAL_DATABASE,
        WELDING_PROCESS_DATABASE,
    )

    return {
        "tolerance": {
            "source_tables": {
                "tolerance_grades": len(TOLERANCE_GRADES),
                "size_ranges": len(SIZE_RANGES),
                "grade_applications": len(GRADE_APPLICATIONS),
                "common_fits": len(COMMON_FITS),
                "shaft_symbols": len(SHAFT_SYMBOLS),
                "hole_symbols": len(HOLE_SYMBOLS),
                "fit_selection_rules": len(FIT_SELECTION_RULES),
            }
        },
        "standards": {
            "source_tables": {
                "metric_coarse_data": len(METRIC_COARSE_DATA),
                "metric_fine_data": len(METRIC_FINE_DATA),
                "metric_threads": len(METRIC_THREADS),
                "series_60_data": len(SERIES_60_DATA),
                "series_62_data": len(SERIES_62_DATA),
                "series_63_data": len(SERIES_63_DATA),
                "bearing_database": len(BEARING_DATABASE),
                "standard_cross_sections": len(STANDARD_CROSS_SECTIONS),
                "oring_tolerances": len(ORING_TOLERANCES),
                "metric_oring_data": len(METRIC_ORING_DATA),
                "oring_database": len(ORING_DATABASE),
            }
        },
        "design_standards": {
            "source_tables": {
                "surface_finish_table": len(SURFACE_FINISH_TABLE),
                "surface_finish_applications": len(SURFACE_FINISH_APPLICATIONS),
                "linear_tolerance_table": len(LINEAR_TOLERANCE_TABLE),
                "angular_tolerance_table": len(ANGULAR_TOLERANCE_TABLE),
                "chamfer_fillet_tolerance": len(CHAMFER_FILLET_TOLERANCE),
                "preferred_diameters": len(PREFERRED_DIAMETERS),
                "standard_chamfers": len(STANDARD_CHAMFERS),
                "standard_fillets": len(STANDARD_FILLETS),
            }
        },
        "gdt": {
            "source_tables": {
                "gdt_symbols": len(GDT_SYMBOLS),
                "tolerance_recommendations": len(TOLERANCE_RECOMMENDATIONS),
                "tolerance_zone_shapes": len(TOLERANCE_ZONE_SHAPES),
                "datum_feature_types": len(DATUM_FEATURE_TYPES),
                "common_applications": len(COMMON_APPLICATIONS),
            }
        },
        "machining": {
            "source_tables": {
                "tool_database": len(TOOL_DATABASE),
                "tool_material_matrix": len(TOOL_MATERIAL_MATRIX),
                "geometry_recommendations": len(GEOMETRY_RECOMMENDATIONS),
                "cutting_speed_table": len(CUTTING_SPEED_TABLE),
                "feed_table": len(FEED_TABLE),
                "depth_table": len(DEPTH_TABLE),
                "machinability_database": len(MACHINABILITY_DATABASE),
            }
        },
        "welding": {
            "source_tables": {
                "joint_design_database": len(JOINT_DESIGN_DATABASE),
                "fillet_weld_sizes": len(FILLET_WELD_SIZES),
                "weldability_database": len(WELDABILITY_DATABASE),
                "preheat_chart": len(PREHEAT_CHART),
                "welding_process_database": len(WELDING_PROCESS_DATABASE),
                "filler_material_database": len(FILLER_MATERIAL_DATABASE),
            }
        },
        "surface_treatment": {
            "source_tables": {
                "anodizing_database": len(ANODIZING_DATABASE),
                "anodize_colors": len(ANODIZE_COLORS),
                "coating_database": len(COATING_DATABASE),
                "corrosivity_recommendations": len(CORROSIVITY_RECOMMENDATIONS),
                "plating_database": len(PLATING_DATABASE),
                "thickness_specifications": len(THICKNESS_SPECIFICATIONS),
            }
        },
        "heat_treatment": {
            "source_tables": {
                "heat_treatment_database": len(HEAT_TREATMENT_DATABASE),
                "quench_media_data": len(QUENCH_MEDIA_DATA),
                "hardening_database": len(HARDENING_DATABASE),
                "tempering_curves": len(TEMPERING_CURVES),
                "annealing_database": len(ANNEALING_DATABASE),
                "stress_relief_guidelines": len(STRESS_RELIEF_GUIDELINES),
            }
        },
    }


def _source_group_row(name: str, counts: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    tables = counts.get("source_tables") or {}
    normalized_tables = {
        table_name: int(table_count or 0)
        for table_name, table_count in tables.items()
    }
    source_table_count = len(normalized_tables)
    ready_source_table_count = sum(1 for count in normalized_tables.values() if count > 0)
    source_item_count = sum(normalized_tables.values())
    reference_standard_count = int(config.get("reference_standard_count") or 0)
    allow_reference_free = bool(config.get("allow_reference_free"))
    reference_ready = reference_standard_count > 0 or allow_reference_free
    if (
        source_table_count > 0
        and ready_source_table_count == source_table_count
        and source_item_count > 0
        and reference_ready
    ):
        status = "ready"
    elif ready_source_table_count > 0 or source_item_count > 0:
        status = "partial"
    else:
        status = "missing"
    return {
        "name": name,
        "label": config.get("label") or name,
        "domain": config.get("domain") or name,
        "status": status,
        "priority": "high" if status == "missing" else "medium" if status == "partial" else "low",
        "source_table_count": source_table_count,
        "ready_source_table_count": ready_source_table_count,
        "missing_source_table_count": max(source_table_count - ready_source_table_count, 0),
        "source_item_count": source_item_count,
        "reference_standard_count": reference_standard_count,
        "allow_reference_free": allow_reference_free,
        "source_tables": normalized_tables,
        "missing_source_tables": [
            table_name for table_name, count in normalized_tables.items() if count <= 0
        ],
    }


def _domain_rows(source_groups: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for domain in DOMAIN_ORDER:
        group_rows = [
            row
            for row in source_groups.values()
            if _text(row.get("domain")) == domain
        ]
        statuses = [_text(row.get("status")) for row in group_rows]
        if statuses and all(status == "ready" for status in statuses):
            status = "ready"
        elif any(status in {"ready", "partial"} for status in statuses):
            status = "partial"
        else:
            status = "missing"
        rows[domain] = {
            "domain": domain,
            "label": DOMAIN_LABELS.get(domain, domain),
            "status": status,
            "priority": (
                "high"
                if status == "missing"
                else "medium"
                if status == "partial"
                else "low"
            ),
            "source_groups": [row.get("name") for row in group_rows],
            "focus_source_groups": [
                row.get("name") for row in group_rows if _text(row.get("status")) != "ready"
            ],
            "source_table_count": sum(
                int(row.get("source_table_count") or 0) for row in group_rows
            ),
            "source_item_count": sum(
                int(row.get("source_item_count") or 0) for row in group_rows
            ),
            "reference_standard_count": sum(
                int(row.get("reference_standard_count") or 0) for row in group_rows
            ),
        }
    return rows


def build_knowledge_source_coverage_status(
    snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a benchmark summary for built-in knowledge source coverage."""
    snapshot = snapshot or {}
    source_groups = {
        name: _source_group_row(name, snapshot.get(name) or {}, config)
        for name, config in CORE_SOURCE_GROUPS.items()
    }
    expansion_candidates = [
        _source_group_row(name, snapshot.get(name) or {}, config)
        for name, config in EXPANSION_SOURCE_GROUPS.items()
    ]
    ready_group_count = sum(
        1 for row in source_groups.values() if _text(row.get("status")) == "ready"
    )
    partial_group_count = sum(
        1 for row in source_groups.values() if _text(row.get("status")) == "partial"
    )
    missing_group_count = sum(
        1 for row in source_groups.values() if _text(row.get("status")) == "missing"
    )
    if ready_group_count == len(source_groups):
        status = "knowledge_source_coverage_ready"
    elif ready_group_count > 0 or partial_group_count > 0:
        status = "knowledge_source_coverage_partial"
    else:
        status = "knowledge_source_coverage_missing"

    focus_areas_detail = [
        row for row in source_groups.values() if _text(row.get("status")) != "ready"
    ]
    domains = _domain_rows(source_groups)
    priority_domains = [
        row.get("domain")
        for row in domains.values()
        if _text(row.get("status")) != "ready"
    ]
    expansion_candidates = sorted(
        expansion_candidates,
        key=lambda row: (
            0 if _text(row.get("status")) == "ready" else 1,
            -int(row.get("source_item_count") or 0),
            _text(row.get("name")),
        ),
    )

    return {
        "status": status,
        "ready_source_group_count": ready_group_count,
        "partial_source_group_count": partial_group_count,
        "missing_source_group_count": missing_group_count,
        "total_source_group_count": len(source_groups),
        "total_source_table_count": sum(
            int(row.get("source_table_count") or 0) for row in source_groups.values()
        ),
        "total_source_item_count": sum(
            int(row.get("source_item_count") or 0) for row in source_groups.values()
        ),
        "total_reference_standard_count": sum(
            int(row.get("reference_standard_count") or 0)
            for row in source_groups.values()
        ),
        "source_groups": source_groups,
        "domains": domains,
        "focus_areas": [row.get("name") for row in focus_areas_detail],
        "focus_areas_detail": focus_areas_detail,
        "priority_domains": priority_domains,
        "expansion_candidates": expansion_candidates,
        "ready_expansion_candidate_count": sum(
            1 for row in expansion_candidates if _text(row.get("status")) == "ready"
        ),
    }


def knowledge_source_coverage_recommendations(component: Dict[str, Any]) -> List[str]:
    status = _text(component.get("status")).lower()
    items: List[str] = []
    for row in component.get("focus_areas_detail") or []:
        name = _text(row.get("name")) or "unknown"
        if row.get("missing_source_tables"):
            items.append(
                f"Restore {name} source tables: "
                + ", ".join(_compact(row.get("missing_source_tables") or [], limit=4))
            )
        items.append(
            f"Raise {name} coverage to ready by keeping table counts and source items "
            + (
                "non-zero."
                if row.get("allow_reference_free")
                else "and reference standards non-zero."
            )
        )
    expansion_candidates = component.get("expansion_candidates") or []
    ready_expansions = [
        row.get("name")
        for row in expansion_candidates
        if _text(row.get("status")) == "ready"
    ]
    if ready_expansions:
        items.append(
            "Promote next-wave manufacturing knowledge into benchmark views: "
            + ", ".join(_compact(ready_expansions, limit=4))
        )
    if status == "knowledge_source_coverage_ready" and not items:
        items.append(
            "Built-in knowledge sources are covered; expand benchmark views to expose "
            "more manufacturing domains."
        )
    return _compact(items, limit=10)


def render_knowledge_source_coverage_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    component = payload.get("knowledge_source_coverage") or {}
    lines = [
        f"# {title}",
        "",
        "## Overview",
        "",
        f"- `status`: `{component.get('status', 'knowledge_source_coverage_missing')}`",
        f"- `ready_source_group_count`: `{component.get('ready_source_group_count', 0)}`",
        f"- `partial_source_group_count`: `{component.get('partial_source_group_count', 0)}`",
        f"- `missing_source_group_count`: `{component.get('missing_source_group_count', 0)}`",
        f"- `total_source_table_count`: `{component.get('total_source_table_count', 0)}`",
        f"- `total_source_item_count`: `{component.get('total_source_item_count', 0)}`",
        f"- `total_reference_standard_count`: "
        f"`{component.get('total_reference_standard_count', 0)}`",
        f"- `priority_domains`: `{', '.join(component.get('priority_domains') or []) or 'none'}`",
        "",
        "## Source Groups",
        "",
    ]
    for name, row in (component.get("source_groups") or {}).items():
        lines.append(
            "- "
            f"`{name}` status=`{row.get('status')}` "
            f"source_tables=`{row.get('source_table_count')}` "
            f"source_items=`{row.get('source_item_count')}` "
            f"reference_standards=`{row.get('reference_standard_count')}`"
        )
    lines.extend(["", "## Domains", ""])
    for name, row in (component.get("domains") or {}).items():
        lines.append(
            "- "
            f"`{name}` status=`{row.get('status')}` "
            f"focus_source_groups=`{', '.join(row.get('focus_source_groups') or []) or 'none'}`"
        )
    lines.extend(["", "## Expansion Candidates", ""])
    expansion_candidates = component.get("expansion_candidates") or []
    if expansion_candidates:
        for row in expansion_candidates:
            lines.append(
                "- "
                f"`{row.get('name')}` status=`{row.get('status')}` "
                f"source_tables=`{row.get('source_table_count')}` "
                f"source_items=`{row.get('source_item_count')}`"
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
