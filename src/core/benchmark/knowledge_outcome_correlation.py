"""Correlate knowledge-domain matrix state with real-data benchmark outcomes."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


DOMAIN_SURFACE_SPECS: Dict[str, Dict[str, Any]] = {
    "tolerance": {
        "label": "Tolerance & Fits",
        "surfaces": ("hybrid_dxf", "history_h5", "step_dir"),
        "action": (
            "Raise tolerance benchmark value by improving hybrid/history accuracy and "
            "STEP directory coverage on real data."
        ),
    },
    "standards": {
        "label": "Standards & Design Tables",
        "surfaces": ("hybrid_dxf", "history_h5"),
        "action": (
            "Tie standards knowledge to stronger DXF/history benchmark outcomes before "
            "treating standards readiness as production-safe."
        ),
    },
    "gdt": {
        "label": "GD&T & Datums",
        "surfaces": ("hybrid_dxf", "step_smoke", "step_dir"),
        "action": (
            "Raise GD&T benchmark value by keeping DXF evidence aligned with STEP smoke "
            "and STEP directory validations."
        ),
    },
}

READY_MATRIX_STATUSES = {"ready", "knowledge_domain_matrix_ready"}
PARTIAL_MATRIX_STATUSES = {"partial", "knowledge_domain_matrix_partial"}
BLOCKED_MATRIX_STATUSES = {
    "blocked",
    "missing",
    "knowledge_domain_matrix_missing",
}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _surface_score(name: str, row: Dict[str, Any]) -> float:
    if name in {"hybrid_dxf", "history_h5"}:
        return max(
            _to_float(row.get("coarse_accuracy")),
            _to_float(row.get("exact_accuracy")),
        )
    if name == "step_dir":
        return max(
            _to_float(row.get("coverage_ratio")),
            _to_float(row.get("hint_coverage_ratio")),
        )
    if name == "step_smoke":
        status = _text(row.get("status")).lower()
        if status == "ready":
            return 1.0
        if status in {"partial", "weak"}:
            return 0.5
        return 0.0
    return 0.0


def _surface_status(score: float) -> str:
    if score >= 0.8:
        return "ready"
    if score >= 0.5:
        return "partial"
    if score > 0.0:
        return "weak"
    return "missing"


def _domain_status(matrix_status: str, best_score: float) -> str:
    normalized = _text(matrix_status).lower() or "missing"
    if normalized in READY_MATRIX_STATUSES and best_score >= 0.8:
        return "ready"
    if best_score > 0.0 and normalized not in BLOCKED_MATRIX_STATUSES:
        return "partial"
    if normalized in PARTIAL_MATRIX_STATUSES and best_score > 0.0:
        return "partial"
    if normalized in BLOCKED_MATRIX_STATUSES:
        return "blocked"
    return "missing"


def _priority(status: str) -> str:
    if status in {"blocked", "missing"}:
        return "high"
    if status == "partial":
        return "medium"
    return "low"


def _best_surface(surface_scores: Dict[str, float]) -> Tuple[str, float]:
    if not surface_scores:
        return "none", 0.0
    name, score = max(surface_scores.items(), key=lambda item: item[1])
    return name, round(score, 6)


def build_knowledge_outcome_correlation_status(
    knowledge_domain_matrix_summary: Dict[str, Any],
    realdata_scorecard_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a domain-aware view of real-data benchmark outcomes."""
    matrix_component = (
        knowledge_domain_matrix_summary.get("knowledge_domain_matrix")
        or knowledge_domain_matrix_summary
        or {}
    )
    scorecard_component = (
        realdata_scorecard_summary.get("realdata_scorecard")
        or realdata_scorecard_summary
        or {}
    )
    matrix_domains = matrix_component.get("domains") or {}
    scorecard_components = scorecard_component.get("components") or {}

    domains: Dict[str, Dict[str, Any]] = {}
    focus_areas_detail: List[Dict[str, Any]] = []
    ready_count = 0
    partial_count = 0
    blocked_count = 0

    for name, spec in DOMAIN_SURFACE_SPECS.items():
        matrix_row = matrix_domains.get(name) or {}
        matrix_status = _text(matrix_row.get("status")) or "missing"
        surface_scores = {
            surface_name: round(
                _surface_score(surface_name, scorecard_components.get(surface_name) or {}),
                6,
            )
            for surface_name in spec["surfaces"]
        }
        surface_statuses = {
            surface_name: _surface_status(score)
            for surface_name, score in surface_scores.items()
        }
        best_surface, best_score = _best_surface(surface_scores)
        status = _domain_status(matrix_status, best_score)
        row = {
            "domain": name,
            "label": _text(matrix_row.get("label")) or _text(spec.get("label")) or name,
            "status": status,
            "priority": _priority(status),
            "matrix_status": matrix_status,
            "best_surface": best_surface,
            "best_surface_score": best_score,
            "surface_scores": surface_scores,
            "surface_statuses": surface_statuses,
            "ready_surfaces": [
                surface_name
                for surface_name, surface_status in surface_statuses.items()
                if surface_status == "ready"
            ],
            "partial_surfaces": [
                surface_name
                for surface_name, surface_status in surface_statuses.items()
                if surface_status == "partial"
            ],
            "weak_surfaces": [
                surface_name
                for surface_name, surface_status in surface_statuses.items()
                if surface_status == "weak"
            ],
            "missing_surfaces": [
                surface_name
                for surface_name, surface_status in surface_statuses.items()
                if surface_status == "missing"
            ],
            "focus_components": list(matrix_row.get("focus_components") or []),
            "missing_metrics": list(matrix_row.get("missing_metrics") or []),
            "action": _text(matrix_row.get("action")) or _text(spec.get("action")),
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

    if ready_count == len(DOMAIN_SURFACE_SPECS):
        overall_status = "knowledge_outcome_correlation_ready"
    elif ready_count > 0 or partial_count > 0:
        overall_status = "knowledge_outcome_correlation_partial"
    else:
        overall_status = "knowledge_outcome_correlation_missing"

    return {
        "status": overall_status,
        "ready_domain_count": ready_count,
        "partial_domain_count": partial_count,
        "blocked_domain_count": blocked_count,
        "total_domain_count": len(DOMAIN_SURFACE_SPECS),
        "priority_domains": [
            row["domain"]
            for row in focus_areas_detail
            if _text(row.get("priority")) == "high"
        ],
        "focus_areas": [row["domain"] for row in focus_areas_detail],
        "focus_areas_detail": focus_areas_detail,
        "domains": domains,
    }


def knowledge_outcome_correlation_recommendations(summary: Dict[str, Any]) -> List[str]:
    status = _text(summary.get("status")).lower()
    if status == "knowledge_outcome_correlation_missing":
        return [
            "Tie tolerance, standards, and GD&T benchmark domains to real-data outcomes "
            "before claiming benchmark knowledge superiority."
        ]

    items: List[str] = []
    for row in summary.get("focus_areas_detail") or []:
        domain = _text(row.get("domain")) or "unknown"
        matrix_status = _text(row.get("matrix_status")) or "missing"
        best_surface = _text(row.get("best_surface")) or "none"
        weak_surfaces = ", ".join(row.get("weak_surfaces") or [])
        missing_surfaces = ", ".join(row.get("missing_surfaces") or [])
        action = _text(row.get("action"))
        if matrix_status in BLOCKED_MATRIX_STATUSES:
            items.append(f"Backfill {domain} knowledge foundation first: {action}")
        if best_surface == "none":
            detail = missing_surfaces or "all linked real-data surfaces"
            items.append(f"Add {domain} real-data benchmark outcomes: {detail}")
            continue
        if weak_surfaces:
            items.append(
                f"Raise {domain} outcome strength beyond `{best_surface}` by improving: "
                f"{weak_surfaces}"
            )
        if missing_surfaces:
            items.append(
                f"Extend {domain} outcome coverage to missing surfaces: {missing_surfaces}"
            )
    if status == "knowledge_outcome_correlation_partial":
        items.append(
            "Use companion, bundle, release decision, and runbook surfaces to track "
            "which knowledge domains are still weak on real-data outcomes."
        )
    return _compact(items, limit=12)


def render_knowledge_outcome_correlation_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    component = payload.get("knowledge_outcome_correlation") or {}
    lines = [
        f"# {title}",
        "",
        "## Overview",
        "",
        f"- `status`: `{component.get('status', 'knowledge_outcome_correlation_missing')}`",
        f"- `ready_domain_count`: `{component.get('ready_domain_count', 0)}`",
        f"- `partial_domain_count`: `{component.get('partial_domain_count', 0)}`",
        f"- `blocked_domain_count`: `{component.get('blocked_domain_count', 0)}`",
        f"- `priority_domains`: "
        f"`{', '.join(component.get('priority_domains') or []) or 'none'}`",
        "",
        "## Domains",
        "",
    ]
    for name, row in (component.get("domains") or {}).items():
        lines.append(f"### {row.get('label') or name}")
        lines.append("")
        lines.append(f"- `status`: `{row.get('status', 'missing')}`")
        lines.append(f"- `matrix_status`: `{row.get('matrix_status', 'missing')}`")
        lines.append(f"- `best_surface`: `{row.get('best_surface', 'none')}`")
        lines.append(f"- `best_surface_score`: `{row.get('best_surface_score', 0)}`")
        for surface_name, score in (row.get("surface_scores") or {}).items():
            status = (row.get("surface_statuses") or {}).get(surface_name, "missing")
            lines.append(f"- `{surface_name}`: `{score}` (`{status}`)")
        if row.get("missing_metrics"):
            lines.append(
                f"- `missing_metrics`: `{', '.join(row.get('missing_metrics') or [])}`"
            )
        if row.get("focus_components"):
            lines.append(
                f"- `focus_components`: "
                f"`{', '.join(row.get('focus_components') or [])}`"
            )
        lines.append("")

    lines.extend(["## Recommendations", ""])
    recommendations = payload.get("recommendations") or []
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- none")
    return "\n".join(lines).rstrip() + "\n"
