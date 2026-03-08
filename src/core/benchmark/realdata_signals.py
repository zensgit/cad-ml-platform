"""Reusable benchmark helpers for real-data validation signals."""

from __future__ import annotations

from typing import Any, Dict, List


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _status_from_score(score: float) -> str:
    if score >= 0.8:
        return "ready"
    if score >= 0.5:
        return "partial"
    return "weak"


def _hybrid_component(summary: Dict[str, Any]) -> Dict[str, Any]:
    sample_size = _to_int(summary.get("sample_size"))
    coarse_scores = summary.get("coarse_scores") or {}
    exact_scores = summary.get("exact_scores") or {}
    hybrid_coarse = _to_float(
        ((coarse_scores.get("hybrid_label") or {}).get("accuracy"))
        or summary.get("coarse_accuracy_overall")
    )
    hybrid_exact = _to_float(
        ((exact_scores.get("hybrid_label") or {}).get("accuracy"))
        or summary.get("accuracy_overall")
    )
    graph2d_coarse = _to_float(
        ((coarse_scores.get("graph2d_label") or {}).get("accuracy"))
        or summary.get("graph2d_accuracy_overall")
    )
    confidence_stats = summary.get("confidence_stats") or {}
    low_conf_rate = _to_float((confidence_stats.get("hybrid_label") or {}).get("low_conf_rate"))

    if sample_size <= 0:
        status = "missing"
    else:
        status = _status_from_score(hybrid_coarse)
    return {
        "status": status,
        "sample_size": sample_size,
        "hybrid_coarse_accuracy": round(hybrid_coarse, 6),
        "hybrid_exact_accuracy": round(hybrid_exact, 6),
        "graph2d_coarse_accuracy": round(graph2d_coarse, 6),
        "low_conf_rate": round(low_conf_rate, 6),
    }


def _history_component(report: Dict[str, Any]) -> Dict[str, Any]:
    validation = report.get("h5_validation") or {}
    status = str(validation.get("status") or "missing")
    prediction = validation.get("prediction") or {}
    ready_status = {
        "ok": "ready",
        "skipped_no_h5py": "environment_blocked",
        "missing": "missing",
    }.get(status, "partial")
    return {
        "status": ready_status,
        "input_status": status,
        "sequence_length": _to_int(validation.get("tokens_length")),
        "confidence": round(_to_float(prediction.get("confidence")), 6),
        "label": str(prediction.get("label") or ""),
        "source": str(prediction.get("source") or ""),
        "vec_shape": list(validation.get("vec_shape") or []),
    }


def _step_smoke_component(report: Dict[str, Any]) -> Dict[str, Any]:
    validation = report.get("step_validation") or {}
    status = str(validation.get("status") or "missing")
    graph = validation.get("brep_graph") or {}
    features = validation.get("brep_features") or {}
    ready_status = {
        "ok": "ready",
        "skipped_no_occ": "environment_blocked",
        "missing": "missing",
        "load_failed": "load_failed",
    }.get(status, "partial")
    return {
        "status": ready_status,
        "input_status": status,
        "shape_loaded": bool(validation.get("shape_loaded")),
        "valid_3d": bool((graph.get("valid_3d")) or (features.get("valid_3d"))),
        "graph_schema_version": str(graph.get("graph_schema_version") or ""),
        "faces": _to_int((features.get("faces"))),
        "node_count": _to_int(graph.get("node_count")),
        "edge_count": _to_int(graph.get("edge_count")),
    }


def _step_dir_component(summary: Dict[str, Any]) -> Dict[str, Any]:
    sample_size = _to_int(summary.get("sample_size"))
    status_counts = summary.get("status_counts") or {}
    ok_count = _to_int(status_counts.get("ok"))
    valid_3d_count = _to_int(summary.get("valid_3d_count"))
    hint_coverage_count = _to_int(summary.get("hint_coverage_count"))
    schema_counts = summary.get("graph_schema_version_counts") or {}
    if sample_size <= 0:
        status = "missing"
    elif ok_count == sample_size and valid_3d_count == sample_size:
        status = "ready"
    elif ok_count > 0:
        status = "partial"
    else:
        status = "load_failed"
    return {
        "status": status,
        "sample_size": sample_size,
        "ok_count": ok_count,
        "valid_3d_count": valid_3d_count,
        "hint_coverage_count": hint_coverage_count,
        "graph_schema_versions": dict(schema_counts),
    }


def build_realdata_signals_status(
    *,
    hybrid_summary: Dict[str, Any],
    online_example_report: Dict[str, Any],
    step_dir_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a benchmark-oriented summary of real-data validation coverage."""
    hybrid = _hybrid_component(hybrid_summary)
    history = _history_component(online_example_report)
    step_smoke = _step_smoke_component(online_example_report)
    step_dir = _step_dir_component(step_dir_summary)

    components = {
        "hybrid_dxf": hybrid,
        "history_h5": history,
        "step_smoke": step_smoke,
        "step_dir": step_dir,
    }
    ready_count = sum(1 for row in components.values() if row.get("status") == "ready")
    partial_count = sum(
        1 for row in components.values() if row.get("status") in {"partial", "weak"}
    )
    blocked_count = sum(
        1 for row in components.values() if row.get("status") == "environment_blocked"
    )
    available_count = sum(
        1 for row in components.values() if row.get("status") not in {"missing"}
    )
    if ready_count >= 3 and available_count == len(components):
        overall_status = "realdata_foundation_ready"
    elif ready_count >= 2 or (ready_count >= 1 and partial_count >= 1):
        overall_status = "realdata_foundation_partial"
    else:
        overall_status = "realdata_foundation_missing"

    return {
        "status": overall_status,
        "component_statuses": {name: row.get("status") for name, row in components.items()},
        "ready_component_count": ready_count,
        "partial_component_count": partial_count,
        "environment_blocked_count": blocked_count,
        "available_component_count": available_count,
        "components": components,
    }


def realdata_signals_recommendations(component: Dict[str, Any]) -> List[str]:
    """Return operator-facing next steps for real-data benchmark readiness."""
    rows = component.get("components") or {}
    recommendations: List[str] = []
    hybrid = rows.get("hybrid_dxf") or {}
    if hybrid.get("status") in {"weak", "missing"}:
        recommendations.append(
            "Strengthen DXF hybrid real-data coverage before treating Graph2D "
            "or hybrid scores as release-grade."
        )
    history = rows.get("history_h5") or {}
    if history.get("status") == "environment_blocked":
        recommendations.append(
            "Install h5py or standardize the history-sequence environment so "
            "real `.h5` evidence is reproducible."
        )
    if history.get("status") == "missing":
        recommendations.append(
            "Add a representative `.h5` history dataset so benchmark reports "
            "cover history-sequence evidence."
        )
    step_smoke = rows.get("step_smoke") or {}
    if step_smoke.get("status") == "environment_blocked":
        recommendations.append(
            "Provide an OCC-enabled runtime so STEP smoke validation is "
            "available outside local micromamba setups."
        )
    step_dir = rows.get("step_dir") or {}
    if step_dir.get("status") in {"partial", "missing", "load_failed"}:
        recommendations.append(
            "Expand STEP/B-Rep directory validation so benchmark release "
            "surfaces include broader 3D evidence."
        )
    if not recommendations:
        recommendations.append(
            "Real-data benchmark signals cover DXF hybrid, history `.h5`, and "
            "STEP/B-Rep validations."
        )
    return recommendations


def render_realdata_signals_markdown(payload: Dict[str, Any], title: str) -> str:
    """Render a concise Markdown summary for real-data benchmark signals."""
    component = payload.get("realdata_signals") or {}
    rows = component.get("components") or {}
    lines = [f"# {title}", "", "## Overview", ""]
    lines.append(f"- `status`: `{component.get('status', 'unknown')}`")
    lines.append(f"- `ready_component_count`: `{component.get('ready_component_count', 0)}`")
    lines.append(
        f"- `partial_component_count`: `{component.get('partial_component_count', 0)}`"
    )
    lines.append(
        f"- `environment_blocked_count`: `{component.get('environment_blocked_count', 0)}`"
    )
    lines.extend(["", "## Components", ""])
    for name in ("hybrid_dxf", "history_h5", "step_smoke", "step_dir"):
        row = rows.get(name) or {}
        lines.append(f"### {name}")
        lines.append("")
        for key, value in row.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    lines.extend(["## Recommendations", ""])
    recommendations = payload.get("recommendations") or []
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- none")
    return "\n".join(lines).rstrip() + "\n"
