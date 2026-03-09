"""Reusable benchmark helpers for real-data benchmark scorecards."""

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


def _score_status(score: float, *, missing_when_zero: bool = False) -> str:
    if missing_when_zero and score <= 0.0:
        return "missing"
    if score >= 0.8:
        return "ready"
    if score >= 0.5:
        return "partial"
    if score > 0.0:
        return "weak"
    return "missing"


def _hybrid_component(summary: Dict[str, Any]) -> Dict[str, Any]:
    sample_size = _to_int(summary.get("sample_size"))
    coarse_scores = summary.get("coarse_scores") or {}
    exact_scores = summary.get("exact_scores") or {}
    confidence_stats = summary.get("confidence_stats") or {}
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
    low_conf_rate = _to_float((confidence_stats.get("hybrid_label") or {}).get("low_conf_rate"))
    status = "missing" if sample_size <= 0 else _score_status(hybrid_coarse)
    return {
        "status": status,
        "sample_size": sample_size,
        "coarse_accuracy": round(hybrid_coarse, 6),
        "exact_accuracy": round(hybrid_exact, 6),
        "graph2d_coarse_accuracy": round(graph2d_coarse, 6),
        "hybrid_minus_graph2d": round(hybrid_coarse - graph2d_coarse, 6),
        "low_conf_rate": round(low_conf_rate, 6),
    }


def _history_component(summary: Dict[str, Any], online_report: Dict[str, Any]) -> Dict[str, Any]:
    if summary:
        total = _to_int(summary.get("total"))
        coarse_accuracy = _to_float(summary.get("coarse_accuracy_overall"))
        exact_accuracy = _to_float(summary.get("accuracy_overall"))
        low_conf_rate = _to_float(summary.get("low_conf_rate"))
        status = "missing" if total <= 0 else _score_status(coarse_accuracy)
        return {
            "status": status,
            "sample_size": total,
            "ok_count": _to_int(summary.get("ok_count")),
            "coarse_accuracy": round(coarse_accuracy, 6),
            "exact_accuracy": round(exact_accuracy, 6),
            "coarse_macro_f1": round(_to_float(summary.get("coarse_macro_f1_overall")), 6),
            "exact_macro_f1": round(_to_float(summary.get("macro_f1_overall")), 6),
            "low_conf_rate": round(low_conf_rate, 6),
            "top_mismatches": list(summary.get("coarse_top_mismatches") or []),
        }

    validation = online_report.get("h5_validation") or {}
    raw_status = str(validation.get("status") or "missing")
    prediction = validation.get("prediction") or {}
    mapped_status = {
        "ok": "smoke_only",
        "skipped_no_h5py": "environment_blocked",
        "missing": "missing",
    }.get(raw_status, "partial")
    return {
        "status": mapped_status,
        "input_status": raw_status,
        "sample_size": 1 if raw_status == "ok" else 0,
        "sequence_length": _to_int(validation.get("tokens_length")),
        "confidence": round(_to_float(prediction.get("confidence")), 6),
        "label": str(prediction.get("label") or ""),
        "source": str(prediction.get("source") or ""),
        "vec_shape": list(validation.get("vec_shape") or []),
    }


def _step_smoke_component(report: Dict[str, Any]) -> Dict[str, Any]:
    validation = report.get("step_validation") or {}
    raw_status = str(validation.get("status") or "missing")
    graph = validation.get("brep_graph") or {}
    features = validation.get("brep_features") or {}
    status = {
        "ok": "ready",
        "skipped_no_occ": "environment_blocked",
        "missing": "missing",
        "load_failed": "load_failed",
    }.get(raw_status, "partial")
    return {
        "status": status,
        "input_status": raw_status,
        "shape_loaded": bool(validation.get("shape_loaded")),
        "valid_3d": bool(graph.get("valid_3d") or features.get("valid_3d")),
        "graph_schema_version": str(graph.get("graph_schema_version") or ""),
        "faces": _to_int(features.get("faces")),
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
        "coverage_ratio": round(ok_count / sample_size, 6) if sample_size else 0.0,
    }


def _best_surface(components: Dict[str, Dict[str, Any]]) -> str:
    candidates = {
        "hybrid_dxf": _to_float((components.get("hybrid_dxf") or {}).get("coarse_accuracy")),
        "history_h5": _to_float((components.get("history_h5") or {}).get("coarse_accuracy")),
    }
    name, score = max(candidates.items(), key=lambda item: item[1])
    if score > 0:
        return name
    step_dir_score = _to_float((components.get("step_dir") or {}).get("coverage_ratio"))
    return "step_dir" if step_dir_score > 0 else "none"


def build_realdata_scorecard_status(
    *,
    hybrid_summary: Dict[str, Any],
    history_summary: Dict[str, Any],
    online_example_report: Dict[str, Any],
    step_dir_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a real-data scorecard across DXF, history, and STEP signals."""
    hybrid = _hybrid_component(hybrid_summary)
    history = _history_component(history_summary, online_example_report)
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
        1 for row in components.values() if row.get("status") in {"partial", "weak", "smoke_only"}
    )
    blocked_count = sum(
        1 for row in components.values() if row.get("status") == "environment_blocked"
    )
    available_count = sum(
        1 for row in components.values() if row.get("status") not in {"missing"}
    )
    if ready_count >= 3 and available_count == len(components):
        status = "realdata_scorecard_ready"
    elif ready_count >= 1 or partial_count >= 1:
        status = "realdata_scorecard_partial"
    else:
        status = "realdata_scorecard_missing"

    return {
        "status": status,
        "component_statuses": {name: row.get("status") for name, row in components.items()},
        "ready_component_count": ready_count,
        "partial_component_count": partial_count,
        "environment_blocked_count": blocked_count,
        "available_component_count": available_count,
        "best_surface": _best_surface(components),
        "components": components,
    }


def realdata_scorecard_recommendations(component: Dict[str, Any]) -> List[str]:
    """Return operator-facing next steps for real-data benchmark scorecards."""
    rows = component.get("components") or {}
    recommendations: List[str] = []
    hybrid = rows.get("hybrid_dxf") or {}
    if hybrid.get("status") in {"weak", "missing"}:
        recommendations.append(
            "Improve DXF hybrid real-data accuracy before treating the semantic scorecard "
            "as benchmark-ready."
        )
    history = rows.get("history_h5") or {}
    if history.get("status") == "environment_blocked":
        recommendations.append(
            "Standardize the history `.h5` runtime so history-sequence evidence is "
            "reproducible in CI and release reviews."
        )
    elif history.get("status") in {"missing", "smoke_only"}:
        recommendations.append(
            "Add a larger `.h5` evaluation set so history-sequence moves beyond smoke-only "
            "evidence."
        )
    step_smoke = rows.get("step_smoke") or {}
    if step_smoke.get("status") == "environment_blocked":
        recommendations.append(
            "Provide an OCC-enabled runtime outside local-only environments so STEP smoke "
            "validation is portable."
        )
    step_dir = rows.get("step_dir") or {}
    if step_dir.get("status") in {"missing", "partial", "load_failed"}:
        recommendations.append(
            "Expand STEP/B-Rep directory evaluation so 3D real-data evidence is broad enough "
            "for benchmark release decisions."
        )
    if not recommendations:
        recommendations.append(
            "Real-data scorecard covers DXF hybrid, history `.h5`, and STEP/B-Rep evidence "
            "with benchmark-ready breadth."
        )
    return recommendations


def render_realdata_scorecard_markdown(payload: Dict[str, Any], title: str) -> str:
    """Render a concise Markdown summary for the real-data scorecard."""
    component = payload.get("realdata_scorecard") or {}
    rows = component.get("components") or {}
    lines = [f"# {title}", "", "## Overview", ""]
    lines.append(f"- `status`: `{component.get('status', 'unknown')}`")
    lines.append(f"- `best_surface`: `{component.get('best_surface', 'none')}`")
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
