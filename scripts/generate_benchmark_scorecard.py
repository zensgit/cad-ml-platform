#!/usr/bin/env python3
"""Generate a unified benchmark scorecard from existing validation artifacts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path_text: str) -> Dict[str, Any]:
    path = Path(path_text).expanduser()
    if not path.exists():
        raise SystemExit(f"JSON input not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected object JSON in {path}")
    return payload


def _maybe_load_json(path_text: str) -> Dict[str, Any]:
    if not str(path_text or "").strip():
        return {}
    return _load_json(path_text)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _branch_accuracy(summary: Dict[str, Any], branch: str, *, exact: bool) -> float:
    key = "exact_accuracy" if exact else "coarse_accuracy"
    payload = summary.get(key, {}) or {}
    if not isinstance(payload, dict):
        return 0.0
    branch_payload = payload.get(branch, {}) or {}
    if not isinstance(branch_payload, dict):
        return 0.0
    return _to_float(branch_payload.get("accuracy"))


def _confidence_low_conf(summary: Dict[str, Any], branch: str) -> float:
    payload = summary.get("confidence", {}) or {}
    if not isinstance(payload, dict):
        return 0.0
    branch_payload = payload.get(branch, {}) or {}
    if not isinstance(branch_payload, dict):
        return 0.0
    return _to_float(branch_payload.get("low_conf_rate"))


def _graph2d_status(
    metrics: Dict[str, Any],
    diagnose: Dict[str, Any],
    diagnose_blind: Dict[str, Any],
) -> Dict[str, Any]:
    best_val_acc = _to_float(metrics.get("best_val_acc"))
    default_acc = _to_float(diagnose.get("accuracy"))
    blind_acc = _to_float(diagnose_blind.get("accuracy"))
    default_low_conf = _to_float(diagnose.get("low_conf_rate"))
    if default_acc < 0.2 or default_low_conf >= 0.9:
        status = "weak_signal_only"
    elif default_acc < 0.5:
        status = "diagnostic_only"
    else:
        status = "candidate_primary_signal"
    return {
        "status": status,
        "best_val_acc": round(best_val_acc, 6),
        "diagnose_accuracy": round(default_acc, 6),
        "blind_accuracy": round(blind_acc, 6),
        "low_conf_rate": round(default_low_conf, 6),
    }


def _hybrid_status(summary: Dict[str, Any]) -> Dict[str, Any]:
    hybrid_exact = _branch_accuracy(summary, "hybrid_label", exact=True)
    hybrid_coarse = _branch_accuracy(summary, "hybrid_label", exact=False)
    final_exact = _branch_accuracy(summary, "final_part_type", exact=True)
    graph2d_exact = _branch_accuracy(summary, "graph2d_label", exact=True)
    output_gap = max(hybrid_exact - final_exact, 0.0)
    if hybrid_exact >= 0.8:
        status = "strong_primary"
    elif hybrid_exact >= 0.6:
        status = "usable_primary"
    else:
        status = "insufficient_primary"
    return {
        "status": status,
        "sample_size": _to_int(summary.get("sample_size")),
        "hybrid_exact_accuracy": round(hybrid_exact, 6),
        "hybrid_coarse_accuracy": round(hybrid_coarse, 6),
        "final_exact_accuracy": round(final_exact, 6),
        "graph2d_exact_accuracy": round(graph2d_exact, 6),
        "graph2d_low_conf_rate": round(
            _confidence_low_conf(summary, "graph2d_label"),
            6,
        ),
        "output_normalization_gap": round(output_gap, 6),
        "has_output_gap": output_gap >= 0.15,
    }


def _history_status(summary: Dict[str, Any]) -> Dict[str, Any]:
    total = _to_int(summary.get("total"))
    coverage = _to_float(summary.get("coverage"))
    coarse_acc = _to_float(summary.get("coarse_accuracy_overall"))
    exact_acc = _to_float(summary.get("accuracy_overall"))
    low_conf = _to_float(summary.get("low_conf_rate"))
    if total == 0:
        status = "missing"
    elif total < 10:
        status = "smoke_only"
    elif coarse_acc >= 0.8 and coverage >= 0.6:
        status = "evidence_ready"
    else:
        status = "needs_more_evidence"
    return {
        "status": status,
        "sample_size": total,
        "coverage": round(coverage, 6),
        "exact_accuracy_overall": round(exact_acc, 6),
        "coarse_accuracy_overall": round(coarse_acc, 6),
        "low_conf_rate": round(low_conf, 6),
    }


def _brep_status(summary: Dict[str, Any]) -> Dict[str, Any]:
    sample_size = _to_int(summary.get("sample_size"))
    valid = _to_int(summary.get("valid_3d_count"))
    hints = _to_int(summary.get("hint_coverage_count"))
    schema_counts = summary.get("graph_schema_version_counts", {}) or {}
    has_v2 = bool(_to_int(schema_counts.get("v2")))
    if sample_size == 0:
        status = "missing"
    elif valid == 0:
        status = "prep_only"
    elif has_v2:
        status = "graph_ready"
    else:
        status = "legacy_graph"
    return {
        "status": status,
        "sample_size": sample_size,
        "valid_3d_count": valid,
        "hint_coverage_count": hints,
        "graph_schema_version_counts": schema_counts,
    }


def _governance_status(summary: Dict[str, Any]) -> Dict[str, Any]:
    if not summary:
        return {"status": "missing"}
    plan_ready = bool(summary.get("plan_ready"))
    coverage_complete = bool(summary.get("coverage_complete"))
    blocking = summary.get("blocking_reasons") or []
    if plan_ready and coverage_complete:
        status = "operationally_ready"
    elif plan_ready:
        status = "partially_ready"
    else:
        status = "blocked"
    return {
        "status": status,
        "plan_ready": plan_ready,
        "coverage_complete": coverage_complete,
        "blocking_reasons": list(blocking) if isinstance(blocking, list) else [],
        "recommended_from_versions": list(summary.get("recommended_from_versions") or []),
        "planned_pending_ratio": summary.get("planned_pending_ratio"),
        "estimated_total_runs": _to_int(summary.get("estimated_total_runs")),
    }


def _overall_status(
    hybrid: Dict[str, Any],
    history: Dict[str, Any],
    brep: Dict[str, Any],
    governance: Dict[str, Any],
) -> str:
    if hybrid.get("status") not in {"strong_primary", "usable_primary"}:
        return "baseline_not_ready"
    if governance.get("status") not in {"operationally_ready", "partially_ready"}:
        return "benchmark_ready_without_governance"
    if history.get("status") in {"missing", "smoke_only", "needs_more_evidence"}:
        return "benchmark_ready_with_history_gap"
    if brep.get("status") in {"missing", "prep_only"}:
        return "benchmark_ready_with_3d_gap"
    return "benchmark_ready_with_multisignal_evidence"


def _recommendations(
    hybrid: Dict[str, Any],
    graph2d: Dict[str, Any],
    history: Dict[str, Any],
    brep: Dict[str, Any],
    governance: Dict[str, Any],
) -> List[str]:
    items: List[str] = []
    if graph2d.get("status") == "weak_signal_only":
        items.append("Keep Graph2D behind hybrid gating and review prioritization.")
    if hybrid.get("has_output_gap"):
        items.append("Prefer coarse/fine hybrid outputs over legacy final_part_type.")
    if history.get("status") in {"missing", "smoke_only", "needs_more_evidence"}:
        items.append("Collect labeled .h5 manifests and expand history benchmark runs.")
    if brep.get("status") in {"missing", "prep_only"}:
        items.append("Run STEP/B-Rep evaluation under OCC-enabled environment.")
    if governance.get("status") == "blocked":
        items.append("Resolve migration blocking reasons before production rollout.")
    if not items:
        items.append(
            "Current scorecard is healthy; freeze this run as the next benchmark baseline."
        )
    return items


def build_scorecard(
    *,
    title: str,
    hybrid_summary: Dict[str, Any],
    graph2d_metrics: Dict[str, Any],
    graph2d_diagnose: Dict[str, Any],
    graph2d_blind_diagnose: Dict[str, Any],
    history_summary: Dict[str, Any],
    brep_summary: Dict[str, Any],
    migration_summary: Dict[str, Any],
) -> Dict[str, Any]:
    hybrid = _hybrid_status(hybrid_summary)
    graph2d = _graph2d_status(
        graph2d_metrics,
        graph2d_diagnose,
        graph2d_blind_diagnose,
    )
    history = _history_status(history_summary)
    brep = _brep_status(brep_summary)
    governance = _governance_status(migration_summary)
    overall_status = _overall_status(hybrid, history, brep, governance)
    return {
        "title": title,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall_status": overall_status,
        "components": {
            "hybrid": hybrid,
            "graph2d": graph2d,
            "history_sequence": history,
            "brep": brep,
            "migration_governance": governance,
        },
        "recommendations": _recommendations(
            hybrid,
            graph2d,
            history,
            brep,
            governance,
        ),
    }


def _render_markdown(scorecard: Dict[str, Any]) -> str:
    components = scorecard.get("components", {}) or {}
    lines = [
        f"# {scorecard.get('title') or 'Benchmark Scorecard'}",
        "",
        f"- generated_at: `{scorecard.get('generated_at')}`",
        f"- overall_status: `{scorecard.get('overall_status')}`",
        "",
        "## Components",
        "",
        "| Component | Status | Key metrics |",
        "| --- | --- | --- |",
    ]
    hybrid = components.get("hybrid", {}) or {}
    lines.append(
        "| hybrid | "
        f"`{hybrid.get('status')}` | "
        f"exact={hybrid.get('hybrid_exact_accuracy')}, "
        f"coarse={hybrid.get('hybrid_coarse_accuracy')}, "
        f"gap={hybrid.get('output_normalization_gap')} |"
    )
    graph2d = components.get("graph2d", {}) or {}
    lines.append(
        "| graph2d | "
        f"`{graph2d.get('status')}` | "
        f"best_val={graph2d.get('best_val_acc')}, "
        f"diag={graph2d.get('diagnose_accuracy')}, "
        f"low_conf={graph2d.get('low_conf_rate')} |"
    )
    history = components.get("history_sequence", {}) or {}
    lines.append(
        "| history_sequence | "
        f"`{history.get('status')}` | "
        f"sample={history.get('sample_size')}, "
        f"coverage={history.get('coverage')}, "
        f"coarse_acc={history.get('coarse_accuracy_overall')} |"
    )
    brep = components.get("brep", {}) or {}
    lines.append(
        "| brep | "
        f"`{brep.get('status')}` | "
        f"sample={brep.get('sample_size')}, "
        f"valid_3d={brep.get('valid_3d_count')}, "
        f"hints={brep.get('hint_coverage_count')} |"
    )
    governance = components.get("migration_governance", {}) or {}
    lines.append(
        "| migration_governance | "
        f"`{governance.get('status')}` | "
        f"plan_ready={governance.get('plan_ready')}, "
        f"coverage_complete={governance.get('coverage_complete')}, "
        f"estimated_runs={governance.get('estimated_total_runs')} |"
    )
    lines.extend(
        [
            "",
            "## Recommendations",
            "",
        ]
    )
    for item in scorecard.get("recommendations", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a unified benchmark scorecard from validation artifacts."
    )
    parser.add_argument("--title", default="Benchmark Scorecard")
    parser.add_argument("--hybrid-summary", default="")
    parser.add_argument("--graph2d-metrics", default="")
    parser.add_argument("--graph2d-diagnose", default="")
    parser.add_argument("--graph2d-blind-diagnose", default="")
    parser.add_argument("--history-summary", default="")
    parser.add_argument("--brep-summary", default="")
    parser.add_argument("--migration-summary", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args(argv)

    if not args.output_json and not args.output_md:
        raise SystemExit("At least one of --output-json or --output-md is required")

    scorecard = build_scorecard(
        title=str(args.title),
        hybrid_summary=_maybe_load_json(args.hybrid_summary),
        graph2d_metrics=_maybe_load_json(args.graph2d_metrics),
        graph2d_diagnose=_maybe_load_json(args.graph2d_diagnose),
        graph2d_blind_diagnose=_maybe_load_json(args.graph2d_blind_diagnose),
        history_summary=_maybe_load_json(args.history_summary),
        brep_summary=_maybe_load_json(args.brep_summary),
        migration_summary=_maybe_load_json(args.migration_summary),
    )

    if args.output_json:
        output_json = Path(args.output_json).expanduser()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(scorecard, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.output_md:
        output_md = Path(args.output_md).expanduser()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_render_markdown(scorecard), encoding="utf-8")

    print(json.dumps(scorecard, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
