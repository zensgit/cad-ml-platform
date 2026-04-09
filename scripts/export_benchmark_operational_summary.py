#!/usr/bin/env python3
"""Export a compact operational benchmark summary from existing artifacts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


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


def _benchmark_component(scorecard: Dict[str, Any], name: str) -> Dict[str, Any]:
    components = scorecard.get("components") or {}
    if not isinstance(components, dict):
        return {}
    value = components.get(name) or {}
    return value if isinstance(value, dict) else {}


def _resolve_assistant_status(
    scorecard: Dict[str, Any],
    assistant_summary: Dict[str, Any],
) -> str:
    component = _benchmark_component(scorecard, "assistant_explainability")
    status = str(component.get("status") or "").strip()
    if status:
        return status
    total_records = _to_int(assistant_summary.get("total_records"))
    evidence_cov = _to_float(
        ((assistant_summary.get("coverage") or {}).get("records_with_evidence") or {}).get(
            "coverage_pct"
        ),
        default=_to_float(assistant_summary.get("records_with_evidence_pct")),
    )
    if total_records <= 0:
        return "missing"
    if evidence_cov >= 0.8:
        return "explainability_ready"
    return "partial_coverage"


def _resolve_review_queue_status(
    scorecard: Dict[str, Any],
    review_queue_summary: Dict[str, Any],
) -> str:
    component = _benchmark_component(scorecard, "review_queue")
    status = str(component.get("status") or "").strip()
    if status:
        return status
    return str(review_queue_summary.get("operational_status") or "missing").strip()


def _resolve_ocr_status(
    scorecard: Dict[str, Any],
    ocr_summary: Dict[str, Any],
) -> str:
    component = _benchmark_component(scorecard, "ocr_review")
    status = str(component.get("status") or "").strip()
    if status:
        return status
    if not ocr_summary:
        return "missing"
    review_candidates = _to_int(ocr_summary.get("review_candidate_count"))
    automation_ready = _to_int(ocr_summary.get("automation_ready_count"))
    if review_candidates <= 0:
        return "ocr_ready"
    if automation_ready > 0:
        return "managed_review"
    return "review_heavy"


def _resolve_feedback_status(
    scorecard: Dict[str, Any],
    feedback_summary: Dict[str, Any],
) -> str:
    component = _benchmark_component(scorecard, "feedback_flywheel")
    status = str(component.get("status") or "").strip()
    if status:
        return status
    inner = feedback_summary.get("feedback_flywheel") or {}
    if isinstance(inner, dict):
        return str(inner.get("status") or "missing").strip()
    return "missing"


def _resolve_operator_adoption_status(
    scorecard: Dict[str, Any],
    operator_adoption_summary: Dict[str, Any],
) -> str:
    component = _benchmark_component(scorecard, "operator_adoption")
    status = str(component.get("status") or "").strip()
    if status:
        return status
    return str(operator_adoption_summary.get("adoption_readiness") or "missing").strip()


def build_operational_summary(
    *,
    title: str,
    benchmark_scorecard: Dict[str, Any],
    feedback_flywheel: Dict[str, Any],
    assistant_evidence: Dict[str, Any],
    review_queue: Dict[str, Any],
    ocr_review: Dict[str, Any],
    benchmark_operator_adoption: Dict[str, Any],
    artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
    overall_status = str(benchmark_scorecard.get("overall_status") or "").strip() or "unknown"
    feedback_status = _resolve_feedback_status(benchmark_scorecard, feedback_flywheel)
    assistant_status = _resolve_assistant_status(benchmark_scorecard, assistant_evidence)
    review_queue_status = _resolve_review_queue_status(benchmark_scorecard, review_queue)
    ocr_status = _resolve_ocr_status(benchmark_scorecard, ocr_review)
    operator_adoption_status = _resolve_operator_adoption_status(
        benchmark_scorecard,
        benchmark_operator_adoption,
    )

    blockers: List[str] = []
    if feedback_status in {"missing", "passive_feedback_only", "feedback_collected"}:
        blockers.append(f"feedback_flywheel:{feedback_status}")
    if assistant_status in {"missing", "partial_coverage", "weak_coverage"}:
        blockers.append(f"assistant_explainability:{assistant_status}")
    if review_queue_status in {"critical_backlog", "managed_backlog", "evidence_gap"}:
        blockers.append(f"review_queue:{review_queue_status}")
    if ocr_status in {"missing", "review_heavy"}:
        blockers.append(f"ocr_review:{ocr_status}")
    if operator_adoption_status in {"missing", "unknown", "guided_manual", "blocked"}:
        blockers.append(f"operator_adoption:{operator_adoption_status}")

    recommendations = benchmark_scorecard.get("recommendations") or []
    if not isinstance(recommendations, list):
        recommendations = []
    recommendations = list(recommendations)
    for item in benchmark_operator_adoption.get("recommended_actions") or []:
        text = str(item).strip()
        if text and text not in recommendations:
            recommendations.append(text)

    payload = {
        "title": title,
        "generated_at": int(time.time()),
        "overall_status": overall_status,
        "component_statuses": {
            "feedback_flywheel": feedback_status,
            "assistant_explainability": assistant_status,
            "review_queue": review_queue_status,
            "ocr_review": ocr_status,
            "operator_adoption": operator_adoption_status,
        },
        "key_metrics": {
            "feedback_total": _to_int(
                ((feedback_flywheel.get("feedback_flywheel") or {}).get("feedback_total"))
            ),
            "feedback_corrections": _to_int(
                ((feedback_flywheel.get("feedback_flywheel") or {}).get("correction_count"))
            ),
            "assistant_records": _to_int(assistant_evidence.get("total_records")),
            "review_queue_total": _to_int(review_queue.get("total")),
            "ocr_review_candidates": _to_int(ocr_review.get("review_candidate_count")),
            "operator_blocking_signal_count": _to_int(
                len(benchmark_operator_adoption.get("blocking_signals") or [])
            ),
            "operator_recommended_action_count": _to_int(
                len(benchmark_operator_adoption.get("recommended_actions") or [])
            ),
        },
        "blockers": blockers,
        "artifact_paths": artifact_paths,
        "recommendations": recommendations[:5],
        "operator_adoption_knowledge_outcome_drift_status": str(
            benchmark_operator_adoption.get("knowledge_outcome_drift_status") or "unknown"
        ),
        "operator_adoption_knowledge_outcome_drift_summary": str(
            benchmark_operator_adoption.get("knowledge_outcome_drift_summary") or "none"
        ),
    }
    return payload


def render_markdown(payload: Dict[str, Any]) -> str:
    component_statuses = payload.get("component_statuses") or {}
    key_metrics = payload.get("key_metrics") or {}
    artifact_paths = payload.get("artifact_paths") or {}
    recommendations = payload.get("recommendations") or []
    lines = [
        f"# {payload.get('title', 'Benchmark Operational Summary')}",
        "",
        "## Overall",
        "",
        f"- `overall_status`: `{payload.get('overall_status', 'unknown')}`",
        f"- `blockers`: `{payload.get('blockers', [])}`",
        "- `operator_adoption_knowledge_outcome_drift_status`: "
        f"`{payload.get('operator_adoption_knowledge_outcome_drift_status', 'unknown')}`",
        "- `operator_adoption_knowledge_outcome_drift_summary`: "
        f"{payload.get('operator_adoption_knowledge_outcome_drift_summary', 'none')}",
        "",
        "## Component Statuses",
        "",
    ]
    for name, status in component_statuses.items():
        lines.append(f"- `{name}`: `{status}`")
    lines.extend(["", "## Key Metrics", ""])
    for name, value in key_metrics.items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(["", "## Artifacts", ""])
    for name, value in artifact_paths.items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(["", "## Recommendations", ""])
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- No additional recommendations.")
    lines.append("")
    return "\n".join(lines)


def _write_output(path_text: str, content: str) -> None:
    output_path = Path(path_text).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export benchmark operational summary.")
    parser.add_argument("--title", default="Benchmark Operational Summary")
    parser.add_argument("--benchmark-scorecard", default="")
    parser.add_argument("--feedback-flywheel", default="")
    parser.add_argument("--assistant-evidence", default="")
    parser.add_argument("--review-queue", default="")
    parser.add_argument("--ocr-review", default="")
    parser.add_argument("--benchmark-operator-adoption", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    artifact_paths = {
        "benchmark_scorecard": args.benchmark_scorecard,
        "feedback_flywheel": args.feedback_flywheel,
        "assistant_evidence": args.assistant_evidence,
        "review_queue": args.review_queue,
        "ocr_review": args.ocr_review,
        "benchmark_operator_adoption": args.benchmark_operator_adoption,
    }
    payload = build_operational_summary(
        title=args.title,
        benchmark_scorecard=_maybe_load_json(args.benchmark_scorecard),
        feedback_flywheel=_maybe_load_json(args.feedback_flywheel),
        assistant_evidence=_maybe_load_json(args.assistant_evidence),
        review_queue=_maybe_load_json(args.review_queue),
        ocr_review=_maybe_load_json(args.ocr_review),
        benchmark_operator_adoption=_maybe_load_json(args.benchmark_operator_adoption),
        artifact_paths=artifact_paths,
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        _write_output(args.output_json, rendered + "\n")
    if args.output_md:
        _write_output(args.output_md, render_markdown(payload))
    print(rendered)


if __name__ == "__main__":
    main()
