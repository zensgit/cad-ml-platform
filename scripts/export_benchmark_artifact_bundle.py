#!/usr/bin/env python3
"""Export a compact benchmark artifact bundle manifest and markdown summary."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.append(str(Path(__file__).resolve().parents[1]))


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


def _write_output(path_text: str, content: str) -> None:
    output_path = Path(path_text).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def _artifact_row(
    *,
    name: str,
    path_text: str,
    payload: Dict[str, Any],
    fallback_status: str = "missing",
) -> Dict[str, Any]:
    path_value = str(path_text or "").strip()
    return {
        "name": name,
        "path": path_value,
        "present": bool(payload) or bool(path_value),
        "status": str(payload.get("overall_status") or payload.get("status") or fallback_status),
    }


def _component_statuses(
    scorecard: Dict[str, Any],
    operational_summary: Dict[str, Any],
    benchmark_companion_summary: Dict[str, Any],
) -> Dict[str, str]:
    components = scorecard.get("components") or {}
    companion_components = benchmark_companion_summary.get("component_statuses") or {}
    component_rows = {
        "hybrid": str(
            companion_components.get("hybrid")
            or (components.get("hybrid") or {}).get("status")
            or "unknown"
        ),
        "history_sequence": str(
            (components.get("history_sequence") or {}).get("status") or "unknown"
        ),
        "brep": str((components.get("brep") or {}).get("status") or "unknown"),
        "migration_governance": str(
            (components.get("migration_governance") or {}).get("status") or "unknown"
        ),
    }
    operational_components = operational_summary.get("component_statuses") or {}
    component_rows["feedback_flywheel"] = str(
        operational_components.get("feedback_flywheel")
        or (components.get("feedback_flywheel") or {}).get("status")
        or "unknown"
    )
    component_rows["assistant_explainability"] = str(
        companion_components.get("assistant_explainability")
        or operational_components.get("assistant_explainability")
        or (components.get("assistant_explainability") or {}).get("status")
        or "unknown"
    )
    component_rows["review_queue"] = str(
        companion_components.get("review_queue")
        or operational_components.get("review_queue")
        or (components.get("review_queue") or {}).get("status")
        or "unknown"
    )
    component_rows["ocr_review"] = str(
        companion_components.get("ocr_review")
        or operational_components.get("ocr_review")
        or (components.get("ocr_review") or {}).get("status")
        or "unknown"
    )
    component_rows["qdrant_backend"] = str(
        companion_components.get("qdrant_backend")
        or (components.get("qdrant_backend") or {}).get("status")
        or "unknown"
    )
    return component_rows


def _compact_list(items: Iterable[Any]) -> List[str]:
    return [str(item).strip() for item in items if str(item).strip()]


def _pick_summary_items(
    benchmark_companion_summary: Dict[str, Any],
    benchmark_operational_summary: Dict[str, Any],
    benchmark_scorecard: Dict[str, Any],
) -> tuple[str, List[str], List[str]]:
    overall_status = (
        str(benchmark_companion_summary.get("overall_status") or "").strip()
        or str(benchmark_operational_summary.get("overall_status") or "").strip()
        or str(benchmark_scorecard.get("overall_status") or "").strip()
        or "unknown"
    )
    blockers = _compact_list(
        benchmark_companion_summary.get("blockers")
        or benchmark_operational_summary.get("blockers")
        or []
    )
    recommendations = _compact_list(
        benchmark_companion_summary.get("recommended_actions")
        or benchmark_operational_summary.get("recommendations")
        or benchmark_scorecard.get("recommendations")
        or []
    )
    return overall_status, blockers, recommendations


def build_bundle(
    *,
    title: str,
    benchmark_scorecard: Dict[str, Any],
    benchmark_operational_summary: Dict[str, Any],
    benchmark_companion_summary: Dict[str, Any],
    feedback_flywheel: Dict[str, Any],
    assistant_evidence: Dict[str, Any],
    review_queue: Dict[str, Any],
    ocr_review: Dict[str, Any],
    artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
    overall_status, blockers, recommendations = _pick_summary_items(
        benchmark_companion_summary,
        benchmark_operational_summary,
        benchmark_scorecard,
    )
    artifact_rows = {
        "benchmark_scorecard": _artifact_row(
            name="benchmark_scorecard",
            path_text=artifact_paths.get("benchmark_scorecard", ""),
            payload=benchmark_scorecard,
        ),
        "benchmark_operational_summary": _artifact_row(
            name="benchmark_operational_summary",
            path_text=artifact_paths.get("benchmark_operational_summary", ""),
            payload=benchmark_operational_summary,
        ),
        "benchmark_companion_summary": _artifact_row(
            name="benchmark_companion_summary",
            path_text=artifact_paths.get("benchmark_companion_summary", ""),
            payload=benchmark_companion_summary,
        ),
        "feedback_flywheel": _artifact_row(
            name="feedback_flywheel",
            path_text=artifact_paths.get("feedback_flywheel", ""),
            payload=feedback_flywheel,
        ),
        "assistant_evidence": _artifact_row(
            name="assistant_evidence",
            path_text=artifact_paths.get("assistant_evidence", ""),
            payload=assistant_evidence,
        ),
        "review_queue": _artifact_row(
            name="review_queue",
            path_text=artifact_paths.get("review_queue", ""),
            payload=review_queue,
        ),
        "ocr_review": _artifact_row(
            name="ocr_review",
            path_text=artifact_paths.get("ocr_review", ""),
            payload=ocr_review,
        ),
    }
    available_count = len([row for row in artifact_rows.values() if row["present"]])
    return {
        "title": title,
        "generated_at": int(time.time()),
        "overall_status": overall_status,
        "available_artifact_count": available_count,
        "component_statuses": _component_statuses(
            benchmark_scorecard,
            benchmark_operational_summary,
            benchmark_companion_summary,
        ),
        "blockers": blockers,
        "recommendations": recommendations,
        "artifacts": artifact_rows,
    }


def render_markdown(payload: Dict[str, Any]) -> str:
    components = payload.get("component_statuses") or {}
    artifacts = payload.get("artifacts") or {}
    lines = [
        f"# {payload.get('title') or 'Benchmark Artifact Bundle'}",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- overall_status: `{payload.get('overall_status')}`",
        f"- available_artifact_count: `{payload.get('available_artifact_count')}`",
        "",
        "## Component Statuses",
        "",
    ]
    for name, status in components.items():
        lines.append(f"- `{name}`: `{status}`")
    lines.extend(["", "## Artifacts", ""])
    for name, row in artifacts.items():
        lines.append(
            f"- `{name}`: present=`{row.get('present')}` status=`{row.get('status')}` "
            f"path=`{row.get('path')}`"
        )
    blockers = payload.get("blockers") or []
    recommendations = payload.get("recommendations") or []
    lines.extend(["", "## Blockers", ""])
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- none")
    lines.extend(["", "## Recommendations", ""])
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a compact benchmark artifact bundle."
    )
    parser.add_argument("--title", default="Benchmark Artifact Bundle")
    parser.add_argument("--benchmark-scorecard", default="")
    parser.add_argument("--benchmark-operational-summary", default="")
    parser.add_argument("--benchmark-companion-summary", default="")
    parser.add_argument("--feedback-flywheel", default="")
    parser.add_argument("--assistant-evidence", default="")
    parser.add_argument("--review-queue", default="")
    parser.add_argument("--ocr-review", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    artifact_paths = {
        "benchmark_scorecard": args.benchmark_scorecard,
        "benchmark_operational_summary": args.benchmark_operational_summary,
        "benchmark_companion_summary": args.benchmark_companion_summary,
        "feedback_flywheel": args.feedback_flywheel,
        "assistant_evidence": args.assistant_evidence,
        "review_queue": args.review_queue,
        "ocr_review": args.ocr_review,
    }
    payload = build_bundle(
        title=args.title,
        benchmark_scorecard=_maybe_load_json(args.benchmark_scorecard),
        benchmark_operational_summary=_maybe_load_json(args.benchmark_operational_summary),
        benchmark_companion_summary=_maybe_load_json(args.benchmark_companion_summary),
        feedback_flywheel=_maybe_load_json(args.feedback_flywheel),
        assistant_evidence=_maybe_load_json(args.assistant_evidence),
        review_queue=_maybe_load_json(args.review_queue),
        ocr_review=_maybe_load_json(args.ocr_review),
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
