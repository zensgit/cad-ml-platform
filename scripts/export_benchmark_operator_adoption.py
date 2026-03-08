#!/usr/bin/env python3
"""Export a benchmark operator-adoption summary."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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


def _compact(items: Iterable[Any], *, limit: int = 6) -> List[str]:
    out: List[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _artifact_row(name: str, path_text: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    path_value = str(path_text or "").strip()
    return {
        "name": name,
        "path": path_value,
        "present": bool(path_value) or bool(payload),
    }


def _pick_status(
    benchmark_release_decision: Dict[str, Any],
    benchmark_release_runbook: Dict[str, Any],
    review_queue: Dict[str, Any],
    feedback_flywheel: Dict[str, Any],
) -> Dict[str, str]:
    return {
        "release_decision": str(
            benchmark_release_decision.get("release_status") or "unknown"
        ).strip()
        or "unknown",
        "release_runbook": str(
            benchmark_release_runbook.get("release_status") or "unknown"
        ).strip()
        or "unknown",
        "review_queue": str(review_queue.get("operational_status") or "unknown").strip()
        or "unknown",
        "feedback_flywheel": str(feedback_flywheel.get("status") or "unknown").strip()
        or "unknown",
    }


def _knowledge_drift_payload(
    benchmark_release_decision: Dict[str, Any],
    benchmark_release_runbook: Dict[str, Any],
    benchmark_knowledge_drift: Dict[str, Any],
) -> Dict[str, Any]:
    component = (
        benchmark_release_runbook.get("knowledge_drift")
        or benchmark_release_decision.get("knowledge_drift")
        or benchmark_knowledge_drift.get("knowledge_drift")
        or benchmark_knowledge_drift
        or {}
    )
    counts = component.get("counts") or {}
    return {
        "status": (
            _text(component.get("status"))
            or _text(benchmark_release_runbook.get("knowledge_drift_status"))
            or _text(benchmark_release_decision.get("knowledge_drift_status"))
            or "unknown"
        ),
        "summary": (
            _text(component.get("summary"))
            or _text(benchmark_release_runbook.get("knowledge_drift_summary"))
            or _text(benchmark_release_decision.get("knowledge_drift_summary"))
        ),
        "recommendations": _compact(
            component.get("recommendations")
            or benchmark_knowledge_drift.get("recommendations")
            or [],
            limit=4,
        ),
        "counts": {
            "regressions": int(counts.get("regressions") or 0),
            "improvements": int(counts.get("improvements") or 0),
            "new_focus_areas": int(counts.get("new_focus_areas") or 0),
            "resolved_focus_areas": int(counts.get("resolved_focus_areas") or 0),
        },
    }


def _adoption_readiness(
    statuses: Dict[str, str],
    freeze_ready: bool,
    blockers: List[str],
    review_signals: List[str],
    knowledge_drift: Dict[str, Any],
) -> str:
    if blockers or statuses["release_decision"] == "blocked":
        return "blocked"
    if knowledge_drift.get("status") == "regressed":
        return "guided_manual"
    if freeze_ready and not review_signals and statuses["review_queue"] not in {
        "critical_backlog",
        "managed_backlog",
    }:
        return "operator_ready"
    return "guided_manual"


def _operator_mode(
    statuses: Dict[str, str],
    next_action: str,
    blockers: List[str],
    review_signals: List[str],
    freeze_ready: bool,
    knowledge_drift: Dict[str, Any],
) -> str:
    if blockers or statuses["release_decision"] == "blocked":
        return "clear_blockers"
    if knowledge_drift.get("status") == "regressed":
        return "stabilize_knowledge"
    if next_action == "collect_artifacts":
        return "stabilize_inputs"
    if next_action == "review_signals" or review_signals:
        return "drive_review"
    if freeze_ready:
        return "freeze_ready"
    return "monitor"


def _recommended_actions(
    benchmark_release_runbook: Dict[str, Any],
    blockers: List[str],
    review_signals: List[str],
    review_queue: Dict[str, Any],
    knowledge_drift: Dict[str, Any],
) -> List[str]:
    actions: List[str] = []
    for item in knowledge_drift.get("recommendations") or []:
        if item and item not in actions:
            actions.append(str(item))
    for step in benchmark_release_runbook.get("operator_steps") or []:
        if not isinstance(step, dict):
            continue
        if str(step.get("status") or "").strip() in {"required", "blocked"}:
            action = str(step.get("action") or "").strip()
            if action and action not in actions:
                actions.append(action)
    if not actions and blockers:
        actions.extend(blockers[:2])
    if not actions and review_signals:
        actions.extend(review_signals[:2])
    if not actions:
        queue_status = str(review_queue.get("operational_status") or "").strip()
        if queue_status in {"critical_backlog", "managed_backlog"}:
            actions.append("Drain the active-learning review queue before promotion.")
    if not actions:
        actions.append("Continue benchmark monitoring and rerun evaluation on new data.")
    return _compact(actions, limit=4)


def build_operator_adoption(
    *,
    title: str,
    benchmark_release_decision: Dict[str, Any],
    benchmark_release_runbook: Dict[str, Any],
    review_queue: Dict[str, Any],
    feedback_flywheel: Dict[str, Any],
    benchmark_knowledge_drift: Dict[str, Any],
    artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
    statuses = _pick_status(
        benchmark_release_decision,
        benchmark_release_runbook,
        review_queue,
        feedback_flywheel,
    )
    blockers = _compact(
        benchmark_release_runbook.get("blocking_signals")
        or benchmark_release_decision.get("blocking_signals")
        or [],
        limit=6,
    )
    review_signals = _compact(
        benchmark_release_runbook.get("review_signals")
        or benchmark_release_decision.get("review_signals")
        or [],
        limit=6,
    )
    next_action = (
        str(benchmark_release_runbook.get("next_action") or "unknown").strip()
        or "unknown"
    )
    freeze_ready = bool(benchmark_release_runbook.get("ready_to_freeze_baseline"))
    automation_ready = bool(benchmark_release_decision.get("automation_ready"))
    critical_count = int(review_queue.get("critical_count") or 0)
    high_count = int(review_queue.get("high_count") or 0)
    correction_count = int(feedback_flywheel.get("correction_count") or 0)
    feedback_total = int(feedback_flywheel.get("feedback_total") or 0)
    knowledge_drift = _knowledge_drift_payload(
        benchmark_release_decision,
        benchmark_release_runbook,
        benchmark_knowledge_drift,
    )

    adoption_readiness = _adoption_readiness(
        statuses,
        freeze_ready,
        blockers,
        review_signals,
        knowledge_drift,
    )
    operator_mode = _operator_mode(
        statuses,
        next_action,
        blockers,
        review_signals,
        freeze_ready,
        knowledge_drift,
    )
    recommended_actions = _recommended_actions(
        benchmark_release_runbook,
        blockers,
        review_signals,
        review_queue,
        knowledge_drift,
    )
    artifacts = {
        "benchmark_release_decision": _artifact_row(
            "benchmark_release_decision",
            artifact_paths.get("benchmark_release_decision", ""),
            benchmark_release_decision,
        ),
        "benchmark_release_runbook": _artifact_row(
            "benchmark_release_runbook",
            artifact_paths.get("benchmark_release_runbook", ""),
            benchmark_release_runbook,
        ),
        "review_queue": _artifact_row(
            "review_queue",
            artifact_paths.get("review_queue", ""),
            review_queue,
        ),
        "feedback_flywheel": _artifact_row(
            "feedback_flywheel",
            artifact_paths.get("feedback_flywheel", ""),
            feedback_flywheel,
        ),
        "benchmark_knowledge_drift": _artifact_row(
            "benchmark_knowledge_drift",
            artifact_paths.get("benchmark_knowledge_drift", ""),
            benchmark_knowledge_drift,
        ),
    }

    return {
        "title": title,
        "generated_at": int(time.time()),
        "adoption_readiness": adoption_readiness,
        "operator_mode": operator_mode,
        "next_action": next_action,
        "automation_ready": automation_ready,
        "freeze_ready": freeze_ready,
        "statuses": statuses,
        "blocking_signals": blockers,
        "review_signals": review_signals,
        "review_queue_critical_count": critical_count,
        "review_queue_high_count": high_count,
        "feedback_total": feedback_total,
        "correction_count": correction_count,
        "knowledge_drift_status": knowledge_drift["status"],
        "knowledge_drift_summary": knowledge_drift["summary"],
        "knowledge_drift": knowledge_drift,
        "recommended_actions": recommended_actions,
        "artifacts": artifacts,
    }


def render_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        f"# {payload.get('title') or 'Benchmark Operator Adoption'}",
        "",
        f"- `adoption_readiness`: `{payload.get('adoption_readiness')}`",
        f"- `operator_mode`: `{payload.get('operator_mode')}`",
        f"- `next_action`: `{payload.get('next_action')}`",
        f"- `automation_ready`: `{payload.get('automation_ready')}`",
        f"- `freeze_ready`: `{payload.get('freeze_ready')}`",
        f"- `knowledge_drift_status`: `{payload.get('knowledge_drift_status')}`",
        "",
        "## Statuses",
        "",
    ]
    for name, status in (payload.get("statuses") or {}).items():
        lines.append(f"- `{name}`: `{status}`")
    lines.extend(
        [
            "",
            "## Queue / Feedback",
            "",
            f"- `review_queue_critical_count`: `{payload.get('review_queue_critical_count')}`",
            f"- `review_queue_high_count`: `{payload.get('review_queue_high_count')}`",
            f"- `feedback_total`: `{payload.get('feedback_total')}`",
            f"- `correction_count`: `{payload.get('correction_count')}`",
            "",
            "## Blocking Signals",
            "",
        ]
    )
    blockers = payload.get("blocking_signals") or []
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- none")
    lines.extend(["", "## Review Signals", ""])
    review = payload.get("review_signals") or []
    lines.extend(f"- {item}" for item in review) if review else lines.append("- none")
    lines.extend(["", "## Knowledge Drift", ""])
    lines.append(
        "- `summary`: "
        + (_text(payload.get("knowledge_drift_summary")) or "none")
    )
    drift = payload.get("knowledge_drift") or {}
    counts = drift.get("counts") or {}
    lines.append(
        "- `counts`: "
        f"regressions={counts.get('regressions', 0)} "
        f"improvements={counts.get('improvements', 0)} "
        f"new_focus_areas={counts.get('new_focus_areas', 0)} "
        f"resolved_focus_areas={counts.get('resolved_focus_areas', 0)}"
    )
    lines.extend(["", "## Recommended Actions", ""])
    actions = payload.get("recommended_actions") or []
    lines.extend(f"- {item}" for item in actions) if actions else lines.append("- none")
    lines.extend(["", "## Artifacts", ""])
    for name, row in (payload.get("artifacts") or {}).items():
        lines.append(
            f"- `{name}`: present=`{row.get('present')}` path=`{row.get('path')}`"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export benchmark operator-adoption summary."
    )
    parser.add_argument("--title", default="Benchmark Operator Adoption")
    parser.add_argument("--benchmark-release-decision", default="")
    parser.add_argument("--benchmark-release-runbook", default="")
    parser.add_argument("--review-queue", default="")
    parser.add_argument("--feedback-flywheel", default="")
    parser.add_argument("--benchmark-knowledge-drift", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    payload = build_operator_adoption(
        title=args.title,
        benchmark_release_decision=_maybe_load_json(args.benchmark_release_decision),
        benchmark_release_runbook=_maybe_load_json(args.benchmark_release_runbook),
        review_queue=_maybe_load_json(args.review_queue),
        feedback_flywheel=_maybe_load_json(args.feedback_flywheel),
        benchmark_knowledge_drift=_maybe_load_json(args.benchmark_knowledge_drift),
        artifact_paths={
            "benchmark_release_decision": args.benchmark_release_decision,
            "benchmark_release_runbook": args.benchmark_release_runbook,
            "review_queue": args.review_queue,
            "feedback_flywheel": args.feedback_flywheel,
            "benchmark_knowledge_drift": args.benchmark_knowledge_drift,
        },
    )
    _write_output(args.output_json, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _write_output(args.output_md, render_markdown(payload))


if __name__ == "__main__":
    main()
