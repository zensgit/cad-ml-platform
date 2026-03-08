#!/usr/bin/env python3
"""Export an operator-facing benchmark release runbook."""

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
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _artifact_row(name: str, path_text: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    path_value = _text(path_text)
    return {
        "name": name,
        "path": path_value,
        "present": bool(path_value) or bool(payload),
    }


def _artifacts(
    *,
    benchmark_release_decision: Dict[str, Any],
    benchmark_companion_summary: Dict[str, Any],
    benchmark_artifact_bundle: Dict[str, Any],
    benchmark_knowledge_readiness: Dict[str, Any],
    benchmark_engineering_signals: Dict[str, Any],
    benchmark_operator_adoption: Dict[str, Any],
    artifact_paths: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    decision_artifacts = benchmark_release_decision.get("artifacts") or {}

    def pick_path(name: str) -> str:
        direct = _text(artifact_paths.get(name, ""))
        if direct:
            return direct
        nested = decision_artifacts.get(name) or {}
        if isinstance(nested, dict):
            return _text(nested.get("path"))
        return ""

    return {
        "benchmark_release_decision": _artifact_row(
            "benchmark_release_decision",
            artifact_paths.get("benchmark_release_decision", ""),
            benchmark_release_decision,
        ),
        "benchmark_companion_summary": _artifact_row(
            "benchmark_companion_summary",
            pick_path("benchmark_companion_summary"),
            benchmark_companion_summary,
        ),
        "benchmark_artifact_bundle": _artifact_row(
            "benchmark_artifact_bundle",
            pick_path("benchmark_artifact_bundle"),
            benchmark_artifact_bundle,
        ),
        "benchmark_knowledge_readiness": _artifact_row(
            "benchmark_knowledge_readiness",
            artifact_paths.get("benchmark_knowledge_readiness", ""),
            benchmark_knowledge_readiness,
        ),
        "benchmark_engineering_signals": _artifact_row(
            "benchmark_engineering_signals",
            pick_path("benchmark_engineering_signals")
            or artifact_paths.get("benchmark_engineering_signals", ""),
            benchmark_engineering_signals,
        ),
        "benchmark_operator_adoption": _artifact_row(
            "benchmark_operator_adoption",
            artifact_paths.get("benchmark_operator_adoption", ""),
            benchmark_operator_adoption,
        ),
        "benchmark_scorecard": _artifact_row(
            "benchmark_scorecard",
            pick_path("benchmark_scorecard"),
            decision_artifacts.get("benchmark_scorecard") or {},
        ),
        "benchmark_operational_summary": _artifact_row(
            "benchmark_operational_summary",
            pick_path("benchmark_operational_summary"),
            decision_artifacts.get("benchmark_operational_summary") or {},
        ),
    }


def _release_status(
    benchmark_release_decision: Dict[str, Any],
    benchmark_companion_summary: Dict[str, Any],
    benchmark_artifact_bundle: Dict[str, Any],
) -> str:
    return (
        str(benchmark_release_decision.get("release_status") or "").strip()
        or str(benchmark_companion_summary.get("overall_status") or "").strip()
        or str(benchmark_artifact_bundle.get("overall_status") or "").strip()
        or "unknown"
    )


def _primary_signal_source(
    benchmark_release_decision: Dict[str, Any],
    benchmark_companion_summary: Dict[str, Any],
    benchmark_artifact_bundle: Dict[str, Any],
) -> str:
    return (
        str(benchmark_release_decision.get("primary_signal_source") or "").strip()
        or ("benchmark_companion_summary" if benchmark_companion_summary else "")
        or ("benchmark_artifact_bundle" if benchmark_artifact_bundle else "")
        or "none"
    )


def _operator_adoption_payload(
    benchmark_operator_adoption: Dict[str, Any]
) -> Dict[str, Any]:
    status = (
        _text(benchmark_operator_adoption.get("status"))
        or _text(benchmark_operator_adoption.get("overall_status"))
        or _text(benchmark_operator_adoption.get("adoption_status"))
        or ("provided" if benchmark_operator_adoption else "unknown")
    )
    summary = (
        _text(benchmark_operator_adoption.get("summary"))
        or _text(benchmark_operator_adoption.get("headline"))
        or _text(benchmark_operator_adoption.get("status_summary"))
    )
    signals = _compact(
        benchmark_operator_adoption.get("signals")
        or benchmark_operator_adoption.get("adoption_signals")
        or benchmark_operator_adoption.get("risks")
        or [],
    )
    actions = _compact(
        benchmark_operator_adoption.get("actions")
        or benchmark_operator_adoption.get("recommended_actions")
        or benchmark_operator_adoption.get("guidance")
        or [],
    )
    return {
        "status": status,
        "summary": summary,
        "signals": signals,
        "actions": actions,
        "has_guidance": bool(signals or actions),
    }


def _step(
    *,
    order: int,
    key: str,
    status: str,
    title: str,
    reason: str,
    action: str,
) -> Dict[str, Any]:
    return {
        "order": order,
        "key": key,
        "status": status,
        "title": title,
        "reason": reason,
        "action": action,
    }


def build_release_runbook(
    *,
    title: str,
    benchmark_release_decision: Dict[str, Any],
    benchmark_companion_summary: Dict[str, Any],
    benchmark_artifact_bundle: Dict[str, Any],
    benchmark_knowledge_readiness: Dict[str, Any],
    benchmark_engineering_signals: Dict[str, Any],
    benchmark_operator_adoption: Dict[str, Any] | None = None,
    artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
    benchmark_operator_adoption = benchmark_operator_adoption or {}
    knowledge_component = (
        benchmark_knowledge_readiness.get("knowledge_readiness")
        or benchmark_knowledge_readiness
        or {}
    )
    knowledge_status = (
        str(knowledge_component.get("status") or "unknown").strip() or "unknown"
    )
    knowledge_focus_areas = list(knowledge_component.get("focus_areas_detail") or [])
    release_status = _release_status(
        benchmark_release_decision,
        benchmark_companion_summary,
        benchmark_artifact_bundle,
    )
    automation_ready = bool(benchmark_release_decision.get("automation_ready"))
    blockers = _compact(
        benchmark_release_decision.get("blocking_signals")
        or benchmark_companion_summary.get("blockers")
        or benchmark_artifact_bundle.get("blockers")
        or [],
    )
    review_signals = _compact(
        benchmark_release_decision.get("review_signals")
        or benchmark_companion_summary.get("recommended_actions")
        or benchmark_artifact_bundle.get("recommendations")
        or [],
    )
    engineering_component = (
        benchmark_engineering_signals.get("engineering_signals")
        or benchmark_engineering_signals
        or {}
    )
    engineering_status = str(engineering_component.get("status") or "unknown").strip() or "unknown"
    if str(engineering_component.get("status") or "").strip() not in {
        "",
        "unknown",
        "engineering_semantics_ready",
    }:
        for item in _compact(benchmark_engineering_signals.get("recommendations") or []):
            if item not in review_signals:
                review_signals.append(item)
    if knowledge_status not in {"", "unknown", "knowledge_foundation_ready"}:
        for item in _compact(benchmark_knowledge_readiness.get("recommendations") or []):
            if item not in review_signals:
                review_signals.append(item)
    artifacts = _artifacts(
        benchmark_release_decision=benchmark_release_decision,
        benchmark_companion_summary=benchmark_companion_summary,
        benchmark_artifact_bundle=benchmark_artifact_bundle,
        benchmark_knowledge_readiness=benchmark_knowledge_readiness,
        benchmark_engineering_signals=benchmark_engineering_signals,
        benchmark_operator_adoption=benchmark_operator_adoption,
        artifact_paths=artifact_paths,
    )
    operator_adoption = _operator_adoption_payload(benchmark_operator_adoption)
    missing_artifacts = [
        name
        for name, row in artifacts.items()
        if name
        not in {
            "benchmark_release_decision",
            "benchmark_operator_adoption",
            "benchmark_knowledge_readiness",
        }
        and not row["present"]
    ]

    operator_steps: List[Dict[str, Any]] = []
    operator_steps.append(
        _step(
            order=1,
            key="collect_artifacts",
            status="required" if missing_artifacts else "ready",
            title="Collect benchmark artifacts",
            reason=(
                "Missing artifacts: " + ", ".join(missing_artifacts)
                if missing_artifacts
                else "All required benchmark artifacts are present."
            ),
            action=(
                "Regenerate or attach the missing benchmark scorecard / operational "
                "summary / companion / bundle artifacts before freezing a release."
                if missing_artifacts
                else "No artifact backfill required."
            ),
        )
    )
    operator_steps.append(
        _step(
            order=2,
            key="resolve_blockers",
            status="required" if blockers else "ready",
            title="Resolve blocking signals",
            reason=(
                "; ".join(blockers)
                if blockers
                else "No release blockers were reported by the benchmark decision."
            ),
            action=(
                "Address the blocking components, then rerun evaluation-report.yml "
                "with benchmark exporters enabled."
                if blockers
                else "Blocker resolution is not required."
            ),
        )
    )
    operator_steps.append(
        _step(
            order=3,
            key="review_signals",
            status="required" if release_status == "review_required" or review_signals else "ready",
            title="Review non-blocking benchmark gaps",
            reason=(
                "; ".join(review_signals)
                if review_signals
                else "No additional review-only signals were emitted."
            ),
            action=(
                "Route the highlighted items through review queue / OCR guidance, "
                "then rerun the benchmark before promoting the next baseline."
                if release_status == "review_required" or review_signals
                else "No manual review escalation is required."
            ),
        )
    )
    operator_steps.append(
        _step(
            order=4,
            key="operator_adoption_guidance",
            status="guidance" if operator_adoption["has_guidance"] else "ready",
            title="Review operator adoption guidance",
            reason=(
                " | ".join(
                    part
                    for part in [
                        operator_adoption.get("summary") or "",
                        (
                            "Signals: "
                            + "; ".join(operator_adoption.get("signals") or [])
                        )
                        if operator_adoption.get("signals")
                        else "",
                    ]
                    if part
                )
                or "No operator adoption guidance was supplied."
            ),
            action=(
                "Use operator adoption actions as low-priority guidance after "
                "required blockers, artifact gaps, and review signals are cleared: "
                + "; ".join(operator_adoption.get("actions") or [])
                if operator_adoption["has_guidance"]
                else "No operator adoption follow-up is required."
            ),
        )
    )
    operator_steps.append(
        _step(
            order=5,
            key="rerun_benchmark",
            status=(
                "required"
                if blockers or review_signals or missing_artifacts
                else "ready"
            ),
            title="Rerun benchmark workflow",
            reason=(
                "Release evidence changed or remains incomplete."
                if blockers or review_signals or missing_artifacts
                else "Current evidence is already consistent."
            ),
            action=(
                "Trigger evaluation-report.yml and verify scorecard, companion "
                "summary, artifact bundle, and release decision artifacts."
            ),
        )
    )
    operator_steps.append(
        _step(
            order=6,
            key="freeze_release_baseline",
            status=(
                "ready"
                if (
                    automation_ready
                    and not blockers
                    and not review_signals
                    and not missing_artifacts
                )
                else "blocked"
            ),
            title="Freeze release benchmark baseline",
            reason=(
                "Automation-ready release decision with no outstanding blockers."
                if (
                    automation_ready
                    and not blockers
                    and not review_signals
                    and not missing_artifacts
                )
                else "Outstanding gaps still prevent freezing the next baseline."
            ),
            action=(
                "Promote this run as the benchmark baseline and attach the generated "
                "artifact bundle to the release record."
                if (
                    automation_ready
                    and not blockers
                    and not review_signals
                    and not missing_artifacts
                )
                else "Do not freeze the release baseline until earlier steps are green."
            ),
        )
    )

    next_action = next(
        (
            step["key"]
            for step in operator_steps
            if step["status"] in {"required", "blocked"}
        ),
        "freeze_release_baseline" if automation_ready else "rerun_benchmark",
    )
    ready_to_freeze = (
        automation_ready
        and release_status == "ready"
        and not blockers
        and not review_signals
        and not missing_artifacts
    )
    return {
        "title": title,
        "generated_at": int(time.time()),
        "release_status": release_status,
        "automation_ready": automation_ready,
        "ready_to_freeze_baseline": ready_to_freeze,
        "engineering_status": engineering_status,
        "knowledge_status": knowledge_status,
        "knowledge_focus_areas": knowledge_focus_areas,
        "primary_signal_source": _primary_signal_source(
            benchmark_release_decision,
            benchmark_companion_summary,
            benchmark_artifact_bundle,
        ),
        "missing_artifacts": missing_artifacts,
        "blocking_signals": blockers,
        "review_signals": review_signals,
        "operator_adoption": operator_adoption,
        "next_action": next_action,
        "operator_steps": operator_steps,
        "artifacts": artifacts,
    }


def render_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        f"# {payload.get('title') or 'Benchmark Release Runbook'}",
        "",
        f"- `release_status`: `{payload.get('release_status')}`",
        f"- `automation_ready`: `{payload.get('automation_ready')}`",
        f"- `ready_to_freeze_baseline`: `{payload.get('ready_to_freeze_baseline')}`",
        f"- `engineering_status`: `{payload.get('engineering_status')}`",
        f"- `knowledge_status`: `{payload.get('knowledge_status')}`",
        f"- `primary_signal_source`: `{payload.get('primary_signal_source')}`",
        f"- `next_action`: `{payload.get('next_action')}`",
        "",
        "## Missing Artifacts",
        "",
    ]
    missing_artifacts = payload.get("missing_artifacts") or []
    if missing_artifacts:
        lines.extend(f"- `{name}`" for name in missing_artifacts)
    else:
        lines.append("- none")
    lines.extend(["", "## Blocking Signals", ""])
    blockers = payload.get("blocking_signals") or []
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- none")
    lines.extend(["", "## Review Signals", ""])
    review_signals = payload.get("review_signals") or []
    if review_signals:
        lines.extend(f"- {item}" for item in review_signals)
    else:
        lines.append("- none")
    lines.extend(["", "## Knowledge Focus Areas", ""])
    focus_areas = payload.get("knowledge_focus_areas") or []
    if focus_areas:
        for row in focus_areas:
            lines.append(
                "- "
                f"`{row.get('component')}` "
                f"status=`{row.get('status')}` "
                f"priority=`{row.get('priority')}` "
                f"action=`{row.get('action')}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Operator Adoption", ""])
    operator_adoption = payload.get("operator_adoption") or {}
    lines.append(f"- `status`: `{operator_adoption.get('status')}`")
    lines.append(f"- `has_guidance`: `{operator_adoption.get('has_guidance')}`")
    lines.append(
        "- `summary`: "
        + (_text(operator_adoption.get("summary")) or "none")
    )
    operator_signals = operator_adoption.get("signals") or []
    if operator_signals:
        lines.extend(f"- signal: {item}" for item in operator_signals)
    else:
        lines.append("- signal: none")
    operator_actions = operator_adoption.get("actions") or []
    if operator_actions:
        lines.extend(f"- action: {item}" for item in operator_actions)
    else:
        lines.append("- action: none")
    lines.extend(["", "## Operator Steps", ""])
    for step in payload.get("operator_steps") or []:
        lines.append(
            f"{step.get('order')}. `{step.get('key')}` `{step.get('status')}` "
            f"{step.get('title')}"
        )
        lines.append(f"   - reason: {step.get('reason')}")
        lines.append(f"   - action: {step.get('action')}")
    lines.extend(["", "## Artifacts", ""])
    for name, row in (payload.get("artifacts") or {}).items():
        lines.append(
            f"- `{name}`: present=`{row.get('present')}` path=`{row.get('path')}`"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a benchmark release operator runbook."
    )
    parser.add_argument("--title", default="Benchmark Release Runbook")
    parser.add_argument("--benchmark-release-decision", default="")
    parser.add_argument("--benchmark-companion-summary", default="")
    parser.add_argument("--benchmark-artifact-bundle", default="")
    parser.add_argument("--benchmark-knowledge-readiness", default="")
    parser.add_argument("--benchmark-engineering-signals", default="")
    parser.add_argument("--benchmark-operator-adoption", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    artifact_paths = {
        "benchmark_release_decision": args.benchmark_release_decision,
        "benchmark_companion_summary": args.benchmark_companion_summary,
        "benchmark_artifact_bundle": args.benchmark_artifact_bundle,
        "benchmark_knowledge_readiness": args.benchmark_knowledge_readiness,
        "benchmark_engineering_signals": args.benchmark_engineering_signals,
        "benchmark_operator_adoption": args.benchmark_operator_adoption,
    }
    payload = build_release_runbook(
        title=args.title,
        benchmark_release_decision=_maybe_load_json(args.benchmark_release_decision),
        benchmark_companion_summary=_maybe_load_json(args.benchmark_companion_summary),
        benchmark_artifact_bundle=_maybe_load_json(args.benchmark_artifact_bundle),
        benchmark_knowledge_readiness=_maybe_load_json(
            args.benchmark_knowledge_readiness
        ),
        benchmark_engineering_signals=_maybe_load_json(
            args.benchmark_engineering_signals
        ),
        benchmark_operator_adoption=_maybe_load_json(
            args.benchmark_operator_adoption
        ),
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
