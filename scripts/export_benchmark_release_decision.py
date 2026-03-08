#!/usr/bin/env python3
"""Export a benchmark release decision summary."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CRITICAL_STATUSES = {
    "blocked",
    "critical_backlog",
    "missing",
}

REVIEW_STATUSES = {
    "attention_required",
    "gap_detected",
    "managed_backlog",
    "partial_coverage",
    "review_heavy",
    "weak_coverage",
    "feedback_collected",
    "passive_feedback_only",
}


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


def _compact(items: Iterable[Any], *, limit: int = 5) -> List[str]:
    out: List[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _artifact_row(name: str, path_text: str) -> Dict[str, Any]:
    path_value = str(path_text or "").strip()
    return {
        "name": name,
        "path": path_value,
        "present": bool(path_value),
    }


def _component_statuses(
    benchmark_scorecard: Dict[str, Any],
    benchmark_operational_summary: Dict[str, Any],
    benchmark_artifact_bundle: Dict[str, Any],
    benchmark_companion_summary: Dict[str, Any],
    benchmark_knowledge_readiness: Dict[str, Any],
    benchmark_engineering_signals: Dict[str, Any],
    benchmark_operator_adoption: Dict[str, Any],
) -> Dict[str, str]:
    scorecard_components = benchmark_scorecard.get("components") or {}
    operational_components = benchmark_operational_summary.get("component_statuses") or {}
    bundle_components = benchmark_artifact_bundle.get("component_statuses") or {}
    companion_components = benchmark_companion_summary.get("component_statuses") or {}
    knowledge_component = (
        benchmark_knowledge_readiness.get("knowledge_readiness")
        or benchmark_knowledge_readiness
        or {}
    )
    engineering_component = (
        benchmark_engineering_signals.get("engineering_signals")
        or benchmark_engineering_signals
        or {}
    )

    def pick(name: str) -> str:
        if isinstance(companion_components, dict) and companion_components.get(name):
            return str(companion_components.get(name))
        if isinstance(bundle_components, dict) and bundle_components.get(name):
            return str(bundle_components.get(name))
        if isinstance(operational_components, dict) and operational_components.get(name):
            return str(operational_components.get(name))
        scorecard_row = scorecard_components.get(name) or {}
        if isinstance(scorecard_row, dict) and scorecard_row.get("status"):
            return str(scorecard_row.get("status"))
        return "unknown"

    return {
        "hybrid": pick("hybrid"),
        "history_sequence": pick("history_sequence"),
        "brep": pick("brep"),
        "migration_governance": pick("migration_governance"),
        "feedback_flywheel": pick("feedback_flywheel"),
        "assistant_explainability": pick("assistant_explainability"),
        "review_queue": pick("review_queue"),
        "ocr_review": pick("ocr_review"),
        "qdrant_backend": pick("qdrant_backend"),
        "knowledge_readiness": str(
            companion_components.get("knowledge_readiness")
            or bundle_components.get("knowledge_readiness")
            or knowledge_component.get("status")
            or (scorecard_components.get("knowledge_readiness") or {}).get("status")
            or "unknown"
        ),
        "engineering_signals": str(
            companion_components.get("engineering_signals")
            or bundle_components.get("engineering_signals")
            or engineering_component.get("status")
            or (scorecard_components.get("engineering_signals") or {}).get("status")
            or "unknown"
        ),
        "operator_adoption": str(
            companion_components.get("operator_adoption")
            or bundle_components.get("operator_adoption")
            or benchmark_operator_adoption.get("adoption_readiness")
            or "unknown"
        ),
    }


def _pick_signal_source(
    benchmark_companion_summary: Dict[str, Any],
    benchmark_artifact_bundle: Dict[str, Any],
    benchmark_operational_summary: Dict[str, Any],
    benchmark_scorecard: Dict[str, Any],
) -> str:
    if benchmark_companion_summary:
        return "benchmark_companion_summary"
    if benchmark_artifact_bundle:
        return "benchmark_artifact_bundle"
    if benchmark_operational_summary:
        return "benchmark_operational_summary"
    if benchmark_scorecard:
        return "benchmark_scorecard"
    return "none"


def _decision(
    component_statuses: Dict[str, str], blockers: List[str], review_signals: List[str]
) -> Tuple[str, bool]:
    decision_statuses = {
        name: status
        for name, status in component_statuses.items()
        if name != "operator_adoption"
    }
    if blockers:
        return "blocked", False
    if any(status in CRITICAL_STATUSES for status in decision_statuses.values()):
        return "blocked", False
    if review_signals or any(status in REVIEW_STATUSES for status in decision_statuses.values()):
        return "review_required", False
    return "ready", True


def _engineering_review_signals(
    benchmark_engineering_signals: Dict[str, Any],
    component_statuses: Dict[str, str],
) -> List[str]:
    status = str(component_statuses.get("engineering_signals") or "").strip()
    if status in {"engineering_semantics_ready", "unknown", ""}:
        return []
    return _compact(benchmark_engineering_signals.get("recommendations") or [], limit=6)


def _knowledge_review_signals(
    benchmark_knowledge_readiness: Dict[str, Any],
    component_statuses: Dict[str, str],
) -> List[str]:
    status = str(component_statuses.get("knowledge_readiness") or "").strip()
    if status in {"knowledge_foundation_ready", "unknown", ""}:
        return []
    return _compact(benchmark_knowledge_readiness.get("recommendations") or [], limit=6)


def build_release_decision(
    *,
    title: str,
    benchmark_scorecard: Dict[str, Any],
    benchmark_operational_summary: Dict[str, Any],
    benchmark_artifact_bundle: Dict[str, Any],
    benchmark_companion_summary: Dict[str, Any],
    benchmark_knowledge_readiness: Dict[str, Any],
    benchmark_engineering_signals: Dict[str, Any],
    benchmark_operator_adoption: Dict[str, Any],
    artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
    component_statuses = _component_statuses(
        benchmark_scorecard,
        benchmark_operational_summary,
        benchmark_artifact_bundle,
        benchmark_companion_summary,
        benchmark_knowledge_readiness,
        benchmark_engineering_signals,
        benchmark_operator_adoption,
    )
    blockers = _compact(
        benchmark_companion_summary.get("blockers")
        or benchmark_artifact_bundle.get("blockers")
        or benchmark_operational_summary.get("blockers")
        or [],
        limit=6,
    )
    if not blockers:
        blockers = _compact(
            benchmark_operator_adoption.get("blocking_signals") or [],
            limit=6,
        )
    review_signals = _compact(
        benchmark_companion_summary.get("recommended_actions")
        or benchmark_artifact_bundle.get("recommendations")
        or benchmark_operational_summary.get("recommendations")
        or benchmark_scorecard.get("recommendations")
        or [],
        limit=6,
    )
    review_signals.extend(
        item
        for item in _engineering_review_signals(
            benchmark_engineering_signals,
            component_statuses,
        )
        if item not in review_signals
    )
    review_signals.extend(
        item
        for item in _knowledge_review_signals(
            benchmark_knowledge_readiness,
            component_statuses,
        )
        if item not in review_signals
    )
    operator_adoption_status = str(
        benchmark_operator_adoption.get("adoption_readiness") or ""
    ).strip()
    if not review_signals and operator_adoption_status not in {
        "",
        "operator_ready",
        "ready",
        "unknown",
    }:
        review_signals.extend(
            item
            for item in _compact(
                benchmark_operator_adoption.get("recommended_actions")
                or benchmark_operator_adoption.get("review_signals")
                or [],
                limit=6,
            )
            if item not in review_signals
        )
    release_status, automation_ready = _decision(
        component_statuses,
        blockers,
        review_signals,
    )
    primary_signal_source = _pick_signal_source(
        benchmark_companion_summary,
        benchmark_artifact_bundle,
        benchmark_operational_summary,
        benchmark_scorecard,
    )
    return {
        "title": title,
        "generated_at": int(time.time()),
        "release_status": release_status,
        "automation_ready": automation_ready,
        "primary_signal_source": primary_signal_source,
        "component_statuses": component_statuses,
        "blocking_signals": blockers,
        "review_signals": review_signals,
        "artifacts": {
            "benchmark_scorecard": _artifact_row(
                "benchmark_scorecard", artifact_paths.get("benchmark_scorecard", "")
            ),
            "benchmark_operational_summary": _artifact_row(
                "benchmark_operational_summary",
                artifact_paths.get("benchmark_operational_summary", ""),
            ),
            "benchmark_artifact_bundle": _artifact_row(
                "benchmark_artifact_bundle",
                artifact_paths.get("benchmark_artifact_bundle", ""),
            ),
            "benchmark_companion_summary": _artifact_row(
                "benchmark_companion_summary",
                artifact_paths.get("benchmark_companion_summary", ""),
            ),
            "benchmark_knowledge_readiness": _artifact_row(
                "benchmark_knowledge_readiness",
                artifact_paths.get("benchmark_knowledge_readiness", ""),
            ),
            "benchmark_engineering_signals": _artifact_row(
                "benchmark_engineering_signals",
                artifact_paths.get("benchmark_engineering_signals", ""),
            ),
            "benchmark_operator_adoption": _artifact_row(
                "benchmark_operator_adoption",
                artifact_paths.get("benchmark_operator_adoption", ""),
            ),
        },
    }


def render_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        f"# {payload.get('title') or 'Benchmark Release Decision'}",
        "",
        f"- `release_status`: `{payload.get('release_status')}`",
        f"- `automation_ready`: `{payload.get('automation_ready')}`",
        f"- `primary_signal_source`: `{payload.get('primary_signal_source')}`",
        "",
        "## Component Statuses",
        "",
    ]
    for name, status in (payload.get("component_statuses") or {}).items():
        lines.append(f"- `{name}`: `{status}`")
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
    lines.extend(["", "## Artifacts", ""])
    for name, row in (payload.get("artifacts") or {}).items():
        lines.append(
            f"- `{name}`: present=`{row.get('present')}` path=`{row.get('path')}`"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a benchmark release decision summary."
    )
    parser.add_argument("--title", default="Benchmark Release Decision")
    parser.add_argument("--benchmark-scorecard", default="")
    parser.add_argument("--benchmark-operational-summary", default="")
    parser.add_argument("--benchmark-artifact-bundle", default="")
    parser.add_argument("--benchmark-companion-summary", default="")
    parser.add_argument("--benchmark-knowledge-readiness", default="")
    parser.add_argument("--benchmark-engineering-signals", default="")
    parser.add_argument("--benchmark-operator-adoption", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    artifact_paths = {
        "benchmark_scorecard": args.benchmark_scorecard,
        "benchmark_operational_summary": args.benchmark_operational_summary,
        "benchmark_artifact_bundle": args.benchmark_artifact_bundle,
        "benchmark_companion_summary": args.benchmark_companion_summary,
        "benchmark_knowledge_readiness": args.benchmark_knowledge_readiness,
        "benchmark_engineering_signals": args.benchmark_engineering_signals,
        "benchmark_operator_adoption": args.benchmark_operator_adoption,
    }
    payload = build_release_decision(
        title=args.title,
        benchmark_scorecard=_maybe_load_json(args.benchmark_scorecard),
        benchmark_operational_summary=_maybe_load_json(
            args.benchmark_operational_summary
        ),
        benchmark_artifact_bundle=_maybe_load_json(args.benchmark_artifact_bundle),
        benchmark_companion_summary=_maybe_load_json(
            args.benchmark_companion_summary
        ),
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
