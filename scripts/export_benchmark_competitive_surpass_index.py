#!/usr/bin/env python3
"""Export a unified benchmark competitive-surpass index."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.benchmark import (  # noqa: E402
    build_competitive_surpass_index,
    competitive_surpass_index_recommendations,
    render_competitive_surpass_markdown,
)


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


def build_competitive_surpass_summary(
    *,
    title: str,
    benchmark_engineering_signals: Dict[str, Any],
    benchmark_knowledge_readiness: Dict[str, Any],
    benchmark_knowledge_application: Dict[str, Any],
    benchmark_knowledge_realdata_correlation: Dict[str, Any],
    benchmark_knowledge_domain_matrix: Dict[str, Any],
    benchmark_knowledge_domain_action_plan: Dict[str, Any] | None = None,
    benchmark_knowledge_source_coverage: Dict[str, Any] | None = None,
    benchmark_knowledge_source_action_plan: Dict[str, Any] | None = None,
    benchmark_knowledge_outcome_correlation: Dict[str, Any],
    benchmark_knowledge_outcome_drift: Dict[str, Any],
    benchmark_realdata_signals: Dict[str, Any],
    benchmark_realdata_scorecard: Dict[str, Any],
    benchmark_operator_adoption: Dict[str, Any],
    artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
    component = build_competitive_surpass_index(
        benchmark_engineering_signals=benchmark_engineering_signals,
        benchmark_knowledge_readiness=benchmark_knowledge_readiness,
        benchmark_knowledge_application=benchmark_knowledge_application,
        benchmark_knowledge_realdata_correlation=benchmark_knowledge_realdata_correlation,
        benchmark_knowledge_domain_matrix=benchmark_knowledge_domain_matrix,
        benchmark_knowledge_domain_action_plan=benchmark_knowledge_domain_action_plan,
        benchmark_knowledge_source_coverage=benchmark_knowledge_source_coverage,
        benchmark_knowledge_source_action_plan=benchmark_knowledge_source_action_plan,
        benchmark_knowledge_outcome_correlation=benchmark_knowledge_outcome_correlation,
        benchmark_knowledge_outcome_drift=benchmark_knowledge_outcome_drift,
        benchmark_realdata_signals=benchmark_realdata_signals,
        benchmark_realdata_scorecard=benchmark_realdata_scorecard,
        benchmark_operator_adoption=benchmark_operator_adoption,
    )
    return {
        "title": title,
        "generated_at": int(time.time()),
        "competitive_surpass_index": component,
        "recommendations": competitive_surpass_index_recommendations(component),
        "artifacts": artifact_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a unified benchmark competitive-surpass index."
    )
    parser.add_argument("--title", default="Benchmark Competitive Surpass Index")
    parser.add_argument("--benchmark-engineering-signals", default="")
    parser.add_argument("--benchmark-knowledge-readiness", default="")
    parser.add_argument("--benchmark-knowledge-application", default="")
    parser.add_argument("--benchmark-knowledge-realdata-correlation", default="")
    parser.add_argument("--benchmark-knowledge-domain-matrix", default="")
    parser.add_argument("--benchmark-knowledge-domain-action-plan", default="")
    parser.add_argument("--benchmark-knowledge-source-coverage", default="")
    parser.add_argument("--benchmark-knowledge-source-action-plan", default="")
    parser.add_argument("--benchmark-knowledge-outcome-correlation", default="")
    parser.add_argument("--benchmark-knowledge-outcome-drift", default="")
    parser.add_argument("--benchmark-realdata-signals", default="")
    parser.add_argument("--benchmark-realdata-scorecard", default="")
    parser.add_argument("--benchmark-operator-adoption", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    artifact_paths = {
        "benchmark_engineering_signals": args.benchmark_engineering_signals,
        "benchmark_knowledge_readiness": args.benchmark_knowledge_readiness,
        "benchmark_knowledge_application": args.benchmark_knowledge_application,
        "benchmark_knowledge_realdata_correlation": (
            args.benchmark_knowledge_realdata_correlation
        ),
        "benchmark_knowledge_domain_matrix": args.benchmark_knowledge_domain_matrix,
        "benchmark_knowledge_domain_action_plan": (
            args.benchmark_knowledge_domain_action_plan
        ),
        "benchmark_knowledge_source_coverage": (
            args.benchmark_knowledge_source_coverage
        ),
        "benchmark_knowledge_source_action_plan": (
            args.benchmark_knowledge_source_action_plan
        ),
        "benchmark_knowledge_outcome_correlation": (
            args.benchmark_knowledge_outcome_correlation
        ),
        "benchmark_knowledge_outcome_drift": args.benchmark_knowledge_outcome_drift,
        "benchmark_realdata_signals": args.benchmark_realdata_signals,
        "benchmark_realdata_scorecard": args.benchmark_realdata_scorecard,
        "benchmark_operator_adoption": args.benchmark_operator_adoption,
    }
    payload = build_competitive_surpass_summary(
        title=args.title,
        benchmark_engineering_signals=_maybe_load_json(args.benchmark_engineering_signals),
        benchmark_knowledge_readiness=_maybe_load_json(args.benchmark_knowledge_readiness),
        benchmark_knowledge_application=_maybe_load_json(args.benchmark_knowledge_application),
        benchmark_knowledge_realdata_correlation=_maybe_load_json(
            args.benchmark_knowledge_realdata_correlation
        ),
        benchmark_knowledge_domain_matrix=_maybe_load_json(
            args.benchmark_knowledge_domain_matrix
        ),
        benchmark_knowledge_domain_action_plan=_maybe_load_json(
            args.benchmark_knowledge_domain_action_plan
        ),
        benchmark_knowledge_source_coverage=_maybe_load_json(
            args.benchmark_knowledge_source_coverage
        ),
        benchmark_knowledge_source_action_plan=_maybe_load_json(
            args.benchmark_knowledge_source_action_plan
        ),
        benchmark_knowledge_outcome_correlation=_maybe_load_json(
            args.benchmark_knowledge_outcome_correlation
        ),
        benchmark_knowledge_outcome_drift=_maybe_load_json(
            args.benchmark_knowledge_outcome_drift
        ),
        benchmark_realdata_signals=_maybe_load_json(args.benchmark_realdata_signals),
        benchmark_realdata_scorecard=_maybe_load_json(args.benchmark_realdata_scorecard),
        benchmark_operator_adoption=_maybe_load_json(args.benchmark_operator_adoption),
        artifact_paths=artifact_paths,
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        _write_output(args.output_json, rendered + "\n")
    if args.output_md:
        _write_output(
            args.output_md,
            render_competitive_surpass_markdown(payload, args.title),
        )
    print(rendered)


if __name__ == "__main__":
    main()
