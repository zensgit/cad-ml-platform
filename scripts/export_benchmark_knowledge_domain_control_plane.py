#!/usr/bin/env python3
"""Export the standards/tolerance/GD&T benchmark control-plane."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from src.core.benchmark import (
    build_knowledge_domain_control_plane,
    knowledge_domain_control_plane_recommendations,
    render_knowledge_domain_control_plane_markdown,
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


def _write_output(path_text: str, content: str) -> None:
    output_path = Path(path_text).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def build_payload(
    *,
    title: str,
    benchmark_knowledge_domain_capability_matrix: Dict[str, Any],
    benchmark_knowledge_domain_capability_drift: Dict[str, Any],
    benchmark_knowledge_realdata_correlation: Dict[str, Any],
    benchmark_knowledge_outcome_correlation: Dict[str, Any],
    benchmark_knowledge_domain_action_plan: Dict[str, Any],
    artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
    component = build_knowledge_domain_control_plane(
        benchmark_knowledge_domain_capability_matrix=(
            benchmark_knowledge_domain_capability_matrix
        ),
        benchmark_knowledge_domain_capability_drift=(
            benchmark_knowledge_domain_capability_drift
        ),
        benchmark_knowledge_realdata_correlation=(
            benchmark_knowledge_realdata_correlation
        ),
        benchmark_knowledge_outcome_correlation=benchmark_knowledge_outcome_correlation,
        benchmark_knowledge_domain_action_plan=benchmark_knowledge_domain_action_plan,
    )
    return {
        "title": title,
        "generated_at": int(time.time()),
        "artifact_paths": artifact_paths,
        "knowledge_domain_control_plane": component,
        "recommendations": knowledge_domain_control_plane_recommendations(component),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export benchmark knowledge domain control-plane."
    )
    parser.add_argument("--title", default="Benchmark Knowledge Domain Control Plane")
    parser.add_argument("--benchmark-knowledge-domain-capability-matrix", required=True)
    parser.add_argument("--benchmark-knowledge-domain-capability-drift", required=True)
    parser.add_argument("--benchmark-knowledge-realdata-correlation", required=True)
    parser.add_argument("--benchmark-knowledge-outcome-correlation", required=True)
    parser.add_argument("--benchmark-knowledge-domain-action-plan", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    artifact_paths = {
        "benchmark_knowledge_domain_capability_matrix": (
            args.benchmark_knowledge_domain_capability_matrix
        ),
        "benchmark_knowledge_domain_capability_drift": (
            args.benchmark_knowledge_domain_capability_drift
        ),
        "benchmark_knowledge_realdata_correlation": (
            args.benchmark_knowledge_realdata_correlation
        ),
        "benchmark_knowledge_outcome_correlation": (
            args.benchmark_knowledge_outcome_correlation
        ),
        "benchmark_knowledge_domain_action_plan": (
            args.benchmark_knowledge_domain_action_plan
        ),
    }
    payload = build_payload(
        title=args.title,
        benchmark_knowledge_domain_capability_matrix=_load_json(
            args.benchmark_knowledge_domain_capability_matrix
        ),
        benchmark_knowledge_domain_capability_drift=_load_json(
            args.benchmark_knowledge_domain_capability_drift
        ),
        benchmark_knowledge_realdata_correlation=_load_json(
            args.benchmark_knowledge_realdata_correlation
        ),
        benchmark_knowledge_outcome_correlation=_load_json(
            args.benchmark_knowledge_outcome_correlation
        ),
        benchmark_knowledge_domain_action_plan=_load_json(
            args.benchmark_knowledge_domain_action_plan
        ),
        artifact_paths=artifact_paths,
    )
    _write_output(args.output_json, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    if args.output_md:
        _write_output(
            args.output_md,
            render_knowledge_domain_control_plane_markdown(payload, args.title),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
