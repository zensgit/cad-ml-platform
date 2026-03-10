#!/usr/bin/env python3
"""Export benchmark knowledge-domain release-gate signals."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.benchmark import (  # noqa: E402
    build_knowledge_domain_release_gate,
    knowledge_domain_release_gate_recommendations,
    render_knowledge_domain_release_gate_markdown,
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


def build_summary(
    *,
    title: str,
    benchmark_knowledge_domain_capability_matrix: Dict[str, Any],
    benchmark_knowledge_domain_capability_drift: Dict[str, Any],
    benchmark_knowledge_domain_action_plan: Dict[str, Any],
    benchmark_knowledge_domain_control_plane: Dict[str, Any],
    benchmark_knowledge_domain_control_plane_drift: Dict[str, Any],
    benchmark_knowledge_domain_release_surface_alignment: Dict[str, Any],
    artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
    component = build_knowledge_domain_release_gate(
        benchmark_knowledge_domain_capability_matrix=(
            benchmark_knowledge_domain_capability_matrix
        ),
        benchmark_knowledge_domain_capability_drift=(
            benchmark_knowledge_domain_capability_drift
        ),
        benchmark_knowledge_domain_action_plan=benchmark_knowledge_domain_action_plan,
        benchmark_knowledge_domain_control_plane=benchmark_knowledge_domain_control_plane,
        benchmark_knowledge_domain_control_plane_drift=(
            benchmark_knowledge_domain_control_plane_drift
        ),
        benchmark_knowledge_domain_release_surface_alignment=(
            benchmark_knowledge_domain_release_surface_alignment
        ),
    )
    return {
        "title": title,
        "generated_at": int(time.time()),
        "knowledge_domain_release_gate": component,
        "recommendations": knowledge_domain_release_gate_recommendations(component),
        "artifacts": artifact_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export benchmark knowledge-domain release-gate signals."
    )
    parser.add_argument("--title", default="Benchmark Knowledge Domain Release Gate")
    parser.add_argument("--benchmark-knowledge-domain-capability-matrix", default="")
    parser.add_argument("--benchmark-knowledge-domain-capability-drift", default="")
    parser.add_argument("--benchmark-knowledge-domain-action-plan", default="")
    parser.add_argument("--benchmark-knowledge-domain-control-plane", default="")
    parser.add_argument("--benchmark-knowledge-domain-control-plane-drift", default="")
    parser.add_argument(
        "--benchmark-knowledge-domain-release-surface-alignment",
        default="",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    artifact_paths = {
        "benchmark_knowledge_domain_capability_matrix": (
            args.benchmark_knowledge_domain_capability_matrix
        ),
        "benchmark_knowledge_domain_capability_drift": (
            args.benchmark_knowledge_domain_capability_drift
        ),
        "benchmark_knowledge_domain_action_plan": (
            args.benchmark_knowledge_domain_action_plan
        ),
        "benchmark_knowledge_domain_control_plane": (
            args.benchmark_knowledge_domain_control_plane
        ),
        "benchmark_knowledge_domain_control_plane_drift": (
            args.benchmark_knowledge_domain_control_plane_drift
        ),
        "benchmark_knowledge_domain_release_surface_alignment": (
            args.benchmark_knowledge_domain_release_surface_alignment
        ),
    }
    payload = build_summary(
        title=args.title,
        benchmark_knowledge_domain_capability_matrix=_maybe_load_json(
            args.benchmark_knowledge_domain_capability_matrix
        ),
        benchmark_knowledge_domain_capability_drift=_maybe_load_json(
            args.benchmark_knowledge_domain_capability_drift
        ),
        benchmark_knowledge_domain_action_plan=_maybe_load_json(
            args.benchmark_knowledge_domain_action_plan
        ),
        benchmark_knowledge_domain_control_plane=_maybe_load_json(
            args.benchmark_knowledge_domain_control_plane
        ),
        benchmark_knowledge_domain_control_plane_drift=_maybe_load_json(
            args.benchmark_knowledge_domain_control_plane_drift
        ),
        benchmark_knowledge_domain_release_surface_alignment=_maybe_load_json(
            args.benchmark_knowledge_domain_release_surface_alignment
        ),
        artifact_paths=artifact_paths,
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        _write_output(args.output_json, rendered + "\n")
    if args.output_md:
        _write_output(
            args.output_md,
            render_knowledge_domain_release_gate_markdown(payload, args.title),
        )
    print(rendered)


if __name__ == "__main__":
    main()
