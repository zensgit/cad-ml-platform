#!/usr/bin/env python3
"""Export knowledge-domain release-surface alignment signals."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from src.core.benchmark import (
    build_knowledge_domain_release_surface_alignment,
    knowledge_domain_release_surface_alignment_recommendations,
    render_knowledge_domain_release_surface_alignment_markdown,
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
    benchmark_release_decision: Dict[str, Any],
    benchmark_release_runbook: Dict[str, Any],
    artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
    component = build_knowledge_domain_release_surface_alignment(
        benchmark_release_decision=benchmark_release_decision,
        benchmark_release_runbook=benchmark_release_runbook,
    )
    return {
        "title": title,
        "generated_at": int(time.time()),
        "artifact_paths": artifact_paths,
        "knowledge_domain_release_surface_alignment": component,
        "recommendations": knowledge_domain_release_surface_alignment_recommendations(
            component
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export benchmark knowledge-domain release-surface alignment."
    )
    parser.add_argument(
        "--title", default="Benchmark Knowledge Domain Release Surface Alignment"
    )
    parser.add_argument("--benchmark-release-decision", required=True)
    parser.add_argument("--benchmark-release-runbook", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    artifact_paths = {
        "benchmark_release_decision": args.benchmark_release_decision,
        "benchmark_release_runbook": args.benchmark_release_runbook,
    }
    payload = build_payload(
        title=args.title,
        benchmark_release_decision=_load_json(args.benchmark_release_decision),
        benchmark_release_runbook=_load_json(args.benchmark_release_runbook),
        artifact_paths=artifact_paths,
    )
    _write_output(args.output_json, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if args.output_md:
        _write_output(
            args.output_md,
            render_knowledge_domain_release_surface_alignment_markdown(
                payload, args.title
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
