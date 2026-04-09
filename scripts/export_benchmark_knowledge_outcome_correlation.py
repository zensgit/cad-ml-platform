#!/usr/bin/env python3
"""Export knowledge-domain benchmark outcome correlation summaries."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from src.core.benchmark import (
    build_knowledge_outcome_correlation_status,
    knowledge_outcome_correlation_recommendations,
    render_knowledge_outcome_correlation_markdown,
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
    benchmark_knowledge_domain_matrix: Dict[str, Any],
    benchmark_realdata_scorecard: Dict[str, Any],
    artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
    component = build_knowledge_outcome_correlation_status(
        benchmark_knowledge_domain_matrix,
        benchmark_realdata_scorecard,
    )
    return {
        "title": title,
        "generated_at": int(time.time()),
        "artifact_paths": artifact_paths,
        "knowledge_outcome_correlation": component,
        "recommendations": knowledge_outcome_correlation_recommendations(component),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export benchmark knowledge outcome correlation."
    )
    parser.add_argument(
        "--title",
        default="Benchmark Knowledge Outcome Correlation",
    )
    parser.add_argument("--benchmark-knowledge-domain-matrix", required=True)
    parser.add_argument("--benchmark-realdata-scorecard", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    artifact_paths = {
        "benchmark_knowledge_domain_matrix": args.benchmark_knowledge_domain_matrix,
        "benchmark_realdata_scorecard": args.benchmark_realdata_scorecard,
    }
    payload = build_payload(
        title=args.title,
        benchmark_knowledge_domain_matrix=_load_json(
            args.benchmark_knowledge_domain_matrix
        ),
        benchmark_realdata_scorecard=_load_json(args.benchmark_realdata_scorecard),
        artifact_paths=artifact_paths,
    )
    _write_output(args.output_json, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    if args.output_md:
        _write_output(
            args.output_md,
            render_knowledge_outcome_correlation_markdown(payload, args.title),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
