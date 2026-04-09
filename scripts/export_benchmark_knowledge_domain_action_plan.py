#!/usr/bin/env python3
"""Export benchmark knowledge domain action plan signals."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.benchmark import (  # noqa: E402
    build_knowledge_domain_action_plan,
    knowledge_domain_action_plan_recommendations,
    render_knowledge_domain_action_plan_markdown,
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
    knowledge_domain_matrix_summary: Dict[str, Any],
    artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
    component = build_knowledge_domain_action_plan(knowledge_domain_matrix_summary)
    return {
        "title": title,
        "generated_at": int(time.time()),
        "knowledge_domain_action_plan": component,
        "recommendations": knowledge_domain_action_plan_recommendations(component),
        "artifacts": artifact_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export benchmark knowledge domain action plan signals."
    )
    parser.add_argument(
        "--title",
        default="Benchmark Knowledge Domain Action Plan",
    )
    parser.add_argument("--benchmark-knowledge-domain-matrix", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    payload = build_summary(
        title=args.title,
        knowledge_domain_matrix_summary=_maybe_load_json(
            args.benchmark_knowledge_domain_matrix
        ),
        artifact_paths={
            "benchmark_knowledge_domain_matrix": args.benchmark_knowledge_domain_matrix,
        },
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        _write_output(args.output_json, rendered + "\n")
    if args.output_md:
        _write_output(
            args.output_md,
            render_knowledge_domain_action_plan_markdown(payload, args.title),
        )
    print(rendered)


if __name__ == "__main__":
    main()
