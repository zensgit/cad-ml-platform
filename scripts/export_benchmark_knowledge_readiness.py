#!/usr/bin/env python3
"""Export benchmark knowledge-readiness signals from built-in knowledge modules."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.benchmark import (  # noqa: E402
    build_knowledge_readiness_status,
    collect_builtin_knowledge_snapshot,
    knowledge_readiness_recommendations,
    render_knowledge_readiness_markdown,
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


def build_knowledge_readiness_summary(
    *,
    title: str,
    knowledge_snapshot: Dict[str, Any],
    artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
    component = build_knowledge_readiness_status(knowledge_snapshot)
    return {
        "title": title,
        "generated_at": int(time.time()),
        "knowledge_readiness": component,
        "recommendations": knowledge_readiness_recommendations(component),
        "artifacts": artifact_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export benchmark knowledge-readiness signals."
    )
    parser.add_argument("--title", default="Benchmark Knowledge Readiness")
    parser.add_argument(
        "--knowledge-snapshot",
        default="",
        help="Optional JSON snapshot override. When omitted, use built-in modules.",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    snapshot = (
        _load_json(args.knowledge_snapshot)
        if str(args.knowledge_snapshot or "").strip()
        else collect_builtin_knowledge_snapshot()
    )
    payload = build_knowledge_readiness_summary(
        title=args.title,
        knowledge_snapshot=snapshot,
        artifact_paths={"knowledge_snapshot": args.knowledge_snapshot},
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        _write_output(args.output_json, rendered + "\n")
    if args.output_md:
        _write_output(
            args.output_md,
            render_knowledge_readiness_markdown(payload, args.title),
        )
    print(rendered)


if __name__ == "__main__":
    main()
