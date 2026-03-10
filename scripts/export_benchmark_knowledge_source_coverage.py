#!/usr/bin/env python3
"""Export benchmark knowledge source coverage signals."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.benchmark import (  # noqa: E402
    build_knowledge_source_coverage_status,
    collect_builtin_knowledge_source_snapshot,
    knowledge_source_coverage_recommendations,
    render_knowledge_source_coverage_markdown,
)


def _write_output(path_text: str, content: str) -> None:
    output_path = Path(path_text).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def build_summary(*, title: str, artifact_paths: Dict[str, str]) -> Dict[str, Any]:
    component = build_knowledge_source_coverage_status(
        collect_builtin_knowledge_source_snapshot()
    )
    return {
        "title": title,
        "generated_at": int(time.time()),
        "knowledge_source_coverage": component,
        "recommendations": knowledge_source_coverage_recommendations(component),
        "artifacts": artifact_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export benchmark knowledge source coverage signals."
    )
    parser.add_argument("--title", default="Benchmark Knowledge Source Coverage")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    payload = build_summary(title=args.title, artifact_paths={})
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        _write_output(args.output_json, rendered + "\n")
    if args.output_md:
        _write_output(
            args.output_md,
            render_knowledge_source_coverage_markdown(payload, args.title),
        )
    print(rendered)


if __name__ == "__main__":
    main()
