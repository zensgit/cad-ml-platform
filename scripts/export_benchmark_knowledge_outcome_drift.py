#!/usr/bin/env python3
"""Export benchmark knowledge outcome correlation drift against a previous baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.benchmark import (  # noqa: E402
    build_knowledge_outcome_drift_status,
    knowledge_outcome_drift_recommendations,
    render_knowledge_outcome_drift_markdown,
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


def build_knowledge_outcome_drift_summary(
    *,
    title: str,
    current_summary: Dict[str, Any],
    previous_summary: Dict[str, Any],
    artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
    component = build_knowledge_outcome_drift_status(current_summary, previous_summary)
    return {
        "title": title,
        "generated_at": int(time.time()),
        "knowledge_outcome_drift": component,
        "recommendations": knowledge_outcome_drift_recommendations(component),
        "artifacts": artifact_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export benchmark knowledge outcome correlation drift summary."
    )
    parser.add_argument("--title", default="Benchmark Knowledge Outcome Drift")
    parser.add_argument("--current-summary", required=True)
    parser.add_argument("--previous-summary", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    previous_summary = (
        _load_json(args.previous_summary)
        if str(args.previous_summary or "").strip()
        else {}
    )
    payload = build_knowledge_outcome_drift_summary(
        title=args.title,
        current_summary=_load_json(args.current_summary),
        previous_summary=previous_summary,
        artifact_paths={
            "current_summary": args.current_summary,
            "previous_summary": args.previous_summary,
        },
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        _write_output(args.output_json, rendered + "\n")
    if args.output_md:
        _write_output(
            args.output_md,
            render_knowledge_outcome_drift_markdown(payload, args.title),
        )
    print(rendered)


if __name__ == "__main__":
    main()
