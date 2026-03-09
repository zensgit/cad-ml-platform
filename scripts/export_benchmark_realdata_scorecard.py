#!/usr/bin/env python3
"""Export a benchmark real-data scorecard from evaluation reports."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.benchmark import (  # noqa: E402
    build_realdata_scorecard_status,
    realdata_scorecard_recommendations,
    render_realdata_scorecard_markdown,
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


def build_realdata_scorecard_summary(
    *,
    title: str,
    hybrid_summary: Dict[str, Any],
    history_summary: Dict[str, Any],
    online_example_report: Dict[str, Any],
    step_dir_summary: Dict[str, Any],
    artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
    component = build_realdata_scorecard_status(
        hybrid_summary=hybrid_summary,
        history_summary=history_summary,
        online_example_report=online_example_report,
        step_dir_summary=step_dir_summary,
    )
    return {
        "title": title,
        "generated_at": int(time.time()),
        "realdata_scorecard": component,
        "recommendations": realdata_scorecard_recommendations(component),
        "artifacts": artifact_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a benchmark real-data scorecard."
    )
    parser.add_argument("--title", default="Benchmark Real-Data Scorecard")
    parser.add_argument("--hybrid-summary", default="")
    parser.add_argument("--history-summary", default="")
    parser.add_argument("--online-example-report", default="")
    parser.add_argument("--step-dir-summary", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    payload = build_realdata_scorecard_summary(
        title=args.title,
        hybrid_summary=_maybe_load_json(args.hybrid_summary),
        history_summary=_maybe_load_json(args.history_summary),
        online_example_report=_maybe_load_json(args.online_example_report),
        step_dir_summary=_maybe_load_json(args.step_dir_summary),
        artifact_paths={
            "hybrid_summary": args.hybrid_summary,
            "history_summary": args.history_summary,
            "online_example_report": args.online_example_report,
            "step_dir_summary": args.step_dir_summary,
        },
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        _write_output(args.output_json, rendered + "\n")
    if args.output_md:
        _write_output(
            args.output_md,
            render_realdata_scorecard_markdown(payload, args.title),
        )
    print(rendered)


if __name__ == "__main__":
    main()
