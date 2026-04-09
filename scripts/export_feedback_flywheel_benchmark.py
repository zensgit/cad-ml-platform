#!/usr/bin/env python3
"""Export a standalone feedback flywheel benchmark artifact."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.benchmark.feedback_flywheel import (  # noqa: E402
    build_feedback_flywheel_status,
    feedback_flywheel_recommendations,
    render_feedback_flywheel_markdown,
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


def build_payload(
    *,
    title: str,
    feedback_summary: Dict[str, Any],
    finetune_summary: Dict[str, Any],
    metric_train_summary: Dict[str, Any],
) -> Dict[str, Any]:
    feedback_flywheel = build_feedback_flywheel_status(
        feedback_summary,
        finetune_summary,
        metric_train_summary,
    )
    return {
        "title": title,
        "generated_at": int(time.time()),
        "feedback_flywheel": feedback_flywheel,
        "recommendations": feedback_flywheel_recommendations(feedback_flywheel),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a standalone feedback flywheel benchmark artifact."
    )
    parser.add_argument(
        "--title",
        default="Feedback Flywheel Benchmark",
        help="Display title for the exported benchmark summary",
    )
    parser.add_argument(
        "--feedback-summary",
        default="",
        help="Optional feedback stats JSON path",
    )
    parser.add_argument(
        "--finetune-summary",
        default="",
        help="Optional fine-tune summary JSON path",
    )
    parser.add_argument(
        "--metric-train-summary",
        default="",
        help="Optional metric-train summary JSON path",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to persist the JSON payload",
    )
    parser.add_argument(
        "--output-md",
        default="",
        help="Optional path to persist the Markdown report",
    )
    args = parser.parse_args()

    payload = build_payload(
        title=args.title,
        feedback_summary=_maybe_load_json(args.feedback_summary),
        finetune_summary=_maybe_load_json(args.finetune_summary),
        metric_train_summary=_maybe_load_json(args.metric_train_summary),
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        _write_output(args.output_json, rendered + "\n")
    if args.output_md:
        markdown = render_feedback_flywheel_markdown(payload, args.title)
        _write_output(args.output_md, markdown)
    print(rendered)


if __name__ == "__main__":
    main()
