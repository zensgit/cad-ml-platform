#!/usr/bin/env python3
"""Export benchmark competitive-surpass trend status."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.benchmark import (  # noqa: E402
    build_competitive_surpass_trend_status,
    competitive_surpass_trend_recommendations,
    render_competitive_surpass_trend_markdown,
)


def _load_json(path_text: str) -> Dict[str, Any]:
    path = Path(path_text).expanduser()
    if not path.exists():
        raise SystemExit(f"JSON input not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected object JSON in {path}")
    return payload


def _write_output(path_text: str, content: str) -> None:
    output_path = Path(path_text).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def _summary(component: Dict[str, Any]) -> str:
    parts = [f"status={component.get('status') or 'unknown'}"]
    parts.append(f"score_delta={component.get('score_delta') or 0}")
    new_gaps = component.get("new_primary_gaps") or []
    resolved = component.get("resolved_primary_gaps") or []
    regressions = component.get("pillar_regressions") or []
    improvements = component.get("pillar_improvements") or []
    if new_gaps:
        parts.append("new_primary_gaps=" + ", ".join(str(item) for item in new_gaps))
    if resolved:
        parts.append(
            "resolved_primary_gaps=" + ", ".join(str(item) for item in resolved)
        )
    if regressions:
        parts.append(
            "pillar_regressions=" + ", ".join(str(item) for item in regressions)
        )
    if improvements:
        parts.append(
            "pillar_improvements=" + ", ".join(str(item) for item in improvements)
        )
    return "; ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export benchmark competitive-surpass trend status."
    )
    parser.add_argument("--title", default="Benchmark Competitive Surpass Trend")
    parser.add_argument("--current-summary", required=True)
    parser.add_argument("--previous-summary", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    current_summary = _load_json(args.current_summary)
    previous_summary = _load_json(args.previous_summary) if args.previous_summary else {}
    component = build_competitive_surpass_trend_status(
        current_summary=current_summary,
        previous_summary=previous_summary,
    )
    payload = {
        "title": args.title,
        "generated_at": int(time.time()),
        "competitive_surpass_trend": component,
        "summary": _summary(component),
        "recommendations": competitive_surpass_trend_recommendations(component),
        "artifacts": {
            "current_summary": args.current_summary,
            "previous_summary": args.previous_summary,
        },
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        _write_output(args.output_json, rendered + "\n")
    if args.output_md:
        _write_output(
            args.output_md,
            render_competitive_surpass_trend_markdown(payload, args.title),
        )
    print(rendered)


if __name__ == "__main__":
    main()
