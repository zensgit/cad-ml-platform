from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


METRICS = [
    "total_lines",
    "total_circles",
    "total_arcs",
    "avg_ink_ratio",
    "avg_components",
]


def _format_thresholds(thresholds: Dict[str, Any]) -> str:
    if not thresholds:
        return "-"
    parts = []
    for key in sorted(thresholds.keys()):
        parts.append(f"{key}={thresholds[key]}")
    return ", ".join(parts)


def _render_summary_table(summary: Dict[str, Any]) -> str:
    lines = [
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in METRICS:
        lines.append(f"| {key} | {summary.get(key, '-')} |")
    return "\n".join(lines)


def _build_report(payload: Dict[str, Any]) -> str:
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("input JSON missing results list")

    lines: List[str] = ["# CAD Feature Benchmark Baseline Summary", ""]
    for idx, entry in enumerate(results, start=1):
        thresholds = entry.get("thresholds", {})
        summary = entry.get("summary", {})
        if not isinstance(thresholds, dict):
            thresholds = {}
        if not isinstance(summary, dict):
            summary = {}
        lines.append(f"## Combo {idx}")
        lines.append(f"**Thresholds**: {_format_thresholds(thresholds)}")
        lines.append("")
        lines.append(_render_summary_table(summary))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render CAD feature benchmark baseline report.")
    parser.add_argument("--input-json", type=Path, required=True, help="Benchmark JSON with results")
    parser.add_argument("--output-md", type=Path, help="Optional markdown output file")
    args = parser.parse_args()

    payload = json.loads(args.input_json.read_text())
    report = _build_report(payload)
    if args.output_md:
        args.output_md.write_text(report)
    else:
        print(report)


if __name__ == "__main__":
    main()
