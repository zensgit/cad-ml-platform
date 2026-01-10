from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _format_thresholds(thresholds: Dict[str, Any]) -> str:
    if not thresholds:
        return "-"
    parts = []
    for key in sorted(thresholds.keys()):
        parts.append(f"{key}={thresholds[key]}")
    return ", ".join(parts)


def _render_summary_table(summary_delta: Dict[str, Any]) -> str:
    headers = [
        "total_lines",
        "total_circles",
        "total_arcs",
        "avg_ink_ratio",
        "avg_components",
    ]
    lines = [
        "| metric | delta |",
        "| --- | --- |",
    ]
    for key in headers:
        lines.append(f"| {key} | {summary_delta.get(key, '-')} |")
    return "\n".join(lines)


def _render_sample_table(sample_deltas: List[Dict[str, Any]], top_n: int) -> str:
    if not sample_deltas:
        return "No sample deltas available."
    ranked = sorted(
        sample_deltas,
        key=lambda item: abs(item.get("components_delta", 0)),
        reverse=True,
    )
    lines = [
        "| sample | lines | circles | arcs | components |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in ranked[:top_n]:
        lines.append(
            "| {name} | {lines_delta} | {circles_delta} | {arcs_delta} | {components_delta} |".format(
                **item
            )
        )
    return "\n".join(lines)


def _build_report(payload: Dict[str, Any], top_n: int) -> str:
    comparison = payload.get("comparison")
    if not isinstance(comparison, dict):
        raise ValueError("input JSON missing comparison block")
    combo_deltas = comparison.get("combo_deltas", [])
    if not isinstance(combo_deltas, list):
        raise ValueError("comparison.combo_deltas must be a list")

    results = payload.get("results", [])
    lines: List[str] = ["# CAD Feature Benchmark Comparison Summary", ""]
    for idx, combo in enumerate(combo_deltas, start=1):
        lines.append(f"## Combo {idx}")
        if combo.get("missing_baseline"):
            lines.append("Baseline entry missing; no delta available.")
            lines.append("")
            continue

        thresholds = {}
        if idx - 1 < len(results):
            thresholds = results[idx - 1].get("thresholds", {})
        lines.append(f"**Thresholds**: {_format_thresholds(thresholds)}")
        lines.append("")
        lines.append("**Summary Delta**")
        lines.append(_render_summary_table(combo.get("summary_delta", {})))
        lines.append("")
        lines.append("**Top Sample Deltas**")
        lines.append(_render_sample_table(combo.get("sample_deltas", []), top_n))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render CAD feature benchmark comparison report.")
    parser.add_argument("--input-json", type=Path, required=True, help="Benchmark JSON with comparison block")
    parser.add_argument("--output-md", type=Path, help="Optional markdown output file")
    parser.add_argument("--top-samples", type=int, default=10, help="Top sample deltas to include")
    args = parser.parse_args()

    payload = json.loads(args.input_json.read_text())
    report = _build_report(payload, args.top_samples)
    if args.output_md:
        args.output_md.write_text(report)
    else:
        print(report)


if __name__ == "__main__":
    main()
