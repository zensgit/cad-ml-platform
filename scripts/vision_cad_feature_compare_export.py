from __future__ import annotations

import argparse
import csv
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


def _select_top_samples(sample_deltas: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    ranked = sorted(
        sample_deltas,
        key=lambda item: abs(item.get("components_delta", 0)),
        reverse=True,
    )
    return ranked[:top_n]


def _build_exports(
    payload: Dict[str, Any],
    top_n: int,
    combo_index: int | None,
) -> List[Dict[str, Any]]:
    comparison = payload.get("comparison")
    if not isinstance(comparison, dict):
        raise ValueError("input JSON missing comparison block")
    combo_deltas = comparison.get("combo_deltas", [])
    if not isinstance(combo_deltas, list):
        raise ValueError("comparison.combo_deltas must be a list")

    results = payload.get("results", [])
    exports: List[Dict[str, Any]] = []
    for idx, combo in enumerate(combo_deltas, start=1):
        if combo_index is not None and idx != combo_index:
            continue
        status = "missing_baseline" if combo.get("missing_baseline") else "ok"
        thresholds: Dict[str, Any] = {}
        if idx - 1 < len(results):
            thresholds = results[idx - 1].get("thresholds", {})
        summary_delta = combo.get("summary_delta", {}) if status == "ok" else {}
        sample_deltas = combo.get("sample_deltas", []) if status == "ok" else []
        exports.append(
            {
                "combo_index": idx,
                "status": status,
                "thresholds": thresholds,
                "summary_delta": summary_delta,
                "top_samples": _select_top_samples(sample_deltas, top_n),
            }
        )
    return exports


def _write_csv(output_path: Path, exports: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "combo_index",
        "status",
        "sample",
        "lines_delta",
        "circles_delta",
        "arcs_delta",
        "ink_ratio_delta",
        "components_delta",
        "thresholds",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in exports:
            thresholds = _format_thresholds(entry.get("thresholds", {}))
            status = entry.get("status")
            samples = entry.get("top_samples", [])
            if not samples:
                writer.writerow(
                    {
                        "combo_index": entry.get("combo_index"),
                        "status": status,
                        "sample": "",
                        "lines_delta": "",
                        "circles_delta": "",
                        "arcs_delta": "",
                        "ink_ratio_delta": "",
                        "components_delta": "",
                        "thresholds": thresholds,
                    }
                )
                continue
            for sample in samples:
                writer.writerow(
                    {
                        "combo_index": entry.get("combo_index"),
                        "status": status,
                        "sample": sample.get("name"),
                        "lines_delta": sample.get("lines_delta"),
                        "circles_delta": sample.get("circles_delta"),
                        "arcs_delta": sample.get("arcs_delta"),
                        "ink_ratio_delta": sample.get("ink_ratio_delta"),
                        "components_delta": sample.get("components_delta"),
                        "thresholds": thresholds,
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export CAD feature benchmark comparison deltas."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Benchmark JSON with comparison block",
    )
    parser.add_argument("--output-json", type=Path, help="Optional JSON output file")
    parser.add_argument("--output-csv", type=Path, help="Optional CSV output file")
    parser.add_argument(
        "--top-samples",
        type=int,
        default=10,
        help="Top sample deltas to include",
    )
    parser.add_argument(
        "--combo-index",
        type=int,
        help="Filter output to a single combo index (1-based)",
    )
    args = parser.parse_args()

    if args.combo_index is not None and args.combo_index < 1:
        raise ValueError("combo index must be >= 1")

    payload = json.loads(args.input_json.read_text())
    exports = _build_exports(payload, args.top_samples, args.combo_index)
    if args.combo_index is not None and not exports:
        raise ValueError("combo index out of range")

    output_payload = {
        "top_samples": args.top_samples,
        "combo_exports": exports,
    }

    if args.output_json:
        args.output_json.write_text(json.dumps(output_payload, indent=2))
    if args.output_csv:
        _write_csv(args.output_csv, exports)
    if not args.output_json and not args.output_csv:
        print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
