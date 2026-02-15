#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _bar(value: float, width: int = 16) -> str:
    v = max(0.0, min(1.0, float(value)))
    filled = int(round(v * width))
    return f"[{'#' * filled}{'.' * (width - filled)}]"


def build_trend_markdown(
    *, summary: Dict[str, Any], rows: List[Dict[str, Any]], title: str
) -> str:
    low_thr = _safe_float(summary.get("strict_low_conf_threshold"), 0.2)
    sorted_rows = sorted(rows, key=lambda r: _safe_int(r.get("seed"), 0))

    out: List[str] = []
    out.append(f"### {title} Trend")
    out.append("")
    if not sorted_rows:
        out.append("No per-seed rows found.")
        out.append("")
        return "\n".join(out)

    out.append(
        "| Seed | Status | Strict acc | Top-pred ratio | Low-conf ratio | Trend (top-pred / low-conf) |"
    )
    out.append("|---:|---|---:|---:|---:|---|")
    for row in sorted_rows:
        seed = _safe_int(row.get("seed"), 0)
        status = str(row.get("status", ""))
        acc = _safe_float(row.get("strict_accuracy"), -1.0)
        top_pred_ratio = _safe_float(row.get("strict_top_pred_ratio"), 0.0)
        low_conf_ratio = _safe_float(row.get("strict_low_conf_ratio"), 0.0)
        top_bar = _bar(top_pred_ratio)
        low_bar = _bar(low_conf_ratio)
        out.append(
            f"| {seed} | `{status}` | `{acc:.4f}` | `{top_pred_ratio:.4f}` | "
            f"`{low_conf_ratio:.4f} (<{low_thr:.3f})` | `{top_bar} / {low_bar}` |"
        )

    out.append("")
    out.append(
        "Aggregates: "
        f"`strict_accuracy_mean={_safe_float(summary.get('strict_accuracy_mean'), -1.0):.6f}`, "
        f"`strict_top_pred_ratio_mean/max={_safe_float(summary.get('strict_top_pred_ratio_mean'), 0.0):.6f}/"
        f"{_safe_float(summary.get('strict_top_pred_ratio_max'), 0.0):.6f}`, "
        f"`strict_low_conf_ratio_mean/max={_safe_float(summary.get('strict_low_conf_ratio_mean'), 0.0):.6f}/"
        f"{_safe_float(summary.get('strict_low_conf_ratio_max'), 0.0):.6f}`"
    )
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render Graph2D seed gate trend markdown from seed sweep outputs."
    )
    parser.add_argument("--summary-json", required=True, help="Path to seed_sweep_summary.json")
    parser.add_argument(
        "--results-json",
        default="",
        help="Path to seed_sweep_results.json (default: next to summary-json).",
    )
    parser.add_argument("--title", required=True, help="Trend title")
    parser.add_argument(
        "--output-md",
        default="",
        help="Optional markdown file output path (print to stdout when omitted).",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    summary_payload = _read_json(summary_path)
    summary = summary_payload if isinstance(summary_payload, dict) else {}

    if args.results_json:
        results_path = Path(args.results_json)
    else:
        results_path = summary_path.parent / "seed_sweep_results.json"
    rows_payload = _read_json(results_path)
    rows = rows_payload if isinstance(rows_payload, list) else []

    markdown = build_trend_markdown(summary=summary, rows=rows, title=args.title)
    if args.output_md:
        out_path = Path(args.output_md)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

