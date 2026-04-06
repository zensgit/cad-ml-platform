#!/usr/bin/env python3
"""Generate the top-level eval reporting discovery / index artifact."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_report_data_helpers import load_json_dict


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def build_index(
    eval_history_dir: Path,
    *,
    bundle_json_path: Optional[Path] = None,
    health_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    bundle_path = bundle_json_path or (eval_history_dir / "eval_reporting_bundle.json")
    health_path = health_json_path or (eval_history_dir / "eval_reporting_bundle_health_report.json")

    bundle = load_json_dict(bundle_path)
    top = bundle if isinstance(bundle, dict) else {}

    return {
        "status": "ok" if top else "no_bundle",
        "surface_kind": "eval_reporting_index",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eval_history_dir": str(eval_history_dir),
        "eval_reporting_bundle_json": str(bundle_path),
        "eval_reporting_bundle_health_json": str(health_path),
        "eval_signal_bundle_json": _safe_str(top.get("eval_signal_bundle_json")),
        "history_sequence_bundle_json": _safe_str(top.get("history_sequence_bundle_json")),
        "static_report_html": _safe_str(top.get("static_report_html")),
        "interactive_report_html": _safe_str(top.get("interactive_report_html")),
        "plots_dir": _safe_str(top.get("plots_dir")),
        "landing_page_html": str(eval_history_dir / "index.html"),
    }


def _build_index_markdown(index: Dict[str, Any]) -> str:
    lines = [
        "# Eval Reporting Index",
        "",
        f"- Status: `{index.get('status', '')}`",
        f"- Generated at: `{index.get('generated_at', '')}`",
        f"- Eval history dir: `{index.get('eval_history_dir', '')}`",
        "",
        "## Artifacts",
        "",
        f"- Bundle JSON: `{index.get('eval_reporting_bundle_json', '')}`",
        f"- Health report: `{index.get('eval_reporting_bundle_health_json', '')}`",
        f"- Eval signal bundle: `{index.get('eval_signal_bundle_json', '')}`",
        f"- History sequence bundle: `{index.get('history_sequence_bundle_json', '')}`",
        f"- Static report: `{index.get('static_report_html', '')}`",
        f"- Interactive report: `{index.get('interactive_report_html', '')}`",
        f"- Plots dir: `{index.get('plots_dir', '')}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the top-level eval reporting discovery / index artifact."
    )
    parser.add_argument(
        "--eval-history-dir",
        default="reports/eval_history",
        help="Directory containing eval-history artifacts.",
    )
    parser.add_argument(
        "--bundle-json",
        default="",
        help="Top-level bundle JSON path.",
    )
    parser.add_argument(
        "--health-json",
        default="",
        help="Health report JSON path.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Index JSON output path.",
    )
    parser.add_argument(
        "--output-md",
        default="",
        help="Index Markdown output path.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    eval_history_dir = Path(str(args.eval_history_dir))
    bundle_json_path = Path(str(args.bundle_json)) if str(args.bundle_json).strip() else None
    health_json_path = Path(str(args.health_json)) if str(args.health_json).strip() else None
    output_json = (
        Path(str(args.output_json))
        if str(args.output_json).strip()
        else eval_history_dir / "eval_reporting_index.json"
    )
    output_md = (
        Path(str(args.output_md))
        if str(args.output_md).strip()
        else eval_history_dir / "eval_reporting_index.md"
    )

    index = build_index(
        eval_history_dir,
        bundle_json_path=bundle_json_path,
        health_json_path=health_json_path,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(_build_index_markdown(index), encoding="utf-8")

    print(f"Eval reporting index JSON: {output_json}")
    print(f"Eval reporting index Markdown: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
