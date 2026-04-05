#!/usr/bin/env python3
"""Top-level eval reporting bundle: orchestrate sub-bundles and HTML reports."""
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

from scripts.ci import generate_eval_signal_reporting_bundle as eval_signal_bundle_mod
from scripts.ci import generate_history_sequence_reporting_bundle as history_sequence_bundle_mod
from scripts import generate_eval_report as static_report_mod
from scripts import generate_eval_report_v2 as interactive_report_mod


def _build_bundle_markdown(manifest: Dict[str, Any]) -> str:
    lines = [
        "# Eval Reporting Bundle",
        "",
        f"- Generated at: `{manifest.get('generated_at', '')}`",
        f"- Eval history dir: `{manifest.get('eval_history_dir', '')}`",
        "",
        "## Sub-Bundles",
        "",
        f"- Eval signal bundle: `{manifest.get('eval_signal_bundle_json', '')}`",
        f"- History sequence bundle: `{manifest.get('history_sequence_bundle_json', '')}`",
        "",
        "## Reports",
        "",
        f"- Static report: `{manifest.get('static_report_html', '')}`",
        f"- Interactive report: `{manifest.get('interactive_report_html', '')}`",
        f"- Plots dir: `{manifest.get('plots_dir', '')}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize top-level eval reporting bundle (orchestrates sub-bundles + HTML reports)."
    )
    parser.add_argument(
        "--eval-history-dir",
        default="reports/eval_history",
        help="Directory containing eval-history JSON records.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Weekly summary rolling window in days.",
    )
    parser.add_argument(
        "--static-report-out",
        default="",
        help="Output directory for static HTML report (default: <eval-history-dir>/report_static).",
    )
    parser.add_argument(
        "--interactive-report-out",
        default="",
        help="Output directory for interactive HTML report (default: <eval-history-dir>/report_interactive).",
    )
    parser.add_argument(
        "--bundle-json",
        default="",
        help="Top-level bundle manifest JSON path (default: <eval-history-dir>/eval_reporting_bundle.json).",
    )
    parser.add_argument(
        "--bundle-md",
        default="",
        help="Top-level bundle manifest Markdown path (default: <eval-history-dir>/eval_reporting_bundle.md).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    eval_history_dir = Path(str(args.eval_history_dir))
    days_str = str(max(1, int(args.days)))

    static_report_out = (
        Path(str(args.static_report_out))
        if str(args.static_report_out).strip()
        else eval_history_dir / "report_static"
    )
    interactive_report_out = (
        Path(str(args.interactive_report_out))
        if str(args.interactive_report_out).strip()
        else eval_history_dir / "report_interactive"
    )
    bundle_json = (
        Path(str(args.bundle_json))
        if str(args.bundle_json).strip()
        else eval_history_dir / "eval_reporting_bundle.json"
    )
    bundle_md = (
        Path(str(args.bundle_md))
        if str(args.bundle_md).strip()
        else eval_history_dir / "eval_reporting_bundle.md"
    )

    eval_signal_bundle_json = eval_history_dir / "eval_signal_reporting_bundle.json"
    history_sequence_bundle_json = eval_history_dir / "history_sequence_reporting_bundle.json"
    plots_dir = eval_history_dir / "plots"

    # 1. eval_signal sub-bundle
    print("=== Step 1/4: eval-signal reporting bundle ===")
    rc = eval_signal_bundle_mod.main([
        "--eval-history-dir", str(eval_history_dir),
        "--days", days_str,
        "--bundle-json", str(eval_signal_bundle_json),
    ])
    if rc != 0:
        print(f"eval-signal reporting bundle failed with rc={rc}")
        return rc

    # 2. history_sequence sub-bundle
    print("=== Step 2/4: history-sequence reporting bundle ===")
    rc = history_sequence_bundle_mod.main([
        "--eval-history-dir", str(eval_history_dir),
        "--days", days_str,
        "--bundle-json", str(history_sequence_bundle_json),
    ])
    if rc != 0:
        print(f"history-sequence reporting bundle failed with rc={rc}")
        return rc

    # 3. static HTML report
    print("=== Step 3/4: static HTML report ===")
    rc = static_report_mod.main([
        "--history-dir", str(eval_history_dir),
        "--out", str(static_report_out),
    ])
    if rc != 0:
        print(f"static HTML report failed with rc={rc}")
        return rc

    # 4. interactive HTML report (v2)
    print("=== Step 4/4: interactive HTML report (v2) ===")
    rc = interactive_report_mod.main([
        "--dir", str(eval_history_dir),
        "--out", str(interactive_report_out),
    ])
    if rc is None:
        rc = 0
    if rc != 0:
        print(f"interactive HTML report failed with rc={rc}")
        return rc

    static_report_html = static_report_out / "index.html"
    interactive_report_html = interactive_report_out / "index.html"

    manifest: Dict[str, Any] = {
        "status": "ok",
        "surface_kind": "eval_reporting_bundle",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eval_history_dir": str(eval_history_dir),
        "eval_signal_bundle_json": str(eval_signal_bundle_json),
        "history_sequence_bundle_json": str(history_sequence_bundle_json),
        "static_report_html": str(static_report_html),
        "interactive_report_html": str(interactive_report_html),
        "plots_dir": str(plots_dir),
    }

    bundle_json.parent.mkdir(parents=True, exist_ok=True)
    bundle_md.parent.mkdir(parents=True, exist_ok=True)
    bundle_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    bundle_md.write_text(_build_bundle_markdown(manifest), encoding="utf-8")

    print(f"Top-level eval reporting bundle JSON: {bundle_json}")
    print(f"Top-level eval reporting bundle Markdown: {bundle_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
