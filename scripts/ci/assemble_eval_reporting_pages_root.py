#!/usr/bin/env python3
"""Assemble a Pages-ready root from existing eval reporting stack artifacts.

Copies existing canonical files into a flat Pages root directory.
Does NOT regenerate, recompute, or re-render any content.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]


def assemble(
    eval_history_dir: Path,
    pages_root: Path,
) -> int:
    pages_root.mkdir(parents=True, exist_ok=True)

    # Landing page → root index.html
    landing = eval_history_dir / "index.html"
    if landing.exists():
        shutil.copy2(landing, pages_root / "index.html")
    else:
        print(f"Warning: landing page not found at {landing}")

    # Static report
    static_src = eval_history_dir / "report_static"
    if static_src.is_dir():
        dst = pages_root / "report_static"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(static_src, dst)
    else:
        print(f"Warning: static report not found at {static_src}")

    # Interactive report
    interactive_src = eval_history_dir / "report_interactive"
    if interactive_src.is_dir():
        dst = pages_root / "report_interactive"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(interactive_src, dst)
    else:
        print(f"Warning: interactive report not found at {interactive_src}")

    # Canonical JSON/MD discovery assets (additive)
    for name in [
        "eval_reporting_bundle.json",
        "eval_reporting_bundle.md",
        "eval_reporting_bundle_health_report.json",
        "eval_reporting_bundle_health_report.md",
        "eval_reporting_index.json",
        "eval_reporting_index.md",
    ]:
        src = eval_history_dir / name
        if src.exists():
            shutil.copy2(src, pages_root / name)

    print(f"Pages root assembled: {pages_root}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble a Pages-ready root from existing eval reporting stack artifacts."
    )
    parser.add_argument(
        "--eval-history-dir",
        default="reports/eval_history",
        help="Directory containing eval-history artifacts.",
    )
    parser.add_argument(
        "--pages-root",
        default="reports/eval_pages",
        help="Output Pages root directory.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    return assemble(
        eval_history_dir=Path(str(args.eval_history_dir)),
        pages_root=Path(str(args.pages_root)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
